"""
Runtime join detection on small result sets.

Brute-forces all column pairs × transform grammar to find the link between two
result sets. Much more tractable than schema-level sampling because Phase 2
queries return at most LIMIT 50 rows per DB.

Discovered joins are stored in JoinRegistry so subsequent queries skip detection.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

SAMPLE_ROWS = 500  # cap on rows used for detection; large enough for 70%+ overlap on genuine joins

def _has_float_values(values: list[str]) -> bool:
    """Return True if >20% of sampled values are non-integer floats (e.g. price = 8.23)."""
    check = values[:20]
    if not check:
        return False
    decimal_count = 0
    for v in check:
        try:
            f = float(v)
            if f != int(f):
                decimal_count += 1
        except (ValueError, TypeError):
            pass
    return decimal_count / len(check) > 0.2


@dataclass
class JoinMatch:
    col_a: str
    col_b: str
    db_a: str
    db_b: str
    table_a: str
    table_b: str
    transform: str          # name from TRANSFORM_GRAMMAR
    transform_sql: str      # SQL expression describing the join condition
    overlap: float          # fraction of A values that matched B


def detect_join(
    rows_a: list[dict],
    db_a: str,
    table_a: str,
    rows_b: list[dict],
    db_b: str,
    table_b: str,
    min_overlap: float = 0.90,
) -> JoinMatch | None:
    """
    Try every column pair × every transform to find a join between two result sets.
    Also tries JSON field extraction on string columns.
    Returns the best match above min_overlap, or None.
    """
    from databridge.schema.joins.transforms import TRANSFORM_GRAMMAR

    sample_a = rows_a[:SAMPLE_ROWS]
    sample_b = rows_b[:SAMPLE_ROWS]

    if not sample_a or not sample_b:
        return None

    # Build expanded raw index for B: {col_key → list of raw string values}
    # col_key is either "col" or "col__jsonfield" for JSON-extracted values
    raw_index_b: dict[str, list[str]] = {}
    for col in sample_b[0].keys():
        raw = [r[col] for r in sample_b if r.get(col) is not None]
        if raw:
            raw_index_b[col] = [str(v).strip() for v in raw]
        # Try JSON extraction
        for json_key, json_vals in _extract_json_fields(raw, col).items():
            raw_index_b[json_key] = [str(v).strip() for v in json_vals]

    best: JoinMatch | None = None
    best_overlap = min_overlap - 0.001

    cols_a = list(sample_a[0].keys())

    for col_a in cols_a:
        vals_a = [r[col_a] for r in sample_a if r.get(col_a) is not None]
        if not vals_a:
            continue
        if _has_float_values(vals_a):
            continue

        for col_b_key, raw_vals_b in raw_index_b.items():
            if not raw_vals_b:
                continue
            if _has_float_values(raw_vals_b):
                continue

            for transform_name, fn in TRANSFORM_GRAMMAR:
                # extract_digits / zfill transforms are designed for string IDs like
                # 'bookid_5' → '5'. Applied to pure-integer columns (helpful_vote,
                # rating) they behave like identity but create false matches with string
                # ID columns. Skip these transforms when either side is purely numeric.
                if transform_name in ("extract_digits", "zfill_5", "zfill_7"):
                    a_all_int = all(str(v).lstrip("-").isdigit() for v in vals_a[:20] if v is not None)
                    b_all_int = all(v.lstrip("-").isdigit() for v in raw_vals_b[:20])
                    if a_all_int or b_all_int:
                        continue

                # Apply the same transform to BOTH sides so symmetric normalizations
                # (e.g. extract_digits: 'bookid_8' and 'purchaseid_8' both → '8') are found.
                transformed_a: set[str] = set()
                for v in vals_a:
                    try:
                        transformed_a.add(fn(v))
                    except Exception:
                        continue
                # A join key must have enough cardinality on both sides.
                # Single-value sets (e.g. all author names → "00000" via zfill_5)
                # trivially match and produce false positives.
                if len(transformed_a) < 3:
                    continue

                transformed_b: set[str] = set()
                for v in raw_vals_b:
                    try:
                        transformed_b.add(fn(v))
                    except Exception:
                        continue
                if len(transformed_b) < 3:
                    continue

                # One-directional coverage from the FK (high-coverage) side.
                # A genuine FK has nearly all its values present in the PK column,
                # but the PK may have entities with no FK rows yet (e.g. unreviewed books).
                # min() penalises genuine joins; max() picks the FK-side coverage.
                #
                # Guard: the high-coverage (FK) side must have ≥15% unique rate.
                # Bounded-domain measures (rating 1-5, score 0-24) have very low
                # unique rates and should not be classified as join keys.
                inter = len(transformed_a & transformed_b)
                cov_a = inter / len(transformed_a)
                cov_b = inter / len(transformed_b)
                if cov_a >= cov_b:
                    # A is FK: check uniqueness of raw A values
                    raw_uniq = len(set(str(v) for v in vals_a)) / max(len(vals_a), 1)
                    if raw_uniq < 0.15:
                        continue
                    overlap = cov_a
                else:
                    # B is FK: check uniqueness of raw B values
                    raw_uniq = len(set(raw_vals_b)) / max(len(raw_vals_b), 1)
                    if raw_uniq < 0.15:
                        continue
                    overlap = cov_b
                if overlap > best_overlap:
                    best_overlap = overlap
                    actual_col_b = col_b_key.split("__")[0]
                    json_field = col_b_key.split("__", 1)[1] if "__" in col_b_key else None
                    best = JoinMatch(
                        col_a=col_a,
                        col_b=actual_col_b,
                        db_a=db_a,
                        db_b=db_b,
                        table_a=table_a,
                        table_b=table_b,
                        transform=transform_name,
                        transform_sql=_build_transform_sql(col_a, actual_col_b, transform_name, json_field),
                        overlap=overlap,
                    )

    if best:
        log.info(
            "join detected: %s.%s.%s ↔ %s.%s.%s via '%s' (overlap=%.2f)",
            db_a, table_a, best.col_a,
            db_b, table_b, best.col_b,
            best.transform, best.overlap,
        )
    else:
        log.debug("no join found between %s.%s and %s.%s", db_a, table_a, db_b, table_b)

    return best


def _extract_json_fields(values: list[Any], parent_col: str) -> dict[str, list[str]]:
    """Extract scalar fields from JSON string values. Returns {parent_col__field: [values]}."""
    result: dict[str, list[str]] = {}
    for v in values:
        if not isinstance(v, str):
            continue
        stripped = v.strip()
        if not stripped.startswith("{"):
            continue
        try:
            obj = json.loads(stripped)
            if not isinstance(obj, dict):
                continue
            for field, val in obj.items():
                if isinstance(val, (str, int, float)) and val is not None:
                    key = f"{parent_col}__{field}"
                    result.setdefault(key, []).append(str(val))
        except Exception:
            continue
    return result


def _build_transform_sql(col_a: str, col_b: str, transform: str, json_field: str | None) -> str:
    """Human-readable SQL expression for the discovered join condition."""
    b_expr = f"JSON_EXTRACT({col_b}, '$.{json_field}')" if json_field else col_b
    exprs = {
        "identity":          f"CAST({col_a} AS VARCHAR) = CAST({b_expr} AS VARCHAR)",
        "lowercase":         f"LOWER(CAST({col_a} AS VARCHAR)) = LOWER(CAST({b_expr} AS VARCHAR))",
        "extract_digits":    f"REGEXP_REPLACE(CAST({col_a} AS VARCHAR),'[^0-9]','','g') = REGEXP_REPLACE(CAST({b_expr} AS VARCHAR),'[^0-9]','','g')",
        "strip_prefix_3":    f"CAST({col_a} AS VARCHAR) = SUBSTRING(CAST({b_expr} AS VARCHAR),4)",
        "strip_prefix_4":    f"CAST({col_a} AS VARCHAR) = SUBSTRING(CAST({b_expr} AS VARCHAR),5)",
        "strip_prefix_5":    f"CAST({col_a} AS VARCHAR) = SUBSTRING(CAST({b_expr} AS VARCHAR),6)",
        "zfill_5":           f"LPAD(CAST({col_a} AS VARCHAR),5,'0') = CAST({b_expr} AS VARCHAR)",
        "zfill_7":           f"LPAD(CAST({col_a} AS VARCHAR),7,'0') = CAST({b_expr} AS VARCHAR)",
        "remove_separators": f"REPLACE(REPLACE(CAST({col_a} AS VARCHAR),'-',''),'_','') = REPLACE(REPLACE(CAST({b_expr} AS VARCHAR),'-',''),'_','')",
        "cast_int":          f"CAST({col_a} AS INTEGER) = CAST({b_expr} AS INTEGER)",
    }
    return exprs.get(transform, f"CAST({col_a} AS VARCHAR) = CAST({b_expr} AS VARCHAR)")
