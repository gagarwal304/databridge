from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import product
from typing import Any, Awaitable, Callable

from rapidfuzz import fuzz

from databridge.connectors.base import TableMeta
from databridge.connectors.registry import ConnectorRegistry
from databridge.schema.joins.transforms import TRANSFORM_GRAMMAR, detect_transform

# Receives (col_a_label, sample_a, col_b_label, sample_b) and returns a transform
# name from TRANSFORM_GRAMMAR, or None if it can't determine one.
SamplingCallback = Callable[[str, list[Any], str, list[Any]], Awaitable[str | None]]

_VALID_TRANSFORMS = {name for name, _ in TRANSFORM_GRAMMAR}


_NLTK_READY = False


def _ensure_nltk() -> None:
    global _NLTK_READY
    if _NLTK_READY:
        return
    import nltk
    for pkg in ("wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)
    _NLTK_READY = True


# Curated database-convention synonyms that WordNet doesn't cover
_DB_SYNONYMS: dict[str, set[str]] = {
    "id": {"_id", "uid", "pk", "ref", "key"},
    "ts": {"timestamp", "created_at", "updated_at", "datetime", "date"},
    "amt": {"amount", "total", "sum", "price", "cost", "value"},
    "qty": {"quantity", "count", "num", "number"},
    "addr": {"address", "street", "location"},
    "dob": {"birth_date", "birthdate", "birth_day", "date_of_birth"},
    "fname": {"first_name", "given_name", "forename"},
    "lname": {"last_name", "surname", "family_name"},
}

# Build reverse lookup
_SYNONYM_MAP: dict[str, set[str]] = {}
for canonical, aliases in _DB_SYNONYMS.items():
    all_terms = {canonical} | aliases
    for term in all_terms:
        _SYNONYM_MAP.setdefault(term, set()).update(all_terms - {term})


def _normalize_col(name: str) -> str:
    """snake_case → tokens, drop common suffixes."""
    tokens = re.split(r"[_\s]+", name.lower())
    drop = {"id", "code", "num", "no", "ref", "key", "cd", "fk"}
    filtered = [t for t in tokens if t and t not in drop]
    return " ".join(filtered) if filtered else name.lower()


def _wordnet_synonyms(word: str) -> set[str]:
    try:
        _ensure_nltk()
        from nltk.corpus import wordnet
        syns: set[str] = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                syns.add(lemma.name().replace("_", " ").lower())
        return syns
    except Exception:
        return set()


def _column_similarity(name_a: str, name_b: str) -> float:
    """
    Returns a similarity score [0, 1] combining:
    - rapidfuzz edit distance on normalised tokens
    - WordNet synonym expansion
    - curated DB synonym dict
    """
    norm_a = _normalize_col(name_a)
    norm_b = _normalize_col(name_b)

    # Direct fuzzy match
    base_score = fuzz.token_sort_ratio(norm_a, norm_b) / 100.0

    # Synonym expansion for each token
    tokens_a = set(norm_a.split())
    tokens_b = set(norm_b.split())

    expanded_a = set(tokens_a)
    for tok in tokens_a:
        expanded_a |= _wordnet_synonyms(tok)
        expanded_a |= _SYNONYM_MAP.get(tok, set())

    expanded_b = set(tokens_b)
    for tok in tokens_b:
        expanded_b |= _wordnet_synonyms(tok)
        expanded_b |= _SYNONYM_MAP.get(tok, set())

    # If any token from A matches any token from B after expansion, boost score
    if expanded_a & expanded_b:
        base_score = max(base_score, 0.85)

    return base_score


@dataclass
class JoinCandidate:
    db_a: str
    table_a: str
    column_a: str
    db_b: str
    table_b: str
    column_b: str
    name_similarity: float
    overlap: float
    transform: str | None
    confidence: float

    @property
    def join_id(self) -> str:
        return f"{self.db_a}.{self.table_a}.{self.column_a}__{self.db_b}.{self.table_b}.{self.column_b}"


class JoinDiscovery:
    def __init__(
        self,
        registry: ConnectorRegistry,
        name_similarity_threshold: float = 0.70,
        value_sample_size: int = 50,
        overlap_threshold: float = 0.70,
        min_confidence: float = 0.60,
    ) -> None:
        self._registry = registry
        self._name_sim_threshold = name_similarity_threshold
        self._sample_size = value_sample_size
        self._overlap_threshold = overlap_threshold
        self._min_confidence = min_confidence

    async def discover(
        self,
        schema: dict[str, dict[str, TableMeta]],
        sampling_callback: SamplingCallback | None = None,
    ) -> list[JoinCandidate]:
        """
        Phase 1: name-similarity filter across all (db, table, column) pairs.
        Phase 2: value sampling + transform grammar on survivors.
        Phase 3 (optional): LLM sampling callback for ambiguous pairs that
          passed name-similarity but whose transform couldn't be identified
          by the grammar alone.

        Returns candidates above min_confidence, ranked descending.
        """
        column_index: list[tuple[str, str, str]] = []  # (db, table, col)
        for db_alias, tables in schema.items():
            for tname, table in tables.items():
                for cname in table.columns:
                    column_index.append((db_alias, tname, cname))

        confirmed: list[JoinCandidate] = []
        # Ambiguous: name-similar pairs where the grammar found no transform.
        # Stored as (db_a, ta, ca, db_b, tb, cb, name_sim, vals_a, vals_b).
        ambiguous: list[tuple] = []

        # Only cross-database pairs
        for (db_a, ta, ca), (db_b, tb, cb) in product(column_index, column_index):
            if db_a >= db_b:  # avoid duplicates and self-joins
                continue
            name_sim = _column_similarity(ca, cb)
            if name_sim < self._name_sim_threshold:
                continue

            # Value sampling
            try:
                vals_a = await self._registry.get(db_a).sample_column(ta, ca, self._sample_size)
                vals_b = await self._registry.get(db_b).sample_column(tb, cb, self._sample_size)
            except Exception:
                continue

            if not vals_a or not vals_b:
                continue

            transform_name, overlap = detect_transform(vals_a, vals_b, self._overlap_threshold)

            # Also try B→A direction
            if overlap < self._overlap_threshold:
                transform_name_ba, overlap_ba = detect_transform(vals_b, vals_a, self._overlap_threshold)
                if overlap_ba > overlap:
                    db_a, db_b = db_b, db_a
                    ta, tb = tb, ta
                    ca, cb = cb, ca
                    transform_name, overlap = transform_name_ba, overlap_ba

            confidence = _compute_confidence(name_sim, overlap)
            if confidence >= self._min_confidence:
                confirmed.append(JoinCandidate(
                    db_a=db_a, table_a=ta, column_a=ca,
                    db_b=db_b, table_b=tb, column_b=cb,
                    name_similarity=name_sim,
                    overlap=overlap,
                    transform=transform_name,
                    confidence=confidence,
                ))
            elif name_sim >= 0.80 and sampling_callback is not None:
                # High name similarity but grammar couldn't find a transform —
                # worth asking the LLM.
                ambiguous.append((db_a, ta, ca, db_b, tb, cb, name_sim, vals_a, vals_b))

        # Phase 3: LLM sampling for ambiguous pairs
        if sampling_callback is not None:
            for db_a, ta, ca, db_b, tb, cb, name_sim, vals_a, vals_b in ambiguous:
                try:
                    col_a_label = f"{db_a}.{ta}.{ca}"
                    col_b_label = f"{db_b}.{tb}.{cb}"
                    transform_name = await sampling_callback(col_a_label, vals_a, col_b_label, vals_b)
                except Exception:
                    continue

                if transform_name not in _VALID_TRANSFORMS:
                    continue

                # Validate the suggested transform against the actual samples
                _, overlap = detect_transform(vals_a, vals_b, threshold=0.0)
                # Re-run with the specific transform to get its actual overlap
                from databridge.schema.joins.transforms import TRANSFORM_GRAMMAR as _TG
                fn = next((f for n, f in _TG if n == transform_name), None)
                if fn is None:
                    continue
                set_b = {str(v).strip() for v in vals_b if v is not None}
                transformed = {fn(v) for v in vals_a if v is not None}
                if not transformed:
                    continue
                overlap = len(transformed & set_b) / len(transformed)
                if overlap < self._overlap_threshold:
                    continue

                confidence = _compute_confidence(name_sim, overlap)
                if confidence < self._min_confidence:
                    continue

                confirmed.append(JoinCandidate(
                    db_a=db_a, table_a=ta, column_a=ca,
                    db_b=db_b, table_b=tb, column_b=cb,
                    name_similarity=name_sim,
                    overlap=overlap,
                    transform=transform_name,
                    confidence=confidence,
                ))

        confirmed.sort(key=lambda c: c.confidence, reverse=True)
        return confirmed

    async def verify_rules(self, rules: list) -> list[tuple]:
        """
        Run value sampling on agent-loop-discovered joins to compute proper confidence.

        For each rule: sample both columns, detect transform + overlap, recompute
        confidence with the same formula as full discovery. Returns
        (rule, new_confidence, new_transform) for rules that pass min_confidence.
        Rules with unknown/missing table names are silently skipped.
        """
        results = []
        for rule in rules:
            if not rule.table_a or not rule.table_b:
                continue
            if rule.table_a == "unknown" or rule.table_b == "unknown":
                continue
            try:
                conn_a = self._registry.get(rule.db_a)
                conn_b = self._registry.get(rule.db_b)
                if conn_a is None or conn_b is None:
                    continue
                vals_a = await conn_a.sample_column(rule.table_a, rule.column_a, self._sample_size)
                vals_b = await conn_b.sample_column(rule.table_b, rule.column_b, self._sample_size)
            except Exception:
                continue
            if not vals_a or not vals_b:
                continue
            transform_name, overlap = detect_transform(vals_a, vals_b, self._overlap_threshold)
            # Try reverse direction — agent may have written the join in either order
            transform_rev, overlap_rev = detect_transform(vals_b, vals_a, self._overlap_threshold)
            if overlap_rev > overlap:
                transform_name, overlap = transform_rev, overlap_rev
            name_sim = _column_similarity(rule.column_a, rule.column_b)
            confidence = _compute_confidence(name_sim, overlap)
            if confidence >= self._min_confidence:
                results.append((rule, confidence, transform_name))
        return results


def _compute_confidence(name_sim: float, overlap: float) -> float:
    """
    Weighted combination: name similarity matters, but value overlap is the
    stronger signal.
    """
    return round(0.35 * name_sim + 0.65 * overlap, 4)
