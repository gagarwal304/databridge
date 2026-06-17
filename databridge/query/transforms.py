"""
Post-query transform pipeline.

The LLM declares intent (what to compute); Python executes it deterministically.
This replaces fragile in-SQL regex, TRY_CAST patterns, and recursive CTEs for
operations like numeric extraction from prose, JSON unpacking, date normalisation,
and exponential moving averages.
"""
from __future__ import annotations

import json as _json
import re as _re
from collections import defaultdict as _defaultdict
from typing import Any

_METRIC_PATTERNS: dict[str, list[str]] = {
    "stars": [
        r"([\d,]+)\s+stars?\b",
        r"stars?\s+count\s+of\s+([\d,]+)",
    ],
    "forks": [
        r"([\d,]+)\s+forks?\b",
        r"forks?\s+count\s+of\s+([\d,]+)",
        r"forked\s+([\d,]+)\s+times",
    ],
    "issues": [
        r"([\d,]+)\s+(?:open\s+)?issues?\b",
        r"issues?\s+count\s+of\s+([\d,]+)",
    ],
}


def _parse_int_str(s: str | None) -> int | None:
    if not s:
        return None
    try:
        return int(_re.sub(r"[,\s]", "", str(s)))
    except ValueError:
        return None


def _extract_number(text: str, patterns: list[str]) -> int | None:
    for pat in patterns:
        m = _re.search(pat, text, _re.IGNORECASE)
        if m:
            val = _parse_int_str(m.group(1))
            if val is not None:
                return val
    return None


def apply_transforms(rows: list[dict[str, Any]], transforms: list[dict]) -> list[dict[str, Any]]:
    """Apply a sequence of post-processing operations to a list of row dicts."""
    for op_spec in transforms:
        op = op_spec.get("op", "")

        if op == "extract_number":
            col = op_spec.get("column", "")
            metric = op_spec.get("metric", "")
            patterns = op_spec.get("patterns") or _METRIC_PATTERNS.get(metric, [])
            output = op_spec.get("output", metric or col)
            for row in rows:
                row[output] = _extract_number(str(row.get(col) or ""), patterns)

        elif op == "cast_number":
            col = op_spec.get("column", "")
            output = op_spec.get("output", col)
            for row in rows:
                row[output] = _parse_int_str(str(row.get(col) or ""))

        elif op == "project_name_from_text":
            col = op_spec.get("column", "")
            output = op_spec.get("output", "project_name")
            for row in rows:
                text = str(row.get(col) or "")
                m = _re.search(r"project\s+([^\s]+)\s+on\s+github", text, _re.IGNORECASE)
                row[output] = m.group(1) if m else None

        elif op == "sort":
            by = op_spec.get("column", op_spec.get("by", ""))
            reverse = op_spec.get("direction", "desc").lower() == "desc"
            rows = sorted(rows, key=lambda r: (r.get(by) is None, r.get(by) or 0), reverse=reverse)

        elif op == "top_n_with_ties":
            col = op_spec.get("column", "")
            n = int(op_spec.get("n", 5))
            reverse = op_spec.get("direction", "desc").lower() == "desc"
            rows = sorted(rows, key=lambda r: (r.get(col) is None, r.get(col) or 0), reverse=reverse)
            valid = [r for r in rows if r.get(col) is not None]
            if len(valid) > n:
                nth_val = valid[n - 1][col]
                rows = [r for r in valid if (r[col] >= nth_val if reverse else r[col] <= nth_val)]
            else:
                rows = valid

        elif op == "json_array_extract":
            # Extract a named field from every element of a JSON array stored as text.
            # e.g. cpc column: '[{"code":"A61B",...},...]' → ["A61B",...]
            col = op_spec.get("column", "")
            field = op_spec.get("field")
            output = op_spec.get("output", col)
            for row in rows:
                raw = row.get(col)
                try:
                    if isinstance(raw, str):
                        raw = _json.loads(raw)
                    if isinstance(raw, list):
                        row[output] = (
                            [item.get(field) if isinstance(item, dict) else item for item in raw]
                            if field else list(raw)
                        )
                    elif isinstance(raw, dict) and field:
                        row[output] = raw.get(field)
                    else:
                        row[output] = raw
                except (ValueError, AttributeError):
                    row[output] = None

        elif op == "json_explode":
            # Expand a list column into multiple rows — one row per element.
            col = op_spec.get("column", "")
            output = op_spec.get("output", col)
            new_rows: list[dict] = []
            for row in rows:
                val = row.get(col)
                if isinstance(val, list):
                    base = {k: v for k, v in row.items() if k != col}
                    for item in val:
                        new_rows.append({**base, output: item})
                else:
                    new_row = dict(row)
                    if output != col:
                        new_row[output] = val
                    new_rows.append(new_row)
            rows = new_rows

        elif op == "string_explode":
            # Split a string column by a delimiter and create one row per element.
            # Use instead of MongoDB $split+$unwind when the pipeline causes JSON
            # escaping errors — get the raw string from MongoDB, then split here.
            # e.g. "Restaurants, Pizza, Italian Food" → three rows, one per category.
            col = op_spec.get("column", "")
            delimiter = op_spec.get("delimiter", ", ")
            output = op_spec.get("output", col)
            strip = op_spec.get("strip", True)  # strip whitespace from each element
            new_rows_se: list[dict] = []
            for row in rows:
                val = row.get(col)
                if val is None:
                    continue
                parts = str(val).split(delimiter)
                base = {k: v for k, v in row.items() if k != col}
                for part in parts:
                    elem = part.strip() if strip else part
                    if elem:
                        new_rows_se.append({**base, output: elem})
            rows = new_rows_se

        elif op == "parse_date":
            # Parse natural-language date strings via dateutil (fuzzy), format with strftime.
            col = op_spec.get("column", "")
            output = op_spec.get("output", col)
            fmt = op_spec.get("output_format", "%Y-%m-%d")
            try:
                from dateutil import parser as _du  # type: ignore
                _fuzzy_parse = _du.parse
            except ImportError:
                _fuzzy_parse = None
            for row in rows:
                text = str(row.get(col) or "")
                parsed = None
                if _fuzzy_parse:
                    try:
                        parsed = _fuzzy_parse(text, fuzzy=True).strftime(fmt)
                    except Exception:
                        parsed = None
                if parsed is None:
                    m = _re.search(r"\b((?:19|20)\d{2})\b", text)
                    if m:
                        import datetime as _dt
                        parsed = _dt.datetime(int(m.group(1)), 1, 1).strftime(fmt)
                row[output] = parsed

        elif op == "group_count":
            # Count rows per unique combination of group_by columns.
            group_by = op_spec.get("group_by", [])
            if isinstance(group_by, str):
                group_by = [group_by]
            output = op_spec.get("output", "count")
            counts: dict = _defaultdict(int)
            first_row: dict = {}
            for row in rows:
                key = tuple(row.get(c) for c in group_by)
                counts[key] += 1
                if key not in first_row:
                    first_row[key] = row
            new_rows = []
            for key, count in counts.items():
                new_row = {c: key[i] for i, c in enumerate(group_by)}
                for k, v in first_row[key].items():
                    if k not in new_row:
                        new_row[k] = v
                new_row[output] = count
                new_rows.append(new_row)
            rows = new_rows

        elif op == "group_sum":
            # Sum a numeric column per unique group_by combination.
            # Use after a cross-DB join when the same logical entity (e.g. same
            # song title+artist) has multiple rows (e.g. different track_id recordings).
            group_by = op_spec.get("group_by", [])
            if isinstance(group_by, str):
                group_by = [group_by]
            value_col = op_spec.get("value", op_spec.get("column", ""))
            output = op_spec.get("output", f"sum_{value_col}")
            sums: dict = _defaultdict(float)
            first_row: dict = {}
            for row in rows:
                key = tuple(row.get(c) for c in group_by)
                try:
                    sums[key] += float(row.get(value_col) or 0)
                except (TypeError, ValueError):
                    pass
                if key not in first_row:
                    first_row[key] = row
            new_rows = []
            for key, total in sums.items():
                new_row = {c: key[i] for i, c in enumerate(group_by)}
                for k, v in first_row[key].items():
                    if k not in new_row:
                        new_row[k] = v
                new_row[output] = round(total, 4)
                new_rows.append(new_row)
            rows = new_rows

        elif op == "round_down":
            # Floor a numeric column to the nearest multiple of `to`.
            # e.g. year 1987 → decade 1980 with to=10
            col = op_spec.get("column", "")
            to = int(op_spec.get("to", 10))
            output = op_spec.get("output", col)
            for row in rows:
                try:
                    v = int(float(row.get(col) or 0))
                    row[output] = (v // to) * to
                except (TypeError, ValueError):
                    row[output] = None

        elif op == "filter":
            # Keep rows where a column satisfies min/max/equals constraints.
            # Numeric comparisons for min/max; string equality for equals.
            column = op_spec.get("column", "")
            min_val = op_spec.get("min")
            max_val = op_spec.get("max")
            equals = op_spec.get("equals")
            filtered = []
            for row in rows:
                val = row.get(column)
                if val is None:
                    continue
                keep = True
                try:
                    fval = float(val)
                    if min_val is not None and fval < float(min_val):
                        keep = False
                    if max_val is not None and fval > float(max_val):
                        keep = False
                except (TypeError, ValueError):
                    fval = None
                if equals is not None and str(val) != str(equals):
                    keep = False
                if keep:
                    filtered.append(row)
            rows = filtered

        elif op == "group_avg":
            # Average a numeric column per unique group_by combination.
            # If the rows already contain a cnt/count/n column (from COUNT(*) in the sub-query),
            # also computes a weighted average and stores it as {output}_weighted so the caller
            # can use the true population average instead of the mean-of-means.
            # Specify "weight" explicitly to get a weighted (population) average.
            # Without "weight", computes a simple unweighted average.
            group_by = op_spec.get("group_by", [])
            if isinstance(group_by, str):
                group_by = [group_by]
            value_col = op_spec.get("value", op_spec.get("column", ""))
            output = op_spec.get("output", f"avg_{value_col}")
            weight_col: str | None = op_spec.get("weight")
            sums: dict = _defaultdict(float)
            counts_g: dict = _defaultdict(int)
            wsums: dict = _defaultdict(float)
            wtotals: dict = _defaultdict(float)
            for row in rows:
                key = tuple(row.get(c) for c in group_by)
                try:
                    v = float(row.get(value_col) or 0)
                    sums[key] += v
                    counts_g[key] += 1
                    if weight_col:
                        w = float(row.get(weight_col) or 1)
                        wsums[key] += v * w
                        wtotals[key] += w
                except (TypeError, ValueError):
                    pass
            new_rows = []
            for key in sums:
                new_row = {c: key[i] for i, c in enumerate(group_by)}
                avg = sums[key] / counts_g[key] if counts_g[key] > 0 else None
                new_row[f"{output}_n"] = counts_g[key]
                if weight_col:
                    wavg = wsums[key] / wtotals[key] if wtotals[key] > 0 else None
                    new_row[output] = round(wavg, 4) if wavg is not None else None
                else:
                    new_row[output] = round(avg, 4) if avg is not None else None
                new_rows.append(new_row)
            rows = new_rows

        elif op == "weighted_group_avg":
            # Weighted average per group: use when sub-query pre-aggregated per entity
            # (e.g. AVG(rating) per book) and you need the true average over all raw rows.
            # weight col holds the row count per entity (COUNT(*) from the sub-query).
            group_by = op_spec.get("group_by", [])
            if isinstance(group_by, str):
                group_by = [group_by]
            value_col = op_spec.get("value", op_spec.get("column", ""))
            weight_col = op_spec.get("weight", "count")
            output = op_spec.get("output", f"wavg_{value_col}")
            wsums: dict = _defaultdict(float)
            wtotals: dict = _defaultdict(float)
            for row in rows:
                key = tuple(row.get(c) for c in group_by)
                try:
                    v = float(row.get(value_col) or 0)
                    w = float(row.get(weight_col) or 1)
                    wsums[key] += v * w
                    wtotals[key] += w
                except (TypeError, ValueError):
                    pass
            new_rows = []
            for key in wsums:
                new_row = {c: key[i] for i, c in enumerate(group_by)}
                wavg = wsums[key] / wtotals[key] if wtotals[key] > 0 else None
                new_row[output] = round(wavg, 4) if wavg is not None else None
                new_rows.append(new_row)
            rows = new_rows

        elif op == "regex_extract":
            # Apply a regex to a column and extract a match into a new column.
            # By default, returns capture group 1 (or the full match if no groups).
            # ⚠ Case sensitivity: this op uses DOTALL only — [A-Z] matches ONLY uppercase.
            #   Add (?i) inline in the pattern if you need case-insensitive keywords.
            #   Do NOT use [A-Z] to require uppercase and also set ignore_case=True —
            #   that would make [A-Z] match lowercase too, capturing the wrong text.
            # Use (?s) flag in the pattern for DOTALL (. matches newlines).
            # Trick: prefix the pattern with "(?s).*" to find the LAST match in the string:
            #   "(?s).*\b(?:in|including|for|featuring)\s+([A-Z][^.]+)\.?\s*$"
            # This is useful for extracting a category list embedded at the END of prose.
            col = op_spec.get("column", "")
            pattern = op_spec.get("pattern", "")
            output = op_spec.get("output", col)
            group = int(op_spec.get("group", 1))
            flags = _re.DOTALL  # IGNORECASE is NOT applied — [A-Z] means uppercase only
            for row in rows:
                text = str(row.get(col) or "")
                m = _re.search(pattern, text, flags)
                if m:
                    try:
                        row[output] = m.group(group) if m.lastindex and group <= m.lastindex else m.group(0)
                    except IndexError:
                        row[output] = m.group(0)
                else:
                    row[output] = None

        elif op == "string_map":
            # Apply a regex substitution to each value in a column.
            # Use to clean up extracted text: strip leading "and ", trailing ".", quotes, etc.
            # Example: strip "and " prefix → pattern="^and ", replacement=""
            # Example: strip trailing period → pattern="\\.\\s*$", replacement=""
            col = op_spec.get("column", "")
            pattern = op_spec.get("pattern", "")
            replacement = op_spec.get("replacement", "")
            output = op_spec.get("output", col)
            flags = _re.IGNORECASE if op_spec.get("ignore_case", True) else 0
            for row in rows:
                text = str(row.get(col) or "")
                row[output] = _re.sub(pattern, replacement, text, flags=flags).strip()

        elif op == "string_filter":
            # Keep or drop rows based on whether a column matches a regex pattern.
            # mode: "keep" (default) — keep rows where the pattern matches.
            #       "drop" — drop rows where the pattern matches.
            # Use to filter out non-category fragments after string_explode:
            #   {"op": "string_filter", "column": "cat", "pattern": "^[a-z]|^[A-Z]{2}$"}  (drop)
            col = op_spec.get("column", "")
            pattern = op_spec.get("pattern", "")
            mode = op_spec.get("mode", "keep")
            flags = _re.IGNORECASE if op_spec.get("ignore_case", False) else 0
            filtered = []
            for row in rows:
                text = str(row.get(col) or "")
                matches = bool(_re.search(pattern, text, flags))
                if (mode == "keep" and matches) or (mode == "drop" and not matches):
                    filtered.append(row)
            rows = filtered

        elif op == "text_extract_list":
            # Find the comma-separated list of capitalized items inside prose text.
            # Scans all substrings that look like "Item1, Item2, Item3" (each starting uppercase),
            # and returns the one with the MOST comma-separated items — typically the category/tag
            # list rather than incidental "City, State" patterns elsewhere in the text.
            # No regex needed from the caller: just specify column and output.
            # min_items (default 2): only return a match if it has at least this many items.
            col = op_spec.get("column", "")
            output = op_spec.get("output", col)
            min_items = int(op_spec.get("min_items", 2))
            _cap_list_pat = _re.compile(r"[A-Z][^,\.\n!?;:]+(?:,\s*[A-Z][^,\.\n!?;:]+)*")
            for row in rows:
                text = str(row.get(col) or "")
                best: str | None = None
                best_n = 0
                for m in _cap_list_pat.finditer(text):
                    items = [x.strip() for x in m.group(0).split(",") if x.strip()]
                    if len(items) > best_n:
                        best_n = len(items)
                        best = m.group(0)
                row[output] = best if best_n >= min_items else None

        elif op == "string_after_last":
            # Return everything after the last occurrence of a fixed substring.
            # Use to isolate a list embedded in prose after a known keyword, e.g.:
            #   "...services in TypeA, TypeB." + separator="in " → "TypeA, TypeB."
            # No regex needed — just specify the literal separator string.
            # If the separator is not found, output is None (row is kept, value is None).
            col = op_spec.get("column", "")
            separator = op_spec.get("separator", "")
            output = op_spec.get("output", col)
            for row in rows:
                text = str(row.get(col) or "")
                idx = text.rfind(separator)
                row[output] = text[idx + len(separator):].strip() if idx >= 0 else None

        elif op == "string_after_any_last":
            # Like string_after_last, but tries a list of separators and picks the one
            # whose last occurrence is furthest into the text. Useful when different rows
            # use different keywords before the same type of list, e.g.:
            #   separators=["including ", "in ", "of ", "featuring "]
            # Finds whichever separator appears LAST overall, returns everything after it.
            col = op_spec.get("column", "")
            separators = op_spec.get("separators", [])
            output = op_spec.get("output", col)
            for row in rows:
                text = str(row.get(col) or "")
                best_idx = -1
                best_sep_len = 0
                for sep in separators:
                    idx = text.rfind(sep)
                    if idx > best_idx:
                        best_idx = idx
                        best_sep_len = len(sep)
                row[output] = text[best_idx + best_sep_len:].strip() if best_idx >= 0 else None

        elif op == "string_before_first":
            # Return everything before the first occurrence of a fixed substring.
            # Use to trim trailing punctuation or a continuing sentence, e.g.:
            #   "TypeA, TypeB, TypeC. Open daily." + separator="." → "TypeA, TypeB, TypeC"
            # If the separator is not found, the original value is returned unchanged.
            col = op_spec.get("column", "")
            separator = op_spec.get("separator", "")
            output = op_spec.get("output", col)
            for row in rows:
                text = str(row.get(col) or "")
                idx = text.find(separator)
                row[output] = text[:idx].strip() if idx >= 0 else text

        elif op == "string_strip_prefix":
            # Strip a leading pattern from a column without writing a regex.
            # preset="leading_digits" — strips a leading digit sequence + separator char:
            #   "007-Title" → "Title",  "12. Item" → "Item",  "3 Name" → "Name"
            # preset="before_dash" — strips everything up to and including the first " - ":
            #   "Artist - Title" → "Title"  (for records where artist is prefixed to title)
            col = op_spec.get("column", "")
            output = op_spec.get("output", col)
            preset = op_spec.get("preset", "leading_digits")
            if preset == "leading_digits":
                pat = _re.compile(r"^\d+[-.\s]+")
            elif preset == "before_dash":
                pat = _re.compile(r"^.+?\s+-\s+")
            else:
                pat = _re.compile(r"^\d+[-.\s]+")
            for row in rows:
                text = str(row.get(col) or "")
                row[output] = pat.sub("", text).strip()

        elif op == "string_drop_between":
            # Remove all text between matching open/close delimiter characters.
            # Use to strip parenthetical subtitles or qualifiers:
            #   "Title (Subtitle)" + open="(" close=")" → "Title"
            #   "Item [note]" + open="[" close="]" → "Item"
            col = op_spec.get("column", "")
            open_ch = op_spec.get("open", "(")
            close_ch = op_spec.get("close", ")")
            output = op_spec.get("output", col)
            pat = _re.compile(
                _re.escape(open_ch) + r"[^" + _re.escape(close_ch) + r"]*" + _re.escape(close_ch)
            )
            for row in rows:
                text = str(row.get(col) or "")
                row[output] = pat.sub("", text).strip()

        elif op == "compute_ema":
            # Exponential moving average per group sorted by a time column.
            # EMA[0] = value[0]; EMA[t] = α×value[t] + (1-α)×EMA[t-1]
            # fill_gaps=true (default): zero-fills missing integer steps.
            # summarize=true (default): returns one row per group at peak EMA step.
            group_col = op_spec.get("group_by", "")
            sort_col = op_spec.get("sort_by", "")
            value_col = op_spec.get("value_col", "count")
            alpha = float(op_spec.get("alpha", 0.2))
            output = op_spec.get("output", "ema")
            fill_gaps = op_spec.get("fill_gaps", True)
            summarize = op_spec.get("summarize", True)

            groups: dict = _defaultdict(list)
            for row in rows:
                groups[row.get(group_col)].append(row)

            result_rows: list[dict] = []
            for grp_val, grp_rows in groups.items():
                grp_rows = sorted(
                    grp_rows,
                    key=lambda r: (r.get(sort_col) is None, r.get(sort_col)),
                )
                if fill_gaps:
                    try:
                        steps = [int(r[sort_col]) for r in grp_rows if r.get(sort_col) is not None]
                        if steps:
                            lo, hi = min(steps), max(steps)
                            step_map = {int(r[sort_col]): r for r in grp_rows if r.get(sort_col) is not None}
                            grp_rows = [
                                step_map.get(s, {group_col: grp_val, sort_col: s, value_col: 0})
                                for s in range(lo, hi + 1)
                            ]
                    except (ValueError, TypeError):
                        pass

                ema_val: float | None = None
                best_row: dict | None = None
                best_ema: float | None = None
                annotated: list[dict] = []
                for row in grp_rows:
                    try:
                        val = float(row.get(value_col) or 0)
                    except (TypeError, ValueError):
                        val = 0.0
                    ema_val = val if ema_val is None else alpha * val + (1 - alpha) * ema_val
                    annotated_row = {**row, output: round(ema_val, 6)}
                    annotated.append(annotated_row)
                    if best_ema is None or ema_val > best_ema:
                        best_ema = ema_val
                        best_row = annotated_row

                if summarize:
                    if best_row is not None:
                        result_rows.append(best_row)
                else:
                    result_rows.extend(annotated)

            rows = result_rows

    return rows
