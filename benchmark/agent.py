"""
LLM agent loop for the DataAgentBench evaluation harness.

The agent receives a natural language question, calls DataBridge tools
(db_schema, db_query) to explore the schema and run SQL, then returns
a final text answer string for validation by each query's validate.py.

DAB's validate.py checks if a ground-truth value (e.g. "2020s") is present
in the agent's text output — so the agent must produce a human-readable
answer, not just raw rows.

Supported providers: "anthropic", "openai"
"""
from __future__ import annotations

import asyncio
import collections
import json
import logging
import math as _math
import re
import time as _time
from abc import ABC, abstractmethod
from typing import Any

log = logging.getLogger(__name__)

from databridge.engine import DataBridgeEngine


def _extract_answer(text: str) -> str:
    """Return content between <answer> tags if present, otherwise the full text."""
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return m.group(1).strip() if m else text


# ── Tool definitions ───────────────────────────────────────────────────────────

_TOOLS: list[dict] = [
    {
        "name": "db_schema",
        "description": (
            "Return schema for connected databases. "
            "Call with no arguments to discover all databases and tables. "
            "Pass database + table to get detailed column info including column data types."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "database": {
                    "type": "string",
                    "description": "Database alias (optional — omit for all).",
                },
                "table": {
                    "type": "string",
                    "description": "Table name to inspect (optional).",
                },
            },
        },
    },
    {
        "name": "db_query",
        "description": (
            "Execute a SQL SELECT query and return rows. "
            "Use this to retrieve data needed to answer the question.\n\n"
            "For CROSS-DATABASE queries or post-processing, pass a JSON spec:\n"
            '{"sub_queries": ['
            '{"db": "postgresql", "query": "SELECT id, title FROM title WHERE ...", "key": "pg"}, '
            '{"db": "sqlite", "query": "SELECT movie_id, person_id FROM cast_info WHERE ...", "key": "sq"}'
            '], "join_on": [["pg.id", "sq.movie_id"]], "transform": [...]}\n'
            "The engine runs sub-queries in parallel, merges on key columns, then applies transforms.\n\n"
            "transform ops (applied to merged rows in order):\n"
            '  {"op": "extract_number", "column": "col", "metric": "stars", "output": "stars"}'
            " — parse numeric metric from prose text; metric: \"stars\", \"forks\", \"issues\";"
            " handles all text format variants and comma-separated numbers automatically\n"
            '  {"op": "top_n_with_ties", "column": "col", "n": 5}'
            " — return top-N rows by column, including ALL rows tied at position N\n"
            '  {"op": "sort", "column": "col", "direction": "desc"} — sort rows\n'
            '  {"op": "cast_number", "column": "col", "output": "col_int"}'
            " — strip commas/spaces and cast to integer\n"
            '  {"op": "project_name_from_text", "column": "col", "output": "proj"}'
            " — extract owner/repo from 'The project X on GitHub' prose\n"
            '  {"op": "json_array_extract", "column": "col", "field": "code", "output": "codes"}'
            " — extract a named field from every element of a JSON array stored as text\n"
            '  {"op": "json_explode", "column": "codes", "output": "code"}'
            " — expand a list column into multiple rows (one row per list element; other columns copied)\n"
            '  {"op": "string_explode", "column": "tags_col", "delimiter": ", ", "output": "tag"}'
            " — split a string column by delimiter into multiple rows (one per element)."
            " Use when a field stores delimited values as a string and MongoDB $split+$unwind"
            " in the pipeline causes JSON escaping errors — project the raw string instead.\n"
            '  {"op": "parse_date", "column": "date_col", "output_format": "%Y", "output": "year"}'
            " — parse natural-language date strings (fuzzy) and format via strftime; default format: %Y-%m-%d\n"
            '  {"op": "group_count", "group_by": ["code", "year"], "output": "count"}'
            " — count rows per unique group, collapsing to one row per group\n"
            '  {"op": "group_sum", "group_by": ["name", "category"], "value": "metric", "output": "total"}'
            " — sum a numeric column per group; use after a cross-DB join when the same logical entity"
            " (e.g. same product or person) has multiple rows with different IDs in the metadata table\n"
            '  {"op": "round_down", "column": "year", "to": 10, "output": "decade"}'
            " — floor a numeric column to the nearest multiple of `to` (e.g. 1987 → 1980)\n"
            '  {"op": "filter", "column": "avg_score_n", "min": 10}'
            " — keep only rows satisfying a condition: min (>=), max (<=), equals (==)."
            " Use after group_avg to apply HAVING-style constraints (e.g. at least 10 entities per group).\n"
            '  {"op": "group_avg", "group_by": ["decade"], "value": "score", "output": "avg_score"}'
            " — average a numeric column per group, one row per group."
            " Always outputs {output}_n = count of entities in each group (use with filter for HAVING)."
            " By default computes a simple (unweighted) average."
            " To get a true population average, specify weight= with the COUNT(*) column name:"
            ' {"op": "group_avg", ..., "weight": "sq__cnt"}  ← weighted by review/event count.\n'
            '  {"op": "weighted_group_avg", "group_by": ["decade"], "value": "avg_score", "weight": "sq__cnt", "output": "true_avg"}'
            " — alias for group_avg with an explicit weight column.\n"
            '  {"op": "text_extract_list", "column": "desc", "output": "list_str"}'
            " — find the comma-separated list of capitalized items inside prose text."
            " Scans for all 'Item1, Item2, Item3...' sequences (each starting uppercase)"
            " and returns the one with the most items — correctly picks the category list"
            " over incidental 'City, State' patterns. No regex needed. Use min_items (default 2)"
            " to require a minimum number of items.\n"
            '  {"op": "string_after_last", "column": "text", "separator": "including ", "output": "list_str"}'
            " — return everything after the last occurrence of a fixed substring. No regex needed."
            " Use only when the text is short and the list is guaranteed to be the LAST thing in the text."
            " For multi-sentence prose descriptions (where a location or sentence may follow the list),"
            " use text_extract_list instead — rfind picks the wrong occurrence and gives location names."
            " Output is None when the separator is not found.\n"
            '  {"op": "string_after_any_last", "column": "text", "separators": ["including ", "in ", "of "], "output": "list_str"}'
            " — like string_after_last but tries multiple separators, picking the one whose last"
            " occurrence is furthest into the text. Use when different rows use different keywords"
            " before the list AND the list ends the text. Prefer text_extract_list for prose"
            " descriptions that continue after the list (e.g. address/location suffix).\n"
            '  {"op": "string_before_first", "column": "list_str", "separator": ".", "output": "list_str"}'
            " — return everything before the first occurrence of a fixed substring."
            " Use to trim a trailing period or sentence continuation.\n"
            '  {"op": "string_strip_prefix", "column": "title", "preset": "leading_digits"}'
            ' — strip a leading pattern without writing a regex. preset="leading_digits" removes a'
            ' leading digit sequence and separator (e.g. "007-Name" → "Name", "12. Item" → "Item").'
            ' preset="before_dash" removes everything up to the first " - " (e.g. "Prefix - Name" → "Name").\n'
            '  {"op": "string_drop_between", "column": "title", "open": "(", "close": ")"}'
            " — remove all text between matching delimiter characters, including the delimiters"
            ' (e.g. "Name (Qualifier)" → "Name").\n'
            '  {"op": "string_map", "column": "col", "pattern": "^and ", "replacement": ""}'
            " — regex substitution on each value; use for cleanup not covered by the above.\n"
            '  {"op": "string_filter", "column": "col", "pattern": ".{2,}", "mode": "keep"}'
            " — keep or drop rows where a column matches a regex (mode: 'keep' or 'drop').\n"
            '  {"op": "regex_extract", "column": "text", "pattern": "your-pattern", "output": "out"}'
            " — apply a regex and extract group 1 (or full match). Uses DOTALL only."
            " Use as a last resort when the higher-level ops cannot express what you need.\n"
            '  {"op": "compute_ema", "group_by": "code", "sort_by": "year", "value_col": "count",'
            ' "alpha": 0.2, "output": "ema"}'
            " — EMA per group sorted by time column; fill_gaps:true (default) zero-fills missing integer steps;"
            " summarize:true (default) returns one row per group at peak EMA step\n\n"
            "⚠ REQUIRED CORRECTNESS RULES — these transforms exist because SQL/regex alternatives"
            " produce systematically wrong answers:\n"
            "  1. Category extraction from prose descriptions → MUST use text_extract_list."
            " Direct string_explode, regex_extract, or string_after_last on a multi-sentence"
            " description picks city/state/address fragments instead of the category list.\n"
            "  2. Average ratings over pre-aggregated SQL rows → MUST use group_avg with weight=cnt."
            " group_avg without weight= gives avg-of-averages (biased; wrong number)."
            " Your SQL must return COUNT(*) AS cnt alongside AVG(score) AS avg_s;\n"
            "  3. Song/title search → MUST use normalize_text(col) LIKE '%concatwords%' (no spaces)."
            " Plain LIKE with spaces misses titles that have spacing variants (e.g. 'GetMe Bodied').\n"
            "  4. Metric totals per entity across databases → MUST join ALL fact rows with ALL metadata"
            " (no LIMIT) and use group_sum by entity name."
            " NEVER: aggregate the fact table + LIMIT 1, then look up the entity name —"
            " this silently ignores other IDs for the same entity that contribute more metric."
            " JOIN first (no LIMIT), group_sum after."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "SQL SELECT query string, or a JSON spec string for cross-database queries. "
                        "Alternatively omit this and use sub_queries + join_on + transform directly."
                    ),
                },
                "databases": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Database aliases to target (optional, for single-DB queries).",
                },
                "sub_queries": {
                    "type": "array",
                    "description": (
                        "Cross-database sub-queries. Each item: {db, query, key}. "
                        "For MongoDB sub-queries, 'query' may be an object "
                        "(e.g. {collection: 'name', pipeline: [...]}) instead of a string — "
                        "this avoids JSON escaping errors."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "db": {"type": "string"},
                            "query": {
                                "description": (
                                    "SQL string for SQL databases, or a MongoDB pipeline object "
                                    "{collection, pipeline} for MongoDB."
                                )
                            },
                            "key": {"type": "string"},
                        },
                        "required": ["db", "query", "key"],
                    },
                },
                "join_on": {
                    "type": "array",
                    "description": "Column pairs to join sub-query results on. Each item: [left_col, right_col].",
                    "items": {"type": "array", "items": {"type": "string"}},
                },
                "transform": {
                    "type": "array",
                    "description": "Post-processing transform ops applied to merged rows (see tool description).",
                    "items": {"type": "object"},
                },
            },
        },
    },
    {
        "name": "math_compute",
        "description": (
            "Fetch data from the database and compute a mathematical result — in one step.\n"
            "Supply a query (same format as db_query) and either an expression or a named operation.\n"
            "The tool runs the query internally; you never need to pass raw row values.\n\n"
            "── Data source (same interface as db_query) ──\n"
            "  Single-DB:   query='SELECT col AS x FROM t WHERE ...', databases=['db_alias']\n"
            "  Cross-DB:    sub_queries=[{db,query,key},...], join_on=[[left,right],...]\n"
            "  Inline data: rows=[...] or variables={...}  (for small/pre-computed values)\n\n"
            "── Mode 1: expression ──\n"
            "  expression: Python math string. Query results are auto-injected as:\n"
            "    rows — list of dicts (all result rows)\n"
            "    <col> — list of values for each column (named by SQL alias)\n"
            "  Plus: math.* (sqrt/log/exp/pi/...), sum/max/min/abs/round/len/sorted/enumerate/zip/range/int/float/str/dict/set\n"
            "  Example — std dev:  expression='math.sqrt(sum((x-sum(v)/len(v))**2 for x in v)/len(v))'\n"
            "                      query='SELECT value AS v FROM measurements', databases=['mydb']\n"
            "  Example — Pearson:  expression='(n*sxy-sx*sy)/math.sqrt((n*sxx-sx**2)*(n*syy-sy**2))'\n"
            "                      variables={'n':100,'sx':450,'sy':380,'sxy':1800,'sxx':2100,'syy':1520}\n\n"
            "── Mode 1 (expression) notes ──\n"
            "  Single-line: result returned directly.  Multi-line: assign final answer to `result`.\n"
            "  Multi-line example:\n"
            "    expression='totals = {r[\"code\"]: r[\"cnt\"] for r in rows}\\nresult = max(totals, key=totals.get)'\n\n"
            "── Mode 2: operation='chi_square' ──\n"
            "  Two input styles:\n"
            "  A) Pre-aggregated (preferred): query returns GROUP BY counts — one row per (row_dim, col_dim) pair.\n"
            "     Example: SELECT ht, mut, COUNT(*) AS cnt FROM ... GROUP BY ht, mut\n"
            "     → operation='chi_square', row_col='ht', col_col='mut', count_col='cnt', min_marginal=10\n"
            "  B) Raw observations: query returns one row per entity (no count column).\n"
            "     Omit count_col — each row is automatically treated as count=1.\n"
            "     ⚠ MUST include BOTH values of the column dimension (e.g., flag=1 AND flag=0 rows).\n"
            "  row_col, col_col: column names for the two categorical dimensions (defaults: row_label/col_label)\n"
            "  min_marginal: exclude categories whose marginal total is <= this value (default: 0)\n"
            "  Returns: {chi_square, grand_total, rows_included, cols_included, rows_excluded, cols_excluded}\n\n"
            "── Mode 3: operation='ema' ──\n"
            "  Compute exponential moving average per group over a time series.\n"
            "  Query should return rows with: a group column, a sort/time column (integer), a value column.\n"
            "  Parameters:\n"
            "    group_col  — column to group by (e.g. 'code', 'category')  [default: 'group']\n"
            "    sort_col   — integer time column to sort within group (e.g. 'year')  [default: 'year']\n"
            "    value_col  — numeric column to compute EMA over (e.g. 'count')  [default: 'count']\n"
            "    alpha      — smoothing factor 0–1 (default 0.3; higher = more weight on recent values)\n"
            "    fill_gaps  — zero-fill missing integer time steps (default true)\n"
            "    summarize  — return one row per group at peak EMA, sorted desc (default true)\n"
            "  Returns: {rows: [{group_col, sort_col, value_col, ema}, ...], row_count}\n"
            "  Example — top CPC codes by EMA of annual patent counts:\n"
            "    sub_queries=[{db:'patents',query:'SELECT code, year, COUNT(*) AS cnt FROM t GROUP BY code,year',key:'k'}]\n"
            "    operation='ema', group_col='code', sort_col='year', value_col='cnt', alpha=0.3"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "SQL SELECT string (single-DB). Use sub_queries instead for cross-DB.",
                },
                "databases": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Database alias(es) for a single-DB query.",
                },
                "sub_queries": {
                    "type": "array",
                    "description": "Cross-DB sub-queries — same format as db_query sub_queries.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "db": {"type": "string"},
                            "query": {},
                            "key": {"type": "string"},
                        },
                        "required": ["db", "query", "key"],
                    },
                },
                "join_on": {
                    "type": "array",
                    "description": "Column pairs to join sub-query results on.",
                    "items": {"type": "array", "items": {"type": "string"}},
                },
                "expression": {
                    "type": "string",
                    "description": "Python math expression (Mode 1). Query result columns are available by name.",
                },
                "variables": {
                    "type": "object",
                    "description": "Extra named scalar/list values for the expression (supplement query data).",
                },
                "operation": {
                    "type": "string",
                    "enum": ["chi_square", "ema"],
                    "description": "Named statistical operation. 'chi_square' for chi-square test; 'ema' for exponential moving average.",
                },
                "rows": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Inline row data (alternative to query, for small datasets).",
                },
                "row_col": {"type": "string", "description": "Row-dimension column name for chi_square. Default: 'row_label'."},
                "col_col": {"type": "string", "description": "Column-dimension column name for chi_square. Default: 'col_label'."},
                "count_col": {"type": "string", "description": "Count column name for chi_square. Default: 'count'."},
                "min_marginal": {
                    "type": "number",
                    "description": "chi_square: exclude categories with marginal total <= this. Default: 0.",
                },
                "group_col": {"type": "string", "description": "ema: column to group by (e.g. 'code', 'category'). Default: 'group'."},
                "sort_col": {"type": "string", "description": "ema: integer time column to sort within each group (e.g. 'year'). Default: 'year'."},
                "value_col": {"type": "string", "description": "ema: numeric column to compute EMA over (e.g. 'count'). Default: 'count'."},
                "alpha": {
                    "type": "number",
                    "description": "ema: smoothing factor 0–1 (default 0.3; higher = more weight on recent values).",
                },
                "fill_gaps": {
                    "type": "boolean",
                    "description": "ema: zero-fill missing integer time steps within each group. Default: true.",
                },
                "summarize": {
                    "type": "boolean",
                    "description": "ema: return one row per group at peak EMA, sorted descending. Default: true.",
                },
            },
        },
    },
]

_PROMPT_BASE = """\
You are a data analysis agent with access to database tools.

## CRITICAL — each table belongs to exactly one database

The schema lists MULTIPLE databases (postgresql, sqlite, duckdb, etc.).
SQL cannot join tables from different databases in a single query — it will always fail.

⚠ NEVER prefix table names with the database name in SQL. Reference tables by name only:
  WRONG: SELECT * FROM postgresql.items   ← fails with "relation does not exist"
  RIGHT: SELECT * FROM items
  Specify the target database via the `databases` parameter, not in the SQL.

⚠ In a spec sub-query, the `db` field MUST match the database that OWNS the table.
  If you see "no such table: X", check that `db` is set to the right database alias.

Use the JSON spec format for any query that spans multiple databases:
  {"sub_queries": [
    {"db": "postgresql", "query": "SELECT id, category FROM items WHERE ...", "key": "pg"},
    {"db": "sqlite",     "query": "SELECT item_id, score  FROM events WHERE ...", "key": "sq"}
  ], "join_on": [["pg.id", "sq.item_id"]]}

The engine runs each sub-query against its own database, then merges on the join keys.
Add a "transform" array to the spec for post-merge processing.

⚠ The "key" in each sub_query MUST match the prefix used in join_on — mismatch → zero rows.

⚠ After merging, RIGHT-side columns are prefixed with "{key}__":
  LEFT side = FIRST sub-query → columns as-is    RIGHT side = SECOND sub-query → prefixed
  Example: right-side key="sq", column "score" becomes "sq__score" in transforms.
  Tip: put the sub-query whose columns you need in transforms as the FIRST (left side).

**Aggregate INSIDE the spec, not in a separate follow-up query.**
After a cross-DB join the merged rows can't be re-queried via SQL. Compute aggregations either:
  (a) inside each sub-query's SQL before the join, or  (b) using transform ops on the merged result.

**Pre-join same-database tables in SQL first.**
If 3 tables are involved and 2 share a database, JOIN those 2 in one SQL sub-query before
the cross-DB spec join. Never use spec format for tables in the same database.

## Schema is pre-loaded

The full schema (tables, columns, types, foreign keys, cross-database join hints) is already in
your context. Do NOT call db_schema() for discovery — only call it if you need detailed column
stats (e.g. row counts, value distributions) for a specific table.

## Follow inline metric definitions exactly

When the question includes a section titled "## … Policy", "## Definition", or similar with
bullet points — that IS the algorithm to implement. Read it verbatim:
- Use the exact fields named (e.g. "company signed date on the contract" ≠ "opportunity close date").
- Apply the stated exclusions (e.g. "do not compute handle time for transferred cases").
- A simpler approximation almost always produces wrong results.

## Cross-database query strategy

Phase 1 — understand the join columns (1–2 queries, one per database):
  Sample the join columns on both sides to see actual values.
  If the schema shows a join hint between columns with DIFFERENT names, they likely encode the
  same ID in different formats (e.g. "prefix_42" vs "other_42"). Extract the integer:
    PostgreSQL: CAST(REGEXP_REPLACE(col_a, '[^0-9]', '', 'g') AS INTEGER) AS id_num
    SQLite:     CAST(REGEXP_REPLACE(col_b, '[^0-9]', '') AS INTEGER) AS id_num
    join_on:    [["pg.id_num", "sq.id_num"]]

Phase 2 — write and verify the spec query:
  After each spec result, challenge the row count before accepting it:
    "Given what I'm computing, does this count make sense?"
  If each sub-query returns thousands of rows on its own but the joined result has 0 or very
  few rows: ask whether 0/few is a plausible answer OR a join key mismatch. Check both:
    (a) Run COUNT(*) on each sub-query alone to confirm they have data.
    (b) Sample both key columns (LIMIT 20) and compare formats — if they don't look like the
        same values in different formats, the extraction is wrong. Fix it and retry.
  Do NOT simply accept a low count if the question implies joining large datasets.

When searching for entities, search across ALL relevant text fields with OR and synonyms:
  WHERE name ILIKE '%keyword%' OR description ILIKE '%keyword%' OR category ILIKE '%keyword%'
Do NOT manually inspect a sample to find matching entities — the dataset is larger than any sample.
Always sample a filter column before using it — values often differ from expectation.

**Categorical vs. text filtering**: when filtering by a type or status value, prefer a
dedicated categorical column over LIKE on a description/prose column — text search misses
records and silently under-counts. Check the schema for a categorical column first.

## Entity resolution — duplicate IDs for the same entity

When a table may have duplicate entries for the same real-world entity (different IDs, same person
or product): search across MULTIPLE fields with OR, collect ALL matching IDs, pass them downstream
(WHERE id IN (id1, id2, ...)). For ranking use LIMIT 200 then re-sum across all IDs.

## Relationship type discriminators

Association tables often have a type/role column distinguishing different relationships.
⚠ Always check: SELECT DISTINCT type_col FROM table LIMIT 20 — then filter to the correct type.
Joining without this filter silently mixes unrelated rows and produces wrong results.

## Lookup tables — always resolve IDs before filtering

NEVER hardcode a numeric ID like `WHERE status_id = 4` without first querying the lookup table.
Hardcoded IDs silently return 0 rows when wrong.

## normalize_text() — accent- and whitespace-insensitive matching

`normalize_text(col)` strips diacritics, lowercases, and removes ALL whitespace.
Use for name/title filtering: WHERE normalize_text(title) LIKE '%bluemoon%'
(matches "Blue Moon", "BlueMoon", "007-Blue Moon" — all with one LIKE pattern)

## Alphabetically first / MIN ordering

SQL sort follows ASCII: special chars < digits < letters. The alphabetically first value may
start with '!' — never add WHERE name LIKE '[A-Za-z]%'. Use MIN() or ORDER BY name ASC LIMIT 1.
When two tables both have a "name" column, qualify explicitly (e.g. MIN(n.name) vs MIN(ak.name)).

## Finding MIN from a filtered set

NEVER look up one record's value and assume it is the minimum.
Always: SELECT MIN(name) FROM table WHERE id IN (all_matching_ids)

## Metric totals, grouping, and transforms

**group_sum — total metric per entity across databases:**
Join ALL fact rows with ALL metadata (no LIMIT in either sub-query), then group_sum by name.
⚠ NEVER LIMIT the fact sub-query before the join — silently drops other IDs for the same entity.
  fact DB:  SELECT entity_id, SUM(metric) AS total FROM events GROUP BY entity_id        ← no LIMIT
  meta DB:  SELECT entity_id, name FROM entities
  join_on:  [["fk.entity_id", "mk.entity_id"]]
  transforms: [{"op": "group_sum", "group_by": ["name"], "value": "fk__total", "output": "grand_total"},
               {"op": "sort", "column": "grand_total", "direction": "desc"}]

**Normalization — always in transforms, never in SQL CTEs:**
  {"op": "string_strip_prefix", "column": "name_col", "preset": "leading_digits"}
  {"op": "string_drop_between", "column": "name_col", "open": "(", "close": ")"}

**Computing group averages correctly:**
SQL MUST include COUNT(*) AS cnt alongside AVG(score) AS avg_s:
  → {"op": "group_avg", "group_by": ["decade"], "value": "sq__avg_s", "weight": "sq__cnt", "output": "avg"}
  AVG without weight gives biased avg-of-averages.

**Deduplication — top entity ≠ top ID:**
The same real-world entity can appear under multiple IDs. Before concluding "highest total":
  1. Sample the metadata table to check for duplicate entries (SELECT DISTINCT on name/title columns).
  2. If duplicates exist, derive the grouping rule from the data — which columns are stable
     across duplicates? Use those as the group-by key.
The correct aggregation order:
  WRONG: aggregate fact table → LIMIT 1 → look up that one ID's name  ← misses other IDs for same entity
  RIGHT: join ALL fact rows with ALL metadata (no LIMIT) → normalize name → group_sum by name → sort
Normalization before grouping: strip numeric prefixes (string_strip_prefix preset "leading_digits"),
drop parenthetical variants (string_drop_between open "(" close ")"), then group_sum.

**Returning the single best vs. all groups:**
"Most frequent/highest/lowest" → ONE result (use LIMIT 1).
"Best for each group" → one row per group.
For top-N ties: use `top_n_with_ties` transform to include ALL rows tied at position N.
"Which/what [plural noun] had more/at least/exceeded …" → ALL qualifying items (threshold condition);
  do NOT use LIMIT 1. Order by the metric DESC and return every row that passes the threshold.

**Prefer transforms over SQL** when a transform op covers the use case:
  extract_number    — numbers from prose text ("1.2k forks", "4 stars")
  compute_ema       — exponential moving averages (not recursive CTEs)
  parse_date        — natural-language date strings ("dated 5th March 2019")
  text_extract_list — comma-separated category list from prose descriptions
  group_avg + weight — true population average over pre-aggregated SQL rows
  group_sum         — total metric across duplicate entity IDs after a join
SQL is for filtering, joining, and raw aggregation on structured columns.

## Row limit and unstructured text

Results are capped at 1,000 rows per query — if a result returns exactly 1,000 rows, it was
truncated; rewrite with tighter filters or aggregation. Cap applies per sub-query too.

For unstructured text columns, always sample 5–10 rows first (SELECT col FROM table LIMIT 10).
Use LIKE patterns matching the exact format you observed. Use extract_number or parse_date
transforms rather than SQL regex for numbers or dates embedded in prose.

## Statistical measures

Use SQL (AVG, SUM, COUNT, MIN, MAX) for simple aggregates — these run efficiently server-side.
Use `math_compute` for formulas SQL cannot express (EMA, chi-square, Pearson, std dev, log transforms).
`math_compute` fetches data and computes in ONE call — never pass raw rows through your context.

**Chi-square** — use operation="chi_square" (handles contingency table reshaping automatically):
  math_compute(
    sub_queries=[{"db":"pg","query":"SELECT cat, CASE WHEN flag=1 THEN 'yes' ELSE 'no' END AS grp, COUNT(*) AS cnt FROM entities GROUP BY cat, grp","key":"k"}],
    operation="chi_square", row_col="cat", col_col="grp", count_col="cnt", min_marginal=10
  )
  ⚠ NEVER compute chi-square via expression mode — reshaping a flat result into a contingency
  matrix cannot be done in one expression. Always use operation="chi_square".
  ⚠ The query MUST include ALL rows for BOTH values of the column dimension — if you only query
  rows where the flag is present, the contingency table has one column and chi-square cannot be
  computed. Use a LEFT JOIN or CASE WHEN to mark presence/absence for every entity.

**EMA (exponential moving average)** — use operation="ema":
  math_compute(
    sub_queries=[{"db":"mydb","query":"SELECT code, year, COUNT(*) AS cnt FROM t GROUP BY code, year","key":"k"}],
    operation="ema", group_col="code", sort_col="year", value_col="cnt", alpha=0.3
  )
  Returns rows sorted by peak EMA descending (summarize=True), each with an `ema` column.
  Use this whenever a question asks for EMA, exponential moving average, or trend weighting over time.
  ⚠ NEVER compute EMA via expression mode — the multi-group iterative logic is complex.
    Always use operation="ema".

**Arbitrary formula** — supply a query + expression; each SQL alias becomes a list variable:
  math_compute(
    query="SELECT value AS v FROM measurements", databases=["mydb"],
    expression="math.sqrt(sum((x - sum(v)/len(v))**2 for x in v) / len(v))"
  )
  Scalar pre-aggregates (from SQL: SUM, COUNT, AVG) work as inline variables:
  math_compute(expression="(n*sxy - sx*sy)/math.sqrt((n*sxx-sx**2)*(n*syy-sy**2))",
               variables={"n":100,"sx":450,"sy":380,"sxy":1800,"sxx":2100,"syy":1520})
  Available: math.*, sum/max/min/abs/round/len/sorted/enumerate/zip/range/int/float,
             `rows` (list of dicts), each SQL alias as a list.
  Single-line: use a single eval()-compatible expression.
  Multi-line: write statements on separate lines and assign the final answer to `result`.
    Example: expression='totals={r["code"]:r["cnt"] for r in rows}\nresult=max(totals,key=totals.get)'

## Filtering by quality / reliability

When a query asks for "reliable", "valid", "high-confidence", or "trusted" records:
sample the FILTER, QUALITY, or STATUS column first (SELECT DISTINCT filter_col FROM t LIMIT 20)
to identify what value means "reliable" (commonly 'PASS', 'PASS_FILTER', 'verified', etc.).
Never skip this check — the wrong filter value silently changes the entire result.

## Log transforms for count or expression data

When a query asks for log-transformed values (log10, log2, ln) of count or expression data,
use LOG10(value + 1) / LOG2(value + 1) / LN(value + 1) rather than LOG10(value).
The +1 offset (log1p) prevents errors on zero values and is the standard convention for
normalized count or expression columns that may contain zeros.

## Classification codes and bracket-notation nulls

Tables often have BOTH a human-readable name column AND a numeric/alphanumeric code column for the
same concept (e.g., a numeric/alphanumeric code column alongside a human-readable name column).
When a query mentions filtering out entries "enclosed in square brackets" or "with bracket notation"
(e.g., [Not Available], [Discrepancy], [Not Reported]), that bracket pattern appears in code/
classification columns. Sample BOTH columns, filter out bracket-enclosed entries from the code
column, and GROUP BY the code column — not the name column — when the question concerns
classification categories.

## Location filtering

When filtering by city, always match city AND state together — state alone matches multiple cities.
If location is embedded in a text field, match the compound pattern "City, StateCode" rather than
just the state abbreviation. The city name must appear explicitly in the filter.

## Record and document identification

For policy/compliance questions, fetch the FULL text of ALL candidate documents
(title + body/answer/summary) and match content against entity data before deciding.
Keyword-matching on titles alone picks the wrong article — read the actual policy text.

## Output format

⚠ CRITICAL: Wrap your final answer in `<answer>` and `</answer>` tags.
Inside the tags: data values only — no intro sentences, no labels, no explanation.
You may reason freely before the tags, but only the tagged content is evaluated.
Do NOT wrap values in backticks, bold, or quotes.
For "most frequent", "most common", "highest", "lowest" — return exactly ONE result.

**Output the exact stored value — never substitute your own labels:**
When the answer is a code, ID, or short identifier stored in the database, output it exactly as
it appears — do not replace it with a human-readable label you know from domain knowledge.
When a table has both a code column and a name column for the same concept, GROUP BY and report
the column that matches what the question asks for (code vs. name); sampling both reveals which is which.

**Record identity:** When the question asks "which record/article/document", the record's ID or
title IS the answer — output it directly. Only append IDs when the question asks for both.

Example (multi-value answer):
<answer>
Widget A
Widget B
Widget C
</answer>

Only use SELECT. Never use INSERT, UPDATE, DELETE, DROP, or CREATE.

## Response length

Keep intermediate reasoning short — one or two sentences before each tool call is enough.
Do NOT write long analysis paragraphs between tool calls; the evaluation only reads `<answer>` tags.
When the final answer is a list of values (IDs, names, codes), output ALL values inside `<answer>` tags
without preamble. If your answer has more than 50 items, verify the question actually asks for all of
them before listing every one.\
"""

_PROMPT_POSTGRES = """
## PostgreSQL notes

Double-quote mixed-case column names everywhere: SELECT "titleFull" FROM t
(PostgreSQL folds unquoted identifiers to lowercase — omitting quotes silently reads the wrong column.)
Supports REGEXP_REPLACE, ILIKE, SUBSTRING(col FROM 'pattern'), window functions, col::text / CAST(x AS TEXT).
Use REGEXP_REPLACE(col::text, ...) when the column is INTEGER.

**INTEGER vs TEXT**: Never compare an INTEGER column to a TEXT literal — raises
"operator does not exist: integer = text". Cast: col::text = 'value', or col = 1234.

**Empty-string columns**: Columns may store '' instead of NULL. Bare casts fail with
"invalid input syntax for type integer: ''". Use NULLIF before casting:
  NULLIF(col, '')::INTEGER   — returns NULL on empty string (safe)
  NULLIF(col, '')::FLOAT     — same for floats
Apply this whenever you cast text columns to numeric types.

**Year or decade from prose text** — use parse_date + round_down transforms, not SQL regex
(embedded ISBNs, page counts, dimensions cause false matches):
  {"op": "parse_date", "column": "text_col", "output_format": "%Y", "output": "year"}
  {"op": "round_down", "column": "year", "to": 10, "output": "decade"}

**JSON/JSONB columns**:
  col->>'key'                    — extract string value
  jsonb_array_elements_text(col) — unnest JSON array into rows

**Spec routing — PostgreSQL tables in DuckDB sub-queries**:
If a spec returns 'Catalog Error: Table does not exist' or 'No files found' from DuckDB for a
table that lives in PostgreSQL, do NOT switch to DuckDB's postgres_scanner or postgres_query()
— those require localhost credentials that are not available. Fix the sub-query's `db` field
to the correct PostgreSQL alias instead.\
"""

_PROMPT_SQLITE = """
## SQLite notes

No ILIKE (use LIKE). REGEXP_REPLACE IS available with 3 or 4 args:
  REGEXP_REPLACE(col, '[^0-9]', '')       — strips non-digits (use for ID normalization)
  REGEXP_REPLACE(col, '[^0-9]', '', 'g') — same with 'g' flag
⚠ Do NOT use REGEXP_REPLACE to extract years from prose text — use the parse_date transform.
No cross-database joins in SQL — always use the spec format.

**VALUES alias workaround**:
  WRONG: SELECT MIN(col) FROM (VALUES ('a'),('b')) AS t(col)
  RIGHT: SELECT MIN(col) FROM (SELECT 'a' AS col UNION ALL SELECT 'b' AS col)

**Date columns stored as Unix timestamps (integer seconds since epoch)**:
  WRONG: strftime('%Y', col)                             — treats col as a string, returns NULL
  RIGHT: strftime('%Y', col, 'unixepoch')                — correctly converts epoch to year
  RIGHT: col >= CAST(strftime('%s', '2020-01-01') AS INTEGER)   — filter from a date
  Always sample the column first to confirm it holds epoch seconds vs. a date string.

**Large table performance**: Filter to a small ID set before joining large fact tables:
  Step 1: SELECT DISTINCT fk FROM large_table WHERE category='X' LIMIT 500
  Step 2: SELECT MIN(name) FROM lookup WHERE id IN (<step 1 results>)

**EMA** — Use compute_ema transform, NOT a recursive CTE (recursive CTEs seed incorrectly):
  {"op": "compute_ema", "group_by": "grp", "sort_by": "yr", "value_col": "cnt", "alpha": 0.2}\
"""

_PROMPT_DUCKDB = """
## DuckDB notes

**Date columns: always sample first, then build the format string from what you see.**
  SELECT col FROM t LIMIT 5   ← look at actual values before writing any strptime call
  Use TRY_STRPTIME (not strptime) so non-matching rows return NULL instead of erroring.
  To extract year: YEAR(TRY_STRPTIME(col, '<exact-format>'))
  Natural-language strings ("January 02, 1980"): use the parse_date transform instead.

**Always use TRY_CAST — never bare CAST — in DuckDB.**
  Bare CAST raises "Conversion Error: Could not convert string '' to INT32" on empty strings.
  Use TRY_CAST in every SELECT, ORDER BY, and WHERE that casts a column:
    TRY_CAST(col AS INTEGER)   — returns NULL on empty string or invalid value

**Time/duration strings** ('0:00', '1:23', '90:00') are MM:SS or HH:MM — DuckDB errors on bare CAST.
  To convert to seconds: CAST(split_part(col, ':', 1) AS INT) * 60 + CAST(split_part(col, ':', 2) AS INT)

**Text columns that embed numbers** ('0 stars', '1.2k forks'): use the extract_number transform.

**Joining a prose table to a structured table in the same database:**
When one table has an entity name as a structured column and another embeds the same name in a
prose/description column, join them in SQL using LIKE rather than a cross-DB spec:
  JOIN prose_table p ON p.description_col LIKE '%' || s.name_col || '%'

**JSON columns:** extract fields with json_extract():
  json_extract(col, '$.field')   → scalar value (string/int/bool)
  Latest record per group = WHERE json_extract(col, '$.rank_field') = (SELECT MAX(...) ...)

**Subqueries must have an alias** — DuckDB requires every subquery in FROM to be named:
  SELECT col FROM (SELECT col FROM t) AS sub   ← required alias
  Without it DuckDB raises "Referenced column not found in FROM clause".

**Table names with special characters** (e.g. `#`, `-`, `.`): DuckDB requires double-quotes.
  FROM "CARR#"  — not  FROM CARR#  (which causes a parse error)
  If a query fails with "parse error" or "Expected table name", re-run with the table name double-quoted.

**Many-table databases**: when the schema shows "⚠ N tables (all share the same structure)",
all table names are listed — reference them directly in SQL.\
"""

_PROMPT_MONGODB = """
## MongoDB

Pass a standalone MongoDB query as a JSON string:
  {"collection": "articles", "pipeline": [{"$match": {"cat": "Sports"}}, {"$limit": 10}]}

**In a cross-DB spec, pass the MongoDB sub-query as a NATIVE OBJECT — never as an escaped string.**
Escaped strings cause "Expecting ',' delimiter" parse errors. Use the sub_queries parameter:

  RIGHT — sub_queries parameter with MongoDB query as a native dict:
    db_query(
      sub_queries=[
        {"db": "mongodb", "query": {"collection": "items", "pipeline": [...]}, "key": "mg"},
        {"db": "sqlite",  "query": "SELECT id_num, SUM(amount) AS total FROM events GROUP BY id_num", "key": "sq"}
      ],
      join_on=[["mg.id_num", "sq.id_num"]]
    )

⚠ Pipeline JSON must be strict: double quotes only, no trailing commas.
If you get parse errors: run the MongoDB query standalone first to confirm it works, then join.

**Attribute fields may contain Python dict strings** (e.g. "{'free': True}") — NOT JSON objects.
Use $regex to match: {"$match": {"attributes.WiFi": {"$regex": "free", "$options": "i"}}}
For boolean attributes: {"$match": {"attributes.BusinessParking": {"$regex": "True"}}}

**List fields stored as strings** — sample first. If it's a concatenated string ("tag1, tag2"):
Use the string_explode transform (NOT $split + $unwind in MongoDB — causes JSON parse errors):
  transforms: [{"op": "string_explode", "column": "tags_col", "delimiter": ", ", "output": "tag"},
               {"op": "group_count", "group_by": ["tag"], "output": "count"},
               {"op": "sort", "column": "count", "direction": "desc"}]

**Cross-DB joins with MongoDB: extract numeric ID on BOTH sides.**
  MongoDB: {"$addFields": {"id_num": {"$toInt": {"$arrayElemAt": [{"$split": ["$item_id", "_"]}, -1]}}}}
  SQL:     CAST(REGEXP_REPLACE(ref_col, '[^0-9]', '', 'g') AS INTEGER) AS id_num
  join_on: [["mg.id_num", "sq.id_num"]]
⚠ If one side is string and the other integer, the join returns 0 rows.

**3-table cross-DB queries (2 SQL tables + 1 MongoDB):**
1. SQL sub-query: JOIN the 2 SQL tables in one query (include COUNT(*) AS cnt for avg ratings).
2. MongoDB sub-query: project the fields you need.
3. Cross-DB join on the numeric ID extracted from both sides.
4. Transforms: extract labels from prose (text_extract_list), then group_sum or group_avg.
Tip: put the MongoDB sub-query FIRST (left side) when you need its text columns in transforms.

**Extracting a comma-separated list from prose text**: use text_extract_list (NOT string_explode
on the raw description — it produces address fragments). See 3-table example above.

For average ratings after extraction: use group_avg with explicit weight="key__cnt".

**Simple aggregation — use the pipeline directly, not math_compute:**
For AVG/COUNT/SUM over a filtered set, compute in the pipeline itself:
  {"collection": "businesses", "pipeline": [
    {"$match": {"description": {"$regex": "Indianapolis, IN"}}},
    {"$group": {"_id": null, "avg_rating": {"$avg": "$stars"}, "count": {"$sum": 1}}}
  ]}
Use math_compute only for formulas MongoDB cannot express (EMA, chi-square, Pearson, etc.).

**Category and text filtering — always use case-insensitive regex:**
When matching a category or keyword in a text/description field:
  {"$regex": "restaurants", "$options": "i"}   ← case-insensitive (correct)
  {"$regex": "Restaurants"}                    ← case-sensitive (misses lowercase occurrences)
Case-sensitive matching silently under-counts entries where the term appears in a different case.\
"""


# ── Kimi-specific compact prompt ─────────────────────────────────────────────
# (Removed — merged into _PROMPT_BASE above. All providers now use _build_system_prompt.)


def _build_kimi_system_prompt(db_types: set[str]) -> str:
    """Alias kept for backward compatibility — now identical to _build_system_prompt."""
    return _build_system_prompt(db_types)


def _build_system_prompt(db_types: set[str]) -> str:
    """Assemble a system prompt containing only sections relevant to the DB types in use."""
    parts = [_PROMPT_BASE]
    if "postgresql" in db_types:
        parts.append(_PROMPT_POSTGRES)
    if "sqlite" in db_types:
        parts.append(_PROMPT_SQLITE)
    if "duckdb" in db_types:
        parts.append(_PROMPT_DUCKDB)
    if "mongodb" in db_types:
        parts.append(_PROMPT_MONGODB)
    return "\n".join(parts)


class _TooLong:
    """Sentinel returned by _call() when the prompt is too long, carrying the excess token count."""
    __slots__ = ("over_tokens",)
    def __init__(self, over_tokens: int):
        self.over_tokens = over_tokens


def _smart_drop(messages: list[dict], over_tokens: int) -> None:
    """
    Drop the earliest non-essential assistant+tool turn from messages (in-place).

    Anthropic format: [user(q), asst, user(tool_result), asst, user(tool_result), ...]
    OpenAI format:    [system, user(q), asst, tool, tool, asst, tool, ...]
    Works for both: finds the first "assistant" message at index ≥ 1 (Anthropic) or ≥ 2 (OpenAI)
    and removes it together with the immediately following same-role-cluster messages.
    """
    # Detect format: if first message is "system", it's OpenAI; otherwise Anthropic.
    start_idx = 2 if messages and messages[0].get("role") == "system" else 1

    drop_start = None
    for i, m in enumerate(messages):
        if i < start_idx:
            continue
        if m.get("role") == "assistant":
            drop_start = i
            break

    if drop_start is None:
        return  # nothing to drop

    # Drop the assistant message + all following tool/user messages until the next assistant.
    drop_end = drop_start + 1
    while drop_end < len(messages) and messages[drop_end].get("role") in ("tool", "user"):
        # For Anthropic, "user" holds tool_result; for OpenAI, "tool" holds results.
        # Stop if we hit the next question-style user message (non-tool-result).
        if messages[drop_end].get("role") == "user":
            content = messages[drop_end].get("content", "")
            # Anthropic tool results have content as a list with type "tool_result"
            if isinstance(content, list) and any(
                (isinstance(c, dict) and c.get("type") == "tool_result") for c in content
            ):
                drop_end += 1
                continue
            break  # plain user message — stop here
        drop_end += 1

    del messages[drop_start:drop_end]


# ── Tool executor ──────────────────────────────────────────────────────────────

_MAX_TOOL_RESULT_CHARS = 40_000  # ~10K tokens; keeps 20-turn context well under 200K

def _json_serialise(obj: object) -> object:
    """Convert types that json.dumps can't handle (e.g. Decimal from PostgreSQL AVG/SUM)."""
    import decimal
    import datetime
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


_MATH_SANDBOX: dict = {
    "math": _math,
    "sum": sum, "max": max, "min": min, "abs": abs,
    "round": round, "len": len, "sorted": sorted,
    "enumerate": enumerate, "zip": zip, "range": range,
    "list": list, "dict": dict, "set": set,
    "int": int, "float": float, "str": str,
    "True": True, "False": False, "None": None,
}


def _safe_eval(expression: str, variables: dict | None = None) -> dict:
    """Evaluate a Python math expression or multi-line code block in a restricted sandbox.

    Single-line: eval() — the expression result is returned directly.
    Multi-line:  exec() — the code MUST assign the final answer to a variable named `result`.
    """
    # Merge into globals (not locals) so comprehensions can see variables.
    # Comprehensions create their own scope that only inherits from globals.
    globs: dict = {"__builtins__": {}, **_MATH_SANDBOX}
    if variables:
        globs.update(variables)

    is_multiline = "\n" in expression.strip()

    if is_multiline:
        local_vars: dict = {}
        try:
            exec(compile(expression, "<math_compute>", "exec"), globs, local_vars)
        except Exception as e:
            return {"error": str(e)}
        if "result" not in local_vars:
            return {
                "error": (
                    "Multi-line code block must assign the final answer to a variable named 'result'. "
                    "Example: result = sum(r['count'] for r in rows if r['type'] == 'A')"
                )
            }
        return {"result": local_vars["result"]}

    try:
        result = eval(compile(expression, "<math_compute>", "eval"), globs)
        return {"result": result}
    except SyntaxError as e:
        return {
            "error": (
                f"SyntaxError: {e}. "
                "For a single expression: use nested constructs (sum(...), [f(x) for x in v])."
                " For multi-statement code: write each statement on a separate line and assign"
                " the final answer to `result` (e.g. result = my_computed_value)."
                " For chi-square, use operation='chi_square' instead."
                " For EMA, use operation='ema' instead."
            )
        }
    except Exception as e:
        return {"error": str(e)}


def _compute_chi_square(
    rows: list[dict],
    row_col: str = "row_label",
    col_col: str = "col_label",
    count_col: str = "count",
    min_marginal: float = 0,
) -> dict:
    """Compute chi-square from contingency table rows (one dict per cell)."""
    from collections import defaultdict

    counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for r in rows:
        rl = str(r.get(row_col, ""))
        cl = str(r.get(col_col, ""))
        raw = r.get(count_col)
        cnt = 1.0 if raw is None else float(raw or 0)
        counts[rl][cl] += cnt

    all_row_labels: set[str] = set(counts.keys())
    all_col_labels: set[str] = set()
    for rc in counts.values():
        all_col_labels.update(rc.keys())

    row_totals = {r: sum(counts[r].values()) for r in all_row_labels}
    col_totals = {c: sum(counts[r].get(c, 0) for r in all_row_labels) for c in all_col_labels}

    valid_rows = {r for r, t in row_totals.items() if t > min_marginal}
    valid_cols = {c for c, t in col_totals.items() if t > min_marginal}

    row_totals_f = {r: sum(counts[r].get(c, 0) for c in valid_cols) for r in valid_rows}
    col_totals_f = {c: sum(counts[r].get(c, 0) for r in valid_rows) for c in valid_cols}
    grand_total = sum(row_totals_f.values())

    if grand_total == 0:
        return {
            "error": (
                f"Grand total is 0 after filtering — no valid cells remain. "
                f"Ensure: (1) count_col='{count_col}' matches your query column name "
                f"(if missing, each row is treated as count=1), "
                f"(2) query returns rows for ALL category combinations (both values of each dimension), "
                f"(3) count values are not all null/zero."
            )
        }

    chi2 = 0.0
    for r in valid_rows:
        for c in valid_cols:
            o = counts[r].get(c, 0)
            e = row_totals_f[r] * col_totals_f[c] / grand_total
            if e > 0:
                chi2 += (o - e) ** 2 / e

    return {
        "chi_square": chi2,
        "grand_total": int(grand_total),
        "rows_included": len(valid_rows),
        "cols_included": len(valid_cols),
        "rows_excluded": len(all_row_labels) - len(valid_rows),
        "cols_excluded": len(all_col_labels) - len(valid_cols),
    }


def _compute_ema(
    rows: list[dict],
    group_col: str,
    sort_col: str,
    value_col: str,
    alpha: float = 0.3,
    fill_gaps: bool = True,
    summarize: bool = True,
) -> dict:
    """Compute EMA per group over a sorted time column.

    Returns one row per group at peak EMA (summarize=True) or the full EMA series.
    """
    from collections import defaultdict

    if not rows:
        return {"rows": [], "row_count": 0}

    groups: dict = defaultdict(list)
    for r in rows:
        key = str(r.get(group_col, ""))
        groups[key].append(r)

    results: list[dict] = []

    for group_key, group_rows in groups.items():
        try:
            sorted_rows = sorted(
                group_rows,
                key=lambda r: (r.get(sort_col) is None, r.get(sort_col) or 0),
            )
        except TypeError:
            sorted_rows = group_rows

        if fill_gaps:
            try:
                min_t = int(sorted_rows[0].get(sort_col) or 0)
                max_t = int(sorted_rows[-1].get(sort_col) or 0)
                val_by_t = {
                    int(r.get(sort_col) or 0): float(r.get(value_col) or 0)
                    for r in sorted_rows
                }
                sorted_rows = [
                    {group_col: group_key, sort_col: t, value_col: val_by_t.get(t, 0.0)}
                    for t in range(min_t, max_t + 1)
                ]
            except (TypeError, ValueError):
                pass

        ema: float | None = None
        best_ema: float = float("-inf")
        best_row: dict | None = None
        ema_series: list[dict] = []

        for r in sorted_rows:
            try:
                val = float(r.get(value_col) or 0)
            except (TypeError, ValueError):
                val = 0.0
            ema = val if ema is None else alpha * val + (1 - alpha) * ema
            row_with_ema = {**r, "ema": round(ema, 6)}
            if ema > best_ema:
                best_ema = ema
                best_row = row_with_ema
            ema_series.append(row_with_ema)

        if summarize:
            if best_row is not None:
                results.append(best_row)
        else:
            results.extend(ema_series)

    if summarize:
        results.sort(key=lambda r: r.get("ema", 0), reverse=True)

    return {"rows": results, "row_count": len(results)}


class ToolExecutor:
    def __init__(self, engine: DataBridgeEngine, session_id: str) -> None:
        self._engine = engine
        self._session_id = session_id

    def reset(self) -> None:
        """Clear per-question dedup state. Call before each new question."""
        self._engine.reset_session(self._session_id)

    async def call(self, name: str, arguments: dict, turn: int = 0) -> str:
        try:
            if name == "db_schema":
                db = arguments.get("database") or None
                tbl = arguments.get("table") or None
                result_dict = await self._engine.schema(db, tbl)
                result = json.dumps(result_dict)
                if log.isEnabledFor(logging.DEBUG):
                    if tbl:
                        ncols = len(result_dict.get(db or "", {}).get("columns", {}))
                        log.debug("[t%d] schema %s.%s → %d cols", turn, db, tbl, ncols)
                    else:
                        summary = {k: len(v) if isinstance(v, dict) else "?" for k, v in result_dict.items()}
                        log.debug("[t%d] schema → %s", turn, summary)
            elif name == "db_query":
                # Models may pass sub_queries/join_on/transform as direct args instead
                # of JSON-encoding them into `query`. Assemble into the spec string here.
                if "sub_queries" in arguments and not arguments.get("query"):
                    sq = arguments["sub_queries"]
                    if isinstance(sq, str):
                        try:
                            sq = json.loads(sq)
                        except json.JSONDecodeError:
                            pass
                    # Handle double-encoded list: ["[{...}]"] → [{...}]
                    if isinstance(sq, list) and len(sq) == 1 and isinstance(sq[0], str):
                        try:
                            parsed = json.loads(sq[0])
                            if isinstance(parsed, list):
                                sq = parsed
                        except json.JSONDecodeError:
                            pass
                    spec = {"sub_queries": sq}
                    if arguments.get("join_on"):
                        spec["join_on"] = arguments["join_on"]
                    if arguments.get("transform"):
                        spec["transform"] = arguments["transform"]
                    raw_q = json.dumps(spec)
                else:
                    raw_q = arguments.get("query", "")
                    if isinstance(raw_q, dict):
                        raw_q = json.dumps(raw_q)
                dbs = arguments.get("databases")
                is_spec = raw_q.strip().startswith("{")
                # Guard: cannot run a single SQL query against multiple databases.
                # Direct the model to use the JSON spec format instead.
                if dbs and len(dbs) > 1 and not is_spec:
                    result_dict = {
                        "error": (
                            f"Cannot run a single SQL query against multiple databases "
                            f"({', '.join(dbs)}). "
                            "Use the JSON spec format with sub_queries to query each "
                            "database separately, then join the results:\n"
                            '{"sub_queries": [{"db": "' + dbs[0] + '", "query": "SELECT ...", "key": "a"}, '
                            '{"db": "' + dbs[1] + '", "query": "SELECT ...", "key": "b"}], '
                            '"join_on": [["a.col", "b.col"]]}'
                        )
                    }
                    result = json.dumps(result_dict)
                    log.debug(
                        "[t%d] query db=%s | blocked multi-DB single SQL — redirecting to spec",
                        turn, dbs,
                    )
                else:
                    db_label = "spec" if is_spec else dbs
                    q = next(
                        (ln.strip() for ln in raw_q.splitlines() if ln.strip()), ""
                    )[:60]
                    log.debug("[t%d] query db=%s | %s | executing...", turn, db_label, q)
                    result_dict = await self._engine.query(raw_q, dbs, self._session_id, row_limit=1_000)
                    result = json.dumps(result_dict, default=_json_serialise)
                    if len(result) > _MAX_TOOL_RESULT_CHARS:
                        result = result[:_MAX_TOOL_RESULT_CHARS] + (
                            f'"... [TRUNCATED — result exceeded {_MAX_TOOL_RESULT_CHARS} chars.'
                            ' Re-query with tighter WHERE filters or fewer columns.]'
                        )
                    if log.isEnabledFor(logging.DEBUG):
                        if "error" in result_dict:
                            log.debug("[t%d] query db=%s | %s | ERROR: %s",
                                      turn, db_label, q, str(result_dict.get("error", ""))[:60])
                        else:
                            log.debug("[t%d] query db=%s | %s | → %d rows",
                                      turn, db_label, q, result_dict.get("row_count", 0))
            elif name == "math_compute":
                # Step 1: resolve data rows (from inline or by running a query)
                data_rows: list[dict] = list(arguments.get("rows") or [])
                query_error: str | None = None
                if "sub_queries" in arguments or "query" in arguments:
                    if "sub_queries" in arguments:
                        _sq = arguments["sub_queries"]
                        if isinstance(_sq, str):
                            try:
                                _sq = json.loads(_sq)
                            except json.JSONDecodeError:
                                pass
                        # Handle double-encoded list: ["[{...}]"] → [{...}]
                        if isinstance(_sq, list) and len(_sq) == 1 and isinstance(_sq[0], str):
                            try:
                                parsed = json.loads(_sq[0])
                                if isinstance(parsed, list):
                                    _sq = parsed
                            except json.JSONDecodeError:
                                pass
                        _spec: dict = {"sub_queries": _sq}
                        if arguments.get("join_on"):
                            _spec["join_on"] = arguments["join_on"]
                        raw_q = json.dumps(_spec)
                    else:
                        raw_q = arguments["query"]
                        if isinstance(raw_q, dict):
                            raw_q = json.dumps(raw_q)
                    dbs = arguments.get("databases")
                    qr = await self._engine.query(raw_q, dbs, self._session_id, row_limit=1_000)
                    if "error" in qr:
                        query_error = str(qr["error"])
                        log.debug("[t%d] math_compute query error: %s", turn, query_error)
                    else:
                        data_rows = qr.get("rows", [])
                        log.debug("[t%d] math_compute query → %d rows", turn, len(data_rows))

                # Step 2: compute (skip if query failed)
                if query_error is not None:
                    result_dict = {"error": f"Query failed: {query_error}"}
                elif "expression" in arguments:
                    # Inject rows + per-column lists so expression can reference them by SQL alias
                    _vars: dict = dict(arguments.get("variables") or {})
                    _vars["rows"] = data_rows
                    if data_rows:
                        for _col in data_rows[0]:
                            if _col not in _vars:
                                _vars[_col] = [r.get(_col) for r in data_rows]
                    result_dict = _safe_eval(arguments["expression"], _vars)
                    log.debug("[t%d] math_compute expr → %s", turn, str(result_dict)[:120])
                else:
                    op = arguments.get("operation", "")
                    if op == "chi_square":
                        result_dict = _compute_chi_square(
                            rows=data_rows,
                            row_col=arguments.get("row_col", "row_label"),
                            col_col=arguments.get("col_col", "col_label"),
                            count_col=arguments.get("count_col", "count"),
                            min_marginal=float(arguments.get("min_marginal", 0)),
                        )
                    elif op == "ema":
                        result_dict = _compute_ema(
                            rows=data_rows,
                            group_col=arguments.get("group_col", "group"),
                            sort_col=arguments.get("sort_col", "year"),
                            value_col=arguments.get("value_col", "count"),
                            alpha=float(arguments.get("alpha", 0.3)),
                            fill_gaps=bool(arguments.get("fill_gaps", True)),
                            summarize=bool(arguments.get("summarize", True)),
                        )
                    else:
                        result_dict = {"error": f"Unknown math_compute operation: {op!r}. Use 'expression' for ad-hoc formulas, 'chi_square' for chi-square test, or 'ema' for exponential moving average."}
                    log.debug("[t%d] math_compute op=%s → %s", turn, op, str(result_dict)[:120])
                result = json.dumps(result_dict)
            else:
                result = json.dumps({"error": f"Unknown tool: {name!r}"})
        except Exception as e:
            log.warning("[t%d] %s failed: %s", turn, name, e)
            result = json.dumps({"error": str(e)})
        return result


# ── Abstract agent ─────────────────────────────────────────────────────────────

_FORCE_CONCLUDE = (
    "This is your last turn. Do not call any more tools. "
    "State the answer value only — no reasoning, no explanation, no labels. "
    "Wrap it in <answer> and </answer> tags. "
    "Examples: <answer>1990s</answer>  <answer>42</answer>  <answer>Alice Smith</answer>. "
    "Give your best answer from the data you have collected, even if not 100% certain. "
    "Only output <answer>no answer</answer> if you retrieved no relevant data at all."
)

_WARN_LAST_TURN = (
    "One turn remaining. If you already have enough data to answer, do so now "
    "instead of making another tool call."
)


class BenchmarkAgent(ABC):
    MAX_TURNS = 20

    def __init__(self, model: str, tools: ToolExecutor) -> None:
        self._model = model
        self._tools = tools

    @abstractmethod
    async def answer(self, question: str) -> str:
        """
        Run the agentic loop and return a text answer string.
        DAB's validate.py checks for ground-truth containment in this string.
        """
        ...


# ── Anthropic agent ────────────────────────────────────────────────────────────

class AnthropicAgent(BenchmarkAgent):
    _LLM_TIMEOUT = 300  # hard wall-clock seconds per LLM call

    def __init__(self, model: str, tools: ToolExecutor, api_key: str | None = None, system_prompt: str = "") -> None:
        super().__init__(model, tools)
        import anthropic
        self._anthropic = anthropic
        self._api_key = api_key
        self._system_prompt = system_prompt
        self._client = self._make_client()

    def _make_client(self):
        return self._anthropic.AsyncAnthropic(
            api_key=self._api_key,
            # httpx read timeout resets per-chunk on chunked responses, so it
            # cannot bound total wall-clock time. We rely on asyncio.wait_for
            # in _call() for that; keep these generous to not interfere.
            timeout=self._anthropic.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0),
            max_retries=0,
            default_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
        )

    async def _call(self, turn: int, **kwargs):
        """
        Call messages.create with a hard wall-clock timeout.
        On timeout, recreate the client so the hung connection never
        poisons subsequent queries. Returns None on timeout/error.
        """
        n_msgs = len(kwargs.get("messages", []))
        log.debug("[t%d] → LLM call start (%d messages in context)", turn, n_msgs)
        try:
            result = await asyncio.wait_for(
                self._client.messages.create(**kwargs),
                timeout=self._LLM_TIMEOUT,
            )
            log.debug("[t%d] → LLM responded: stop_reason=%s", turn, result.stop_reason)
            return result
        except asyncio.TimeoutError:
            log.warning("[t%d] → LLM call exceeded %ds — resetting HTTP client", turn, self._LLM_TIMEOUT)
            self._client = self._make_client()
            return None
        except self._anthropic.APIError as e:
            if "prompt is too long" in str(e):
                m = re.search(r'(\d+) tokens > (\d+) maximum', str(e))
                over = int(m.group(1)) - int(m.group(2)) if m else 20_000
                log.warning("[t%d] → context too long (+%d tokens) — will trim and retry", turn, over)
                return _TooLong(over)
            log.warning("[t%d] → LLM API error: %s", turn, e)
            return None

    async def answer(self, question: str) -> str:
        # Build tools list; mark the last tool for prompt caching so the full
        # tools + system prompt prefix is cached across turns and queries.
        anthropic_tools = [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["parameters"],
            }
            for t in _TOOLS
        ]
        anthropic_tools[-1]["cache_control"] = {"type": "ephemeral"}

        # System prompt as a content block so we can attach cache_control.
        cached_system = [
            {
                "type": "text",
                "text": self._system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]

        messages: list[dict] = [{"role": "user", "content": question}]
        final_text = ""
        trim_count = 0

        for turn in range(self.MAX_TURNS):
            response = await self._call(
                turn=turn + 1,
                model=self._model,
                max_tokens=16384,
                system=cached_system,
                tools=anthropic_tools,
                messages=messages,
            )
            if isinstance(response, _TooLong):
                if len(messages) < 3:
                    log.warning("[t%d] context too long but nothing left to trim — stopping", turn + 1)
                    break
                n_before = len(messages)
                _smart_drop(messages, response.over_tokens)
                trim_count += 1
                log.warning(
                    "[t%d] trimmed context %d→%d messages (trim #%d, +%d tokens over)",
                    turn + 1, n_before, len(messages), trim_count, response.over_tokens,
                )
                if trim_count >= 5:
                    log.warning("[t%d] too many context trims — stopping to avoid loop", turn + 1)
                    break
                continue

            if response is None:
                log.warning("[t%d] stopping — no LLM response", turn + 1)
                break

            messages.append({"role": "assistant", "content": response.content})

            for block in response.content:
                if hasattr(block, "text"):
                    final_text = block.text

            if response.stop_reason != "tool_use":
                break

            last_turn = (turn == self.MAX_TURNS - 1)

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result_str = await self._tools.call(block.name, block.input, turn=turn + 1)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_str,
                })

            messages.append({"role": "user", "content": tool_results})

            if last_turn:
                conclusion = await self._call(
                    turn=turn + 1,
                    model=self._model,
                    max_tokens=1024,
                    system=cached_system,
                    messages=messages + [{"role": "user", "content": _FORCE_CONCLUDE}],
                )
                if conclusion:
                    for block in conclusion.content:
                        if hasattr(block, "text"):
                            final_text = block.text
                break

        return _extract_answer(final_text)


# ── OpenAI agent ───────────────────────────────────────────────────────────────

class OpenAIAgent(BenchmarkAgent):
    _LLM_TIMEOUT = 120  # hard wall-clock seconds per LLM call
    _MAX_OUTPUT_TOKENS = 4096
    _FORCE_CONCLUDE_TOKENS = 1024  # tokens for the forced final-answer call
    _MAX_API_RETRIES = 3           # total attempts on 429 (1 initial + 2 retries)
    _RETRY_429_BASE = 5            # seconds; doubles each retry: 5 → 10 → 20 (capped 60)

    def __init__(
        self,
        model: str,
        tools: ToolExecutor,
        api_key: str | None = None,
        system_prompt: str = "",
        base_url: str | None = None,
    ) -> None:
        super().__init__(model, tools)
        self._system_prompt = system_prompt
        import openai
        self._openai = openai
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=openai.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0),
            max_retries=0,
        )
        # Newer models (o1/o3/o4/gpt-5+) require max_completion_tokens; older use max_tokens.
        _new_token_models = ("o1", "o3", "o4", "gpt-5")
        self._max_tokens_key = (
            "max_completion_tokens"
            if any(model.startswith(p) for p in _new_token_models)
            else "max_tokens"
        )

    async def _call(self, turn: int, **kwargs):
        n_msgs = len(kwargs.get("messages", []))
        log.debug("[t%d] → LLM call start (%d messages in context)", turn, n_msgs)
        max_retries = self._MAX_API_RETRIES
        for attempt in range(max_retries):
            try:
                result = await asyncio.wait_for(
                    self._client.chat.completions.create(**kwargs),
                    timeout=self._LLM_TIMEOUT,
                )
                log.debug("[t%d] → LLM responded: finish_reason=%s", turn, result.choices[0].finish_reason)
                return result
            except asyncio.TimeoutError:
                log.warning("[t%d] → LLM call exceeded %ds — skipping turn", turn, self._LLM_TIMEOUT)
                return None
            except self._openai.APIError as e:
                status = getattr(e, "status_code", None)
                if status == 429 and attempt < max_retries - 1:
                    wait = min(self._RETRY_429_BASE * (2 ** attempt), 60)
                    log.warning(
                        "[t%d] → engine overloaded (429) — waiting %ds (retry %d/%d)",
                        turn, wait, attempt + 1, max_retries - 1,
                    )
                    await asyncio.sleep(wait)
                    continue
                err_str = str(e).lower()
                if status == 400 and any(k in err_str for k in ("context", "length", "too long", "maximum")):
                    # Context window exceeded — extract token counts if present.
                    m = re.search(r'(\d+)[^\d]+(\d+)', str(e))
                    over = (int(m.group(1)) - int(m.group(2))) if m else 30_000
                    log.warning("[t%d] → context too long (+%d tokens) — will trim and retry", turn, over)
                    return _TooLong(over)
                log.warning("[t%d] → LLM API error: %s", turn, e)
                return None

    async def answer(self, question: str) -> str:
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in _TOOLS
        ]

        messages: list[dict] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": question},
        ]
        final_text = ""
        trim_count = 0

        for turn in range(self.MAX_TURNS):
            response = await self._call(
                turn=turn + 1,
                model=self._model,
                **{self._max_tokens_key: self._MAX_OUTPUT_TOKENS},
                tools=openai_tools,
                tool_choice="auto",
                messages=messages,
            )
            if isinstance(response, _TooLong):
                if len(messages) < 4:
                    log.warning("[t%d] context too long but nothing left to trim — stopping", turn + 1)
                    break
                n_before = len(messages)
                _smart_drop(messages, response.over_tokens)
                trim_count += 1
                log.warning(
                    "[t%d] trimmed context %d→%d messages (trim #%d, +%d tokens over)",
                    turn + 1, n_before, len(messages), trim_count, response.over_tokens,
                )
                continue  # retry same turn with trimmed context
            if response is None:
                log.warning("[t%d] stopping — no LLM response", turn + 1)
                break
            choice = response.choices[0]

            if choice.message.content:
                final_text = choice.message.content

            tc_list = choice.message.tool_calls or []
            asst: dict[str, Any] = {
                "role": "assistant",
                "content": choice.message.content,
            }
            if tc_list:
                asst["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tc_list
                ]
            messages.append(asst)

            if choice.finish_reason == "length" and not tc_list:
                # Model hit output token cap mid-response.
                if turn < self.MAX_TURNS - 1:
                    log.warning("[t%d] finish_reason=length — injecting continuation prompt", turn + 1)
                    messages.append({"role": "user", "content": "Your response was cut off. Please continue and provide your final answer inside <answer> tags."})
                    continue
                # Last turn hit length — still run force_conclude to extract any answer.
                log.warning("[t%d] finish_reason=length on last turn — running force_conclude", turn + 1)
                conclusion = await self._call(
                    turn=turn + 1,
                    model=self._model,
                    **{self._max_tokens_key: self._FORCE_CONCLUDE_TOKENS},
                    tools=openai_tools,
                    tool_choice="none",
                    messages=messages + [{"role": "user", "content": _FORCE_CONCLUDE}],
                )
                if conclusion:
                    c = conclusion.choices[0]
                    if c.message.content:
                        final_text = c.message.content
                break

            if choice.finish_reason != "tool_calls" or not tc_list:
                break

            last_turn = (turn == self.MAX_TURNS - 1)
            penultimate_turn = (turn == self.MAX_TURNS - 2)

            for tc in tc_list:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                result_str = await self._tools.call(tc.function.name, args, turn=turn + 1)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })

            if penultimate_turn:
                messages.append({"role": "user", "content": _WARN_LAST_TURN})

            if last_turn:
                conclusion = await self._call(
                    turn=turn + 1,
                    model=self._model,
                    **{self._max_tokens_key: self._FORCE_CONCLUDE_TOKENS},
                    tools=openai_tools,
                    tool_choice="none",
                    messages=messages + [{"role": "user", "content": _FORCE_CONCLUDE}],
                )
                if conclusion:
                    c = conclusion.choices[0]
                    if c.message.content:
                        final_text = c.message.content
                break

        return _extract_answer(final_text)


# ── Groq ───────────────────────────────────────────────────────────────────────

class GroqAgent(OpenAIAgent):
    """OpenAI-compatible agent pointed at Groq's inference endpoint."""

    _GROQ_BASE_URL = "https://api.groq.com/openai/v1"
    _MAX_OUTPUT_TOKENS = 16384  # Qwen3 thinking mode needs headroom beyond default 4096

    def __init__(self, model: str, tools: ToolExecutor, api_key: str | None = None, system_prompt: str = "") -> None:
        super().__init__(model, tools, api_key=api_key, system_prompt=system_prompt, base_url=self._GROQ_BASE_URL)


# ── Kimi (Moonshot AI) ─────────────────────────────────────────────────────────

class KimiAgent(OpenAIAgent):
    """OpenAI-compatible agent pointed at Moonshot AI's Kimi endpoint.

    Base URL : https://api.moonshot.ai/v1
    API key  : MOONSHOT_API_KEY
    Note     : tool_choice="required" is unsupported; temperature is capped at 1.0.
    """

    _KIMI_BASE_URL = "https://api.moonshot.ai/v1"
    _MAX_OUTPUT_TOKENS = 32768  # kimi uses extended chain-of-thought; 8192 truncates mid-reasoning
    _LLM_TIMEOUT = 300          # kimi-k2.7 extended reasoning can exceed the 120s default
    _FORCE_CONCLUDE_TOKENS = 4096   # answer-only response; explicit "no reasoning" keeps content short
    _MAX_API_RETRIES = 5        # kimi hits engine_overloaded more often; 5 attempts (5→10→20→40s)
    _RETRY_429_BASE = 5         # exponential backoff: 5, 10, 20, 40s (75s total wait across retries)
    _TPM_LIMIT = 3_000_000      # Tier-2 tokens-per-minute limit
    _TPM_RESERVE = 100_000      # headroom to keep before sleeping
    _TPM_WINDOW = 60.0          # sliding window in seconds

    def __init__(self, model: str, tools: ToolExecutor, api_key: str | None = None, system_prompt: str = "") -> None:
        super().__init__(model, tools, api_key=api_key, system_prompt=system_prompt, base_url=self._KIMI_BASE_URL)
        # Sliding window of (monotonic_timestamp, tokens_used) tuples for TPM tracking.
        self._tpm_window: collections.deque = collections.deque()

    async def _call(self, turn: int, **kwargs):
        # Enforce TPM limit: prune entries older than 60s, then sleep until headroom exists.
        now = _time.monotonic()
        cutoff = now - self._TPM_WINDOW
        while self._tpm_window and self._tpm_window[0][0] < cutoff:
            self._tpm_window.popleft()

        window_tokens = sum(t for _, t in self._tpm_window)
        if window_tokens >= self._TPM_LIMIT - self._TPM_RESERVE:
            oldest_ts = self._tpm_window[0][0]
            sleep_secs = (oldest_ts + self._TPM_WINDOW) - now + 1.0
            if sleep_secs > 0:
                log.warning(
                    "[t%d] TPM %.0f/%.0fM — sleeping %.0fs for window reset",
                    turn, window_tokens, self._TPM_LIMIT / 1_000_000, sleep_secs,
                )
                await asyncio.sleep(sleep_secs)
                # Re-prune after waking up.
                now = _time.monotonic()
                cutoff = now - self._TPM_WINDOW
                while self._tpm_window and self._tpm_window[0][0] < cutoff:
                    self._tpm_window.popleft()

        result = await super()._call(turn, **kwargs)

        # Record token usage for this call in the sliding window.
        if result is not None and not isinstance(result, _TooLong):
            usage = getattr(result, "usage", None)
            if usage:
                tokens = getattr(usage, "total_tokens", 0) or 0
                self._tpm_window.append((_time.monotonic(), tokens))

        return result


# ── Together AI ────────────────────────────────────────────────────────────────

class TogetherAgent(OpenAIAgent):
    """OpenAI-compatible agent pointed at Together AI's inference endpoint.

    Base URL : https://api.together.ai/v1
    API key  : TOGETHER_API_KEY
    Models   : use namespaced IDs, e.g. "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"

    Rate limits are dynamic (no fixed TPM/RPM published). On 429, Together returns:
      x-ratelimit-reset  — seconds until the limit resets (used as the exact sleep duration)
      error_type         — "dynamic_request_limited" or "dynamic_token_limited"
    We sleep for x-ratelimit-reset + 1s buffer, then retry up to _MAX_API_RETRIES times.
    """

    _TOGETHER_BASE_URL = "https://api.together.ai/v1"
    _LLM_TIMEOUT = 600          # Together models can be slow; large models need headroom
    _MAX_OUTPUT_TOKENS = 16384  # avoid finish_reason=length cutting off mid-response
    _FORCE_CONCLUDE_TOKENS = 4096
    _MAX_API_RETRIES = 6        # more retries since dynamic limits can reset quickly
    _RETRY_429_BASE = 10        # fallback if x-ratelimit-reset header is absent
    _RETRY_503_BASE = 15        # 503 service unavailable — slightly longer initial wait
    _RETRY_MAX_WAIT = 120       # cap on any single sleep (2 minutes)

    def __init__(self, model: str, tools: ToolExecutor, api_key: str | None = None, system_prompt: str = "") -> None:
        super().__init__(model, tools, api_key=api_key, system_prompt=system_prompt, base_url=self._TOGETHER_BASE_URL)

    async def _call(self, turn: int, **kwargs):
        max_retries = self._MAX_API_RETRIES
        for attempt in range(max_retries):
            try:
                result = await asyncio.wait_for(
                    self._client.chat.completions.create(**kwargs),
                    timeout=self._LLM_TIMEOUT,
                )
                log.debug("[t%d] → LLM responded: finish_reason=%s", turn, result.choices[0].finish_reason)
                return result
            except asyncio.TimeoutError:
                log.warning("[t%d] → LLM call exceeded %ds — skipping turn", turn, self._LLM_TIMEOUT)
                return None
            except self._openai.APIError as e:
                status = getattr(e, "status_code", None)
                err_body = getattr(e, "body", {}) or {}

                if status == 429 and attempt < max_retries - 1:
                    # Prefer x-ratelimit-reset header (exact seconds Together says to wait).
                    headers = getattr(getattr(e, "response", None), "headers", {}) or {}
                    reset_secs = headers.get("x-ratelimit-reset")
                    try:
                        wait = min(float(reset_secs) + 1.0, self._RETRY_MAX_WAIT)
                    except (TypeError, ValueError):
                        wait = min(self._RETRY_429_BASE * (2 ** attempt), self._RETRY_MAX_WAIT)
                    limit_type = err_body.get("error_type", "rate_limited")
                    log.warning(
                        "[t%d] → 429 %s — sleeping %.0fs (attempt %d/%d)",
                        turn, limit_type, wait, attempt + 1, max_retries - 1,
                    )
                    await asyncio.sleep(wait)
                    continue

                if status in (500, 503) and attempt < max_retries - 1:
                    wait = min(self._RETRY_503_BASE * (2 ** attempt), self._RETRY_MAX_WAIT)
                    log.warning(
                        "[t%d] → %d server error — sleeping %.0fs (attempt %d/%d)",
                        turn, status, wait, attempt + 1, max_retries - 1,
                    )
                    await asyncio.sleep(wait)
                    continue

                err_str = str(e).lower()
                if status == 400 and any(k in err_str for k in ("context", "length", "too long", "maximum")):
                    m = re.search(r'(\d+)[^\d]+(\d+)', str(e))
                    over = (int(m.group(1)) - int(m.group(2))) if m else 30_000
                    log.warning("[t%d] → context too long (+%d tokens) — will trim and retry", turn, over)
                    return _TooLong(over)

                log.warning("[t%d] → Together API error %s: %s", turn, status, e)
                return None


# ── Ollama (local) ─────────────────────────────────────────────────────────────

class OllamaAgent(OpenAIAgent):
    """OpenAI-compatible agent pointed at a local Ollama instance.

    Base URL : http://localhost:11434/v1
    API key  : not required (uses 'ollama' as dummy)
    Models   : any model pulled via `ollama pull <model>` (e.g. qwen3:4b)
    """

    _OLLAMA_BASE_URL = "http://localhost:11434/v1"
    _MAX_OUTPUT_TOKENS = 2048  # cap per-turn output; prevents verbose models filling context
    _LLM_TIMEOUT = 600  # local inference is slow; 10-min ceiling per call

    def __init__(self, model: str, tools: ToolExecutor, api_key: str | None = None, system_prompt: str = "") -> None:
        super().__init__(model, tools, api_key=api_key or "ollama", system_prompt=system_prompt, base_url=self._OLLAMA_BASE_URL)

    async def _call(self, turn: int, **kwargs):
        # Pass think=False at the Ollama API level — the /no_think system-prompt directive
        # is unreliable; this option is the authoritative way to disable qwen3 reasoning chains.
        extra = kwargs.pop("extra_body", {})
        extra["think"] = False
        return await super()._call(turn, extra_body=extra, **kwargs)


# ── Factory ────────────────────────────────────────────────────────────────────

def build_agent(
    provider: str,
    model: str,
    tools: ToolExecutor,
    api_key: str | None = None,
    db_types: set[str] | None = None,
) -> BenchmarkAgent:
    dbt = db_types or set()
    if provider == "anthropic":
        return AnthropicAgent(model, tools, api_key, system_prompt=_build_system_prompt(dbt))
    if provider == "openai":
        return OpenAIAgent(model, tools, api_key, system_prompt=_build_system_prompt(dbt))
    if provider == "groq":
        return GroqAgent(model, tools, api_key, system_prompt=_build_system_prompt(dbt))
    if provider == "kimi":
        return KimiAgent(model, tools, api_key, system_prompt=_build_kimi_system_prompt(dbt))
    if provider == "together":
        return TogetherAgent(model, tools, api_key, system_prompt=_build_system_prompt(dbt))
    if provider == "ollama":
        return OllamaAgent(model, tools, api_key, system_prompt=_build_system_prompt(dbt))
    raise ValueError(f"Unknown provider {provider!r}. Choose 'anthropic', 'openai', 'groq', 'kimi', 'together', or 'ollama'.")
