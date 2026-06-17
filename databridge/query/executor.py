from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from databridge.connectors.base import DbType, QueryResult
from databridge.connectors.registry import ConnectorRegistry
from databridge.query.planner import QueryPlan, SubQuery
from databridge.query.translator import QueryTranslator
from databridge.safety.enforcement import SafetyEnforcer
from databridge.schema.joins.registry import JoinRule


@dataclass
class ExecutionResult:
    rows: list[dict[str, Any]]
    row_count: int
    columns: list[str]
    sub_results: dict[str, QueryResult] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    total_ms: float = 0.0
    truncated: bool = False


class QueryExecutor:
    def __init__(
        self,
        registry: ConnectorRegistry,
        enforcer: SafetyEnforcer,
        translator: QueryTranslator,
        default_row_limit: int = 10_000,
    ) -> None:
        self._registry = registry
        self._enforcer = enforcer
        self._translator = translator
        self._row_limit = default_row_limit

    async def execute(self, plan: QueryPlan, row_limit: int | None = None) -> ExecutionResult:
        limit = row_limit or self._row_limit
        t0 = time.monotonic()

        # Safety check all sub-queries
        for sq in plan.sub_queries:
            connector = self._registry.get(sq.db_alias)
            dialect = connector.db_type.sqlglot_dialect
            if connector.db_type != DbType.MONGODB:
                self._enforcer.check(sq.query, dialect=dialect)

        # Translate dialects and inject limits
        translated = []
        effective_limits: list[int | None] = []
        for sq in plan.sub_queries:
            connector = self._registry.get(sq.db_alias)
            if connector.db_type == DbType.MONGODB:
                # MongoDB pipelines must not go through SQL translation or LIMIT injection —
                # the connector appends {"$limit": row_limit} to the pipeline itself.
                translated.append(SubQuery(db_alias=sq.db_alias, query=sq.query, result_key=sq.result_key))
                effective_limits.append(limit)
                continue
            dialect = connector.db_type.sqlglot_dialect
            # translate() is a no-op when source == target dialect (agent writes for the target DB).
            # It only matters if a query is written in a different dialect than the target.
            q = self._translator.translate(sq.query, dialect, connector.db_type)
            if self._translator.is_aggregate_query(q, dialect=dialect):
                q = self._translator.push_aggregation(q, dialect=dialect)
                effective_limits.append(None)  # no limit on aggregates
            else:
                q = self._translator.inject_limit(q, limit, dialect=dialect)
                effective_limits.append(limit)
            translated.append(SubQuery(db_alias=sq.db_alias, query=q, result_key=sq.result_key))

        # Execute in parallel
        tasks = [
            self._execute_one(sq, row_limit)
            for sq, row_limit in zip(translated, effective_limits)
        ]
        results: list[QueryResult] = await asyncio.gather(*tasks)

        sub_map = {sq.result_key: res for sq, res in zip(translated, results)}

        # Merge results
        if len(results) == 1:
            rows = results[0].rows
            cols = results[0].columns
        else:
            rows, cols = _merge_results(sub_map, plan.join_rules, plan.merge_on)

        total_ms = (time.monotonic() - t0) * 1000
        warnings = _collect_warnings(sub_map)

        return ExecutionResult(
            rows=rows,
            row_count=len(rows),
            columns=cols,
            sub_results=sub_map,
            provenance={
                **plan.provenance,
                "databases": [sq.db_alias for sq in translated],
                "join_rules_applied": [r.join_id for r in plan.join_rules],
                "db_key_map": {sq.result_key: sq.db_alias for sq in translated},
                "merge_on": plan.merge_on,
                "sub_queries": [{"key": sq.result_key, "db": sq.db_alias, "query": sq.query} for sq in translated],
            },
            warnings=warnings,
            total_ms=total_ms,
            truncated=any(r.truncated for r in results),
        )

    async def _execute_one(self, sq: SubQuery, limit: int | None) -> QueryResult:
        connector = self._registry.get(sq.db_alias)
        return await connector.execute_query(sq.query, row_limit=limit)


def _merge_results(
    sub_map: dict[str, QueryResult],
    join_rules: list[JoinRule],
    merge_on: list[tuple[str, str]] | None = None,
) -> tuple[list[dict], list[str]]:
    """
    In-memory merge across sub-results.

    When the query spec includes explicit join_on, the planner populates merge_on
    and leaves join_rules empty — model-provided columns take precedence.
    When no join_on is given, the planner populates join_rules from the registry.
    merge_on entries are "result_key.column" pairs, e.g. ("pg.id", "sq.movie_id").
    """
    from databridge.schema.joins.transforms import TRANSFORM_GRAMMAR

    transform_fns = {name: fn for name, fn in TRANSFORM_GRAMMAR}

    keys = list(sub_map.keys())
    if not keys:
        return [], []

    merged = sub_map[keys[0]].rows

    # Build a unified list of (left_key, left_col, right_key, right_col, transform_fn) joins
    joins: list[tuple[str, str, str, str, Any]] = []

    for rule in join_rules:
        right_key = keys[1] if len(keys) > 1 else keys[0]
        fn = transform_fns.get(rule.transform or "identity", str)
        joins.append((keys[0], rule.column_a, right_key, rule.column_b, fn))

    if not joins and merge_on:
        # Parse "result_key.column" pairs from the spec's join_on field
        for left_ref, right_ref in merge_on:
            left_parts = left_ref.split(".", 1)
            right_parts = right_ref.split(".", 1)
            if len(left_parts) == 2 and len(right_parts) == 2:
                joins.append((left_parts[0], left_parts[1], right_parts[0], right_parts[1], str))

    for left_key, col_a, right_key, col_b, transform_fn in joins:
        left_rows = sub_map.get(left_key, QueryResult([], 0, [])).rows if left_key != keys[0] else merged
        right_rows = sub_map.get(right_key, QueryResult([], 0, [])).rows
        if not right_rows:
            continue

        # Build hash index on right side
        right_index: dict[str, list[dict]] = {}
        for row in right_rows:
            val = row.get(col_b)
            if val is not None:
                right_index.setdefault(str(val), []).append(row)

        # Hash join
        result_rows = []
        for left_row in left_rows:
            left_val = left_row.get(col_a)
            if left_val is None:
                continue
            try:
                transformed = transform_fn(left_val)
            except Exception:
                transformed = str(left_val)
            for right_row in right_index.get(transformed, []):
                merged_row = {**left_row, **{f"{right_key}__{k}": v for k, v in right_row.items()}}
                result_rows.append(merged_row)

        # Always update merged — if result_rows is empty, the join produced no
        # matches (key format mismatch or genuine no-overlap). Return empty so
        # the caller sees 0 rows and can retry, rather than silently getting the
        # unfiltered left-side rows.
        merged = result_rows

    cols = list(merged[0].keys()) if merged else []
    return merged, cols


def _collect_warnings(sub_map: dict[str, QueryResult]) -> list[str]:
    warnings = []
    for key, result in sub_map.items():
        if result.truncated:
            warnings.append(
                f"Sub-query '{key}' returned exactly {result.row_count} rows (the row limit) "
                f"— the table has more rows and the result is a partial sample. "
                f"Averages, counts, and aggregations computed from this data will be WRONG. "
                f"Fix: add GROUP BY + AVG/COUNT inside the sub-query to aggregate before "
                f"joining, instead of returning raw rows."
            )
    return warnings
