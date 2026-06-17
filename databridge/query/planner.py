from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from databridge.connectors.base import DbType
from databridge.connectors.registry import ConnectorRegistry
from databridge.schema.joins.registry import JoinRegistry, JoinRule


@dataclass
class SubQuery:
    db_alias: str
    query: str  # SQL string or JSON-serialised MongoDB pipeline dict
    result_key: str  # alias used when merging results


@dataclass
class QueryPlan:
    sub_queries: list[SubQuery] = field(default_factory=list)
    join_rules: list[JoinRule] = field(default_factory=list)
    merge_on: list[tuple[str, str]] = field(default_factory=list)  # (left_col, right_col)
    provenance: dict[str, Any] = field(default_factory=dict)


class QueryPlanner:
    def __init__(
        self,
        registry: ConnectorRegistry,
        join_registry: JoinRegistry,
    ) -> None:
        self._registry = registry
        self._join_registry = join_registry

    async def plan(
        self,
        query: str,
        target_databases: list[str] | None = None,
    ) -> QueryPlan:
        """
        Build an execution plan for a query.

        For single-database queries: one SubQuery, no joins.
        For cross-database queries: multiple SubQueries with join rules applied.

        query: SQL string targeting one or more databases.
        target_databases: if provided, restrict to these aliases; otherwise use all.
        """
        aliases = target_databases or self._registry.aliases()
        plan = QueryPlan()

        # Simple heuristic: split the query across databases by detecting table names
        # that are present in each database's schema cache.
        # For cross-DB queries the agent passes structured hints in the query metadata.
        if isinstance(query, str) and query.strip().startswith("{"):
            spec = json.loads(query)
            return await self._plan_from_spec(spec, aliases)

        # Single-database fallback: route to first matching alias
        plan.sub_queries.append(SubQuery(db_alias=aliases[0], query=query, result_key="main"))
        plan.provenance["strategy"] = "single_database"
        plan.provenance["target"] = aliases[0]
        return plan

    async def _plan_from_spec(self, spec: dict, aliases: list[str]) -> QueryPlan:
        """
        spec format:
        {
            "sub_queries": [
                {"db": "postgresql", "query": "SELECT ...", "key": "orders"},
                {"db": "mongodb",    "query": {...},          "key": "users"}
            ],
            "join_on": [["orders.customer_id", "users._id"]]
        }

        Also accepts MongoDB-native format (no sub_queries wrapper):
        {"collection": "articles", "pipeline": [...]}
        {"db": "mongodb", "collection": "articles", "pipeline": [...]}
        """
        plan = QueryPlan()

        # MongoDB-native format: {"collection": "...", "pipeline": [...]}
        if not spec.get("sub_queries") and "collection" in spec:
            db_hint = spec.get("db")
            if db_hint and db_hint in aliases:
                mongo_alias = db_hint
            else:
                mongo_alias = next(
                    (a for a in aliases if self._registry.get(a).db_type == DbType.MONGODB),
                    None,
                )
            if mongo_alias:
                q = json.dumps({k: v for k, v in spec.items() if k != "db"})
                plan.sub_queries.append(SubQuery(db_alias=mongo_alias, query=q, result_key="main"))
                plan.provenance["strategy"] = "mongodb_native"
            return plan

        sub_queries_raw = spec.get("sub_queries", [])
        # Some models double-encode the array as a JSON string (with possible
        # trailing content). raw_decode stops at the first complete JSON value.
        if isinstance(sub_queries_raw, str):
            sub_queries_raw, _ = json.JSONDecoder().raw_decode(sub_queries_raw.strip())

        for sq in sub_queries_raw:
            q = sq["query"]
            if isinstance(q, dict):
                q = json.dumps(q)
            plan.sub_queries.append(SubQuery(db_alias=sq["db"], query=q, result_key=sq["key"]))

        for pair in spec.get("join_on", []):
            if len(pair) == 2:
                plan.merge_on.append((pair[0], pair[1]))

        # Attach known join rules for result merging — only when the model did NOT
        # provide an explicit join_on. When join_on is present, the model has already
        # reasoned about the join columns (potentially with computed aliases like
        # "id_num") and the registry rule would reference different column names,
        # causing a silent 0-row join if used instead.
        if not plan.merge_on:
            for sq_a, sq_b in zip(plan.sub_queries, plan.sub_queries[1:]):
                db_a, db_b = sq_a.db_alias, sq_b.db_alias
                # Extract table names naively from the first word after FROM
                ta = _extract_table(sq_a.query)
                tb = _extract_table(sq_b.query)
                rules = await self._join_registry.find_for_tables(db_a, ta, db_b, tb)
                plan.join_rules.extend(r for r in rules if r.is_reliable)

        plan.provenance["strategy"] = "spec_plan"
        return plan


def _extract_table(query: str) -> str:
    import re
    m = re.search(r"\bFROM\b\s+[\"']?(\w+)[\"']?", query, re.IGNORECASE)
    return m.group(1) if m else ""
