from __future__ import annotations

import re
import time

from databridge.query.executor import ExecutionResult
from databridge.schema.joins.registry import JoinRegistry, JoinRule


class SessionLearner:
    """
    Observes query results and updates join registry confidence scores.
    Zero-row results on joins flag the rule as suspect.
    Non-zero results reinforce confidence.
    """

    def __init__(self, join_registry: JoinRegistry) -> None:
        self._registry = join_registry

    async def observe(self, result: ExecutionResult) -> None:
        join_ids = result.provenance.get("join_rules_applied", [])
        if not join_ids:
            return

        success = result.row_count > 0

        for join_id in join_ids:
            await self._registry.record_outcome(join_id, success=success)

    async def propose_from_result(
        self,
        result: ExecutionResult,
        session_id: str,
    ) -> None:
        """
        When a cross-DB spec query succeeds, propose the merge columns as
        unconfirmed join rules (confidence 0.65) if not already registered.
        This seeds the registry from agent-validated queries so future
        auto-discovery and human confirmation are easier.
        """
        if result.row_count == 0:
            return

        merge_on: list[tuple[str, str]] = result.provenance.get("merge_on", [])
        if not merge_on:
            return

        db_key_map: dict[str, str] = result.provenance.get("db_key_map", {})
        sub_queries: list[dict] = result.provenance.get("sub_queries", [])

        # Build result_key → table_name map by parsing FROM clause of each sub-query
        key_table: dict[str, str] = {}
        for sq in sub_queries:
            table = _extract_table(sq.get("query", ""))
            if table:
                key_table[sq["key"]] = table

        for left_ref, right_ref in merge_on:
            left_parts = left_ref.split(".", 1)
            right_parts = right_ref.split(".", 1)
            if len(left_parts) != 2 or len(right_parts) != 2:
                continue

            left_key, left_col = left_parts
            right_key, right_col = right_parts

            db_a = db_key_map.get(left_key)
            db_b = db_key_map.get(right_key)
            if not db_a or not db_b:
                continue

            table_a = key_table.get(left_key, "unknown")
            table_b = key_table.get(right_key, "unknown")

            join_id = f"{db_a}.{table_a}.{left_col}__{db_b}.{table_b}.{right_col}"

            existing = await self._registry.get(join_id)
            if existing is not None:
                # Already known — just reinforce if this was a success
                await self._registry.record_outcome(join_id, success=True)
                continue

            rule = JoinRule(
                join_id=join_id,
                db_a=db_a,
                table_a=table_a,
                column_a=left_col,
                db_b=db_b,
                table_b=table_b,
                column_b=right_col,
                transform=None,
                confidence=0.65,
                confirmed=False,
                verified_at=time.time(),
                verified_by=f"session:{session_id}",
                success_count=1,
                failure_count=0,
            )
            await self._registry.save(rule)


def _extract_table(query: str) -> str:
    m = re.search(r"\bFROM\b\s+[\"']?(\w+)[\"']?", query, re.IGNORECASE)
    return m.group(1) if m else ""
