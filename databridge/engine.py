"""
DataBridgeEngine — unified business logic shared by the MCP server and benchmark harness.

Both entry points construct an engine from the same components and call the same methods.
All product features (transforms, plausibility, join discovery, dedup, timeouts, audit,
session learning, large-result warnings) live here — not split across two codebases.
"""
from __future__ import annotations

import asyncio
import json
import re
import uuid
from typing import Any

from databridge.audit.log import AuditLog
from databridge.connectors.registry import ConnectorRegistry
from databridge.learning.session import SessionLearner
from databridge.query.executor import ExecutionResult, QueryExecutor
from databridge.query.planner import QueryPlanner
from databridge.query.transforms import apply_transforms
from databridge.safety.enforcement import SafetyViolation
from databridge.schema.joins.discovery import JoinDiscovery
from databridge.schema.joins.registry import JoinRegistry, JoinRule
from databridge.schema.scanner import SchemaScanner
from databridge.verification.plausibility import PlausibilityChecker


class DataBridgeEngine:
    """
    All DataBridge tool logic in one place.

    MCP server wraps each method with @mcp.tool().
    Benchmark harness calls methods directly, passing session_id per question.
    """

    QUERY_TIMEOUT = 180  # outer wall-clock guard — must exceed any connector-level timeout (120s)

    def __init__(
        self,
        registry: ConnectorRegistry,
        scanner: SchemaScanner,
        planner: QueryPlanner,
        executor: QueryExecutor,
        join_registry: JoinRegistry,
        checker: PlausibilityChecker | None = None,
        audit: AuditLog | None = None,
        learner: SessionLearner | None = None,
        discovery: JoinDiscovery | None = None,
        max_cost_budget: float = float("inf"),
    ) -> None:
        self._registry = registry
        self._scanner = scanner
        self._planner = planner
        self._executor = executor
        self._join_registry = join_registry
        self._checker = checker
        self._audit = audit
        self._learner = learner
        self._discovery = discovery
        self._max_cost_budget = max_cost_budget
        # Per-session dedup: session_id → set of (normalised_query, sorted_dbs) tuples
        self._seen: dict[str, set[tuple]] = {}

    def reset_session(self, session_id: str) -> None:
        """Clear dedup state for a session. Call between benchmark questions."""
        self._seen.pop(session_id, None)

    # ── db_query ──────────────────────────────────────────────────────────────

    async def query(
        self,
        query: str,
        databases: list[str] | None = None,
        session_id: str | None = None,
        row_limit: int | None = None,
    ) -> dict[str, Any]:
        """
        Execute a query and return rows, warnings, and plausibility info.

        Handles the full pipeline: transform stripping → dedup → cost check →
        plan → execute (with timeout) → apply transforms → warnings → plausibility
        → session learning → audit.
        """
        sid = session_id or str(uuid.uuid4())

        # Dedup: computed from raw query (before transform stripping) so that
        # the same spec WITH transforms is distinct from the same spec WITHOUT.
        dedup_key = (_normalise_query(query), tuple(sorted(databases or [])))

        # Strip "transform" from spec before passing to planner
        transforms: list[dict] = []
        if isinstance(query, str) and query.strip().startswith("{"):
            try:
                spec_obj = json.loads(query)
                transforms = spec_obj.pop("transform", None) or []
                if transforms:
                    query = json.dumps(spec_obj)
            except (json.JSONDecodeError, AttributeError):
                pass
        session_seen = self._seen.setdefault(sid, set())
        if dedup_key in session_seen:
            return {
                "error": (
                    "Duplicate query: you already ran this exact query earlier in this "
                    "conversation. Use the result already in your context — "
                    "do not re-run it. Form your final answer from the data you have."
                )
            }
        session_seen.add(dedup_key)

        is_spec = isinstance(query, str) and query.strip().startswith("{")

        # Plan
        try:
            plan = await self._planner.plan(query, target_databases=databases)
        except Exception as e:
            return {"error": str(e), "type": "planning_error"}

        # Cost budget check
        if self._max_cost_budget < float("inf"):
            total_cost = 0.0
            for sq in plan.sub_queries:
                try:
                    conn = self._registry.get(sq.db_alias)
                    if conn.db_type.value != "mongodb":
                        total_cost += await conn.explain_cost(sq.query)
                except Exception:
                    pass
            if total_cost > self._max_cost_budget:
                return {
                    "error": (
                        f"Query cost estimate ({total_cost:.0f}) exceeds the configured budget "
                        f"({self._max_cost_budget:.0f}). Add WHERE filters or aggregations to "
                        "reduce the scan scope."
                    ),
                    "type": "cost_overrun",
                    "estimated_cost": total_cost,
                    "budget": self._max_cost_budget,
                }

        # Execute with outer timeout
        try:
            result = await asyncio.wait_for(
                self._executor.execute(plan, row_limit=row_limit),
                timeout=self.QUERY_TIMEOUT,
            )
        except asyncio.TimeoutError:
            return {
                "error": (
                    f"Query timed out after {self.QUERY_TIMEOUT}s. "
                    "Add WHERE filters, aggregations (MIN/MAX/COUNT), or LIMIT clauses "
                    "to reduce the data scanned."
                ),
                "type": "timeout",
            }
        except SafetyViolation as e:
            return {"error": str(e), "type": "safety_violation"}
        except Exception as e:
            return {"error": str(e), "type": "execution_error"}

        # Apply transforms (runs on all rows before the 100-row display cap)
        rows: list[dict] = list(result.rows)
        if transforms:
            rows = apply_transforms(rows, transforms)

        columns = list(rows[0].keys()) if rows else result.columns
        total = len(rows)
        warnings: list[str] = list(result.warnings)

        # Execution cap warning: fires when the connector hit the row_limit entirely.
        # Skip spec queries — executor._collect_warnings() already handles them.
        if result.truncated and not is_spec:
            q_upper = query.upper() if isinstance(query, str) else ""
            if "GROUP BY" in q_upper:
                warnings.append(
                    "⚠ EXECUTION CAP — GROUP BY INCOMPLETE: this query hit the 1000-row "
                    "execution limit before all groups were fetched. There are MORE groups "
                    "not included in this result at all. MAX, MIN, and top-N from this "
                    "result are WRONG — the highest-value groups may not be among the "
                    "1000 returned. Fix: add ORDER BY <metric> DESC LIMIT <N> inside "
                    "the SQL to have the database compute the true top-N before the cap."
                )
            else:
                warnings.append(
                    "⚠ EXECUTION CAP: this query hit the 1000-row limit — more rows exist "
                    "but were not fetched. Add WHERE filters or aggregations to reduce scope."
                )

        # Large-result warning with actionable guidance
        if total > 100:
            if total > 5000:
                warnings.append(
                    f"Only the first 100 of {total} rows are shown. "
                    f"A {total}-row result CANNOT be used to find a specific answer — "
                    "you MUST add MIN/MAX/COUNT/GROUP BY aggregations INSIDE your sub-queries "
                    "to compute the answer directly. Do not re-run this query as-is."
                )
            else:
                warnings.append(
                    f"Only the first 100 of {total} rows are shown. "
                    "Refine your query with WHERE filters, use GROUP BY / COUNT / AVG "
                    "aggregations, or add a LIMIT clause to get a targeted result set."
                )

        # Plausibility check
        plausibility_data: dict | None = None
        if self._checker:
            try:
                schema = await self._scanner.scan_all()
                pr = self._checker.check(result, schema, query)
                if pr.warnings:
                    warnings.extend(pr.warnings)
                plausibility_data = {
                    "score": pr.score,
                    "is_plausible": pr.is_plausible,
                    "failure_mode": pr.failure_mode,
                }
            except Exception:
                pass

        # Session learning: reinforce join rules from successful spec queries
        if self._learner:
            try:
                await self._learner.observe(result)
                await self._learner.propose_from_result(result, sid)
            except Exception:
                pass

        # Persist new join rules discovered from agent-constructed spec queries
        if is_spec:
            await self._record_spec_joins(query)

        # Audit
        if self._audit:
            try:
                await self._audit.record(
                    session_id=sid,
                    query=query,
                    databases=result.provenance.get("databases", []),
                    row_count=total,
                    execution_ms=result.total_ms,
                    plausibility_score=plausibility_data["score"] if plausibility_data else 1.0,
                    warnings=warnings,
                    truncated=total > 100,
                )
            except Exception:
                pass

        response: dict[str, Any] = {}
        if warnings:
            response["warnings"] = warnings
        response.update({
            "row_count": total,
            "columns": columns,
            "rows": rows[:100],
            "truncated": total > 100,
            "execution_ms": result.total_ms,
            "provenance": result.provenance,
        })
        if plausibility_data:
            response["plausibility"] = plausibility_data
        return response

    # ── db_schema ─────────────────────────────────────────────────────────────

    async def schema(
        self,
        database: str | None = None,
        table: str | None = None,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        aliases = [database] if database else self._registry.aliases()
        out: dict[str, Any] = {}
        for alias in aliases:
            try:
                tables = await self._scanner.scan(alias, force=force_refresh)
            except Exception as e:
                out[alias] = {"error": str(e)}
                continue
            if table:
                t = tables.get(table)
                out[alias] = (
                    {
                        "table": table,
                        "row_count_approx": t.row_count_approx,
                        "columns": {
                            k: {
                                "dtype": v.dtype,
                                "nullable": v.nullable,
                                "null_rate": v.null_rate,
                                "unique_rate": v.unique_rate,
                                "p50": v.p50,
                                "p95": v.p95,
                                "notes": v.notes,
                            }
                            for k, v in t.columns.items()
                        },
                    }
                    if t
                    else {"error": f"Table '{table}' not found in '{alias}'"}
                )
            else:
                out[alias] = {
                    tname: {
                        "row_count_approx": t.row_count_approx,
                        "column_count": len(t.columns),
                        "columns": list(t.columns.keys()),
                    }
                    for tname, t in tables.items()
                }
        return out

    # ── db_joins ──────────────────────────────────────────────────────────────

    async def joins(
        self,
        discover: bool = False,
        confirm: str | None = None,
        reject: str | None = None,
        sampling_callback: Any = None,
    ) -> dict[str, Any]:
        if confirm:
            await self._join_registry.confirm(confirm, session_id="agent")
            return {"confirmed": confirm}
        if reject:
            await self._join_registry.reject(reject)
            return {"rejected": reject}
        if discover:
            if self._discovery is None:
                return {"error": "Join discovery not configured"}
            schema = await self._scanner.scan_all()
            candidates = await self._discovery.discover(schema, sampling_callback=sampling_callback)
            for c in candidates:
                rule = JoinRule(
                    join_id=c.join_id,
                    db_a=c.db_a, table_a=c.table_a, column_a=c.column_a,
                    db_b=c.db_b, table_b=c.table_b, column_b=c.column_b,
                    transform=c.transform,
                    confidence=c.confidence,
                    confirmed=False,
                )
                if await self._join_registry.get(c.join_id) is None:
                    await self._join_registry.save(rule)
            return {
                "proposed": [
                    {
                        "join_id": c.join_id,
                        "source": f"{c.db_a}.{c.table_a}.{c.column_a}",
                        "target": f"{c.db_b}.{c.table_b}.{c.column_b}",
                        "transform": c.transform,
                        "confidence": c.confidence,
                        "action": "call db_joins(confirm=<join_id>) to confirm",
                    }
                    for c in candidates
                ],
                "count": len(candidates),
            }
        rules = await self._join_registry.get_all()
        return {
            "joins": [
                {
                    "join_id": r.join_id,
                    "source": f"{r.db_a}.{r.table_a}.{r.column_a}",
                    "target": f"{r.db_b}.{r.table_b}.{r.column_b}",
                    "transform": r.transform,
                    "confidence": r.confidence,
                    "confirmed": r.confirmed,
                    "success_count": r.success_count,
                    "failure_count": r.failure_count,
                }
                for r in rules
            ]
        }

    # ── db_verify ─────────────────────────────────────────────────────────────

    async def verify(self, rows: list[dict], query: str = "") -> dict[str, Any]:
        if not self._checker:
            return {"error": "Plausibility checker not configured"}
        result = ExecutionResult(
            rows=rows,
            row_count=len(rows),
            columns=list(rows[0].keys()) if rows else [],
        )
        schema = await self._scanner.scan_all()
        pr = self._checker.check(result, schema, query)
        return {
            "score": pr.score,
            "is_plausible": pr.is_plausible,
            "failure_mode": pr.failure_mode,
            "warnings": pr.warnings,
        }

    # ── db_plan ───────────────────────────────────────────────────────────────

    async def plan_query(
        self,
        query: str,
        databases: list[str] | None = None,
    ) -> dict[str, Any]:
        try:
            plan = await self._planner.plan(query, target_databases=databases)
        except Exception as e:
            return {"error": str(e)}
        cost_estimates: dict[str, float] = {}
        for sq in plan.sub_queries:
            try:
                conn = self._registry.get(sq.db_alias)
                cost_estimates[sq.result_key] = await conn.explain_cost(sq.query)
            except Exception:
                cost_estimates[sq.result_key] = -1.0
        return {
            "sub_queries": [
                {
                    "key": sq.result_key,
                    "database": sq.db_alias,
                    "query": sq.query,
                    "estimated_cost": cost_estimates.get(sq.result_key),
                }
                for sq in plan.sub_queries
            ],
            "join_rules": [
                {
                    "join_id": r.join_id,
                    "source": f"{r.db_a}.{r.table_a}.{r.column_a}",
                    "target": f"{r.db_b}.{r.table_b}.{r.column_b}",
                }
                for r in plan.join_rules
            ],
            "provenance": plan.provenance,
        }

    # ── db_audit ──────────────────────────────────────────────────────────────

    async def audit_log(
        self,
        session_id: str | None = None,
        recent: int = 20,
        replay_id: int | None = None,
    ) -> dict[str, Any]:
        if not self._audit:
            return {"error": "Audit log not configured"}
        if replay_id:
            entry = await self._audit.replay(replay_id)
            if not entry:
                return {"error": f"No audit entry with id={replay_id}"}
            return {
                "id": entry.id, "session_id": entry.session_id, "query": entry.query,
                "databases": entry.databases, "row_count": entry.row_count,
                "execution_ms": entry.execution_ms, "plausibility_score": entry.plausibility_score,
                "warnings": entry.warnings, "logged_at": entry.logged_at,
            }
        entries = (
            await self._audit.get_session(session_id) if session_id
            else await self._audit.recent(recent)
        )
        return {
            "entries": [
                {
                    "id": e.id, "session_id": e.session_id,
                    "query": e.query[:120] + ("..." if len(e.query) > 120 else ""),
                    "databases": e.databases, "row_count": e.row_count,
                    "execution_ms": e.execution_ms, "plausibility_score": e.plausibility_score,
                    "warnings": e.warnings, "logged_at": e.logged_at,
                }
                for e in entries
            ],
            "count": len(entries),
        }

    # ── db_connections ────────────────────────────────────────────────────────

    async def connections(self) -> dict[str, Any]:
        result = []
        for alias, connector in self._registry.all().items():
            status = "unknown"
            try:
                probe = (
                    '{"collection":"_health","pipeline":[{"$limit":1}]}'
                    if connector.db_type.value == "mongodb"
                    else "SELECT 1"
                )
                await connector.execute_query(probe, row_limit=1)
                status = "connected"
            except Exception as e:
                status = f"error: {e}"
            result.append({"alias": alias, "type": connector.db_type.value, "status": status})
        return {"connections": result, "count": len(result)}


    async def _record_spec_joins(self, spec_json: str) -> None:
        """Parse a cross-DB spec query and persist any new join rules discovered from it."""
        if not self._join_registry:
            return
        try:
            import hashlib
            import sqlglot
            import sqlglot.expressions as sg_exp
            from databridge.schema.joins.registry import JoinRule

            spec = json.loads(spec_json)
            join_on = spec.get("join_on", [])
            if not join_on:
                return

            key_to_db: dict[str, str] = {}
            key_to_table: dict[str, str] = {}
            key_to_alias_map: dict[str, dict[str, str]] = {}

            for sq in spec.get("sub_queries", []):
                key = sq.get("key", "")
                db = sq.get("db", "")
                if not (key and db):
                    continue
                key_to_db[key] = db
                try:
                    ast = sqlglot.parse_one(sq["query"])
                    tbls = [t.name for t in ast.find_all(sg_exp.Table)]
                    key_to_table[key] = tbls[0] if tbls else ""
                    alias_map: dict[str, str] = {}
                    for sel in ast.selects:
                        alias = sel.alias or (sel.name if hasattr(sel, "name") else "")
                        if not alias:
                            continue
                        src_cols = [c.name for c in sel.find_all(sg_exp.Column)]
                        if len(src_cols) == 1:
                            alias_map[alias] = src_cols[0]
                        elif not src_cols and hasattr(sel, "name"):
                            alias_map[alias] = sel.name
                    key_to_alias_map[key] = alias_map
                except Exception:
                    key_to_table[key] = ""
                    key_to_alias_map[key] = {}

            for pair in join_on:
                if len(pair) != 2:
                    continue
                left, right = pair
                lkey, lcol = left.split(".", 1) if "." in left else ("", left)
                rkey, rcol = right.split(".", 1) if "." in right else ("", right)
                db_a = key_to_db.get(lkey, "")
                db_b = key_to_db.get(rkey, "")
                if not (db_a and db_b and lcol and rcol):
                    continue
                col_a = key_to_alias_map.get(lkey, {}).get(lcol, lcol)
                col_b = key_to_alias_map.get(rkey, {}).get(rcol, rcol)
                table_a = key_to_table.get(lkey, "")
                table_b = key_to_table.get(rkey, "")
                join_id = hashlib.sha256(
                    f"{db_a}.{table_a}.{col_a}|{db_b}.{table_b}.{col_b}".encode()
                ).hexdigest()[:16]
                rule = JoinRule(
                    join_id=join_id,
                    db_a=db_a, table_a=table_a, column_a=col_a,
                    db_b=db_b, table_b=table_b, column_b=col_b,
                    confidence=0.6,
                )
                if await self._join_registry.get(join_id) is None:
                    await self._join_registry.save(rule)
        except Exception:
            pass


def _normalise_query(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().lower())
