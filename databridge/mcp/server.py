from __future__ import annotations

from typing import Any

import mcp.types as mcp_types
from mcp.server.fastmcp import Context, FastMCP

from databridge.audit.log import AuditLog
from databridge.config import DatabridgeConfig
from databridge.connectors.registry import ConnectorRegistry
from databridge.engine import DataBridgeEngine
from databridge.learning.session import SessionLearner
from databridge.query.executor import QueryExecutor
from databridge.query.planner import QueryPlanner
from databridge.query.translator import QueryTranslator
from databridge.safety.enforcement import SafetyEnforcer
from databridge.schema.cache import SchemaCache
from databridge.schema.joins.discovery import JoinDiscovery
from databridge.schema.joins.registry import JoinRegistry
from databridge.schema.scanner import SchemaScanner
from databridge.verification.plausibility import PlausibilityChecker


def create_server(config: DatabridgeConfig) -> FastMCP:
    config.ensure_dirs()

    registry = ConnectorRegistry.from_uris(config.database_uris)
    cache = SchemaCache(config.resolved_cache_path(), config.schema_cache_ttl_hours)
    join_registry = JoinRegistry(config.resolved_cache_path().parent / "joins.db")
    scanner = SchemaScanner(registry, cache)
    enforcer = SafetyEnforcer()
    translator = QueryTranslator()
    planner = QueryPlanner(registry, join_registry)
    executor = QueryExecutor(registry, enforcer, translator, config.default_row_limit)
    checker = PlausibilityChecker(config.zero_row_warning_threshold, config.numeric_range_tolerance)
    audit = AuditLog(config.resolved_audit_path())
    learner = SessionLearner(join_registry)
    discovery = JoinDiscovery(
        registry,
        name_similarity_threshold=config.name_similarity_threshold,
        value_sample_size=config.value_sample_size,
        overlap_threshold=config.overlap_threshold,
        min_confidence=config.min_confidence_to_propose,
    )

    engine = DataBridgeEngine(
        registry=registry,
        scanner=scanner,
        planner=planner,
        executor=executor,
        join_registry=join_registry,
        checker=checker,
        audit=audit,
        learner=learner,
        discovery=discovery,
        max_cost_budget=config.max_cost_budget,
    )

    mcp = FastMCP(config.mcp_server_name)

    # ── db_query ──────────────────────────────────────────────────────────────
    @mcp.tool()
    async def db_query(
        query: str,
        databases: list[str] | None = None,
        session_id: str | None = None,
        row_limit: int | None = None,
    ) -> dict[str, Any]:
        """
        Execute a SQL SELECT query (or MongoDB aggregation pipeline JSON) across one
        or more connected databases.

        Returns rows, column names, plausibility score, warnings, and provenance.

        For CROSS-DATABASE queries or post-processing, pass a JSON spec:
          {"sub_queries": [
            {"db": "postgresql", "query": "SELECT id, title FROM title WHERE ...", "key": "pg"},
            {"db": "sqlite",     "query": "SELECT movie_id, name FROM cast_info WHERE ...", "key": "sq"}
          ], "join_on": [["pg.id", "sq.movie_id"]], "transform": [...]}

        transform ops (applied to merged rows in order):
          {"op": "extract_number", "column": "col", "metric": "stars", "output": "stars"}
            — parse numeric metric from prose text (stars/forks/issues); handles all format variants
          {"op": "top_n_with_ties", "column": "col", "n": 5}
            — top-N rows including ALL tied rows at position N
          {"op": "sort", "column": "col", "direction": "desc"} — sort rows
          {"op": "cast_number", "column": "col", "output": "col_int"} — strip commas/spaces, cast to int
          {"op": "project_name_from_text", "column": "col", "output": "proj"}
            — extract owner/repo from 'The project X on GitHub' prose
          {"op": "json_array_extract", "column": "cpc", "field": "code", "output": "codes"}
            — extract a named field from every element of a JSON array stored as text
          {"op": "json_explode", "column": "codes", "output": "code"}
            — expand a list column into multiple rows (one row per element)
          {"op": "parse_date", "column": "filing_date", "output_format": "%Y", "output": "year"}
            — parse natural-language dates (fuzzy) and format via strftime
          {"op": "group_count", "group_by": ["code", "year"], "output": "count"}
            — count rows per unique group, collapse to one row per group
          {"op": "compute_ema", "group_by": "code", "sort_by": "year",
           "value_col": "count", "alpha": 0.2, "output": "ema"}
            — EMA per group; fill_gaps=true (default) zero-fills missing integer steps;
              summarize=true (default) returns one row per group at peak EMA step
        """
        return await engine.query(query, databases, session_id, row_limit)

    # ── db_schema ─────────────────────────────────────────────────────────────
    @mcp.tool()
    async def db_schema(
        database: str | None = None,
        table: str | None = None,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """
        Return schema context for a database, specific table, or all databases.
        Results are cached — use force_refresh=True to re-scan.
        """
        return await engine.schema(database, table, force_refresh)

    # ── db_joins ──────────────────────────────────────────────────────────────
    @mcp.tool()
    async def db_joins(
        discover: bool = False,
        confirm: str | None = None,
        reject: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """
        List known cross-database join relationships.

        discover=True  — run auto-discovery (WordNet + rapidfuzz + transform grammar,
                         plus MCP sampling callback for ambiguous pairs).
        confirm=<id>   — confirm a proposed join by its join_id.
        reject=<id>    — reject and remove a proposed join.
        """
        sampling_callback = None
        if discover and ctx is not None:
            from databridge.schema.joins.transforms import TRANSFORM_GRAMMAR as _TG
            _valid = {name for name, _ in _TG}

            async def sampling_callback(col_a: str, vals_a: list, col_b: str, vals_b: list):
                prompt = (
                    f"Column A ({col_a}) sample values: {vals_a[:10]}\n"
                    f"Column B ({col_b}) sample values: {vals_b[:10]}\n\n"
                    "These columns are name-similar but the transform between their values "
                    "is not obvious. What string transform converts A values to match B values?\n"
                    f"Reply with exactly one word from: {', '.join(sorted(_valid))}, or 'none'."
                )
                try:
                    result = await ctx.session.create_message(
                        messages=[mcp_types.SamplingMessage(
                            role="user",
                            content=mcp_types.TextContent(type="text", text=prompt),
                        )],
                        max_tokens=20,
                    )
                    text = ""
                    if hasattr(result.content, "text"):
                        text = result.content.text.strip().lower()
                    return text if text in _valid else None
                except Exception:
                    return None

        return await engine.joins(discover, confirm, reject, sampling_callback)

    # ── db_plan ───────────────────────────────────────────────────────────────
    @mcp.tool()
    async def db_plan(
        query: str,
        databases: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Return the execution plan for a query without running it.
        Shows which databases will be queried and which join rules will be applied.
        """
        return await engine.plan_query(query, databases)

    # ── db_verify ─────────────────────────────────────────────────────────────
    @mcp.tool()
    async def db_verify(
        rows: list[dict],
        query: str = "",
    ) -> dict[str, Any]:
        """
        Check plausibility of a result set against known schema statistics.
        Returns a score (0–1), failure mode classification, and specific warnings.
        """
        return await engine.verify(rows, query)

    # ── db_audit ──────────────────────────────────────────────────────────────
    @mcp.tool()
    async def db_audit(
        session_id: str | None = None,
        recent: int = 20,
        replay_id: int | None = None,
    ) -> dict[str, Any]:
        """
        Get query history.

        session_id  — return all queries for a specific session.
        recent      — return N most recent queries across all sessions (default 20).
        replay_id   — return the entry with this ID for replay.
        """
        return await engine.audit_log(session_id, recent, replay_id)

    # ── db_connections ────────────────────────────────────────────────────────
    @mcp.tool()
    async def db_connections() -> dict[str, Any]:
        """List all active database connections and their status."""
        return await engine.connections()

    return mcp
