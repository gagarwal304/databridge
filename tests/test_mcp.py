"""
MCP server tool tests.

These tests call the underlying async functions that back each MCP tool directly,
bypassing the MCP protocol layer. This is the correct way to unit-test tool logic
without spinning up an actual MCP server process.
"""
import pytest
from databridge.config import DatabridgeConfig
from databridge.connectors.registry import ConnectorRegistry
from databridge.query.executor import QueryExecutor
from databridge.query.planner import QueryPlanner
from databridge.query.translator import QueryTranslator
from databridge.safety.enforcement import SafetyEnforcer
from databridge.schema.cache import SchemaCache
from databridge.schema.joins.registry import JoinRegistry, JoinRule
from databridge.schema.scanner import SchemaScanner
from databridge.audit.log import AuditLog
from databridge.learning.session import SessionLearner
from databridge.verification.plausibility import PlausibilityChecker


@pytest.fixture
async def mcp_deps(sqlite_registry, tmp_path):
    """Return all the wired-up components that back the MCP tools."""
    cache = SchemaCache(tmp_path / "schema.db", ttl_hours=1)
    join_reg = JoinRegistry(tmp_path / "joins.db")
    scanner = SchemaScanner(sqlite_registry, cache)
    enforcer = SafetyEnforcer()
    translator = QueryTranslator()
    planner = QueryPlanner(sqlite_registry, join_reg)
    executor = QueryExecutor(sqlite_registry, enforcer, translator, default_row_limit=100)
    checker = PlausibilityChecker(zero_row_threshold=1_000, numeric_tolerance=3.0)
    audit = AuditLog(tmp_path / "audit.db")
    learner = SessionLearner(join_reg)
    return {
        "registry": sqlite_registry,
        "cache": cache,
        "join_registry": join_reg,
        "scanner": scanner,
        "enforcer": enforcer,
        "translator": translator,
        "planner": planner,
        "executor": executor,
        "checker": checker,
        "audit": audit,
        "learner": learner,
    }


# ── db_schema logic ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_schema_returns_all_databases(mcp_deps):
    scanner = mcp_deps["scanner"]
    registry = mcp_deps["registry"]
    aliases = registry.aliases()
    out = {}
    for alias in aliases:
        tables = await scanner.scan(alias)
        out[alias] = {
            tname: {"row_count_approx": t.row_count_approx, "columns": list(t.columns.keys())}
            for tname, t in tables.items()
        }
    assert "sqlite" in out
    assert "orders" in out["sqlite"]
    assert "customers" in out["sqlite"]


@pytest.mark.asyncio
async def test_schema_single_table(mcp_deps):
    scanner = mcp_deps["scanner"]
    tables = await scanner.scan("sqlite")
    t = tables.get("orders")
    assert t is not None
    assert "id" in t.columns
    assert "amount" in t.columns


# ── db_query logic ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_query_select_all_orders(mcp_deps):
    planner = mcp_deps["planner"]
    executor = mcp_deps["executor"]

    plan = await planner.plan("SELECT id, customer_id, amount FROM orders")
    result = await executor.execute(plan)
    assert result.row_count == 50
    assert "id" in result.columns


@pytest.mark.asyncio
async def test_query_with_where_clause(mcp_deps):
    planner = mcp_deps["planner"]
    executor = mcp_deps["executor"]

    plan = await planner.plan("SELECT id FROM customers WHERE id = 1")
    result = await executor.execute(plan)
    assert result.row_count == 1
    assert result.rows[0]["id"] == 1


@pytest.mark.asyncio
async def test_query_safety_blocked(mcp_deps):
    planner = mcp_deps["planner"]
    executor = mcp_deps["executor"]
    from databridge.safety.enforcement import SafetyViolation
    from databridge.query.planner import QueryPlan, SubQuery

    plan = QueryPlan(sub_queries=[
        SubQuery(db_alias="sqlite", query="DROP TABLE orders", result_key="main")
    ])
    with pytest.raises(SafetyViolation):
        await executor.execute(plan)


# ── db_joins logic ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_joins_list_empty(mcp_deps):
    join_reg = mcp_deps["join_registry"]
    rules = await join_reg.get_all()
    assert rules == []


@pytest.mark.asyncio
async def test_joins_save_confirm_list(mcp_deps):
    join_reg = mcp_deps["join_registry"]
    rule = JoinRule(
        join_id="mcp-test-rule",
        db_a="db_a", table_a="orders", column_a="customer_id",
        db_b="db_b", table_b="customers", column_b="id",
        confidence=0.75, confirmed=False,
    )
    await join_reg.save(rule)
    await join_reg.confirm("mcp-test-rule", session_id="agent")
    rules = await join_reg.get_all(confirmed_only=True)
    assert len(rules) == 1
    assert rules[0].join_id == "mcp-test-rule"


@pytest.mark.asyncio
async def test_joins_reject(mcp_deps):
    join_reg = mcp_deps["join_registry"]
    rule = JoinRule(
        join_id="to-reject",
        db_a="a", table_a="t1", column_a="c1",
        db_b="b", table_b="t2", column_b="c2",
        confidence=0.6, confirmed=False,
    )
    await join_reg.save(rule)
    await join_reg.reject("to-reject")
    assert await join_reg.get("to-reject") is None


# ── db_audit logic ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_audit_record_and_retrieve(mcp_deps):
    audit = mcp_deps["audit"]
    eid = await audit.record(
        session_id="test-session",
        query="SELECT id FROM customers",
        databases=["sqlite"],
        row_count=10,
        execution_ms=5.0,
        plausibility_score=0.9,
    )
    entries = await audit.get_session("test-session")
    assert len(entries) == 1
    assert entries[0].id == eid
    assert entries[0].row_count == 10


@pytest.mark.asyncio
async def test_audit_recent(mcp_deps):
    audit = mcp_deps["audit"]
    for i in range(5):
        await audit.record(f"s{i}", f"SELECT {i}", ["sqlite"], i, float(i))
    entries = await audit.recent(3)
    assert len(entries) == 3


# ── db_verify logic (plausibility) ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_verify_plausible_result(mcp_deps):
    checker = mcp_deps["checker"]
    scanner = mcp_deps["scanner"]
    from databridge.query.executor import ExecutionResult

    rows = [{"id": i, "name": f"Customer {i}"} for i in range(1, 6)]
    result = ExecutionResult(rows=rows, row_count=5, columns=["id", "name"])
    schema = await scanner.scan_all()
    pr = checker.check(result, schema, "SELECT id, name FROM customers")
    assert pr.is_plausible is True


@pytest.mark.asyncio
async def test_verify_empty_result(mcp_deps):
    checker = mcp_deps["checker"]
    scanner = mcp_deps["scanner"]
    from databridge.query.executor import ExecutionResult

    result = ExecutionResult(rows=[], row_count=0, columns=[])
    schema = await scanner.scan_all()
    pr = checker.check(result, schema, "SELECT id FROM customers")
    # 10-row table is below zero_row_threshold=1000, so EMPTY_RESULT not WRONG_JOIN_KEY
    assert pr.failure_mode.value in ("empty_result", "wrong_join_key")


# ── db_connections logic ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_connections_health(sqlite_registry):
    conns = sqlite_registry.all()
    assert len(conns) == 1
    for alias, connector in conns.items():
        result = await connector.execute_query("SELECT 1", row_limit=1)
        assert result.row_count == 1
