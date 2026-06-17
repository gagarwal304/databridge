import pytest
from databridge.query.translator import QueryTranslator
from databridge.query.planner import QueryPlanner, QueryPlan, SubQuery
from databridge.query.executor import QueryExecutor, ExecutionResult
from databridge.safety.enforcement import SafetyEnforcer, SafetyViolation
from databridge.connectors.base import DbType


# ── QueryTranslator ───────────────────────────────────────────────────────────

@pytest.fixture
def translator():
    return QueryTranslator()


def test_translate_passthrough_same_dialect(translator):
    q = "SELECT id FROM orders"
    result = translator.translate(q, "sqlite", DbType.SQLITE)
    assert "SELECT" in result


def test_inject_limit_adds_limit(translator):
    q = "SELECT id FROM orders"
    result = translator.inject_limit(q, 100, dialect="sqlite")
    assert "100" in result


def test_inject_limit_preserves_existing(translator):
    q = "SELECT id FROM orders LIMIT 5"
    result = translator.inject_limit(q, 100, dialect="sqlite")
    assert "LIMIT 5" in result
    assert "100" not in result


def test_is_aggregate_query_group_by(translator):
    q = "SELECT status, COUNT(*) FROM orders GROUP BY status"
    assert translator.is_aggregate_query(q, dialect="sqlite") is True


def test_is_aggregate_query_count_no_group(translator):
    q = "SELECT COUNT(*) FROM orders"
    assert translator.is_aggregate_query(q, dialect="sqlite") is True


def test_is_aggregate_query_plain_select(translator):
    q = "SELECT id, amount FROM orders"
    assert translator.is_aggregate_query(q, dialect="sqlite") is False


def test_push_aggregation_removes_limit(translator):
    q = "SELECT status, COUNT(*) FROM orders GROUP BY status LIMIT 10"
    result = translator.push_aggregation(q, dialect="sqlite")
    assert "LIMIT" not in result


def test_translate_mongodb_passthrough(translator):
    q = '{"collection": "orders", "pipeline": []}'
    result = translator.translate(q, "mongo", DbType.MONGODB)
    assert result == q


# ── QueryPlanner ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_planner_single_database(sqlite_registry, join_registry):
    planner = QueryPlanner(sqlite_registry, join_registry)
    plan = await planner.plan("SELECT id FROM orders")
    assert len(plan.sub_queries) == 1
    assert plan.sub_queries[0].db_alias == "sqlite"
    assert plan.provenance["strategy"] == "single_database"


@pytest.mark.asyncio
async def test_planner_spec_format(sqlite_registry, join_registry):
    import json
    spec = {
        "sub_queries": [
            {"db": "sqlite", "query": "SELECT id FROM orders", "key": "orders"},
        ],
        "join_on": [],
    }
    planner = QueryPlanner(sqlite_registry, join_registry)
    plan = await planner.plan(json.dumps(spec))
    assert plan.provenance["strategy"] == "spec_plan"
    assert len(plan.sub_queries) == 1
    assert plan.sub_queries[0].result_key == "orders"


# ── QueryExecutor ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_executor_basic_query(sqlite_registry, join_registry):
    planner = QueryPlanner(sqlite_registry, join_registry)
    enforcer = SafetyEnforcer()
    translator = QueryTranslator()
    executor = QueryExecutor(sqlite_registry, enforcer, translator, default_row_limit=50)

    plan = await planner.plan("SELECT id, amount FROM orders")
    result = await executor.execute(plan)

    assert isinstance(result, ExecutionResult)
    assert result.row_count > 0
    assert "id" in result.columns
    assert result.total_ms > 0


@pytest.mark.asyncio
async def test_executor_blocks_write(sqlite_registry, join_registry):
    planner = QueryPlanner(sqlite_registry, join_registry)
    enforcer = SafetyEnforcer()
    translator = QueryTranslator()
    executor = QueryExecutor(sqlite_registry, enforcer, translator)

    plan = QueryPlan(sub_queries=[
        SubQuery(db_alias="sqlite", query="DELETE FROM orders", result_key="main")
    ])
    with pytest.raises(SafetyViolation):
        await executor.execute(plan)


@pytest.mark.asyncio
async def test_executor_aggregate_no_limit(sqlite_registry, join_registry):
    planner = QueryPlanner(sqlite_registry, join_registry)
    enforcer = SafetyEnforcer()
    translator = QueryTranslator()
    executor = QueryExecutor(sqlite_registry, enforcer, translator, default_row_limit=5)

    plan = await planner.plan("SELECT customer_id, SUM(amount) as total FROM orders GROUP BY customer_id")
    result = await executor.execute(plan)
    # All 10 customers should appear (aggregate should not be limited to 5)
    assert result.row_count == 10


@pytest.mark.asyncio
async def test_executor_respects_row_limit(sqlite_registry, join_registry):
    planner = QueryPlanner(sqlite_registry, join_registry)
    enforcer = SafetyEnforcer()
    translator = QueryTranslator()
    executor = QueryExecutor(sqlite_registry, enforcer, translator, default_row_limit=10_000)

    plan = await planner.plan("SELECT id FROM orders")
    result = await executor.execute(plan, row_limit=3)
    assert result.row_count == 3
    assert result.truncated is True


@pytest.mark.asyncio
async def test_executor_provenance(sqlite_registry, join_registry):
    planner = QueryPlanner(sqlite_registry, join_registry)
    executor = QueryExecutor(sqlite_registry, SafetyEnforcer(), QueryTranslator())

    plan = await planner.plan("SELECT id FROM customers")
    result = await executor.execute(plan)
    assert "databases" in result.provenance
    assert "sqlite" in result.provenance["databases"]
