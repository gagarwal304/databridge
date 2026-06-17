import pytest
from databridge.connectors.base import DbType, QueryResult, TableMeta
from databridge.connectors.registry import ConnectorRegistry
from databridge.connectors.sqlite_conn import SQLiteConnector


@pytest.mark.asyncio
async def test_sqlite_connect_disconnect(tmp_path):
    db = tmp_path / "test.db"
    conn = SQLiteConnector(f"sqlite:///{db}", "test")
    await conn.connect()
    await conn.disconnect()
    assert conn._conn is None


@pytest.mark.asyncio
async def test_sqlite_execute_query(sqlite_registry):
    conn = sqlite_registry.get("sqlite")
    result = await conn.execute_query("SELECT id, amount FROM orders LIMIT 5")
    assert isinstance(result, QueryResult)
    assert result.row_count == 5
    assert "id" in result.columns
    assert result.database == "sqlite"


@pytest.mark.asyncio
async def test_sqlite_row_limit(sqlite_registry):
    conn = sqlite_registry.get("sqlite")
    result = await conn.execute_query("SELECT id FROM orders", row_limit=3)
    assert result.row_count == 3
    assert result.truncated is True


@pytest.mark.asyncio
async def test_sqlite_no_truncation_when_under_limit(sqlite_registry):
    conn = sqlite_registry.get("sqlite")
    result = await conn.execute_query("SELECT id FROM customers", row_limit=100)
    assert result.truncated is False


@pytest.mark.asyncio
async def test_sqlite_introspect_schema(sqlite_registry):
    conn = sqlite_registry.get("sqlite")
    schema = await conn.introspect_schema()
    assert "orders" in schema
    assert "customers" in schema
    orders = schema["orders"]
    assert isinstance(orders, TableMeta)
    assert orders.row_count_approx == 50
    assert "id" in orders.columns
    assert "amount" in orders.columns


@pytest.mark.asyncio
async def test_sqlite_sample_column(sqlite_registry):
    conn = sqlite_registry.get("sqlite")
    samples = await conn.sample_column("orders", "amount", 5)
    assert len(samples) <= 5
    assert all(isinstance(v, float) for v in samples)


@pytest.mark.asyncio
async def test_sqlite_db_type(tmp_path):
    db = tmp_path / "t.db"
    conn = SQLiteConnector(f"sqlite:///{db}", "alias")
    assert conn.db_type == DbType.SQLITE


@pytest.mark.asyncio
async def test_sqlite_explain_cost(sqlite_registry):
    conn = sqlite_registry.get("sqlite")
    cost = await conn.explain_cost("SELECT id FROM orders")
    assert isinstance(cost, float)


def test_registry_get(sqlite_registry):
    conn = sqlite_registry.get("sqlite")
    assert conn is not None
    assert conn.db_alias == "sqlite"


def test_registry_all(sqlite_registry):
    all_conns = sqlite_registry.all()
    assert "sqlite" in all_conns


def test_registry_aliases(sqlite_registry):
    assert "sqlite" in sqlite_registry.aliases()


@pytest.mark.asyncio
async def test_registry_from_uris_multi(tmp_path):
    db1 = tmp_path / "a.db"
    db2 = tmp_path / "b.db"
    reg = ConnectorRegistry.from_uris([f"sqlite:///{db1}", f"sqlite:///{db2}"])
    assert len(reg.aliases()) == 2


@pytest.mark.asyncio
async def test_not_connected_raises(tmp_path):
    db = tmp_path / "noop.db"
    conn = SQLiteConnector(f"sqlite:///{db}", "noop")
    with pytest.raises(RuntimeError, match="not connected"):
        await conn.execute_query("SELECT 1")
