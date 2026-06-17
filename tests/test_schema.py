import pytest
from databridge.connectors.base import ColumnMeta, TableMeta
from databridge.schema.cache import SchemaCache
from databridge.schema.scanner import SchemaScanner


def _make_table(name: str, row_count: int = 10) -> TableMeta:
    t = TableMeta(name=name, row_count_approx=row_count)
    t.columns["id"] = ColumnMeta(name="id", dtype="INTEGER", nullable=False)
    t.columns["value"] = ColumnMeta(name="value", dtype="TEXT", nullable=True)
    return t


@pytest.mark.asyncio
async def test_cache_save_and_load(schema_cache):
    table = _make_table("products", row_count=500)
    await schema_cache.save("mydb", {"products": table})

    loaded = await schema_cache.load("mydb")
    assert loaded is not None
    assert "products" in loaded
    assert loaded["products"].row_count_approx == 500
    assert "id" in loaded["products"].columns
    assert loaded["products"].columns["id"].dtype == "INTEGER"


@pytest.mark.asyncio
async def test_cache_load_empty_on_miss(schema_cache):
    result = await schema_cache.load("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_cache_invalidate(schema_cache):
    table = _make_table("events")
    await schema_cache.save("evdb", {"events": table})
    await schema_cache.invalidate("evdb")
    result = await schema_cache.load("evdb")
    assert result is None


@pytest.mark.asyncio
async def test_cache_get_hash(schema_cache):
    table = _make_table("logs")
    await schema_cache.save("logdb", {"logs": table})
    h = await schema_cache.get_hash("logdb", "logs")
    assert h is not None
    assert len(h) == 16


@pytest.mark.asyncio
async def test_cache_hash_missing_table(schema_cache):
    h = await schema_cache.get_hash("logdb", "ghost")
    assert h is None


@pytest.mark.asyncio
async def test_cache_ttl_expired(tmp_path):
    cache = SchemaCache(tmp_path / "ttl.db", ttl_hours=0)
    table = _make_table("stale")
    await cache.save("db", {"stale": table})
    loaded = await cache.load("db")
    assert loaded is None


@pytest.mark.asyncio
async def test_scanner_scan(sqlite_registry, schema_cache):
    scanner = SchemaScanner(sqlite_registry, schema_cache)
    tables = await scanner.scan("sqlite")
    assert "orders" in tables
    assert "customers" in tables


@pytest.mark.asyncio
async def test_scanner_scan_all(sqlite_registry, schema_cache):
    scanner = SchemaScanner(sqlite_registry, schema_cache)
    all_schemas = await scanner.scan_all()
    assert "sqlite" in all_schemas
    assert "orders" in all_schemas["sqlite"]


@pytest.mark.asyncio
async def test_scanner_force_refresh(sqlite_registry, schema_cache):
    scanner = SchemaScanner(sqlite_registry, schema_cache)
    first = await scanner.scan("sqlite")
    second = await scanner.scan("sqlite", force=True)
    assert set(first.keys()) == set(second.keys())


@pytest.mark.asyncio
async def test_scanner_diff_no_change(sqlite_registry, schema_cache):
    scanner = SchemaScanner(sqlite_registry, schema_cache)
    await scanner.scan("sqlite")
    diff = await scanner.diff("sqlite")
    assert diff == []
