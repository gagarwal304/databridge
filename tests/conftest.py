from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from databridge.config import DatabridgeConfig, reset_config
from databridge.connectors.registry import ConnectorRegistry
from databridge.schema.cache import SchemaCache
from databridge.schema.joins.registry import JoinRegistry


@pytest.fixture(autouse=True)
def reset_global_config():
    reset_config()
    yield
    reset_config()


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def test_config(tmp_path: Path) -> DatabridgeConfig:
    db_path = tmp_path / "test.db"
    return DatabridgeConfig(
        database_uris=[f"sqlite:///{db_path}"],
        schema_cache_path=str(tmp_path / "schema.db"),
        audit_log_path=str(tmp_path / "audit.db"),
        default_row_limit=100,
        value_sample_size=10,
    )


@pytest.fixture
async def sqlite_registry(tmp_path: Path) -> ConnectorRegistry:
    db_path = tmp_path / "test.db"
    import aiosqlite
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute(
            "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL)"
        )
        await conn.execute(
            "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, email TEXT)"
        )
        await conn.executemany(
            "INSERT INTO orders VALUES (?, ?, ?)",
            [(i, i % 10 + 1, float(i * 100)) for i in range(1, 51)],
        )
        await conn.executemany(
            "INSERT INTO customers VALUES (?, ?, ?)",
            [(i, f"Customer {i}", f"c{i}@example.com") for i in range(1, 11)],
        )
        await conn.commit()

    registry = ConnectorRegistry.from_uris([f"sqlite:///{db_path}"])
    await registry.connect_all()
    yield registry
    await registry.disconnect_all()


@pytest.fixture
async def schema_cache(tmp_path: Path) -> SchemaCache:
    return SchemaCache(tmp_path / "schema.db", ttl_hours=1)


@pytest.fixture
async def join_registry(tmp_path: Path) -> JoinRegistry:
    return JoinRegistry(tmp_path / "joins.db")
