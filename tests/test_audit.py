import pytest
from databridge.audit.log import AuditLog


@pytest.fixture
def audit(tmp_path):
    return AuditLog(tmp_path / "audit.db")


@pytest.mark.asyncio
async def test_record_returns_id(audit):
    entry_id = await audit.record(
        session_id="sess-1",
        query="SELECT id FROM orders",
        databases=["sqlite"],
        row_count=10,
        execution_ms=42.5,
    )
    assert isinstance(entry_id, int)
    assert entry_id >= 1


@pytest.mark.asyncio
async def test_record_and_replay(audit):
    entry_id = await audit.record(
        session_id="sess-replay",
        query="SELECT id FROM customers",
        databases=["mydb"],
        row_count=5,
        execution_ms=10.0,
        plausibility_score=0.95,
        warnings=["test warning"],
    )
    entry = await audit.replay(entry_id)
    assert entry is not None
    assert entry.query == "SELECT id FROM customers"
    assert entry.row_count == 5
    assert entry.plausibility_score == pytest.approx(0.95)
    assert "test warning" in entry.warnings


@pytest.mark.asyncio
async def test_replay_missing_returns_none(audit):
    result = await audit.replay(99999)
    assert result is None


@pytest.mark.asyncio
async def test_get_session(audit):
    await audit.record("sess-a", "SELECT 1", ["db1"], 1, 5.0)
    await audit.record("sess-a", "SELECT 2", ["db1"], 2, 6.0)
    await audit.record("sess-b", "SELECT 3", ["db2"], 3, 7.0)

    entries = await audit.get_session("sess-a")
    assert len(entries) == 2
    assert all(e.session_id == "sess-a" for e in entries)


@pytest.mark.asyncio
async def test_get_session_empty(audit):
    entries = await audit.get_session("no-such-session")
    assert entries == []


@pytest.mark.asyncio
async def test_recent_returns_n(audit):
    for i in range(5):
        await audit.record(f"sess-{i}", f"SELECT {i}", ["db"], i, float(i))

    entries = await audit.recent(3)
    assert len(entries) == 3


@pytest.mark.asyncio
async def test_recent_returns_all_when_fewer_than_n(audit):
    await audit.record("s", "SELECT 1", ["db"], 1, 1.0)
    entries = await audit.recent(50)
    assert len(entries) == 1


@pytest.mark.asyncio
async def test_truncated_flag(audit):
    entry_id = await audit.record(
        session_id="sess-trunc",
        query="SELECT id FROM big_table",
        databases=["db"],
        row_count=10000,
        execution_ms=100.0,
        truncated=True,
    )
    entry = await audit.replay(entry_id)
    assert entry.truncated is True


@pytest.mark.asyncio
async def test_warnings_round_trip(audit):
    warnings = ["possible wrong join key", "zero rows from large table"]
    entry_id = await audit.record(
        "s", "SELECT ...", ["db"], 0, 5.0, warnings=warnings
    )
    entry = await audit.replay(entry_id)
    assert entry.warnings == warnings


@pytest.mark.asyncio
async def test_databases_round_trip(audit):
    dbs = ["postgresql", "mongodb"]
    entry_id = await audit.record("s", "SELECT ...", dbs, 10, 20.0)
    entry = await audit.replay(entry_id)
    assert entry.databases == dbs
