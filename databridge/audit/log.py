from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiosqlite
from contextlib import asynccontextmanager


_DDL = """
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    query TEXT NOT NULL,
    databases TEXT NOT NULL,
    estimated_cost REAL,
    row_count INTEGER,
    execution_ms REAL,
    plausibility_score REAL,
    warnings TEXT,
    truncated INTEGER DEFAULT 0,
    logged_at REAL NOT NULL
);
"""


@dataclass
class AuditEntry:
    id: int
    session_id: str
    query: str
    databases: list[str]
    estimated_cost: float | None
    row_count: int
    execution_ms: float
    plausibility_score: float | None
    warnings: list[str]
    truncated: bool
    logged_at: float


class AuditLog:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def _db(self):
        async with aiosqlite.connect(self._path) as conn:
            conn.row_factory = aiosqlite.Row
            await conn.executescript(_DDL)
            await conn.commit()
            yield conn

    async def record(
        self,
        session_id: str,
        query: str,
        databases: list[str],
        row_count: int,
        execution_ms: float,
        estimated_cost: float | None = None,
        plausibility_score: float | None = None,
        warnings: list[str] | None = None,
        truncated: bool = False,
    ) -> int:
        import json
        async with self._db() as conn:
            cur = await conn.execute(
                """INSERT INTO audit_log
                   (session_id, query, databases, estimated_cost, row_count,
                    execution_ms, plausibility_score, warnings, truncated, logged_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    query,
                    json.dumps(databases),
                    estimated_cost,
                    row_count,
                    execution_ms,
                    plausibility_score,
                    json.dumps(warnings or []),
                    int(truncated),
                    time.time(),
                ),
            )
            await conn.commit()
            return cur.lastrowid  # type: ignore[return-value]

    async def get_session(self, session_id: str) -> list[AuditEntry]:
        import json
        async with self._db() as conn:
            async with conn.execute(
                "SELECT * FROM audit_log WHERE session_id = ? ORDER BY logged_at",
                (session_id,),
            ) as cur:
                rows = await cur.fetchall()
        return [_row_to_entry(r) for r in rows]

    async def replay(self, entry_id: int) -> AuditEntry | None:
        async with self._db() as conn:
            async with conn.execute(
                "SELECT * FROM audit_log WHERE id = ?", (entry_id,)
            ) as cur:
                row = await cur.fetchone()
        return _row_to_entry(row) if row else None

    async def recent(self, n: int = 20) -> list[AuditEntry]:
        async with self._db() as conn:
            async with conn.execute(
                "SELECT * FROM audit_log ORDER BY logged_at DESC LIMIT ?", (n,)
            ) as cur:
                rows = await cur.fetchall()
        return [_row_to_entry(r) for r in rows]


def _row_to_entry(row: aiosqlite.Row) -> AuditEntry:
    import json
    return AuditEntry(
        id=row["id"],
        session_id=row["session_id"],
        query=row["query"],
        databases=json.loads(row["databases"]),
        estimated_cost=row["estimated_cost"],
        row_count=row["row_count"],
        execution_ms=row["execution_ms"],
        plausibility_score=row["plausibility_score"],
        warnings=json.loads(row["warnings"]),
        truncated=bool(row["truncated"]),
        logged_at=row["logged_at"],
    )
