from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import aiosqlite
from contextlib import asynccontextmanager


_DDL = """
CREATE TABLE IF NOT EXISTS join_rules (
    join_id TEXT PRIMARY KEY,
    db_a TEXT NOT NULL,
    table_a TEXT NOT NULL,
    column_a TEXT NOT NULL,
    db_b TEXT NOT NULL,
    table_b TEXT NOT NULL,
    column_b TEXT NOT NULL,
    transform TEXT,
    confidence REAL DEFAULT 0.0,
    confirmed INTEGER DEFAULT 0,
    verified_at REAL,
    verified_by TEXT,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0
);
"""


@dataclass
class JoinRule:
    join_id: str
    db_a: str
    table_a: str
    column_a: str
    db_b: str
    table_b: str
    column_b: str
    transform: str | None = None
    confidence: float = 0.0
    confirmed: bool = False
    verified_at: float | None = None
    verified_by: str | None = None
    success_count: int = 0
    failure_count: int = 0

    @property
    def is_reliable(self) -> bool:
        return self.confirmed and self.confidence >= 0.70

    def record_success(self) -> None:
        self.success_count += 1
        self.confidence = min(1.0, self.confidence + 0.02)

    def record_failure(self) -> None:
        self.failure_count += 1
        self.confidence = max(0.0, self.confidence - 0.05)


class JoinRegistry:
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

    async def save(self, rule: JoinRule) -> None:
        async with self._db() as conn:
            await conn.execute(
                """INSERT OR REPLACE INTO join_rules
                   (join_id, db_a, table_a, column_a, db_b, table_b, column_b,
                    transform, confidence, confirmed, verified_at, verified_by,
                    success_count, failure_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    rule.join_id, rule.db_a, rule.table_a, rule.column_a,
                    rule.db_b, rule.table_b, rule.column_b,
                    rule.transform, rule.confidence, int(rule.confirmed),
                    rule.verified_at, rule.verified_by,
                    rule.success_count, rule.failure_count,
                ),
            )
            await conn.commit()

    async def save_if_new(self, rule: JoinRule) -> None:
        """Insert only if no rule with this join_id exists — never overwrites confirmed rules."""
        async with self._db() as conn:
            await conn.execute(
                """INSERT OR IGNORE INTO join_rules
                   (join_id, db_a, table_a, column_a, db_b, table_b, column_b,
                    transform, confidence, confirmed, verified_at, verified_by,
                    success_count, failure_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    rule.join_id, rule.db_a, rule.table_a, rule.column_a,
                    rule.db_b, rule.table_b, rule.column_b,
                    rule.transform, rule.confidence, int(rule.confirmed),
                    rule.verified_at, rule.verified_by,
                    rule.success_count, rule.failure_count,
                ),
            )
            await conn.commit()

    async def confirm(self, join_id: str, session_id: str = "human") -> None:
        async with self._db() as conn:
            await conn.execute(
                """UPDATE join_rules SET confirmed = 1, verified_at = ?, verified_by = ?
                   WHERE join_id = ?""",
                (time.time(), session_id, join_id),
            )
            await conn.commit()

    async def reject(self, join_id: str) -> None:
        async with self._db() as conn:
            await conn.execute("DELETE FROM join_rules WHERE join_id = ?", (join_id,))
            await conn.commit()

    async def replace_for_tables(self, rule: JoinRule) -> None:
        """Remove all existing joins for a table pair and save the new authoritative rule.

        Called by Phase 3 runtime detection to evict stale or false-positive joins that
        schema-level discovery may have written (e.g. rating↔rating_number superseded by
        the real book_id↔purchase_id join detected from actual sample data).
        """
        async with self._db() as conn:
            await conn.execute(
                """DELETE FROM join_rules
                   WHERE (db_a = ? AND table_a = ? AND db_b = ? AND table_b = ?)
                      OR (db_a = ? AND table_a = ? AND db_b = ? AND table_b = ?)""",
                (
                    rule.db_a, rule.table_a, rule.db_b, rule.table_b,
                    rule.db_b, rule.table_b, rule.db_a, rule.table_a,
                ),
            )
            await conn.execute(
                """INSERT INTO join_rules
                   (join_id, db_a, table_a, column_a, db_b, table_b, column_b,
                    transform, confidence, confirmed, verified_at, verified_by,
                    success_count, failure_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    rule.join_id, rule.db_a, rule.table_a, rule.column_a,
                    rule.db_b, rule.table_b, rule.column_b,
                    rule.transform, rule.confidence, int(rule.confirmed),
                    rule.verified_at, rule.verified_by,
                    rule.success_count, rule.failure_count,
                ),
            )
            await conn.commit()

    async def record_outcome(self, join_id: str, success: bool) -> None:
        async with self._db() as conn:
            if success:
                await conn.execute(
                    """UPDATE join_rules
                       SET success_count = success_count + 1,
                           confidence = MIN(1.0, confidence + 0.02)
                       WHERE join_id = ?""",
                    (join_id,),
                )
            else:
                await conn.execute(
                    """UPDATE join_rules
                       SET failure_count = failure_count + 1,
                           confidence = MAX(0.0, confidence - 0.05)
                       WHERE join_id = ?""",
                    (join_id,),
                )
            await conn.commit()

    async def get_all(self, confirmed_only: bool = False) -> list[JoinRule]:
        async with self._db() as conn:
            query = "SELECT * FROM join_rules"
            if confirmed_only:
                query += " WHERE confirmed = 1"
            query += " ORDER BY confidence DESC"
            async with conn.execute(query) as cur:
                rows = await cur.fetchall()
        return [_row_to_rule(r) for r in rows]

    async def get(self, join_id: str) -> JoinRule | None:
        async with self._db() as conn:
            async with conn.execute(
                "SELECT * FROM join_rules WHERE join_id = ?", (join_id,)
            ) as cur:
                row = await cur.fetchone()
        return _row_to_rule(row) if row else None

    async def find_for_tables(
        self, db_a: str, table_a: str, db_b: str, table_b: str
    ) -> list[JoinRule]:
        async with self._db() as conn:
            async with conn.execute(
                """SELECT * FROM join_rules
                   WHERE (db_a = ? AND table_a = ? AND db_b = ? AND table_b = ?)
                      OR (db_a = ? AND table_a = ? AND db_b = ? AND table_b = ?)
                   ORDER BY confidence DESC""",
                (db_a, table_a, db_b, table_b, db_b, table_b, db_a, table_a),
            ) as cur:
                rows = await cur.fetchall()
        return [_row_to_rule(r) for r in rows]


def _row_to_rule(row: aiosqlite.Row) -> JoinRule:
    return JoinRule(
        join_id=row["join_id"],
        db_a=row["db_a"],
        table_a=row["table_a"],
        column_a=row["column_a"],
        db_b=row["db_b"],
        table_b=row["table_b"],
        column_b=row["column_b"],
        transform=row["transform"],
        confidence=row["confidence"],
        confirmed=bool(row["confirmed"]),
        verified_at=row["verified_at"],
        verified_by=row["verified_by"],
        success_count=row["success_count"],
        failure_count=row["failure_count"],
    )
