from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import aiosqlite
from contextlib import asynccontextmanager

from databridge.connectors.base import ColumnMeta, ForeignKey, TableMeta


_DDL = """
CREATE TABLE IF NOT EXISTS schema_tables (
    db_alias TEXT NOT NULL,
    table_name TEXT NOT NULL,
    row_count_approx INTEGER DEFAULT 0,
    schema_name TEXT DEFAULT 'public',
    schema_hash TEXT,
    scanned_at REAL NOT NULL,
    PRIMARY KEY (db_alias, table_name)
);

CREATE TABLE IF NOT EXISTS schema_columns (
    db_alias TEXT NOT NULL,
    table_name TEXT NOT NULL,
    column_name TEXT NOT NULL,
    dtype TEXT,
    nullable INTEGER DEFAULT 1,
    unique_rate REAL,
    null_rate REAL,
    p50 REAL,
    p95 REAL,
    is_unstructured INTEGER DEFAULT 0,
    notes TEXT DEFAULT '',
    PRIMARY KEY (db_alias, table_name, column_name)
);

CREATE TABLE IF NOT EXISTS schema_fk (
    db_alias TEXT NOT NULL,
    table_name TEXT NOT NULL,
    column_name TEXT NOT NULL,
    ref_table TEXT NOT NULL,
    ref_column TEXT NOT NULL,
    PRIMARY KEY (db_alias, table_name, column_name, ref_table, ref_column)
);
"""


class SchemaCache:
    def __init__(self, path: Path, ttl_hours: int = 24) -> None:
        self._path = path
        self._ttl_seconds = ttl_hours * 3600
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def _db(self):
        async with aiosqlite.connect(self._path) as conn:
            conn.row_factory = aiosqlite.Row
            await conn.executescript(_DDL)
            await conn.commit()
            yield conn

    async def save(self, db_alias: str, tables: dict[str, TableMeta]) -> None:
        now = time.time()
        async with self._db() as conn:
            for tname, table in tables.items():
                h = _hash_table(table)
                await conn.execute(
                    """INSERT OR REPLACE INTO schema_tables
                       (db_alias, table_name, row_count_approx, schema_name, schema_hash, scanned_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (db_alias, tname, table.row_count_approx, table.schema, h, now),
                )
                for cname, col in table.columns.items():
                    await conn.execute(
                        """INSERT OR REPLACE INTO schema_columns
                           (db_alias, table_name, column_name, dtype, nullable,
                            unique_rate, null_rate, p50, p95, is_unstructured, notes)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            db_alias, tname, cname, col.dtype,
                            int(col.nullable), col.unique_rate, col.null_rate,
                            col.p50, col.p95, int(col.is_unstructured), col.notes,
                        ),
                    )
                # Replace FKs for this table entirely on each scan
                await conn.execute(
                    "DELETE FROM schema_fk WHERE db_alias = ? AND table_name = ?",
                    (db_alias, tname),
                )
                for fk in table.foreign_keys:
                    await conn.execute(
                        """INSERT INTO schema_fk
                           (db_alias, table_name, column_name, ref_table, ref_column)
                           VALUES (?, ?, ?, ?, ?)""",
                        (db_alias, tname, fk.column, fk.ref_table, fk.ref_column),
                    )
            await conn.commit()

    async def load(self, db_alias: str) -> dict[str, TableMeta] | None:
        cutoff = time.time() - self._ttl_seconds
        async with self._db() as conn:
            async with conn.execute(
                "SELECT * FROM schema_tables WHERE db_alias = ? AND scanned_at > ?",
                (db_alias, cutoff),
            ) as cur:
                table_rows = await cur.fetchall()
            if not table_rows:
                return None

            tables: dict[str, TableMeta] = {}
            for row in table_rows:
                tname = row["table_name"]
                tables[tname] = TableMeta(
                    name=tname,
                    row_count_approx=row["row_count_approx"],
                    schema=row["schema_name"] or "public",
                )
            async with conn.execute(
                "SELECT * FROM schema_columns WHERE db_alias = ?", (db_alias,)
            ) as cur:
                col_rows = await cur.fetchall()
            for row in col_rows:
                tname = row["table_name"]
                if tname not in tables:
                    continue
                cname = row["column_name"]
                tables[tname].columns[cname] = ColumnMeta(
                    name=cname,
                    dtype=row["dtype"] or "",
                    nullable=bool(row["nullable"]),
                    unique_rate=row["unique_rate"],
                    null_rate=row["null_rate"],
                    p50=row["p50"],
                    p95=row["p95"],
                    is_unstructured=bool(row["is_unstructured"]),
                    notes=row["notes"] or "",
                )
            async with conn.execute(
                "SELECT table_name, column_name, ref_table, ref_column FROM schema_fk WHERE db_alias = ?",
                (db_alias,),
            ) as cur:
                fk_rows = await cur.fetchall()
            for row in fk_rows:
                tname = row["table_name"]
                if tname in tables:
                    tables[tname].foreign_keys.append(
                        ForeignKey(
                            column=row["column_name"],
                            ref_table=row["ref_table"],
                            ref_column=row["ref_column"],
                        )
                    )
        return tables

    async def invalidate(self, db_alias: str) -> None:
        async with self._db() as conn:
            await conn.execute("DELETE FROM schema_tables WHERE db_alias = ?", (db_alias,))
            await conn.execute("DELETE FROM schema_columns WHERE db_alias = ?", (db_alias,))
            await conn.commit()

    async def get_hash(self, db_alias: str, table_name: str) -> str | None:
        async with self._db() as conn:
            async with conn.execute(
                "SELECT schema_hash FROM schema_tables WHERE db_alias = ? AND table_name = ?",
                (db_alias, table_name),
            ) as cur:
                row = await cur.fetchone()
        return row["schema_hash"] if row else None


def _hash_table(table: TableMeta) -> str:
    payload = {
        "columns": {k: {"dtype": v.dtype, "nullable": v.nullable} for k, v in table.columns.items()}
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
