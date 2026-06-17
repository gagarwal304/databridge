from __future__ import annotations

import asyncio
import time
import unicodedata
from typing import Any
from urllib.parse import urlparse

import aiosqlite

from databridge.connectors.base import BaseConnector, ColumnMeta, DbType, ForeignKey, QueryResult, TableMeta


def _normalize_text(s: str | None) -> str | None:
    """Lowercase, strip diacritics, remove all whitespace — usable as normalize_text() in SQL."""
    if s is None:
        return None
    s = unicodedata.normalize("NFKD", str(s).lower())
    s = s.encode("ascii", "ignore").decode("ascii")
    return "".join(s.split())


def _regexp_replace_3(string: str | None, pattern: str, replacement: str) -> str | None:
    """REGEXP_REPLACE(string, pattern, replacement) — matches PostgreSQL 3-arg form."""
    import re
    if string is None:
        return None
    return re.sub(pattern, replacement, str(string))


def _regexp_replace_4(string: str | None, pattern: str, replacement: str, flags: str) -> str | None:
    """REGEXP_REPLACE(string, pattern, replacement, flags) — 'g' flag is the default (re.sub replaces all)."""
    import re
    if string is None:
        return None
    return re.sub(pattern, replacement, str(string))


class SQLiteConnector(BaseConnector):
    db_type = DbType.SQLITE

    def __init__(self, uri: str, db_alias: str) -> None:
        super().__init__(uri, db_alias)
        parsed = urlparse(uri)
        self._path = parsed.path or parsed.netloc
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self._conn = await aiosqlite.connect(self._path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.create_function("normalize_text", 1, _normalize_text)
        await self._conn.create_function("REGEXP_REPLACE", 3, _regexp_replace_3)
        await self._conn.create_function("REGEXP_REPLACE", 4, _regexp_replace_4)

    async def disconnect(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    def _conn_or_raise(self) -> aiosqlite.Connection:
        if not self._conn:
            raise RuntimeError(f"Connector '{self.db_alias}' is not connected")
        return self._conn

    async def _fetch(self, query: str) -> tuple[list, list[str]]:
        async with self._conn_or_raise().execute(query) as cursor:
            rows = await cursor.fetchall()
            cols = [d[0] for d in cursor.description or []]
        return rows, cols

    async def _interrupt_and_reopen(self) -> None:
        """
        Abort the running SQLite query and reopen a clean connection.

        aiosqlite runs one worker thread per connection. asyncio.wait_for cancels
        our coroutine but the thread keeps running, blocking every subsequent query.
        sqlite3.interrupt() is thread-safe and signals SQLite to abort immediately,
        freeing the thread. We then reopen so the connection is clean.
        """
        if self._conn is not None:
            raw: object = getattr(self._conn, "_conn", None)
            if raw is not None:
                try:
                    raw.interrupt()  # type: ignore[attr-defined]
                except Exception:
                    pass
            try:
                await asyncio.wait_for(self._conn.close(), timeout=5)
            except Exception:
                pass
            self._conn = None
        self._conn = await aiosqlite.connect(self._path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.create_function("normalize_text", 1, _normalize_text)
        await self._conn.create_function("REGEXP_REPLACE", 3, _regexp_replace_3)
        await self._conn.create_function("REGEXP_REPLACE", 4, _regexp_replace_4)

    async def execute_query(self, query: str, row_limit: int | None = None) -> QueryResult:
        if row_limit:
            query = _inject_limit(query, row_limit)
        t0 = time.monotonic()
        try:
            rows_raw, cols = await asyncio.wait_for(self._fetch(query), timeout=120)
        except asyncio.TimeoutError:
            await self._interrupt_and_reopen()
            raise RuntimeError(
                "SQLite query timed out after 120s — add WHERE filters or aggregations to reduce the scan scope"
            )
        elapsed = (time.monotonic() - t0) * 1000
        dicts = [dict(zip(cols, row)) for row in rows_raw]
        return QueryResult(
            rows=dicts,
            row_count=len(dicts),
            columns=cols,
            execution_ms=elapsed,
            database=self.db_alias,
            query=query,
            truncated=row_limit is not None and len(dicts) == row_limit,
        )

    async def introspect_schema(self) -> dict[str, TableMeta]:
        conn = self._conn_or_raise()
        tables: dict[str, TableMeta] = {}
        async with conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ) as cur:
            table_names = [row[0] async for row in cur]

        for tname in table_names:
            async with conn.execute(f"PRAGMA table_info('{tname}')") as cur:
                col_rows = await cur.fetchall()
            async with conn.execute(f"PRAGMA foreign_key_list('{tname}')") as cur:
                fk_rows = await cur.fetchall()
            async with conn.execute(f"SELECT COUNT(*) FROM '{tname}'") as cur:
                count_row = await cur.fetchone()
            count = count_row[0] if count_row else 0
            columns = {}
            for row in col_rows:
                cname = row[1]
                columns[cname] = ColumnMeta(
                    name=cname,
                    dtype=row[2],
                    nullable=row[3] == 0,
                    is_unstructured=_is_unstructured(cname, row[2]),
                )
            # PRAGMA foreign_key_list columns: id, seq, table, from, to, on_update, on_delete, match
            fks = [
                ForeignKey(column=row[3], ref_table=row[2], ref_column=row[4])
                for row in fk_rows
            ]
            tables[tname] = TableMeta(name=tname, row_count_approx=count, columns=columns, foreign_keys=fks)
        return tables

    async def sample_column(self, table: str, column: str, n: int) -> list[Any]:
        conn = self._conn_or_raise()
        async with conn.execute(
            f'SELECT "{column}" FROM "{table}" WHERE "{column}" IS NOT NULL LIMIT ?', (n,)
        ) as cur:
            rows = await cur.fetchall()
        return [r[0] for r in rows]

    async def explain_cost(self, query: str) -> float:
        conn = self._conn_or_raise()
        async with conn.execute(f"EXPLAIN QUERY PLAN {query}") as cur:
            rows = await cur.fetchall()
        # SQLite EXPLAIN QUERY PLAN doesn't give numeric cost; estimate from row count
        return float(len(rows))


def _inject_limit(query: str, limit: int) -> str:
    q = query.rstrip().rstrip(";")
    if "LIMIT" not in q.upper():
        q = f"{q} LIMIT {limit}"
    return q


_UNSTRUCTURED_NAME_TOKENS = {
    "description", "desc", "bio", "biography", "text", "body", "content",
    "note", "notes", "comment", "comments", "summary", "detail", "details",
    "narrative", "abstract", "overview", "message", "info", "information",
    "remark", "remarks", "review", "reviews", "feedback",
}
_UNSTRUCTURED_SQLITE_TYPES = {"text", "varchar", "clob", "blob", "json", ""}


def _is_unstructured(col_name: str, dtype: str) -> bool:
    if dtype.lower() not in _UNSTRUCTURED_SQLITE_TYPES:
        return False
    tokens = {t.lower() for t in col_name.replace("-", "_").split("_")}
    return bool(tokens & _UNSTRUCTURED_NAME_TOKENS)
