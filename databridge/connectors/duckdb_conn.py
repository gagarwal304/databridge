from __future__ import annotations

import asyncio
import time
from typing import Any
from urllib.parse import urlparse

import duckdb

from databridge.connectors.base import BaseConnector, ColumnMeta, DbType, ForeignKey, QueryResult, TableMeta

_QUERY_TIMEOUT = 120  # seconds — must be less than engine-level timeout (180s)

# SQL macro registered on every connection — strips non-ASCII chars and lowercases.
# Avoids the numpy dependency that DuckDB Python UDFs require.
_NORMALIZE_TEXT_MACRO = r"""
CREATE OR REPLACE MACRO normalize_text(s) AS
  regexp_replace(
    regexp_replace(lower(CAST(s AS VARCHAR)), '[^\x00-\x7f]', '', 'g'),
    '\s+', '', 'g'
  )
"""


class DuckDBConnector(BaseConnector):
    db_type = DbType.DUCKDB

    def __init__(self, uri: str, db_alias: str) -> None:
        super().__init__(uri, db_alias)
        parsed = urlparse(uri)
        self._path = parsed.path or ":memory:"
        self._conn: duckdb.DuckDBPyConnection | None = None

    async def connect(self) -> None:
        self._conn = duckdb.connect(self._path)
        self._conn.execute(_NORMALIZE_TEXT_MACRO)

    async def disconnect(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def _conn_or_raise(self) -> duckdb.DuckDBPyConnection:
        if not self._conn:
            raise RuntimeError(f"Connector '{self.db_alias}' is not connected")
        return self._conn

    async def _interrupt_and_reopen(self) -> None:
        """
        Abort the running DuckDB query and reopen a clean connection.

        run_in_executor wraps the blocking call in a thread; asyncio.wait_for
        cancels our coroutine but the thread keeps running. conn.interrupt()
        signals DuckDB to abort immediately, then we reopen for a clean state.
        """
        if self._conn is not None:
            try:
                self._conn.interrupt()
            except Exception:
                pass
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        self._conn = duckdb.connect(self._path)
        self._conn.execute(_NORMALIZE_TEXT_MACRO)

    async def execute_query(self, query: str, row_limit: int | None = None) -> QueryResult:
        conn = self._conn_or_raise()
        if row_limit:
            query = _inject_limit(query, row_limit)
        t0 = time.monotonic()

        def _run() -> tuple[list, list[str]]:
            rel = conn.execute(query)
            rows = rel.fetchall()
            cols = [d[0] for d in (rel.description or [])]
            return rows, cols

        loop = asyncio.get_running_loop()
        try:
            rows, cols = await asyncio.wait_for(
                loop.run_in_executor(None, _run),
                timeout=_QUERY_TIMEOUT,
            )
        except asyncio.TimeoutError:
            await self._interrupt_and_reopen()
            raise RuntimeError(
                f"DuckDB query timed out after {_QUERY_TIMEOUT}s — add WHERE filters or aggregations to reduce the scan scope"
            )

        elapsed = (time.monotonic() - t0) * 1000
        dicts = [dict(zip(cols, row)) for row in rows]
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
        col_rows = conn.execute("""
            SELECT table_name, column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'main'
            ORDER BY table_name, ordinal_position
        """).fetchall()

        count_map: dict[str, int] = {}
        table_rows = conn.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'main' AND table_type = 'BASE TABLE'
        """).fetchall()
        for (tname,) in table_rows:
            try:
                result = conn.execute(f'SELECT COUNT(*) FROM "{tname}"').fetchone()
                count_map[tname] = result[0] if result else 0
            except Exception:
                count_map[tname] = 0

        try:
            fk_rows = conn.execute("""
                SELECT
                    tc.table_name,
                    kcu.column_name,
                    ccu.table_name  AS ref_table,
                    ccu.column_name AS ref_column
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                    ON tc.constraint_name = kcu.constraint_name
                    AND tc.constraint_schema = kcu.table_schema
                JOIN information_schema.constraint_column_usage ccu
                    ON tc.constraint_name = ccu.constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY'
                AND   tc.table_schema    = 'main'
            """).fetchall()
            fk_map: dict[str, list[ForeignKey]] = {}
            for tname, col, ref_table, ref_col in fk_rows:
                fk_map.setdefault(tname, []).append(ForeignKey(column=col, ref_table=ref_table, ref_column=ref_col))
        except Exception:
            fk_map = {}

        tables: dict[str, TableMeta] = {}
        for tname, cname, dtype, nullable in col_rows:
            if tname not in tables:
                tables[tname] = TableMeta(name=tname, row_count_approx=count_map.get(tname, 0))
            tables[tname].columns[cname] = ColumnMeta(
                name=cname,
                dtype=dtype,
                nullable=nullable == "YES",
                is_unstructured=_is_unstructured(cname, dtype),
            )
        for tname, table in tables.items():
            table.foreign_keys = fk_map.get(tname, [])
        return tables

    async def sample_column(self, table: str, column: str, n: int) -> list[Any]:
        conn = self._conn_or_raise()
        rows = conn.execute(
            f'SELECT "{column}" FROM "{table}" WHERE "{column}" IS NOT NULL LIMIT {n}'
        ).fetchall()
        return [r[0] for r in rows]

    async def explain_cost(self, query: str) -> float:
        conn = self._conn_or_raise()
        plan = conn.execute(f"EXPLAIN {query}").fetchall()
        return float(len(plan))


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
_UNSTRUCTURED_DUCKDB_TYPES = {"varchar", "text", "string", "json"}


def _is_unstructured(col_name: str, dtype: str) -> bool:
    if dtype.lower() not in _UNSTRUCTURED_DUCKDB_TYPES:
        return False
    tokens = {t.lower() for t in col_name.replace("-", "_").split("_")}
    return bool(tokens & _UNSTRUCTURED_NAME_TOKENS)
