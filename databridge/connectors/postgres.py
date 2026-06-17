from __future__ import annotations

import asyncio
import time
from typing import Any

import asyncpg

from databridge.connectors.base import BaseConnector, ColumnMeta, DbType, ForeignKey, QueryResult, TableMeta

_QUERY_TIMEOUT = 120  # seconds

# Created once per connect(); persistent in the DB so subsequent connections find it already.
# STRICT means NULL input returns NULL (consistent with SQLite and DuckDB behavior).
_NORMALIZE_TEXT_SQL = """
CREATE OR REPLACE FUNCTION normalize_text(s TEXT)
RETURNS TEXT
LANGUAGE SQL
STRICT
IMMUTABLE
AS $$
  SELECT regexp_replace(
    regexp_replace(lower(s), '[^[:ascii:]]', '', 'g'),
    '\\s', '', 'g'
  )
$$
"""


class PostgresConnector(BaseConnector):
    db_type = DbType.POSTGRES

    def __init__(self, uri: str, db_alias: str) -> None:
        super().__init__(uri, db_alias)
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(self.uri)
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(_NORMALIZE_TEXT_SQL)
        except Exception:
            pass  # non-fatal: DB may lack CREATE FUNCTION privilege

    async def disconnect(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    def _pool_or_raise(self) -> asyncpg.Pool:
        if not self._pool:
            raise RuntimeError(f"Connector '{self.db_alias}' is not connected")
        return self._pool

    async def execute_query(self, query: str, row_limit: int | None = None) -> QueryResult:
        pool = self._pool_or_raise()
        if row_limit:
            query = _inject_limit(query, row_limit)
        t0 = time.monotonic()
        try:
            async with asyncio.timeout(_QUERY_TIMEOUT):
                async with pool.acquire() as conn:
                    rows = await conn.fetch(query)
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"PostgreSQL query timed out after {_QUERY_TIMEOUT}s — add WHERE filters or aggregations to reduce the scan scope — "
                "try a more targeted query with WHERE filters or aggregation"
            )
        elapsed = (time.monotonic() - t0) * 1000
        dicts = [dict(r) for r in rows]
        cols = list(rows[0].keys()) if rows else []
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
        pool = self._pool_or_raise()
        async with pool.acquire() as conn:
            col_rows = await conn.fetch("""
                SELECT
                    c.table_name,
                    c.column_name,
                    c.data_type,
                    c.is_nullable
                FROM information_schema.columns c
                WHERE c.table_schema = 'public'
                ORDER BY c.table_name, c.ordinal_position
            """)
            count_rows = await conn.fetch("""
                SELECT relname AS table_name, reltuples::bigint AS row_count
                FROM pg_class
                JOIN pg_namespace ON pg_namespace.oid = pg_class.relnamespace
                WHERE nspname = 'public' AND relkind = 'r'
            """)
            fk_rows = await conn.fetch("""
                SELECT
                    tc.table_name,
                    kcu.column_name,
                    ccu.table_name  AS ref_table,
                    ccu.column_name AS ref_column
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                    ON tc.constraint_name = kcu.constraint_name
                    AND tc.table_schema  = kcu.table_schema
                JOIN information_schema.constraint_column_usage AS ccu
                    ON tc.constraint_name = ccu.constraint_name
                    AND tc.table_schema  = ccu.table_schema
                WHERE tc.constraint_type = 'FOREIGN KEY'
                AND   tc.table_schema    = 'public'
            """)
        counts = {r["table_name"]: r["row_count"] for r in count_rows}
        fk_map: dict[str, list[ForeignKey]] = {}
        for row in fk_rows:
            fk_map.setdefault(row["table_name"], []).append(
                ForeignKey(column=row["column_name"], ref_table=row["ref_table"], ref_column=row["ref_column"])
            )
        tables: dict[str, TableMeta] = {}
        for row in col_rows:
            tname = row["table_name"]
            if tname not in tables:
                tables[tname] = TableMeta(name=tname, row_count_approx=counts.get(tname, 0))
            tables[tname].columns[row["column_name"]] = ColumnMeta(
                name=row["column_name"],
                dtype=row["data_type"],
                nullable=row["is_nullable"] == "YES",
                is_unstructured=_is_unstructured(row["column_name"], row["data_type"]),
            )
        for tname, table in tables.items():
            table.foreign_keys = fk_map.get(tname, [])
        return tables

    async def sample_column(self, table: str, column: str, n: int) -> list[Any]:
        pool = self._pool_or_raise()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f'SELECT "{column}" FROM "{table}" WHERE "{column}" IS NOT NULL LIMIT $1', n
            )
        return [r[column] for r in rows]

    async def explain_cost(self, query: str) -> float:
        pool = self._pool_or_raise()
        async with pool.acquire() as conn:
            rows = await conn.fetch(f"EXPLAIN (FORMAT JSON) {query}")
        plan = rows[0][0]
        return float(plan[0]["Plan"].get("Total Cost", 0.0))


def _inject_limit(query: str, limit: int) -> str:
    q = query.rstrip().rstrip(";")
    upper = q.upper()
    if "LIMIT" not in upper:
        q = f"{q} LIMIT {limit}"
    return q


_UNSTRUCTURED_NAME_TOKENS = {
    "description", "desc", "bio", "biography", "text", "body", "content",
    "note", "notes", "comment", "comments", "summary", "detail", "details",
    "narrative", "abstract", "overview", "message", "info", "information",
    "remark", "remarks", "review", "reviews", "feedback",
}
_UNSTRUCTURED_PG_TYPES = {"text", "character varying", "varchar", "json", "jsonb", "xml"}


def _is_unstructured(col_name: str, dtype: str) -> bool:
    if dtype.lower() not in _UNSTRUCTURED_PG_TYPES:
        return False
    tokens = {t.lower() for t in col_name.replace("-", "_").split("_")}
    return bool(tokens & _UNSTRUCTURED_NAME_TOKENS)
