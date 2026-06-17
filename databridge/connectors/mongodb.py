from __future__ import annotations

import asyncio
import time
from typing import Any
from urllib.parse import urlparse

import motor.motor_asyncio
from pymongo import MongoClient

from databridge.connectors.base import BaseConnector, ColumnMeta, DbType, QueryResult, TableMeta

_QUERY_TIMEOUT = 120  # seconds


class MongoConnector(BaseConnector):
    db_type = DbType.MONGODB

    def __init__(self, uri: str, db_alias: str) -> None:
        super().__init__(uri, db_alias)
        parsed = urlparse(uri)
        # Extract db name from path, default to first path segment
        path = parsed.path.lstrip("/")
        self._db_name = path.split("/")[0] if path else "test"
        self._client: motor.motor_asyncio.AsyncIOMotorClient | None = None
        self._db: motor.motor_asyncio.AsyncIOMotorDatabase | None = None

    async def connect(self) -> None:
        self._client = motor.motor_asyncio.AsyncIOMotorClient(self.uri)
        self._db = self._client[self._db_name]

    async def disconnect(self) -> None:
        if self._client:
            self._client.close()
            self._client = None
            self._db = None

    def _db_or_raise(self) -> motor.motor_asyncio.AsyncIOMotorDatabase:
        if self._db is None:
            raise RuntimeError(f"Connector '{self.db_alias}' is not connected")
        return self._db

    async def execute_query(self, query: str, row_limit: int | None = None) -> QueryResult:
        """
        query is expected to be a JSON-serialisable dict describing a MongoDB
        aggregation pipeline: {"collection": "...", "pipeline": [...]}
        Plain SQL strings are not supported on MongoDB — the query planner always
        sends pre-translated pipeline dicts.
        """
        import json

        db = self._db_or_raise()
        spec = json.loads(query) if isinstance(query, str) else query
        if isinstance(spec, list):
            raise ValueError(
                'MongoDB query must be a JSON object: {"collection": "<name>", "pipeline": [<stages>]}. '
                'Do not send a bare array. '
                'Example: {"collection": "articles", "pipeline": [{"$limit": 5}]}'
            )
        collection_name: str = spec["collection"]
        pipeline: list[dict] = spec.get("pipeline", [])
        if row_limit:
            pipeline = [*pipeline, {"$limit": row_limit}]

        t0 = time.monotonic()
        cursor = db[collection_name].aggregate(pipeline)
        try:
            rows = await asyncio.wait_for(
                cursor.to_list(length=row_limit or 10_000),
                timeout=_QUERY_TIMEOUT,
            )
        except asyncio.TimeoutError:
            # Reconnect — cursor/connection state is unknown after cancellation
            if self._client:
                self._client.close()
            self._client = motor.motor_asyncio.AsyncIOMotorClient(self.uri)
            self._db = self._client[self._db_name]
            raise RuntimeError(
                f"MongoDB query timed out after {_QUERY_TIMEOUT}s — simplify the pipeline or add $match filters"
            )
        elapsed = (time.monotonic() - t0) * 1000

        # Serialise ObjectId and other BSON types
        rows = [_bson_to_dict(r) for r in rows]
        cols = list(rows[0].keys()) if rows else []
        return QueryResult(
            rows=rows,
            row_count=len(rows),
            columns=cols,
            execution_ms=elapsed,
            database=self.db_alias,
            query=str(spec),
            truncated=row_limit is not None and len(rows) == row_limit,
        )

    async def introspect_schema(self) -> dict[str, TableMeta]:
        db = self._db_or_raise()
        tables: dict[str, TableMeta] = {}
        collection_names = await db.list_collection_names()
        for cname in collection_names:
            count = await db[cname].estimated_document_count()
            # Sample documents to infer schema
            sample = await db[cname].aggregate([{"$sample": {"size": 20}}]).to_list(20)
            columns: dict[str, ColumnMeta] = {}
            for doc in sample:
                for key, val in doc.items():
                    if key not in columns:
                        columns[key] = ColumnMeta(
                            name=key,
                            dtype=type(val).__name__,
                            nullable=True,
                        )
            tables[cname] = TableMeta(
                name=cname,
                row_count_approx=count,
                columns=columns,
            )
        return tables

    async def sample_column(self, table: str, column: str, n: int) -> list[Any]:
        db = self._db_or_raise()
        pipeline = [
            {"$match": {column: {"$exists": True, "$ne": None}}},
            {"$project": {column: 1, "_id": 0}},
            {"$limit": n},
        ]
        docs = await db[table].aggregate(pipeline).to_list(n)
        return [_bson_to_dict(d).get(column) for d in docs]

    async def explain_cost(self, query: str) -> float:
        # MongoDB does not expose a numeric cost estimate; return 0 to satisfy interface
        return 0.0


def _bson_to_dict(doc: dict) -> dict:
    result = {}
    for k, v in doc.items():
        if hasattr(v, "__str__") and type(v).__module__.startswith("bson"):
            result[k] = str(v)
        else:
            result[k] = v
    return result
