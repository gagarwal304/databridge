from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DbType(str, Enum):
    POSTGRES = "postgresql"
    MONGODB = "mongodb"
    SQLITE = "sqlite"
    DUCKDB = "duckdb"

    @property
    def sqlglot_dialect(self) -> str:
        """Return the dialect name recognized by sqlglot for this DbType."""
        _MAP = {
            DbType.POSTGRES: "postgres",
            DbType.SQLITE: "sqlite",
            DbType.DUCKDB: "duckdb",
            DbType.MONGODB: "mongo",
        }
        return _MAP.get(self, self.value)


@dataclass
class ColumnMeta:
    name: str
    dtype: str
    nullable: bool = True
    unique_rate: float | None = None
    null_rate: float | None = None
    p50: float | None = None
    p95: float | None = None
    is_unstructured: bool = False
    notes: str = ""


@dataclass
class ForeignKey:
    column: str       # local column name
    ref_table: str    # referenced table (same DB)
    ref_column: str   # referenced column


@dataclass
class TableMeta:
    name: str
    row_count_approx: int = 0
    columns: dict[str, ColumnMeta] = field(default_factory=dict)
    schema: str = "public"
    foreign_keys: list[ForeignKey] = field(default_factory=list)


@dataclass
class QueryResult:
    rows: list[dict[str, Any]]
    row_count: int
    columns: list[str]
    execution_ms: float = 0.0
    database: str = ""
    query: str = ""
    truncated: bool = False


class BaseConnector(ABC):
    db_type: DbType

    def __init__(self, uri: str, db_alias: str) -> None:
        self.uri = uri
        self.db_alias = db_alias

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def disconnect(self) -> None: ...

    @abstractmethod
    async def execute_query(
        self, query: str, row_limit: int | None = None
    ) -> QueryResult: ...

    @abstractmethod
    async def introspect_schema(self) -> dict[str, TableMeta]: ...

    @abstractmethod
    async def sample_column(
        self, table: str, column: str, n: int
    ) -> list[Any]: ...

    @abstractmethod
    async def explain_cost(self, query: str) -> float: ...

    async def __aenter__(self) -> "BaseConnector":
        await self.connect()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.disconnect()
