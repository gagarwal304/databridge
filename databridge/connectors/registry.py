from __future__ import annotations

from urllib.parse import urlparse

from databridge.connectors.base import BaseConnector, DbType


class ConnectorRegistry:
    def __init__(self) -> None:
        self._connectors: dict[str, BaseConnector] = {}

    def register(self, connector: BaseConnector) -> None:
        self._connectors[connector.db_alias] = connector

    def get(self, alias: str) -> BaseConnector:
        if alias not in self._connectors:
            raise KeyError(f"No connector registered for alias '{alias}'")
        return self._connectors[alias]

    def all(self) -> dict[str, BaseConnector]:
        return dict(self._connectors)

    def aliases(self) -> list[str]:
        return list(self._connectors.keys())

    async def connect_all(self) -> None:
        for connector in self._connectors.values():
            await connector.connect()

    async def disconnect_all(self) -> None:
        for connector in self._connectors.values():
            await connector.disconnect()

    @staticmethod
    def from_uri(uri: str, alias: str | None = None) -> BaseConnector:
        from databridge.connectors.postgres import PostgresConnector
        from databridge.connectors.mongodb import MongoConnector
        from databridge.connectors.sqlite_conn import SQLiteConnector
        from databridge.connectors.duckdb_conn import DuckDBConnector

        scheme = urlparse(uri).scheme.lower()
        db_alias = alias or scheme

        drivers = {
            "postgresql": PostgresConnector,
            "postgres": PostgresConnector,
            "mongodb": MongoConnector,
            "mongodb+srv": MongoConnector,
            "sqlite": SQLiteConnector,
            "duckdb": DuckDBConnector,
        }
        if scheme not in drivers:
            raise ValueError(f"Unsupported database scheme: '{scheme}'")
        return drivers[scheme](uri=uri, db_alias=db_alias)

    @classmethod
    def from_uris(cls, uris: list[str]) -> "ConnectorRegistry":
        registry = cls()
        seen: dict[str, int] = {}
        for uri in uris:
            scheme = urlparse(uri).scheme.lower()
            seen[scheme] = seen.get(scheme, 0) + 1
            alias = scheme if seen[scheme] == 1 else f"{scheme}_{seen[scheme]}"
            registry.register(cls.from_uri(uri, alias))
        return registry
