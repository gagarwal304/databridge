from __future__ import annotations

import asyncio

from databridge.connectors.base import TableMeta
from databridge.connectors.registry import ConnectorRegistry
from databridge.schema.cache import SchemaCache


class SchemaScanner:
    def __init__(self, registry: ConnectorRegistry, cache: SchemaCache) -> None:
        self._registry = registry
        self._cache = cache

    async def scan(self, db_alias: str, force: bool = False) -> dict[str, TableMeta]:
        if not force:
            cached = await self._cache.load(db_alias)
            if cached is not None:
                return cached
        connector = self._registry.get(db_alias)
        tables = await connector.introspect_schema()
        await self._cache.save(db_alias, tables)
        return tables

    async def scan_all(self, force: bool = False) -> dict[str, dict[str, TableMeta]]:
        results = await asyncio.gather(
            *[self.scan(alias, force=force) for alias in self._registry.aliases()],
            return_exceptions=True,
        )
        out: dict[str, dict[str, TableMeta]] = {}
        for alias, result in zip(self._registry.aliases(), results):
            if isinstance(result, Exception):
                # Surface as empty schema with a note rather than crashing
                out[alias] = {}
            else:
                out[alias] = result  # type: ignore[assignment]
        return out

    async def diff(self, db_alias: str) -> list[str]:
        """Return list of changed table names since last scan."""
        fresh = await self._registry.get(db_alias).introspect_schema()
        changed = []
        for tname, table in fresh.items():
            cached_hash = await self._cache.get_hash(db_alias, tname)
            from databridge.schema.cache import _hash_table
            if cached_hash != _hash_table(table):
                changed.append(tname)
        return changed
