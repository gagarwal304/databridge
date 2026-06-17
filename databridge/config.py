from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabridgeConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DATABRIDGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Connections ──────────────────────────────────────────────────────────
    database_uris: list[str] = Field(default_factory=list)

    @field_validator("database_uris", mode="before")
    @classmethod
    def parse_uris(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [u.strip() for u in v.split(",") if u.strip()]
        return v

    # ── Schema cache ─────────────────────────────────────────────────────────
    schema_cache_path: str = Field(default="~/.databridge/schema.db")
    schema_cache_ttl_hours: int = Field(default=24, gt=0)

    # ── Safety ───────────────────────────────────────────────────────────────
    default_row_limit: int = Field(default=10_000, gt=0)
    max_cost_budget: float = Field(default=1_000.0, gt=0)
    hidden_tables: list[str] = Field(default_factory=list)

    @field_validator("hidden_tables", mode="before")
    @classmethod
    def parse_hidden(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [t.strip() for t in v.split(",") if t.strip()]
        return v

    # ── Join discovery ───────────────────────────────────────────────────────
    name_similarity_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    value_sample_size: int = Field(default=50, gt=0)
    overlap_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    min_confidence_to_propose: float = Field(default=0.60, ge=0.0, le=1.0)

    # ── Audit ────────────────────────────────────────────────────────────────
    audit_log_path: str = Field(default="~/.databridge/audit.db")

    # ── MCP ──────────────────────────────────────────────────────────────────
    mcp_server_name: str = Field(default="databridge")

    # ── Verification ─────────────────────────────────────────────────────────
    zero_row_warning_threshold: int = Field(default=1_000, gt=0)
    numeric_range_tolerance: float = Field(default=3.0, gt=0)

    def resolved_cache_path(self) -> Path:
        return Path(self.schema_cache_path).expanduser().resolve()

    def resolved_audit_path(self) -> Path:
        return Path(self.audit_log_path).expanduser().resolve()

    def ensure_dirs(self) -> None:
        self.resolved_cache_path().parent.mkdir(parents=True, exist_ok=True)
        self.resolved_audit_path().parent.mkdir(parents=True, exist_ok=True)


_config: DatabridgeConfig | None = None


def get_config() -> DatabridgeConfig:
    global _config
    if _config is None:
        _config = DatabridgeConfig()
    return _config


def reset_config() -> None:
    global _config
    _config = None
