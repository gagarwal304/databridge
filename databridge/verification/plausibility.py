from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from databridge.connectors.base import TableMeta
from databridge.query.executor import ExecutionResult


class FailureMode(str, Enum):
    WRONG_JOIN_KEY = "wrong_join_key"
    SCHEMA_MISMATCH = "schema_mismatch"
    EMPTY_RESULT = "empty_result"
    COST_OVERRUN = "cost_overrun"
    PLAUSIBLE = "plausible"


@dataclass
class PlausibilityResult:
    score: float  # 0.0 (suspicious) – 1.0 (plausible)
    failure_mode: FailureMode
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def is_plausible(self) -> bool:
        return self.score >= 0.5


class PlausibilityChecker:
    def __init__(
        self,
        zero_row_threshold: int = 1_000,
        numeric_tolerance: float = 3.0,
    ) -> None:
        self._zero_row_threshold = zero_row_threshold
        self._numeric_tolerance = numeric_tolerance

    def check(
        self,
        result: ExecutionResult,
        schema: dict[str, dict[str, TableMeta]],
        query: str = "",
    ) -> PlausibilityResult:
        warnings: list[str] = []
        details: dict[str, Any] = {}

        # Zero-row check against known large tables
        if result.row_count == 0:
            for db_alias, tables in schema.items():
                for tname, table in tables.items():
                    if table.row_count_approx >= self._zero_row_threshold:
                        if tname.lower() in query.lower():
                            warnings.append(
                                f"Zero rows returned from '{db_alias}.{tname}' which has "
                                f"~{table.row_count_approx:,} rows. Possible wrong join key or filter."
                            )
            if warnings:
                return PlausibilityResult(
                    score=0.2,
                    failure_mode=FailureMode.WRONG_JOIN_KEY,
                    warnings=warnings,
                    details=details,
                )
            return PlausibilityResult(
                score=0.6,
                failure_mode=FailureMode.EMPTY_RESULT,
                warnings=["Query returned zero rows — may be correct or may indicate an issue."],
            )

        # Numeric range check against known schema stats
        for row in result.rows[:100]:
            for col, val in row.items():
                if not isinstance(val, (int, float)):
                    continue
                for db_alias, tables in schema.items():
                    for tname, table in tables.items():
                        col_meta = table.columns.get(col)
                        if col_meta and col_meta.p95 is not None:
                            if abs(val) > self._numeric_tolerance * col_meta.p95:
                                warnings.append(
                                    f"Value {val} in column '{col}' is {self._numeric_tolerance}x "
                                    f"above historical p95 ({col_meta.p95})."
                                )

        score = 1.0 - min(0.5, len(warnings) * 0.1)
        return PlausibilityResult(
            score=score,
            failure_mode=FailureMode.PLAUSIBLE if not warnings else FailureMode.SCHEMA_MISMATCH,
            warnings=warnings,
            details={"row_count": result.row_count},
        )
