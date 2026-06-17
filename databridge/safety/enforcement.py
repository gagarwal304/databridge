from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import sqlglot
import sqlglot.expressions as exp


class ViolationType(str, Enum):
    WRITE_OPERATION = "write_operation"
    DDL_OPERATION = "ddl_operation"
    MULTIPLE_STATEMENTS = "multiple_statements"
    PARSE_ERROR = "parse_error"


@dataclass
class SafetyViolation(Exception):
    violation_type: ViolationType
    detail: str

    def __str__(self) -> str:
        return f"[{self.violation_type}] {self.detail}"


_WRITE_NODE_TYPES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
)

_DDL_NODE_TYPES = (
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.TruncateTable,
    exp.Command,
)


class SafetyEnforcer:
    """
    Deterministic read-only enforcement at the SQL AST level.
    MongoDB pipelines are always read-only by construction and skip this check.
    """

    def check(self, query: str, dialect: str = "postgres") -> None:
        """Raises SafetyViolation if the query is not a safe read-only SELECT."""
        if not query or not query.strip():
            return

        # Mongo pipeline dicts are passed as JSON strings — always read-only
        stripped = query.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            return

        try:
            statements = sqlglot.parse(query, dialect=dialect, error_level=sqlglot.ErrorLevel.RAISE)
        except sqlglot.errors.ParseError as e:
            raise SafetyViolation(ViolationType.PARSE_ERROR, str(e)) from e

        if not statements:
            return

        if len(statements) > 1:
            raise SafetyViolation(
                ViolationType.MULTIPLE_STATEMENTS,
                "Multiple SQL statements are not allowed in a single query.",
            )

        stmt = statements[0]

        for node_type in _WRITE_NODE_TYPES:
            if isinstance(stmt, node_type):
                raise SafetyViolation(
                    ViolationType.WRITE_OPERATION,
                    f"{node_type.__name__} operations are blocked. DataBridge is read-only by default.",
                )

        for node_type in _DDL_NODE_TYPES:
            if isinstance(stmt, node_type):
                raise SafetyViolation(
                    ViolationType.DDL_OPERATION,
                    f"{node_type.__name__} operations are unconditionally blocked.",
                )

    def is_safe(self, query: str, dialect: str = "postgres") -> bool:
        try:
            self.check(query, dialect)
            return True
        except SafetyViolation:
            return False
