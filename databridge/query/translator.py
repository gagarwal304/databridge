from __future__ import annotations

import sqlglot
import sqlglot.expressions as exp
from databridge.connectors.base import DbType


class QueryTranslator:
    def translate(self, query: str, from_dialect: str, to_db_type: DbType) -> str:
        """Transpile SQL between dialects. MongoDB handled by planner, not here."""
        if to_db_type == DbType.MONGODB:
            return query  # planner generates pipeline dicts directly

        to_dialect = to_db_type.sqlglot_dialect
        if from_dialect == to_dialect:
            return query

        try:
            return sqlglot.transpile(
                query,
                read=from_dialect,
                write=to_dialect,
                pretty=False,
            )[0]
        except Exception:
            return query  # fall back to original if transpilation fails

    def inject_limit(self, query: str, limit: int, dialect: str = "postgres") -> str:
        try:
            ast = sqlglot.parse_one(query, dialect=dialect)
            if isinstance(ast, exp.Select):
                if ast.args.get("limit") is None:
                    ast = ast.limit(limit)
            return ast.sql(dialect=dialect)
        except Exception:
            q = query.rstrip().rstrip(";")
            return f"{q} LIMIT {limit}"

    def is_aggregate_query(self, query: str, dialect: str = "postgres") -> bool:
        try:
            ast = sqlglot.parse_one(query, dialect=dialect)
            if not isinstance(ast, exp.Select):
                return False
            return bool(ast.args.get("group")) or any(
                isinstance(node, (exp.AggFunc,)) for node in ast.walk()
            )
        except Exception:
            return False

    def push_aggregation(self, query: str, dialect: str = "postgres") -> str:
        """Ensure GROUP BY queries without ORDER BY don't strip aggregated rows via LIMIT.

        If the query has ORDER BY (i.e. it asks for top-N groups), keep the LIMIT —
        removing it would return all groups instead of just the requested top-N.
        Only remove the LIMIT when there is no ORDER BY and the intent is to return
        all groups without a row cap.
        """
        try:
            ast = sqlglot.parse_one(query, dialect=dialect)
            if isinstance(ast, exp.Select) and ast.args.get("group"):
                if not ast.args.get("order"):
                    # No ORDER BY → remove any accidental LIMIT so all groups are returned
                    ast.args.pop("limit", None)
                return ast.sql(dialect=dialect)
        except Exception:
            pass
        return query
