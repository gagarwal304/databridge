"""
DataAgentBench (DAB) evaluation harness.

Wires an LLM agent to DataBridge's MCP tools and evaluates it against
DAB's directory-structured benchmark.

DAB structure (per dataset):
    query_<dataset>/
    ├── db_config.yaml          ← database connections for this dataset
    ├── db_description.txt
    └── query<N>/
        ├── query.json          ← the question (a double-quoted string)
        ├── ground_truth.csv    ← expected answer
        └── validate.py         ← validate(llm_output: str) -> (bool, str)

Usage:
    databridge benchmark run \\
        --dab-root /path/to/DataAgentBench \\
        --provider anthropic \\
        --model claude-opus-4-8

Or directly:
    python -m benchmark.dab \\
        --dab-root /path/to/DataAgentBench \\
        --provider anthropic \\
        --model claude-opus-4-8

Results are written to benchmark/results/ locally. Never auto-submitted.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _load_dotenv(path: str = ".env") -> None:
    """Load key=value pairs from .env into os.environ (existing vars are not overwritten)."""
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

_load_dotenv()

log = logging.getLogger(__name__)


@dataclass
class QueryResult:
    dataset: str
    query_id: str
    question: str
    answer: str
    passed: bool
    reason: str
    execution_ms: float
    error: str | None = None


@dataclass
class BenchmarkReport:
    provider: str
    model: str
    total: int
    passed: int
    failed: int
    errors: int
    pass_at_1: float
    datasets: dict[str, float]   # dataset → pass rate
    results: list[QueryResult]
    run_at: float = field(default_factory=time.time)

    @property
    def summary(self) -> str:
        return (
            f"Provider: {self.provider}/{self.model} | "
            f"Pass@1: {self.pass_at_1:.1%} ({self.passed}/{self.total}) | "
            f"Errors: {self.errors}"
        )


# ── DAB directory helpers ──────────────────────────────────────────────────────

# The 12 datasets officially listed in the DataAgentBench benchmark overview.
# Folder names match query_<name> directories in the repo.
OFFICIAL_DATASETS: frozenset[str] = frozenset({
    "DEPS_DEV_V1",
    "GITHUB_REPOS",
    "PANCANCER_ATLAS",
    "PATENTS",
    "agnews",
    "bookreview",
    "crmarenapro",
    "googlelocal",
    "music_brainz_20k",
    "stockindex",
    "stockmarket",
    "yelp",
})

# Official run order: fastest/simplest first so early feedback is quick;
# heavy/flaky datasets (crmarenapro) run last.
OFFICIAL_DATASET_ORDER: list[str] = [
    "bookreview",
    "googlelocal",
    "PATENTS",
    "DEPS_DEV_V1",
    "music_brainz_20k",
    "stockindex",
    "stockmarket",
    "GITHUB_REPOS",
    "PANCANCER_ATLAS",
    "agnews",
    "yelp",
    "crmarenapro",
]


def _discover_queries(
    dab_root: Path,
    dataset_filter: str | list[str] | None = None,
) -> list[tuple[str, str, Path]]:
    """
    Walk dab_root and return (dataset_name, query_id, query_dir) for every
    query_<dataset>/query<N> that has a query.json and validate.py.

    dataset_filter: None = all datasets; str = one dataset; list[str] = exact set.
    """
    if isinstance(dataset_filter, str):
        allowed: set[str] | None = {dataset_filter}
    elif isinstance(dataset_filter, (list, frozenset, set)):
        allowed = set(dataset_filter)
    else:
        allowed = None

    found = []
    for dataset_dir in sorted(dab_root.iterdir()):
        if not dataset_dir.is_dir() or not dataset_dir.name.startswith("query_"):
            continue
        dataset_name = dataset_dir.name[len("query_"):]
        if allowed is not None and dataset_name not in allowed:
            continue
        for query_dir in sorted(dataset_dir.iterdir()):
            if not query_dir.is_dir() or not query_dir.name.startswith("query"):
                continue
            if (query_dir / "query.json").exists() and (query_dir / "validate.py").exists():
                found.append((dataset_name, query_dir.name, query_dir))
    return found


def _read_question(query_dir: Path) -> str:
    """Read the natural-language question from query.json (a double-quoted string)."""
    raw = (query_dir / "query.json").read_text(encoding="utf-8").strip()
    return json.loads(raw)


def _call_validate(query_dir: Path, answer: str) -> tuple[bool, str]:
    """Dynamically import validate.py and call validate(answer)."""
    import sys
    validate_path = query_dir / "validate.py"
    # Some validators import from common_scaffold, which lives in the DAB root.
    dab_root = query_dir.parent.parent
    if str(dab_root) not in sys.path:
        sys.path.insert(0, str(dab_root))
    spec = importlib.util.spec_from_file_location("_dab_validate", validate_path)
    mod = importlib.util.module_from_spec(spec)          # type: ignore[arg-type]
    spec.loader.exec_module(mod)                          # type: ignore[union-attr]
    result = mod.validate(answer)
    if isinstance(result, tuple):
        is_valid, reason = result
    else:
        is_valid, reason = bool(result), ""
    return bool(is_valid), str(reason)


def _pg_env() -> tuple[str, str, str, str, dict]:
    """Return (user, password, host, port, env) from PG_* or PGUSER/PGPASSWORD env vars."""
    user = os.environ.get("PG_USER") or os.environ.get("PGUSER") or os.environ.get("USER", "postgres")
    password = os.environ.get("PG_PASSWORD") or os.environ.get("PGPASSWORD", "")
    host = os.environ.get("PG_HOST") or os.environ.get("PGHOST", "localhost")
    port = os.environ.get("PG_PORT") or os.environ.get("PGPORT", "5432")
    env = os.environ.copy()
    if password:
        env["PGPASSWORD"] = password
    return user, password, host, port, env


def _pg_db_exists(db_name: str) -> bool:
    user, _, host, port, env = _pg_env()
    result = subprocess.run(
        ["psql", f"-h{host}", f"-U{user}", f"-p{port}", "-lqt"],
        capture_output=True, text=True, env=env,
    )
    return any(
        line.split("|")[0].strip() == db_name
        for line in result.stdout.splitlines()
    )


def _ensure_postgres_db(db_name: str, sql_file: Path) -> None:
    """Create the database and load the SQL dump if it doesn't already exist."""
    if _pg_db_exists(db_name):
        log.info(f"PostgreSQL database '{db_name}' already exists — skipping load.")
        return

    if not sql_file.exists():
        raise RuntimeError(
            f"SQL dump not found: {sql_file}\n"
            "Run 'git lfs pull' inside the DataAgentBench directory to download data files."
        )

    user, _, host, port, env = _pg_env()
    log.info(f"Creating PostgreSQL database '{db_name}' and loading {sql_file.name}…")

    subprocess.run(
        ["createdb", f"-h{host}", f"-U{user}", f"-p{port}",
         "--encoding=UTF8", "--lc-collate=C", "--lc-ctype=C",
         "--template=template0", db_name],
        check=True, env=env,
    )
    result = subprocess.run(
        ["psql", f"-h{host}", f"-U{user}", f"-p{port}",
         "-d", db_name, "-f", str(sql_file)],
        capture_output=True, text=True, env=env,
    )
    # psql exits 0 even when individual statements error (e.g. OWNER TO role not found).
    # Surface only genuine failures (non-zero exit), not benign OWNER/GRANT warnings.
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, "psql", result.stderr)
    log.info(f"Loaded '{db_name}' successfully.")


def _mongo_db_exists(db_name: str, host: str = "localhost", port: int = 27017) -> bool:
    """Return True if the MongoDB database already has at least one collection."""
    result = subprocess.run(
        ["mongosh", "--quiet", "--host", host, "--port", str(port),
         "--eval", f"db.getSiblingDB('{db_name}').getCollectionNames().length"],
        capture_output=True, text=True,
    )
    try:
        return int(result.stdout.strip()) > 0
    except (ValueError, TypeError):
        return False


def _ensure_mongo_db(db_name: str, dump_folder: Path, host: str = "localhost", port: int = 27017) -> None:
    """Restore a MongoDB dump if the database doesn't already have collections."""
    if _mongo_db_exists(db_name, host, port):
        log.info(f"MongoDB database '{db_name}' already exists — skipping restore.")
        return

    if not dump_folder.exists():
        raise RuntimeError(
            f"MongoDB dump folder not found: {dump_folder}\n"
            "Run 'git lfs pull' inside the DataAgentBench directory to download data files."
        )

    log.info(f"Restoring MongoDB database '{db_name}' from {dump_folder.name}…")
    result = subprocess.run(
        ["mongorestore", f"--host={host}", f"--port={port}",
         f"--nsInclude={db_name}.*", str(dump_folder)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, "mongorestore", result.stderr)
    log.info(f"Restored MongoDB '{db_name}' successfully.")


def _build_uris_from_config(dataset_dir: Path) -> list[str]:
    """
    Parse db_config.yaml, auto-load any missing PostgreSQL/MongoDB databases,
    and return DataBridge-compatible URIs.
    """
    config_path = dataset_dir / "db_config.yaml"
    if not config_path.exists():
        return []

    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    uris: list[str] = []

    user, password, host, port, _ = _pg_env()

    for _alias, db in config.get("db_clients", {}).items():
        db_type = db.get("db_type", "").lower()

        if db_type in ("postgres", "postgresql"):
            db_name = db.get("db_name", "")
            sql_file_rel = db.get("sql_file", "")
            if sql_file_rel:
                _ensure_postgres_db(db_name, (dataset_dir / sql_file_rel).resolve())
            auth = f"{user}:{password}@" if password else f"{user}@"
            uris.append(f"postgresql://{auth}{host}:{port}/{db_name}")

        elif db_type == "sqlite":
            full_path = (dataset_dir / db.get("db_path", "")).resolve()
            uris.append(f"sqlite:///{full_path}")

        elif db_type == "duckdb":
            full_path = (dataset_dir / db.get("db_path", "")).resolve()
            uris.append(f"duckdb:///{full_path}")

        elif db_type in ("mongo", "mongodb"):
            host_m = db.get("host", "localhost")
            port_m = db.get("port", 27017)
            db_name_m = db.get("db_name", "")
            dump_folder_rel = db.get("dump_folder", "")
            if dump_folder_rel:
                _ensure_mongo_db(db_name_m, (dataset_dir / dump_folder_rel).resolve(), host_m, port_m)
            uris.append(f"mongodb://{host_m}:{port_m}/{db_name_m}")

    return uris


def _read_db_description(dataset_dir: Path, use_hints: bool = True) -> str:
    """Return the database description text to include in the agent's context."""
    hint_path = dataset_dir / "db_description_withhint.txt"
    plain_path = dataset_dir / "db_description.txt"
    if use_hints and hint_path.exists():
        return hint_path.read_text(encoding="utf-8")
    if plain_path.exists():
        return plain_path.read_text(encoding="utf-8")
    return ""


_SCHEMA_TABLE_LIMIT = 30  # max tables per DB before switching to compact listing

_TRANSFORM_SQL_TEMPLATES: dict[str, str] = {
    "identity":          "CAST({a} AS VARCHAR) = CAST({b} AS VARCHAR)",
    "lowercase":         "LOWER(CAST({a} AS VARCHAR)) = LOWER(CAST({b} AS VARCHAR))",
    "extract_digits":    "REGEXP_REPLACE(CAST({a} AS VARCHAR),'[^0-9]','','g') = REGEXP_REPLACE(CAST({b} AS VARCHAR),'[^0-9]','','g')",
    "strip_prefix_3":    "CAST({a} AS VARCHAR) = SUBSTRING(CAST({b} AS VARCHAR),4)",
    "strip_prefix_4":    "CAST({a} AS VARCHAR) = SUBSTRING(CAST({b} AS VARCHAR),5)",
    "strip_prefix_5":    "CAST({a} AS VARCHAR) = SUBSTRING(CAST({b} AS VARCHAR),6)",
    "zfill_5":           "LPAD(CAST({a} AS VARCHAR),5,'0') = CAST({b} AS VARCHAR)",
    "zfill_7":           "LPAD(CAST({a} AS VARCHAR),7,'0') = CAST({b} AS VARCHAR)",
    "remove_separators": "REPLACE(REPLACE(CAST({a} AS VARCHAR),'-',''),'_','') = REPLACE(REPLACE(CAST({b} AS VARCHAR),'-',''),'_','')",
    "cast_int":          "CAST({a} AS INTEGER) = CAST({b} AS INTEGER)",
}


def _build_schema_context(
    schema: dict,
    join_rules: list,
    sample_rows: "dict[str, dict[str, list]] | None" = None,
) -> str:
    """
    Build a compact schema context string from the cached schema and join registry.
    Injected into each question so the agent starts with full structural knowledge.
    sample_rows: optional {db_alias: {table_name: [row_dict, ...]}} for sample data.
    """
    lines = ["## Database Schema (pre-loaded — skip db_schema discovery call)\n"]

    for db_alias, tables in schema.items():
        if not tables:
            continue
        lines.append(f"### {db_alias}")

        table_names = list(tables.keys())
        n_tables = len(table_names)

        if n_tables > _SCHEMA_TABLE_LIMIT:
            # Many same-structure tables (e.g. per-ticker stock tables).
            # List all names compactly + one representative column schema to avoid
            # bloating the prompt with thousands of identical table descriptions.
            rep_tname, rep_table = next(iter(tables.items()))
            col_parts = []
            for cname, col in rep_table.columns.items():
                display = f'"{cname}"' if (db_alias == "postgresql" and cname != cname.lower()) else cname
                part = f"{display} ({col.dtype})"
                if col.is_unstructured:
                    part += "*"
                col_parts.append(part)
            _NAME_DISPLAY_LIMIT = 50
            if n_tables > _NAME_DISPLAY_LIMIT:
                shown = table_names[:_NAME_DISPLAY_LIMIT]
                name_str = f"{', '.join(shown)}, ... and {n_tables - _NAME_DISPLAY_LIMIT} more (use db_schema to list all)"
            else:
                name_str = ', '.join(table_names)
            lines.append(
                f"⚠ {n_tables} tables (all share the same structure). "
                f"Table names: {name_str}"
            )
            lines.append(
                f"  Columns (representative — same for all {n_tables} tables): {', '.join(col_parts)}"
            )
        else:
            for tname, table in tables.items():
                col_parts = []
                for cname, col in table.columns.items():
                    # PostgreSQL folds unquoted identifiers to lowercase — show mixed-case
                    # column names already quoted so the agent writes them correctly in SQL.
                    display = f'"{cname}"' if (db_alias == "postgresql" and cname != cname.lower()) else cname
                    part = f"{display} ({col.dtype})"
                    if col.is_unstructured:
                        part += "*"
                    col_parts.append(part)
                row_info = f"~{table.row_count_approx:,}" if table.row_count_approx else "?"
                # PostgreSQL: quote table names that are mixed-case or SQL reserved words
                # (e.g. "Case", "Order", "User") so the agent writes them correctly.
                display_tname = f'"{tname}"' if (db_alias == "postgresql" and tname != tname.lower()) else tname
                lines.append(f"**{display_tname}** ({row_info} rows): {', '.join(col_parts)}")
                for fk in table.foreign_keys:
                    lines.append(f"  FK: {tname}.{fk.column} → {fk.ref_table}.{fk.ref_column}")
                # Show sample rows so the agent knows real data formats without extra queries
                if sample_rows:
                    rows = (sample_rows.get(db_alias) or {}).get(tname, [])
                    if rows:
                        sample = {k: (str(v)[:80] if v is not None else "") for k, v in rows[0].items()}
                        lines.append(f"  sample row: {json.dumps(sample)}")
        lines.append("")

    lines.append("*unstructured text field — sample before filtering\n")

    # Show a join rule if:
    # 1. It has been confirmed by a successful cross-DB query.
    # 2. It is an identity transform between the same-named column on both sides
    #    at high confidence — safe to show without risking normalization confusion
    #    (e.g. gmap_id ↔ gmap_id).
    # 3. It is a non-identity (i.e. requires data transformation) join at high
    #    confidence — helps the model understand *how* to join prefixed IDs like
    #    book_id / purchase_id. Identity joins with different column names (e.g.
    #    rating ↔ rating_number) are excluded because they are common false
    #    positives from auto-discovery.
    seen_pairs: set = set()
    confirmed_joins = []
    for r in join_rules:
        if not (
            r.confirmed
            or (r.transform == "identity" and r.column_a == r.column_b and r.confidence >= 0.85)
            or (r.transform != "identity" and r.confidence >= 0.80)
        ):
            continue
        pair = (r.db_a, r.table_a, r.column_a, r.db_b, r.table_b, r.column_b)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        confirmed_joins.append(r)
    _JOIN_DISPLAY_LIMIT = 30
    if len(confirmed_joins) > _JOIN_DISPLAY_LIMIT:
        # Too many joins to show all — deduplicate to unique column-pair patterns
        # (e.g. 1743 ticker-level joins collapse to one representative)
        seen_col_pairs: set = set()
        deduped: list = []
        for r in confirmed_joins:
            col_pair = (r.column_a, r.column_b, r.transform)
            if col_pair not in seen_col_pairs:
                seen_col_pairs.add(col_pair)
                deduped.append(r)
        confirmed_joins = deduped[:_JOIN_DISPLAY_LIMIT]

    if confirmed_joins:
        lines.append("## Known Cross-Database Joins")
        for rule in confirmed_joins:
            status = "confirmed" if rule.confirmed else f"confidence={rule.confidence:.2f}"
            ta = f".{rule.table_a}" if rule.table_a else ""
            tb = f".{rule.table_b}" if rule.table_b else ""
            lines.append(
                f"- {rule.db_a}{ta}.{rule.column_a} ↔ {rule.db_b}{tb}.{rule.column_b} ({status})"
            )
            # Show the join transform SQL so the agent doesn't have to guess ID formats
            tmpl = _TRANSFORM_SQL_TEMPLATES.get(rule.transform or "")
            if tmpl and rule.transform not in ("identity", "lowercase", "cast_int"):
                expr = tmpl.format(a=rule.column_a, b=rule.column_b)
                lines.append(
                    f"  ⚠ IDs use different formats — pre-transform in each sub-query before joining:"
                )
                lines.append(
                    f"  {rule.db_a}: SELECT REGEXP_REPLACE(CAST({rule.column_a} AS VARCHAR),'[^0-9]','','g') AS jk, ..."
                    if "REGEXP_REPLACE" in expr else
                    f"  Transform: {expr}"
                )
                lines.append(
                    f"  {rule.db_b}: SELECT REGEXP_REPLACE(CAST({rule.column_b} AS VARCHAR),'[^0-9]','','g') AS jk, ..."
                    if "REGEXP_REPLACE" in expr else
                    f"  Use aliased columns in join_on: [[\"key_a.jk\", \"key_b.jk\"]]"
                )
                lines.append(f"  join_on: [[\"pg.jk\", \"sq.jk\"]] (use your sub-query key prefixes)")
        lines.append("")

    return "\n".join(lines)


# ── Evaluator ─────────────────────────────────────────────────────────────────

class DABEvaluator:
    def __init__(
        self,
        dab_root: Path,
        results_dir: Path,
        config_path: Path | None = None,
        provider: str = "anthropic",
        model: str = "claude-opus-4-8",
        api_key: str | None = None,
        dataset: str | None = None,
        datasets: list[str] | None = None,
        use_hints: bool = True,
        run: int = 0,
    ) -> None:
        self._dab_root = dab_root
        self._results_dir = results_dir
        self._config_path = config_path
        self._provider = provider
        self._model = model
        self._api_key = api_key
        # Support both single-dataset (--dataset) and multi-dataset (--datasets/--official) filters.
        if datasets is not None:
            self._dataset: str | list[str] | None = datasets
        else:
            self._dataset = dataset
        self._use_hints = use_hints
        self._run = run
        self._results_dir.mkdir(parents=True, exist_ok=True)

    async def run(self) -> BenchmarkReport:
        _env_vars = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "groq": "GROQ_API_KEY",
            "kimi": "MOONSHOT_API_KEY",
            "together": "TOGETHER_API_KEY",
        }
        # Resolve API key from provider-specific env var if not explicitly supplied.
        api_key = self._api_key
        if self._provider == "ollama":
            api_key = api_key or "ollama"  # Ollama runs locally — no real key needed
        else:
            if api_key is None:
                env_var = _env_vars.get(self._provider)
                if env_var:
                    api_key = os.environ.get(env_var)
                if not api_key:
                    raise RuntimeError(
                        f"No API key for provider '{self._provider}'. "
                        f"Set {env_var or 'the appropriate API key env var'} or pass --api-key."
                    )

        # Sanity-check: catch Anthropic key used for wrong provider.
        if self._provider in ("openai", "groq", "kimi", "together") and api_key and api_key.startswith("sk-ant-"):
            raise RuntimeError(
                f"API key looks wrong for provider '{self._provider}': key starts with 'sk-ant-', "
                f"which is an Anthropic key. Set {_env_vars[self._provider]} or pass --api-key."
            )

        from databridge.audit.log import AuditLog
        from databridge.connectors.registry import ConnectorRegistry
        from databridge.engine import DataBridgeEngine
        from databridge.query.executor import QueryExecutor
        from databridge.query.planner import QueryPlanner
        from databridge.query.translator import QueryTranslator
        from databridge.safety.enforcement import SafetyEnforcer
        from databridge.schema.cache import SchemaCache
        from databridge.schema.joins.discovery import JoinDiscovery
        from databridge.schema.joins.registry import JoinRegistry
        from databridge.schema.scanner import SchemaScanner
        from databridge.verification.plausibility import PlausibilityChecker
        from benchmark.agent import ToolExecutor, build_agent

        queries = _discover_queries(self._dab_root, dataset_filter=self._dataset)
        if not queries:
            raise RuntimeError(
                f"No queries found in {self._dab_root}. "
                f"Check --dab-root points to the DataAgentBench directory."
            )

        results: list[QueryResult] = []

        # Group by dataset so we connect/disconnect once per dataset
        datasets: dict[str, list[tuple[str, Path]]] = {}
        for dataset_name, query_id, query_dir in queries:
            datasets.setdefault(dataset_name, []).append((query_id, query_dir))

        _order_idx = {name: i for i, name in enumerate(OFFICIAL_DATASET_ORDER)}

        def _log_dataset_summary(name: str) -> None:
            ds_r = [r for r in results if r.dataset == name]
            ds_p = sum(1 for r in ds_r if r.passed)
            log.info("─" * 56)
            log.info("[%s]  %d/%d passed", name, ds_p, len(ds_r))
            for r in ds_r:
                ans = (r.answer or r.error or "")[:55].replace("\n", " ")
                log.info("  %-10s  %s  %s", r.query_id, "✓" if r.passed else "✗", ans)
            log.info("─" * 56)

        for dataset_name, dataset_queries in sorted(
            datasets.items(),
            key=lambda kv: (_order_idx.get(kv[0], len(OFFICIAL_DATASET_ORDER)), kv[0]),
        ):
            dataset_dir = self._dab_root / f"query_{dataset_name}"
            uris = _build_uris_from_config(dataset_dir)

            if not uris:
                for query_id, query_dir in dataset_queries:
                    question = _read_question(query_dir)
                    results.append(QueryResult(
                        dataset=dataset_name,
                        query_id=query_id,
                        question=question,
                        answer="",
                        passed=False,
                        reason="No database URIs found in db_config.yaml",
                        execution_ms=0.0,
                        error="no_db_config",
                    ))
                _log_dataset_summary(dataset_name)
                continue

            # Set up DataBridge stack for this dataset
            cache_dir = Path(f"~/.databridge/bench_{dataset_name}").expanduser()
            cache_dir.mkdir(parents=True, exist_ok=True)

            # On run 0, clear stale join cache so each submission round starts
            # with a clean discovery pass. Agent-loop joins from prior runs can
            # corrupt the registry with false-positive joins that mislead the model.
            if self._run == 0:
                joins_db = cache_dir / "joins.db"
                if joins_db.exists():
                    joins_db.unlink()
                    log.info("Run 0: cleared join cache for '%s' (fresh discovery)", dataset_name)

            registry = ConnectorRegistry.from_uris(uris)
            try:
                await registry.connect_all()
            except Exception as e:
                for query_id, query_dir in dataset_queries:
                    question = _read_question(query_dir)
                    results.append(QueryResult(
                        dataset=dataset_name,
                        query_id=query_id,
                        question=question,
                        answer="",
                        passed=False,
                        reason=f"DB connection failed: {e}",
                        execution_ms=0.0,
                        error=str(e),
                    ))
                _log_dataset_summary(dataset_name)
                continue

            cache = SchemaCache(cache_dir / "schema.db", ttl_hours=24)
            join_registry = JoinRegistry(cache_dir / "joins.db")
            scanner = SchemaScanner(registry, cache)
            planner = QueryPlanner(registry, join_registry)
            enforcer = SafetyEnforcer()
            translator = QueryTranslator()
            executor = QueryExecutor(registry, enforcer, translator, 10_000)
            checker = PlausibilityChecker()
            audit = AuditLog(cache_dir / "audit.db")
            discovery = JoinDiscovery(registry)

            engine = DataBridgeEngine(
                registry=registry,
                scanner=scanner,
                planner=planner,
                executor=executor,
                join_registry=join_registry,
                checker=checker,
                audit=audit,
                learner=None,
                discovery=discovery,
            )

            # Pre-scan schema — use the 24h cache on repeat runs; only force a
            # fresh scan when the cache is empty (first run or cache evicted).
            cached_schema = await scanner.scan_all(force=False)
            cache_empty = not any(cached_schema.values())
            log.info(f"Pre-scanning schema for dataset '{dataset_name}'…")
            fresh_schema = await scanner.scan_all(force=cache_empty)
            for alias, tables in fresh_schema.items():
                if tables:
                    log.info(f"  {alias}: {list(tables.keys())}")
                else:
                    log.warning(f"  {alias}: empty schema (connection may have failed)")

            # Auto-discover cross-database joins — skip if already cached.
            existing_joins = await join_registry.get_all()
            if not existing_joins:
                log.info(f"Discovering joins for dataset '{dataset_name}'…")
                try:
                    await engine.joins(discover=True)
                except Exception as e:
                    log.warning(f"Join discovery failed: {e}")
            else:
                log.info(
                    "Using %d cached join(s) for '%s' — skipping discovery",
                    len(existing_joins), dataset_name,
                )


            # Sample 2 rows per table so the schema context shows real data formats.
            # This lets the model know column formats (e.g. date text in "details")
            # without spending turns on exploratory queries.
            session_id = f"bench_{dataset_name}"
            sample_rows: dict[str, dict[str, list]] = {}
            for alias, tables in fresh_schema.items():
                sample_rows[alias] = {}
                if len(tables) > _SCHEMA_TABLE_LIMIT:
                    log.info("  %s: skipping pre-scan sample (%d tables — compact schema used)", alias, len(tables))
                    continue
                for tname in tables:
                    try:
                        q = f'SELECT * FROM "{tname}" LIMIT 2' if alias == "postgresql" else f"SELECT * FROM \"{tname}\" LIMIT 2"
                        res = await engine.query(q, databases=[alias], session_id=session_id)
                        sample_rows[alias][tname] = res.get("rows", [])
                    except Exception as e:
                        log.debug("Pre-scan sample failed for %s.%s: %s", alias, tname, e)

            tool_executor = ToolExecutor(engine, session_id)
            db_description = _read_db_description(dataset_dir, self._use_hints)

            for query_id, query_dir in dataset_queries:
                question = _read_question(query_dir)

                # Rebuild context before each question so newly discovered joins are included
                join_rules = await join_registry.get_all()
                schema_context = _build_schema_context(fresh_schema, join_rules, sample_rows=sample_rows)

                context_parts = []
                if db_description:
                    context_parts.append(f"Database context:\n{db_description}")
                context_parts.append(schema_context)
                full_question = "\n\n".join(context_parts) + f"\n\nQuestion: {question}"

                log.info("[%s/%s] schema_context=%d chars full_question=%d chars",
                         dataset_name, query_id, len(schema_context), len(full_question))
                log.info("[%s/%s] %s", dataset_name, query_id, question[:80])
                tool_executor.reset()
                db_types = {registry.get(alias).db_type.value for alias in registry.aliases()}
                agent = build_agent(self._provider, self._model, tool_executor, api_key, db_types=db_types)

                t0 = time.monotonic()
                answer = ""
                error = None

                try:
                    answer = await agent.answer(full_question)
                except Exception as e:
                    error = str(e)

                elapsed = (time.monotonic() - t0) * 1000

                passed, reason = False, ""
                if not error:
                    try:
                        passed, reason = _call_validate(query_dir, answer)
                    except Exception as e:
                        error = f"validate.py error: {e}"

                status = "✓" if passed else "✗"
                ans_short = (answer or error or "")[:60].replace("\n", " ")
                log.info("[%s/%s] %s %.0fs — %s", dataset_name, query_id, status, elapsed / 1000, ans_short)

                results.append(QueryResult(
                    dataset=dataset_name,
                    query_id=query_id,
                    question=question,
                    answer=answer,
                    passed=passed,
                    reason=reason,
                    execution_ms=elapsed,
                    error=error,
                ))

            await registry.disconnect_all()
            _log_dataset_summary(dataset_name)

            # Write both the submission file and a partial debug report after each
            # dataset so a crash mid-run doesn't lose results already collected.
            _model_slug_inc = self._model.replace("/", "-")
            _ds_tag_inc = (
                self._dataset if isinstance(self._dataset, str)
                else ("official" if isinstance(self._dataset, list) and set(self._dataset) == OFFICIAL_DATASETS else "custom")
                if self._dataset else "all"
            )

            # Partial debug report — cumulative, overwritten after each dataset
            _partial_total = len(results)
            _partial_passed = sum(1 for r in results if r.passed)
            _partial_errors = sum(1 for r in results if r.error)
            _partial_ds_stats: dict[str, float] = {}
            for _ds in set(r.dataset for r in results):
                _ds_rs = [r for r in results if r.dataset == _ds]
                _partial_ds_stats[_ds] = sum(1 for r in _ds_rs if r.passed) / len(_ds_rs)
            _partial_report = BenchmarkReport(
                provider=self._provider,
                model=self._model,
                total=_partial_total,
                passed=_partial_passed,
                failed=_partial_total - _partial_passed - _partial_errors,
                errors=_partial_errors,
                pass_at_1=_partial_passed / _partial_total if _partial_total else 0.0,
                datasets=_partial_ds_stats,
                results=results,
            )
            _partial_path = self._results_dir / f"dab_{self._provider}_{_model_slug_inc}_{_ds_tag_inc}_run{self._run}_partial.json"
            _partial_path.write_text(json.dumps(asdict(_partial_report), indent=2, default=str))

            _sub_path = self._results_dir / f"submission_{_model_slug_inc}.json"
            _existing: list[dict] = []
            if _sub_path.exists():
                try:
                    _existing = json.loads(_sub_path.read_text())
                except Exception:
                    _existing = []
            _entries = {(e["dataset"], e["query"], e["run"]): e for e in _existing}
            _run_str = str(self._run)
            for r in results:
                _entries[(r.dataset.lower(), r.query_id.removeprefix("query"), _run_str)] = {
                    "dataset": r.dataset.lower(),
                    "query": r.query_id.removeprefix("query"),
                    "run": _run_str,
                    "answer": r.answer,
                }
            _sub_path.write_text(
                json.dumps(sorted(_entries.values(), key=lambda e: (e["dataset"], e["query"], e["run"])), indent=2)
            )

        # Aggregate
        total = len(results)
        passed_count = sum(1 for r in results if r.passed)
        error_count = sum(1 for r in results if r.error)

        # Per-dataset pass rates
        dataset_stats: dict[str, float] = {}
        for ds in set(r.dataset for r in results):
            ds_results = [r for r in results if r.dataset == ds]
            ds_pass = sum(1 for r in ds_results if r.passed)
            dataset_stats[ds] = ds_pass / len(ds_results) if ds_results else 0.0

        report = BenchmarkReport(
            provider=self._provider,
            model=self._model,
            total=total,
            passed=passed_count,
            failed=total - passed_count - error_count,
            errors=error_count,
            pass_at_1=passed_count / total if total else 0.0,
            datasets=dataset_stats,
            results=results,
        )

        # Internal format — full detail for development/debugging
        _ds_tag = (
            self._dataset if isinstance(self._dataset, str)
            else ("official" if isinstance(self._dataset, list) and set(self._dataset) == OFFICIAL_DATASETS else "custom")
            if self._dataset else "all"
        )
        _model_slug = self._model.replace("/", "-")
        out_path = (
            self._results_dir
            / f"dab_{self._provider}_{_model_slug}_{_ds_tag}_{int(time.time())}.json"
        )
        out_path.write_text(json.dumps(asdict(report), indent=2, default=str))

        # DAB submission format — accumulates across runs in one file
        # Required shape: [{"dataset": str, "query": str, "run": str, "answer": str}, ...]
        submission_path = (
            self._results_dir
            / f"submission_{self._model.replace('/', '-')}.json"
        )
        submission_entries: list[dict] = []
        if submission_path.exists():
            try:
                submission_entries = json.loads(submission_path.read_text())
            except Exception:
                submission_entries = []

        # Remove any existing entries for this (dataset, query, run) to allow re-runs
        run_str = str(self._run)
        new_entries = {
            (e["dataset"], e["query"], e["run"]): e
            for e in submission_entries
        }
        for r in results:
            ds = r.dataset.lower()
            q = r.query_id.removeprefix("query")  # "query1" → "1"
            new_entries[(ds, q, run_str)] = {
                "dataset": ds,
                "query": q,
                "run": run_str,
                "answer": r.answer,
            }
        submission_path.write_text(
            json.dumps(sorted(new_entries.values(), key=lambda e: (e["dataset"], e["query"], e["run"])), indent=2)
        )
        log.info("Results saved to: %s", self._results_dir)
        return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dab-root", required=True, type=Path,
                        help="Path to cloned DataAgentBench directory")
    parser.add_argument("--results-dir", default=Path("benchmark/results"), type=Path)
    parser.add_argument("--provider", default="anthropic",
                        choices=["anthropic", "openai", "groq", "kimi", "together", "ollama"])
    parser.add_argument("--model", default="claude-opus-4-8")
    parser.add_argument("--dataset", default=None,
                        help="Run one dataset only (e.g. bookreview)")
    parser.add_argument("--datasets", default=None,
                        help="Comma-separated list of datasets to run (e.g. agnews,yelp,bookreview)")
    parser.add_argument("--official", action="store_true",
                        help="Run only the 12 officially listed DataAgentBench datasets (excludes imdb, cve, krama, civic_unstructured, usaspending)")
    parser.add_argument("--no-hints", action="store_true",
                        help="Use db_description.txt instead of db_description_withhint.txt")
    parser.add_argument("--run", type=int, default=0,
                        help="Run index 0–4 (for 5-run leaderboard submission)")
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    # Resolve dataset filter: --official > --datasets > --dataset > default (official 12)
    if args.official:
        datasets_arg = list(OFFICIAL_DATASETS)
        dataset_arg = None
    elif args.datasets:
        datasets_arg = [d.strip() for d in args.datasets.split(",") if d.strip()]
        dataset_arg = None
    elif args.dataset:
        datasets_arg = None
        dataset_arg = args.dataset
    else:
        # Default: run only the 12 official datasets; skips krama, imdb, cve, etc.
        datasets_arg = list(OFFICIAL_DATASETS)
        dataset_arg = None

    evaluator = DABEvaluator(
        dab_root=args.dab_root,
        results_dir=args.results_dir,
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        dataset=dataset_arg,
        datasets=datasets_arg,
        use_hints=not args.no_hints,
        run=args.run,
    )
    report = asyncio.run(evaluator.run())
    print(report.summary)
    for ds, rate in sorted(report.datasets.items()):
        print(f"  {ds}: {rate:.1%}")
