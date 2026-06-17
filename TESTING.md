# DataBridge Testing Guide

## Unit Tests

DataBridge uses [pytest](https://docs.pytest.org) with asyncio support. Tests use in-memory SQLite databases — no external services or API keys required.

### Install

```bash
pip install -e ".[dev]"
```

### Run

```bash
pytest tests/          # all tests
pytest tests/ -v       # verbose
pytest tests/test_safety.py -v          # single file
pytest tests/test_joins.py::test_join_registry_confirm -v   # single test
pytest tests/ --cov=databridge --cov-report=term-missing    # with coverage
pytest tests/ -k "transform"            # keyword filter
```

### Test structure

```
tests/
├── conftest.py          # Shared fixtures (SQLite registry, schema cache, join registry)
├── test_safety.py       # SQL write/DDL enforcement (parser level)
├── test_connectors.py   # Database connector layer
├── test_schema.py       # Schema cache + SchemaScanner
├── test_joins.py        # Transform grammar + JoinRegistry CRUD
├── test_query.py        # QueryTranslator, QueryPlanner, QueryExecutor
├── test_verification.py # PlausibilityChecker
├── test_audit.py        # AuditLog (record, replay, session, recent)
└── test_mcp.py          # MCP tool logic (no MCP protocol overhead)
```

The `conftest.py` fixture creates a temporary SQLite database per test with `orders` (50 rows) and `customers` (10 rows) tables. All temp files are cleaned up after each test.

---

## DAB Benchmark

The benchmark is a development signal. Results are written locally and never auto-submitted.

### How it works

```
NL question → LLM calls db_schema → LLM calls db_query(SQL) → answer compared to ground truth
```

DataBridge is the tool layer. The LLM is the planner. The benchmark score reflects the full system.

### Prerequisites

1. Install benchmark dependencies:
   ```bash
   pip install -e ".[benchmark]"
   ```

2. Clone the DAB dataset:
   ```bash
   git clone https://github.com/ucbepic/DataAgentBench
   ```

3. MongoDB must be running locally for MongoDB datasets (`mongod`).

4. Set your API key:
   ```bash
   export TOGETHER_API_KEY=...    # for GLM-5.1 (cheap, recommended)
   export ANTHROPIC_API_KEY=...   # for Claude
   export OPENAI_API_KEY=...      # for OpenAI
   ```

### Run

```bash
# Single dataset (fastest for iteration)
databridge benchmark run \
  --dab-root DataAgentBench \
  --dataset bookreview \
  --provider together \
  --model zai-org/GLM-5.1

# All 12 official datasets
databridge benchmark run \
  --dab-root DataAgentBench \
  --provider together \
  --model zai-org/GLM-5.1 \
  --official

# Verbose (shows each tool call)
databridge benchmark run \
  --dab-root DataAgentBench \
  --dataset bookreview \
  --provider anthropic \
  --model claude-opus-4-8 \
  --verbose
```

Results are written to `benchmark/results/`.

### Leaderboard submission (5 runs required)

```bash
for i in 0 1 2 3 4; do
  databridge benchmark run \
    --dab-root DataAgentBench \
    --provider together \
    --model zai-org/GLM-5.1 \
    --official \
    --run $i
done
```

Submit the `benchmark/results/` directory to the DAB leaderboard.

### Official datasets (54 queries, 12 datasets)

| Dataset | DBMSes | Queries |
|---------|--------|---------|
| `bookreview` | PostgreSQL + SQLite | 3 |
| `googlelocal` | PostgreSQL + SQLite | 4 |
| `patents` | PostgreSQL + SQLite | 3 |
| `deps_dev_v1` | DuckDB + SQLite | 2 |
| `music_brainz_20k` | DuckDB + SQLite | 3 |
| `stockindex` | DuckDB + SQLite | 2 |
| `stockmarket` | DuckDB + SQLite | 5 |
| `github_repos` | DuckDB + SQLite | 4 |
| `pancancer_atlas` | DuckDB + PostgreSQL | 3 |
| `agnews` | MongoDB + SQLite | 4 |
| `yelp` | DuckDB + MongoDB | 7 |
| `crmarenapro` | DuckDB + PostgreSQL + SQLite | 13 |

### Resetting benchmark cache

Each dataset caches schema and join discovery at `~/.databridge/bench_<dataset>/`.

```bash
rm -rf ~/.databridge/bench_bookreview/   # reset one dataset
rm -rf ~/.databridge/                    # reset all
```

Reset when: you've changed a test database schema, or a previous run cached stale data.

**Reset a PostgreSQL dataset** (so the harness recreates it from the SQL file):
```bash
psql -U postgres -c "DROP DATABASE IF EXISTS review_GoogleLocal;"
```

---

## Environment Variables

Unit tests require no environment variables. The benchmark uses:

```bash
TOGETHER_API_KEY=...        # Together AI (GLM-5.1)
ANTHROPIC_API_KEY=...       # Anthropic (Claude)
OPENAI_API_KEY=...          # OpenAI

DATABRIDGE_DATABASE_URIS=postgresql://...   # for manual server usage
```

All config is read via `pydantic-settings` with the `DATABRIDGE_` prefix.

---

## CI

All tests in `tests/` are safe to run in CI with no external services. The benchmark harness is excluded from CI — it requires the DAB dataset, real databases, and an LLM API key.
