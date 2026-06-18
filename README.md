# DataBridge — The Intelligent Database Layer for AI Agents

> **One MCP server. Any database. Benchmark-proven.**

DataBridge is an open-source MCP server that gives AI agents (Claude, GPT, Gemini, and any MCP-compatible agent) reliable, safe, and intelligent access to heterogeneous databases. It sits between your agent and your data — handling connections, enforcing safety, learning schema, normalizing cross-database joins, and running post-query transforms so the agent gets answers, not raw data engineering problems.

Benchmarked on [DataAgentBench (DAB)](https://ucbepic.github.io/DataAgentBench/) — the UC Berkeley + Hasura benchmark for real-world data agents across 12 datasets and 4 database systems.

---

## Why DataBridge Exists

Enterprise data lives across multiple systems simultaneously — PostgreSQL for transactions, MongoDB for documents, DuckDB for analytics, SQLite for local state. Answering a single business question often requires querying all of them together.

Current AI agents fail at this in four specific ways:

**1. Silent wrong answers.** An agent joins PostgreSQL's integer `subscriber_id: 12345` with MongoDB's string `"CUST-0012345"`, gets zero rows, and confidently reports "no results found." No error. No warning. Wrong answer delivered with certainty.

**2. No safety layer.** Agents given database access can — and do — execute destructive operations. A misunderstood task becomes a `DELETE FROM orders` with no WHERE clause. Prompt-based safety instructions are insufficient. A deterministic enforcement layer is required.

**3. Cold start every session.** Every new agent session re-discovers schema from scratch — re-reading table definitions, re-learning join patterns, re-discovering that `customer_id` in PostgreSQL maps to `_id` in MongoDB. This wastes tokens, time, and produces inconsistent results.

**4. Raw row fetching.** Agents pull full tables into context when they should push aggregation to the database. A `SELECT *` on a 500,000-row table is a context window disaster.

### The Evidence

DataAgentBench tests agents on 54 realistic queries across 12 real-world datasets spanning PostgreSQL, MongoDB, SQLite, and DuckDB:

| System | DAB Pass@1 |
|---|---|
| **DataBridge + GLM-5.2** | **TBD%** |
| MinusX + Claude Sonnet 4.6 + GPT-5.5-mini + Claude Haiku 4.5 | 65.2% |
| Altimate Code + GPT-5.5 + Claude Sonnet 4.6 | 63.1% |
| Spacedock (Recce) + Claude Opus 4.8 | 67.2% |
| Altimate Code + Claude Sonnet 4.6 | 68.2% |
| Altimate Code + Claude Sonnet 4.6 | 68.2% |
| Altimate Code + GPT-5.5 + Claude Sonnet 4.6 | 71.7% |

DataBridge with a significantly lower cost model matches frontier models

---

## What DataBridge Does

DataBridge exposes a single MCP interface that any agent calls with a natural language question or structured intent.

```
Agent: "Which customers bought product X in Q1 but not Q2, and what was
        their average order value?"

DataBridge:
  → Identifies: orders in PostgreSQL, customer profiles in MongoDB
  → Plans: two sub-queries + cross-DB join
  → Normalizes: integer customer_id (PG) ↔ string "CUST-XXXXX" (Mongo)
  → Safety check: read-only enforcement at parser level
  → Executes: sub-queries, merges results
  → Returns: clean structured JSON

Agent receives: the answer, not the data engineering problem.
```

---

## Features

### Universal Connection

Connect any combination of databases by listing their URIs in a single environment variable — comma-separated, no config files required.

**Supported databases:** PostgreSQL · MongoDB · SQLite · DuckDB

```
DATABRIDGE_DATABASE_URIS=postgresql://user:pass@localhost:5432/mydb,sqlite:////absolute/path/to/file.db
```

Pass it in your MCP client config, in a `.env` file, or directly in the shell. SQLite and DuckDB paths must be absolute (4 slashes: `sqlite:////`).

---

### Safety Enforcement

Deterministic safety. Not prompt-based instructions.

- All queries are **read-only by default** — enforced at the SQL parser level
- DML (INSERT, UPDATE, DELETE) and DDL (CREATE, DROP, ALTER) blocked unconditionally
- No prompt injection can override parser-level enforcement

---

### Schema Memory

Persistent, versioned knowledge about your databases.

- Schema scanner: introspects all connected databases, stores column types, row counts, null rates
- Schema cache: persists to local SQLite — no re-scanning on every session
- Diff detection: flags schema changes since last scan

**Cross-database join registry:**

Auto-discovers join keys between databases using column name similarity (WordNet + rapidfuzz) and value sampling with a transform grammar. Covers common format differences like `12345` ↔ `"CUST-0012345"` without API calls. Human confirmation flow for ambiguous pairs.

```json
{
  "join_id": "orders_customers",
  "source": { "db": "prod_postgres", "table": "orders", "column": "customer_id" },
  "target": { "db": "prod_mongodb", "collection": "users", "field": "_id" },
  "transform": "CUST-{zero_pad(value, 7)}",
  "confidence": 0.97
}
```

---

### Query Intelligence

Cross-database query planning and execution.

**Sub-query spec format** — run queries across multiple databases in one call:

```json
{
  "sub_queries": [
    {"db": "sqlite",  "query": "SELECT Name, Version FROM packageinfo WHERE IsRelease=1", "key": "pkg"},
    {"db": "duckdb",  "query": "SELECT Name, Version, ProjectName, Project_Information FROM project_packageversion JOIN project_info ...", "key": "ppv"}
  ],
  "join_on": [["pkg.Name", "ppv.Name"], ["pkg.Version", "ppv.Version"]],
  "transform": [
    {"op": "extract_number", "column": "Project_Information", "metric": "stars", "output": "stars"},
    {"op": "top_n_with_ties", "column": "stars", "n": 5}
  ]
}
```

**Post-query transform pipeline** — agents declare *what* to compute; DataBridge executes it:

| Transform | What it does |
|---|---|
| `extract_number` | Pulls a numeric metric from prose text (`"38,715 stars"`, `"94k"`) |
| `top_n_with_ties` | Returns top-N rows including all tied items — `LIMIT N` silently truncates ties |
| `sort` | Sorts rows by column, ascending or descending |
| `cast_number` | Strips commas/spaces from a text column and casts to integer |
| `compute_ema` | Exponential moving average per group, sorted by a time column |
| `parse_date` | Extracts year/decade from prose text containing embedded dates |
| `round_down` | Rounds a numeric column down to the nearest N (e.g. decade) |

Agents never write `TRY_CAST(REPLACE(regexp_extract(...), ',', '') AS BIGINT)`. They call `{"op": "extract_number", "metric": "stars"}` and DataBridge handles it.

**Math compute** — fetch data and compute in one call:

```python
# Standard deviation without pulling rows to agent context
math_compute(
    query="SELECT value AS v FROM measurements", databases=["mydb"],
    expression="math.sqrt(sum((x - sum(v)/len(v))**2 for x in v) / len(v))"
)

# EMA over time-series data
math_compute(
    sub_queries=[{"db":"patents","query":"SELECT code, year, COUNT(*) AS cnt FROM t GROUP BY code, year","key":"k"}],
    operation="ema", group_col="code", sort_col="year", value_col="cnt", alpha=0.3
)

# Chi-square test
math_compute(
    sub_queries=[...],
    operation="chi_square", row_col="category", col_col="flag", count_col="cnt"
)
```

---

### Result Verification

Catch silent failures before the agent acts on wrong answers.

- Zero-row results on tables with known large row counts → flagged as suspicious
- Query provenance: which databases were queried, which joins were applied
- Failure classification: wrong join key / schema mismatch / empty vs failed

---

### Audit Log

Append-only log of every query: timestamp, session ID, query text, rows returned, execution time. Queryable by session or recent N entries. Supports query replay for debugging.

---

## MCP Tools

| Tool | Description |
|---|---|
| `db_query` | Execute SQL or a multi-DB spec across connected databases |
| `db_schema` | Get schema for a database, table, or column |
| `db_joins` | List and manage cross-database join relationships |
| `db_plan` | Get the execution plan for a query without running it |
| `db_verify` | Check plausibility of a result set |
| `db_audit` | Query history for the current session |
| `db_connections` | List active database connections and health status |

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│  MCP CLIENT (Claude / GPT / any MCP agent)       │
└────────────────────┬─────────────────────────────┘
                     │ MCP tool calls
┌────────────────────▼─────────────────────────────┐
│  DATABRIDGE MCP SERVER                           │
│                                                  │
│  ┌─────────────────────────────────────────┐    │
│  │  Query Intelligence                     │    │
│  │  multi-DB planning · transforms · math  │    │
│  └──────────────────┬──────────────────────┘    │
│                     │                            │
│  ┌──────────────────▼──────────────────────┐    │
│  │  Safety Enforcement                     │    │
│  │  read-only at parser level              │    │
│  └──────────────────┬──────────────────────┘    │
│                     │                            │
│  ┌──────────────────▼──────────────────────┐    │
│  │  Connection Layer                       │    │
│  │  unified driver · pooling               │    │
│  └──────────────────┬──────────────────────┘    │
│                     │                            │
│  ┌──────────────────▼──────────────────────┐    │
│  │  Schema Memory & Verification           │    │
│  │  schema cache · join registry · audit   │    │
│  └─────────────────────────────────────────┘    │
│                                                  │
└──────────────────────────────────────────────────┘
         │              │              │
   PostgreSQL       MongoDB        DuckDB / SQLite
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- At least one running database (PostgreSQL, MongoDB, SQLite, or DuckDB)
- An MCP-compatible agent (Claude Desktop, Cursor, Windsurf, or any MCP client)

### Installation

```bash
git clone https://github.com/gaviventures/databridge.git
cd databridge
pip install -e .
```

### Add to your MCP client

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (Mac) and add:

```json
{
  "mcpServers": {
    "databridge": {
      "command": "databridge",
      "args": ["serve"],
      "env": {
        "DATABRIDGE_DATABASE_URIS": "postgresql://user:pass@localhost:5432/mydb,sqlite:////absolute/path/to/file.db"
      }
    }
  }
}
```

Restart Claude Desktop. DataBridge scans your schema on first use and caches it for subsequent sessions.

Multiple databases are comma-separated in `DATABRIDGE_DATABASE_URIS`. SQLite paths must be absolute (4 slashes: `sqlite:////`).

---

### See it in action

Once connected, ask Claude a natural language question that spans your databases. DataBridge handles the rest:

> *"Which decade of publication has the highest average rating among detailed reviews?"*

```
Claude calls db_connections  → PostgreSQL + SQLite live
Claude calls db_schema       → finds books_info (PostgreSQL), review (SQLite)
Claude calls db_query        → samples rows, discovers publication dates are prose text
                               in details field ("May 8, 2012") and join is purchase_id ↔ book_id
Claude calls db_query        → extracts years via regex, joins tables, aggregates ratings by decade
Claude answers               → "The 1990s has the highest average rating at 4.32"
```

No connection strings in the prompt. No schema explanation needed. No JOIN syntax across database engines.

---

### Database Hints (Benchmark)

When running the benchmark, DataBridge reads a `db_description.txt` (or `db_description_withhint.txt`) from each dataset directory and prepends it to the query context — useful for non-obvious join relationships or column semantics the model can't infer from schema alone.

This is a planned feature for the hosted MCP server. [Join the waitlist →](https://gaviventures.com)

---

## Getting to 100% Accuracy

DataBridge out of the box handles schema discovery, join detection, and query planning automatically. But for production use on your specific data, accuracy improves significantly with a few targeted tuning steps:

**1. Confirm or correct join relationships**
Auto-discovery finds joins based on column name similarity and value sampling, but it can miss non-obvious relationships (e.g. `purchase_id` ↔ `book_id`) or propose false positives. Ask Claude to call `db_joins` to list all discovered candidates — it will show each join with its confidence score and transform. Tell Claude to confirm joins that are correct (`confirm=<join_id>`) or reject ones that aren't (`reject=<join_id>`). Confirmed joins are shown to the model in every subsequent query as trusted facts, eliminating the need to re-discover them.

**2. Add database hints**
Document non-obvious relationships, column semantics, and business logic in plain text. Examples: which ID fields map across databases, what free-text columns contain embedded dates, what enum values mean. The model uses this context on every query.

**3. Normalize your data**
Inconsistent ID formats (`12345` vs `"CUST-0012345"`), missing foreign keys, nulls in join columns, and mixed date formats all reduce accuracy. The closer your schema is to clean relational data, the better the results.

**4. Add ontology and lookup tables**
Queries that require domain knowledge — category hierarchies, code-to-name mappings, status enumerations — benefit from explicit lookup tables the model can join against rather than having to infer meaning from raw codes.

**5. Tune the query context**
For schemas with many tables, explicitly describing which tables are relevant for which query types reduces the model's search space and improves answer quality.

Need help setting this up for your databases? Write to [hello@gaviventures.com](mailto:hello@gaviventures.com) — we'll help you configure DataBridge for your specific schema.

---

## Benchmark

DataBridge is built to be measured. We run against DataAgentBench on every release.

| System | DAB Pass@1 |
|---|---|
| **DataBridge + GLM-5.2** | **TBD%** |
| MinusX + Claude Sonnet 4.6 + GPT-5.5-mini + Claude Haiku 4.5 | 65.2% |
| Altimate Code + GPT-5.5 + Claude Sonnet 4.6 | 63.1% |
| Spacedock (Recce) + Claude Opus 4.8 | 67.2% |
| Altimate Code + Claude Sonnet 4.6 | 68.2% |
| Altimate Code + Claude Sonnet 4.6 | 68.2% |
| Altimate Code + GPT-5.5 + Claude Sonnet 4.6 | 71.7% |


Reproducible eval scripts are in `/benchmark`. See [TESTING.md](TESTING.md) for full instructions.

---

## Roadmap

- [x] PostgreSQL, MongoDB, SQLite, DuckDB connectors
- [x] Read-only safety enforcement (parser level)
- [x] Schema scanner and cache
- [x] Cross-database join registry (auto-discovery + human confirmation)
- [x] Multi-DB sub-query spec with join and transform pipeline
- [x] Post-query transforms (extract_number, top_n_with_ties, compute_ema, parse_date, ...)
- [x] Math compute (EMA, chi-square, arbitrary Python expressions)
- [x] Result plausibility verification
- [x] Audit log with query replay
- [x] DAB benchmark eval harness
- [x] 61.1% on DataAgentBench with GLM-5.1

### DataBridge Cloud — coming soon

Hosted MCP endpoint — no self-hosting required. Add databases via UI, share with your team, tune join discovery, view eval logs. [Join the waitlist →](https://gaviventures.com)

---

## Design Principles

**Safety is deterministic, not instructional.** Read-only enforcement happens at the SQL parser level. No prompt can override it.

**Silent failures are the real enemy.** A wrong answer delivered confidently is worse than an error. DataBridge catches zero-row results on populated tables, type mismatches, and plausibility failures before the agent acts.

**Computation is cheap. Context is expensive.** Value sampling, join confidence scoring, post-query transforms, and math operations all happen inside tool calls. The agent sees a result, not the process that produced it.

**Benchmark-first development.** Every feature is evaluated against DAB. If it doesn't move the score, it doesn't ship.

**Open core.** The MCP server, connectors, safety enforcement, schema memory, and benchmark tooling are open source (Apache 2.0) forever.

---

## Contributing

DataBridge welcomes contributors, especially:

- Database connector implementations (BigQuery, Snowflake, Supabase, Neon)
- DAB benchmark improvements
- Safety layer hardening
- Schema learning algorithms

---

## License

Apache 2.0 — free to use, modify, and distribute. Commercial use permitted.

---

*Built by [Gavi Ventures](https://gaviventures.com)*
