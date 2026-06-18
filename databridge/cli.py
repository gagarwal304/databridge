from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Optional

# Load .env at import time so all env vars (API keys, PG*) are available
# without requiring the user to export them manually in the shell.
def _load_dotenv(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if key and key not in os.environ:  # don't override already-set env vars
            os.environ[key] = value

_load_dotenv()

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="DataBridge — The Intelligent Database Layer for AI Agents")
console = Console()


@app.command()
def serve(
    config_path: Optional[str] = typer.Option(None, "--config", envvar="DATABRIDGE_CONFIG"),
) -> None:
    """Start the DataBridge MCP server."""
    from databridge.config import get_config
    from databridge.mcp.server import create_server

    cfg = get_config()
    server = create_server(cfg)
    import sys
    print(f"DataBridge MCP server starting ({cfg.mcp_server_name})", file=sys.stderr)
    server.run()


@app.command()
def connect(
    uri: str = typer.Argument(..., help="Database URI to connect"),
    alias: Optional[str] = typer.Option(None, "--alias", "-a"),
) -> None:
    """Register a database URI in the local config."""
    import os
    from pathlib import Path

    env_path = Path(".env")
    existing = ""
    if env_path.exists():
        existing = env_path.read_text()

    # Append to DATABRIDGE_DATABASE_URIS
    if "DATABRIDGE_DATABASE_URIS=" in existing:
        lines = existing.splitlines()
        updated = []
        for line in lines:
            if line.startswith("DATABRIDGE_DATABASE_URIS="):
                current = line.split("=", 1)[1].strip()
                line = f"DATABRIDGE_DATABASE_URIS={current},{uri}" if current else f"DATABRIDGE_DATABASE_URIS={uri}"
            updated.append(line)
        env_path.write_text("\n".join(updated) + "\n")
    else:
        with env_path.open("a") as f:
            f.write(f"\nDATABRIDGE_DATABASE_URIS={uri}\n")

    console.print(f"[green]Registered:[/green] {uri}")


@app.command()
def schema(
    action: str = typer.Argument("scan", help="scan | diff | export"),
    database: Optional[str] = typer.Option(None, "--database", "-d"),
    force: bool = typer.Option(False, "--force", "-f"),
) -> None:
    """Manage schema cache: scan, diff, or export."""

    async def _run() -> None:
        from databridge.config import get_config
        from databridge.connectors.registry import ConnectorRegistry
        from databridge.schema.cache import SchemaCache
        from databridge.schema.scanner import SchemaScanner

        cfg = get_config()
        cfg.ensure_dirs()
        reg = ConnectorRegistry.from_uris(cfg.database_uris)
        await reg.connect_all()
        cache = SchemaCache(cfg.resolved_cache_path(), cfg.schema_cache_ttl_hours)
        scanner = SchemaScanner(reg, cache)

        if action == "scan":
            aliases = [database] if database else reg.aliases()
            for alias in aliases:
                tables = await scanner.scan(alias, force=force)
                console.print(f"[green]{alias}[/green]: {len(tables)} tables scanned")
                t = Table("Table", "Rows (approx)", "Columns")
                for tname, tm in tables.items():
                    t.add_row(tname, str(tm.row_count_approx), str(len(tm.columns)))
                console.print(t)

        elif action == "diff":
            aliases = [database] if database else reg.aliases()
            for alias in aliases:
                changed = await scanner.diff(alias)
                if changed:
                    console.print(f"[yellow]{alias}[/yellow] changed tables: {', '.join(changed)}")
                else:
                    console.print(f"[green]{alias}[/green]: no schema changes detected")

        elif action == "export":
            result = await scanner.scan_all()
            console.print(json.dumps(
                {
                    alias: {
                        tname: {"row_count": t.row_count_approx, "columns": list(t.columns.keys())}
                        for tname, t in tables.items()
                    }
                    for alias, tables in result.items()
                },
                indent=2,
            ))

        await reg.disconnect_all()

    asyncio.run(_run())


@app.command()
def joins(
    action: str = typer.Argument("list", help="list | discover"),
) -> None:
    """Manage cross-database join registry."""

    async def _run() -> None:
        from databridge.config import get_config
        from databridge.connectors.registry import ConnectorRegistry
        from databridge.schema.cache import SchemaCache
        from databridge.schema.joins.registry import JoinRegistry
        from databridge.schema.joins.discovery import JoinDiscovery
        from databridge.schema.scanner import SchemaScanner

        cfg = get_config()
        cfg.ensure_dirs()
        reg = ConnectorRegistry.from_uris(cfg.database_uris)
        await reg.connect_all()
        cache = SchemaCache(cfg.resolved_cache_path(), cfg.schema_cache_ttl_hours)
        join_reg = JoinRegistry(cfg.resolved_cache_path().parent / "joins.db")
        scanner = SchemaScanner(reg, cache)

        if action == "list":
            rules = await join_reg.get_all()
            t = Table("Join ID", "Source", "Target", "Transform", "Confidence", "Confirmed")
            for r in rules:
                t.add_row(
                    r.join_id[:30],
                    f"{r.db_a}.{r.table_a}.{r.column_a}",
                    f"{r.db_b}.{r.table_b}.{r.column_b}",
                    r.transform or "—",
                    f"{r.confidence:.2f}",
                    "✓" if r.confirmed else "○",
                )
            console.print(t)

        elif action == "discover":
            discovery = JoinDiscovery(
                reg,
                name_similarity_threshold=cfg.name_similarity_threshold,
                value_sample_size=cfg.value_sample_size,
                overlap_threshold=cfg.overlap_threshold,
                min_confidence=cfg.min_confidence_to_propose,
            )
            schema = await scanner.scan_all()
            candidates = await discovery.discover(schema)
            t = Table("Join ID", "Source", "Target", "Transform", "Confidence")
            for c in candidates:
                t.add_row(
                    c.join_id[:30],
                    f"{c.db_a}.{c.table_a}.{c.column_a}",
                    f"{c.db_b}.{c.table_b}.{c.column_b}",
                    c.transform or "—",
                    f"{c.confidence:.2f}",
                )
            console.print(t)
            console.print(f"[green]{len(candidates)} candidates found.[/green]")

        await reg.disconnect_all()

    asyncio.run(_run())


@app.command()
def benchmark(
    action: str = typer.Argument("run", help="run"),
    dab_root: str = typer.Option(..., "--dab-root", help="Path to cloned DataAgentBench directory"),
    results_dir: str = typer.Option("benchmark/results", "--results-dir"),
    provider: str = typer.Option("anthropic", "--provider", "-p", help="anthropic | openai | groq | kimi | together | ollama"),
    model: str = typer.Option("claude-opus-4-8", "--model", "-m", help="Model name"),
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d", help="Run one dataset only (e.g. bookreview)"),
    datasets: Optional[str] = typer.Option(None, "--datasets", help="Comma-separated list of datasets (e.g. agnews,yelp)"),
    official: bool = typer.Option(False, "--official", help="Run only the 12 official DataAgentBench datasets"),
    no_hints: bool = typer.Option(False, "--no-hints", help="Disable schema hints"),
    run: int = typer.Option(0, "--run", help="Run index 0–4 for 5-run leaderboard submission"),
    api_key: Optional[str] = typer.Option(None, "--api-key"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show tool call debug logs"),
) -> None:
    """Run the DataAgentBench evaluation suite with an LLM agent in the loop."""
    import logging
    import os
    from pathlib import Path
    from benchmark.dab import DABEvaluator, OFFICIAL_DATASETS, OFFICIAL_DATASET_ORDER

    # Always keep third-party libraries quiet; only elevate our own loggers.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger("benchmark").setLevel(level)
    logging.getLogger("databridge").setLevel(level)

    _env_vars = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "groq": "GROQ_API_KEY",
        "kimi": "MOONSHOT_API_KEY",
        "together": "TOGETHER_API_KEY",
    }
    if provider == "ollama":
        api_key = api_key or "ollama"  # Ollama runs locally — no real key needed
    else:
        if api_key is None:
            api_key = os.environ.get(_env_vars.get(provider, "OPENAI_API_KEY"))
        if not api_key:
            env_var = _env_vars.get(provider, "API_KEY")
            console.print(f"[red]No API key found. Set {env_var} or pass --api-key.[/red]")
            raise typer.Exit(1)

    # Resolve dataset filter: --official > --datasets > --dataset
    if official:
        datasets_list = list(OFFICIAL_DATASETS)
        dataset_single = None
        scope = "official 12 datasets"
    elif datasets:
        datasets_list = [d.strip() for d in datasets.split(",") if d.strip()]
        dataset_single = None
        scope = f"datasets: {', '.join(datasets_list)}"
    else:
        datasets_list = None
        dataset_single = dataset
        scope = dataset or "all datasets"

    evaluator = DABEvaluator(
        dab_root=Path(dab_root),
        results_dir=Path(results_dir),
        provider=provider,
        model=model,
        api_key=api_key,
        dataset=dataset_single,
        datasets=datasets_list,
        use_hints=not no_hints,
        run=run,
    )
    console.print(f"[green]Running DAB benchmark[/green] | {scope} | {provider}/{model}")

    async def _run() -> None:
        report = await evaluator.run()
        console.print(f"\n[bold]{report.summary}[/bold]")

        ds_table = Table("Dataset", "Pass@1", "Passed", "Total")
        _order = {name: i for i, name in enumerate(OFFICIAL_DATASET_ORDER)}
        for ds, rate in sorted(report.datasets.items(), key=lambda kv: (_order.get(kv[0], 999), kv[0])):
            ds_results = [r for r in report.results if r.dataset == ds]
            ds_pass = sum(1 for r in ds_results if r.passed)
            ds_table.add_row(ds, f"{rate:.1%}", str(ds_pass), str(len(ds_results)))
        console.print(ds_table)

        q_table = Table("Dataset", "Query", "Pass", "Answer (truncated)", "Error")
        for r in report.results:
            status = "[green]✓[/green]" if r.passed else "[red]✗[/red]"
            ans_short = (r.answer or "")[:50] + ("…" if len(r.answer or "") > 50 else "")
            err = (r.error or "")[:40]
            q_table.add_row(r.dataset, r.query_id, status, ans_short, err)
        console.print(q_table)
        console.print(f"Results saved to: {results_dir}/")

    asyncio.run(_run())


if __name__ == "__main__":
    app()
