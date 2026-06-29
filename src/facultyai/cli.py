"""Click-based CLI entry point for FacultyAI."""

from __future__ import annotations

import asyncio
import json
import sys

import click
from rich.console import Console
from rich.table import Table

from . import __version__
from .cache import CacheManager
from .config import load_config, mask_secrets
from .database import Database
from .lock_manager import LockManager
from .schema import load_schema

console = Console()


def _run_async(coro):
    return asyncio.run(coro)


@click.group()
@click.version_option(__version__, prog_name="facultyai")
def cli() -> None:
    """FacultyAI – AI-driven faculty information scraper."""


@cli.command()
@click.option("--config-path", default="config.yaml", help="Path to config file.")
@click.option("--retry-failed", is_flag=True, default=False, help="Retry previously failed jobs.")
def run(config_path: str, retry_failed: bool) -> None:
    """Start/run all pending jobs."""
    _run_with_lock(config_path, retry_failed=retry_failed)


@cli.command()
@click.option("--config-path", default="config.yaml", help="Path to config file.")
@click.option("--retry-failed", is_flag=True, default=False, help="Also retry previously failed jobs.")
def resume(config_path: str, retry_failed: bool) -> None:
    """Resume incomplete jobs (resets running jobs, optionally retries failed)."""
    cfg = load_config(config_path)
    lock = LockManager()

    if not lock.acquire():
        console.print("[red]Another facultyai process is already running.[/]")
        sys.exit(1)

    try:

        async def _run() -> None:
            db = Database(cfg.files.database)
            cache = CacheManager(cfg.files.cache_dir)
            schema = load_schema(cfg.files.schema_file)

            async with db:
                try:
                    running = await db.get_jobs_by_status("running")
                    for j in running:
                        await db.update_job_status(j["job_id"], "pending")

                    if retry_failed:
                        failed = await db.get_jobs_by_status("failed")
                        for j in failed:
                            await db.update_job_status(j["job_id"], "pending")

                    from .orchestrator import run_pipeline

                    summary = await run_pipeline(cfg, schema, db, cache, retry_failed=retry_failed)
                    if summary["failed"] > 0:
                        sys.exit(1)
                finally:
                    cache.close()

        _run_async(_run())
    finally:
        lock.release()


@cli.command()
@click.option("--config-path", default="config.yaml", help="Path to config file.")
@click.argument("university")
@click.argument("department", required=False)
def retry(config_path: str, university: str, department: str | None) -> None:
    """Retry a specific failed job by university (and optional department)."""
    cfg = load_config(config_path)
    lock = LockManager()

    if not lock.acquire():
        console.print("[red]Another facultyai process is already running.[/]")
        sys.exit(1)

    try:

        async def _run() -> None:
            db = Database(cfg.files.database)
            cache = CacheManager(cfg.files.cache_dir)
            schema = load_schema(cfg.files.schema_file)

            async with db:
                try:
                    from .database import _job_id

                    jid = _job_id(university, department)
                    job = await db.get_job(jid)
                    if not job:
                        console.print(
                            f"[red]No job found for {university}/{department or 'All'}[/]"
                        )
                        return

                    if job["status"] not in ("failed", "completed"):
                        console.print(
                            f"[yellow]Job is not in a retryable state (status: {job['status']}). "
                            f"Use 'facultyai run --retry-failed' instead.[/]"
                        )
                        return

                    await db.update_job_status(jid, "pending")

                    from .orchestrator import run_pipeline

                    summary = await run_pipeline(
                        cfg, schema, db, cache, retry_failed=True
                    )
                    if summary["failed"] > 0:
                        sys.exit(1)
                finally:
                    cache.close()

        _run_async(_run())
    finally:
        lock.release()


@cli.command()
@click.option("--config-path", default="config.yaml", help="Path to config file.")
def status(config_path: str) -> None:
    """Show job statuses and run history."""
    cfg = load_config(config_path)

    async def _run() -> None:
        db = Database(cfg.files.database)
        async with db:
            jobs = await db.list_jobs()
            if not jobs:
                console.print("[yellow]No jobs yet. Run 'facultyai run' first.[/]")
            else:
                table = Table(title="Job Status")
                table.add_column("University", style="cyan")
                table.add_column("Department")
                table.add_column("Type")
                table.add_column("Status")
                table.add_column("URL", style="dim")
                table.add_column("Error", style="red")

                status_styles = {
                    "completed": "[green]completed[/]",
                    "running": "[yellow]running[/]",
                    "failed": "[red]failed[/]",
                    "pending": "[dim]pending[/]",
                }

                for j in jobs:
                    dept = j["department"] or "All"
                    url = j["listing_url"] or ""
                    if len(url) > 40:
                        url = url[:37] + "..."
                    table.add_row(
                        j["university"],
                        dept,
                        j["job_type"],
                        status_styles.get(j["status"], j["status"]),
                        url,
                        (j["error"] or "")[:60],
                    )

                console.print(table)

                counts = {}
                for j in jobs:
                    s = j["status"]
                    counts[s] = counts.get(s, 0) + 1

                console.print(
                    f"\n  [green]{counts.get('completed', 0)} completed[/], "
                    f"[yellow]{counts.get('running', 0)} running[/], "
                    f"[dim]{counts.get('pending', 0)} pending[/], "
                    f"[red]{counts.get('failed', 0)} failed[/]"
                )

            # Run history
            assert db._connection is not None
            async with db._connection.execute(
                "SELECT * FROM run_history ORDER BY id DESC LIMIT 5"
            ) as cursor:
                history = [dict(r) for r in await cursor.fetchall()]

            if history:
                console.print()
                htable = Table(title="Run History (last 5)")
                htable.add_column("ID")
                htable.add_column("Started")
                htable.add_column("Finished")
                htable.add_column("Total")
                htable.add_column("Success")
                htable.add_column("Failed")

                for h in history:
                    started = (h["started_at"] or "")[:16]
                    finished = (h["finished_at"] or "")[:16]
                    htable.add_row(
                        str(h["id"]),
                        started,
                        finished,
                        str(h["total_jobs"] or "-"),
                        str(h["successful"] or "-"),
                        str(h["failed"] or "-"),
                    )
                console.print(htable)

    _run_async(_run())


@cli.command()
@click.option("--config-path", default="config.yaml", help="Path to config file.")
def export(config_path: str) -> None:
    """Regenerate Excel output from database."""
    cfg = load_config(config_path)

    async def _run() -> None:
        db = Database(cfg.files.database)
        schema = load_schema(cfg.files.schema_file)
        async with db:
            from .exporter import export_to_excel

            rows = await export_to_excel(db, schema, cfg.files.output_excel)
            console.print(f"[green]{cfg.files.output_excel}[/] written ({rows} rows).")

    _run_async(_run())


@cli.command()
@click.option("--config-path", default="config.yaml", help="Path to config file.")
def chat(config_path: str) -> None:
    """Interactive chat agent for configuration & queries."""
    cfg = load_config(config_path)

    async def _run() -> None:
        db = Database(cfg.files.database)
        schema = load_schema(cfg.files.schema_file)
        async with db:
            from .chat import run_chat

            await run_chat(cfg, schema, db)

    _run_async(_run())


@cli.group()
def config() -> None:
    """Configuration commands."""


@config.command("validate")
@click.option("--path", default="config.yaml", help="Path to config file.")
def config_validate(path: str) -> None:
    """Validate config.yaml."""
    try:
        cfg = load_config(path)
        console.print(f"[green]Config valid.[/] version={cfg.version} provider={cfg.llm.provider}")
    except Exception as e:
        console.print(f"[red]Config invalid:[/] {e}")
        sys.exit(1)


@config.command("show")
@click.option("--path", default="config.yaml", help="Path to config file.")
def config_show(path: str) -> None:
    """Show current config (with secrets masked)."""
    cfg = load_config(path)
    console.print_json(json.dumps(mask_secrets(cfg)))


def _run_with_lock(config_path: str, retry_failed: bool = False) -> None:
    cfg = load_config(config_path)
    lock = LockManager()

    if not lock.acquire():
        console.print("[red]Another facultyai process is already running.[/]")
        sys.exit(1)

    try:

        async def _run() -> None:
            db = Database(cfg.files.database)
            cache = CacheManager(cfg.files.cache_dir)
            schema = load_schema(cfg.files.schema_file)

            async with db:
                try:
                    from .orchestrator import run_pipeline

                    summary = await run_pipeline(
                        cfg, schema, db, cache, retry_failed=retry_failed
                    )
                    if summary["failed"] > 0:
                        sys.exit(1)
                finally:
                    cache.close()

        _run_async(_run())
    finally:
        lock.release()


if __name__ == "__main__":
    cli()
