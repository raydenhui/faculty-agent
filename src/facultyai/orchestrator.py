"""Async orchestrator that manages the end-to-end pipeline."""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .cache import CacheManager
from .config import AppConfig
from .database import Database, _job_id
from .exporter import export_to_excel
from .input_manager import sync_input_excel
from .llm_factory import get_llm
from .logging_config import get_logger
from .schema import Schema
from .scraper_graph import build_agent_graph

log = get_logger("orch")


async def run_pipeline(
    config: AppConfig,
    schema: Schema,
    db: Database,
    cache: CacheManager,
    retry_failed: bool = False,
) -> dict[str, Any]:
    """Run all pending jobs and export results. Returns run summary dict."""
    llm = get_llm(config.llm)
    console = Console()
    log.info("pipeline start  provider=%s model=%s", config.llm.provider, config.llm.model)

    console.print("[bold blue]Orchestrator[/] Loading input from Excel...")
    inserted, deleted = await sync_input_excel(db, config.files.input_excel)
    console.print(f"  Input sync: {inserted} rows kept, {deleted} removed.")

    uni_rows = await db.get_input_universities()
    console.print(f"  Loaded {len(uni_rows)} university entries from DB.")

    if not uni_rows:
        console.print("[yellow]No university entries to process.[/]")
        return {"total": 0, "successful": 0, "failed": 0}

    existing_jobs = await db.list_jobs()
    existing_job_ids = {j["job_id"] for j in existing_jobs}
    existing_job_map = {j["job_id"]: j for j in existing_jobs}

    discovery_jobs = 0
    scrape_jobs = 0

    async with AsyncSqliteSaver.from_conn_string(str(db.db_path)) as checkpointer:
        agent = build_agent_graph(config, schema, llm, cache, checkpointer=checkpointer)

        for r in uni_rows:
            uni = r["university"]
            dept = r.get("department")

            jid = _job_id(uni, dept)

            if jid in existing_job_ids:
                existing = existing_job_map[jid]
                if retry_failed and existing["status"] in ("failed", "running", "completed"):
                    await db.update_job_status(jid, "pending")
                    if existing["job_type"] == "discovery":
                        discovery_jobs += 1
                    else:
                        scrape_jobs += 1
                elif existing["status"] in ("completed", "running"):
                    if existing["job_type"] == "discovery":
                        discovery_jobs += 1
                    else:
                        scrape_jobs += 1
                continue

            if dept is None and config.department.discovery_enabled:
                await db.upsert_job(uni, None, job_type="discovery", status="pending")
                discovery_jobs += 1
            else:
                await db.upsert_job(uni, dept, job_type="scrape", status="pending")
                scrape_jobs += 1

        console.print(f"  Queued {discovery_jobs} discovery jobs, {scrape_jobs} scrape jobs.")

        run_id = await db.start_run()

        discovery_pending = await db.get_jobs_by_status("pending")
        discovery_pending = [j for j in discovery_pending if j["job_type"] == "discovery"]

        for job in discovery_pending:
            await db.update_job_status(job["job_id"], "running")
            try:
                state = {
                    "university": job["university"],
                    "department": None,
                    "need_discovery": True,
                }
                result = await agent.ainvoke(
                    state,
                    {"configurable": {"thread_id": job["job_id"]}},
                )

                departments = result.get("discovered_departments", [])
                console.print(
                    f"[cyan]Discovery[/] {job['university']}: found {len(departments)} departments"
                )

                existing_job_ids = {j["job_id"] for j in await db.list_jobs()}

                for d in departments:
                    d_jid = _job_id(job["university"], d)
                    if d_jid not in existing_job_ids:
                        await db.upsert_job(
                            job["university"], d, job_type="scrape", status="pending"
                        )
                        scrape_jobs += 1

                await db.update_job_status(job["job_id"], "completed")
            except Exception as e:
                console.print(f"[red]Discovery failed[/] {job['university']}: {e}")
                await db.update_job_status(job["job_id"], "failed", str(e))

        scrape_pending = await db.get_jobs_by_status("pending")
        scrape_pending = [j for j in scrape_pending if j["job_type"] == "scrape"]

        semaphore = asyncio.Semaphore(config.scraping.max_concurrent_jobs)
        successful = 0
        failed = 0

        if scrape_pending:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("[blue]Scraping...", total=len(scrape_pending))

                async def _process_job(job: dict[str, Any]) -> None:
                    nonlocal successful, failed
                    async with semaphore:
                        jid = job["job_id"]
                        await db.update_job_status(jid, "running")
                        try:
                            state = {
                                "university": job["university"],
                                "department": job.get("department"),
                                "need_discovery": False,
                            }
                            result = await agent.ainvoke(
                                state,
                                {"configurable": {"thread_id": jid}},
                            )

                            listing_url = result.get("listing_url")
                            if listing_url:
                                await db.upsert_job(
                                    job["university"],
                                    job.get("department"),
                                    listing_url=listing_url,
                                    status="running",
                                )

                            records = result.get("extracted_records", [])
                            log.info(
                                "job done  uni=%s dept=%s records=%d url=%s error=%s",
                                job["university"],
                                job.get("department"),
                                len(records),
                                result.get("listing_url", "?"),
                                result.get("error") or "-",
                            )

                            # Log graph-level errors
                            graph_error = result.get("error")
                            if graph_error:
                                progress.console.print(
                                    f"[yellow]Warning[/] {job['university']}/"
                                    f"{job.get('department', 'All')}: {graph_error}"
                                )

                            for rec in records:
                                unique_vals: dict[str, Any] = {}
                                for key in config.output.unique_keys:
                                    val = rec.get(key, "")
                                    if val:
                                        unique_vals[key] = val
                                await db.upsert_faculty(
                                    university=job["university"],
                                    department=job.get("department"),
                                    unique_vals=unique_vals,
                                    data=rec,
                                    profile_url=rec.get(
                                        "Profile URL", result.get("listing_url")
                                    ),
                                )

                            seen_ids = []
                            for rec in records:
                                uv: dict[str, Any] = {}
                                for key in config.output.unique_keys:
                                    val = rec.get(key, "")
                                    if val:
                                        uv[key] = val
                                rid = _build_record_id(
                                    job["university"], job.get("department"), uv
                                )
                                seen_ids.append(rid)

                            await db.mark_not_seen(
                                job["university"], job.get("department"), seen_ids
                            )

                            await db.update_job_status(jid, "completed")
                            successful += 1
                            progress.console.print(
                                f"[green]Scrape[/] {job['university']}/"
                                f"{job.get('department', 'All')}: {len(records)} faculty"
                            )
                        except Exception as e:
                            console.print(
                                f"[red]Scrape failed[/] {job['university']}/"
                                f"{job.get('department', 'All')}: {e}"
                            )
                            await db.update_job_status(jid, "failed", str(e))
                            failed += 1
                        finally:
                            progress.update(task, advance=1)

                tasks_list = [asyncio.create_task(_process_job(j)) for j in scrape_pending]
                await asyncio.gather(*tasks_list)

        archived = await db.archive_old_not_found(config.output.archive_after_not_found_runs)
        if archived:
            console.print(f"  Archived {len(archived)} stale records.")

        total = discovery_jobs + scrape_jobs
        await db.finish_run(run_id, total, successful, failed)

    console.print("[bold blue]Export[/] Generating Excel output...")
    row_count = await export_to_excel(db, schema, config.files.output_excel)
    console.print(f"  [green]{config.files.output_excel}[/] written ({row_count} rows).")

    return {"total": total, "successful": successful, "failed": failed}


def _build_record_id(university: str, department: str | None, unique_vals: dict[str, Any]) -> str:
    payload = json.dumps(
        {"university": university, "department": department or "", "keys": unique_vals},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:32]
