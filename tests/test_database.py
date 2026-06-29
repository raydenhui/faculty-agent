"""Tests for the async database layer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from facultyai.database import Database


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "test.db")
    await database.connect()
    yield database
    await database.close()


@pytest.mark.asyncio
async def test_upsert_and_get_input_universities(db: Database) -> None:
    await db.upsert_input_university("MIT", "EECS", "test info")
    await db.upsert_input_university("MIT", "Physics")

    rows = await db.get_input_universities()
    assert len(rows) == 2
    assert rows[0]["university"] == "MIT"
    assert rows[0]["extra_info"] == "test info"
    assert rows[1]["department"] == "Physics"
    assert rows[1]["extra_info"] is None


@pytest.mark.asyncio
async def test_upsert_updates_existing(db: Database) -> None:
    await db.upsert_input_university("MIT", "EECS", "old")
    await db.upsert_input_university("MIT", "EECS", "new")
    rows = await db.get_input_universities()
    assert len(rows) == 1
    assert rows[0]["extra_info"] == "new"


@pytest.mark.asyncio
async def test_delete_input_university(db: Database) -> None:
    await db.upsert_input_university("MIT", "EECS")
    await db.upsert_input_university("MIT", "Physics")
    await db.delete_input_university("MIT", "EECS")
    rows = await db.get_input_universities()
    assert len(rows) == 1
    assert rows[0]["department"] == "Physics"


@pytest.mark.asyncio
async def test_job_lifecycle(db: Database) -> None:
    jid = await db.upsert_job("MIT", "EECS")
    job = await db.get_job(jid)
    assert job is not None
    assert job["university"] == "MIT"
    assert job["status"] == "pending"

    await db.update_job_status(jid, "running")
    job = await db.get_job(jid)
    assert job["status"] == "running"

    await db.update_job_status(jid, "completed")
    job = await db.get_job(jid)
    assert job["status"] == "completed"
    assert job["completed_at"] is not None


@pytest.mark.asyncio
async def test_get_jobs_by_status(db: Database) -> None:
    await db.upsert_job("MIT", "EECS", status="pending")
    await db.upsert_job("MIT", "Physics", status="running")
    await db.upsert_job("Stanford", "CS", status="completed")

    pending = await db.get_jobs_by_status("pending")
    assert len(pending) == 1
    assert pending[0]["department"] == "EECS"


@pytest.mark.asyncio
async def test_faculty_upsert_and_query(db: Database) -> None:
    rid = await db.upsert_faculty(
        "MIT",
        "EECS",
        unique_vals={"Email": "jsmith@mit.edu"},
        data={"English Full Name": "John Smith", "Email": "jsmith@mit.edu"},
        profile_url="https://mit.edu/jsmith",
    )
    assert rid

    active = await db.get_active_faculty()
    assert len(active) == 1
    assert json.loads(active[0]["data_json"])["English Full Name"] == "John Smith"
    assert active[0]["status"] == "active"


@pytest.mark.asyncio
async def test_mark_not_seen_and_archive(db: Database) -> None:
    await db.upsert_faculty("MIT", "EECS", {"Email": "a@mit.edu"}, {"name": "A"})
    await db.upsert_faculty("MIT", "EECS", {"Email": "b@mit.edu"}, {"name": "B"})
    await db.upsert_faculty("MIT", "EECS", {"Email": "c@mit.edu"}, {"name": "C"})

    all_fac = await db.get_faculty_by_university("MIT", "EECS")
    record_ids = [r["record_id"] for r in all_fac]

    seen = record_ids[:1]
    await db.mark_not_seen("MIT", "EECS", seen)
    # 2 records now not_found with runs_not_found=1

    # Second run: same record seen again → other 2 get runs_not_found=2
    await db.mark_not_seen("MIT", "EECS", seen)
    archived = await db.archive_old_not_found(threshold=2)
    assert len(archived) == 2

    active = await db.get_active_faculty()
    assert len(active) == 1


@pytest.mark.asyncio
async def test_run_history(db: Database) -> None:
    rid = await db.start_run()
    assert rid > 0
    await db.finish_run(rid, 10, 8, 2)
    # No assertion needed; just verifying no errors


@pytest.mark.asyncio
async def test_schema_versions_logging(db: Database) -> None:
    vid = await db.log_schema_version([{"name": "Test", "type": "extracted"}])
    assert vid > 0


@pytest.mark.asyncio
async def test_context_manager() -> None:
    db_path = Path("test_context.db")
    try:
        async with Database(db_path) as database:
            await database.upsert_input_university("MIT")
            rows = await database.get_input_universities()
            assert len(rows) == 1
    finally:
        db_path.unlink(missing_ok=True)
