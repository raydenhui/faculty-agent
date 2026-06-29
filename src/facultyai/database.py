"""Async SQLite database layer for FacultyAI.

Handles schema creation, migrations, and CRUD operations.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
-- User input (synced from Excel)
CREATE TABLE IF NOT EXISTS input_universities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    university TEXT NOT NULL,
    department TEXT,
    extra_info TEXT,
    added_at TEXT DEFAULT (datetime('now')),
    UNIQUE(university, department)
);

-- Job queue & progress
CREATE TABLE IF NOT EXISTS job (
    job_id TEXT PRIMARY KEY,
    university TEXT NOT NULL,
    department TEXT,
    job_type TEXT DEFAULT 'scrape',
    status TEXT DEFAULT 'pending',
    listing_url TEXT,
    error TEXT,
    last_attempted TEXT,
    completed_at TEXT
);

-- Core faculty records
CREATE TABLE IF NOT EXISTS faculty (
    record_id TEXT PRIMARY KEY,
    university TEXT NOT NULL,
    department TEXT,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    status TEXT DEFAULT 'active',
    runs_not_found INTEGER DEFAULT 0,
    data_json TEXT,
    profile_url TEXT
);

-- History of schema changes
CREATE TABLE IF NOT EXISTS schema_versions (
    version_id INTEGER PRIMARY KEY AUTOINCREMENT,
    applied_at TEXT DEFAULT (datetime('now')),
    columns_json TEXT
);

-- Archived records
CREATE TABLE IF NOT EXISTS faculty_archive (
    record_id TEXT PRIMARY KEY,
    university TEXT,
    department TEXT,
    first_seen TEXT,
    last_seen TEXT,
    data_json TEXT,
    archived_at TEXT DEFAULT (datetime('now'))
);

-- Run metadata
CREATE TABLE IF NOT EXISTS run_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT,
    finished_at TEXT,
    total_jobs INTEGER,
    successful INTEGER,
    failed INTEGER
);
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _job_id(university: str, department: str | None) -> str:
    payload = f"{university}::{department or ''}"
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def _record_id(university: str, department: str | None, unique_vals: dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "university": university,
            "department": department or "",
            "keys": unique_vals,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Database class
# ---------------------------------------------------------------------------


class Database:
    """Async SQLite facade for FacultyAI."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._connection: aiosqlite.Connection | None = None

    # -- lifecycle ----------------------------------------------------------

    async def connect(self) -> Database:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = await aiosqlite.connect(self.db_path)
        self._connection.row_factory = aiosqlite.Row
        await self._connection.executescript(_SCHEMA_SQL)
        await self._connection.commit()
        return self

    async def close(self) -> None:
        if self._connection:
            await self._connection.close()
            self._connection = None

    async def __aenter__(self) -> Database:
        return await self.connect()

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # -- input_universities -------------------------------------------------

    async def upsert_input_university(
        self,
        university: str,
        department: str | None = None,
        extra_info: str | None = None,
    ) -> None:
        assert self._connection is not None
        await self._connection.execute(
            """
            INSERT INTO input_universities(university, department, extra_info)
            VALUES (?, ?, ?)
            ON CONFLICT(university, department) DO UPDATE SET
                extra_info=excluded.extra_info
            """,
            (university, department, extra_info),
        )
        await self._connection.commit()

    async def get_input_universities(self) -> list[dict[str, Any]]:
        assert self._connection is not None
        async with self._connection.execute(
            "SELECT * FROM input_universities ORDER BY id"
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def delete_input_university(self, university: str, department: str | None) -> None:
        assert self._connection is not None
        if department is None:
            await self._connection.execute(
                "DELETE FROM input_universities WHERE university=? AND department IS NULL",
                (university,),
            )
        else:
            await self._connection.execute(
                "DELETE FROM input_universities WHERE university=? AND department=?",
                (university, department),
            )
        await self._connection.commit()

    # -- jobs ---------------------------------------------------------------

    async def upsert_job(
        self,
        university: str,
        department: str | None = None,
        job_type: str = "scrape",
        status: str = "pending",
        listing_url: str | None = None,
        error: str | None = None,
    ) -> str:
        assert self._connection is not None
        jid = _job_id(university, department)
        await self._connection.execute(
            """
            INSERT INTO job(job_id, university, department, job_type, status, listing_url, error)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                status=excluded.status,
                listing_url=COALESCE(excluded.listing_url, job.listing_url),
                error=excluded.error,
                last_attempted=CASE WHEN excluded.status='running' THEN ? ELSE job.last_attempted END
            """,
            (jid, university, department, job_type, status, listing_url, error, _now()),
        )
        await self._connection.commit()
        return jid

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        assert self._connection is not None
        async with self._connection.execute(
            "SELECT * FROM job WHERE job_id=?", (job_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def get_jobs_by_status(self, status: str) -> list[dict[str, Any]]:
        assert self._connection is not None
        async with self._connection.execute(
            "SELECT * FROM job WHERE status=? ORDER BY university, department", (status,)
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def update_job_status(self, job_id: str, status: str, error: str | None = None) -> None:
        assert self._connection is not None
        completed = _now() if status == "completed" else None
        await self._connection.execute(
            """
            UPDATE job SET status=?, error=?, last_attempted=?,
                completed_at=COALESCE(?, completed_at)
            WHERE job_id=?
            """,
            (status, error, _now(), completed, job_id),
        )
        await self._connection.commit()

    async def list_jobs(self) -> list[dict[str, Any]]:
        assert self._connection is not None
        async with self._connection.execute(
            "SELECT * FROM job ORDER BY university, department"
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    # -- faculty ------------------------------------------------------------

    async def upsert_faculty(
        self,
        university: str,
        department: str | None,
        unique_vals: dict[str, Any],
        data: dict[str, Any],
        profile_url: str | None = None,
    ) -> str:
        assert self._connection is not None
        rid = _record_id(university, department, unique_vals)
        now = _now()
        data_json = json.dumps(data, ensure_ascii=False)
        await self._connection.execute(
            """
            INSERT INTO faculty(
                record_id, university, department, first_seen, last_seen,
                status, runs_not_found, data_json, profile_url
            )
            VALUES (?, ?, ?, ?, ?, 'active', 0, ?, ?)
            ON CONFLICT(record_id) DO UPDATE SET
                last_seen=excluded.last_seen,
                status='active',
                runs_not_found=0,
                data_json=excluded.data_json,
                profile_url=COALESCE(excluded.profile_url, faculty.profile_url)
            """,
            (rid, university, department, now, now, data_json, profile_url),
        )
        await self._connection.commit()
        return rid

    async def get_active_faculty(self) -> list[dict[str, Any]]:
        assert self._connection is not None
        async with self._connection.execute(
            "SELECT * FROM faculty WHERE status='active' ORDER BY university, department"
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_faculty_by_university(
        self, university: str, department: str | None = None
    ) -> list[dict[str, Any]]:
        assert self._connection is not None
        if department is not None:
            async with self._connection.execute(
                "SELECT * FROM faculty WHERE university=? AND department=? ORDER BY record_id",
                (university, department),
            ) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
        async with self._connection.execute(
            "SELECT * FROM faculty WHERE university=? ORDER BY record_id", (university,)
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def mark_not_seen(
        self, university: str, department: str | None, seen_ids: list[str]
    ) -> None:
        """Mark faculty not in *seen_ids* as not_found and increment counter."""
        assert self._connection is not None
        placeholders = ",".join("?" * len(seen_ids)) if seen_ids else "''"
        dept_clause = "department IS NULL" if department is None else "department=?"
        sql = f"""
            UPDATE faculty
            SET status='not_found',
                runs_not_found = runs_not_found + 1,
                last_seen = ?
            WHERE university=? AND {dept_clause}
              AND record_id NOT IN ({placeholders})
        """
        params: list[Any] = [_now(), university]
        if department is not None:
            params.append(department)
        params.extend(seen_ids)
        await self._connection.execute(sql, params)
        await self._connection.commit()

    async def archive_old_not_found(self, threshold: int) -> list[str]:
        """Archive faculty with runs_not_found >= threshold. Returns archived record_ids."""
        assert self._connection is not None
        async with self._connection.execute(
            "SELECT * FROM faculty WHERE status='not_found' AND runs_not_found >= ?",
            (threshold,),
        ) as cursor:
            rows = await cursor.fetchall()

        archived: list[str] = []
        for row in rows:
            rid = row["record_id"]
            await self._connection.execute(
                """
                INSERT INTO faculty_archive(record_id, university, department, first_seen, last_seen, data_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(record_id) DO UPDATE SET
                    last_seen=excluded.last_seen,
                    data_json=excluded.data_json,
                    archived_at=?
                """,
                (
                    rid,
                    row["university"],
                    row["department"],
                    row["first_seen"],
                    row["last_seen"],
                    row["data_json"],
                    _now(),
                ),
            )
            await self._connection.execute("DELETE FROM faculty WHERE record_id=?", (rid,))
            archived.append(rid)

        await self._connection.commit()
        return archived

    # -- schema_versions ----------------------------------------------------

    async def log_schema_version(self, columns: list[dict[str, Any]]) -> int:
        assert self._connection is not None
        cursor = await self._connection.execute(
            "INSERT INTO schema_versions(columns_json) VALUES (?) RETURNING version_id",
            (json.dumps(columns, ensure_ascii=False),),
        )
        row = await cursor.fetchone()
        await self._connection.commit()
        return row[0] if row else -1

    # -- run_history --------------------------------------------------------

    async def start_run(self) -> int:
        assert self._connection is not None
        cursor = await self._connection.execute(
            "INSERT INTO run_history(started_at) VALUES (?) RETURNING id",
            (_now(),),
        )
        row = await cursor.fetchone()
        await self._connection.commit()
        return row[0] if row else -1

    async def finish_run(self, run_id: int, total_jobs: int, successful: int, failed: int) -> None:
        assert self._connection is not None
        await self._connection.execute(
            """
            UPDATE run_history
            SET finished_at=?, total_jobs=?, successful=?, failed=?
            WHERE id=?
            """,
            (_now(), total_jobs, successful, failed, run_id),
        )
        await self._connection.commit()
