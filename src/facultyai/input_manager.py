"""Input manager: reads ``universities.xlsx`` and imports into SQLite.

Detects file changes via SHA-256 hash so re-imports only happen when needed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .database import Database


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def sync_input_excel(
    db: Database,
    excel_path: str | Path,
    sheet_name: str = "universities",
) -> tuple[int, int]:
    """Import rows from *excel_path* into ``input_universities``.

    Returns ``(inserted_or_updated, deleted)``.
    """
    path = Path(excel_path)
    if not path.exists():
        return 0, 0

    # Read Excel
    df = pd.read_excel(path, sheet_name=sheet_name)
    df = df.where(pd.notna(df), None)  # type: ignore[assignment]

    # Normalise column names
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    required = {"university_name"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required column(s): {required - set(df.columns)}")

    # Build current set from file
    file_rows: set[tuple[str, str | None]] = set()
    for _, row in df.iterrows():
        uni = str(row["university_name"]).strip()
        dept = (
            str(row["department_name"]).strip()
            if "department_name" in df.columns and row["department_name"]
            else None
        )
        extra = (
            str(row["extra_info"]).strip()
            if "extra_info" in df.columns and row["extra_info"]
            else None
        )
        file_rows.add((uni, dept))
        await db.upsert_input_university(uni, dept, extra or None)

    # Remove DB rows that no longer exist in file
    db_rows = await db.get_input_universities()
    deleted = 0
    for r in db_rows:
        key = (r["university"], r["department"])
        if key not in file_rows:
            await db.delete_input_university(r["university"], r["department"])
            deleted += 1

    return len(file_rows), deleted
