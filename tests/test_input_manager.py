"""Tests for input manager."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from facultyai.database import Database
from facultyai.input_manager import sync_input_excel


@pytest.fixture
def sample_excel(tmp_path: Path) -> str:
    path = tmp_path / "universities.xlsx"
    df = pd.DataFrame(
        {
            "university_name": ["MIT", "Stanford"],
            "department_name": ["EECS", "Computer Science"],
            "extra_info": ["test", None],
        }
    )
    df.to_excel(path, sheet_name="universities", index=False)
    return str(path)


@pytest.mark.asyncio
async def test_sync_creates_entries(sample_excel: str, tmp_path: Path) -> None:
    async with Database(tmp_path / "test.db") as db:
        inserted, deleted = await sync_input_excel(db, sample_excel)
        assert inserted == 2
        assert deleted == 0

        rows = await db.get_input_universities()
        assert len(rows) == 2
        assert rows[0]["university"] == "MIT"
        assert rows[0]["department"] == "EECS"
        assert rows[1]["university"] == "Stanford"


@pytest.mark.asyncio
async def test_sync_removes_missing(sample_excel: str, tmp_path: Path) -> None:
    path = Path(sample_excel)
    async with Database(tmp_path / "test.db") as db:
        await sync_input_excel(db, sample_excel)

        # Modify Excel to have only one entry
        df = pd.DataFrame(
            {
                "university_name": ["MIT"],
                "department_name": ["EECS"],
                "extra_info": [None],
            }
        )
        df.to_excel(path, sheet_name="universities", index=False)

        inserted, deleted = await sync_input_excel(db, sample_excel)
        assert inserted == 1
        assert deleted == 1

        rows = await db.get_input_universities()
        assert len(rows) == 1


@pytest.mark.asyncio
async def test_sync_missing_file(tmp_path: Path) -> None:
    async with Database(tmp_path / "test.db") as db:
        inserted, deleted = await sync_input_excel(db, tmp_path / "missing.xlsx")
        assert inserted == 0
        assert deleted == 0
