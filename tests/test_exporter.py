"""Tests for Excel exporter."""

from __future__ import annotations

from pathlib import Path

import pytest

from facultyai.database import Database
from facultyai.exporter import _col_letter, _resolve_formula, export_to_excel
from facultyai.schema import ColumnDef, Schema


def test_col_letter() -> None:
    assert _col_letter(0) == "A"
    assert _col_letter(25) == "Z"
    assert _col_letter(26) == "AA"
    assert _col_letter(27) == "AB"
    assert _col_letter(51) == "AZ"
    assert _col_letter(52) == "BA"


def test_resolve_formula() -> None:
    col_index = {"English Full Name": 0, "Last Name": 1, "Email": 2}
    formula = '=TEXTAFTER([@[English Full Name]]," ")'
    result = _resolve_formula(formula, col_index)
    assert result == '=TEXTAFTER(A{row}," ")'

    formula2 = '=CONCAT([@[Last Name]],", ",[@[Email]])'
    result2 = _resolve_formula(formula2, col_index)
    assert result2 == '=CONCAT(B{row},", ",C{row})'


@pytest.mark.asyncio
async def test_export_empty_db(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    schema = Schema(
        columns=[
            ColumnDef(name="Name", type="extracted"),
            ColumnDef(name="Institution", type="static", value_from="university_name"),
        ]
    )

    async with Database(db_path) as db:
        count = await export_to_excel(db, schema, tmp_path / "output.xlsx")
        assert count == 0


@pytest.mark.asyncio
async def test_export_with_data(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    schema = Schema(
        columns=[
            ColumnDef(name="English Full Name", type="extracted"),
            ColumnDef(name="Email", type="extracted"),
            ColumnDef(name="Institution", type="static", value_from="university_name"),
            ColumnDef(
                name="Last Name",
                type="formula",
                formula='=TEXTAFTER([@[English Full Name]]," ")',
            ),
        ]
    )

    async with Database(db_path) as db:
        await db.upsert_faculty(
            "MIT",
            "EECS",
            {"Email": "jsmith@mit.edu"},
            {"English Full Name": "John Smith", "Email": "jsmith@mit.edu"},
        )
        await db.upsert_faculty(
            "MIT",
            "Physics",
            {"Email": "aeinstein@mit.edu"},
            {"English Full Name": "Albert Einstein", "Email": "aeinstein@mit.edu"},
        )

        output = tmp_path / "output.xlsx"
        count = await export_to_excel(db, schema, output)
        assert count == 2
        assert output.exists()
