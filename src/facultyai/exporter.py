"""Excel exporter: writes ``faculty_data.xlsx`` from the database."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .database import Database
from .schema import ColumnDef, Schema


def _col_letter(index: int) -> str:
    """Convert 0-based column index to Excel column letter(s)."""
    result = ""
    while index >= 0:
        result = chr(index % 26 + 65) + result
        index = index // 26 - 1
    return result


def _resolve_formula(formula: str, column_index: dict[str, int]) -> str:
    """Replace ``[@[Column Name]]`` with cell-relative references.

    Since we write formulas per-row in openpyxl we convert
    ``[@[English Full Name]]`` → ``A{row}``.
    The row placeholder is kept as ``{row}`` for later substitution.
    """

    def _replace(m: re.Match[str]) -> str:
        col_name = m.group(1)
        idx = column_index.get(col_name)
        if idx is None:
            return f"[@{col_name}]"
        return _col_letter(idx) + "{row}"

    return re.sub(r"\[@\[([^\]]+)\]\]", _replace, formula)


async def export_to_excel(
    db: Database,
    schema: Schema,
    output_path: str | Path,
) -> int:
    """Regenerate *output_path* from current DB contents. Returns row count."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Gather column definitions
    col_map: dict[str, int] = {}
    headers: list[str] = []
    formula_columns: list[tuple[int, ColumnDef]] = []
    static_columns: list[tuple[int, ColumnDef]] = []

    for col in schema.columns:
        idx = len(headers)
        col_map[col.name] = idx
        headers.append(col.name)
        if col.is_formula():
            formula_columns.append((idx, col))
        elif col.is_static():
            static_columns.append((idx, col))

    # 2. Query active faculty
    rows = await db.get_active_faculty()
    if not rows:
        pd.DataFrame(columns=headers).to_excel(path, index=False, engine="openpyxl")
        return 0

    # 3. Build DataFrame rows
    data_rows: list[dict[str, Any]] = []
    for r in rows:
        parsed = json.loads(r["data_json"] or "{}")
        parsed["university_name"] = r["university"]
        parsed["department"] = r["department"]
        data_rows.append(parsed)

    # 4. Build records
    records: list[list[Any]] = []
    for i, d in enumerate(data_rows):
        row_num = i + 2  # Excel rows are 1-indexed, header=1
        record: list[Any] = [None] * len(headers)
        # Extracted values
        for col in schema.extracted_columns():
            idx = col_map[col.name]
            record[idx] = d.get(col.name, "")
        # Static values
        for idx, col in static_columns:
            if col.value_from:
                record[idx] = d.get(col.value_from, "")
            else:
                record[idx] = col.value or ""
        # Formula columns – store as strings for openpyxl
        for idx, col in formula_columns:
            raw = _resolve_formula(col.formula or "", col_map)
            record[idx] = raw.replace("{row}", str(row_num))
        records.append(record)

    # 5. Write via pandas + openpyxl
    df = pd.DataFrame(records, columns=headers)
    df = df.astype(str)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Faculty")

    return len(records)
