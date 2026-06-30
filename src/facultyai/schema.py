"""Schema definition handling for FacultyAI.

Parses ``schema.json`` and resolves column types (extracted, formula, static).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

ColumnType = Literal["extracted", "formula", "static"]


class ColumnDef(BaseModel):
    name: str
    type: ColumnType
    hint: str | None = None
    formula: str | None = None
    comment: str | None = None
    value: Any | None = None
    value_from: str | None = None

    def is_extracted(self) -> bool:
        return self.type == "extracted"

    def is_formula(self) -> bool:
        return self.type == "formula"

    def is_static(self) -> bool:
        return self.type == "static"


class Schema(BaseModel):
    columns: list[ColumnDef] = Field(default_factory=list)

    def extracted_columns(self) -> list[ColumnDef]:
        return [c for c in self.columns if c.is_extracted()]

    def formula_columns(self) -> list[ColumnDef]:
        return [c for c in self.columns if c.is_formula()]

    def static_columns(self) -> list[ColumnDef]:
        return [c for c in self.columns if c.is_static()]

    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def fingerprint(self) -> str:
        """A stable hash of the extracted columns for cache keys."""
        import hashlib

        payload = json.dumps(
            [{"name": c.name, "hint": c.hint} for c in self.extracted_columns()],
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()


def load_schema(path: str | Path = "schema.json") -> Schema:
    """Load a schema from a JSON file. Returns empty schema if missing."""
    path = Path(path)
    if not path.exists():
        return Schema()
    data = json.loads(path.read_text(encoding="utf-8"))
    return Schema.model_validate(data)


def build_extraction_prompt(schema: Schema) -> str:
    """Build the ScrapeGraphAI prompt from extracted columns and hints.

    Designed for DepthSearchGraph (depth=1): the LLM visits the listing page
    AND individual profile pages, so it can extract listing-level fields
    (name, title, position) AND detail-level fields (email, profile URL).
    """
    lines = []
    for c in schema.extracted_columns():
        hint_text = f"  — {c.hint}" if c.hint else ""
        lines.append(f'  "{c.name}"{hint_text}')
    fields = "\n".join(lines)
    names = ", ".join(f'"{c.name}"' for c in schema.extracted_columns())
    return (
        "You are crawling a university faculty directory. You will visit the main "
        "listing page AND follow links to individual professor profile pages. "
        "Extract the following fields for every faculty member from ALL pages "
        "you visit:\n\n"
        f"{fields}\n\n"
        "Important:\n"
        "- Basic info (name, title, position) is usually on the listing page.\n"
        "- Email, profile URL, and detailed bio are usually on the individual "
        "profile page — look for lines like 'Email: xxx@xxx.edu'.\n"
        "- Fill in every field you can find. Leave missing fields as empty strings.\n\n"
        "Return ONLY a JSON array of objects. Each object MUST use exactly these keys: "
        f"[{names}].\n"
        "Example: [{" + f'"{schema.extracted_columns()[0].name}": "Dr. John Smith"' + "}]"
    )
