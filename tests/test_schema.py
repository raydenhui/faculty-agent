"""Tests for schema loading and prompt building."""

from __future__ import annotations

from pathlib import Path

from facultyai.schema import Schema, build_extraction_prompt, load_schema


class TestSchemaLoading:
    def test_load_from_file(self, sample_schema_file: Path) -> None:
        schema = load_schema(sample_schema_file)
        assert len(schema.columns) == 4
        assert schema.columns[0].name == "English Full Name"
        assert schema.columns[0].is_extracted()

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        schema = load_schema(tmp_path / "missing.json")
        assert len(schema.columns) == 0

    def test_column_type_helpers(self, sample_schema_file: Path) -> None:
        schema = load_schema(sample_schema_file)
        assert len(schema.extracted_columns()) == 2
        assert len(schema.formula_columns()) == 1
        assert len(schema.static_columns()) == 1

    def test_column_names(self, sample_schema_file: Path) -> None:
        schema = load_schema(sample_schema_file)
        assert schema.column_names() == [
            "English Full Name",
            "Last Name",
            "Email",
            "Institution",
        ]


class TestFingerprint:
    def test_fingerprint_stable(self, sample_schema_file: Path) -> None:
        s1 = load_schema(sample_schema_file)
        s2 = load_schema(sample_schema_file)
        assert s1.fingerprint() == s2.fingerprint()

    def test_fingerprint_changes_with_hint(self, sample_schema_dict: dict) -> None:
        s1 = Schema.model_validate(sample_schema_dict)
        modified = sample_schema_dict.copy()
        modified["columns"] = [
            {**sample_schema_dict["columns"][0], "hint": "Different hint"},
            *sample_schema_dict["columns"][1:],
        ]
        s2 = Schema.model_validate(modified)
        assert s1.fingerprint() != s2.fingerprint()


class TestExtractionPrompt:
    def test_prompt_contains_column_names(self, sample_schema_file: Path) -> None:
        schema = load_schema(sample_schema_file)
        prompt = build_extraction_prompt(schema)
        assert "English Full Name" in prompt
        assert "Email" in prompt
        assert "Institution" not in prompt  # static, not extracted
        assert "JSON" in prompt
