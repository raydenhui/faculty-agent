"""Shared pytest fixtures for FacultyAI tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def sample_schema_dict() -> dict:
    return {
        "columns": [
            {"name": "English Full Name", "type": "extracted", "hint": "Full name"},
            {
                "name": "Last Name",
                "type": "formula",
                "formula": '=TEXTAFTER([@[English Full Name]]," ")',
            },
            {"name": "Email", "type": "extracted", "hint": "Email address"},
            {"name": "Institution", "type": "static", "value_from": "university_name"},
        ]
    }


@pytest.fixture
def sample_schema_file(tmp_path: Path, sample_schema_dict: dict) -> Path:
    p = tmp_path / "schema.json"
    p.write_text(json.dumps(sample_schema_dict), encoding="utf-8")
    return p


@pytest.fixture
def sample_config_dict() -> dict:
    return {
        "version": 2,
        "llm": {
            "provider": "deepseek",
            "model": "deepseek-chat",
            "base_url": "https://api.deepseek.com/v1",
            "api_key": "${DEEPSEEK_API_KEY}",
        },
        "scraping": {"headless": False, "max_concurrent_jobs": 5},
    }


@pytest.fixture
def sample_config_file(tmp_path: Path, sample_config_dict: dict) -> Path:
    p = tmp_path / "config.yaml"
    import yaml

    p.write_text(yaml.safe_dump(sample_config_dict), encoding="utf-8")
    return p
