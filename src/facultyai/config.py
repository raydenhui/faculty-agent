"""Configuration loading and validation for FacultyAI."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _resolve_env_vars(value: Any) -> Any:
    """Recursively replace ``${VAR}`` placeholders with environment variables."""
    if isinstance(value, str):
        return _ENV_VAR_PATTERN.sub(lambda m: os.getenv(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(v) for v in value]
    return value


class LLMConfig(BaseModel):
    provider: str = "openai_compatible"
    model: str = "gpt-4-turbo-preview"
    temperature: float = 0.2
    max_tokens: int = 4096
    base_url: str = ""
    api_key: str = ""
    azure_endpoint: str = ""
    api_version: str = ""


class SearchConfig(BaseModel):
    provider: str = "duckduckgo"
    bing_api_key: str = ""


class ScrapingConfig(BaseModel):
    headless: bool = True
    browser_timeout: int = 30
    max_concurrent_jobs: int = 3
    max_retries_per_step: int = 3
    request_delay_sec: float = 1.0
    use_scrapegraphai: bool = True


class FilesConfig(BaseModel):
    input_excel: str = "universities.xlsx"
    output_excel: str = "faculty_data.xlsx"
    schema_file: str = "schema.json"
    database: str = "facultyai.db"
    cache_dir: str = "./cache"


class OutputConfig(BaseModel):
    unique_keys: list[str] = Field(default_factory=lambda: ["Email", "English Full Name"])
    preserve_deprecated: bool = True
    archive_after_not_found_runs: int = 3


class DepartmentConfig(BaseModel):
    discovery_enabled: bool = True
    find_similar_department: bool = False
    similar_search_confidence: float = 0.7


class ChatConfig(BaseModel):
    allow_config_changes: bool = True


class AppConfig(BaseModel):
    version: int = 2
    llm: LLMConfig = Field(default_factory=LLMConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    scraping: ScrapingConfig = Field(default_factory=ScrapingConfig)
    files: FilesConfig = Field(default_factory=FilesConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    department: DepartmentConfig = Field(default_factory=DepartmentConfig)
    chat: ChatConfig = Field(default_factory=ChatConfig)

    @field_validator("version")
    @classmethod
    def _check_version(cls, v: int) -> int:
        if v != 2:
            raise ValueError(f"Unsupported config version: {v}. Expected 2.")
        return v


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    """Load configuration from a YAML file, resolving env-var placeholders."""
    path = Path(path)
    if not path.exists():
        return AppConfig()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return AppConfig.model_validate(_resolve_env_vars(raw))


def mask_secrets(config: AppConfig) -> dict:
    """Return a dict representation of the config with secret fields masked."""
    data = config.model_dump()
    if data["llm"].get("api_key"):
        data["llm"]["api_key"] = _mask(data["llm"]["api_key"])
    if data["search"].get("bing_api_key"):
        data["search"]["bing_api_key"] = _mask(data["search"]["bing_api_key"])
    return data


def _mask(value: str) -> str:
    if len(value) <= 4:
        return "****"
    return value[:2] + "*" * (len(value) - 4) + value[-2:]
