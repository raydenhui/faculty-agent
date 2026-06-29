"""FacultyAI – AI-driven async CLI tool for scraping university faculty information."""

from __future__ import annotations

__version__ = "0.1.0"

from .config import AppConfig, load_config, mask_secrets
from .database import Database
from .schema import Schema, load_schema

__all__ = [
    "AppConfig",
    "Database",
    "Schema",
    "__version__",
    "load_config",
    "load_schema",
    "mask_secrets",
]
