"""Caching layer for FacultyAI using diskcache.

Caches URL content and LLM extraction results to avoid redundant API calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import diskcache


class CacheManager:
    """Disk-based LRU cache with TTL support."""

    def __init__(self, cache_dir: str | Path = "./cache") -> None:
        self._cache = diskcache.Cache(Path(cache_dir))

    def get_url_content(self, url: str) -> str | None:
        key = _url_key(url)
        return self._cache.get(key, default=None)

    def set_url_content(self, url: str, content: str, ttl_sec: int = 604800) -> None:
        key = _url_key(url)
        self._cache.set(key, content, expire=ttl_sec)

    def get_extraction(self, input_hash: str) -> list[dict[str, Any]] | None:
        key = f"extract:{input_hash}"
        raw = self._cache.get(key, default=None)
        return json.loads(raw) if raw else None

    def set_extraction(
        self,
        input_hash: str,
        records: list[dict[str, Any]],
        ttl_sec: int = 2592000,
    ) -> None:
        key = f"extract:{input_hash}"
        self._cache.set(key, json.dumps(records), expire=ttl_sec)

    def close(self) -> None:
        self._cache.close()


def _url_key(url: str) -> str:
    import hashlib

    return f"url:{hashlib.sha256(url.encode()).hexdigest()[:40]}"
