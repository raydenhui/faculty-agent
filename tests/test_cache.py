"""Tests for caching layer."""

from __future__ import annotations

from pathlib import Path

from facultyai.cache import CacheManager


def test_url_cache_roundtrip(tmp_path: Path) -> None:
    cm = CacheManager(tmp_path / "cache")
    url = "https://example.com/faculty"
    cm.set_url_content(url, "<html>test</html>")

    cached = cm.get_url_content(url)
    assert cached == "<html>test</html>"

    cm.close()


def test_url_cache_miss(tmp_path: Path) -> None:
    cm = CacheManager(tmp_path / "cache")
    assert cm.get_url_content("https://nonexistent.com") is None
    cm.close()


def test_extraction_cache_roundtrip(tmp_path: Path) -> None:
    cm = CacheManager(tmp_path / "cache")
    records = [{"name": "John", "email": "j@mit.edu"}]
    cm.set_extraction("hash123", records)

    cached = cm.get_extraction("hash123")
    assert cached == records

    cm.close()


def test_extraction_cache_miss(tmp_path: Path) -> None:
    cm = CacheManager(tmp_path / "cache")
    assert cm.get_extraction("nonexistent") is None
    cm.close()
