"""Tests for lock manager."""

from __future__ import annotations

from pathlib import Path

from facultyai.lock_manager import LockManager


def test_acquire_and_release(tmp_path: Path) -> None:
    lock = LockManager(tmp_path / "test.lock")
    assert lock.acquire() is True
    assert lock.locked is True
    lock.release()
    assert lock.locked is False


def test_double_acquire_fails(tmp_path: Path) -> None:
    lock1 = LockManager(tmp_path / "test.lock")
    lock2 = LockManager(tmp_path / "test.lock")

    assert lock1.acquire() is True
    assert lock2.acquire() is False

    lock1.release()
    lock2.release()
