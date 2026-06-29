"""File-based locking for exclusive access to the database and output."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path


class LockManager:
    """Simple file-based advisory lock."""

    def __init__(self, lock_path: str | Path = "facultyai.lock") -> None:
        self._path = Path(lock_path)
        self._fd: int | None = None

    def acquire(self) -> bool:
        if self._fd is not None:
            return True
        try:
            fd = os.open(
                self._path,
                os.O_CREAT | os.O_EXCL | os.O_RDWR,
            )
            self._fd = fd
            return True
        except FileExistsError:
            return False

    def release(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
            with contextlib.suppress(FileNotFoundError):
                self._path.unlink()

    @property
    def locked(self) -> bool:
        return self._fd is not None
