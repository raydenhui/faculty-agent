"""Centralised logging configuration for FacultyAI.

Usage:
    from facultyai.logging_config import get_logger
    log = get_logger(__name__)
    log.debug("detail"), log.info("milestone"), log.warning("issue"), log.error("fail")
"""

from __future__ import annotations

import logging
import sys

# Module-level flag – set by CLI before pipeline runs.
_verbose: bool = False
_debug: bool = False

LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)-25s %(message)s"
LOG_FORMAT_VERBOSE = "%(asctime)s [%(levelname)-7s] %(name)-25s %(funcName)s:%(lineno)d  %(message)s"


def configure(verbose: bool = False, debug: bool = False) -> None:
    """Call once at startup to configure global logging."""
    global _verbose, _debug
    _verbose = verbose
    _debug = debug

    level = logging.DEBUG if debug else (logging.INFO if verbose else logging.WARNING)
    fmt = LOG_FORMAT_VERBOSE if debug else LOG_FORMAT

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))

    root = logging.getLogger("facultyai")
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a logger child of the facultyai namespace."""
    return logging.getLogger(f"facultyai.{name}")


def is_verbose() -> bool:
    return _verbose


def is_debug() -> bool:
    return _debug
