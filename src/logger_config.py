"""Central logging configuration for BioMedStatX.

Single entry point: ``configure_logging()`` — idempotent, safe to call from any
module. Subsequent imports use ``logging.getLogger(__name__)`` and inherit the
handler chain configured here.

Log targets:
    * Console (stderr) at INFO level — visible when launched from a terminal.
    * Rotating file handler at DEBUG level — survives GUI sessions where stdout
      is invisible. Location:
        Windows : %LOCALAPPDATA%\\BioMedStatX\\logs\\biomedstatx.log
        macOS   : ~/Library/Logs/BioMedStatX/biomedstatx.log
        Linux   : ~/.local/state/biomedstatx/biomedstatx.log

Rotation: 5 files × 2 MB each.
"""
from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIGURED = False
_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s :: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _log_dir() -> Path:
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
        return Path(base) / "BioMedStatX" / "logs"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / "BioMedStatX"
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "biomedstatx"


def configure_logging(level: int = logging.INFO) -> Path | None:
    """Initialize root logger. Returns log file path, or None if file handler failed.

    Idempotent: repeated calls are no-ops.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return getattr(configure_logging, "_log_path", None)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Console handler
    console = logging.StreamHandler(stream=sys.stderr)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
    root.addHandler(console)

    log_path: Path | None = None
    try:
        log_dir = _log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "biomedstatx.log"
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=2 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        root.addHandler(file_handler)
    except OSError as exc:
        # Filesystem failure (permissions, read-only volume) — keep console only.
        root.warning("logger_config: file handler disabled: %s", exc)
        log_path = None

    _CONFIGURED = True
    configure_logging._log_path = log_path  # type: ignore[attr-defined]
    return log_path


def get_logger(name: str) -> logging.Logger:
    """Convenience wrapper. Ensures ``configure_logging`` ran at least once."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
