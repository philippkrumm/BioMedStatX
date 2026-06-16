"""Central logging configuration for BioMedStatX.

Replaces scattered print() diagnostics. Modules use
``logger = logging.getLogger(__name__)`` and log at DEBUG/INFO/WARNING/ERROR;
this module decides where output goes.

Defaults (keep the console clean for the beta):
- console (stderr): WARNING and above
- rotating file ``biomedstatx.log``: DEBUG and above

Override via environment:
- ``BIOMEDSTATX_LOG_LEVEL``   console level (e.g. DEBUG to see everything)
- ``BIOMEDSTATX_LOG_FILE``    log file path ("" disables the file handler)
"""
import logging
import logging.handlers
import os

_CONFIGURED = False


def setup_logging(console_level=None, log_file=None, force=False):
    """Configure root logging once. Idempotent unless force=True."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    console_level = (console_level or os.environ.get("BIOMEDSTATX_LOG_LEVEL", "WARNING")).upper()
    if log_file is None:
        log_file = os.environ.get("BIOMEDSTATX_LOG_FILE", "biomedstatx.log")

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # handlers gate the actual output level
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    console = logging.StreamHandler()
    console.setLevel(getattr(logging, console_level, logging.WARNING))
    console.setFormatter(fmt)
    root.addHandler(console)

    if log_file:
        try:
            fh = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(fmt)
            root.addHandler(fh)
        except Exception as exc:  # file not writable -> console only, never crash
            root.warning("Could not open log file %s: %s", log_file, exc)

    _CONFIGURED = True
