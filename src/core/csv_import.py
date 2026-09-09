"""Locale-aware CSV reading with an explicitly declared number format.

The user declares the CSV number format at import time — there is NO autodetect
and no silent assumption. Wave-4b established (class A) why: a naive half-fix
that sets ``decimal=","`` but omits ``thousands="."`` turns a value like
``1.234,56`` into ``NaN`` — silent TOTAL data loss, worse than the visibly wrong
number the raw pandas default produces. So ``thousands`` is part of the minimum
contract here, not an optional extra, and every preset states it explicitly.

Excel import is unaffected: .xlsx stores numbers as floats, immune to display
locale (Wave-4b CHECK 1, positive control). This module is CSV-only.
"""
from __future__ import annotations

import pandas as pd

# preset name -> parsing contract. ``thousands=None`` means "no thousands
# separator". ``label`` is the human-readable dialog text.
CSV_FORMAT_PRESETS: dict[str, dict] = {
    "international": {
        "label": "International — comma separated, dot decimal, no thousands separator",
        "sep": ",",
        "decimal": ".",
        "thousands": None,
    },
    "german": {
        "label": "German — semicolon separated, comma decimal, dot thousands separator",
        "sep": ";",
        "decimal": ",",
        "thousands": ".",
    },
}

# Shown as the dialog default, but the user must actively confirm it — it is
# never applied automatically without a choice.
DEFAULT_CSV_FORMAT = "international"


def read_csv_localized(filepath_or_buffer, *, sep: str, decimal: str,
                       thousands: str | None, **kwargs):
    """Read a CSV with an explicitly chosen number format.

    ``thousands`` is required (may be ``None``). Passing a comma ``decimal``
    while omitting ``thousands`` silently NaN-destroys grouped values such as
    ``1.234,56`` — the exact failure Wave-4b documented — so this signature
    forces the caller to state it.
    """
    return pd.read_csv(
        filepath_or_buffer,
        sep=sep,
        decimal=decimal,
        thousands=thousands,
        **kwargs,
    )


def read_csv_with_preset(filepath_or_buffer, preset: str, **kwargs):
    """Read a CSV using one of ``CSV_FORMAT_PRESETS``."""
    try:
        spec = CSV_FORMAT_PRESETS[preset]
    except KeyError:
        raise ValueError(
            f"Unknown CSV format preset '{preset}'. "
            f"Choose from: {', '.join(CSV_FORMAT_PRESETS)}."
        )
    return read_csv_localized(
        filepath_or_buffer,
        sep=spec["sep"],
        decimal=spec["decimal"],
        thousands=spec["thousands"],
        **kwargs,
    )
