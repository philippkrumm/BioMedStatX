"""Wave-4b BLOCKER 1 repair: locale-aware CSV reading via an explicit,
user-declared number format (no autodetect).

Wave-4b proved (class A) that pd.read_csv with pandas defaults silently
corrupts German-format CSV, and — critically — that a naive half-fix
(decimal="," WITHOUT thousands=".") turns 1.234,56 into NaN, i.e. silent TOTAL
loss, worse than the visibly-wrong number. So `thousands` is part of the
minimum contract, not a nice-to-have.

The parser core is a pure, Qt-free helper (core.csv_import); the import dialog
is UI plumbing on top of it. These tests pin the parser core on the three
demonstrated cases plus a positive control.
"""
import os
import sys
import io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from core.csv_import import read_csv_with_preset, read_csv_localized, CSV_FORMAT_PRESETS


# --- German-format sources a real "German Excel -> Save as CSV" produces ---
GERMAN_SMALL = "Group;Value\nA;1,5\nA;2,3\nB;4,1\nB;3,8\n"          # comma decimal
GERMAN_THOUSANDS = "Group;Value\nA;1.234,56\nA;2.500,00\nB;987,25\n"  # dot thousands + comma decimal
INTERNATIONAL = "Group,Value\nA,1.5\nA,2.3\nB,4.1\nB,3.8\n"          # already clean


def test_before_default_read_corrupts_german_csv():
    """Anchor: the current behaviour (pandas defaults) on German CSV is wrong."""
    df = pd.read_csv(io.StringIO(GERMAN_SMALL))            # what pipeline:967 does today
    # everything collapses into one column, values garbled
    assert list(df.columns) == ["Group;Value"]
    assert "Value" not in df.columns


def test_german_preset_parses_comma_decimal():
    df = read_csv_with_preset(io.StringIO(GERMAN_SMALL), "german")
    assert list(df.columns) == ["Group", "Value"]
    assert df["Value"].tolist() == [1.5, 2.3, 4.1, 3.8]
    assert df["Value"].dtype.kind == "f"


def test_german_preset_parses_thousands_separator():
    """The RLU-range case: 1.234,56 must become 1234.56, not NaN."""
    df = read_csv_with_preset(io.StringIO(GERMAN_THOUSANDS), "german")
    assert df["Value"].tolist() == [1234.56, 2500.0, 987.25]
    assert not df["Value"].isna().any()


def test_thousands_is_mandatory_naive_halffix_would_nan():
    """Guard the exact regression Wave-4b found: decimal="," WITHOUT thousands="."
    NaN-destroys 1.234,56. The german preset must NOT reproduce that."""
    naive = read_csv_localized(io.StringIO(GERMAN_THOUSANDS), sep=";", decimal=",", thousands=None)
    # read leaves them as unparseable object strings ("1.234,56"), which the
    # downstream to_numeric(errors="coerce") then turns into all-NaN — total loss.
    assert naive["Value"].dtype.kind == "O", "sanity: naive half-fix cannot parse these"
    assert pd.to_numeric(naive["Value"], errors="coerce").isna().all(), \
        "the naive half-fix NaN-destroys 1.234,56 downstream"
    # the preset includes thousands="." and therefore does not
    assert CSV_FORMAT_PRESETS["german"]["thousands"] == "."


def test_international_preset_is_unchanged_positive_control():
    """A clean English CSV (as the Wave-0-4 test data uses) must round-trip
    correctly under the International preset — no regression risk."""
    df = read_csv_with_preset(io.StringIO(INTERNATIONAL), "international")
    assert list(df.columns) == ["Group", "Value"]
    assert df["Value"].tolist() == [1.5, 2.3, 4.1, 3.8]


def test_preset_choice_matters_negative_control():
    """German data under the International preset stays broken — proving the
    user's choice is what fixes it, not a silent guess."""
    df = read_csv_with_preset(io.StringIO(GERMAN_SMALL), "international")
    assert "Value" not in df.columns or df.get("Value", pd.Series(dtype=float)).dtype.kind != "f"
