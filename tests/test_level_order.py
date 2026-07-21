"""natural_order: human-friendly, ROW-ORDER-INDEPENDENT level ordering."""
import random

import pytest

from core.level_order import natural_order


@pytest.mark.parametrize("given,expected", [
    (["24h", "48h", "6h"], ["6h", "24h", "48h"]),                       # numeric
    (["48h", "Baseline", "6h", "24h"], ["Baseline", "6h", "24h", "48h"]),  # mixed
    (["After", "Before"], ["Before", "After"]),                        # hierarchy
    (["KO", "WT"], ["WT", "KO"]),
    (["Trt", "ctrl"], ["ctrl", "Trt"]),
    (["OE", "EmptyVector"], ["EmptyVector", "OE"]),                    # vector control
    (["scrambled", "shRNA-A"], ["scrambled", "shRNA-A"]),             # scrambled control first
    (["0.5", "10", "2"], ["0.5", "2", "10"]),                         # decimals numeric
])
def test_expected_order(given, expected):
    assert natural_order(given) == expected


def test_order_is_independent_of_input_row_order():
    base = ["Baseline", "6h", "24h", "48h", "WT", "KO", "DrugA", "DrugX"]
    canonical = natural_order(base)
    rng = random.Random(0)
    for _ in range(30):
        shuffled = base[:]
        rng.shuffle(shuffled)
        assert natural_order(shuffled) == canonical, shuffled


def test_original_dtype_preserved():
    # ints stay ints (safe for DataFrame masking), ordered numerically
    assert natural_order([10, 2, 1]) == [1, 2, 10]
    assert all(isinstance(x, int) for x in natural_order([10, 2, 1]))


def test_unknown_levels_note_fires_once():
    notes = []
    out = natural_order(["DrugX", "DrugA"], notes=notes)
    assert out == ["DrugA", "DrugX"]                 # alphabetical fallback
    assert len(notes) == 1
    assert "alphabetically" in notes[0]
    assert "DrugA" in notes[0] and "DrugX" in notes[0]


def test_no_note_for_numeric_or_recognized_levels():
    for levels in (["6h", "24h", "48h"], ["Baseline", "6h", "24h"], ["WT", "KO"]):
        notes = []
        natural_order(levels, notes=notes)
        assert notes == [], levels


def test_single_unknown_level_no_note():
    notes = []
    natural_order(["DrugX", "6h"], notes=notes)  # only one non-numeric unknown
    assert notes == []


def test_report_assumption_summary_omits_ordering_note_but_logs_it(caplog):
    """(b): the alphabetical-fallback note is a debug-log diagnostic only — it is
    NOT surfaced in the user report (it fired on every composite interaction-cell
    label and was noise, not actionable). natural_order still logs the warning
    once per analysis for transparency."""
    import logging
    from export.report_summaries import _SummariesMixin

    with caplog.at_level(logging.WARNING, logger="core.level_order"):
        unknown = _SummariesMixin._build_assumption_summary({"groups": ["DrugX", "DrugA"]})
    assert "level_ordering_note" not in unknown          # not in the report payload
    assert any("alphabetically" in r.message for r in caplog.records)  # but logged

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="core.level_order"):
        known = _SummariesMixin._build_assumption_summary({"groups": ["6h", "24h", "48h"]})
    assert "level_ordering_note" not in known
    assert not any("alphabetically" in r.message for r in caplog.records)  # numeric -> no warning
