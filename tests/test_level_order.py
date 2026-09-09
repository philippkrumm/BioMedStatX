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


def test_a_lone_unknown_beside_a_numeric_level_is_still_a_guess():
    """Behaviour change: the old rule needed two unrecognized non-numeric levels.

    "6h" ahead of "DrugX" rests on nothing but the alphabet -- one is a
    duration, the other a compound, and no number separates them. The previous
    test asked whether each label was *entirely* numeric and counted unknowns,
    which stayed silent here while flagging "Week 4, Week 12", where a number
    had in fact done the ordering. The predicate now asks which chunk the sort
    keys first disagree on, so both cases come out the other way round.
    """
    notes = []
    natural_order(["DrugX", "6h"], notes=notes)
    assert len(notes) == 1
    assert "DrugX" in notes[0]

    numerically_separated = []
    natural_order(["Week 12", "Week 4"], notes=numerically_separated)
    assert numerically_separated == []


def test_the_ordering_note_reaches_the_report_only_when_the_order_is_guessed(caplog):
    """The note was muted rather than corrected; it is back, and it is precise.

    It used to fire on every composite interaction-cell label -- because the old
    predicate could not see the level behind a "factor=level" prefix -- so it
    was demoted to a debug log. With the corrected predicate it speaks only for
    a genuinely alphabetical order, which is worth saying: it is also why a plot
    will not connect subjects across those levels.

    It is deliberately NOT part of data_health_warnings, which renders as the
    red pre-analysis data-quality table. An alphabetical axis order is neither a
    data defect nor a danger.
    """
    from export.report_summaries import _SummariesMixin

    guessed = _SummariesMixin._build_assumption_summary({"groups": ["DrugX", "DrugA"]})
    assert "alphabetically" in guessed["level_order_note"]
    assert guessed["data_health_warnings"] == []

    numeric = _SummariesMixin._build_assumption_summary({"groups": ["6h", "24h", "48h"]})
    assert numeric["level_order_note"] == ""

    # The composite form that caused the muting in the first place.
    composite = _SummariesMixin._build_assumption_summary(
        {"groups": ["Timepoint=Pre", "Timepoint=Post"]})
    assert composite["level_order_note"] == ""
