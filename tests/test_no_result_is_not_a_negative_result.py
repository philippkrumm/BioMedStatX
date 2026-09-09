"""A p-value that does not exist is not a p-value above alpha.

``isinstance(nan, float)`` is True and ``nan < 0.05`` is False, so the usual
``isinstance(p, (int, float)) and p < 0.05`` quietly files "the model produced
no answer" under "the model produced a negative answer".

Found by the fuzzer, seed 20: a Firth logistic fit on separated data with a
collinear covariate overflowed in ``np.exp``, returned ``p_value = nan`` and
``statistic = nan``, and was not marked blocked. The 4.6 MB report it wrote
carried a **Not significant** badge and the sentence "Logistic Regression did
not show evidence against the null hypothesis" -- a claim about the data, drawn
from a number that does not exist. The reader has no way to tell that apart from
a genuine null finding, which is the whole problem.
"""
from __future__ import annotations

import math

import pytest

# Root conftest.py puts src/ on sys.path and forces headless Qt.
from export.html_exporter import HTMLExporter


def _hero(p_value):
    return HTMLExporter._build_hero_context(
        {"test": "Logistic Regression", "p_value": p_value, "alpha": 0.05})


@pytest.mark.parametrize("p_value", [float("nan"), float("inf"), float("-inf"), None, "n/a"])
def test_a_missing_p_value_is_never_badged_not_significant(p_value):
    hero = _hero(p_value)
    assert hero["significance_label"] == "No result", (
        f"p_value {p_value!r} produced the badge {hero['significance_label']!r}"
    )
    assert hero["significance_class"] == "is-danger"
    assert hero["is_significant"] is False


@pytest.mark.parametrize("p_value", [float("nan"), float("inf"), None])
def test_a_missing_p_value_makes_no_claim_about_the_null_hypothesis(p_value):
    note = HTMLExporter._build_summary_note(
        {"p_value": p_value}, "Logistic Regression", p_value)
    assert "did not show evidence" not in note, note
    assert "detected evidence" not in note, note
    assert "without a numeric p-value" in note, note


def test_a_missing_p_value_says_why_when_the_engine_knows():
    """"No result" without a cause leaves the reader nothing to act on.

    The Firth fit that raised this recorded `converged = False` and a data-health
    warning naming quasi-perfect separation; the reader saw only "completed
    without a numeric p-value".
    """
    note = HTMLExporter._build_summary_note(
        {"p_value": float("nan"), "converged": False}, "Logistic Regression",
        float("nan"))
    assert "did not converge" in note, note
    assert "separation" in note, note
    assert "did not show evidence" not in note, note


def test_a_missing_p_value_invents_no_cause_when_none_was_recorded():
    """Only the engine's own verdict is repeated; nothing is guessed."""
    note = HTMLExporter._build_summary_note(
        {"p_value": None}, "Some test", None)
    assert note == "Some test completed without a numeric p-value.", note


def test_a_real_null_result_still_reads_as_not_significant():
    """The three states have to stay three: this is the one that must not move."""
    hero = _hero(0.42)
    assert hero["significance_label"] == "Not significant"
    assert hero["significance_class"] == "is-neutral"
    assert hero["is_significant"] is False
    assert "did not show evidence" in HTMLExporter._build_summary_note(
        {"p_value": 0.42}, "t-test", 0.42)


def test_a_significant_result_is_unchanged():
    hero = _hero(0.001)
    assert hero["significance_label"] == "Significant"
    assert hero["significance_class"] == "is-significant"
    assert hero["is_significant"] is True


@pytest.mark.parametrize("value,expected", [
    (0.04, True), (0.05, False), (0.06, False), (0.0, True),
    (float("nan"), False), (float("inf"), False), (None, False),
    (True, False),  # a bool is not a p-value, however much it looks like an int
])
def test_the_shared_gate_agrees_with_arithmetic(value, expected):
    assert HTMLExporter._significant_at(value) is expected


@pytest.mark.parametrize("value,expected", [
    (0.04, True), (0.99, True), (0.0, True),
    (float("nan"), False), (float("inf"), False), (float("-inf"), False),
    (None, False), ("0.04", False), (True, False),
])
def test_having_a_p_value_is_separate_from_being_significant(value, expected):
    assert HTMLExporter._has_p_value(value) is expected
    if expected:
        assert math.isfinite(float(value))


def test_the_badge_classes_exist_in_both_templates():
    """A label is only honest if the template can draw it.

    ``is-danger`` is the class the "No result" badge asks for; if a template
    stopped styling it the badge would render unstyled and read as ordinary
    text, which is the opposite of what it is for.
    """
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent / "src" / "templates"
    for name in ("report_single.html.j2", "report_multi.html.j2"):
        css = (root / name).read_text(encoding="utf-8")
        assert ".is-danger{" in css, f"{name} does not style .is-danger"
        assert ".hero .badge.is-danger{" in css, f"{name} has no hero variant"
