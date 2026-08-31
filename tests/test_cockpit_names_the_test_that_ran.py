"""The design card names the model that ran, not the one that was planned.

The cockpit read `context["inferred_test"]` -- the model chosen from the SHAPE
of the data, before the assumption checks look at the numbers. Where those
checks switch the analysis, the card kept announcing the plan: a real report on
three groups with unequal spread said

    Model: One-Way ANOVA

while the results section, the post-hoc and the methodology all said Welch's
ANOVA, which is what actually ran. Found by eye in the shipped 2.0 build, on the
first dataset anybody put through it.

The card directly below it already reads the RESULT for the post-hoc name. Two
lines of the same panel, one reporting what happened and one reporting what was
intended.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from autopilot.statistical_analyzer_autopilot_pipeline import _ap_format_context_design


class _Window:
    """Only what the formatter reaches for."""

    from autopilot.statistical_analyzer_autopilot_pipeline import (
        _ap_detected_test_label as _detected_test_label,
    )


def _design(inferred, results):
    context = {"inferred_test": inferred, "factor_columns": ["Group"],
               "subject_column": None}
    return _ap_format_context_design(_Window(), context, results)


@pytest.mark.parametrize("performed", [
    "Welch's ANOVA",            # the case from the shipped build
    "Kruskal-Wallis Test",      # normality switched it
    "Brunner-Langer ATS Test",
])
def test_the_card_names_what_ran(performed):
    text = _design("one_way_anova", {"test": performed})
    assert f"Model: {performed}" in text, text
    assert "One-way design" not in text, (
        "the planned model is still announced beside the one that ran: %r" % text)


def test_the_planned_model_is_used_when_nothing_ran():
    """A blocked or refused run has no performed test to name."""
    assert "Model: One-way design (independent groups)" in _design("one_way_anova", {"blocked": True})
    assert "Model: One-way design (independent groups)" in _design("one_way_anova", {})


@pytest.mark.parametrize("sentinel", ["Not performed", "", "   ", None])
def test_a_sentinel_is_not_a_model_name(sentinel):
    """`test` carries placeholders on paths where no analysis happened."""
    text = _design("one_way_anova", {"test": sentinel})
    assert "Model: One-way design (independent groups)" in text, text


def test_the_rest_of_the_card_is_unchanged():
    text = _design("one_way_anova", {"test": "Welch's ANOVA"})
    assert "Factors: Group" in text
    assert "Subject ID: None" in text


@pytest.mark.parametrize("key,label", [
    # The router picks Welch unconditionally for independent groups, so a
    # classic one-way ANOVA and a Student t-test are never what runs. A label
    # naming either announces an analysis this program does not perform.
    ("one_way_anova", "One-way design (independent groups)"),
    ("independent_ttest", "Two independent groups"),
])
def test_a_design_label_does_not_name_a_test_that_never_runs(key, label):
    from autopilot.statistical_analyzer_autopilot_pipeline import _ap_detected_test_label
    got = _ap_detected_test_label(_Window(), {"inferred_test": key})
    assert got == label
    assert "ANOVA" not in got and "t-test" not in got


@pytest.mark.parametrize("key", ["two_way_anova", "mixed_anova", "repeated_measures_anova"])
def test_designs_that_do_run_under_their_own_name_keep_it(key):
    """The rename is narrow on purpose -- these really are what runs."""
    from autopilot.statistical_analyzer_autopilot_pipeline import _ap_detected_test_label
    assert "ANOVA" in _ap_detected_test_label(_Window(), {"inferred_test": key})
