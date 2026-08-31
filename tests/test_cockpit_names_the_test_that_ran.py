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


# --- a number that is not a number must not be printed as one -----------------
#
# `p_value is None` was the only guard, and NaN is not None: the metric line
# rendered "Welch's ANOVA; p = nan" and the effect size "Eta-squared = nan".
# The report itself already grew a third badge state for exactly this ("No
# result"), because a p-value that does not exist is not a p-value above alpha.
# The cockpit kept printing it as a value.
#
# A NaN should be blocked upstream now -- but that guard was found sitting off a
# whole code path this morning, and the multi-dataset render site checks
# `blocked` on the lead result only. Two lines here cost nothing and do not
# depend on being right about which paths are covered.

_NAN = float("nan")


def _main_test(results):
    from autopilot.statistical_analyzer_autopilot_pipeline import _ap_format_main_test_metric
    return _ap_format_main_test_metric(_Window(), results)


def _effect(results):
    from autopilot.statistical_analyzer_autopilot_pipeline import _ap_format_effect_size_metric
    return _ap_format_effect_size_metric(_Window(), results)


@pytest.mark.parametrize("p", [_NAN, float("inf"), float("-inf")])
def test_a_p_value_that_is_not_a_number_is_not_printed_as_one(p):
    text = _main_test({"test": "Welch's ANOVA", "p_value": p})
    assert "nan" not in text.lower() and "inf" not in text.lower(), text
    assert "Welch's ANOVA" in text


@pytest.mark.parametrize("value", [_NAN, float("inf")])
def test_an_effect_size_that_is_not_a_number_is_not_printed_as_one(value):
    text = _effect({"effect_size": value, "effect_size_type": "eta_squared"})
    assert "nan" not in text.lower() and "inf" not in text.lower(), text


def test_real_numbers_are_still_printed():
    assert "0.0031" in _main_test({"test": "Welch's ANOVA", "p_value": 0.0031})
    assert "< 0.0001" in _main_test({"test": "Welch's ANOVA", "p_value": 1e-9})
    assert "0.4200" in _effect({"effect_size": 0.42, "effect_size_type": "eta_squared"})


def test_a_missing_p_value_still_reads_as_missing():
    assert "N/A" in _main_test({"test": "Welch's ANOVA", "p_value": None})
