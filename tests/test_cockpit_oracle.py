"""The cockpit is checked by something other than a person looking at it.

Three defects were found in this panel by eye, in a shipped build, in one
sitting -- the design card naming the planned model instead of the one that ran,
two design labels naming tests this program never performs, and `p = nan`
printed as a number. All three fuzzers read the exported HTML, and the cockpit
is not in the HTML, so none of them could have found any of it.

Two things are tested here, and they are different questions:

* the oracles CATCH the defect shapes -- each check is handed a panel carrying
  the defect and must report it. A check that never fails is decoration.
* the oracles FIRE on an ordinary run -- a check whose precondition is never met
  passes every seed and guards nothing, which is the failure this repository has
  paid for more than once.

The structural test at the bottom guards the seam that makes any of it possible:
the oracle reads the summary the widget is handed, so a renderer that builds its
own dict would leave the panel unchecked again while every test here still
passed.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from fuzzing.cockpit_oracles import (ORACLES, build_summary, check_cockpit,
                                     cockpit_target)

CONTEXT = {"inferred_test": "one_way_anova", "factor_columns": ["Group"],
           "subject_column": None, "group_labels": ["A", "B", "C"],
           "display_group_col": "Group"}

RESULT = {
    "test": "Welch's ANOVA",
    "p_value": 0.0031,
    "effect_size": 0.42,
    "effect_size_type": "eta_squared",
    "groups": ["A", "B", "C"],
    "selected_groups": ["A", "B", "C"],
    "posthoc_test": "Games-Howell Test",
    "n_total": 24,
    "raw_data": {"A": [1] * 8, "B": [2] * 8, "C": [3] * 8},
    "normality_tests": {"model_residuals": {"is_normal": True, "p_value": 0.71}},
    "variance_test": {"equal_variance": False, "p_value": 0.02},
}


def _clean():
    return build_summary(CONTEXT, RESULT)


def _check(summary, results=None, context=None):
    """Run every oracle over a panel, returning (violations, fired)."""
    violations, fired = [], []
    for name, oracle in ORACLES:
        if oracle(summary, context or CONTEXT, results or RESULT, violations):
            fired.append(name)
    return violations, fired


def test_an_ordinary_run_reports_nothing():
    violations, _ = _check(_clean())
    assert violations == [], violations


@pytest.mark.parametrize("name", [name for name, _ in ORACLES
                                  if name != "cockpit_agrees_with_report"])
def test_every_oracle_fires_on_an_ordinary_run(name):
    """A check that never runs is not a check.

    The report seam is excluded: it needs the exported HTML, which only a real
    fuzz run has.
    """
    _, fired = _check(_clean())
    assert name in fired, "oracle %s never fired on a complete result" % name


# --- the three defects that were found by eye ----------------------------------

def test_a_planned_model_beside_the_one_that_ran_is_caught():
    summary = _clean()
    summary["context_design"] = summary["context_design"].replace(
        "Model: Welch's ANOVA", "Model: One-Way ANOVA")
    violations, _ = _check(summary)
    assert any("One-Way ANOVA" in v and "Welch's ANOVA" in v for v in violations), violations


@pytest.mark.parametrize("printed", ["p = nan", "p = inf", "p = -inf"])
def test_a_number_that_is_not_one_is_caught(printed):
    summary = _clean()
    summary["inference_main_test"] = "Welch's ANOVA; " + printed
    violations, _ = _check(summary)
    assert any("as a value" in v for v in violations), violations


def test_an_effect_size_that_is_not_a_number_is_caught():
    summary = _clean()
    summary["inference_effect_size"] = "Eta-squared = nan"
    violations, _ = _check(summary)
    assert any("as a value" in v for v in violations), violations


# --- round trips ----------------------------------------------------------------

def test_a_p_value_that_does_not_match_the_result_is_caught():
    summary = _clean()
    summary["inference_main_test"] = "Welch's ANOVA; p = 0.4210"
    violations, _ = _check(summary)
    assert any("0.421" in v for v in violations), violations


def test_a_p_value_dropped_from_the_card_is_caught():
    summary = _clean()
    summary["inference_main_test"] = "Welch's ANOVA; p = N/A"
    violations, _ = _check(summary)
    assert any("states no p-value" in v for v in violations), violations


def test_the_card_naming_a_different_test_than_the_result_is_caught():
    summary = _clean()
    summary["inference_main_test"] = "Kruskal-Wallis Test; p = 0.0031"
    violations, _ = _check(summary)
    assert any("Kruskal-Wallis" in v for v in violations), violations


def test_an_effect_size_that_does_not_match_is_caught():
    summary = _clean()
    summary["inference_effect_size"] = "Eta-squared = 0.9900"
    violations, _ = _check(summary)
    assert any("0.99" in v for v in violations), violations


def test_an_effect_size_whose_kind_was_dropped_is_caught():
    summary = _clean()
    summary["inference_effect_size"] = "Effect size = 0.4200"
    violations, _ = _check(summary)
    assert any("names no kind" in v for v in violations), violations


def test_a_sample_size_that_is_not_the_sample_is_caught():
    summary = _clean()
    summary["context_sample_overview"] = summary["context_sample_overview"].replace(
        "Sample size (N): 24", "Sample size (N): 0")
    violations, _ = _check(summary)
    assert any("N" in v and "24" in v for v in violations), violations


def test_a_sample_size_the_data_does_not_have_is_caught():
    """The N is held against the data, not only against the field behind it."""
    results = dict(RESULT, n_total=99)
    summary = build_summary(CONTEXT, results)
    violations, _ = _check(summary, results)
    assert any("the analysed data holds 24 values" in v for v in violations), violations


def test_a_group_the_analysis_never_saw_is_caught():
    summary = _clean()
    summary["context_sample_overview"] = summary["context_sample_overview"].replace(
        "Groups: A, B, C", "Groups: A, B, D")
    violations, _ = _check(summary)
    assert any("'D'" in v for v in violations), violations


# --- claims that must be true, not merely present -------------------------------

def test_a_three_group_run_told_it_has_two_is_caught():
    summary = _clean()
    summary["context_analysis_scope"] = (
        "Covariates: None\n"
        "Post-hoc: No post-hoc applicable for t-tests (two groups only).")
    violations, _ = _check(summary)
    assert any("ran on 3" in v for v in violations), violations


def test_a_significant_result_called_not_significant_is_caught():
    summary = _clean()
    summary["context_analysis_scope"] = (
        "Covariates: None\n"
        "Post-hoc: No post-hoc required because the omnibus test was not significant.")
    violations, _ = _check(summary)
    assert any("not significant at p=" in v for v in violations), violations


def test_a_post_hoc_name_the_result_does_not_carry_is_caught():
    summary = _clean()
    summary["context_analysis_scope"] = "Covariates: None\nPost-hoc: Tukey HSD."
    violations, _ = _check(summary)
    assert any("Tukey HSD" in v for v in violations), violations


def test_normality_called_ok_while_every_test_failed_is_caught():
    results = dict(RESULT,
                   normality_tests={"model_residuals": {"is_normal": False}})
    summary = build_summary(CONTEXT, results)
    summary["metric_normality"] = "OK"
    violations, _ = _check(summary, results)
    assert any("every normality test failed" in v for v in violations), violations


def test_a_transformation_credited_but_never_applied_is_caught():
    summary = _clean()
    summary["metric_normality"] = "OK (after transformation)"
    violations, _ = _check(summary)
    assert any("credits a transformation" in v for v in violations), violations


def test_a_model_named_nowhere_in_the_report_is_caught():
    violations, _ = check_cockpit(CONTEXT, RESULT,
                                  report_text="<h1>Kruskal-Wallis Test</h1>")
    assert any("appears nowhere in the report" in v for v in violations), violations


def test_a_model_the_report_agrees_with_passes():
    violations, fired = check_cockpit(
        CONTEXT, RESULT, report_text="<h1>Welch's ANOVA</h1> ...")
    assert violations == [], violations
    assert "cockpit_agrees_with_report" in fired


# --- what the panel never shows -------------------------------------------------

@pytest.mark.parametrize("result", [
    {"blocked": True, "block_reason": "no usable values"},
    {"cancelled": True, "cancel_reason": "user backed out"},
    {"error": "boom"},
])
def test_a_state_the_panel_never_reaches_is_not_judged(result):
    """Blocked, cancelled and errored runs return before the cards are built."""
    assert cockpit_target(result, CONTEXT) is None


def test_a_multi_run_is_judged_on_the_lead_dataset_only():
    lead = dict(RESULT, test="Kruskal-Wallis Test")
    result = {"type": "multi_dataset_analysis",
              "results": {"DS1": lead, "DS2": dict(RESULT)},
              "successful_datasets": ["DS1", "DS2"]}
    target = cockpit_target(result, CONTEXT)
    assert target is not None
    assert target[1] is lead


def test_a_multi_run_whose_lead_was_cancelled_is_not_judged():
    """A cancelled dataset is still filed under ``results`` by the wrapper."""
    result = {"type": "multi_dataset_analysis",
              "results": {"DS1": {"cancelled": True}},
              "successful_datasets": ["DS1"]}
    assert cockpit_target(result, CONTEXT) is None


# --- the seam that makes the whole file possible --------------------------------

def test_the_renderer_reads_the_same_summary_the_oracle_does():
    """The window must not build its own panel.

    ``_build_result_summary`` exists so an oracle can see what the cockpit is
    about to claim. If the renderer assembles its own dict instead, every test
    above keeps passing while the panel the user sees goes unchecked -- and that
    is precisely how this surface got to a release unguarded.
    """
    import inspect

    from autopilot.statistical_analyzer_autopilot_pipeline import (
        _ap_build_result_summary, _ap_render_result_summary)

    source = inspect.getsource(_ap_render_result_summary)
    assert "self._build_result_summary(" in source, (
        "the cockpit renderer no longer goes through _build_result_summary")
    assert "_format_context_design" not in source, (
        "the renderer builds part of the summary itself again")

    built = inspect.getsource(_ap_build_result_summary)
    for key in ("metric_normality", "metric_variance", "inference_main_test",
                "inference_effect_size", "context_design",
                "context_sample_overview", "context_analysis_scope"):
        assert key in built, "the builder no longer produces %s" % key
