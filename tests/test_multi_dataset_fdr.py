"""Several measurement columns tested at once is a family, and it is corrected.

The window's "Multi-Dataset Analysis" analyses each mapped measurement column in
turn -- several genes, several markers -- and combines them into one overview.
That is a multiple-testing problem, and the combined report is built to say so:
each card can carry an FDR badge and the overview can carry a note naming the
family size.

Nothing filled either. The correction existed in exactly one place: inside the
sheet-loop reached by ``analyze(selected_datasets=...)``, which had no caller
anywhere in the program. Measured on three real columns before the fix, at
p = 0.00013 / 0.0012 / 0.014, the overview reported all three uncorrected and
mentioned no correction at all. The dead code's own trace text gave the intent
away -- it says "across N simultaneously tested DEPENDENT VARIABLES", which
describes the column loop it was never reachable from.

The last test here is the one that matters most: it holds the SEAM. The
correction being right is worth nothing if the loop does not call it, which is
precisely the state this feature shipped in.
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from analysis.analysis_core import AnalysisManager


def _results(*p_values):
    return {"DS%d" % (i + 1): {"p_value": p} for i, p in enumerate(p_values)}


def test_benjamini_hochberg_values_are_the_published_ones():
    """m = 3: p(1)*3/1, p(2)*3/2, p(3)*3/3, then enforced monotone."""
    results = _results(0.01, 0.04, 0.20)
    assert AnalysisManager.apply_across_dataset_fdr(results) == 3
    assert results["DS1"]["p_value_fdr"] == pytest.approx(0.03)
    assert results["DS2"]["p_value_fdr"] == pytest.approx(0.06)
    assert results["DS3"]["p_value_fdr"] == pytest.approx(0.20)


def test_an_adjusted_p_is_never_smaller_than_its_raw_one():
    results = _results(0.001, 0.002, 0.003, 0.5, 0.9)
    AnalysisManager.apply_across_dataset_fdr(results)
    for result in results.values():
        assert result["p_value_fdr"] >= result["p_value"] - 1e-12


def test_one_dataset_is_not_a_family():
    results = _results(0.01)
    assert AnalysisManager.apply_across_dataset_fdr(results) == 0
    assert "p_value_fdr" not in results["DS1"]


def test_one_usable_p_value_among_several_is_not_a_family():
    """Correcting a family of one would only ever return the raw value."""
    results = {"DS1": {"p_value": 0.01}, "DS2": {"p_value": None},
               "DS3": {"p_value": float("nan")}}
    assert AnalysisManager.apply_across_dataset_fdr(results) == 0
    assert "p_value_fdr" not in results["DS1"]


def test_a_dataset_without_a_usable_p_value_stays_out_of_the_family():
    results = {"DS1": {"p_value": 0.01}, "DS2": {"p_value": 0.04},
               "DS3": {"p_value": None}, "DS4": {"p_value": float("nan")}}
    assert AnalysisManager.apply_across_dataset_fdr(results) == 2
    assert results["DS1"]["p_value_fdr"] == pytest.approx(0.02)
    assert "p_value_fdr" not in results["DS3"]
    assert "p_value_fdr" not in results["DS4"]


def test_the_family_size_reaches_the_methodology_trace():
    """The report's methods section has to be able to state m."""
    results = _results(0.01, 0.04, 0.20)
    AnalysisManager.apply_across_dataset_fdr(results)
    trace = results["DS1"].get("methodology_trace")
    assert trace is not None, "no trace entry was written for the correction"
    assert "m = 3" in str(trace.__dict__) or "m = 3" in repr(trace), repr(trace)


def test_a_boolean_is_not_a_p_value():
    results = {"DS1": {"p_value": True}, "DS2": {"p_value": 0.04},
               "DS3": {"p_value": 0.20}}
    assert AnalysisManager.apply_across_dataset_fdr(results) == 2


# --- the seam this feature shipped without -------------------------------------

def test_the_multi_loop_actually_applies_the_correction():
    """A correct correction nothing calls is what was already there.

    For the whole life of this feature the implementation existed and the loop
    that needed it did not call it -- which no test of the correction itself can
    detect. This reads the loop.
    """
    from autopilot.statistical_analyzer_autopilot_pipeline import _ap_determine_and_run_test

    source = inspect.getsource(_ap_determine_and_run_test)
    assert "apply_across_dataset_fdr(" in source, (
        "the multi loop no longer corrects across the datasets it analyses")
    assert "_ap_split_multi_results(" in source, (
        "the multi loop no longer separates failed columns from analysed ones")

    # ... and what it corrects must be the successes, not everything: an errored
    # column has no p-value to adjust and would only inflate the family.
    corrected = source.split("apply_across_dataset_fdr(")[1].split(")")[0]
    assert corrected.strip() == "analysed", corrected
