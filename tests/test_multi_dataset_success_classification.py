"""A successful dataset must not be counted as a failed one.

The multi loop decided whether a dataset had failed by asking whether its result
dict *contains* an ``error`` key -- not whether that key holds anything.
``StatisticalTester._standardize_results`` fills every result it returns with a
full set of standard keys, ``"error": None`` among them, so every design that
returns through it was classified as a failure the moment it succeeded.

Two consequences, and the second is the serious one:

* the combined overview reported that nothing had been analysed, while the
  per-dataset reports sat next to it perfectly intact; and
* the cross-dataset FDR correction is gated on having two or more results, so it
  silently never ran. A user comparing several outcome variables received
  uncorrected p-values with no indication that the correction had been skipped.

The clinical models (correlation, regression, ANCOVA, LMM, Firth) return
``model.as_results_dict()`` without passing through the standardizer, which is
why multi-dataset runs of *those* designs worked and hid the bug.

What is left here is the PRODUCER side -- that a real success really does carry
``"error": None`` -- plus the two guarantees the window's loop owns, read off
the loop itself. The consumer side moved with the code: the rule now lives in
``_ap_split_multi_results`` and is exercised in
``test_multi_report_failed_datasets.py``; the correction it gates is in
``test_multi_dataset_fdr.py``.
"""

def _standardized_success(name):
    """A success as the product actually produces it -- via the real
    standardizer, not a hand-written dict that could agree by construction."""
    from analysis.statisticaltester import StatisticalTester

    return StatisticalTester._standardize_results({
        "test": "One-way ANOVA",
        "p_value": 0.01 if name == "DS1" else 0.04,
        "statistic": 7.5,
        "effect_size": 0.42,
        "effect_size_type": "eta_squared",
        "dataset_name": name,
    })


def test_a_standardized_success_carries_a_null_error_key():
    """The producer side of the mismatch, pinned so it cannot drift silently."""
    result = _standardized_success("DS1")

    assert "error" in result
    assert result["error"] is None
    assert result["p_value"] == 0.01


# --- what the loop itself promises ---------------------------------------------
#
# These read the window's multi loop rather than driving it. The loop needs a
# QFileDialog, a live window and a mapped frame to run at all, and the two
# guarantees below were each broken once by a line being absent -- which is
# exactly what a source-level check can see and a mocked run of the old
# sheet-loop could not.

def _multi_loop_source():
    import inspect

    from autopilot.statistical_analyzer_autopilot_pipeline import _ap_determine_and_run_test

    source = inspect.getsource(_ap_determine_and_run_test)
    start = source.index('context["mode"] == "single"')
    return source[start:]


def test_a_cancelled_column_aborts_the_whole_batch():
    """Consent withdrawn for one column is not consent to analyse the rest."""
    source = _multi_loop_source()
    assert 'get("cancelled")' in source, source[-1200:]
    assert "_handle_cancelled_result" in source


def test_only_the_analysed_columns_are_exported():
    """An errored column must not enter the overview as an ordinary card."""
    source = _multi_loop_source()
    assert "_ap_split_multi_results(all_results)" in source
    exported = source.split("export_multi_dataset_results(")[1]
    assert exported.split(")")[0].replace("\n", " ").split(",")[0].strip() == "analysed", exported[:120]
    assert "failed" in exported.split(")")[0], exported[:200]
