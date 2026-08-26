"""A successful dataset must not be counted as a failed one.

``_analyze_multiple_datasets`` decided whether a dataset had failed by asking
whether its result dict *contains* an ``error`` key -- not whether that key holds
anything. ``StatisticalTester._standardize_results`` fills every result it
returns with a full set of standard keys, ``"error": None`` among them, so every
design that returns through it was classified as a failure the moment it
succeeded.

Two consequences, and the second is the serious one:

* the combined overview reported that nothing had been analysed (and, before the
  failure channel existed, was not written at all), while the per-dataset
  reports sat next to it perfectly intact; and
* the cross-dataset FDR correction is gated on ``len(all_results) >= 2``, so it
  silently never ran. A user comparing several outcome variables received
  uncorrected p-values with no indication that the correction had been skipped.

The clinical models (correlation, regression, ANCOVA, LMM, Firth) return
``model.as_results_dict()`` without passing through the standardizer, which is
why multi-dataset runs of *those* designs worked and hid the bug.
"""

from pathlib import Path


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


def _run_two(tmp_path, monkeypatch):
    from analysis.analysis_core import AnalysisManager

    def fake_single(**kwargs):
        return _standardized_success(kwargs["dataset_name"])

    monkeypatch.setattr(AnalysisManager, "_analyze_single_dataset",
                        staticmethod(fake_single))

    return AnalysisManager._analyze_multiple_datasets(
        file_path="unused.xlsx", group_col="Group", groups=["A", "B"],
        selected_datasets=["DS1", "DS2"], value_cols=["Value"],
        combine_columns=False, width=8, height=6, dependent=False, compare=None,
        colors=None, hatches=None, title=None, x_label=None, y_label=None,
        file_name=str(tmp_path / "run"), save_plot=False, skip_plots=True,
        error_type="sd", additional_factors=None, show_individual_lines=False,
    )


def test_a_successful_dataset_is_not_counted_as_failed(tmp_path, monkeypatch):
    summary = _run_two(tmp_path, monkeypatch)

    assert summary["failed_datasets"] == {}
    assert summary["successful_datasets"] == ["DS1", "DS2"]
    assert summary["summary"]["successful"] == 2
    assert summary["summary"]["failed"] == 0


def test_the_cross_dataset_fdr_correction_actually_runs(tmp_path, monkeypatch):
    """The quiet half of the bug: no survivors means no correction."""
    summary = _run_two(tmp_path, monkeypatch)

    per_dataset = summary["results"]
    assert set(per_dataset) == {"DS1", "DS2"}
    for name, sub in per_dataset.items():
        assert sub.get("p_value_fdr") is not None, f"{name} was never FDR-adjusted"
    # Benjamini-Hochberg over {0.01, 0.04}: both scale to 0.02 and 0.04.
    assert per_dataset["DS1"]["p_value_fdr"] > per_dataset["DS1"]["p_value"]


def test_a_genuinely_failed_dataset_is_still_counted_as_failed(tmp_path, monkeypatch):
    """The fix must not swing the other way and swallow real failures."""
    from analysis.analysis_core import AnalysisManager

    def fake_single(**kwargs):
        name = kwargs["dataset_name"]
        if name == "DS2":
            return {"error": "engine blew up"}
        return _standardized_success(name)

    monkeypatch.setattr(AnalysisManager, "_analyze_single_dataset",
                        staticmethod(fake_single))

    summary = AnalysisManager._analyze_multiple_datasets(
        file_path="unused.xlsx", group_col="Group", groups=["A", "B"],
        selected_datasets=["DS1", "DS2"], value_cols=["Value"],
        combine_columns=False, width=8, height=6, dependent=False, compare=None,
        colors=None, hatches=None, title=None, x_label=None, y_label=None,
        file_name=str(tmp_path / "run"), save_plot=False, skip_plots=True,
        error_type="sd", additional_factors=None, show_individual_lines=False,
    )

    assert summary["successful_datasets"] == ["DS1"]
    assert summary["failed_datasets"] == {"DS2": "engine blew up"}


def test_the_overview_names_both_surviving_datasets(tmp_path, monkeypatch):
    """End of the chain: what the reader is handed."""
    summary = _run_two(tmp_path, monkeypatch)

    combined = Path(summary["combined_report"])
    report = combined.with_name(combined.stem + "_report.html")
    assert report.exists(), f"no combined report at {report}"

    text = report.read_text(encoding="utf-8")
    assert "DS1" in text and "DS2" in text
    assert "2 datasets summarized" in text
