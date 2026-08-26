"""A dataset whose analysis fails must not vanish from the combined report.

The run knows exactly which datasets errored -- ``_analyze_multiple_datasets``
collects them in ``failed_datasets`` and reports a success rate. The reader,
however, only ever receives the HTML overview, and that overview was built from
``all_results`` alone. A dataset the user explicitly selected therefore
disappeared without a trace, and the headline counted only the survivors: with
two datasets selected and one failing, the report said "1 datasets summarized"
and never mentioned the other.

The sibling export path gets this right --
``outlier_html_exporter.export_multi(all_results, failed_datasets, ...)``
renders "N failed" plus a section per failed dataset -- which is what these
tests hold the analysis path to.
"""

import re
from pathlib import Path


def _result(name, p_value=0.01):
    """A minimal result dict with the fields the overview card reads."""
    return {
        "test": "One-way ANOVA",
        "final_test_label": "One-way ANOVA",
        "p_value": p_value,
        "effect_size": 0.42,
        "effect_size_type": "eta_squared",
        "alpha": 0.05,
        "dataset_name": name,
        "descriptive": {},
    }


def _render(tmp_path, all_results, failed_datasets, stem="combined"):
    from export.export_dispatcher import ExportDispatcher

    out = ExportDispatcher.export_multi_dataset_results(
        all_results, str(tmp_path / stem), failed_datasets=failed_datasets
    )
    assert out["warning"] is None, out["warning"]
    assert out["html_path"] is not None
    return Path(out["html_path"]).read_text(encoding="utf-8")


def test_a_failed_dataset_is_named_in_the_overview(tmp_path):
    text = _render(tmp_path, {"DS1": _result("DS1")}, {"DS2": "engine blew up"})

    assert "DS1" in text
    assert "DS2" in text, "the failed dataset vanished from the report entirely"
    assert "engine blew up" in text, "the reader is not told why it failed"


def test_the_headline_still_counts_only_the_summarized_datasets(tmp_path):
    """"N datasets summarized" must keep meaning the cards behind it.

    The failure count is additional information, not a correction of that
    number -- the same split the outlier overview makes.
    """
    text = _render(tmp_path, {"DS1": _result("DS1")}, {"DS2": "engine blew up"})

    match = re.search(r"(\d+)\s+datasets? summarized", text)
    assert match, "the overview no longer states how many datasets it summarizes"
    assert match.group(1) == "1"
    assert re.search(r"1\s+failed", text), "the failure count is not stated"


def test_no_failure_section_when_every_dataset_succeeded(tmp_path):
    text = _render(tmp_path, {"DS1": _result("DS1"), "DS2": _result("DS2", 0.2)}, {})

    match = re.search(r"(\d+)\s+datasets? summarized", text)
    assert match and match.group(1) == "2"
    assert "could not be analysed" not in text, "a clean run must not mention failures"


def test_a_report_is_still_written_when_every_dataset_failed(tmp_path):
    """The most extreme form of the same bug: the user gets nothing at all."""
    text = _render(tmp_path, {}, {"DS1": "bad input", "DS2": "engine blew up"})

    assert "DS1" in text and "DS2" in text
    assert "bad input" in text and "engine blew up" in text
    assert re.search(r"2\s+failed", text)


def test_the_analysis_run_hands_its_failures_to_the_exporter(tmp_path, monkeypatch):
    """The wiring, not the rendering: the seam where the failures were dropped.

    ``_analyze_multiple_datasets`` knew about the failures all along and simply
    never passed them on. Only the per-dataset analysis is stubbed here; the
    loop, the export call, the dispatcher, the exporter and the template are the
    real ones.
    """
    from analysis.analysis_core import AnalysisManager

    def fake_single(**kwargs):
        name = kwargs["dataset_name"]
        if name == "DS2":
            return {"error": "engine blew up"}
        return _result(name)

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

    assert summary["failed_datasets"] == {"DS2": "engine blew up"}
    combined = Path(summary["combined_report"])
    report = combined.with_name(combined.stem + "_report.html")
    assert report.exists(), f"no combined report at {report}"

    text = report.read_text(encoding="utf-8")
    assert "DS2" in text, "the failed dataset never reached the report"
    assert "engine blew up" in text
