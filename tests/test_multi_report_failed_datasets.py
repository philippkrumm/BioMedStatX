"""A dataset whose analysis fails must not vanish from the combined report.

The run knows exactly which columns errored -- the multi loop holds each
result as it comes back. The reader, however, only ever receives the HTML
overview, and that overview was built from the successes alone. A dataset the user explicitly selected therefore
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


def test_the_multi_run_hands_its_failures_to_the_exporter(tmp_path):
    """The wiring, not the rendering: the seam where the failures were dropped.

    The window analyses one measurement column at a time and combines the
    results afterwards. That loop only ever asked whether a column had been
    CANCELLED, so a column whose analysis came back with an error went into the
    overview as an ordinary card with nothing in it -- and the exporter's
    failure map, which has existed since it was written, stayed empty.

    ``_ap_split_multi_results`` is the rule the loop uses, and the fuzzer uses
    the same function rather than a copy. Everything downstream of it here --
    the dispatcher, the exporter, the template -- is real.
    """
    from autopilot.statistical_analyzer_autopilot_pipeline import _ap_split_multi_results

    all_results = {"DS1": _result("DS1"), "DS2": {"error": "engine blew up"}}
    analysed, failed = _ap_split_multi_results(all_results)

    assert list(analysed) == ["DS1"], "an errored column was counted as analysed"
    assert failed == {"DS2": "engine blew up"}

    text = _render(tmp_path, analysed, failed)
    assert "DS2" in text, "the failed dataset never reached the report"
    assert "engine blew up" in text
    assert re.search(r"1\s+datasets? summarized", text), text


def test_a_column_with_no_error_is_not_called_a_failure():
    from autopilot.statistical_analyzer_autopilot_pipeline import _ap_split_multi_results

    # `_standardize_results` gives every result it returns "error": None, so a
    # membership test marks every success a failure -- the mistake this rule has
    # already been fixed for once.
    analysed, failed = _ap_split_multi_results(
        {"DS1": dict(_result("DS1"), error=None), "DS2": _result("DS2")})
    assert list(analysed) == ["DS1", "DS2"]
    assert failed == {}
