"""Oracles for the exported HTML report -- the artefact the user actually gets.

The statistical oracles in :mod:`fuzzing.oracles` inspect the in-memory result
dict. That is one layer short of the product: every bug of the "engine fixed,
display not" family lives between the dict and the file, and none of them are
visible from the dict alone. Until now the report was written into a
``TemporaryDirectory`` and deleted before anything looked at it.

These checks read the written file. They are deliberately browser-free -- the
report carries its own machine-readable payloads (``pd-data-*`` script tags plus
the embedded Plotly figures), so parsing is enough to verify the invariants that
matter, at a cost that survives hundreds of seeds.

Each check returns violations *and* declares whether it fired at all. An oracle
whose precondition is never met across a whole run contributes nothing but the
appearance of coverage, so the orchestrator reports the firing counts alongside
the outcomes.
"""
from __future__ import annotations

import math
import re
from typing import Any, List, Tuple

# The report/export layer of these checks now lives in ``src`` -- see
# ``export/report_selfcheck``. It is imported rather than duplicated because the
# export path itself runs the same checks to write its sidecar, and two copies
# of a check drifting apart is a failure this repository has paid for more than
# once. What stays here is what only a fuzz run needs: the two oracles that
# depend on the fuzzer's own notion of a result, the multi-dataset overview, and
# the violation plumbing the orchestrator reports on.
from export.report_selfcheck import (
    GROUP_CHART_DIV,
    _is_number,
    _oracle_axis_order_is_ranked,
    _oracle_brackets_mode_has_no_letters,
    _oracle_letters_gate_is_complete,
    _oracle_letters_match_the_pairwise_table,
    _oracle_one_plot_font,
    _oracle_p_precision_capped,
    _oracle_payloads_parse,
    _oracle_result_number_is_rendered,
    _oracle_sections_present,
    _oracle_transform_display_is_earned,
    _plottable_groups,
    load_report,
)

# Re-exported deliberately: the tests and the visual worker read a report
# through this module, and moving the reader out from under them would be a
# rename dressed up as a refactor.
__all__ = ["ORACLES", "MULTI_ORACLES", "check_report", "check_multi_report",
           "check_report_without_result", "report_stats", "load_report",
           "GROUP_CHART_DIV"]


def _oracle_designer_present_when_plottable(report, result, violations) -> bool:
    """A result with drawable groups must reach the reader with a figure builder.

    The designer is switched on by the presence of plottable data, so a result
    carrying groups but a report carrying no payloads means the figure silently
    fell out somewhere between the two.
    """
    groups = _plottable_groups(result)
    if not groups:
        return False
    if not report.payloads:
        violations.append(
            f"result holds {groups} plottable group(s) but the report has no "
            "figure-builder payloads"
        )
    return True



def _oracle_paired_line_gate(report, result, violations) -> bool:
    """Lines only where a line is defensible, and a reason wherever it is not.

    The verdict is recomputed from the result the report was built from, so the
    payload cannot drift away from the rule while still looking plausible.
    """
    from analysis.paired_lines import paired_lines_supported

    payload = report.payloads.get("pd-data-paired-lines")
    if not isinstance(payload, dict):
        return False

    expected, expected_reason = paired_lines_supported(
        report.order, result.get("raw_data_subjects") or {})
    shown = bool(payload.get("supported"))
    if shown != expected:
        violations.append(
            f"paired-line gate says supported={shown} but the data says "
            f"{expected} ({expected_reason or 'supported'})"
        )
    if not shown:
        if not str(payload.get("reason") or "").strip():
            violations.append("paired lines refused without a reason")
        if payload.get("trajectories"):
            violations.append("paired lines refused but trajectories were emitted anyway")

    # A mixed design has no single ordered axis -- its cells are combinations,
    # so a line across them would assert a path that does not exist.
    if shown and str(result.get("design_type", "")).lower().startswith("mixed"):
        violations.append("paired lines drawn across the cells of a mixed design")
    return True



ORACLES = (
    ("payloads_parse", _oracle_payloads_parse),
    ("designer_when_plottable", _oracle_designer_present_when_plottable),
    ("sections_present", _oracle_sections_present),
    ("result_number_rendered", _oracle_result_number_is_rendered),
    ("p_precision_capped", _oracle_p_precision_capped),
    ("transform_display_earned", _oracle_transform_display_is_earned),
    ("letters_gate_complete", _oracle_letters_gate_is_complete),
    ("letters_match_pairs", _oracle_letters_match_the_pairwise_table),
    ("brackets_have_no_letters", _oracle_brackets_mode_has_no_letters),
    ("paired_line_gate", _oracle_paired_line_gate),
    ("axis_order_ranked", _oracle_axis_order_is_ranked),
    ("one_plot_font", _oracle_one_plot_font),
)


# Oracles that read only the file. A dataset whose analysis errored still leaves
# a report behind, but never reaches ``results``, so there is no result dict to
# compare it against -- skipping those files entirely would have silently
# dropped substantial reports from the run. The ones listed here need nothing
# but the report itself; the rest would recompute an expectation from an empty
# result and invent violations.
RESULT_FREE_ORACLES = ("payloads_parse", "sections_present", "letters_gate_complete",
                       "letters_match_pairs", "brackets_have_no_letters",
                       "axis_order_ranked", "one_plot_font",
                       "transform_display_earned")


def check_report_without_result(path: str) -> Tuple[List[str], List[str]]:
    """Check what a report asserts on its own, with no result to compare to."""
    subset = tuple((name, fn) for name, fn in ORACLES if name in RESULT_FREE_ORACLES)
    return _run(path, {}, subset, min_bytes=1000)


# --- the combined report of a multi-dataset run ---------------------------------
# A different artefact with a different template: no sections, no figure builder,
# no significance layer -- a lean overview of one card per dataset. It had no
# coverage at all, because the generator only ever built mode="single".


def _oracle_multi_lists_every_dataset(report, result, violations) -> bool:
    """Every analysed dataset must appear in the overview by name."""
    names = [str(n) for n in (result.get("successful_datasets") or [])]
    if not names:
        return False
    for name in names:
        if name not in report.rendered:
            violations.append(f"dataset {name!r} was analysed but is absent from the overview")
    return True


def _oracle_multi_count_matches(report, result, violations) -> bool:
    """The headline count must be the number of cards behind it."""
    names = result.get("successful_datasets") or []
    if not names:
        return False
    match = re.search(r"(\d+)\s+datasets? summarized", report.rendered)
    if not match:
        violations.append("the overview does not state how many datasets it summarizes")
        return True
    if int(match.group(1)) != len(names):
        violations.append(
            f"the overview says {match.group(1)} datasets but {len(names)} were analysed"
        )
    return True


def _oracle_multi_p_values_rendered(report, result, violations) -> bool:
    """Each card must carry its dataset's p-value, raw and FDR-adjusted."""
    from export.report_formatting import _FormattingMixin

    per_dataset = result.get("results") or {}
    if not isinstance(per_dataset, dict) or not per_dataset:
        return False

    fired = False
    for name, sub in per_dataset.items():
        if not isinstance(sub, dict):
            continue
        p = sub.get("p_value")
        if _is_number(p) and math.isfinite(p) and sub.get("blocked") is not True:
            fired = True
            shown = _FormattingMixin._format_p_value(p, sub.get("p_value_resolution"))
            if shown not in report.rendered:
                violations.append(
                    f"dataset {name!r} has p={p!r} but its rendering {shown!r} is "
                    "absent from the overview"
                )
        adjusted = sub.get("p_value_fdr")
        if _is_number(adjusted) and math.isfinite(adjusted):
            fired = True
            shown = _FormattingMixin._format_p_value(adjusted)
            if shown not in report.rendered:
                violations.append(
                    f"dataset {name!r} carries an FDR-adjusted p but {shown!r} is "
                    "absent from the overview"
                )
    return fired


def _oracle_multi_failures_surfaced(report, result, violations) -> bool:
    """A dataset that failed must not vanish from the report.

    The run knows which datasets errored -- it returns them in
    ``failed_datasets`` -- but the reader only ever receives the combined
    report. A dataset that silently disappears from it reads as one the user
    never selected.
    """
    failed = result.get("failed_datasets") or {}
    if not isinstance(failed, dict) or not failed:
        return False
    for name in failed:
        if str(name) not in report.rendered:
            violations.append(
                f"dataset {name!r} failed but is not mentioned in the overview at all"
            )
    return True


MULTI_ORACLES = (
    ("multi_lists_datasets", _oracle_multi_lists_every_dataset),
    ("multi_count_matches", _oracle_multi_count_matches),
    ("multi_p_values_rendered", _oracle_multi_p_values_rendered),
    ("multi_failures_surfaced", _oracle_multi_failures_surfaced),
)


def check_multi_report(path: str, result: Any) -> Tuple[List[str], List[str]]:
    """Check the combined overview of a multi-dataset run."""
    return _run(path, result, MULTI_ORACLES, min_bytes=2000)


def _run(path, result, oracles, min_bytes) -> Tuple[List[str], List[str]]:
    violations: List[str] = []
    if not isinstance(result, dict):
        return [f"result is not a dict: {type(result).__name__}"], []

    try:
        report = load_report(path)
    except Exception as exc:
        return [f"report at {path} could not be read: {type(exc).__name__}: {exc}"], []

    if len(report.text) < min_bytes:
        return [f"report at {path} is {len(report.text)} bytes -- effectively empty"], []

    fired: List[str] = []
    for name, oracle in oracles:
        try:
            if oracle(report, result, violations):
                fired.append(name)
        except Exception as exc:  # an oracle that throws is a finding about itself
            violations.append(f"oracle {name} raised {type(exc).__name__}: {exc}")
    return violations, fired


def check_report(path: str, result: Any) -> Tuple[List[str], List[str]]:
    """Check a single-analysis report. Returns ``(violations, oracles_that_fired)``."""
    return _run(path, result, ORACLES, min_bytes=1000)


def report_stats(path: str) -> dict:
    """Cheap shape of a written report: how much of it is actually filled in.

    A blocked run still writes a report -- honestly, with placeholders -- and
    almost every oracle then has nothing to say. Counting bytes and empty-state
    blocks separates "thin because it was rightly gated" from "thin because the
    checks did not apply", which the firing count alone cannot.
    """
    try:
        text = open(path, encoding="utf-8", errors="replace").read()
    except Exception:
        return {}
    return {"bytes": len(text),
            "empty_states": text.count("empty-state"),
            "figures": text.count("Plotly.newPlot(")}
