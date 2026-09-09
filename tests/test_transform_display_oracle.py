"""The report may not show transformed values for a run that transformed nothing.

Built from a real defect: a "Transformed value" column standing in a report for
an analysis where no transformation was applied. The unit tests that froze that
fix check the builders; this checks the finished page, which is the only place
the three separate gates -- the column's printed-value comparison, the charts'
``grouped_samples_changed``, the note's dict inequality -- can be caught
disagreeing with each other.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from export.report_selfcheck import (_oracle_transform_display_is_earned,  # noqa: E402
                                     load_report)

_ROWS = [("0.771219", "0.504369"), ("1.150088", "0.553047"), ("0.735494", "0.499484")]


def _page(tmp_path, *, declared="log10", rows=_ROWS, subjects=False,
          column=True, note=True, after=2, name="report.html"):
    """A raw data vault and an assumptions badge, which is all this check reads."""
    headers = ["Group"] + (["Subject"] if subjects else []) + ["Raw value"]
    if column:
        headers.append("Transformed value")
    head = "".join(f"<th>{h}</th>" for h in headers)

    body = ""
    for index, (raw, transformed) in enumerate(rows):
        cells = ['<td data-csv="G1">G1</td>']
        if subjects:
            cells.append(f'<td data-csv="S{index}">S{index}</td>')
        cells.append(f'<td data-csv="{raw}">{raw}</td>')
        if column:
            cells.append(f'<td data-csv="{transformed}">{transformed}</td>')
        body += "<tr>" + "".join(cells) + "</tr>"

    parts = []
    if declared is not None:
        parts.append(f'<div class="badge is-info">Transformation: {declared}</div>')
    parts.append(f'<table id="raw-data-table"><thead><tr>{head}</tr></thead>'
                 f"<tbody>{body}</tbody></table>")
    if note:
        parts.append("<p>Transformed-scale means (Mean &plusmn; SD): G1: 0.5 &plusmn; 0.1</p>")
    for _ in range(after):
        parts.append('<div class="section-kicker">Q-Q Diagnostic - After log10</div>')

    path = tmp_path / name
    path.write_text("<html><body>" + "".join(parts) + "</body></html>", encoding="utf-8")
    return load_report(str(path))


def _judge(report):
    violations = []
    fired = _oracle_transform_display_is_earned(report, {}, violations)
    return fired, violations


def test_a_real_transformation_passes(tmp_path):
    fired, violations = _judge(_page(tmp_path))
    assert fired and not violations, violations


def test_a_transformed_column_with_nothing_declared_is_a_finding(tmp_path):
    """The defect itself: transformed values shown for an untransformed run."""
    fired, violations = _judge(_page(tmp_path, declared="None"))

    assert fired
    assert violations and "Transformed value column" in violations[0]


def test_the_lowercase_spelling_is_caught_too(tmp_path):
    """Both spellings are live at once.

    The standard path renders "None"; the correlation path stores the string
    "none" and renders that. An oracle that knew only the first would pass the
    second by accident -- which is how a check ends up covering half its
    subject without anyone noticing.
    """
    for spelling in ("none", "None", "Keine", "skip", ""):
        fired, violations = _judge(_page(tmp_path, declared=spelling))
        assert fired and violations, f"{spelling!r} was accepted as a transformation"


def test_a_column_that_mirrors_the_raw_column_is_a_finding(tmp_path):
    """A named transform that changed nothing still claims a change on the page."""
    mirrored = [(raw, raw) for raw, _ in _ROWS]
    fired, violations = _judge(_page(tmp_path, rows=mirrored))

    assert fired
    assert violations and "repeats the raw column" in violations[0]


def test_an_empty_transformed_column_is_a_finding(tmp_path):
    fired, violations = _judge(_page(tmp_path, rows=[(raw, "N/A") for raw, _ in _ROWS]))

    assert fired
    assert violations and "cells are empty" in violations[0]


def test_after_transformation_charts_alone_are_enough_to_be_caught(tmp_path):
    """Each of the three claims is gated separately, so each must be checked."""
    fired, violations = _judge(
        _page(tmp_path, declared="None", column=False, note=False, after=2))

    assert fired
    assert violations and "after-transformation" in violations[0]


def test_a_subject_column_does_not_shift_the_comparison(tmp_path):
    """Cells are located by header, not by counting from the left.

    The subject column is conditional, so a positional read would compare the
    wrong two cells on exactly the designs that carry subjects -- and would do
    it silently, since the values it compared would still differ.
    """
    mirrored = [(raw, raw) for raw, _ in _ROWS]
    fired, violations = _judge(_page(tmp_path, rows=mirrored, subjects=True))

    assert fired
    assert violations and "repeats the raw column" in violations[0]


def test_a_report_claiming_nothing_and_declaring_nothing_does_not_apply(tmp_path):
    """No badge and no claim is not a pass -- there was nothing to judge."""
    fired, violations = _judge(
        _page(tmp_path, declared=None, column=False, note=False, after=0))

    assert not fired and not violations


def test_a_clean_untransformed_report_passes(tmp_path):
    """The common case: a badge saying None and no transformed anything."""
    fired, violations = _judge(
        _page(tmp_path, declared="None", column=False, note=False, after=0))

    assert fired and not violations, violations
