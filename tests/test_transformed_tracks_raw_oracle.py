"""Each printed row must pair a measurement with its own transformed value.

The raw data vault assembles its two numeric columns from two separate dicts.
Where those dicts were ordered differently the page showed one subject's raw
value beside another subject's transformed value -- every column still holding
the right multiset, so every mean, SD, Q-Q plot and distribution chart built
from them stayed correct. Nothing but the row-wise pairing was wrong, and
nothing was looking at it.

The check needs no knowledge of which transformation ran: log10, sqrt, Box-Cox
at any lambda and arcsin-sqrt are all monotonically increasing, so within a
group the ranking of the raw values must be the ranking of the transformed
ones. Where the badge names log10 there is a second, tolerance-free check --
a value below 1 has a negative base-10 logarithm.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from export.report_selfcheck import (  # noqa: E402
    _oracle_transformed_column_tracks_the_raw_one, load_report)

# Right-skewed positives, the shape that actually reaches a log10.
_RAW = [0.354201, 14.637243, 1.079895, 224.090231, 0.825990, 31.815742]


def _log_rows(values=_RAW):
    return [(f"{v:.6f}", f"{math.log10(v):.6f}") for v in values]


def _page(tmp_path, rows, *, declared="log10", groups=None, subjects=False,
          name="report.html"):
    """A raw data vault and an assumptions badge, which is all this check reads."""
    groups = groups or ["G1"] * len(rows)
    headers = ["Group"] + (["Subject"] if subjects else []) + \
              ["Raw value", "Transformed value"]
    head = "".join(f"<th>{h}</th>" for h in headers)

    body = ""
    for index, (raw, transformed) in enumerate(rows):
        cells = [f'<td data-csv="{groups[index]}">{groups[index]}</td>']
        if subjects:
            cells.append(f'<td data-csv="S{index}">S{index}</td>')
        cells.append(f'<td data-csv="{raw}">{raw}</td>')
        cells.append(f'<td data-csv="{transformed}">{transformed}</td>')
        body += "<tr>" + "".join(cells) + "</tr>"

    parts = []
    if declared is not None:
        parts.append(f'<div class="badge is-info">Transformation: {declared}</div>')
    parts.append(f'<table id="raw-data-table"><thead><tr>{head}</tr></thead>'
                 f"<tbody>{body}</tbody></table>")
    path = tmp_path / name
    path.write_text("<html><body>" + "".join(parts) + "</body></html>", encoding="utf-8")
    return load_report(str(path))


def _judge(report):
    violations = []
    fired = _oracle_transformed_column_tracks_the_raw_one(report, {}, violations)
    return fired, violations


def test_a_correctly_paired_column_passes(tmp_path):
    fired, violations = _judge(_page(tmp_path, _log_rows()))
    assert fired and not violations, violations


def test_a_rotated_transformed_column_is_caught(tmp_path):
    """The defect's exact shape: the right values, in the wrong rows."""
    rows = _log_rows()
    rotated = [(raw, rows[(i + 2) % len(rows)][1]) for i, (raw, _) in enumerate(rows)]
    fired, violations = _judge(_page(tmp_path, rotated))
    assert fired
    assert violations, "a rotated transformed column went unnoticed"
    assert "does not follow the raw one" in violations[0]


def test_two_swapped_rows_are_caught(tmp_path):
    """Not only a wholesale rotation -- one transposed pair is enough."""
    rows = _log_rows()
    swapped = list(rows)
    swapped[0] = (rows[0][0], rows[1][1])
    swapped[1] = (rows[1][0], rows[0][1])
    fired, violations = _judge(_page(tmp_path, swapped))
    assert fired and violations


def test_the_log10_sign_check_catches_an_impossible_row(tmp_path):
    """A raw value below 1 cannot carry a positive base-10 logarithm.

    Stated in one group of two rows that are themselves correctly ordered, so
    the monotonicity check stays silent and only the sign check can fire.
    """
    rows = [("0.500000", "0.100000"), ("2.000000", "0.301030")]
    fired, violations = _judge(_page(tmp_path, rows))
    assert fired
    assert any("log10" in v for v in violations), violations


def test_the_sign_check_stays_quiet_when_no_log_is_declared(tmp_path):
    """Only log10 has that sign property; sqrt and Box-Cox do not."""
    rows = [("0.500000", "0.100000"), ("2.000000", "0.301030")]
    fired, violations = _judge(_page(tmp_path, rows, declared="box_cox"))
    assert fired and not violations, violations


def test_ordering_is_checked_within_a_group_not_across_groups(tmp_path):
    """Groups on different transformed scales are not out of order with each other.

    The two halves must be transformed differently, or this proves nothing: one
    monotone function applied to everything is monotone across the whole table
    too, so a group-blind check would pass such a fixture and the test would
    assert nothing. Here group B sits on a scale that puts its transformed
    values *below* group A's while its raw values are *above* them -- so reading
    the table as one sequence yields a violation and reading it per group does
    not.
    """
    rows = [("1.000000", "0.000000"), ("2.000000", "0.301030"),
            ("3.000000", "0.477121"),
            ("4.000000", "0.040000"), ("5.000000", "0.050000"),
            ("6.000000", "0.060000")]
    groups = ["A", "A", "A", "B", "B", "B"]

    # The premise: read group-blind, this table IS out of order.
    blind = [(float(r), float(t)) for r, t in rows]
    assert any((a[0] < b[0]) != (a[1] < b[1])
               for i, a in enumerate(blind) for b in blind[i + 1:]), \
        "fixture no longer distinguishes a group-blind reading from a scoped one"

    fired, violations = _judge(_page(tmp_path, rows, groups=groups,
                                     declared="box_cox"))
    assert fired and not violations, violations


def test_the_group_column_is_actually_read(tmp_path):
    """Named in the finding, so a table that stops being grouped cannot hide.

    ``<th[^>]*>`` matched ``<thead>`` as well, which pushed every header along
    by one and left the Group column unfindable by name -- silently collapsing
    every row into a single unnamed bucket.
    """
    rows = _log_rows([1.0, 2.0, 3.0])
    swapped = [rows[0], (rows[1][0], rows[2][1]), (rows[2][0], rows[1][1])]
    fired, violations = _judge(_page(tmp_path, swapped, groups=["Treated"] * 3))
    assert fired and violations
    assert "'Treated'" in violations[0], violations


def test_a_subject_column_does_not_shift_the_comparison(tmp_path):
    """The subject column is conditional; the cells are found by header."""
    fired, violations = _judge(_page(tmp_path, _log_rows(), subjects=True))
    assert fired and not violations, violations

    rows = _log_rows()
    rotated = [(raw, rows[(i + 1) % len(rows)][1]) for i, (raw, _) in enumerate(rows)]
    fired, violations = _judge(_page(tmp_path, rotated, subjects=True,
                                     name="rotated.html"))
    assert fired and violations, "the subject column hid a mispaired table"


def test_ties_and_unreadable_cells_carry_no_ordering_claim(tmp_path):
    """Equal raw values and N/A cells are skipped rather than guessed at."""
    rows = [("1.000000", "0.000000"), ("1.000000", "0.000000"),
            ("5.000000", "N/A"), ("10.000000", "1.000000")]
    fired, violations = _judge(_page(tmp_path, rows))
    assert fired and not violations, violations


def test_a_report_with_no_transformed_column_does_not_fire(tmp_path):
    path = tmp_path / "plain.html"
    path.write_text(
        '<html><body><div class="badge is-info">Transformation: None</div>'
        '<table id="raw-data-table"><thead><tr><th>Group</th>'
        "<th>Raw value</th></tr></thead><tbody>"
        '<tr><td data-csv="G1">G1</td><td data-csv="1.0">1.0</td></tr>'
        "</tbody></table></body></html>",
        encoding="utf-8")
    fired, violations = _judge(load_report(str(path)))
    assert not fired and not violations


def test_a_single_row_group_does_not_fire(tmp_path):
    """One value has no ordering, so there is nothing to judge."""
    fired, violations = _judge(_page(tmp_path, _log_rows([7.0])))
    assert not fired and not violations
