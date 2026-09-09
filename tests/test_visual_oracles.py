"""Unit-level proof that the browser oracles judge what they claim to judge.

``fuzzing/visual_selfcheck.py`` is the stronger proof -- it breaks a real
report and opens it in a real browser -- but it needs Playwright and a Chromium
download, so it cannot run in the ordinary suite. These tests do the fast half:
they hand each oracle a synthetic snapshot in the exact shape
``_visual_worker._SNAPSHOT_JS`` produces and check it says so.

The snapshot shape is the thing that could drift. It is pinned in one place
here, so a field renamed in the worker breaks these tests rather than silently
turning every oracle into a no-op.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fuzzing.visual_oracles import (ORACLES, check_download,  # noqa: E402
                                    check_stage)

_BY_NAME = dict(ORACLES)


def snapshot(**overrides):
    """A clean, plottable page -- every oracle should pass on this."""
    base = {
        "stage": "test",
        "console_errors": [],
        "page_errors": [],
        "figures": [{"id": "pd-plot", "traces": 4, "svgs": 3, "w": 869, "h": 680}],
        "designer": True,
        "has_plot_payload": True,
        "payload_order": ["Vehicle", "Dose1", "Dose2", "Dose10"],
        "order_user_modified": False,
        "pd": {
            "traces": 4, "w": 869, "h": 680, "overflow": [],
            "bracket_shapes": 0, "other_shapes": 0,
            "letter_annotations": 4, "annotations": 4,
            "categories": ["Vehicle", "Dose1", "Dose2", "Dose10"],
            "categories_y": ["8", "10", "12"],
            "sig_mode": "letters", "sig_disabled": False, "warning": "",
        },
    }
    pd_over = overrides.pop("pd", None)
    base.update(overrides)
    if pd_over is not None:
        base["pd"] = {**base["pd"], **pd_over}
    return base


def run(name, snap):
    found = []
    fired = _BY_NAME[name](snap, found)
    return fired, found


def test_a_clean_page_violates_nothing():
    violations, fired = check_stage(snapshot())
    assert violations == [], violations
    # A silent run would pass this test just as happily, so the oracles that
    # should have been in a position to judge are named explicitly.
    for name in ("no_script_error", "figures_render", "designer_live_when_plottable",
                 "labels_not_clipped", "significance_matches_mode",
                 "designer_keeps_report_order"):
        assert name in fired, f"{name} did not fire on a page it should have judged"


@pytest.mark.parametrize("field", ["console_errors", "page_errors"])
def test_script_errors_are_findings(field):
    fired, found = run("no_script_error", snapshot(**{field: ["TypeError: x is not a function"]}))
    assert fired and found and "x is not a function" in found[0]


@pytest.mark.parametrize("broken,needle", [
    ({"traces": 0}, "rendered no trace"),
    ({"svgs": 0}, "produced no SVG"),
    ({"w": 0, "h": 0}, "zero size"),
])
def test_a_figure_that_is_not_drawn_is_a_finding(broken, needle):
    figure = {"id": "biomedstatx-group-chart", "traces": 2, "svgs": 3, "w": 600, "h": 400}
    figure.update(broken)
    fired, found = run("figures_render", snapshot(figures=[figure]))
    assert fired and found and needle in found[0]


def test_plot_data_without_a_designer_is_a_finding():
    fired, found = run("designer_live_when_plottable", snapshot(designer=False))
    assert fired and found and "no designer panel" in found[0]


def test_an_empty_designer_figure_is_a_finding():
    fired, found = run("designer_live_when_plottable", snapshot(pd={"traces": 0}))
    assert fired and found and "no trace" in found[0]


def test_an_announced_refusal_is_not_a_finding():
    """Some designs have no forest plot to draw; saying so is correct behaviour.

    What must not pass is a *silent* empty canvas, which is why the excuse is
    the warning text and not the emptiness itself.
    """
    excused = snapshot(pd={"traces": 0, "warning": "No plottable data found."},
                       figures=[{"id": "pd-plot", "traces": 0, "svgs": 3, "w": 869, "h": 680}])
    violations, fired = check_stage(excused)
    assert violations == [], violations
    assert "designer_live_when_plottable" in fired and "figures_render" in fired


def test_the_excuse_covers_only_the_designer_canvas():
    """A report chart that draws nothing is a finding whatever the designer says."""
    snap = snapshot(pd={"traces": 0, "warning": "No plottable data found."},
                    figures=[{"id": "pd-plot", "traces": 0, "svgs": 3, "w": 869, "h": 680},
                             {"id": "biomedstatx-group-chart", "traces": 0, "svgs": 3,
                              "w": 600, "h": 400}])
    fired, found = run("figures_render", snap)
    assert fired and len(found) == 1 and "biomedstatx-group-chart" in found[0], found


def test_the_designer_oracle_stays_silent_without_plot_data():
    fired, found = run("designer_live_when_plottable",
                       snapshot(has_plot_payload=False, designer=False))
    assert not fired and not found


def test_text_outside_the_container_is_a_finding():
    fired, found = run("labels_not_clipped", snapshot(
        pd={"overflow": [{"text": "Concentration [nmol/L]", "cls": "ytick", "over": 17}]}))
    assert fired and found and "17px" in found[0]


def test_brackets_in_letters_mode_are_a_finding():
    fired, found = run("significance_matches_mode",
                       snapshot(pd={"sig_mode": "letters", "bracket_shapes": 18}))
    assert fired and found and "letters mode" in found[0]


def test_anything_drawn_in_none_mode_is_a_finding():
    fired, found = run("significance_matches_mode",
                       snapshot(pd={"sig_mode": "none", "bracket_shapes": 6,
                                    "letter_annotations": 0}))
    assert fired and found and "significance is off" in found[0]


def test_a_disabled_significance_control_is_not_judged():
    fired, found = run("significance_matches_mode",
                       snapshot(pd={"sig_disabled": True, "bracket_shapes": 3}))
    assert not fired and not found


def test_an_alphabetically_resorted_axis_is_a_finding():
    """The exact shape of the bug fixed centrally in analysis_core."""
    fired, found = run("designer_keeps_report_order", snapshot(
        pd={"categories": ["Dose1", "Dose10", "Dose2", "Vehicle"]}))
    assert fired and found and "does not match" in found[0]


def test_groups_on_the_y_axis_are_read_there():
    """Forest and the horizontal layouts put a numeric scale on x."""
    fired, found = run("designer_keeps_report_order", snapshot(pd={
        "categories": ["-4", "-2", "0", "2", "4"],
        "categories_y": ["Vehicle", "Dose1", "Dose2", "Dose10"]}))
    assert fired and not found
    # Plotly numbers a categorical y axis from the bottom, so the reverse is
    # the same figure read the other way and must not be called a defect.
    fired, found = run("designer_keeps_report_order", snapshot(pd={
        "categories": ["-4", "0", "4"],
        "categories_y": ["Dose10", "Dose2", "Dose1", "Vehicle"]}))
    assert fired and not found


def test_a_numeric_axis_alone_is_not_judged():
    """No overlap at all means a different kind of figure, not a broken one."""
    fired, found = run("designer_keeps_report_order", snapshot(pd={
        "categories": ["-4", "-2", "0"], "categories_y": ["1", "2", "3"]}))
    assert not fired and not found


def test_a_group_missing_from_the_axis_is_a_finding():
    fired, found = run("designer_keeps_report_order", snapshot(pd={
        "categories": ["Vehicle", "Dose1", "Dose2"], "categories_y": ["8", "10"]}))
    assert fired and found and "Dose10" in found[0]


def test_a_user_reorder_is_allowed_but_must_keep_every_group():
    reordered = ["Dose1", "Vehicle", "Dose2", "Dose10"]
    fired, found = run("designer_keeps_report_order",
                       snapshot(order_user_modified=True, pd={"categories": reordered}))
    assert fired and not found, found

    fired, found = run("designer_keeps_report_order", snapshot(
        order_user_modified=True,
        pd={"categories": ["Vehicle", "Dose1", "Dose2", "Dose10", "Dose1"]}))
    assert fired and not found, "a duplicate is not a lost group"

    # Losing one is a finding in its own right, and it is the case the oracle
    # nearly missed: the surviving groups no longer cover the reported set, so
    # an axis-identification that only accepts a full match would have filed a
    # dropped group as "no group axis here" and passed.
    snap = snapshot(order_user_modified=True, pd={"categories": ["Vehicle", "Dose1"],
                                                 "categories_y": ["8", "10"]})
    fired, found = run("designer_keeps_report_order", snap)
    assert fired and found and "reached no axis" in found[0], found


def test_the_designer_warning_is_collected_not_judged():
    snap = snapshot(pd={"warning": "Letters need a complete comparison matrix."})
    fired, found = run("designer_warning_text", snap)
    assert fired and not found
    assert snap["warnings_seen"] == ["Letters need a complete comparison matrix."]


@pytest.mark.parametrize("fmt,payload,error,needle", [
    ("png", None, "", "produced no file"),
    ("svg", b"<svg/>", "", "empty canvas"),
    ("png", b"not a png" * 900, "", "does not start with"),
    ("svg", None, "TimeoutError: no download", "failed"),
])
def test_a_download_that_is_not_a_figure_is_a_finding(tmp_path, fmt, payload, error, needle):
    target = str(tmp_path / f"out.{fmt}")
    if payload is not None:
        with open(target, "wb") as fh:
            fh.write(payload)
    elif not error:
        target = str(tmp_path / f"absent.{fmt}")
    violations, fired = check_download(fmt, "" if error else target, error)
    assert fired == [f"download_{fmt}"]
    assert violations and needle in violations[0]


def test_a_real_looking_download_passes(tmp_path):
    png = tmp_path / "figure.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20000)
    violations, fired = check_download("png", str(png))
    assert fired == ["download_png"] and violations == []


def test_a_broken_oracle_reports_itself():
    """A snapshot the oracles cannot read must not pass as clean."""
    violations, _ = check_stage({"stage": "broken", "pd": "not-a-dict",
                                 "figures": [], "console_errors": [], "page_errors": []})
    assert any("raised" in v for v in violations), violations
