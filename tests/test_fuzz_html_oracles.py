"""The report oracles must be able to fail.

An oracle that cannot fail is worse than none: it reports coverage it does not
have, and a green fuzz run then means only that nothing was looked at. Each test
below breaks exactly one invariant in an otherwise valid report and demands that
the matching oracle -- and, where it matters, only that one -- says so.

The reports here are synthetic. They carry the structure the real exporter
emits, which was read off actual renders, and keep the tests fast enough to run
with the ordinary suite; that the structure matches the product is verified by
the fuzzer itself, which runs these same oracles against real renders.
"""

import json

import pytest

from fuzzing.html_oracles import ORACLES, check_report
from visualization import style_tokens

GROUPS = ["G1", "G2", "G3", "G4"]

# A~B and B~C but A#C: letters must keep A and C apart.
PAIRS = [
    {"pair_id": 0, "group1": "G1", "group2": "G2", "p_value": 0.4, "stars": "", "significant": False},
    {"pair_id": 1, "group1": "G1", "group2": "G3", "p_value": 0.01, "stars": "*", "significant": True},
    {"pair_id": 2, "group1": "G1", "group2": "G4", "p_value": 0.001, "stars": "**", "significant": True},
    {"pair_id": 3, "group1": "G2", "group2": "G3", "p_value": 0.3, "stars": "", "significant": False},
    {"pair_id": 4, "group1": "G2", "group2": "G4", "p_value": 0.002, "stars": "**", "significant": True},
    {"pair_id": 5, "group1": "G3", "group2": "G4", "p_value": 0.02, "stars": "*", "significant": True},
]

LETTERS = {0: "c", 1: "bc", 2: "b", 3: "a"}

RESULT = {
    "test": "One-Way ANOVA",
    "p_value": 0.002,
    "groups": GROUPS,
    "raw_data": {g: [1.0, 2.0, 3.0] for g in GROUPS},
    "pairwise_comparisons": PAIRS,
}

SECTIONS = ("hd-results", "hd-assumptions", "hd-charts", "hd-decision",
            "hd-descriptive", "hd-methods", "hd-raw", "hd-pairwise")


def _figure(letters):
    annotations = [{"text": f"<b>{text}</b>", "x": x, "y": 1.0 + x,
                    "showarrow": False, "font": {"color": "#16313a", "size": 13}}
                   for x, text in sorted(letters.items())]
    data = [{"type": "box", "name": g, "y": [1.0, 2.0, 3.0],
             "marker": {"color": "#4E79A7"}} for g in GROUPS]
    layout = {"annotations": annotations, "xaxis": {"title": {"text": "Group"}},
              "font": {"family": style_tokens.FONT_FAMILY_STACK, "size": 12}}
    return (f'Plotly.newPlot("biomedstatx-group-chart", {json.dumps(data)}, '
            f'{json.dumps(layout)}, {{"responsive": true}});')


def _report_html(*, sections=SECTIONS, payloads=None, letters=None, sig_mode="letters",
                 p_text="p = 0.002 **", extra=""):
    letters = LETTERS if letters is None else letters
    payloads = _payloads() if payloads is None else payloads
    blocks = "".join(
        f'<script id="{pid}" type="application/json">{body}</script>'
        for pid, body in payloads.items())
    heads = "".join(f'<h2 id="{s}">Section</h2>' for s in sections)
    return (
        "<html><head><style>body{font-family:\"Segoe UI\",Arial,sans-serif}</style></head><body>"
        + heads
        + f'<div class="metric-value">{p_text.replace("<", "&lt;")}</div>'
        + blocks
        + f'<script>{_figure(letters)}</script>'
        + f'<script>const SIG_MODE="{sig_mode}";</script>'
        + extra
        + "<p>" + "padding " * 200 + "</p></body></html>"
    )


def _payloads():
    return {
        "pd-data-plot": json.dumps({g: [1.0, 2.0, 3.0] for g in GROUPS}),
        "pd-data-order": json.dumps(GROUPS),
        "pd-data-pairs": json.dumps(PAIRS),
        "pd-data-stats": json.dumps({g: {"n": 3} for g in GROUPS}),
        "pd-data-style": json.dumps({"default_palette": "grayscale"}),
        "pd-data-paired-lines": json.dumps({
            "supported": False,
            "reason": ("No subject identity in this result, so there is nothing to "
                       "connect. Independent designs measure different individuals per group."),
            "max_subjects": 30,
            "trajectories": [],
        }),
    }


def _write(tmp_path, html, name="report.html"):
    path = tmp_path / name
    path.write_text(html, encoding="utf-8")
    return str(path)


def _check(tmp_path, html, result=None):
    return check_report(_write(tmp_path, html), dict(result or RESULT))


def test_a_faithful_report_passes_every_oracle(tmp_path):
    violations, fired = _check(tmp_path, _report_html())
    assert violations == []
    # Everything except the resolution check, which needs an estimated p-value,
    # and the bracket-mode guard, which is the other half of this report's mode.
    assert set(fired) == ({name for name, _ in ORACLES}
                          - {"p_precision_capped", "brackets_have_no_letters"})


def test_an_unparseable_payload_is_caught(tmp_path):
    payloads = _payloads()
    payloads["pd-data-pairs"] = "{not json,}"
    violations, _ = _check(tmp_path, _report_html(payloads=payloads))
    assert any("pd-data-pairs is not valid JSON" in v for v in violations)


def test_a_missing_payload_is_caught(tmp_path):
    payloads = _payloads()
    del payloads["pd-data-order"]
    violations, _ = _check(tmp_path, _report_html(payloads=payloads))
    assert any("pd-data-order missing" in v for v in violations)


def test_a_dropped_figure_builder_is_caught(tmp_path):
    """Plottable groups in the result, no designer in the file."""
    violations, fired = _check(tmp_path, _report_html(payloads={}))
    assert "designer_when_plottable" in fired
    assert any("no figure-builder payloads" in v for v in violations)


def test_a_design_without_groups_does_not_demand_a_designer(tmp_path):
    """Correlation and regression have no group chart; that is not a defect."""
    result = {"test": "Pearson correlation", "p_value": 0.01, "raw_data": {}}
    violations, fired = check_report(
        _write(tmp_path, _report_html(payloads={}, sections=SECTIONS[:-1],
                                      p_text="p = 0.010 *", sig_mode="brackets",
                                      letters={})),
        result)
    assert violations == []
    assert "designer_when_plottable" not in fired
    assert "payloads_parse" not in fired


def test_a_missing_section_is_caught(tmp_path):
    violations, _ = _check(tmp_path, _report_html(sections=tuple(s for s in SECTIONS
                                                                if s != "hd-descriptive")))
    assert any("hd-descriptive missing" in v for v in violations)


def test_comparisons_without_a_pairwise_section_are_caught(tmp_path):
    """The engine produced comparisons and the report shows none of them."""
    violations, _ = _check(tmp_path, _report_html(
        sections=tuple(s for s in SECTIONS if s != "hd-pairwise")))
    assert any("no pairwise section" in v for v in violations)


def test_a_blocked_run_may_omit_the_pairwise_section(tmp_path):
    result = dict(RESULT, pairwise_comparisons=[], blocked=True)
    violations, _ = check_report(
        _write(tmp_path, _report_html(sections=tuple(s for s in SECTIONS if s != "hd-pairwise"),
                                      sig_mode="brackets", letters={})),
        result)
    assert violations == []


def test_an_n_a_where_a_number_belongs_is_caught(tmp_path):
    violations, _ = _check(tmp_path, _report_html(p_text="N/A"))
    assert any("is absent from the report" in v for v in violations)


def test_letters_drawn_from_an_incomplete_matrix_are_caught(tmp_path):
    """A many-to-one post-hoc cannot carry letters: the untested pairs would
    silently read as 'not different'."""
    dunnett = [p for p in PAIRS if "G1" in (p["group1"], p["group2"])]
    payloads = _payloads()
    payloads["pd-data-pairs"] = json.dumps(dunnett)
    violations, fired = _check(
        tmp_path, _report_html(payloads=payloads),
        result=dict(RESULT, pairwise_comparisons=dunnett))
    assert "letters_gate_complete" in fired
    assert any("3 comparisons, 6 required" in v for v in violations)


def test_a_shared_letter_between_different_groups_is_caught(tmp_path):
    """The exact failure of the superseded star-and-absorb implementation."""
    collapsed = {i: "a" for i in range(len(GROUPS))}
    violations, fired = _check(tmp_path, _report_html(letters=collapsed))
    assert "letters_match_pairs" in fired
    assert any("G1 vs G3 is significant but they share" in v for v in violations)


def test_a_missing_shared_letter_between_equal_groups_is_caught(tmp_path):
    """The converse: a plot claiming a difference the test did not find."""
    split = {0: "a", 1: "b", 2: "c", 3: "d"}
    violations, _ = _check(tmp_path, _report_html(letters=split))
    assert any("G1 vs G2 is not significant but shares no letter" in v for v in violations)


def test_a_group_without_a_letter_is_caught(tmp_path):
    partial = {0: "c", 1: "bc", 2: "b"}
    violations, _ = _check(tmp_path, _report_html(letters=partial))
    assert any("groups without a letter: ['G4']" in v for v in violations)


def test_letters_left_on_the_chart_in_bracket_mode_are_caught(tmp_path):
    """The two layers disagreeing about which form won is the drift itself."""
    violations, fired = _check(tmp_path, _report_html(sig_mode="brackets"))
    assert "brackets_have_no_letters" in fired
    assert any("mode is 'brackets' but the chart carries letters" in v for v in violations)


def test_stars_are_not_mistaken_for_letters(tmp_path):
    """Bracket stars are annotations too; only alphabetic text is a letter."""
    violations, fired = _check(tmp_path, _report_html(sig_mode="brackets",
                                                      letters={0: "***", 1: "*"}))
    assert violations == []
    assert "letters_match_pairs" not in fired


def test_a_paired_line_verdict_that_drifted_is_caught(tmp_path):
    payloads = _payloads()
    payloads["pd-data-paired-lines"] = json.dumps(
        {"supported": True, "reason": "", "max_subjects": 30, "trajectories": []})
    violations, fired = _check(tmp_path, _report_html(payloads=payloads))
    assert "paired_line_gate" in fired
    assert any("paired-line gate says supported=True" in v for v in violations)


def test_a_refusal_without_a_reason_is_caught(tmp_path):
    payloads = _payloads()
    payloads["pd-data-paired-lines"] = json.dumps(
        {"supported": False, "reason": "  ", "max_subjects": 30,
         "trajectories": [{"subject": "S1", "points": []}]})
    violations, _ = _check(tmp_path, _report_html(payloads=payloads))
    assert any("refused without a reason" in v for v in violations)
    assert any("trajectories were emitted anyway" in v for v in violations)


def test_lines_across_the_cells_of_a_mixed_design_are_caught(tmp_path):
    """Mixed cells are combinations, so a line across them asserts no real path."""
    order = ["Between=B0, Time=T0", "Between=B0, Time=T1"]
    subjects = {g: ["S1", "S2"] for g in order}
    payloads = _payloads()
    payloads["pd-data-order"] = json.dumps(order)
    payloads["pd-data-paired-lines"] = json.dumps(
        {"supported": True, "reason": "", "max_subjects": 30,
         "trajectories": [{"subject": "S1", "points": []}]})
    result = dict(RESULT, design_type="mixed", raw_data_subjects=subjects,
                  raw_data={g: [1.0, 2.0] for g in order},
                  pairwise_comparisons=[])
    violations, _ = check_report(
        _write(tmp_path, _report_html(payloads=payloads,
                                      sections=tuple(s for s in SECTIONS if s != "hd-pairwise"),
                                      sig_mode="brackets", letters={})),
        result)
    assert any("mixed design" in v for v in violations)


def test_an_alphabetically_sorted_axis_is_caught(tmp_path):
    payloads = _payloads()
    payloads["pd-data-order"] = json.dumps(["Week 12", "Week 4", "Baseline"])
    violations, fired = _check(tmp_path, _report_html(payloads=payloads))
    assert "axis_order_ranked" in fired
    assert any("is not the ranked order" in v for v in violations)


def test_a_stray_font_family_is_caught(tmp_path):
    violations, fired = _check(
        tmp_path, _report_html(extra='<script>var x = {"family": "Comic Sans MS"};</script>'))
    assert "one_plot_font" in fired
    assert any("Comic Sans MS" in v for v in violations)


@pytest.mark.parametrize("printed, expect_violation", [
    ("bound", False),      # the bound the resolution allows
    ("figure", True),      # the figure the method never measured
])
def test_a_simulated_headline_p_printed_in_full_is_caught(tmp_path, printed, expect_violation):
    """A permutation p at the grid floor means 'nothing beat it', not a magnitude."""
    from export.report_formatting import _FormattingMixin

    resolution = 1.0 / 5001
    p = 1e-9
    bound = _FormattingMixin._format_p_value(p, resolution)
    text = bound if printed == "bound" else _FormattingMixin._format_p_value(p)
    result = dict(RESULT, p_value=p, p_value_resolution=resolution)
    violations, fired = _check(tmp_path, _report_html(p_text=text), result=result)
    assert "p_precision_capped" in fired
    assert bool([v for v in violations if "headline" in v]) is expect_violation


def test_a_contrast_row_that_lost_its_resolution_is_caught(tmp_path):
    """The row builder dropping the resolution is the display-side half of the bug.

    The engine attaches it and the report renders through _build_pairwise_rows;
    if that call stops passing it on, the number reaches the reader with a
    precision the Monte-Carlo integration never had.
    """
    from analysis.posthoc_core import PostHocAnalyzer
    from export.report_formatting import _FormattingMixin

    resolution = 1e-6
    p = 4.2e-8
    result = PostHocAnalyzer.create_result_template("EMM")
    PostHocAnalyzer.add_comparison(
        result, group1="G1", group2="G2", test="EMM + multivariate-t",
        p_value=p, statistic=12.0, significant=True, p_value_resolution=resolution)
    payload = dict(RESULT, pairwise_comparisons=result["pairwise_comparisons"])

    bound = _FormattingMixin._format_p_value(p, resolution)
    violations, fired = _check(
        tmp_path, _report_html(extra=f"<td>{bound}</td>"), result=payload)
    assert "p_precision_capped" in fired
    assert violations == [], violations

    # Now the same report without the bound anywhere in it.
    violations, _ = _check(tmp_path, _report_html(), result=payload)
    assert any("never reaches the report" in v for v in violations)


def test_an_analytic_twin_of_the_same_number_is_not_a_violation(tmp_path):
    """Two p-values, one estimated and one derived, can render identically.

    In a two-level repeated-measures design F is t squared, so the analytic
    omnibus and the Monte-Carlo contrast are the same quantity. A document-wide
    search for the full figure found the omnibus -- correctly printed -- and
    reported it as the contrast printed without its bound.
    """
    from analysis.posthoc_core import PostHocAnalyzer
    from export.report_formatting import _FormattingMixin

    resolution = 1e-6
    p = 4.2310417862949984e-08
    omnibus_p = 4.2310417862950904e-08
    result = PostHocAnalyzer.create_result_template("EMM")
    PostHocAnalyzer.add_comparison(
        result, group1="G1", group2="G2", test="EMM + multivariate-t",
        p_value=p, statistic=12.0, significant=True, p_value_resolution=resolution)
    payload = dict(RESULT, p_value=omnibus_p, p_value_resolution=None,
                   pairwise_comparisons=result["pairwise_comparisons"])

    bound = _FormattingMixin._format_p_value(p, resolution)
    analytic = _FormattingMixin._format_p_value(omnibus_p)
    assert analytic != bound
    html = _report_html(p_text=analytic, extra=f"<span>{bound}</span>")
    violations, fired = _check(tmp_path, html, result=payload)
    assert "p_precision_capped" in fired
    assert violations == [], violations
