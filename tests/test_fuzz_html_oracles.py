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


# A raw data vault with a correctly paired transformed column, so the check that
# reads it fires on the faithful fixture instead of sitting out the "passes every
# oracle" claim. log10 of each raw value, printed the way the template prints it.
RAW_TABLE = (
    '<table id="raw-data-table"><thead><tr><th>Group</th>'
    '<th>Raw value</th><th>Transformed value</th></tr></thead><tbody>'
    '<tr><td data-csv="G1">G1</td><td data-csv="2.000000">2.000000</td>'
    '<td data-csv="0.301030">0.301030</td></tr>'
    '<tr><td data-csv="G1">G1</td><td data-csv="20.000000">20.000000</td>'
    '<td data-csv="1.301030">1.301030</td></tr>'
    '<tr><td data-csv="G2">G2</td><td data-csv="5.000000">5.000000</td>'
    '<td data-csv="0.698970">0.698970</td></tr>'
    '<tr><td data-csv="G2">G2</td><td data-csv="50.000000">50.000000</td>'
    '<td data-csv="1.698970">1.698970</td></tr>'
    "</tbody></table>")


def _report_html(*, sections=SECTIONS, payloads=None, letters=None, sig_mode="letters",
                 p_text="p = 0.002 **", extra="", transformation="log10",
                 raw_table=RAW_TABLE):
    letters = LETTERS if letters is None else letters
    payloads = _payloads() if payloads is None else payloads
    blocks = "".join(
        f'<script id="{pid}" type="application/json">{body}</script>'
        for pid, body in payloads.items())
    heads = "".join(f'<h2 id="{s}">Section</h2>' for s in sections)
    return (
        "<html><head><style>body{font-family:\"Segoe UI\",Arial,sans-serif}</style></head><body>"
        + heads
        # Every real report carries this badge, transformed or not, and the
        # transform check reads it. Leaving it out would make the fixture a
        # report no export produces, and would quietly excuse that oracle from
        # the "passes every oracle" claim below.
        + f'<div class="badge is-info">Transformation: {transformation}</div>'
        + f'<div class="metric-value">{p_text.replace("<", "&lt;")}</div>'
        + blocks
        + f'<script>{_figure(letters)}</script>'
        + f'<script>const SIG_MODE="{sig_mode}";</script>'
        + raw_table
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


def test_transformed_values_in_a_report_that_transformed_nothing_are_caught(tmp_path):
    """The whole report, not the builder: the page declares None and shows a column.

    Checked here as well as in the oracle's own tests because this is the path a
    fuzz run actually takes -- through check_report on a written file -- and a
    check that works in isolation but is not wired into that path finds nothing.
    """
    column = ('<table id="raw-data-table"><thead><tr><th>Group</th>'
              '<th>Raw value</th><th>Transformed value</th></tr></thead><tbody>'
              '<tr><td data-csv="G1">G1</td><td data-csv="1.0">1.0</td>'
              '<td data-csv="0.0">0.0</td></tr></tbody></table>')
    violations, fired = _check(tmp_path, _report_html(transformation="None",
                                                      raw_table=column))

    assert "transform_display_earned" in fired
    assert any("declares its transformation as" in v for v in violations), violations


def test_a_mispaired_transformed_column_is_caught(tmp_path):
    """The defect's real shape: right values, wrong rows, through check_report.

    Group G1's two rows carry each other's transformed value. Every number on
    the page is one the transformation genuinely produced, the column's mean and
    SD are untouched, and only the row-wise pairing is wrong -- which is what
    made the original go unnoticed.
    """
    swapped = RAW_TABLE.replace(
        '<td data-csv="0.301030">0.301030</td>', '<td data-csv="__A__">__A__</td>'
    ).replace(
        '<td data-csv="1.301030">1.301030</td>', '<td data-csv="0.301030">0.301030</td>'
    ).replace(
        '<td data-csv="__A__">__A__</td>', '<td data-csv="1.301030">1.301030</td>')
    violations, fired = _check(tmp_path, _report_html(raw_table=swapped))

    assert "transformed_tracks_raw" in fired
    assert any("does not follow the raw one" in v for v in violations), violations
    assert any("'G1'" in v for v in violations), violations


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


# --- the statistical oracles, whose stale assumptions the fuzzer exposed ------


def test_a_negative_t_is_not_reported_as_an_impossible_f():
    """Welch's *t*-test matched the "welch" keyword meant for Welch's ANOVA.

    Every negative t it produced was filed as an impossible F -- a finding that
    looked like a bug and was noise from the checker.
    """
    from fuzzing.oracles import check_result

    t_test = {"test": "Welch's t-test (unequal variances)", "statistic": -1.29,
              "p_value": 0.21}
    assert check_result(t_test) == []

    anova = {"test": "Welch ANOVA", "statistic": -1.29, "p_value": 0.21}
    assert any("negative" in v for v in check_result(anova))


def test_a_cancelled_analysis_is_a_self_identifying_result():
    """Backing out of a mid-analysis dialog is correct behaviour, not a defect."""
    from fuzzing.oracles import check_result

    cancelled = {"cancelled": True, "cancel_reason": "unequal sample sizes"}
    assert check_result(cancelled) == []

    nameless = {"analysis_log": "..."}
    assert any("test' label" in v for v in check_result(nameless))


# --- the combined report of a multi-dataset run --------------------------------


def _multi_html(*, names=("DS1", "DS2"), summarized=None, p_texts=(), extra=""):
    summarized = len(names) if summarized is None else summarized
    cards = "".join(f'<article class="dataset-card"><h3>{n}</h3></article>' for n in names)
    ps = "".join(f'<div class="metric-value">{t.replace("<", "&lt;")}</div>' for t in p_texts)
    return (
        "<html><body><header class='hero'>"
        f"<p class='hero-subtitle'>{summarized} datasets summarized, 1 significant "
        "main results.</p></header>"
        f'<div class="dataset-grid">{cards}</div>{ps}{extra}'
        + "<p>" + "padding " * 400 + "</p></body></html>"
    )


def _multi_result(names=("DS1", "DS2"), failed=None, p_values=None, fdr=None):
    p_values = p_values or {n: 0.02 for n in names}
    results = {}
    for n in names:
        sub = {"test": "One-Way ANOVA", "p_value": p_values[n]}
        if fdr and n in fdr:
            sub["p_value_fdr"] = fdr[n]
        results[n] = sub
    return {"type": "multi_dataset_analysis",
            "successful_datasets": list(names),
            "failed_datasets": failed or {},
            "results": results}


def test_a_faithful_overview_passes(tmp_path):
    from export.report_formatting import _FormattingMixin
    from fuzzing.html_oracles import check_multi_report

    result = _multi_result()
    texts = [_FormattingMixin._format_p_value(0.02)]
    path = _write(tmp_path, _multi_html(p_texts=texts), name="combined.html")
    violations, fired = check_multi_report(path, result)
    assert violations == []
    assert set(fired) == {"multi_lists_datasets", "multi_count_matches",
                          "multi_p_values_rendered"}


def test_a_dataset_missing_from_the_overview_is_caught(tmp_path):
    from export.report_formatting import _FormattingMixin
    from fuzzing.html_oracles import check_multi_report

    result = _multi_result()
    texts = [_FormattingMixin._format_p_value(0.02)]
    path = _write(tmp_path, _multi_html(names=("DS1",), summarized=2, p_texts=texts),
                  name="combined.html")
    violations, _ = check_multi_report(path, result)
    assert any("DS2" in v and "absent from the overview" in v for v in violations)


def test_a_headline_count_that_lies_is_caught(tmp_path):
    from export.report_formatting import _FormattingMixin
    from fuzzing.html_oracles import check_multi_report

    result = _multi_result()
    texts = [_FormattingMixin._format_p_value(0.02)]
    path = _write(tmp_path, _multi_html(summarized=7, p_texts=texts), name="combined.html")
    violations, _ = check_multi_report(path, result)
    assert any("says 7 datasets but 2 were analysed" in v for v in violations)


def test_an_fdr_adjusted_p_that_never_reaches_the_reader_is_caught(tmp_path):
    from export.report_formatting import _FormattingMixin
    from fuzzing.html_oracles import check_multi_report

    result = _multi_result(fdr={"DS1": 0.043})
    texts = [_FormattingMixin._format_p_value(0.02)]
    path = _write(tmp_path, _multi_html(p_texts=texts), name="combined.html")
    violations, fired = check_multi_report(path, result)
    assert "multi_p_values_rendered" in fired
    assert any("FDR-adjusted p" in v for v in violations)


def test_a_failed_dataset_that_vanishes_is_caught(tmp_path):
    """The run records the failure; the reader only ever gets the report.

    `_prepare_multi_report_context` takes `all_results` alone, so a dataset that
    errored has no channel into the overview at all -- it simply disappears, and
    the headline counts the survivors. The sibling outlier export does carry its
    failures through.
    """
    from export.report_formatting import _FormattingMixin
    from fuzzing.html_oracles import check_multi_report

    result = _multi_result(names=("DS1",), failed={"DS2": "engine blew up"})
    texts = [_FormattingMixin._format_p_value(0.02)]
    path = _write(tmp_path, _multi_html(names=("DS1",), summarized=1, p_texts=texts),
                  name="combined.html")
    violations, fired = check_multi_report(path, result)
    assert "multi_failures_surfaced" in fired
    assert any("DS2" in v and "not mentioned" in v for v in violations)


def test_the_multi_summary_is_checked_one_level_down(tmp_path):
    """A multi run returns a summary, not a test; the analyses live in `results`."""
    from fuzzing.oracles import check_result

    clean = _multi_result()
    assert check_result(clean) == []

    broken = _multi_result()
    broken["results"]["DS2"]["p_value"] = 1.7
    violations = check_result(broken)
    assert any(v.startswith("[DS2]") and "outside [0, 1]" in v for v in violations)
