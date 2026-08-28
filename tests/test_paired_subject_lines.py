"""Connecting subjects across levels: when it is allowed, and what it draws.

A paired test analyses within-subject differences. Two datasets can have
identical distributions at both levels and opposite paired results, so a plot
that shows only the boxes hides the thing that was tested. Lines put it back --
but only where they are defensible, which is a structural question, not taste.
"""

import pandas as pd
import pytest
from scipy import stats

from analysis.paired_lines import (PAIRED_LINE_MAX_SUBJECTS, build_paired_trajectories,
                                   paired_lines_supported)
from core.level_order import order_is_defined
from export.report_charts import _ChartsMixin

ORDER = ["Baseline", "Week 4", "Week 12"]
# The second block is stored in a different subject order, which is legal input
# and exactly what a positional pairing gets wrong.
SUBJECTS = {"Baseline": ["S1", "S2", "S3"],
            "Week 4": ["S3", "S1", "S2"],
            "Week 12": ["S1", "S2", "S3"]}
RAW = {"Baseline": [10, 20, 30], "Week 4": [33, 11, 22], "Week 12": [12, 23, 34]}


def test_a_trajectory_follows_the_subject_not_the_position():
    supported, reason = paired_lines_supported(ORDER, SUBJECTS)
    assert supported, reason
    paths = {t["subject"]: [p["value"] for p in t["points"]]
             for t in build_paired_trajectories(ORDER, RAW, SUBJECTS)}
    assert paths == {"S1": [10.0, 11.0, 12.0],
                     "S2": [20.0, 22.0, 23.0],
                     "S3": [30.0, 33.0, 34.0]}


def test_lines_show_a_difference_the_distributions_cannot():
    """The reason the feature exists, as a measurement rather than a claim.

    Both datasets hold the same values at baseline and the same values after,
    so every box, violin and summary statistic is identical. Only the pairing
    differs -- and with it the result of the test.
    """
    base = [10 + i for i in range(12)]
    after = [b + 5 for b in base]

    uniform = (base, after)                  # every subject moves the same way
    reversal = (base, list(reversed(after)))  # same collectives, mixed direction

    assert sorted(uniform[0]) == sorted(reversal[0])
    assert sorted(uniform[1]) == sorted(reversal[1])

    p_uniform = stats.ttest_rel(uniform[1], uniform[0]).pvalue
    p_reversal = stats.ttest_rel(reversal[1], reversal[0]).pvalue
    assert p_uniform < p_reversal, "identical boxes, different paired results"

    order = ["Baseline", "Post"]
    subjects = {g: [f"S{i}" for i in range(12)] for g in order}
    directions = {}
    for name, (before, post) in (("uniform", uniform), ("reversal", reversal)):
        trajectories = build_paired_trajectories(
            order, {"Baseline": before, "Post": post}, subjects)
        rising = sum(1 for t in trajectories
                     if t["points"][1]["value"] > t["points"][0]["value"])
        directions[name] = (rising, len(trajectories) - rising)

    assert directions["uniform"] == (12, 0)
    assert directions["reversal"][1] > 0, "the falling subjects must be visible"


def test_an_independent_design_is_refused():
    supported, reason = paired_lines_supported(["A", "B"], {})
    assert not supported
    assert "subject" in reason.lower()


CELLS = ["Arm=A, Time=T0", "Arm=A, Time=T1", "Arm=B, Time=T0", "Arm=B, Time=T1"]
CELL_SUBJECTS = {CELLS[0]: ["S1", "S2"], CELLS[1]: ["S1", "S2"],
                 CELLS[2]: ["S3", "S4"], CELLS[3]: ["S3", "S4"]}


def test_a_mixed_design_draws_lines_inside_each_between_group():
    """A subject never changes its between group, so no line can cross one.

    Each between group's cells are a block, and inside a block the only thing
    that moves is the within factor -- structurally the repeated-measures case,
    drawn once per block. The earlier blanket refusal of two-factor axes was
    more conservative than the risk it was guarding against.
    """
    supported, reason = paired_lines_supported(CELLS, CELL_SUBJECTS)
    assert supported, reason

    trajectories = build_paired_trajectories(
        CELLS, {level: [1.0, 2.0] for level in CELLS}, CELL_SUBJECTS)
    assert len(trajectories) == 4
    for trajectory in trajectories:
        groups = [point["group"] for point in trajectory["points"]]
        arms = {group.split(",")[0] for group in groups}
        assert len(arms) == 1, f"{trajectory['subject']} crossed between groups: {groups}"
        assert len(groups) == 2


def test_the_between_factors_own_order_does_not_gate_the_lines():
    """Aachen before Bonn is a question about the axis, not about the path."""
    cells = ["Site=Aachen, Time=T0", "Site=Aachen, Time=T1",
             "Site=Bonn, Time=T0", "Site=Bonn, Time=T1"]
    subjects = {cells[0]: ["S1"], cells[1]: ["S1"],
                cells[2]: ["S2"], cells[3]: ["S2"]}
    # The full labels are an alphabetical guess ...
    assert not order_is_defined(cells)[0]
    # ... and the lines are still fine, because no line runs along that axis.
    supported, reason = paired_lines_supported(cells, subjects)
    assert supported, reason


def test_an_unordered_within_factor_still_refuses():
    """The gate that does apply inside a block has to keep applying."""
    cells = ["Arm=A, Drug=DrugX", "Arm=A, Drug=DrugA",
             "Arm=B, Drug=DrugX", "Arm=B, Drug=DrugA"]
    subjects = {cells[0]: ["S1"], cells[1]: ["S1"],
                cells[2]: ["S2"], cells[3]: ["S2"]}
    supported, reason = paired_lines_supported(cells, subjects)
    assert not supported
    assert "alphabetical" in reason


def test_a_block_split_across_the_axis_is_refused():
    """A line may not reach across another group's bars to close itself."""
    interleaved = ["Arm=A, Time=T0", "Arm=B, Time=T0",
                   "Arm=A, Time=T1", "Arm=B, Time=T1"]
    subjects = {interleaved[0]: ["S1"], interleaved[2]: ["S1"],
                interleaved[1]: ["S2"], interleaved[3]: ["S2"]}
    supported, reason = paired_lines_supported(interleaved, subjects)
    assert not supported
    assert "next to each other" in reason


def test_a_single_factor_design_is_unaffected():
    prefixed = ["Time=T0", "Time=T1"]
    supported, reason = paired_lines_supported(
        prefixed, {"Time=T0": ["S1", "S2"], "Time=T1": ["S1", "S2"]})
    assert supported, reason


def test_an_alphabetical_level_order_is_refused():
    """A line asserts a path; between Drug A and Drug B there is none."""
    subjects = {"Drug A": ["S1", "S2"], "Drug B": ["S1", "S2"]}
    supported, reason = paired_lines_supported(["Drug A", "Drug B"], subjects)
    assert not supported
    assert "alphabetical" in reason
    # The refusal and the report's own note come from one computation.
    assert order_is_defined(["Drug A", "Drug B"])[1] in reason


def test_too_many_subjects_is_refused_with_the_count():
    subjects = {g: [f"S{i}" for i in range(PAIRED_LINE_MAX_SUBJECTS + 5)]
                for g in ("T0", "T1")}
    supported, reason = paired_lines_supported(["T0", "T1"], subjects)
    assert not supported
    assert str(PAIRED_LINE_MAX_SUBJECTS + 5) in reason
    assert str(PAIRED_LINE_MAX_SUBJECTS) in reason


def test_the_threshold_is_a_parameter_not_a_hardcoded_number():
    subjects = {g: [f"S{i}" for i in range(5)] for g in ("T0", "T1")}
    assert paired_lines_supported(["T0", "T1"], subjects, max_subjects=10)[0]
    assert not paired_lines_supported(["T0", "T1"], subjects, max_subjects=4)[0]


def test_a_subject_seen_at_one_level_only_is_dropped():
    order = ["T0", "T1"]
    subjects = {"T0": ["S1", "S2"], "T1": ["S1"]}
    raw = {"T0": [1.0, 2.0], "T1": [3.0]}
    trajectories = build_paired_trajectories(order, raw, subjects)
    assert [t["subject"] for t in trajectories] == ["S1"]


def test_the_static_chart_draws_and_explains():
    import plotly.graph_objects as go

    figure = go.Figure()
    for group in ORDER:
        figure.add_trace(go.Box(y=RAW[group], name=group))
    skipped = _ChartsMixin._build_paired_line_layer(
        figure, {"raw_data": RAW, "raw_data_subjects": SUBJECTS}, ORDER)
    assert skipped == ""
    line_trace = next(t for t in figure.data if t.name == "Subject")
    # One trace with gaps between subjects rather than one trace per subject.
    assert list(line_trace.x).count(None) == 3

    bare = go.Figure()
    for group in ORDER:
        bare.add_trace(go.Box(y=RAW[group], name=group))
    reason = _ChartsMixin._build_paired_line_layer(bare, {"raw_data": RAW}, ORDER)
    assert reason, "a refusal must come with its reason, not silence"
    assert not any(t.name == "Subject" for t in bare.data)


@pytest.mark.parametrize("labels, defined", [
    (["6h", "24h", "48h"], True),
    (["Baseline", "Week 4", "Week 12"], True),
    (["Timepoint 1", "Timepoint 2", "Timepoint 3"], True),
    (["Timepoint=Pre", "Timepoint=Post"], True),
    (["Cond=Hypoxia", "Cond=Normoxia"], True),
    (["Drug A", "Drug B"], False),
    (["DrugX", "6h"], False),
])
def test_order_predicate_asks_which_chunk_decided(labels, defined):
    """Numbers and recognized reference terms order; the alphabet guesses.

    The earlier test asked whether a label was *entirely* numeric, which called
    "Week 4, Week 12" a guess although their numbers had ordered them, and could
    not see a level behind a "factor=level" prefix at all.
    """
    assert order_is_defined(labels)[0] is defined
