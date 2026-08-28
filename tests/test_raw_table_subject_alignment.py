"""A value and the subject printed beside it must come from the same row.

The raw-data table prints ``Group | Subject | Raw value`` side by side, and the
figure connects subjects across levels from the same two dicts. Both were filled
by different extractions:

* ``raw_data`` by the advanced pipeline, from the original frame;
* ``raw_data_subjects`` by the logged test wrapper, from the frame it analyses --
  which for a design with technical replicates has been averaged to one row per
  subject and level.

Two consequences, both measured before the fix. Without replicates the keys
themselves disagreed (``T0`` against ``Time=T0``), so the Subject column vanished
and the figure refused subject lines with "No subject was measured at more than
one level" -- false about every subject in the design. With replicates the keys
agreed and the *lengths* did not: 24 values per level were labelled from a list
of 8, and all 24 printed rows named the wrong subject.

The rule these tests hold to is not "the labels are present" but "the label
belongs to the value it stands next to".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Root conftest.py puts src/ on sys.path and forces headless Qt.
from analysis.paired_lines import build_paired_trajectories, paired_lines_supported
from statistical_testing.advanced_pipeline import _attach_raw_data, perform_advanced_test_pipeline

DV, SUBJECT, WITHIN, BETWEEN = "Val", "Subject", "Time", "Between"
LEVELS = ("T0", "T1", "T2")


def _rm_frame(n_subjects=8, replicates=1, shuffle=True):
    rng = np.random.default_rng(3)
    offsets = rng.normal(0, 1.0, size=n_subjects)
    rows = [
        {SUBJECT: f"S{s}", WITHIN: level,
         DV: 10.0 + 1.5 * index + offsets[s] + rng.normal(0, 0.3)}
        for s in range(n_subjects)
        for index, level in enumerate(LEVELS)
        for _ in range(replicates)
    ]
    frame = pd.DataFrame(rows)
    # Row order is the caller's, not ours. Shuffling is what breaks any pairing
    # that secretly relies on position.
    return frame.sample(frac=1.0, random_state=1).reset_index(drop=True) if shuffle else frame


def _run_rm(frame):
    samples = {level: frame[frame[WITHIN] == level][DV].tolist() for level in LEVELS}
    return perform_advanced_test_pipeline(
        frame, "repeated_measures_anova", DV, SUBJECT, between=None, within=[WITHIN],
        alpha=0.05, transformed_samples={k: list(v) for k, v in samples.items()},
        recommendation="parametric", force_parametric=True,
    )


def _ownership(frame):
    """Every value each subject really holds at each level."""
    return frame.groupby([SUBJECT, WITHIN])[DV].apply(list)


def _mispaired(result, frame):
    """Printed rows whose value does not belong to the subject beside it."""
    raw = result.get("raw_data") or {}
    subjects = result.get("raw_data_subjects") or {}
    owned = _ownership(frame)
    wrong, checked = 0, 0
    for level, labels in subjects.items():
        values = raw.get(level) or []
        for position, label in enumerate(labels):
            if position >= len(values):
                continue
            checked += 1
            mine = owned.get((label, level))
            if mine is None or not any(
                    np.isclose(float(values[position]), value, rtol=1e-9) for value in mine):
                wrong += 1
    return checked, wrong


def test_the_keys_of_both_halves_are_the_same_vocabulary():
    result = _run_rm(_rm_frame())
    raw, subjects = result.get("raw_data") or {}, result.get("raw_data_subjects") or {}
    assert subjects, "a repeated-measures design has subject identity to report"
    assert set(raw) == set(subjects), (raw.keys(), subjects.keys())


def test_every_printed_row_names_the_subject_the_value_came_from():
    frame = _rm_frame()
    checked, wrong = _mispaired(_run_rm(frame), frame)
    assert checked == len(frame), f"only {checked} of {len(frame)} rows were checked"
    assert wrong == 0, f"{wrong} of {checked} printed rows named the wrong subject"


def test_technical_replicates_do_not_shift_the_labels():
    """The wrapper averages replicates; the pipeline does not. One frame wins.

    Before the fix this was the worst case: keys matched, so nothing looked
    wrong, and every single printed row was mislabelled.
    """
    frame = _rm_frame(replicates=3)
    result = _run_rm(frame)
    raw, subjects = result.get("raw_data") or {}, result.get("raw_data_subjects") or {}
    for level in LEVELS:
        assert len(raw[level]) == len(subjects[level]), (
            f"{level}: {len(raw[level])} values labelled from {len(subjects[level])} names"
        )
    checked, wrong = _mispaired(result, frame)
    assert checked == len(frame)
    assert wrong == 0, f"{wrong} of {checked} printed rows named the wrong subject"


def test_subject_lines_are_offered_where_every_subject_spans_the_levels():
    """The refusal used to be a false statement about the data, not an absence."""
    result = _run_rm(_rm_frame())
    order = list((result.get("raw_data") or {}).keys())
    supported, reason = paired_lines_supported(order, result.get("raw_data_subjects") or {})
    assert supported, reason
    trajectories = build_paired_trajectories(
        order, result["raw_data"], result["raw_data_subjects"])
    assert len(trajectories) == 8
    assert all(len(t["points"]) == len(LEVELS) for t in trajectories)


def test_a_mixed_design_keeps_its_cell_vocabulary():
    rng = np.random.default_rng(5)
    offsets = rng.normal(0, 1.0, size=12)
    frame = pd.DataFrame([
        {SUBJECT: f"S{s}", WITHIN: level, BETWEEN: f"B{s % 2}",
         DV: 10.0 + 1.1 * index + 0.9 * (s % 2) + offsets[s] + rng.normal(0, 0.4)}
        for s in range(12) for index, level in enumerate(("T0", "T1"))
    ])
    cells = {f"{BETWEEN}={b}, {WITHIN}={w}":
             frame[(frame[BETWEEN] == b) & (frame[WITHIN] == w)][DV].tolist()
             for b in ("B0", "B1") for w in ("T0", "T1")}
    result = perform_advanced_test_pipeline(
        frame, "mixed_anova", DV, SUBJECT, between=[BETWEEN], within=[WITHIN],
        alpha=0.05, transformed_samples={k: list(v) for k, v in cells.items()},
        recommendation="parametric", force_parametric=True,
    )
    raw, subjects = result.get("raw_data") or {}, result.get("raw_data_subjects") or {}
    assert set(raw) == set(subjects) == set(cells)


@pytest.mark.parametrize("subjects,kept", [
    ({"A": ["s1", "s2"], "B": ["s1", "s2"]}, True),
    # Fewer labels than values -- the replicate-averaging case.
    ({"A": ["s1"], "B": ["s1"]}, False),
    # A key the values do not have.
    ({"A": ["s1", "s2"], "C": ["s1", "s2"]}, False),
    (None, False),
    ({}, False),
])
def test_labels_that_do_not_line_up_are_dropped_not_printed(subjects, kept):
    """An absent Subject column says nothing; a wrong one says something false."""
    result = {"raw_data_subjects": {"stale": ["x"]}}
    _attach_raw_data(result, {"A": [1.0, 2.0], "B": [3.0, 4.0]}, subjects)
    assert ("raw_data_subjects" in result) is kept
    if kept:
        assert result["raw_data_subjects"] == subjects
