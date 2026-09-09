"""Unit coverage for the interactive "Select Data Ranges" parsing core.

This is the same input-layer risk class as the Wave-4b import audit: the raw
cell/range selection sits BEFORE every downstream computation, so a wrong parse
here produces an analysis where every reported number is computed correctly on
the wrong input -- exactly the failure mode (silent, invisible in downstream
tests) that the group-label and CSV-locale bugs had. The range selector had no
test coverage; this file adds it.

Only the Qt-free, module-level core is exercised here (the dialog ships a
`_FakeIdx` QModelIndex surrogate specifically for this):
  _selected_indexes_to_ranges  -- selected cells -> connected-component boxes
  _cells_in_ranges             -- range boxes -> covered (row, col) set
  extract_from_coordinates     -- range boxes over a raw grid -> values + NaN report

The dialog's own `_row_count` / `_cell_count` are local closures over the same
range dicts, so their counting logic is the `_cells_in_ranges` cell set and the
box row span checked below. The interactive drag/assign flow itself is left to
the manual click-test.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

from autopilot.statistical_analyzer_autopilot_ui import (
    _FakeIdx,
    _selected_indexes_to_ranges,
    _cells_in_ranges,
    extract_from_coordinates,
)


def _boxset(ranges):
    """Order-independent view of the returned range dicts."""
    return {(r["rows"], r["cols"]) for r in ranges}


# ---------- _selected_indexes_to_ranges: connected components ----------

def test_single_rectangle_is_one_box():
    idx = [_FakeIdx(r, c) for r in (1, 2) for c in (1, 2)]
    assert _boxset(_selected_indexes_to_ranges(idx)) == {((1, 2), (1, 2))}


def test_two_disconnected_blocks_stay_separate():
    # Ctrl+click: cols 0-1 and cols 5-6 on row 0, gap at 2-4 -> two components.
    idx = [_FakeIdx(0, 0), _FakeIdx(0, 1), _FakeIdx(0, 5), _FakeIdx(0, 6)]
    assert _boxset(_selected_indexes_to_ranges(idx)) == {
        ((0, 0), (0, 1)),
        ((0, 0), (5, 6)),
    }


def test_L_shape_is_one_box_that_tolerates_the_gap():
    # Connected but non-rectangular (column 0 rows 0-2, plus (2,1),(2,2)).
    # One component; its bounding box spans rows 0-2, cols 0-2 and INCLUDES the
    # unselected corner cells -- the documented "gaps inside the box are
    # tolerated" behaviour (blanks coerce to NaN downstream).
    idx = [_FakeIdx(0, 0), _FakeIdx(1, 0), _FakeIdx(2, 0), _FakeIdx(2, 1), _FakeIdx(2, 2)]
    assert _boxset(_selected_indexes_to_ranges(idx)) == {((0, 2), (0, 2))}


def test_diagonal_only_cells_do_not_merge():
    # 4-connectivity: diagonal adjacency must NOT join cells (by design, since
    # plate layouts are row/column organised). Three lone cells -> three boxes.
    idx = [_FakeIdx(0, 0), _FakeIdx(1, 1), _FakeIdx(2, 2)]
    assert _boxset(_selected_indexes_to_ranges(idx)) == {
        ((0, 0), (0, 0)),
        ((1, 1), (1, 1)),
        ((2, 2), (2, 2)),
    }


def test_empty_selection_returns_no_ranges():
    assert _selected_indexes_to_ranges([]) == []


# ---------- _cells_in_ranges: cell counting vs hand calculation ----------

def test_cell_count_single_box():
    # rows 0-2 (3), cols 0-1 (2) -> 3*2 = 6 cells.
    cells = _cells_in_ranges([{"rows": (0, 2), "cols": (0, 1)}])
    assert len(cells) == 6
    assert (2, 1) in cells and (0, 0) in cells and (3, 0) not in cells


def test_cell_count_two_boxes_dedupes_overlap():
    # (0-1,0-1) = 4 cells, (1-2,1-2) = 4 cells, sharing (1,1) -> union is 7.
    cells = _cells_in_ranges([
        {"rows": (0, 1), "cols": (0, 1)},
        {"rows": (1, 2), "cols": (1, 2)},
    ])
    assert len(cells) == 7


# ---------- extract_from_coordinates: values + NaN report ----------

def _grid():
    # header=None, dtype=str raw grid. Blank at (1,2), non-numeric at (2,2).
    return pd.DataFrame([
        ["x", "1.0", "2.0", "x"],
        ["x", "3.0", "",    "x"],
        ["x", "4.0", "abc", "x"],
        ["x", "5.0", "6.0", "x"],
    ], dtype=str)


def test_biological_extraction_values_and_nan_report():
    sel = {"A": [{"rows": (0, 3), "cols": (1, 2)}]}  # 8 cells, row-major
    df, nan_report = extract_from_coordinates(_grid(), sel, replicate_type="biological")
    # blank + "abc" -> 2 NaN, counted, then dropped from the Value column.
    assert nan_report == {"A": 2}
    assert sorted(df["Value"].tolist()) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert (df["Group"] == "A").all()
    assert df["n_replicates"].isna().all()  # biological rows carry NaN reps


def test_technical_extraction_is_block_mean():
    sel = {"A": [{"rows": (0, 3), "cols": (1, 2)}]}
    df, nan_report = extract_from_coordinates(_grid(), sel, replicate_type="technical")
    assert nan_report == {"A": 2}
    assert len(df) == 1
    # nanmean of the 6 valid values = 21/6 = 3.5, over n_valid = 6 replicates.
    assert df["Value"].iloc[0] == pytest.approx(3.5)
    assert df["n_replicates"].iloc[0] == 6.0


def test_disconnected_selection_round_trips_into_two_group_ranges():
    # Two Ctrl+click blocks for one group extract as two source ranges.
    idx = [_FakeIdx(0, 1), _FakeIdx(3, 1)]  # single cells, rows 0 and 3, col 1
    ranges = _selected_indexes_to_ranges(idx)
    assert _boxset(ranges) == {((0, 0), (1, 1)), ((3, 3), (1, 1))}
    df, nan_report = extract_from_coordinates(_grid(), {"A": ranges},
                                              replicate_type="biological")
    assert sorted(df["Value"].tolist()) == [1.0, 5.0]  # (0,1)="1.0", (3,1)="5.0"
    assert nan_report == {"A": 0}
