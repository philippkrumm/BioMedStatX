"""Regression tests for SheetSelectionDialog range geometry.

Covers Blind Spot #5: connected-components BFS split and overlap detection.
Pure-function level — no QApplication needed.
"""
import os
import sys
from pathlib import Path

import pytest

# Headless Qt — module top-level imports PyQt5.QtWidgets.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from autopilot.statistical_analyzer_autopilot_ui import (  # noqa: E402
    _FakeIdx,
    _cells_in_ranges,
    _selected_indexes_to_ranges,
    _to_display_coords,
    extract_from_coordinates,
)


def _norm(ranges):
    """Sort ranges canonically: by (rows[0], cols[0])."""
    return sorted(
        ({"rows": tuple(r["rows"]), "cols": tuple(r["cols"])} for r in ranges),
        key=lambda r: (r["rows"][0], r["cols"][0]),
    )


# ---------------------------------------------------------------------------
# _selected_indexes_to_ranges
# ---------------------------------------------------------------------------

def test_empty_input_returns_empty():
    assert _selected_indexes_to_ranges([]) == []


def test_single_rectangular_block_collapses_to_one_range():
    idxs = [_FakeIdx(r, c) for r in range(2, 6) for c in range(1, 4)]
    ranges = _selected_indexes_to_ranges(idxs)
    assert ranges == [{"rows": (2, 5), "cols": (1, 3)}]


def test_non_contiguous_columns_split_into_two_ranges():
    """Col B (idx 1) rows 2-6 + Col D (idx 3) rows 2-6 with gap on Col C."""
    cells = [(r, 1) for r in range(1, 6)] + [(r, 3) for r in range(1, 6)]
    idxs = [_FakeIdx(r, c) for r, c in cells]
    ranges = _norm(_selected_indexes_to_ranges(idxs))
    assert ranges == _norm([
        {"rows": (1, 5), "cols": (1, 1)},
        {"rows": (1, 5), "cols": (3, 3)},
    ])


def test_diagonal_cells_do_not_merge():
    """4-connectivity: diagonal neighbours are NOT adjacent."""
    idxs = [_FakeIdx(2, 2), _FakeIdx(3, 3)]
    ranges = _norm(_selected_indexes_to_ranges(idxs))
    assert ranges == _norm([
        {"rows": (2, 2), "cols": (2, 2)},
        {"rows": (3, 3), "cols": (3, 3)},
    ])


def test_l_shape_is_single_component():
    """L: (0,0),(1,0),(2,0),(2,1),(2,2) — connected via shared edges."""
    cells = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
    idxs = [_FakeIdx(r, c) for r, c in cells]
    ranges = _selected_indexes_to_ranges(idxs)
    assert ranges == [{"rows": (0, 2), "cols": (0, 2)}]


def test_three_disconnected_singletons():
    idxs = [_FakeIdx(0, 0), _FakeIdx(5, 5), _FakeIdx(10, 10)]
    ranges = _norm(_selected_indexes_to_ranges(idxs))
    assert ranges == _norm([
        {"rows": (0, 0), "cols": (0, 0)},
        {"rows": (5, 5), "cols": (5, 5)},
        {"rows": (10, 10), "cols": (10, 10)},
    ])


# ---------------------------------------------------------------------------
# _cells_in_ranges
# ---------------------------------------------------------------------------

def test_cells_in_ranges_inclusive_bounds():
    cells = _cells_in_ranges([{"rows": (1, 2), "cols": (3, 4)}])
    assert cells == {(1, 3), (1, 4), (2, 3), (2, 4)}


def test_cells_in_ranges_union_across_blocks():
    cells = _cells_in_ranges([
        {"rows": (0, 0), "cols": (0, 0)},
        {"rows": (2, 2), "cols": (2, 2)},
    ])
    assert cells == {(0, 0), (2, 2)}


# ---------------------------------------------------------------------------
# Overlap detection — pure-function emulation of _assign_selection logic
# ---------------------------------------------------------------------------

def test_overlap_detected_between_groups():
    group_a = [{"rows": (0, 4), "cols": (1, 1)}]
    new_b = [{"rows": (3, 7), "cols": (1, 1)}]
    conflict = _cells_in_ranges(group_a) & _cells_in_ranges(new_b)
    assert conflict == {(3, 1), (4, 1)}


def test_no_overlap_when_columns_disjoint():
    group_a = [{"rows": (0, 9), "cols": (1, 1)}]
    new_b = [{"rows": (0, 9), "cols": (3, 3)}]
    conflict = _cells_in_ranges(group_a) & _cells_in_ranges(new_b)
    assert conflict == set()


def test_remove_cells_via_bfs_resplits_ranges():
    """Removing middle cells from a column range must split it into two ranges."""
    original_cells = {(r, 1) for r in range(0, 10)}
    to_remove = {(4, 1), (5, 1)}
    kept = original_cells - to_remove
    idxs = [_FakeIdx(r, c) for r, c in kept]
    ranges = _norm(_selected_indexes_to_ranges(idxs))
    assert ranges == _norm([
        {"rows": (0, 3), "cols": (1, 1)},
        {"rows": (6, 9), "cols": (1, 1)},
    ])


# ---------------------------------------------------------------------------
# _to_display_coords Beyond AZ
# ---------------------------------------------------------------------------

def test_to_display_coords_standard_and_large():
    assert _to_display_coords(0, 0) == "Row 1, Col A"
    assert _to_display_coords(0, 25) == "Row 1, Col Z"
    assert _to_display_coords(0, 26) == "Row 1, Col AA"
    assert _to_display_coords(0, 51) == "Row 1, Col AZ"
    assert _to_display_coords(0, 52) == "Row 1, Col BA"
    assert _to_display_coords(4, 701) == "Row 5, Col ZZ"
    assert _to_display_coords(9, 702) == "Row 10, Col AAA"


# ---------------------------------------------------------------------------
# extract_from_coordinates Float Validation, NaN Handling, and Same-Group Merge
# ---------------------------------------------------------------------------

import pandas as pd
import numpy as np

def test_extract_from_coordinates_skips_empty_and_text_typos():
    df_raw = pd.DataFrame([
        ["1.2", "", "3.4"],
        ["abc", "5.6", "N/A"],
        [None, "7.8", "   "]
    ])
    selection_map = {
        "Group_A": [{"rows": (0, 2), "cols": (0, 2)}]
    }
    
    # Run extraction
    result_df, nan_report = extract_from_coordinates(df_raw, selection_map, replicate_type="biological")
    
    # Out of 9 cells: 
    # Valid numeric: 1.2, 3.4, 5.6, 7.8 (4 values)
    # Empty: "", None, "   " (3 cells)
    # Non-numeric: "abc", "N/A" (2 cells)
    # Total NaN should be 5
    assert nan_report["Group_A"] == 5
    assert len(result_df) == 4
    assert sorted(result_df["Value"].tolist()) == [1.2, 3.4, 5.6, 7.8]
    assert all(result_df["Group"] == "Group_A")


def test_extract_technical_replicates_two_groups_triplicates():
    """Verify standard lab pattern: WT and KO, each n=5 biological samples, measured in triplicates."""
    import numpy as np
    import pandas as pd

    # 5 rows x 6 cols: cols 0-2 = WT triplicates, cols 3-5 = KO triplicates
    grid = [
        ["1.0", "1.2", "1.4",  "2.0", "2.1", "2.2"],  # animal 1
        ["1.1", "1.3", "1.5",  "2.3", "2.2", "2.4"],  # animal 2
        ["1.0", "1.0", "1.0",  "3.0", "3.0", "3.0"],  # animal 3
        ["1.2", "1.4", "1.6",  "2.5", "2.5", "2.5"],  # animal 4
        ["1.5", "1.5", "1.5",  "2.8", "2.9", "3.0"],  # animal 5
    ]
    df_raw = pd.DataFrame(grid, dtype=str)
    selection_map = {
        "WT": [{"rows": (0, 4), "cols": (0, 2)}],
        "KO": [{"rows": (0, 4), "cols": (3, 5)}],
    }

    result_df, nan_report = extract_from_coordinates(
        df_raw, selection_map, replicate_type="technical", replicate_axis="row"
    )

    assert nan_report == {"WT": 0, "KO": 0}
    assert len(result_df) == 10
    wt_df = result_df[result_df["Group"] == "WT"]
    ko_df = result_df[result_df["Group"] == "KO"]

    assert len(wt_df) == 5
    assert len(ko_df) == 5
    assert (wt_df["n_replicates"] == 3.0).all()
    assert (ko_df["n_replicates"] == 3.0).all()

    # Animal 1 WT mean: (1.0 + 1.2 + 1.4) / 3 = 1.2
    assert wt_df["Value"].iloc[0] == pytest.approx(1.2)
    # Animal 3 WT mean: 1.0
    assert wt_df["Value"].iloc[2] == pytest.approx(1.0)
    # Animal 1 KO mean: (2.0 + 2.1 + 2.2) / 3 = 2.1
    assert ko_df["Value"].iloc[0] == pytest.approx(2.1)


def test_touching_range_merging_simulation():
    # Emulate the touching-range merging logic in _assign_selection
    existing_ranges = [{"rows": (0, 1), "cols": (0, 1)}]
    new_range = {"rows": (1, 2), "cols": (1, 2)}
    
    touching_ranges = []
    non_touching_ranges = []
    for r in existing_ranges:
        touch = not (new_range["rows"][1] < r["rows"][0] - 1 or new_range["rows"][0] > r["rows"][1] + 1 or
                     new_range["cols"][1] < r["cols"][0] - 1 or new_range["cols"][0] > r["cols"][1] + 1)
        if touch:
            touching_ranges.append(r)
        else:
            non_touching_ranges.append(r)
            
    assert len(touching_ranges) == 1
    
    combined_cells = _cells_in_ranges(touching_ranges + [new_range])
    fake_indexes = [_FakeIdx(r, c) for r, c in combined_cells]
    merged = _norm(_selected_indexes_to_ranges(fake_indexes))
    
    assert merged == [{"rows": (0, 2), "cols": (0, 2)}]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
