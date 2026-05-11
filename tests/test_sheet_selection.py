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

from statistical_analyzer_autopilot_ui import (  # noqa: E402
    _FakeIdx,
    _cells_in_ranges,
    _selected_indexes_to_ranges,
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
