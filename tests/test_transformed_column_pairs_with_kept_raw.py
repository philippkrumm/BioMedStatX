"""The Transformed column must pair with the raw values that were KEPT.

Four writers populate ``raw_data_transformed`` -- the standard path, the
tester's own, and both branches of the advanced pipeline -- and the raw half is
CHOSEN between two extractions after all of them have run. So a write that
paired correctly at the time can be left unpaired by that choice, and guarding
the writers is not the same as guarding the page.

Measured: fuzz seed 51307 printed a Box-Cox column against a raw column of a
different length (9 raw values against 10 transformed in one cell) with the
standard path's writer already guarded -- the earlier fix reached one door of
four, and the pairing here was broken by the choice rather than by any write.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from statistical_testing.validators import (
    drop_unpaired_transformed,
    transformed_pairs_up,
)


def test_same_groups_and_lengths_pair_up():
    raw = {"A": [1.0, 2.0], "B": [3.0]}
    assert transformed_pairs_up(raw, {"A": [0.0, 0.3], "B": [0.47]})


def test_a_dropped_row_does_not_pair_up():
    """The exact shape of the defect: one column kept a NaN row, one dropped it."""
    raw = {"A": [1.0, 2.0, 3.0]}
    assert not transformed_pairs_up(raw, {"A": [0.0, 0.3, 0.47, 0.6]})


def test_a_group_missing_on_the_transformed_side_does_not_pair_up():
    assert not transformed_pairs_up({"A": [1.0], "B": [2.0]}, {"A": [0.0]})


def test_an_empty_raw_side_pairs_with_nothing():
    """Otherwise "no groups at all" would vacuously count as aligned."""
    assert not transformed_pairs_up({}, {"A": [1.0]})


def test_extra_groups_on_the_transformed_side_are_not_a_mismatch():
    """The table walks the raw side; a group it never prints cannot mispair.

    The advanced pipeline emits an empty entry for a cell that was never run,
    and refusing the whole column for that would drop a correct table.
    """
    raw = {"A": [1.0, 2.0]}
    assert transformed_pairs_up(raw, {"A": [0.0, 0.3], "B": []})


def test_a_stale_transformed_dict_is_dropped_from_the_result():
    """Written upstream against a raw dict that a later choice replaced."""
    results = {
        "raw_data": {"A": [1.0, 2.0, 3.0]},
        "raw_data_transformed": {"A": [0.0, 0.3, 0.47, 0.6]},
        "transformed_data": {"A": [0.0, 0.3, 0.47, 0.6]},
    }
    dropped = drop_unpaired_transformed(results)

    assert set(dropped) == {"raw_data_transformed", "transformed_data"}
    assert "raw_data_transformed" not in results
    # The second key matters on its own: the report falls back to it when the
    # first is absent, so dropping one alone leaves the column reaching the
    # page by the other door.
    assert "transformed_data" not in results
    assert results["raw_data"] == {"A": [1.0, 2.0, 3.0]}


def test_a_column_that_does_pair_is_left_alone():
    """The drop must not be the easy answer for every run."""
    results = {
        "raw_data": {"A": [1.0, 2.0]},
        "raw_data_transformed": {"A": [0.0, 0.301]},
    }
    assert drop_unpaired_transformed(results) == []
    assert results["raw_data_transformed"] == {"A": [0.0, 0.301]}


@pytest.mark.parametrize("results", [
    {},                                      # nothing to judge
    {"raw_data": {}},                        # no raw values at all
    {"raw_data": None, "raw_data_transformed": {"A": [1.0]}},
])
def test_nothing_to_pair_against_drops_nothing(results):
    """A missing raw half is not evidence that the transformed one is wrong."""
    before = dict(results)
    assert drop_unpaired_transformed(results) == []
    assert results == before
