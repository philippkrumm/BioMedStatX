"""A reordering is not a transformation.

Found by the report self-check on fuzz seed 116 (repeated-measures, no
mutation, no transformation selected): the report showed a "Transformed value"
column, a transformed-scale means note and two "After transformation"
diagnostic charts, while its own badge correctly read "Transformation: None".

The cause was not the display. ``transformed_samples`` held the same fourteen
numbers per level as ``filtered_samples``, in a different order -- and the gate
that decides whether anything was transformed compared position by position, so
a permutation read as a change. The 2026-08 hardening replaced a label check
with a value check; this is the same gate failing for the remaining reason,
because "the values at these positions differ" is not what "a value was
altered" means.

Order-insensitivity cannot hide a real transformation: a transformation that
leaves the multiset intact has not altered any value, which is exactly the case
the gate exists to suppress.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from statistical_testing.validators import grouped_samples_changed  # noqa: E402

# The shape of the real finding: a block of values moved to the end.
_RAW = [1.1665, 5.412, -1.0379, 2.6836, -0.1912, 3.46, 0.0769, 0.6693]
_ROTATED = [1.1665, 5.412, 0.0769, 0.6693, -1.0379, 2.6836, -0.1912, 3.46]


def test_a_permutation_is_not_a_transformation():
    """The finding itself, reduced to the gate."""
    assert grouped_samples_changed({"T0": _RAW}, {"T0": _ROTATED}) is False


def test_a_permutation_in_one_level_of_several_is_not_a_transformation():
    """The real case had two levels; only reporting per level would still leak."""
    raw = {"T0": _RAW, "T1": [2.7649, 9.0936, 2.2687, 5.144]}
    permuted = {"T0": _ROTATED, "T1": [5.144, 2.2687, 9.0936, 2.7649]}

    assert grouped_samples_changed(raw, permuted) is False


def test_an_actual_transformation_is_still_a_change():
    """The gate must not become one that never fires."""
    raw = {"T0": [1.0, 10.0, 100.0]}
    logged = {"T0": [0.0, 1.0, 2.0]}

    assert grouped_samples_changed(raw, logged) is True


def test_one_altered_value_among_many_is_still_a_change():
    """Sorting must not swallow a single moved value."""
    raw = {"T0": list(_RAW)}
    nudged = {"T0": list(_RAW)}
    nudged["T0"][3] = 99.0

    assert grouped_samples_changed(raw, nudged) is True


def test_a_different_length_is_still_a_change():
    assert grouped_samples_changed({"T0": _RAW}, {"T0": _RAW[:-1]}) is True


def test_an_identity_copy_with_nan_is_still_unchanged():
    """NaN sorts to the end on both sides; the pre-existing guarantee holds."""
    values = [1.0, float("nan"), 3.0]

    assert grouped_samples_changed({"T0": values}, {"T0": list(values)}) is False
    assert grouped_samples_changed({"T0": values}, {"T0": [3.0, float("nan"), 1.0]}) is False


def test_a_nan_appearing_where_there_was_none_is_a_change():
    """A transformation that puts a value out of domain has altered it."""
    assert grouped_samples_changed(
        {"T0": [1.0, 2.0, 3.0]}, {"T0": [1.0, float("nan"), 3.0]}) is True


def test_ndarray_containers_behave_like_lists():
    """The numpy path is the one the pipeline actually takes."""
    assert grouped_samples_changed(
        {"T0": np.array(_RAW)}, {"T0": np.array(_ROTATED)}) is False
    assert grouped_samples_changed(
        {"T0": np.array([1.0, 10.0])}, {"T0": np.array([0.0, 1.0])}) is True
