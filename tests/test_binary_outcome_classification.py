"""Guards the operator-precedence fix in the binary-outcome classifier found in the
Help Hub content audit. The intended semantics (per the original code's own
comment): exactly 2 values that are 0/1 (or two strings), AND the column name does
not hint at a grouping variable. Because `and` binds tighter than `or`, the
un-parenthesized original expression let a numeric 2-value column bypass the
grouping-name guard, and let an all-string column bypass the len==2 guard entirely.
"""
from autopilot.statistical_analyzer_autopilot_pipeline import _classify_binary_outcome


def test_numeric_two_value_grouping_named_column_is_not_binary():
    # Regression case: a numeric 2-value column named like a grouping variable
    # (e.g. "Treatment_Group" coded 1/2) must NOT be classified as binary, even
    # though it has exactly 2 values. Before the fix, the numeric branch of the
    # buggy `or`-split expression bypassed the "not grouping" guard entirely.
    assert _classify_binary_outcome([1, 2], "Treatment_Group") is False


def test_string_column_with_more_than_two_values_is_not_binary():
    # Regression case: an all-string column with more than 2 unique values must
    # NOT be classified as binary. Before the fix, the string branch of the buggy
    # expression had no len==2 gate at all.
    assert _classify_binary_outcome(["Low", "Medium", "High"], "Outcome") is False


def test_numeric_01_column_is_binary():
    assert _classify_binary_outcome([0, 1], "Died") is True


def test_numeric_two_value_non_01_column_is_binary_if_not_grouping_named():
    # A numeric 2-value column that ISN'T 0/1 and ISN'T grouping-named should still
    # be rejected per the stated "0/1 (or two strings)" contract -- only is_01 or
    # is_str values count, not any 2-value numeric column.
    assert _classify_binary_outcome([5, 12], "ScoreCode") is False


def test_yes_no_string_column_is_binary():
    assert _classify_binary_outcome(["Yes", "No"], "Survived") is True


def test_yes_no_grouping_named_column_is_not_binary():
    assert _classify_binary_outcome(["Yes", "No"], "Treatment_Arm") is False
