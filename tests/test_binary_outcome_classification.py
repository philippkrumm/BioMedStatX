"""Guards the binary-outcome classifier used for logistic-vs-correlation routing.

The classifier is tri-state:

  "binary"       -- unambiguous: exactly 2 values that are 0/1 (or booleans, or
                    two strings), non-grouping name. Routed to logistic, no prompt.
  "maybe_binary" -- exactly 2 numeric values NOT coded 0/1 (e.g. 1/2, 5/12),
                    non-grouping name. Ambiguous: the caller must confirm with the
                    user before treating it as binary. This closes the silent
                    footgun where a binary outcome coded 1/2 fell through to a
                    plausible-but-wrong Pearson correlation.
  "not_binary"   -- everything else (!=2 values, mixed types, grouping-named).

Also guards the original operator-precedence fix (Help Hub content audit): the
grouping-name guard must reject grouping-named columns, and the len==2 guard must
reject all-string columns with more than two values.
"""
import numpy as np

from autopilot.statistical_analyzer_autopilot_pipeline import _classify_binary_outcome


# --- unambiguous binary -----------------------------------------------------

def test_numeric_01_column_is_binary():
    assert _classify_binary_outcome([0, 1], "Died") == "binary"


def test_numpy_01_column_is_binary():
    # Real DataFrames yield numpy scalars, not python ints -- the 0/1 gate must
    # still recognise them (the template's Responder column is np.int64 0/1).
    assert _classify_binary_outcome([np.int64(0), np.int64(1)], "Responder") == "binary"


def test_bool_column_is_binary():
    assert _classify_binary_outcome([True, False], "Flag") == "binary"


def test_yes_no_string_column_is_binary():
    assert _classify_binary_outcome(["Yes", "No"], "Survived") == "binary"


# --- ambiguous (the footgun) ------------------------------------------------

def test_numeric_1_2_column_is_maybe_binary():
    # THE FOOTGUN: a binary outcome coded 1/2 (not 0/1). Old code returned False
    # and silently shipped a Pearson correlation; now it is flagged ambiguous so
    # the caller prompts instead of guessing.
    assert _classify_binary_outcome([1, 2], "Responder") == "maybe_binary"


def test_numpy_1_2_column_is_maybe_binary():
    assert _classify_binary_outcome([np.int64(1), np.int64(2)], "Responder") == "maybe_binary"


def test_float_two_value_column_is_maybe_binary():
    assert _classify_binary_outcome([2.0, 5.0], "Level") == "maybe_binary"


def test_numeric_two_value_non_01_column_is_maybe_binary_even_if_not_grouping_named():
    # A numeric 2-value column that isn't 0/1 and isn't grouping-named is now an
    # ambiguous candidate (was hard-rejected as "not binary" before the fix).
    assert _classify_binary_outcome([5, 12], "ScoreCode") == "maybe_binary"


# --- not binary -------------------------------------------------------------

def test_numeric_two_value_grouping_named_column_is_not_binary():
    # A numeric 2-value column named like a grouping variable (e.g.
    # "Treatment_Group" coded 1/2) must NOT be treated as an outcome -- the
    # grouping-name guard rejects it before the maybe_binary branch.
    assert _classify_binary_outcome([1, 2], "Treatment_Group") == "not_binary"


def test_string_column_with_more_than_two_values_is_not_binary():
    assert _classify_binary_outcome(["Low", "Medium", "High"], "Outcome") == "not_binary"


def test_yes_no_grouping_named_column_is_not_binary():
    assert _classify_binary_outcome(["Yes", "No"], "Treatment_Arm") == "not_binary"


def test_empty_unique_values_is_not_binary():
    assert _classify_binary_outcome([], "Outcome") == "not_binary"


def test_single_value_column_is_not_binary():
    assert _classify_binary_outcome([1], "Outcome") == "not_binary"


def test_mixed_type_unique_values_is_not_binary():
    # Neither all-numeric-01 nor all-string nor all-numeric: no category holds.
    assert _classify_binary_outcome([0, "No"], "Outcome") == "not_binary"


# --- Help Hub hint stays a conservative bool --------------------------------

class _FakeBucket:
    def __init__(self, columns):
        self._columns = columns

    def get_assigned_columns(self):
        return self._columns


class _FakeAppForHelpDetection:
    def __init__(self, df, dv_column):
        self.df = df
        self.dv_bucket = _FakeBucket([dv_column])


def test_help_hint_and_real_routing_agree_on_grouping_named_binary_column():
    import pandas as pd
    from autopilot.statistical_analyzer_autopilot_pipeline import _ap_is_binary_outcome_for_help

    df = pd.DataFrame({"Treatment_Arm": ["Yes", "No", "Yes", "No"]})
    fake_app = _FakeAppForHelpDetection(df, "Treatment_Arm")
    assert _ap_is_binary_outcome_for_help(fake_app) is False


def test_help_hint_still_true_for_genuine_binary_outcome():
    import pandas as pd
    from autopilot.statistical_analyzer_autopilot_pipeline import _ap_is_binary_outcome_for_help

    df = pd.DataFrame({"Survived": ["Yes", "No", "Yes", "No"]})
    fake_app = _FakeAppForHelpDetection(df, "Survived")
    assert _ap_is_binary_outcome_for_help(fake_app) is True


def test_help_hint_stays_false_for_ambiguous_1_2_outcome():
    # Help Hub is conservative: an ambiguous 1/2 outcome does NOT pre-emptively
    # suggest the logistic recipe; the confirmation prompt at routing time owns
    # that decision. Returned value must be a plain bool, not "maybe_binary".
    import pandas as pd
    from autopilot.statistical_analyzer_autopilot_pipeline import _ap_is_binary_outcome_for_help

    df = pd.DataFrame({"Responder": [1, 2, 1, 2]})
    fake_app = _FakeAppForHelpDetection(df, "Responder")
    result = _ap_is_binary_outcome_for_help(fake_app)
    assert result is False
