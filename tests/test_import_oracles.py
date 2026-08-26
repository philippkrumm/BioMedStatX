"""The import oracles must be able to fail.

Same standard as the report oracles: a check that has never been seen to fail is
indistinguishable from one that cannot, and a fuzz run made of such checks
reports coverage it does not have. Each test below takes a state the app could
genuinely be in, breaks exactly one thing, and demands that the matching oracle
-- named, not merely "something" -- says so.

The states here are synthetic so the suite stays fast. That they match the real
window is established separately by the fuzzer, which builds these same states
from an actual ``StatisticalAnalyzerApp`` after opening an actual file, and by
the mutation run recorded in ``fuzzing/README.md``, where six deliberate breaks
in real ``src/`` code were each caught by the oracle named here.
"""

import pytest

from fuzzing.import_generators import ImportCase
from fuzzing.import_oracles import ORACLES, check_import

LEVELS = ["Ctrl"] * 3 + ["Treated"] * 3
VALUES = [10.0, 11.0, 12.0, 14.0, 15.0, 16.0]


def _case(**overrides):
    base = dict(
        seed=1, file_path="/tmp/x.csv", file_format="csv", mutations=[],
        dv_name="Value", factor_name="Group",
        levels=list(LEVELS), values=list(VALUES),
        n_rows=len(VALUES), n_cols=2,
        declared_format={"sep": ",", "decimal": ".", "thousands": None},
        written_format={"sep": ",", "decimal": ".", "thousands": None},
        extra={"header_row": 0},
    )
    base.update(overrides)
    return ImportCase(**base)


def _state(**overrides):
    base = dict(
        loaded=True,
        columns=["Group", "Value"],
        dtypes={"Group": "object", "Value": "float64"},
        n_rows=len(VALUES),
        dv_is_numeric=True,
        dv_values=list(VALUES),
        factor_levels=list(LEVELS),
        wide_pivoted=False,
        dv_bucket=["Value"],
        factor1_bucket=["Group"],
        factor2_bucket=[],
        subject_bucket=[],
        mapping_feedback="Independent t-test will run.",
        start_enabled=True,
        messages=[],
    )
    base.update(overrides)
    return base


def _names(violations):
    return {v.split("]")[0].lstrip("[") for v in violations}


def test_a_clean_import_violates_nothing_and_checks_a_lot():
    violations, fired = check_import(_state(), _case())

    assert violations == []
    # If this ever thins out, the suite is passing on fewer checks than it looks.
    assert set(fired) >= {
        "file_loaded", "shape_survives", "dv_is_numeric", "values_survive",
        "levels_survive", "no_phantom_levels", "dv_reaches_bucket",
        "factor_reaches_bucket", "measurement_is_not_a_subject",
    }


def test_a_file_that_never_loaded_is_reported():
    violations, fired = check_import(_state(loaded=False, n_rows=0), _case())

    assert "file_loaded" in _names(violations)


def test_a_silent_load_failure_is_reported():
    """Not loading is forgivable. Not saying so is not."""
    case = _case(extra={"header_row": 2})       # app may legitimately refuse
    violations, fired = check_import(_state(loaded=False, messages=[]), case)

    assert "load_failure_is_announced" in fired
    assert "load_failure_is_announced" in _names(violations)


def test_an_announced_load_failure_passes():
    case = _case(extra={"header_row": 2})
    violations, fired = check_import(
        _state(loaded=False, messages=[{"kind": "critical", "text": "Error loading file"}]),
        case)

    assert "load_failure_is_announced" in fired
    assert violations == []


def test_lost_rows_are_reported():
    violations, _ = check_import(
        _state(n_rows=3, dv_values=VALUES[:3], factor_levels=LEVELS[:3]), _case())

    assert "shape_survives" in _names(violations)


def test_a_column_of_numbers_arriving_as_text_is_reported():
    violations, _ = check_import(
        _state(dv_is_numeric=False, dtypes={"Group": "object", "Value": "object"},
               dv_values=None), _case())

    assert "dv_is_numeric" in _names(violations)


def test_values_changed_by_a_factor_of_a_thousand_are_reported():
    """The dangerous shape: the right count of plausible, wrong numbers.

    A thousands separator read as a decimal point does not fail, does not warn,
    and does not look wrong -- it just divides everything by 1000.
    """
    violations, _ = check_import(
        _state(dv_values=[v / 1000 for v in VALUES]), _case())

    assert "values_survive" in _names(violations)


def test_a_blank_label_turning_into_a_group_is_reported():
    case = _case(levels=["Ctrl", "", "", "Treated", "", ""])
    violations, _ = check_import(
        _state(factor_levels=["Ctrl", "nan", "nan", "Treated", "nan", "nan"]), case)

    assert "no_phantom_levels" in _names(violations)


def test_a_blank_label_left_blank_is_accepted():
    """Merged cells really do leave blanks; the app cannot invent the label and
    must not be blamed for the gap -- only for filling it in."""
    case = _case(levels=["Ctrl", "", "", "Treated", "", ""])
    violations, fired = check_import(
        _state(factor_levels=["Ctrl", "", "", "Treated", "", ""]), case)

    assert "no_phantom_levels" in fired
    assert violations == []


def test_changed_group_labels_are_reported():
    violations, _ = check_import(
        _state(factor_levels=["Ctrl", "Ctrl", "Ctrl", "TREATED", "TREATED", "TREATED"]),
        _case())

    assert "levels_survive" in _names(violations)


def test_a_measurement_column_that_misses_its_bucket_is_reported():
    violations, _ = check_import(_state(dv_bucket=[]), _case())

    assert "dv_reaches_bucket" in _names(violations)


def test_a_group_column_that_misses_factor_one_is_reported():
    violations, _ = check_import(_state(factor1_bucket=[]), _case())

    assert "factor_reaches_bucket" in _names(violations)


def test_a_measurement_filed_as_a_subject_is_reported():
    violations, _ = check_import(
        _state(dv_bucket=[], subject_bucket=["Value"]), _case())

    assert "measurement_is_not_a_subject" in _names(violations)


# --- files the app is only expected to refuse visibly -----------------------------


def _unreadable_case():
    """Notes above the header: the app may misread it, but must not pretend."""
    return _case(mutations=["notes_above_header"], extra={"header_row": 2})


def test_a_misread_file_presented_as_ready_is_reported():
    violations, fired = check_import(
        _state(dv_values=[v / 1000 for v in VALUES],
               mapping_feedback="Independent t-test will run.",
               start_enabled=True),
        _unreadable_case())

    assert "broken_import_fails_visibly" in fired
    assert "broken_import_fails_visibly" in _names(violations)


def test_a_misread_file_that_disables_the_button_passes():
    violations, fired = check_import(
        _state(dv_values=[v / 1000 for v in VALUES], start_enabled=False),
        _unreadable_case())

    assert "broken_import_fails_visibly" in fired
    assert violations == []


def test_a_misread_file_that_names_the_problem_passes():
    violations, fired = check_import(
        _state(dv_is_numeric=False, dv_values=None, dv_bucket=[],
               mapping_feedback="Assign at least one measurement column.",
               start_enabled=True),
        _unreadable_case())

    assert "broken_import_fails_visibly" in fired
    assert violations == []


def test_an_empty_bucket_alone_does_not_count_as_refusing():
    """The hole a mutant walked through: nothing assigned, yet the app says it
    is ready and lights the button. That is the contradiction, not the refusal."""
    violations, _ = check_import(
        _state(dv_is_numeric=False, dv_values=None, dv_bucket=[],
               mapping_feedback="Ready.", start_enabled=True),
        _unreadable_case())

    assert "broken_import_fails_visibly" in _names(violations)


def test_a_file_the_app_may_refuse_is_not_held_to_the_faithful_checks():
    """Preconditions must actually gate: none of the value-level oracles may
    fire on a file the app was never expected to parse."""
    _, fired = check_import(_state(loaded=True), _unreadable_case())

    for name in ("file_loaded", "shape_survives", "dv_is_numeric",
                 "values_survive", "levels_survive", "dv_reaches_bucket"):
        assert name not in fired, f"{name} fired on a file the app may refuse"


@pytest.mark.parametrize("name", [name for name, _ in ORACLES])
def test_every_oracle_is_reachable(name):
    """A named oracle nobody can make fire is decoration."""
    clean_fired = set(check_import(_state(), _case())[1])
    refuse_fired = set(check_import(_state(loaded=False), _unreadable_case())[1])

    assert name in (clean_fired | refuse_fired)
