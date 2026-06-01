"""Isolated unit tests for the pure formatting / numeric helpers extracted into
``export.report_formatting._FormattingMixin`` (Phase 1 of the html_exporter
split). These are deterministic pure functions, so they are tested directly on
the mixin — no HTMLExporter, no Qt, no I/O.
"""
import math

import numpy as np
import pandas as pd
import pytest

from export.report_formatting import _FormattingMixin as F


# --------------------------------------------------------------------------- #
# _format_metric
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value,expected", [
    (3.14159, "3.1416"),
    (0.0, "0.0000"),
    (-2.5, "-2.5000"),
    (1500, "1.500e+03"),       # |x| >= 1000 -> scientific
    (0.0001, "1.000e-04"),     # |x| < 0.001 -> scientific
    (None, "N/A"),
    (float("nan"), "N/A"),
    (float("inf"), "Infinity"),
    (float("-inf"), "-Infinity"),
    ("already a string", "already a string"),
])
def test_format_metric_scalar(value, expected):
    assert F._format_metric(value) == expected


def test_format_metric_digits_arg():
    assert F._format_metric(3.14159, digits=2) == "3.14"


def test_format_metric_list_joins():
    assert F._format_metric([1.0, 2.0]) == "1.0000, 2.0000"


# --------------------------------------------------------------------------- #
# _sci_notation
# --------------------------------------------------------------------------- #
def test_sci_notation_negative_exponent():
    assert F._sci_notation(0.0005) == "5.00 × 10⁻⁴"


def test_sci_notation_positive_exponent():
    assert F._sci_notation(12345.0) == "1.23 × 10⁴"


# --------------------------------------------------------------------------- #
# _format_p_value
# --------------------------------------------------------------------------- #
def test_format_p_value_highly_significant():
    out = F._format_p_value(0.0005)
    assert out.startswith("p < 0.001")
    assert out.endswith("***")
    assert "× 10" in out  # sci-notation of the exact value embedded


@pytest.mark.parametrize("value,stars", [
    (0.005, "**"),
    (0.03, "*"),
    (0.2, "ns"),
])
def test_format_p_value_star_tiers(value, stars):
    out = F._format_p_value(value)
    assert out == f"p = {value:.3f} {stars}"


@pytest.mark.parametrize("value", [None, "", "N/A", float("nan")])
def test_format_p_value_missing(value):
    assert F._format_p_value(value) == "N/A"


def test_format_p_value_numpy_scalar():
    assert F._format_p_value(np.float64(0.03)) == "p = 0.030 *"


# --------------------------------------------------------------------------- #
# _format_confidence_interval
# --------------------------------------------------------------------------- #
def test_format_confidence_interval_pair():
    assert F._format_confidence_interval([1.2, 3.4]) == "[1.2000, 3.4000]"


def test_format_confidence_interval_scalar_falls_back_to_metric():
    assert F._format_confidence_interval(2.5) == "2.5000"


# --------------------------------------------------------------------------- #
# _prettify_label / _bool_label / _bool_class
# --------------------------------------------------------------------------- #
def test_prettify_label():
    assert F._prettify_label("mean_diff-value") == "Mean Diff Value"


@pytest.mark.parametrize("value,expected", [(True, "Passed"), (False, "Flagged"), (None, "Not available")])
def test_bool_label(value, expected):
    assert F._bool_label(value) == expected


@pytest.mark.parametrize("value,expected", [
    (True, "is-significant"), (False, "is-danger"), (None, "is-neutral"),
])
def test_bool_class(value, expected):
    assert F._bool_class(value) == expected


# --------------------------------------------------------------------------- #
# _p_heat_style
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("p,expected", [
    (0.0005, "background:rgba(31,122,90,.22)"),
    (0.005, "background:rgba(31,122,90,.13)"),
    (0.03, "background:rgba(183,121,31,.13)"),
    (0.08, "background:rgba(159,58,56,.08)"),
    (0.5, ""),
    (float("nan"), ""),
    ("not a number", ""),
])
def test_p_heat_style(p, expected):
    assert F._p_heat_style(p) == expected


# --------------------------------------------------------------------------- #
# _has_display_value
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value,expected", [
    (None, False),
    ("", False),
    ("x", True),
    ([], False),
    ([1], True),
    ({}, False),
    ({"a": 1}, True),
    (np.array([]), False),
    (np.array([1, 2]), True),
    (0, True),       # a present scalar, even falsy, counts as displayable
])
def test_has_display_value(value, expected):
    assert F._has_display_value(value) is expected


# --------------------------------------------------------------------------- #
# _normalize_for_json
# --------------------------------------------------------------------------- #
def test_normalize_for_json_float_specials():
    assert F._normalize_for_json(float("nan")) is None
    assert F._normalize_for_json(float("inf")) == "Infinity"
    assert F._normalize_for_json(float("-inf")) == "-Infinity"
    assert F._normalize_for_json(1.5) == 1.5


def test_normalize_for_json_numpy_and_path():
    assert F._normalize_for_json(np.float64(2.0)) == 2.0
    assert F._normalize_for_json(np.array([1, 2])) == [1, 2]
    assert F._normalize_for_json(pd.Timestamp("2024-01-02")) == "2024-01-02T00:00:00"


def test_normalize_for_json_nested_dict_keys_stringified():
    out = F._normalize_for_json({1: [float("nan"), 2.0]})
    assert out == {"1": [None, 2.0]}


def test_normalize_for_json_dataframe():
    df = pd.DataFrame({"a": [1, np.nan], "b": [3, 4]})
    out = F._normalize_for_json(df)
    assert out["columns"] == ["a", "b"]
    assert out["data"] == [[1.0, 3], [None, 4]]


# --------------------------------------------------------------------------- #
# _coerce_numeric_sequence / _downsample_for_display / _summarize_numeric_group
# --------------------------------------------------------------------------- #
def test_coerce_numeric_sequence_filters_non_finite():
    assert F._coerce_numeric_sequence([1, "2", "x", float("nan"), float("inf"), 3.5]) == [1.0, 2.0, 3.5]
    assert F._coerce_numeric_sequence(None) == []


def test_downsample_passthrough_when_small():
    values = [1.0, 2.0, 3.0]
    assert F._downsample_for_display(values, max_points=10) is values


def test_downsample_is_deterministic_and_capped():
    values = list(range(10_000))
    a = F._downsample_for_display(values, max_points=100)
    b = F._downsample_for_display(values, max_points=100)
    assert len(a) == 100
    assert a == b                      # seeded RNG -> reproducible
    assert a == sorted(a)              # indices kept in order


def test_summarize_numeric_group_basic():
    out = F._summarize_numeric_group([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    assert out["n"] == 8
    assert out["mean"] == pytest.approx(5.0)
    assert out["median"] == pytest.approx(4.5)
    assert out["min"] == 2.0 and out["max"] == 9.0
    assert out["ci95_lower"] < out["mean"] < out["ci95_upper"]


def test_summarize_numeric_group_empty():
    assert F._summarize_numeric_group([]) == {}
