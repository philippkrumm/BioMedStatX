"""OutlierDetector._convert_values_to_float unconditionally applies German decimal-format
conversion (dot=thousands, comma=decimal) whenever the value column isn't already numeric
dtype - silently multiplying plain US/international-formatted decimal strings like "1.5" by
10-1000x ("1.5" -> 15.0). Fix: try a direct float() parse first; only fall back to the
German-format substitution for values that fail it.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import pytest

from analysis.outlier_core import OutlierDetector


def test_us_formatted_decimal_strings_are_not_corrupted():
    df = pd.DataFrame({
        "Group": ["a", "a", "b", "b"],
        "Value": ["1.5", "2.75", "3.0", "4.25"],
    })
    # The fixture must start as text (not already float): that is when
    # _convert_values_to_float runs. pandas >=3 infers a StringDtype for text
    # columns instead of object, so assert on the intent -- non-numeric -- rather
    # than a specific dtype.
    assert not pd.api.types.is_numeric_dtype(df["Value"]), "test fixture must start as text, not already numeric"

    detector = OutlierDetector(df, "Group", "Value")

    assert list(detector.df["Value"]) == pytest.approx([1.5, 2.75, 3.0, 4.25]), (
        "US-formatted decimal strings must parse as-is, not get reinterpreted as "
        "German-formatted thousands separators"
    )


def test_german_formatted_decimal_strings_still_convert_correctly():
    df = pd.DataFrame({
        "Group": ["a", "a", "b", "b"],
        "Value": ["1,5", "2,75", "1.234,5", "4,25"],
    })
    detector = OutlierDetector(df, "Group", "Value")

    assert list(detector.df["Value"]) == pytest.approx([1.5, 2.75, 1234.5, 4.25]), (
        "genuine German-formatted values (comma decimal, dot thousands) must still convert"
    )


def test_already_numeric_column_is_left_untouched():
    df = pd.DataFrame({"Group": ["a", "b"], "Value": [1.5, 2.75]})
    detector = OutlierDetector(df, "Group", "Value")
    assert list(detector.df["Value"]) == pytest.approx([1.5, 2.75])
