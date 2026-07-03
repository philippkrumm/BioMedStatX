"""_detect_wide_format must not treat an entirely-empty (all-NaN) numeric
column as a usable value column — today it passes the dtype check (NaN
columns are float64) and only fails later with a cryptic empty-group error
deep in analysis_core.py.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from autopilot.statistical_analyzer_autopilot_ui import _detect_wide_format


def test_all_nan_value_column_is_excluded_not_crashed_on():
    df = pd.DataFrame({
        "Subject": ["S1", "S2", "S3", "S4", "S5"],
        "Time1": [10.5, 11.2, 12.1, 13.5, 14.0],
        "Time2": [11.2, 13.1, 14.0, 15.1, 15.5],
        "Time3": [np.nan, np.nan, np.nan, np.nan, np.nan],
    })
    result = _detect_wide_format(df)
    assert result is not None
    assert "Time3" not in result["value_cols"], (
        "an all-NaN column must not be treated as a usable measurement column"
    )
    assert set(result["value_cols"]) == {"Time1", "Time2"}


def test_all_columns_nan_returns_none_not_a_bogus_signature():
    df = pd.DataFrame({
        "Subject": ["S1", "S2", "S3", "S4", "S5"],
        "Time1": [np.nan] * 5,
        "Time2": [np.nan] * 5,
    })
    result = _detect_wide_format(df)
    assert result is None, (
        "with zero usable value columns, this must not be reported as wide-format data"
    )
