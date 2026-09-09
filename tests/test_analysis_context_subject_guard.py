"""_ap_build_analysis_context must reject a Subject-ID column containing NaN
before any factor/balance-detection logic runs — this is the one choke point
every analysis path (wide-pivoted or manually-mapped long-format) passes
through, unlike _detect_wide_format which only covers the auto-pivot path.
Uses a minimal fake `self` (not a real Qt widget) exposing only the bucket
API _ap_build_analysis_context actually calls (get_assigned_columns /
get_filter / isChecked) — building a full QApplication harness for this one
guard would be disproportionate (see HANDOFF.md's noted preference for
extracting/testing pure logic over fragile full-Qt-app harnesses).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from autopilot.statistical_analyzer_autopilot_pipeline import _ap_build_analysis_context


class _FakeBucket:
    def __init__(self, columns=None):
        self._columns = columns or []

    def get_assigned_columns(self):
        return list(self._columns)


class _FakeFilterBucket(_FakeBucket):
    def get_filter(self):
        return None


class _FakeCheckbox:
    def __init__(self, checked=False):
        self._checked = checked

    def isChecked(self):
        return self._checked


class _FakeApp:
    def __init__(self, df, subject_col):
        self.df = df
        self.dv_bucket = _FakeBucket(["Value"])
        self.factor1_bucket = _FakeBucket(["Group"])
        self.factor2_bucket = _FakeBucket([])
        self.subject_bucket = _FakeBucket([subject_col] if subject_col else [])
        self.covariates_bucket = _FakeBucket([])
        self.filter_bucket = _FakeFilterBucket([])
        self.multi_mode_button = _FakeCheckbox(False)
        self.analysis_selected_groups = []


def test_build_analysis_context_rejects_missing_subject_id():
    # Genuine repeated-measures shape (each subject appears at both Group
    # levels) except one row's Subject is NaN — the guard must fire before
    # the later repeated-measures-structure check even looks at this.
    df = pd.DataFrame({
        "Subject": ["S1", "S1", "S2", np.nan],
        "Group": ["A", "B", "A", "B"],
        "Value": [1.0, 2.0, 3.0, 4.0],
    })
    fake_self = _FakeApp(df, subject_col="Subject")
    with pytest.raises(ValueError, match=r"1 missing"):
        _ap_build_analysis_context(fake_self)


def test_build_analysis_context_allows_complete_subject_id():
    # Each subject appears at both Group levels — valid repeated-measures shape.
    df = pd.DataFrame({
        "Subject": ["S1", "S1", "S2", "S2"],
        "Group": ["A", "B", "A", "B"],
        "Value": [1.0, 2.0, 3.0, 4.0],
    })
    fake_self = _FakeApp(df, subject_col="Subject")
    context = _ap_build_analysis_context(fake_self)
    assert context["subject_column"] == "Subject"
