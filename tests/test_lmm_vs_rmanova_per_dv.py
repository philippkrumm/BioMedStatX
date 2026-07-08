"""U2: multi-DV batch mode must decide LMM-vs-RM-ANOVA/paired-ttest per DV column, not once
from dv_columns[0] for the whole batch - each DV column can have its own missingness pattern.
_ap_lmm_vs_rmanova_needed is the extracted pure function this decision should go through,
callable both from _ap_build_analysis_context (the original single-shot site) and from the
multi-DV loop in _ap_determine_and_run_test (per column).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from autopilot.statistical_analyzer_autopilot_pipeline import (
    _ap_lmm_vs_rmanova_needed,
    _ap_build_analysis_context,
)


def test_complete_data_does_not_need_lmm():
    df = pd.DataFrame({
        "Subject": ["S1", "S1", "S2", "S2"],
        "Time": ["T1", "T2", "T1", "T2"],
        "Gene_A": [1.0, 2.0, 3.0, 4.0],
    })
    assert _ap_lmm_vs_rmanova_needed(df, "Subject", "Time", "Gene_A") is False


def test_structurally_missing_timepoint_needs_lmm():
    df = pd.DataFrame({
        "Subject": ["S1", "S1", "S2"],
        "Time": ["T1", "T2", "T1"],
        "Gene_A": [1.0, 2.0, 3.0],
    })
    assert _ap_lmm_vs_rmanova_needed(df, "Subject", "Time", "Gene_A") is True


def test_nan_measurement_needs_lmm():
    df = pd.DataFrame({
        "Subject": ["S1", "S1", "S2", "S2"],
        "Time": ["T1", "T2", "T1", "T2"],
        "Gene_A": [1.0, 2.0, 3.0, np.nan],
    })
    assert _ap_lmm_vs_rmanova_needed(df, "Subject", "Time", "Gene_A") is True


def test_different_dv_columns_can_disagree():
    df = pd.DataFrame({
        "Subject": ["S1", "S1", "S2", "S2"],
        "Time": ["T1", "T2", "T1", "T2"],
        "Gene_A": [1.0, 2.0, 3.0, 4.0],
        "Gene_B": [1.0, 2.0, 3.0, np.nan],
    })
    assert _ap_lmm_vs_rmanova_needed(df, "Subject", "Time", "Gene_A") is False
    assert _ap_lmm_vs_rmanova_needed(df, "Subject", "Time", "Gene_B") is True


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
    def __init__(self, df, subject_col, dv_col="Value"):
        self.df = df
        self.dv_bucket = _FakeBucket([dv_col])
        self.factor1_bucket = _FakeBucket(["Time"])
        self.factor2_bucket = _FakeBucket([])
        self.subject_bucket = _FakeBucket([subject_col] if subject_col else [])
        self.covariates_bucket = _FakeBucket([])
        self.filter_bucket = _FakeFilterBucket([])
        self.multi_mode_button = _FakeCheckbox(False)
        self.analysis_selected_groups = []


def test_build_analysis_context_upgrades_to_lmm_and_stores_pre_upgrade_choice():
    df = pd.DataFrame({
        "Subject": ["S1", "S1", "S2"],
        "Time": ["T1", "T2", "T1"],
        "Value": [1.0, 2.0, 3.0],
    })
    fake_self = _FakeApp(df, subject_col="Subject")
    context = _ap_build_analysis_context(fake_self)
    assert context["inferred_test"] == "lmm"
    assert context["_test_before_lmm_upgrade"] == "paired_ttest"


def test_build_analysis_context_leaves_complete_data_as_paired_ttest():
    df = pd.DataFrame({
        "Subject": ["S1", "S1", "S2", "S2"],
        "Time": ["T1", "T2", "T1", "T2"],
        "Value": [1.0, 2.0, 3.0, 4.0],
    })
    fake_self = _FakeApp(df, subject_col="Subject")
    context = _ap_build_analysis_context(fake_self)
    assert context["inferred_test"] == "paired_ttest"
    assert context["_test_before_lmm_upgrade"] == "paired_ttest"
