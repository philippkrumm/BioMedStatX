"""Observed (not merely code-read): the control-group dialog is prompted at most
once in the analysis_core Dunnett path.

The redundant pre-prompt at analysis_core was removed so the refactored post-hoc
function is the single source that prompts for a Dunnett control group. Without
that removal, once the dialog returns None on cancel (the Bug-2 fix), the caller
would pre-prompt AND the function would prompt again -- a double dialog. This
counts real invocations through AnalysisManager.analyze to prove exactly one.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from analysis.analysis_core import AnalysisManager


@pytest.fixture
def dummy_file(tmp_path):
    path = tmp_path / "dummy.xlsx"
    pd.DataFrame({"a": [1]}).to_excel(path, index=False)
    return str(path)


def test_control_group_prompted_exactly_once_in_dunnett_path(dummy_file, tmp_path, monkeypatch):
    from PyQt5.QtWidgets import QApplication, QDialog
    QApplication.instance() or QApplication([])
    monkeypatch.setattr(QDialog, "exec_", lambda self, *a, **k: 0, raising=False)

    from analysis.statisticaltester import UIDialogManager

    calls = {"control": 0}

    def _count_control(groups, parent=None):
        calls["control"] += 1
        return groups[0]  # pick a valid control so Dunnett actually runs

    monkeypatch.setattr(UIDialogManager, "select_transformation_dialog",
                        staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(UIDialogManager, "select_posthoc_test_dialog",
                        staticmethod(lambda *a, **k: "dunnett"))
    monkeypatch.setattr(UIDialogManager, "select_control_group_dialog",
                        staticmethod(_count_control))

    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "Group": ["A"] * 8 + ["B"] * 8 + ["C"] * 8,
        "Value": list(rng.normal(10, 2, 8)) + list(rng.normal(13, 2, 8)) + list(rng.normal(16, 2, 8)),
    })
    ctx = {
        "injected_df": df, "factor_columns": ["Group"], "between_factors": ["Group"],
        "dv_columns": ["Value"], "group_labels": ["A", "B", "C"], "mode": "single",
        "dependent": False,
    }
    AnalysisManager.analyze(
        file_path=dummy_file, group_col="Group", groups=["A", "B", "C"],
        value_cols=["Value"], save_plot=False, skip_plots=True, dependent=False,
        file_name=str(tmp_path / "dunnett_once"), analysis_context=ctx,
    )

    assert calls["control"] <= 1, (
        f"control-group dialog prompted {calls['control']} times -- double prompt regressed"
    )
    assert calls["control"] == 1, (
        f"expected exactly one control-group prompt in the Dunnett path, got {calls['control']}"
    )
