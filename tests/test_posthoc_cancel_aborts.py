"""Cancelling the post-hoc selection dialog aborts the whole analysis.

Product decision: the post-hoc dialog appears only AFTER a significant omnibus,
but pressing Cancel there must abort the run entirely -- no results, no report
file written, no confetti (the pipeline resets to the mapping state). This is
distinct from the explicit "none" choice, which keeps the omnibus and simply
omits pairwise comparisons.

The dialog returning None (cancel) raises AnalysisCancelledError before the
report is exported, so no HTML is written; analyze() turns it into a
{"cancelled": True} result the pipeline honours.
"""
import glob
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _qt(monkeypatch):
    from PyQt5.QtWidgets import QApplication, QDialog
    QApplication.instance() or QApplication([])
    monkeypatch.setattr(QDialog, "exec_", lambda self, *a, **k: 0, raising=False)


def _significant_three_group():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "Group": ["A"] * 8 + ["B"] * 8 + ["C"] * 8,
        "Value": (list(rng.normal(10, 1, 8)) + list(rng.normal(13, 3, 8))
                  + list(rng.normal(16, 6, 8))),
    })


def _run(monkeypatch, posthoc_return):
    from analysis.statisticaltester import UIDialogManager
    from analysis.analysis_core import AnalysisManager
    monkeypatch.setattr(UIDialogManager, "select_transformation_dialog",
                        staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(UIDialogManager, "select_posthoc_test_dialog",
                        staticmethod(lambda *a, **k: posthoc_return))
    monkeypatch.setattr(UIDialogManager, "select_control_group_dialog",
                        staticmethod(lambda *a, **k: None))

    d = tempfile.mkdtemp()
    dummy = os.path.join(d, "x.xlsx")
    pd.DataFrame({"a": [1]}).to_excel(dummy, index=False)
    out = os.path.join(d, "out")
    df = _significant_three_group()
    ctx = {"injected_df": df, "factor_columns": ["Group"], "between_factors": ["Group"],
           "dv_columns": ["Value"], "group_labels": ["A", "B", "C"], "mode": "single",
           "dependent": False}
    result = AnalysisManager.analyze(
        file_path=dummy, group_col="Group", groups=["A", "B", "C"], value_cols=["Value"],
        save_plot=False, skip_plots=True, dependent=False, file_name=out,
        analysis_context=ctx)
    reports = glob.glob(out + "*.html")
    return result, reports


def test_posthoc_cancel_aborts_with_no_report(monkeypatch):
    result, reports = _run(monkeypatch, posthoc_return=None)   # CANCEL
    assert result.get("cancelled") is True, f"expected cancelled result, got keys {list(result)}"
    assert "p_value" not in result and "posthoc_test" not in result, \
        "aborted analysis must not carry partial results"
    assert reports == [], f"no report may be written on cancel, found {reports}"


def test_nonparametric_posthoc_cancel_aborts(monkeypatch):
    """Kruskal-Wallis path: cancelling the non-parametric (Dunn) post-hoc dialog
    aborts too -- the abort reaches through perform_refactored_posthoc_testing's
    except-Exception guards (AnalysisCancelledError is a BaseException)."""
    from analysis.statisticaltester import UIDialogManager
    from analysis.analysis_core import AnalysisManager
    monkeypatch.setattr(UIDialogManager, "select_transformation_dialog",
                        staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(UIDialogManager, "select_nonparametric_posthoc_dialog",
                        staticmethod(lambda *a, **k: None))  # CANCEL
    d = tempfile.mkdtemp()
    dummy = os.path.join(d, "x.xlsx")
    pd.DataFrame({"a": [1]}).to_excel(dummy, index=False)
    out = os.path.join(d, "out")
    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        "Group": ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
        "Value": (list(5 + rng.lognormal(1, 1.2, 10)) + list(20 + rng.lognormal(1, 1.2, 10))
                  + list(40 + rng.lognormal(1, 1.2, 10))),
    })
    ctx = {"injected_df": df, "factor_columns": ["Group"], "between_factors": ["Group"],
           "dv_columns": ["Value"], "group_labels": ["A", "B", "C"], "mode": "single",
           "dependent": False}
    result = AnalysisManager.analyze(
        file_path=dummy, group_col="Group", groups=["A", "B", "C"], value_cols=["Value"],
        save_plot=False, skip_plots=True, dependent=False, file_name=out, analysis_context=ctx)
    assert result.get("cancelled") is True, f"nonparametric cancel must abort, got {list(result)}"
    assert glob.glob(out + "*.html") == [], "no report on nonparametric cancel"


def test_comparison_selection_cancel_aborts(monkeypatch):
    """Advanced paired_custom path: cancelling the ComparisonSelectionDialog
    aborts (was a silent 'select all pairs' default). The _qt fixture makes the
    real dialog's exec_ return Rejected; transform/post-hoc are stubbed so the
    comparison dialog is the one that cancels."""
    from analysis.statisticaltester import UIDialogManager
    from analysis.analysis_core import AnalysisManager
    monkeypatch.setattr(UIDialogManager, "select_transformation_dialog",
                        staticmethod(lambda *a, **k: "skip"))
    monkeypatch.setattr(UIDialogManager, "select_posthoc_test_dialog",
                        staticmethod(lambda *a, **k: "paired_custom"))
    d = tempfile.mkdtemp()
    dummy = os.path.join(d, "x.xlsx")
    pd.DataFrame({"a": [1]}).to_excel(dummy, index=False)
    out = os.path.join(d, "out")
    rng = np.random.default_rng(1)
    rows = []
    for i in range(10):
        base = rng.normal(0, 1)
        for t, eff in [("T1", 0), ("T2", 5), ("T3", 10)]:
            rows.append({"Subject": f"S{i}", "Time": t, "Value": base + eff + rng.normal(0, 1)})
    df = pd.DataFrame(rows)
    ctx = {"injected_df": df, "factor_columns": ["Time"], "between_factors": [],
           "within_factors": ["Time"], "dv_columns": ["Value"], "group_labels": ["T1", "T2", "T3"],
           "mode": "single", "dependent": True, "subject_column": "Subject",
           "inferred_test": "repeated_measures_anova"}
    result = AnalysisManager.analyze(
        file_path=dummy, group_col="Time", groups=["T1", "T2", "T3"], value_cols=["Value"],
        save_plot=False, skip_plots=True, dependent=True, file_name=out,
        analysis_context=ctx, subject_column="Subject", test="repeated_measures_anova")
    assert result.get("cancelled") is True, f"comparison-select cancel must abort, got {list(result)}"
    assert glob.glob(out + "*.html") == [], "no report on comparison-select cancel"


def test_advanced_posthoc_cancel_aborts(monkeypatch):
    """RM-ANOVA (advanced engine) path: cancelling the post-hoc dialog aborts
    too, reaching through the advanced engine's except-Exception guards."""
    from analysis.statisticaltester import UIDialogManager
    from analysis.analysis_core import AnalysisManager
    monkeypatch.setattr(UIDialogManager, "select_transformation_dialog",
                        staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(UIDialogManager, "select_posthoc_test_dialog",
                        staticmethod(lambda *a, **k: None))  # CANCEL (advanced uses this)
    d = tempfile.mkdtemp()
    dummy = os.path.join(d, "x.xlsx")
    pd.DataFrame({"a": [1]}).to_excel(dummy, index=False)
    out = os.path.join(d, "out")
    rng = np.random.default_rng(1)
    rows = []
    for i in range(10):
        base = rng.normal(0, 1)
        for t, eff in [("T1", 0), ("T2", 5), ("T3", 10)]:
            rows.append({"Subject": f"S{i}", "Time": t, "Value": base + eff + rng.normal(0, 1)})
    df = pd.DataFrame(rows)
    ctx = {"injected_df": df, "factor_columns": ["Time"], "between_factors": [],
           "within_factors": ["Time"], "dv_columns": ["Value"], "group_labels": ["T1", "T2", "T3"],
           "mode": "single", "dependent": True, "subject_column": "Subject",
           "inferred_test": "repeated_measures_anova"}
    result = AnalysisManager.analyze(
        file_path=dummy, group_col="Time", groups=["T1", "T2", "T3"], value_cols=["Value"],
        save_plot=False, skip_plots=True, dependent=True, file_name=out,
        analysis_context=ctx, subject_column="Subject", test="repeated_measures_anova")
    assert result.get("cancelled") is True, f"advanced cancel must abort, got {list(result)}"
    assert glob.glob(out + "*.html") == [], "no report on advanced cancel"
