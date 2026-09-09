"""The classic decision tree presented variance homogeneity as a gate it isn't.

Nodes B and C read "Are the data normally distributed and variances equal?"
and "Assumptions met/violated" based on normality AND Brown-Forsythe. But the
engine uses an unconditional Welch default (Feature B): select_comparison_test
never reads the variance verdict. So the diagram showed a decision dimension
the engine does not have -- structurally the same "diagram says X, engine does
Y" defect, in visual form.

The fix keeps the engine untouched and makes the diagram honest: normality is
the gate (parametric vs non-parametric), Brown-Forsythe is shown for
information only, and the paired/RM case where no variance test runs renders a
clean "N/A" instead of a bare True.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from analysis.analysis_core import AnalysisManager
from visualization.decisiontreevisualizer import DecisionTreeVisualizer


@pytest.fixture(scope="module", autouse=True)
def _qt_and_dialogs():
    try:
        from PyQt5.QtWidgets import QApplication, QDialog
    except Exception:
        yield
        return
    app = QApplication.instance() or QApplication([])
    mp = pytest.MonkeyPatch()
    mp.setattr(QDialog, "exec_", lambda self, *a, **k: 0, raising=False)
    mp.setattr(QDialog, "exec", lambda self, *a, **k: 0, raising=False)
    try:
        from analysis.statisticaltester import UIDialogManager
        mp.setattr(UIDialogManager, "select_transformation_dialog", staticmethod(lambda *a, **k: None), raising=False)
        for name in ("select_posthoc_test_dialog", "select_nonparametric_posthoc_dialog",
                     "select_control_group_dialog", "select_custom_pairs_dialog"):
            mp.setattr(UIDialogManager, name, staticmethod(lambda *a, **k: None), raising=False)
    except Exception:
        pass
    yield app
    mp.undo()


@pytest.fixture
def dummy_file(tmp_path):
    path = tmp_path / "dummy.xlsx"
    pd.DataFrame({"a": [1]}).to_excel(path, index=False)
    return str(path)


def _two_group(df, dummy_file, tmp_path, tag, dependent):
    ctx = {
        "injected_df": df, "factor_columns": ["Group"],
        "between_factors": [] if dependent else ["Group"], "dv_columns": ["Value"],
        "group_labels": ["A", "B"], "mode": "single", "dependent": dependent,
    }
    if dependent:
        ctx["subject_column"] = "Subject"
    results = AnalysisManager.analyze(
        file_path=dummy_file, group_col="Group", groups=["A", "B"],
        value_cols=["Value"], save_plot=False, skip_plots=True, dependent=dependent,
        file_name=str(tmp_path / tag), analysis_context=ctx,
        subject_column="Subject" if dependent else None,
    )
    tree = DecisionTreeVisualizer.get_tree_json(results)
    labels = {n["id"]: n["label"] for n in tree["nodes"]}
    return results, labels


def _heteroscedastic_independent():
    """seed 21: normal groups, Brown-Forsythe flags unequal variance (p=0.01)."""
    rng = np.random.default_rng(21)
    n = 20
    a = rng.normal(10, 2, n)
    b = rng.normal(12, 2, n)
    return pd.DataFrame({"Group": ["A"] * n + ["B"] * n,
                         "Value": np.concatenate([a, b]),
                         "Subject": list(range(n)) * 2})


def test_independent_variance_is_informational_not_a_gate(dummy_file, tmp_path):
    results, labels = _two_group(_heteroscedastic_independent(), dummy_file, tmp_path,
                                 "indep", dependent=False)
    node_b, node_c = labels["B"], labels["C"]

    # node B no longer poses variance equality as a co-equal assumption question
    assert "variances equal?" not in node_b
    # normality is the stated gate
    assert "normal" in node_b.lower()
    # Brown-Forsythe still shown, but marked informational
    assert "Brown-Forsythe" in node_b
    assert "informational" in node_b.lower() or "not used" in node_b.lower()

    # the engine ran Welch (variance unequal is fine, not a violation)
    assert results.get("test") == "Welch's t-test (unequal variances)"
    # node C's verdict tracks normality alone; the groups are normal -> "met"
    assert "met" in node_c.lower()
    assert "violated" not in node_c.lower()


def test_paired_variance_renders_na_cleanly(dummy_file, tmp_path):
    results, labels = _two_group(_heteroscedastic_independent(), dummy_file, tmp_path,
                                 "paired", dependent=True)
    node_b = labels["B"]
    assert results.get("test") == "Paired t-test"
    # no variance test runs for a paired design -> honest N/A, never a bare True
    assert "N/A" in node_b
    assert "Brown-Forsythe: True" not in node_b
    assert "variances equal?" not in node_b
