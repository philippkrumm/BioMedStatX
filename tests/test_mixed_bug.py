import sys

import numpy as np
import pandas as pd
import pytest
from PyQt5.QtWidgets import QApplication

from analysis.analysis_core import AnalysisManager

# Reuse an existing QApplication if another Qt test already created one in this
# process; only construct one if none exists. Avoids a double-initialization.
app = QApplication.instance() or QApplication(sys.argv)


@pytest.fixture(autouse=True)
def _qt_and_dialogs(monkeypatch):
    """Self-contained dialog suppression. This module builds no dialog stubs of
    its own; it was only passing because an earlier module's (now torn-down)
    global patches leaked in. Neutralise every modal here so a headless run can
    never block on a real dialog (function-scoped monkeypatch auto-reverts)."""
    try:
        from PyQt5.QtWidgets import QDialog
        from analysis.statisticaltester import UIDialogManager
    except Exception:
        return
    monkeypatch.setattr(QDialog, "exec_", lambda self, *a, **k: 0, raising=False)
    monkeypatch.setattr(QDialog, "exec", lambda self, *a, **k: 0, raising=False)
    monkeypatch.setattr(UIDialogManager, "select_transformation_dialog",
                        staticmethod(lambda *a, **k: None), raising=False)
    monkeypatch.setattr(UIDialogManager, "select_posthoc_test_dialog",
                        staticmethod(lambda *a, **k: "tukey"), raising=False)
    monkeypatch.setattr(UIDialogManager, "select_nonparametric_posthoc_dialog",
                        staticmethod(lambda *a, **k: "dunn"), raising=False)
    for name in ("select_control_group_dialog", "select_custom_pairs_dialog"):
        monkeypatch.setattr(UIDialogManager, name,
                            staticmethod(lambda *a, **k: None), raising=False)


def test_mixed_anova_autopilot_flow(tmp_path):
    # Regression guard: autopilot infers two_way_anova, but a subject column is
    # mapped, so the analysis must be upgraded to a Mixed ANOVA (5afaa56/38dee1e).
    np.random.seed(42)
    subjects = [f"S{i:02d}" for i in range(1, 21)] * 2
    groups = ["KO"] * 10 + ["WT"] * 10
    groups = groups * 2
    timepoints = ["0h"] * 20 + ["2h"] * 20
    values = np.random.randn(40) + 10

    df = pd.DataFrame({
        "Subject ID": subjects,
        "Factor 2": groups,
        "Factor 1": timepoints,
        "Dependent Variable": values,
    })
    xlsx_path = tmp_path / "temp_mixed_bug.xlsx"
    df.to_excel(xlsx_path, index=False)

    analysis_context = {
        "dv_columns": ["Dependent Variable"],
        "factor_columns": ["Factor 1", "Factor 2"],
        "subject_column": "Subject ID",
        # Simulating the bug: autopilot somehow inferred two_way_anova
        "inferred_test": "two_way_anova",
        "selected_group_column": None,
        "selected_groups": [],
        "dependent": False,
        "display_group_col": "__AUTO_GROUP__",
    }

    # Simulate how autopilot calls AnalysisManager
    res = AnalysisManager.analyze(
        file_path=str(xlsx_path),
        group_col="__AUTO_GROUP__",
        groups=[],
        sheet_name="Sheet1",
        value_cols=["Dependent Variable"],
        combine_columns=False,
        dependent=False, compare=False,
        colors=[], hatches=[], title="", x_label="", y_label="",
        file_name="", save_plot=False, skip_plots=True,
        error_type="se", dataset_name="",
        # Crucially, autopilot does NOT pass additional_factors directly in kwargs.
        test="two_way_anova",
        analysis_context=analysis_context,
        subject_column="Subject ID",
    )

    assert res.get("test") == "Mixed ANOVA", f"Expected Mixed ANOVA, got {res.get('test')}"

    types = [f.get("type") for f in res.get("factors", [])]
    assert "within" in types and "between" in types, f"Expected within and between factors, got {types}"
