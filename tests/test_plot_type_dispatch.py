"""Guards the fix for the dead 'Strip' plot-type branch and the silent
wrong-plot-type fallback in the export dispatch, found in the Help Hub content
audit. An unrecognized plot_type must raise, matching the preview dispatch's
behavior, instead of silently rendering a Bar plot.
"""
import pandas as pd
import pytest

from analysis.analysis_core import AnalysisManager


@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch):
    from PyQt5.QtWidgets import QDialog
    monkeypatch.setattr(QDialog, "exec_", lambda self, *a, **k: 0, raising=False)
    monkeypatch.setattr(QDialog, "exec", lambda self, *a, **k: 0, raising=False)
    from analysis.statisticaltester import UIDialogManager
    monkeypatch.setattr(UIDialogManager, "select_transformation_dialog",
                        staticmethod(lambda *a, **k: "log10"), raising=False)
    monkeypatch.setattr(UIDialogManager, "select_posthoc_test_dialog",
                        staticmethod(lambda *a, **k: "tukey"), raising=False)
    for name in ("select_nonparametric_posthoc_dialog",
                 "select_control_group_dialog", "select_custom_pairs_dialog"):
        monkeypatch.setattr(UIDialogManager, name,
                            staticmethod(lambda *a, **k: None), raising=False)


@pytest.fixture
def dummy_file(tmp_path):
    path = tmp_path / "dummy.xlsx"
    pd.DataFrame({"a": [1]}).to_excel(path, index=False)
    return str(path)


def _run(dummy_file, tmp_path, plot_type):
    df = pd.DataFrame({
        "Grp": ["Control", "Control", "Control", "Treatment", "Treatment", "Treatment"],
        "Val": [1.0, 2.0, 1.5, 5.0, 6.0, 5.5],
    })
    ctx = {
        "injected_df": df,
        "factor_columns": ["Grp"],
        "dv_columns": ["Val"],
        "group_labels": ["Control", "Treatment"],
        "mode": "single",
    }
    return AnalysisManager.analyze(
        file_path=dummy_file,
        group_col="Grp",
        groups=["Control", "Treatment"],
        value_cols=["Val"],
        save_plot=False,
        skip_plots=False,
        plot_type=plot_type,
        file_name=str(tmp_path / "out"),
        analysis_context=ctx,
    )


def test_unrecognized_plot_type_raises(dummy_file, tmp_path):
    # AnalysisManager._analyze_single_dataset wraps its whole body in a
    # try/except Exception that converts any raised error (by design, per
    # the graceful-degradation/blocked-result contract) into a blocked
    # result dict rather than letting it propagate. So the ValueError raised
    # by the plot-type dispatch surfaces here as a blocked result carrying
    # the original message, not as a propagated exception.
    result = _run(dummy_file, tmp_path, plot_type="NotARealPlotType")
    assert result.get("blocked") is True
    assert result.get("block_code") == "UNHANDLED_EXCEPTION"
    assert "Unknown plot type" in (result.get("block_reason") or "")


def test_strip_is_no_longer_a_special_case(dummy_file, tmp_path):
    result = _run(dummy_file, tmp_path, plot_type="Strip")
    assert result.get("blocked") is True
    assert result.get("block_code") == "UNHANDLED_EXCEPTION"
    assert "Unknown plot type" in (result.get("block_reason") or "")


def test_bar_still_renders(dummy_file, tmp_path):
    result = _run(dummy_file, tmp_path, plot_type="Bar")
    assert result is not None
