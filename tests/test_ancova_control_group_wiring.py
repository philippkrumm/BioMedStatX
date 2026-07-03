"""analysis_core.py's ANCOVA dispatch never passed control_group into
ANCOVAModel.fit(), even though the model supports it and the LMM branch
right below it already does this correctly via control_group_callback. This
made the vs-control multivariate-t EMM post-hoc unreachable from the primary
dispatch path (AnalysisManager.analyze -> _analyze_single_dataset), even
though advanced_pipeline.py's secondary dispatch path already had it wired.

ANCOVAModel is imported locally inside _analyze_single_dataset
(`from analysis.clinical_models import (ANCOVAModel, ...)`), so it must be
patched at its source module (analysis.clinical_models), not at
analysis.analysis_core — patching the re-exported name there would not be
picked up by the function's own fresh import.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

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


class _FakeAncovaModel:
    """Records the kwargs .fit() was called with instead of doing real work."""
    last_fit_kwargs = None

    def fit(self, df, **kwargs):
        _FakeAncovaModel.last_fit_kwargs = kwargs

    def as_results_dict(self):
        return {"model_type": "ANCOVA", "p_value": 0.5, "adjusted_means": {}}


def test_ancova_dispatch_passes_control_group_callback_result(dummy_file, tmp_path, monkeypatch):
    import analysis.clinical_models as clinical_models_module
    monkeypatch.setattr(clinical_models_module, "ANCOVAModel", _FakeAncovaModel)

    df = pd.DataFrame({
        "Group": ["ctrl", "ctrl", "a", "a", "b", "b"],
        "Value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "Cov": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    })
    ctx = {
        "injected_df": df,
        "factor_columns": ["Group"],
        "between_factors": ["Group"],
        "dv_columns": ["Value"],
        "group_labels": ["ctrl", "a", "b"],
        "mode": "single",
    }
    control_cb_calls = []

    def _control_cb(levels):
        control_cb_calls.append(levels)
        return "ctrl"

    AnalysisManager.analyze(
        file_path=dummy_file,
        group_col="Group",
        groups=["ctrl", "a", "b"],
        value_cols=["Value"],
        save_plot=False,
        skip_plots=True,
        file_name=str(tmp_path / "out"),
        analysis_context=ctx,
        test="ancova",
        covariates=["Cov"],
        control_group_callback=_control_cb,
    )

    assert len(control_cb_calls) == 1, "control_group_callback must be invoked for ANCOVA"
    assert _FakeAncovaModel.last_fit_kwargs is not None, "ANCOVAModel.fit was never called"
    assert _FakeAncovaModel.last_fit_kwargs.get("control_group") == "ctrl"


def test_ancova_dispatch_without_callback_passes_none(dummy_file, tmp_path, monkeypatch):
    import analysis.clinical_models as clinical_models_module
    monkeypatch.setattr(clinical_models_module, "ANCOVAModel", _FakeAncovaModel)

    df = pd.DataFrame({
        "Group": ["ctrl", "ctrl", "a", "a", "b", "b"],
        "Value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "Cov": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    })
    ctx = {
        "injected_df": df,
        "factor_columns": ["Group"],
        "between_factors": ["Group"],
        "dv_columns": ["Value"],
        "group_labels": ["ctrl", "a", "b"],
        "mode": "single",
    }

    AnalysisManager.analyze(
        file_path=dummy_file,
        group_col="Group",
        groups=["ctrl", "a", "b"],
        value_cols=["Value"],
        save_plot=False,
        skip_plots=True,
        file_name=str(tmp_path / "out"),
        analysis_context=ctx,
        test="ancova",
        covariates=["Cov"],
    )

    assert _FakeAncovaModel.last_fit_kwargs is not None
    assert _FakeAncovaModel.last_fit_kwargs.get("control_group") is None
