"""The continuous predictor was cast to str, which killed RegressionHealthScanner.

``_prepare_contextual_inputs`` stringifies ``display_group_col`` so the
group-splitting equality join below it matches when group labels are numeric.
For correlation and linear regression ``display_group_col`` is not a grouping
column at all -- it is the continuous predictor. Casting it to str meant:

  * ``RegressionHealthScanner`` died on ``np.median`` over an object-dtype
    array, and the bare ``except`` in analysis_core swallowed the TypeError,
    so every correlation/regression report silently lost its data-health
    findings (missing values, predictor outliers, VIF).
  * the Raw Data Vault exported the predictor as strings.

The cast is still required for the categorical branch, where a numeric factor
column has to become categorical -- the tests below pin both sides.
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from analysis.analysis_core import AnalysisManager
from analysis.correlation_models import RegressionHealthScanner


@pytest.fixture(scope="module", autouse=True)
def _qt_and_dialogs():
    """The multi-group guard tests reach the post-hoc path, which constructs Qt
    dialogs. Same neutralisation the golden-core suite uses."""
    try:
        from PyQt5.QtWidgets import QApplication, QDialog
    except Exception:
        yield
        return
    app = QApplication.instance() or QApplication([])
    QDialog.exec_ = lambda self, *a, **k: 0
    QDialog.exec = lambda self, *a, **k: 0
    try:
        from analysis.statisticaltester import UIDialogManager
        UIDialogManager.select_transformation_dialog = staticmethod(lambda *a, **k: None)
        UIDialogManager.select_posthoc_test_dialog = staticmethod(lambda *a, **k: "tukey")
        for name in ("select_nonparametric_posthoc_dialog", "select_control_group_dialog",
                     "select_custom_pairs_dialog"):
            setattr(UIDialogManager, name, staticmethod(lambda *a, **k: None))
    except Exception:
        pass
    yield app


@pytest.fixture
def dummy_file(tmp_path):
    path = tmp_path / "dummy.xlsx"
    pd.DataFrame({"a": [1]}).to_excel(path, index=False)
    return str(path)


def _unhealthy_regression_df():
    """Audit fixture: seed 12, n=60, one missing covariate value, one outlier."""
    rng = np.random.default_rng(12)
    n = 60
    x = rng.normal(0, 1, n)
    c = rng.normal(0, 1, n)
    y = 1 + 0.8 * x + 1.2 * c + rng.normal(0, 1, n)
    df = pd.DataFrame({"X": x, "Y": y, "C": c})
    df.loc[0, "C"] = np.nan
    df.loc[1, "C"] = 900.0
    return df


def _run(df, dummy_file, tmp_path, tag, test, covariates=None):
    out = str(tmp_path / tag)
    ctx = {
        "injected_df": df, "x_variable": "X", "factor_columns": ["X"],
        "dv_columns": ["Y"], "mode": "single", "covariates": covariates or [],
    }
    results = AnalysisManager.analyze(
        file_path=dummy_file, group_col="X", groups=[], value_cols=["Y"],
        save_plot=False, skip_plots=True, file_name=out,
        analysis_context=ctx, test=test, covariates=covariates or [],
    )
    return results, out + "_results.html"


def _visible_text(path):
    doc = open(path, errors="ignore").read()
    doc = re.sub(r"<script.*?</script>", "", doc, flags=re.S | re.I)
    doc = re.sub(r"<style.*?</style>", "", doc, flags=re.S | re.I)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", doc))


def test_premise_scanner_fires_on_this_fixture_when_x_is_numeric():
    """Positive control from the audit: numeric X -> 2 warnings, str X -> TypeError."""
    df = _unhealthy_regression_df()
    report = RegressionHealthScanner(df, x_col="X", y_col="Y", covariates=["C"]).run()
    assert len(report["warnings"]) == 2
    assert any("Missing values" in w for w in report["warnings"])
    assert any("mod. Z-score" in w for w in report["warnings"])

    stringified = df.copy()
    stringified["X"] = stringified["X"].astype(str)
    with pytest.raises(TypeError):
        RegressionHealthScanner(stringified, x_col="X", y_col="Y", covariates=["C"]).run()


@pytest.mark.parametrize("test,covariates", [
    ("linear_regression", ["C"]),
    ("correlation", []),
])
def test_health_report_survives_the_pipeline(dummy_file, tmp_path, test, covariates):
    results, _ = _run(_unhealthy_regression_df(), dummy_file, tmp_path,
                      f"health_{test}", test, covariates)
    health = results.get("data_health") or {}
    assert health.get("checks"), (
        f"{test}: RegressionHealthScanner produced nothing — it died and the bare "
        f"except swallowed the reason"
    )
    assert "missing_data" in health["checks"]


def test_health_warnings_reach_the_rendered_regression_report(dummy_file, tmp_path):
    results, report = _run(_unhealthy_regression_df(), dummy_file, tmp_path,
                           "health_html", "linear_regression", ["C"])
    assert os.path.exists(report)
    text = _visible_text(report)

    # positive control: the report really did render engine content
    assert "Linear Regression" in text

    assert "Missing values" in text, "missing-data warning still does not reach the report"
    assert "mod. Z-score" in text, "predictor outlier warning still does not reach the report"


def test_raw_data_vault_keeps_the_predictor_numeric(dummy_file, tmp_path):
    results, _ = _run(_unhealthy_regression_df(), dummy_file, tmp_path,
                      "vault", "linear_regression", ["C"])
    columns = results.get("raw_data_columns") or {}
    assert "X" in columns and columns["X"]
    assert all(isinstance(v, float) for v in columns["X"]), (
        f"predictor exported as {type(columns['X'][0]).__name__}, not float"
    )


def test_numeric_group_labels_still_split_correctly(dummy_file, tmp_path):
    """Guard for the branch the cast exists for: numeric group labels must still
    match the stringified group_labels used for sample splitting."""
    rng = np.random.default_rng(4)
    n = 15
    df = pd.DataFrame({
        "Dose": [1] * n + [2] * n + [3] * n,
        "Value": np.concatenate([rng.normal(10, 2, n), rng.normal(13, 2, n),
                                 rng.normal(16, 2, n)]),
    })
    ctx = {
        "injected_df": df, "factor_columns": ["Dose"], "between_factors": ["Dose"],
        "dv_columns": ["Value"], "group_labels": ["1", "2", "3"], "mode": "single",
    }
    results = AnalysisManager.analyze(
        file_path=dummy_file, group_col="Dose", groups=["1", "2", "3"],
        value_cols=["Value"], save_plot=False, skip_plots=True,
        file_name=str(tmp_path / "numeric_groups"), analysis_context=ctx,
    )
    assert results.get("error") is None
    descriptive = results.get("descriptive") or {}
    assert set(descriptive) == {"1", "2", "3"}, (
        f"numeric group labels no longer split into groups: {sorted(descriptive)}"
    )
    for group in ("1", "2", "3"):
        assert descriptive[group]["n"] == n
    assert results.get("p_value") is not None


def test_numeric_factor_stays_categorical_for_ancova(dummy_file, tmp_path):
    """The cast is what makes a numeric factor categorical downstream. ANCOVA
    must keep seeing Dose as three groups, not as a continuous regressor."""
    rng = np.random.default_rng(5)
    n = 20
    dose = np.array([1] * n + [2] * n + [3] * n)
    cov = rng.normal(10, 2, 3 * n)
    df = pd.DataFrame({
        "Dose": dose,
        "Cov": cov,
        "Value": 2 + 0.5 * cov + dose * 1.5 + rng.normal(0, 1, 3 * n),
    })
    ctx = {
        "injected_df": df, "factor_columns": ["Dose"], "between_factors": ["Dose"],
        "dv_columns": ["Value"], "group_labels": ["1", "2", "3"], "mode": "single",
        "covariates": ["Cov"],
    }
    results = AnalysisManager.analyze(
        file_path=dummy_file, group_col="Dose", groups=["1", "2", "3"],
        value_cols=["Value"], save_plot=False, skip_plots=True,
        file_name=str(tmp_path / "ancova_numeric"), analysis_context=ctx,
        test="ancova", covariates=["Cov"],
    )
    assert results.get("error") is None
    assert results.get("p_value") is not None
    adjusted = results.get("adjusted_means") or {}
    levels = adjusted.get("Dose") or {}
    assert set(levels) == {"1", "2", "3"}, (
        f"ANCOVA no longer treats the numeric factor as three levels: {sorted(levels)}"
    )
