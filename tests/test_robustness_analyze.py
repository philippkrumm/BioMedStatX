"""analyze()-level robustness suite.

Drives the full AnalysisManager.analyze entry (the documented injected-df path)
with each pathological dataset, exercising NaN ingestion, numeric coercion, group
extraction, the data-quality chokepoint and assumption checks end-to-end. Every
case must degrade gracefully.
"""
import pandas as pd
import pytest

from analysis.analysis_core import AnalysisManager

from tests.robustness.contract import assert_graceful
from tests.robustness.edge_cases import CATALOG, to_long_df


@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch):
    """analyze() opens Qt dialogs when assumptions fail (transformation choice)
    or a multi-group result is significant (post-hoc choice). In an automated
    run those modal dialogs abort the process. Replace them with non-interactive
    defaults so the suite exercises data flow, not the GUI."""
    # Catch-all so a directly-constructed modal (e.g. ComparisonSelectionDialog)
    # can't block the event-less test process.
    from PyQt5.QtWidgets import QDialog
    monkeypatch.setattr(QDialog, "exec_", lambda self, *a, **k: 0, raising=False)
    monkeypatch.setattr(QDialog, "exec", lambda self, *a, **k: 0, raising=False)

    from analysis.statisticaltester import UIDialogManager
    monkeypatch.setattr(UIDialogManager, "select_transformation_dialog",
                        staticmethod(lambda *a, **k: "log10"), raising=False)
    # "tukey" avoids the custom-pairs follow-up modal.
    monkeypatch.setattr(UIDialogManager, "select_posthoc_test_dialog",
                        staticmethod(lambda *a, **k: "tukey"), raising=False)
    for name in ("select_nonparametric_posthoc_dialog",
                 "select_control_group_dialog", "select_custom_pairs_dialog"):
        monkeypatch.setattr(UIDialogManager, name,
                            staticmethod(lambda *a, **k: None), raising=False)


@pytest.fixture(scope="module")
def dummy_file(tmp_path_factory):
    # analyze() requires an existing path; injected_df overrides the content.
    path = tmp_path_factory.mktemp("robustness") / "dummy.xlsx"
    pd.DataFrame({"a": [1]}).to_excel(path, index=False)
    return str(path)


def _analyze(samples, dependent, file_path, output_dir):
    df = to_long_df(samples)
    groups = list(samples.keys())
    ctx = {
        "injected_df": df,
        "factor_columns": ["Grp"],
        "dv_columns": ["Val"],
        "group_labels": groups,
        "mode": "single",
    }
    return AnalysisManager.analyze(
        file_path=file_path,
        group_col="Grp",
        groups=groups,
        value_cols=["Val"],
        dependent=dependent,
        save_plot=False,
        skip_plots=True,
        file_name=str(output_dir / "out"),
        analysis_context=ctx,
    )


@pytest.mark.parametrize("ec", CATALOG, ids=lambda e: e.name)
def test_analyze_edge_case_is_graceful(ec, dummy_file, tmp_path):
    try:
        result = _analyze(ec.samples, ec.dependent, dummy_file, tmp_path)
    except Exception as exc:  # pragma: no cover - failure path
        pytest.fail(f"{ec.name}: analyze() raised instead of degrading: {exc!r}")

    assert isinstance(result, dict)
    assert_graceful(result, ec.expectation)

    if ec.expectation == "blocked" and ec.expected_code:
        assert result.get("block_code") == ec.expected_code, (
            f"{ec.name}: expected block_code {ec.expected_code}, got {result.get('block_code')}"
        )


def test_nonfinite_block_converts_inf_statistic():
    """Regression for the fuzzer finding: an advanced engine (RM-ANOVA -> LMM on a
    degenerate design) emitted statistic=-inf and it was presented as valid. The
    nonfinite_block safety net must convert any non-finite p/statistic into a
    clean block. Unit-tested on the guard logic so it stays valid regardless of
    which generator seed currently reproduces the LMM path."""
    from analysis.statisticaltester import StatisticalTester as T

    # -inf statistic, not blocked -> must be blocked
    b = T.nonfinite_block({"test": "Linear Mixed Model", "p_value": None, "statistic": float("-inf")})
    assert b is not None and b["blocked"] is True and b["block_code"] == "NON_FINITE_RESULT"

    # NaN p-value -> blocked
    b2 = T.nonfinite_block({"test": "Welch's ANOVA", "p_value": float("nan"), "statistic": 1.0})
    assert b2 is not None and b2["block_code"] == "NON_FINITE_RESULT"

    # finite, valid -> not touched
    assert T.nonfinite_block({"test": "t-test", "p_value": 0.04, "statistic": 2.1}) is None
    # already blocked -> not double-wrapped
    assert T.nonfinite_block({"blocked": True, "p_value": float("inf")}) is None


def test_clinical_model_constant_dv_is_blocked(dummy_file, tmp_path):
    """Clinical models (ANCOVA/LMM/regression) bypass the group chokepoint. A
    constant outcome must be blocked at the clinical pre-flight (VAR_ZERO) rather
    than fitting a meaningless/singular model."""
    import numpy as np
    rng = np.random.default_rng(4)
    rows = []
    for g in ("A", "B"):
        for _ in range(8):
            rows.append({"Grp": g, "Cov": float(rng.normal(0, 1)), "Val": 7.0})  # constant DV
    df = pd.DataFrame(rows)
    ctx = {"injected_df": df, "factor_columns": ["Grp"], "dv_columns": ["Val"],
           "group_labels": ["A", "B"], "covariates": ["Cov"], "between_factors": ["Grp"],
           "mode": "single"}
    result = AnalysisManager.analyze(
        file_path=dummy_file, group_col="Grp", groups=["A", "B"], value_cols=["Val"],
        test="ancova", covariates=["Cov"], save_plot=False, skip_plots=True,
        file_name=str(tmp_path / "out"), analysis_context=ctx,
    )
    assert result.get("blocked") is True
    assert result.get("block_code") == "VAR_ZERO"
