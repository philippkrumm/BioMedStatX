import math
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from analysis.analysis_core import AnalysisManager
from tests.robustness.contract import assert_graceful

@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch):
    try:
        from PyQt5.QtWidgets import QDialog
    except Exception:
        return
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

def generate_fuzz_dataset(seed: int, n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    
    # 1. Base numeric
    val1 = rng.normal(10, 2, n)
    val2 = rng.exponential(1, n)
    
    # 2. Inject edge cases
    # NaNs
    val1[rng.choice(n, int(n * 0.1), replace=False)] = np.nan
    # Extreme values
    val2[rng.choice(n, int(n * 0.05), replace=False)] = 1e10
    # Infs
    val1[rng.choice(n, int(n * 0.02), replace=False)] = np.inf
    
    # 3. Covariates with collinearity
    cov1 = val1 * 2 + rng.normal(0, 0.1, n)
    
    # 4. Binary outcome
    binary = rng.binomial(1, 0.5, n)
    
    # 5. Factors (balanced and unbalanced)
    factor1 = rng.choice(["A", "B", "C"], n, p=[0.5, 0.4, 0.1])
    factor2 = rng.choice(["X", "Y"], n)
    
    # 6. Rank ties (discrete data)
    discrete = rng.poisson(2, n)
    
    df = pd.DataFrame({
        "Subject": np.arange(1, n + 1),
        "Val1": val1,
        "Val2": val2,
        "Cov1": cov1,
        "Binary": binary,
        "Factor1": factor1,
        "Factor2": factor2,
        "Discrete": discrete,
    })
    return df

@pytest.fixture(scope="module")
def fuzz_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("fuzz")

# 5 seeds
SEEDS = [42, 101, 1337, 9999, 12345]

@pytest.mark.parametrize("seed", SEEDS)
def test_fuzz_analyze_is_graceful(seed, fuzz_dir):
    df = generate_fuzz_dataset(seed)
    
    dummy_file = str(fuzz_dir / f"dummy_{seed}.xlsx")
    df.to_excel(dummy_file, index=False)
    
    # Fuzz a few test types
    # 1. Logistic Regression
    ctx_log = {
        "injected_df": df.copy(),
        "mode": "single",
        "test_type_hint": "logistic_regression",
        "inferred_test": "logistic_regression",
        "factor_columns": ["Val1"],
        "dv_columns": ["Binary"]
    }
    try:
        res1 = AnalysisManager.analyze(
            file_path=dummy_file, group_col="Val1", groups=[], value_cols=["Binary"],
            dependent="Binary", save_plot=False, skip_plots=True, file_name=str(fuzz_dir / f"out_log_{seed}"),
            analysis_context=ctx_log, test="logistic_regression", covariates=["Cov1", "Factor1"]
        )
        assert isinstance(res1, dict)
        assert_graceful(res1, "any")
    except Exception as exc:
        pytest.fail(f"Logistic fuzz {seed} failed: {exc!r}")

    # 2. Two-Way ANOVA
    ctx_two = {
        "injected_df": df.copy(),
        "mode": "single",
        "test_type_hint": "two_way_anova",
        "inferred_test": "two_way_anova",
        "factor_columns": ["Factor1"],
        "dv_columns": ["Val2"]
    }
    try:
        res2 = AnalysisManager.analyze(
            file_path=dummy_file, group_col="Factor1", groups=[], value_cols=["Val2"],
            dependent="Val2", save_plot=False, skip_plots=True, file_name=str(fuzz_dir / f"out_two_{seed}"),
            analysis_context=ctx_two, test="two_way_anova", additional_factors=["Factor2"]
        )
        assert isinstance(res2, dict)
        assert_graceful(res2, "any")
    except Exception as exc:
        pytest.fail(f"Two-Way ANOVA fuzz {seed} failed: {exc!r}")
        
    # 3. Kruskal-Wallis (testing rank ties)
    ctx_kw = {
        "injected_df": df.copy(),
        "mode": "single",
        "test_type_hint": "kruskal_wallis",
        "inferred_test": "kruskal_wallis",
        "factor_columns": ["Factor1"],
        "dv_columns": ["Discrete"]
    }
    try:
        res3 = AnalysisManager.analyze(
            file_path=dummy_file, group_col="Factor1", groups=[], value_cols=["Discrete"],
            dependent="Discrete", save_plot=False, skip_plots=True, file_name=str(fuzz_dir / f"out_kw_{seed}"),
            analysis_context=ctx_kw, test="kruskal_wallis"
        )
        assert isinstance(res3, dict)
        assert_graceful(res3, "any")
    except Exception as exc:
        pytest.fail(f"Kruskal-Wallis fuzz {seed} failed: {exc!r}")
