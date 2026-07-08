"""advanced_pipeline.py explicitly excludes logistic_regression from the shared
validate_samples_for_test pre-flight gate (`if test not in ["logistic_regression"]:`) with no
substitute - a constant (all-0 or all-1) binary outcome reaches LogisticRegressionModel.fit()
with no pre-flight net, unlike every other advanced test.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import pytest

from statistical_testing.advanced_pipeline import perform_advanced_test_pipeline


def test_constant_binary_outcome_is_blocked_before_fitting():
    df = pd.DataFrame({
        "Outcome": [0, 0, 0, 0, 0, 0],  # zero variance - logistic regression is meaningless
        "Predictor": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })

    result = perform_advanced_test_pipeline(
        df=df,
        test="logistic_regression",
        dv="Outcome",
        subject=None,
        between=["Predictor"],
        within=None,
        force_parametric=True,
    )

    assert result.get("blocked") is True, (
        f"expected a blocked result for a constant outcome, got: {result}"
    )
    assert result.get("block_code") == "VAR_ZERO"


def test_normal_binary_outcome_is_not_blocked():
    import numpy as np
    rng = np.random.RandomState(0)
    n = 60
    predictor = rng.randn(n)
    df = pd.DataFrame({
        "Outcome": (predictor + rng.randn(n) * 0.5 > 0).astype(int),
        "Predictor": predictor,
    })

    result = perform_advanced_test_pipeline(
        df=df,
        test="logistic_regression",
        dv="Outcome",
        subject=None,
        between=["Predictor"],
        within=None,
        force_parametric=True,
    )

    assert result.get("blocked") is not True, f"a normal binary outcome should not be blocked: {result}"
    assert result.get("error") is None, f"a normal binary outcome should not error: {result}"
