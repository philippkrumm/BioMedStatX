"""LogisticRegressionModel.fit() hard-requires len(predictors) >= 1 (clinical_models.py:1079-1080,
added 2026-07-03), but the UI's own dispatch logic (statistical_analyzer_autopilot_pipeline.py:
761-763) recommends "logistic_regression" for a covariates-only setup (binary outcome, covariates
bucket, no factor1/between) - that configuration reaches fit() with predictors=[] and raises
ModelDesignError, contradicting the UI's own recommendation. Statistically there is no reason a
covariates-only logistic regression should be rejected: the model's own formula construction
already handles predictors=[] gracefully (terms = [] + covariate names), and odds_ratios() reads
from the fitted model's actual parameter index, not from self._predictors directly.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from analysis.clinical_models import LogisticRegressionModel


def test_fit_succeeds_with_covariates_only_no_predictors():
    rng = np.random.RandomState(0)
    n = 100
    x = rng.randn(n)
    y = (x + rng.randn(n) * 0.5 > 0).astype(int)
    df = pd.DataFrame({"y": y, "x": x})

    model = LogisticRegressionModel()
    model.fit(df, dv="y", predictors=[], covariates=["x"])

    rows = model.odds_ratios()
    assert len(rows) == 1, f"expected exactly one coefficient row for the single covariate, got: {rows}"
    assert "x" in rows[0]["parameter"]


def test_fit_still_rejects_zero_predictors_and_zero_covariates():
    rng = np.random.RandomState(0)
    n = 20
    y = rng.randint(0, 2, n)
    df = pd.DataFrame({"y": y})

    model = LogisticRegressionModel()
    with pytest.raises(Exception):
        model.fit(df, dv="y", predictors=[], covariates=[])
