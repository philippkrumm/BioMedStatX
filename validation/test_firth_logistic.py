"""
test_firth_logistic.py — Regression tests for Firth logistic inference.

Background (audit 2026-06): when the Firth fallback engaged, p-values came
from Wald z = coef/se. Wald inference is unreliable in exactly the situations
Firth exists for (separation / sparse events). The reference method (R logistf
default) is the penalized likelihood-ratio (PLR) test:

    PLR_j = 2 * [ pl(beta_hat) - max_{beta: beta_j = 0} pl(beta) ]  ~  chi2(1)

where pl is the Firth-penalized log-likelihood  ll + 0.5*log|X'WX|.

Validation strategy (logistf not installable on this machine):
  1. Large n, no separation: the Jeffreys penalty vanishes asymptotically, so
     Firth PLR p-values must approach the standard likelihood-ratio test
     p-values from statsmodels GLM.
  2. Quasi-separation: Firth must engage, coefficients stay finite, PLR
     p-values are valid and more powerful than Wald for the separated
     true-effect predictor.
"""

import numpy as np
import pandas as pd
import pytest

from analysis.clinical_models import LogisticRegressionModel


def _make_large_n(seed=1, n=800):
    """Well-behaved data: x1 true effect, x2 null."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    eta = -0.3 + 0.6 * x1 + 0.0 * x2
    y = rng.binomial(1, 1 / (1 + np.exp(-eta)))
    X = np.column_stack([np.ones(n), x1, x2])
    return X, y.astype(float)


def _glm_lr_pvalue(X, y, j):
    """Standard (unpenalized) likelihood-ratio test p for dropping column j."""
    import statsmodels.api as sm
    from scipy import stats

    full = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    Xr = np.delete(X, j, axis=1)
    red = sm.GLM(y, Xr, family=sm.families.Binomial()).fit()
    stat = 2 * (full.llf - red.llf)
    return float(stats.chi2.sf(max(stat, 0.0), 1))


@pytest.mark.parametrize("j", [1, 2])  # effect and null predictor
def test_firth_plr_approaches_glm_lr_at_large_n(j):
    X, y = _make_large_n()
    model = LogisticRegressionModel()
    beta, _, _ = model._fit_firth_logistic(X, y)
    p_plr = model._firth_plr_pvalue(X, y, beta, j)
    p_lr = _glm_lr_pvalue(X, y, j)
    # Penalty is O(p/n): at n=800 the tests must agree closely.
    assert p_plr == pytest.approx(p_lr, abs=5e-3), (
        f"col {j}: Firth PLR p={p_plr:.5f}, GLM LR p={p_lr:.5f}"
    )


def _make_separated_df(seed=3, n=30):
    """Quasi-complete separation in the binary factor 'group'.

    fit() wraps predictors as C(...) categorical terms, so the separated
    predictor must be a factor; 'noise' is a continuous null covariate.
    """
    rng = np.random.default_rng(seed)
    group = np.array(["A"] * (n // 2) + ["B"] * (n - n // 2))
    outcome = (group == "B").astype(int)
    # one overlap point so the model is barely identifiable
    outcome[0] = 1
    noise = rng.normal(0, 1, n)
    return pd.DataFrame({"outcome": outcome, "group": group, "noise": noise})


def test_firth_engages_on_separation_and_reports_plr():
    df = _make_separated_df()
    model = LogisticRegressionModel().fit(df, "outcome", ["group"], covariates=["noise"])

    assert model._model_variant == "Firth Penalized Likelihood"

    rows = model.odds_ratios()
    assert rows, "no odds-ratio rows returned"
    for row in rows:
        assert row["p_value_method"].startswith("PLR"), row
        assert 0.0 < row["p_value"] <= 1.0
        assert np.isfinite(row["odds_ratio"])
        assert np.isfinite(row["coefficient"])


def test_firth_plr_beats_wald_under_separation():
    """For the separated true-effect predictor, PLR must be more powerful than Wald."""
    from scipy import stats

    df = _make_separated_df()
    model = LogisticRegressionModel().fit(df, "outcome", ["group"], covariates=["noise"])
    assert model._model_variant == "Firth Penalized Likelihood"

    group_row = next(r for r in model.odds_ratios() if "group" in r["parameter"].lower())
    p_wald = 2 * (1 - stats.norm.cdf(abs(group_row["z_value"])))
    assert group_row["p_value"] < p_wald, (
        f"PLR p={group_row['p_value']:.4f} not smaller than Wald p={p_wald:.4f}"
    )
