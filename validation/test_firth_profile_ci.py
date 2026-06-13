"""
test_firth_profile_ci.py — Penalized profile-likelihood CI for Firth logistic.

Background (audit 2026-06): under separation (the case Firth is invoked for),
the Wald CI coef +/- 1.96*se is unreliable. The correct interval is the
penalized profile-likelihood CI used by R's logistf: the set of beta_j with
2*[pl(beta_hat) - pl_profile(beta_j)] <= chi2(1, 1-alpha), where pl_profile
re-maximizes the penalized log-likelihood over the other coefficients.

R's logistf could not be installed in this environment, so the production
Newton/bisection implementation (ANCOVAModel-style) is cross-validated against
an INDEPENDENT scipy.optimize + brentq reference implemented here. Two
independent implementations of the same well-defined estimator must agree.
"""

import numpy as np
import pytest
from scipy import optimize
from scipy.stats import chi2

from analysis.clinical_models import LogisticRegressionModel


# --- independent reference implementation (scipy.optimize + brentq) ---

def _pen_loglik(beta, X, y):
    eta = X @ beta
    pi = 1.0 / (1.0 + np.exp(-eta))
    pi = np.clip(pi, 1e-12, 1 - 1e-12)
    ll = np.sum(y * np.log(pi) + (1 - y) * np.log(1 - pi))
    w = pi * (1 - pi)
    xtwx = X.T @ (w[:, None] * X)
    sign, logdet = np.linalg.slogdet(xtwx)
    if sign <= 0:
        return -np.inf
    return ll + 0.5 * logdet


def _fit_full(X, y):
    p = X.shape[1]
    res = optimize.minimize(lambda b: -_pen_loglik(b, X, y), np.zeros(p),
                            method="BFGS", options={"maxiter": 2000})
    return res.x


def _profile_max(X, y, j, value):
    p = X.shape[1]
    free = [k for k in range(p) if k != j]

    def negpl(bfree):
        b = np.empty(p)
        b[j] = value
        b[free] = bfree
        return -_pen_loglik(b, X, y)

    res = optimize.minimize(negpl, np.zeros(len(free)), method="BFGS",
                            options={"maxiter": 2000})
    return -res.fun


def _reference_profile_ci(X, y, j, alpha=0.05):
    beta_hat = _fit_full(X, y)
    pll_hat = _pen_loglik(beta_hat, X, y)
    crit = chi2.ppf(1 - alpha, 1)

    def g(c):
        return 2 * (pll_hat - _profile_max(X, y, j, c)) - crit

    se_guess = max(abs(beta_hat[j]) * 0.5, 1.0)
    # expand outward until bracketed
    def find_root(direction):
        step = se_guess
        c0 = beta_hat[j]
        c1 = beta_hat[j] + direction * step
        for _ in range(60):
            if g(c1) > 0:
                return optimize.brentq(g, c0, c1, xtol=1e-6)
            c0, c1 = c1, c1 + direction * step
        return np.nan
    return find_root(-1), find_root(+1)


def _separation_data():
    x = np.array([1,2,3,4,5,6,7,8,9,10,2,4,6,8,3,7], dtype=float)
    y = np.array([0,0,0,0,0,1,1,1,1,1,0,0,1,1,0,1], dtype=float)
    return x, y


def _fit_app():
    import pandas as pd
    x, y = _separation_data()
    df = pd.DataFrame({"x": x, "y": y})
    model = LogisticRegressionModel()
    # continuous predictor -> force the Firth path by construction below
    model.fit(df, dv="y", predictors=[], covariates=["x"])
    return model


def test_firth_profile_ci_matches_independent_reference():
    model = _fit_app()
    assert model._model_variant == "Firth Penalized Likelihood", "expected Firth path on separated data"

    X = model.result.model.exog
    y = model.result.model.endog
    # locate the x coefficient index
    names = list(model.result.params.index)
    j = names.index("x")

    lo_ref, hi_ref = _reference_profile_ci(X, y, j, alpha=0.05)
    lo_app, hi_app = model._firth_profile_ci(X, y, model._firth_coefs, j, alpha=0.05)

    assert lo_app == pytest.approx(lo_ref, abs=0.05), f"lo app={lo_app} ref={lo_ref}"
    assert hi_app == pytest.approx(hi_ref, abs=0.05), f"hi app={hi_app} ref={hi_ref}"


def test_profile_ci_contains_point_estimate():
    model = _fit_app()
    X, y = model.result.model.exog, model.result.model.endog
    j = list(model.result.params.index).index("x")
    lo, hi = model._firth_profile_ci(X, y, model._firth_coefs, j, alpha=0.05)
    assert lo < model._firth_coefs[j] < hi


def test_odds_ratios_report_profile_ci():
    model = _fit_app()
    rows = model.odds_ratios()
    row = next(r for r in rows if r["parameter"] == "x")
    assert row["ci_method"] == "Profile likelihood"
    # CI on OR scale must bracket the point OR and be ordered
    assert row["ci_lower"] < row["odds_ratio"] < row["ci_upper"]
