"""
test_correlation_ci.py — Regression tests for correlation confidence intervals.

Pearson CI uses the standard Fisher-z SE = 1/sqrt(n-3).
Spearman CI must use the Bonett-Wright SE = sqrt((1 + r^2/2) / (n-3)),
which is wider than the Pearson SE (the rank statistic is less efficient).

Background (audit 2026-06): _fisher_z_ci applied the Pearson SE to Spearman
correlations, producing CIs ~8% too narrow (anti-conservative).
"""

import numpy as np
import pandas as pd
import pytest

from analysis.correlation_models import _fisher_z_ci, CorrelationModel


def _bonett_wright_ci(r, n, alpha=0.05):
    z = np.arctanh(r)
    se = np.sqrt((1.0 + r**2 / 2.0) / (n - 3))
    zc = scipy_norm_ppf(1 - alpha / 2)
    return float(np.tanh(z - zc * se)), float(np.tanh(z + zc * se))


def scipy_norm_ppf(q):
    from scipy import stats
    return float(stats.norm.ppf(q))


def _pearson_ci(r, n, alpha=0.05):
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    zc = scipy_norm_ppf(1 - alpha / 2)
    return float(np.tanh(z - zc * se)), float(np.tanh(z + zc * se))


@pytest.mark.parametrize("r,n", [(0.6, 30), (-0.45, 50), (0.8, 15)])
def test_spearman_ci_uses_bonett_wright(r, n):
    lo, hi = _fisher_z_ci(r, n, alpha=0.05, method="spearman")
    exp_lo, exp_hi = _bonett_wright_ci(r, n)
    assert lo == pytest.approx(exp_lo, abs=1e-9)
    assert hi == pytest.approx(exp_hi, abs=1e-9)


@pytest.mark.parametrize("r,n", [(0.6, 30), (-0.45, 50)])
def test_pearson_ci_unchanged(r, n):
    lo, hi = _fisher_z_ci(r, n, alpha=0.05, method="pearson")
    exp_lo, exp_hi = _pearson_ci(r, n)
    assert lo == pytest.approx(exp_lo, abs=1e-9)
    assert hi == pytest.approx(exp_hi, abs=1e-9)


def test_spearman_ci_wider_than_pearson():
    r, n = 0.6, 30
    s_lo, s_hi = _fisher_z_ci(r, n, method="spearman")
    p_lo, p_hi = _fisher_z_ci(r, n, method="pearson")
    assert (s_hi - s_lo) > (p_hi - p_lo)


def test_default_method_is_pearson_backward_compat():
    """Calling without method= must keep the original Pearson behaviour."""
    r, n = 0.5, 40
    assert _fisher_z_ci(r, n) == _fisher_z_ci(r, n, method="pearson")


def test_correlation_model_spearman_uses_wider_ci():
    """End-to-end: a Spearman fit must report the Bonett-Wright CI."""
    rng = np.random.default_rng(11)
    x = rng.normal(0, 1, 40)
    # monotone but non-linear -> Spearman path under method='spearman'
    y = np.sign(x) * np.abs(x) ** 1.5 + rng.normal(0, 0.3, 40)
    df = pd.DataFrame({"X": x, "Y": y})

    model = CorrelationModel().fit(df, "X", "Y", method="spearman")
    out = model.as_results_dict()

    exp_lo, exp_hi = _bonett_wright_ci(out["r"], out["n"])
    assert out["ci_lower"] == pytest.approx(exp_lo, abs=1e-9)
    assert out["ci_upper"] == pytest.approx(exp_hi, abs=1e-9)
