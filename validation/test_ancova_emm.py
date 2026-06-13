"""
test_ancova_emm.py — Regression tests for ANCOVA estimated marginal means.

Background (audit 2026-06): adjusted_means() predicted over the empirical
subset rows of each factor level, i.e. the OTHER between-factors entered with
their observed (sample-weighted) distribution. True EMMs average predictions
over a balanced reference grid: every level of the other factors weighted
equally, covariates at their grand mean (R emmeans / SPSS EMMEANS behaviour).

With one between-factor both definitions coincide; with two or more factors
and an unbalanced design they diverge.

Test strategy: noise-free data generated from a known additive model. The
correct EMM difference between factor levels then equals the true effect
EXACTLY, independent of the imbalance — no tolerance fudging needed.
"""

import numpy as np
import pandas as pd
import pytest

from analysis.clinical_models import ANCOVAModel


A_EFFECT = 2.0     # true effect of A: a2 vs a1
B_EFFECT = 5.0     # true effect of B: b2 vs b1
COV_SLOPE = 1.5


def _make_unbalanced_two_factor():
    """Noise-free additive data; B-distribution differs strongly per A level.

    A=a1 cells: 12x b1, 3x b2.  A=a2 cells: 3x b1, 12x b2.
    Sample-weighted averaging therefore biases the A contrast by
    B_EFFECT * (12-3)/15 = 3.0; the true EMM contrast is exactly A_EFFECT.
    """
    rng = np.random.default_rng(8)
    rows = []
    layout = {("a1", "b1"): 12, ("a1", "b2"): 3, ("a2", "b1"): 3, ("a2", "b2"): 12}
    for (a, b), n in layout.items():
        cov = rng.uniform(0, 10, n)
        y = (
            A_EFFECT * (a == "a2")
            + B_EFFECT * (b == "b2")
            + COV_SLOPE * cov
        )
        for c, v in zip(cov, y):
            rows.append({"A": a, "B": b, "cov": c, "y": v})
    return pd.DataFrame(rows)


def test_emm_two_factors_unbalanced_recovers_true_effect():
    df = _make_unbalanced_two_factor()
    model = ANCOVAModel()
    model.fit(df, dv="y", between_factors=["A", "B"], covariates=["cov"], alpha=0.05)

    means = model.adjusted_means()
    a = means["A"]
    diff_a = a["a2"]["adjusted_mean"] - a["a1"]["adjusted_mean"]
    assert diff_a == pytest.approx(A_EFFECT, abs=1e-6), (
        f"A contrast {diff_a:.4f} != true effect {A_EFFECT} "
        f"(sample-weighted bias would give ~{A_EFFECT + 3.0:.1f})"
    )

    b = means["B"]
    diff_b = b["b2"]["adjusted_mean"] - b["b1"]["adjusted_mean"]
    assert diff_b == pytest.approx(B_EFFECT, abs=1e-6)


def test_emm_single_factor_at_covariate_grand_mean():
    """One factor: EMM must equal the model prediction at the covariate grand mean."""
    rng = np.random.default_rng(4)
    n = 40
    grp = np.array(["g1"] * (n // 2) + ["g2"] * (n // 2))
    cov = rng.uniform(0, 10, n)
    y = 3.0 * (grp == "g2") + 0.8 * cov  # noise-free
    df = pd.DataFrame({"grp": grp, "cov": cov, "y": y})

    model = ANCOVAModel()
    model.fit(df, dv="y", between_factors=["grp"], covariates=["cov"], alpha=0.05)
    means = model.adjusted_means()["grp"]

    cov_mean = float(df["cov"].mean())
    assert means["g1"]["adjusted_mean"] == pytest.approx(0.8 * cov_mean, abs=1e-6)
    assert means["g2"]["adjusted_mean"] == pytest.approx(3.0 + 0.8 * cov_mean, abs=1e-6)
    # raw/n bookkeeping intact
    assert means["g1"]["n"] == n // 2
