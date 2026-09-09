"""The multivariate-t adjusted p-values were not reproducible.

`_mvt_pvalues` (ANCOVAModel and LinearMixedModel each carry a copy) calls
`scipy.stats.multivariate_t.cdf`, which integrates by Monte Carlo and accepts a
`random_state`. No seed was passed, so the same dataset analysed twice produced
different post-hoc p-values. The audit measured five consecutive runs of one
ANCOVA giving 2.380e-04, 2.333e-04, 2.346e-04, 2.282e-04, 2.324e-04 for the same
contrast — and at a t-value whose true adjusted p sits on 0.05, 300 identical
calls split 210 "significant" / 90 "not significant".

This affects every vs-control post-hoc in ANCOVA and LMM, i.e. the default path
whenever a control group is set. The Holm/pairwise path was always bit-stable.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from analysis.clinical_models import ANCOVAModel, LinearMixedModel


def _ancova_df():
    rng = np.random.default_rng(4242)
    n = 20
    df = pd.DataFrame({
        "Group": ["ctrl"] * n + ["low"] * n + ["high"] * n,
        "Cov": rng.normal(10, 2, 3 * n),
    })
    df["Value"] = (2 + 0.8 * df["Cov"]
                   + df["Group"].map({"ctrl": 0.0, "low": 1.5, "high": 3.0})
                   + rng.normal(0, 1, 3 * n))
    return df


def test_mvt_helper_is_deterministic():
    """Direct call, fixed arguments, five repetitions."""
    R = np.array([[1.0, 0.5], [0.5, 1.0]])
    runs = [tuple(ANCOVAModel._mvt_pvalues([2.3, 2.9], R, 32)) for _ in range(5)]
    assert len(set(runs)) == 1, (
        f"multivariate-t p-values still vary across identical calls: {runs}"
    )
    runs_lmm = [tuple(LinearMixedModel._mvt_pvalues([2.3, 2.9], R, 28))
                for _ in range(5)]
    assert len(set(runs_lmm)) == 1, (
        f"LMM copy still varies across identical calls: {runs_lmm}"
    )


def test_ancova_vs_control_posthoc_is_reproducible():
    df = _ancova_df()
    runs = []
    for _ in range(5):
        model = ANCOVAModel()
        model.fit(df, dv="Value", between_factors=["Group"], covariates=["Cov"],
                  control_group="ctrl")
        res = model.as_results_dict()
        runs.append(tuple(round(c["p_value"], 15)
                          for c in res["pairwise_comparisons"]))
    assert len(set(runs)) == 1, f"vs-control p-values still drift: {runs}"


def test_lmm_vs_control_posthoc_is_reproducible():
    rng = np.random.default_rng(4242)
    rows = []
    for i in range(1, 21):
        subject, intercept = f"S{i:02d}", rng.normal(0, 1.5)
        for level, effect in [("t0", 0.0), ("t1", 1.2), ("t2", 2.4)]:
            rows.append({"Subject": subject, "Time": level,
                         "Value": 10 + intercept + effect + rng.normal(0, 0.8)})
    df = pd.DataFrame(rows)

    runs = []
    for _ in range(5):
        model = LinearMixedModel()
        model.fit(df, dv="Value", fixed_effects=["Time"],
                  random_intercept="Subject", control_group="t0")
        res = model.as_results_dict()
        runs.append(tuple(round(c["p_value"], 15)
                          for c in res["pairwise_comparisons"]))
    assert len(set(runs)) == 1, f"LMM vs-control p-values still drift: {runs}"


def test_verdict_near_alpha_is_stable():
    """The audit's coin-flip case: a t-value whose adjusted p sits on 0.05 must
    produce one verdict, not a distribution of them."""
    R = np.array([[1.0, 0.5], [0.5, 1.0]])
    t_on_the_line = 2.313783
    verdicts = {ANCOVAModel._mvt_pvalues([t_on_the_line, t_on_the_line], R, 32)[0] < 0.05
                for _ in range(50)}
    assert len(verdicts) == 1, (
        "significance verdict still flips between runs at the alpha boundary"
    )


def test_holm_path_stays_reproducible():
    """Control: the pairwise/Holm branch was never affected and must stay so."""
    df = _ancova_df()
    runs = []
    for _ in range(3):
        model = ANCOVAModel()
        model.fit(df, dv="Value", between_factors=["Group"], covariates=["Cov"])
        res = model.as_results_dict()
        runs.append(tuple(c["p_value"] for c in res["pairwise_comparisons"]))
    assert len(set(runs)) == 1
