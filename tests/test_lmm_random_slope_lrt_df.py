"""LinearMixedModel.fit() hardcoded the random-slope LRT to df=1, with the
comment "Diagonal RE structure adds 1 parameter (variance of slope)". Both
premises were wrong: statsmodels' MixedLM default random-effects covariance is
unstructured (not diagonal), and the random-slope candidate handed in by the
pipeline (`within_factors[0]`) is a *categorical* factor, so `re_formula="~F"`
adds one random slope column per non-reference level.

For a 3-level within factor the RI-only model has a 1x1 cov_re (1 free
parameter) and the RI+slope model a 3x3 cov_re (6 free parameters) — a df
difference of 5, not 1. Testing a 5-df statistic against chi2(1) is wildly
anti-conservative: on data generated WITHOUT any random slope the overparameter-
ised structure was selected in ~40% of runs, and because `self.result` is then
switched to that model, every downstream number (fixed-effect p-values, ICC,
residual variance) came from it.

Seed 1011 is the flip case from the audit: D = 3.9713, chi2.sf(D, 1) = 0.0463
(selects the random slope) vs chi2.sf(D, 5) = 0.5535 (correctly keeps
random-intercept-only).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

from analysis.clinical_models import LinearMixedModel


def _flip_case_df():
    """The audit's seed-1011 dataset: no random slope in the data-generating
    process, three timepoints, 15 subjects."""
    rng = np.random.default_rng(1011)
    rows = []
    for i in range(1, 16):
        subject = f"S{i:02d}"
        intercept = rng.normal(0, 1.5)
        for level, effect in [("t0", 0.0), ("t1", 1.0), ("t2", 2.0)]:
            rows.append({
                "Subject": subject,
                "Time": level,
                "Value": 10 + intercept + effect + rng.normal(0, 0.8),
            })
    return pd.DataFrame(rows)


def _reference_fits(df):
    """Fit both candidate structures directly, independent of the code under
    test, and derive the expected df from the cov_re dimensions."""
    import statsmodels.formula.api as smf

    ri = smf.mixedlm("Value ~ C(Time)", df, groups=df["Subject"]).fit(reml=True)
    rs = smf.mixedlm("Value ~ C(Time)", df, groups=df["Subject"],
                     re_formula="~Time").fit(reml=True)

    def n_free_cov_params(fit):
        k = fit.cov_re.shape[0]
        return k * (k + 1) // 2

    expected_df = n_free_cov_params(rs) - n_free_cov_params(ri)
    D = 2 * (rs.llf - ri.llf)
    return ri, rs, expected_df, D


def test_expected_df_for_a_three_level_within_factor_is_five():
    """Guards the premise: the structure really does add 5 parameters."""
    df = _flip_case_df()
    ri, rs, expected_df, _ = _reference_fits(df)
    assert ri.cov_re.shape == (1, 1)
    assert rs.cov_re.shape == (3, 3), (
        "re_formula='~Time' on a 3-level categorical must produce a 3x3 "
        "unstructured cov_re (intercept + 2 slope columns)"
    )
    assert expected_df == 5


def test_lrt_uses_the_real_covariance_parameter_difference():
    df = _flip_case_df()
    _, _, expected_df, D = _reference_fits(df)

    model = LinearMixedModel()
    model.fit(df, dv="Value", fixed_effects=["Time"],
              random_intercept="Subject", random_slope="Time")
    res = model.as_results_dict()

    assert res["lrt_performed"] is True
    assert res["lrt_statistic"] == pytest.approx(D, rel=1e-6)
    assert res["lrt_p_value"] == pytest.approx(chi2.sf(D, expected_df), rel=1e-6), (
        f"LRT p must be evaluated against chi2({expected_df}); "
        f"chi2(1) would give {chi2.sf(D, 1):.6f}"
    )


def test_flip_case_keeps_random_intercept_only():
    """The regression itself: with the correct df this dataset must NOT get a
    random slope."""
    df = _flip_case_df()
    model = LinearMixedModel()
    model.fit(df, dv="Value", fixed_effects=["Time"],
              random_intercept="Subject", random_slope="Time")
    res = model.as_results_dict()

    assert res["lrt_p_value"] > 0.05
    assert res["random_structure_chosen"] == "Random Intercept Only"


def test_false_selection_rate_is_near_nominal():
    """Data generated without a random slope must not select one at ~40%.

    The audit measured 21/53 with the hardcoded df=1. With the correct df the
    rate has to collapse to roughly the nominal alpha level.
    """
    selected = 0
    usable = 0
    for sim in range(30):
        rng = np.random.default_rng(1000 + sim)
        rows = []
        for i in range(1, 16):
            subject = f"S{i:02d}"
            intercept = rng.normal(0, 1.5)
            for level, effect in [("t0", 0.0), ("t1", 1.0), ("t2", 2.0)]:
                rows.append({"Subject": subject, "Time": level,
                             "Value": 10 + intercept + effect + rng.normal(0, 0.8)})
        df = pd.DataFrame(rows)
        model = LinearMixedModel()
        model.fit(df, dv="Value", fixed_effects=["Time"],
                  random_intercept="Subject", random_slope="Time")
        res = model.as_results_dict()
        if not res["lrt_performed"]:
            continue
        usable += 1
        if res["random_structure_chosen"].startswith("Random Intercept + Random Slope"):
            selected += 1

    assert usable >= 20, f"too few usable simulations ({usable}) to judge the rate"
    rate = selected / usable
    assert rate <= 0.15, (
        f"random slope selected in {selected}/{usable} = {rate:.1%} of runs on "
        "data without one; the hardcoded df=1 produced ~40%"
    )
