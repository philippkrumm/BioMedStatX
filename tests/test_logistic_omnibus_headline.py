"""LogisticRegressionModel.as_results_dict() published `or_table[0]` — the first
non-intercept coefficient — as the model's headline `p_value` / `statistic`.
Same defect as the LMM headline: for a predictor with more than two levels that
is one dummy contrast, not the effect of the predictor.

The audit's case: three groups where a and b are identical by construction and c
has a 97% event rate. The b-vs-a contrast is n.s. (p ~ 1.0), the predictor is
overwhelming (LR chi2 ~ 51 on 2 df). The report hero printed "Logistic
Regression did not show evidence against the null hypothesis" and the decision
tree highlighted "No meaningful predictors found".

The omnibus here is a likelihood-ratio test, not Wald: with a near-separated
level the Wald statistic collapses (Hauck-Donner), and the module already
prefers likelihood-ratio inference elsewhere (the Firth branch reports penalized
LR p-values rather than Wald).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

from analysis.clinical_models import LogisticRegressionModel


def _audit_df():
    """a == b by construction, c has a 97% event rate."""
    rng = np.random.default_rng(20260722)
    n = 60
    grp = np.array(["a"] * n + ["b"] * n + ["c"] * n)
    outcome = rng.binomial(1, np.where(grp == "c", 0.95, 0.5))
    return pd.DataFrame({"Grp": grp, "Outcome": outcome})


def _reference_lr(df):
    """Likelihood-ratio test for the Grp term, computed independently."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    full = smf.glm("Outcome ~ C(Grp)", df, family=sm.families.Binomial()).fit()
    null = smf.glm("Outcome ~ 1", df, family=sm.families.Binomial()).fit()
    stat = 2 * (full.llf - null.llf)
    return float(stat), 2, float(chi2.sf(stat, 2))


def test_the_scenario_really_has_a_nonsignificant_first_coefficient():
    """Premise guard."""
    df = _audit_df()
    model = LogisticRegressionModel()
    model.fit(df, dv="Outcome", predictors=["Grp"])
    rows = model.as_results_dict()["odds_ratios"]
    by_param = {r["parameter"]: r["p_value"] for r in rows}
    assert by_param["C(Grp)[T.b]"] > 0.05, "first dummy contrast must be n.s. here"
    assert by_param["C(Grp)[T.c]"] < 0.01, "third group must be strongly significant"


def test_headline_pvalue_is_the_omnibus_not_the_first_coefficient():
    df = _audit_df()
    stat, df_num, p_omnibus = _reference_lr(df)

    model = LogisticRegressionModel()
    model.fit(df, dv="Outcome", predictors=["Grp"])
    res = model.as_results_dict()

    assert res["p_value"] == pytest.approx(p_omnibus, rel=1e-6, abs=1e-300)
    assert res["statistic"] == pytest.approx(stat, rel=1e-6)
    assert res["statistic_type"] == "chi2"
    assert res["p_value"] < 1e-8, (
        "group c has a 97% event rate; a headline p near 1.0 means the first "
        "dummy coefficient leaked into the headline again"
    )


def test_binary_predictor_omnibus_agrees_with_its_single_coefficient():
    """Control: with two levels the LR test and the single coefficient test the
    same hypothesis, so both must reach the same verdict."""
    rng = np.random.default_rng(11)
    n = 80
    grp = np.array(["a"] * n + ["b"] * n)
    outcome = rng.binomial(1, np.where(grp == "b", 0.8, 0.35))
    df = pd.DataFrame({"Grp": grp, "Outcome": outcome})

    model = LogisticRegressionModel()
    model.fit(df, dv="Outcome", predictors=["Grp"])
    res = model.as_results_dict()
    coef_p = {r["parameter"]: r["p_value"] for r in res["odds_ratios"]}["C(Grp)[T.b]"]

    assert res["p_value"] < 0.05
    assert coef_p < 0.05


def test_firth_variant_reports_a_penalized_omnibus():
    """Under complete separation the model switches to Firth; the headline must
    still be an omnibus for the predictor, and it must stay significant."""
    df = pd.DataFrame({"Grp": ["a"] * 20 + ["b"] * 20,
                       "Outcome": [0] * 20 + [1] * 20})
    model = LogisticRegressionModel()
    model.fit(df, dv="Outcome", predictors=["Grp"])
    res = model.as_results_dict()

    assert res["model_variant"] == "Firth Penalized Likelihood"
    assert res["statistic_type"] == "chi2"
    assert res["p_value"] is not None and res["p_value"] < 0.001
