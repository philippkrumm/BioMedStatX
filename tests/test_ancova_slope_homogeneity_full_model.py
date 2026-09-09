"""Two-Way ANCOVA tested homogeneity of regression slopes on marginal models.

check_regression_slope_homogeneity() fitted one model per (factor, covariate)
pair — `Value ~ C(Factor, Sum) * Cov` — leaving every *other* between factor out.
When the omitted factor interacts with the covariate and is unbalanced across
the tested factor's levels, that interaction leaks into the tested term.

The audit's case: Sex genuinely interacts with Cov, Group does not, and Sex is
unbalanced across Group (90/50/10% male). The marginal model reported
Group:Cov F = 4.55, p = 0.0118, assumption_holds = False. The full model
`Value ~ Group * Sex * Cov` puts Group:Cov at F = 1.39, p = 0.2509 — no
interaction at all.

That false alarm is not cosmetic. `slopes_heterogeneous` gates
run_simple_slopes_and_jn(), so a follow-up analysis ran that should not have,
and the flag reaches the decision tree twice over: once through
slope_homogeneity itself and again through the mere presence of
simple_slopes_analysis.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from analysis.clinical_models import ANCOVAModel


def _leaky_df():
    """Sex x Cov is real, Group x Cov is not, Sex is unbalanced across Group."""
    rng = np.random.default_rng(3)
    records = []
    for group, p_male in [("A", 0.9), ("B", 0.5), ("C", 0.1)]:
        for _ in range(60):
            sex = "m" if rng.random() < p_male else "f"
            cov = rng.normal(10, 3)
            slope = 3.0 if sex == "m" else -3.0
            records.append({"Group": group, "Sex": sex, "Cov": cov,
                            "Value": 5 + slope * cov + rng.normal(0, 2)})
    return pd.DataFrame(records)


def _reference_full_model(df):
    """Type III interaction terms from the full model, computed independently."""
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm

    fit = smf.ols("Value ~ C(Group, Sum) * C(Sex, Sum) * Cov", data=df).fit()
    table = anova_lm(fit, typ=3)
    return {
        "group": float(table.loc["C(Group, Sum):Cov", "PR(>F)"]),
        "sex": float(table.loc["C(Sex, Sum):Cov", "PR(>F)"]),
    }


def test_the_leak_scenario_is_set_up_correctly():
    """Premise guard: Sex x Cov must be real and Group x Cov must not be."""
    truth = _reference_full_model(_leaky_df())
    assert truth["sex"] < 1e-20, "Sex x Cov must be overwhelming here"
    assert truth["group"] > 0.05, "Group x Cov must be absent in the full model"


def test_group_slope_verdict_matches_the_full_model():
    df = _leaky_df()
    truth = _reference_full_model(df)

    model = ANCOVAModel()
    model.fit(df, dv="Value", between_factors=["Group", "Sex"], covariates=["Cov"])
    homogeneity = model.check_regression_slope_homogeneity()

    group = homogeneity["Group:Cov"]
    assert group["p_value"] == pytest.approx(truth["group"], rel=1e-6), (
        f"Group:Cov still tested on a marginal model "
        f"(reported p={group['p_value']:.4g}, full model p={truth['group']:.4g})"
    )
    assert group["assumption_holds"] is True

    sex = homogeneity["Sex:Cov"]
    assert sex["p_value"] == pytest.approx(truth["sex"], rel=1e-6)
    assert sex["assumption_holds"] is False


def test_false_alarm_no_longer_triggers_the_follow_up_analysis():
    """The Group false positive must not by itself pull in simple slopes."""
    df = _leaky_df()
    model = ANCOVAModel()
    model.fit(df, dv="Value", between_factors=["Group", "Sex"], covariates=["Cov"])
    res = model.as_results_dict()

    # Sex genuinely violates the assumption, so the flag stays True overall —
    # but the Group verdict, which is what the false alarm was about, is clean.
    assert res["slope_homogeneity"]["Group:Cov"]["assumption_holds"] is True


def test_single_factor_ancova_is_unchanged():
    """Control: with one between factor the marginal model *is* the full model,
    so the numbers must not move."""
    rng = np.random.default_rng(21)
    n = 40
    df = pd.DataFrame({"Group": ["ctrl"] * n + ["low"] * n + ["high"] * n,
                       "Cov": rng.normal(10, 2, 3 * n)})
    df["Value"] = (2 + 0.8 * df["Cov"]
                   + df["Group"].map({"ctrl": 0.0, "low": 1.5, "high": 3.0})
                   + rng.normal(0, 1, 3 * n))

    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    fit = smf.ols("Value ~ C(Group, Sum) * Cov", data=df).fit()
    expected = float(anova_lm(fit, typ=3).loc["C(Group, Sum):Cov", "PR(>F)"])

    model = ANCOVAModel()
    model.fit(df, dv="Value", between_factors=["Group"], covariates=["Cov"])
    got = model.check_regression_slope_homogeneity()["Group:Cov"]
    assert got["p_value"] == pytest.approx(expected, rel=1e-9)
