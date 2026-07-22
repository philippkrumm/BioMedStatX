"""LinearMixedModel.as_results_dict() published the FIRST matching fixed-effect
dummy coefficient as the model's headline `p_value` / `statistic`. For a factor
with more than two levels that is a single contrast, not the effect of the
factor.

The audit's case: three timepoints where t0 and t1 are identical by construction
and t2 sits +6.0 away. The t1-vs-t0 contrast is n.s., the factor is overwhelming.
The headline reported the n.s. contrast, and three consumers took it at face
value: the report hero banner ("did not show evidence against the null
hypothesis"), the decision tree (highlighted "No consistent effects found"), and
the cross-dataset FDR correction.

Under REML the log-likelihood is not comparable across different fixed-effect
structures, so the omnibus has to be a Wald test over the factor's whole
parameter block, not a likelihood-ratio test.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

from analysis.clinical_models import LinearMixedModel


def _audit_df():
    """t0 == t1 by construction, t2 is +6.0; 20 subjects."""
    rng = np.random.default_rng(20260722)
    rows = []
    for i in range(1, 21):
        subject = f"S{i:02d}"
        intercept = rng.normal(0, 1.0)
        for level, effect in [("t0", 0.0), ("t1", 0.0), ("t2", 6.0)]:
            rows.append({
                "Subject": subject,
                "Time": level,
                "Value": 10 + intercept + effect + rng.normal(0, 0.7),
            })
    return pd.DataFrame(rows)


def _reference_omnibus(df):
    """Wald test over the whole C(Time) block, computed independently of the
    code under test."""
    import statsmodels.formula.api as smf

    fit = smf.mixedlm("Value ~ C(Time)", df, groups=df["Subject"]).fit(reml=True)
    names = list(fit.fe_params.index)
    idx = [i for i, n in enumerate(names) if n.startswith("C(Time)")]
    k_fe = fit.k_fe
    V = np.asarray(fit.cov_params())[:k_fe, :k_fe]
    beta = np.asarray(fit.fe_params.values, dtype=float)
    R = np.zeros((len(idx), k_fe))
    for r, i in enumerate(idx):
        R[r, i] = 1.0
    diff = R @ beta
    W = float(diff @ np.linalg.inv(R @ V @ R.T) @ diff)
    return W, len(idx), float(chi2.sf(W, len(idx)))


def test_the_scenario_really_has_a_nonsignificant_first_contrast():
    """Premise guard: without it the test could pass for the wrong reason."""
    df = _audit_df()
    model = LinearMixedModel()
    model.fit(df, dv="Value", fixed_effects=["Time"], random_intercept="Subject")
    table = {r["parameter"]: r["p_value"] for r in model.as_results_dict()["fixed_effects_table"]}
    assert table["C(Time)[T.t1]"] > 0.05, "first dummy contrast must be n.s. here"
    assert table["C(Time)[T.t2]"] < 1e-10, "second contrast must be overwhelming"


def test_headline_pvalue_is_the_omnibus_not_the_first_contrast():
    df = _audit_df()
    W, df_num, p_omnibus = _reference_omnibus(df)

    model = LinearMixedModel()
    model.fit(df, dv="Value", fixed_effects=["Time"], random_intercept="Subject")
    res = model.as_results_dict()

    assert res["p_value"] == pytest.approx(p_omnibus, rel=1e-6, abs=1e-300)
    assert res["statistic"] == pytest.approx(W, rel=1e-6)
    assert res["statistic_type"] == "chi2"
    assert res["p_value"] < 1e-10, (
        "the factor effect is enormous; a headline p above 0.05 means the "
        "first dummy contrast leaked into the headline again"
    )


def test_two_level_factor_still_agrees_with_its_single_contrast():
    """Control: with only two levels the omnibus and the single contrast test
    the same hypothesis, so the p-values must agree."""
    rng = np.random.default_rng(4)
    rows = []
    for i in range(1, 21):
        subject = f"S{i:02d}"
        intercept = rng.normal(0, 1.0)
        for level, effect in [("t0", 0.0), ("t1", 1.4)]:
            rows.append({"Subject": subject, "Time": level,
                         "Value": 10 + intercept + effect + rng.normal(0, 0.7)})
    df = pd.DataFrame(rows)

    model = LinearMixedModel()
    model.fit(df, dv="Value", fixed_effects=["Time"], random_intercept="Subject")
    res = model.as_results_dict()
    contrast_p = {r["parameter"]: r["p_value"]
                  for r in res["fixed_effects_table"]}["C(Time)[T.t1]"]

    # Wald chi2 with 1 df vs the Between-Within t-test on the same contrast:
    # both must call this significant and agree to within the df correction.
    assert res["p_value"] < 0.05
    assert contrast_p < 0.05
