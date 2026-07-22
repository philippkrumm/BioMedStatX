"""The Between-Within degrees of freedom were assigned by substring match.

as_results_dict() classifies every predictor into `between_cols` / `within_cols`
correctly (a column is between-subject when it is constant inside every
subject). It then throws that result away and re-derives the same decision for
each *parameter* with

    for col in within_cols:
        if col in param_name:      # substring, not membership

so a between-subject covariate whose name merely contains a within factor's
name is treated as within. The audit's case: fixed effect "Dose" (within),
covariate "Dose_base" (between, constant per subject). 'Dose' in 'Dose_base' is
True, so Dose_base received df = n_obs - n_subjects - n_within = 39 instead of
the correct n_subjects - 1 - n_between = 18, changing its p-value from 0.1108
to 0.1015.

This is the third independent derivation of the same between/within split in
this file, and the only one that is wrong.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from analysis.clinical_models import LinearMixedModel


def _collision_df():
    """'Dose' varies within subject; 'Dose_base' is constant per subject."""
    rng = np.random.default_rng(7)
    rows = []
    for i in range(1, 21):
        subject = f"S{i:02d}"
        intercept = rng.normal(0, 1.5)
        baseline = 40 + (i % 10)
        for level, effect in [("d0", 0.0), ("d1", 1.2), ("d2", 2.4)]:
            rows.append({
                "Subject": subject, "Dose": level, "Dose_base": baseline,
                "Value": 10 + intercept + effect + 0.02 * baseline + rng.normal(0, 0.8),
            })
    return pd.DataFrame(rows)


def _renamed_df():
    """Numerically identical, but the covariate name no longer collides."""
    return _collision_df().rename(columns={"Dose_base": "Baseline"})


def _fit(df, covariate):
    model = LinearMixedModel()
    model.fit(df, dv="Value", fixed_effects=["Dose"], random_intercept="Subject",
              covariates=[covariate])
    return model.as_results_dict()


def _df_of(res, parameter):
    return {r["parameter"]: r["df"] for r in res["fixed_effects_table"]}[parameter]


def test_classification_itself_was_never_wrong():
    """Premise guard: the sets are right; only the df lookup is not."""
    res = _fit(_collision_df(), "Dose_base")
    assert res["between_effects"] == ["Dose_base"]
    assert res["within_effects"] == ["Dose"]


def test_between_covariate_gets_the_between_df_despite_the_name_collision():
    res = _fit(_collision_df(), "Dose_base")
    n_subj, n_obs = res["n_subjects"], res["n_observations"]
    expected_between = n_subj - 1 - len(res["between_effects"])
    expected_within = n_obs - n_subj - len(res["within_effects"])

    assert _df_of(res, "Dose_base") == expected_between, (
        f"between covariate got df={_df_of(res, 'Dose_base')}; the within df "
        f"would be {expected_within}"
    )
    assert _df_of(res, "C(Dose)[T.d1]") == expected_within


def test_renaming_the_covariate_changes_nothing():
    collision = _fit(_collision_df(), "Dose_base")
    renamed = _fit(_renamed_df(), "Baseline")

    assert _df_of(collision, "Dose_base") == _df_of(renamed, "Baseline")
    collision_p = {r["parameter"]: r["p_value"]
                   for r in collision["fixed_effects_table"]}["Dose_base"]
    renamed_p = {r["parameter"]: r["p_value"]
                 for r in renamed["fixed_effects_table"]}["Baseline"]
    assert collision_p == pytest.approx(renamed_p, rel=1e-12), (
        f"renaming the covariate moved its p-value: {collision_p} vs {renamed_p}"
    )


def test_within_factor_keeps_the_within_df_when_names_do_not_collide():
    """Control: the ordinary case must be unaffected."""
    res = _fit(_renamed_df(), "Baseline")
    n_subj, n_obs = res["n_subjects"], res["n_observations"]
    assert _df_of(res, "C(Dose)[T.d1]") == n_obs - n_subj - len(res["within_effects"])
    assert _df_of(res, "Baseline") == n_subj - 1 - len(res["between_effects"])
