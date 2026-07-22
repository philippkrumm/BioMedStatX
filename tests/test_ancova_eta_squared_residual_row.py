"""Partial eta-squared picked the wrong row out of the Type III ANOVA table.

The residual row was located with a substring search:

    next((k for k in self.anova_table.index if "residual" in k.lower()), None)

The table index runs [Intercept, factors..., covariates..., Residual], so a
covariate whose *name* contains "residual" comes first and wins. "Residual
Volume" is a standard lung-function parameter, so this is a realistic column
name for this application, not a contrived one.

Audit case: the same data analysed with the covariate named "Volume" reports
partial eta-squared 0.6240; renaming it to "Residual_Volume" drops the reported
effect size to 0.3266, with no warning anywhere.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from analysis.clinical_models import ANCOVAModel


def _base_df(covariate_name):
    rng = np.random.default_rng(11)
    n = 20
    df = pd.DataFrame({
        "Group": ["ctrl"] * n + ["low"] * n + ["high"] * n,
        covariate_name: rng.normal(10, 2, 3 * n),
    })
    df["Value"] = (2 + 0.8 * df[covariate_name]
                   + df["Group"].map({"ctrl": 0.0, "low": 1.5, "high": 3.0})
                   + rng.normal(0, 1, 3 * n))
    return df


def _expected_partial_eta(res):
    """ss_factor / (ss_factor + ss_residual), read off the emitted table."""
    table = {row["source"]: row["sum_sq"] for row in res["anova_table"]}
    ss_factor = table["C(Group, Sum)"]
    ss_residual = table["Residual"]
    return ss_factor / (ss_factor + ss_residual)


def _fit(covariate_name):
    df = _base_df(covariate_name)
    model = ANCOVAModel()
    model.fit(df, dv="Value", between_factors=["Group"], covariates=[covariate_name])
    return model.as_results_dict()


def test_control_covariate_without_the_substring():
    res = _fit("Volume")
    assert res["effect_size"] == pytest.approx(_expected_partial_eta(res), rel=1e-12)


def test_covariate_named_residual_volume_does_not_hijack_the_residual_row():
    res = _fit("Residual_Volume")
    expected = _expected_partial_eta(res)
    assert res["effect_size"] == pytest.approx(expected, rel=1e-12), (
        "the covariate row was matched instead of the model's Residual row"
    )


def test_both_namings_give_the_same_effect_size():
    """The two datasets are numerically identical, so renaming a column must
    not move the effect size at all."""
    benign = _fit("Volume")
    hostile = _fit("Residual_Volume")
    assert benign["effect_size"] == pytest.approx(hostile["effect_size"], rel=1e-12), (
        f"renaming the covariate changed partial eta-squared: "
        f"{benign['effect_size']} vs {hostile['effect_size']}"
    )


def test_effect_size_is_absent_when_there_is_no_residual_row():
    """Guard the fallback: an exact match must not silently pick something else
    if the row is genuinely missing."""
    res = _fit("Volume")
    model = ANCOVAModel()
    df = _base_df("Volume")
    model.fit(df, dv="Value", between_factors=["Group"], covariates=["Volume"])
    model.anova_table = model.anova_table.drop(index="Residual")
    stripped = model.as_results_dict()
    assert stripped["effect_size"] is None
