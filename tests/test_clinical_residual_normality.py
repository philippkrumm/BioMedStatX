"""ANCOVA and LMM never tested residual normality and never emitted residuals.

The report row said "Residual Normality — Assessed visually via Q-Q plot", and
the Q-Q plot builder (report_summaries._build_assumption_visuals) fell through
its `model_residuals` -> `residuals` -> `raw_data` chain to the last option,
plotting the raw dependent variable pooled across groups. With group means far
apart that pooled distribution is multi-modal by construction, so a user
following the report's own instruction concluded the assumption was violated
when the residuals were textbook normal.

Audit case: residuals drawn N(0,1), group means 0 / 25 / 50. Shapiro on the
pooled raw DV gives p = 3.6e-10; Shapiro on the model residuals gives p = 0.76.

The fix routes the residuals of the model that was ACTUALLY fitted (covariate
and random effects included) through the app's shared Shapiro input contract,
`validators.validate_residuals_for_shapiro`, and emits them in the
`normality_tests` shape report_summaries already reads — which is what makes the
previously dead ANCOVA branch there come alive.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from analysis.clinical_models import ANCOVAModel, LinearMixedModel
from export.report_summaries import _SummariesMixin


def _ancova_df():
    rng = np.random.default_rng(5)
    n = 40
    df = pd.DataFrame({
        "Group": ["ctrl"] * n + ["low"] * n + ["high"] * n,
        "Cov": rng.normal(10, 2, 3 * n),
    })
    resid = rng.normal(0, 1.0, 3 * n)
    df["Value"] = (0.5 * df["Cov"]
                   + df["Group"].map({"ctrl": 0.0, "low": 25.0, "high": 50.0})
                   + resid)
    return df


def _lmm_df():
    rng = np.random.default_rng(5)
    rows = []
    for i in range(1, 21):
        subject, intercept = f"S{i:02d}", rng.normal(0, 1.0)
        for level, effect in [("t0", 0.0), ("t1", 25.0), ("t2", 50.0)]:
            rows.append({"Subject": subject, "Time": level,
                         "Value": 10 + intercept + effect + rng.normal(0, 0.8)})
    return pd.DataFrame(rows)


def test_pooled_raw_data_really_would_look_non_normal():
    """Premise guard: the scenario has to be one where the old behaviour lies."""
    df = _ancova_df()
    pooled = df["Value"].values
    assert stats.shapiro(pooled).pvalue < 1e-6, (
        "pooled raw DV must look strongly non-normal for this test to mean anything"
    )


def test_ancova_emits_model_residuals_and_tests_them():
    df = _ancova_df()
    model = ANCOVAModel()
    model.fit(df, dv="Value", between_factors=["Group"], covariates=["Cov"])
    res = model.as_results_dict()

    assert "model_residuals" in res, (
        "the Q-Q plot fallback chain looks for model_residuals first; without it "
        "the plot silently falls back to the pooled raw DV"
    )
    expected = np.asarray(model.result.resid, dtype=float)
    assert np.allclose(np.asarray(res["model_residuals"], dtype=float), expected)

    nt = res.get("normality_tests") or {}
    assert nt, "ANCOVA must publish normality_tests"
    payload = next(iter(nt.values()))
    shapiro_ref = stats.shapiro(expected)
    assert payload["p_value"] == pytest.approx(float(shapiro_ref.pvalue), rel=1e-9)
    assert payload["statistic"] == pytest.approx(float(shapiro_ref.statistic), rel=1e-9)
    assert payload["is_normal"] is True
    assert payload["p_value"] > 0.1, (
        "residuals are normal by construction; a tiny p means the pooled raw "
        "data got tested again"
    )


def test_lmm_emits_model_residuals_and_tests_them():
    df = _lmm_df()
    model = LinearMixedModel()
    model.fit(df, dv="Value", fixed_effects=["Time"], random_intercept="Subject")
    res = model.as_results_dict()

    assert "model_residuals" in res
    nt = res.get("normality_tests") or {}
    assert nt, "LMM must publish normality_tests"
    payload = next(iter(nt.values()))
    assert isinstance(payload.get("p_value"), float)
    assert payload.get("statistic") is not None


def test_report_shows_a_real_statistic_not_a_visual_disclaimer():
    for build, kwargs in [
        (ANCOVAModel, dict(dv="Value", between_factors=["Group"], covariates=["Cov"])),
        (LinearMixedModel, dict(dv="Value", fixed_effects=["Time"],
                                random_intercept="Subject")),
    ]:
        df = _ancova_df() if build is ANCOVAModel else _lmm_df()
        model = build()
        model.fit(df, **kwargs)
        res = model.as_results_dict()

        rows = _SummariesMixin._build_assumption_summary(res).get("rows", [])
        normality_rows = [r for r in rows if "Normal" in r["name"]]
        assert normality_rows, f"{build.__name__}: no normality row rendered"
        row = normality_rows[0]
        assert row["statistic"] not in ("N/A", "—"), (
            f"{build.__name__}: report still shows a placeholder instead of a "
            f"computed Shapiro-Wilk statistic ({row!r})"
        )
        assert "Assessed visually" not in str(row.get("status_label", "")), (
            f"{build.__name__}: report still defers to the Q-Q plot instead of "
            "reporting the test it now actually runs"
        )


def test_qq_plot_is_built_from_residuals_not_pooled_raw_data():
    df = _ancova_df()
    model = ANCOVAModel()
    model.fit(df, dv="Value", between_factors=["Group"], covariates=["Cov"])
    res = model.as_results_dict()
    # emulate the pipeline attaching grouped raw data alongside the results
    res["raw_data"] = {g: sub["Value"].tolist()
                       for g, sub in df.groupby("Group")}

    assert _SummariesMixin._build_assumption_visuals(res) is not None
    # The chain in _build_assumption_visuals prefers model_residuals; proving the
    # key is present and correct is what decides which branch it takes.
    assert "model_residuals" in res
    assert np.allclose(np.asarray(res["model_residuals"], dtype=float),
                       np.asarray(model.result.resid, dtype=float))
