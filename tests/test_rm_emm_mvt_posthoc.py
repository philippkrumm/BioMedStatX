"""Level-vs-baseline EMM + multivariate-t post-hoc for pure RM ANOVA (P2).

Pins the closed form against the classical univariate RM error term and checks
that the contrasts are genuinely vs-baseline with a joint mvt adjustment.
"""
import numpy as np
import pandas as pd
import pytest

from analysis.emm_posthoc import (
    rm_strata, rm_dunnett_emm_mvt, UnsupportedDesignError,
)


def _rm_df(n=10, levels=("T0", "T1", "T2", "T3"), seed=7):
    rng = np.random.default_rng(seed)
    eff = {"T0": 0.0, "T1": 1.5, "T2": 2.0, "T3": 0.2}
    rows = []
    for s in range(n):
        subj_intercept = rng.normal(0, 2.0)  # between-subject variance
        for lvl in levels:
            y = subj_intercept + eff[lvl] + rng.normal(0, 1.0)
            rows.append({"y": y, "subject": f"S{s:02d}", "time": lvl})
    return pd.DataFrame(rows)


def test_ms_res_matches_statsmodels_anovarm():
    from statsmodels.stats.anova import AnovaRM

    df = _rm_df()
    s = rm_strata(df, dv="y", subject="subject", within="time")
    aov = AnovaRM(df, depvar="y", subject="subject", within=["time"]).fit()
    # AnovaRM exposes F and df; reconstruct MS_error from the within SS table.
    # Cross-check via the residual: MS_error = SS_resid / df_resid.
    # statsmodels stores the anova table; pull the within-factor F + dfs.
    tbl = aov.anova_table
    f_val = float(tbl.loc["time", "F Value"])
    df_num = float(tbl.loc["time", "Num DF"])
    df_den = float(tbl.loc["time", "Den DF"])
    assert s.df_res == int(df_den)
    # MS_factor / MS_error = F  =>  MS_error = MS_factor / F
    grand = df["y"].mean()
    cell = df.groupby("time")["y"].mean()
    ss_factor = s.n * sum((cell - grand) ** 2)
    ms_factor = ss_factor / df_num
    ms_error_expected = ms_factor / f_val
    assert s.ms_res == pytest.approx(ms_error_expected, rel=1e-6)


def test_contrast_se_and_df_closed_form():
    df = _rm_df()
    s = rm_strata(df, dv="y", subject="subject", within="time")
    contrasts = rm_dunnett_emm_mvt(df, dv="y", subject="subject",
                                   within="time", control_level="T0")
    expected_se = np.sqrt(2.0 * s.ms_res / s.n)
    assert len(contrasts) == 3  # W-1, baseline excluded
    for c in contrasts:
        assert c["control"] == "T0"
        assert c["se"] == pytest.approx(expected_se, abs=1e-9)
        assert c["df"] == (10 - 1) * (4 - 1)
        assert 0.0 <= c["p_value"] <= 1.0


def test_estimate_equals_cell_mean_difference():
    df = _rm_df()
    cell = df.groupby("time")["y"].mean()
    contrasts = rm_dunnett_emm_mvt(df, dv="y", subject="subject",
                                   within="time", control_level="T0")
    by_lvl = {c["level"]: c for c in contrasts}
    for lvl in ("T1", "T2", "T3"):
        assert by_lvl[lvl]["estimate"] == pytest.approx(cell[lvl] - cell["T0"], abs=1e-9)


def test_mvt_less_conservative_than_bonferroni():
    from scipy.stats import t as student_t
    from analysis.emm_posthoc import _mvt_adjusted_p

    df = _rm_df()
    s = rm_strata(df, dv="y", subject="subject", within="time")
    contrasts = rm_dunnett_emm_mvt(df, dv="y", subject="subject",
                                   within="time", control_level="T0")
    k = len(contrasts)

    # scipy integrates the multivariate-t CDF by Monte Carlo, so every adjusted
    # p carries ~1e-4 of absolute noise. Deep in the tail the mvt bound and the
    # Bonferroni bound converge (1-(1-p)^k -> k*p), so the true gap between them
    # shrinks below that noise and its SIGN stops being resolvable: measured on
    # this fixture, the gap is +18% at t=2.2 but -3% at t=5.2, decided purely by
    # the RNG draw. The former `<= bound + 1e-9` form therefore tested the draw,
    # not the method — it failed for 10 of 40 Monte-Carlo seeds. Raising maxpts
    # is no remedy either (5x the budget costs ~4300 ms per call instead of 1 ms).
    # So the bound is asserted down to the noise floor here. Note this loop is a
    # COARSE sanity bound only — at this fixture's steep t-values (1.93/4.23/4.60)
    # both p and the bound are far below MC_SLACK, so it would also pass with a
    # broken correlation structure. It guards gross breakage (inverted CDF, df
    # blow-up); the correlation-aware property is guarded by the strict check
    # below, which is the real correctness assertion of this test.
    MC_SLACK = 5e-4          # ~8x the worst excess observed across 40 seeds
    for c in contrasts:
        p_raw = 2.0 * student_t.sf(abs(c["t"]), s.df_res)
        assert c["p_value"] <= min(1.0, k * p_raw) + MC_SLACK

    # ...and the property itself is asserted where it IS resolvable. At a
    # moderate t the correlation benefit dwarfs the noise: p_mvt/Bonferroni sits
    # at a stable 0.835-0.837 over 30 seeds, while dropping the equicorrelation
    # (shape=I, i.e. the Sidak case) moves it to 0.945-0.946. The threshold below
    # is placed in that gap and was mutation-checked against exactly that break —
    # a looser 0.95 would NOT have caught it.
    t_moderate = 2.2
    p_mvt = _mvt_adjusted_p([t_moderate] * k, float(s.df_res))[0]
    bonferroni = min(1.0, k * 2.0 * student_t.sf(t_moderate, s.df_res))
    assert p_mvt < 0.90 * bonferroni


def test_incomplete_design_raises():
    df = _rm_df()
    df = df[~((df.subject == "S00") & (df.time == "T2"))]
    with pytest.raises(UnsupportedDesignError):
        rm_strata(df, dv="y", subject="subject", within="time")
