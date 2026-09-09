"""Effect-driven Mixed-ANOVA post-hoc (feature B): simple main effects +
marginal-mean contrasts, gated on which omnibus effects are significant.
"""
import numpy as np
import pandas as pd
import pytest

pg = pytest.importorskip("pingouin")
from scipy.stats import ttest_ind, ttest_rel  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

from analysis.mixed_simple_effects import (  # noqa: E402
    simple_main_effects, marginal_within, marginal_between, mixed_effect_driven_posthoc,
)


def _frame(seed=3, groups=("WT", "KO"), times=("0h", "6h", "24h"), n=8):
    rng = np.random.default_rng(seed)
    rows = []
    for g in groups:
        for s in range(n):
            se = rng.normal(0, 1)
            for ti, t in enumerate(times):
                rows.append({"subj": f"{g}{s}", "grp": g, "time": t,
                             "y": se + (1.5 if g == "KO" else 0) + 0.5 * ti + rng.normal(0, 1)})
    return pd.DataFrame(rows)


def test_no_cross_cell_comparisons():
    comps = simple_main_effects(_frame(), "y", "subj", "grp", "time")
    # 2 groups x C(3,2)=3 within + 3 times x C(2,2)=1 between = 6 + 3 = 9
    assert len(comps) == 9
    for c in comps:
        if c["comparison_type"] == "within_subject":
            # same group, different time  ("g:t")
            assert c["group1"].split(":")[0] == c["group2"].split(":")[0]
            assert c["group1"].split(":")[1] != c["group2"].split(":")[1]
        else:
            # same time, different group  ("g:t")
            assert c["group1"].split(":")[1] == c["group2"].split(":")[1]
            assert c["group1"].split(":")[0] != c["group2"].split(":")[0]


def test_raw_values_and_correction_match_the_routed_error_term():
    """Raw test values + per-family Holm-Sidak, against the error term the app ROUTED.

    CHANGED (assumption-driven error term): the previous version of this test
    hard-wired one convention — ttest_rel (isolated) for every within contrast and
    ttest_ind(equal_var=True) (Student) for every between contrast. Feature B now
    picks the error term per contrast via Levene on the subject differences
    (within) / on the raw values (between). Those two asserts are REPLACED, not
    dropped: the raw-value and Holm-Sidak checks still run, but reproduce whichever
    term the app reports, and the routing decision itself is independently
    re-derived from Levene here.
    """
    from scipy.stats import levene, ttest_1samp, t as tdist

    df = _frame()
    comps = simple_main_effects(df, "y", "subj", "grp", "time")

    def _diffs(grp, t1, t2):
        sub = df[df.grp == grp]
        a = sub[sub.time == t1].sort_values("subj").y.to_numpy()
        b = sub[sub.time == t2].sort_values("subj").y.to_numpy()
        return a - b

    for family in ("within_simple_effect", "between_simple_effect"):
        fam = [c for c in comps if c["family"] == family]
        assert fam, family
        raw = []
        for c in fam:
            g1, g2 = c["group1"], c["group2"]
            if c["comparison_type"] == "within_subject":
                grp, t1 = g1.split(":")
                t2 = g2.split(":")[1]
                per = {g: _diffs(g, t1, t2) for g in sorted(df.grp.unique())}
                lev_p = float(levene(*per.values(), center="median")[1])
                expect_term = "pooled" if lev_p > 0.05 else "isolated"
                assert c["error_term"] == expect_term, (g1, g2, lev_p, c["error_term"])
                assert c["variance_check"]["p_value"] == pytest.approx(lev_p, abs=1e-12)
                d = per[grp]
                if expect_term == "pooled":
                    dfp = sum(len(x) - 1 for x in per.values())
                    s2p = sum((len(x) - 1) * np.var(x, ddof=1) for x in per.values()) / dfp
                    t_stat = np.mean(d) / np.sqrt(s2p / len(d))
                    raw.append(float(2 * tdist.sf(abs(t_stat), dfp)))
                    assert c["df"] == dfp
                else:
                    raw.append(float(ttest_1samp(d, 0.0)[1]))
                    assert c["df"] == len(d) - 1
            else:
                t = g1.split(":")[1]
                g1n, g2n = g1.split(":")[0], g2.split(":")[0]
                sub = df[df.time == t]
                a = sub[sub.grp == g1n].y.to_numpy()
                b = sub[sub.grp == g2n].y.to_numpy()
                lev_p = float(levene(a, b, center="median")[1])
                equal_var = lev_p > 0.05
                assert c["error_term"] == ("pooled" if equal_var else "isolated")
                assert c["variance_check"]["p_value"] == pytest.approx(lev_p, abs=1e-12)
                raw.append(float(ttest_ind(a, b, equal_var=equal_var)[1]))

        # the app must expose exactly the raw p-values it then corrected
        np.testing.assert_allclose([c["p_value_raw"] for c in fam], raw, atol=1e-12)
        expected = multipletests(raw, method="holm-sidak")[1]
        np.testing.assert_allclose([c["p_value"] for c in fam], expected, atol=1e-9)


def test_within_raw_stat_matches_pingouin():
    df = _frame(groups=("WT",))  # single group -> pure within, 1 subject group
    pt = pg.pairwise_tests(data=df, dv="y", within="time", subject="subj")
    ref = {frozenset((str(r["A"]), str(r["B"]))): float(r["p_unc"]) for _, r in pt.iterrows()}
    # marginal_within over a single group == within pairwise; recover raw by
    # re-running holm-sidak inversely is fiddly, so check the 1-group case where
    # the between family is empty and within raw matches when only one pair.
    two = _frame(groups=("WT",), times=("0h", "6h"))
    out = marginal_within(two, "y", "subj", "time")
    assert len(out) == 1  # single pair -> holm leaves p unchanged
    ptt = pg.pairwise_tests(data=two, dv="y", within="time", subject="subj")
    assert out[0]["p_value"] == pytest.approx(float(ptt.iloc[0]["p_unc"]), abs=1e-9)


def test_order_independent_of_row_order():
    df = _frame()
    a = simple_main_effects(df, "y", "subj", "grp", "time")
    shuffled = df.sample(frac=1.0, random_state=1).reset_index(drop=True)
    b = simple_main_effects(shuffled, "y", "subj", "grp", "time")
    key = lambda comps: [(c["group1"], c["group2"], round(c["p_value"], 12)) for c in comps]
    assert key(a) == key(b)


def test_gating_routes_by_significant_effect():
    df = _frame()
    assert mixed_effect_driven_posthoc(df, "y", "subj", "grp", "time",
                                       interaction_p=1e-4, within_p=1e-9, between_p=1e-3)[1] == "simple_main_effects"
    assert mixed_effect_driven_posthoc(df, "y", "subj", "grp", "time",
                                       interaction_p=0.5, within_p=1e-9, between_p=0.5)[1] == "marginal_within"
    assert mixed_effect_driven_posthoc(df, "y", "subj", "grp", "time",
                                       interaction_p=0.5, within_p=0.5, between_p=1e-3)[1] == "marginal_between"
    assert mixed_effect_driven_posthoc(df, "y", "subj", "grp", "time",
                                       interaction_p=0.5, within_p=0.5, between_p=0.5)[1] == "none"
    # no effect info -> defaults to simple main effects (never cross-cells)
    assert mixed_effect_driven_posthoc(df, "y", "subj", "grp", "time")[1] == "simple_main_effects"
