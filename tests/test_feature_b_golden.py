"""Golden + structural guards for the Feature-B Mixed default post-hoc.

Three layers, because the SC2 bug class is "right numbers, wrong routing" and a
value check alone cannot catch it:

1. VALUES vs R  — each contrast's raw t/df/p must match the R golden for the
   error term the app actually routed to (emmeans-pooled or isolated/Welch).
2. ROUTING      — the *set* of contrasts is asserted hard against the known
   dataset structure (within-per-group + between-per-timepoint, no cross-cells),
   and the pooled/isolated assignment is asserted hard against an independently
   computed Levene decision.
3. GATING       — driven through the REAL analyzer entry point. The previous
   gating test called mixed_effect_driven_posthoc() directly with hand-supplied
   p-values, which is exactly why a broken pingouin column lookup ("p-unc" vs
   "p_unc") silently disabled the gate in production without any test failing.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from analysis.mixed_simple_effects import simple_main_effects, marginal_within, _subject_diffs
from analysis.posthoc_core import MixedAnovaPostHocAnalyzer

TOL = 1e-8
GOLDEN_DIR = os.path.join("tests", "golden")
GROUPS = ["Ctrl", "Trt"]
TIMES = ["T1", "T2", "T3"]
WPAIRS = [("T1", "T2"), ("T1", "T3"), ("T2", "T3")]


def _load(name):
    with open(os.path.join(GOLDEN_DIR, name)) as fh:
        g = json.load(fh)
    return g, pd.DataFrame(g["data"])


@pytest.fixture(scope="module")
def homo():
    return _load("references_feature_b.json")


@pytest.fixture(scope="module")
def het():
    return _load("references_feature_b_het.json")


def _levene_p(arrays):
    from scipy.stats import levene
    return float(levene(*arrays, center="median")[1])


# ---------------------------------------------------------------- 1. VALUES

def test_seed123_each_contrast_matches_the_golden_for_its_routed_error_term(homo):
    """Canonical seed-123 set routes BOTH ways (T1-T2/T1-T3 isolated, T2-T3 pooled).
    Each contrast must match the golden belonging to the branch it took."""
    g, df = homo
    iso, emm = g["results"]["isolated"], g["results"]["emmeans_pooled"]
    comps = simple_main_effects(df, "y_mixed", "subj", "groupA", "time")
    seen = 0
    for c in comps:
        key = f"{c['group1']}|{c['group2']}"
        if c["comparison_type"] == "within_subject":
            grp, w1 = c["group1"].split(":")
            w2 = c["group2"].split(":")[1]
            if c["error_term"] == "pooled":
                ref = emm["within"][f"{grp}:{w1}-{w2}"]
                assert c["statistic"] == pytest.approx(ref["t_ratio"], abs=TOL)
                assert c["df"] == ref["df"]
            else:
                ref = iso["within"][key]
                assert c["statistic"] == pytest.approx(ref["t"], abs=TOL)
                assert c["df"] == ref["df"]
        else:
            ref = iso["between"][key]
            assert c["statistic"] == pytest.approx(ref["t"], abs=TOL)
            assert c["p_value_raw"] == pytest.approx(ref["p"], abs=TOL)
        seen += 1
    assert seen == 9


def test_heterogeneous_between_takes_welch_and_matches_r(het):
    """PRIMARY guard for the Welch-Satterthwaite branch, which the canonical
    dataset never triggers (all its between contrasts are homogeneous)."""
    g, df = het
    welch = g["results"]["between_y_bet_het"]["welch"]
    comps = simple_main_effects(df, "y_bet_het", "subj", "groupA", "time")
    between = [c for c in comps if c["comparison_type"] == "between_subject"]
    assert len(between) == 3
    for c in between:
        assert c["error_term"] == "isolated", c            # Welch = separate variances
        assert "Welch" in c["test"], c["test"]
        ref = welch[f"{c['group1']}|{c['group2']}"]
        assert c["statistic"] == pytest.approx(ref["t"], abs=TOL)
        assert c["df"] == pytest.approx(ref["df"], abs=1e-6)   # Satterthwaite df
        assert c["p_value_raw"] == pytest.approx(ref["p"], abs=TOL)


def test_heterogeneous_within_splits_isolated_and_pooled_against_r(het):
    """y_win_mix is built so T1-T2 difference variances are decisively
    heterogeneous and T2-T3 decisively homogeneous (not borderline)."""
    g, df = het
    iso = g["results"]["within_y_win_mix"]["isolated"]
    emm = g["results"]["within_y_win_mix"]["emmeans_pooled"]
    comps = simple_main_effects(df, "y_win_mix", "subj", "groupA", "time")
    for c in [x for x in comps if x["comparison_type"] == "within_subject"]:
        grp, w1 = c["group1"].split(":")
        w2 = c["group2"].split(":")[1]
        if (w1, w2) == ("T2", "T3"):
            assert c["error_term"] == "pooled", c["variance_check"]
            ref = emm[f"{grp}:{w1}-{w2}"]
            assert c["statistic"] == pytest.approx(ref["t_ratio"], abs=TOL)
        else:
            assert c["error_term"] == "isolated", c["variance_check"]
            ref = iso[f"{c['group1']}|{c['group2']}"]
            assert c["statistic"] == pytest.approx(ref["t"], abs=TOL)


# --------------------------------------------------------------- 2. ROUTING

@pytest.mark.parametrize("fixture_name,dv", [("homo", "y_mixed"), ("het", "y_win_mix")])
def test_contrast_set_is_exactly_within_per_group_plus_between_per_timepoint(
        fixture_name, dv, request):
    """Hard structural assert against the KNOWN dataset layout: no cross-cell
    pairing, nothing missing, nothing extra."""
    _g, df = request.getfixturevalue(fixture_name)
    comps = simple_main_effects(df, dv, "subj", "groupA", "time")

    expect_within = {(f"{g}:{w1}", f"{g}:{w2}") for g in GROUPS for w1, w2 in WPAIRS}
    expect_between = {(f"Ctrl:{w}", f"Trt:{w}") for w in TIMES}

    got_within = {(c["group1"], c["group2"]) for c in comps
                  if c["comparison_type"] == "within_subject"}
    got_between = {(c["group1"], c["group2"]) for c in comps
                   if c["comparison_type"] == "between_subject"}

    assert got_within == expect_within
    assert got_between == expect_between
    assert len(comps) == 9

    for c in comps:
        g1, t1 = c["group1"].split(":")
        g2, t2 = c["group2"].split(":")
        # never confound both factors in one contrast
        assert not (g1 != g2 and t1 != t2), f"cross-cell contrast: {c['group1']} vs {c['group2']}"
        if c["comparison_type"] == "within_subject":
            assert g1 == g2 and t1 != t2
            assert "Paired" in c["test"]
            assert c["family"] == "within_simple_effect"
        else:
            assert t1 == t2 and g1 != g2
            assert ("Independent" in c["test"]) or ("Welch" in c["test"])
            assert c["family"] == "between_simple_effect"


@pytest.mark.parametrize("fixture_name,dv", [("homo", "y_mixed"), ("het", "y_win_mix")])
def test_pooled_isolated_assignment_matches_independent_levene(fixture_name, dv, request):
    """The error term must follow the Levene decision on the subject differences
    (within) / raw values (between) -- recomputed here, not read from the output."""
    _g, df = request.getfixturevalue(fixture_name)
    comps = simple_main_effects(df, dv, "subj", "groupA", "time")
    for c in comps:
        if c["comparison_type"] == "within_subject":
            w1 = c["group1"].split(":")[1]
            w2 = c["group2"].split(":")[1]
            diffs = [_subject_diffs(df[df.groupA == g], dv, "subj", "time", w1, w2)
                     for g in GROUPS]
            p = _levene_p(diffs)
        else:
            w = c["group1"].split(":")[1]
            sub = df[df.time == w]
            p = _levene_p([sub[sub.groupA == g][dv].to_numpy() for g in GROUPS])
        expected = "pooled" if p > 0.05 else "isolated"
        assert c["error_term"] == expected, (c["group1"], c["group2"], p)
        assert c["variance_check"]["p_value"] == pytest.approx(p, abs=1e-12)


def test_rm_within_only_frame_yields_only_within_contrasts(homo):
    """RM (no between factor): every contrast must be within/paired, and the
    collapsed frame has no between-group variance comparison -> isolated."""
    _g, df = homo
    comps = marginal_within(df, "y_mixed", "subj", "time")
    assert {(c["group1"], c["group2"]) for c in comps} == {(w1, w2) for w1, w2 in WPAIRS}
    for c in comps:
        assert c["comparison_type"] == "within_subject"
        assert c["family"] == "within_marginal"
        assert c["error_term"] == "isolated"
        assert "Paired" in c["test"]


# ---------------------------------------------------------------- 3. GATING

@pytest.mark.parametrize("dv,expected_mode,expected_n", [
    ("y_mixed", "simple_main_effects", 9),      # interaction p = 6.4e-4 -> significant
    ("y_mixed_noia", "marginal_within", 3),     # interaction p = 0.129  -> not significant
])
def test_gating_through_the_real_analyzer_entry_point(homo, dv, expected_mode, expected_n):
    """REGRESSION GUARD for the silent-gate bug: MixedAnovaPostHocAnalyzer read
    pingouin's uncorrected-p column as "p-unc" while pingouin >=0.6 emits
    "p_unc". The KeyError was swallowed, every effect p became None, and the gate
    always fell through to simple main effects -- so y_mixed_noia produced 9
    simple-main-effect contrasts instead of 3 marginal ones. Driving the real
    entry point (not mixed_effect_driven_posthoc directly) is what catches this.
    """
    _g, df = homo
    res = MixedAnovaPostHocAnalyzer.perform_test(
        df, between="groupA", within="time", dv=dv, subject="subj", alpha=0.05)
    assert res.get("gating_applied") is True, res.get("gating_fallback_reason")
    assert res["posthoc_mode"] == expected_mode
    assert len(res["pairwise_comparisons"]) == expected_n
