"""SC2 fix (pre-2.0 audit): the RM/Mixed default post-hoc must NOT be the
hand-rolled studentized-range Tukey.

The hand-rolled Tukey (posthoc_core.py) fed the paired-t df (n-1) and a per-pair
SD into scipy's studentized_range, giving systematically too-conservative
p-values on the DEFAULT path a normal user hits. It is also incoherent with the
omnibus, which corrects for sphericity by default (Greenhouse-Geisser) — a
studentized-range post-hoc assumes the sphericity the omnibus just corrected for.

Fix (Opt 2): drop the hand-rolled formula; default RM/Mixed to `paired_custom`
(Holm-Šidák over per-pair tests that are type-correct — paired-t for
within-subject pairs, independent-t for between-subject pairs). These tests lock
that behavior and give the path its first external (pingouin) numeric anchor.
"""
import numpy as np
import pandas as pd
import pytest

pg = pytest.importorskip("pingouin")

from statistical_testing.engines.advanced_posthoc import AdvancedPostHocEngine  # noqa: E402


def _rm_frame():
    rng = np.random.default_rng(4)
    k, n = 4, 10
    rows = []
    subj_effect = rng.normal(0, 1.0, n)
    for s in range(n):
        for c in range(k):
            rows.append({
                "subj": f"s{s}",
                "cond": f"c{c}",
                "y": subj_effect[s] + 0.5 * c + rng.normal(0, 1.0),
            })
    return pd.DataFrame(rows)


def _run_default_rm_posthoc(df):
    """Drive the real default path: no posthoc_method_callback => engine uses
    its own default_method (the thing the fix changes)."""
    engine = AdvancedPostHocEngine()
    res = engine.execute({
        "mode": "advanced_parametric",
        "test": "repeated_measures_anova",
        "df_transformed": df,
        "dv": "y",
        "subject": "subj",
        "within": ["cond"],
        "alpha": 0.05,
    })
    return res.metadata


def test_rm_default_posthoc_is_not_handrolled_tukey():
    meta = _run_default_rm_posthoc(_rm_frame())
    comps = meta.get("pairwise_comparisons", [])
    assert comps, f"no pairwise comparisons produced: {meta}"
    # RED before fix: default_method='tukey' => posthoc_test "Tukey HSD (RM)" and
    # each comp["test"] == "Paired t-test (Tukey HSD (RM))".
    assert "Tukey" not in str(meta.get("posthoc_test", "")), meta.get("posthoc_test")
    for c in comps:
        label = str(c.get("test", ""))
        assert "Tukey" not in label, f"default RM post-hoc still hand-rolled Tukey: {label}"
        assert "Holm" in label, f"expected Holm-Šidák default, got: {label}"


def test_rm_default_posthoc_matches_pingouin_paired_reference():
    """External anchor: the app's default RM pairwise p-values equal
    pingouin's raw paired-t p-values put through the documented Holm-Šidák
    adjustment (statsmodels). Anchors the raw statistic to pingouin."""
    from statsmodels.stats.multitest import multipletests

    df = _rm_frame()
    meta = _run_default_rm_posthoc(df)
    comps = meta.get("pairwise_comparisons", [])

    # Reference raw paired-t p-values from pingouin, keyed by unordered pair.
    pt = pg.pairwise_tests(data=df, dv="y", within="cond", subject="subj")
    raw = {frozenset((str(r["A"]), str(r["B"]))): float(r["p_unc"]) for _, r in pt.iterrows()}

    keys = [frozenset((str(c["group1"]), str(c["group2"]))) for c in comps]
    ref_raw = [raw[k] for k in keys]
    ref_corr = multipletests(ref_raw, alpha=0.05, method="holm-sidak")[1]

    app_corr = np.array([float(c["p_value"]) for c in comps])
    np.testing.assert_allclose(app_corr, ref_corr, atol=1e-9)


def _mixed_frame():
    """Mixed design with a REAL interaction.

    The effects were deliberately strengthened (group 1.0, time 0.8/step, plus a
    1.2/step interaction, noise 0.6): with the earlier weak version every omnibus
    effect was n.s. (p = 0.12 / 0.14 / 0.58). That only went unnoticed because the
    effect-driven gate was broken and always produced simple main effects anyway.
    With a working gate the weak frame correctly yields mode='none' and zero
    contrasts, which would make the assertion below vacuous.
    """
    rng = np.random.default_rng(1)
    rows = []
    for g in ("ctrl", "trt"):
        for s in range(8):
            se = rng.normal(0, 1)
            for t in range(3):
                rows.append({
                    "subj": f"{g}{s}", "grp": g, "time": f"t{t}",
                    "y": (se + (1.0 if g == "trt" else 0.0) + 0.8 * t
                          + (1.2 * t if g == "trt" else 0.0) + rng.normal(0, 0.6)),
                })
    return pd.DataFrame(rows)


def test_mixed_paired_custom_never_pairs_between_subject_contrasts():
    """The user's blast-radius question: in a Mixed design, between-subject pairs
    must NOT go through a paired t-test. paired_custom classifies each pair and
    uses ttest_ind for between-group contrasts, ttest_rel only for within.

    Between contrasts may now legitimately be Student OR Welch (the variance
    assumption is Levene-gated), so the check asserts the invariant directly —
    never paired — instead of pinning one independent-test label.
    """
    from analysis.posthoc_core import MixedAnovaPostHocAnalyzer

    res = MixedAnovaPostHocAnalyzer.perform_test(
        _mixed_frame(), between="grp", within="time", dv="y", subject="subj",
        method="paired_custom",
    )
    comps = res["pairwise_comparisons"]
    assert comps, res
    for c in comps:
        label = str(c.get("test", ""))
        assert "Tukey" not in label
        g1, g2 = str(c["group1"]), str(c["group2"])
        same_between = g1.split(":")[0] == g2.split(":")[0]
        if same_between:
            assert "Paired" in label, (g1, g2, label)
        else:
            # between-group contrast: independent (Student or Welch), never paired
            assert "Paired" not in label, (g1, g2, label)
            assert ("Independent" in label) or ("Welch" in label), (g1, g2, label)
