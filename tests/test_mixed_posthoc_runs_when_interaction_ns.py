"""BLOCKER (pre-2.0 audit): the Mixed-ANOVA post-hoc for a non-significant
interaction was a row-order-dependent paired t-test, and the checked
implementation was switched off for exactly that case.

`advanced_pipeline` gates the post-hoc on `res["p_value"] < alpha`, and for a
mixed design `res["p_value"]` is the INTERACTION's p (statisticaltester.py sets
it from the GG-corrected interaction row). So whenever the interaction is not
significant -- the common case in a mixed design with a real main effect --
AdvancedPostHocEngine never ran, and the inline block in
`StatisticalTester._run_mixed_anova` supplied the contrasts instead:

    data1 = df[df[rm_factor] == group1][dv].values
    data2 = df[df[rm_factor] == group2][dv].values
    t_stat, p_val = stats.ttest_rel(data1, data2)

That pairs by POSITION in the dataframe, not by subject. Measured on the frame
below, reordering the rows moved T1-T2 from p = 0.0117 (significant) to
p = 0.2982 (not), and flipped comparison directions (T2-T3 became T3-T2). The
results were merged into the standard `pairwise_comparisons` key
(statisticaltester.py:1908-1912) and rendered under the label
"Paired t-tests (Holm-Bonferroni)", so nothing about the report looked wrong.

Feature B already handles this case correctly -- `mixed_effect_driven_posthoc`
returns `marginal_within` contrasts, paired by subject and Levene-gated -- it was
simply never asked.
"""
import numpy as np
import pandas as pd
import pytest

from statistical_testing.advanced_pipeline import perform_advanced_test_pipeline

ALPHA = 0.05


def _frame(shuffled=False):
    """Strong within effect (p ~ 7e-05), clearly non-significant interaction."""
    rng = np.random.default_rng(3)
    rows = []
    for g in ("ctrl", "trt"):
        for s in range(12):
            base = rng.normal(0, 1)
            off = 1.0 if g == "trt" else 0.0
            for i, t in enumerate(("T1", "T2", "T3")):
                rows.append({"subj": f"{g}{s}", "grp": g, "time": t,
                             "y": base + off + 0.8 * i + rng.normal(0, 0.8)})
    df = pd.DataFrame(rows)
    if shuffled:
        df = df.sample(frac=1.0, random_state=99).reset_index(drop=True)
    return df


def _run(df):
    return perform_advanced_test_pipeline(
        df=df, test="mixed_anova", dv="y", subject="subj",
        between=["grp"], within=["time"], force_parametric=True, alpha=ALPHA,
    )


@pytest.fixture(scope="module")
def result():
    return _run(_frame())


def test_the_scenario_is_the_one_we_mean(result):
    """Interaction not significant, within main effect strongly significant."""
    assert result["p_value"] > ALPHA, result["p_value"]
    within = next(f for f in result["factors"] if f["factor"] == "time")
    assert within["p_value"] < 1e-3, within


def test_a_post_hoc_is_still_reported(result):
    """Guard against the opposite failure: removing the broken branch must not
    leave a significant main effect with no contrasts at all."""
    assert result.get("pairwise_comparisons"), result.get("posthoc_test")


def test_feature_b_supplies_the_contrasts(result):
    assert result.get("posthoc_mode") == "marginal_within", result.get("posthoc_mode")
    label = str(result.get("posthoc_test") or "")
    assert "marginal" in label.lower(), label


def test_result_is_row_order_invariant():
    """The core defect: identical data, different row order, different verdict."""
    a = _run(_frame(shuffled=False))
    b = _run(_frame(shuffled=True))

    def by_pair(res):
        out = {}
        for c in res["pairwise_comparisons"]:
            key = tuple(sorted((str(c["group1"]), str(c["group2"]))))
            out[key] = float(c["p_value"])
        return out

    pa, pb = by_pair(a), by_pair(b)
    assert set(pa) == set(pb), (sorted(pa), sorted(pb))
    for key in pa:
        assert pa[key] == pytest.approx(pb[key], abs=1e-12), (
            f"{key}: p changed with row order, {pa[key]} -> {pb[key]}")


def test_contrasts_are_paired_by_subject_not_position(result):
    """Anchor one contrast to a subject-aligned paired t-test."""
    from scipy import stats

    df = _frame()
    wide = df.pivot(index="subj", columns="time", values="y")
    ref = stats.ttest_rel(wide["T1"], wide["T2"])

    comp = next(c for c in result["pairwise_comparisons"]
                if {str(c["group1"]), str(c["group2"])} == {"T1", "T2"})
    assert float(comp["statistic"]) == pytest.approx(ref.statistic, abs=1e-9), (
        comp["statistic"], ref.statistic)


def test_the_inline_blocks_leftover_keys_are_gone(result):
    for key in ("within_pairwise_comparisons", "between_pairwise_comparisons",
                "within_posthoc_test", "between_posthoc_test"):
        assert not result.get(key), (key, result.get(key))
