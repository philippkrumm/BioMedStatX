"""EMM post-hoc contrasts for ANCOVA (P1).

The contrasts must run on the covariate-adjusted means (EMMs), NOT on the raw
group means. These tests pin three properties:

1. A vs-control contrast estimate equals the difference of adjusted means
   (``adjusted_means``), i.e. the contrast is genuinely covariate-adjusted.
2. The adjusted contrast differs from the naive raw-mean contrast whenever the
   covariate is unbalanced across groups (the whole point of ANCOVA post-hocs).
3. Family shapes: ``vs_control`` yields G-1 comparisons (control excluded);
   ``pairwise`` yields C(G,2). Both carry p-values and significance flags.
"""
import numpy as np
import pandas as pd
import pytest

from analysis.clinical_models import ANCOVAModel


def _ancova_df():
    """3 groups, one covariate deliberately confounded with group.

    Group means on the covariate differ (A<B<C), so adjusting for the covariate
    pulls the group means apart from their raw values.
    """
    rng = np.random.default_rng(42)
    rows = []
    # covariate centre shifts by group; true treatment effect is small/none so
    # that the raw contrast and the adjusted contrast diverge.
    cov_centre = {"A": 5.0, "B": 8.0, "C": 11.0}
    slope = 2.0
    group_offset = {"A": 0.0, "B": 1.0, "C": 0.5}
    for g in ("A", "B", "C"):
        for _ in range(20):
            x = cov_centre[g] + rng.normal(0, 1.0)
            y = group_offset[g] + slope * x + rng.normal(0, 1.0)
            rows.append({"y": y, "group": g, "cov": x})
    return pd.DataFrame(rows)


def _fit():
    m = ANCOVAModel()
    m.fit(_ancova_df(), dv="y", between_factors=["group"], covariates=["cov"])
    return m


def test_vs_control_estimate_equals_adjusted_mean_difference():
    m = _fit()
    contrasts = m.emm_contrasts(method="vs_control", control_group="A")
    adj = m.adjusted_means()["group"]
    by_trt = {c["group1"]: c for c in contrasts}
    for trt in ("B", "C"):
        expected = adj[trt]["adjusted_mean"] - adj["A"]["adjusted_mean"]
        assert by_trt[trt]["estimate"] == pytest.approx(expected, abs=1e-8)


def test_adjusted_contrast_differs_from_raw_mean_contrast():
    df = _ancova_df()
    m = _fit()
    contrasts = m.emm_contrasts(method="vs_control", control_group="A")
    by_trt = {c["group1"]: c for c in contrasts}
    raw = df.groupby("group")["y"].mean()
    raw_diff_C = raw["C"] - raw["A"]
    adj_diff_C = by_trt["C"]["estimate"]
    # Covariate is strongly unbalanced => adjusted and raw contrasts diverge.
    assert abs(adj_diff_C - raw_diff_C) > 1.0


def test_vs_control_family_shape_and_mvt_pvalues():
    m = _fit()
    contrasts = m.emm_contrasts(method="vs_control", control_group="A")
    assert len(contrasts) == 2  # G-1, control excluded
    assert all(c["group2"] == "A" for c in contrasts)
    for c in contrasts:
        assert 0.0 <= c["p_value"] <= 1.0
        assert isinstance(c["significant"], bool)
        assert c["se"] > 0 and c["df"] > 0


def test_pairwise_family_shape():
    m = _fit()
    contrasts = m.emm_contrasts(method="pairwise")
    assert len(contrasts) == 3  # C(3,2)
    pairs = {tuple(sorted((c["group1"], c["group2"]))) for c in contrasts}
    assert pairs == {("A", "B"), ("A", "C"), ("B", "C")}


def test_results_dict_emits_pairwise_comparisons_and_falls_back_to_pairwise():
    # No control group identifiable -> fallback to pairwise (EMM + Holm).
    m = _fit()
    res = m.as_results_dict()
    assert res.get("pairwise_comparisons"), "ANCOVA must emit EMM contrasts"
    assert "EMM" in (res.get("posthoc_test") or "")
    assert len(res["pairwise_comparisons"]) == 3  # pairwise fallback


def test_results_dict_uses_vs_control_when_control_group_set():
    m = ANCOVAModel()
    m.fit(_ancova_df(), dv="y", between_factors=["group"],
          covariates=["cov"], control_group="A")
    res = m.as_results_dict()
    assert len(res["pairwise_comparisons"]) == 2  # vs-control family
    assert "control" in (res.get("posthoc_test") or "").lower()
