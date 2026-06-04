import sys
sys.path.insert(0, "/Users/philippkrumm/Documents/BioMedStatX/src")
import numpy as np, pandas as pd
from analysis.nonparametricanovas import perform_freedman_lane_test
from statistical_testing.engines.advanced_posthoc import AdvancedPostHocEngine


def test_freedman_lane_integration():
    # Stub dialog (headless): select all candidates
    AdvancedPostHocEngine._select_comparisons_dialog = staticmethod(lambda pairs: pairs)

    rng = np.random.default_rng(0)
    rows = []
    for ai, a in enumerate(["a1", "a2", "a3"]):
        for b in ["b1", "b2"]:
            for k in range(8):
                rows.append({"y": 2.0 * ai + rng.normal(0, 1), "A": a, "B": b})
    df = pd.DataFrame(rows)
    res = perform_freedman_lane_test(df, "y", "A", "B", alpha=0.05, n_permutations=200, seed=1)

    eng = AdvancedPostHocEngine()
    upd = eng._run_nonparametric_fallback_posthoc({
        "res": res, "test": "two_way_anova", "df_original": df, "dv": "y",
        "subject": None, "between": ["A", "B"], "within": None, "alpha": 0.05,
    })
    assert upd.get("pairwise_comparisons"), "FL branch must populate pairwise_comparisons"
    assert "Mann-Whitney" in (upd.get("posthoc_test") or ""), upd.get("posthoc_test")

    # Non-significant omnibus -> posthoc_skipped passthrough
    rows2 = [{"y": rng.normal(0, 1), "A": a, "B": b} for a in ["a1", "a2"] for b in ["b1", "b2"] for _ in range(8)]
    df2 = pd.DataFrame(rows2)
    res2 = perform_freedman_lane_test(df2, "y", "A", "B", alpha=0.05, n_permutations=200, seed=2)
    upd2 = eng._run_nonparametric_fallback_posthoc({
        "res": res2, "test": "two_way_anova", "df_original": df2, "dv": "y",
        "subject": None, "between": ["A", "B"], "within": None, "alpha": 0.05,
    })
    if res2["p_value"] >= 0.05:
        assert upd2.get("posthoc_skipped") is True, upd2


def test_freedman_lane_integration_nan():
    """Full pipeline (_run_nonparametric_fallback_posthoc) must work with NaN df_original."""
    AdvancedPostHocEngine._select_comparisons_dialog = staticmethod(lambda pairs: pairs)
    rng = np.random.default_rng(42)
    rows = []
    for ai, a in enumerate(["a1", "a2", "a3"]):
        for b in ["b1", "b2"]:
            for k in range(12):
                rows.append({"y": 2.5 * ai + rng.normal(0, 1), "A": a, "B": b})
    df = pd.DataFrame(rows)

    # NaN in DV and one factor column
    nan_dv = rng.choice(df.index, size=len(df) // 5, replace=False)
    df.loc[nan_dv, "y"] = np.nan
    nan_fac = rng.choice(df.index, size=4, replace=False)
    df.loc[nan_fac, "A"] = np.nan

    res = perform_freedman_lane_test(df, "y", "A", "B", alpha=0.05, n_permutations=200, seed=3)
    assert res["error"] is None, f"FL raised on NaN data: {res['error']}"

    eng = AdvancedPostHocEngine()
    upd = eng._run_nonparametric_fallback_posthoc({
        "res": res, "test": "two_way_anova", "df_original": df, "dv": "y",
        "subject": None, "between": ["A", "B"], "within": None, "alpha": 0.05,
    })
    # Must return a dict; pairwise_comparisons must be a list without None entries
    comps = upd.get("pairwise_comparisons", [])
    assert isinstance(comps, list)
    for c in comps:
        assert c is not None, "None comparison leaked through"
        assert 0.0 <= c["p_value"] <= 1.0


if __name__ == "__main__":
    test_freedman_lane_integration()
    test_freedman_lane_integration_nan()
    print("INTEGRATION ASSERTS PASS")
