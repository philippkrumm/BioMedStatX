import sys
sys.path.insert(0, "/Users/philippkrumm/Documents/BioMedStatX/src")
import numpy as np, pandas as pd
from analysis.nonparametricanovas import perform_freedman_lane_test
from statistical_testing.engines.advanced_posthoc import AdvancedPostHocEngine


def test_freedman_lane_dialog():
    rng = np.random.default_rng(0)
    A = ["a1", "a2", "a3"]   # 3 levels -> 3 marginal A pairs
    B = ["b1", "b2"]         # 2 levels -> 1 marginal B pair
    rows = []
    for ai, a in enumerate(A):
        for bi, b in enumerate(B):
            for k in range(8):  # N = 48
                rows.append({"y": 2.0 * ai + rng.normal(0, 1), "A": a, "B": b})
    df = pd.DataFrame(rows)

    res = perform_freedman_lane_test(df, "y", "A", "B", alpha=0.05, n_permutations=300, seed=1)
    assert res["error"] is None, f"FL failed on clean data: {res['error']}"
    pA = res["factors"][0]["p_value"]; pB = res["factors"][1]["p_value"]; pAB = res["interactions"][0]["p_value"]
    assert pA < 0.05, "DGP sanity: A should be significant"

    eng = AdvancedPostHocEngine()
    specs = eng._freedman_lane_candidate_specs(res, df, ["A", "B"], 0.05)
    kinds = set(s[2] for s in specs)

    # Gating must match omnibus p-values exactly
    assert ("marginal_a" in kinds) == (pA < 0.05), kinds
    assert ("marginal_b" in kinds) == (pB < 0.05), kinds
    assert ("cell" in kinds) == (pAB < 0.05), kinds
    if pA < 0.05:
        assert sum(1 for s in specs if s[2] == "marginal_a") == 3  # C(3,2)

    # Compute on all candidates -> non-empty, MWU, Holm bounded
    all_pairs = [(s[0], s[1]) for s in specs]
    out = eng._freedman_lane_compute(specs, all_pairs, df, "y", ["A", "B"], 0.05)
    assert out["pairwise_comparisons"]
    for c in out["pairwise_comparisons"]:
        assert c["test"] == "Mann-Whitney U"
        assert 0.0 <= c["p_value"] <= 1.0

    # Consistency vs FL-internal when all candidates selected
    internal = {(c["group1"], c["group2"]): round(c["p_value"], 6) for c in res["pairwise_comparisons"]}
    engine_out = {(c["group1"], c["group2"]): round(c["p_value"], 6) for c in out["pairwise_comparisons"]}
    assert internal == engine_out, f"engine != FL-internal\n internal={internal}\n engine={engine_out}"

    # Orchestrator with stubbed selection (pick first 2 pairs)
    AdvancedPostHocEngine._select_comparisons_dialog = staticmethod(lambda all_pairs, cb=None: all_pairs[:2])
    out2 = eng._freedman_lane_dialog_posthoc(res, df, "y", ["A", "B"], 0.05)
    assert len(out2["pairwise_comparisons"]) == 2


def test_freedman_lane_dialog_nan():
    """Dialog post-hoc must work when df_original still contains NaN rows."""
    rng = np.random.default_rng(42)
    A = ["a1", "a2", "a3"]
    B = ["b1", "b2"]
    rows = []
    for ai, a in enumerate(A):
        for bi, b in enumerate(B):
            for k in range(12):  # extra obs to survive NaN loss
                rows.append({"y": 2.5 * ai + rng.normal(0, 1), "A": a, "B": b})
    df = pd.DataFrame(rows)
    N_full = len(df)

    # ~18% NaN in DV; a few NaN factor labels
    nan_dv = rng.choice(df.index, size=N_full // 5, replace=False)
    df.loc[nan_dv, "y"] = np.nan
    nan_fac = rng.choice(df.index, size=3, replace=False)
    df.loc[nan_fac, "B"] = np.nan

    res = perform_freedman_lane_test(df, "y", "A", "B", alpha=0.05, n_permutations=200, seed=2)
    assert res["error"] is None, f"FL raised on NaN data: {res['error']}"

    AdvancedPostHocEngine._select_comparisons_dialog = staticmethod(lambda pairs, cb=None: pairs)
    eng = AdvancedPostHocEngine()
    specs = eng._freedman_lane_candidate_specs(res, df, ["A", "B"], 0.05)
    all_pairs = [(s[0], s[1]) for s in specs]
    out = eng._freedman_lane_compute(specs, all_pairs, df, "y", ["A", "B"], 0.05)

    assert isinstance(out["pairwise_comparisons"], list)
    for c in out["pairwise_comparisons"]:
        assert c is not None, "None comparison in result"
        assert 0.0 <= c["p_value"] <= 1.0
        assert c["test"] == "Mann-Whitney U"

    # Orchestrator path must not raise either
    out2 = eng._freedman_lane_dialog_posthoc(res, df, "y", ["A", "B"], 0.05)
    assert isinstance(out2["pairwise_comparisons"], list)


if __name__ == "__main__":
    test_freedman_lane_dialog()
    test_freedman_lane_dialog_nan()
    print("ALL STUFE-2 ASSERTS PASS")
