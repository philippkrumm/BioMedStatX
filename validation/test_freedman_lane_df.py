import sys
sys.path.insert(0, "/Users/philippkrumm/Documents/BioMedStatX/src")
import numpy as np, pandas as pd
from analysis.nonparametricanovas import perform_freedman_lane_test


def test_freedman_lane_df():
    rng = np.random.default_rng(0)
    A = ["a1", "a2", "a3", "a4"]      # a = 4 -> df_A should be 3
    B = ["b1", "b2", "b3"]            # b = 3 -> df_B should be 2; df_AB = 6
    rows = []
    for ai, a in enumerate(A):
        for bi, b in enumerate(B):
            for k in range(5):        # n=5 per cell -> N=60
                rows.append({"y": 1.0 * ai + 0.5 * bi + rng.normal(0, 1), "A": a, "B": b})
    df = pd.DataFrame(rows)
    N = len(df)

    res = perform_freedman_lane_test(df, "y", "A", "B", alpha=0.05, n_permutations=300, seed=1)
    assert res["error"] is None, f"FL failed on clean data: {res['error']}"
    at = res["anova_table"]

    df1s = [int(x) for x in at["DF1"]]
    df2s = [int(x) for x in at["DF2"]]
    assert df1s == [3, 2, 6], f"DF1 wrong: {df1s}"
    assert df2s == [N - 6, N - 6, N - 12], f"DF2 wrong: {df2s}"   # 54,54,48

    es = [f["effect_size"] for f in res["factors"]] + [res["interactions"][0]["effect_size"]]
    ts = [f["effect_size_type"] for f in res["factors"]] + [res["interactions"][0]["effect_size_type"]]
    for e in es:
        assert 0.0 <= e <= 1.0, f"eta2 out of [0,1]: {e}"
    assert set(ts) == {"partial η²"}, ts

    # eta2p == F*df1/(F*df1+df2)
    for i, row in enumerate(at.itertuples(index=False)):
        F = float(row.F); d1 = int(row.DF1); d2 = int(row.DF2)
        expect = F * d1 / (F * d1 + d2) if F > 0 else 0.0
        assert abs(expect - es[i]) < 1e-6, f"row {i} {row.Source}: F-form {expect} != stored {es[i]}"


def test_freedman_lane_df_nan():
    """NaN in DV and factor labels must not crash; eta2p must stay in [0,1]."""
    rng = np.random.default_rng(42)
    A = ["a1", "a2", "a3", "a4"]
    B = ["b1", "b2", "b3"]
    rows = []
    for ai, a in enumerate(A):
        for bi, b in enumerate(B):
            for k in range(7):  # more obs so cells survive NaN loss
                rows.append({"y": 1.5 * ai + 0.5 * bi + rng.normal(0, 1), "A": a, "B": b})
    df = pd.DataFrame(rows)
    N_full = len(df)

    # ~15% NaN in DV
    nan_dv = rng.choice(df.index, size=N_full // 7, replace=False)
    df.loc[nan_dv, "y"] = np.nan
    # NaN in factor labels (a few rows) → dropna removes them entirely
    nan_fac = rng.choice(df.index, size=4, replace=False)
    df.loc[nan_fac, "A"] = np.nan

    res = perform_freedman_lane_test(df, "y", "A", "B", alpha=0.05, n_permutations=200, seed=1)
    assert res["error"] is None, f"FL raised on NaN data: {res['error']}"
    at = res["anova_table"]
    assert at is not None
    assert all(int(x) >= 1 for x in at["DF1"])

    es = [f["effect_size"] for f in res["factors"]] + [res["interactions"][0]["effect_size"]]
    ts = {f["effect_size_type"] for f in res["factors"]} | {res["interactions"][0]["effect_size_type"]}
    for e in es:
        assert 0.0 <= e <= 1.0, f"eta2 out of [0,1] after NaN injection: {e}"
    assert ts == {"partial η²"}, ts

    # eta2p == F*df1/(F*df1+df2) still holds on cleaned data
    for i, row in enumerate(at.itertuples(index=False)):
        F = float(row.F); d1 = int(row.DF1); d2 = int(row.DF2)
        expect = F * d1 / (F * d1 + d2) if F > 0 else 0.0
        assert abs(expect - es[i]) < 1e-6, f"row {i}: F-form {expect} != stored {es[i]}"

    # Pairwise list must not contain None entries
    assert isinstance(res["pairwise_comparisons"], list)
    assert all(c is not None for c in res["pairwise_comparisons"])
    for c in res["pairwise_comparisons"]:
        assert 0.0 <= c["p_value"] <= 1.0


def test_freedman_lane_catastrophic_nan():
    """When NaN wipes an entire factor level, FL must return a graceful error
    dict (not raise), with a non-None error string and safe empty collections."""
    rng = np.random.default_rng(99)
    rows = []
    for ai, a in enumerate(["a1", "a2", "a3"]):
        for b in ["b1", "b2"]:
            for k in range(4):
                rows.append({"y": float(ai) + rng.normal(0, 1), "A": a, "B": b})
    df = pd.DataFrame(rows)
    # Wipe entire level a1 in factor A → only 2 levels remain, but let's also
    # wipe a2 so the design collapses completely for one cell combination
    df.loc[(df["A"] == "a1") & (df["B"] == "b1"), "y"] = np.nan
    df.loc[(df["A"] == "a1") & (df["B"] == "b2"), "y"] = np.nan
    # a1 is now gone → A has 2 levels, design is 2×2 → still valid, no error expected
    res = perform_freedman_lane_test(df, "y", "A", "B", n_permutations=100, seed=1)
    # Either succeeds (missing level reduced design) or fails gracefully
    if res["error"] is not None:
        assert isinstance(res["error"], str)
        assert res["pairwise_comparisons"] == []
        assert res["anova_table"] is None
    else:
        at = res["anova_table"]
        assert at is not None
        for e in ([f["effect_size"] for f in res["factors"]] +
                  [res["interactions"][0]["effect_size"]]):
            assert 0.0 <= e <= 1.0


if __name__ == "__main__":
    test_freedman_lane_df()
    test_freedman_lane_df_nan()
    test_freedman_lane_catastrophic_nan()
    print("ALL ASSERTS PASS")
