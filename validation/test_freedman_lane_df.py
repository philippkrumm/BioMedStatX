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


if __name__ == "__main__":
    test_freedman_lane_df()
    print("ALL ASSERTS PASS")
