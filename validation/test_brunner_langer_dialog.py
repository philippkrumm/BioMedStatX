import sys
sys.path.insert(0, "/Users/philippkrumm/Documents/BioMedStatX/src")
import numpy as np, pandas as pd
from analysis.nonparametricanovas import perform_brunner_langer_ats
from statistical_testing.engines.advanced_posthoc import AdvancedPostHocEngine


def test_brunner_langer_dialog():
    rng = np.random.default_rng(0)
    groups = ["g1", "g2"]          # between, 2 levels -> 1 between pair
    times = ["t1", "t2", "t3"]     # within, 3 levels -> 3 within pairs
    group_eff = {"g1": 0.0, "g2": 2.5}
    time_eff = {"t1": 0.0, "t2": 1.2, "t3": 2.4}
    rows = []
    sid = 0
    for g in groups:
        for _ in range(12):        # 12 subjects per group
            sid += 1
            subj_intercept = rng.normal(0, 0.4)
            for t in times:
                y = group_eff[g] + time_eff[t] + subj_intercept + rng.normal(0, 0.4)
                rows.append({"y": y, "G": g, "Time": t, "subj": f"s{sid}"})
    df = pd.DataFrame(rows)

    res = perform_brunner_langer_ats(df, "y", "G", "Time", "subj", alpha=0.05)
    pB = next(f["p_value"] for f in res["factors"] if f["factor"] == "G")
    pW = next(f["p_value"] for f in res["factors"] if f["factor"] == "Time")
    pBW = res["interactions"][0]["p_value"]

    eng = AdvancedPostHocEngine()
    specs = eng._brunner_langer_candidate_specs(res, df, ["G"], ["Time"], 0.05)
    kinds = set(s[2] for s in specs)

    # Gating matches omnibus p-values exactly
    assert ("between" in kinds) == (pB < 0.05), kinds
    assert ("within" in kinds) == (pW < 0.05), kinds
    assert ("interaction" in kinds) == (pBW < 0.05), kinds

    all_pairs = [(s[0], s[1]) for s in specs]
    out = eng._brunner_langer_compute(specs, all_pairs, df, "y", ["G"], ["Time"], "subj", 0.05)

    # Test-type routing: within -> Wilcoxon, between/interaction -> MWU
    spec_kind = {(s[0], s[1]): s[2] for s in specs}
    for c in out["pairwise_comparisons"]:
        kind = spec_kind[(c["group1"], c["group2"])]
        if kind == "within":
            assert c["test"] == "Wilcoxon Signed-Rank", (kind, c["test"])
        else:
            assert c["test"] == "Mann-Whitney U", (kind, c["test"])

    # Consistency vs BL-internal (all candidates selected)
    internal = {(c["group1"], c["group2"]): round(c["p_value"], 6) for c in res["pairwise_comparisons"]}
    engine_out = {(c["group1"], c["group2"]): round(c["p_value"], 6) for c in out["pairwise_comparisons"]}
    assert internal == engine_out, f"engine != BL-internal\n internal={internal}\n engine={engine_out}"

    # Pipeline wrapper + stubbed selection
    AdvancedPostHocEngine._select_comparisons_dialog = staticmethod(lambda pairs: pairs)
    upd = eng._run_nonparametric_fallback_posthoc({
        "res": res, "test": "mixed_anova", "df_original": df, "dv": "y",
        "subject": "subj", "between": ["G"], "within": ["Time"], "alpha": 0.05,
    })
    assert upd.get("pairwise_comparisons"), upd.keys()
    assert "Wilcoxon" in upd["posthoc_test"] and "Mann-Whitney" in upd["posthoc_test"]


if __name__ == "__main__":
    test_brunner_langer_dialog()
    print("ALL BL ASSERTS PASS")
