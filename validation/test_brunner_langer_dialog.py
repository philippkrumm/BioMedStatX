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


def test_brunner_langer_dialog_nan():
    """BL dialog post-hoc must work when df_original has NaN in DV and missing
    within-measurements (subjects with incomplete time points are dropped from
    the paired Wilcoxon wide matrix but kept in MWU between comparisons)."""
    rng = np.random.default_rng(42)
    groups = ["g1", "g2"]
    times = ["t1", "t2", "t3"]
    group_eff = {"g1": 0.0, "g2": 3.0}
    time_eff = {"t1": 0.0, "t2": 1.5, "t3": 3.0}
    rows = []
    sid = 0
    for g in groups:
        for _ in range(18):  # more subjects to survive NaN-induced list-wise deletion
            sid += 1
            subj_intercept = rng.normal(0, 0.3)
            for t in times:
                y = group_eff[g] + time_eff[t] + subj_intercept + rng.normal(0, 0.3)
                rows.append({"y": y, "G": g, "Time": t, "subj": f"s{sid}"})
    df = pd.DataFrame(rows)

    # Structural NaN: 5 subjects each miss one time point → excluded from Wilcoxon wide
    subjects = df["subj"].unique()
    nan_subj = rng.choice(subjects, size=5, replace=False)
    df.loc[(df["subj"].isin(nan_subj)) & (df["Time"] == "t1"), "y"] = np.nan

    # Random NaN in DV (not tied to a specific time point)
    nan_idx = rng.choice(df.index, size=8, replace=False)
    df.loc[nan_idx, "y"] = np.nan

    # NaN in subject label on a few rows → those rows dropped by omnibus dropna
    nan_subj_label = rng.choice(df.index, size=3, replace=False)
    df.loc[nan_subj_label, "subj"] = np.nan

    res = perform_brunner_langer_ats(df, "y", "G", "Time", "subj", alpha=0.05)
    assert res["error"] is None, f"BL raised on NaN data: {res['error']}"
    assert res["factors"] and res["interactions"]

    AdvancedPostHocEngine._select_comparisons_dialog = staticmethod(lambda pairs: pairs)
    eng = AdvancedPostHocEngine()
    specs = eng._brunner_langer_candidate_specs(res, df, ["G"], ["Time"], 0.05)
    all_pairs = [(s[0], s[1]) for s in specs]
    out = eng._brunner_langer_compute(specs, all_pairs, df, "y", ["G"], ["Time"], "subj", 0.05)

    comps = out["pairwise_comparisons"]
    assert isinstance(comps, list)
    for c in comps:
        assert c is not None, "None comparison leaked through"
        assert 0.0 <= c["p_value"] <= 1.0
        assert c["test"] in {"Wilcoxon Signed-Rank", "Mann-Whitney U"}

    # Pipeline wrapper
    upd = eng._run_nonparametric_fallback_posthoc({
        "res": res, "test": "mixed_anova", "df_original": df, "dv": "y",
        "subject": "subj", "between": ["G"], "within": ["Time"], "alpha": 0.05,
    })
    pipeline_comps = upd.get("pairwise_comparisons", [])
    assert isinstance(pipeline_comps, list)
    for c in pipeline_comps:
        assert c is not None
        assert 0.0 <= c["p_value"] <= 1.0


if __name__ == "__main__":
    test_brunner_langer_dialog()
    test_brunner_langer_dialog_nan()
    print("ALL BL ASSERTS PASS")
