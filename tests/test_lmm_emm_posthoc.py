"""EMM post-hoc contrasts for Linear Mixed Model (LMM) (P4).

Pins three properties:
1. test_lmm_between_factor: df_bw should be N_clusters - 1 - k_between. vs_control -> mvt.
2. test_lmm_within_factor: df_bw should be N_obs - N_clusters - k_within. pairwise -> Holm.
3. test_lmm_missing_data_fallback: dynamic df_bw on unbalanced data, fallback to pairwise if control group missing.
"""
import numpy as np
import pandas as pd
import pytest

from analysis.clinical_models import LinearMixedModel


def test_lmm_between_factor():
    # 10 subjects (clusters), 3 observations per subject => N_obs = 30.
    # Genotype is constant within subject (WT vs KO).
    rng = np.random.default_rng(42)
    rows = []
    # WT for subjects S0-S4, KO for S5-S9
    for s in range(10):
        genotype = "WT" if s < 5 else "KO"
        subj_intercept = rng.normal(0, 1.0)
        for t in range(3):
            # Genotype KO has a small positive effect
            y = subj_intercept + (1.5 if genotype == "KO" else 0.0) + rng.normal(0, 0.5)
            rows.append({"y": y, "subject": f"S{s}", "genotype": genotype})
    df = pd.DataFrame(rows)

    m = LinearMixedModel()
    m.fit(df, dv="y", fixed_effects=["genotype"], random_intercept="subject",
          alpha=0.05, control_group="WT")
    
    res = m.as_results_dict()
    assert res.get("pairwise_comparisons"), "Should compute contrasts"
    contrasts = res["pairwise_comparisons"]
    
    # 1 comparison: KO vs WT
    assert len(contrasts) == 1
    c = contrasts[0]
    assert c["group1"] == "KO"
    assert c["group2"] == "WT"
    
    # Check df: N_clusters (10) - 1 - k_between (1) = 8.
    assert c["df"] == 8
    assert c["correction"] == "multivariate-t"
    assert 0.0 <= c["p_value"] <= 1.0
    assert isinstance(c["significant"], bool)


def test_lmm_within_factor():
    # 10 subjects (clusters), 3 timepoints (T0, T1, T2) within subject.
    # Time varies within subject.
    rng = np.random.default_rng(42)
    rows = []
    for s in range(10):
        subj_intercept = rng.normal(0, 1.0)
        for t in ("T0", "T1", "T2"):
            eff = {"T0": 0.0, "T1": 1.0, "T2": 2.0}[t]
            y = subj_intercept + eff + rng.normal(0, 0.5)
            rows.append({"y": y, "subject": f"S{s}", "time": t})
    df = pd.DataFrame(rows)

    m = LinearMixedModel()
    m.fit(df, dv="y", fixed_effects=["time"], random_intercept="subject", alpha=0.05)
    
    res = m.as_results_dict()
    assert res.get("pairwise_comparisons"), "Should compute pairwise contrasts"
    contrasts = res["pairwise_comparisons"]
    
    # 3 pairwise comparisons: T1-T0, T2-T0, T2-T1
    assert len(contrasts) == 3
    for c in contrasts:
        # Check df: N_obs (30) - N_clusters (10) - k_within (2) = 18.
        assert c["df"] == 18
        assert c["correction"] == "Holm-Bonferroni"
        assert 0.0 <= c["p_value"] <= 1.0
        assert isinstance(c["significant"], bool)


def test_lmm_missing_data_fallback():
    # 10 subjects, unbalanced timepoints (T0, T1, T2)
    rng = np.random.default_rng(42)
    rows = []
    for s in range(10):
        subj_intercept = rng.normal(0, 1.0)
        for t in ("T0", "T1", "T2"):
            eff = {"T0": 0.0, "T1": 1.0, "T2": 2.0}[t]
            y = subj_intercept + eff + rng.normal(0, 0.5)
            rows.append({"y": y, "subject": f"S{s}", "time": t})
    df = pd.DataFrame(rows)
    # Remove 2 rows to make it unbalanced
    df = df.iloc[:-2]
    
    m = LinearMixedModel()
    # No control group -> fallback to pairwise Holm-Bonferroni
    m.fit(df, dv="y", fixed_effects=["time"], random_intercept="subject", alpha=0.05)
    res = m.as_results_dict()
    assert res.get("pairwise_comparisons")
    contrasts = res["pairwise_comparisons"]
    
    assert len(contrasts) == 3
    for c in contrasts:
        # N_obs is now 28. df should be 28 - 10 - 2 = 16.
        assert c["df"] == 16
        assert c["correction"] == "Holm-Bonferroni"
