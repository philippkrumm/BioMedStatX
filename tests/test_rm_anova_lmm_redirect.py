import pytest
import pandas as pd
import numpy as np
from analysis.statisticaltester import StatisticalTester

def test_rm_anova_lmm_redirect():
    """
    Test E2-LMM feature: 
    When > 5% of subjects have missing data in RM-ANOVA, it should redirect to LMM.
    """
    np.random.seed(42)
    n_subjects = 40
    # 40 subjects * 3 timepoints
    subjects = np.repeat(np.arange(1, n_subjects + 1), 3)
    timepoints = np.tile(['T1', 'T2', 'T3'], n_subjects)
    values = np.random.normal(50, 10, size=n_subjects * 3)
    
    df = pd.DataFrame({
        'Subject': subjects,
        'Time': timepoints,
        'Value': values
    })
    # Force a main effect so we get posthoc pairwise comparisons
    df.loc[df['Time'] == 'T1', 'Value'] += 15
    
    # 1. 0% missing data -> should return RepeatedMeasuresANOVA
    res_complete = StatisticalTester()._run_repeated_measures_anova(
        df=df.copy(), dv='Value', subject='Subject', within=['Time']
    )
    assert res_complete["model_type"] == "RepeatedMeasuresANOVA"
    assert "analysis_note" not in res_complete or "redirected to a Linear Mixed Model" not in str(res_complete.get("analysis_note", ""))
    
    # TV-1: Assert cohen_d_rm value/sign
    pairs = res_complete.get("pairwise_comparisons", [])
    assert len(pairs) == 3
    t1_t2 = next(p for p in pairs if set([p['group1'], p['group2']]) == {'T1', 'T2'})
    assert t1_t2['effect_size_type'] == "Cohen's d (RM)"
    # The expected cohen d rm is ~1.226 (calculated with ddof=1)
    assert abs(t1_t2['effect_size']) > 1.0

    # 2. Add > 5% missing data (e.g. drop 4 values from 4 different subjects -> 4/40 = 10% missing)
    df_missing = df.copy()
    drop_indices = df_missing.groupby('Subject').apply(lambda x: x.sample(1)).index.get_level_values(1)[:4]
    df_missing.loc[drop_indices, 'Value'] = np.nan
    
    res_redirect = StatisticalTester()._run_repeated_measures_anova(
        df=df_missing, dv='Value', subject='Subject', within=['Time']
    )
    
    # Should be redirected to LMM
    assert res_redirect["model_type"] == "LMM"
    assert "redirected to a Linear Mixed Model" in res_redirect["analysis_note"]
    
    # Check if the posthoc info warning is included
    warnings = res_redirect.get("warnings", [])
    assert any("pairwise contrasts computed via EMM" in w for w in warnings)
    assert len(res_redirect.get("pairwise_comparisons", [])) > 0
