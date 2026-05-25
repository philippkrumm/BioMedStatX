import numpy as np
import pandas as pd
import pytest
from clinical_models import LinearMixedModel, LogisticRegressionModel

def test_lmm_bw_degrees_of_freedom():
    # Verify df = 28 for group/between and df = 59 for time/within under N = 90
    # Let's create a dataset with:
    # 30 subjects (groups)
    # 3 observations per subject -> N = 90
    # predictor 'group_var': between-subject variable (constant per subject)
    # predictor 'time_var': within-subject variable (0, 1, 2 for each subject)
    np.random.seed(42)
    n_groups = 30
    obs_per_group = 3
    n_obs = n_groups * obs_per_group
    
    subjects = []
    group_vars = []
    time_vars = []
    y = []
    
    # 15 subjects in group 0, 15 in group 1
    for s_idx in range(n_groups):
        g_val = 0 if s_idx < 15 else 1
        for t_val in range(obs_per_group):
            subjects.append(f"Subj_{s_idx}")
            group_vars.append(g_val)
            time_vars.append(t_val)
            y_val = 10.0 + 2.0 * g_val + 0.5 * t_val + np.random.normal(0, 0.5)
            y.append(y_val)
            
    df = pd.DataFrame({
        "subject": subjects,
        "group_var": group_vars,
        "time_var": time_vars,
        "y": y
    })
    
    model = LinearMixedModel()
    model.fit(
        df,
        dv="y",
        fixed_effects=["group_var"],
        covariates=["time_var"],
        random_intercept="subject"
    )
    res = model.as_results_dict()
    
    assert "fixed_effects_table" in res
    fe_table = res["fixed_effects_table"]
    fe_by_param = {fe["parameter"]: fe for fe in fe_table}
    
    # Check that df_method indicates Between-Within
    assert res["df_method"].startswith("Between-Within")
    
    # Check degrees of freedom:
    # For between-subject factor (group_var):
    # n_groups - 1 - n_between_predictors = 30 - 1 - 1 = 28
    # For within-subject factor (time_var):
    # n_obs - n_groups - n_within_predictors = 90 - 30 - 1 = 59
    # For Intercept: n_groups - 1 = 29
    
    group_param = [k for k in fe_by_param.keys() if "group_var" in k][0]
    time_param = [k for k in fe_by_param.keys() if "time_var" in k][0]
    
    assert fe_by_param[group_param]["df"] == 28
    assert fe_by_param[time_param]["df"] == 59
    assert fe_by_param["Intercept"]["df"] == 29

def test_logistic_firth_separation():
    # Verify standard MLE logit fails/has high SE, Firth fallback is triggered,
    # coefficients are finite, and "model_variant": "Firth Penalized Likelihood" is set.
    # Complete separation dataset:
    # N = 50. X is binary, Y is binary.
    # For X = 0, Y = 0 (25 cases)
    # For X = 1, Y = 1 (25 cases)
    np.random.seed(42)
    n_cases = 25
    x = np.concatenate([np.zeros(n_cases), np.ones(n_cases)])
    y = np.concatenate([np.zeros(n_cases), np.ones(n_cases)])
    cov = np.random.normal(0, 1, len(x))
    
    df = pd.DataFrame({
        "x": x,
        "y": y,
        "cov": cov
    })
    
    model = LogisticRegressionModel()
    model.fit(
        df,
        dv="y",
        predictors=["x"],
        covariates=["cov"]
    )
    res = model.as_results_dict()
    
    assert res["model_variant"] == "Firth Penalized Likelihood"
    assert res["brier_score"] is not None
    assert np.isfinite(res["brier_score"])
    assert res["calibration_slope"] is not None
    assert np.isfinite(res["calibration_slope"])
    assert res["calibration_intercept"] is not None
    assert np.isfinite(res["calibration_intercept"])
    
    # Check that coefficients are finite
    for fe in res["odds_ratios"]:
        assert np.isfinite(fe["coefficient"])
        assert np.isfinite(fe["odds_ratio"])
        assert np.isfinite(fe["std_err"])
