import pandas as pd
import numpy as np
from analysis.clinical_models import LogisticRegressionModel

def test_convergence_keys():
    # 1. Logistic: Complete separation (should trigger Firth, which usually converges, but we test the mechanism)
    # If we want it to completely fail, we can make the design matrix singular or just test the fallback.
    df_log = pd.DataFrame({'x': [1, 2, 3, 4, 5, 6], 'y': [0, 0, 0, 1, 1, 1]})
    mod_log = LogisticRegressionModel()
    mod_log.fit(df_log, 'y', ['x'])
    res_log = mod_log.as_results_dict()
    assert "converged" in res_log
    
    # To truly fail Logistic (both standard and Firth), we provide a singular design matrix (x1 == x2)
    # or a dataset with perfect separation that fails Firth. Let's try perfect separation again.
    # Actually, singular design matrix reliably triggers LinAlgError and forces non-convergence.
    df_log_fail = pd.DataFrame({'x1': [1, 2, 3, 4, 5, 6], 'x2': [1, 2, 3, 4, 5, 6], 'y': [0, 0, 0, 1, 1, 1]})
    mod_log_fail = LogisticRegressionModel()
    mod_log_fail.fit(df_log_fail, 'y', ['x1', 'x2'])
    res_log_fail = mod_log_fail.as_results_dict()
    assert res_log_fail.get("converged") is False
    assert any("not converge" in w or "unreliable" in w for w in res_log_fail.get("warnings", []))
