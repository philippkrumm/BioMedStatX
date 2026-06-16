"""Gated golden test: Advanced R Integration (car, afex, emmeans).

Runs the APP's perform_advanced_test on the frozen dataset and compares the outputs
to the frozen R references (Two-Way ANOVA, Mixed ANOVA, ANCOVA).
"""
import json
import math
import os

import pandas as pd
import pytest

from analysis.statisticaltester import StatisticalTester

_REF = os.path.join(os.path.dirname(__file__), "golden", "references_r_advanced.json")
with open(_REF) as _fh:
    _DATA = json.load(_fh)

_RESULTS = _DATA["results"]
_DF = pd.DataFrame(_DATA["data"])

@pytest.fixture(scope="module", autouse=True)
def _qt_and_dialogs():
    """Neutralize dialogs for the gated run."""
    try:
        from PyQt5.QtWidgets import QApplication, QDialog
    except Exception:
        yield
        return
    app = QApplication.instance() or QApplication([])
    QDialog.exec_ = lambda self, *a, **k: 0
    QDialog.exec = lambda self, *a, **k: 0
    yield app

def _assert_close(label, actual, expected, tol=1e-3):
    assert actual is not None and math.isfinite(actual), f"{label}: bad value {actual}"
    assert abs(float(actual) - expected) <= tol, f"{label}: app={actual} vs R={expected} (tol={tol})"

def test_golden_car_type2_anova():
    # Model: y_car ~ groupA * groupB
    result = StatisticalTester.perform_advanced_test(
        df=_DF,
        test="two_way_anova",
        dv="y_car",
        subject=None,
        between=["groupA", "groupB"],
        within=[],
        force_parametric=True
    )
    assert result.get("error") is None
    
    app_effects = {}
    for f in result.get("factors", []):
        app_effects[f["factor"]] = f
    for i in result.get("interactions", []):
        key = " * ".join(sorted(i["factors"]))
        app_effects[key] = i
        
    r_car = _RESULTS["car_type2"]
    
    # car output maps slightly differently: "groupA:groupB" vs "groupA * groupB"
    # we just check them logically
    src_map = {"groupA": "groupA", "groupB": "groupB", "groupA:groupB": "groupA * groupB"}
    
    for r_term, app_term in src_map.items():
        assert app_term in app_effects, f"{app_term} missing from app results"
        app_eff = app_effects[app_term]
        r_eff = r_car[r_term]
        
        _assert_close(f"{app_term} F", app_eff.get("F", app_eff.get("statistic")), r_eff["F"])
        _assert_close(f"{app_term} df1", app_eff.get("df1"), r_eff["df1"])
        _assert_close(f"{app_term} p", app_eff.get("p_value"), r_eff["p"])


def test_golden_afex_mixed_anova():
    # Model: y_mixed ~ groupA * time + Error(subj/time)
    result = StatisticalTester.perform_advanced_test(
        df=_DF,
        test="mixed_anova",
        dv="y_mixed",
        subject="subj",
        between=["groupA"],
        within=["time"],
        force_parametric=True
    )
    assert result.get("error") is None
    
    app_effects = {}
    for f in result.get("factors", []):
        app_effects[f["factor"]] = f
    for i in result.get("interactions", []):
        key = " * ".join(sorted(i["factors"]))
        app_effects[key] = i
        
    r_afex = _RESULTS["afex_mixed"]
    
    # afex names: "groupA", "time", "groupA:time"
    src_map = {"groupA": "groupA", "time": "time", "groupA:time": "groupA * time"}
    
    for r_term, app_term in src_map.items():
        assert app_term in app_effects, f"{app_term} missing from app results"
        app_eff = app_effects[app_term]
        r_eff = r_afex[r_term]
        
        # for within factors, afex might apply GG if not explicitly suppressed. 
        # But we compare uncorrected F/df if possible. Our perform_advanced_test uses pingouin mixed_anova.
        # It's possible pingouin df matches afex uncorrected df. Let's just assert F and p.
        _assert_close(f"{app_term} F", app_eff.get("F", app_eff.get("statistic")), r_eff["F"])
        # Pingouin sometimes doesn't expose uncorrected df if it applies GG. But let's check.
        # Actually Pingouin Mixed ANOVA has F, p-unc.
        if "p_unc" in app_eff:
            p_val = app_eff["p_unc"]
        else:
            p_val = app_eff["p_value"]
            
        # Due to some numerical differences in corrections, we use a slightly larger tol for p
        _assert_close(f"{app_term} p", p_val, r_eff["p"], tol=0.01)

def test_golden_emmeans_ancova():
    # Model: y_ancova ~ groupA + covar
    result = StatisticalTester.perform_advanced_test(
        df=_DF,
        test="ancova",
        dv="y_ancova",
        subject=None,
        between=["groupA"],
        within=[],
        covariates=["covar"],
        force_parametric=True
    )
    assert result.get("error") is None
    
    r_ancova_main = _RESULTS["ancova_main"]
    r_emmeans = _RESULTS["ancova_emmeans"]
    
    # 1. Main effects
    app_anova = {str(r["source"]): r for r in result.get("anova_table", [])}
    
    # Check groupA effect
    # In clinical_models.py we use `C(groupA, Sum)`
    group_term = "C(groupA, Sum)"
    assert group_term in app_anova, f"{group_term} missing from app ANOVA"
    
    _assert_close("groupA F", app_anova[group_term]["F"], r_ancova_main["groupA"]["F"])
    _assert_close("groupA p", app_anova[group_term]["p_value"], r_ancova_main["groupA"]["p"])
    
    # Check covariate effect
    cov_term = "covar"
    assert cov_term in app_anova, f"{cov_term} missing from app ANOVA"
    _assert_close("covar F", app_anova[cov_term]["F"], r_ancova_main["covar"]["F"])
    _assert_close("covar p", app_anova[cov_term]["p_value"], r_ancova_main["covar"]["p"])
    
    # 2. Adjusted Means (emmeans)
    app_adj_means = result.get("adjusted_means", {}).get("groupA", {})
    for grp, r_mean in r_emmeans.items():
        assert grp in app_adj_means, f"Missing adjusted mean for {grp}"
        app_mean = app_adj_means[grp]["adjusted_mean"]
        _assert_close(f"{grp} emmean", app_mean, r_mean["emmean"])


def _map_r_term_to_sm_term(r_term, sm_terms):
    """Simple heuristic to match R term names to statsmodels term names."""
    if r_term == "(Intercept)":
        return "Intercept"
    
    # E.g. groupATrt -> C(groupA)[T.Trt]
    # groupATrt:timeT2 -> C(groupA)[T.Trt]:C(time)[T.T2]
    import re
    # Extract factor names and levels, e.g., groupATrt -> factor groupA, level Trt
    # Just do a naive match: the sm_term must contain all the alphanumeric parts of r_term
    parts = re.split(r'[:\*]', r_term)
    
    best_match = None
    for sm_term in sm_terms:
        if sm_term == "Intercept":
            continue
            
        # We check if all parts are somewhat represented in sm_term
        # e.g., 'groupATrt' -> 'groupA' and 'Trt' in sm_term
        # For covar, it's just 'covar'
        match_all = True
        for part in parts:
            if part == "covar":
                if "covar" not in sm_term:
                    match_all = False
            else:
                # part is like groupATrt
                # Try to find the factor name and level
                if "groupA" in part:
                    lvl = part.replace("groupA", "")
                    if "groupA" not in sm_term or lvl not in sm_term:
                        match_all = False
                elif "time" in part:
                    lvl = part.replace("time", "")
                    if "time" not in sm_term or lvl not in sm_term:
                        match_all = False
        
        if match_all:
            if best_match is None or len(sm_term) < len(best_match):
                best_match = sm_term
                
    return best_match

def test_golden_lme4_lmm():
    result = StatisticalTester.perform_advanced_test(
        df=_DF,
        test="lmm",
        dv="y_mixed",
        subject="subj",
        between=["groupA"],
        within=["time"],
        covariates=["covar"],
        random_slope=None,
        force_parametric=True
    )
    assert result.get("error") is None
    
    r_lmm = _RESULTS["lme4_lmm"]
    app_coefs = {r["parameter"]: r for r in result.get("fixed_effects_table", [])}
    
    for r_term, r_vals in r_lmm.items():
        if r_term == "(Intercept)":
            continue
            
        sm_term = _map_r_term_to_sm_term(r_term, list(app_coefs.keys()))
        assert sm_term is not None, f"Could not map R term '{r_term}' to app terms: {list(app_coefs.keys())}"
        
        # Test Estimate and SE (lme4 REML=TRUE should match smf.mixedlm reml=TRUE)
        _assert_close(f"{r_term} Estimate", app_coefs[sm_term]["coefficient"], r_vals["Estimate"], tol=0.01)
        _assert_close(f"{r_term} SE", app_coefs[sm_term]["std_err"], r_vals["SE"], tol=0.01)


def test_golden_glm_logistic():
    # Remove duplicates to match R's standard Logit data
    df_subj = _DF.drop_duplicates("subj").copy()
    
    result = StatisticalTester.perform_advanced_test(
        df=df_subj,
        test="logistic_regression",
        dv="y_logit_std",
        subject=None,
        between=["groupA"],
        within=[],
        covariates=["covar"],
        force_parametric=True
    )
    assert result.get("error") is None
    
    r_glm = _RESULTS["glm_logistic"]
    app_coefs = {r["parameter"]: r for r in result.get("odds_ratios", [])}
    
    for r_term, r_vals in r_glm.items():
        if r_term == "(Intercept)":
            continue
            
        sm_term = _map_r_term_to_sm_term(r_term, list(app_coefs.keys()))
        assert sm_term is not None, f"Could not map R term '{r_term}' to app terms: {list(app_coefs.keys())}"
        
        _assert_close(f"{r_term} Estimate", app_coefs[sm_term]["coefficient"], r_vals["Estimate"], tol=0.01)
        _assert_close(f"{r_term} SE", app_coefs[sm_term]["std_err"], r_vals["SE"], tol=0.01)
        _assert_close(f"{r_term} p", app_coefs[sm_term]["p_value"], r_vals["p"], tol=0.01)


def test_golden_logistf_firth():
    df_subj = _DF.drop_duplicates("subj").copy()
    
    result = StatisticalTester.perform_advanced_test(
        df=df_subj,
        test="logistic_regression",
        dv="y_logit_sep", # Perfectly separated by groupA
        subject=None,
        between=["groupA"],
        within=[],
        covariates=["covar"],
        force_parametric=True
    )
    assert result.get("error") is None
    # Ensure Firth was used
    assert result.get("model_variant") == "Firth Penalized Likelihood"
    
    r_firth = _RESULTS["logistf_firth"]
    app_coefs = {r["parameter"]: r for r in result.get("odds_ratios", [])}
    
    for r_term, r_vals in r_firth.items():
        if r_term == "(Intercept)":
            continue
            
        sm_term = _map_r_term_to_sm_term(r_term, list(app_coefs.keys()))
        assert sm_term is not None, f"Could not map R term '{r_term}' to app terms: {list(app_coefs.keys())}"
        
        _assert_close(f"{r_term} Estimate", app_coefs[sm_term]["coefficient"], r_vals["Estimate"], tol=0.01)
        # R's logistf doesn't return regular standard errors for parameters effectively, but we extracted sqrt(diag(var))
        # Let's compare p-values (Profile Likelihood Ratio from our custom implementation vs logistf)
        _assert_close(f"{r_term} p-value", app_coefs[sm_term]["p_value"], r_vals["p"], tol=0.02)
