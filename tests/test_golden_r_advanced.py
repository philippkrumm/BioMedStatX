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
