"""Gated golden test: Advanced R Integration (car, afex, emmeans).

Runs the APP's perform_advanced_test on the frozen dataset and compares the outputs
to the frozen R references (Two-Way ANOVA, Mixed ANOVA, ANCOVA).

.. note:: Tolerance calibration (2026-06-16)

   Empirical |app − R| deltas on this frozen dataset:
   - Point estimators (β̂, F, df, emmeans): Δ ≤ 1e-7 → guarded at **tol=1e-4**
   - Standard errors (SE):                  Δ ≤ 1e-7 → guarded at **tol=1e-4**
   - p-values (Wald / ANOVA):               Δ ≤ 1e-4 → guarded at **tol=1e-3** (default)
   - p-values (Firth Profile-LR):           Δ ≤ 1e-7 → guarded at **tol=1e-4**,
     but kept at **1e-3** because profile-likelihood ratio tests use a different
     optimisation path (logistf's penalised IRLS vs. our Newton-Raphson + PLR
     bisection) and future scipy/numpy updates could shift convergence.

   If any assertion trips after a dependency upgrade, compare the new delta
   against the tolerance *before* loosening it — a delta jump from 1e-11 to
   1e-2 signals a regression, not a tolerance problem.
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
    """Assert numerical closeness with labeled diagnostics.

    Default tol=1e-3 is used for ANOVA p-values where GG-correction rounding
    may introduce small cross-implementation drift. For point estimators and
    coefficients, callers pass tol=1e-4 explicitly (see module docstring).
    """
    assert actual is not None and math.isfinite(actual), f"{label}: bad value {actual}"
    delta = abs(float(actual) - expected)
    assert delta <= tol, (
        f"{label}: app={actual} vs R={expected} "
        f"(|Δ|={delta:.2e}, tol={tol})"
    )

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
        
        # Point estimators: algorithmic identity expected (Δ ≤ 1e-7 empirically)
        _assert_close(f"{app_term} F", app_eff.get("F", app_eff.get("statistic")), r_eff["F"], tol=1e-4)
        _assert_close(f"{app_term} df1", app_eff.get("df1"), r_eff["df1"], tol=1e-4)
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
        # F-statistic: algorithmic identity expected (Δ ≤ 1e-7 empirically)
        _assert_close(f"{app_term} F", app_eff.get("F", app_eff.get("statistic")), r_eff["F"], tol=1e-4)
        # Pingouin exposes p-unc (uncorrected) alongside GG-corrected p.
        if "p_unc" in app_eff:
            p_val = app_eff["p_unc"]
        else:
            p_val = app_eff["p_value"]

        # p-value: pingouin vs afex may differ by GG-correction rounding;
        # empirical Δ ≤ 1.5e-4 (groupA:time), so 1e-3 provides a safe margin.
        _assert_close(f"{app_term} p", p_val, r_eff["p"])

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
    
    # Point estimators: algorithmic identity expected (Δ ≤ 1e-7 empirically)
    _assert_close("groupA F", app_anova[group_term]["F"], r_ancova_main["groupA"]["F"], tol=1e-4)
    _assert_close("groupA p", app_anova[group_term]["p_value"], r_ancova_main["groupA"]["p"])
    
    # Check covariate effect
    cov_term = "covar"
    assert cov_term in app_anova, f"{cov_term} missing from app ANOVA"
    _assert_close("covar F", app_anova[cov_term]["F"], r_ancova_main["covar"]["F"], tol=1e-4)
    _assert_close("covar p", app_anova[cov_term]["p_value"], r_ancova_main["covar"]["p"])
    
    # 2. Adjusted Means (emmeans)
    app_adj_means = result.get("adjusted_means", {}).get("groupA", {})
    for grp, r_mean in r_emmeans.items():
        assert grp in app_adj_means, f"Missing adjusted mean for {grp}"
        app_mean = app_adj_means[grp]["adjusted_mean"]
        _assert_close(f"{grp} emmean", app_mean, r_mean["emmean"], tol=1e-4)


# ---------------------------------------------------------------------------
# ARCHITECTURAL NOTE — Golden-test-only nomenclature bridge
# ---------------------------------------------------------------------------
# The following function maps R's coefficient names (e.g. "groupATrt") to the
# Patsy/statsmodels equivalents (e.g. "C(groupA)[T.Trt]") using a hardcoded
# heuristic that is ONLY safe for the deterministic, sanitized factor levels
# in the frozen golden reference dataset (groupA ∈ {Ctrl, Trt}, time ∈
# {T1, T2, T3}, covar is continuous).
#
# ⚠  DO NOT import or reuse this function in production code (src/).
#    It will produce incorrect mappings when factor levels contain
#    special characters, whitespace, or overlapping substrings.
# ---------------------------------------------------------------------------

def _map_r_term_to_sm_term(r_term: str, sm_terms: list[str]) -> str | None:
    """Match an R coefficient name to a statsmodels term name.

    **Scope**: This is a test-internal helper for the frozen golden reference
    suite.  It exploits the fact that the golden dataset's factor levels
    (``Ctrl``, ``Trt``, ``T1``–``T3``, ``covar``) are short, ASCII-only,
    and non-overlapping — properties that a general-purpose mapper cannot
    assume.

    **DO NOT USE** outside ``test_golden_r_advanced.py``.
    """
    if r_term == "(Intercept)":
        return "Intercept"

    import re
    parts = re.split(r'[:\*]', r_term)

    best_match = None
    for sm_term in sm_terms:
        if sm_term == "Intercept":
            continue

        match_all = True
        for part in parts:
            if part == "covar":
                if "covar" not in sm_term:
                    match_all = False
            else:
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
        
        # Coefficients and SEs: lme4 REML vs smf.mixedlm REML use the same
        # likelihood; empirical Δ ≤ 1e-7.  Tol=1e-4 catches solver drift.
        _assert_close(f"{r_term} Estimate", app_coefs[sm_term]["coefficient"], r_vals["Estimate"], tol=1e-4)
        _assert_close(f"{r_term} SE", app_coefs[sm_term]["std_err"], r_vals["SE"], tol=1e-4)


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
        
        # Standard GLM (no separation): statsmodels IRLS == R glm() IRLS.
        # Empirical Δ ≤ 1e-11.  Tol=1e-4 catches solver drift.
        _assert_close(f"{r_term} Estimate", app_coefs[sm_term]["coefficient"], r_vals["Estimate"], tol=1e-4)
        _assert_close(f"{r_term} SE", app_coefs[sm_term]["std_err"], r_vals["SE"], tol=1e-4)
        _assert_close(f"{r_term} p", app_coefs[sm_term]["p_value"], r_vals["p"], tol=1e-4)


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
        
        # Firth coefficient: penalised IRLS (logistf) vs our Newton-Raphson.
        # Empirical Δ ≤ 3.2e-7.  Tol=1e-4 is safe; if this trips after a
        # scipy upgrade, compare the new Δ magnitude before loosening.
        _assert_close(f"{r_term} Estimate", app_coefs[sm_term]["coefficient"], r_vals["Estimate"], tol=1e-4)
        # Profile-Likelihood-Ratio p-values: logistf's PLR vs our bisection
        # PLR.  Empirical Δ ≤ 1e-7.  Tol=1e-3 provides margin for the
        # different optimisation paths (see module docstring).
        _assert_close(f"{r_term} p-value", app_coefs[sm_term]["p_value"], r_vals["p"])
