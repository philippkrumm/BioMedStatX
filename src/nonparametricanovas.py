# --- Minimal test for posthoc_marginaleffects ---
# (Moved to end of file to ensure all symbols are defined)
# --- Utility: Modern post hoc analysis using marginaleffects ---
def posthoc_marginaleffects(
    result,
    by=None,
    variables=None,
    plot=False,
    plot_type="predictions",
    to_pandas=True,
    **kwargs
):
    """
    Compute marginal means, pairwise comparisons, and optionally plot post hoc results
    for a fitted GLMM/GEE/MixedLM model using the marginaleffects package.

    Parameters
    ----------
    result : statsmodels result object
        The fitted model result (e.g., from GLMMMixedANOVA, GLMMTwoWayANOVA, GEERMANOVA).
    by : str or list, optional
        Factor(s) to group by for marginal means (e.g., ["FactorA", "FactorB"])
    variables : str or list, optional
        Factor(s) for pairwise comparisons (e.g., "FactorB")
    plot : bool, default False
        If True, show a plot of marginal means or comparisons
    plot_type : str, default "predictions"
        "predictions" for marginal means, "comparisons" for pairwise contrasts
    to_pandas : bool, default True
        If True, convert results to pandas DataFrame
    **kwargs :
        Additional arguments passed to marginaleffects functions

    Returns
    -------
    dict with keys:
        "marginal_means": marginal means table
        "comparisons": pairwise comparisons table
        "plot": plot object (if plot=True)

    Example
    -------
    >>> model = GLMMMixedANOVA().fit(df, dv="Value", between=["FactorA"], within=["FactorB"], subject="Subject")
    >>> res = model.result
    >>> out = posthoc_marginaleffects(res, by=["FactorA", "FactorB"], variables="FactorB", plot=True)
    >>> print(out["marginal_means"])
    >>> print(out["comparisons"])

    Notes
    -----
    - Requires marginaleffects >= 0.12.0 (pip install marginaleffects)
    - For MixedLM, only fixed effects are supported (see marginaleffects roadmap)
    - Outputs are Polars DataFrames by default; set to_pandas=True to convert
    - For more advanced options, see marginaleffects documentation
    """
    if avg_predictions is None or comparisons is None:
        raise ImportError("marginaleffects is not installed. Please run 'pip install marginaleffects'.")
    # Marginal means
    mm = avg_predictions(result, by=by, **kwargs)
    if to_pandas:
        mm = mm.to_pandas()
    # Pairwise comparisons
    cmp = None
    if variables is not None:
        cmp = comparisons(result, variables=variables, by=by, **kwargs)
        if to_pandas:
            cmp = cmp.to_pandas()
    # Plot
    plt_obj = None
    if plot:
        if plot_type == "predictions":
            plt_obj = plot_predictions(result, by=by)
        elif plot_type == "comparisons" and variables is not None:
            plt_obj = plot_comparisons(result, variables=variables, by=by)
        if plt_obj is not None:
            plt_obj.show()
    return {"marginal_means": mm, "comparisons": cmp, "plot": plt_obj}
# --- Assumption checks and automated decision logic ---
from scipy.stats import shapiro, levene

# --- marginaleffects: modern post hoc analysis for GLMM/GEE ---
# To use the post hoc utility below, install marginaleffects:
#   pip install marginaleffects
try:
    from marginaleffects import avg_predictions, comparisons, plot_predictions, plot_comparisons
except ImportError:
    avg_predictions = comparisons = plot_predictions = plot_comparisons = None
    # The posthoc_marginaleffects function will raise an error if called without marginaleffects
import warnings
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.linalg import block_diag
import statsmodels.formula.api as smf


def _holm_correct(p_values):
    """Holm step-down correction. Returns list of corrected p-values (same order)."""
    n = len(p_values)
    if n == 0:
        return []
    order = np.argsort(p_values)
    corrected = np.array(p_values, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = float(p_values[idx]) * (n - rank)
        running_max = max(running_max, adj)
        corrected[idx] = min(running_max, 1.0)
    return corrected.tolist()


def _wilcoxon_posthoc_comp(arr1, arr2, label1, label2, alpha):
    """Paired Wilcoxon signed-rank comparison dict (raw p; Holm applied by caller)."""
    diffs = np.asarray(arr1, float) - np.asarray(arr2, float)
    diffs = diffs[~np.isnan(diffs)]
    n = len(diffs)
    if n < 3 or np.all(diffs == 0):
        return None
    try:
        stat, p_raw = sp_stats.wilcoxon(diffs, alternative='two-sided', zero_method='wilcox')
    except Exception:
        return None
    total = n * (n + 1) / 2.0
    rbc = abs((2.0 * float(stat) - total) / total)   # rank-biserial correlation
    return {
        "group1": label1, "group2": label2,
        "test": "Wilcoxon Signed-Rank",
        "statistic": float(stat),
        "p_value": float(p_raw),    # overwritten after Holm by caller
        "corrected": False, "correction": "None",
        "significant": float(p_raw) < alpha,
        "effect_size": round(rbc, 4), "effect_size_type": "rank_biserial_r",
        "confidence_interval": None,
    }


def _mwu_posthoc_comp(arr1, arr2, label1, label2, alpha):
    """Mann-Whitney U comparison dict (raw p; Holm applied by caller)."""
    a1 = np.asarray(arr1, float); a1 = a1[~np.isnan(a1)]
    a2 = np.asarray(arr2, float); a2 = a2[~np.isnan(a2)]
    n1, n2 = len(a1), len(a2)
    if n1 < 2 or n2 < 2:
        return None
    try:
        stat, p_raw = sp_stats.mannwhitneyu(a1, a2, alternative='two-sided', use_continuity=True)
    except Exception:
        return None
    rbc = abs((2.0 * float(stat) - n1 * n2) / (n1 * n2))   # rank-biserial correlation
    return {
        "group1": label1, "group2": label2,
        "test": "Mann-Whitney U",
        "statistic": float(stat),
        "p_value": float(p_raw),    # overwritten after Holm by caller
        "corrected": False, "correction": "None",
        "significant": float(p_raw) < alpha,
        "effect_size": round(rbc, 4), "effect_size_type": "rank_biserial_r",
        "confidence_interval": None,
    }


def _apply_holm(raw_comps, alpha):
    """In-place Holm correction of p_value fields. Returns the list."""
    if not raw_comps:
        return []
    ps = [c["p_value"] for c in raw_comps]
    corrected = _holm_correct(ps)
    for comp, cp in zip(raw_comps, corrected):
        comp["p_value"] = round(cp, 6)
        comp["corrected"] = True
        comp["correction"] = "Holm"
        comp["significant"] = cp < alpha
    return raw_comps


# ---------------------------------------------------------------------------

def perform_friedman_test(data, dv, within_factor, subject_col, alpha=0.05):
    """
    Friedman test as nonparametric fallback for Repeated-Measures ANOVA.
    Valid for small samples (n>=3 per cell). Uses scipy.stats.friedmanchisquare.

    Returns a result dict compatible with the downstream exporter and
    _run_modern_fallback_posthoc post-hoc infrastructure.
    """
    warnings_list = []
    error = None

    try:
        # --- Wide pivot (subjects × within-levels) ---
        wide = data.pivot_table(index=subject_col, columns=within_factor, values=dv, aggfunc='mean')
        wide = wide.dropna()  # complete cases only

        level_cols = list(wide.columns)
        k = len(level_cols)
        n_subjects = len(wide)

        if k < 2:
            raise ValueError(f"Friedman test requires at least 2 within-levels, got {k}.")
        if n_subjects < 3:
            raise ValueError(f"Friedman test requires at least 3 subjects, got {n_subjects}.")

        if k == 2:
            warnings_list.append("Only 2 time points: consider paired Wilcoxon instead of Friedman.")
        if n_subjects < 5:
            warnings_list.append(f"Very few subjects (n={n_subjects}). Friedman test may have low power.")

        # --- Run Friedman test ---
        chi2_stat, p_value = sp_stats.friedmanchisquare(*[wide[col].values for col in level_cols])
        df1 = k - 1
        chi2_stat = float(chi2_stat)
        p_value = float(p_value)

        # --- Kendall's W: W = χ² / (n * (k-1)), range [0, 1] ---
        kendall_w = float(np.clip(chi2_stat / (n_subjects * (k - 1)), 0.0, 1.0))

        # --- Descriptive stats per within-level ---
        descriptive = {}
        for col in level_cols:
            vals = wide[col].dropna()
            n = len(vals)
            mean = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            stderr = float(sd / np.sqrt(n)) if n > 0 else None
            descriptive[f"{within_factor}={col}"] = {
                "n": n, "mean": mean, "sd": sd, "std": sd, "stderr": stderr,
                "median": float(np.median(vals)),
                "min": float(np.min(vals)), "max": float(np.max(vals)),
                "ci_lower": None, "ci_upper": None,
            }

        # --- ANOVA table ---
        anova_table = pd.DataFrame([{
            "Source": within_factor,
            "Chi2": chi2_stat,
            "F": chi2_stat,           # alias for downstream compatibility
            "Wald_Chi2": chi2_stat,
            "DF1": df1,
            "DF2": None,
            "p_unc": p_value,
            "StatisticType": "Friedman Chi-square",
        }])

        # --- factors list ---
        factors = [{
            "factor": within_factor,
            "type": "within",
            "F": chi2_stat,
            "Wald_Chi2": chi2_stat,
            "p_value": p_value,
            "df1": df1,
            "df2": None,
            "effect_size": kendall_w,
            "effect_size_type": "Kendall's W",
        }]

        primary_effect = {
            "source": within_factor,
            "kind": "main",
            "policy": "interaction_first",
            "p_value": p_value,
            "wald_chi2": chi2_stat,
        }

        analysis_note = (
            f"Assumptions for parametric Repeated Measures ANOVA were violated. "
            f"A Friedman test was applied as the nonparametric alternative "
            f"(Chi\u00b2({df1}) = {chi2_stat:.3f}, p = {p_value:.4f}, "
            f"n = {n_subjects} subjects, k = {k} measurements)."
        )

        # --- Post-hoc: pairwise Wilcoxon signed-rank (Holm-corrected), only if significant ---
        posthoc_comps = []
        posthoc_name = None
        if p_value < alpha and k >= 2:
            from itertools import combinations as _comb
            raw = []
            for c1, c2 in _comb(level_cols, 2):
                comp = _wilcoxon_posthoc_comp(
                    wide[c1].values, wide[c2].values,
                    f"{within_factor}={c1}", f"{within_factor}={c2}", alpha
                )
                if comp is not None:
                    raw.append(comp)
            posthoc_comps = _apply_holm(raw, alpha)
            if posthoc_comps:
                posthoc_name = f"Pairwise Wilcoxon Signed-Rank (Holm, n={n_subjects} subjects)"

        return {
            "test": "Friedman Test",
            "p_value": p_value,
            "statistic": chi2_stat,
            "posthoc_test": posthoc_name,
            "pairwise_comparisons": posthoc_comps,
            "descriptive": descriptive,
            "descriptive_transformed": {},
            "alpha": alpha,
            "null_hypothesis": "The distribution is the same across all repeated measurements.",
            "alternative_hypothesis": "At least one repeated measurement differs from the others.",
            "confidence_interval": None,
            "ci_level": 0.95,
            "power": None,
            "effect_size": kendall_w,
            "effect_size_type": "Kendall's W",
            "error": None,
            "df1": df1,
            "df2": None,
            "model_type": "Friedman",
            "model_class": "Friedman",
            "model_family": None,
            "model_link": None,
            "family_diagnostics": {},
            "cov_struct_used": None,
            "covariance_estimator": None,
            "factors": factors,
            "interactions": [],
            "anova_table": anova_table,
            "primary_effect": primary_effect,
            "primary_effect_policy": "interaction_first",
            "interaction_significant": False,
            "interpretation_order": ["main_effects"],
            "recommendation": "non_parametric",
            "fallback_model_used": False,
            "analysis_note": analysis_note,
            "warnings": warnings_list,
            "StatisticType": "Friedman Chi-square",
        }

    except Exception as exc:
        return {
            "test": "Friedman Test",
            "p_value": None, "statistic": None,
            "posthoc_test": None, "pairwise_comparisons": [],
            "descriptive": {}, "descriptive_transformed": {},
            "alpha": alpha,
            "null_hypothesis": "The distribution is the same across all repeated measurements.",
            "alternative_hypothesis": "At least one repeated measurement differs from the others.",
            "confidence_interval": None, "ci_level": 0.95,
            "power": None, "effect_size": None, "effect_size_type": None,
            "error": str(exc), "df1": None, "df2": None,
            "model_type": "Friedman",
            "model_class": "Friedman",
            "model_family": None, "model_link": None,
            "family_diagnostics": {}, "cov_struct_used": None, "covariance_estimator": None,
            "factors": [], "interactions": [], "anova_table": None,
            "primary_effect": None, "primary_effect_policy": "interaction_first",
            "interaction_significant": False, "interpretation_order": ["main_effects"],
            "recommendation": "non_parametric", "fallback_model_used": False,
            "analysis_note": f"Friedman test failed: {exc}",
            "warnings": warnings_list, "StatisticType": "Friedman Chi-square",
        }


def perform_freedman_lane_test(data, dv, factor_a, factor_b, alpha=0.05, n_permutations=5000, seed=None):
    """
    Freedman-Lane permutation test as nonparametric fallback for Two-Way ANOVA.

    For each effect (A, B, A×B):
      1. Fit the reduced OLS model (without the tested effect).
      2. Permute residuals of the reduced model, reconstruct pseudo-outcomes.
      3. Refit the full model on pseudo-outcomes and record F-statistics.
      4. p_perm = (#{F_perm >= F_obs} + 1) / (n_perm + 1)

    Returns a result dict compatible with the downstream exporter.
    """
    warnings_list = []
    rng = np.random.default_rng(seed)

    try:
        df = data[[dv, factor_a, factor_b]].dropna().copy()

        # Sanitize column names for patsy formulas
        import re
        _safe = lambda s: re.sub(r"\W+", "_", str(s)).strip("_") or "col"
        safe_dv = _safe(dv)
        safe_a  = _safe(factor_a)
        safe_b  = _safe(factor_b)
        rename_map = {dv: safe_dv, factor_a: safe_a, factor_b: safe_b}
        df = df.rename(columns=rename_map)

        # Ensure factors are categorical strings
        df[safe_a] = df[safe_a].astype(str)
        df[safe_b] = df[safe_b].astype(str)

        n_total = len(df)
        cell_counts = df.groupby([safe_a, safe_b]).size()
        min_cell_n  = int(cell_counts.min()) if len(cell_counts) > 0 else 0

        if min_cell_n < 5:
            warnings_list.append(
                f"Very small cell sizes (min n={min_cell_n}). "
                "Permutation p-values have limited resolution."
            )
        if n_total < 12:
            warnings_list.append(
                f"Total N < 12: very few unique permutations possible. Results are exploratory."
            )

        formula_full    = f"{safe_dv} ~ C({safe_a}) + C({safe_b}) + C({safe_a}):C({safe_b})"
        # Freedman-Lane reduced models: nuisance-only (no interaction for main-effect tests).
        # Using "Y ~ C(B) + C(A):C(B)" as reduced for A is wrong in balanced designs:
        # that model spans the same column space as the full model, making F_obs ≈ 0.
        # Standard Freedman-Lane approach: reduced contains only the other main effect.
        formula_no_a    = f"{safe_dv} ~ C({safe_b})"
        formula_no_b    = f"{safe_dv} ~ C({safe_a})"
        formula_no_inter= f"{safe_dv} ~ C({safe_a}) + C({safe_b})"

        full_model = smf.ols(formula_full, data=df).fit()
        RSS_full   = full_model.ssr
        df_resid   = full_model.df_resid

        def _f_obs_and_perm(formula_reduced, effect_label):
            """Return (F_obs, df_effect, p_perm, p_parametric)."""
            red_model  = smf.ols(formula_reduced, data=df).fit()
            RSS_red    = red_model.ssr
            df_effect  = max(red_model.df_model - full_model.df_model, 1)
            # Recompute df_effect as difference in residual df (more reliable)
            df_effect  = int(round(red_model.df_resid - full_model.df_resid))
            if df_effect < 1:
                df_effect = 1
            F_obs = ((RSS_red - RSS_full) / df_effect) / (RSS_full / df_resid)
            F_obs = max(F_obs, 0.0)

            y_hat_red = red_model.fittedvalues.values
            e_red     = red_model.resid.values

            F_perm_arr = np.empty(n_permutations)
            for i in range(n_permutations):
                e_perm  = rng.permutation(e_red)
                y_perm  = y_hat_red + e_perm
                df_perm = df.copy()
                df_perm[safe_dv] = y_perm
                fm = smf.ols(formula_full,    data=df_perm).fit()
                rm = smf.ols(formula_reduced, data=df_perm).fit()
                rss_f = fm.ssr; rss_r = rm.ssr
                F_p = ((rss_r - rss_f) / df_effect) / (rss_f / df_resid)
                F_perm_arr[i] = max(F_p, 0.0)

            p_perm = (np.sum(F_perm_arr >= F_obs) + 1) / (n_permutations + 1)
            # Parametric reference p: derived from the same reduced vs full comparison
            F_para = float(F_obs)
            p_parametric = float(sp_stats.f.sf(F_para, dfn=df_effect, dfd=df_resid)) if F_para > 0 else 1.0
            return float(F_obs), int(df_effect), float(p_perm), p_parametric

        # Compute for each effect
        F_A,   df_A,   p_perm_A,   p_par_A   = _f_obs_and_perm(formula_no_a,    "A")
        F_B,   df_B,   p_perm_B,   p_par_B   = _f_obs_and_perm(formula_no_b,    "B")
        F_AB,  df_AB,  p_perm_AB,  p_par_AB  = _f_obs_and_perm(formula_no_inter,"AB_interaction")
        df2 = int(df_resid)

        # --- Descriptive stats ---
        descriptive = {}
        for a_val in sorted(df[safe_a].unique()):
            for b_val in sorted(df[safe_b].unique()):
                subset = df[(df[safe_a] == a_val) & (df[safe_b] == b_val)][safe_dv].dropna()
                key = f"{factor_a}={a_val}, {factor_b}={b_val}"
                n = len(subset)
                mean = float(np.mean(subset)) if n > 0 else None
                sd   = float(np.std(subset, ddof=1)) if n > 1 else 0.0
                descriptive[key] = {
                    "n": n, "mean": mean, "sd": sd, "std": sd,
                    "stderr": float(sd / np.sqrt(n)) if n > 0 else None,
                    "median": float(np.median(subset)) if n > 0 else None,
                    "min": float(np.min(subset)) if n > 0 else None,
                    "max": float(np.max(subset)) if n > 0 else None,
                    "ci_lower": None, "ci_upper": None,
                }

        # --- ANOVA table ---
        anova_table = pd.DataFrame([
            {"Source": factor_a,              "F": F_A,  "p-perm": p_perm_A,  "p-parametric": p_par_A,  "DF1": df_A,  "DF2": df2, "StatisticType": "Permutation F (Freedman-Lane)", "Wald_Chi2": F_A},
            {"Source": factor_b,              "F": F_B,  "p-perm": p_perm_B,  "p-parametric": p_par_B,  "DF1": df_B,  "DF2": df2, "StatisticType": "Permutation F (Freedman-Lane)", "Wald_Chi2": F_B},
            {"Source": f"{factor_a}:{factor_b}", "F": F_AB, "p-perm": p_perm_AB, "p-parametric": p_par_AB, "DF1": df_AB, "DF2": df2, "StatisticType": "Permutation F (Freedman-Lane)", "Wald_Chi2": F_AB},
        ])
        # Also expose p-perm as p_unc for exporter compatibility
        anova_table["p_unc"] = anova_table["p-perm"]

        # Cohen's f (approx.) from permutation F: f = sqrt(F * df_effect / n_total)
        # Labelled "approx." because F comes from permutations, not OLS SS decomposition.
        def _cohens_f_approx(F_val, df_val):
            return float(np.sqrt(max(F_val * df_val / n_total, 0.0)))

        factors = [
            {"factor": factor_a, "type": "between", "F": F_A,  "Wald_Chi2": F_A,  "p_value": p_perm_A,  "df1": df_A,  "df2": df2, "effect_size": _cohens_f_approx(F_A, df_A),   "effect_size_type": "Cohen's f (approx.)"},
            {"factor": factor_b, "type": "between", "F": F_B,  "Wald_Chi2": F_B,  "p_value": p_perm_B,  "df1": df_B,  "df2": df2, "effect_size": _cohens_f_approx(F_B, df_B),   "effect_size_type": "Cohen's f (approx.)"},
        ]
        interactions = [
            {"factors": [factor_a, factor_b], "F": F_AB, "Wald_Chi2": F_AB, "p_value": p_perm_AB, "df1": df_AB, "df2": df2, "effect_size": _cohens_f_approx(F_AB, df_AB), "effect_size_type": "Cohen's f (approx.)"},
        ]

        # --- Primary effect (interaction_first policy) ---
        interaction_significant = p_perm_AB < alpha
        if interaction_significant:
            primary_source = f"{factor_a}:{factor_b}"
            primary_p      = p_perm_AB
            primary_F      = F_AB
            primary_kind   = "interaction"
            interpretation_order = ["interaction", "main_effects_cautious"]
        else:
            if p_perm_A <= p_perm_B:
                primary_source, primary_p, primary_F = factor_a, p_perm_A, F_A
            else:
                primary_source, primary_p, primary_F = factor_b, p_perm_B, F_B
            primary_kind = "main"
            interpretation_order = ["main_effects", "interaction"]

        primary_effect = {
            "source": primary_source,
            "kind": primary_kind,
            "policy": "interaction_first",
            "p_value": primary_p,
            "wald_chi2": primary_F,
        }

        analysis_note = (
            f"Assumptions for parametric Two-Way ANOVA were violated. "
            f"A Freedman-Lane permutation test ({n_permutations} permutations, seed={seed}) "
            f"was used as nonparametric alternative for factors '{factor_a}' and '{factor_b}'."
        )

        # --- Post-hoc: pairwise MWU for significant main effects and interaction (Holm) ---
        posthoc_comps = []
        posthoc_name = None
        from itertools import combinations as _comb
        raw = []
        a_levels = sorted(df[safe_a].unique())
        b_levels = sorted(df[safe_b].unique())

        if p_perm_A < alpha and len(a_levels) >= 2:
            for v1, v2 in _comb(a_levels, 2):
                comp = _mwu_posthoc_comp(
                    df[df[safe_a] == v1][safe_dv].values,
                    df[df[safe_a] == v2][safe_dv].values,
                    f"{factor_a}={v1}", f"{factor_a}={v2}", alpha
                )
                if comp is not None:
                    raw.append(comp)

        if p_perm_B < alpha and len(b_levels) >= 2:
            for v1, v2 in _comb(b_levels, 2):
                comp = _mwu_posthoc_comp(
                    df[df[safe_b] == v1][safe_dv].values,
                    df[df[safe_b] == v2][safe_dv].values,
                    f"{factor_b}={v1}", f"{factor_b}={v2}", alpha
                )
                if comp is not None:
                    raw.append(comp)

        if p_perm_AB < alpha:
            cells = [(av, bv) for av in a_levels for bv in b_levels]
            for (a1, b1), (a2, b2) in _comb(cells, 2):
                comp = _mwu_posthoc_comp(
                    df[(df[safe_a] == a1) & (df[safe_b] == b1)][safe_dv].values,
                    df[(df[safe_a] == a2) & (df[safe_b] == b2)][safe_dv].values,
                    f"{factor_a}={a1}, {factor_b}={b1}",
                    f"{factor_a}={a2}, {factor_b}={b2}", alpha
                )
                if comp is not None:
                    raw.append(comp)

        if raw:
            posthoc_comps = _apply_holm(raw, alpha)
            posthoc_name = "Pairwise Mann-Whitney U (Holm-corrected)"

        return {
            "test": "Freedman-Lane Permutation Test",
            "p_value": primary_p,
            "statistic": primary_F,
            "posthoc_test": posthoc_name,
            "pairwise_comparisons": posthoc_comps,
            "descriptive": descriptive,
            "descriptive_transformed": {},
            "alpha": alpha,
            "null_hypothesis": "The means of all groups are equal for all factors and their interaction.",
            "alternative_hypothesis": "At least one group mean differs.",
            "confidence_interval": None,
            "ci_level": 0.95,
            "power": None,
            "effect_size": _cohens_f_approx(primary_F, df_AB if interaction_significant else (df_A if p_perm_A <= p_perm_B else df_B)),
            "effect_size_type": "Cohen's f (approx.)",
            "error": None,
            "df1": df_AB if interaction_significant else (df_A if p_perm_A <= p_perm_B else df_B),
            "df2": df2,
            "model_type": "FreedmanLanePermutation",
            "model_class": "Freedman-Lane Permutation",
            "model_family": None,
            "model_link": None,
            "family_diagnostics": {},
            "cov_struct_used": None,
            "covariance_estimator": None,
            "factors": factors,
            "interactions": interactions,
            "anova_table": anova_table,
            "primary_effect": primary_effect,
            "primary_effect_policy": "interaction_first",
            "interaction_significant": interaction_significant,
            "interpretation_order": interpretation_order,
            "recommendation": "non_parametric",
            "fallback_model_used": False,
            "analysis_note": analysis_note,
            "warnings": warnings_list,
            "StatisticType": "Permutation F (Freedman-Lane)",
            "n_permutations": n_permutations,
        }

    except Exception as exc:
        return {
            "test": "Freedman-Lane Permutation Test",
            "p_value": None, "statistic": None,
            "posthoc_test": None, "pairwise_comparisons": [],
            "descriptive": {}, "descriptive_transformed": {},
            "alpha": alpha,
            "null_hypothesis": "The means of all groups are equal.",
            "alternative_hypothesis": "At least one group mean differs.",
            "confidence_interval": None, "ci_level": 0.95,
            "power": None, "effect_size": None, "effect_size_type": None,
            "error": str(exc), "df1": None, "df2": None,
            "model_type": "FreedmanLanePermutation",
            "model_class": "Freedman-Lane Permutation",
            "model_family": None, "model_link": None,
            "family_diagnostics": {}, "cov_struct_used": None, "covariance_estimator": None,
            "factors": [], "interactions": [], "anova_table": None,
            "primary_effect": None, "primary_effect_policy": "interaction_first",
            "interaction_significant": False, "interpretation_order": ["main_effects"],
            "recommendation": "non_parametric", "fallback_model_used": False,
            "analysis_note": f"Freedman-Lane permutation test failed: {exc}",
            "warnings": warnings_list, "StatisticType": "Permutation F (Freedman-Lane)",
            "n_permutations": n_permutations,
        }


def perform_brunner_langer_ats(data, dv, between_factor, within_factor, subject_col, alpha=0.05):
    """
    Brunner-Langer ANOVA-Type Statistic (ATS) as nonparametric fallback for Mixed ANOVA.

    Implements the F1-LD-F1 design (1 Between × 1 Within factor) from:
      Brunner, Domhof, Langer (2002). "Nonparametric Analysis of Longitudinal Data in
      Factorial Experiments." Wiley.

    Projection matrices used directly (idempotent, no pseudoinverse needed).
    df2 for Between effect via Satterthwaite marginal-covariance approximation.
    """
    warnings_list = []

    try:
        df = data[[dv, between_factor, within_factor, subject_col]].dropna().copy()

        between_levels = sorted(df[between_factor].dropna().unique())
        within_levels  = sorted(df[within_factor].dropna().unique())
        a = len(between_levels)
        t = len(within_levels)

        if a < 2:
            raise ValueError(f"Between factor must have at least 2 levels, got {a}.")
        if t < 2:
            raise ValueError(f"Within factor must have at least 2 levels, got {t}.")

        # --- Global mid-ranks (all observations ranked together, ties → average) ---
        all_vals = df[dv].values.astype(float)
        ranks    = sp_stats.rankdata(all_vals, method='average')
        N        = len(ranks)
        df['_rank'] = ranks

        # --- Per-group wide rank matrices and covariance ---
        group_ns  = []
        V_hats    = []   # (t×t) per-group rank covariance / N  (Ŝ_i per Brunner)
        RTE_rows  = []   # For output DataFrame

        for i, b_val in enumerate(between_levels):
            grp = df[df[between_factor] == b_val]
            # Pivot to wide: rows=subjects, cols=within_levels
            wide = grp.pivot_table(index=subject_col, columns=within_factor, values='_rank', aggfunc='mean')
            # Ensure columns are in sorted within_levels order
            wide = wide.reindex(columns=within_levels)
            # Drop subjects with any missing within-level
            wide = wide.dropna()
            n_i  = len(wide)
            group_ns.append(n_i)

            if n_i < 2:
                raise ValueError(
                    f"Group '{b_val}' has only {n_i} complete subjects. "
                    "At least 2 are needed for covariance estimation."
                )
            if n_i < 3:
                warnings_list.append(
                    f"Group '{b_val}' has n={n_i} < 3. Covariance estimation may be unreliable."
                )

            R_i = wide.values.astype(float)  # raw ranks, shape (n_i, t)
            # Ŝ_i = cov(R_i)/N per Brunner et al. (2002); V̂_N = block_diag(Ŝ_i/n_i)
            V_hat_i = np.cov(R_i.T, ddof=1) / N  # shape (t, t)
            V_hats.append(V_hat_i)

            # RTEs per cell
            for s, w_val in enumerate(within_levels):
                cell_ranks = grp[grp[within_factor] == w_val]['_rank'].dropna().values
                rte = (np.mean(cell_ranks) - 0.5) / N if len(cell_ranks) > 0 else np.nan
                RTE_rows.append({
                    "between_group": b_val,
                    "within_level": w_val,
                    "RTE": rte,
                    "n": len(cell_ranks),
                })

        if min(n_i * t for n_i in group_ns) < 6:
            warnings_list.append(
                "Very few observations per cell. ATS may have reduced power. Interpret cautiously."
            )

        # --- Block-diagonal total covariance V_N ---
        V_N = block_diag(*[V_hats[i] / group_ns[i] for i in range(a)])  # shape (a*t, a*t)

        # --- RTE vector p_hat (row-major: group0_time0, group0_time1, ..., group1_time0, ...) ---
        RTE_df  = pd.DataFrame(RTE_rows)
        RTE_mat = np.array([[
            RTE_df[(RTE_df['between_group'] == b_val) & (RTE_df['within_level'] == w_val)]['RTE'].values[0]
            for w_val in within_levels
        ] for b_val in between_levels])   # shape (a, t)
        p_hat = RTE_mat.flatten(order='C')   # (a*t,)

        # --- Idempotent projection matrices (a*t × a*t) ---
        I_a = np.eye(a); J_a = np.ones((a, a))
        I_t = np.eye(t); J_t = np.ones((t, t))
        T_between = np.kron(I_a - J_a / a, J_t / t)   # Between: rank a-1
        T_within  = np.kron(J_a / a,       I_t - J_t / t)  # Within:  rank t-1
        T_inter   = np.kron(I_a - J_a / a, I_t - J_t / t)  # Interaction: rank (a-1)(t-1)

        def _ats_and_df(T_mat):
            """Compute ATS, Box df1, and trace product for a given projection matrix."""
            TV     = T_mat @ V_N
            tr_TV  = np.trace(TV)
            if tr_TV <= 0:
                return 0.0, 1.0
            ATS  = float(N * (p_hat @ T_mat @ p_hat) / tr_TV)
            tr_TV2 = np.trace(TV @ TV)
            f_hat  = float(tr_TV ** 2 / tr_TV2) if tr_TV2 > 0 else 1.0
            return ATS, max(f_hat, 1.0)

        ATS_A,  f_A  = _ats_and_df(T_between)
        ATS_T,  f_T  = _ats_and_df(T_within)
        ATS_AT, f_AT = _ats_and_df(T_inter)

        # --- p-values ---
        # Within + Interaction: F(f_hat, ∞) ≡ Chi²(f_hat)/f_hat
        p_T  = float(1.0 - sp_stats.chi2.cdf(ATS_T  * f_T,  df=f_T))
        p_AT = float(1.0 - sp_stats.chi2.cdf(ATS_AT * f_AT, df=f_AT))

        # Between: finite df2 via Satterthwaite marginal-covariance approximation
        # λ_i = (1_t^T V_hat_i 1_t) / (t² · n_i) — marginal variance of group-average RTE
        ones_t = np.ones(t)
        lambda_i  = [float(ones_t @ V_hats[i] @ ones_t) / (t ** 2 * group_ns[i]) for i in range(a)]
        lambda_sum = sum(lambda_i)
        denom_f2   = sum(
            li ** 2 / (group_ns[i] - 1)
            for i, li in enumerate(lambda_i)
            if group_ns[i] > 1
        )
        f_hat_2 = float(lambda_sum ** 2 / denom_f2) if denom_f2 > 0 else np.inf
        p_A  = float(1.0 - sp_stats.f.cdf(ATS_A, dfn=f_A, dfd=f_hat_2))

        df2_between = f_hat_2 if np.isfinite(f_hat_2) else None
        df2_inf     = None  # Represent ∞ as None for JSON/Excel compatibility

        # --- Descriptive stats ---
        descriptive = {}
        for b_val in between_levels:
            for w_val in within_levels:
                subset = df[(df[between_factor] == b_val) & (df[within_factor] == w_val)][dv].dropna()
                key = f"{between_factor}={b_val}, {within_factor}={w_val}"
                n = len(subset)
                mean = float(np.mean(subset)) if n > 0 else None
                sd   = float(np.std(subset, ddof=1)) if n > 1 else 0.0
                descriptive[key] = {
                    "n": n, "mean": mean, "sd": sd, "std": sd,
                    "stderr": float(sd / np.sqrt(n)) if n > 0 else None,
                    "median": float(np.median(subset)) if n > 0 else None,
                    "min": float(np.min(subset)) if n > 0 else None,
                    "max": float(np.max(subset)) if n > 0 else None,
                    "ci_lower": None, "ci_upper": None,
                }

        # --- ANOVA table ---
        anova_table = pd.DataFrame([
            {"Source": between_factor,                  "ATS": ATS_A,  "F": ATS_A,  "Wald_Chi2": ATS_A,  "df1": f_A,  "df2": round(df2_between, 2) if df2_between else None, "p-value": p_A,  "p_unc": p_A,  "StatisticType": "ANOVA-Type Statistic (ATS)"},
            {"Source": within_factor,                   "ATS": ATS_T,  "F": ATS_T,  "Wald_Chi2": ATS_T,  "df1": f_T,  "df2": None,                                            "p-value": p_T,  "p_unc": p_T,  "StatisticType": "ANOVA-Type Statistic (ATS)"},
            {"Source": f"{between_factor}:{within_factor}", "ATS": ATS_AT, "F": ATS_AT, "Wald_Chi2": ATS_AT, "df1": f_AT, "df2": None,                                            "p-value": p_AT, "p_unc": p_AT, "StatisticType": "ANOVA-Type Statistic (ATS)"},
        ])

        factors = [
            {"factor": between_factor, "type": "between", "F": ATS_A,  "Wald_Chi2": ATS_A,  "p_value": p_A,  "df1": f_A,  "df2": round(df2_between, 2) if df2_between else None, "effect_size": None, "effect_size_type": None},
            {"factor": within_factor,  "type": "within",  "F": ATS_T,  "Wald_Chi2": ATS_T,  "p_value": p_T,  "df1": f_T,  "df2": None,                                            "effect_size": None, "effect_size_type": None},
        ]
        interactions = [
            {"factors": [between_factor, within_factor], "F": ATS_AT, "Wald_Chi2": ATS_AT, "p_value": p_AT, "df1": f_AT, "df2": None, "effect_size": None, "effect_size_type": None},
        ]

        # --- Primary effect (interaction_first) ---
        interaction_significant = p_AT < alpha
        if interaction_significant:
            primary_source, primary_p, primary_F, primary_df1, primary_kind = (
                f"{between_factor}:{within_factor}", p_AT, ATS_AT, f_AT, "interaction"
            )
            interpretation_order = ["interaction", "main_effects_cautious"]
        else:
            if p_A <= p_T:
                primary_source, primary_p, primary_F, primary_df1 = between_factor, p_A, ATS_A, f_A
            else:
                primary_source, primary_p, primary_F, primary_df1 = within_factor, p_T, ATS_T, f_T
            primary_kind = "main"
            interpretation_order = ["main_effects", "interaction"]

        primary_effect = {
            "source": primary_source,
            "kind": primary_kind,
            "policy": "interaction_first",
            "p_value": primary_p,
            "wald_chi2": primary_F,
        }

        analysis_note = (
            f"Assumptions for parametric Mixed ANOVA were violated. "
            f"A Brunner-Langer ANOVA-Type Statistic (ATS) was computed using global mid-ranks "
            f"(F1-LD-F1 design: {a} groups \u00d7 {t} time points, N = {N} total observations). "
            f"Between-effect df2 ({df2_between:.1f}) uses Satterthwaite marginal-covariance approximation "
            f"(Brunner et al. 2002)."
        )
        # Append RTE table so it appears in the Excel Summary sheet
        rte_lines = ["Relative Treatment Effects (RTE, range 0–1; 0.5 = no effect):"]
        for _, rte_row in RTE_df.iterrows():
            rte_lines.append(
                f"  {between_factor}={rte_row['between_group']}, "
                f"{within_factor}={rte_row['within_level']}: "
                f"RTE={rte_row['RTE']:.4f} (n={int(rte_row['n'])})"
            )
        analysis_note += "\n" + "\n".join(rte_lines)

        # --- Post-hoc: Wilcoxon (within), MWU (between/interaction), Holm-corrected ---
        posthoc_comps = []
        posthoc_name = None
        from itertools import combinations as _comb
        raw = []

        # Between-factor: MWU between groups collapsed over within levels
        if p_A < alpha and a >= 2:
            for b1, b2 in _comb(between_levels, 2):
                comp = _mwu_posthoc_comp(
                    df[df[between_factor] == b1][dv].values,
                    df[df[between_factor] == b2][dv].values,
                    f"{between_factor}={b1}", f"{between_factor}={b2}", alpha
                )
                if comp is not None:
                    raw.append(comp)

        # Within-factor: paired Wilcoxon between time points (all subjects)
        if p_T < alpha and t >= 2:
            wide_all = df.pivot_table(
                index=subject_col, columns=within_factor, values=dv, aggfunc='mean'
            ).reindex(columns=within_levels).dropna()
            for w1, w2 in _comb(within_levels, 2):
                comp = _wilcoxon_posthoc_comp(
                    wide_all[w1].values, wide_all[w2].values,
                    f"{within_factor}={w1}", f"{within_factor}={w2}", alpha
                )
                if comp is not None:
                    raw.append(comp)

        # Interaction: MWU for each between-group pair at each within level
        if p_AT < alpha:
            for w_val in within_levels:
                for b1, b2 in _comb(between_levels, 2):
                    comp = _mwu_posthoc_comp(
                        df[(df[between_factor] == b1) & (df[within_factor] == w_val)][dv].values,
                        df[(df[between_factor] == b2) & (df[within_factor] == w_val)][dv].values,
                        f"{between_factor}={b1}, {within_factor}={w_val}",
                        f"{between_factor}={b2}, {within_factor}={w_val}", alpha
                    )
                    if comp is not None:
                        raw.append(comp)

        if raw:
            posthoc_comps = _apply_holm(raw, alpha)
            posthoc_name = "Pairwise Wilcoxon/MWU (Holm-corrected)"

        return {
            "test": "Brunner-Langer ATS Test",
            "p_value": primary_p,
            "statistic": primary_F,
            "posthoc_test": posthoc_name,
            "pairwise_comparisons": posthoc_comps,
            "descriptive": descriptive,
            "descriptive_transformed": {},
            "alpha": alpha,
            "null_hypothesis": "The relative treatment effects are equal across all groups and time points.",
            "alternative_hypothesis": "At least one relative treatment effect differs.",
            "confidence_interval": None,
            "ci_level": 0.95,
            "power": None,
            "effect_size": None,
            "effect_size_type": None,
            "error": None,
            "df1": primary_df1,
            "df2": round(df2_between, 2) if (primary_kind == "main" and primary_source == between_factor and df2_between) else None,
            "model_type": "BrunnerLangerATS",
            "model_class": "Brunner-Langer ATS",
            "model_family": None,
            "model_link": None,
            "family_diagnostics": {},
            "cov_struct_used": None,
            "covariance_estimator": None,
            "factors": factors,
            "interactions": interactions,
            "anova_table": anova_table,
            "primary_effect": primary_effect,
            "primary_effect_policy": "interaction_first",
            "interaction_significant": interaction_significant,
            "interpretation_order": interpretation_order,
            "recommendation": "non_parametric",
            "fallback_model_used": False,
            "analysis_note": analysis_note,
            "warnings": warnings_list,
            "StatisticType": "ANOVA-Type Statistic (ATS)",
            "RTE": RTE_df,
        }

    except Exception as exc:
        return {
            "test": "Brunner-Langer ATS Test",
            "p_value": None, "statistic": None,
            "posthoc_test": None, "pairwise_comparisons": [],
            "descriptive": {}, "descriptive_transformed": {},
            "alpha": alpha,
            "null_hypothesis": "The relative treatment effects are equal across all groups and time points.",
            "alternative_hypothesis": "At least one relative treatment effect differs.",
            "confidence_interval": None, "ci_level": 0.95,
            "power": None, "effect_size": None, "effect_size_type": None,
            "error": str(exc), "df1": None, "df2": None,
            "model_type": "BrunnerLangerATS",
            "model_class": "Brunner-Langer ATS",
            "model_family": None, "model_link": None,
            "family_diagnostics": {}, "cov_struct_used": None, "covariance_estimator": None,
            "factors": [], "interactions": [], "anova_table": None,
            "primary_effect": None, "primary_effect_policy": "interaction_first",
            "interaction_significant": False, "interpretation_order": ["main_effects"],
            "recommendation": "non_parametric", "fallback_model_used": False,
            "analysis_note": f"Brunner-Langer ATS failed: {exc}",
            "warnings": warnings_list, "StatisticType": "ANOVA-Type Statistic (ATS)",
            "RTE": pd.DataFrame(),
        }

