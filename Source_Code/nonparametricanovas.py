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


def fallback_modern_models(data, dependent_var, formula, design_type, subject_col=None):
    import re
    import warnings
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from pandas.api.types import is_numeric_dtype
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    def _base_result(test_label):
        if design_type in ["rm", "mixed"]:
            analysis_note = (
                "Computed via Generalized Estimating Equations (GEE) due to "
                "violated parametric assumptions."
            )
        else:
            analysis_note = (
                "Computed via Generalized Linear Model (GLM) due to "
                "violated parametric assumptions."
            )
        return {
            "test": test_label,
            "p_value": None,
            "statistic": None,
            "posthoc_test": None,
            "pairwise_comparisons": [],
            "descriptive": {},
            "descriptive_transformed": {},
            "alpha": 0.05,
            "null_hypothesis": "The means/medians of all groups are equal.",
            "alternative_hypothesis": "At least one mean/median differs from the others.",
            "confidence_interval": None,
            "ci_level": 0.95,
            "power": None,
            "effect_size": None,
            "effect_size_type": None,
            "error": None,
            "df1": None,
            "df2": None,
            "factors": [],
            "interactions": [],
            "anova_table": None,
            "analysis_note": analysis_note,
            "recommendation": "modern_model_fallback",
            "fallback_model_used": True,
            "model_class": None,
            "model_family": None,
        }

    def _as_float(value):
        if value is None:
            return None
        if isinstance(value, (float, int, np.floating, np.integer)):
            return float(value)
        if hasattr(value, "item"):
            try:
                return float(value.item())
            except Exception:
                return None
        if isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0:
            try:
                return float(np.asarray(value).ravel()[0])
            except Exception:
                return None
        return None

    def _as_int(value):
        value = _as_float(value)
        if value is None or np.isnan(value):
            return None
        return int(value)

    def _extract_factor_order(formula_text):
        factor_matches = re.findall(r"C\(\s*`?([^`)]+?)`?\s*\)", formula_text)
        cleaned = []
        for factor_name in factor_matches:
            factor_name = factor_name.strip()
            if factor_name not in cleaned:
                cleaned.append(factor_name)
        return cleaned

    def _sanitize_dataframe(df):
        rename_map = {}
        used_names = set()

        for col in df.columns:
            safe_name = re.sub(r"\W+", "_", str(col)).strip("_")
            if not safe_name:
                safe_name = "col"
            if safe_name[0].isdigit():
                safe_name = f"col_{safe_name}"

            candidate = safe_name
            suffix = 1
            while candidate in used_names:
                suffix += 1
                candidate = f"{safe_name}_{suffix}"

            rename_map[col] = candidate
            used_names.add(candidate)

        return df.rename(columns=rename_map).copy(), rename_map

    def _sanitize_formula(formula_text, rename_map):
        lhs, rhs = [part.strip() for part in formula_text.split("~", 1)]
        lhs = lhs.strip("`")
        sanitized_lhs = rename_map.get(lhs, lhs)
        sanitized_rhs = rhs

        for original, safe in sorted(rename_map.items(), key=lambda item: len(str(item[0])), reverse=True):
            original_escaped = re.escape(str(original))
            sanitized_rhs = re.sub(
                rf"C\(\s*`?{original_escaped}`?\s*\)",
                f"C({safe})",
                sanitized_rhs,
            )

        return f"{sanitized_lhs} ~ {sanitized_rhs}"

    def _restore_term(term_name, reverse_map):
        restored = str(term_name)
        for safe, original in sorted(reverse_map.items(), key=lambda item: len(item[0]), reverse=True):
            restored = restored.replace(f"`{safe}`", str(original))
            restored = restored.replace(safe, str(original))
        return restored

    def _term_factors(term_name, reverse_map):
        restored = _restore_term(term_name, reverse_map)
        factors = re.findall(r"C\(([^)]+)\)", restored)
        factors = [factor.replace("`", "").strip() for factor in factors]
        if factors:
            return factors

        if ":" in restored:
            return [part.strip() for part in restored.split(":")]

        restored = restored.replace("`", "").strip()
        if restored and restored.lower() not in ["intercept", "const"]:
            return [restored]
        return []

    def _term_type(factor_name, factor_order):
        if design_type == "two_way":
            return "between"
        if design_type == "rm":
            return "within"
        if design_type == "mixed" and len(factor_order) >= 2:
            between_factor = factor_order[0]
            within_factor = factor_order[1]
            if factor_name == within_factor:
                return "within"
            if factor_name == between_factor:
                return "between"
        return None

    def _build_descriptive(df, factor_order, dv):
        descriptive = {}

        if design_type in ["two_way", "mixed"] and len(factor_order) >= 2:
            factor_a, factor_b = factor_order[0], factor_order[1]
            for a_val in df[factor_a].dropna().unique():
                for b_val in df[factor_b].dropna().unique():
                    subset = df[(df[factor_a] == a_val) & (df[factor_b] == b_val)][dv].dropna()
                    key = f"{factor_a}={a_val}, {factor_b}={b_val}"
                    if len(subset) == 0:
                        descriptive[key] = {
                            "n": 0,
                            "mean": None,
                            "sd": None,
                            "stderr": None,
                            "ci_lower": None,
                            "ci_upper": None,
                            "min": None,
                            "max": None,
                            "median": None
                        }
                        continue

                    n = len(subset)
                    mean = float(np.mean(subset))
                    sd = float(np.std(subset, ddof=1)) if n > 1 else 0.0
                    stderr = float(sd / np.sqrt(n)) if n > 0 else None
                    descriptive[key] = {
                        "n": n,
                        "mean": mean,
                        "sd": sd,
                        "stderr": stderr,
                        "ci_lower": None,
                        "ci_upper": None,
                        "min": float(np.min(subset)),
                        "max": float(np.max(subset)),
                        "median": float(np.median(subset))
                    }

        elif design_type == "rm" and factor_order:
            within_factor = factor_order[0]
            for level in df[within_factor].dropna().unique():
                subset = df[df[within_factor] == level][dv].dropna()
                key = f"{within_factor}={level}"
                if len(subset) == 0:
                    descriptive[key] = {
                        "n": 0,
                        "mean": None,
                        "sd": None,
                        "stderr": None,
                        "ci_lower": None,
                        "ci_upper": None,
                        "min": None,
                        "max": None,
                        "median": None
                    }
                    continue

                n = len(subset)
                mean = float(np.mean(subset))
                sd = float(np.std(subset, ddof=1)) if n > 1 else 0.0
                stderr = float(sd / np.sqrt(n)) if n > 0 else None
                descriptive[key] = {
                    "n": n,
                    "mean": mean,
                    "sd": sd,
                    "stderr": stderr,
                    "ci_lower": None,
                    "ci_upper": None,
                    "min": float(np.min(subset)),
                    "max": float(np.max(subset)),
                    "median": float(np.median(subset))
                }

        return descriptive

    factor_order = _extract_factor_order(formula)
    if design_type == "two_way" and len(factor_order) >= 2:
        test_label = f"Two-Way ANOVA ({factor_order[0]} * {factor_order[1]}) [GLM Fallback]"
    elif design_type == "rm" and factor_order:
        test_label = f"Repeated Measures ANOVA ({factor_order[0]}) [GEE Fallback]"
    elif design_type == "mixed" and len(factor_order) >= 2:
        test_label = f"Mixed ANOVA ({factor_order[1]} * {factor_order[0]}) [GEE Fallback]"
    else:
        test_label = f"{design_type} [Fallback]"

    results = _base_result(test_label)

    try:
        if dependent_var not in data.columns:
            results["error"] = f"Dependent variable '{dependent_var}' not found."
            return results
        if not is_numeric_dtype(data[dependent_var]):
            results["error"] = f"Dependent variable '{dependent_var}' must be numeric."
            return results

        required_columns = [dependent_var] + factor_order
        if subject_col:
            required_columns.append(subject_col)
        missing_columns = [column for column in required_columns if column not in data.columns]
        if missing_columns:
            results["error"] = f"Missing required columns: {', '.join(missing_columns)}"
            return results

        model_data = data[required_columns].copy()
        model_data[dependent_var] = pd.to_numeric(model_data[dependent_var], errors="coerce")

        categorical_columns = list(factor_order)
        if subject_col:
            categorical_columns.append(subject_col)
        for column in categorical_columns:
            if column in model_data.columns:
                model_data[column] = model_data[column].astype("category")

        model_data = model_data.dropna(subset=required_columns)

        if model_data.empty:
            results["error"] = "No valid data available for fallback model."
            return results

        dependent_values = model_data[dependent_var].to_numpy(dtype=float)
        if np.any(~np.isfinite(dependent_values)):
            results["error"] = "Dependent variable contains non-finite values."
            return results

        is_integer_like = np.all(np.isclose(dependent_values, np.round(dependent_values)))
        if is_integer_like:
            if np.any(dependent_values < 0):
                results["error"] = "Negative binomial fallback is not possible: count data must not contain negative values."
                return results
            family = sm.families.NegativeBinomial()
            results["model_family"] = "NegativeBinomial"
        else:
            if np.any(dependent_values <= 0):
                results["error"] = "Gamma fallback is not possible: continuous data must be strictly greater than 0."
                return results
            family = sm.families.Gamma(link=sm.families.links.Log())
            results["model_family"] = "Gamma(log)"

        sanitized_df, rename_map = _sanitize_dataframe(model_data)
        reverse_map = {safe: original for original, safe in rename_map.items()}
        sanitized_formula = _sanitize_formula(formula, rename_map)
        sanitized_subject = rename_map.get(subject_col) if subject_col else None

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            if design_type == "two_way":
                model = smf.glm(formula=sanitized_formula, data=sanitized_df, family=family)
                fitted = model.fit()
                results["model_class"] = "GLM"
            elif design_type in ["rm", "mixed"]:
                if sanitized_subject is None:
                    results["error"] = "subject_col is required for rm/mixed fallback models."
                    return results
                model = smf.gee(
                    formula=sanitized_formula,
                    groups=sanitized_df[sanitized_subject],
                    data=sanitized_df,
                    family=family,
                    cov_struct=sm.cov_struct.Exchangeable()
                )
                fitted = model.fit()
                results["model_class"] = "GEE"
            else:
                results["error"] = f"Unknown design_type '{design_type}'."
                return results

        convergence_messages = []
        for warning in caught_warnings:
            warning_text = str(warning.message).lower()
            if issubclass(warning.category, ConvergenceWarning) or "converg" in warning_text:
                convergence_messages.append(str(warning.message))

        if convergence_messages or getattr(fitted, "converged", True) is False:
            results["error"] = "Model could not converge"
            return results

        try:
            wald = fitted.wald_test_terms()
            wald_table = wald.table.copy()
        except Exception as exc:
            results["error"] = f"Wald test could not be computed: {exc}"
            return results

        statistic_column = "statistic" if "statistic" in wald_table.columns else None
        pvalue_column = "pvalue" if "pvalue" in wald_table.columns else None
        df1_column = "df_constraint" if "df_constraint" in wald_table.columns else None

        if statistic_column is None or pvalue_column is None:
            results["error"] = "Wald test table does not contain the expected columns."
            return results

        factor_entries = {}
        interaction_entries = []
        anova_rows = []

        for term_name, row in wald_table.iterrows():
            restored_term = _restore_term(term_name, reverse_map)
            if restored_term.lower() in ["intercept", "const"]:
                continue

            statistic = _as_float(row.get(statistic_column))
            p_value = _as_float(row.get(pvalue_column))
            df1 = _as_int(row.get(df1_column))
            term_factors = _term_factors(term_name, reverse_map)

            anova_rows.append({
                "Source": restored_term,
                "F": statistic,
                "p-unc": p_value,
                "DF1": df1,
                "DF2": None,
                "StatisticType": "Wald Chi-square"
            })

            if len(term_factors) == 1:
                factor_name = term_factors[0]
                factor_entries[factor_name] = {
                    "factor": factor_name,
                    "type": _term_type(factor_name, factor_order),
                    "F": statistic,
                    "p_value": p_value,
                    "df1": df1,
                    "df2": None,
                    "effect_size": None,
                    "effect_size_type": None
                }
            elif len(term_factors) >= 2:
                interaction_entries.append({
                    "factors": term_factors,
                    "F": statistic,
                    "p_value": p_value,
                    "df1": df1,
                    "df2": None,
                    "effect_size": None,
                    "effect_size_type": None
                })

        if design_type == "mixed" and len(factor_order) >= 2:
            ordered_factor_names = [factor_order[1], factor_order[0]]
        else:
            ordered_factor_names = factor_order

        results["factors"] = [
            factor_entries[name] for name in ordered_factor_names if name in factor_entries
        ]
        results["interactions"] = interaction_entries
        results["anova_table"] = pd.DataFrame(anova_rows)
        results["descriptive"] = _build_descriptive(model_data, factor_order, dependent_var)
        results["fitted_model"] = fitted

        if design_type in ["two_way", "mixed"] and results["interactions"]:
            primary_effect = results["interactions"][0]
        elif results["factors"]:
            primary_effect = results["factors"][0]
        else:
            primary_effect = None

        if primary_effect is not None:
            results["p_value"] = primary_effect.get("p_value")
            results["statistic"] = primary_effect.get("F")
            results["df1"] = primary_effect.get("df1")
            results["df2"] = primary_effect.get("df2")

        return results

    except ConvergenceWarning:
        results["error"] = "Model could not converge"
        return results
    except Exception as exc:
        results["error"] = f"Error in modern-model fallback: {exc}"
        return results

def check_normality(residuals):
    """
    Perform Shapiro–Wilk test for normality on residuals.
    Returns (statistic, p-value).
    """
    if len(residuals) < 3:
        return (np.nan, 1.0)  # Not enough data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p = shapiro(residuals)
    return stat, p

def check_levene(*groups):
    """
    Perform Levene's test for homogeneity of variances (median-centered).
    Returns (statistic, p-value).
    """
    if any(len(g) < 2 for g in groups):
        return (np.nan, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p = levene(*groups, center='median')
    return stat, p

def auto_anova_decision(df, dv, factors, subject=None, design='two_way', alpha=0.05, family=None, verbose=True, **kwargs):
    """
    Automatically choose between parametric ANOVA and robust GLMM/GEE based on normality and variance tests.
    
    Parameters:
        df: DataFrame
        dv: dependent variable (str)
        factors: list of factor names (str)
        subject: subject column (for RM or mixed)
        design: 'two_way', 'rm', or 'mixed'
        alpha: significance threshold for assumption tests
        family: statsmodels family (for GLMM/GEE)
        verbose: print decision process
        kwargs: passed to model fit
    Returns:
        result: fitted model result (parametric or GLMM/GEE)
        info: dict with assumption test results and model type
    """
    # 1. Fit OLS and get residuals for normality
    import statsmodels.formula.api as smf
    if design == 'two_way':
        formula = f"{dv} ~ {' * '.join(factors)}"
        ols_model = smf.ols(formula, data=df).fit()
        residuals = ols_model.resid
        # Group data for Levene
        groups = [df[df[factors[0]] == lvl][dv].values for lvl in df[factors[0]].unique()]
        if len(factors) > 1:
            groups += [df[df[factors[1]] == lvl][dv].values for lvl in df[factors[1]].unique()]
    elif design == 'rm':
        # For RM, use within-subject residuals
        formula = f"{dv} ~ {' + '.join(factors)}"
        ols_model = smf.ols(formula, data=df).fit()
        residuals = ols_model.resid
        groups = [df[df[factors[0]] == lvl][dv].values for lvl in df[factors[0]].unique()]
    elif design == 'mixed':
        formula = f"{dv} ~ {' * '.join(factors)}"
        ols_model = smf.ols(formula, data=df).fit()
        residuals = ols_model.resid
        groups = [df[df[factors[0]] == lvl][dv].values for lvl in df[factors[0]].unique()]
        if len(factors) > 1:
            groups += [df[df[factors[1]] == lvl][dv].values for lvl in df[factors[1]].unique()]
    else:
        raise ValueError(f"Unknown design: {design}")

    stat_norm, p_norm = check_normality(residuals)
    stat_lev, p_lev = check_levene(*groups)

    if verbose:
        print(f"Normality p={p_norm:.3g}, Levene p={p_lev:.3g}")

    # 2. Decision logic
    if p_norm > alpha and p_lev > alpha:
        if verbose:
            print("Assumptions met: using parametric ANOVA.")
        if design == 'two_way':
            import statsmodels.api as sm
            aov_table = sm.stats.anova_lm(ols_model, typ=2)
            return aov_table, {'parametric': True, 'p_norm': p_norm, 'p_levene': p_lev, 'model': 'ANOVA'}
        elif design == 'rm':
            from statsmodels.stats.anova import AnovaRM
            result = AnovaRM(df, depvar=dv, subject=subject, within=factors).fit()
            return result, {'parametric': True, 'p_norm': p_norm, 'p_levene': p_lev, 'model': 'AnovaRM'}
        elif design == 'mixed':
            import statsmodels.formula.api as smf
            # Use MixedLM for parametric
            model = smf.mixedlm(formula, data=df, groups=df[subject])
            result = model.fit()
            return result, {'parametric': True, 'p_norm': p_norm, 'p_levene': p_lev, 'model': 'MixedLM'}
    else:
        if verbose:
            print("Assumptions violated: using robust GLMM/GEE.")
        if design == 'two_way':
            model = GLMMTwoWayANOVA(family=family or sm.families.Gaussian())
            result = model.fit(df, dv, factors[0], factors[1], random_group=subject, **kwargs)
            return result, {'parametric': False, 'p_norm': p_norm, 'p_levene': p_lev, 'model': 'GLMMTwoWayANOVA'}
        elif design == 'rm':
            model = GEERMANOVA(family=family or sm.families.Gaussian())
            result = model.fit(df, dv, time=factors[0], subject=subject, **kwargs)
            return result, {'parametric': False, 'p_norm': p_norm, 'p_levene': p_lev, 'model': 'GEERMANOVA'}
        elif design == 'mixed':
            model = GLMMMixedANOVA(family=family or sm.families.Gaussian())
            result = model.fit(df, dv, between=[factors[0]], within=[factors[1]], subject=subject, **kwargs)
            return result, {'parametric': False, 'p_norm': p_norm, 'p_levene': p_lev, 'model': 'GLMMMixedANOVA'}
    raise RuntimeError("Could not select or fit a model.")
"""
GLMM/GEE-based robust alternatives for Two-Way, RM, and Mixed ANOVA
using statsmodels (GLM, MixedLM, GEE), with support for non-Gaussian outcomes,
bootstrapped p-values, diagnostics, and Bayesian GLMMs.

Example usage:
    # Two-Way ANOVA (Gaussian)
    model = GLMMTwoWayANOVA(family=sm.families.Gaussian())
    result = model.fit(df, dv='y', factor_a='A', factor_b='B', random_group='subject')
    print(result.summary())

    # Two-Way ANOVA (Binomial)
    model = GLMMTwoWayANOVA(family=sm.families.Binomial())
    result = model.fit(df, dv='y_bin', factor_a='A', factor_b='B', random_group='subject')
    print(result.summary())

    # Repeated Measures ANOVA (GEE, Poisson)
    model = GEERMANOVA(family=sm.families.Poisson())
    result = model.fit(df, dv='y_count', time='time', subject='subject')
    print(result.summary())

    # Mixed ANOVA (random slopes, Gaussian)
    model = GLMMMixedANOVA()
    result = model.fit(df, dv='y', between=['A'], within=['time'], subject='subject')
    print(result.summary())

    # Bayesian GLMM (Binomial)
    bayes_result = fit_bayesian_glmm_binomial(df, formula='y_bin ~ A * B', group='subject')
    print(bayes_result.summary())
"""


import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from scipy.stats import norm

# Utility function for standardized result output
class GLMMResult:
    """
    Standardized result wrapper for GLMM/GEE models.
    Provides summary, dictionary output, and diagnostics.
    """
    def __init__(self, result, model_type):
        self.result = result
        self.model_type = model_type
        self.diagnostics = {}

    def check_convergence(self):
        """Return convergence status and message if available."""
        if hasattr(self.result, 'converged'):
            return self.result.converged
        if hasattr(self.result, 'mle_retvals'):
            return self.result.mle_retvals.get('converged', None)
        return None

    def plot_residuals(self):
        """Plot residuals vs. fitted values if available."""
        if hasattr(self.result, 'resid') and hasattr(self.result, 'fittedvalues'):
            plt.scatter(self.result.fittedvalues, self.result.resid)
            plt.xlabel('Fitted values')
            plt.ylabel('Residuals')
            plt.title('Residuals vs Fitted')
            plt.show()
        else:
            print("Residuals or fitted values not available for this model.")

    def summary(self):
        return self.result.summary() if self.result is not None else 'Model not fit yet.'
    def as_dict(self):
        if self.result is None:
            return {'error': 'Model not fit yet.'}
        params = self.result.params.to_dict() if hasattr(self.result, 'params') else {}
        pvalues = self.result.pvalues.to_dict() if hasattr(self.result, 'pvalues') else {}
        return {
            'model_type': self.model_type,
            'params': params,
            'pvalues': pvalues,
            'aic': getattr(self.result, 'aic', None),
            'bic': getattr(self.result, 'bic', None),
            'converged': getattr(self.result, 'converged', None),
            'diagnostics': self.diagnostics,
        }


class GLMMTwoWayANOVA:
    def run(self, *args, **kwargs):
        self.fit(*args, **kwargs)
        # Return a dict matching the expected output structure
        if self._glmm_result is not None:
            return {
                'recommendation': 'robust_fallback',
                'result': self._glmm_result.as_dict(),
                'model_type': self.model_type or 'GLMMTwoWayANOVA',
            }
        elif hasattr(self, 'error'):
            return {'recommendation': 'robust_fallback', 'error': self.error}
        else:
            return {'recommendation': 'robust_fallback', 'error': 'Unknown error'}
    """
    Two-Way ANOVA using GLM (fixed effects), MixedLM (random effects), or GLM for non-Gaussian outcomes.
    Supports Gaussian, Binomial, Poisson, etc. via the `family` argument.
    
    Example:
        model = GLMMTwoWayANOVA(family=sm.families.Binomial())
        result = model.fit(df, dv='y_bin', factor_a='A', factor_b='B', random_group='subject')
        print(result.summary())
    """
    def __init__(self, family=None, **kwargs):
        self.family = family or sm.families.Gaussian()
        self.result = None
        self.model_type = None
        self._glmm_result = None

    def fit(self, df=None, dv=None, factor_a=None, factor_b=None, random_group=None, bootstrap_p=False, n_boot=1000, seed=None, **kwargs):
        # Accept both positional and keyword arguments for backward compatibility
        # If called with positional args, map them to keywords
        if df is None and len(kwargs) > 0:
            df = kwargs.get('df')
        if dv is None and 'dv' in kwargs:
            dv = kwargs['dv']
        if factor_a is None and 'factor_a' in kwargs:
            factor_a = kwargs['factor_a']
        if factor_b is None and 'factor_b' in kwargs:
            factor_b = kwargs['factor_b']
        if random_group is None and 'random_group' in kwargs:
            random_group = kwargs['random_group']
        # Allow for alternative argument names (between, within, subject)
        if factor_a is None and 'between' in kwargs:
            ba = kwargs['between']
            if isinstance(ba, (list, tuple)) and len(ba) > 0:
                factor_a = ba[0]
        if factor_b is None and 'within' in kwargs:
            wb = kwargs['within']
            if isinstance(wb, (list, tuple)) and len(wb) > 0:
                factor_b = wb[0]
        if random_group is None and 'subject' in kwargs:
            random_group = kwargs['subject']
        # Defensive: ensure all required args are present
        if df is None or dv is None or factor_a is None or factor_b is None:
            self.result = None
            self._glmm_result = None
            self.error = 'Missing required arguments for GLMMTwoWayANOVA.fit()'
            return self
        formula = f"{dv} ~ {factor_a} * {factor_b}"
        try:
            if random_group is None:
                model = smf.glm(formula, data=df, family=self.family)
                self.result = model.fit()
                self.model_type = 'GLM'
            else:
                if isinstance(self.family, sm.families.Gaussian):
                    model = smf.mixedlm(formula, data=df, groups=df[random_group])
                    self.result = model.fit()
                    self.model_type = 'MixedLM'
                else:
                    # For non-Gaussian, fallback to GEE for random effects
                    model = smf.gee(formula, random_group, data=df, family=self.family)
                    self.result = model.fit()
                    self.model_type = 'GEE'
            self._glmm_result = GLMMResult(self.result, self.model_type)
            # Bootstrapped p-values if requested
            if bootstrap_p:
                self._glmm_result.diagnostics['bootstrap_p'] = self._bootstrap_p(df, formula, random_group, n_boot, seed)
        except Exception as e:
            self.result = None
            self._glmm_result = None
            self.error = str(e)
        return self

    def _bootstrap_p(self, df, formula, random_group, n_boot, seed):
        np.random.seed(seed)
        coefs = []
        for _ in range(n_boot):
            sample = df.sample(frac=1, replace=True)
            try:
                if random_group is None:
                    model = smf.glm(formula, data=sample, family=self.family)
                    res = model.fit()
                else:
                    if isinstance(self.family, sm.families.Gaussian):
                        model = smf.mixedlm(formula, data=sample, groups=sample[random_group])
                        res = model.fit()
                    else:
                        model = smf.gee(formula, random_group, data=sample, family=self.family)
                        res = model.fit()
                coefs.append(res.params.values)
            except Exception:
                continue
        coefs = np.array(coefs)
        # Two-sided p-value for each coefficient
        orig_params = self.result.params.values
        pvals = [2 * min((coef > orig).mean(), (coef < orig).mean()) for coef, orig in zip(coefs.T, orig_params)]
        return dict(zip(self.result.params.index, pvals))

    def summary(self):
        if self._glmm_result:
            return self._glmm_result.summary()
        elif hasattr(self, 'error'):
            return f'Error: {self.error}'
        else:
            return 'Model not fit yet.'
    def as_dict(self):
        if self._glmm_result:
            return self._glmm_result.as_dict()
        elif hasattr(self, 'error'):
            return {'error': self.error}
        else:
            return {'error': 'Model not fit yet.'}


class GEERMANOVA:
    def run(self, *args, **kwargs):
        self.fit(*args, **kwargs)
        if self._glmm_result is not None:
            return {
                'recommendation': 'robust_fallback',
                'result': self._glmm_result.as_dict(),
                'model_type': 'GEERMANOVA',
            }
        elif hasattr(self, 'error'):
            return {'recommendation': 'robust_fallback', 'error': self.error}
        else:
            return {'recommendation': 'robust_fallback', 'error': 'Unknown error'}
    """
    Repeated Measures ANOVA using GEE.
    Supports Gaussian, Binomial, Poisson, etc. via the `family` argument.
    Example:
        model = GEERMANOVA(family=sm.families.Poisson())
        result = model.fit(df, dv='y_count', time='time', subject='subject')
        print(result.summary())
    """
    def __init__(self, family=None, cov_struct=None, **kwargs):
        self.family = family or sm.families.Gaussian()
        self.cov_struct = cov_struct or sm.cov_struct.Exchangeable()
        self.result = None
        self._glmm_result = None

    def fit(self, df=None, dv=None, time=None, subject=None, other_factors=None, bootstrap_p=False, n_boot=1000, seed=None, **kwargs):
        # Accept both positional and keyword arguments for backward compatibility
        if df is None and len(kwargs) > 0:
            df = kwargs.get('df')
        if dv is None and 'dv' in kwargs:
            dv = kwargs['dv']
        if time is None and 'time' in kwargs:
            time = kwargs['time']
        if subject is None and 'subject' in kwargs:
            subject = kwargs['subject']
        if other_factors is None and 'other_factors' in kwargs:
            other_factors = kwargs['other_factors']
        # Defensive: ensure all required args are present
        if df is None or dv is None or time is None or subject is None:
            self.result = None
            self._glmm_result = None
            self.error = 'Missing required arguments for GEERMANOVA.fit()'
            return self
        formula = f"{dv} ~ {time}"
        if other_factors:
            formula += ' + ' + ' + '.join(other_factors)
        try:
            model = smf.gee(formula, subject, data=df, cov_struct=self.cov_struct, family=self.family)
            self.result = model.fit()
            self._glmm_result = GLMMResult(self.result, 'GEE')
            if bootstrap_p:
                self._glmm_result.diagnostics['bootstrap_p'] = self._bootstrap_p(df, formula, subject, n_boot, seed)
        except Exception as e:
            self.result = None
            self._glmm_result = None
            self.error = str(e)
        return self

    def _bootstrap_p(self, df, formula, subject, n_boot, seed):
        np.random.seed(seed)
        coefs = []
        for _ in range(n_boot):
            sample = df.groupby(subject, group_keys=False).apply(lambda x: x.sample(frac=1, replace=True))
            try:
                model = smf.gee(formula, subject, data=sample, cov_struct=self.cov_struct, family=self.family)
                res = model.fit()
                coefs.append(res.params.values)
            except Exception:
                continue
        coefs = np.array(coefs)
        orig_params = self.result.params.values
        pvals = [2 * min((coef > orig).mean(), (coef < orig).mean()) for coef, orig in zip(coefs.T, orig_params)]
        return dict(zip(self.result.params.index, pvals))

    def summary(self):
        if self._glmm_result:
            return self._glmm_result.summary()
        elif hasattr(self, 'error'):
            return f'Error: {self.error}'
        else:
            return 'Model not fit yet.'
    def as_dict(self):
        if self._glmm_result:
            return self._glmm_result.as_dict()
        elif hasattr(self, 'error'):
            return {'error': self.error}
        else:
            return {'error': 'Model not fit yet.'}


class GLMMMixedANOVA:
    def run(self, *args, **kwargs):
        self.fit(*args, **kwargs)
        if self._glmm_result is not None:
            return {
                'recommendation': 'robust_fallback',
                'result': self._glmm_result.as_dict(),
                'model_type': 'GLMMMixedANOVA',
            }
        elif hasattr(self, 'error'):
            return {'recommendation': 'robust_fallback', 'error': self.error}
        else:
            return {'recommendation': 'robust_fallback', 'error': 'Unknown error'}
    """
    Mixed ANOVA using MixedLM with random slopes/intercepts.
    Example:
        model = GLMMMixedANOVA()
        result = model.fit(df, dv='y', between=['A'], within=['time'], subject='subject')
        print(result.summary())
    """
    def __init__(self, family=None, **kwargs):
        self.family = family or sm.families.Gaussian()
        self.result = None
        self._glmm_result = None

    def fit(self, df=None, dv=None, between=None, within=None, subject=None, bootstrap_p=False, n_boot=1000, seed=None, **kwargs):
        # Accept both positional and keyword arguments for backward compatibility
        if df is None and len(kwargs) > 0:
            df = kwargs.get('df')
        if dv is None and 'dv' in kwargs:
            dv = kwargs['dv']
        if between is None and 'between' in kwargs:
            between = kwargs['between']
        if within is None and 'within' in kwargs:
            within = kwargs['within']
        if subject is None and 'subject' in kwargs:
            subject = kwargs['subject']
        # Defensive: ensure all required args are present
        if df is None or dv is None or between is None or within is None or subject is None:
            self.result = None
            self._glmm_result = None
            self.error = 'Missing required arguments for GLMMMixedANOVA.fit()'
            return self
        between_str = ' * '.join(between) if isinstance(between, (list, tuple)) else str(between)
        within_str = ' * '.join(within) if isinstance(within, (list, tuple)) else str(within)
        formula = f"{dv} ~ {between_str} * {within_str}"
        re_formula = f"~{within_str}"
        try:
            if isinstance(self.family, sm.families.Gaussian):
                model = smf.mixedlm(formula, data=df, groups=df[subject], re_formula=re_formula)
                self.result = model.fit()
                self._glmm_result = GLMMResult(self.result, 'MixedLM')
            else:
                # For non-Gaussian, fallback to GEE
                model = smf.gee(formula, subject, data=df, family=self.family)
                self.result = model.fit()
                self._glmm_result = GLMMResult(self.result, 'GEE')
            if bootstrap_p:
                self._glmm_result.diagnostics['bootstrap_p'] = self._bootstrap_p(df, formula, subject, n_boot, seed)
        except Exception as e:
            self.result = None
            self._glmm_result = None
            self.error = str(e)
        return self

    def _bootstrap_p(self, df, formula, subject, n_boot, seed):
        np.random.seed(seed)
        coefs = []
        # Extract within_str from formula if possible
        import re
        m = re.search(r'~.*\*(.*)', formula)
        within_str = None
        if m:
            within_str = m.group(1).strip()
        for _ in range(n_boot):
            sample = df.groupby(subject, group_keys=False).apply(lambda x: x.sample(frac=1, replace=True))
            try:
                if isinstance(self.family, sm.families.Gaussian):
                    # Use within_str if available, else default to None
                    re_formula = f"~{within_str}" if within_str else None
                    model = smf.mixedlm(formula, data=sample, groups=sample[subject], re_formula=re_formula)
                    res = model.fit()
                else:
                    model = smf.gee(formula, subject, data=sample, family=self.family)
                    res = model.fit()
                coefs.append(res.params.values)
            except Exception:
                continue
        coefs = np.array(coefs)
        orig_params = self.result.params.values
        pvals = [2 * min((coef > orig).mean(), (coef < orig).mean()) for coef, orig in zip(coefs.T, orig_params)]
        return dict(zip(self.result.params.index, pvals))

# Bayesian GLMM for Binomial outcomes
def fit_bayesian_glmm_binomial(df, formula, group):
    """
    Fit a Bayesian GLMM for binomial outcomes using BinomialBayesMixedGLM.
    Example:
        result = fit_bayesian_glmm_binomial(df, 'y_bin ~ A * B', group='subject')
        print(result.summary())
    """
    # Random intercept for group
    exog_vc = {f'0|{group}': f'0 + C({group})'}
    md = BinomialBayesMixedGLM.from_formula(formula, exog_vc, df)
    fit = md.fit_vb()
    return fit

    def summary(self):
        if self._glmm_result:
            return self._glmm_result.summary()
        elif hasattr(self, 'error'):
            return f'Error: {self.error}'
        else:
            return 'Model not fit yet.'
    def as_dict(self):
        if self._glmm_result:
            return self._glmm_result.as_dict()
        elif hasattr(self, 'error'):
            return {'error': self.error}
        else:
            return {'error': 'Model not fit yet.'}
