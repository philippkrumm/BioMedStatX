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


def fallback_modern_models(data, dependent_var, formula, design_type, subject_col=None, cov_struct_option=None, time_col=None):
    import re
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
            "model_link": None,
            "family_diagnostics": {},
            "cov_struct_used": None,
            "covariance_estimator": None,
            "primary_effect_policy": "minimum_p_across_omnibus_effects",
            "primary_effect": None,
            "interaction_significant": False,
            "interpretation_order": ["main_effects", "interaction"],
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

    def _is_time_like_levels(series):
        values = pd.Series(series).dropna().unique().tolist()
        if len(values) < 3:
            return False
        try:
            numeric_values = pd.to_numeric(pd.Series(values), errors="coerce")
            if numeric_values.notna().all():
                return True
        except Exception:
            pass

        values_as_text = [str(value).strip() for value in values]
        has_digit = [bool(re.search(r"\d", text)) for text in values_as_text]
        return all(has_digit)

    def _select_family(values):
        diagnostics = {
            "n": int(len(values)),
            "zero_fraction": float(np.mean(values == 0)) if len(values) > 0 else None,
            "negative_fraction": float(np.mean(values < 0)) if len(values) > 0 else None,
            "integer_like": bool(np.all(np.isclose(values, np.round(values)))) if len(values) > 0 else False,
            "mean": float(np.mean(values)) if len(values) > 0 else None,
            "variance": float(np.var(values, ddof=1)) if len(values) > 1 else 0.0,
            "overdispersion_ratio": None,
            "selection_reason": None,
        }

        if diagnostics["integer_like"]:
            if np.any(values < 0):
                return None, None, diagnostics, "Count-data fallback is not possible because negative values were detected."

            mean_value = diagnostics["mean"] if diagnostics["mean"] is not None else 0.0
            variance_value = diagnostics["variance"] if diagnostics["variance"] is not None else 0.0
            if mean_value > 0:
                diagnostics["overdispersion_ratio"] = float(variance_value / mean_value)

            diagnostics["selection_reason"] = (
                "Count outcome detected. Initial fit uses Poisson(log), then switches to NegativeBinomial(log) "
                "if Pearson phi exceeds 1.2."
            )
            return sm.families.Poisson(), "Poisson", "log", diagnostics, None

        if np.any(values <= 0):
            diagnostics["selection_reason"] = (
                "Continuous outcome contains non-positive values; Gamma(log) assumptions violated. "
                "Using Gaussian(identity) fallback."
            )
            return sm.families.Gaussian(), "Gaussian", "identity", diagnostics, None

        diagnostics["selection_reason"] = "Strictly positive continuous outcome; selected Gamma(log)."
        return sm.families.Gamma(link=sm.families.links.Log()), "Gamma", "log", diagnostics, None

    def _resolve_cov_struct(sanitized_df, sanitized_time, requested_cov):
        if design_type not in ["rm", "mixed"]:
            return None, None, "not_applicable"

        option = (requested_cov or "").strip().lower()
        if option in ["", "auto"]:
            if sanitized_time and sanitized_time in sanitized_df.columns and _is_time_like_levels(sanitized_df[sanitized_time]):
                return sm.cov_struct.Autoregressive(), sanitized_df[sanitized_time].cat.codes.to_numpy(), "Autoregressive"
            return sm.cov_struct.Exchangeable(), None, "Exchangeable"

        if option in ["ar1", "autoregressive"]:
            if not sanitized_time or sanitized_time not in sanitized_df.columns:
                return sm.cov_struct.Exchangeable(), None, "Exchangeable"
            return sm.cov_struct.Autoregressive(), sanitized_df[sanitized_time].cat.codes.to_numpy(), "Autoregressive"

        if option in ["independence", "independent"]:
            return sm.cov_struct.Independence(), None, "Independence"

        if option in ["exchangeable", "exchange"]:
            return sm.cov_struct.Exchangeable(), None, "Exchangeable"

        return sm.cov_struct.Exchangeable(), None, "Exchangeable"

    def _compute_pearson_phi(y_true, mu, variance_fn, n_params):
        try:
            y_array = np.asarray(y_true, dtype=float)
            mu_array = np.asarray(mu, dtype=float)
            var_mu = np.asarray(variance_fn(mu_array), dtype=float)
            var_mu = np.where(var_mu <= 1e-12, 1e-12, var_mu)

            valid_mask = np.isfinite(y_array) & np.isfinite(mu_array) & np.isfinite(var_mu)
            if not np.any(valid_mask):
                return None

            y_array = y_array[valid_mask]
            mu_array = mu_array[valid_mask]
            var_mu = var_mu[valid_mask]

            numerator = np.sum(((y_array - mu_array) ** 2) / var_mu)
            denominator = len(y_array) - max(1, int(n_params))
            if denominator <= 0:
                return None
            return float(numerator / denominator)
        except Exception:
            return None

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

        family, family_name, link_name, family_diagnostics, selection_error = _select_family(dependent_values)
        if selection_error:
            results["error"] = selection_error
            return results

        results["model_family"] = family_name
        results["model_link"] = link_name
        results["family_diagnostics"] = family_diagnostics

        if family_diagnostics.get("zero_fraction") is not None and family_diagnostics["zero_fraction"] > 0.30:
            warnings_list = results.setdefault("warnings", [])
            warnings_list.append(
                "High zero fraction detected (>30%). Consider zero-inflated models for confirmatory analyses."
            )

        sanitized_df, rename_map = _sanitize_dataframe(model_data)
        reverse_map = {safe: original for original, safe in rename_map.items()}
        sanitized_formula = _sanitize_formula(formula, rename_map)
        sanitized_subject = rename_map.get(subject_col) if subject_col else None
        sanitized_time = rename_map.get(time_col) if time_col else None
        sanitized_dependent = rename_map.get(dependent_var, dependent_var)

        cov_struct = None
        gee_time = None
        cov_struct_name = "not_applicable"
        if design_type in ["rm", "mixed"]:
            cov_struct, gee_time, cov_struct_name = _resolve_cov_struct(
                sanitized_df,
                sanitized_time,
                cov_struct_option,
            )
        results["cov_struct_used"] = cov_struct_name

        if design_type in ["rm", "mixed"] and sanitized_subject in sanitized_df.columns:
            n_clusters = int(sanitized_df[sanitized_subject].nunique())
            if n_clusters <= 30:
                warnings_list = results.setdefault("warnings", [])
                warnings_list.append(
                    f"Small number of clusters detected (n={n_clusters}). Working-correlation choice can materially affect precision."
                )

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            if design_type == "two_way":
                model = smf.glm(formula=sanitized_formula, data=sanitized_df, family=family)
                fit_kwargs = {}
                if family_name == "Gaussian":
                    fit_kwargs["cov_type"] = "HC3"
                    results["covariance_estimator"] = "sandwich_hc3"
                else:
                    results["covariance_estimator"] = "model_based"
                fitted = model.fit(**fit_kwargs)
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
                    cov_struct=cov_struct,
                    time=gee_time
                )
                fitted = model.fit()
                results["model_class"] = "GEE"
                results["covariance_estimator"] = "robust_gee_sandwich"
            else:
                results["error"] = f"Unknown design_type '{design_type}'."
                return results

            # For count outcomes, use Pearson phi from the initial Poisson fit to decide NB refit.
            if (
                family_name == "Poisson"
                and family_diagnostics.get("integer_like")
                and sanitized_dependent in sanitized_df.columns
            ):
                pearson_phi = _compute_pearson_phi(
                    y_true=sanitized_df[sanitized_dependent],
                    mu=fitted.fittedvalues,
                    variance_fn=family.variance,
                    n_params=len(getattr(fitted, "params", [])),
                )
                family_diagnostics["pearson_phi"] = pearson_phi

                if pearson_phi is not None and pearson_phi > 1.2:
                    family = sm.families.NegativeBinomial()
                    family_name = "NegativeBinomial"
                    link_name = "log"
                    family_diagnostics["selection_reason"] = (
                        f"Count outcome with Pearson phi={pearson_phi:.3f} > 1.2; switched to NegativeBinomial(log)."
                    )

                    if design_type == "two_way":
                        model = smf.glm(formula=sanitized_formula, data=sanitized_df, family=family)
                        fitted = model.fit()
                        results["model_class"] = "GLM"
                        results["covariance_estimator"] = "model_based"
                    else:
                        model = smf.gee(
                            formula=sanitized_formula,
                            groups=sanitized_df[sanitized_subject],
                            data=sanitized_df,
                            family=family,
                            cov_struct=cov_struct,
                            time=gee_time
                        )
                        fitted = model.fit()
                        results["model_class"] = "GEE"
                        results["covariance_estimator"] = "robust_gee_sandwich"

                    results["model_family"] = family_name
                    results["model_link"] = link_name

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
                "Wald_Chi2": statistic,
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
                    "Wald_Chi2": statistic,
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
                    "Wald_Chi2": statistic,
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

        interaction_significant = any(
            interaction.get("p_value") is not None and interaction.get("p_value") < results.get("alpha", 0.05)
            for interaction in results["interactions"]
        )
        results["interaction_significant"] = interaction_significant
        if interaction_significant:
            results["interpretation_order"] = ["interaction", "main_effects_cautious"]
            analysis_note = str(results.get("analysis_note") or "")
            interaction_note = (
                " Significant interaction detected: interpret interaction effects first; "
                "main effects represent averaged effects across the other factor."
            )
            if interaction_note.strip() not in analysis_note:
                results["analysis_note"] = f"{analysis_note}{interaction_note}".strip()

        all_effects = []
        for factor_effect in results["factors"]:
            all_effects.append({"source": factor_effect.get("factor"), "kind": "main", "effect": factor_effect})
        for interaction_effect in results["interactions"]:
            interaction_name = " * ".join(interaction_effect.get("factors", []))
            all_effects.append({"source": interaction_name, "kind": "interaction", "effect": interaction_effect})

        primary_effect = None
        candidate_effects = [entry for entry in all_effects if entry["effect"].get("p_value") is not None]
        if candidate_effects:
            primary_effect = min(candidate_effects, key=lambda entry: entry["effect"].get("p_value"))

        if primary_effect is not None:
            effect_payload = primary_effect.get("effect", {})
            results["p_value"] = effect_payload.get("p_value")
            results["statistic"] = effect_payload.get("Wald_Chi2", effect_payload.get("F"))
            results["df1"] = effect_payload.get("df1")
            results["df2"] = effect_payload.get("df2")
            results["primary_effect"] = {
                "source": primary_effect.get("source"),
                "kind": primary_effect.get("kind"),
                "policy": results.get("primary_effect_policy"),
                "p_value": results["p_value"],
                "wald_chi2": results["statistic"],
            }

        return results

    except ConvergenceWarning:
        results["error"] = "Model could not converge"
        return results
    except Exception as exc:
        results["error"] = f"Error in modern-model fallback: {exc}"
        return results

# ---------------------------------------------------------------------------
# Post-hoc helpers shared by all three nonparametric ANOVA functions
# ---------------------------------------------------------------------------

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
            "p-unc": p_value,
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
            "effect_size": None,
            "effect_size_type": None,
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
            "test": f"Repeated Measures ANOVA [Friedman Fallback]",
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
            "effect_size": None,
            "effect_size_type": None,
            "error": None,
            "df1": df1,
            "df2": None,
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
            "test": "Repeated Measures ANOVA [Friedman Fallback]",
            "p_value": None, "statistic": None,
            "posthoc_test": None, "pairwise_comparisons": [],
            "descriptive": {}, "descriptive_transformed": {},
            "alpha": alpha,
            "null_hypothesis": "The distribution is the same across all repeated measurements.",
            "alternative_hypothesis": "At least one repeated measurement differs from the others.",
            "confidence_interval": None, "ci_level": 0.95,
            "power": None, "effect_size": None, "effect_size_type": None,
            "error": str(exc), "df1": None, "df2": None,
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
        # Also expose p-perm as p-unc for exporter compatibility
        anova_table["p-unc"] = anova_table["p-perm"]

        factors = [
            {"factor": factor_a, "type": "between", "F": F_A,  "Wald_Chi2": F_A,  "p_value": p_perm_A,  "df1": df_A,  "df2": df2, "effect_size": None, "effect_size_type": None},
            {"factor": factor_b, "type": "between", "F": F_B,  "Wald_Chi2": F_B,  "p_value": p_perm_B,  "df1": df_B,  "df2": df2, "effect_size": None, "effect_size_type": None},
        ]
        interactions = [
            {"factors": [factor_a, factor_b], "F": F_AB, "Wald_Chi2": F_AB, "p_value": p_perm_AB, "df1": df_AB, "df2": df2, "effect_size": None, "effect_size_type": None},
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
            "test": "Two-Way ANOVA [Freedman-Lane Permutation Fallback]",
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
            "effect_size": None,
            "effect_size_type": None,
            "error": None,
            "df1": df_AB if interaction_significant else (df_A if p_perm_A <= p_perm_B else df_B),
            "df2": df2,
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
            "test": "Two-Way ANOVA [Freedman-Lane Permutation Fallback]",
            "p_value": None, "statistic": None,
            "posthoc_test": None, "pairwise_comparisons": [],
            "descriptive": {}, "descriptive_transformed": {},
            "alpha": alpha,
            "null_hypothesis": "The means of all groups are equal.",
            "alternative_hypothesis": "At least one group mean differs.",
            "confidence_interval": None, "ci_level": 0.95,
            "power": None, "effect_size": None, "effect_size_type": None,
            "error": str(exc), "df1": None, "df2": None,
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
            {"Source": between_factor,                  "ATS": ATS_A,  "F": ATS_A,  "Wald_Chi2": ATS_A,  "df1": f_A,  "df2": round(df2_between, 2) if df2_between else None, "p-value": p_A,  "p-unc": p_A,  "StatisticType": "ANOVA-Type Statistic (ATS)"},
            {"Source": within_factor,                   "ATS": ATS_T,  "F": ATS_T,  "Wald_Chi2": ATS_T,  "df1": f_T,  "df2": None,                                            "p-value": p_T,  "p-unc": p_T,  "StatisticType": "ANOVA-Type Statistic (ATS)"},
            {"Source": f"{between_factor}:{within_factor}", "ATS": ATS_AT, "F": ATS_AT, "Wald_Chi2": ATS_AT, "df1": f_AT, "df2": None,                                            "p-value": p_AT, "p-unc": p_AT, "StatisticType": "ANOVA-Type Statistic (ATS)"},
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
            "test": "Mixed ANOVA [Brunner-Langer ATS Fallback]",
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
            "test": "Mixed ANOVA [Brunner-Langer ATS Fallback]",
            "p_value": None, "statistic": None,
            "posthoc_test": None, "pairwise_comparisons": [],
            "descriptive": {}, "descriptive_transformed": {},
            "alpha": alpha,
            "null_hypothesis": "The relative treatment effects are equal across all groups and time points.",
            "alternative_hypothesis": "At least one relative treatment effect differs.",
            "confidence_interval": None, "ci_level": 0.95,
            "power": None, "effect_size": None, "effect_size_type": None,
            "error": str(exc), "df1": None, "df2": None,
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
            aov_table = sm.stats.anova_lm(ols_model, typ=2)
            return aov_table, {'parametric': True, 'p_norm': p_norm, 'p_levene': p_lev, 'model': 'ANOVA'}
        elif design == 'rm':
            from statsmodels.stats.anova import AnovaRM
            result = AnovaRM(df, depvar=dv, subject=subject, within=factors).fit()
            return result, {'parametric': True, 'p_norm': p_norm, 'p_levene': p_lev, 'model': 'AnovaRM'}
        elif design == 'mixed':
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
