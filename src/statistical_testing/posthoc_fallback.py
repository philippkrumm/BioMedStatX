import logging

import numpy as np
import pandas as pd
from scipy import stats

from lazy_imports import get_statsmodels_multitest
from nonparametricanovas import posthoc_marginaleffects
from stats_functions import UIDialogManager, PostHocFactory, PostHocAnalyzer, PostHocStatistics

logger = logging.getLogger(__name__)


def _get_ui_dialog_manager():
    """Resolve dialog manager through statisticaltester to honor test-time monkeypatches."""
    try:
        from statisticaltester import UIDialogManager as patched_dialog_manager
        return patched_dialog_manager
    except Exception:
        return UIDialogManager


class PosthocFallbackEngine:
    @staticmethod
    def _prefix_pairwise_labels(pairwise_comparisons, prefix):
        updated = []
        for comparison in pairwise_comparisons or []:
            comp = dict(comparison)
            comp["group1"] = f"{prefix}{comp.get('group1', '')}"
            comp["group2"] = f"{prefix}{comp.get('group2', '')}"
            updated.append(comp)
        return updated

    @staticmethod
    def _build_rm_aligned_samples(df, dv, subject, within_factor):
        samples = {}
        complete_subjects = []
        within_levels = sorted(df[within_factor].dropna().unique())

        subject_level_counts = df.groupby(subject)[within_factor].nunique()
        expected_levels = len(within_levels)
        complete_subjects = sorted(subject_level_counts[subject_level_counts == expected_levels].index.tolist())

        if not complete_subjects:
            return [], {}

        df_complete = df[df[subject].isin(complete_subjects)].copy()
        for level in within_levels:
            level_df = df_complete[df_complete[within_factor] == level].sort_values(by=subject)
            samples[str(level)] = level_df[dv].tolist()

        valid_groups = [group for group in within_levels if str(group) in samples and len(samples[str(group)]) > 0]
        return [str(group) for group in valid_groups], samples

    @staticmethod
    def _apply_pairwise_multiplicity(pairwise_rows, alpha=0.05, correction_method="holm-sidak"):
        if not pairwise_rows:
            return pairwise_rows

        valid_indices = []
        p_values = []
        for idx, comparison in enumerate(pairwise_rows):
            p_val = comparison.get("p_value")
            if isinstance(p_val, (float, int)) and not np.isnan(float(p_val)):
                valid_indices.append(idx)
                p_values.append(float(p_val))

        if not p_values:
            return pairwise_rows

        try:
            multipletests = get_statsmodels_multitest()
            _, corrected_p, _, _ = multipletests(p_values, alpha=alpha, method=correction_method)
        except Exception:
            return pairwise_rows

        for idx, corrected_value in zip(valid_indices, corrected_p):
            pairwise_rows[idx]["p_value"] = float(corrected_value)
            pairwise_rows[idx]["significant"] = bool(float(corrected_value) < alpha)
            pairwise_rows[idx]["corrected"] = True
            pairwise_rows[idx]["correction"] = correction_method

        return pairwise_rows

    @staticmethod
    def _map_marginaleffects_to_exporter(
        comparisons_df,
        alpha=0.05,
        by_cols=None,
        test_label="Marginal Effects (Robust GLM)",
        apply_correction=True,
        correction_method="holm-sidak",
    ):
        mapped = []
        by_cols = by_cols or []
        excluded_columns = {
            "term", "contrast", "estimate", "statistic", "p.value", "p_value", "pvalue",
            "s.e.", "std.error", "std_error", "conf.low", "conf.high", "conf_low", "conf_high",
            "z", "t", "df", "rowid", "predicted"
        }

        p_col = next((col for col in ["p.value", "p_value", "pvalue"] if col in comparisons_df.columns), None)
        stat_col = next((col for col in ["statistic", "z", "t"] if col in comparisons_df.columns), None)
        ci_low_col = next((col for col in ["conf.low", "conf_low"] if col in comparisons_df.columns), None)
        ci_high_col = next((col for col in ["conf.high", "conf_high"] if col in comparisons_df.columns), None)

        if p_col is None:
            return mapped

        if not by_cols:
            by_cols = [col for col in comparisons_df.columns if col not in excluded_columns]

        for _, row in comparisons_df.iterrows():
            raw_contrast = str(row.get("contrast", "N/A"))
            if " - " in raw_contrast:
                contrast_parts = [part.strip() for part in raw_contrast.split(" - ", 1)]
                group1, group2 = (contrast_parts + ["N/A"])[:2]
            else:
                group1, group2 = "N/A", "N/A"

            prefix_parts = []
            for col in by_cols:
                if col in row and pd.notna(row[col]):
                    prefix_parts.append(f"{col}={row[col]}")
            prefix = " | ".join(prefix_parts)

            p_value = row.get(p_col)
            statistic = row.get(stat_col) if stat_col else None
            ci_low = row.get(ci_low_col) if ci_low_col else None
            ci_high = row.get(ci_high_col) if ci_high_col else None

            try:
                p_value = float(p_value) if pd.notna(p_value) else None
            except Exception:
                p_value = None

            try:
                statistic = float(statistic) if statistic is not None and pd.notna(statistic) else None
            except Exception:
                statistic = None

            confidence_interval = None
            if ci_low is not None and ci_high is not None and pd.notna(ci_low) and pd.notna(ci_high):
                try:
                    confidence_interval = (float(ci_low), float(ci_high))
                except Exception:
                    confidence_interval = None

            mapped.append({
                "group1": f"{prefix} | {group1}" if prefix else group1,
                "group2": f"{prefix} | {group2}" if prefix else group2,
                "test": test_label,
                "p_value": p_value,
                "statistic": statistic,
                "significant": bool(p_value is not None and p_value < alpha),
                "corrected": bool(apply_correction),
                "correction": correction_method if apply_correction else None,
                "effect_size": None,
                "effect_size_type": None,
                "confidence_interval": confidence_interval
            })

        if apply_correction:
            mapped = PosthocFallbackEngine._apply_pairwise_multiplicity(
                mapped,
                alpha=alpha,
                correction_method=correction_method,
            )

        return mapped

    @staticmethod
    def _run_two_way_marginaleffects_posthoc(results, between, alpha=0.05):
        fitted_model = results.get("fitted_model")
        if fitted_model is None or not between or len(between) < 2:
            return None

        primary_factor = between[0]
        by_factor = between[1]

        try:
            posthoc = posthoc_marginaleffects(
                fitted_model,
                variables=primary_factor,
                by=[by_factor],
                to_pandas=True
            )
            comparisons_df = posthoc.get("comparisons")
            if comparisons_df is None or getattr(comparisons_df, "empty", True):
                return None

            mapped = PosthocFallbackEngine._map_marginaleffects_to_exporter(
                comparisons_df,
                alpha=alpha,
                by_cols=[by_factor],
                test_label="Marginal Effects (Robust GLM)",
                apply_correction=True,
                correction_method="holm-sidak",
            )
            if not mapped:
                return None

            return {
                "posthoc_test": "Marginal Effects Pairwise Comparisons",
                "pairwise_comparisons": mapped,
                "error": None
            }
        except Exception as exc:
            return {
                "posthoc_test": None,
                "pairwise_comparisons": [],
                "error": f"Marginal effects post-hoc failed: {str(exc)}"
            }

    @staticmethod
    def _run_rm_marginaleffects_posthoc(results, within_factor, alpha=0.05):
        fitted_model = results.get("fitted_model")
        if fitted_model is None or not within_factor:
            return None
        # GEE not yet implemented — this function is disabled until GEE model_class is set
        if results.get("model_class") != "GEE":
            return None

        try:
            posthoc = posthoc_marginaleffects(
                fitted_model,
                variables=within_factor,
                by=None,
                to_pandas=True
            )
            comparisons_df = posthoc.get("comparisons")
            if comparisons_df is None or getattr(comparisons_df, "empty", True):
                return None

            mapped = PosthocFallbackEngine._map_marginaleffects_to_exporter(
                comparisons_df,
                alpha=alpha,
                by_cols=[],
                test_label="Marginal Effects (RM)",
                apply_correction=True,
                correction_method="holm-sidak",
            )
            if not mapped:
                return None

            return {
                "posthoc_test": "Marginal Effects Pairwise Comparisons (GEE-based)",
                "pairwise_comparisons": mapped,
                "error": None
            }
        except Exception as exc:
            return {
                "posthoc_test": None,
                "pairwise_comparisons": [],
                "error": f"Marginal effects RM post-hoc failed: {str(exc)}"
            }

    @staticmethod
    def _run_mixed_marginaleffects_posthoc(results, between, within, alpha=0.05):
        fitted_model = results.get("fitted_model")
        if fitted_model is None or not between or not within:
            return None
        # GEE not yet implemented — this function is disabled until GEE model_class is set
        if results.get("model_class") != "GEE":
            return None

        between_factor = between[0]
        within_factor = within[0]
        pairwise_comparisons = []
        errors = []

        # Pass 1: Between-factor comparisons at fixed within levels
        try:
            posthoc_between = posthoc_marginaleffects(
                fitted_model,
                variables=between_factor,
                by=[within_factor],
                to_pandas=True
            )
            comparisons_df = posthoc_between.get("comparisons")
            if comparisons_df is not None and not getattr(comparisons_df, "empty", True):
                mapped = PosthocFallbackEngine._map_marginaleffects_to_exporter(
                    comparisons_df,
                    alpha=alpha,
                    by_cols=[within_factor],
                    test_label="Marginal Effects (Mixed: Between at fixed Within)",
                    apply_correction=False,
                )
                pairwise_comparisons.extend(mapped)
        except Exception as exc:
            error_msg = f"Mixed marginaleffects between-pass failed: {str(exc)}"
            errors.append(error_msg)
            warnings_list = results.setdefault("warnings", [])
            if error_msg not in warnings_list:
                warnings_list.append(error_msg)

        # Pass 2: Within-factor comparisons at fixed between levels
        try:
            posthoc_within = posthoc_marginaleffects(
                fitted_model,
                variables=within_factor,
                by=[between_factor],
                to_pandas=True
            )
            comparisons_df = posthoc_within.get("comparisons")
            if comparisons_df is not None and not getattr(comparisons_df, "empty", True):
                mapped = PosthocFallbackEngine._map_marginaleffects_to_exporter(
                    comparisons_df,
                    alpha=alpha,
                    by_cols=[between_factor],
                    test_label="Marginal Effects (Mixed: Within at fixed Between)",
                    apply_correction=False,
                )
                pairwise_comparisons.extend(mapped)
        except Exception as exc:
            error_msg = f"Mixed marginaleffects within-pass failed: {str(exc)}"
            errors.append(error_msg)
            warnings_list = results.setdefault("warnings", [])
            if error_msg not in warnings_list:
                warnings_list.append(error_msg)

        if not pairwise_comparisons:
            if errors:
                return {
                    "posthoc_test": None,
                    "pairwise_comparisons": [],
                    "error": " | ".join(errors)
                }
            return None

        pairwise_comparisons = PosthocFallbackEngine._apply_pairwise_multiplicity(
            pairwise_comparisons,
            alpha=alpha,
            correction_method="holm-sidak",
        )

        return {
            "posthoc_test": "Marginal Effects Pairwise Comparisons (Mixed GEE-based)",
            "pairwise_comparisons": pairwise_comparisons,
            "error": " | ".join(errors) if errors else None
        }

    @staticmethod
    def _run_modern_fallback_posthoc(df, test, dv, subject=None, between=None, within=None, alpha=0.05):
        result = {
            "posthoc_test": "Non-parametric pairwise tests (Holm-Sidak corrected)",
            "pairwise_comparisons": [],
            "error": None
        }

        try:
            if test == "two_way_anova":
                factor_a, factor_b = between
                samples = {}
                group_labels = []
                for a_val in sorted(df[factor_a].dropna().unique()):
                    for b_val in sorted(df[factor_b].dropna().unique()):
                        label = f"{factor_a}={a_val}, {factor_b}={b_val}"
                        subset = df[(df[factor_a] == a_val) & (df[factor_b] == b_val)][dv].dropna().tolist()
                        if subset:
                            samples[label] = subset
                            group_labels.append(label)

                if len(group_labels) > 1:
                    posthoc = PosthocFallbackEngine.perform_refactored_posthoc_testing(
                        group_labels,
                        samples,
                        "non_parametric",
                        alpha=alpha,
                        posthoc_choice="dunn"
                    )
                    if posthoc:
                        result["pairwise_comparisons"].extend(posthoc.get("pairwise_comparisons", []))
                        if posthoc.get("error"):
                            result["error"] = posthoc["error"]

            elif test == "repeated_measures_anova":
                within_factor = within[0]
                valid_groups, samples = PosthocFallbackEngine._build_rm_aligned_samples(df, dv, subject, within_factor)
                if len(valid_groups) > 1:
                    posthoc = PosthocFallbackEngine.perform_dependent_posthoc_tests(
                        samples,
                        valid_groups,
                        alpha=alpha,
                        parametric=False
                    )
                    if posthoc:
                        result["pairwise_comparisons"].extend(posthoc.get("pairwise_comparisons", []))
                        if posthoc.get("error"):
                            result["error"] = posthoc["error"]

            elif test == "mixed_anova":
                between_factor = between[0]
                within_factor = within[0]

                # Between-subject comparisons within each within-factor level
                for within_level in sorted(df[within_factor].dropna().unique()):
                    subset = df[df[within_factor] == within_level].copy()
                    samples = {}
                    groups = []
                    for between_level in sorted(subset[between_factor].dropna().unique()):
                        label = str(between_level)
                        values = subset[subset[between_factor] == between_level][dv].dropna().tolist()
                        if values:
                            samples[label] = values
                            groups.append(label)

                    if len(groups) > 1:
                        posthoc = PosthocFallbackEngine.perform_refactored_posthoc_testing(
                            groups,
                            samples,
                            "non_parametric",
                            alpha=alpha,
                            posthoc_choice="dunn"
                        )
                        if posthoc:
                            prefixed = PosthocFallbackEngine._prefix_pairwise_labels(
                                posthoc.get("pairwise_comparisons", []),
                                f"{within_factor}={within_level} | "
                            )
                            result["pairwise_comparisons"].extend(prefixed)
                            if posthoc.get("error") and result["error"] is None:
                                result["error"] = posthoc["error"]

                # Within-subject comparisons within each between-factor level
                for between_level in sorted(df[between_factor].dropna().unique()):
                    subset = df[df[between_factor] == between_level].copy()
                    valid_groups, samples = PosthocFallbackEngine._build_rm_aligned_samples(
                        subset,
                        dv,
                        subject,
                        within_factor
                    )
                    if len(valid_groups) > 1:
                        posthoc = PosthocFallbackEngine.perform_dependent_posthoc_tests(
                            samples,
                            valid_groups,
                            alpha=alpha,
                            parametric=False
                        )
                        if posthoc:
                            prefixed = PosthocFallbackEngine._prefix_pairwise_labels(
                                posthoc.get("pairwise_comparisons", []),
                                f"{between_factor}={between_level} | "
                            )
                            result["pairwise_comparisons"].extend(prefixed)
                            if posthoc.get("error") and result["error"] is None:
                                result["error"] = posthoc["error"]

            if not result["pairwise_comparisons"] and result["error"] is None:
                result["error"] = "No valid non-parametric pairwise comparisons could be performed."

            return result
        except Exception as exc:
            result["error"] = f"Error performing fallback post-hoc tests: {str(exc)}"
            return result
    
    @staticmethod
    def perform_dependent_posthoc_tests(data_dict, groups, alpha=0.05, parametric=True):
        """
        Performs post-hoc tests for dependent samples.
    
        Parameters:
        -----------
        data_dict : dict
            Dictionary with group names as keys and lists of values as values
        groups : list
            List of groups to analyze
        alpha : float
            Significance level
        parametric : bool
            Whether to perform parametric tests
    
        Returns:
        --------
        dict
            Results of the post-hoc tests
        """
        # In this transition phase, use the new implementation
        return PosthocFallbackEngine.perform_refactored_posthoc_testing(
            groups,
            data_dict,
            "parametric" if parametric else "non_parametric",
            alpha=alpha,
            posthoc_choice="dependent",
            is_dependent=True
        )
    @staticmethod
    def perform_refactored_posthoc_testing(
        valid_groups,
        samples,
        test_recommendation,
        alpha=0.05,
        posthoc_choice=None,
        control_group=None,
        is_dependent=False,
        test_info=None,
    ):
        """
        Central function for performing post-hoc tests with the new framework.
        Can be used as a replacement for the existing perform_posthoc_testing.

        Parameters:
        -----------
        valid_groups : list
            List of groups to analyze
        samples : dict
            Dictionary with group names as keys and lists of values
        test_recommendation : str
            "parametric" or "non_parametric", based on normality tests
        alpha : float, optional
            Significance level (default: 0.05)
        posthoc_choice : str, optional
            Specific post-hoc test: "tukey", "dunnett", "dunn" or None
        control_group : str, optional
            Control group for Dunnett test
        is_dependent : bool, optional
            Indicates whether samples are dependent

        Returns:
        --------
        dict
            Dictionary with post-hoc test results and pairwise comparisons
        """
        # Initialize default result
        result = {
            "posthoc_test": None,
            "pairwise_comparisons": [],
            "error": None
        }
        ui_dialog_manager = _get_ui_dialog_manager()

        # Check validity
        if len(valid_groups) <= 1:
            result["error"] = "At least two groups are required for post-hoc tests."
            return result

        # Automatic test selection if not explicitly specified
        if posthoc_choice is None:
            if is_dependent:
                posthoc_choice = "dependent"
            elif test_recommendation in ("parametric", "welch"):
                # Show dialog for parametric post-hoc test selection.
                # When Welch ANOVA was used (unequal variances), pre-select Games-Howell.
                default_method = "games_howell" if test_recommendation == "welch" else "tukey"
                try:
                    variance_info = test_info.get("variance_test", {}) if isinstance(test_info, dict) else {}
                    equal_variance = None
                    if default_method == "games_howell":
                        equal_variance = False
                    elif isinstance(variance_info, dict):
                        transformed_variance = variance_info.get("transformed")
                        if isinstance(transformed_variance, dict) and transformed_variance.get("equal_variance") is not None:
                            equal_variance = transformed_variance.get("equal_variance")
                        elif variance_info.get("equal_variance") is not None:
                            equal_variance = variance_info.get("equal_variance")
                    posthoc_choice = ui_dialog_manager.select_posthoc_test_dialog(
                        parent=None, progress_text=None, column_name=None,
                        default_method=default_method, equal_variance=equal_variance
                    )
                    logger.debug(f"DEBUG: Parametric post-hoc dialog returned: {posthoc_choice}")
                    if posthoc_choice is None:
                        posthoc_choice = default_method
                        print(f"DEBUG: Parametric post-hoc dialog cancelled, defaulting to {default_method}")
                except Exception as e:
                    logger.debug(f"DEBUG: Error showing parametric post-hoc dialog: {e}")
                    posthoc_choice = default_method
            else:
                # NEW: Dialog for non-parametric post-hoc tests
                print("DEBUG: About to show non-parametric post-hoc dialog")
                try:
                    posthoc_choice = ui_dialog_manager.select_nonparametric_posthoc_dialog(
                        parent=None, progress_text=None, column_name=None
                    )
                    logger.debug(f"DEBUG: Non-parametric post-hoc dialog returned: {posthoc_choice}")
                    # If dialog was cancelled or returned None, default to Dunn
                    if posthoc_choice is None:
                        posthoc_choice = "dunn"
                        print("DEBUG: Non-parametric post-hoc dialog cancelled, defaulting to Dunn test")
                except Exception as e:
                    logger.debug(f"DEBUG: Error showing non-parametric post-hoc dialog: {e}")
                    import traceback
                    traceback.print_exc()
                    posthoc_choice = "dunn"  # Fallback to Dunn test
        
        # If Dunnett was selected, we need a control group
        if posthoc_choice == "dunnett" and control_group is None:
            try:
                control_group = ui_dialog_manager.select_control_group_dialog(valid_groups)
                logger.debug(f"DEBUG: Control group selected for Dunnett test: {control_group}")
                if control_group is None:
                    print("DEBUG: No control group selected, defaulting to Tukey HSD")
                    posthoc_choice = "tukey"
            except Exception as e:
                logger.debug(f"DEBUG: Error selecting control group: {e}")
                posthoc_choice = "tukey"  # Fallback to Tukey if control selection fails

        try:
            is_parametric = test_recommendation == "parametric"

            # Create the appropriate test
            if posthoc_choice == "dependent":
                test_instance = PostHocFactory.create_test(None, is_parametric=is_parametric, is_dependent=True)
                if test_instance:
                    return test_instance.perform_test(valid_groups, samples, alpha=alpha, parametric=is_parametric)
                
            elif posthoc_choice == "paired_custom":
                # Open dialog for pair selection
                pairs = ui_dialog_manager.select_custom_pairs_dialog(valid_groups)
                if not pairs:
                    result["error"] = "No pairs selected."
                    return result
                # Paired t-tests for the selected pairs
                pvals, stats_list = [], []
                for g1, g2 in pairs:
                    x, y = np.array(samples[g1]), np.array(samples[g2])
                    tstat, p = stats.ttest_rel(x, y)
                    stats_list.append(tstat)
                    pvals.append(p)
                # Holm-Sidak-Korrektur
                multipletests = get_statsmodels_multitest()
                reject, p_adj, _, _ = multipletests(pvals, alpha=alpha, method='holm-sidak')
                # Ergebnisse sammeln
                for i, (g1, g2) in enumerate(pairs):
                    ci = PostHocStatistics.calculate_ci_mean_diff(samples[g1], samples[g2], alpha=alpha, paired=True)
                    d = PostHocStatistics.calculate_cohens_d(samples[g1], samples[g2], paired=True)
                    PostHocAnalyzer.add_comparison(
                        result,
                        group1=g1,
                        group2=g2,
                        test="Paired t-test",
                        p_value=p_adj[i],
                        statistic=stats_list[i],
                        corrected=True,
                        correction_method="Holm-Sidak",
                        effect_size=d,
                        effect_size_type="cohen_d",
                        confidence_interval=ci,
                        alpha=alpha
                    )
                result["posthoc_test"] = "Custom paired t-tests (Holm-Sidak)"
                return result
                
            elif posthoc_choice == "mw_custom":
                # NEW: Pairwise Mann-Whitney-U (Šidák, custom pairs)
                pairs = ui_dialog_manager.select_custom_pairs_dialog(valid_groups)
                if not pairs:
                    result["error"] = "No pairs selected."
                    return result
                mannwhitneyu = stats.mannwhitneyu
                pvals, stats_list = [], []
                for g1, g2 in pairs:
                    x, y = np.array(samples[g1]), np.array(samples[g2])
                    stat, p = mannwhitneyu(x, y, alternative='two-sided')
                    stats_list.append(stat)
                    pvals.append(p)
                k = len(pvals)
                sidak_ps = [1 - (1 - p)**k for p in pvals]
                sidak_ps = [min(p, 1.0) for p in sidak_ps]
                for i, (g1, g2) in enumerate(pairs):
                    n1, n2 = len(samples[g1]), len(samples[g2])
                    u = stats_list[i]
                    mean_u = n1 * n2 / 2
                    std_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
                    z = (u - mean_u) / std_u if std_u > 0 else 0
                    r = abs(z) / np.sqrt(n1 + n2)
                    PostHocAnalyzer.add_comparison(
                        result,
                        group1=g1,
                        group2=g2,
                        test="Mann-Whitney-U",
                        p_value=sidak_ps[i],
                        statistic=stats_list[i],
                        corrected=True,
                        correction_method="Sidak",
                        effect_size=r,
                        effect_size_type="r",
                        confidence_interval=(None, None),
                        alpha=alpha
                    )
                result["posthoc_test"] = "Custom Mann-Whitney-U tests (Sidak)"
                return result

            elif posthoc_choice == "dunnett":
                if control_group is None or control_group not in valid_groups:
                    result["error"] = "A valid control group must be specified for the Dunnett test."
                    return result
                test_instance = PostHocFactory.create_test("dunnett", is_parametric=is_parametric, is_dependent=False)
                if test_instance:
                    return test_instance.perform_test(valid_groups, samples, control_group=control_group, alpha=alpha)
            else:
                # tukey or dunn
                test_instance = PostHocFactory.create_test(posthoc_choice, is_parametric=is_parametric, is_dependent=False)
                if test_instance:
                    return test_instance.perform_test(valid_groups, samples, alpha=alpha)

            # If no suitable implementation was found
            result["error"] = f"No suitable test available for {posthoc_choice} (parametric: {is_parametric}, dependent: {is_dependent})"
            return result
        except Exception as e:
            import traceback
            result["error"] = f"Error performing post-hoc test: {str(e)}"
            traceback.print_exc()
            return result

    @staticmethod
    def process_results(results):
        print("Processing results:")
        print(f"Keys in results: {list(results.keys())}")
        if 'interactions' in results:
            print(f"Interaction effect p-value: {results['interactions'][0]['p_value'] if results['interactions'] else 'No interaction found'}")
        if 'pairwise_comparisons' in results and results['pairwise_comparisons']:
            print("Pairwise comparisons found:")
            for comparison in results['pairwise_comparisons']:
                print(f"{comparison['group1']} vs {comparison['group2']}: p = {comparison['p_value']}")
        else:
            print("No pairwise comparisons found.")

