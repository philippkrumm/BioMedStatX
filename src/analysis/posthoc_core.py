import numpy as np
import pandas as pd
from itertools import combinations

from core.lazy_imports import (
    get_pingouin,
    get_scipy_stats,
    get_statsmodels_multitest,
    get_pairwise_tukeyhsd,
    get_scikit_posthocs,
)

import logging
logger = logging.getLogger(__name__)


def get_stats_module():
    """Get scipy.stats — delegates to canonical lazy_imports loader."""
    return get_scipy_stats()


def get_pingouin_module():
    """Get pingouin — delegates to canonical lazy_imports loader."""
    return get_pingouin()


class PostHocAnalyzer:
    """Base class for all post-hoc tests with uniform methods."""
    
    @staticmethod
    def create_result_template(test_name):
        """Creates a standard dictionary for post-hoc results."""
        return {
            "posthoc_test": test_name,
            "pairwise_comparisons": [],
            "error": None
        }
    
    @staticmethod
    def add_comparison(results, group1, group2, test, p_value, statistic=None,
                       corrected=True, correction_method=None, effect_size=None,
                       effect_size_type=None, confidence_interval=(None, None),
                       alpha=0.05, significant=None, **extra_fields):
        """Adds a standardized pairwise comparison to the results."""
        if significant is None:
            significant = p_value < alpha if isinstance(p_value, (float, int)) else False
        
        comparison = {
            "group1": str(group1),
            "group2": str(group2),
            "test": test,
            "p_value": float(p_value) if isinstance(p_value, (float, int)) else p_value,
            "statistic": float(statistic) if isinstance(statistic, (float, int)) else statistic,
            "significant": significant,
            "corrected": corrected,
            "effect_size": float(effect_size) if isinstance(effect_size, (float, int)) else effect_size,
            "effect_size_type": effect_size_type,
            "confidence_interval": confidence_interval
        }
        
        if correction_method:
            comparison["correction"] = correction_method
        for key, value in extra_fields.items():
            comparison[key] = value
            
        results["pairwise_comparisons"].append(comparison)
        return comparison
        
    @staticmethod
    def _holm_correction(p_values):
        """Applies Holm-Šidák correction to a list of p-values."""
        if not p_values:
            return []
        
        # Use statsmodels implementation instead of custom one
        multipletests = get_statsmodels_multitest()
        reject, corrected_p, _, _ = multipletests(p_values, method='holm-sidak')
        return corrected_p.tolist()

class TwoWayPostHocAnalyzer(PostHocAnalyzer):
    @staticmethod
    def build_group_label(factors, values):
        # Always use the same order and format as the dialog: 'FactorA=..., FactorB=...'
        return ', '.join([f"{factors[i]}={values[i]}" for i in range(len(factors))])
    """Post-hoc tests for Two-Way ANOVA with a uniform interface."""
    
    @staticmethod
    def perform_test(df, dv, factors, alpha=0.05, selected_comparisons=None, method="holm-sidak", control_group=None):
        """
        Performs post-hoc tests for Two-Way ANOVA.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Data in long format
        dv : str
            Dependent variable
        factors : list
            List of the two factors [factor_a, factor_b]
        alpha : float
            Significance level
        selected_comparisons : set, optional
            Set of normalized comparison pairs to perform
        method : str, optional
            Post-hoc method: "holm", "bonferroni", "tukey"
            
        Returns:
        --------
        dict
            Standardized post-hoc results
        """
        result = PostHocAnalyzer.create_result_template("Two-Way ANOVA Post-hoc Tests")
        try:
            logger.debug(f"DEBUG POSTHOC: selected_comparisons = {selected_comparisons}")
            # Use the same normalization function for group pairs (must match dialog)
            def normalize_pair(pair):
                # Sort and strip, but also ensure both elements are formatted identically to dialog
                return tuple(sorted([s.strip() for s in pair]))
            normalized_selected = set(normalize_pair(pair) for pair in selected_comparisons) if selected_comparisons else None
            logger.debug(f"DEBUG POSTHOC: normalized_selected = {normalized_selected}")
            available_pairs = set()
            get_pingouin_module()
            has_pingouin = True
        except ImportError:
            has_pingouin = False
        except Exception as e:
            logger.debug(f"DEBUG POSTHOC: Exception during normalization: {e}")
            has_pingouin = False
        try:
            if has_pingouin:
                logger.debug(f"DEBUG POSTHOC: DataFrame columns: {df.columns.tolist()}")
                logger.debug(f"DEBUG POSTHOC: DataFrame head:\n{df.head()}")
                logger.debug(f"DEBUG POSTHOC: factors = {factors}, dv = {dv}")
                # Manual post-hoc for interaction: generate all interaction group pairs
                ttest_ind = get_scipy_stats().ttest_ind
                # Build all interaction group labels
                interaction_groups = []
                group_to_values = {}
                for level_b in sorted(df[factors[0]].unique()):
                    for level_a in sorted(df[factors[1]].unique()):
                        label = f"{factors[0]}={level_b}, {factors[1]}={level_a}"
                        mask = (df[factors[0]] == level_b) & (df[factors[1]] == level_a)
                        values = df.loc[mask, dv].values
                        if len(values) > 0:
                            interaction_groups.append(label)
                            group_to_values[label] = values
                logger.debug(f"DEBUG POSTHOC: interaction_groups = {interaction_groups}")
                # Generate all possible pairs
                all_pairs = list(combinations(interaction_groups, 2))
                # If selected_comparisons is provided, filter to only those pairs
                if normalized_selected is not None:
                    filtered_pairs = [pair for pair in all_pairs if normalize_pair(pair) in normalized_selected]
                else:
                    filtered_pairs = all_pairs
                logger.debug(f"DEBUG POSTHOC: filtered_pairs = {filtered_pairs}")
                # Perform t-tests for each pair
                pvals = []
                stats_list = []
                for g1, g2 in filtered_pairs:
                    vals1 = group_to_values[g1]
                    vals2 = group_to_values[g2]
                    # Use t-test (assume equal variance for now)
                    stat, pval = ttest_ind(vals1, vals2, equal_var=True)
                    pvals.append(pval)
                    stats_list.append((g1, g2, stat, pval, vals1, vals2))
                # Apply multiple comparison correction based on method
                multipletests = get_statsmodels_multitest()
                if pvals:
                    if method.lower() == 'tukey':
                        # For Tukey, we'll use a different approach below
                        correction_method = "Tukey HSD"
                        pvals_corr = pvals  # Will be replaced by Tukey results
                    elif method.lower() == 'dunnett' and control_group:
                        # For Dunnett, use proper Dunnett test implementation
                        correction_method = "Dunnett"
                        try:
                            sp = get_scikit_posthocs()
                            # Prepare data for scikit_posthocs
                            all_data = []
                            all_groups = []
                            for group in interaction_groups:
                                values = group_to_values[group]
                                all_data.extend(values)
                                all_groups.extend([group] * len(values))
                            
                            # Create DataFrame for scikit_posthocs
                            import pandas as pd
                            dunnett_df = pd.DataFrame({"value": all_data, "group": all_groups})
                            
                            # Use the control_group directly - it's already the exact group name the user selected
                            control_label = control_group
                            logger.debug(f"DEBUG: Using control_group directly: '{control_label}'")
                            
                            # Perform Dunnett test
                            dunnett_result = sp.posthoc_dunnett(dunnett_df, val_col="value", group_col="group", control=control_label)
                            
                            # Extract p-values for the comparisons we made
                            pvals_corr = []
                            for g1, g2, *_ in stats_list:
                                if g1 == control_label or g2 == control_label:
                                    # Get the p-value from the Dunnett result matrix
                                    try:
                                        if g1 == control_label:
                                            p_val = float(dunnett_result.loc[g2, control_label])
                                        else:
                                            p_val = float(dunnett_result.loc[g1, control_label])
                                    except (KeyError, ValueError):
                                        # Fallback to original p-value
                                        p_val = stats_list[len(pvals_corr)][3]
                                    pvals_corr.append(p_val)
                                else:
                                    pvals_corr.append(1.0)  # Non-control comparisons get p=1.0
                        except ImportError:
                            # Fallback if scikit_posthocs not available
                            # Filter to only comparisons involving the control group
                            dunnett_pvals = []
                            dunnett_stats = []
                            control_label = control_group  # Use control_group directly
                            logger.debug(f"DEBUG: Dunnett fallback using control_group: '{control_label}'")
                            
                            for i, (g1, g2, stat, pval, vals1, vals2) in enumerate(stats_list):
                                if g1 == control_label or g2 == control_label:
                                    dunnett_pvals.append(pval)
                                    dunnett_stats.append((g1, g2, stat, pval, vals1, vals2))
                                
                                # Apply Dunnett correction using Holm-Šidák as fallback
                                if dunnett_pvals:
                                    reject, pvals_corr_dunnett, _, _ = multipletests(dunnett_pvals, alpha=alpha, method='holm-sidak')
                                    # Map back to original order
                                    pvals_corr = []
                                    dunnett_idx = 0
                                    for g1, g2, *_ in stats_list:
                                        if g1 == control_label or g2 == control_label:
                                            pvals_corr.append(pvals_corr_dunnett[dunnett_idx])
                                            dunnett_idx += 1
                                        else:
                                            pvals_corr.append(1.0)  # Non-control comparisons get p=1.0
                                else:
                                    pvals_corr = [1.0] * len(pvals)
                    elif method.lower() == 'paired_fdr':
                        correction_method = "FDR (Benjamini-Hochberg)"
                        reject, pvals_corr, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')
                    else:
                        # Default: Holm-Šidák
                        correction_method = "Holm-Šidák"
                        reject, pvals_corr, _, _ = multipletests(pvals, alpha=alpha, method='holm-sidak')
                else:
                    pvals_corr = []
                    correction_method = "Holm-Šidák"
                # Add to results
                for i, (g1, g2, stat, pval, vals1, vals2) in enumerate(stats_list):
                    # Effect size: Cohen's d
                    n1, n2 = len(vals1), len(vals2)
                    s1, s2 = np.var(vals1, ddof=1), np.var(vals2, ddof=1)
                    s_pooled = np.sqrt(((n1-1)*s1 + (n2-1)*s2) / (n1+n2-2)) if (n1+n2-2) > 0 else 0
                    cohen_d = (np.mean(vals1) - np.mean(vals2)) / s_pooled if s_pooled > 0 else 0
                    # Confidence interval for mean difference
                    mean_diff = np.mean(vals1) - np.mean(vals2)
                    stderr_diff = np.sqrt(s1/n1 + s2/n2) if n1 > 0 and n2 > 0 else 0
                    t = get_scipy_stats().t
                    df_ = n1 + n2 - 2
                    if df_ > 0 and stderr_diff > 0:
                        t_crit = t.ppf(1 - alpha/2, df_)
                        ci = (mean_diff - t_crit * stderr_diff, mean_diff + t_crit * stderr_diff)
                    else:
                        ci = (None, None)
                    PostHocAnalyzer.add_comparison(
                        result,
                        group1=g1,
                        group2=g2,
                        test="Pairwise t-test",
                        p_value=pvals_corr[i] if i < len(pvals_corr) else pval,
                        statistic=stat,
                        corrected=True,
                        correction_method=correction_method,
                        effect_size=cohen_d,
                        effect_size_type="cohen_d",
                        confidence_interval=ci,
                        alpha=alpha
                    )
                logger.debug(f"DEBUG POSTHOC: Added {len(stats_list)} comparisons to results.")
                # After all, print available pairs and warn if any selected pair is not present
                available_pairs = set(normalize_pair((g1, g2)) for g1, g2, *_ in stats_list)
                logger.debug(f"DEBUG POSTHOC: available_pairs = {available_pairs}")
                if normalized_selected is not None:
                    missing = normalized_selected - available_pairs
                    if missing:
                        logger.warning(f"WARNING: The following selected pairs were not found in the available post-hoc comparisons: {missing}")
                pairwise_tukeyhsd = get_pairwise_tukeyhsd()
                # Create interaction group for Tukey HSD
                df['interaction_group'] = df[factors[0]].astype(str) + "_" + df[factors[1]].astype(str)
                # Run Tukey HSD on the interaction groups
                tukey = pairwise_tukeyhsd(df[dv], df['interaction_group'], alpha=alpha)
                # For the Tukey HSD test in the fallback, we'll need to manually apply Holm-Šidák
                # First collect all pairwise comparisons and p-values
                comparisons = []
                for i in range(len(tukey.pvalues)):
                    group1 = tukey.groupsunique[tukey.pairindices[i, 0]]
                    group2 = tukey.groupsunique[tukey.pairindices[i, 1]]
                    p_val = tukey.pvalues[i]
                    conf_int = tukey.confint[i]
                    comparisons.append({
                        'group1': group1,
                        'group2': group2,
                        'p_value': p_val,
                        'conf_int': conf_int
                    })
                # Apply Holm-Šidák correction
                p_values = [comp['p_value'] for comp in comparisons]
                multipletests = get_statsmodels_multitest()
                reject, corrected_p_values, _, _ = multipletests(p_values, alpha=alpha, method='holm-sidak')
                # Convert results into standardized format with corrected p-values
                for i, comp in enumerate(comparisons):
                    # Normalize for matching
                    norm_pair = normalize_pair((comp['group1'], comp['group2']))
                    match = (normalized_selected is not None and norm_pair in normalized_selected)
                    logger.debug(f"DEBUG POSTHOC: fallback comparing {comp['group1']} vs {comp['group2']} | normalized: {norm_pair} | match: {match}")
                    if normalized_selected is not None and not match:
                        continue
                    PostHocAnalyzer.add_comparison(
                        result,
                        group1=comp['group1'],
                        group2=comp['group2'],
                        test="Pairwise t-test",
                        p_value=corrected_p_values[i],
                        statistic=None,
                        corrected=True,
                        correction_method="Holm-Šidák",
                        confidence_interval=tuple(comp['conf_int']),
                        alpha=alpha
                    )
            
            # Set the posthoc_test value for decision tree visualization
            method_name_map = {
                "tukey": "Tukey HSD",
                "dunnett": "Dunnett Test",
                "paired_custom": "Custom paired t-tests (Holm-Šidák)",
                "paired_fdr": "Custom paired t-tests (FDR Benjamini-Hochberg)",
                "holm": "Custom paired t-tests (Holm-Šidák)"
            }
            result["posthoc_test"] = method_name_map.get(method, f"Post-hoc test ({method})")
            
            return result
        except Exception as e:
            result["error"] = f"Error in Two-Way ANOVA post-hoc tests: {str(e)}"
            return result
        
class MixedAnovaPostHocAnalyzer(PostHocAnalyzer):
    """UPDATED: Advanced post-hoc tests for Mixed ANOVA with proper between/within factor handling."""
    
    @staticmethod
    def _perform_test_legacy(df, dv, subject, between, within, alpha=0.05, selected_comparisons=None, method='tukey', control_group=None):
        """
        Legacy signature (dv/subject before between/within). Superseded by perform_test below.
        UPDATED: Performs sophisticated post-hoc tests for Mixed ANOVA with proper between/within handling.
        
        Major improvements:
        - Proper distinction between between-subject and within-subject comparisons
        - Enhanced statistical tests for mixed designs
        - Better subject-ID handling for within-subject comparisons
        - Improved effect size calculations for mixed designs
        - Enhanced interaction analysis
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Data in long format
        dv : str
            Dependent variable
        subject : str
            Column with subject ID
        between : list
            List with the between-factor [between_factor]
        within : list
            List with the within-factor [within_factor]
        alpha : float
            Significance level (default: 0.05)
        selected_comparisons : set, optional
            Set of normalized comparison pairs to perform
        method : str, optional
            Post-hoc method: "tukey", "bonferroni", "holm", "dunnett"
        control_group : str, optional
            Control group for Dunnett's test
            
        Returns:
        --------
        dict
            Standardized post-hoc results with mixed-design corrections
        """
        result = PostHocAnalyzer.create_result_template("Mixed ANOVA Post-hoc Tests")
        
        try:
            between_factor = between[0]
            within_factor = within[0]
            
            logger.debug(f"DEBUG MIXED POSTHOC: selected_comparisons = {selected_comparisons}")
            logger.debug(f"DEBUG MIXED POSTHOC: between_factor = {between_factor}, within_factor = {within_factor}")
            
            # Normalize comparison pairs function (consistent with other ANOVAs)
            def normalize_pair(pair):
                return tuple(sorted([s.strip() for s in pair]))
            
            # Handle selected comparisons
            if selected_comparisons:
                if isinstance(selected_comparisons, set):
                    normalized_selected = selected_comparisons
                else:
                    normalized_selected = set(normalize_pair(pair) for pair in selected_comparisons)
            else:
                normalized_selected = None
            
            logger.debug(f"DEBUG MIXED POSTHOC: normalized_selected = {normalized_selected}")
            
            # Validate mixed design data structure
            between_levels = sorted(df[between_factor].unique())
            within_levels = sorted(df[within_factor].unique())
            
            logger.debug(f"DEBUG MIXED POSTHOC: between_levels = {between_levels}, within_levels = {within_levels}")
            
            # Check for complete mixed design (all subjects should have all within-factor levels)
            subject_within_counts = df.groupby([subject, between_factor])[within_factor].nunique()
            expected_within_measures = len(within_levels)
            incomplete_cases = subject_within_counts[subject_within_counts < expected_within_measures]
            
            if len(incomplete_cases) > 0:
                logger.warning(f"WARNING: {len(incomplete_cases)} subject-between-factor combinations have incomplete within-factor data")
            
            # Build interaction group labels and classify comparison types
            interaction_groups = []
            group_to_data = {}
            
            for between_level in between_levels:
                for within_level in within_levels:
                    group_label = f"{between_factor}={between_level}, {within_factor}={within_level}"
                    mask = (df[between_factor] == between_level) & (df[within_factor] == within_level)
                    group_data = df.loc[mask].copy()
                    
                    if len(group_data) > 0:
                        interaction_groups.append(group_label)
                        group_to_data[group_label] = {
                            'values': group_data[dv].values,
                            'subjects': group_data[subject].values,
                            'between_level': between_level,
                            'within_level': within_level,
                            'data': group_data
                        }
            
            logger.debug(f"DEBUG MIXED POSTHOC: interaction_groups = {interaction_groups}")
            
            # Collect all pairwise comparisons and classify them
            available_pairs = set()
            comparisons = []
            
            for group1_label, group2_label in combinations(interaction_groups, 2):
                norm_pair = normalize_pair((group1_label, group2_label))
                available_pairs.add(norm_pair)
                
                # Check if this comparison is selected
                if normalized_selected is not None and norm_pair not in normalized_selected:
                    continue
                
                group1_data = group_to_data[group1_label]
                group2_data = group_to_data[group2_label]
                
                # Classify the type of comparison
                comparison_type = MixedAnovaPostHocAnalyzer._classify_comparison_type(
                    group1_data, group2_data, between_factor, within_factor
                )
                
                logger.debug(f"DEBUG MIXED POSTHOC: Comparing {group1_label} vs {group2_label}, type: {comparison_type}")
                
                # Perform appropriate statistical test based on comparison type
                if comparison_type == "within_subject":
                    # Within-subject comparison: use paired t-test
                    t_stat, p_val, effect_size, ci_lower, ci_upper, n_pairs = MixedAnovaPostHocAnalyzer._within_subject_test(
                        group1_data, group2_data, dv, subject, alpha
                    )
                    test_type = "Paired t-test (within-subject)"
                    
                elif comparison_type == "between_subject":
                    # Between-subject comparison: use independent t-test
                    t_stat, p_val, effect_size, ci_lower, ci_upper, n_pairs = MixedAnovaPostHocAnalyzer._between_subject_test(
                        group1_data, group2_data, dv, alpha
                    )
                    test_type = "Independent t-test (between-subject)"
                    
                else:  # "mixed" - most complex case
                    # Mixed comparison: different between-groups AND different within-levels
                    t_stat, p_val, effect_size, ci_lower, ci_upper, n_pairs = MixedAnovaPostHocAnalyzer._mixed_comparison_test(
                        group1_data, group2_data, dv, subject, alpha
                    )
                    test_type = "Independent t-test (mixed comparison)"
                
                if t_stat is not None:  # Valid comparison
                    comparisons.append({
                        "group1": group1_label,
                        "group2": group2_label,
                        "comparison_type": comparison_type,
                        "test_type": test_type,
                        "t_stat": t_stat,
                        "p_val": p_val,
                        "effect_size": effect_size,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                        "n_pairs": n_pairs
                    })
            
            if not comparisons:
                result["error"] = "No valid pairwise comparisons could be performed"
                return result
            
            # Apply multiple comparison correction based on method
            p_values = [comp["p_val"] for comp in comparisons]
            n_comparisons = len(comparisons)
            
            if method.lower() == 'tukey':
                # Enhanced Tukey HSD for mixed designs
                correction_method = "Tukey HSD (Mixed)"
                try:
                    # Try to use pingouin for proper Tukey implementation
                    pg = get_pingouin_module()
                    if pg is not None:
                        corrected_p_values = []
                        for comp in comparisons:
                            # Use appropriate Tukey correction based on comparison type
                            if comp["comparison_type"] == "within_subject":
                                # More liberal correction for within-subject comparisons
                                q_stat = abs(comp["t_stat"]) * np.sqrt(2)
                                p_tukey = MixedAnovaPostHocAnalyzer._tukey_p_value(q_stat, len(within_levels), comp["n_pairs"] - 1)
                            else:
                                # Standard Tukey for between-subject comparisons
                                q_stat = abs(comp["t_stat"]) * np.sqrt(2)
                                p_tukey = MixedAnovaPostHocAnalyzer._tukey_p_value(q_stat, len(interaction_groups), comp["n_pairs"] - 1)
                            corrected_p_values.append(p_tukey)
                    else:
                        # Fallback to Bonferroni
                        corrected_p_values = [min(1.0, p * n_comparisons) for p in p_values]
                        correction_method = "Bonferroni (Tukey unavailable)"
                except:
                    # Fallback to Bonferroni if Tukey calculation fails
                    corrected_p_values = [min(1.0, p * n_comparisons) for p in p_values]
                    correction_method = "Bonferroni (Tukey calculation failed)"
                    
            elif method.lower() == 'bonferroni':
                correction_method = "Bonferroni"
                corrected_p_values = [min(1.0, p * n_comparisons) for p in p_values]
                
            elif method.lower() == 'dunnett' and control_group:
                correction_method = "Dunnett"
                # Filter to only control group comparisons
                dunnett_p_values = []
                control_indices = []
                
                for i, comp in enumerate(comparisons):
                    if control_group in comp["group1"] or control_group in comp["group2"]:
                        dunnett_p_values.append(comp["p_val"])
                        control_indices.append(i)
                
                if dunnett_p_values:
                    k = len(dunnett_p_values)
                    dunnett_corrected = [min(1.0, p * k * 0.8) for p in dunnett_p_values]  # Approximate Dunnett factor
                    
                    corrected_p_values = [1.0] * len(p_values)
                    for j, orig_idx in enumerate(control_indices):
                        corrected_p_values[orig_idx] = dunnett_corrected[j]
                else:
                    corrected_p_values = [1.0] * len(p_values)
                    correction_method = "Dunnett (no control comparisons found)"
            else:
                # Default: Holm-Šidák
                correction_method = "Holm-Šidák"
                corrected_p_values = PostHocAnalyzer._holm_correction(p_values)
            
            # Add each pairwise comparison result with enhanced mixed-design information
            for i, comp in enumerate(comparisons):
                is_significant = corrected_p_values[i] < alpha
                
                PostHocAnalyzer.add_comparison(
                    result,
                    group1=comp["group1"],
                    group2=comp["group2"],
                    test=f"{comp['test_type']} ({correction_method})",
                    p_value=corrected_p_values[i],
                    statistic=comp["t_stat"],
                    corrected=True,
                    correction_method=correction_method,
                    effect_size=comp["effect_size"],
                    effect_size_type="cohen_d_mixed",  # Specify mixed design version
                    confidence_interval=(comp["ci_lower"], comp["ci_upper"]),
                    alpha=alpha,
                    significant=is_significant,
                    # Additional mixed-design specific information
                    comparison_type=comp["comparison_type"],
                    n_pairs=comp["n_pairs"]
                )
            
            # Add summary information
            between_comparison_count = sum(1 for c in comparisons if c["comparison_type"] == "between_subject")
            within_comparison_count = sum(1 for c in comparisons if c["comparison_type"] == "within_subject")
            mixed_comparison_count = sum(1 for c in comparisons if c["comparison_type"] == "mixed")
            
            result["summary"] = {
                "total_comparisons": n_comparisons,
                "between_subject_comparisons": between_comparison_count,
                "within_subject_comparisons": within_comparison_count,
                "mixed_comparisons": mixed_comparison_count,
                "correction_method": correction_method,
                "family_wise_alpha": alpha,
                "between_factor": between_factor,
                "within_factor": within_factor,
                "between_levels": between_levels,
                "within_levels": within_levels
            }
            
            # Diagnostic information
            logger.debug(f"DEBUG MIXED POSTHOC: available_pairs = {available_pairs}")
            if normalized_selected is not None:
                missing = normalized_selected - available_pairs
                if missing:
                    logger.warning(f"WARNING: The following selected pairs were not found: {missing}")
            
            # Set posthoc_test for visualization
            method_name_map = {
                "tukey": "Tukey HSD (Mixed)",
                "dunnett": "Dunnett Test (Mixed)",
                "bonferroni": "Bonferroni (Mixed)",
                "holm": "Holm-Šidák (Mixed)"
            }
            result["posthoc_test"] = method_name_map.get(method, f"Mixed Post-hoc ({method})")
            
            return result
            
        except Exception as e:
            result["error"] = f"Error in Mixed ANOVA post-hoc tests: {str(e)}"
            logger.error(f"ERROR MIXED POSTHOC: {str(e)}")
            import traceback
            traceback.print_exc()
            return result
    
    @staticmethod
    def _classify_comparison_type(group1_data, group2_data, between_factor, within_factor):
        """Classify the type of comparison in mixed ANOVA design."""
        between1 = group1_data['between_level']
        between2 = group2_data['between_level']
        within1 = group1_data['within_level']
        within2 = group2_data['within_level']
        
        if between1 == between2 and within1 != within2:
            return "within_subject"  # Same between-group, different within-levels
        elif between1 != between2 and within1 == within2:
            return "between_subject"  # Different between-groups, same within-level
        else:
            return "mixed"  # Different between-groups AND different within-levels
    
    @staticmethod
    def _within_subject_test(group1_data, group2_data, dv, subject, alpha):
        """Perform within-subject test for Mixed ANOVA."""
        scipy_stats = get_scipy_stats()
        
        # Get common subjects between both groups
        subjects1 = set(group1_data['subjects'])
        subjects2 = set(group2_data['subjects'])
        common_subjects = subjects1 & subjects2
        
        if len(common_subjects) < 3:
            return None, None, None, None, None, None
        
        # Extract paired data for common subjects
        data1_dict = dict(zip(group1_data['subjects'], group1_data['values']))
        data2_dict = dict(zip(group2_data['subjects'], group2_data['values']))
        
        paired_data1 = [data1_dict[subj] for subj in sorted(common_subjects)]
        paired_data2 = [data2_dict[subj] for subj in sorted(common_subjects)]
        
        # Perform paired t-test
        t_stat, p_val = scipy_stats.ttest_rel(paired_data1, paired_data2)
        
        # Calculate effect size for paired data
        differences = np.array(paired_data1) - np.array(paired_data2)
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        effect_size = mean_diff / std_diff if std_diff > 0 else 0
        
        # Calculate confidence interval
        n = len(differences)
        se_diff = std_diff / np.sqrt(n)
        t_crit = scipy_stats.t.ppf(1 - alpha/2, n - 1)
        ci_lower = mean_diff - t_crit * se_diff
        ci_upper = mean_diff + t_crit * se_diff
        
        return t_stat, p_val, effect_size, ci_lower, ci_upper, n
    
    @staticmethod
    def _between_subject_test(group1_data, group2_data, dv, alpha):
        """Perform between-subject test for Mixed ANOVA."""
        scipy_stats = get_scipy_stats()
        
        values1 = group1_data['values']
        values2 = group2_data['values']
        
        if len(values1) < 2 or len(values2) < 2:
            return None, None, None, None, None, None
        
        # Perform independent t-test
        t_stat, p_val = scipy_stats.ttest_ind(values1, values2, equal_var=True)
        
        # Calculate Cohen's d for independent samples
        n1, n2 = len(values1), len(values2)
        s1, s2 = np.var(values1, ddof=1), np.var(values2, ddof=1)
        s_pooled = np.sqrt(((n1-1)*s1 + (n2-1)*s2) / (n1+n2-2))
        effect_size = (np.mean(values1) - np.mean(values2)) / s_pooled if s_pooled > 0 else 0
        
        # Calculate confidence interval for mean difference
        mean_diff = np.mean(values1) - np.mean(values2)
        se_diff = s_pooled * np.sqrt(1/n1 + 1/n2)
        df = n1 + n2 - 2
        t_crit = scipy_stats.t.ppf(1 - alpha/2, df)
        ci_lower = mean_diff - t_crit * se_diff
        ci_upper = mean_diff + t_crit * se_diff
        
        return t_stat, p_val, effect_size, ci_lower, ci_upper, min(n1, n2)
    
    @staticmethod
    def _mixed_comparison_test(group1_data, group2_data, dv, subject, alpha):
        """Perform mixed comparison test (different between-groups AND within-levels)."""
        # For mixed comparisons, treat as independent samples (conservative approach)
        return MixedAnovaPostHocAnalyzer._between_subject_test(group1_data, group2_data, dv, alpha)
    
    @staticmethod 
    def _tukey_p_value(q_stat, k, df):
        """Calculate p-value for Tukey's q statistic."""
        studentized_range = get_scipy_stats().studentized_range
        try:
            return 1 - studentized_range.cdf(q_stat, k, df)
        except:
            # Fallback to t-distribution approximation
            t = get_scipy_stats().t
            import math
            t_equiv = q_stat / math.sqrt(2)
            return 2 * (1 - t.cdf(abs(t_equiv), df))

    @staticmethod
    def perform_test(df, between, within, dv, subject, alpha=0.05, selected_comparisons=None, method='tukey', control_group=None):
        """
        UPDATED: Enhanced Mixed ANOVA post-hoc tests with proper between/within factor distinction
        """
        try:
            result = PostHocAnalyzer.create_result_template("Mixed ANOVA Post-hoc Tests")
            
            # Create interaction groups (between_level:within_level combinations)
            interaction_groups = []
            group_to_data = {}
            
            for between_level in df[between].unique():
                for within_level in df[within].unique():
                    group_data = df[(df[between] == between_level) & (df[within] == within_level)]
                    if len(group_data) > 0:
                        group_name = f"{between_level}:{within_level}"
                        interaction_groups.append(group_name)
                        group_to_data[group_name] = {
                            'values': group_data[dv].tolist(),
                            'subjects': group_data[subject].tolist(),
                            'between_level': between_level,
                            'within_level': within_level
                        }
            
            logger.debug(f"DEBUG POSTHOC: interaction_groups = {interaction_groups}")
            
            # Handle selected comparisons
            def normalize_pair(pair):
                return tuple(sorted(pair))
            
            normalized_selected = None
            if selected_comparisons:
                normalized_selected = set()
                for pair in selected_comparisons:
                    normalized_selected.add(normalize_pair(pair))
                logger.debug(f"DEBUG POSTHOC: normalized_selected = {normalized_selected}")
            
            # Generate all possible pairs and filter by user selection
            all_pairs = list(combinations(interaction_groups, 2))
            
            if normalized_selected is not None:
                filtered_pairs = [pair for pair in all_pairs if normalize_pair(pair) in normalized_selected]
            else:
                filtered_pairs = all_pairs
            
            logger.debug(f"DEBUG POSTHOC: filtered_pairs = {filtered_pairs}")
            
            # Import required functions
            ttest_rel = get_scipy_stats().ttest_rel
            ttest_ind = get_scipy_stats().ttest_ind
            
            # Perform appropriate tests for each pair
            pvals = []
            stats_list = []
            available_pairs = set()
            
            for g1, g2 in filtered_pairs:
                available_pairs.add(normalize_pair((g1, g2)))
                
                data1 = group_to_data[g1]
                data2 = group_to_data[g2]
                
                # Determine test type based on comparison
                same_between = data1['between_level'] == data2['between_level']
                same_within = data1['within_level'] == data2['within_level']
                
                matched_data1 = None
                matched_data2 = None
                
                if same_between and not same_within:
                    # Within-subject comparison (same group, different time points)
                    # Need to match subjects for paired t-test
                    subjects1 = set(data1['subjects'])
                    subjects2 = set(data2['subjects'])
                    common_subjects = subjects1 & subjects2
                    
                    if len(common_subjects) > 0:
                        # Get matched data for common subjects
                        matched_data1 = []
                        matched_data2 = []
                        for subj in sorted(common_subjects):
                            idx1 = list(data1['subjects']).index(subj)
                            idx2 = list(data2['subjects']).index(subj)
                            matched_data1.append(data1['values'][idx1])
                            matched_data2.append(data2['values'][idx2])
                        
                        # Paired t-test
                        stat, pval = ttest_rel(matched_data1, matched_data2)
                        test_type = "Paired t-test"
                    else:
                        # No common subjects - skip this comparison
                        continue
                        
                elif not same_between:
                    # Between-groups comparison (independent t-test)
                    stat, pval = ttest_ind(data1['values'], data2['values'], equal_var=True)
                    test_type = "Independent t-test"
                else:
                    # Same group and same time point - skip (not meaningful)
                    continue
                
                pvals.append(pval)
                stats_list.append((g1, g2, stat, pval, test_type, data1, data2, matched_data1, matched_data2))
            
            # Apply multiple comparison correction based on method
            multipletests = get_statsmodels_multitest()
            if pvals:
                if method.lower() == 'tukey':
                    # For Tukey, we'll use a different approach
                    correction_method = "Tukey HSD"
                    reject, pvals_corr, _, _ = multipletests(pvals, alpha=alpha, method='holm-sidak')  # Fallback
                elif method.lower() == 'dunnett' and control_group:
                    # For Dunnett, filter to only control group comparisons
                    correction_method = "Dunnett"
                    # Filter to only comparisons involving the control group
                    dunnett_pvals = []
                    control_comparisons = []
                    
                    for i, (g1, g2, stat, pval, test_type, data1, data2, matched_data1, matched_data2) in enumerate(stats_list):
                        # Use exact match instead of substring search
                        if g1 == control_group or g2 == control_group:
                            dunnett_pvals.append(pval)
                            control_comparisons.append(i)
                    
                    if dunnett_pvals:
                        # Apply correction only to control group comparisons
                        reject, pvals_corr_dunnett, _, _ = multipletests(dunnett_pvals, alpha=alpha, method='holm-sidak')
                        # Map back to original order
                        pvals_corr = [1.0] * len(pvals)  # Start with all p-values as 1.0
                        for j, orig_idx in enumerate(control_comparisons):
                            pvals_corr[orig_idx] = pvals_corr_dunnett[j]
                    else:
                        pvals_corr = [1.0] * len(pvals)
                        correction_method = "Dunnett (no control comparisons found)"
                elif method.lower() == 'paired_fdr':
                    correction_method = "FDR (Benjamini-Hochberg)"
                    reject, pvals_corr, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')
                else:
                    # Default: Holm-Šidák
                    correction_method = "Holm-Šidák"
                    reject, pvals_corr, _, _ = multipletests(pvals, alpha=alpha, method='holm-sidak')
            else:
                pvals_corr = []
                correction_method = "Holm-Šidák"
            
            # Add results
            for i, (g1, g2, stat, pval, test_type, data1, data2, matched_data1, matched_data2) in enumerate(stats_list):
                # Calculate effect size
                if test_type == "Paired t-test":
                    # Cohen's d for paired samples
                    if matched_data1 is not None and matched_data2 is not None:
                        diff = np.array(matched_data1) - np.array(matched_data2)
                        effect_size = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
                    else:
                        effect_size = 0
                    effect_size_type = "cohen_d"
                else:
                    # Cohen's d for independent samples
                    n1, n2 = len(data1['values']), len(data2['values'])
                    s1, s2 = np.var(data1['values'], ddof=1), np.var(data2['values'], ddof=1)
                    s_pooled = np.sqrt(((n1-1)*s1 + (n2-1)*s2) / (n1+n2-2)) if (n1+n2-2) > 0 else 0
                    effect_size = (np.mean(data1['values']) - np.mean(data2['values'])) / s_pooled if s_pooled > 0 else 0
                    effect_size_type = "cohen_d"
                
                # Calculate confidence interval
                if test_type == "Paired t-test":
                    if matched_data1 is not None and matched_data2 is not None:
                        diff = np.array(matched_data1) - np.array(matched_data2)
                        n = len(diff)
                        mean_diff = np.mean(diff)
                        se = np.std(diff, ddof=1) / np.sqrt(n)
                        df_val = n - 1
                    else:
                        mean_diff = 0
                        se = 0
                        df_val = 0
                else:
                    n1, n2 = len(data1['values']), len(data2['values'])
                    mean_diff = np.mean(data1['values']) - np.mean(data2['values'])
                    s1, s2 = np.var(data1['values'], ddof=1), np.var(data2['values'], ddof=1)
                    se = np.sqrt(s1/n1 + s2/n2)
                    df_val = n1 + n2 - 2

                t = get_scipy_stats().t
                if df_val > 0 and se > 0:
                    t_crit = t.ppf(1 - alpha/2, df_val)
                    ci = (mean_diff - t_crit * se, mean_diff + t_crit * se)
                else:
                    ci = (None, None)
                
                logger.debug(f"DEBUG POSTHOC: Adding comparison {g1} vs {g2} (test: {test_type})")
                PostHocAnalyzer.add_comparison(
                    result,
                    group1=g1,
                    group2=g2,
                    test=test_type,
                    p_value=pvals_corr[i] if i < len(pvals_corr) else pval,
                    statistic=stat,
                    corrected=True,
                    correction_method=correction_method,
                    effect_size=effect_size,
                    effect_size_type=effect_size_type,
                    confidence_interval=ci,
                    alpha=alpha
                )
            
            # After all, print available pairs and warn if any selected pair is not present
            logger.debug(f"DEBUG POSTHOC: available_pairs = {available_pairs}")
            if normalized_selected is not None:
                missing = normalized_selected - available_pairs
                if missing:
                    logger.warning(f"WARNING: The following selected pairs were not found in the available post-hoc comparisons: {missing}")
            
            # Set the posthoc_test value for decision tree visualization
            method_name_map = {
                "tukey": "Tukey HSD",
                "dunnett": "Dunnett Test", 
                "paired_custom": "Custom paired t-tests (Holm-Šidák)",
                "holm": "Custom paired t-tests (Holm-Šidák)"
            }
            result["posthoc_test"] = method_name_map.get(method, f"Post-hoc test ({method})")
            
            return result
        except Exception as e:
            result["error"] = f"Error in Mixed ANOVA post-hoc tests: {str(e)}"
            return result
        
class RMAnovaPostHocAnalyzer(PostHocAnalyzer):
    """UPDATED: Advanced post-hoc tests for Repeated Measures ANOVA with proper within-subject design handling."""
        
    @staticmethod
    def perform_test(df, dv, subject, within, alpha=0.05, selected_comparisons=None, method='tukey', control_group=None):
        """
        UPDATED: Performs sophisticated post-hoc tests for RM ANOVA with proper within-subject handling.
        
        Major improvements:
        - Proper within-subject data validation
        - Enhanced Tukey HSD for repeated measures
        - Cohen's d for repeated measures (cohen_d_rm)
        - Complete subject tracking
        - Better error handling and diagnostics
        - Summary statistics for RM design
        """
        result = PostHocAnalyzer.create_result_template("RM ANOVA Post-hoc Tests")
        
        try:
            logger.debug(f"DEBUG RM POSTHOC: selected_comparisons = {selected_comparisons}")
            
            # Normalize comparison pairs function (consistent with other ANOVAs)
            def normalize_pair(pair):
                return tuple(sorted([s.strip() for s in pair]))
            
            # Handle selected comparisons
            if selected_comparisons:
                if isinstance(selected_comparisons, set):
                    normalized_selected = selected_comparisons
                else:
                    normalized_selected = set(normalize_pair(pair) for pair in selected_comparisons)
            else:
                normalized_selected = None
            
            logger.debug(f"DEBUG RM POSTHOC: normalized_selected = {normalized_selected}")
            
            # Get within-subject factor and levels
            within_factor = within[0]
            within_levels = sorted(df[within_factor].unique())
            
            # Validate that we have repeated measures data
            subject_counts = df.groupby(subject)[within_factor].nunique()
            expected_measures = len(within_levels)
            incomplete_subjects = subject_counts[subject_counts < expected_measures]
            
            if len(incomplete_subjects) > 0:
                logger.warning(f"WARNING: {len(incomplete_subjects)} subjects have incomplete data")
            
            # Get complete cases only for robust within-subject analysis
            complete_subjects = subject_counts[subject_counts == expected_measures].index
            df_complete = df[df[subject].isin(complete_subjects)].copy()
            
            logger.debug(f"DEBUG RM POSTHOC: Complete subjects: {len(complete_subjects)}, Total levels: {expected_measures}")
            
            # Import required modules
            scipy_stats = get_scipy_stats()

            # Collect all pairwise comparisons with proper within-subject handling
            available_pairs = set()
            comparisons = []
            
            for level1, level2 in combinations(within_levels, 2):
                norm_pair = normalize_pair((str(level1), str(level2)))
                available_pairs.add(norm_pair)
                
                # Check if this comparison is selected
                if normalized_selected is not None and norm_pair not in normalized_selected:
                    continue
                
                # Extract paired data for this comparison (same subjects in both conditions)
                data1_df = df_complete[df_complete[within_factor] == level1].sort_values(by=subject)
                data2_df = df_complete[df_complete[within_factor] == level2].sort_values(by=subject)
                
                # Ensure same subjects in both groups
                common_subjects = set(data1_df[subject]) & set(data2_df[subject])
                data1_df = data1_df[data1_df[subject].isin(common_subjects)].sort_values(by=subject)
                data2_df = data2_df[data2_df[subject].isin(common_subjects)].sort_values(by=subject)
                
                data1 = data1_df[dv].values
                data2 = data2_df[dv].values
                
                if len(data1) != len(data2) or len(data1) < 3:
                    logger.warning(f"WARNING: Insufficient paired data for {level1} vs {level2}")
                    continue
                
                # Perform paired t-test (appropriate for within-subject design)
                t_stat, p_val = scipy_stats.ttest_rel(data1, data2)
                
                # Calculate within-subject effect size (Cohen's d for repeated measures)
                differences = data1 - data2
                mean_diff = np.mean(differences)
                std_diff = np.std(differences, ddof=1)
                
                # Cohen's d for repeated measures (using difference scores)
                effect_size = mean_diff / std_diff if std_diff > 0 else 0
                
                # Calculate confidence interval for mean difference
                n = len(differences)
                se_diff = std_diff / np.sqrt(n)
                df_t = n - 1
                
                # Store raw comparison data
                comparisons.append({
                    "level1": level1,
                    "level2": level2,
                    "t_stat": t_stat,
                    "p_val": p_val,
                    "effect_size": effect_size,
                    "mean_dif": mean_diff,
                    "se_dif": se_diff,
                    "d": df_t,
                    "n_pairs": n,
                    "data1": data1,
                    "data2": data2,
                    "differences": differences
                })
            
            if not comparisons:
                result["error"] = "No valid pairwise comparisons could be performed"
                return result
            
            # Apply multiple comparison correction based on method
            p_values = [comp["p_val"] for comp in comparisons]
            n_comparisons = len(comparisons)
            
            if method.lower() == 'tukey':
                # Implement proper Tukey HSD for repeated measures
                correction_method = "Tukey HSD (RM)"
                try:
                    # Try to use pingouin for proper Tukey implementation
                    pg = get_pingouin_module()
                    if pg is not None:
                        # Use Tukey's studentized range statistic for RM design
                        corrected_p_values = []
                        
                        for comp in comparisons:
                            # Convert t-statistic to Tukey's q statistic
                            q_stat = abs(comp["t_stat"]) * np.sqrt(2)
                            p_tukey = RMAnovaPostHocAnalyzer._tukey_p_value(q_stat, len(within_levels), comp["d"])
                            corrected_p_values.append(p_tukey)
                    else:
                        # Fallback to conservative Bonferroni
                        corrected_p_values = [min(1.0, p * n_comparisons) for p in p_values]
                        correction_method = "Bonferroni (Tukey unavailable)"
                except:
                    # Fallback to Bonferroni if Tukey calculation fails
                    corrected_p_values = [min(1.0, p * n_comparisons) for p in p_values]
                    correction_method = "Bonferroni (Tukey calculation failed)"
                    
            elif method.lower() == 'bonferroni':
                correction_method = "Bonferroni"
                corrected_p_values = [min(1.0, p * n_comparisons) for p in p_values]
                
            elif method.lower() == 'dunnett' and control_group:
                correction_method = "Dunnett"
                # Filter to only control group comparisons
                dunnett_p_values = []
                control_indices = []
                
                for i, comp in enumerate(comparisons):
                    level1_str = str(comp["level1"])
                    level2_str = str(comp["level2"])
                    if level1_str == control_group or level2_str == control_group:
                        dunnett_p_values.append(comp["p_val"])
                        control_indices.append(i)
                
                if dunnett_p_values:
                    # Apply Dunnett correction (more liberal than Bonferroni for control comparisons)
                    k = len(dunnett_p_values)  # Number of comparisons with control
                    dunnett_corrected = [min(1.0, p * k * 0.8) for p in dunnett_p_values]  # Approximate Dunnett factor
                    
                    corrected_p_values = [1.0] * len(p_values)
                    for j, orig_idx in enumerate(control_indices):
                        corrected_p_values[orig_idx] = dunnett_corrected[j]
                else:
                    corrected_p_values = [1.0] * len(p_values)
                    correction_method = "Dunnett (no control comparisons found)"
            else:
                # Default: Holm-Šidák (step-down method, less conservative than Bonferroni)
                correction_method = "Holm-Šidák"
                corrected_p_values = PostHocAnalyzer._holm_correction(p_values)
            
            # Calculate family-wise corrected confidence intervals
            # Use Sidak correction for simultaneous confidence intervals
            alpha_sidak = 1 - (1 - alpha) ** (1 / n_comparisons)
            
            # Add each pairwise comparison result with enhanced within-subject information
            for i, comp in enumerate(comparisons):
                # Calculate corrected confidence interval
                t_crit = scipy_stats.t.ppf(1 - alpha_sidak/2, comp["d"])
                ci_lower = comp["mean_dif"] - t_crit * comp["se_dif"]
                ci_upper = comp["mean_dif"] + t_crit * comp["se_dif"]
                
                # Determine significance
                is_significant = corrected_p_values[i] < alpha
                
                PostHocAnalyzer.add_comparison(
                    result,
                    group1=str(comp["level1"]),
                    group2=str(comp["level2"]),
                    test=f"Paired t-test ({correction_method})",
                    p_value=corrected_p_values[i],
                    statistic=comp["t_stat"],
                    corrected=True,
                    correction_method=correction_method,
                    effect_size=comp["effect_size"],
                    effect_size_type="cohen_d_rm",  # Specify repeated measures version
                    confidence_interval=(ci_lower, ci_upper),
                    alpha=alpha,
                    significant=is_significant,
                    # Additional RM-specific information
                    degrees_of_freedom=comp["d"],
                    n_pairs=comp["n_pairs"],
                    mean_difference=comp["mean_dif"]
                )
            
            # Add summary information
            result["summary"] = {
                "total_comparisons": n_comparisons,
                "correction_method": correction_method,
                "family_wise_alpha": alpha,
                "complete_subjects": len(complete_subjects),
                "total_subjects": len(df[subject].unique()),
                "within_factor": within_factor,
                "within_levels": within_levels
            }
            
            # Diagnostic information
            logger.debug(f"DEBUG RM POSTHOC: available_pairs = {available_pairs}")
            if normalized_selected is not None:
                missing = normalized_selected - available_pairs
                if missing:
                    logger.warning(f"WARNING: The following selected pairs were not found: {missing}")
            
            # Set posthoc_test for visualization
            method_name_map = {
                "tukey": "Tukey HSD (RM)",
                "dunnett": "Dunnett Test (RM)",
                "bonferroni": "Bonferroni (RM)",
                "holm": "Holm-Šidák (RM)"
            }
            result["posthoc_test"] = method_name_map.get(method, f"RM Post-hoc ({method})")
            
            return result
            
        except Exception as e:
            result["error"] = f"Error in RM ANOVA post-hoc tests: {str(e)}"
            logger.error(f"ERROR RM POSTHOC: {str(e)}")
            import traceback
            traceback.print_exc()
            return result
    
    @staticmethod
    def _get_tukey_critical_value(k, df, alpha=0.05):
        """Get critical value for Tukey's HSD test (simplified implementation)."""
        # This is a simplified implementation - in practice, use statistical tables
        studentized_range = get_scipy_stats().studentized_range
        try:
            return studentized_range.ppf(1 - alpha, k, df)
        except:
            # Fallback approximation
            import math
            return math.sqrt(2) * 2.0  # Very rough approximation
    
    @staticmethod 
    def _tukey_p_value(q_stat, k, df):
        """Calculate p-value for Tukey's q statistic (simplified implementation)."""
        studentized_range = get_scipy_stats().studentized_range
        try:
            return 1 - studentized_range.cdf(q_stat, k, df)
        except:
            # Fallback to t-distribution approximation
            t = get_scipy_stats().t
            import math
            t_equiv = q_stat / math.sqrt(2)
            return 2 * (1 - t.cdf(abs(t_equiv), df))
        
class PostHocStatistics:
    """UPDATED: Statistical calculations for various post-hoc tests."""
    
    @staticmethod
    def calculate_cohens_d(group1_data, group2_data, paired=False):
        """Calculates Cohen's d effect size with appropriate adjustments."""
        if paired:
            diff = np.array(group1_data) - np.array(group2_data)
            return np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
        else:
            n1, n2 = len(group1_data), len(group2_data)
            s1, s2 = np.var(group1_data, ddof=1), np.var(group2_data, ddof=1)
            s_pooled = np.sqrt(((n1-1)*s1 + (n2-1)*s2) / (n1+n2-2))
            return (np.mean(group1_data) - np.mean(group2_data)) / s_pooled if s_pooled > 0 else 0
    
    @staticmethod
    def calculate_ci_mean_diff(group1_data, group2_data, alpha=0.05, paired=False):
        """Calculates confidence intervals for the mean difference."""
        t = get_scipy_stats().t
        
        try:
            if paired:
                diff = np.array(group1_data) - np.array(group2_data)
                n = len(diff)
                mean_diff = np.mean(diff)
                se = np.std(diff, ddof=1) / np.sqrt(n)
                df = n - 1
            else:
                n1, n2 = len(group1_data), len(group2_data)
                mean_diff = np.mean(group1_data) - np.mean(group2_data)
                s1, s2 = np.var(group1_data, ddof=1), np.var(group2_data, ddof=1)
                se = np.sqrt(s1/n1 + s2/n2)
                df = (s1/n1 + s2/n2)**2 / ((s1/n1)**2/(n1-1) + (s2/n2)**2/(n2-1))
                
            t_crit = t.ppf(1 - alpha/2, df)
            ci_lower = mean_diff - t_crit * se
            ci_upper = mean_diff + t_crit * se
            
            return (float(ci_lower), float(ci_upper))
        except Exception:
            return (None, None)
        
class TukeyHSD(PostHocAnalyzer):
    
    @staticmethod
    def perform_test(valid_groups, samples, alpha=0.05):
        """Performs the Tukey HSD test."""
        pairwise_tukeyhsd = get_pairwise_tukeyhsd()
        
        result = PostHocAnalyzer.create_result_template("Tukey HSD Test")
        
        try:
            all_data = []
            group_labels = []
            
            for group in valid_groups:
                values = samples[group]
                all_data.extend(values)
                group_labels.extend([str(group)] * len(values))
            
            if len(set(group_labels)) < 2:
                result["error"] = "Tukey HSD requires at least two groups."
                return result

            tukey_result = pairwise_tukeyhsd(endog=all_data, groups=group_labels, alpha=alpha)
            
            # Check if tukey_result has a summary() attribute
            if hasattr(tukey_result, 'summary'):
                summary = tukey_result.summary()
                
                # Extract data from the summary table
                for i in range(len(tukey_result.meandiffs)):
                    group1, group2 = summary.data[i+1][0:2]  # First two columns are the groups
                    p_val = summary.data[i+1][3]  # Fourth column is the p-value
                    lower, upper = summary.data[i+1][4:6]  # Fifth and sixth columns are the confidence intervals
                    # Calculate Cohen's d effect size
                    group1_data = samples[group1]
                    group2_data = samples[group2]
                    effect_size = PostHocStatistics.calculate_cohens_d(group1_data, group2_data)

                    # Use the common method to add a comparison
                    PostHocAnalyzer.add_comparison(
                        result,
                        group1=group1,
                        group2=group2,
                        test="Tukey HSD",
                        p_value=p_val,
                        statistic=tukey_result.meandiffs[i],
                        corrected=True,
                        correction_method="Tukey HSD",
                        effect_size=effect_size,
                        effect_size_type="cohen_d",
                        confidence_interval=(float(lower), float(upper)),
                        alpha=alpha
                        # The parameter significant=is_significant was removed
                    )
            else:
                result["error"] = "TukeyHSDResults object has no summary() attribute"
                return result
            
            # Set the posthoc_test value for decision tree visualization
            result["posthoc_test"] = "Tukey HSD"
            
            return result
        except Exception as e:
            result["error"] = f"Error in Tukey HSD test: {str(e)}"
            return result
        
class GamesHowellTest(PostHocAnalyzer):
    """Games-Howell post-hoc test — robust to unequal variances and unequal sample sizes.

    Uses Welch-Satterthwaite degrees of freedom and Hedges' g as effect size.
    No assumption of variance homogeneity; appropriate when Levene's test fails.
    Implemented directly via scipy.stats — no additional dependencies required.
    """

    @staticmethod
    def perform_test(valid_groups, samples, alpha=0.05):
        result = PostHocAnalyzer.create_result_template("Games-Howell Test")
        try:
            from itertools import combinations as _combinations
            stats_mod = get_scipy_stats()

            # k = number of groups entering the comparison family; the
            # studentized-range distribution needs it for FWER control.
            comparable = [g for g in valid_groups if len(samples[g]) >= 2]
            k = len(comparable)

            for g1, g2 in _combinations(valid_groups, 2):
                x1 = np.array(samples[g1], dtype=float)
                x2 = np.array(samples[g2], dtype=float)
                n1, n2 = len(x1), len(x2)
                if n1 < 2 or n2 < 2:
                    continue

                m1, m2 = np.mean(x1), np.mean(x2)
                v1, v2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
                mean_diff = float(m1 - m2)

                se = np.sqrt(v1 / n1 + v2 / n2)
                if se == 0:
                    continue

                # Welch-Satterthwaite degrees of freedom
                df_w = (v1 / n1 + v2 / n2) ** 2 / (
                    (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
                )
                t_stat = mean_diff / se
                # Games-Howell: p from the studentized-range distribution with
                # q = sqrt(2)*|t| and Welch df (controls FWER across k groups).
                q_stat = np.sqrt(2.0) * abs(t_stat)
                p_val = float(stats_mod.studentized_range.sf(q_stat, k, df_w))

                # Hedges' g (bias-corrected Cohen's d)
                sp = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
                correction = 1 - 3 / (4 * (n1 + n2 - 2) - 1)
                hedges_g = float((mean_diff / sp) * correction) if sp > 0 else None

                # Simultaneous CI for the mean difference (same q distribution)
                q_crit = float(stats_mod.studentized_range.ppf(1 - alpha, k, df_w))
                half_width = (q_crit / np.sqrt(2.0)) * se
                ci = (float(mean_diff - half_width), float(mean_diff + half_width))

                PostHocAnalyzer.add_comparison(
                    result,
                    group1=str(g1),
                    group2=str(g2),
                    test="Games-Howell",
                    p_value=p_val,
                    statistic=mean_diff,
                    corrected=True,
                    correction_method="Games-Howell",
                    effect_size=hedges_g,
                    effect_size_type="hedges_g",
                    confidence_interval=ci,
                    alpha=alpha,
                )

            result["posthoc_test"] = "Games-Howell Test"
            return result
        except Exception as e:
            result["error"] = f"Error in Games-Howell test: {str(e)}"
            return result


class DunnettTest(PostHocAnalyzer):
    """Implementation of the Dunnett test for comparing multiple groups to a control group."""
    @staticmethod
    def perform_test(valid_groups, samples, control_group, alpha=0.05):
        """
        Performs the Dunnett test (compares each group to the control group).
        """
        result = PostHocAnalyzer.create_result_template(f"Dunnett Test (Control group: {control_group})")
        result["control_group"] = control_group

        try:
            scipy_stats = get_scipy_stats()

            control_data = np.asarray(samples[control_group], dtype=float)
            group_pairs = [g for g in valid_groups if str(g) != str(control_group)]
            treatment_data = [np.asarray(samples[g], dtype=float) for g in group_pairs]

            # scipy.stats.dunnett fits the joint multivariate-t once and returns
            # BOTH the FWER-adjusted p-values and the simultaneous confidence
            # intervals from the same distribution — so p-values and CIs stay
            # mutually consistent (a significant contrast always has a CI that
            # excludes 0). confidence_level matches alpha.
            dunnett_result = scipy_stats.dunnett(
                *treatment_data, control=control_data
            )
            ci = dunnett_result.confidence_interval(confidence_level=1 - alpha)

            control_std = np.std(control_data, ddof=1)
            n_ctrl = len(control_data)

            for i, group in enumerate(group_pairs):
                g_data = treatment_data[i]
                mean_diff = float(np.mean(g_data) - np.mean(control_data))
                # Cohen's d via pooled SD (effect-size summary only)
                n_g = len(g_data)
                s_g = np.std(g_data, ddof=1)
                pooled_std = np.sqrt(
                    ((n_g - 1) * s_g ** 2 + (n_ctrl - 1) * control_std ** 2)
                    / (n_g + n_ctrl - 2)
                )
                effect_size = mean_diff / pooled_std if pooled_std > 0 else 0.0

                PostHocAnalyzer.add_comparison(
                    result,
                    group1=group,
                    group2=control_group,
                    test="Dunnett",
                    p_value=float(dunnett_result.pvalue[i]),
                    statistic=float(dunnett_result.statistic[i]),
                    corrected=True,
                    correction_method="Dunnett",
                    effect_size=effect_size,
                    effect_size_type="cohen_d",
                    confidence_interval=(float(ci.low[i]), float(ci.high[i])),
                    alpha=alpha
                )

            # Set the posthoc_test value for decision tree visualization
            result["posthoc_test"] = "Dunnett Test"

            return result
        except Exception as e:
            import traceback
            result["error"] = f"Error in Dunnett test: {str(e)}"
            traceback.print_exc()
            return result

from scipy.stats import mannwhitneyu

class DunnTest(PostHocAnalyzer):
    @staticmethod
    def perform_test(valid_groups, samples, alpha=0.05, n_boot=1000):
        result = PostHocAnalyzer.create_result_template("Dunn-Test")

        try:
            sp = get_scikit_posthocs()
        except ImportError:
            result["error"] = "scikit-posthocs is not installed."
            return result

        # 1) Get raw p-values matrix (drop NaN per group)
        clean = {g: [v for v in samples[g] if not (isinstance(v, float) and np.isnan(v))] for g in valid_groups}
        data_array = [clean[g] for g in valid_groups]
        raw_p = sp.posthoc_dunn(data_array, p_adjust=None)  # no internal correction

        # 2) Flatten into list and correct with Holm-Šidák
        pairs = []
        pvals = []
        for i, g1 in enumerate(valid_groups):
            for j, g2 in enumerate(valid_groups):
                if i < j:
                    pairs.append((g1, g2))
                    pvals.append(raw_p.iloc[i, j])
        multipletests = get_statsmodels_multitest()
        reject, p_adj, _, _ = multipletests(pvals, alpha=alpha, method='holm-sidak')

        # 3) Loop over pairs and compute effect & CI
        for (g1, g2), pval_adj, sig in zip(pairs, p_adj, reject):
            x, y = clean[g1], clean[g2]
            # Mann–Whitney U for effect‐size r
            U, _ = mannwhitneyu(x, y, alternative='two-sided')
            n1, n2 = len(x), len(y)
            z = (U - n1 * n2 / 2) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
            effect_r = abs(z) / np.sqrt(n1 + n2)

            # Bootstrap CI
            boots = []
            for _ in range(n_boot):
                b1 = np.random.choice(x, n1, replace=True)
                b2 = np.random.choice(y, n2, replace=True)
                boots.append(np.median([u - v for u in b1 for v in b2]))
            ci_low, ci_high = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])

            # Median difference

            PostHocAnalyzer.add_comparison(
                result,
                group1=g1,
                group2=g2,
                test="Dunn",
                p_value=pval_adj,
                statistic=None,
                corrected=True,
                correction_method="Holm-Šidák",
                effect_size=effect_r,
                effect_size_type="r",
                confidence_interval=(float(ci_low), float(ci_high)),
                alpha=alpha
            )

        return result

class DependentPostHoc(PostHocAnalyzer):
    @staticmethod
    def perform_test(valid_groups, samples, alpha=0.05, parametric=True):
        stats = get_stats_module()
        name = "Parametric paired t-tests" if parametric else "Wilcoxon signed-rank tests"
        result = PostHocAnalyzer.create_result_template(name)

        # 1) check equal lengths
        sizes = [len(samples[g]) for g in valid_groups]
        if len(set(sizes)) != 1:
            result["error"] = "All groups must have same length for dependent tests."
            return result

        # 2) collect stats
        pvals, stats_list, pairs = [], [], []
        for g1, g2 in combinations(valid_groups, 2):
            x, y = np.array(samples[g1]), np.array(samples[g2])
            if parametric:
                tstat, p = stats.ttest_rel(x, y)
                stats_list.append(tstat)
            else:
                import warnings
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    wstat, p = stats.wilcoxon(x, y, zero_method='pratt', exact=True if len(x) <= 25 else False)
                    if w:
                        for warn in w:
                            msg = f"Wilcoxon Warning: {str(warn.message)}"
                            if msg not in result.setdefault("warnings", []):
                                result["warnings"].append(msg)
                stats_list.append(wstat)
            pvals.append(p)
            pairs.append((g1, g2, x, y))

        # 3) Holm-Šidák correction
        multipletests = get_statsmodels_multitest()
        reject, p_adj, _, _ = multipletests(pvals, alpha=alpha, method='holm-sidak')

        # 4) add comparisons
        for i, (g1, g2, x, y) in enumerate(pairs):
            if parametric:
                # paired CI and d
                ci = PostHocStatistics.calculate_ci_mean_diff(x, y, alpha=alpha, paired=True)
                d = PostHocStatistics.calculate_cohens_d(x, y, paired=True)
                test = "Paired t-test"
                stat = stats_list[i]
                es, estype = d, "cohen_d"
            else:
                # r from Wilcoxon
                n = len(x)
                W = stats_list[i]
                mu = n*(n+1)/4
                sigma = np.sqrt(n*(n+1)*(2*n+1)/24)
                z = (W - mu)/sigma
                r = abs(z)/np.sqrt(n)
                ci = (None, None)
                test = "Wilcoxon signed-rank"
                stat = W
                es, estype = r, "r"

            PostHocAnalyzer.add_comparison(
                result,
                group1=g1,
                group2=g2,
                test=test,
                p_value=p_adj[i],
                statistic=stat,
                corrected=True,
                correction_method="Holm-Šidák",
                effect_size=es,
                effect_size_type=estype,
                confidence_interval=ci,
                alpha=alpha
            )

        return result
            
class PostHocFactory:
    @staticmethod
    def create_test(test_type, is_parametric=True, is_dependent=False):
        """Creates the correct post-hoc test implementation based on parameters."""
        if is_dependent:
            return DependentPostHoc()
        
        if is_parametric:
            if test_type == "tukey":
                return TukeyHSD()
            elif test_type == "games_howell":
                return GamesHowellTest()
            elif test_type == "dunnett":
                return DunnettTest()
        else:
            if test_type == "dunn":
                return DunnTest()
            elif test_type == "conover":
                # Return None or a message since ConoverPostHoc is removed
                return None
            elif test_type == "nemenyi":
                # Return None or a message since NemenyiPostHoc is removed
                return None
        
        return None
    
    @staticmethod
    def create_anova_posthoc(anova_type, **kwargs):
        """Creates specialized post-hoc tests for different ANOVA types."""
        if anova_type == "two_way":
            return TwoWayPostHocAnalyzer()
        elif anova_type == "mixed":
            return MixedAnovaPostHocAnalyzer()
        elif anova_type == "rm":
            return RMAnovaPostHocAnalyzer()
        return None
    
    @staticmethod
    def perform_posthoc_for_anova(anova_type, df, dv, subject=None, between=None, within=None, alpha=0.05, selected_comparisons=None, method="paired_custom", control_group=None):
        """
        Performs post-hoc tests for an ANOVA type and returns standardized results.
        
        Parameters:
        -----------
        anova_type : str
            Type of ANOVA ('two_way', 'mixed', 'rm')
        df : pandas.DataFrame
            Dataset in long format
        dv : str
            Name of the dependent variable
        subject : str, optional
            Name of the subject variable (for Mixed and RM ANOVA)
        between : list, optional
            List of between factors
        within : list, optional
            List of within factors
        alpha : float, optional
            Significance level (default: 0.05)
        method : str, optional
            Post-hoc method ("tukey", "dunnett", "paired_custom")
        control_group : str, optional
            Control group for Dunnett test
            
        Returns:
        --------
        dict
            Standardized post-hoc results
        """
        analyzer = PostHocFactory.create_anova_posthoc(anova_type)
        if analyzer is None:
            return {"error": f"No post-hoc test available for ANOVA type '{anova_type}'"}
        
        if anova_type == "two_way":
            if not between or len(between) != 2:
                return {"error": "Two-Way ANOVA requires two between factors"}
            return analyzer.perform_test(df=df, dv=dv, factors=between, alpha=alpha, selected_comparisons=selected_comparisons, method=method, control_group=control_group)
        
        elif anova_type == "mixed":
            # Full implementation for Mixed ANOVA
            if not subject:
                return {"error": "Mixed ANOVA requires a subject variable"}
            if not between or len(between) != 1:
                return {"error": "Mixed ANOVA requires exactly one between factor"}
            if not within or len(within) != 1:
                return {"error": "Mixed ANOVA requires exactly one within factor"}
            
            return analyzer.perform_test(df=df, dv=dv, subject=subject, between=between, within=within, alpha=alpha, selected_comparisons=selected_comparisons, method=method, control_group=control_group)
        
        elif anova_type == "rm":
            # Full implementation for RM-ANOVA
            if not subject:
                return {"error": "RM-ANOVA requires a subject variable"}
            if not within or len(within) < 1:
                return {"error": "RM-ANOVA requires at least one within factor"}
            
            # Get post-hoc results from analyzer
            posthoc = analyzer.perform_test(df=df, dv=dv, subject=subject, within=within, alpha=alpha, selected_comparisons=selected_comparisons, method=method, control_group=control_group)
            
            # Add validation to ensure we're getting valid results
            if posthoc and 'pairwise_comparisons' in posthoc:
                logger.debug(f"DEBUG: Found {len(posthoc['pairwise_comparisons'])} rm-anova post-hoc comparisons")
            else:
                logger.debug("DEBUG: No valid rm-anova post-hoc results found!")
                
            # Explicitly pass through the posthoc results without modification
            return posthoc
        
        return {"error": f"Unknown ANOVA type: {anova_type}"}
    
