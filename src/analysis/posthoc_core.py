import numpy as np
from core.level_order import natural_order
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
                for level_b in natural_order(df[factors[0]].unique()):
                    for level_a in natural_order(df[factors[1]].unique()):
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

                if method.lower() == 'tukey':
                    pairwise_tukeyhsd = get_pairwise_tukeyhsd()
                    # Create interaction group for Tukey HSD matching the exact label formats
                    df['interaction_group'] = factors[0] + "=" + df[factors[0]].astype(str) + ", " + factors[1] + "=" + df[factors[1]].astype(str)
                    # Run Tukey HSD on the interaction groups
                    tukey = pairwise_tukeyhsd(df[dv], df['interaction_group'], alpha=alpha)
                    
                    for i in range(len(tukey.pvalues)):
                        group1 = str(tukey.groupsunique[tukey.pairindices[i, 0]])
                        group2 = str(tukey.groupsunique[tukey.pairindices[i, 1]])
                        p_val = tukey.pvalues[i]
                        conf_int = tukey.confint[i]
                        
                        norm_pair = normalize_pair((group1, group2))
                        match = (normalized_selected is not None and norm_pair in normalized_selected)
                        if normalized_selected is not None and not match:
                            continue
                            
                        # No Holm-Sidak correction needed for Tukey! It's already family-wise corrected.
                        PostHocAnalyzer.add_comparison(
                            result,
                            group1=group1,
                            group2=group2,
                            test="Tukey HSD",
                            p_value=p_val,
                            statistic=None,
                            corrected=True,
                            correction_method="Tukey HSD",
                            confidence_interval=tuple(conf_int),
                            alpha=alpha
                        )

                elif method.lower() == 'dunnett' and control_group:
                    # Flatten the data for scipy.stats.dunnett
                    samples = []
                    control_sample = None
                    treatment_labels = []
                    
                    for group_label in interaction_groups:
                        vals = group_to_values[group_label]
                        if group_label == control_group:
                            control_sample = vals
                        else:
                            samples.append(vals)
                            treatment_labels.append(group_label)
                    
                    if control_sample is None:
                        raise ValueError(f"Control group '{control_group}' not found in data.")
                        
                    # Safeguard: Check n >= 2 for all samples (including control)
                    if len(control_sample) < 2:
                        raise ValueError(f"Control group '{control_group}' has n={len(control_sample)} < 2, cannot compute Dunnett.")
                    for label, sample in zip(treatment_labels, samples):
                        if len(sample) < 2:
                            raise ValueError(f"Treatment group '{label}' has n={len(sample)} < 2, cannot compute Dunnett.")
                            
                    if samples:
                        scipy_stats = get_scipy_stats()
                        dunnett_res = scipy_stats.dunnett(*samples, control=control_sample)
                        
                        try:
                            ci = dunnett_res.confidence_interval(confidence_level=1 - alpha)
                            lowers, uppers = ci.low, ci.high
                        except AttributeError:
                            lowers, uppers = [None]*len(samples), [None]*len(samples)
                        
                        for i, label in enumerate(treatment_labels):
                            norm_pair = normalize_pair((control_group, label))
                            if normalized_selected is not None and norm_pair not in normalized_selected:
                                continue
                                
                            # If scipy returns an array or scalar depending on number of treatments
                            p_val = float(np.atleast_1d(dunnett_res.pvalue)[i])
                            stat = float(np.atleast_1d(dunnett_res.statistic)[i])
                            
                            c_int = (None, None)
                            if lowers[i] is not None and uppers[i] is not None:
                                c_int = (float(np.atleast_1d(lowers)[i]), float(np.atleast_1d(uppers)[i]))
                            
                            PostHocAnalyzer.add_comparison(
                                result,
                                group1=control_group,
                                group2=label,
                                test="Dunnett Test",
                                p_value=p_val,
                                statistic=stat,
                                corrected=True,
                                correction_method="Dunnett",
                                confidence_interval=c_int,
                                alpha=alpha
                            )
                else:
                    # Perform t-tests for each pair
                    pvals = []
                    stats_list = []
                    for g1, g2 in filtered_pairs:
                        vals1 = group_to_values[g1]
                        vals2 = group_to_values[g2]
                        # Use t-test (assume equal variance for now)
                        stat, pval = get_scipy_stats().ttest_ind(vals1, vals2, equal_var=True)
                        pvals.append(pval)
                        stats_list.append((g1, g2, stat, pval, vals1, vals2))
                    
                    # Apply multiple comparison correction based on method
                    multipletests = get_statsmodels_multitest()
                    if pvals:
                        if method.lower() == 'paired_fdr':
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

                # After all, warn if any selected pair is not present
                if normalized_selected is not None:
                    added_pairs = set(normalize_pair((c["group1"], c["group2"])) for c in result["pairwise_comparisons"])
                    missing = normalized_selected - added_pairs
                    if missing:
                        logger.warning(f"WARNING: The following selected pairs were not found in the available post-hoc comparisons: {missing}")
            
            # Set the posthoc_test value for decision tree visualization
            method_name_map = {
                "tukey": "Tukey HSD",
                "dunnett": "Dunnett Test",
                "paired_custom": "Pairwise t-tests (independent, Holm-Šidák)",
                "paired_fdr": "Pairwise t-tests (independent, FDR Benjamini-Hochberg)",
                "holm": "Pairwise t-tests (independent, Holm-Šidák)"
            }
            result["posthoc_test"] = method_name_map.get(method, f"Post-hoc test ({method})")
            
            return result
        except Exception as e:
            result["error"] = f"Error in Two-Way ANOVA post-hoc tests: {str(e)}"
            return result
        
class MixedAnovaPostHocAnalyzer(PostHocAnalyzer):
    """UPDATED: Advanced post-hoc tests for Mixed ANOVA with proper between/within factor handling."""
    
    
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
    def perform_test(df, between, within, dv, subject, alpha=0.05, selected_comparisons=None, method='tukey', control_group=None):
        """
        UPDATED: Enhanced Mixed ANOVA post-hoc tests with proper between/within factor distinction
        """
        try:
            if method and method.lower() == "emm_mvt":
                from analysis.emm_posthoc import mixed_dunnett_emm_mvt, UnsupportedDesignError
                try:
                    contrasts = mixed_dunnett_emm_mvt(
                        df, dv=dv, subject=subject, between=between,
                        within=within, control_group=control_group, alpha=alpha,
                    )
                except UnsupportedDesignError as exc:
                    logger.warning("EMM/mvt unavailable (%s); falling back to isolated t-tests", exc)
                else:
                    emm_result = PostHocAnalyzer.create_result_template(
                        "Dunnett-type (EMM + multivariate-t, Mixed)")
                    for c in contrasts:
                        PostHocAnalyzer.add_comparison(
                            emm_result,
                            group1=f"{c['control']}:{c['within_level']}",
                            group2=f"{c['treatment']}:{c['within_level']}",
                            test="EMM + multivariate-t",
                            p_value=c["p_value"],
                            statistic=c["t"],
                            significant=c["significant"],
                            correction_method="multivariate-t (within level)",
                        )
                    return emm_result

            result = PostHocAnalyzer.create_result_template("Mixed ANOVA Post-hoc Tests")

            # Effect-driven post-hoc (feature B): after a significant Mixed ANOVA
            # the follow-up is gated on which omnibus effects are significant.
            # Interaction sig -> simple main effects (within-per-group +
            # between-per-within-level, NO cross-cells); else the significant main
            # effect's marginal-mean contrasts. Holm-Sidak per effect family.
            from analysis.mixed_simple_effects import mixed_effect_driven_posthoc

            bcol = between[0] if isinstance(between, (list, tuple)) else between
            wcol = within[0] if isinstance(within, (list, tuple)) else within

            interaction_p = within_p = between_p = None
            _gating_fallback = None
            try:
                _pg = get_pingouin_module()
                _aov = _pg.mixed_anova(data=df, dv=dv, within=wcol, subject=subject, between=bcol)

                # pingouin renamed the uncorrected-p column between releases
                # ("p-unc" up to 0.5.x, "p_unc" from 0.6). Reading only one
                # spelling raised KeyError, which the except below swallowed, so
                # every effect p stayed None and the effect-driven gate silently
                # degraded to simple main effects for EVERY design -- including
                # the interaction-n.s. case that must use marginal means.
                _p_col = next((c for c in ("p-unc", "p_unc") if c in _aov.columns), None)
                if _p_col is None:
                    raise KeyError(
                        f"no uncorrected-p column in mixed_anova output: {list(_aov.columns)}")

                def _effect_p(source):
                    _row = _aov[_aov["Source"] == source]
                    return float(_row[_p_col].iloc[0]) if not _row.empty else None

                within_p = _effect_p(wcol)
                between_p = _effect_p(bcol)
                interaction_p = _effect_p("Interaction")
            except Exception as exc:
                _gating_fallback = str(exc)
                logger.warning(
                    "Mixed post-hoc: omnibus effect p-values unavailable (%s); the effect-driven "
                    "gate is SKIPPED and simple main effects are reported without checking which "
                    "omnibus effects are significant.", exc)

            comps, mode = mixed_effect_driven_posthoc(
                df, dv=dv, subject=subject, between=bcol, within=wcol, alpha=alpha,
                interaction_p=interaction_p, within_p=within_p, between_p=between_p,
            )
            for comp in comps:
                PostHocAnalyzer.add_comparison(
                    result,
                    group1=comp["group1"],
                    group2=comp["group2"],
                    test=f"{comp['test']} (Holm-Sidak)",
                    p_value=comp["p_value"],
                    statistic=comp["statistic"],
                    corrected=True,
                    correction_method=comp["correction_method"],
                    effect_size=comp["effect_size"],
                    effect_size_type=comp["effect_size_type"],
                    alpha=alpha,
                    significant=comp["significant"],
                    comparison_type=comp["comparison_type"],
                )
            _mode_label = {
                "simple_main_effects": "Simple main effects (Holm-Sidak per family)",
                "marginal_within": "Within-factor marginal means (Holm-Sidak)",
                "marginal_between": "Between-factor marginal means (Holm-Sidak)",
                "none": "No pairwise post-hoc (no significant effect to break down)",
            }
            result["posthoc_test"] = _mode_label.get(mode, "Mixed post-hoc")
            result["posthoc_mode"] = mode
            # Make a degraded gate visible in the result, not just in the log.
            result["gating_applied"] = _gating_fallback is None
            if _gating_fallback is not None:
                result["gating_fallback_reason"] = _gating_fallback
                result.setdefault("warnings", []).append(
                    "Effect-driven post-hoc gating unavailable (%s): simple main effects were "
                    "reported without checking which omnibus effects are significant."
                    % _gating_fallback)
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
            if method and method.lower() == "emm_mvt":
                from analysis.emm_posthoc import rm_dunnett_emm_mvt, UnsupportedDesignError
                within_factor = within[0] if isinstance(within, (list, tuple)) else within
                try:
                    contrasts = rm_dunnett_emm_mvt(
                        df, dv=dv, subject=subject, within=within_factor,
                        control_level=control_group, alpha=alpha,
                    )
                except UnsupportedDesignError as exc:
                    logger.warning("RM EMM/mvt unavailable (%s); falling back to isolated t-tests", exc)
                else:
                    emm_result = PostHocAnalyzer.create_result_template(
                        "Dunnett-type (EMM + multivariate-t, RM level-vs-baseline)")
                    for c in contrasts:
                        PostHocAnalyzer.add_comparison(
                            emm_result,
                            group1=str(c["control"]),
                            group2=str(c["level"]),
                            test="EMM + multivariate-t",
                            p_value=c["p_value"],
                            statistic=c["t"],
                            significant=c["significant"],
                            correction_method="multivariate-t (level vs baseline)",
                        )
                    return emm_result

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
            within_levels = natural_order(df[within_factor].unique())
            
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
                    msg = f"Insufficient paired data for {level1} vs {level2}"
                    logger.warning(f"WARNING: {msg}")
                    result.setdefault("warnings", []).append(msg)
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
                    "df": df_t,
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
            
            if method.lower() == 'bonferroni':
                correction_method = "Bonferroni"
                corrected_p_values = [min(1.0, p * n_comparisons) for p in p_values]
                
            elif method.lower() == 'dunnett' and control_group:
                # Repeated-measures contrasts are all within-subject (paired) and
                # thus dependent, so scipy.stats.dunnett / the exact Dunnett
                # multivariate-t (which assume independent groups with shared
                # equicorrelation) do not apply, and only the per-comparison
                # p-values are available. We control the family-wise error rate
                # over the many-to-one family (control vs each level) with
                # Holm-Bonferroni, valid under arbitrary dependence. Labelled
                # honestly as Holm-adjusted, not exact Dunnett.
                correction_method = "Dunnett-type (Holm-adjusted, repeated measures)"
                dunnett_p_values = []
                control_indices = []

                for i, comp in enumerate(comparisons):
                    level1_str = str(comp["level1"])
                    level2_str = str(comp["level2"])
                    if level1_str == control_group or level2_str == control_group:
                        dunnett_p_values.append(comp["p_val"])
                        control_indices.append(i)

                corrected_p_values = [1.0] * len(p_values)
                if dunnett_p_values:
                    # Plain Holm-Bonferroni (not Holm-Sidak): only the Bonferroni
                    # variant guarantees FWER control under arbitrary dependence,
                    # which is required because these contrasts can be negatively
                    # correlated (within-subject pairing).
                    multipletests = get_statsmodels_multitest()
                    holm_adjusted = multipletests(
                        dunnett_p_values, alpha=alpha, method='holm')[1].tolist()
                    for j, orig_idx in enumerate(control_indices):
                        corrected_p_values[orig_idx] = holm_adjusted[j]
                else:
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
                if method.lower() == 'dunnett' and control_group:
                    if str(comp["level1"]) != control_group and str(comp["level2"]) != control_group:
                        continue
                
                # Calculate corrected confidence interval
                t_crit = scipy_stats.t.ppf(1 - alpha_sidak/2, comp["df"])
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
                    degrees_of_freedom=comp["df"],
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
            # No "tukey" entry -- see the Mixed analyzer above (audit SC2).
            method_name_map = {
                "dunnett": "Dunnett Test (RM)",
                "bonferroni": "Bonferroni (RM)",
                "holm": "Holm-Šidák (RM)"
            }
            result["posthoc_test"] = method_name_map.get(method, f"RM Post-hoc ({method})")
            
            return result
            
        except Exception as e:
            result["error"] = f"Error in RM ANOVA post-hoc tests: {str(e)}"
            logger.exception(f"ERROR RM POSTHOC: {str(e)}")
            return result
    
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
    def perform_test(valid_groups, samples, alpha=0.05, n_boot=1000, seed=12345):
        result = PostHocAnalyzer.create_result_template("Dunn-Test")

        # Seed a local Generator so the bootstrap CI below is reproducible. The
        # unseeded global np.random drew fresh samples every run, so the reported
        # median-difference CI drifted between identical analyses.
        rng = np.random.default_rng(seed)

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

            # Bootstrap CI - np.subtract.outer(b1, b2) computes the identical
            # n1×n2 pairwise-difference matrix as the equivalent nested
            # Python loop, vectorized (was ~13.5s per pair at n=500/group).
            boots = []
            for _ in range(n_boot):
                b1 = rng.choice(x, n1, replace=True)
                b2 = rng.choice(y, n2, replace=True)
                boots.append(np.median(np.subtract.outer(b1, b2)))
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
                    wstat, p = stats.wilcoxon(
                        x, y, zero_method='pratt',
                        method='exact' if len(x) <= 25 else 'approx',
                    )
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
                es, estype = d, "Cohen's d (RM)"
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
    
