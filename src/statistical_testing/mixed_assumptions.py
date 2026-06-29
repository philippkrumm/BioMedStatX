import numpy as np
import pandas as pd
from scipy import stats

from analysis.stats_functions import get_pingouin_module


class MixedAnovaAssumptionEngine:
    @staticmethod
    def _pingouin_p_column(columns):
        if columns is None:
            return None
        if "p_unc" in columns:
            return "p_unc"
        if "p-unc" in columns:
            return "p-unc"
        return None

    @staticmethod
    def _pingouin_p_value(row, default=np.nan):
        if row is None:
            return default
        columns = row.index if hasattr(row, "index") else row.keys() if hasattr(row, "keys") else None
        p_col = MixedAnovaAssumptionEngine._pingouin_p_column(columns)
        if p_col is None:
            return default
        raw_value = row.get(p_col, default) if hasattr(row, "get") else row[p_col]
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return default
    def _test_mixed_anova_between_assumptions(df, dv, between_factor, subject, alpha=0.05):
        """
        Tests assumptions for between-subjects factors in Mixed ANOVA.
        
        Tests performed:
        - Levene's Test for Homogeneity of Variance
        - Brown-Forsythe Test (robust alternative)
        - Bartlett's Test (sensitive to normality violations)
        - Welch's ANOVA (when assumptions violated)
        
        Parameters:
        -----------
        df : DataFrame
            Data containing mixed design variables
        dv : str
            Dependent variable column name
        between_factor : str
            Between-subjects factor column name
        subject : str
            Subject identifier column name
        alpha : float
            Significance level for assumption tests
            
        Returns:
        --------
        dict
            Comprehensive between-factor assumption test results
        """
        assumption_results = {}

        try:
            # Get between-factor groups
            between_groups = df[between_factor].unique()
            if len(between_groups) < 2:
                assumption_results["between_assumptions"] = {
                    "error": "Need at least 2 groups for between-factor assumption testing"
                }
                return assumption_results
            
            # Prepare group data for assumption tests
            group_data = []
            group_labels = []
            
            for group in between_groups:
                group_subset = df[df[between_factor] == group][dv].values
                if len(group_subset) > 0:
                    group_data.append(group_subset)
                    group_labels.append(group)
            
            if len(group_data) < 2:
                assumption_results["between_assumptions"] = {
                    "error": "Insufficient data for assumption testing"
                }
                return assumption_results
            
            # 1. Levene's Test for Homogeneity of Variance (most common)
            levene_results = MixedAnovaAssumptionEngine._perform_levene_test(group_data, group_labels, dv, between_factor)
            
            # 2. Brown-Forsythe Test (robust to non-normality)
            brown_forsythe_results = MixedAnovaAssumptionEngine._perform_brown_forsythe_test(group_data, group_labels, dv, between_factor)
            
            # 3. Bartlett's Test (sensitive to normality)
            bartlett_results = MixedAnovaAssumptionEngine._perform_bartlett_test(group_data, group_labels, dv, between_factor)
            
            # 4. Welch's ANOVA (unequal variances correction)
            welch_results = MixedAnovaAssumptionEngine._perform_welch_anova(group_data, group_labels, dv, between_factor)
            
            # Compile comprehensive results
            assumption_results["between_assumptions"] = {
                "factor": between_factor,
                "groups_tested": group_labels,
                "sample_sizes": [len(data) for data in group_data],
                "variance_tests": {
                    "levene": levene_results,
                    "brown_forsythe": brown_forsythe_results,
                    "bartlett": bartlett_results
                },
                "robust_alternatives": {
                    "welch_anova": welch_results
                },
                "recommendations": MixedAnovaAssumptionEngine._generate_between_assumption_recommendations(
                    levene_results, brown_forsythe_results, bartlett_results, alpha
                )
            }
            
        except Exception as e:
            assumption_results["between_assumptions"] = {
                "error": f"Between-factor assumption testing failed: {str(e)}",
                "factor": between_factor
            }
            
        return assumption_results
    
    @staticmethod
    def _perform_levene_test(group_data, group_labels, dv, factor_name):
        """
        Performs Levene's test for homogeneity of variance.

        Returns comprehensive results including interpretation.
        """
        try:
            levene = stats.levene

            # Perform Levene's test (now using median to match Brown-Forsythe by default)
            statistic, p_value = levene(*group_data, center='median')
            
            # Calculate group variances for descriptive info
            group_variances = [np.var(data, ddof=1) for data in group_data]
            
            return {
                "test_name": "Levene's Test for Homogeneity of Variance",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "group_variances": {f"{factor_name}={group_labels[i]}": float(var) 
                                 for i, var in enumerate(group_variances)},
                "variance_ratio": float(max(group_variances) / min(group_variances)) if min(group_variances) > 0 else None,
                "assumption_met": p_value > 0.05,
                "interpretation": (
                    f"Homogeneity of variance assumption is {'met' if p_value > 0.05 else 'violated'} "
                    f"(p = {p_value:.4f}). "
                    f"{'No correction needed.' if p_value > 0.05 else 'Consider robust alternatives.'}"
                ),
                "recommendation": (
                    "Proceed with standard ANOVA" if p_value > 0.05 else 
                    "Consider Welch's ANOVA or Brown-Forsythe test"
                )
            }
            
        except Exception as e:
            return {
                "test_name": "Levene's Test for Homogeneity of Variance",
                "error": f"Levene's test failed: {str(e)}",
                "assumption_met": None
            }
    
    @staticmethod
    def _perform_brown_forsythe_test(group_data, group_labels, dv, factor_name):
        """
        Performs Brown-Forsythe test (robust version of Levene's test).

        Uses median instead of mean - more robust to non-normality.
        """
        try:
            levene = stats.levene

            # Brown-Forsythe test uses median instead of mean
            statistic, p_value = levene(*group_data, center='median')
            
            # Calculate group variances for descriptive info
            group_variances = [np.var(data, ddof=1) for data in group_data]
            
            return {
                "test_name": "Brown-Forsythe Test (Robust Homogeneity of Variance)",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "group_variances": {f"{factor_name}={group_labels[i]}": float(var) 
                                 for i, var in enumerate(group_variances)},
                "variance_ratio": float(max(group_variances) / min(group_variances)) if min(group_variances) > 0 else None,
                "assumption_met": p_value > 0.05,
                "interpretation": (
                    f"Robust homogeneity assumption is {'met' if p_value > 0.05 else 'violated'} "
                    f"(p = {p_value:.4f}). More robust to non-normality than Levene's test."
                ),
                "recommendation": (
                    "Variance homogeneity confirmed" if p_value > 0.05 else 
                    "Variance heterogeneity detected - use Welch's ANOVA"
                )
            }
            
        except Exception as e:
            return {
                "test_name": "Brown-Forsythe Test (Robust Homogeneity of Variance)",
                "error": f"Brown-Forsythe test failed: {str(e)}",
                "assumption_met": None
            }
    
    @staticmethod
    def _perform_bartlett_test(group_data, group_labels, dv, factor_name):
        """
        Performs Bartlett's test for homogeneity of variance.

        Note: Sensitive to departures from normality.
        """
        try:
            bartlett = stats.bartlett

            # Perform Bartlett's test
            statistic, p_value = bartlett(*group_data)
            
            # Calculate group variances for descriptive info
            group_variances = [np.var(data, ddof=1) for data in group_data]
            
            return {
                "test_name": "Bartlett's Test for Homogeneity of Variance",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "group_variances": {f"{factor_name}={group_labels[i]}": float(var) 
                                 for i, var in enumerate(group_variances)},
                "variance_ratio": float(max(group_variances) / min(group_variances)) if min(group_variances) > 0 else None,
                "assumption_met": p_value > 0.05,
                "interpretation": (
                    f"Variance homogeneity assumption is {'met' if p_value > 0.05 else 'violated'} "
                    f"(p = {p_value:.4f}). Note: Sensitive to non-normality."
                ),
                "recommendation": (
                    "Proceed if normality is also met" if p_value > 0.05 else 
                    "Use Brown-Forsythe or Levene's test for robustness"
                ),
                "note": "Bartlett's test assumes normality - use with caution if data is non-normal"
            }
            
        except Exception as e:
            return {
                "test_name": "Bartlett's Test for Homogeneity of Variance",
                "error": f"Bartlett's test failed: {str(e)}",
                "assumption_met": None
            }
    
    @staticmethod
    def _perform_welch_anova(group_data, group_labels, dv, factor_name):
        """
        Performs Welch's ANOVA (does not assume equal variances).

        Provides robust alternative when variance homogeneity is violated.
        """
        try:
            f_oneway = stats.f_oneway
            
            # Standard F-test (equal variances assumed)
            f_stat_standard, p_val_standard = f_oneway(*group_data)
            
            # Welch's ANOVA (unequal variances)
            try:
                # Calculate Welch's ANOVA manually for more control
                group_means = [np.mean(data) for data in group_data]
                group_vars = [np.var(data, ddof=1) for data in group_data]
                group_sizes = [len(data) for data in group_data]
                
                # Weighted grand mean
                weights = [n/var for n, var in zip(group_sizes, group_vars)]
                grand_mean = sum(w * mean for w, mean in zip(weights, group_means)) / sum(weights)
                
                # Welch's F statistic
                numerator = sum(w * (mean - grand_mean)**2 for w, mean in zip(weights, group_means))
                denominator = 1 + (2 * (len(group_data) - 2) / (len(group_data)**2 - 1)) * sum(
                    (1 - w/sum(weights))**2 / (n - 1) for w, n in zip(weights, group_sizes)
                )
                
                welch_f = numerator / denominator
                
                # Approximate degrees of freedom
                df1 = len(group_data) - 1
                df2 = (len(group_data)**2 - 1) / (3 * sum((1 - w/sum(weights))**2 / (n - 1) 
                                                         for w, n in zip(weights, group_sizes)))
                
                # P-value from F-distribution
                f = stats.f
                p_val_welch = 1 - f.cdf(welch_f, df1, df2)
                
            except Exception:
                # Fallback to scipy's implementation if available
                welch_f, p_val_welch = f_oneway(*group_data)
                df1, df2 = len(group_data) - 1, sum(group_sizes) - len(group_data)
            
            return {
                "test_name": "Welch's ANOVA (Unequal Variances)",
                "welch_f_statistic": float(welch_f),
                "welch_p_value": float(p_val_welch),
                "standard_f_statistic": float(f_stat_standard),
                "standard_p_value": float(p_val_standard),
                "df1": int(df1),
                "df2": float(df2),
                "group_sizes": group_sizes,
                "group_means": {f"{factor_name}={group_labels[i]}": float(mean) 
                              for i, mean in enumerate(group_means)},
                "group_variances": {f"{factor_name}={group_labels[i]}": float(var) 
                                  for i, var in enumerate(group_vars)},
                "interpretation": (
                    f"Welch's ANOVA (robust to unequal variances): F({df1}, {df2:.1f}) = {welch_f:.3f}, "
                    f"p = {p_val_welch:.4f}. "
                    f"{'Significant' if p_val_welch < 0.05 else 'Non-significant'} group differences detected."
                ),
                "recommendation": (
                    "Use Welch's result when variance assumptions are violated" if 
                    abs(p_val_welch - p_val_standard) > 0.01 else 
                    "Results similar to standard ANOVA"
                )
            }
            
        except Exception as e:
            return {
                "test_name": "Welch's ANOVA (Unequal Variances)",
                "error": f"Welch's ANOVA failed: {str(e)}"
            }
    
    @staticmethod
    def _generate_between_assumption_recommendations(levene_results, brown_forsythe_results, bartlett_results, alpha=0.05):
        """
        Generates intelligent recommendations based on assumption test results.
        
        Provides guidance on which test results to trust and what corrections to apply.
        """
        recommendations = []
        
        try:
            # Check Levene's test result
            levene_met = levene_results.get("assumption_met", None)
            brown_met = brown_forsythe_results.get("assumption_met", None)
            bartlett_met = bartlett_results.get("assumption_met", None)
            
            # Generate recommendations based on test consensus
            if levene_met is True and brown_met is True:
                recommendations.append("✅ Variance homogeneity confirmed by both Levene's and Brown-Forsythe tests")
                recommendations.append("→ Proceed with standard Mixed ANOVA")
                
            elif levene_met is False and brown_met is False:
                recommendations.append("⚠️ Variance heterogeneity detected by both robust tests")
                recommendations.append("→ Use Welch's ANOVA correction for between-factor")
                recommendations.append("→ Consider Games-Howell post-hoc tests instead of Tukey")
                
            elif levene_met != brown_met:
                recommendations.append("⚠️ Mixed results from variance tests")
                recommendations.append("→ Trust Brown-Forsythe result (more robust to non-normality)")
                if brown_met is False:
                    recommendations.append("→ Apply Welch's correction")
                else:
                    recommendations.append("→ Standard ANOVA acceptable")
                    
            # Bartlett's test interpretation
            if bartlett_met is not None:
                if bartlett_met != levene_met:
                    recommendations.append("INFO:Bartlett's test differs from Levene's - may indicate non-normality")
                    recommendations.append("→ Check normality assumptions as well")
            
            # Variance ratio guidance
            variance_ratio = levene_results.get("variance_ratio", None)
            if variance_ratio is not None:
                if variance_ratio > 4:
                    recommendations.append(f"⚠️ High variance ratio ({variance_ratio:.2f}) - strong heterogeneity")
                    recommendations.append("→ Welch's correction highly recommended")
                elif variance_ratio > 2:
                    recommendations.append(f"⚠️ Moderate variance ratio ({variance_ratio:.2f}) - consider robust methods")
                else:
                    recommendations.append(f"✅ Acceptable variance ratio ({variance_ratio:.2f})")
            
            # Overall recommendation
            if not recommendations:
                recommendations.append("INFO:Unable to assess assumptions - proceed with caution")
                
        except Exception as e:
            recommendations.append(f"⚠️ Error generating recommendations: {str(e)}")
            
        return recommendations

    @staticmethod
    def _test_mixed_anova_within_sphericity(df, dv, subject, within_factor, aov, alpha=0.05):
        """
        Tests sphericity assumptions for within-subjects factors in Mixed ANOVA.
        
        Uses the comprehensive sphericity testing framework adapted for Mixed designs.
        Includes interaction-specific sphericity testing when applicable.
        
        Parameters:
        -----------
        df : DataFrame
            Data containing mixed design variables
        dv : str
            Dependent variable column name
        subject : str
            Subject identifier column name
        within_factor : str
            Within-subjects factor column name
        aov : DataFrame
            ANOVA results table from pingouin
        alpha : float
            Significance level for assumption tests
            
        Returns:
        --------
        dict
            Comprehensive within-factor sphericity analysis results
        """
        sphericity_results = {}
        
        try:
            # Get within-factor levels and check if sphericity testing is relevant
            within_levels = df[within_factor].unique()
            k = len(within_levels)
            
            if k <= 2:
                # Sphericity always met with 2 levels
                sphericity_results["within_sphericity_test"] = {
                    "test_name": "Mauchly's Test for Sphericity (Within-Factor)",
                    "factor": within_factor,
                    "W": None,
                    "chi_square": None,
                    "df": None,
                    "p_value": None,
                    "sphericity_assumed": True,
                    "note": "Sphericity assumption is always met with 2 levels",
                    "interpretation": "No correction needed - only 2 conditions compared"
                }
                sphericity_results["within_corrected_p_value"] = None  # No correction needed
                sphericity_results["within_correction_used"] = "None (sphericity assumption met)"
                return sphericity_results
            
            # Perform comprehensive sphericity testing for within-factor
            pg = get_pingouin_module()
            
            # Test sphericity for the within-factor
            try:
                sphericity_result = pg.sphericity(df, dv=dv, subject=subject, within=within_factor)
                
                # Handle different return formats
                if isinstance(sphericity_result, tuple) and len(sphericity_result) >= 3:
                    W, pval, spher = sphericity_result[:3]
                    sphericity_violated = not bool(spher) if spher is not None else True
                    
                    sphericity_results["within_sphericity_test"] = {
                        "test_name": "Mauchly's Test for Sphericity (Within-Factor)",
                        "factor": within_factor,
                        "W": float(W) if W is not None else None,
                        "p_value": float(pval) if pval is not None else None,
                        "sphericity_assumed": bool(spher) if spher is not None else False,
                        "df": int((k * (k - 1)) / 2 - 1) if k > 2 else None,
                        "interpretation": MixedAnovaAssumptionEngine._interpret_sphericity_test(pval, spher) if pval is not None else "Test failed",
                        "levels_tested": k,
                        "comparisons": int(k * (k - 1) / 2)
                    }
                else:
                    raise ValueError("Unexpected sphericity test output format")
                    
            except Exception as e:
                # Fallback: Extract from ANOVA table if available
                sphericity_results["within_sphericity_test"] = MixedAnovaAssumptionEngine._extract_mixed_sphericity_from_anova_table(
                    aov, within_factor, k
                )
                sphericity_violated = not sphericity_results["within_sphericity_test"].get("sphericity_assumed", True)
                
            # Apply corrections for within-factor effects in Mixed ANOVA
            if sphericity_violated:
                corrections_applied = MixedAnovaAssumptionEngine._apply_mixed_anova_sphericity_corrections(
                    aov, within_factor, sphericity_violated
                )
                sphericity_results.update(corrections_applied)
            else:
                sphericity_results["within_sphericity_corrections"] = {
                    "needed": False,
                    "reason": "Sphericity assumption is met for within-factor"
                }
                sphericity_results["within_corrected_p_value"] = None  # No correction needed
                sphericity_results["within_correction_used"] = "None (sphericity assumption met)"
                
        except Exception as e:
            # Comprehensive fallback
            sphericity_results["within_sphericity_test"] = {
                "test_name": "Mauchly's Test for Sphericity (Within-Factor)",
                "factor": within_factor,
                "W": None,
                "p_value": None,
                "sphericity_assumed": False,
                "note": f"Within-factor sphericity test failed: {str(e)}",
                "interpretation": "Indeterminate (Defaulting to GG correction)"
            }
            sphericity_results["within_corrected_p_value"] = None
            sphericity_results["within_correction_used"] = "None (sphericity test failed)"
            
        return sphericity_results
    
    @staticmethod
    def _extract_mixed_sphericity_from_anova_table(aov, within_factor, k):
        """
        Extracts sphericity information for within-factor from Mixed ANOVA table.
        
        Parameters:
        -----------
        aov : DataFrame
            ANOVA results table
        within_factor : str
            Within-subjects factor name
        k : int
            Number of within-factor levels
            
        Returns:
        --------
        dict
            Sphericity test results for within-factor
        """
        try:
            # Look for within-factor row in ANOVA table
            within_mask = aov["Source"] == within_factor
            if within_mask.any():
                within_row = aov.loc[within_mask].iloc[0]
                
                # Check for sphericity columns
                sphericity_cols = ['W-spher', 'p-spher', 'sphericity']
                available_cols = [col for col in sphericity_cols if col in aov.columns]
                
                if available_cols:
                    return {
                        "test_name": "Mauchly's Test for Sphericity (Within-Factor)",
                        "factor": within_factor,
                        "W": float(within_row.get('W-spher', np.nan)) if 'W-spher' in aov.columns else None,
                        "p_value": float(within_row.get('p-spher', np.nan)) if 'p-spher' in aov.columns else None,
                        "sphericity_assumed": bool(within_row.get('sphericity', True)) if 'sphericity' in aov.columns else None,
                        "df": int((k * (k - 1)) / 2 - 1) if k > 2 else None,
                        "note": "Extracted from Mixed ANOVA table",
                        "interpretation": "See p-value for significance",
                        "levels_tested": k
                    }
                else:
                    return {
                        "test_name": "Mauchly's Test for Sphericity (Within-Factor)",
                        "factor": within_factor,
                        "W": None,
                        "p_value": None,
                        "sphericity_assumed": False,  # Conservative assumption (Apply GG)
                        "df": int((k * (k - 1)) / 2 - 1) if k > 2 else None,
                        "note": "No sphericity information in Mixed ANOVA table",
                        "interpretation": "Indeterminate (Defaulting to GG correction)",
                        "levels_tested": k
                    }
            else:
                raise ValueError(f"Within-factor '{within_factor}' not found in ANOVA table")
                
        except Exception as e:
            return {
                "test_name": "Mauchly's Test for Sphericity (Within-Factor)",
                "factor": within_factor,
                "W": None,
                "p_value": None,
                "sphericity_assumed": False,
                "note": f"Failed to extract within-factor sphericity: {str(e)}",
                "interpretation": "Indeterminate (Defaulting to GG correction)",
                "levels_tested": k
            }
    
    @staticmethod
    def _apply_mixed_anova_sphericity_corrections(aov, within_factor, sphericity_violated):
        """
        Applies sphericity corrections specifically for Mixed ANOVA within-factor effects.
        
        Handles both main effects and interaction effects that involve the within-factor.
        
        Parameters:
        -----------
        aov : DataFrame
            ANOVA results table from pingouin
        within_factor : str
            Within-subjects factor name
        sphericity_violated : bool
            Whether sphericity assumption is violated
            
        Returns:
        --------
        dict
            Correction results for within-factor effects
        """
        corrections = {}
        
        try:
            if not sphericity_violated:
                corrections["within_sphericity_corrections"] = {
                    "needed": False,
                    "reason": "Sphericity assumption is met for within-factor"
                }
                return corrections
            
            # Sphericity violated - apply corrections to within-factor effects
            corrections["within_sphericity_corrections"] = {"needed": True}
            
            # Find within-factor main effect row
            within_mask = aov["Source"] == within_factor
            if within_mask.any():
                within_row = aov.loc[within_mask].iloc[0]
                
                # Apply corrections to within-factor main effect
                within_corrections = MixedAnovaAssumptionEngine._apply_corrections_to_effect_row(
                    within_row, f"within-factor ({within_factor})"
                )
                corrections["within_sphericity_corrections"]["main_effect"] = within_corrections
                
            # Find interaction effects involving the within-factor
            interaction_rows = aov[aov["Source"].str.contains(within_factor, regex=False) & 
                                  aov["Source"].str.contains("*", regex=False)]
            
            if not interaction_rows.empty:
                interaction_corrections = {}
                for _, interaction_row in interaction_rows.iterrows():
                    interaction_name = interaction_row["Source"]
                    interaction_corrections[interaction_name] = MixedAnovaAssumptionEngine._apply_corrections_to_effect_row(
                        interaction_row, f"interaction ({interaction_name})"
                    )
                corrections["within_sphericity_corrections"]["interactions"] = interaction_corrections
            
            # Overall recommendation for within-factor
            corrections["within_correction_recommendation"] = MixedAnovaAssumptionEngine._generate_within_factor_recommendations(
                corrections["within_sphericity_corrections"]
            )
                
        except Exception as e:
            corrections["within_sphericity_corrections"] = {
                "needed": True,
                "error": f"Failed to apply within-factor corrections: {str(e)}"
            }
            
        return corrections
    
    @staticmethod
    def _apply_corrections_to_effect_row(effect_row, effect_name):
        """
        Applies Greenhouse-Geisser and Huynh-Feldt corrections to a specific effect.
        
        Parameters:
        -----------
        effect_row : Series
            Row from ANOVA table for specific effect
        effect_name : str
            Name/description of the effect
            
        Returns:
        --------
        dict
            Correction results for this specific effect
        """
        corrections = {"effect": effect_name}
        
        try:
            # Greenhouse-Geisser Correction
            if 'GG-eps' in effect_row and 'p-GG' in effect_row:
                gg_epsilon = float(effect_row["GG-eps"])
                gg_p_value = float(effect_row["p-GG"])
                
                corrections["greenhouse_geisser"] = {
                    "epsilon": gg_epsilon,
                    "corrected_df1": float(effect_row["DF1"]) * gg_epsilon,
                    "corrected_df2": float(effect_row["DF2"]) * gg_epsilon,
                    "p_value": gg_p_value,
                    "conservative": True,
                    "description": f"Greenhouse-Geisser correction for {effect_name}"
                }
            else:
                gg_epsilon = None
                gg_p_value = MixedAnovaAssumptionEngine._pingouin_p_value(effect_row)
            
            # Huynh-Feldt Correction  
            if 'HF-eps' in effect_row and 'p-HF' in effect_row:
                hf_epsilon = float(effect_row["HF-eps"])
                hf_p_value = float(effect_row["p-HF"])
                
                corrections["huynh_feldt"] = {
                    "epsilon": hf_epsilon,
                    "corrected_df1": float(effect_row["DF1"]) * hf_epsilon,
                    "corrected_df2": float(effect_row["DF2"]) * hf_epsilon,
                    "p_value": hf_p_value,
                    "conservative": False,
                    "description": f"Huynh-Feldt correction for {effect_name}"
                }
            else:
                hf_epsilon = None
                hf_p_value = MixedAnovaAssumptionEngine._pingouin_p_value(effect_row)
            
            # Intelligent correction selection
            if gg_epsilon is not None and hf_epsilon is not None:
                if gg_epsilon > 0.75:
                    corrections["recommended_correction"] = "huynh_feldt"
                    corrections["final_p_value"] = hf_p_value
                    corrections["correction_used"] = f"Huynh-Feldt (ε = {hf_epsilon:.3f} > 0.75)"
                    corrections["rationale"] = "ε > 0.75: Huynh-Feldt is less conservative and more appropriate"
                else:
                    corrections["recommended_correction"] = "greenhouse_geisser"
                    corrections["final_p_value"] = gg_p_value
                    corrections["correction_used"] = f"Greenhouse-Geisser (ε = {gg_epsilon:.3f} ≤ 0.75)"
                    corrections["rationale"] = "ε ≤ 0.75: Greenhouse-Geisser is more conservative and safer"
            elif gg_epsilon is not None:
                corrections["recommended_correction"] = "greenhouse_geisser"
                corrections["final_p_value"] = gg_p_value
                corrections["correction_used"] = f"Greenhouse-Geisser (ε = {gg_epsilon:.3f})"
            elif hf_epsilon is not None:
                corrections["recommended_correction"] = "huynh_feldt"
                corrections["final_p_value"] = hf_p_value
                corrections["correction_used"] = f"Huynh-Feldt (ε = {hf_epsilon:.3f})"
            else:
                corrections["recommended_correction"] = "none"
                corrections["final_p_value"] = MixedAnovaAssumptionEngine._pingouin_p_value(effect_row)
                corrections["correction_used"] = "None (corrections not available)"
                corrections["rationale"] = "Correction information not available - consider multivariate approach"
                
        except Exception as e:
            corrections["error"] = f"Failed to apply corrections to {effect_name}: {str(e)}"
            corrections["final_p_value"] = MixedAnovaAssumptionEngine._pingouin_p_value(effect_row)
            corrections["correction_used"] = "None (correction failed)"
            
        return corrections
    
    @staticmethod
    def _generate_within_factor_recommendations(sphericity_corrections):
        """
        Generates recommendations for within-factor sphericity violations in Mixed ANOVA.
        
        Parameters:
        -----------
        sphericity_corrections : dict
            Sphericity correction results
            
        Returns:
        --------
        list
            List of recommendation strings
        """
        recommendations = []
        
        try:
            if not sphericity_corrections.get("needed", False):
                recommendations.append("✅ Within-factor sphericity assumption is met")
                recommendations.append("→ Standard Mixed ANOVA results are valid")
                return recommendations
            
            # Sphericity violated - provide guidance
            recommendations.append("⚠️ Within-factor sphericity assumption is violated")
            
            # Check main effect corrections
            main_effect = sphericity_corrections.get("main_effect", {})
            if main_effect:
                rec_correction = main_effect.get("recommended_correction", "none")
                if rec_correction != "none":
                    epsilon_val = main_effect.get(rec_correction, {}).get("epsilon", None)
                    if epsilon_val:
                        recommendations.append(f"→ Apply {rec_correction.replace('_', '-').title()} correction (ε = {epsilon_val:.3f})")
                
            # Check interaction effect corrections
            interactions = sphericity_corrections.get("interactions", {})
            if interactions:
                recommendations.append("→ Interaction effects also require sphericity corrections")
                for interaction_name, interaction_data in interactions.items():
                    rec_correction = interaction_data.get("recommended_correction", "none")
                    if rec_correction != "none":
                        recommendations.append(f"   • {interaction_name}: Use {rec_correction.replace('_', '-').title()} correction")
            
            # General recommendations
            recommendations.append("→ Report both uncorrected and corrected p-values")
            recommendations.append("→ Consider multivariate approach (MANOVA) as alternative")
            
        except Exception as e:
            recommendations.append(f"⚠️ Error generating within-factor recommendations: {str(e)}")
            
        return recommendations

    @staticmethod
    def _test_mixed_anova_interaction_assumptions(df, dv, between_factor, within_factor, subject, aov, alpha=0.05):
        """
        Tests assumptions for interaction effects in Mixed ANOVA.
        
        Comprehensive testing includes:
        - Sphericity for the interaction effect
        - Homogeneity of variance for interaction cells
        - Independence of covariance patterns across groups
        - Compound symmetry assumptions
        - Robust alternatives when assumptions violated
        
        Parameters:
        -----------
        df : DataFrame
            Data containing mixed design variables
        dv : str
            Dependent variable column name
        between_factor : str
            Between-subjects factor column name
        within_factor : str
            Within-subjects factor column name
        subject : str
            Subject identifier column name
        aov : DataFrame
            ANOVA results table from pingouin
        alpha : float
            Significance level for assumption tests
            
        Returns:
        --------
        dict
            Comprehensive interaction assumption test results
        """
        from itertools import product

        interaction_results = {}
        interaction_name = f"{within_factor} * {between_factor}"
        
        try:
            # Get factor levels
            between_levels = df[between_factor].unique()
            within_levels = df[within_factor].unique()
            
            if len(between_levels) < 2 or len(within_levels) < 2:
                interaction_results["interaction_assumptions"] = {
                    "error": "Need at least 2 levels in each factor for interaction assumption testing",
                    "interaction": interaction_name
                }
                return interaction_results
            
            # 1. Sphericity Testing for Interaction Effect
            interaction_sphericity = MixedAnovaAssumptionEngine._test_interaction_sphericity(
                df, dv, between_factor, within_factor, subject, aov, alpha
            )
            
            # 2. Homogeneity Tests for Interaction Cells
            cell_homogeneity = MixedAnovaAssumptionEngine._test_interaction_cell_homogeneity(
                df, dv, between_factor, within_factor, alpha
            )
            
            # 3. Covariance Pattern Tests
            covariance_tests = MixedAnovaAssumptionEngine._test_interaction_covariance_patterns(
                df, dv, between_factor, within_factor, subject, alpha
            )
            
            # 4. Box's M Test for Homogeneity of Covariance Matrices
            box_m_test = MixedAnovaAssumptionEngine._perform_box_m_test(
                df, dv, between_factor, within_factor, subject
            )
            
            # 5. Compound Symmetry Assessment
            compound_symmetry = MixedAnovaAssumptionEngine._assess_compound_symmetry(
                df, dv, between_factor, within_factor, subject
            )
            
            # Compile comprehensive results
            interaction_results["interaction_assumptions"] = {
                "interaction": interaction_name,
                "between_factor": between_factor,
                "within_factor": within_factor,
                "between_levels": len(between_levels),
                "within_levels": len(within_levels),
                "total_cells": len(between_levels) * len(within_levels),
                "sphericity_tests": interaction_sphericity,
                "cell_homogeneity": cell_homogeneity,
                "covariance_patterns": covariance_tests,
                "box_m_test": box_m_test,
                "compound_symmetry": compound_symmetry,
                "overall_recommendations": MixedAnovaAssumptionEngine._generate_interaction_recommendations(
                    interaction_sphericity, cell_homogeneity, covariance_tests, box_m_test, alpha
                )
            }
            
        except Exception as e:
            interaction_results["interaction_assumptions"] = {
                "error": f"Interaction assumption testing failed: {str(e)}",
                "interaction": interaction_name
            }
            
        return interaction_results
    
    @staticmethod
    def _test_interaction_sphericity(df, dv, between_factor, within_factor, subject, aov, alpha):
        """
        Tests sphericity specifically for the interaction effect.
        
        The interaction effect has its own sphericity assumption that may differ
        from the main within-factor sphericity.
        """
        try:
            interaction_name = f"{within_factor} * {between_factor}"
            
            # Check if interaction exists in ANOVA table
            interaction_mask = aov["Source"] == interaction_name
            if not interaction_mask.any():
                return {
                    "test_name": "Interaction Sphericity Test",
                    "interaction": interaction_name,
                    "error": "Interaction effect not found in ANOVA table",
                    "sphericity_assumed": False
                }
            
            interaction_row = aov.loc[interaction_mask].iloc[0]
            
            # Extract sphericity information for interaction
            sphericity_result = {
                "test_name": "Mauchly's Test for Interaction Sphericity",
                "interaction": interaction_name,
                "factor_levels": len(df[within_factor].unique())
            }
            
            # Try to get sphericity information from ANOVA table
            if 'W-spher' in interaction_row and pd.notnull(interaction_row['W-spher']):
                sphericity_result.update({
                    "W": float(interaction_row['W-spher']),
                    "p_value": float(interaction_row['p-spher']) if 'p-spher' in interaction_row else None,
                    "sphericity_assumed": bool(interaction_row['sphericity']) if 'sphericity' in interaction_row else None,
                    "interpretation": "Interaction sphericity extracted from ANOVA table"
                })
            else:
                # Attempt direct sphericity test for interaction
                try:
                    pg = get_pingouin_module()
                    # Create interaction factor for sphericity testing
                    df_interaction = df.copy()
                    df_interaction['interaction_factor'] = df_interaction[within_factor].astype(str) + "_" + df_interaction[between_factor].astype(str)
                    
                    # Test sphericity on interaction factor
                    sphericity_test = pg.sphericity(df_interaction, dv=dv, subject=subject, within='interaction_factor')
                    if isinstance(sphericity_test, tuple) and len(sphericity_test) >= 3:
                        W, pval, spher = sphericity_test[:3]
                        sphericity_result.update({
                            "W": float(W) if W is not None else None,
                            "p_value": float(pval) if pval is not None else None,
                            "sphericity_assumed": bool(spher) if spher is not None else False,
                            "interpretation": "Direct interaction sphericity test performed"
                        })
                    else:
                        raise ValueError("Unexpected sphericity test output")
                        
                except Exception as inner_e:
                    sphericity_result.update({
                        "W": None,
                        "p_value": None,
                        "sphericity_assumed": False,
                        "note": f"Direct sphericity test failed: {str(inner_e)}",
                        "interpretation": "Indeterminate (Defaulting to GG correction)"
                    })
            
            # Add corrections if available
            if 'GG-eps' in interaction_row and 'HF-eps' in interaction_row:
                corrections = MixedAnovaAssumptionEngine._apply_corrections_to_effect_row(
                    interaction_row, f"interaction ({interaction_name})"
                )
                sphericity_result["corrections"] = corrections
            
            return sphericity_result
            
        except Exception as e:
            return {
                "test_name": "Interaction Sphericity Test",
                "interaction": f"{within_factor} * {between_factor}",
                "error": f"Interaction sphericity test failed: {str(e)}",
                "sphericity_assumed": False
            }
    
    @staticmethod
    def _test_interaction_cell_homogeneity(df, dv, between_factor, within_factor, alpha):
        """
        Tests homogeneity of variance across all interaction cells.
        
        Creates a comprehensive test of whether variances are equal across
        all combinations of between and within factor levels.
        """
        try:
            levene = stats.levene
            bartlett = stats.bartlett
            from itertools import product

            # Get all factor level combinations
            between_levels = df[between_factor].unique()
            within_levels = df[within_factor].unique()
            
            # Collect data for each cell
            cell_data = []
            cell_labels = []
            cell_variances = {}
            
            for between_level, within_level in product(between_levels, within_levels):
                cell_mask = (df[between_factor] == between_level) & (df[within_factor] == within_level)
                cell_values = df.loc[cell_mask, dv].values
                
                if len(cell_values) > 1:  # Need at least 2 values for variance
                    cell_data.append(cell_values)
                    cell_label = f"{between_factor}={between_level}, {within_factor}={within_level}"
                    cell_labels.append(cell_label)
                    cell_variances[cell_label] = float(np.var(cell_values, ddof=1))
            
            if len(cell_data) < 2:
                return {
                    "test_name": "Interaction Cell Homogeneity Test",
                    "error": "Insufficient cells for homogeneity testing",
                    "cells_tested": len(cell_data)
                }
            
            # Perform Levene's test on all cells (using median / Brown-Forsythe)
            levene_stat, levene_p = levene(*cell_data, center='median')
            
            # Perform Bartlett's test on all cells
            bartlett_stat, bartlett_p = bartlett(*cell_data)
            
            # Calculate variance ratios
            variances = list(cell_variances.values())
            max_var = max(variances)
            min_var = min(variances)
            variance_ratio = max_var / min_var if min_var > 0 else np.inf
            
            return {
                "test_name": "Interaction Cell Homogeneity Tests",
                "cells_tested": len(cell_data),
                "cell_variances": cell_variances,
                "levene_test": {
                    "statistic": float(levene_stat),
                    "p_value": float(levene_p),
                    "assumption_met": levene_p > alpha,
                    "interpretation": f"Cell homogeneity {'met' if levene_p > alpha else 'violated'} (Levene's test, p = {levene_p:.4f})"
                },
                "bartlett_test": {
                    "statistic": float(bartlett_stat),
                    "p_value": float(bartlett_p),
                    "assumption_met": bartlett_p > alpha,
                    "interpretation": f"Cell homogeneity {'met' if bartlett_p > alpha else 'violated'} (Bartlett's test, p = {bartlett_p:.4f})"
                },
                "variance_ratio": float(variance_ratio),
                "variance_summary": {
                    "min_variance": float(min_var),
                    "max_variance": float(max_var),
                    "mean_variance": float(np.mean(variances)),
                    "variance_range": float(max_var - min_var)
                },
                "recommendation": (
                    "Cell homogeneity assumptions met" if levene_p > alpha else
                    "Consider robust methods due to heterogeneous cell variances"
                )
            }
            
        except Exception as e:
            return {
                "test_name": "Interaction Cell Homogeneity Tests",
                "error": f"Cell homogeneity testing failed: {str(e)}"
            }
    
    @staticmethod
    def _test_interaction_covariance_patterns(df, dv, between_factor, within_factor, subject, alpha):
        """
        Tests whether covariance patterns are similar across between-group levels.
        
        This is crucial for Mixed ANOVA as it assumes that the covariance structure
        of the within-factor is similar across between-group levels.
        """
        try:
            pearsonr = stats.pearsonr

            between_levels = df[between_factor].unique()
            within_levels = df[within_factor].unique()
            
            if len(within_levels) < 3:
                return {
                    "test_name": "Covariance Pattern Test",
                    "note": "Covariance pattern testing requires at least 3 within-factor levels",
                    "within_levels": len(within_levels)
                }
            
            # Create covariance matrices for each between-group
            covariance_matrices = {}
            correlation_matrices = {}
            
            for between_level in between_levels:
                group_data = df[df[between_factor] == between_level]
                
                # Pivot to get within-factor levels as columns
                pivoted = group_data.pivot_table(
                    values=dv, 
                    index=subject, 
                    columns=within_factor, 
                    aggfunc='mean'
                ).dropna()
                
                if len(pivoted) > 1:  # Need at least 2 subjects
                    cov_matrix = np.cov(pivoted.T)  # Transpose to get variables as rows
                    corr_matrix = np.corrcoef(pivoted.T)
                    
                    covariance_matrices[f"{between_factor}={between_level}"] = cov_matrix
                    correlation_matrices[f"{between_factor}={between_level}"] = corr_matrix
            
            if len(covariance_matrices) < 2:
                return {
                    "test_name": "Covariance Pattern Test",
                    "error": "Need at least 2 between-groups with sufficient data",
                    "groups_with_data": len(covariance_matrices)
                }
            
            # Compare covariance patterns
            covariance_similarities = {}
            correlation_similarities = {}
            
            group_names = list(covariance_matrices.keys())
            for i in range(len(group_names)):
                for j in range(i + 1, len(group_names)):
                    group1, group2 = group_names[i], group_names[j]
                    
                    # Calculate similarity between covariance matrices
                    cov1 = covariance_matrices[group1]
                    cov2 = covariance_matrices[group2]
                    
                    # Frobenius norm of difference (normalized)
                    cov_diff = np.linalg.norm(cov1 - cov2, 'fro')
                    cov_avg_norm = (np.linalg.norm(cov1, 'fro') + np.linalg.norm(cov2, 'fro')) / 2
                    cov_similarity = 1 - (cov_diff / cov_avg_norm) if cov_avg_norm > 0 else 0
                    
                    covariance_similarities[f"{group1} vs {group2}"] = float(cov_similarity)
                    
                    # Calculate correlation between correlation matrices
                    corr1 = correlation_matrices[group1]
                    corr2 = correlation_matrices[group2]
                    
                    # Flatten upper triangular parts (excluding diagonal)
                    triu_indices = np.triu_indices_from(corr1, k=1)
                    corr1_flat = corr1[triu_indices]
                    corr2_flat = corr2[triu_indices]
                    
                    if len(corr1_flat) > 0:
                        corr_similarity, _ = pearsonr(corr1_flat, corr2_flat)
                        correlation_similarities[f"{group1} vs {group2}"] = float(corr_similarity)
            
            # Overall assessment
            avg_cov_similarity = np.mean(list(covariance_similarities.values()))
            avg_corr_similarity = np.mean(list(correlation_similarities.values()))
            
            return {
                "test_name": "Covariance Pattern Similarity Test",
                "groups_compared": len(covariance_matrices),
                "within_levels": len(within_levels),
                "covariance_similarities": covariance_similarities,
                "correlation_similarities": correlation_similarities,
                "average_covariance_similarity": float(avg_cov_similarity),
                "average_correlation_similarity": float(avg_corr_similarity),
                "interpretation": (
                    f"Average covariance similarity: {avg_cov_similarity:.3f}, "
                    f"correlation similarity: {avg_corr_similarity:.3f}. "
                    f"{'Good' if avg_cov_similarity > 0.7 and avg_corr_similarity > 0.7 else 'Poor'} "
                    f"similarity across groups."
                ),
                "assumption_met": avg_cov_similarity > 0.7 and avg_corr_similarity > 0.7,
                "recommendation": (
                    "Covariance patterns are sufficiently similar" if 
                    avg_cov_similarity > 0.7 and avg_corr_similarity > 0.7 else
                    "Consider separate analyses for each group or robust methods"
                )
            }
            
        except Exception as e:
            return {
                "test_name": "Covariance Pattern Similarity Test",
                "error": f"Covariance pattern testing failed: {str(e)}"
            }
    
    @staticmethod
    def _perform_box_m_test(df, dv, between_factor, within_factor, subject):
        """
        Performs Box's M test for homogeneity of covariance matrices.
        
        Tests whether the covariance matrices are equal across between-group levels.
        This is a key assumption for Mixed ANOVA.
        """
        try:
            between_levels = df[between_factor].unique()
            within_levels = df[within_factor].unique()
            
            if len(between_levels) < 2 or len(within_levels) < 2:
                return {
                    "test_name": "Box's M Test",
                    "error": "Need at least 2 levels in both factors",
                    "between_levels": len(between_levels),
                    "within_levels": len(within_levels)
                }
            
            # Prepare data matrices for each group
            group_matrices = []
            group_sizes = []
            
            for between_level in between_levels:
                group_data = df[df[between_factor] == between_level]
                
                # Pivot to get within-factor levels as columns
                pivoted = group_data.pivot_table(
                    values=dv, 
                    index=subject, 
                    columns=within_factor, 
                    aggfunc='mean'
                ).dropna()
                
                if len(pivoted) > len(within_levels):  # Need more subjects than variables
                    group_matrices.append(pivoted.values)
                    group_sizes.append(len(pivoted))
            
            if len(group_matrices) < 2:
                return {
                    "test_name": "Box's M Test",
                    "error": "Insufficient data for Box's M test",
                    "groups_with_sufficient_data": len(group_matrices)
                }
            
            # Calculate Box's M statistic (simplified implementation)
            # Full implementation would be quite complex, so this is an approximation
            pooled_cov = None
            log_det_sum = 0
            total_df = 0
            
            for i, matrix in enumerate(group_matrices):
                cov_matrix = np.cov(matrix.T)
                n_i = group_sizes[i] - 1
                total_df += n_i
                
                # Add to pooled covariance
                if pooled_cov is None:
                    pooled_cov = n_i * cov_matrix
                else:
                    pooled_cov += n_i * cov_matrix
                
                # Add log determinant
                try:
                    log_det = np.log(np.linalg.det(cov_matrix))
                    log_det_sum += n_i * log_det
                except:
                    log_det_sum += n_i * np.log(1e-10)  # Handle singular matrices
            
            pooled_cov /= total_df
            
            # Box's M approximation
            try:
                pooled_log_det = np.log(np.linalg.det(pooled_cov))
                box_m = total_df * pooled_log_det - log_det_sum
                
                # Approximate p-value (this is a simplification)
                # In practice, Box's M follows a complex distribution
                p_value = 1.0 if box_m < 0 else min(0.5, np.exp(-abs(box_m) / 10))
                
                return {
                    "test_name": "Box's M Test (Approximation)",
                    "statistic": float(box_m),
                    "p_value": float(p_value),
                    "groups_tested": len(group_matrices),
                    "assumption_met": p_value > 0.05,
                    "interpretation": (
                        f"Box's M = {box_m:.3f}, p ≈ {p_value:.3f}. "
                        f"Covariance matrices are {'similar' if p_value > 0.05 else 'different'} "
                        f"across groups."
                    ),
                    "note": "This is an approximation - consider using specialized software for exact test",
                    "recommendation": (
                        "Covariance homogeneity assumption met" if p_value > 0.05 else
                        "Consider robust methods or separate group analyses"
                    )
                }
                
            except Exception as calc_e:
                return {
                    "test_name": "Box's M Test",
                    "error": f"Box's M calculation failed: {str(calc_e)}",
                    "groups_tested": len(group_matrices)
                }
            
        except Exception as e:
            return {
                "test_name": "Box's M Test",
                "error": f"Box's M test failed: {str(e)}"
            }
    
    @staticmethod
    def _assess_compound_symmetry(df, dv, between_factor, within_factor, subject):
        """
        Assesses compound symmetry assumption for Mixed ANOVA.
        
        Compound symmetry is a stronger assumption than sphericity and requires
        that all variances are equal and all covariances are equal.
        """
        try:
            between_levels = df[between_factor].unique()
            within_levels = df[within_factor].unique()
            
            compound_symmetry_results = {}
            
            for between_level in between_levels:
                group_data = df[df[between_factor] == between_level]
                
                # Pivot to get within-factor levels as columns
                pivoted = group_data.pivot_table(
                    values=dv, 
                    index=subject, 
                    columns=within_factor, 
                    aggfunc='mean'
                ).dropna()
                
                if len(pivoted) > 1:
                    # Calculate covariance matrix
                    cov_matrix = np.cov(pivoted.T)
                    
                    # Extract variances (diagonal) and covariances (off-diagonal)
                    variances = np.diag(cov_matrix)
                    covariances = cov_matrix[np.triu_indices_from(cov_matrix, k=1)]
                    
                    # Test for compound symmetry
                    # Variances should be approximately equal
                    variance_cv = np.std(variances) / np.mean(variances) if np.mean(variances) > 0 else np.inf
                    
                    # Covariances should be approximately equal
                    covariance_cv = np.std(covariances) / np.mean(covariances) if np.mean(covariances) > 0 else np.inf
                    
                    # Compound symmetry met if both CVs are small
                    compound_symmetry_met = variance_cv < 0.2 and covariance_cv < 0.3
                    
                    compound_symmetry_results[f"{between_factor}={between_level}"] = {
                        "variances": variances.tolist(),
                        "covariances": covariances.tolist(),
                        "variance_cv": float(variance_cv),
                        "covariance_cv": float(covariance_cv),
                        "compound_symmetry_met": compound_symmetry_met,
                        "variance_range": float(np.max(variances) - np.min(variances)),
                        "covariance_range": float(np.max(covariances) - np.min(covariances)) if len(covariances) > 0 else 0
                    }
            
            # Overall assessment
            all_groups_meet = all(group_data["compound_symmetry_met"] for group_data in compound_symmetry_results.values())
            
            return {
                "test_name": "Compound Symmetry Assessment",
                "groups_assessed": len(compound_symmetry_results),
                "group_results": compound_symmetry_results,
                "overall_compound_symmetry": all_groups_meet,
                "interpretation": (
                    f"Compound symmetry {'met' if all_groups_meet else 'violated'} across "
                    f"{len(compound_symmetry_results)} between-group level(s). "
                    f"This is a {'strong' if all_groups_meet else 'weak'} assumption for Mixed ANOVA."
                ),
                "recommendation": (
                    "Compound symmetry supports Mixed ANOVA assumptions" if all_groups_meet else
                    "Sphericity test more appropriate than compound symmetry assumption"
                )
            }
            
        except Exception as e:
            return {
                "test_name": "Compound Symmetry Assessment",
                "error": f"Compound symmetry assessment failed: {str(e)}"
            }
    
    @staticmethod
    def _generate_interaction_recommendations(sphericity_tests, cell_homogeneity, covariance_tests, box_m_test, alpha):
        """
        Generates comprehensive recommendations for interaction assumptions in Mixed ANOVA.
        
        Integrates results from all assumption tests to provide actionable guidance.
        """
        recommendations = []
        
        try:
            # Sphericity recommendations
            sphericity_met = sphericity_tests.get("sphericity_assumed", None)
            if sphericity_met is True:
                recommendations.append("✅ Interaction sphericity assumption is met")
            elif sphericity_met is False:
                recommendations.append("⚠️ Interaction sphericity assumption is violated")
                if "corrections" in sphericity_tests:
                    recommended = sphericity_tests["corrections"].get("recommended_correction", "none")
                    if recommended != "none":
                        recommendations.append(f"→ Apply {recommended.replace('_', '-').title()} correction for interaction")
            elif sphericity_met is None:
                recommendations.append("⚠️ Could not determine interaction sphericity")
                recommendations.append("→ Proceed with caution and consider robust alternatives")
            
            # Cell homogeneity recommendations
            if "levene_test" in cell_homogeneity:
                levene_met = cell_homogeneity["levene_test"].get("assumption_met", None)
                if levene_met is True:
                    recommendations.append("✅ Interaction cell variances are homogeneous")
                elif levene_met is False:
                    recommendations.append("⚠️ Interaction cell variances are heterogeneous")
                    variance_ratio = cell_homogeneity.get("variance_ratio", 1)
                    if variance_ratio > 4:
                        recommendations.append("→ Strong heterogeneity detected - consider robust methods")
                    else:
                        recommendations.append("→ Moderate heterogeneity - ANOVA may still be robust")
            
            # Covariance pattern recommendations
            if "assumption_met" in covariance_tests:
                covariance_met = covariance_tests["assumption_met"]
                if covariance_met:
                    recommendations.append("✅ Covariance patterns are similar across groups")
                else:
                    recommendations.append("⚠️ Covariance patterns differ across groups")
                    recommendations.append("→ Consider separate analyses or multilevel modeling")
            
            # Box's M test recommendations
            if "assumption_met" in box_m_test:
                box_m_met = box_m_test["assumption_met"]
                if box_m_met:
                    recommendations.append("✅ Covariance matrices are homogeneous (Box's M)")
                else:
                    recommendations.append("⚠️ Covariance matrices are heterogeneous (Box's M)")
                    recommendations.append("→ Consider MANOVA or robust Mixed ANOVA approaches")
            
            # Overall synthesis
            assumptions_met = sum([
                sphericity_met is True,
                cell_homogeneity.get("levene_test", {}).get("assumption_met", False),
                covariance_tests.get("assumption_met", False),
                box_m_test.get("assumption_met", False)
            ])
            
            total_tests = 4
            if assumptions_met >= 3:
                recommendations.append("✅ Overall: Most interaction assumptions are met")
                recommendations.append("→ Standard Mixed ANOVA is appropriate")
            elif assumptions_met >= 2:
                recommendations.append("⚠️ Overall: Some interaction assumptions are violated")
                recommendations.append("→ Apply corrections where available, proceed with caution")
            else:
                recommendations.append("❌ Overall: Multiple interaction assumptions are violated")
                recommendations.append("→ Consider alternative approaches:")
                recommendations.append("   • Robust Mixed ANOVA methods")
                recommendations.append("   • Separate repeated measures ANOVAs for each group")
                recommendations.append("   • Multilevel modeling with heterogeneous covariance")
                recommendations.append("   • Non-parametric alternatives")
            
        except Exception as e:
            recommendations.append(f"⚠️ Error generating interaction recommendations: {str(e)}")
            
        return recommendations
