import numpy as np
import pandas as pd
import os
from datetime import datetime
from scipy import stats
from lazy_imports import get_pingouin, get_scipy_stats, get_statsmodels_multitest
from stats_functions import UIDialogManager, PostHocFactory, get_pingouin_module, get_output_path, PostHocAnalyzer, PostHocStatistics
from resultsexporter import ResultsExporter
from nonparametricanovas import GLMMTwoWayANOVA, GEERMANOVA, GLMMMixedANOVA, auto_anova_decision

class StatisticalTester:
    @staticmethod
    def _standardize_results(results):
        """
        Ensures that all results dictionaries have the same structure with consistent keys.
        Fills in any missing keys with None or appropriate default values.
        
        Parameters:
        -----------
        results : dict
            The results dictionary to standardize
            
        Returns:
        --------
        dict
            The standardized results dictionary
        """
        standard_keys = {
            "test": None,
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
            "df2": None
        }
        
        # Debug logging
        if 'pairwise_comparisons' in results:
            print(f"DEBUG: Standardizing results with {len(results['pairwise_comparisons'])} pairwise comparisons")
        
        # Copy all existing values first
        standardized = dict(results)
        
        # Fill in any missing top-level keys with default values (without overwriting existing ones)
        for key, default_value in standard_keys.items():
            if key not in standardized:
                standardized[key] = default_value

        # Define the standard keys for each post-hoc comparison entry
        comparison_keys = {
            "group1": "",
            "group2": "",
            "test": "",
            "p_value": None,
            "statistic": None,
            "significant": False,
            "corrected": False,
            "effect_size": None,
            "effect_size_type": None,
            "confidence_interval": (None, None),
            "correction": None  # optional key if a correction method was applied
        }
        
        # If the standardized results contain pairwise comparisons,
        # ensure each comparison has all the expected keys.
        if isinstance(standardized.get("pairwise_comparisons"), list):
            for comp in standardized.get("pairwise_comparisons", []):
                for key, default in comparison_keys.items():
                    if key not in comp:
                        comp[key] = default

        return standardized

    @staticmethod
    def check_normality_and_variance(
        groups, samples, dataset_name=None, progress_text=None, column_name=None, already_transformed=False,
        formula="Value ~ C(Group)", model_type="oneway"
    ):
        """
        Checks normality and homogeneity of variance using model residuals (before and after transformation).
        
        Parameters:
        - model_type: str, one of "oneway", "twoway", "ttest", "rm" (repeated measures)
        - formula: str, formula for statsmodels OLS (e.g., "Value ~ C(Group)" for one-way)
        
        Always fits the specified model and tests residuals for normality using Shapiro-Wilk test.
        Levene test is performed on the raw values for variance homogeneity.
        """
        from scipy.stats import boxcox, boxcox_normmax
        import numpy as np
        import pandas as pd
        from statsmodels.formula.api import ols

        print(f"DEBUG check_normality_and_variance: Starting assumption tests")
        print(f"DEBUG check_normality_and_variance: model_type={model_type}, formula={formula}")
        print(f"DEBUG check_normality_and_variance: Groups: {groups}")

        test_info = {"pre_transformation": {}, "post_transformation": {}, "transformation": None}
        valid_groups = [g for g in groups if g in samples and len(samples[g]) > 0]
        transformed_samples = {g: samples[g].copy() for g in valid_groups}
        test_recommendation = "parametric"

        # Fit model and check residuals normality on raw data
        def make_df(samps):
            if model_type == "twoway":
                # For Two-Way ANOVA, extract factors from group labels like "FactorA=val1, FactorB=val2"
                data_rows = []
                factor_names = []
                for g in valid_groups:
                    for v in samps[g]:
                        # Parse group label to extract factor values
                        if "=" in g and "," in g:
                            parts = [part.strip() for part in g.split(",")]
                            if len(parts) == 2:
                                factor_a_part = parts[0].split("=")
                                factor_b_part = parts[1].split("=")
                                if len(factor_a_part) == 2 and len(factor_b_part) == 2:
                                    factor_a_name, factor_a_val = factor_a_part
                                    factor_b_name, factor_b_val = factor_b_part
                                    # Store original factor names for later
                                    if not factor_names:
                                        factor_names = [factor_a_name.strip(), factor_b_name.strip()]
                                    data_rows.append({
                                        "Group": g,
                                        "Value": v,
                                        "FactorA": factor_a_val.strip(),  # Use simple column names without spaces
                                        "FactorB": factor_b_val.strip()
                                    })
                                    continue
                        # Fallback for malformed group labels
                        data_rows.append({"Group": g, "Value": v})
                
                df = pd.DataFrame(data_rows)
                # Store the original factor names for reference
                df.attrs['original_factor_names'] = factor_names if factor_names else ['FactorA', 'FactorB']
                return df
            else:
                # For other models, use simple Group/Value structure
                return pd.DataFrame([
                    {"Group": g, "Value": v} for g in valid_groups for v in samps[g]
                ])
        df_raw = make_df(samples)
        
        # Adjust formula for Two-Way ANOVA if needed
        adjusted_formula = formula
        if model_type == "twoway" and "FactorA" in df_raw.columns and "FactorB" in df_raw.columns:
            adjusted_formula = "Value ~ C(FactorA) * C(FactorB)"
            print(f"DEBUG SHAPIRO: Adjusted formula for Two-Way ANOVA: {adjusted_formula}")
        
        from scipy import stats
        try:
            # ROBUST FORMULA CREATION: Use actual DataFrame columns, not assumptions
            print(f"DEBUG SHAPIRO: Available columns in df_raw: {list(df_raw.columns)}")
            print(f"DEBUG SHAPIRO: Original formula: {formula}")
            print(f"DEBUG SHAPIRO: Adjusted formula: {adjusted_formula}")
            
            # Create formula based on ACTUAL columns present in DataFrame
            actual_columns = list(df_raw.columns)
            
            if "Value" not in actual_columns:
                print(f"DEBUG SHAPIRO ERROR: 'Value' column missing in DataFrame!")
                stat, pval = None, None
            else:
                # Find factor columns (everything except 'Value')
                factor_columns = [col for col in actual_columns if col != "Value"]
                print(f"DEBUG SHAPIRO: Found factor columns: {factor_columns}")
                
                if not factor_columns:
                    # No factors, use intercept-only model
                    working_formula = "Value ~ 1"
                    print(f"DEBUG SHAPIRO: Using intercept-only model: {working_formula}")
                elif len(factor_columns) == 1:
                    # Single factor
                    working_formula = f"Value ~ C({factor_columns[0]})"
                    print(f"DEBUG SHAPIRO: Using single factor model: {working_formula}")
                elif len(factor_columns) == 2 and model_type == "twoway":
                    # Two factors for two-way ANOVA
                    working_formula = f"Value ~ C({factor_columns[0]}) * C({factor_columns[1]})"
                    print(f"DEBUG SHAPIRO: Using two-factor model: {working_formula}")
                else:
                    # Multiple factors, use first one only to avoid complexity
                    working_formula = f"Value ~ C({factor_columns[0]})"
                    print(f"DEBUG SHAPIRO: Using first factor only: {working_formula}")
                
                # Try to fit the model with actual column names
                model_raw = ols(working_formula, data=df_raw).fit()
                resid_raw = model_raw.resid
                print(f"DEBUG SHAPIRO: Raw residuals length: {len(resid_raw)}, unique values: {len(set(resid_raw))}")
                stat, pval = stats.shapiro(resid_raw) if len(resid_raw) >= 3 and len(set(resid_raw)) > 1 else (None, None)
                print(f"DEBUG SHAPIRO: Pre-transformation Shapiro-Wilk: W={stat}, p={pval}")
                
        except Exception as e:
            print(f"DEBUG SHAPIRO ERROR: Failed pre-transformation Shapiro-Wilk test: {str(e)}")
            print(f"DEBUG SHAPIRO ERROR: DataFrame info - Shape: {df_raw.shape}, Columns: {list(df_raw.columns)}")
            print(f"DEBUG SHAPIRO ERROR: DataFrame head:\n{df_raw.head()}")
            stat, pval = None, None
        test_info["pre_transformation"]["residuals_normality"] = {
            "statistic": stat, "p_value": pval, "is_normal": (pval > 0.05 if pval is not None else False)
        }

        # Levene test on raw data (Brown-Forsythe test using median)
        try:
            data_for_levene = [samples[g] for g in valid_groups]
            print(f"DEBUG BROWN-FORSYTHE: Pre-transformation - Groups: {len(valid_groups)}, Data lengths: {[len(v) for v in data_for_levene]}")
            if len(valid_groups) >= 2 and all(len(v) >= 3 for v in data_for_levene):
                stat, pval = stats.levene(*data_for_levene, center='median')
                has_equal_variance = pval > 0.05
                print(f"DEBUG BROWN-FORSYTHE: Pre-transformation - Statistic: {stat}, p-value: {pval}, Equal variance: {has_equal_variance}")
            else:
                stat, pval, has_equal_variance = None, None, False
                print(f"DEBUG BROWN-FORSYTHE: Pre-transformation - Insufficient data for test")
        except Exception as e:
            print(f"DEBUG BROWN-FORSYTHE ERROR: Pre-transformation failed: {str(e)}")
            stat, pval, has_equal_variance = None, None, False
        test_info["pre_transformation"]["variance"] = {
            "statistic": stat, "p_value": pval, "equal_variance": has_equal_variance
        }

        need_transform = not (
            test_info["pre_transformation"]["residuals_normality"]["is_normal"]
            and test_info["pre_transformation"]["variance"]["equal_variance"]
        )

        # Transformation if needed
        if need_transform:
            if already_transformed:
                test_info["transformation"] = "No further"
                return transformed_samples, "non_parametric", test_info

            transformation_type = None
            try:
                transformation_type = UIDialogManager.select_transformation_dialog(
                    parent=None, progress_text=progress_text, column_name=column_name
                )
            except Exception:
                transformation_type = "log10"
            if not transformation_type:
                transformation_type = "log10"
            test_info["transformation"] = transformation_type

            # Apply transformation
            for group in valid_groups:
                values = samples[group]
                min_val = min(values)
                shift = -min_val + 1 if min_val <= 0 else 0
                if transformation_type == "log10":
                    transformed_samples[group] = [np.log10(v + shift) for v in values]
                elif transformation_type == "boxcox":
                    shifted = [v + shift for v in values]
                    try:
                        lambda_val = boxcox_normmax(shifted)
                        transformed_samples[group] = list(boxcox(shifted, lambda_val))
                        test_info["boxcox_lambda"] = lambda_val
                    except Exception as e:
                        test_info["transformation_error"] = str(e)
                        transformed_samples[group] = [np.log10(v + shift) for v in values]
                elif transformation_type == "arcsin_sqrt":
                    max_val = max(values)
                    # Scale to 0-1 if needed
                    if min_val < 0 or max_val > 1:
                        scaled = [(v - min_val) / (max_val - min_val) for v in values]
                    else:
                        scaled = values
                    transformed_samples[group] = [np.arcsin(np.sqrt(v)) for v in scaled]

        # Fit model and check residuals normality on transformed data
        df_tr = make_df(transformed_samples)
        
        # Use the same robust formula approach for transformed data
        try:
            # ROBUST FORMULA FOR TRANSFORMED DATA: Use actual DataFrame columns
            print(f"DEBUG SHAPIRO: Available columns in df_tr: {list(df_tr.columns)}")
            
            # Create formula based on ACTUAL columns present in transformed DataFrame
            actual_columns_tr = list(df_tr.columns)
            
            if "Value" not in actual_columns_tr:
                print(f"DEBUG SHAPIRO ERROR: 'Value' column missing in transformed DataFrame!")
                stat2, pval2 = None, None
            else:
                # Find factor columns (everything except 'Value')
                factor_columns_tr = [col for col in actual_columns_tr if col != "Value"]
                print(f"DEBUG SHAPIRO: Found factor columns in transformed data: {factor_columns_tr}")
                
                if not factor_columns_tr:
                    # No factors, use intercept-only model
                    working_formula_tr = "Value ~ 1"
                    print(f"DEBUG SHAPIRO: Using intercept-only model for transformed: {working_formula_tr}")
                elif len(factor_columns_tr) == 1:
                    # Single factor
                    working_formula_tr = f"Value ~ C({factor_columns_tr[0]})"
                    print(f"DEBUG SHAPIRO: Using single factor model for transformed: {working_formula_tr}")
                elif len(factor_columns_tr) == 2 and model_type == "twoway":
                    # Two factors for two-way ANOVA
                    working_formula_tr = f"Value ~ C({factor_columns_tr[0]}) * C({factor_columns_tr[1]})"
                    print(f"DEBUG SHAPIRO: Using two-factor model for transformed: {working_formula_tr}")
                else:
                    # Multiple factors, use first one only to avoid complexity
                    working_formula_tr = f"Value ~ C({factor_columns_tr[0]})"
                    print(f"DEBUG SHAPIRO: Using first factor only for transformed: {working_formula_tr}")
                
                # Try to fit the model with actual column names
                model_tr = ols(working_formula_tr, data=df_tr).fit()
                resid_tr = model_tr.resid
                print(f"DEBUG SHAPIRO: Transformed residuals length: {len(resid_tr)}, unique values: {len(set(resid_tr))}")
                stat2, pval2 = stats.shapiro(resid_tr) if len(resid_tr) >= 3 and len(set(resid_tr)) > 1 else (None, None)
                print(f"DEBUG SHAPIRO: Post-transformation Shapiro-Wilk: W={stat2}, p={pval2}")
                
        except Exception as e:
            print(f"DEBUG SHAPIRO ERROR: Failed post-transformation Shapiro-Wilk test: {str(e)}")
            print(f"DEBUG SHAPIRO ERROR: Transformed DataFrame info - Shape: {df_tr.shape}, Columns: {list(df_tr.columns)}")
            print(f"DEBUG SHAPIRO ERROR: Transformed DataFrame head:\n{df_tr.head()}")
            stat2, pval2 = None, None
        test_info["post_transformation"]["residuals_normality"] = {
            "statistic": stat2, "p_value": pval2, "is_normal": (pval2 > 0.05 if pval2 is not None else False)
        }

        # Levene test on transformed data
        try:
            data_for_levene_tr = [transformed_samples[g] for g in valid_groups]
            print(f"DEBUG BROWN-FORSYTHE: Post-transformation - Groups: {len(valid_groups)}, Data lengths: {[len(v) for v in data_for_levene_tr]}")
            if len(valid_groups) >= 2 and all(len(v) >= 3 for v in data_for_levene_tr):
                stat_tr, pval_tr = stats.levene(*data_for_levene_tr, center='median')
                has_equal_variance_tr = pval_tr > 0.05
                print(f"DEBUG BROWN-FORSYTHE: Post-transformation - Statistic: {stat_tr}, p-value: {pval_tr}, Equal variance: {has_equal_variance_tr}")
            else:
                stat_tr, pval_tr, has_equal_variance_tr = None, None, False
                print(f"DEBUG BROWN-FORSYTHE: Post-transformation - Insufficient data for test")
        except Exception as e:
            print(f"DEBUG BROWN-FORSYTHE ERROR: Post-transformation failed: {str(e)}")
            stat_tr, pval_tr, has_equal_variance_tr = None, None, False
        test_info["post_transformation"]["variance"] = {
            "statistic": stat_tr, "p_value": pval_tr, "equal_variance": has_equal_variance_tr
        }

        # Recommend test based on assumptions
        post_norm = test_info["post_transformation"]["residuals_normality"]["is_normal"]
        post_var = test_info["post_transformation"]["variance"]["equal_variance"]

        if post_norm and post_var:
            test_recommendation = "parametric"
        elif post_norm and not post_var and model_type == "ttest" and len(valid_groups) == 2:
            test_recommendation = "parametric"  # Welch-Test
            test_info["note"] = "Residuen normal, Varianzen ungleich – Welch-Test wird verwendet"
        elif post_norm and not post_var and model_type == "oneway" and len(valid_groups) > 2:
            test_recommendation = "parametric"  # Welch-ANOVA
            test_info["note"] = "Residuen normal, Varianzen ungleich – Welch-ANOVA wird verwendet"
        elif post_norm and not post_var and model_type in ["twoway", "mixed", "rm"]:
            test_recommendation = "parametric"  # Advanced ANOVAs can often handle unequal variances
            test_info["note"] = f"Residuen normal, Varianzen ungleich – {model_type.upper()}-ANOVA wird trotzdem verwendet (robust gegen Varianzheterogenität)"
        elif post_norm and not post_var and len(valid_groups) == 2:
            test_recommendation = "parametric"  # Welch-Test
            test_info["note"] = "Normal distribution but unequal variances – Welch's t-test will be used."
        elif post_norm and not post_var and len(valid_groups) > 2:
            test_recommendation = "parametric"  # Welch-ANOVA
            test_info["note"] = "Normal distribution but unequal variances – Welch's ANOVA will be used."
        else:
            test_recommendation = "non_parametric"

        print(f"DEBUG check_normality_and_variance: Final test_info structure:")
        print(f"  Pre-transformation normality: {test_info['pre_transformation'].get('residuals_normality', 'Missing')}")
        print(f"  Pre-transformation variance: {test_info['pre_transformation'].get('variance', 'Missing')}")
        print(f"  Post-transformation normality: {test_info['post_transformation'].get('residuals_normality', 'Missing')}")
        print(f"  Post-transformation variance: {test_info['post_transformation'].get('variance', 'Missing')}")
        print(f"  Test recommendation: {test_recommendation}")

        return transformed_samples, test_recommendation, test_info
    
    @staticmethod
    def perform_statistical_test(
    groups, transformed_samples, original_samples, 
    dependent=False, test_recommendation="parametric", alpha=0.05, test_info=None
    ):
        """
        Robustly performs the statistical test (t-test, ANOVA, Mann-Whitney, Kruskal-Wallis, posthoc, etc.).
        Catches all typical sources of error and always returns a meaningful result dict.
        Important: For parametric tests, transformed data is used,
        for non-parametric tests, the original data is used.
        """
        results = {
            "test": None,
            "p_value": None,
            "statistic": None,
            "posthoc_test": None,
            "pairwise_comparisons": [],
            "descriptive": {},
            "descriptive_transformed": {},
            "alpha": alpha,
            "null_hypothesis": "The means/medians of all groups are equal.",
            "alternative_hypothesis": "At least one mean/median differs from the others.",
            "confidence_interval": None,
            "ci_level": 0.95,
            "power": None
        }

        valid_groups = [g for g in groups if g in transformed_samples and len(transformed_samples[g]) > 0]

        # Always compute descriptive stats for both original and transformed data
        results["descriptive"] = {g: StatisticalTester._compute_descriptive_stats(original_samples[g]) for g in valid_groups}
        
        # IMPORTANT: Always include transformed descriptive stats when transformation was performed
        if any(original_samples[g] != transformed_samples[g] for g in valid_groups):
            results["descriptive_transformed"] = {g: StatisticalTester._compute_descriptive_stats(transformed_samples[g]) for g in valid_groups}
        
        # Store raw data for both original and transformed values
        results["raw_data"] = {g: original_samples[g].copy() for g in valid_groups}
        if original_samples != transformed_samples:
            results["raw_data_transformed"] = {g: transformed_samples[g].copy() for g in valid_groups}

        if len(valid_groups) == 0:
            return StatisticalTester._stat_test_no_valid_groups(results)

        if len(valid_groups) == 1:
            return StatisticalTester._stat_test_one_group(results, valid_groups, original_samples, transformed_samples)

        # Descriptive statistics for all groups
        results["descriptive"] = {g: StatisticalTester._compute_descriptive_stats(original_samples[g]) for g in valid_groups}
        if original_samples != transformed_samples:
            results["descriptive_transformed"] = {g: StatisticalTester._compute_descriptive_stats(transformed_samples[g]) for g in valid_groups}

        samples_to_use = transformed_samples if test_recommendation == "parametric" else original_samples

        if len(valid_groups) == 2:
            return StatisticalTester._stat_test_two_groups(
                results, valid_groups, samples_to_use, original_samples, dependent, test_recommendation, alpha, test_info=test_info
            )

        return StatisticalTester._stat_test_multi_groups(
        results, valid_groups, samples_to_use, dependent, test_recommendation, alpha, test_info=test_info
        )

    @staticmethod
    def _stat_test_no_valid_groups(results):
        results["test"] = "No test possible"
        results["error"] = "No valid groups found"
        return StatisticalTester._standardize_results(results)

    @staticmethod
    def _stat_test_one_group(results, valid_groups, original_samples, transformed_samples):
        g = valid_groups[0]
        results["descriptive"] = {
            g: StatisticalTester._compute_descriptive_stats(original_samples[g])
        }
        if original_samples[g] != transformed_samples[g]:
            results["descriptive_transformed"] = {
                g: StatisticalTester._compute_descriptive_stats(transformed_samples[g])
            }
        results["test"] = "Descriptive statistics only"
        return StatisticalTester._standardize_results(results)

    @staticmethod
    def _stat_test_two_groups(results, valid_groups, samples_to_use, original_samples, dependent, test_recommendation, alpha, test_info=None):
        g1, g2 = valid_groups
        data1, data2 = samples_to_use[g1], samples_to_use[g2]
        try:
            if dependent:
                if test_recommendation == "parametric":
                    return StatisticalTester._paired_ttest(results, g1, g2, data1, data2, alpha)
                else:
                    return StatisticalTester._wilcoxon_test(results, g1, g2, data1, data2, alpha)
            else:
                if test_recommendation == "parametric":
                    # Check variance homogeneity for Welch test
                    equal_var = True
                    if test_info is not None:
                        if test_info.get("transformation"):
                            equal_var = test_info.get("variance_test", {}).get("transformed", {}).get("equal_variance", True)
                        else:
                            equal_var = test_info.get("variance_test", {}).get("equal_variance", True)
                    return StatisticalTester._independent_ttest(results, g1, g2, data1, data2, alpha, equal_var=equal_var)
        except Exception as e:
            results["test"] = "Error during test"
            results["error"] = str(e)
            results["posthoc_test"] = "Not performed (error in main test)"
            results["pairwise_comparisons"] = []
            return StatisticalTester._standardize_results(results)

    @staticmethod
    def _paired_ttest(results, g1, g2, data1, data2, alpha):
        statistic, p_value = stats.ttest_rel(data1, data2)
        test_name = "Paired t-test"
        diff = np.array(data1) - np.array(data2)
        cohen_d = np.mean(diff) / np.std(diff, ddof=1)
        results["effect_size"] = cohen_d
        results["effect_size_type"] = "cohen_d"
        n = len(diff)
        stderr = np.std(diff, ddof=1) / np.sqrt(n)
        from scipy.stats import t
        ci = t.interval(0.95, n-1, loc=np.mean(diff), scale=stderr)
        results["confidence_interval"] = ci
        try:
            from statsmodels.stats.power import TTestPower
            effect_size = abs(cohen_d)
            power_analysis = TTestPower()
            results["power"] = float(power_analysis.power(effect_size=effect_size, nobs=n, alpha=alpha))
        except Exception as e:
            results["power"] = None
        results["test"] = test_name
        results["statistic"] = statistic
        results["p_value"] = p_value
        results["pairwise_comparisons"] = [{
            "group1": g1, "group2": g2, "test": test_name,
            "p_value": p_value, "statistic": statistic,
            "significant": p_value < alpha, "corrected": False,
            "effect_size": results.get("effect_size"),
            "effect_size_type": results.get("effect_size_type"),
            "confidence_interval": results.get("confidence_interval"),
            "power": results.get("power")
        }]
        return StatisticalTester._standardize_results(results)

    @staticmethod
    def _wilcoxon_test(results, g1, g2, data1, data2, alpha):
        statistic, p_value = stats.wilcoxon(data1, data2)
        test_name = "Wilcoxon test"
        n = len(data1)
        r = statistic / (n * (n + 1) / 2)
        results["effect_size"] = r
        results["effect_size_type"] = "r"
        results["confidence_interval"] = (None, None)
        try:
            from statsmodels.stats.power import TTestPower
            effect_size_corrected = r * 0.955
            power_analysis = TTestPower()
            results["power"] = float(power_analysis.power(effect_size=effect_size_corrected, nobs=n, alpha=alpha))
        except Exception as e:
            results["power"] = None
        results["test"] = test_name
        results["statistic"] = statistic
        results["p_value"] = p_value
        results["pairwise_comparisons"] = [{
            "group1": g1, "group2": g2, "test": test_name,
            "p_value": p_value, "statistic": statistic,
            "significant": p_value < alpha, "corrected": False,
            "effect_size": results.get("effect_size"),
            "effect_size_type": results.get("effect_size_type"),
            "confidence_interval": results.get("confidence_interval"),
            "power": results.get("power")
        }]
        return StatisticalTester._standardize_results(results)
    
    @staticmethod
    def _independent_ttest(results, g1, g2, data1, data2, alpha, equal_var=True):
        statistic, p_value = stats.ttest_ind(data1, data2, equal_var=equal_var)
        test_name = "t-test (independent)"
        if not equal_var:
            test_name = "Welch's t-test (unequal variances)"
        n1, n2 = len(data1), len(data2)
        s1, s2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
        
        # Different calculations for equal vs unequal variances
        if equal_var:
            # Pooled standard deviation for equal variances
            s_pooled = np.sqrt(((n1-1)*s1 + (n2-1)*s2) / (n1+n2-2))
            cohen_d = (np.mean(data1) - np.mean(data2)) / s_pooled
            stderr_diff = s_pooled * np.sqrt(1/n1 + 1/n2)
            df = n1 + n2 - 2
        else:
            # For Welch's t-test (unequal variances)
            s_pooled = np.sqrt((s1 + s2) / 2)  # Simple average for Cohen's d
            cohen_d = (np.mean(data1) - np.mean(data2)) / s_pooled
            stderr_diff = np.sqrt(s1/n1 + s2/n2)
            # Welch-Satterthwaite degrees of freedom
            df = (s1/n1 + s2/n2)**2 / ((s1/n1)**2/(n1-1) + (s2/n2)**2/(n2-1))
        
        results["effect_size"] = cohen_d
        results["effect_size_type"] = "cohen_d"
        mean_diff = np.mean(data1) - np.mean(data2)
        from scipy.stats import t
        ci = t.interval(0.95, df, loc=mean_diff, scale=stderr_diff)
        results["confidence_interval"] = ci
        try:
            from statsmodels.stats.power import TTestIndPower
            effect_size = abs(cohen_d)
            power_analysis = TTestIndPower()
            results["power"] = float(power_analysis.power(effect_size=effect_size, nobs1=n1, ratio=n2/n1, alpha=alpha))
        except Exception as e:
            results["power"] = None
        results["test"] = test_name
        results["statistic"] = statistic
        results["p_value"] = p_value
        results["pairwise_comparisons"] = [{
            "group1": g1, "group2": g2, "test": test_name,
            "p_value": p_value, "statistic": statistic,
            "significant": p_value < alpha, "corrected": False,
            "effect_size": results.get("effect_size"),
            "effect_size_type": results.get("effect_size_type"),
            "confidence_interval": results.get("confidence_interval"),
            "power": results.get("power")
        }]
        return StatisticalTester._standardize_results(results)

    @staticmethod
    def _mannwhitney_test(results, g1, g2, data1, data2, alpha):
        statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        test_name = "Mann-Whitney-U"
        n1, n2 = len(data1), len(data2)
        u = statistic
        mean_u = n1 * n2 / 2
        std_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        z = (u - mean_u) / std_u if std_u > 0 else 0
        r = abs(z) / np.sqrt(n1 + n2)
        results["effect_size"] = r
        results["effect_size_type"] = "r"
        results["confidence_interval"] = (None, None)
        try:
            from statsmodels.stats.power import TTestIndPower
            effect_size_corrected = r * 0.955
            power_analysis = TTestIndPower()
            results["power"] = float(power_analysis.power(effect_size=effect_size_corrected, nobs1=n1, ratio=n2/n1, alpha=alpha))
        except Exception as e:
            results["power"] = None
        results["test"] = test_name
        results["statistic"] = statistic
        results["p_value"] = p_value
        results["pairwise_comparisons"] = [{
            "group1": g1, "group2": g2, "test": test_name,
            "p_value": p_value, "statistic": statistic,
            "significant": p_value < alpha, "corrected": False,
            "effect_size": results.get("effect_size"),
            "effect_size_type": results.get("effect_size_type"),
            "confidence_interval": results.get("confidence_interval"),
            "power": results.get("power")
        }]
        return StatisticalTester._standardize_results(results)

    @staticmethod
    def _stat_test_multi_groups(results, valid_groups, samples_to_use, dependent, test_recommendation, alpha, test_info=None, df=None, dv=None, subject=None, within=None):
        # Welch ANOVA: If explicitly requested OR if parametric but variances are unequal
        welch_condition = (
            test_recommendation == "welch"  # Direct request for Welch ANOVA
            or (
                test_recommendation == "parametric" 
                and test_info is not None
                and (
                    # Check if transformed data has unequal variance (new structure)
                    (test_info.get("transformation") and not test_info.get("post_transformation", {}).get("variance", {}).get("equal_variance", True))
                    # Or if original data has unequal variance (when no transformation) (new structure)
                    or (not test_info.get("transformation") and not test_info.get("pre_transformation", {}).get("variance", {}).get("equal_variance", True))
                )
                # Ensure we're still normally distributed (new structure)
                and (
                    (test_info.get("transformation") and test_info.get("post_transformation", {}).get("residuals_normality", {}).get("is_normal", False))
                    or (not test_info.get("transformation") and test_info.get("pre_transformation", {}).get("residuals_normality", {}).get("is_normal", False))
                )
            )
        )
        
        print(f"DEBUG WELCH CHECK: test_recommendation = {test_recommendation}")
        if test_info and test_info.get("transformation"):
            print(f"DEBUG WELCH CHECK: transformed data normality = {test_info.get('post_transformation', {}).get('residuals_normality', {}).get('is_normal', False)}")
            print(f"DEBUG WELCH CHECK: transformed variance equal = {test_info.get('post_transformation', {}).get('variance', {}).get('equal_variance', True)}")
        elif test_info:
            print(f"DEBUG WELCH CHECK: original data normality = {test_info.get('pre_transformation', {}).get('residuals_normality', {}).get('is_normal', False)}")
            print(f"DEBUG WELCH CHECK: original variance equal = {test_info.get('pre_transformation', {}).get('variance', {}).get('equal_variance', True)}")
        else:
            print(f"DEBUG WELCH CHECK: test_info is None")
        print(f"DEBUG WELCH CHECK: welch_condition = {welch_condition}")
        
        if welch_condition:
            # Use Welch ANOVA on transformed data
            print("DEBUG WELCH CHECK: Using Welch ANOVA!")
            welch_result = StatisticalTester._welch_anova_test(results, valid_groups, samples_to_use, alpha)
            if welch_result is not None:
                return welch_result
            else:
                print("DEBUG WELCH CHECK: Welch ANOVA returned None, falling back to regular ANOVA")
                # Fall through to regular ANOVA
        
        try:
            if dependent and len(valid_groups) > 2:
                results["test"] = "Repeated Measures ANOVA not supported in simple pipeline"
                results["error"] = "Please use perform_advanced_test for RM-ANOVA."
                return StatisticalTester._standardize_results(results)
            else:
                if test_recommendation == "parametric":
                    try:
                        print("DEBUG: Attempting to use Pingouin for ANOVA...")
                        pg = get_pingouin_module()
                        
                        # Create a completely fresh DataFrame with proper numeric types
                        data_for_anova = []
                        
                        # Use standard column names that won't confuse Pingouin
                        for i, group in enumerate(valid_groups):
                            values = samples_to_use[group]
                            for val in values:
                                data_for_anova.append({
                                    'dependent_var': float(val),
                                    'group_var': i  # Integer group identifier
                                })
                        
                        # Convert to DataFrame with explicit dtypes
                        df_pg = pd.DataFrame(data_for_anova)
                        df_pg['dependent_var'] = df_pg['dependent_var'].astype(float)
                        df_pg['group_var'] = df_pg['group_var'].astype('category')
                        
                        print(f"DEBUG: New df_pg shape = {df_pg.shape}")
                        print(f"DEBUG: New groups = {df_pg['group_var'].unique()}")
                        print(f"DEBUG: New dependent_var min={df_pg['dependent_var'].min()}, max={df_pg['dependent_var'].max()}")
                        
                        # Run ANOVA with robust column names
                        aov = pg.anova(data=df_pg, dv='dependent_var', between='group_var', detailed=True)
                        results["anova_table"] = aov.copy()
                        
                        # Extract results more carefully
                        print(f"DEBUG: ANOVA table columns: {list(aov.columns)}")
                        print(f"DEBUG: ANOVA table index/rows: {list(aov.index)}")
                        
                        # Get the between-groups row (should be the first row)
                        row_between = aov.iloc[0]
                        
                        # Get residuals row (should be the second row)
                        if len(aov) > 1:
                            row_residual = aov.iloc[1]
                            df2 = row_residual['DF']  # This should be numeric already
                            results["df2"] = int(df2) if pd.notnull(df2) else None
                        else:
                            # Fallback: calculate residual df from total observations - groups
                            total_observations = sum(len(samples_to_use[g]) for g in valid_groups)
                            results["df2"] = total_observations - len(valid_groups)
                        
                        # Extract the main ANOVA results
                        results["test"] = "One-way ANOVA (Pingouin)"
                        results["statistic"] = float(row_between["F"])
                        results["p_value"] = float(row_between["p-unc"])
                        results["effect_size"] = float(row_between["np2"])
                        results["effect_size_type"] = "partial_eta_squared"
                        results["confidence_interval"] = (None, None)
                        results["power"] = None
                        
                        # Map group identifiers back to names for easier reading
                        group_map = {i: group for i, group in enumerate(valid_groups)}
                        
                        print("DEBUG: Successfully used Pingouin for ANOVA!")
                        
                    except Exception as e:
                        print(f"DEBUG: Pingouin failed with error: {str(e)}")
                        # Detailed traceback for better debugging
                        import traceback
                        print(f"DEBUG: Full traceback: {traceback.format_exc()}")
                        # Fallback to scipy if Pingouin fails
                        teststat, pval = stats.f_oneway(*[samples_to_use[g] for g in valid_groups])
                        results["test"] = "One-way ANOVA (scipy fallback)"
                        results["p_value"] = pval
                        results["statistic"] = teststat
                        all_data = np.concatenate([samples_to_use[g] for g in valid_groups])
                        grand_mean = np.mean(all_data)
                        ss_between = sum(len(samples_to_use[g]) * (np.mean(samples_to_use[g]) - grand_mean) ** 2 for g in valid_groups)
                        ss_total = sum((x - grand_mean) ** 2 for x in all_data)
                        eta_sq = ss_between / ss_total if ss_total > 0 else None
                        results["effect_size"] = eta_sq
                        results["effect_size_type"] = "eta_squared"
                        results["anova_table"] = None
                        try:
                            from statsmodels.stats.power import FTestAnovaPower
                            k = len(valid_groups)
                            n = sum(len(samples_to_use[g]) for g in valid_groups)
                            eta_sq = results.get("effect_size", 0)
                            f2 = eta_sq / (1 - eta_sq) if eta_sq < 1 else 0
                            power_analysis = FTestAnovaPower()
                            results["power"] = float(power_analysis.power(
                                effect_size=f2, k_groups=k, nobs=n, alpha=alpha
                            ))
                        except Exception:
                            results["power"] = None
                        results["confidence_interval"] = (None, None)
                else:
                    teststat, pval = stats.kruskal(*[samples_to_use[g] for g in valid_groups])
                    results["test"] = "Kruskal-Wallis test"
                    results["p_value"] = pval
                    results["statistic"] = teststat
                    n = sum(len(samples_to_use[g]) for g in valid_groups)
                    h = teststat
                    k = len(valid_groups)
                    epsilon_sq = (h - k + 1) / (n - k) if n > k else None
                    results["effect_size"] = epsilon_sq
                    results["effect_size_type"] = "epsilon_squared"
                    results["anova_table"] = None
                    try:
                        from statsmodels.stats.power import FTestAnovaPower
                        k = len(valid_groups)
                        n = sum(len(samples_to_use[g]) for g in valid_groups)
                        epsilon_sq = results.get("effect_size", 0)
                        f2_approx = (epsilon_sq / (1 - epsilon_sq)) * 0.955 if epsilon_sq < 1 else 0
                        power_analysis = FTestAnovaPower()
                        results["power"] = float(power_analysis.power(
                            effect_size=f2_approx, k_groups=k, nobs=n, alpha=alpha
                        ))
                    except Exception:
                        results["power"] = None
                    results["confidence_interval"] = (None, None)
            if test_recommendation == "parametric" and results.get("p_value") is not None and results["p_value"] < alpha:
                # Use the unified post-hoc testing approach instead of separate dialogs
                print("DEBUG: Significant parametric test, calling perform_refactored_posthoc_testing")
                posthoc_results = StatisticalTester.perform_refactored_posthoc_testing(
                    valid_groups, samples_to_use, test_recommendation="parametric", alpha=alpha,
                    posthoc_choice=None  # Let the function show the dialog
                )
                
                if posthoc_results:
                    results["posthoc_test"] = posthoc_results.get("posthoc_test")
                    results["pairwise_comparisons"] = posthoc_results.get("pairwise_comparisons", [])
                else:
                    results["posthoc_test"] = "No post-hoc tests performed"
                    results["pairwise_comparisons"] = []
            elif test_recommendation == "non_parametric" and results.get("p_value") is not None and results["p_value"] < alpha:
                # Let the perform_refactored_posthoc_testing function handle the dialog selection
                print("DEBUG: Significant non-parametric test, calling perform_refactored_posthoc_testing without preset posthoc_choice")
                posthoc_results = StatisticalTester.perform_refactored_posthoc_testing(
                    valid_groups, samples_to_use, test_recommendation,
                    alpha=alpha, posthoc_choice=None  # Let the function show the dialog
                )
                
                if posthoc_results:
                    results["posthoc_test"] = posthoc_results.get("posthoc_test")
                    results["pairwise_comparisons"] = posthoc_results.get("pairwise_comparisons", [])
                else:
                    results["posthoc_test"] = "No post-hoc tests performed (error or unsupported test)"
                    results["pairwise_comparisons"] = []
        except Exception as e:
            results["test"] = "Error during test"
            results["p_value"] = None
            results["statistic"] = None
            results["effect_size"] = None
            results["effect_size_type"] = None
            results["confidence_interval"] = None
            results["power"] = None
            results["posthoc_test"] = "Not performed (error in main test)"
            results["pairwise_comparisons"] = []
            results["error"] = str(e)
            for key in ["test", "p_value", "statistic", "effect_size", "effect_size_type"]:
                if key not in results or results[key] is None:
                    results[key] = None
            return results
        return StatisticalTester._standardize_results(results)

    @staticmethod
    def _welch_anova_test(results, valid_groups, samples_to_use, alpha):
        """
        Perform Welch's ANOVA (for unequal variances).
        
        Parameters:
        -----------
        results : dict
            Results dictionary to store the output
        valid_groups : list
            List of group identifiers
        samples_to_use : list or dict
            Data samples for each group
        alpha : float
            Significance level
            
        Returns:
        --------
        dict
            Updated results dictionary with Welch ANOVA results
        """
        try:
            pg = get_pingouin_module()
            
            # Prepare data for pingouin
            data_for_pingouin = []
            
            # Handle both list and dict format for samples_to_use
            if isinstance(samples_to_use, dict):
                for group in valid_groups:
                    if group in samples_to_use:
                        for value in samples_to_use[group]:
                            data_for_pingouin.append({
                                'value': float(value),
                                'group': group
                            })
            else:  # Assume it's a list of lists, matching valid_groups order
                for i, (group, samples) in enumerate(zip(valid_groups, samples_to_use)):
                    for value in samples:
                        data_for_pingouin.append({
                            'value': float(value),
                            'group': group
                        })
            
            # Create DataFrame for pingouin
            df_pg = pd.DataFrame(data_for_pingouin)
            
            # Skip test if not enough groups or data
            if len(df_pg['group'].unique()) < 2:
                results["test"] = "Welch's ANOVA (failed)"
                results["error"] = "At least two groups with data are required for Welch's ANOVA"
                results["effects"] = []
                return results
                
            print(f"DEBUG: Welch ANOVA - df_pg shape = {df_pg.shape}")
            print(f"DEBUG: Welch ANOVA - groups = {df_pg['group'].unique()}")
            
            # Perform Welch's ANOVA using pingouin
            welch_results = pg.welch_anova(data=df_pg, dv='value', between='group')
            
            # Extract results
            F_value = float(welch_results['F'].iloc[0])
            p_value = float(welch_results['p-unc'].iloc[0])
            df1 = float(welch_results['ddof1'].iloc[0])
            df2 = float(welch_results['ddof2'].iloc[0])
            
            # Store results in the format expected by the calling code
            results["test"] = "Welch's ANOVA"
            results["test_type"] = "parametric"
            results["error"] = None
            results["effects"] = [{
                "name": "between",
                "F": F_value,
                "p": p_value,
                "significant": p_value < alpha,
                "df_num": df1,
                "df_den": df2,
                "effect_size": None,  # Could calculate eta-squared if needed
                "ci_lower": None,
                "ci_upper": None,
                "posthoc_tests": None  # We'll add post-hoc separately if needed
            }]
            
            # Add test-level results 
            results["p_value"] = p_value
            results["statistic"] = F_value
            results["df1"] = df1
            results["df2"] = df2
            results["anova_table"] = welch_results
            
            # Post-hoc: Dunnett's T3 if significant
            if p_value < alpha:
                try:
                    print("DEBUG: Performing Dunnett's T3 post-hoc tests")
                    pairwise = StatisticalTester._perform_dunnett_t3_posthoc(
                        valid_groups, samples_to_use, alpha=alpha
                    )
                    results["posthoc_test"] = "Dunnett's T3"
                    results["pairwise_comparisons"] = pairwise
                except Exception as ph_err:
                    print(f"DEBUG: Post-hoc Dunnett's T3 failed: {str(ph_err)}")
                    results["posthoc_test"] = "No post-hoc tests performed (error)"
                    results["pairwise_comparisons"] = []
            else:
                results["posthoc_test"] = "No post-hoc tests performed (not significant)"
                results["pairwise_comparisons"] = []
            
            return results
        except Exception as e:
            import traceback
            print(f"DEBUG: Welch ANOVA failed with error: {str(e)}")
            print(f"DEBUG: Traceback: {traceback.format_exc()}")
            results["test"] = "Welch's ANOVA (failed)"
            results["test_type"] = "parametric"
            results["error"] = str(e)
            results["effects"] = []
            results["p_value"] = None
            results["statistic"] = None
            results["df1"] = None
            results["df2"] = None
            return results

    @staticmethod
    def _perform_dunnett_t3_posthoc(valid_groups, samples_to_use, alpha=0.05):
        """
        Performs Dunnett's T3 post-hoc test for unequal variances.
        
        Parameters:
        -----------
        valid_groups : list
            List of group names
        samples_to_use : dict
            Dictionary with group names as keys and lists of values as values
        alpha : float
            Significance level
            
        Returns:
        --------
        list
            List of pairwise comparison results
        """
        try:
            import numpy as np
            from scipy import stats
            from itertools import combinations
            
            pairwise_results = []
            
            # Create all possible pairs of groups
            pairs = list(combinations(valid_groups, 2))
            
            for group1, group2 in pairs:
                # Extract data for the two groups
                data1 = np.array(samples_to_use[group1])
                data2 = np.array(samples_to_use[group2])
                
                # Sample sizes
                n1 = len(data1)
                n2 = len(data2)
                
                # Calculate means
                mean1 = np.mean(data1)
                mean2 = np.mean(data2)
                
                # Calculate variances (with correction for sample size)
                var1 = np.var(data1, ddof=1)
                var2 = np.var(data2, ddof=1)
                
                # Standard error of the difference
                se = np.sqrt(var1/n1 + var2/n2)
                
                # T-statistic
                t_stat = (mean1 - mean2) / se
                
                # Degrees of freedom using Welch-Satterthwaite equation
                df_num = (var1/n1 + var2/n2)**2
                df_den = (var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1)
                df = df_num / df_den
                
                # Number of groups for the critical value calculation
                k = len(valid_groups)
                
                # Get critical value from studentized range distribution and transform for SMM
                try:
                    # For p-value: use studentized range distribution
                    # First, convert t-statistic to q-statistic format
                    q_stat = abs(t_stat) * np.sqrt(2)
                    
                    # Calculate p-value from studentized range distribution
                    # We use 1-cdf because we want P(q > |q_stat|)
                    p_value = 1 - stats.studentized_range.cdf(q_stat, k, df)
                    
                    # Get critical value
                    q_crit = stats.studentized_range.ppf(1-alpha, k, df)
                    crit_value = q_crit / np.sqrt(2)
                    
                    # Determine significance
                    significant = abs(t_stat) > crit_value
                    
                    # Calculate effect size (Cohen's d with pooled SD)
                    cohens_d = (mean1 - mean2) / np.sqrt((var1 + var2) / 2)
                    
                    # Calculate confidence interval
                    # For the CI, we use the critical value from the studentized range distribution
                    ci_lower = (mean1 - mean2) - crit_value * se
                    ci_upper = (mean1 - mean2) + crit_value * se
                    
                    # Add the result to our list
                    pairwise_results.append({
                        "group1": group1,
                        "group2": group2,
                        "test": "Dunnett's T3",
                        "statistic": float(t_stat),
                        "p_value": float(p_value),
                        "significant": significant,
                        "corrected": True,
                        "effect_size": float(cohens_d),
                        "effect_size_type": "cohen_d",
                        "confidence_interval": (float(ci_lower), float(ci_upper))
                    })
                    
                except Exception as err:
                    print(f"WARNING: Error calculating critical value: {str(err)}")
                    # Fallback: use t-distribution (conservative)
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
                    significant = p_value < (alpha / len(pairs))  # Bonferroni correction
                    
                    pairwise_results.append({
                        "group1": group1,
                        "group2": group2,
                        "test": "Dunnett's T3 (t approximation)",
                        "statistic": float(t_stat),
                        "p_value": float(p_value),
                        "significant": significant,
                        "corrected": True,
                        "effect_size": None,
                        "effect_size_type": None,
                        "confidence_interval": (None, None)
                    })
            
            return pairwise_results
        
        except Exception as e:
            print(f"ERROR in Dunnett's T3 post-hoc: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    @staticmethod
    def _compute_descriptive_stats(values):
        if not values or len(values) == 0:
            return {"n": 0, "mean": None, "median": None, "std": None, "stderr": None, "min": None, "max": None}
        arr = np.array(values)
        n = arr.size
        mean = np.mean(arr)
        std = np.std(arr, ddof=1) if n > 1 else 0
        stderr = std / np.sqrt(n) if n > 1 else 0
        # 95% confidence interval
        from scipy.stats import t
        if n > 1:
            ci = t.interval(0.95, n-1, loc=mean, scale=stderr)
        else:
            ci = (None, None)
        return {
            "n": n,
            "mean": mean,
            "median": np.median(arr),
            "std": std,
            "stderr": stderr,
            "ci_lower": ci[0],
            "ci_upper": ci[1],
            "min": np.min(arr),
            "max": np.max(arr)
        }
    
    @staticmethod
    def validate_dependent_data(samples, groups):
        """
        Checks whether the data are suitable for dependent tests.
        
        Parameters:
        -----------
        samples : dict
            Dictionary with group names as keys and lists of measurements as values
        groups : list
            List of groups to analyze
            
        Returns:
        --------
        dict
            Validation results with status and error messages if applicable
        """
        validation = {"valid": True, "messages": []}
        
        # Check if all specified groups exist
        for group in groups:
            if group not in samples:
                validation["valid"] = False
                validation["messages"].append(f"Group '{group}' not found in the data.")
        
        if not validation["valid"]:
            return validation
        
        # Check sample sizes
        sample_sizes = [len(samples[g]) for g in groups]
        if len(set(sample_sizes)) > 1:
            validation["valid"] = False
            validation["messages"].append(
                f"Unequal sample sizes: {', '.join([f'{g}: {len(samples[g])}' for g in groups])}"
            )
            validation["messages"].append(
                "For dependent tests, all groups must have the same number of measurements."
            )
        
        # Check for empty groups
        for group in groups:
            if len(samples[group]) == 0:
                validation["valid"] = False
                validation["messages"].append(f"Group '{group}' contains no data.")
        
        # Check minimum number of measurements
        min_samples = min(sample_sizes) if sample_sizes else 0
        if min_samples < 3:
            validation["valid"] = False
            validation["messages"].append(
                f"Too few measurements ({min_samples}). At least 3 measurements per group are recommended."
            )
        
        return validation
    
    @staticmethod
    def prepare_advanced_test(df, test, dv, subject, between=None, within=None):
        """
        Prepares an advanced statistical test by checking the assumptions.
        Returns test_info, recommendation, samples and groups.
        """
        recommendation = 'parametric'
        
        try:
            samples = {}
            groups = []

            if test == 'mixed_anova':
                if not between or not within:
                    return {"error": "Mixed ANOVA requires between and within factor"}
                b_factor, w_factor = between[0], within[0]
                for b_val in df[b_factor].unique():
                    for w_val in df[w_factor].unique():
                        group_label = f"{b_factor}={b_val}, {w_factor}={w_val}"
                        subset = df[(df[b_factor] == b_val) & (df[w_factor] == w_val)]
                        samples[group_label] = subset[dv].tolist()
                groups = list(samples.keys())
                # Formula for Mixed ANOVA (for assumption checking) - use sanitized names
                sanitized_b_factor = b_factor.replace(' ', '') if ' ' in b_factor else b_factor
                sanitized_w_factor = w_factor.replace(' ', '') if ' ' in w_factor else w_factor
                formula = f"Value ~ C({sanitized_b_factor}) * C({sanitized_w_factor})"

            elif test == 'repeated_measures_anova':
                if not within:
                    return {"error": "RM-ANOVA requires within factor"}
                w_factor = within[0]
                for lvl in df[w_factor].unique():
                    samples[lvl] = df[df[w_factor] == lvl][dv].tolist()
                groups = list(samples.keys())
                # Formula for RM-ANOVA (for assumption checking) - use sanitized names
                sanitized_w_factor = w_factor.replace(' ', '') if ' ' in w_factor else w_factor
                formula = f"Value ~ C({sanitized_w_factor})"

            elif test == 'two_way_anova':
                if not between or len(between) != 2:
                    return {"error": "Two-Way ANOVA requires two between factors"}
                fA, fB = between
                for a_val in df[fA].unique():
                    for b_val in df[fB].unique():
                        group_label = f"{fA}={a_val}, {fB}={b_val}"
                        subset = df[(df[fA] == a_val) & (df[fB] == b_val)]
                        samples[group_label] = subset[dv].tolist()
                groups = list(samples.keys())
                # Formula for Two-Way ANOVA (for assumption checking) - use sanitized names
                sanitized_fA = fA.replace(' ', '') if ' ' in fA else fA
                sanitized_fB = fB.replace(' ', '') if ' ' in fB else fB
                formula = f"Value ~ C({sanitized_fA}) * C({sanitized_fB})"

            else:
                return {"error": f"Unknown test type: {test}"}

            # Assumption checking with appropriate formula
            model_type_map = {
                "One-Way ANOVA": "oneway", 
                "Two-Way ANOVA": "twoway",
                "two_way_anova": "twoway",  # Add lowercase variant
                "t-test": "ttest",
                "mixed_anova": "mixed",
                "repeated_measures_anova": "rm"
            }
            model_type = model_type_map.get(test, "oneway")
            
            transformed_samples, recommendation, test_info = StatisticalTester.check_normality_and_variance(
                groups, samples, dataset_name=dv,
                progress_text=f"{test}",
                column_name=dv,
                formula=formula,
                model_type=model_type
            )

            return {
                "test_info": test_info,
                "recommendation": recommendation,
                "transformed_samples": transformed_samples,
                "samples": samples,
                "groups": groups
            }

        except Exception as e:
            return {"error": str(e)} 
        
    @staticmethod
    def perform_advanced_test(
        df, test, dv, subject, between=None, within=None, alpha=0.05,
        transformed_samples=None, recommendation=None, test_info=None,
        transform_fn=None, force_parametric=False, skip_excel=False, file_name=None, manual_transform=None,
        analysis_log=None  # Add this parameter
    ):
        # Initialize analysis_log if None
        if analysis_log is None:
            analysis_log = []
        """
        Performs advanced statistical tests (Mixed ANOVA, RM-ANOVA, Two-Way ANOVA).
        Checks assumptions, applies transformation if necessary, performs main and post-hoc tests.
        Returns a complete result dict.
        """
        from scipy import stats
        from datetime import datetime

        try:
            # 1. Extract group data
            samples = {}
            groups = []
            df_original = df.copy()

            if test == 'mixed_anova':
                # Mixed ANOVA extraction code...
                if not between or not within:
                    return {"error": "Mixed ANOVA requires between and within factor", "test": test}
                b_factor, w_factor = between[0], within[0]
                for b_val in df[b_factor].unique():
                    for w_val in df[w_factor].unique():
                        group_label = f"{b_factor}={b_val}, {w_factor}={w_val}"
                        subset = df[(df[b_factor] == b_val) & (df[w_factor] == w_val)]
                        samples[group_label] = subset[dv].tolist()
                groups = list(samples.keys())

            elif test == 'repeated_measures_anova':
                if not within or not subject:
                    return {"error": "RM-ANOVA requires within factor and subject", "test": test}
                w_factor = within[0]
                for lvl in df[w_factor].unique():
                    samples[lvl] = df[df[w_factor] == lvl][dv].tolist()
                groups = list(samples.keys())
                group_sizes = [len(samples[g]) for g in groups]
                if len(set(group_sizes)) > 1:
                    return {
                        "error": "For dependent samples, all groups must have the same number of measurements.",
                        "test": "Repeated Measures ANOVA (failed)"
                    }

            elif test == 'two_way_anova':
                # Two-way ANOVA extraction code...
                if not between or len(between) != 2:
                    return {"error": "Two-Way ANOVA requires two between factors", "test": test}
                fA, fB = between
                for a_val in df[fA].unique():
                    for b_val in df[fB].unique():
                        group_label = f"{fA}={a_val}, {fB}={b_val}"
                        subset = df[(df[fA] == a_val) & (df[fB] == b_val)]
                        samples[group_label] = subset[dv].tolist()
                groups = list(samples.keys())

            else:
                return {"error": f"Invalid test type: {test}", "test": test}

            # CRITICAL FIX: Store the original samples BEFORE any transformation
            original_samples = {k: v.copy() for k, v in samples.items()}

            # 2. Check assumptions - BUT ONLY IF NOT ALREADY DONE
            # If transformation was manually specified or already applied, skip the dialog
            transformation_already_applied = manual_transform is not None or transformed_samples is not None
            
            if transformed_samples is None or recommendation is None:
                # We haven't performed checks yet, need to do them
                
                # If manual_transform is provided, we should bypass showing the dialog
                # by directly setting up the test_info and returning the appropriate values
                if manual_transform is not None:
                    print(f"DEBUG: Using manually specified transformation '{manual_transform}'")
                    
                    # Initialize test_info if it doesn't exist
                    if test_info is None:
                        test_info = {
                            "normality_tests": {"all_data": {}, "transformed_data": {}},
                            "variance_test": {},
                            "transformation": manual_transform
                        }
                    else:
                        # Just ensure the transformation is correctly set
                        test_info["transformation"] = manual_transform
                    
                    # Apply the manually selected transformation to generate transformed_samples
                    transformed_samples = {k: v.copy() for k, v in samples.items()}
                    
                    # FIRST: Run tests on ORIGINAL data
                    try:
                        # Combine all raw data for overall normality test
                        all_values = [value for group in groups for value in samples[group]]
                        if len(all_values) >= 3 and len(set(all_values)) > 1:
                            stat, pval = stats.shapiro(all_values)
                            is_normal = pval > 0.05
                            test_info["normality_tests"]["all_data"] = {
                                "statistic": stat, "p_value": pval, "is_normal": is_normal
                            }
                            
                        # Run variance test on raw data (Brown-Forsythe test)
                        if len(groups) >= 2:
                            data_for_levene = [samples[g] for g in groups]
                            if all(len(v) >= 3 for v in data_for_levene):
                                stat, pval = stats.levene(*data_for_levene, center='median')
                                has_equal_variance = pval > 0.05
                                test_info["variance_test"] = {
                                    "statistic": stat, 
                                    "p_value": pval, 
                                    "equal_variance": has_equal_variance
                                }
                    except Exception as e:
                        print(f"DEBUG: Error in original data tests: {str(e)}")
                    
                    # SECOND: Apply the transformation
                    if manual_transform == "log10":
                        # Apply log transformation to each group
                        min_val = min([min(samples[g]) for g in groups])
                        shift = -min_val + 1 if min_val <= 0 else 0
                        for group in groups:
                            transformed_samples[group] = [np.log10(v + shift) for v in samples[group]]
                        print(f"DEBUG: Applied log10 transformation with shift {shift}")
                    elif manual_transform == "boxcox":
                        # Apply Box-Cox transformation
                        min_val = min([min(samples[g]) for g in groups])
                        shift = -min_val + 1 if min_val <= 0 else 0
                        for group in groups:
                            shifted = [v + shift for v in samples[group]]
                            try:
                                lambda_val = boxcox_normmax(shifted)
                                transformed_samples[group] = list(boxcox(shifted, lambda_val))
                                test_info["boxcox_lambda"] = lambda_val
                            except Exception as e:
                                print(f"DEBUG: Error in Box-Cox, falling back to log10: {str(e)}")
                                transformed_samples[group] = [np.log10(v + shift) for v in samples[group]]
                    elif manual_transform == "arcsin_sqrt":
                        # Apply arcsin square root transformation
                        for group in groups:
                            values = samples[group]
                            min_val = min(values)
                            max_val = max(values)
                            # Scale values to 0-1 range if needed
                            if min_val < 0 or max_val > 1:
                                scaled = [(v - min_val) / (max_val - min_val) for v in values]
                            else:
                                scaled = values
                            transformed_samples[group] = [np.arcsin(np.sqrt(v)) for v in scaled]
                    
                    # THIRD: Test the transformed data
                    try:
                        # Combine all transformed data for overall normality test
                        all_values_trans = [value for group in groups for value in transformed_samples[group]]
                        if len(all_values_trans) >= 3 and len(set(all_values_trans)) > 1:
                            stat, pval = stats.shapiro(all_values_trans)
                            is_normal = pval > 0.05
                            test_info["normality_tests"]["transformed_data"] = {
                                "statistic": stat, "p_value": pval, "is_normal": is_normal
                            }
                            
                        # Run variance test on transformed data (Brown-Forsythe test)
                        if len(groups) >= 2:
                            data_for_levene = [transformed_samples[g] for g in groups]
                            if all(len(v) >= 3 for v in data_for_levene):
                                stat, pval = stats.levene(*data_for_levene, center='median')
                                has_equal_variance = pval > 0.05
                                test_info["variance_test"]["transformed"] = {
                                    "statistic": stat, 
                                    "p_value": pval, 
                                    "equal_variance": has_equal_variance
                                }
                    except Exception as e:
                        print(f"DEBUG: Error in transformed data tests: {str(e)}")
                    
                    recommendation = "parametric" if manual_transform != "none" else "non_parametric"
                    # But still check normality after transformation to be sure (new structure)
                    if "post_transformation" in test_info and "residuals_normality" in test_info["post_transformation"]:
                        is_normal = test_info["post_transformation"]["residuals_normality"].get("is_normal", False)
                        if not is_normal:
                            print(f"DEBUG: Model residuals still not normal after manual transformation, using non_parametric")
                            recommendation = "non_parametric"
                else:
                    # We already have the results from prepare_advanced_test, no need to call again
                    print("DEBUG: Using existing test results from prepare_advanced_test")
            

            print("DEBUG: transformed_samples =", transformed_samples)
            print("DEBUG: samples =", samples)
            # Patch: If transformed_samples is None, fallback to a copy of samples
            if transformed_samples is None:
                print("WARNING: transformed_samples is None, falling back to a copy of samples.")
                transformed_samples = {k: v.copy() for k, v in samples.items()}
            valid_groups = [g for g in groups if g in transformed_samples and len(transformed_samples[g]) > 0]
            print("DEBUG: valid_groups =", valid_groups)
            print("DEBUG: recommendation =", recommendation)

            df_transformed = df.copy()

            # 3. Apply transformation (automatically or manually, but never twice!)
            transformation_type = None
            if manual_transform is not None:
                transformation_type = manual_transform
                # Ensure this is also set in test_info for consistent reporting
                if test_info:
                    test_info["transformation"] = manual_transform
            elif test_info and test_info.get("transformation"):
                transformation_type = test_info["transformation"]

            # IMPORTANT: Only apply transformation if we have a valid transformation type
            # and it's not "none" or "None"
            if transformation_type and transformation_type not in ["none", "None", "Keine"]:
                if transformation_type == "log10":
                    min_val = df[dv].min()
                    shift = -min_val + 1 if min_val <= 0 else 0
                    df_transformed[dv] = np.log10(df[dv] + shift)
                    print(f"DEBUG: Applied log10 transformation with shift {shift}")
                elif transformation_type == "boxcox":
                    min_val = df[dv].min()
                    shift = -min_val + 1 if min_val <= 0 else 0
                    if shift > 0:
                        df_transformed[dv] = df[dv] + shift
                    lambda_val = test_info.get("boxcox_lambda")
                    if lambda_val is None:
                        from scipy.stats import boxcox_normmax
                        try:
                            lambda_val = boxcox_normmax(df_transformed[dv])
                        except Exception as e:
                            print(f"DEBUG: boxcox_normmax failed: {e}. Using lambda=0 (log transform)")
                            lambda_val = 0
                    from scipy.stats import boxcox
                    df_transformed[dv] = boxcox(df_transformed[dv], lambda_val)
                elif transformation_type == "arcsin_sqrt":
                    min_val = df[dv].min()
                    max_val = df[dv].max()
                    if min_val < 0 or max_val > 1:
                        df_transformed[dv] = (df[dv] - min_val) / (max_val - min_val)
                    df_transformed[dv] = np.arcsin(np.sqrt(df_transformed[dv]))

                transformed_values = df_transformed[dv].values
                print("DEBUG: Transformed data statistics:")
                print(f"- Min: {np.min(transformed_values)}")
                print(f"- Max: {np.max(transformed_values)}")
                print(f"- Contains NaN: {np.isnan(transformed_values).any()}")
                print(f"- Contains Inf: {np.isinf(transformed_values).any()}")

                # Extract transformed samples again
                samples_for_transform = {}
                if test == 'mixed_anova':
                    # Extract for mixed ANOVA...
                    b_factor, w_factor = between[0], within[0]
                    for b_val in df_transformed[b_factor].unique():
                        for w_val in df_transformed[w_factor].unique():
                            group_label = f"{b_factor}={b_val}, {w_factor}={w_val}"
                            subset = df_transformed[(df_transformed[b_factor] == b_val) & (df_transformed[w_factor] == w_val)]
                            samples_for_transform[group_label] = subset[dv].tolist()
                elif test == 'repeated_measures_anova':
                    # Extract for RM ANOVA...
                    w_factor = within[0]
                    for lvl in df_transformed[w_factor].unique():
                        samples_for_transform[lvl] = df_transformed[df_transformed[w_factor] == lvl][dv].tolist()
                elif test == 'two_way_anova':
                    # Extract for two-way ANOVA...
                    fA, fB = between
                    for a_val in df_transformed[fA].unique():
                        for b_val in df_transformed[fB].unique():
                            group_label = f"{fA}={a_val}, {fB}={b_val}"
                            subset = df_transformed[(df_transformed[fA] == a_val) & (df_transformed[fB] == b_val)]
                            samples_for_transform[group_label] = subset[dv].tolist()
                transformed_samples = samples_for_transform
            else:
                # No transformation applied, use original data
                print("DEBUG: No transformation applied, using original data")

            # 4. Perform test
            result = {"test_info": test_info, "recommendation": recommendation}
            # To:
            if force_parametric:
                print(f"DEBUG: User explicitly forced parametric test, overriding recommendation '{recommendation}'")
                recommendation = 'parametric'
            else:
                # Honor recommendation from normality tests
                print(f"DEBUG: Using recommendation from normality tests: '{recommendation}'")
                # Double-check normality for extra safety (new structure)
                if test_info:
                    # Check post-transformation if available, otherwise pre-transformation
                    if "post_transformation" in test_info and "residuals_normality" in test_info["post_transformation"]:
                        is_normal = test_info["post_transformation"]["residuals_normality"].get("is_normal", False)
                        if not is_normal:
                            print(f"DEBUG: Model residuals are NOT normal after transformation, forcing non_parametric")
                            recommendation = "non_parametric"
                    elif "pre_transformation" in test_info and "residuals_normality" in test_info["pre_transformation"]:
                        is_normal = test_info["pre_transformation"]["residuals_normality"].get("is_normal", False)
                        if not is_normal:
                            print(f"DEBUG: Model residuals are NOT normal, forcing non_parametric")
                            recommendation = "non_parametric"

            if recommendation == 'parametric':
                if test == 'mixed_anova':
                    res = StatisticalTester._run_mixed_anova_logged(df_transformed, dv, subject, between, within, alpha)
                elif test == 'repeated_measures_anova':
                    res = StatisticalTester._run_repeated_measures_anova_logged(
                        df_transformed, dv, subject, within, alpha,
                        test_info=test_info  # Pass test_info here
                    )
                else:
                    res = StatisticalTester._run_two_way_anova_logged(df_transformed, dv, between, alpha, test_info=test_info)
                res.update(result)
                # Set test info at top level
                if test_info:
                    if "test_info" not in res:
                        res["test_info"] = test_info
                    else:
                        # Merge test_info into existing test_info
                        for key, value in test_info.items():
                            if key not in res["test_info"]:
                                res["test_info"][key] = value

                # Ensure normality and variance tests are directly accessible 
                if "test_info" in res:
                    ti = res["test_info"]
                    if "normality_tests" in ti and "normality_tests" not in res:
                        res["normality_tests"] = ti["normality_tests"]
                    if "variance_test" in ti and "variance_test" not in res:
                        res["variance_test"] = ti["variance_test"]
                    if "transformation" in ti and "transformation" not in res:
                        res["transformation"] = ti["transformation"]
                    if "boxcox_lambda" in ti and "boxcox_lambda" not in res:
                        res["boxcox_lambda"] = ti["boxcox_lambda"]
                    
                    # CRITICAL FIX: Convert new structure to old structure for Excel compatibility
                    if "pre_transformation" in ti and "post_transformation" in ti and "normality_tests" not in res:
                        # Create normality_tests structure from pre/post transformation data
                        normality_tests = {"all_data": {}, "transformed_data": {}}
                        
                        # Pre-transformation (original) data
                        if "residuals_normality" in ti["pre_transformation"]:
                            pre_norm = ti["pre_transformation"]["residuals_normality"]
                            normality_tests["all_data"] = {
                                "statistic": pre_norm.get("statistic") if pre_norm.get("statistic") is not None else "N/A",
                                "p_value": pre_norm.get("p_value") if pre_norm.get("p_value") is not None else "N/A",
                                "is_normal": pre_norm.get("is_normal", False)
                            }
                        
                        # Post-transformation data
                        if "residuals_normality" in ti["post_transformation"]:
                            post_norm = ti["post_transformation"]["residuals_normality"]
                            normality_tests["transformed_data"] = {
                                "statistic": post_norm.get("statistic") if post_norm.get("statistic") is not None else "N/A",
                                "p_value": post_norm.get("p_value") if post_norm.get("p_value") is not None else "N/A",
                                "is_normal": post_norm.get("is_normal", False)
                            }
                        
                        res["normality_tests"] = normality_tests
                    
                    # CRITICAL FIX: Convert variance test structure for Excel compatibility
                    if "pre_transformation" in ti and "post_transformation" in ti and "variance_test" not in res:
                        variance_test = {}
                        
                        # Pre-transformation variance test
                        if "variance" in ti["pre_transformation"]:
                            pre_var = ti["pre_transformation"]["variance"]
                            variance_test.update({
                                "statistic": pre_var.get("statistic") if pre_var.get("statistic") is not None else "N/A",
                                "p_value": pre_var.get("p_value") if pre_var.get("p_value") is not None else "N/A",
                                "equal_variance": pre_var.get("equal_variance", False)
                            })
                        
                        # Post-transformation variance test
                        if "variance" in ti["post_transformation"]:
                            post_var = ti["post_transformation"]["variance"]
                            variance_test["transformed"] = {
                                "statistic": post_var.get("statistic") if post_var.get("statistic") is not None else "N/A",
                                "p_value": post_var.get("p_value") if post_var.get("p_value") is not None else "N/A",
                                "equal_variance": post_var.get("equal_variance", False)
                            }
                        
                        res["variance_test"] = variance_test
                    
                    # Also ensure transformation is accessible from test_info at top level
                    if "transformation" in ti and "transformation" not in res:
                        res["transformation"] = ti["transformation"]

                # --- POST-HOC for all advanced tests ---
                if res.get("p_value") is not None and res["p_value"] < alpha:
                    # 1. Generate all possible group comparisons
                    from itertools import combinations
                    if test == "two_way_anova":
                        # For Two-Way ANOVA, create interaction group labels
                        group_names = []
                        factors = between
                        for factor_a_val in sorted(df_transformed[factors[0]].unique()):
                            for factor_b_val in sorted(df_transformed[factors[1]].unique()):
                                group_label = f"{factors[0]}={factor_a_val}, {factors[1]}={factor_b_val}"
                                group_names.append(group_label)
                    elif test == "mixed_anova":
                        # For Mixed ANOVA, create interaction group labels
                        group_names = []
                        b_factor, w_factor = between[0], within[0]
                        for b_val in sorted(df_transformed[b_factor].unique()):
                            for w_val in sorted(df_transformed[w_factor].unique()):
                                group_label = f"{b_factor}={b_val}, {w_factor}={w_val}"
                                group_names.append(group_label)
                    elif test == "repeated_measures_anova":
                        w_factor = within[0]
                        group_names = list(df_transformed[w_factor].unique())
                    else:
                        group_names = []

                    all_comparisons = list(combinations(group_names, 2))

                    # 2. Show post-hoc method selection dialog FIRST
                    posthoc_method = "paired_custom"  # Default to paired t-tests with Holm-Sidak
                    control_group = None
                    try:
                        # For Two-Way ANOVA, default to paired t-tests (better for interaction effects)
                        default_method = "paired_custom" if test == "two_way_anova" else "tukey"
                        posthoc_method = UIDialogManager.select_posthoc_test_dialog(
                            parent=None, progress_text=f"({test})", column_name=dv, default_method=default_method
                        )
                        if posthoc_method is None:
                            posthoc_method = "paired_custom"  # Default fallback
                        print(f"DEBUG: Selected post-hoc method for {test}: {posthoc_method}")
                        
                        # If Dunnett was selected, ask for control group
                        if posthoc_method == "dunnett":
                            control_group = UIDialogManager.select_control_group_dialog(
                                parent=None, groups=group_names
                            )
                            print(f"DEBUG: Selected control group: {control_group}")
                    except Exception as e:
                        print(f"WARNING: Could not show post-hoc method dialog: {e}")
                        posthoc_method = "paired_custom"

                    # 2.5. Show the comparison selection dialog (only for pairwise t-tests)
                    selected_comparisons = None
                    if posthoc_method == "dunnett" and control_group:
                        # For Dunnett, automatically generate comparisons against control group
                        # IMPORTANT: No additional dialog should be shown for Dunnett!
                        selected_comparisons = [(control_group, group) for group in group_names if group != control_group]
                        print(f"DEBUG: Auto-generated Dunnett comparisons against control '{control_group}': {selected_comparisons}")
                        print(f"DEBUG: Dunnett test will compare {len(selected_comparisons)} groups against the control group")
                    elif posthoc_method == "tukey":
                        # For Tukey HSD, automatically use all pairwise comparisons
                        selected_comparisons = all_comparisons
                        print(f"DEBUG: Auto-generated Tukey comparisons (all pairwise): {len(selected_comparisons)} comparisons")
                    elif posthoc_method == "paired_custom":
                        # Only show dialog for pairwise t-tests with custom selection
                        try:
                            from comparison_selection_dialog import ComparisonSelectionDialog
                            import sys
                            from PyQt5.QtWidgets import QApplication
                            app = QApplication.instance()
                            if app is None:
                                app = QApplication(sys.argv)
                            dialog = ComparisonSelectionDialog(all_comparisons, checked_by_default=False)  # Pass flag to deselect all
                            if dialog.exec_() == dialog.Accepted:
                                selected_comparisons = dialog.get_selected_comparisons()
                            else:
                                selected_comparisons = []
                            print(f"DEBUG: User selected {len(selected_comparisons)} comparisons: {selected_comparisons}")
                        except Exception as e:
                            print(f"WARNING: Could not show comparison selection dialog: {e}")
                            selected_comparisons = all_comparisons  # fallback: select all
                    else:
                        # Default fallback: use all comparisons
                        selected_comparisons = all_comparisons

                    # Normalize selected comparisons to sorted, stripped tuples for robust matching
                    def normalize_pair(pair):
                        return tuple(sorted([s.strip() for s in pair]))
                    normalized_selected_comparisons = set(normalize_pair(pair) for pair in selected_comparisons)
                    print(f"DEBUG: Normalized selected comparisons: {normalized_selected_comparisons}")

                    # 3. Pass normalized selected comparisons and method to posthoc
                    posthoc_kwargs = {
                        "selected_comparisons": normalized_selected_comparisons,
                        "method": posthoc_method
                    }
                    if control_group:
                        posthoc_kwargs["control_group"] = control_group
                    if test == "two_way_anova":
                        # Add method selection to kwargs
                        # posthoc_method is not defined; remove or define it before use
                        # posthoc_kwargs["method"] = posthoc_method
                        posthoc = PostHocFactory.perform_posthoc_for_anova(
                            "two_way", df=df_transformed, dv=dv, between=between, alpha=alpha, **posthoc_kwargs
                        )
                    elif test == "mixed_anova":
                        posthoc = PostHocFactory.perform_posthoc_for_anova(
                            "mixed", df=df_transformed, dv=dv, subject=subject, between=between, within=within, alpha=alpha, **posthoc_kwargs
                        )
                    elif test == "repeated_measures_anova":
                        posthoc = PostHocFactory.perform_posthoc_for_anova(
                            "rm", df=df_transformed, dv=dv, subject=subject, within=within, alpha=alpha, **posthoc_kwargs
                        )
                    else:
                        posthoc = None
                    if posthoc and "pairwise_comparisons" in posthoc:
                        res["pairwise_comparisons"] = posthoc["pairwise_comparisons"]
                        # Override posthoc_test if:
                        # 1. It wasn't set by the main ANOVA function
                        # 2. It has a generic default name
                        # 3. It was set by Pingouin but PostHocFactory used a different method
                        current_posthoc = res.get("posthoc_test", "")
                        new_posthoc = posthoc.get("posthoc_test", "")
                        print(f"DEBUG OVERRIDE: Current posthoc_test: '{current_posthoc}'")
                        print(f"DEBUG OVERRIDE: New posthoc_test from PostHocFactory: '{new_posthoc}'")
                        should_override = (
                            not current_posthoc or 
                            current_posthoc == "Two-Way ANOVA Post-hoc Tests" or
                            ("Pingouin" in str(current_posthoc) and new_posthoc and "Tukey" in str(new_posthoc))
                        )
                        print(f"DEBUG OVERRIDE: Should override? {should_override}")
                        if should_override:
                            res["posthoc_test"] = new_posthoc
                            print(f"DEBUG OVERRIDE: Updated posthoc_test to: '{res.get('posthoc_test')}'")
                        else:
                            print(f"DEBUG OVERRIDE: Keeping original posthoc_test: '{current_posthoc}'")

                # CRITICAL FIX: Always store the original and transformed samples as separate entities
                res["raw_data"] = original_samples
                if transformation_type and transformation_type not in ["none", "None", "Keine"]:
                    res["raw_data_transformed"] = transformed_samples

                # Excel export
                if not skip_excel:
                    print(f"DEBUG: Current working directory before export: {os.getcwd()}")
                    excel_file = file_name if file_name else get_output_path(f"{test}_{datetime.now().strftime('%Y%m%d_%H%M%S')}", "xlsx")
                    print("DEBUG: Results dict before Excel export:", res)  # <--- Add this line
                    ResultsExporter.export_results_to_excel(res, excel_file, res.get("analysis_log", None))
                    res["excel_file"] = excel_file
                return res

            elif recommendation == 'non_parametric':
                # Non-parametric alternative is needed but not available
                print(f"DEBUG: Non-parametric alternative required for {test}")
                
                # Create comprehensive results indicating non-parametric alternative is needed
                test_name_map = {
                    'two_way_anova': 'Non-parametric Two-Way ANOVA',
                    'mixed_anova': 'Non-parametric Mixed ANOVA', 
                    'repeated_measures_anova': 'Non-parametric Repeated Measures ANOVA'
                }
                
                suggested_alternatives = {
                    'two_way_anova': 'Rank transformation + permutation test or Friedman test with blocking',
                    'mixed_anova': 'Mixed effects model with robust methods or rank-based approach',
                    'repeated_measures_anova': 'Friedman test or rank-based repeated measures analysis'
                }
                
                res = {
                    "test": test_name_map.get(test, f"Non-parametric {test.replace('_', ' ').title()}") + " (required but not available)",
                    "recommendation": "non_parametric",
                    "parametric_assumptions_violated": True,
                    "test_info": test_info,
                    "raw_data": original_samples,
                    "message": f"Parametric assumptions not met even after transformation. {test_name_map.get(test, 'Non-parametric alternative')} is recommended.",
                    "suggested_alternative": suggested_alternatives.get(test, "Non-parametric alternative"),
                    "p_value": None,
                    "statistic": None,
                    "error": f"Non-parametric alternatives for {test} are not implemented in this version.",
                    "analysis_note": "The data transformation was attempted but parametric assumptions still could not be met. A non-parametric approach would be appropriate for this dataset."
                }
                
                # Include transformation information
                if transformation_type and transformation_type not in ["none", "None", "Keine"]:
                    res["transformation"] = transformation_type
                    res["raw_data_transformed"] = transformed_samples
                
                # Ensure test_info is accessible at top level for Excel export
                if test_info:
                    for key in ["normality_tests", "variance_test", "transformation", "boxcox_lambda"]:
                        if key in test_info and key not in res:
                            res[key] = test_info[key]
                
                # Create analysis log
                analysis_log = []
                analysis_log.append(f"Advanced Test Analysis: {test}")
                analysis_log.append(f"Dataset: {dv}")
                analysis_log.append("Assumption Check Results:")
                if test_info:
                    # Add normality test results
                    if "pre_transformation" in test_info:
                        pre_norm = test_info["pre_transformation"].get("residuals_normality", {})
                        if "p_value" in pre_norm and pre_norm['p_value'] is not None:
                            analysis_log.append(f"- Original data normality: p = {pre_norm['p_value']:.4f} ({'Normal' if pre_norm.get('is_normal', False) else 'Not normal'})")
                    
                    if "post_transformation" in test_info:
                        post_norm = test_info["post_transformation"].get("residuals_normality", {})
                        if "p_value" in post_norm and post_norm['p_value'] is not None:
                            analysis_log.append(f"- After transformation normality: p = {post_norm['p_value']:.4f} ({'Normal' if post_norm.get('is_normal', False) else 'Not normal'})")
                    
                    # Add variance test results
                    if "variance_test" in test_info:
                        var_test = test_info["variance_test"]
                        if "p_value" in var_test and var_test['p_value'] is not None:
                            analysis_log.append(f"- Variance homogeneity: p = {var_test['p_value']:.4f} ({'Equal' if var_test.get('equal_variance', False) else 'Not equal'})")
                
                if transformation_type:
                    analysis_log.append(f"Applied transformation: {transformation_type}")
                
                analysis_log.append(f"CONCLUSION: {res['message']}")
                analysis_log.append(f"RECOMMENDED: {res['suggested_alternative']}")
                
                res["analysis_log"] = "\n".join(analysis_log)
                
                # Excel export
                if not skip_excel:
                    print(f"DEBUG: Current working directory before export: {os.getcwd()}")
                    excel_file = file_name if file_name else get_output_path(f"{test}_nonparametric_needed_{datetime.now().strftime('%Y%m%d_%H%M%S')}", "xlsx")
                    ResultsExporter.export_results_to_excel(res, excel_file, res.get("analysis_log", None))
                    res["excel_file"] = excel_file
                    print(f"DEBUG: Excel file created with non-parametric recommendation: {excel_file}")
                
                return res

            else:  # robust fallback path (GLMM/GEE-based)
                # Select and run the appropriate robust method
                robust_result = None
                robust_method = None
                try:
                    if test == 'two_way_anova':
                        robust_method = GLMMTwoWayANOVA
                        robust_result = GLMMTwoWayANOVA(df=df, dv=dv, factors=between, alpha=alpha).run()
                    elif test == 'repeated_measures_anova':
                        robust_method = GEERMANOVA
                        robust_result = GEERMANOVA(df=df, dv=dv, subject=subject, within=within, alpha=alpha).run()
                    elif test == 'mixed_anova':
                        robust_method = GLMMMixedANOVA
                        robust_result = GLMMMixedANOVA(df=df, dv=dv, subject=subject, between=between, within=within, alpha=alpha).run()
                    else:
                        # Fallback: use auto decision logic
                        robust_method = auto_anova_decision
                        robust_result = auto_anova_decision(df=df, dv=dv, subject=subject, between=between, within=within, alpha=alpha)
                except Exception as robust_e:
                    import traceback
                    print(f"ERROR in robust fallback: {str(robust_e)}")
                    traceback.print_exc()
                    robust_result = {
                        "test": f"Robust {test} (failed)",
                        "test_info": test_info,
                        "recommendation": "robust_fallback",
                        "error": f"Error running robust fallback: {str(robust_e)}",
                        "parametric_violated": True
                    }

                # Format result to match parametric structure
                result = {
                    "test": robust_result.get("test", f"Robust {test}"),
                    "test_info": test_info,
                    "recommendation": "robust_fallback",
                    "robust_method": robust_method.__name__ if robust_method else None,
                    "parametric_violated": True,
                    "raw_data": original_samples,
                    "robust_result": robust_result
                }
                # Copy over key stats if present
                for key in ["p_value", "statistic", "summary", "model", "diagnostics", "pairwise_comparisons", "posthoc_test"]:
                    if key in robust_result:
                        result[key] = robust_result[key]
                # Add transformation info if present
                if transformation_type:
                    result["transformation"] = transformation_type
                # Excel export
                if not skip_excel:
                    analysis_log = []
                    analysis_log.append(f"Advanced Test Analysis: {test}")
                    analysis_log.append(f"Dataset: {dv}")
                    analysis_log.append(f"Test recommendation: robust_fallback (GLMM/GEE)")
                    if transformation_type:
                        analysis_log.append(f"Applied transformation: {transformation_type}")
                    if "p_value" in result:
                        if result["p_value"] is not None and result["p_value"] < alpha:
                            analysis_log.append(f"Result: Significant difference found (p = {result['p_value']:.4f})")
                        elif result["p_value"] is not None:
                            analysis_log.append(f"Result: No significant difference (p = {result['p_value']:.4f})")
                        else:
                            analysis_log.append("Result: No p-value available")
                    else:
                        analysis_log.append("Result: Robust test performed (see details)")
                    analysis_log_text = "\n".join(analysis_log)
                    output_file = f"{file_name}_results.xlsx" if file_name else get_output_path(
                        f"{test}_{datetime.now().strftime('%Y%m%d_%H%M%S')}", "xlsx"
                    )
                    try:
                        ResultsExporter.export_results_to_excel(result, output_file, analysis_log_text)
                        result["excel_file"] = output_file
                        print(f"Results exported to: {output_file}")
                    except Exception as export_e:
                        print(f"Excel export could not be performed: {export_e}")
                        result["excel_export_error"] = str(export_e)
                return result
            
        except Exception as e:
            import traceback
            print(f"ERROR in perform_advanced_test: {str(e)}")
            traceback.print_exc()
            return {
                "error": f"Error performing the test: {str(e)}",
                "test": f"{test} (failed)",
                "p_value": None,
                "statistic": None
            }

    @staticmethod
    def _run_any_parametric_test(
        df, dv, subject=None, between=None, within=None,
        alpha=0.05, test_func=None, extract_raw=None, test_info=None, **kwargs
    ):
        # 1. Initialize logger
        log_messages = []
        def log_step(msg):
            log_messages.append(msg)

        # Add logging about assumptions from test_info if available
        if test_info:
            log_step("Starting with test_info from preceding normality and variance tests.")
            
            # Log normality test results - handle both old and new structure
            if "normality_tests" in test_info:
                norm_tests = test_info["normality_tests"]
                if "all_data" in norm_tests and "p_value" in norm_tests["all_data"]:
                    p_val = norm_tests["all_data"]["p_value"]
                    is_normal = norm_tests["all_data"].get("is_normal", False)
                    log_step(f"Original normality test: p = {p_val:.4f} - {'Normal' if is_normal else 'Not normal'}")
                    
                if "transformed_data" in norm_tests and "p_value" in norm_tests["transformed_data"]:
                    p_val = norm_tests["transformed_data"]["p_value"]
                    is_normal = norm_tests["transformed_data"].get("is_normal", False)
                    log_step(f"Transformed normality test: p = {p_val:.4f} - {'Normal' if is_normal else 'Not normal'}")
            
            # Handle new structure from check_assumptions_and_transform
            elif "pre_transformation" in test_info and "post_transformation" in test_info:
                # Log pre-transformation normality
                pre_norm = test_info["pre_transformation"].get("residuals_normality", {})
                if "p_value" in pre_norm and pre_norm["p_value"] is not None:
                    p_val = pre_norm["p_value"]
                    stat_val = pre_norm.get("statistic", "N/A")
                    is_normal = pre_norm.get("is_normal", False)
                    log_step(f"Original data normality (Shapiro-Wilk): W = {stat_val:.4f}, p = {p_val:.4f} - {'Normal' if is_normal else 'Not normal'}")
                
                # Log post-transformation normality
                post_norm = test_info["post_transformation"].get("residuals_normality", {})
                if "p_value" in post_norm and post_norm["p_value"] is not None:
                    p_val = post_norm["p_value"]
                    stat_val = post_norm.get("statistic", "N/A")
                    is_normal = post_norm.get("is_normal", False)
                    transformation = test_info.get("transformation", "No transformation")
                    log_step(f"After {transformation} transformation normality (Shapiro-Wilk): W = {stat_val:.4f}, p = {p_val:.4f} - {'Normal' if is_normal else 'Not normal'}")
            
            # Log variance test results - handle both old and new structure
            if "variance_test" in test_info:
                var_test = test_info["variance_test"]
                if "p_value" in var_test:
                    p_val = var_test["p_value"]
                    is_equal = var_test.get("equal_variance", False)
                    log_step(f"Original variance test: p = {p_val:.4f} - {'Equal' if is_equal else 'Not equal'}")
                    
                if "transformed" in var_test and "p_value" in var_test["transformed"]:
                    p_val = var_test["transformed"]["p_value"] 
                    is_equal = var_test["transformed"].get("equal_variance", False)
                    log_step(f"Transformed variance test: p = {p_val:.4f} - {'Equal' if is_equal else 'Not equal'}")
            
            # Handle new structure from check_assumptions_and_transform
            elif "pre_transformation" in test_info and "post_transformation" in test_info:
                # Log pre-transformation variance
                pre_var = test_info["pre_transformation"].get("variance", {})
                if "p_value" in pre_var and pre_var["p_value"] is not None:
                    p_val = pre_var["p_value"]
                    stat_val = pre_var.get("statistic", "N/A")
                    is_equal = pre_var.get("equal_variance", False)
                    log_step(f"Original data variance homogeneity (Brown-Forsythe): F = {stat_val:.4f}, p = {p_val:.4f} - {'Equal' if is_equal else 'Unequal'}")
                
                # Log post-transformation variance
                post_var = test_info["post_transformation"].get("variance", {})
                if "p_value" in post_var and post_var["p_value"] is not None:
                    p_val = post_var["p_value"]
                    stat_val = post_var.get("statistic", "N/A")
                    is_equal = post_var.get("equal_variance", False)
                    transformation = test_info.get("transformation", "No transformation")
                    log_step(f"After {transformation} transformation variance homogeneity (Brown-Forsythe): F = {stat_val:.4f}, p = {p_val:.4f} - {'Equal' if is_equal else 'Unequal'}")
                    log_step(f"Transformed variance test: p = {p_val:.4f} - {'Equal' if is_equal else 'Not equal'}")

        log_step(f"Start parametric test: {test_func.__name__}")
        log_step("Check test assumptions (normality, variance)...")
        log_step("Performing test...")

        # Determine which parameters the function actually accepts
        import inspect
        sig = inspect.signature(test_func)
        valid_params = sig.parameters.keys()
        
        # Create a dictionary with the required parameters
        test_params = {}
        if 'df' in valid_params: test_params['df'] = df
        if 'dv' in valid_params: test_params['dv'] = dv
        if 'subject' in valid_params and subject is not None: test_params['subject'] = subject
        if 'between' in valid_params and between is not None: test_params['between'] = between
        if 'within' in valid_params and within is not None: test_params['within'] = within
        if 'alpha' in valid_params: test_params['alpha'] = alpha
        if 'test_info' in valid_params and test_info is not None: test_params['test_info'] = test_info
        
        # Filter kwargs to only include parameters accepted by the test function
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        test_params.update(filtered_kwargs)  # Add filtered additional parameters
        
        # 3. Run test with only the required parameters
        results = test_func(**test_params)

        # 4. Post-hoc tests if significant
        p = results.get("p_value")
        if p is not None and p < alpha:
            log_step("Significant result (p={:.3f}). Performing post-hoc tests...".format(p))

        # 5. Extract raw data
        if extract_raw:
            log_step("Extracting raw data for DV and factors...")
            results["raw_data"] = extract_raw(df, dv, between, within, subject)

        # 6. Add main ANOVA results to log
        # Main and interaction effects
        if "factors" in results:
            for factor in results["factors"]:
                if factor.get('p_value') is not None:
                    log_step(f"Main effect {factor['factor']}: F({factor['df1']}, {factor['df2']}) = {factor['F']:.3f}, p = {factor['p_value']:.4f}, effect size: {factor.get('effect_size', 'N/A')}")
                else:
                    log_step(f"Main effect {factor['factor']}: F({factor['df1']}, {factor['df2']}) = {factor['F']:.3f}, p = N/A, effect size: {factor.get('effect_size', 'N/A')}")
        if "interactions" in results:
            for inter in results["interactions"]:
                if inter.get('p_value') is not None:
                    log_step(f"Interaction {inter['factors'][0]} x {inter['factors'][1]}: F({inter['df1']}, {inter['df2']}) = {inter['F']:.3f}, p = {inter['p_value']:.4f}, effect size: {inter.get('effect_size', 'N/A')}")
                else:
                    log_step(f"Interaction {inter['factors'][0]} x {inter['factors'][1]}: F({inter['df1']}, {inter['df2']}) = {inter['F']:.3f}, p = N/A, effect size: {inter.get('effect_size', 'N/A')}")

        # Post-hoc comparisons
        if "pairwise_comparisons" in results and results["pairwise_comparisons"]:
            log_step("Post-hoc pairwise comparisons:")
            for comp in results["pairwise_comparisons"]:
                g1 = comp.get("group1", "")
                g2 = comp.get("group2", "")
                pval = comp.get("p_value", "")
                signif = "significant" if comp.get("significant", False) else "not significant"
                log_step(f"  {g1} vs {g2}: p = {pval:.4f} ({signif})")

        results["analysis_log"] = log_messages
        
        # Add test_info to results if not already there
        if test_info is not None and "test_info" not in results:
            results["test_info"] = test_info
        
        return StatisticalTester._standardize_results(results)
    
    @staticmethod
    def _run_mixed_anova_logged(df, dv, subject, between, within, alpha=0.05):
        # 'extract_raw' can be a function that extracts raw data
        return StatisticalTester._run_any_parametric_test(
            df=df,
            dv=dv,
            subject=subject,
            between=between,
            within=within,
            alpha=alpha,
            test_func=StatisticalTester._run_mixed_anova,
            extract_raw=StatisticalTester._extract_raw_data_mixed_anova
        )
    
    @staticmethod
    def _extract_raw_data_mixed_anova(df, dv, between, within, subject):
        # Example implementation: return all individual values per group
        raw = {}
        b, w = between[0], within[0]
        for b_val in df[b].unique():
            for w_val in df[w].unique():
                key = f"{b}={b_val}, {w}={w_val}"
                raw[key] = df[(df[b] == b_val) & (df[w] == w_val)][dv].tolist()
        return raw
    
    @staticmethod
    def _run_repeated_measures_anova_logged(df, dv, subject, within, alpha=0.05, force_posthoc=False, custom_posthoc_alpha=None, **kwargs):
        """Wrapper function that includes logging for repeated measures ANOVA"""
        
        # Capture the test_info parameter and pass it through
        test_info = None
        if 'test_info' in kwargs:
            test_info = kwargs.pop('test_info')
            
        results = StatisticalTester._run_any_parametric_test(
            df=df,
            dv=dv,
            subject=subject,
            between=None,
            within=within,
            alpha=alpha,
            force_posthoc=force_posthoc,
            custom_posthoc_alpha=custom_posthoc_alpha,
            test_func=StatisticalTester._run_repeated_measures_anova,
            extract_raw=StatisticalTester._extract_raw_data_rm_anova,
            test_info=test_info  # Pass the test_info parameter
        )
        
        # Ensure test_info is added to results
        if test_info is not None and "test_info" not in results:
            results["test_info"] = test_info
            
        return results
    
    @staticmethod
    def _extract_raw_data_rm_anova(df, dv, between, within, subject):
        raw = {}
        w = within[0]
        for lvl in df[w].unique():
            raw[f"{w}={lvl}"] = df[df[w] == lvl][dv].tolist()
        return raw
    
    @staticmethod
    def _run_two_way_anova_logged(df, dv, between, alpha=0.05, test_info=None):
        return StatisticalTester._run_any_parametric_test(
            df=df,
            dv=dv,
            subject=None,
            between=between,
            within=None,
            alpha=alpha,
            test_func=StatisticalTester._run_two_way_anova,
            extract_raw=StatisticalTester._extract_raw_data_two_way_anova,
            test_info=test_info
        )
    
    @staticmethod
    def _extract_raw_data_two_way_anova(df, dv, between, within, subject):
        raw = {}
        a, b = between
        for a_val in df[a].unique():
            for b_val in df[b].unique():
                key = f"{a}={a_val}, {b}={b_val}"
                raw[key] = df[(df[a] == a_val) & (df[b] == b_val)][dv].tolist()
        return raw
        
    @staticmethod
    def _run_mixed_anova(df, dv, subject, between, within, alpha=0.05):
        """
        Performs a Mixed ANOVA. Prefers pingouin, fallback to statsmodels.
        """
        import numpy as np
        results = {
            "test": "Mixed ANOVA",
            "p_value": None,
            "statistic": None,
            "effect_size": None,
            "effect_size_type": None,
            "descriptive": {},
            "factors": [],
            "interactions": [],
            "error": None
        }

        if not between or not within:
            results["error"] = "Mixed ANOVA requires both between and within factors"
            return StatisticalTester._standardize_results(results)

        between_factor = between[0]
        rm_factor = within[0]

        try:
            pg = get_pingouin_module()
            has_pingouin = True
        except ImportError:
            has_pingouin = False
            results["warning"] = "Pingouin not installed, using statsmodels"

        try:
            if has_pingouin:
                print("DEBUG: DataFrame columns:", df.columns)
                print("DEBUG: Unique values for within factor:", df[within[0]].unique())
                print("DEBUG: Unique values for subject:", df[subject].unique())
                print("DEBUG: First few rows of df:\n", df.head())
                print("DEBUG: Using Pingouin for Mixed ANOVA")
                aov = pg.mixed_anova(data=df, dv=dv, within=rm_factor, between=between_factor, subject=subject)
                results["anova_table"] = aov.copy()
                for factor in [rm_factor, between_factor]:
                    mask = aov["Source"] == factor
                    if mask.any():
                        row = aov.loc[mask].iloc[0]
                        results["factors"].append({
                            "factor": factor,
                            "type": "within" if factor == rm_factor else "between",
                            "F": float(row["F"]),
                            "p_value": float(row["p-unc"]),
                            "df1": int(row["DF1"]),
                            "df2": int(row["DF2"]),
                            "effect_size": float(row["np2"]),
                            "effect_size_type": "partial_eta_squared"
                        })
                    else:
                        results.setdefault("warnings", []).append(f"No result for factor '{factor}' found in Mixed-ANOVA.")
                
                interaction_name = f"{rm_factor} * {between_factor}"
                mask_int = aov["Source"] == interaction_name
                if mask_int.any():
                    row = aov.loc[mask_int].iloc[0]
                    interaction = {
                        "factors": [rm_factor, between_factor],
                        "F": float(row["F"]),
                        "p_value": float(row["p-unc"]),
                        "df1": int(row["DF1"]),
                        "df2": int(row["DF2"]),
                        "effect_size": float(row["np2"]),
                        "effect_size_type": "partial_eta_squared"
                    }
                    results["interactions"].append(interaction)
                    # Set top-level fields
                    results.update({
                        "p_value": interaction["p_value"],
                        "statistic": interaction["F"],
                        "effect_size": interaction["effect_size"],
                        "effect_size_type": interaction["effect_size_type"],
                        "df1": interaction["df1"],
                        "df2": interaction["df2"],
                    })
                else:
                    results.setdefault("warnings", []).append(f"No interaction '{interaction_name}' found in Mixed-ANOVA.")
                
                # Enhanced Between-Factor Assumption Testing for Mixed ANOVA
                between_assumptions = StatisticalTester._test_mixed_anova_between_assumptions(
                    df, dv, between_factor, subject, alpha
                )
                results.update(between_assumptions)
                
                # Enhanced Interaction Assumption Testing for Mixed ANOVA
                interaction_assumptions = StatisticalTester._test_mixed_anova_interaction_assumptions(
                    df, dv, between_factor, rm_factor, subject, aov, alpha
                )
                results.update(interaction_assumptions)
                #  POST-HOC: pairwise t-tests (Bonferroni) if interaction is significant
                try:
                    # 1. Check if interaction is significant
                    int_row = aov.loc[aov["Source"] == interaction_name]
                    if not int_row.empty and float(int_row["p-unc"].iloc[0]) < alpha:
                        # Interaction is significant: t-tests for all combinations
                        ph = pg.pairwise_tests(
                            data=df,
                            dv=dv,
                            between=between_factor,
                            within=rm_factor,
                            subject=subject,
                            padjust="holm"  # Changed from "bonf" to "holm"
                        )
                        results["posthoc_test"] = "Pairwise t-tests for interaction (Holm-Sidak)"  # Changed from "Bonferroni" to "Holm-Sidak"
                        for _, r in ph.iterrows():
                            results.setdefault("pairwise_comparisons", []).append({
                                "group1": f"{between_factor}={r['A']}, {rm_factor}={r['Time']}",
                                "group2": f"{between_factor}={r['B']}, {rm_factor}={r['Time']}",
                                "test": "Paired t-test" if r['Type'] == 'within' else "Independent t-test",
                                "statistic": float(r["T"]),
                                "p_value": float(r["p-corr"]),
                                "significant": bool(r["significant"]),
                                "corrected": True,
                                "effect_size": float(r["hedges"]) if "hedges" in r else None,
                                "effect_size_type": "hedges_g"
                            })
                    else:
                        # 2. Interaction not significant, check main effects
                        # Between-factor post-hoc with Tukey (if significant)
                        between_row = aov.loc[aov["Source"] == between_factor]
                        if not between_row.empty and float(between_row["p-unc"].iloc[0]) < alpha:
                            # Tukey HSD for between-factor
                            from statsmodels.stats.multicomp import pairwise_tukeyhsd
                            between_groups = df[between_factor].unique()
                            if len(between_groups) > 1:
                                tukey = pairwise_tukeyhsd(
                                    endog=df[dv],
                                    groups=df[between_factor],
                                    alpha=alpha
                                )
                                results["between_posthoc_test"] = "Tukey HSD"
                
                                # More robust way to handle various versions of statsmodels
                                try:
                                    # First try with the pairindices attribute
                                    for i in range(len(tukey.pvalues)):
                                        group1 = tukey.groupsunique[tukey.pairindices[i, 0]]
                                        group2 = tukey.groupsunique[tukey.pairindices[i, 1]]
                                        p_val = tukey.pvalues[i]
                                        is_significant = tukey.reject[i]
                                        
                                        results.setdefault("between_pairwise_comparisons", []).append({
                                            "group1": f"{between_factor}={group1}",
                                            "group2": f"{between_factor}={group2}",
                                            "test": "Tukey HSD",
                                            "p_value": float(p_val),
                                            "significant": bool(is_significant),
                                            "corrected": True
                                        })
                                except (AttributeError, IndexError):
                                    # Fall back to using summary() method, which works in newer versions
                                    summary = tukey.summary()
                                    for i in range(len(summary.data) - 1):  # Skip header row
                                        row = summary.data[i+1]
                                        group1, group2 = row[0], row[1]
                                        p_val = row[3]
                                        is_significant = row[6]  # reject column
                                        
                                        results.setdefault("between_pairwise_comparisons", []).append({
                                            "group1": f"{between_factor}={group1}",
                                            "group2": f"{between_factor}={group2}",
                                            "test": "Tukey HSD",
                                            "p_value": float(p_val),
                                            "significant": bool(is_significant),
                                            "corrected": True
                                        })

                        # Within-factor post-hoc with paired t-tests (if significant)
                        within_row = aov.loc[aov["Source"] == rm_factor]
                        if not within_row.empty and float(within_row["p-unc"].iloc[0]) < alpha:
                            # Paired t-tests for within-factor with Bonferroni
                            from itertools import combinations
                            within_groups = df[rm_factor].unique()
                            results["within_posthoc_test"] = "Paired t-tests (Holm-Sidak)"  # Changed from "Bonferroni" to "Holm-Sidak"
                            
                            # Create comparison groups
                            n_comparisons = len(list(combinations(within_groups, 2)))
                            
                            # Perform paired t-tests and store p-values for Holm-Sidak correction
                            p_values = []
                            t_stats = []
                            data_pairs = []
                            for group1, group2 in combinations(within_groups, 2):
                                # Prepare data for paired t-tests
                                data1 = df[df[rm_factor] == group1][dv].values
                                data2 = df[df[rm_factor] == group2][dv].values
                                
                                # Store data pairs for later calculations
                                data_pairs.append((group1, group2, data1, data2))
                                
                                # Calculate t-statistic and p-value
                                t_stat, p_val = stats.ttest_rel(data1, data2)
                                p_values.append(p_val)
                                t_stats.append(t_stat)

                            # Apply Holm-Sidak correction to all p-values at once
                            corrected_p_values = PostHocAnalyzer._holm_sidak_correction(p_values)

                            # Create comparison results using corrected p-values
                            for i, (group1, group2, data1, data2) in enumerate(data_pairs):
                                t_stat = t_stats[i]
                                p_val = p_values[i]  # Original p-value
                                corrected_p = corrected_p_values[i]  # Holm-Sidak corrected p-value
                                
                                # Calculate effect size (Cohen's d)
                                d = (np.mean(data1) - np.mean(data2)) / np.std(np.array(data1) - np.array(data2))
                                
                                results.setdefault("within_pairwise_comparisons", []).append({
                                    "group1": f"{rm_factor}={group1}",
                                    "group2": f"{rm_factor}={group2}",
                                    "test": "Paired t-test (Holm-Sidak)",  # Changed from "Bonferroni" to "Holm-Sidak"
                                    "statistic": float(t_stat),
                                    "p_value": float(corrected_p),
                                    "original_p": float(p_val),
                                    "significant": corrected_p < alpha,
                                    "corrected": True,
                                    "effect_size": float(d),
                                    "effect_size_type": "cohen_d"
                                })
                except Exception as ph_err:
                    results["warnings"] = results.get("warnings", []) + [f"Post-hoc failed: {ph_err}"]
                    
                # Enhanced Within-Factor Sphericity Testing for Mixed ANOVA
                rm_factor = within[0]
                within_sphericity_results = StatisticalTester._test_mixed_anova_within_sphericity(
                    df, dv, subject, rm_factor, aov, alpha
                )
                results.update(within_sphericity_results)
            else:
                print("DEBUG: Using statsmodels for Mixed ANOVA")
                # Fallback with statsmodels
                
                # Sanitize column names for statsmodels compatibility
                sanitized_df, column_mapping = StatisticalTester._sanitize_column_names_for_statsmodels(
                    df, [between_factor, rm_factor, dv]
                )
                
                # Use sanitized column names in formula
                sanitized_between = column_mapping[between_factor]
                sanitized_rm = column_mapping[rm_factor]
                sanitized_dv = column_mapping[dv]
                
                import statsmodels.api as sm
                from statsmodels.formula.api import ols
                formula = f"{sanitized_dv} ~ C({sanitized_between}) + C({sanitized_rm}) + C({sanitized_between}):C({sanitized_rm})"
                print(f"DEBUG: Mixed ANOVA formula with sanitized names: {formula}")
                
                model = ols(formula, data=sanitized_df).fit()
                anova = sm.stats.anova_lm(model, typ=2)

                # Effect sizes not available in fallback
                for factor, orig_factor in zip([sanitized_rm, sanitized_between], [rm_factor, between_factor]):
                    row = anova.loc[f"C({factor})"]
                    results["factors"].append({
                        "factor": orig_factor,  # Use original factor name in results
                        "type": "within" if orig_factor == rm_factor else "between",
                        "F": float(row["F"]),
                        "p_value": float(row["PR(>F)"]),
                        "df1": int(row["df"]),
                        "df2": int(anova.loc["Residual", "df"]),
                        "effect_size": None,
                        "effect_size_type": None
                    })

                interaction_key = f"C({between_factor}):C({rm_factor})"
                row = anova.loc[interaction_key]
                interaction = {
                    "factors": [rm_factor, between_factor],
                    "F": float(row["F"]),
                    "p_value": float(row["PR(>F)"]),
                    "df1": int(row["df"]),
                    "df2": int(anova.loc["Residual", "df"]),
                    "effect_size": None,
                    "effect_size_type": None
                }
                results["interactions"].append(interaction)
                results.update({
                    "p_value": interaction["p_value"],
                    "statistic": interaction["F"],
                    "df1": interaction["df1"],
                    "df2": interaction["df2"],
                    "effect_size": interaction["effect_size"],
                    "effect_size_type": interaction["effect_size_type"],
                    "test": f"Mixed ANOVA ({rm_factor} * {between_factor}) [statsmodels]"
                })
        except Exception as e:
            results["error"] = str(e)

        # Descriptive statistics
        for b in df[between_factor].unique():
            for w in df[rm_factor].unique():
                subset = df[(df[between_factor] == b) & (df[rm_factor] == w)][dv]
                key = f"{between_factor}={b}, {rm_factor}={w}"
                if len(subset) > 0:
                    n = len(subset)
                    mean = float(np.mean(subset))
                    std = float(np.std(subset, ddof=1))
                    stderr = std / np.sqrt(n) if n > 0 else 0
                    
                    # Calculate confidence interval
                    ci_lower = None
                    ci_upper = None
                    if n > 1:
                        try:
                            from scipy.stats import t
                            ci_lower, ci_upper = t.interval(0.95, n-1, loc=mean, scale=stderr)
                        except Exception:
                            pass
                    
                    results["descriptive"][key] = {
                        "n": n,
                        "mean": mean,
                        "sd": std,
                        "stderr": stderr,  # Changed from 'se' to 'stderr' for consistency
                        "ci_lower": ci_lower,  # Added confidence interval
                        "ci_upper": ci_upper,  # Added confidence interval
                        "min": float(np.min(subset)),
                        "max": float(np.max(subset)),
                        "median": float(np.median(subset))
                    }
        
        # Ensure a top-level result if nothing was set above
        if results.get("p_value", None) is None:
            if results["interactions"]:
                iv = results["interactions"][0]
                results["p_value"]         = iv["p_value"]
                results["statistic"]       = iv["F"]
                results["effect_size"]     = iv.get("effect_size")
                results["effect_size_type"]= iv.get("effect_size_type")
                results["df1"]             = iv.get("df1")
                results["df2"]             = iv.get("df2")
            elif results["factors"]:
                fv = results["factors"][0]
                results["p_value"]         = fv["p_value"]
                results["statistic"]       = fv["F"]
                results["effect_size"]     = fv.get("effect_size")
                results["effect_size_type"]= fv.get("effect_size_type")
                results["df1"]             = fv.get("df1")
                results["df2"]             = fv.get("df2")
        # Ensure pairwise_comparisons exists
        if "pairwise_comparisons" not in results:
            results["pairwise_comparisons"] = []

        # Consolidate all post-hoc results into the main pairwise_comparisons array
        if "between_pairwise_comparisons" in results and results["between_pairwise_comparisons"]:
            results["pairwise_comparisons"].extend(results["between_pairwise_comparisons"])
            
        if "within_pairwise_comparisons" in results and results["within_pairwise_comparisons"]:
            results["pairwise_comparisons"].extend(results["within_pairwise_comparisons"])
            
        return StatisticalTester._standardize_results(results)          
    
    @staticmethod
    def _run_repeated_measures_anova(df, dv, subject, within, alpha=0.05):
        """
        Performs a Repeated Measures ANOVA (one or more within factors).
        Prefers pingouin, fallback to statsmodels.
        """
        import numpy as np
        results = {
            "test": "Repeated Measures ANOVA",
            "p_value": None,
            "statistic": None,
            "effect_size": None,
            "effect_size_type": None,
            "factors": [],
            "interactions": [],
            "descriptive": {},
            "error": None
        }

        try:
            pg = get_pingouin_module()
            has_pingouin = True
        except ImportError:
            has_pingouin = False
            results["warning"] = "Pingouin not installed, using statsmodels"

        try:
            if has_pingouin:
                print("DEBUG: DataFrame columns:", df.columns)
                print("DEBUG: Unique values for within factor:", df[within[0]].unique())
                print("DEBUG: Unique values for subject:", df[subject].unique())
                print("DEBUG: First few rows of df:\n", df.head())
                print("DEBUG: Using Pingouin for RM ANOVA")    
                if len(within) == 1:
                    factor = within[0]
                    # Add correction=True to apply corrections for sphericity violation
                    aov = pg.rm_anova(data=df, dv=dv, within=factor, subject=subject, detailed=True, correction=True)
                    print("DEBUG: ANOVA result:", aov)
                    print("DEBUG: Results structure:", results)
                    results["anova_table"] = aov.copy()
                    row = aov.iloc[0]
                    error_row = aov[aov["Source"] == "Error"].iloc[0]
                    results["factors"].append({
                        "factor": factor,
                        "type": "within",
                        "F": float(row["F"]),
                        "p_value": float(row["p-unc"]),
                        "df1": int(row["DF"]),
                        "df2": int(error_row["DF"]),
                        "effect_size": float(row["ng2"]) if "ng2" in row else None,
                        "effect_size_type": "partial_eta_squared"
                    })
                    results.update({
                        "p_value": float(row["p-unc"]),
                        "statistic": float(row["F"]),
                        "effect_size": float(row["ng2"]) if "ng2" in row else None,
                        "effect_size_type": "partial_eta_squared",
                        "df1": int(row["DF"]),
                        "df2": int(error_row["DF"]),
                        "test": f"Repeated Measures ANOVA ({factor})"
                    })

                    # Enhanced Sphericity Testing with Professional Implementation
                    sphericity_results = StatisticalTester._perform_comprehensive_sphericity_test(
                        df, dv, subject, factor, aov, row, error_row
                    )
                    results.update(sphericity_results)
                                
                    # Automatic post-hoc tests for significant main effect
                    if results["p_value"] is not None and results["p_value"] < alpha:
                        try:
                            # Extract data for post-hoc tests
                            factor_levels = df[factor].unique()
                            factor_data = {}
                            for level in factor_levels:
                                factor_data[level] = df[df[factor] == level][dv].tolist()
                            
                            # Perform paired t-tests with Holm-Sidak correction
                            posthoc_results = StatisticalTester.perform_dependent_posthoc_tests(
                                factor_data, list(factor_levels), alpha=alpha, parametric=True
                            )
                            print(f"DEBUG: Post-hoc for RM-ANOVA created with {len(posthoc_results.get('pairwise_comparisons', []))} comparisons")
                            results["posthoc_test"] = posthoc_results.get("posthoc_test", "Paired t-tests (Holm-Sidak)")

                            # Initialize with empty list as default
                            results["pairwise_comparisons"] = []

                            # If we got valid posthoc results, use them
                            if posthoc_results and 'pairwise_comparisons' in posthoc_results and posthoc_results['pairwise_comparisons']:
                                # Deep copy to ensure data isn't lost
                                import copy
                                results["pairwise_comparisons"] = copy.deepcopy(posthoc_results['pairwise_comparisons'])
                                print(f"DEBUG: Added {len(results['pairwise_comparisons'])} pairwise comparisons to RM-ANOVA results")
                        except Exception as ph_err:
                            results["warnings"] = results.get("warnings", []) + [f"Post-hoc failed: {ph_err}"]
                            print(f"DEBUG: Post-hoc test error: {str(ph_err)}")
                else:
                    aov = pg.rm_anova(data=df, dv=dv, within=within, subject=subject, detailed=True)
                    for _, row in aov.iterrows():
                        if "*" in row["Source"]:
                            results["interactions"].append({
                                "factors": row["Source"].split("*"),
                                "F": float(row["F"]),
                                "p_value": float(row["p-unc"]),
                                "df1": int(row["DF1"]),
                                "df2": int(row["DF2"]),
                                "effect_size": float(row["np2"]),
                                "effect_size_type": "partial_eta_squared"
                            })
                        else:
                            results["factors"].append({
                                "factor": row["Source"],
                                "type": "within",
                                "F": float(row["F"]),
                                "p_value": float(row["p-unc"]),
                                "df1": int(row["DF1"]),
                                "df2": int(row["DF2"]),
                                "effect_size": float(row["np2"]),
                                "effect_size_type": "partial_eta_squared"
                            })

                    # Best result for main output
                    best_row = aov.iloc[aov["F"].argmax()]
                    results.update({
                        "p_value": float(best_row["p-unc"]),
                        "statistic": float(best_row["F"]),
                        "effect_size": float(best_row["np2"]),
                        "effect_size_type": "partial_eta_squared",
                        "df1": int(best_row["DF1"]),
                        "df2": int(best_row["DF2"]),
                        "test": "Repeated Measures ANOVA (multiple factors)"
                    })
            else:
                print("DEBUG: Using statsmodels for RM ANOVA")
                # Only simple fallback for one factor
                if len(within) != 1:
                    results["error"] = "Multiple within factors only possible with pingouin"
                    return StatisticalTester._standardize_results(results)
                    
                factor = within[0]
                
                # Sanitize column names for statsmodels compatibility
                sanitized_df, column_mapping = StatisticalTester._sanitize_column_names_for_statsmodels(
                    df, [factor, subject, dv]
                )
                
                # Use sanitized column names in formula
                sanitized_factor = column_mapping[factor]
                sanitized_subject = column_mapping[subject]
                sanitized_dv = column_mapping[dv]
                
                from statsmodels.formula.api import ols
                import statsmodels.api as sm
                formula = f"{sanitized_dv} ~ C({sanitized_factor}) + C({sanitized_subject})"
                print(f"DEBUG: RM ANOVA formula with sanitized names: {formula}")
                
                model = ols(formula, data=sanitized_df).fit()
                anova = sm.stats.anova_lm(model, typ=2)
                row = anova.loc[f"C({sanitized_factor})"]
                results["factors"].append({
                    "factor": factor,
                    "type": "within",
                    "F": float(row["F"]),
                    "p_value": float(row["PR(>F)"]),
                    "df1": int(row["df"]),
                    "df2": int(anova.loc['Residual', 'df']),
                    "effect_size": None,
                    "effect_size_type": None
                })
                results.update({
                    "p_value": float(row["PR(>F)"]),
                    "statistic": float(row["F"]),
                    "df1": int(row["df"]),
                    "df2": int(anova.loc['Residual', 'df']),
                    "test": f"Repeated Measures ANOVA ({factor}) [statsmodels]"
                })
        except Exception as e:
            results["error"] = str(e)
            print(f"DEBUG: Error in RM-ANOVA: {str(e)}")
            import traceback
            traceback.print_exc()

        # Descriptive statistics
        for factor in within:
            for val in df[factor].unique():
                subset = df[df[factor] == val][dv]
                n = len(subset)
                mean = float(np.mean(subset))
                std = float(np.std(subset, ddof=1))
                stderr = std / np.sqrt(n) if n > 0 else 0
                
                # Calculate confidence interval
                ci_lower = None
                ci_upper = None
                if n > 1:
                    try:
                        from scipy.stats import t
                        ci_lower, ci_upper = t.interval(0.95, n-1, loc=mean, scale=stderr)
                    except Exception:
                        pass
                
                results["descriptive"][f"{factor}={val}"] = {
                    "n": n,
                    "mean": mean,
                    "sd": std,
                    "stderr": stderr,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "min": float(np.min(subset)),
                    "max": float(np.max(subset)),
                    "median": float(np.median(subset))
                }

        # Ensure a top-level result if nothing was set above
        if results.get("p_value", None) is None:
            if results["interactions"]:
                iv = results["interactions"][0]
                results["p_value"]         = iv["p_value"]
                results["statistic"]       = iv["F"]
                results["effect_size"]     = iv.get("effect_size")
                results["effect_size_type"]= iv.get("effect_size_type")
                results["df1"]             = iv.get("df1")
                results["df2"]             = iv.get("df2")
            elif results["factors"]:
                fv = results["factors"][0]
                results["p_value"]         = fv["p_value"]
                results["statistic"]       = fv["F"]
                results["effect_size"]     = fv.get("effect_size")
                results["effect_size_type"]= fv.get("effect_size_type")
                results["df1"]             = fv.get("df1")
                results["df2"]             = fv.get("df2")

        return StatisticalTester._standardize_results(results)
    
    @staticmethod
    def _sanitize_column_names_for_statsmodels(df, columns):
        """
        Sanitize column names for statsmodels compatibility by removing spaces.
        Returns sanitized dataframe and mapping of old->new names.
        """
        sanitized_df = df.copy()
        column_mapping = {}
        
        for col in columns:
            if col in df.columns and ' ' in col:
                sanitized_name = col.replace(' ', '')
                sanitized_df = sanitized_df.rename(columns={col: sanitized_name})
                column_mapping[col] = sanitized_name
                print(f"DEBUG: Column name sanitized: '{col}' -> '{sanitized_name}'")
            else:
                column_mapping[col] = col
                
        return sanitized_df, column_mapping
    
    @staticmethod
    def _run_two_way_anova(df, dv, between, alpha=0.05):
        """
        Performs a Two-Way ANOVA (two between factors).

        Parameters:
        -----------
        df : pandas.DataFrame
            Data in long format
        dv : str
            Dependent variable
        between : list
            Two between factors
        alpha : float
            Significance level

        Returns:
        --------
        dict
            Results including main effects, interaction, effect sizes
        """
        import numpy as np
        import pandas as pd
        results = {
            "test": f"Two-Way ANOVA ({between[0]} * {between[1]})",
            "factors": [],
            "interactions": [],
            "p_value": None,
            "statistic": None,
            "df1": None,
            "df2": None,
            "effect_size": None,
            "effect_size_type": "partial_eta_squared",
            "descriptive": {},
            "error": None,
            "pairwise_comparisons": []
        }

        try:
            try:
                pg = get_pingouin_module()
                has_pingouin = True
            except ImportError:
                has_pingouin = False

            factor_a, factor_b = between[0], between[1]

            if has_pingouin:
                aov = pg.anova(data=df, dv=dv, between=between, detailed=True)
                results["anova_table"] = aov.copy()
                if "Residual" not in aov["Source"].values:
                    results["error"] = "Residuals not found in Pingouin ANOVA output. Cannot determine df2."
                    return StatisticalTester._standardize_results(results)

                residual_df_series = aov.loc[aov["Source"] == "Residual", "DF"]
                if residual_df_series.empty:
                    results["error"] = "Residual DF not found in Pingouin ANOVA output."
                    return StatisticalTester._standardize_results(results)
                residual_df = int(residual_df_series.iloc[0])

                # Main effects
                for factor in between:
                    if factor not in aov["Source"].values:
                        results.setdefault("warnings", []).append(f"Factor '{factor}' not found in Pingouin ANOVA output.")
                        continue
                    row = aov.loc[aov["Source"] == factor].iloc[0]
                    results["factors"].append({
                        "factor": factor,
                        "type": "between",
                        "F": float(row["F"]),
                        "p_value": float(row["p-unc"]),
                        "df1": int(row["DF"]),
                        "df2": residual_df,
                        "effect_size": float(row["np2"]),
                        "effect_size_type": "partial_eta_squared"
                    })

                # Interaction
                interaction_label = f"{factor_a} * {factor_b}"
                possible_interaction_labels = [interaction_label, f"{factor_b} * {factor_a}"]
                actual_interaction_label = None
                for label in possible_interaction_labels:
                    if label in aov["Source"].values:
                        actual_interaction_label = label
                        break

                if actual_interaction_label:
                    row = aov.loc[aov["Source"] == actual_interaction_label].iloc[0]
                    interaction_result = {
                        "factors": [factor_a, factor_b],
                        "F": float(row["F"]),
                        "p_value": float(row["p-unc"]),
                        "df1": int(row["DF"]),
                        "df2": residual_df,
                        "effect_size": float(row["np2"]),
                        "effect_size_type": "partial_eta_squared"
                    }
                    results["interactions"].append(interaction_result)
                    results["p_value"] = float(row["p-unc"])
                    results["statistic"] = float(row["F"])
                    results["df1"] = int(row["DF"])
                    results["df2"] = residual_df
                    results["effect_size"] = float(row["np2"])
                else:
                    results.setdefault("warnings", []).append(f"Interaction term for '{factor_a}' and '{factor_b}' not found in Pingouin ANOVA output.")
                    if not results["p_value"] and results["factors"]:
                        results["p_value"] = results["factors"][0]["p_value"]
                        results["statistic"] = results["factors"][0]["F"]
                        results["df1"] = results["factors"][0]["df1"]
                        results["df2"] = results["factors"][0]["df2"]
                        results["effect_size"] = results["factors"][0]["effect_size"]

                if actual_interaction_label and results["p_value"] is not None and results["p_value"] < alpha:
                    pingouin_success = False
                    try:
                        posthoc_df = pg.pairwise_tests(data=df, dv=dv, between=between, padjust='holm', subject=None)
                        if not posthoc_df.empty:
                            # Only set posthoc_test if we successfully process the results
                            temp_comparisons = []
                            for _, ph_row in posthoc_df.iterrows():
                                g1_label = str(ph_row.get('A', 'Group1'))
                                g2_label = str(ph_row.get('B', 'Group2'))
                                if 'Contrast' in ph_row and isinstance(ph_row['Contrast'], list) and len(ph_row['Contrast']) == 2:
                                    g1_label = str(ph_row['Contrast'][0])
                                    g2_label = str(ph_row['Contrast'][1])
                                elif 'Contrast' in ph_row and isinstance(ph_row['Contrast'], str) and 'vs.' in ph_row['Contrast']:
                                    parts = ph_row['Contrast'].split(' vs. ')
                                    if len(parts) == 2:
                                        g1_label = parts[0].strip()
                                        g2_label = parts[1].strip()
                                pval_col = 'p-corr' if 'p-corr' in ph_row else 'p-unc'
                                confidence_interval = (None, None)
                                if 'CI95%' in ph_row and isinstance(ph_row['CI95%'], (list, np.ndarray)) and len(ph_row['CI95%']) == 2:
                                    confidence_interval = tuple(ph_row['CI95%'])
                                elif 'CI95' in ph_row and isinstance(ph_row['CI95'], (list, np.ndarray)) and len(ph_row['CI95']) == 2:
                                    confidence_interval = tuple(ph_row['CI95'])
                                elif 'CLES' in ph_row and isinstance(ph_row['CLES'], (list, np.ndarray)) and len(ph_row['CLES']) == 2:
                                    confidence_interval = tuple(ph_row['CLES'])
                                elif 'ci' in ph_row and isinstance(ph_row['ci'], (list, np.ndarray)) and len(ph_row['ci']) == 2:
                                    confidence_interval = tuple(ph_row['ci'])
                                temp_comparisons.append({
                                    "group1": g1_label,
                                    "group2": g2_label,
                                    "test": "Pairwise t-test",
                                    "p_value": float(ph_row[pval_col]),
                                    "statistic": float(ph_row["T"]) if "T" in ph_row else None,
                                    "significant": float(ph_row[pval_col]) < alpha,
                                    "corrected": "Holm-Sidak",
                                    "confidence_interval": confidence_interval
                                })
                            
                            # Only set posthoc_test and add comparisons if we successfully processed all rows
                            if temp_comparisons:
                                results["posthoc_test"] = "Tukey HSD Test (Pingouin)"
                                results["pairwise_comparisons"].extend(temp_comparisons)
                                pingouin_success = True
                        else:
                            results.setdefault("warnings", []).append("Pingouin pairwise_tests for interaction returned empty.")
                    except Exception as e_ph:
                        results.setdefault("warnings", []).append(f"Post-hoc tests (Pingouin) for interaction failed: {str(e_ph)}")

            else: # Fallback to statsmodels
                import statsmodels.api as sm
                from statsmodels.formula.api import ols
                print("DEBUG: WARNING! Two-Way ANOVA uses statsmodels fallback!")
                print("DEBUG: Pingouin not installed or import failed.")

                # Sanitize column names for statsmodels compatibility
                sanitized_df, column_mapping = StatisticalTester._sanitize_column_names_for_statsmodels(
                    df, [factor_a, factor_b, dv]
                )
                
                # Use sanitized column names in formula
                sanitized_factor_a = column_mapping[factor_a]
                sanitized_factor_b = column_mapping[factor_b] 
                sanitized_dv = column_mapping[dv]

                formula = f"`{sanitized_dv}` ~ C(`{sanitized_factor_a}`) * C(`{sanitized_factor_b}`)"
                print(f"DEBUG: Two-Way ANOVA formula with sanitized names: {formula}")
                
                model = ols(formula, data=sanitized_df).fit()
                aov = sm.stats.anova_lm(model, typ=2)
                if "Residual" not in aov.index:
                    results["error"] = "Residuals not found in statsmodels ANOVA output."
                    return StatisticalTester._standardize_results(results)
                residual_df = int(aov.loc["Residual", "df"])

                # Main effects
                for factor in [factor_a, factor_b]:
                    sanitized_factor = column_mapping[factor]
                    factor_term = f"C(`{sanitized_factor}`)"
                    if factor_term not in aov.index:
                        results.setdefault("warnings", []).append(f"Factor term '{factor_term}' not found in statsmodels ANOVA output.")
                        continue
                    row = aov.loc[factor_term]
                    results["factors"].append({
                        "factor": factor,
                        "type": "between",
                        "F": float(row["F"]),
                        "p_value": float(row["PR(>F)"]),
                        "df1": int(row["df"]),
                        "df2": residual_df,
                        "effect_size": None,
                        "effect_size_type": None
                    })

                # Interaction
                sanitized_factor_a = column_mapping[factor_a]
                sanitized_factor_b = column_mapping[factor_b]
                interaction_term = f"C(`{sanitized_factor_a}`):C(`{sanitized_factor_b}`)"
                if interaction_term in aov.index:
                    row = aov.loc[interaction_term]
                    interaction_result = {
                        "factors": [factor_a, factor_b],
                        "F": float(row["F"]),
                        "p_value": float(row["PR(>F)"]),
                        "df1": int(row["df"]),
                        "df2": residual_df,
                        "effect_size": None,
                        "effect_size_type": None
                    }
                    results["interactions"].append(interaction_result)
                    results["p_value"] = float(row["PR(>F)"])
                    results["statistic"] = float(row["F"])
                    results["df1"] = int(row["df"])
                    results["df2"] = residual_df
                    results["effect_size"] = None
                    results["test"] += " [statsmodels]"
                else:
                    results.setdefault("warnings", []).append(f"Interaction term '{interaction_term}' not found in statsmodels ANOVA output.")
                    if not results["p_value"] and results["factors"]:
                        results["p_value"] = results["factors"][0]["p_value"]
                        results["statistic"] = results["factors"][0]["F"]
                        results["df1"] = results["factors"][0]["df1"]
                        results["df2"] = results["factors"][0]["df2"]

                if (results["p_value"] is not None and results["p_value"] < alpha) or any(factor["p_value"] < alpha for factor in results["factors"]):
                    statsmodels_success = False
                    try:
                        factor_a, factor_b = between[0], between[1]
                        df['interaction_group'] = df[factor_a].astype(str) + "_" + df[factor_b].astype(str)
                        from statsmodels.stats.multicomp import pairwise_tukeyhsd
                        tukey = pairwise_tukeyhsd(df[dv], df['interaction_group'], alpha=alpha)
                        
                        # Only override posthoc_test and add comparisons if we successfully compute all results
                        temp_comparisons = []
                        if "pairwise_comparisons" not in results:
                            results["pairwise_comparisons"] = []
                        for i in range(len(tukey.pvalues)):
                            group1 = tukey.groupsunique[tukey.pairindices[i, 0]]
                            group2 = tukey.groupsunique[tukey.pairindices[i, 1]]
                            p_val = tukey.pvalues[i]
                            is_significant = tukey.reject[i]
                            conf_int = tukey.confint[i]
                            group1_values = df[df['interaction_group'] == group1][dv].values
                            group2_values = df[df['interaction_group'] == group2][dv].values
                            effect_size = None
                            try:
                                from scipy import stats
                                n1, n2 = len(group1_values), len(group2_values)
                                s1, s2 = np.var(group1_values, ddof=1), np.var(group2_values, ddof=1)
                                s_pooled = np.sqrt(((n1-1)*s1 + (n2-1)*s2) / (n1+n2-2))
                                effect_size = (np.mean(group1_values) - np.mean(group2_values)) / s_pooled if s_pooled > 0 else 0
                            except Exception:
                                effect_size = None
                            temp_comparisons.append({
                                "group1": str(group1),
                                "group2": str(group2),
                                "test": "Tukey HSD",
                                "p_value": float(p_val),
                                "significant": bool(is_significant),
                                "corrected": True,
                                "correction": "Tukey HSD",
                                "effect_size": effect_size,
                                "effect_size_type": "cohen_d",
                                "confidence_interval": tuple(conf_int)
                            })
                        
                        # Only set posthoc_test and add comparisons if we successfully processed all results
                        if temp_comparisons:
                            results["posthoc_test"] = "Tukey HSD for interaction effect"
                            results["pairwise_comparisons"].extend(temp_comparisons)
                            statsmodels_success = True
                    except Exception as e_ph:
                        results.setdefault("warnings", []).append(f"Post-hoc Tests failed: {str(e_ph)}")

            # Descriptive statistics
            for a_val in df[factor_a].unique():
                for b_val in df[factor_b].unique():
                    group_key = f"{factor_a}={a_val}, {factor_b}={b_val}"
                    subset = df[(df[factor_a] == a_val) & (df[factor_b] == b_val)][dv]
                    if len(subset) > 0:
                        mean_val = float(np.mean(subset))
                        std_val = float(np.std(subset, ddof=1))
                        n_val = len(subset)
                        se_val = std_val / np.sqrt(n_val) if n_val > 0 else 0
                        ci_desc = (None, None)
                        if n_val > 1:
                            try:
                                from scipy.stats import t
                                ci_desc = t.interval(0.95, n_val - 1, loc=mean_val, scale=se_val)
                            except:
                                pass
                        results["descriptive"][group_key] = {
                            "n": n_val,
                            "mean": mean_val,
                            "sd": std_val,
                            "se": se_val,
                            "ci_lower": ci_desc[0],
                            "ci_upper": ci_desc[1],
                            "min": float(np.min(subset)),
                            "max": float(np.max(subset)),
                            "median": float(np.median(subset))
                        }
                    else:
                        results["descriptive"][group_key] = {
                            "n": 0, "mean": None, "sd": None, "se": None,
                            "ci_lower": None, "ci_upper": None,
                            "min": None, "max": None, "median": None
                        }

        except Exception as e:
            import traceback
            results["error"] = f"Error in Two-Way ANOVA: {str(e)}. Trace: {traceback.format_exc()}"
            results["test"] += " (failed)"

        # Ensure top-level results are set, prioritizing interaction, then first main effect.
        if results.get("p_value") is None:
            if results["interactions"] and results["interactions"][0].get("p_value") is not None:
                iv = results["interactions"][0]
                results["p_value"] = iv["p_value"]
                results["statistic"] = iv["F"]
                results["effect_size"] = iv.get("effect_size")
                results["effect_size_type"] = iv.get("effect_size_type")
                results["df1"] = iv.get("df1")
                results["df2"] = iv.get("df2")
            elif results["factors"] and results["factors"][0].get("p_value") is not None:
                fv = results["factors"][0]
                results["p_value"] = fv["p_value"]
                results["statistic"] = fv["F"]
                results["effect_size"] = fv.get("effect_size")
                results["effect_size_type"] = fv.get("effect_size_type")
                results["df1"] = fv.get("df1")
                results["df2"] = fv.get("df2")
        return StatisticalTester._standardize_results(results)
    
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
        return StatisticalTester.perform_refactored_posthoc_testing(
            groups,
            data_dict,
            "parametric" if parametric else "non_parametric",
            alpha=alpha,
            posthoc_choice="dependent",
            is_dependent=True
        )
    @staticmethod
    def perform_refactored_posthoc_testing(valid_groups, samples, test_recommendation, alpha=0.05, posthoc_choice=None, control_group=None, is_dependent=False):
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

        # Check validity
        if len(valid_groups) <= 1:
            result["error"] = "At least two groups are required for post-hoc tests."
            return result

        # Automatic test selection if not explicitly specified
        if posthoc_choice is None:
            if is_dependent:
                posthoc_choice = "dependent"
            elif test_recommendation == "parametric":
                # Show dialog for parametric post-hoc test selection
                try:
                    posthoc_choice = UIDialogManager.select_posthoc_test_dialog(
                        parent=None, progress_text=None, column_name=None
                    )
                    print(f"DEBUG: Parametric post-hoc dialog returned: {posthoc_choice}")
                    # If dialog was cancelled or returned None, default to Tukey
                    if posthoc_choice is None:
                        posthoc_choice = "tukey"
                        print("DEBUG: Parametric post-hoc dialog cancelled, defaulting to Tukey HSD")
                except Exception as e:
                    print(f"DEBUG: Error showing parametric post-hoc dialog: {e}")
                    posthoc_choice = "tukey"  # Fallback to Tukey HSD
            else:
                # NEW: Dialog for non-parametric post-hoc tests
                print("DEBUG: About to show non-parametric post-hoc dialog")
                try:
                    posthoc_choice = UIDialogManager.select_nonparametric_posthoc_dialog(
                        parent=None, progress_text=None, column_name=None
                    )
                    print(f"DEBUG: Non-parametric post-hoc dialog returned: {posthoc_choice}")
                    # If dialog was cancelled or returned None, default to Dunn
                    if posthoc_choice is None:
                        posthoc_choice = "dunn"
                        print("DEBUG: Non-parametric post-hoc dialog cancelled, defaulting to Dunn test")
                except Exception as e:
                    print(f"DEBUG: Error showing non-parametric post-hoc dialog: {e}")
                    import traceback
                    traceback.print_exc()
                    posthoc_choice = "dunn"  # Fallback to Dunn test
        
        # If Dunnett was selected, we need a control group
        if posthoc_choice == "dunnett" and control_group is None:
            try:
                control_group = UIDialogManager.select_control_group_dialog(valid_groups)
                print(f"DEBUG: Control group selected for Dunnett test: {control_group}")
                if control_group is None:
                    print("DEBUG: No control group selected, defaulting to Tukey HSD")
                    posthoc_choice = "tukey"
            except Exception as e:
                print(f"DEBUG: Error selecting control group: {e}")
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
                pairs = UIDialogManager.select_custom_pairs_dialog(valid_groups)
                if not pairs:
                    result["error"] = "No pairs selected."
                    return result
                # Import required modules
                from scipy import stats
                import numpy as np
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
                pairs = UIDialogManager.select_custom_pairs_dialog(valid_groups)
                if not pairs:
                    result["error"] = "No pairs selected."
                    return result
                from scipy.stats import mannwhitneyu
                import numpy as np
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

    @staticmethod
    def _perform_comprehensive_sphericity_test(df, dv, subject, factor, aov, row, error_row):
        """
        Performs comprehensive sphericity testing with Mauchly's test and corrections.
        
        Includes:
        - Mauchly's Test for Sphericity 
        - Greenhouse-Geisser Correction
        - Huynh-Feldt Correction
        - Intelligent correction selection
        - Detailed interpretation
        
        Parameters:
        -----------
        df : DataFrame
            Data containing repeated measures
        dv : str
            Dependent variable column name
        subject : str
            Subject identifier column name
        factor : str
            Within-subjects factor column name
        aov : DataFrame
            ANOVA results table from pingouin
        row : Series
            Main effect row from ANOVA table
        error_row : Series
            Error term row from ANOVA table
            
        Returns:
        --------
        dict
            Comprehensive sphericity analysis results
        """
        import numpy as np
        import pandas as pd
        
        results = {}
        
        try:
            # Get factor levels and check if sphericity is relevant
            within_levels = df[factor].unique()
            k = len(within_levels)
            
            if k <= 2:
                # Sphericity always met with 2 levels
                results["sphericity_test"] = {
                    "test_name": "Mauchly's Test for Sphericity",
                    "W": None,
                    "chi_square": None,
                    "df": None,
                    "p_value": None,
                    "sphericity_assumed": True,
                    "note": "Sphericity assumption is always met with 2 levels",
                    "interpretation": "No correction needed - only 2 conditions compared"
                }
                results["corrected_p_value"] = float(row["p-unc"])
                results["correction_used"] = "None (sphericity assumption met)"
                return results
            
            # Attempt comprehensive sphericity testing
            pg = get_pingouin_module()
            sphericity_violated = False
            mauchly_results = {}
            
            # Primary method: Use pingouin's sphericity function
            try:
                sphericity_result = pg.sphericity(df, dv=dv, subject=subject, within=factor)
                
                # Handle different return formats
                if isinstance(sphericity_result, tuple) and len(sphericity_result) >= 3:
                    W, pval, spher = sphericity_result[:3]
                    mauchly_results = {
                        "test_name": "Mauchly's Test for Sphericity",
                        "W": float(W) if W is not None else None,
                        "p_value": float(pval) if pval is not None else None,
                        "sphericity_assumed": bool(spher) if spher is not None else None,
                        "df": int((k * (k - 1)) / 2 - 1) if k > 2 else None,
                        "interpretation": StatisticalTester._interpret_sphericity_test(pval, spher) if pval is not None else "Test failed"
                    }
                    sphericity_violated = not bool(spher) if spher is not None else True
                else:
                    raise ValueError("Unexpected sphericity test output format")
                    
            except Exception as e:
                # Fallback: Extract from ANOVA table if available
                mauchly_results = StatisticalTester._extract_sphericity_from_anova_table(aov, k)
                sphericity_violated = not mauchly_results.get("sphericity_assumed", True)
                
            results["sphericity_test"] = mauchly_results
            
            # Apply corrections based on sphericity violation
            corrections_applied = StatisticalTester._apply_sphericity_corrections(
                row, error_row, sphericity_violated, aov
            )
            results.update(corrections_applied)
            
        except Exception as e:
            # Comprehensive fallback
            results["sphericity_test"] = {
                "test_name": "Mauchly's Test for Sphericity",
                "W": None,
                "p_value": None,
                "sphericity_assumed": None,
                "note": f"Sphericity test failed: {str(e)}",
                "interpretation": "Could not determine sphericity - proceeding with caution"
            }
            results["corrected_p_value"] = float(row["p-unc"])
            results["correction_used"] = "None (sphericity test failed)"
            
        return results
    
    @staticmethod
    def _interpret_sphericity_test(p_value, sphericity_met):
        """
        Provides interpretation of Mauchly's sphericity test results.
        
        Parameters:
        -----------
        p_value : float
            P-value from Mauchly's test
        sphericity_met : bool
            Whether sphericity assumption is met
            
        Returns:
        --------
        str
            Human-readable interpretation
        """
        if p_value is None:
            return "Could not determine sphericity"
            
        if sphericity_met:
            return f"Sphericity assumption is met (p = {p_value:.4f}). No correction needed."
        else:
            return f"Sphericity assumption is violated (p = {p_value:.4f}). Corrections recommended."
    
    @staticmethod
    def _extract_sphericity_from_anova_table(aov, k):
        """
        Attempts to extract sphericity information from ANOVA table.
        
        Parameters:
        -----------
        aov : DataFrame
            ANOVA results table
        k : int
            Number of factor levels
            
        Returns:
        --------
        dict
            Sphericity test results
        """
        try:
            # Check for sphericity columns in ANOVA table
            sphericity_cols = ['W-spher', 'p-spher', 'sphericity']
            available_cols = [col for col in sphericity_cols if col in aov.columns]
            
            if available_cols:
                first_row = aov.iloc[0]
                return {
                    "test_name": "Mauchly's Test for Sphericity",
                    "W": float(first_row.get('W-spher', np.nan)) if 'W-spher' in aov.columns else None,
                    "p_value": float(first_row.get('p-spher', np.nan)) if 'p-spher' in aov.columns else None,
                    "sphericity_assumed": bool(first_row.get('sphericity', True)) if 'sphericity' in aov.columns else None,
                    "df": int((k * (k - 1)) / 2 - 1) if k > 2 else None,
                    "note": "Extracted from ANOVA table",
                    "interpretation": "See p-value for significance"
                }
            else:
                return {
                    "test_name": "Mauchly's Test for Sphericity",
                    "W": None,
                    "p_value": None,
                    "sphericity_assumed": True,  # Conservative assumption
                    "df": int((k * (k - 1)) / 2 - 1) if k > 2 else None,
                    "note": "No sphericity information in ANOVA table",
                    "interpretation": "Assuming sphericity (could not test)"
                }
        except Exception:
            return {
                "test_name": "Mauchly's Test for Sphericity",
                "W": None,
                "p_value": None,
                "sphericity_assumed": True,
                "note": "Failed to extract sphericity information",
                "interpretation": "Defaulting to sphericity assumption"
            }
    
    @staticmethod
    def _apply_sphericity_corrections(row, error_row, sphericity_violated, aov):
        """
        Applies appropriate sphericity corrections based on violation status.
        
        Parameters:
        -----------
        row : Series
            Main effect row from ANOVA table
        error_row : Series
            Error term row from ANOVA table
        sphericity_violated : bool
            Whether sphericity assumption is violated
        aov : DataFrame
            Full ANOVA results table
            
        Returns:
        --------
        dict
            Correction results and recommendations
        """
        corrections = {}
        
        try:
            if not sphericity_violated:
                # No correction needed
                corrections["sphericity_corrections"] = {
                    "needed": False,
                    "reason": "Sphericity assumption is met"
                }
                corrections["corrected_p_value"] = float(row["p-unc"])
                corrections["correction_used"] = "None (sphericity assumption met)"
                corrections["final_p_value"] = float(row["p-unc"])
                return corrections
            
            # Sphericity violated - apply corrections
            corrections["sphericity_corrections"] = {"needed": True}
            
            # Greenhouse-Geisser Correction
            if 'GG-eps' in row and 'p-GG' in row:
                gg_epsilon = float(row["GG-eps"])
                gg_p_value = float(row["p-GG"])
                
                corrections["sphericity_corrections"]["greenhouse_geisser"] = {
                    "epsilon": gg_epsilon,
                    "corrected_df1": float(row["DF"]) * gg_epsilon,
                    "corrected_df2": float(error_row["DF"]) * gg_epsilon,
                    "p_value": gg_p_value,
                    "conservative": True,
                    "description": "Conservative correction for sphericity violation"
                }
            else:
                gg_epsilon = None
                gg_p_value = float(row["p-unc"])
            
            # Huynh-Feldt Correction  
            if 'HF-eps' in row and 'p-HF' in row:
                hf_epsilon = float(row["HF-eps"])
                hf_p_value = float(row["p-HF"])
                
                corrections["sphericity_corrections"]["huynh_feldt"] = {
                    "epsilon": hf_epsilon,
                    "corrected_df1": float(row["DF"]) * hf_epsilon,
                    "corrected_df2": float(error_row["DF"]) * hf_epsilon,
                    "p_value": hf_p_value,
                    "conservative": False,
                    "description": "Less conservative correction, preferred when ε > 0.75"
                }
            else:
                hf_epsilon = None
                hf_p_value = float(row["p-unc"])
            
            # Intelligent correction selection
            if gg_epsilon is not None and hf_epsilon is not None:
                if gg_epsilon > 0.75:
                    # Use Huynh-Feldt for higher epsilon values
                    corrections["corrected_p_value"] = hf_p_value
                    corrections["correction_used"] = f"Huynh-Feldt (ε = {hf_epsilon:.3f} > 0.75)"
                    corrections["final_p_value"] = hf_p_value
                    corrections["recommendation"] = "Huynh-Feldt correction recommended (less conservative)"
                else:
                    # Use Greenhouse-Geisser for lower epsilon values
                    corrections["corrected_p_value"] = gg_p_value
                    corrections["correction_used"] = f"Greenhouse-Geisser (ε = {gg_epsilon:.3f} ≤ 0.75)"
                    corrections["final_p_value"] = gg_p_value
                    corrections["recommendation"] = "Greenhouse-Geisser correction recommended (more conservative)"
            elif gg_epsilon is not None:
                corrections["corrected_p_value"] = gg_p_value
                corrections["correction_used"] = f"Greenhouse-Geisser (ε = {gg_epsilon:.3f})"
                corrections["final_p_value"] = gg_p_value
            elif hf_epsilon is not None:
                corrections["corrected_p_value"] = hf_p_value
                corrections["correction_used"] = f"Huynh-Feldt (ε = {hf_epsilon:.3f})"
                corrections["final_p_value"] = hf_p_value
            else:
                # Fallback to uncorrected
                corrections["corrected_p_value"] = float(row["p-unc"])
                corrections["correction_used"] = "None (corrections not available)"
                corrections["final_p_value"] = float(row["p-unc"])
                corrections["recommendation"] = "Consider multivariate approach or non-parametric alternatives"
                
        except Exception as e:
            corrections["sphericity_corrections"] = {
                "needed": True,
                "error": f"Failed to apply corrections: {str(e)}"
            }
            corrections["corrected_p_value"] = float(row["p-unc"])
            corrections["correction_used"] = "None (correction failed)"
            corrections["final_p_value"] = float(row["p-unc"])
            
        return corrections

    @staticmethod
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
        import numpy as np
        import pandas as pd
        from scipy import stats
        
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
            levene_results = StatisticalTester._perform_levene_test(group_data, group_labels, dv, between_factor)
            
            # 2. Brown-Forsythe Test (robust to non-normality)
            brown_forsythe_results = StatisticalTester._perform_brown_forsythe_test(group_data, group_labels, dv, between_factor)
            
            # 3. Bartlett's Test (sensitive to normality)
            bartlett_results = StatisticalTester._perform_bartlett_test(group_data, group_labels, dv, between_factor)
            
            # 4. Welch's ANOVA (unequal variances correction)
            welch_results = StatisticalTester._perform_welch_anova(group_data, group_labels, dv, between_factor)
            
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
                "recommendations": StatisticalTester._generate_between_assumption_recommendations(
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
            from scipy.stats import levene
            
            # Perform Levene's test
            statistic, p_value = levene(*group_data)
            
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
            from scipy.stats import levene
            
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
            from scipy.stats import bartlett
            
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
            from scipy.stats import f_oneway
            
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
                from scipy.stats import f
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
                    recommendations.append("ℹ️ Bartlett's test differs from Levene's - may indicate non-normality")
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
                recommendations.append("ℹ️ Unable to assess assumptions - proceed with caution")
                
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
        import numpy as np
        import pandas as pd
        
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
                        "sphericity_assumed": bool(spher) if spher is not None else None,
                        "df": int((k * (k - 1)) / 2 - 1) if k > 2 else None,
                        "interpretation": StatisticalTester._interpret_sphericity_test(pval, spher) if pval is not None else "Test failed",
                        "levels_tested": k,
                        "comparisons": int(k * (k - 1) / 2)
                    }
                else:
                    raise ValueError("Unexpected sphericity test output format")
                    
            except Exception as e:
                # Fallback: Extract from ANOVA table if available
                sphericity_results["within_sphericity_test"] = StatisticalTester._extract_mixed_sphericity_from_anova_table(
                    aov, within_factor, k
                )
                sphericity_violated = not sphericity_results["within_sphericity_test"].get("sphericity_assumed", True)
                
            # Apply corrections for within-factor effects in Mixed ANOVA
            if sphericity_violated:
                corrections_applied = StatisticalTester._apply_mixed_anova_sphericity_corrections(
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
                "sphericity_assumed": None,
                "note": f"Within-factor sphericity test failed: {str(e)}",
                "interpretation": "Could not determine sphericity for within-factor - proceeding with caution"
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
                        "sphericity_assumed": True,  # Conservative assumption
                        "df": int((k * (k - 1)) / 2 - 1) if k > 2 else None,
                        "note": "No sphericity information in Mixed ANOVA table",
                        "interpretation": "Assuming sphericity for within-factor (could not test)",
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
                "sphericity_assumed": True,
                "note": f"Failed to extract within-factor sphericity: {str(e)}",
                "interpretation": "Defaulting to sphericity assumption for within-factor",
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
                within_corrections = StatisticalTester._apply_corrections_to_effect_row(
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
                    interaction_corrections[interaction_name] = StatisticalTester._apply_corrections_to_effect_row(
                        interaction_row, f"interaction ({interaction_name})"
                    )
                corrections["within_sphericity_corrections"]["interactions"] = interaction_corrections
            
            # Overall recommendation for within-factor
            corrections["within_correction_recommendation"] = StatisticalTester._generate_within_factor_recommendations(
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
                gg_p_value = float(effect_row["p-unc"])
            
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
                hf_p_value = float(effect_row["p-unc"])
            
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
                corrections["final_p_value"] = float(effect_row["p-unc"])
                corrections["correction_used"] = "None (corrections not available)"
                corrections["rationale"] = "Correction information not available - consider multivariate approach"
                
        except Exception as e:
            corrections["error"] = f"Failed to apply corrections to {effect_name}: {str(e)}"
            corrections["final_p_value"] = float(effect_row["p-unc"])
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
        import numpy as np
        import pandas as pd
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
            interaction_sphericity = StatisticalTester._test_interaction_sphericity(
                df, dv, between_factor, within_factor, subject, aov, alpha
            )
            
            # 2. Homogeneity Tests for Interaction Cells
            cell_homogeneity = StatisticalTester._test_interaction_cell_homogeneity(
                df, dv, between_factor, within_factor, alpha
            )
            
            # 3. Covariance Pattern Tests
            covariance_tests = StatisticalTester._test_interaction_covariance_patterns(
                df, dv, between_factor, within_factor, subject, alpha
            )
            
            # 4. Box's M Test for Homogeneity of Covariance Matrices
            box_m_test = StatisticalTester._perform_box_m_test(
                df, dv, between_factor, within_factor, subject
            )
            
            # 5. Compound Symmetry Assessment
            compound_symmetry = StatisticalTester._assess_compound_symmetry(
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
                "overall_recommendations": StatisticalTester._generate_interaction_recommendations(
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
                    "sphericity_assumed": None
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
                            "sphericity_assumed": bool(spher) if spher is not None else None,
                            "interpretation": "Direct interaction sphericity test performed"
                        })
                    else:
                        raise ValueError("Unexpected sphericity test output")
                        
                except Exception as inner_e:
                    sphericity_result.update({
                        "W": None,
                        "p_value": None,
                        "sphericity_assumed": None,
                        "note": f"Direct sphericity test failed: {str(inner_e)}",
                        "interpretation": "Could not determine interaction sphericity"
                    })
            
            # Add corrections if available
            if 'GG-eps' in interaction_row and 'HF-eps' in interaction_row:
                corrections = StatisticalTester._apply_corrections_to_effect_row(
                    interaction_row, f"interaction ({interaction_name})"
                )
                sphericity_result["corrections"] = corrections
            
            return sphericity_result
            
        except Exception as e:
            return {
                "test_name": "Interaction Sphericity Test",
                "interaction": f"{within_factor} * {between_factor}",
                "error": f"Interaction sphericity test failed: {str(e)}",
                "sphericity_assumed": None
            }
    
    @staticmethod
    def _test_interaction_cell_homogeneity(df, dv, between_factor, within_factor, alpha):
        """
        Tests homogeneity of variance across all interaction cells.
        
        Creates a comprehensive test of whether variances are equal across
        all combinations of between and within factor levels.
        """
        try:
            from scipy.stats import levene, bartlett
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
            
            # Perform Levene's test on all cells
            levene_stat, levene_p = levene(*cell_data)
            
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
            import numpy as np
            from scipy.stats import pearsonr
            
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