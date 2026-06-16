import numpy as np
import pandas as pd
import logging
from scipy import stats
from analysis.stats_functions import get_pingouin_module, PostHocAnalyzer, UIDialogManager
from core.methodology_trace import MethodologyTrace
from statistical_testing.decision_logic import (
    extract_assumption_state,
    select_comparison_test,
    strategy_to_recommendation,
)
from statistical_testing.engines.comparison import ComparisonEngine
from statistical_testing.engines.posthoc import PostHocEngine
from statistical_testing.advanced_pipeline import perform_advanced_test_pipeline
from statistical_testing.assumption_checks import AssumptionCheckEngine
from statistical_testing.mixed_assumptions import MixedAnovaAssumptionEngine
from statistical_testing.posthoc_fallback import PosthocFallbackEngine
from statistical_testing.models import StatisticalResult
from statistical_testing.validators import (
    GroupValidationError,
    ModelDesignError,
    PairedDataError,
    ValidationError,
    validate_group_count,
    ensure_equal_group_sizes,
    validate_minimum_n,
    validate_paired_data,
    validate_test_design,
    MIN_N_HARD,
    MIN_N_BLOCK,
)

# LOW-1: module-level logger (use logging.getLogger in each method for context)
logger = logging.getLogger(__name__)

# LOW-2: module-level constants — avoids magic numbers scattered through tests
DEFAULT_ALPHA = 0.05
NORMALITY_THRESHOLD = 0.05
CI_LEVEL = 0.95
MIN_GROUP_SIZE = 2

class StatisticalTester:
    @staticmethod
    def make_blocked_result(reason, *, code, details=None, warnings=None):
        """Build a standardized 'blocked' result for a data-quality pre-flight
        failure. Callers (analysis_core) return this instead of running a test so
        the UI/report can surface a clear reason rather than a crash or a
        silently-wrong number. ``blocked=True`` is the detection flag."""
        blocked = {
            "test": "Not performed",
            "blocked": True,
            "block_reason": reason,
            "block_code": code,
            "error": reason,
            "p_value": None,
            "statistic": None,
            "pairwise_comparisons": [],
            "warnings": list(warnings or []),
            "data_quality": details or {},
        }
        return StatisticalTester._standardize_results(blocked)

    @staticmethod
    def nonfinite_block(results):
        """Safety net for advanced engines (LMM, RM/Mixed/Two-Way ANOVA, ANCOVA)
        that can emit a non-finite statistic / p-value on a degenerate design
        (zero variance, collinear covariate, rank-deficient model) WITHOUT raising.
        Returns a standardized block if `results` carries such a value and is not
        already blocked, else None. Keeps the silent-wrong-number from reaching
        the UI/report."""
        if not isinstance(results, dict) or results.get("blocked"):
            return None

        def _nonfinite(x):
            return (isinstance(x, (int, float)) and not isinstance(x, bool)
                    and not bool(np.isfinite(x)))

        if not (_nonfinite(results.get("p_value")) or _nonfinite(results.get("statistic"))):
            return None

        reason = (
            f"{results.get('test') or 'The test'} produced a non-finite result "
            "(infinite or undefined). This indicates a degenerate design — e.g. "
            "zero variance, perfectly collinear covariates, or a rank-deficient "
            "model — for which the test is not defined."
        )
        return StatisticalTester.make_blocked_result(
            reason, code="NON_FINITE_RESULT", details={"test": results.get("test")},
        )

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
            "final_test_label": None,
            "tested_against": None,
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
            logger.debug(f"DEBUG: Standardizing results with {len(results['pairwise_comparisons'])} pairwise comparisons")
        
        # Copy all existing values first
        standardized = dict(results)

        if standardized.get("test") and not standardized.get("final_test_label"):
            standardized["final_test_label"] = standardized["test"]
        if standardized.get("final_test_label") and not standardized.get("tested_against"):
            standardized["tested_against"] = standardized["final_test_label"]
        
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

        # Check minimum N warnings (G2 Fix)
        from statistical_testing.validators import MIN_N_HARD
        descriptives = standardized.get("descriptive", {})
        if isinstance(descriptives, dict):
            for group, grp_stats in descriptives.items():
                n = grp_stats.get("n", 0) if isinstance(grp_stats, dict) else 0
                if 0 < n < MIN_N_HARD:
                    standardized.setdefault("warnings", []).append(
                        f"WARNING: N={n} for group '{group}' is a small sample (< {MIN_N_HARD}). "
                        "Statistical power is low; interpret results with caution."
                    )

        return standardized

    @staticmethod
    def to_statistical_result(results):
        """Convert a legacy result dict into the canonical StatisticalResult DTO."""
        standardized = StatisticalTester._standardize_results(results or {})
        return StatisticalResult.from_legacy_dict(standardized)

    @staticmethod
    def from_statistical_result(result):
        """Convert StatisticalResult DTO back to legacy dict format for compatibility."""
        if not isinstance(result, StatisticalResult):
            raise TypeError("result must be an instance of StatisticalResult")
        legacy = result.to_legacy_dict()
        return StatisticalTester._standardize_results(legacy)

    @staticmethod
    def _pingouin_p_column(columns):
        """Return the available Pingouin uncorrected p-value column name."""
        if columns is None:
            return None
        if "p_unc" in columns:
            return "p_unc"
        if "p-unc" in columns:
            return "p-unc"
        return None

    @staticmethod
    def _pingouin_p_value(row, default=np.nan):
        """Safely extract uncorrected p-value from a Pingouin result row."""
        if row is None:
            return default
        columns = row.index if hasattr(row, "index") else row.keys() if hasattr(row, "keys") else None
        p_col = StatisticalTester._pingouin_p_column(columns)
        if p_col is None:
            return default
        raw_value = row.get(p_col, default) if hasattr(row, "get") else row[p_col]
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return default

    check_normality_and_variance = staticmethod(AssumptionCheckEngine.check_normality_and_variance)

    @staticmethod
    def perform_statistical_test(
    groups, transformed_samples, original_samples,
    dependent=False, test_recommendation="parametric", alpha=0.05, test_info=None,
    trace: "MethodologyTrace | None" = None
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

        samples_to_use = transformed_samples if test_recommendation in {"parametric", "welch"} else original_samples

        if len(valid_groups) == 2:
            return StatisticalTester._stat_test_two_groups(
                results, valid_groups, samples_to_use, original_samples, dependent, test_recommendation, alpha, test_info=test_info
            )

        return StatisticalTester._stat_test_multi_groups(
        results, valid_groups, samples_to_use, dependent, test_recommendation, alpha, test_info=test_info, trace=trace
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
                validate_paired_data(data1, data2, group_a_label=str(g1), group_b_label=str(g2), min_n=MIN_N_BLOCK)
            else:
                validate_minimum_n(data1, min_n=MIN_N_BLOCK, label=str(g1), allow_missing=False)
                validate_minimum_n(data2, min_n=MIN_N_BLOCK, label=str(g2), allow_missing=False)
        except ValidationError as validation_error:
            results["test"] = "Error during test"
            results["error"] = str(validation_error)
            results["posthoc_test"] = "Not performed (invalid paired input)"
            results["pairwise_comparisons"] = []
            return StatisticalTester._standardize_results(results)

        try:
            assumptions = extract_assumption_state(test_info)
            if test_info is None:
                fallback_is_normal = test_recommendation in {"parametric", "welch"}
                fallback_equal_variance = test_recommendation == "parametric"
            else:
                fallback_is_normal = assumptions.residuals_normal
                fallback_equal_variance = assumptions.equal_variance

            strategy = select_comparison_test(
                is_normal=fallback_is_normal,
                is_homoscedastic=fallback_equal_variance,
                is_paired=dependent,
                group_count=2,
            )

            supported_engine_strategies = {
                "paired_ttest",
                "wilcoxon",
                "student_ttest",
                "welch_ttest",
                "mann_whitney_u",
            }
            if strategy in supported_engine_strategies:
                engine_result = ComparisonEngine().execute(
                    {
                        "strategy": strategy,
                        "groups": [g1, g2],
                        "samples": {g1: data1, g2: data2},
                        "alpha": alpha,
                        "results": results,
                    }
                )
                return StatisticalTester.from_statistical_result(engine_result)

            results["test"] = "Error during test"
            results["error"] = f"Unsupported two-group strategy '{strategy}'"
            results["posthoc_test"] = "Not performed (unsupported strategy)"
            results["pairwise_comparisons"] = []
            return StatisticalTester._standardize_results(results)
        except Exception as e:
            results["test"] = "Error during test"
            results["error"] = str(e)
            results["posthoc_test"] = "Not performed (error in main test)"
            results["pairwise_comparisons"] = []
            return StatisticalTester._standardize_results(results)

    @staticmethod
    def _build_paired_subject_trajectories(group1_name, group2_name, data1, data2):
        trajectories = []
        if len(data1) != len(data2):
            return trajectories
        n = len(data1)
        for index in range(n):
            try:
                value1 = float(data1[index])
                value2 = float(data2[index])
                if np.isnan(value1) or np.isinf(value1) or np.isnan(value2) or np.isinf(value2):
                    continue
            except Exception:
                continue

            trajectories.append({
                "subject_id": f"S{index + 1}",
                "points": [
                    {"group": str(group1_name), "value": value1},
                    {"group": str(group2_name), "value": value2},
                ],
            })
        return trajectories

    @staticmethod
    def _build_subject_trajectories_from_long_df(df, dv, subject_col, label_columns, group_order=None):
        if df is None or subject_col is None or subject_col not in df.columns or dv not in df.columns:
            return []

        label_columns = [column for column in (label_columns or []) if column in df.columns]
        if not label_columns:
            return []

        group_rank = {str(group): idx for idx, group in enumerate(group_order or [])}
        subjects = {}

        for _, row in df.iterrows():
            subject_value = row.get(subject_col)
            if pd.isna(subject_value):
                continue
            try:
                numeric_value = float(row.get(dv))
                if np.isnan(numeric_value) or np.isinf(numeric_value):
                    continue
            except Exception:
                continue

            label_parts = [f"{column}={row.get(column)}" for column in label_columns]
            group_label = ", ".join(label_parts)
            if not group_label:
                continue

            subject_key = str(subject_value)
            if subject_key not in subjects:
                subjects[subject_key] = {}
            if group_label not in subjects[subject_key]:
                subjects[subject_key][group_label] = []
            subjects[subject_key][group_label].append(numeric_value)

        trajectories = []
        for subject_id, point_groups in subjects.items():
            points = []
            for group_label, values in point_groups.items():
                if not values:
                    continue
                points.append({
                    "group": group_label,
                    "value": float(np.mean(values)),
                })
            if len(points) < 2:
                continue
            points.sort(key=lambda point: (group_rank.get(point["group"], 10_000), point["group"]))
            trajectories.append({"subject_id": subject_id, "points": points})

        if len(trajectories) > 2000:
            trajectories = trajectories[:2000]
        return trajectories

    @staticmethod
    def _paired_ttest(results, g1, g2, data1, data2, alpha):
        data1_arr, data2_arr = validate_paired_data(
            data1,
            data2,
            group_a_label=str(g1),
            group_b_label=str(g2),
            min_n=MIN_N_BLOCK,
        )
        statistic, p_value = stats.ttest_rel(data1_arr, data2_arr)
        test_name = "Paired t-test"
        diff = data1_arr - data2_arr
        std_diff = np.std(diff, ddof=1)
        if std_diff == 0:
            cohen_d = None
            results["effect_size_type"] = "cohen_d (undefined — zero variance in differences)"
        else:
            cohen_d = np.mean(diff) / std_diff
            results["effect_size_type"] = "cohen_d"
        results["effect_size"] = cohen_d
        n = len(diff)
        stderr = std_diff / np.sqrt(n) if std_diff > 0 else 0
        t = stats.t
        ci = t.interval(0.95, n-1, loc=np.mean(diff), scale=stderr)
        results["confidence_interval"] = ci
        try:
            from statsmodels.stats.power import TTestPower
            effect_size = abs(cohen_d)
            power_analysis = TTestPower()
            results["power"] = float(power_analysis.power(effect_size=effect_size, nobs=n, alpha=alpha))
        except Exception:
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
        results["plot_subject_trajectories"] = StatisticalTester._build_paired_subject_trajectories(
            g1,
            g2,
            data1_arr.tolist(),
            data2_arr.tolist(),
        )
        return StatisticalTester._standardize_results(results)

    @staticmethod
    def _wilcoxon_test(results, g1, g2, data1, data2, alpha):
        data1_arr, data2_arr = validate_paired_data(
            data1,
            data2,
            group_a_label=str(g1),
            group_b_label=str(g2),
            min_n=MIN_N_BLOCK,
        )
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            statistic, p_value = stats.wilcoxon(
                data1_arr, data2_arr, 
                zero_method='pratt', 
                method='exact' if len(data1_arr) <= 25 else 'approx'
            )
            if w:
                for warn in w:
                    msg = f"Wilcoxon Warning: {str(warn.message)}"
                    if msg not in results.setdefault("warnings", []):
                        results["warnings"].append(msg)
        test_name = "Wilcoxon test"
        # scipy wilcoxon() returns min(T+, T-), not T+.
        # Correct rank-biserial: r = |1 - 2*min / N| where N = n_eff*(n_eff+1)/2
        diffs = data1_arr - data2_arr
        n_eff = int(np.sum(diffs != 0))
        _N = n_eff * (n_eff + 1) / 2
        r = float(abs(1.0 - 2.0 * statistic / _N)) if _N > 0 else 0.0
        results["effect_size"] = r
        results["effect_size_type"] = "r"
        results["confidence_interval"] = (None, None)
        try:
            from statsmodels.stats.power import TTestPower
            effect_size_corrected = r * 0.955
            power_analysis = TTestPower()
            results["power"] = float(power_analysis.power(effect_size=effect_size_corrected, nobs=n_eff, alpha=alpha))
        except Exception:
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
        results["plot_subject_trajectories"] = StatisticalTester._build_paired_subject_trajectories(
            g1,
            g2,
            data1_arr.tolist(),
            data2_arr.tolist(),
        )
        return StatisticalTester._standardize_results(results)
    
    @staticmethod
    def _independent_ttest(results, g1, g2, data1, data2, alpha, equal_var=True):
        data1_arr = validate_minimum_n(data1, min_n=MIN_N_BLOCK, label=str(g1), allow_missing=False)
        data2_arr = validate_minimum_n(data2, min_n=MIN_N_BLOCK, label=str(g2), allow_missing=False)
        n1, n2 = len(data1_arr), len(data2_arr)
        statistic, p_value = stats.ttest_ind(data1_arr, data2_arr, equal_var=equal_var)
        test_name = "t-test (independent)"
        if not equal_var:
            test_name = "Welch's t-test (unequal variances)"
        s1, s2 = np.var(data1_arr, ddof=1), np.var(data2_arr, ddof=1)

        # Different calculations for equal vs unequal variances
        if equal_var:
            # Pooled standard deviation for equal variances
            s_pooled = np.sqrt(((n1-1)*s1 + (n2-1)*s2) / (n1+n2-2))
            # CRITICAL-2: guard against zero pooled SD
            if s_pooled == 0:
                cohen_d = None
                results["effect_size_type"] = "cohen_d (undefined — zero pooled variance)"
            else:
                cohen_d = (np.mean(data1_arr) - np.mean(data2_arr)) / s_pooled
                results["effect_size_type"] = "cohen_d"
            stderr_diff = s_pooled * np.sqrt(1/n1 + 1/n2)
            df = n1 + n2 - 2
        else:
            # HIGH-1: Welch's uses Hedges' g formula (df-weighted pooled SD)
            s_pooled_hedges = np.sqrt(((n1-1)*s1 + (n2-1)*s2) / (n1+n2-2))
            if s_pooled_hedges == 0:
                cohen_d = None
                results["effect_size_type"] = "hedges_g (undefined — zero pooled variance)"
            else:
                cohen_d = (np.mean(data1_arr) - np.mean(data2_arr)) / s_pooled_hedges
                results["effect_size_type"] = "hedges_g"
            stderr_diff = np.sqrt(s1/n1 + s2/n2)
            # Welch-Satterthwaite degrees of freedom (safe: n>=2 guaranteed above)
            df_num = (s1/n1 + s2/n2)**2
            df_den = (s1/n1)**2/(n1-1) + (s2/n2)**2/(n2-1)
            df = df_num / df_den if df_den > 0 else min(n1, n2) - 1

        results["effect_size"] = cohen_d
        mean_diff = np.mean(data1_arr) - np.mean(data2_arr)
        t = stats.t
        ci = t.interval(0.95, df, loc=mean_diff, scale=stderr_diff)
        results["confidence_interval"] = ci
        try:
            if cohen_d is not None:
                from statsmodels.stats.power import TTestIndPower
                effect_size = abs(cohen_d)
                power_analysis = TTestIndPower()
                results["power"] = float(power_analysis.power(effect_size=effect_size, nobs1=n1, ratio=n2/n1, alpha=alpha))
            else:
                results["power"] = None
        except Exception:
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
        data1_arr = validate_minimum_n(data1, min_n=MIN_N_BLOCK, label=str(g1), allow_missing=False)
        data2_arr = validate_minimum_n(data2, min_n=MIN_N_BLOCK, label=str(g2), allow_missing=False)
        n1, n2 = len(data1_arr), len(data2_arr)
        from statistical_testing.validators import MIN_N_SMALL
        _mwu_method = 'exact' if (n1 + n2) < MIN_N_SMALL else 'asymptotic'
        statistic, p_value = stats.mannwhitneyu(data1_arr, data2_arr, alternative='two-sided', method=_mwu_method)
        test_name = f"Mann-Whitney-U ({'exact' if _mwu_method == 'exact' else 'asymptotic'})"
        u = statistic
        mean_u = n1 * n2 / 2
        std_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        z = (u - mean_u) / std_u if std_u > 0 else 0
        r = abs(z) / np.sqrt(n1 + n2)
        results["effect_size"] = r
        results["effect_size_type"] = "r"
        results["confidence_interval"] = (None, None)
        # No closed-form post-hoc power for the Wilcoxon/MWU rank test: the rank
        # effect size r = |Z|/sqrt(N) is not Cohen's d, so feeding it into a
        # t-test power routine yields a meaningless number. Report None and let
        # the rank-biserial r above stand as the effect-size summary.
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
    def _stat_test_multi_groups(results, valid_groups, samples_to_use, dependent, test_recommendation, alpha, test_info=None, df=None, dv=None, subject=None, within=None, trace: "MethodologyTrace | None" = None):
        try:
            validate_group_count(valid_groups, min_groups=3, label="multi_group_tests")
            if isinstance(samples_to_use, dict):
                for group in valid_groups:
                    validate_minimum_n(samples_to_use.get(group, []), min_n=MIN_N_BLOCK, label=str(group), allow_missing=False)
        except ValidationError as validation_error:
            results["test"] = "Error during test"
            results["error"] = str(validation_error)
            results["posthoc_test"] = "Not performed (invalid multi-group input)"
            results["pairwise_comparisons"] = []
            return StatisticalTester._standardize_results(results)

        assumptions = extract_assumption_state(test_info)
        if test_info is None:
            if dependent and len(valid_groups) > 2:
                strategy = "repeated_measures_required"
            elif test_recommendation == "welch":
                strategy = "welch_anova"
            elif test_recommendation == "parametric":
                strategy = "one_way_anova"
            else:
                strategy = "kruskal_wallis"
        else:
            strategy = select_comparison_test(
                is_normal=assumptions.residuals_normal,
                is_homoscedastic=assumptions.equal_variance,
                is_paired=dependent,
                group_count=len(valid_groups),
            )
        recommendation_family = strategy_to_recommendation(strategy)
        if test_info is not None:
            test_info.setdefault("decision", {}).update(
                {
                    "strategy": strategy,
                    "recommendation": recommendation_family,
                    "assumptions": {
                        "residuals_normal": assumptions.residuals_normal,
                        "equal_variance": assumptions.equal_variance,
                    },
                    "group_count": len(valid_groups),
                }
            )

        logger.debug(f"DEBUG DECISION: multigroup strategy={strategy}, recommendation={recommendation_family}")

        if strategy == "welch_anova":
            logger.debug("DEBUG DECISION: using Welch ANOVA path")
            welch_result = StatisticalTester._welch_anova_test(results, valid_groups, samples_to_use, alpha)
            if welch_result is not None:
                return StatisticalTester._standardize_results(welch_result)
            else:
                logger.debug("DEBUG DECISION: Welch ANOVA returned None, falling back to regular ANOVA")
                # Fall through to regular ANOVA
        
        try:
            if strategy == "repeated_measures_required":
                results["test"] = "Repeated Measures ANOVA not supported in simple pipeline"
                results["error"] = "Please use perform_advanced_test for RM-ANOVA."
                return StatisticalTester._standardize_results(results)
            else:
                engine_result = ComparisonEngine().execute(
                    {
                        "strategy": strategy,
                        "groups": valid_groups,
                        "samples": samples_to_use,
                        "alpha": alpha,
                        "results": results,
                    }
                )
                primary_result = StatisticalTester.from_statistical_result(engine_result)
                results.update(primary_result)

                if results.get("test") == "comparison_engine_failed" and results.get("error"):
                    return StatisticalTester._standardize_results(results)
            should_run_posthoc = (
                recommendation_family in {"parametric", "welch", "non_parametric"}
                and results.get("p_value") is not None
                and results["p_value"] < alpha
                and len(valid_groups) >= 3
                and not results.get("pairwise_comparisons")
            )
            if should_run_posthoc:
                posthoc_recommendation = "parametric" if recommendation_family in {"parametric", "welch"} else "non_parametric"
                logger.debug(
                    "DEBUG: Significant %s test, delegating post-hoc to PostHocEngine",
                    posthoc_recommendation,
                )
                if trace:
                    trace.add(
                        4,
                        "Post-hoc",
                        f"Main {posthoc_recommendation} test significant (p<{alpha}) with {len(valid_groups)} groups \u2014 "
                        f"{posthoc_recommendation} post-hoc engine invoked.",
                    )

                posthoc_result = PostHocEngine().execute(
                    {
                        "groups": valid_groups,
                        "samples": samples_to_use,
                        "test_recommendation": posthoc_recommendation,
                        "alpha": alpha,
                        "posthoc_choice": None,
                        "test_info": test_info,
                    }
                )
                posthoc_payload = StatisticalTester.from_statistical_result(posthoc_result)

                if posthoc_payload.get("pairwise_comparisons"):
                    results["posthoc_test"] = posthoc_payload.get("posthoc_test") or posthoc_payload.get("test")
                    results["pairwise_comparisons"] = posthoc_payload.get("pairwise_comparisons", [])
                    if trace and results.get("posthoc_test"):
                        trace.add(
                            4,
                            "Post-hoc",
                            f"{results['posthoc_test']} selected for pairwise comparisons.",
                            detail=f"{len(results['pairwise_comparisons'])} comparisons",
                        )
                else:
                    error_detail = posthoc_payload.get("error")
                    if posthoc_recommendation == "parametric":
                        results["posthoc_test"] = "No post-hoc tests performed"
                    else:
                        results["posthoc_test"] = "No post-hoc tests performed (error or unsupported test)"
                    results["pairwise_comparisons"] = []
                    if error_detail:
                        results["posthoc_error"] = error_detail

            # Explicit skip flag: fires when the main test was not significant and
            # post-hoc was intentionally omitted (groups >= 3, p_value known).
            if (
                not results.get("pairwise_comparisons")
                and isinstance(results.get("p_value"), (float, int))
                and results["p_value"] >= results.get("alpha", alpha)
                and len(valid_groups) >= 3
            ):
                _p_str = f"{results['p_value']:.4f}"
                results["posthoc_skipped"] = True
                results["posthoc_skip_reason"] = (
                    "Post-hoc not performed: main test was not significant "
                    f"(p\u202f=\u202f{_p_str})"
                )

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
        from analysis.posthoc_core import GamesHowellTest
        try:
            pg = get_pingouin_module()
            validate_group_count(valid_groups, min_groups=2, label="welch_anova_groups")
            
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
                
            logger.debug(f"DEBUG: Welch ANOVA - df_pg shape = {df_pg.shape}")
            logger.debug(f"DEBUG: Welch ANOVA - groups = {df_pg['group'].unique()}")
            
            # Perform Welch's ANOVA using pingouin
            welch_results = pg.welch_anova(data=df_pg, dv='value', between='group')
            
            # Extract results
            F_value = float(welch_results['F'].iloc[0])
            p_col = StatisticalTester._pingouin_p_column(welch_results.columns)
            p_value = float(welch_results[p_col].iloc[0]) if p_col else np.nan
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
            
            # Cohen's f (approx.) — no MS_within available for Welch, so use F*df/N
            n_total = len(df_pg)
            cohens_f = float(np.sqrt(max(F_value * df1 / n_total, 0.0)))
            results["effects"][0]["effect_size"] = cohens_f

            # Add test-level results
            results["p_value"] = p_value
            results["statistic"] = F_value
            results["effect_size"] = cohens_f
            results["effect_size_type"] = "Cohen's f"
            results["df1"] = df1
            results["df2"] = df2
            results["anova_table"] = welch_results
            
            # Post-hoc: Games-Howell if significant
            # Post-hoc selection is handled downstream (analysis_core) via the
            # user dialog — the main test no longer pre-selects Games-Howell, so
            # the user can choose Games-Howell / Dunnett / custom, or decline.
            # Games-Howell is supplied as the dialog DEFAULT, not forced here.
            results["pairwise_comparisons"] = []
            if p_value >= alpha:
                results["posthoc_test"] = "No post-hoc tests performed (not significant)"
            
            return results
        except Exception as e:
            import traceback
            logger.debug(f"DEBUG: Welch ANOVA failed with error: {str(e)}")
            logger.debug(f"DEBUG: Traceback: {traceback.format_exc()}")
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
                # HIGH-2: guard against degenerate df (n=1 in a group)
                if df_den == 0 or np.isnan(df_den) or np.isinf(df_den):
                    df = min(n1, n2) - 1
                    df_warning = GroupValidationError(
                        f"Dunnett T3 df calculation degenerate for {group1} vs {group2}; using conservative df={df}."
                    )
                    logger.warning(str(df_warning))
                else:
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
                    critical_value_warning = ValidationError(
                        f"Dunnett T3 critical-value calculation failed ({err}); using t-approximation fallback."
                    )
                    logger.warning(str(critical_value_warning))
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
            logger.error(f"ERROR in Dunnett's T3 post-hoc: {str(e)}")
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
        t = stats.t
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

        for group in groups:
            if group not in samples:
                validation["valid"] = False
                validation["messages"].append(f"Group '{group}' not found in the data.")

        if not validation["valid"]:
            return validation

        try:
            ensure_equal_group_sizes(samples, groups, min_n=1)
        except PairedDataError as exc:
            validation["valid"] = False
            validation["messages"].append(str(exc))
            validation["messages"].append(
                "For dependent tests, all groups must have the same number of measurements."
            )
        except ValidationError as exc:
            validation["valid"] = False
            validation["messages"].append(str(exc))

        for group in groups:
            try:
                validate_minimum_n(samples[group], min_n=3, label=str(group), allow_missing=False)
            except ValidationError as exc:
                validation["valid"] = False
                validation["messages"].append(str(exc))

        return validation
    
    @staticmethod
    def prepare_advanced_test(df, test, dv, subject, between=None, within=None):
        """
        Prepares an advanced statistical test by checking the assumptions.
        Returns test_info, recommendation, samples and groups.
        """
        recommendation = 'parametric'
        
        try:
            validate_test_design(test_name=test, between=between, within=within, subject=subject)
            samples = {}
            groups = []

            if test == 'mixed_anova':
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
                w_factor = within[0]
                for lvl in df[w_factor].unique():
                    samples[lvl] = df[df[w_factor] == lvl][dv].tolist()
                groups = list(samples.keys())
                # Formula for RM-ANOVA (for assumption checking) - use sanitized names
                sanitized_w_factor = w_factor.replace(' ', '') if ' ' in w_factor else w_factor
                formula = f"Value ~ C({sanitized_w_factor})"

            elif test == 'two_way_anova':
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
                raise ModelDesignError(f"Unknown test type: {test}")

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

        except ValidationError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": str(e)} 
        
    @staticmethod
    def perform_advanced_test(
        df, test, dv, subject, between=None, within=None, alpha=0.05,
        transformed_samples=None, recommendation=None, test_info=None,
        transform_fn=None, force_parametric=False, file_name=None, manual_transform=None,
        analysis_log=None  # Add this parameter
    ):
        return perform_advanced_test_pipeline(
            df=df,
            test=test,
            dv=dv,
            subject=subject,
            between=between,
            within=within,
            alpha=alpha,
            transformed_samples=transformed_samples,
            recommendation=recommendation,
            test_info=test_info,
            transform_fn=transform_fn,
            force_parametric=force_parametric,
            file_name=file_name,
            manual_transform=manual_transform,
            analysis_log=analysis_log,
        )

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
        results = {
            "test": "Mixed ANOVA",
            "model_type": "MixedANOVA",
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
        results["n_within_levels"] = len(df[rm_factor].unique())

        try:
            pg = get_pingouin_module()
            has_pingouin = True
        except ImportError:
            has_pingouin = False
            results["warning"] = "Pingouin not installed, using statsmodels"

        try:
            if has_pingouin:
                logger.debug("DEBUG: DataFrame columns: %s", df.columns)
                logger.debug("DEBUG: Unique values for within factor: %s", df[within[0]].unique())
                logger.debug("DEBUG: Unique values for subject: %s", df[subject].unique())
                logger.debug("DEBUG: First few rows of df:\n %s", df.head())
                logger.debug("DEBUG: Using Pingouin for Mixed ANOVA")
                aov = pg.mixed_anova(data=df, dv=dv, within=rm_factor, between=between_factor, subject=subject)
                p_col = "p_unc" if "p_unc" in aov.columns else "p-unc" if "p-unc" in aov.columns else None
                if p_col is None:
                    raise KeyError("No pingouin p-value column found in Mixed ANOVA table")
                results["anova_table"] = aov.copy()
                for factor in [rm_factor, between_factor]:
                    mask = aov["Source"] == factor
                    if mask.any():
                        row = aov.loc[mask].iloc[0]
                        results["factors"].append({
                            "factor": factor,
                            "type": "within" if factor == rm_factor else "between",
                            "F": float(row["F"]),
                            "p_value": float(row[p_col]),
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
                        "p_value": float(row[p_col]),
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
                    if not int_row.empty and float(int_row[p_col].iloc[0]) < alpha:
                        # Interaction is significant: t-tests for all combinations
                        ph = pg.pairwise_tests(
                            data=df,
                            dv=dv,
                            between=between_factor,
                            within=rm_factor,
                            subject=subject,
                            padjust="holm"  # Changed from "bon" to "holm"
                        )
                        results["posthoc_test"] = "Pairwise t-tests for interaction (Holm-Bonferroni)"  # Changed from "Bonferroni" to "Holm-Bonferroni"
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
                        if not between_row.empty and float(between_row[p_col].iloc[0]) < alpha:
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
                        if not within_row.empty and float(within_row[p_col].iloc[0]) < alpha:
                            # Paired t-tests for within-factor with Bonferroni
                            from itertools import combinations
                            within_groups = df[rm_factor].unique()
                            results["within_posthoc_test"] = "Paired t-tests (Holm-Bonferroni)"  # Changed from "Bonferroni" to "Holm-Bonferroni"
                            
                            # Perform paired t-tests and store p-values for Holm-Bonferroni correction
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

                            # Apply Holm-Bonferroni correction to all p-values at once
                            corrected_p_values = PostHocAnalyzer._holm_correction(p_values)

                            # Create comparison results using corrected p-values
                            for i, (group1, group2, data1, data2) in enumerate(data_pairs):
                                t_stat = t_stats[i]
                                p_val = p_values[i]  # Original p-value
                                corrected_p = corrected_p_values[i]  # Holm-Bonferroni corrected p-value
                                
                                # Calculate effect size (Cohen's d)
                                d = (np.mean(data1) - np.mean(data2)) / np.std(np.array(data1) - np.array(data2))
                                
                                results.setdefault("within_pairwise_comparisons", []).append({
                                    "group1": f"{rm_factor}={group1}",
                                    "group2": f"{rm_factor}={group2}",
                                    "test": "Paired t-test (Holm-Bonferroni)",  # Changed from "Bonferroni" to "Holm-Bonferroni"
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
                logger.debug("DEBUG: Using statsmodels for Mixed ANOVA")
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
                # C1-full: Type III SS with Sum contrasts using Q() to protect column names
                formula = f"Q('{sanitized_dv}') ~ C(Q('{sanitized_between}'), Sum) * C(Q('{sanitized_rm}'), Sum)"
                logger.debug(f"DEBUG: Mixed ANOVA formula with sanitized names: {formula}")
                
                model = ols(formula, data=sanitized_df).fit()
                anova = sm.stats.anova_lm(model, typ=3)

                # Effect sizes not available in fallback
                for factor, orig_factor in zip([sanitized_rm, sanitized_between], [rm_factor, between_factor]):
                    row = anova.loc[f"C(Q('{factor}'), Sum)"]
                    results["factors"].append({
                        "factor": orig_factor,  # Use original factor name in results
                        "type": "within" if orig_factor == rm_factor else "between",
                        "F": float(row["F"]),
                        "p_value": float(row["PR(>F)"]),
                        "df1": int(row["d"]),
                        "df2": int(anova.loc["Residual", "d"]),
                        "effect_size": None,
                        "effect_size_type": None
                    })

                interaction_key = f"C(Q('{sanitized_between}'), Sum):C(Q('{sanitized_rm}'), Sum)"
                row = anova.loc[interaction_key]
                interaction = {
                    "factors": [rm_factor, between_factor],
                    "F": float(row["F"]),
                    "p_value": float(row["PR(>F)"]),
                    "df1": int(row["d"]),
                    "df2": int(anova.loc["Residual", "d"]),
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
                            t = stats.t
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
        results = {
            "test": "Repeated Measures ANOVA",
            "model_type": "RepeatedMeasuresANOVA",
            "p_value": None,
            "statistic": None,
            "effect_size": None,
            "effect_size_type": None,
            "factors": [],
            "interactions": [],
            "descriptive": {},
            "error": None
        }
        if within and len(within) > 0 and within[0] in df.columns:
            factor = within[0]
            results["n_within_levels"] = len(df[factor].unique())
            
            # A2 & E2: detect incomplete subjects and emit a warning or redirect to LMM
            df_complete = df.dropna(subset=[factor, dv])
            expected_levels = set(df_complete[factor].unique())
            n_obs_per_subject = df_complete.groupby(subject)[factor].nunique()
            n_excluded = int((n_obs_per_subject < len(expected_levels)).sum())
            n_total = int(n_obs_per_subject.shape[0])
            
            if n_total > 0 and (n_excluded / n_total) > 0.05:
                # E2-LMM Redirect
                msg_redirect = (
                    "Methodological Note: RM-ANOVA was automatically redirected to a Linear Mixed Model (LMM) "
                    f"because > 5% of subjects ({n_excluded} out of {n_total}) had missing data. "
                    "LMMs handle unbalanced longitudinal data without listwise deletion."
                )
                msg_posthoc = "LMM redirect: pairwise contrasts not automatically computed. Interpret fixed effects table directly or re-run with complete data for post-hoc tests."
                
                logger.info(msg_redirect)
                # Trace: LMM-Redirect (2b)
                _lmm_trace = results.get("methodology_trace") or MethodologyTrace()
                _lmm_pct = f"{n_excluded / n_total * 100:.1f}%" if n_total > 0 else "N/A"
                _lmm_trace.add(3, "LMM Redirect",
                               f"RM-ANOVA redirected to LMM ({n_excluded}/{n_total} subjects incomplete, {_lmm_pct}).",
                               detail=("Repeated Measures ANOVA replaced by Linear Mixed Model (REML) because "
                                       f"{n_excluded} of {n_total} subjects ({_lmm_pct}) had incomplete data. "
                                       "LMMs handle unbalanced longitudinal data without listwise deletion "
                                       "and are valid under Missing At Random (MAR) assumptions. "
                                       "Pairwise contrasts not auto-computed — interpret fixed effects table."))
                results["methodology_trace"] = _lmm_trace
                try:
                    from analysis.clinical_models import LinearMixedModel
                    lmm = LinearMixedModel().fit(df, dv=dv, fixed_effects=within, random_intercept=subject)
                    lmm_results = lmm.as_results_dict()
                    # Ensure standard output fields
                    if "analysis_note" not in lmm_results:
                        lmm_results["analysis_note"] = msg_redirect
                    else:
                        lmm_results["analysis_note"] = msg_redirect + "\n\n" + lmm_results["analysis_note"]

                    lmm_results.setdefault("warnings", []).extend([msg_redirect, msg_posthoc])
                    lmm_results["methodology_trace"] = _lmm_trace
                    return StatisticalTester._standardize_results(lmm_results)
                except Exception as e:
                    warn_msg = f"RM-ANOVA LMM redirect failed ({e}). Falling back to listwise deletion."
                    results.setdefault("warnings", []).append(warn_msg)
                    logger.warning(warn_msg)
                    
            if n_excluded > 0 and (n_total == 0 or (n_excluded / n_total) <= 0.05):
                warn_msg = (
                    f"RM-ANOVA: {n_excluded} of {n_total} subjects excluded due to missing "
                    "observations (listwise deletion). Results reflect only complete cases. "
                    "Consider imputation if missingness is not random (MAR/MNAR)."
                )
                results.setdefault("warnings", []).append(warn_msg)
                logger.warning(warn_msg)
                # Trace: Listwise Deletion (2c)
                _ld_trace = results.get("methodology_trace") or MethodologyTrace()
                _ld_pct = f"{n_excluded / n_total * 100:.1f}%" if n_total > 0 else "N/A"
                _ld_trace.add(3, "Missing Data",
                              f"Listwise deletion: {n_excluded} of {n_total} subject(s) excluded ({_ld_pct}).",
                              detail=(f"{n_excluded} of {n_total} subjects ({_ld_pct}) excluded from RM-ANOVA "
                                      "due to missing data at ≥1 measurement occasion. Results reflect only "
                                      f"{n_total - n_excluded} subjects with complete data. Consider imputation "
                                      "if missingness is not completely at random (MCAR)."))
                results["methodology_trace"] = _ld_trace

        try:
            pg = get_pingouin_module()
            has_pingouin = True
        except ImportError:
            has_pingouin = False
            results["warning"] = "Pingouin not installed, using statsmodels"

        try:
            if has_pingouin:
                logger.debug("DEBUG: DataFrame columns: %s", df.columns)
                logger.debug("DEBUG: Unique values for within factor: %s", df[within[0]].unique())
                logger.debug("DEBUG: Unique values for subject: %s", df[subject].unique())
                logger.debug("DEBUG: First few rows of df:\n %s", df.head())
                logger.debug("DEBUG: Using Pingouin for RM ANOVA")    
                if len(within) == 1:
                    factor = within[0]
                    # Add correction=True to apply corrections for sphericity violation
                    aov = pg.rm_anova(data=df, dv=dv, within=factor, subject=subject, detailed=True, correction=True)
                    p_col = "p_unc" if "p_unc" in aov.columns else "p-unc" if "p-unc" in aov.columns else None
                    if p_col is None:
                        raise KeyError("No pingouin p-value column found in RM ANOVA table")
                    logger.debug("DEBUG: ANOVA result: %s", aov)
                    logger.debug("DEBUG: Results structure: %s", results)
                    results["anova_table"] = aov.copy()
                    row = aov.iloc[0]
                    error_row = aov[aov["Source"] == "Error"].iloc[0]
                    results["factors"].append({
                        "factor": factor,
                        "type": "within",
                        "F": float(row["F"]),
                        "p_value": float(row[p_col]),
                        "df1": int(row["DF"]),
                        "df2": int(error_row["DF"]),
                        "effect_size": float(row["ng2"]) if "ng2" in row else None,
                        "effect_size_type": "partial_eta_squared"
                    })
                    results.update({
                        "p_value": float(row[p_col]),
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
                    
                    # E1: write the correction-selected p-value back to the canonical field
                    if sphericity_results.get("final_p_value") is not None:
                        results["p_value"] = sphericity_results["final_p_value"]

                    # Trace: Sphericity + Epsilon-Korrektur (2a)
                    _trace = results.get("methodology_trace") or MethodologyTrace()
                    _spher_met = sphericity_results.get("sphericity_assumed", True)
                    _mauchly_p = sphericity_results.get("mauchly_p")
                    _epsilon = sphericity_results.get("epsilon")
                    _correction = sphericity_results.get("correction_applied", "none")
                    _mp_str = f"p = {_mauchly_p:.3f}" if isinstance(_mauchly_p, (float, int)) else "p = N/A"
                    if _spher_met:
                        _trace.add(3, "Sphericity",
                                   f"Mauchly's test: sphericity assumed ({_mp_str}). No correction applied.",
                                   detail=_mp_str)
                    else:
                        _corr_name = "Greenhouse-Geisser" if "GG" in str(_correction).upper() else "Huynh-Feldt"
                        _eps_str = f"ε = {_epsilon:.3f}" if isinstance(_epsilon, (float, int)) else "ε = N/A"
                        _trace.add(3, "Sphericity Correction",
                                   f"Mauchly's test violated ({_mp_str}) — {_corr_name} correction applied ({_eps_str}).",
                                   detail=f"{_mp_str}, {_eps_str}, correction = {_corr_name}")
                    results["methodology_trace"] = _trace

                    # Automatic post-hoc tests for significant main effect
                    if results["p_value"] is not None and results["p_value"] < alpha:
                        try:
                            # Extract data for post-hoc tests
                            factor_levels = df[factor].unique()
                            factor_data = {}
                            for level in factor_levels:
                                factor_data[level] = df[df[factor] == level][dv].tolist()
                            
                            # Perform paired t-tests with Holm-Bonferroni correction
                            posthoc_results = StatisticalTester.perform_dependent_posthoc_tests(
                                factor_data, list(factor_levels), alpha=alpha, parametric=True
                            )
                            logger.debug(f"DEBUG: Post-hoc for RM-ANOVA created with {len(posthoc_results.get('pairwise_comparisons', []))} comparisons")
                            results["posthoc_test"] = posthoc_results.get("posthoc_test", "Paired t-tests (Holm-Bonferroni)")

                            # Initialize with empty list as default
                            results["pairwise_comparisons"] = []

                            # If we got valid posthoc results, use them
                            if posthoc_results and 'pairwise_comparisons' in posthoc_results and posthoc_results['pairwise_comparisons']:
                                # Deep copy to ensure data isn't lost
                                import copy
                                results["pairwise_comparisons"] = copy.deepcopy(posthoc_results['pairwise_comparisons'])
                                logger.debug(f"DEBUG: Added {len(results['pairwise_comparisons'])} pairwise comparisons to RM-ANOVA results")
                        except Exception as ph_err:
                            results["warnings"] = results.get("warnings", []) + [f"Post-hoc failed: {ph_err}"]
                            logger.debug(f"DEBUG: Post-hoc test error: {str(ph_err)}")
                else:
                    aov = pg.rm_anova(data=df, dv=dv, within=within, subject=subject, detailed=True)
                    p_col = "p_unc" if "p_unc" in aov.columns else "p-unc" if "p-unc" in aov.columns else None
                    if p_col is None:
                        raise KeyError("No pingouin p-value column found in RM ANOVA table")
                    for _, row in aov.iterrows():
                        if "*" in row["Source"]:
                            results["interactions"].append({
                                "factors": row["Source"].split("*"),
                                "F": float(row["F"]),
                                "p_value": float(row[p_col]),
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
                                "p_value": float(row[p_col]),
                                "df1": int(row["DF1"]),
                                "df2": int(row["DF2"]),
                                "effect_size": float(row["np2"]),
                                "effect_size_type": "partial_eta_squared"
                            })

                    # Best result for main output
                    best_row = aov.iloc[aov["F"].argmax()]
                    results.update({
                        "p_value": float(best_row[p_col]),
                        "statistic": float(best_row["F"]),
                        "effect_size": float(best_row["np2"]),
                        "effect_size_type": "partial_eta_squared",
                        "df1": int(best_row["DF1"]),
                        "df2": int(best_row["DF2"]),
                        "test": "Repeated Measures ANOVA (multiple factors)"
                    })
            else:
                logger.debug("DEBUG: Using statsmodels for RM ANOVA")
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
                formula = f"Q('{sanitized_dv}') ~ C(Q('{sanitized_factor}'), Sum) + C(Q('{sanitized_subject}'), Sum)"
                logger.debug(f"DEBUG: RM ANOVA formula with sanitized names: {formula}")
                
                model = ols(formula, data=sanitized_df).fit()
                anova = sm.stats.anova_lm(model, typ=3)
                row = anova.loc[f"C(Q('{sanitized_factor}'), Sum)"]
                results["factors"].append({
                    "factor": factor,
                    "type": "within",
                    "F": float(row["F"]),
                    "p_value": float(row["PR(>F)"]),
                    "df1": int(row["d"]),
                    "df2": int(anova.loc['Residual', 'df']),
                    "effect_size": None,
                    "effect_size_type": None
                })
                results.update({
                    "p_value": float(row["PR(>F)"]),
                    "statistic": float(row["F"]),
                    "df1": int(row["d"]),
                    "df2": int(anova.loc['Residual', 'df']),
                    "test": f"Repeated Measures ANOVA ({factor}) [statsmodels]"
                })
        except Exception as e:
            results["error"] = str(e)
            logger.debug(f"DEBUG: Error in RM-ANOVA: {str(e)}")
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
                        t = stats.t
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
                logger.debug(f"DEBUG: Column name sanitized: '{col}' -> '{sanitized_name}'")
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
        results = {
            "test": f"Two-Way ANOVA ({between[0]} * {between[1]})",
            "model_type": "TwoWayANOVA",
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
            factor_a, factor_b = between[0], between[1]
            
            # D2: Check for empty cells or unbalanced design
            cell_counts = df.groupby([factor_a, factor_b]).size()
            min_cell_size = cell_counts.min() if len(cell_counts) > 0 else 0
            unique_a = df[factor_a].nunique()
            unique_b = df[factor_b].nunique()
            expected_cells = unique_a * unique_b
            actual_cells = len(cell_counts[cell_counts > 0])
            
            if actual_cells < expected_cells or min_cell_size == 0:
                msg = f"Warning (Two-Way ANOVA): The design has empty cells ({expected_cells - actual_cells} missing combinations). A Linear Mixed Model (LMM) is strongly recommended for incomplete designs."
                results.setdefault("warnings", []).append(msg)
            elif cell_counts.max() != cell_counts.min():
                msg = "Note: The design is unbalanced (unequal cell sizes). Type III Sum of Squares are used, but a Linear Mixed Model (LMM) may offer more robust estimates."
                results.setdefault("warnings", []).append(msg)

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
                        "p_value": StatisticalTester._pingouin_p_value(row),
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
                        "p_value": StatisticalTester._pingouin_p_value(row),
                        "df1": int(row["DF"]),
                        "df2": residual_df,
                        "effect_size": float(row["np2"]),
                        "effect_size_type": "partial_eta_squared"
                    }
                    results["interactions"].append(interaction_result)
                    results["p_value"] = StatisticalTester._pingouin_p_value(row)
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
                                pval_col = 'p_corr' if 'p_corr' in ph_row else ('p_unc' if 'p_unc' in ph_row else 'p-unc')
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
                                    "corrected": "Holm-Bonferroni",
                                    "confidence_interval": confidence_interval
                                })
                            
                            # Only set posthoc_test and add comparisons if we successfully processed all rows
                            if temp_comparisons:
                                results["posthoc_test"] = "Tukey HSD Test (Pingouin)"
                                results["pairwise_comparisons"].extend(temp_comparisons)
                        else:
                            results.setdefault("warnings", []).append("Pingouin pairwise_tests for interaction returned empty.")
                    except Exception as e_ph:
                        results.setdefault("warnings", []).append(f"Post-hoc tests (Pingouin) for interaction failed: {str(e_ph)}")

            else: # Fallback to statsmodels
                import statsmodels.api as sm
                from statsmodels.formula.api import ols
                logger.debug("DEBUG: WARNING! Two-Way ANOVA uses statsmodels fallback!")
                logger.debug("DEBUG: Pingouin not installed or import failed.")

                # Sanitize column names for statsmodels compatibility
                sanitized_df, column_mapping = StatisticalTester._sanitize_column_names_for_statsmodels(
                    df, [factor_a, factor_b, dv]
                )
                
                # Use sanitized column names in formula
                sanitized_factor_a = column_mapping[factor_a]
                sanitized_factor_b = column_mapping[factor_b] 
                sanitized_dv = column_mapping[dv]

                formula = f"Q('{sanitized_dv}') ~ C(Q('{sanitized_factor_a}'), Sum) * C(Q('{sanitized_factor_b}'), Sum)"
                logger.debug(f"DEBUG: Two-Way ANOVA formula with sanitized names: {formula}")
                
                model = ols(formula, data=sanitized_df).fit()
                aov = sm.stats.anova_lm(model, typ=3)
                if "Residual" not in aov.index:
                    results["error"] = "Residuals not found in statsmodels ANOVA output."
                    return StatisticalTester._standardize_results(results)
                residual_df = int(aov.loc["Residual", "d"])

                # Main effects
                for factor in [factor_a, factor_b]:
                    sanitized_factor = column_mapping[factor]
                    factor_term = f"C(Q('{sanitized_factor}'), Sum)"
                    if factor_term not in aov.index:
                        results.setdefault("warnings", []).append(f"Factor term '{factor_term}' not found in statsmodels ANOVA output.")
                        continue
                    row = aov.loc[factor_term]
                    results["factors"].append({
                        "factor": factor,
                        "type": "between",
                        "F": float(row["F"]),
                        "p_value": float(row["PR(>F)"]),
                        "df1": int(row["d"]),
                        "df2": residual_df,
                        "effect_size": None,
                        "effect_size_type": None
                    })

                # Interaction
                sanitized_factor_a = column_mapping[factor_a]
                sanitized_factor_b = column_mapping[factor_b]
                interaction_term = f"C(Q('{sanitized_factor_a}'), Sum):C(Q('{sanitized_factor_b}'), Sum)"
                if interaction_term in aov.index:
                    row = aov.loc[interaction_term]
                    interaction_result = {
                        "factors": [factor_a, factor_b],
                        "F": float(row["F"]),
                        "p_value": float(row["PR(>F)"]),
                        "df1": int(row["d"]),
                        "df2": residual_df,
                        "effect_size": None,
                        "effect_size_type": None
                    }
                    results["interactions"].append(interaction_result)
                    results["p_value"] = float(row["PR(>F)"])
                    results["statistic"] = float(row["F"])
                    results["df1"] = int(row["d"])
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
                                t = stats.t
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

        # D1: Interaction significance warning
        interaction_significant = False
        if results.get("interactions"):
            inter = results["interactions"][0]
            if inter.get("p_value") is not None and inter["p_value"] < alpha:
                interaction_significant = True
                
        if interaction_significant:
            msg = "Methodological Warning: The interaction term is significant. Therefore, the main effects cannot be interpreted independently. Focus on the interaction effect and pairwise comparisons between individual cells."
            if msg not in results.setdefault("warnings", []):
                results["warnings"].insert(0, msg) # Insert at beginning to emphasize methodological importance
            # Trace: Interaktionsterm-Warnung (2d)
            _inter = results["interactions"][0]
            _inter_p = _inter.get("p_value")
            _inter_factors = _inter.get("factors", between if between else ["A", "B"])
            _fa = _inter_factors[0] if len(_inter_factors) > 0 else "Factor A"
            _fb = _inter_factors[1] if len(_inter_factors) > 1 else "Factor B"
            _ip_str = f"p = {_inter_p:.4f}" if isinstance(_inter_p, (float, int)) else "p = N/A"
            _int_trace = results.get("methodology_trace") or MethodologyTrace()
            _int_trace.add(4, "Interaction Effect",
                           f"Significant interaction: {_fa} × {_fb} ({_ip_str}) — main effects not independently interpretable.",
                           detail=(f"The interaction term {_fa} × {_fb} was statistically significant ({_ip_str}). "
                                   f"Main effects of {_fa} and {_fb} cannot be interpreted in isolation. "
                                   "A simple effects analysis is required for valid interpretation."))
            results["methodology_trace"] = _int_trace

        return StatisticalTester._standardize_results(results)

    _prefix_pairwise_labels = staticmethod(PosthocFallbackEngine._prefix_pairwise_labels)
    _build_rm_aligned_samples = staticmethod(PosthocFallbackEngine._build_rm_aligned_samples)
    _apply_pairwise_multiplicity = staticmethod(PosthocFallbackEngine._apply_pairwise_multiplicity)
    _map_marginaleffects_to_exporter = staticmethod(PosthocFallbackEngine._map_marginaleffects_to_exporter)
    _run_two_way_marginaleffects_posthoc = staticmethod(PosthocFallbackEngine._run_two_way_marginaleffects_posthoc)
    _run_rm_marginaleffects_posthoc = staticmethod(PosthocFallbackEngine._run_rm_marginaleffects_posthoc)
    _run_mixed_marginaleffects_posthoc = staticmethod(PosthocFallbackEngine._run_mixed_marginaleffects_posthoc)
    _run_modern_fallback_posthoc = staticmethod(PosthocFallbackEngine._run_modern_fallback_posthoc)
    perform_dependent_posthoc_tests = staticmethod(PosthocFallbackEngine.perform_dependent_posthoc_tests)
    perform_refactored_posthoc_testing = staticmethod(PosthocFallbackEngine.perform_refactored_posthoc_testing)
    process_results = staticmethod(PosthocFallbackEngine.process_results)

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
                    "d": None,
                    "p_value": None,
                    "sphericity_assumed": True,
                    "note": "Sphericity assumption is always met with 2 levels",
                    "interpretation": "No correction needed - only 2 conditions compared"
                }
                results["corrected_p_value"] = StatisticalTester._pingouin_p_value(row)
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
                if hasattr(sphericity_result, 'spher'):
                    spher = sphericity_result.spher
                    W = sphericity_result.W
                    pval = sphericity_result.pval
                    dof = sphericity_result.dof
                    
                    mauchly_results = {
                        "test_name": "Mauchly's Test for Sphericity",
                        "W": float(W) if W is not None else None,
                        "p_value": float(pval) if pval is not None else None,
                        "sphericity_assumed": bool(spher) if spher is not None else None,
                        "d": int(dof) if dof is not None else (int((k * (k - 1)) / 2 - 1) if k > 2 else None),
                        "interpretation": StatisticalTester._interpret_sphericity_test(pval, spher) if pval is not None else "Test failed"
                    }
                    sphericity_violated = not bool(spher) if spher is not None else True
                elif isinstance(sphericity_result, tuple) and len(sphericity_result) >= 5:
                    spher, W, chi2, dof, pval = sphericity_result[:5]
                    mauchly_results = {
                        "test_name": "Mauchly's Test for Sphericity",
                        "W": float(W) if W is not None else None,
                        "p_value": float(pval) if pval is not None else None,
                        "sphericity_assumed": bool(spher) if spher is not None else None,
                        "d": int(dof) if dof is not None else (int((k * (k - 1)) / 2 - 1) if k > 2 else None),
                        "interpretation": StatisticalTester._interpret_sphericity_test(pval, spher) if pval is not None else "Test failed"
                    }
                    sphericity_violated = not bool(spher) if spher is not None else True
                else:
                    raise ValueError("Unexpected sphericity test output format")
                    
            except Exception:
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
            results["corrected_p_value"] = StatisticalTester._pingouin_p_value(row)
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
                    "d": int((k * (k - 1)) / 2 - 1) if k > 2 else None,
                    "note": "Extracted from ANOVA table",
                    "interpretation": "See p-value for significance"
                }
            else:
                return {
                    "test_name": "Mauchly's Test for Sphericity",
                    "W": None,
                    "p_value": None,
                    "sphericity_assumed": True,  # Conservative assumption
                    "d": int((k * (k - 1)) / 2 - 1) if k > 2 else None,
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
                corrections["corrected_p_value"] = StatisticalTester._pingouin_p_value(row)
                corrections["correction_used"] = "None (sphericity assumption met)"
                corrections["final_p_value"] = StatisticalTester._pingouin_p_value(row)
                return corrections
            
            # Sphericity violated - apply corrections
            corrections["sphericity_corrections"] = {"needed": True}
            
            # Greenhouse-Geisser Correction
            if 'p_GG_corr' in row and 'eps' in row:
                gg_epsilon = float(row["eps"])
                gg_p_value = float(row["p_GG_corr"])
                
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
                gg_p_value = StatisticalTester._pingouin_p_value(row)
            
            # Huynh-Feldt Correction  
            if 'p_HF_corr' in row and 'eps_HF' in row:
                hf_epsilon = float(row["eps_HF"])
                hf_p_value = float(row["p_HF_corr"])
                
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
                hf_p_value = StatisticalTester._pingouin_p_value(row)
            
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
                corrections["corrected_p_value"] = StatisticalTester._pingouin_p_value(row)
                corrections["correction_used"] = "None (corrections not available)"
                corrections["final_p_value"] = StatisticalTester._pingouin_p_value(row)
                corrections["recommendation"] = "Consider multivariate approach or non-parametric alternatives"
                
        except Exception as e:
            corrections["sphericity_corrections"] = {
                "needed": True,
                "error": f"Failed to apply corrections: {str(e)}"
            }
            corrections["corrected_p_value"] = StatisticalTester._pingouin_p_value(row)
            corrections["correction_used"] = "None (correction failed)"
            corrections["final_p_value"] = StatisticalTester._pingouin_p_value(row)
            
        return corrections

    _test_mixed_anova_between_assumptions = staticmethod(MixedAnovaAssumptionEngine._test_mixed_anova_between_assumptions)
    _perform_levene_test = staticmethod(MixedAnovaAssumptionEngine._perform_levene_test)
    _perform_brown_forsythe_test = staticmethod(MixedAnovaAssumptionEngine._perform_brown_forsythe_test)
    _perform_bartlett_test = staticmethod(MixedAnovaAssumptionEngine._perform_bartlett_test)
    _perform_welch_anova = staticmethod(MixedAnovaAssumptionEngine._perform_welch_anova)
    _generate_between_assumption_recommendations = staticmethod(MixedAnovaAssumptionEngine._generate_between_assumption_recommendations)
    _test_mixed_anova_within_sphericity = staticmethod(MixedAnovaAssumptionEngine._test_mixed_anova_within_sphericity)
    _extract_mixed_sphericity_from_anova_table = staticmethod(MixedAnovaAssumptionEngine._extract_mixed_sphericity_from_anova_table)
    _apply_mixed_anova_sphericity_corrections = staticmethod(MixedAnovaAssumptionEngine._apply_mixed_anova_sphericity_corrections)
    _apply_corrections_to_effect_row = staticmethod(MixedAnovaAssumptionEngine._apply_corrections_to_effect_row)
    _generate_within_factor_recommendations = staticmethod(MixedAnovaAssumptionEngine._generate_within_factor_recommendations)
    _test_mixed_anova_interaction_assumptions = staticmethod(MixedAnovaAssumptionEngine._test_mixed_anova_interaction_assumptions)
    _test_interaction_sphericity = staticmethod(MixedAnovaAssumptionEngine._test_interaction_sphericity)
    _test_interaction_cell_homogeneity = staticmethod(MixedAnovaAssumptionEngine._test_interaction_cell_homogeneity)
    _test_interaction_covariance_patterns = staticmethod(MixedAnovaAssumptionEngine._test_interaction_covariance_patterns)
    _perform_box_m_test = staticmethod(MixedAnovaAssumptionEngine._perform_box_m_test)
    _assess_compound_symmetry = staticmethod(MixedAnovaAssumptionEngine._assess_compound_symmetry)
    _generate_interaction_recommendations = staticmethod(MixedAnovaAssumptionEngine._generate_interaction_recommendations)
