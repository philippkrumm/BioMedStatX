from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    from core.methodology_trace import MethodologyTrace

from statistical_testing.decision_logic import select_comparison_test, strategy_to_recommendation
from statistical_testing.validators import (
    GroupValidationError,
    ValidationError,
    bounded_boxcox_lambda,
    validate_levene_inputs,
    validate_residuals_for_shapiro,
)
from analysis.stats_functions import UIDialogManager

logger = logging.getLogger(__name__)


def _get_ui_dialog_manager():
    """Resolve dialog manager through statisticaltester to honor test-time monkeypatches."""
    try:
        from analysis.statisticaltester import UIDialogManager as patched_dialog_manager
        return patched_dialog_manager
    except Exception:
        return UIDialogManager


class AssumptionCheckEngine:
    @staticmethod
    def check_normality_and_variance(
        groups, samples, dataset_name=None, progress_text=None, column_name=None, already_transformed=False,
        formula="Value ~ C(Group)", model_type="oneway", trace: "MethodologyTrace | None" = None
    ):
        """
        Checks normality and homogeneity of variance using model residuals (before and after transformation).
        
        Parameters:
        - model_type: str, one of "oneway", "twoway", "ttest", "rm" (repeated measures)
        - formula: str, formula for statsmodels OLS (e.g., "Value ~ C(Group)" for one-way)
        
        Always fits the specified model and tests residuals for normality using Shapiro-Wilk test.
        Levene test is performed on the raw values for variance homogeneity.
        """
        boxcox = stats.boxcox
        boxcox_normmax = stats.boxcox_normmax
        from statsmodels.formula.api import ols

        logger.debug("DEBUG check_normality_and_variance: Starting assumption tests")
        logger.debug(f"DEBUG check_normality_and_variance: model_type={model_type}, formula={formula}")
        logger.debug(f"DEBUG check_normality_and_variance: Groups: {groups}")
        ui_dialog_manager = _get_ui_dialog_manager()

        test_info: dict[str, Any] = {
            "pre_transformation": {},
            "post_transformation": {},
            "transformation": None,
            "validation_notes": []
        }
        def add_note(msg: str) -> None:
            notes = test_info.get("validation_notes")
            if isinstance(notes, list):
                notes.append(msg)

        valid_groups = [g for g in groups if g in samples and len(samples[g]) > 0]
        transformed_samples = {g: samples[g].copy() for g in valid_groups}
        test_recommendation = "parametric"

        # B1: For a paired t-test the relevant normality assumption is on the
        # within-pair differences d = x1 - x2, NOT on the pooled OLS residuals
        # of a Value ~ C(Group) model (which encodes the *unpaired* assumption).
        is_paired = model_type == "paired"

        def _paired_diffs(samps):
            if len(valid_groups) != 2:
                return None
            g0, g1 = valid_groups[0], valid_groups[1]
            a, b = list(samps[g0]), list(samps[g1])
            if len(a) != len(b) or len(a) < 3:
                return None
            return np.asarray(a, dtype=float) - np.asarray(b, dtype=float)

        def _rm_diffs(samps):
            if len(valid_groups) < 2:
                return None
            baseline = valid_groups[0]
            base_vals = np.asarray(list(samps[baseline]), dtype=float)
            diffs = []
            for g in valid_groups[1:]:
                vals = np.asarray(list(samps[g]), dtype=float)
                if len(vals) != len(base_vals) or len(vals) < 3:
                    return None
                diffs.extend(vals - base_vals)
            return np.asarray(diffs)


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
                data_rows = []
                for g in valid_groups:
                    for i, v in enumerate(samps[g]):
                        row = {"Group": g, "Value": v}
                        if model_type in ("rm", "paired"):
                            row["Subject"] = f"S{i}"
                        data_rows.append(row)
                return pd.DataFrame(data_rows)
        df_raw = make_df(samples)
        
        # Adjust formula for Two-Way ANOVA if needed
        adjusted_formula = formula
        if model_type == "twoway" and "FactorA" in df_raw.columns and "FactorB" in df_raw.columns:
            adjusted_formula = "Value ~ C(FactorA) * C(FactorB)"
            logger.debug(f"DEBUG SHAPIRO: Adjusted formula for Two-Way ANOVA: {adjusted_formula}")
        
        try:
            actual_columns = list(df_raw.columns)
            
            # ROBUST FORMULA CREATION: Use actual DataFrame columns, not assumptions
            logger.debug(f"DEBUG SHAPIRO: Available columns in df_raw: {actual_columns}")
            logger.debug(f"DEBUG SHAPIRO: Original formula: {formula}")
            logger.debug(f"DEBUG SHAPIRO: Adjusted formula: {adjusted_formula}")
            
            if "Value" not in actual_columns:
                logger.debug("DEBUG SHAPIRO ERROR: 'Value' column missing in DataFrame!")
                stat, pval = None, None
            elif is_paired:
                # B1: paired t-test → Shapiro-Wilk on within-pair differences.
                diffs = _paired_diffs(samples)
                if diffs is not None:
                    try:
                        validate_residuals_for_shapiro(diffs, label="pre_transformation_paired_diffs")
                        stat, pval = stats.shapiro(diffs)
                    except ValidationError as exc:
                        stat, pval = None, None
                        add_note(str(exc))
                        logger.warning(str(exc))
                else:
                    stat, pval = None, None
                    add_note(
                        "Paired normality check skipped: groups not equal length or n<3."
                    )
            else:
                # Find factor columns (everything except 'Value' and 'Subject')
                factor_columns = [col for col in actual_columns if col not in ("Value", "Subject")]
                logger.debug(f"DEBUG SHAPIRO: Found factor columns: {factor_columns}")
                
                if not factor_columns:
                    # No factors, use intercept-only model
                    working_formula = "Value ~ 1"
                    logger.debug(f"DEBUG SHAPIRO: Using intercept-only model: {working_formula}")
                elif len(factor_columns) == 1:
                    # Single factor
                    if model_type == "rm" and "Subject" in actual_columns:
                        working_formula = f"Value ~ C({factor_columns[0]}) + C(Subject)"
                    else:
                        working_formula = f"Value ~ C({factor_columns[0]})"
                    logger.debug(f"DEBUG SHAPIRO: Using single factor model: {working_formula}")
                elif len(factor_columns) == 2 and model_type == "twoway":
                    # Two factors for two-way ANOVA
                    working_formula = f"Value ~ C({factor_columns[0]}) * C({factor_columns[1]})"
                    logger.debug(f"DEBUG SHAPIRO: Using two-factor model: {working_formula}")
                else:
                    # Multiple factors, use first one only to avoid complexity
                    working_formula = f"Value ~ C({factor_columns[0]})"
                    logger.debug(f"DEBUG SHAPIRO: Using first factor only: {working_formula}")
                
                # Try to fit the model with actual column names
                try:
                    model_raw = ols(working_formula, data=df_raw).fit()
                    resid_raw = model_raw.resid
                    logger.debug(f"DEBUG SHAPIRO: Raw residuals length: {len(resid_raw)}, unique values: {len(set(resid_raw))}")
                    # HIGH-3: centralized validation for Shapiro-Wilk applicability
                    try:
                        validate_residuals_for_shapiro(resid_raw, label="pre_transformation_residuals")
                        stat, pval = stats.shapiro(resid_raw)
                    except ValidationError as exc:
                        stat, pval = None, None
                        add_note(str(exc))
                        logger.warning(str(exc))
                    logger.debug(f"DEBUG SHAPIRO: Pre-transformation Shapiro-Wilk: W={stat}, p={pval}")
                except Exception as e:
                    if model_type == "rm" and "Subject" in working_formula:
                        logger.warning(f"Failed to fit RM model with Subject effect: {e}. Falling back to baseline diffs.")
                        diffs = _rm_diffs(samples)
                        if diffs is not None:
                            try:
                                validate_residuals_for_shapiro(diffs, label="pre_transformation_rm_diffs")
                                stat, pval = stats.shapiro(diffs)
                            except ValidationError as exc:
                                stat, pval = None, None
                                add_note(str(exc))
                                logger.warning(str(exc))
                        else:
                            stat, pval = None, None
                            add_note("RM normality check failed (fallback could not compute differences).")
                    else:
                        raise e
                
        except Exception as e:
            logger.debug(f"DEBUG SHAPIRO ERROR: Failed pre-transformation Shapiro-Wilk test: {str(e)}")
            logger.debug(f"DEBUG SHAPIRO ERROR: DataFrame info - Shape: {df_raw.shape}, Columns: {list(df_raw.columns)}")
            logger.debug(f"DEBUG SHAPIRO ERROR: DataFrame head:\n{df_raw.head()}")
            stat, pval = None, None
            
        _pre_is_normal = (pval > 0.05 if pval is not None else False)

        test_info["pre_transformation"]["residuals_normality"] = {
            "statistic": stat, "p_value": pval, "is_normal": _pre_is_normal
        }
        
        _norm_target = "within-pair differences" if is_paired else "model residuals"
        if trace:
            _norm_verdict = "normality assumed" if _pre_is_normal else "normality violated"
            _p_str = f"p={pval:.4f}" if isinstance(pval, (float, int)) else "p=N/A"
            _w_str = f"W={stat:.4f}" if isinstance(stat, (float, int)) else "W=N/A"
            trace.add(1, "Normality",
                        f"Shapiro-Wilk on {_norm_target} yielded {_p_str} \u2014 {_norm_verdict}.",
                        detail=f"{_w_str}, {_p_str}")

        # Levene test on raw data (Brown-Forsythe test using median)
        if model_type in ("rm", "mixed", "paired"):
            stat, pval, has_equal_variance = None, None, True
            if is_paired:
                test_name = "N/A (Paired)"
                add_note(
                    "Levene's test bypassed: variance homogeneity is not an assumption of the paired t-test (it operates on within-pair differences)."
                )
            else:
                test_name = "N/A (Repeated Measures / Mixed)"
                add_note(
                    "Levene's test bypassed: sphericity check (Mauchly's test) is the appropriate variance check for repeated measures/mixed designs."
                )
        else:
            test_name = "Brown-Forsythe"
            try:
                data_for_levene = [[v for v in samples[g] if not (isinstance(v, float) and np.isnan(v))] for g in valid_groups]
                logger.debug(f"DEBUG BROWN-FORSYTHE: Pre-transformation - Groups: {len(valid_groups)}, Data lengths: {[len(v) for v in data_for_levene]}")
                try:
                    validated_levene_data = validate_levene_inputs(
                        data_for_levene,
                        min_groups=2,
                        min_n_per_group=3,
                        label="pre_transformation_levene",
                    )
                    stat, pval = stats.levene(*validated_levene_data, center='median')
                    has_equal_variance = pval > 0.05
                    logger.debug(f"DEBUG BROWN-FORSYTHE: Pre-transformation - Statistic: {stat}, p-value: {pval}, Equal variance: {has_equal_variance}")
                except ValidationError as exc:
                    stat, pval, has_equal_variance = None, None, False
                    add_note(str(exc))
                    logger.warning(str(exc))
                    logger.debug("DEBUG BROWN-FORSYTHE: Pre-transformation - Insufficient data for test")
            except Exception as e:
                logger.debug(f"DEBUG BROWN-FORSYTHE ERROR: Pre-transformation failed: {str(e)}")
                stat, pval, has_equal_variance = None, None, False
        
        test_info["pre_transformation"]["variance"] = {
            "statistic": stat, "p_value": pval, "equal_variance": has_equal_variance,
            "test_name": test_name,
        }
        if trace:
            if model_type in ("rm", "mixed", "paired"):
                _bypass_reason = ("variance homogeneity is not an assumption of the paired t-test"
                                  if is_paired else
                                  "Sphericity is the relevant variance check")
                trace.add(2, "Assumption",
                          f"Homogeneity of variance (Levene's test) bypassed for dependent design; {_bypass_reason}.")
            else:
                _var_verdict = "equal variances" if has_equal_variance else "unequal variances"
                _vp_str = f"p={pval:.4f}" if isinstance(pval, (float, int)) else "p=N/A"
                trace.add(2, "Assumption",
                          f"Brown-Forsythe test yielded {_vp_str} \u2014 {_var_verdict}.",
                          detail=f"F={stat:.4f}, {_vp_str}" if isinstance(stat, (float, int)) else "")

        # Welch-ANOVA (and RM corrections) handles variance heteroscedasticity.
        # Transformation is strictly for correcting non-normality.
        need_transform = not test_info["pre_transformation"]["residuals_normality"]["is_normal"]

        # Transformation if needed
        if need_transform:
            if already_transformed:
                test_info["transformation"] = "No further"
                return transformed_samples, "non_parametric", test_info

            transformation_type = None
            try:
                transformation_type = ui_dialog_manager.select_transformation_dialog(
                    parent=None, progress_text=progress_text, column_name=column_name
                )
            except Exception:
                transformation_type = "log10"
            if not transformation_type:
                transformation_type = "log10"
            test_info["transformation"] = transformation_type

            # Calculate a uniform global shift across the entire raw dependent variable column vector in df_raw
            global_min = df_raw["Value"].min() if (not df_raw.empty and "Value" in df_raw.columns) else 0
            global_shift = -global_min + 1 if global_min <= 0 else 0

            if global_shift > 0 and transformation_type in ("log10", "boxcox") and trace:
                trace.add(2, "Transformation Validation", 
                          f"Data contained values \u2264 0. A constant shift of {global_shift} was added before {transformation_type} transformation to ensure strict positivity.")

            # MEDIUM-3: record very large shifts via validation notes
            if global_shift > 1e6:
                shift_warning = GroupValidationError(
                    f"Log10/Box-Cox transformation requires large global shift ({global_shift:.2e}); "
                    "consider preprocessing."
                )
                add_note(str(shift_warning))
                logger.warning(str(shift_warning))

            # BoxCox: estimate one global lambda from all valid values across groups
            _boxcox_lambda = None
            if transformation_type == "boxcox":
                _all_bc = []
                for _g in valid_groups:
                    for _v in samples[_g]:
                        try:
                            _fv = float(_v)
                        except (TypeError, ValueError):
                            continue
                        if not (np.isnan(_fv) or np.isinf(_fv)):
                            _all_bc.append(_fv + global_shift)
                if len(_all_bc) >= 3 and min(_all_bc) > 0:
                    # Guard against optimizer divergence on extremely skewed assay
                    # data: reject an out-of-bounds lambda and fall back to log
                    # (lambda=0) rather than potentiating the variance.
                    _boxcox_lambda, _reverted = bounded_boxcox_lambda(np.array(_all_bc))
                    test_info["boxcox_lambda"] = _boxcox_lambda
                    if _reverted:
                        _warn = (
                            "Maximum-likelihood estimation of the Box-Cox parameter "
                            "lambda diverged (out of bounds). Fell back to a log "
                            "transformation (lambda = 0)."
                        )
                        add_note(_warn)
                        test_info["transform_warning"] = _warn
                else:
                    add_note("Box-Cox: insufficient valid data globally; falling back to log10.")
                    transformation_type = "log10"

            # Apply transformation
            for group in valid_groups:
                values = samples[group]
                min_val = min(values)
                
                if transformation_type == "log10":
                    transformed_samples[group] = [np.log10(v + global_shift) for v in values]
                    if global_shift > 0:
                        if "log10_shifts" not in test_info or not isinstance(test_info["log10_shifts"], dict):
                            test_info["log10_shifts"] = {}
                        test_info["log10_shifts"][group] = global_shift
                elif transformation_type == "boxcox":
                    transformed = []
                    for v in values:
                        try:
                            fv = float(v)
                        except (TypeError, ValueError):
                            transformed.append(v)
                            continue
                        if np.isnan(fv) or np.isinf(fv):
                            transformed.append(fv)
                        else:
                            transformed.append(float(boxcox(fv + global_shift, _boxcox_lambda)))
                    transformed_samples[group] = transformed
                elif transformation_type == "arcsin_sqrt":
                    max_val = max(values)
                    # Scale to 0-1 if needed
                    if min_val < 0 or max_val > 1:
                        if trace and group == valid_groups[0]:
                            trace.add(2, "Transformation Validation", 
                                      "Data contained values outside [0, 1]. Data was min-max scaled to [0, 1] before arcsin-sqrt transformation.")
                        # CRITICAL-4: guard against zero variance (min == max)
                        if max_val == min_val:
                            variance_warning = GroupValidationError(
                                f"Group '{group}': arcsin-sqrt transformation received zero variance data; using 0.5 fallback."
                            )
                            add_note(str(variance_warning))
                            logger.warning(str(variance_warning))
                            scaled = [0.5] * len(values)
                        else:
                            scaled = [(v - min_val) / (max_val - min_val) for v in values]
                    else:
                        # Values already in [0,1] — still guard against zero variance
                        if len(set(values)) == 1:
                            variance_warning = GroupValidationError(
                                f"Group '{group}': arcsin-sqrt transformation received zero variance data; using 0.5 fallback."
                            )
                            add_note(str(variance_warning))
                            logger.warning(str(variance_warning))
                            scaled = [0.5] * len(values)
                        else:
                            scaled = values
                    transformed_samples[group] = [np.arcsin(np.sqrt(v)) for v in scaled]

        # Fit model and check residuals normality on transformed data
        df_tr = make_df(transformed_samples)
        
        # Use the same robust formula approach for transformed data
        try:
            # ROBUST FORMULA FOR TRANSFORMED DATA: Use actual DataFrame columns
            logger.debug(f"DEBUG SHAPIRO: Available columns in df_tr: {list(df_tr.columns)}")
            
            # Create formula based on ACTUAL columns present in transformed DataFrame
            actual_columns_tr = list(df_tr.columns)
            
            if "Value" not in actual_columns_tr:
                logger.debug("DEBUG SHAPIRO ERROR: 'Value' column missing in transformed DataFrame!")
                stat2, pval2 = None, None
            elif is_paired:
                # B1: paired t-test → Shapiro-Wilk on transformed within-pair differences.
                diffs_tr = _paired_diffs(transformed_samples)
                if diffs_tr is not None:
                    try:
                        validate_residuals_for_shapiro(diffs_tr, label="post_transformation_paired_diffs")
                        stat2, pval2 = stats.shapiro(diffs_tr)
                    except ValidationError as exc:
                        stat2, pval2 = None, None
                        add_note(str(exc))
                        logger.warning(str(exc))
                else:
                    stat2, pval2 = None, None
            else:
                # Find factor columns (everything except 'Value' and 'Subject')
                factor_columns_tr = [col for col in actual_columns_tr if col not in ("Value", "Subject")]
                logger.debug(f"DEBUG SHAPIRO: Found factor columns in transformed data: {factor_columns_tr}")
                
                if not factor_columns_tr:
                    # No factors, use intercept-only model
                    working_formula_tr = "Value ~ 1"
                    logger.debug(f"DEBUG SHAPIRO: Using intercept-only model for transformed: {working_formula_tr}")
                elif len(factor_columns_tr) == 1:
                    # Single factor
                    if model_type == "rm" and "Subject" in actual_columns_tr:
                        working_formula_tr = f"Value ~ C({factor_columns_tr[0]}) + C(Subject)"
                    else:
                        working_formula_tr = f"Value ~ C({factor_columns_tr[0]})"
                    logger.debug(f"DEBUG SHAPIRO: Using single factor model for transformed: {working_formula_tr}")
                elif len(factor_columns_tr) == 2 and model_type == "twoway":
                    # Two factors for two-way ANOVA
                    working_formula_tr = f"Value ~ C({factor_columns_tr[0]}) * C({factor_columns_tr[1]})"
                    logger.debug(f"DEBUG SHAPIRO: Using two-factor model for transformed: {working_formula_tr}")
                else:
                    # Multiple factors, use first one only to avoid complexity
                    working_formula_tr = f"Value ~ C({factor_columns_tr[0]})"
                    logger.debug(f"DEBUG SHAPIRO: Using first factor only for transformed: {working_formula_tr}")
                
                # Try to fit the model with actual column names
                try:
                    model_tr = ols(working_formula_tr, data=df_tr).fit()
                    resid_tr = model_tr.resid
                    logger.debug(f"DEBUG SHAPIRO: Transformed residuals length: {len(resid_tr)}, unique values: {len(set(resid_tr))}")
                    try:
                        validate_residuals_for_shapiro(resid_tr, label="post_transformation_residuals")
                        stat2, pval2 = stats.shapiro(resid_tr)
                    except ValidationError as exc:
                        stat2, pval2 = None, None
                        add_note(str(exc))
                        logger.warning(str(exc))
                    logger.debug(f"DEBUG SHAPIRO: Post-transformation Shapiro-Wilk: W={stat2}, p={pval2}")
                except Exception as e:
                    if model_type == "rm" and "Subject" in working_formula_tr:
                        logger.warning(f"Failed to fit RM model with Subject effect on transformed data: {e}. Falling back to baseline diffs.")
                        diffs_tr = _rm_diffs(transformed_samples)
                        if diffs_tr is not None:
                            try:
                                validate_residuals_for_shapiro(diffs_tr, label="post_transformation_rm_diffs")
                                stat2, pval2 = stats.shapiro(diffs_tr)
                            except ValidationError as exc:
                                stat2, pval2 = None, None
                                add_note(str(exc))
                                logger.warning(str(exc))
                        else:
                            stat2, pval2 = None, None
                            add_note("RM normality check failed (fallback could not compute differences).")
                    else:
                        raise e
                
        except Exception as e:
            logger.debug(f"DEBUG SHAPIRO ERROR: Failed post-transformation Shapiro-Wilk test: {str(e)}")
            logger.debug(f"DEBUG SHAPIRO ERROR: Transformed DataFrame info - Shape: {df_tr.shape}, Columns: {list(df_tr.columns)}")
            logger.debug(f"DEBUG SHAPIRO ERROR: Transformed DataFrame head:\n{df_tr.head()}")
            stat2, pval2 = None, None
        test_info["post_transformation"]["residuals_normality"] = {
            "statistic": stat2, "p_value": pval2, "is_normal": (pval2 > 0.05 if pval2 is not None else False)
        }

        # Levene test on transformed data
        if model_type in ("rm", "mixed", "paired"):
            stat_tr, pval_tr, has_equal_variance_tr = None, None, True
            test_name_tr = "N/A (Paired)" if is_paired else "N/A (Repeated Measures / Mixed)"
        else:
            test_name_tr = "Brown-Forsythe"
            try:
                data_for_levene_tr = [[v for v in transformed_samples[g] if not (isinstance(v, float) and np.isnan(v))] for g in valid_groups]
                logger.debug(f"DEBUG BROWN-FORSYTHE: Post-transformation - Groups: {len(valid_groups)}, Data lengths: {[len(v) for v in data_for_levene_tr]}")
                try:
                    validated_levene_data_tr = validate_levene_inputs(
                        data_for_levene_tr,
                        min_groups=2,
                        min_n_per_group=3,
                        label="post_transformation_levene",
                    )
                    stat_tr, pval_tr = stats.levene(*validated_levene_data_tr, center='median')
                    has_equal_variance_tr = pval_tr > 0.05
                    logger.debug(f"DEBUG BROWN-FORSYTHE: Post-transformation - Statistic: {stat_tr}, p-value: {pval_tr}, Equal variance: {has_equal_variance_tr}")
                except ValidationError as exc:
                    stat_tr, pval_tr, has_equal_variance_tr = None, None, False
                    add_note(str(exc))
                    logger.warning(str(exc))
                    logger.debug("DEBUG BROWN-FORSYTHE: Post-transformation - Insufficient data for test")
            except Exception as e:
                logger.debug(f"DEBUG BROWN-FORSYTHE ERROR: Post-transformation failed: {str(e)}")
                stat_tr, pval_tr, has_equal_variance_tr = None, None, False
        test_info["post_transformation"]["variance"] = {
            "statistic": stat_tr, "p_value": pval_tr, "equal_variance": has_equal_variance_tr,
            "test_name": test_name_tr,
        }

        # Recommend test based on assumptions
        post_norm = test_info["post_transformation"]["residuals_normality"]["is_normal"]
        post_var = test_info["post_transformation"]["variance"]["equal_variance"]

        if need_transform and trace:
            _p_str2 = f"p={pval2:.4f}" if isinstance(pval2, (float, int)) else "p=N/A"
            _w_str2 = f"W={stat2:.4f}" if isinstance(stat2, (float, int)) else "W=N/A"
            _norm_verdict2 = "normality assumption met" if post_norm else "normality assumption violated"
            trace.add(2, "Normality (Post-Transformation)",
                      f"Normality reassessed after {test_info.get('transformation')} transformation: Shapiro-Wilk on {_norm_target} yielded {_p_str2} \u2014 {_norm_verdict2}.",
                      detail=f"{_w_str2}, {_p_str2}")

        if model_type in ["twoway", "mixed", "rm"] and post_norm:
            decision_strategy = f"{model_type}_anova"
            test_recommendation = "parametric"
            if not post_var:
                test_info["note"] = (
                    f"Residuals are normal but variances are unequal - {model_type.upper()} ANOVA will still be used "
                    "(robust to variance heterogeneity)."
                )
        else:
            decision_strategy = select_comparison_test(
                is_normal=post_norm,
                is_homoscedastic=post_var,
                is_paired=False,
                group_count=len(valid_groups),
            )
            test_recommendation = strategy_to_recommendation(decision_strategy)
            if decision_strategy == "welch_ttest":
                if post_var:
                    test_info["note"] = "Residuals are normal and variances are equal - Welch's t-test will be used (robust default)."
                else:
                    test_info["note"] = "Residuals are normal but variances are unequal - Welch's t-test will be used."
            elif decision_strategy == "welch_anova":
                if post_var:
                    test_info["note"] = "Residuals are normal and variances are equal - Welch's ANOVA will be used (robust default)."
                else:
                    test_info["note"] = "Residuals are normal but variances are unequal - Welch's ANOVA will be used."

        test_info["decision"] = {
            "strategy": decision_strategy,
            "recommendation": test_recommendation,
            "assumptions": {
                "residuals_normal": post_norm,
                "equal_variance": post_var,
            },
            "group_count": len(valid_groups),
            "model_type": model_type,
        }

        if trace:
            if test_recommendation == "non_parametric":
                trace.add(3, "Test Selection", "Normality assumption violated \u2014 non-parametric test selected.")
            elif decision_strategy in {"welch_ttest", "welch_anova"}:
                trace.add(3, "Test Selection", "Normality confirmed, variance inequality detected \u2014 Welch correction selected.")
            else:
                trace.add(3, "Test Selection", "Assumptions support parametric testing \u2014 standard parametric route selected.")

        logger.debug("DEBUG check_normality_and_variance: Final test_info structure:")
        logger.debug(f"  Pre-transformation normality: {test_info['pre_transformation'].get('residuals_normality', 'Missing')}")
        logger.debug(f"  Pre-transformation variance: {test_info['pre_transformation'].get('variance', 'Missing')}")
        logger.debug(f"  Post-transformation normality: {test_info['post_transformation'].get('residuals_normality', 'Missing')}")
        logger.debug(f"  Post-transformation variance: {test_info['post_transformation'].get('variance', 'Missing')}")
        logger.debug(f"  Decision strategy: {decision_strategy}; recommendation: {test_recommendation}")

        return transformed_samples, test_recommendation, test_info
    
