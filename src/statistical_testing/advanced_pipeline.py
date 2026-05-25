import logging

from analysis.nonparametricanovas import (
    perform_brunner_langer_ats,
    perform_freedman_lane_test,
    perform_friedman_test,
)

from .engines.advanced_posthoc import AdvancedPostHocEngine
from .engines.assumption_bridge import AssumptionBridgeEngine
from .engines.extraction import ExtractionEngine
from .engines.finalization import FinalizationEngine
from .engines.recommendation import RecommendationEngine
from .engines.reporting import ReportingEngine
from .engines.transformation import TransformationEngine
from .validators import ValidationError, validate_minimum_n, validate_test_design


logger = logging.getLogger(__name__)


def perform_advanced_test_pipeline(
    df,
    test,
    dv,
    subject,
    between=None,
    within=None,
    alpha=0.05,
    transformed_samples=None,
    recommendation=None,
    test_info=None,
    transform_fn=None,
    force_parametric=False,
    skip_excel=False,
    file_name=None,
    manual_transform=None,
    analysis_log=None,
):
    # Late import avoids module-cycle issues while keeping behavior unchanged.
    from analysis.statisticaltester import StatisticalTester
    from datetime import datetime

    if analysis_log is None:
        analysis_log = []

    try:
        validate_test_design(test_name=test, between=between, within=within, subject=subject)
        extraction_result = ExtractionEngine().execute(
            {
                "mode": "advanced_group_extraction",
                "df": df,
                "test": test,
                "dv": dv,
                "between": between,
                "within": within,
            }
        )
        extraction_updates = dict(extraction_result.metadata or {})
        if extraction_updates.get("error"):
            return {
                "error": str(extraction_updates.get("error")),
                "test": str(extraction_updates.get("test") or f"{test} (failed)"),
            }

        samples = dict(extraction_updates.get("samples") or {})
        groups = list(extraction_updates.get("groups") or [])
        df_original = extraction_updates.get("df_original", df.copy())
        original_samples = dict(extraction_updates.get("original_samples") or {})

        if transformed_samples is None or recommendation is None:
            logger.debug("DEBUG: Using existing test results from prepare_advanced_test")

        print("DEBUG: transformed_samples =", transformed_samples)
        print("DEBUG: samples =", samples)
        if transformed_samples is None:
            fallback_warning = ValidationError(
                "transformed_samples missing; falling back to untransformed samples copy."
            )
            logger.warning(str(fallback_warning))
            transformed_samples = {k: v.copy() for k, v in samples.items()}

        valid_groups = [g for g in groups if g in transformed_samples and len(transformed_samples[g]) > 0]
        try:
            for group in valid_groups:
                validate_minimum_n(
                    transformed_samples.get(group, []),
                    min_n=2,
                    label=str(group),
                    allow_missing=False,
                )
        except ValidationError as exc:
            err_msg = str(exc)
            logger.warning(err_msg)
            result = {
                "test_info": test_info,
                "recommendation": recommendation,
                "error": err_msg,
                "test": None,
                "p_value": None,
            }
            return StatisticalTester._standardize_results(result)

        print("DEBUG: valid_groups =", valid_groups)
        print("DEBUG: recommendation =", recommendation)

        transformation_result = TransformationEngine().execute(
            {
                "mode": "advanced_transformation",
                "df": df,
                "dv": dv,
                "test": test,
                "between": between,
                "within": within,
                "test_info": test_info,
                "transformed_samples": transformed_samples,
            }
        )
        transformation_updates = dict(transformation_result.metadata or {})
        transformation_type = transformation_updates.get("transformation_type")
        if transformation_updates.get("error"):
            msg = str(transformation_updates["error"])
            print(msg)
            result = {
                "test_info": test_info,
                "recommendation": recommendation,
                "error": msg,
                "test": transformation_updates.get("test", transformation_type),
                "p_value": None,
            }
            return StatisticalTester._standardize_results(result)

        df_transformed = transformation_updates.get("df_transformed", df.copy())
        transformed_samples = transformation_updates.get("transformed_samples", transformed_samples)

        result = {"test_info": test_info, "recommendation": recommendation}
        recommendation_result = RecommendationEngine().execute(
            {
                "mode": "advanced_recommendation",
                "recommendation": recommendation,
                "force_parametric": force_parametric,
                "test_info": test_info,
            }
        )
        recommendation_updates = dict(recommendation_result.metadata or {})
        effective_recommendation = recommendation_updates.get("effective_recommendation", recommendation)

        if recommendation_updates.get("forced"):
            logger.debug(
                "DEBUG: User explicitly forced parametric test, overriding recommendation '%s'",
                recommendation,
            )
        else:
            logger.debug("DEBUG: Using recommendation from normality tests: '%s'", recommendation)
            if effective_recommendation == "non_parametric" and recommendation != "non_parametric":
                logger.debug("DEBUG: Model residuals are NOT normal, forcing non_parametric")

        if effective_recommendation == "parametric":
            if test == "mixed_anova":
                res = StatisticalTester._run_mixed_anova_logged(df_transformed, dv, subject, between, within, alpha)
            elif test == "repeated_measures_anova":
                res = StatisticalTester._run_repeated_measures_anova_logged(
                    df_transformed,
                    dv,
                    subject,
                    within,
                    alpha,
                    test_info=test_info,
                )
            else:
                res = StatisticalTester._run_two_way_anova_logged(
                    df_transformed,
                    dv,
                    between,
                    alpha,
                    test_info=test_info,
                )
            res.update(result)
            assumption_bridge_result = AssumptionBridgeEngine().execute(
                {
                    "mode": "advanced_assumption_projection",
                    "res": res,
                    "test_info": test_info,
                }
            )
            assumption_updates = dict(assumption_bridge_result.metadata or {})
            for key in ["test_info", "normality_tests", "variance_test", "transformation", "boxcox_lambda"]:
                if key in assumption_updates:
                    res[key] = assumption_updates[key]

            if res.get("p_value") is not None and res["p_value"] < alpha:
                advanced_posthoc_result = AdvancedPostHocEngine().execute(
                    {
                        "mode": "advanced_parametric",
                        "test": test,
                        "df_transformed": df_transformed,
                        "dv": dv,
                        "subject": subject,
                        "between": between,
                        "within": within,
                        "alpha": alpha,
                    }
                )
                advanced_posthoc_updates = dict(advanced_posthoc_result.metadata or {})
                if advanced_posthoc_updates.get("pairwise_comparisons"):
                    res["pairwise_comparisons"] = advanced_posthoc_updates.get("pairwise_comparisons", [])
                    current_posthoc = res.get("posthoc_test", "")
                    new_posthoc = advanced_posthoc_updates.get("posthoc_test") or advanced_posthoc_result.test_name
                    should_override = (
                        not current_posthoc
                        or current_posthoc == "Two-Way ANOVA Post-hoc Tests"
                        or ("Pingouin" in str(current_posthoc) and new_posthoc and "Tukey" in str(new_posthoc))
                    )
                    if should_override:
                        res["posthoc_test"] = new_posthoc
                elif advanced_posthoc_updates.get("error"):
                    warnings_list = res.setdefault("warnings", [])
                    if advanced_posthoc_updates["error"] not in warnings_list:
                        warnings_list.append(advanced_posthoc_updates["error"])

            res["raw_data"] = original_samples
            if transformation_type and transformation_type not in ["none", "None", "Keine"]:
                res["raw_data_transformed"] = transformed_samples

            if test == "repeated_measures_anova" and subject and within:
                res["plot_subject_trajectories"] = StatisticalTester._build_subject_trajectories_from_long_df(
                    df_original,
                    dv,
                    subject,
                    [within[0]],
                    group_order=list(original_samples.keys()),
                )
            elif test == "mixed_anova" and subject and between and within:
                res["plot_subject_trajectories"] = StatisticalTester._build_subject_trajectories_from_long_df(
                    df_original,
                    dv,
                    subject,
                    [between[0], within[0]],
                    group_order=list(original_samples.keys()),
                )

            finalization_result = FinalizationEngine().execute(
                {
                    "mode": "advanced_result",
                    "res": res,
                    "skip_excel": skip_excel,
                    "file_name": file_name,
                    "export_stem": f"{test}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "analysis_log": res.get("analysis_log", None),
                }
            )
            finalization_updates = dict(finalization_result.metadata or {})
            if finalization_updates.get("warning"):
                logger.warning(finalization_updates["warning"])
            for key in ["excel_file", "final_test_label", "tested_against"]:
                if key in finalization_updates:
                    res[key] = finalization_updates[key]
            return res

        if effective_recommendation == "non_parametric":
            logger.debug(f"DEBUG: Nonparametric fallback required for {test}")

            if test == "repeated_measures_anova":
                res = perform_friedman_test(
                    data=df_original.copy(),
                    dv=dv,
                    within_factor=within[0],
                    subject_col=subject,
                    alpha=alpha,
                )
            elif test == "two_way_anova":
                res = perform_freedman_lane_test(
                    data=df_original.copy(),
                    dv=dv,
                    factor_a=between[0],
                    factor_b=between[1],
                    alpha=alpha,
                )
            elif test == "mixed_anova":
                res = perform_brunner_langer_ats(
                    data=df_original.copy(),
                    dv=dv,
                    between_factor=between[0],
                    within_factor=within[0],
                    subject_col=subject,
                    alpha=alpha,
                )
            else:
                res = {
                    "test": f"{test} (non-parametric fallback not available)",
                    "error": f"No non-parametric fallback implemented for test type: {test}",
                    "p_value": None,
                    "statistic": None,
                    "model_class": "Unknown",
                }

            res["test_info"] = test_info
            res["parametric_assumptions_violated"] = True
            res["raw_data"] = original_samples

            if transformation_type and transformation_type not in ["none", "None", "Keine"]:
                res["transformation"] = transformation_type
                res["raw_data_transformed"] = transformed_samples

            if test == "repeated_measures_anova" and subject and within:
                res["plot_subject_trajectories"] = StatisticalTester._build_subject_trajectories_from_long_df(
                    df_original,
                    dv,
                    subject,
                    [within[0]],
                    group_order=list(original_samples.keys()),
                )
            elif test == "mixed_anova" and subject and between and within:
                res["plot_subject_trajectories"] = StatisticalTester._build_subject_trajectories_from_long_df(
                    df_original,
                    dv,
                    subject,
                    [between[0], within[0]],
                    group_order=list(original_samples.keys()),
                )

            assumption_bridge_result = AssumptionBridgeEngine().execute(
                {
                    "mode": "advanced_assumption_projection",
                    "res": res,
                    "test_info": test_info,
                }
            )
            assumption_updates = dict(assumption_bridge_result.metadata or {})
            for key in ["test_info", "normality_tests", "variance_test", "transformation", "boxcox_lambda"]:
                if key in assumption_updates:
                    res[key] = assumption_updates[key]

            nonparam_posthoc_result = AdvancedPostHocEngine().execute(
                {
                    "mode": "nonparametric_fallback",
                    "res": res,
                    "test": test,
                    "df_original": df_original,
                    "dv": dv,
                    "subject": subject,
                    "between": between,
                    "within": within,
                    "alpha": alpha,
                }
            )
            nonparam_posthoc_updates = dict(nonparam_posthoc_result.metadata or {})
            if nonparam_posthoc_updates:
                for key in [
                    "pairwise_comparisons",
                    "posthoc_test",
                    "warnings",
                    "analysis_note",
                    "posthoc_skipped",
                    "posthoc_skip_reason",
                    "error",
                ]:
                    if key in nonparam_posthoc_updates:
                        res[key] = nonparam_posthoc_updates[key]

            reporting_result = ReportingEngine().execute(
                {
                    "mode": "modern_fallback_analysis_log",
                    "res": res,
                    "test": test,
                    "dv": dv,
                    "test_info": test_info,
                    "transformation_type": transformation_type,
                }
            )
            reporting_updates = dict(reporting_result.metadata or {})
            if reporting_updates.get("analysis_note"):
                res["analysis_note"] = reporting_updates["analysis_note"]

            res.pop("fitted_model", None)
            res["analysis_log"] = reporting_updates.get("analysis_log", "")
            res = StatisticalTester._standardize_results(res)

            finalization_result = FinalizationEngine().execute(
                {
                    "mode": "advanced_result",
                    "res": res,
                    "skip_excel": skip_excel,
                    "file_name": file_name,
                    "export_stem": f"{test}_modern_model_fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "analysis_log": res.get("analysis_log", None),
                }
            )
            finalization_updates = dict(finalization_result.metadata or {})
            if finalization_updates.get("warning"):
                logger.warning(finalization_updates["warning"])
            for key in ["excel_file", "final_test_label", "tested_against"]:
                if key in finalization_updates:
                    res[key] = finalization_updates[key]

            return res

        logger.warning(
            "Unknown recommendation '%s' for %s — returning error.",
            recommendation,
            test,
        )
        return {
            "error": f"Unknown test recommendation: {recommendation}",
            "test": test,
            "p_value": None,
            "statistic": None,
        }

    except ValidationError as e:
        return {
            "error": str(e),
            "test": f"{test} (failed)",
            "p_value": None,
            "statistic": None,
        }
    except Exception as e:
        import traceback

        logger.error(f"ERROR in perform_advanced_test: {str(e)}")
        logger.debug(traceback.format_exc())
        return {
            "error": f"Error performing the test: {str(e)}",
            "test": f"{test} (failed)",
            "p_value": None,
            "statistic": None,
        }
