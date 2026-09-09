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
from .validators import (
    ValidationError,
    grouped_samples_changed,
    validate_outcome,
    validate_samples_for_test,
    validate_test_design,
)


logger = logging.getLogger(__name__)


def _attach_transformed(res, original_samples, transformed_samples):
    """Write the transformed values only where they pair with the raw ones.

    Both branches of this pipeline wrote them unconditionally, so a guard added
    to the standard path alone left the misaligned column reaching the page
    through here -- which is how fuzz seed 51307 printed a Box-Cox column
    against a raw column of a different length.
    """
    from statistical_testing.validators import transformed_pairs_up

    raw = res.get("raw_data") or original_samples
    if transformed_pairs_up(raw, transformed_samples):
        res["raw_data_transformed"] = transformed_samples
        return
    logger.warning(
        "transformed values do not line up with the raw ones (%s); the "
        "Transformed column is dropped rather than printed against the wrong "
        "measurements.",
        {g: (len(raw.get(g, [])), len(transformed_samples.get(g, [])))
         for g in raw},
    )


def _attach_raw_data(res, samples, subjects):
    """The raw values and who they belong to are one fact, written in one place.

    The logged test wrappers extract both for themselves, from the frame they
    analyse -- which for a design with technical replicates has been averaged to
    one row per subject and level. This function then replaces the values with
    the untransformed originals. Replacing one half of the pair and leaving the
    other behind is what produced a raw table naming one subject beside another
    subject's value: measured at 24 printed rows out of 24 on a replicated
    repeated-measures design, where 24 raw values per level were being labelled
    from a list of 8.

    So both halves come from the same extraction, or the subject labels are
    dropped and the table shows no Subject column. An absent column says
    nothing; a wrong one says something false.
    """
    res["raw_data"] = samples
    aligned = (
        isinstance(subjects, dict) and subjects
        and set(subjects) == set(samples)
        and all(len(subjects[group]) == len(samples[group]) for group in samples)
    )
    if aligned:
        res["raw_data_subjects"] = subjects
    else:
        if subjects:
            logger.warning(
                "raw-data subject labels do not line up with the extracted values "
                "(%d groups vs %d); the Subject column is dropped rather than guessed.",
                len(subjects), len(samples),
            )
        res.pop("raw_data_subjects", None)


def perform_advanced_test_pipeline(
    df,
    test,
    dv,
    subject,
    between=None,
    within=None,
    covariates=None,
    random_slope=None,
    alpha=0.05,
    transformed_samples=None,
    recommendation=None,
    test_info=None,
    transform_fn=None,
    force_parametric=False,
    file_name=None,
    manual_transform=None,
    analysis_log=None,
    posthoc_method_callback=None,
    control_group_callback=None,
    custom_pairs_callback=None,
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
                "subject": subject,
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
        original_subjects = extraction_updates.get("subjects") or None

        if transformed_samples is None or recommendation is None:
            logger.debug("DEBUG: Using existing test results from prepare_advanced_test")

        logger.debug("DEBUG: transformed_samples = %s", transformed_samples)
        logger.debug("DEBUG: samples = %s", samples)
        if transformed_samples is None:
            fallback_warning = ValidationError(
                "transformed_samples missing; falling back to untransformed samples copy."
            )
            logger.warning(str(fallback_warning))
            transformed_samples = {k: v.copy() for k, v in samples.items()}

        valid_groups = [g for g in groups if g in transformed_samples and len(transformed_samples[g]) > 0]
        # Data-quality pre-flight on the extracted cells
        if test == "logistic_regression":
            # logistic_regression's shape (binary outcome + predictors) doesn't
            # fit validate_samples_for_test's group-based gate - use the
            # single-vector degeneracy gate on the outcome column instead.
            # Previously this test had NO pre-flight gate at all (AT4).
            #
            # validate_outcome coerces via pd.to_numeric, which turns a
            # 2-level TEXT outcome (e.g. "yes"/"no") entirely to NaN - but
            # LogisticRegressionModel.fit() (clinical_models.py) explicitly
            # supports and correctly encodes exactly that shape via its own
            # unique-value identity check. Mirror that encoding here first,
            # so a valid text-labeled outcome isn't miscoerced into looking
            # empty (round-3 audit finding R1).
            _outcome_raw = df[dv]
            _unique_outcome_vals = sorted(_outcome_raw.dropna().unique())
            if len(_unique_outcome_vals) == 2 and set(_unique_outcome_vals) != {0, 1}:
                _outcome_for_check = (_outcome_raw == _unique_outcome_vals[1]).astype(float)
                _outcome_for_check[_outcome_raw.isna()] = float("nan")
            else:
                _outcome_for_check = _outcome_raw
            _outcome_issue = validate_outcome(_outcome_for_check, label=dv, min_n_block=2)
            if _outcome_issue is not None:
                logger.warning("Advanced pre-flight blocked: %s", _outcome_issue.message)
                blocked = StatisticalTester.make_blocked_result(
                    _outcome_issue.message, code=_outcome_issue.code,
                    details={"test": test},
                )
                blocked["test_info"] = test_info
                blocked["recommendation"] = recommendation
                return blocked
        else:
            _quality = validate_samples_for_test(
                transformed_samples, valid_groups, dependent=False, min_n_block=2,
            )
            if _quality.blocking_issue is not None:
                issue = _quality.blocking_issue
                logger.warning("Advanced pre-flight blocked: %s", issue.message)
                blocked = StatisticalTester.make_blocked_result(
                    issue.message, code=issue.code,
                    details={"groups": [str(g) for g in valid_groups], "test": test},
                    warnings=_quality.warnings,
                )
                blocked["test_info"] = test_info
                blocked["recommendation"] = recommendation
                return blocked

        logger.debug("DEBUG: valid_groups = %s", valid_groups)
        logger.debug("DEBUG: recommendation = %s", recommendation)

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
            logger.info(msg)
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
            elif test in ("ancova", "two_way_ancova"):
                # Let the user pick a control level so the EMM post-hoc uses the
                # vs-control (multivariate-t) family; None => pairwise fallback.
                ancova_control = None
                if control_group_callback and between:
                    try:
                        primary_levels = sorted(
                            str(v) for v in df_transformed[between[0]].dropna().unique()
                        )
                        ancova_control = control_group_callback(primary_levels)
                    except Exception as exc:
                        logger.warning("ANCOVA control-group selection failed: %s", exc)
                res = StatisticalTester._run_ancova_logged(
                    df_transformed, dv, between, covariates, alpha,
                    test_info=test_info, control_group=ancova_control,
                )
            elif test == "lmm":
                lmm_control = None
                primary_factor = None
                if between:
                    primary_factor = between[0]
                elif within:
                    primary_factor = within[0]
                if control_group_callback and primary_factor:
                    try:
                        primary_levels = sorted(
                            str(v) for v in df_transformed[primary_factor].dropna().unique()
                        )
                        lmm_control = control_group_callback(primary_levels)
                    except Exception as exc:
                        logger.warning("LMM control-group selection failed: %s", exc)
                res = StatisticalTester._run_lmm_logged(
                    df_transformed, dv, subject, between, within, covariates, random_slope, alpha,
                    test_info=test_info, control_group=lmm_control
                )
            elif test == "logistic_regression":
                res = StatisticalTester._run_logistic_regression_logged(
                    df_transformed, dv, between, covariates, alpha, test_info=test_info
                )
            elif test == "two_way_anova":
                res = StatisticalTester._run_two_way_anova_logged(
                    df_transformed,
                    dv,
                    between,
                    alpha,
                    test_info=test_info,
                )
            else:
                res = {"error": f"Unknown test type: {test}"}
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

            _run_posthoc = res.get("p_value") is not None and res["p_value"] < alpha

            # Mixed designs only. res["p_value"] carries the INTERACTION's p for a
            # mixed ANOVA (statisticaltester.py sets it from the GG-corrected
            # interaction row), so this gate switched the post-hoc off whenever the
            # interaction was not significant -- the common case when there is a
            # real main effect. The effect-driven post-hoc is built precisely for
            # that situation (it returns marginal_within / marginal_between), so it
            # has to be reached before it can decide. Guarded on the test name, so
            # RM and Two-Way keep the original condition unchanged.
            if not _run_posthoc and test == "mixed_anova":
                _effect_ps = [f.get("p_value") for f in (res.get("factors") or [])]
                _effect_ps += [i.get("p_value") for i in (res.get("interactions") or [])]
                _run_posthoc = any(p is not None and p < alpha for p in _effect_ps)

            if _run_posthoc:
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
                        "posthoc_method_callback": posthoc_method_callback,
                        "control_group_callback": control_group_callback,
                        "custom_pairs_callback": custom_pairs_callback,
                    }
                )
                advanced_posthoc_updates = dict(advanced_posthoc_result.metadata or {})
                if advanced_posthoc_updates.get("pairwise_comparisons"):
                    res["pairwise_comparisons"] = advanced_posthoc_updates.get("pairwise_comparisons", [])
                    new_posthoc = advanced_posthoc_updates.get("posthoc_test") or advanced_posthoc_result.test_name
                    res["posthoc_test"] = new_posthoc
                    # Carry the effect-driven post-hoc's own diagnostics across the
                    # layer boundary. The engine already returns them; dropping them
                    # here left "which contrast family did we report, and did the
                    # omnibus gate actually run" unanswerable downstream.
                    for _diag in ("posthoc_mode", "gating_applied", "gating_fallback_reason"):
                        if _diag in advanced_posthoc_updates:
                            res[_diag] = advanced_posthoc_updates[_diag]
                elif advanced_posthoc_updates.get("error"):
                    warnings_list = res.setdefault("warnings", [])
                    if advanced_posthoc_updates["error"] not in warnings_list:
                        warnings_list.append(advanced_posthoc_updates["error"])

            _attach_raw_data(res, original_samples, original_subjects)
            # Store transformed raw data only when the transformation actually
            # changed the values — a truthy-but-no-op label (e.g. "No further",
            # or a name that maps to no transform branch) must not emit a
            # Transformed column identical to Raw (report bug 2026-08).
            if (transformation_type and transformation_type not in ["none", "None", "Keine"]
                    and grouped_samples_changed(original_samples, transformed_samples)):
                _attach_transformed(res, original_samples, transformed_samples)

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
                    "file_name": file_name,
                    "export_stem": f"{test}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "analysis_log": res.get("analysis_log", None),
                }
            )
            finalization_updates = dict(finalization_result.metadata or {})
            if finalization_updates.get("warning"):
                logger.warning(finalization_updates["warning"])
            for key in ["final_test_label", "tested_against"]:
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
            _attach_raw_data(res, original_samples, original_subjects)

            if transformation_type and transformation_type not in ["none", "None", "Keine"]:
                res["transformation"] = transformation_type
                # Gate the transformed dict on an actual value change (see the
                # parametric branch above; report bug 2026-08).
                if grouped_samples_changed(original_samples, transformed_samples):
                    _attach_transformed(res, original_samples, transformed_samples)

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
                    "posthoc_method_callback": posthoc_method_callback,
                    "control_group_callback": control_group_callback,
                    "custom_pairs_callback": custom_pairs_callback,
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
                    "file_name": file_name,
                    "export_stem": f"{test}_modern_model_fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "analysis_log": res.get("analysis_log", None),
                }
            )
            finalization_updates = dict(finalization_result.metadata or {})
            if finalization_updates.get("warning"):
                logger.warning(finalization_updates["warning"])
            for key in ["final_test_label", "tested_against"]:
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
