from typing import Any, Mapping

from ..models import StatisticalResult


class ReportingEngine:
    """Builds reporting payload fragments for advanced fallback flows."""

    def execute(self, data: Mapping[str, Any]) -> StatisticalResult:
        payload = dict(data or {})
        mode = str(payload.get("mode") or "")

        if mode != "modern_fallback_analysis_log":
            return StatisticalResult(
                test_name="reporting_engine_failed",
                statistic_value=None,
                p_value=None,
                metadata={"error": f"Unsupported reporting mode '{mode}'."},
            )

        res = dict(payload.get("res") or {})
        test = payload.get("test")
        dv = payload.get("dv")
        test_info = payload.get("test_info")
        transformation_type = payload.get("transformation_type")

        analysis_log = []
        analysis_log.append(f"Advanced Test Analysis: {test}")
        analysis_log.append(f"Dataset: {dv}")
        analysis_log.append("Fallback path: statsmodels modern model")

        if test_info:
            analysis_log.append("Assumption Check Results:")
            if "pre_transformation" in test_info:
                pre_norm = test_info["pre_transformation"].get("residuals_normality", {})
                if pre_norm.get("p_value") is not None:
                    analysis_log.append(
                        f"- Original data normality: p = {pre_norm['p_value']:.4f} "
                        f"({'Normal' if pre_norm.get('is_normal', False) else 'Not normal'})"
                    )
            if "post_transformation" in test_info:
                post_norm = test_info["post_transformation"].get("residuals_normality", {})
                if post_norm.get("p_value") is not None:
                    analysis_log.append(
                        f"- After transformation normality: p = {post_norm['p_value']:.4f} "
                        f"({'Normal' if post_norm.get('is_normal', False) else 'Not normal'})"
                    )

        if transformation_type and transformation_type not in ["none", "None", "Keine"]:
            analysis_log.append(f"Applied transformation before fallback: {transformation_type}")

        if res.get("analysis_note"):
            analysis_log.append(str(res["analysis_note"]))

        if res.get("model_class") or res.get("model_family"):
            analysis_log.append(
                f"Model used: {res.get('model_class', 'Unknown')} with family {res.get('model_family', 'Unknown')}"
            )

        updates: dict[str, Any] = {}
        if res.get("error"):
            analysis_log.append(f"Error: {res['error']}")
            if res.get("analysis_note"):
                updates["analysis_note"] = f"{res['analysis_note']} Error: {res['error']}"
        elif res.get("p_value") is not None:
            analysis_log.append(f"Fallback result p-value: {res['p_value']:.4f}")
        else:
            analysis_log.append("Fallback result: No p-value available")

        if res.get("posthoc_test"):
            analysis_log.append(f"Post-hoc tests: {res['posthoc_test']}")
            analysis_log.append(
                f"Pairwise comparisons generated: {len(res.get('pairwise_comparisons', []))}"
            )

        updates["analysis_log"] = "\n".join(analysis_log)
        return StatisticalResult(
            test_name="reporting_completed",
            statistic_value=None,
            p_value=None,
            metadata=updates,
        )
