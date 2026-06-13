from typing import Any, Mapping

from ..models import StatisticalResult


class RecommendationEngine:
    """Resolves effective advanced-test recommendation from overrides and assumptions."""

    def execute(self, data: Mapping[str, Any]) -> StatisticalResult:
        payload = dict(data or {})
        mode = str(payload.get("mode") or "")

        if mode != "advanced_recommendation":
            return StatisticalResult(
                test_name="recommendation_failed",
                statistic_value=None,
                p_value=None,
                metadata={"error": f"Unsupported recommendation mode '{mode}'."},
            )

        recommendation = str(payload.get("recommendation") or "")
        force_parametric = bool(payload.get("force_parametric", False))
        test_info = payload.get("test_info")

        if force_parametric:
            return StatisticalResult(
                test_name="recommendation_resolved",
                statistic_value=None,
                p_value=None,
                metadata={
                    "effective_recommendation": "parametric",
                    "forced": True,
                },
            )

        effective = recommendation
        if isinstance(test_info, Mapping):
            post_norm = dict(test_info.get("post_transformation", {})).get("residuals_normality", {})
            if post_norm:
                if not bool(post_norm.get("is_normal", False)):
                    effective = "non_parametric"
            else:
                pre_norm = dict(test_info.get("pre_transformation", {})).get("residuals_normality", {})
                if pre_norm and not bool(pre_norm.get("is_normal", False)):
                    effective = "non_parametric"

        return StatisticalResult(
            test_name="recommendation_resolved",
            statistic_value=None,
            p_value=None,
            metadata={
                "effective_recommendation": effective,
                "forced": False,
            },
        )