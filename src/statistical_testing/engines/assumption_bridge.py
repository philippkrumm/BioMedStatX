from typing import Any, Mapping

from ..models import StatisticalResult


class AssumptionBridgeEngine:
    """Projects advanced test_info assumption data to legacy result fields."""

    def execute(self, data: Mapping[str, Any]) -> StatisticalResult:
        payload = dict(data or {})
        mode = str(payload.get("mode") or "")

        if mode != "advanced_assumption_projection":
            return StatisticalResult(
                test_name="assumption_bridge_failed",
                statistic_value=None,
                p_value=None,
                metadata={"error": f"Unsupported assumption bridge mode '{mode}'."},
            )

        res = dict(payload.get("res") or {})
        incoming_test_info = payload.get("test_info")
        if not isinstance(incoming_test_info, Mapping):
            return StatisticalResult(
                test_name="assumption_bridge_completed",
                statistic_value=None,
                p_value=None,
                metadata={},
            )

        merged_test_info = self._merge_test_info(
            existing=(res.get("test_info") if isinstance(res.get("test_info"), Mapping) else None),
            incoming=incoming_test_info,
        )

        updates: dict[str, Any] = {}
        if not isinstance(res.get("test_info"), Mapping) or dict(res.get("test_info") or {}) != merged_test_info:
            updates["test_info"] = merged_test_info

        if "normality_tests" in merged_test_info and "normality_tests" not in res:
            updates["normality_tests"] = merged_test_info["normality_tests"]
        if "variance_test" in merged_test_info and "variance_test" not in res:
            updates["variance_test"] = merged_test_info["variance_test"]
        if "transformation" in merged_test_info and "transformation" not in res:
            updates["transformation"] = merged_test_info["transformation"]
        if "boxcox_lambda" in merged_test_info and "boxcox_lambda" not in res:
            updates["boxcox_lambda"] = merged_test_info["boxcox_lambda"]

        if (
            "pre_transformation" in merged_test_info
            and "post_transformation" in merged_test_info
            and "normality_tests" not in res
            and "normality_tests" not in updates
        ):
            updates["normality_tests"] = self._build_normality_tests(merged_test_info)

        if (
            "pre_transformation" in merged_test_info
            and "post_transformation" in merged_test_info
            and "variance_test" not in res
            and "variance_test" not in updates
        ):
            updates["variance_test"] = self._build_variance_test(merged_test_info)

        return StatisticalResult(
            test_name="assumption_bridge_completed",
            statistic_value=None,
            p_value=None,
            metadata=updates,
        )

    @staticmethod
    def _merge_test_info(existing: Mapping[str, Any] | None, incoming: Mapping[str, Any]) -> dict[str, Any]:
        merged = dict(existing or {})
        for key, value in dict(incoming).items():
            merged.setdefault(key, value)
        return merged

    @staticmethod
    def _build_normality_tests(test_info: Mapping[str, Any]) -> dict[str, Any]:
        normality_tests: dict[str, Any] = {"all_data": {}, "transformed_data": {}}

        pre_norm = dict(test_info.get("pre_transformation", {})).get("residuals_normality", {})
        if pre_norm:
            normality_tests["all_data"] = {
                "statistic": pre_norm.get("statistic") if pre_norm.get("statistic") is not None else "N/A",
                "p_value": pre_norm.get("p_value") if pre_norm.get("p_value") is not None else "N/A",
                "is_normal": pre_norm.get("is_normal", False),
            }

        post_norm = dict(test_info.get("post_transformation", {})).get("residuals_normality", {})
        if post_norm:
            normality_tests["transformed_data"] = {
                "statistic": post_norm.get("statistic") if post_norm.get("statistic") is not None else "N/A",
                "p_value": post_norm.get("p_value") if post_norm.get("p_value") is not None else "N/A",
                "is_normal": post_norm.get("is_normal", False),
            }

        return normality_tests

    @staticmethod
    def _build_variance_test(test_info: Mapping[str, Any]) -> dict[str, Any]:
        variance_test: dict[str, Any] = {}

        pre_var = dict(test_info.get("pre_transformation", {})).get("variance", {})
        if pre_var:
            variance_test.update(
                {
                    "statistic": pre_var.get("statistic") if pre_var.get("statistic") is not None else "N/A",
                    "p_value": pre_var.get("p_value") if pre_var.get("p_value") is not None else "N/A",
                    "equal_variance": pre_var.get("equal_variance", False),
                }
            )

        post_var = dict(test_info.get("post_transformation", {})).get("variance", {})
        if post_var:
            variance_test["transformed"] = {
                "statistic": post_var.get("statistic") if post_var.get("statistic") is not None else "N/A",
                "p_value": post_var.get("p_value") if post_var.get("p_value") is not None else "N/A",
                "equal_variance": post_var.get("equal_variance", False),
            }

        return variance_test