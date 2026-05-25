from typing import Any, Mapping

from ..models import StatisticalResult


class FinalizationEngine:
    """Applies final advanced-result labels and optional Excel export."""

    def execute(self, data: Mapping[str, Any]) -> StatisticalResult:
        payload = dict(data or {})
        mode = str(payload.get("mode") or "")

        if mode != "advanced_result":
            return StatisticalResult(
                test_name="finalization_failed",
                statistic_value=None,
                p_value=None,
                metadata={"error": f"Unsupported finalization mode '{mode}'."},
            )

        res = dict(payload.get("res") or {})
        # Optional HTML generation logic was previously here, but it is now fully handled
        # by the caller (like AdvancedStatisticalPipeline._execute_single_analysis)
        # to properly handle both Single and Multi-dataset HTML reports uniformly.
        
        # No more Excel export within the engine.

        updates: dict[str, Any] = {}

        if res.get("test") and not res.get("final_test_label"):
            updates["final_test_label"] = res["test"]

        final_test_label = updates.get("final_test_label") or res.get("final_test_label")
        if final_test_label and not res.get("tested_against"):
            updates["tested_against"] = final_test_label

        return StatisticalResult(
            test_name="finalization_completed",
            statistic_value=None,
            p_value=None,
            metadata=updates,
        )