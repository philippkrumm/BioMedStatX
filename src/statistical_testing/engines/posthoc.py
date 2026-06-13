from typing import Any, Mapping

from ..models import StatisticalResult


class PostHocEngine:
    """Executes post-hoc workflows behind a stable strategy interface."""

    def execute(self, data: Mapping[str, Any]) -> StatisticalResult:
        payload = dict(data or {})
        groups = list(payload.get("groups") or [])
        samples = dict(payload.get("samples") or {})
        test_recommendation = str(payload.get("test_recommendation") or "parametric")
        alpha = float(payload.get("alpha", 0.05))
        test_info = payload.get("test_info")
        posthoc_choice = payload.get("posthoc_choice")

        if len(groups) < 3:
            return StatisticalResult(
                test_name="posthoc_engine_failed",
                statistic_value=None,
                p_value=None,
                metadata={"error": f"Post-hoc requires at least 3 groups, got {len(groups)}."},
            )

        try:
            from analysis.statisticaltester import StatisticalTester

            posthoc_results = StatisticalTester.perform_refactored_posthoc_testing(
                groups,
                samples,
                test_recommendation=test_recommendation,
                alpha=alpha,
                posthoc_choice=posthoc_choice,
                test_info=test_info,
            )
            if not posthoc_results:
                return StatisticalResult(
                    test_name="posthoc_engine_failed",
                    statistic_value=None,
                    p_value=None,
                    metadata={"error": "No post-hoc results returned."},
                )

            return StatisticalResult(
                test_name=str(posthoc_results.get("posthoc_test") or "posthoc_completed"),
                statistic_value=None,
                p_value=None,
                metadata=dict(posthoc_results),
            )
        except Exception as exc:
            return StatisticalResult(
                test_name="posthoc_engine_failed",
                statistic_value=None,
                p_value=None,
                metadata={"error": str(exc), "test_recommendation": test_recommendation},
            )
