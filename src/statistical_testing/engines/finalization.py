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
        skip_excel = bool(payload.get("skip_excel", False))
        file_name = payload.get("file_name")
        export_stem = str(payload.get("export_stem") or "analysis_results")
        analysis_log = payload.get("analysis_log")

        updates: dict[str, Any] = {}

        if not skip_excel:
            try:
                from export_dispatcher import ExportDispatcher
                from stats_functions import get_output_path

                excel_file = file_name if file_name else get_output_path(export_stem, "xlsx")
                export_result = ExportDispatcher.export_analysis_results(res, excel_file, analysis_log)
                if export_result.get("warning"):
                    updates["warning"] = export_result["warning"]
                updates["excel_file"] = export_result.get("excel_path", excel_file)
            except Exception as exc:
                updates["warning"] = f"Excel export failed: {exc}"

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