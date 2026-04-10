import os
from pathlib import Path

from resultsexporter import ResultsExporter


class ExportDispatcher:
    @staticmethod
    def export_analysis_results(results, output_file, analysis_log=None) -> dict:
        excel_path = Path(output_file).resolve()
        excel_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate decision tree once; reuse for both Excel and HTML exports
        tree_path = None
        try:
            from decisiontreevisualizer import DecisionTreeVisualizer
            tree_path = DecisionTreeVisualizer.generate_and_save_for_excel(results)
        except Exception as exc:
            print(f"WARNING EXPORT DISPATCHER: Decision tree pre-generation failed: {exc}")

        html_result = None
        warning = None
        try:
            ResultsExporter.export_results_to_excel(
                results, str(excel_path), analysis_log, pre_generated_tree=tree_path
            )

            html_path = excel_path.with_suffix(".html")
            try:
                from html_exporter import HTMLExporter

                html_result = HTMLExporter.export_results_to_html(
                    results, str(html_path), analysis_log, pre_generated_tree=tree_path
                )
                if html_result is None:
                    warning = f"HTML report export failed for '{html_path.name}'."
            except Exception as exc:
                warning = f"HTML report export failed for '{html_path.name}': {exc}"
                print(f"WARNING EXPORT DISPATCHER: {warning}")
        finally:
            if tree_path and os.path.exists(tree_path):
                try:
                    os.remove(tree_path)
                except Exception:
                    pass

        return {
            "excel_path": str(excel_path),
            "html_path": html_result,
            "warning": warning,
        }

    @staticmethod
    def export_multi_dataset_results(all_results, excel_path) -> dict:
        workbook_path = Path(excel_path).resolve()
        workbook_path.parent.mkdir(parents=True, exist_ok=True)

        ResultsExporter.export_multi_dataset_results(all_results, str(workbook_path))

        html_path = workbook_path.with_name(f"{workbook_path.stem}_report.html")
        warning = None
        html_result = None
        try:
            from html_exporter import HTMLExporter

            html_result = HTMLExporter.export_multi_dataset_results_to_html(all_results, str(html_path))
            if html_result is None:
                warning = f"HTML overview export failed for '{html_path.name}'."
        except Exception as exc:
            warning = f"HTML overview export failed for '{html_path.name}': {exc}"
            print(f"WARNING EXPORT DISPATCHER: {warning}")

        return {
            "excel_path": str(workbook_path),
            "html_path": html_result,
            "warning": warning,
        }
