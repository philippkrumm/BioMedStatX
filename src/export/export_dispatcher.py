import os
from pathlib import Path


class ExportDispatcher:
    @staticmethod
    def export_analysis_results(results, output_file, analysis_log=None) -> dict:
        base_path = Path(output_file).resolve()
        base_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate decision tree once; reuse for HTML export
        tree_path = None
        try:
            from visualization.decisiontreevisualizer import DecisionTreeVisualizer
            tree_path = DecisionTreeVisualizer.generate_and_save_for_excel(results)
        except Exception as exc:
            print(f"WARNING EXPORT DISPATCHER: Decision tree pre-generation failed: {exc}")

        html_result = None
        warning = None
        try:
            html_path = base_path.with_suffix(".html")
            try:
                from export.html_exporter import HTMLExporter

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
            "html_path": html_result,
            "warning": warning,
        }

    @staticmethod
    def export_multi_dataset_results(all_results, output_file) -> dict:
        base_path = Path(output_file).resolve()
        base_path.parent.mkdir(parents=True, exist_ok=True)

        html_path = base_path.with_name(f"{base_path.stem}_report.html")
        warning = None
        html_result = None
        try:
            from export.html_exporter import HTMLExporter

            html_result = HTMLExporter.export_multi_dataset_results_to_html(all_results, str(html_path))
            if html_result is None:
                warning = f"HTML overview export failed for '{html_path.name}'."
        except Exception as exc:
            warning = f"HTML overview export failed for '{html_path.name}': {exc}"
            print(f"WARNING EXPORT DISPATCHER: {warning}")

        return {
            "html_path": html_result,
            "warning": warning,
        }
