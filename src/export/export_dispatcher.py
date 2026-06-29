from pathlib import Path

import logging
logger = logging.getLogger(__name__)


class ExportDispatcher:
    @staticmethod
    def export_analysis_results(results, output_file, analysis_log=None) -> dict:
        base_path = Path(output_file).resolve()
        base_path.parent.mkdir(parents=True, exist_ok=True)

        html_result = None
        warning = None
        try:
            html_path = base_path.with_suffix(".html")
            try:
                from export.html_exporter import HTMLExporter

                html_result = HTMLExporter.export_results_to_html(
                    results, str(html_path), analysis_log
                )
                if html_result is None:
                    warning = f"HTML report export failed for '{html_path.name}'."
            except Exception as exc:
                warning = f"HTML report export failed for '{html_path.name}': {exc}"
                logger.warning(f"WARNING EXPORT DISPATCHER: {warning}")
        except Exception as exc:
            warning = f"HTML report export failed for '{html_path.name}' (outer): {exc}"
            logger.exception(f"WARNING EXPORT DISPATCHER: {warning}")

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
            logger.warning(f"WARNING EXPORT DISPATCHER: {warning}")

        return {
            "html_path": html_result,
            "warning": warning,
        }
