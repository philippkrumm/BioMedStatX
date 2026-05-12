"""
HTML exporter validation for BioMedStatX.

Reuses the existing synthetic design fixtures and AnalysisManager flow to
validate that the new HTML companion report is created and contains the
expected scientific-report structure.

Run with:
    cd validation
    pytest test_html_exporter.py -v --tb=short

Or as a standalone script:
    python test_html_exporter.py
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in [str(ROOT), str(SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from conftest import DESIGNS
from html_exporter import HTMLExporter


def _excel_export_enabled() -> bool:
    """True iff ExportDispatcher's Excel call is uncommented in source.

    Excel export is HTML-only-mode-disabled in commit c1b26ef; this lets the
    tests auto-restore their Excel assertions once the dispatcher is flipped
    back. See validation/test_all_paths.py for the same helper.
    """
    dispatcher = SRC / "export_dispatcher.py"
    try:
        src = dispatcher.read_text(encoding="utf-8")
    except OSError:
        return False
    for raw in src.splitlines():
        line = raw.lstrip()
        if line.startswith("#"):
            continue
        if "ResultsExporter.export_results_to_excel" in line:
            return True
    return False


def _temp_tree_files() -> set[str]:
    temp_dir = Path(tempfile.gettempdir())
    return {p.name for p in temp_dir.glob("decision_tree_*.png")}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _assert_single_report(html_path: Path, result: dict):
    text = _read_text(html_path)
    assert "<meta charset=\"utf-8\">" in text.lower()
    assert "BioMedStatX" in text
    assert "Decision Path" in text
    assert "Statistical Engine" in text
    assert "Assumptions" in text
    assert "Descriptive Statistics" in text
    assert "Raw Data Vault" in text
    assert "Methods Snippet" in text
    assert "navigator.clipboard" in text

    if result.get("pairwise_comparisons"):
        assert "Pairwise Comparisons" in text

    if result.get("raw_data") or result.get("samples"):
        assert "Plot Designer" in text
        assert "pd-data-plot" in text
        assert "pd-download-svg" in text
        assert "pd-download-png" in text


def _run_analysis(design: dict, excel_path: str, out_excel_base: str):
    from test_all_paths import build_analysis_context
    from stats_functions import AnalysisManager

    df = design["df_factory"]()
    group_labels = design["group_labels"] or []
    context = build_analysis_context(design, group_labels)
    groups_arg = group_labels or list(df[design["factor_columns"][0]].dropna().unique())

    return AnalysisManager.analyze(
        file_path=excel_path,
        group_col=design["factor_columns"][0],
        groups=groups_arg,
        sheet_name=0,
        value_cols=design["dv_columns"],
        dependent=design["dependent"],
        skip_plots=True,
        skip_excel=False,
        file_name=out_excel_base,
        analysis_context=context,
    )


@pytest.mark.parametrize(
    "design_name",
    [
        "indep_ttest_parametric",
        "one_way_anova_nonparametric",
        "nan_robustness",
    ],
)
def test_single_analysis_html_export(design_name, make_excel_fixture, tmp_path):
    design = next(d for d in DESIGNS if d["name"] == design_name)
    excel_path = make_excel_fixture(design)
    out_excel_base = str(tmp_path / f"{design['name']}_output")

    tree_files_before = _temp_tree_files()

    result = _run_analysis(design, excel_path, out_excel_base)

    assert result is not None
    assert not result.get("error"), result.get("error")

    excel_output = Path(result.get("excel_file", out_excel_base)).resolve()
    html_output = excel_output.with_suffix(".html")
    assert html_output.exists(), f"Expected HTML companion report at {html_output}"
    _assert_single_report(html_output, result)

    tree_files_after = _temp_tree_files()
    assert tree_files_after == tree_files_before, "Temporary decision tree PNGs were not cleaned up"


def test_single_analysis_html_export_preserves_utf8_special_characters(tmp_path):
    design = next(d for d in DESIGNS if d["name"] == "indep_ttest_parametric").copy()
    special_df = design["df_factory"]().copy()
    special_df["Group"] = special_df["Group"].replace({
        "Control": "Kontrolle μg/mL",
        "Treatment": "Behandlung β-Blocker 37°C",
    })
    design["df_factory"] = lambda df=special_df: df.copy()
    design["group_labels"] = ["Kontrolle μg/mL", "Behandlung β-Blocker 37°C"]

    excel_path = tmp_path / "utf8_special_chars.xlsx"
    special_df.to_excel(excel_path, index=False)

    result = _run_analysis(design, str(excel_path), str(tmp_path / "utf8_special_chars_output"))

    assert result is not None
    assert not result.get("error"), result.get("error")

    excel_output = Path(result.get("excel_file") or (str(tmp_path / "utf8_special_chars_output.xlsx"))).resolve()
    html_output = excel_output.with_suffix(".html")
    assert html_output.exists()

    text = _read_text(html_output)
    assert "Kontrolle μg/mL" in text
    assert "Behandlung β-Blocker 37°C" in text
    assert "charset=\"utf-8\"" in text.lower()


def test_multi_dataset_html_export(tmp_path):
    from test_all_paths import build_analysis_context
    from stats_functions import AnalysisManager

    mock_ui = MagicMock()
    mock_ui.select_posthoc_test_dialog.return_value = "tukey"
    mock_ui.select_nonparametric_posthoc_dialog.return_value = "dunn"
    mock_ui.select_control_group_dialog.return_value = None
    mock_ui.select_custom_pairs_dialog.return_value = []
    mock_ui.select_transformation_dialog.return_value = None

    selected = [
        next(d for d in DESIGNS if d["name"] == name)
        for name in (
            "indep_ttest_parametric",
            "paired_ttest_parametric",
            "one_way_anova_parametric",
            "repeated_anova_parametric",
            "nan_robustness",
        )
    ]

    all_results = {}
    with patch("stats_functions.UIDialogManager", mock_ui), patch("statisticaltester.UIDialogManager", mock_ui):
        for design in selected:
            fixture = tmp_path / f"{design['name']}.xlsx"
            design["df_factory"]().to_excel(fixture, index=False)
            group_labels = design["group_labels"] or []
            context = build_analysis_context(design, group_labels)
            groups_arg = group_labels or list(
                design["df_factory"]()[design["factor_columns"][0]].dropna().unique()
            )
            result = AnalysisManager.analyze(
                file_path=str(fixture),
                group_col=design["factor_columns"][0],
                groups=groups_arg,
                sheet_name=0,
                value_cols=design["dv_columns"],
                dependent=design["dependent"],
                skip_plots=True,
                skip_excel=True,
                file_name=str(tmp_path / design["name"]),
                analysis_context=context,
            )
            assert result is not None
            assert not result.get("error"), result.get("error")
            all_results[design["name"]] = result

    from export_dispatcher import ExportDispatcher

    combined_excel = tmp_path / "html_validation_multi.xlsx"
    export_result = ExportDispatcher.export_multi_dataset_results(all_results, str(combined_excel))
    # Excel export currently disabled in ExportDispatcher (HTML-only mode,
    # commit c1b26ef). The dispatcher still returns the requested xlsx path
    # but never writes the file; only assert existence when the export is on.
    if _excel_export_enabled():
        assert Path(export_result["excel_path"]).exists()
    assert export_result["html_path"] is not None

    html_output = Path(export_result["html_path"])
    assert html_output.exists()
    text = _read_text(html_output)
    assert "Overview Report" in text
    # Accordion card markup is the stable structural marker for the multi-report.
    # The old human-readable subtitle ("Summary across exported analyses") was
    # removed during the multi-dataset UX rework — assert on the card class
    # instead, which is required for cards to render.
    assert "multi-accordion" in text or "acc-card" in text
    for design in selected:
        assert design["name"] in text


def test_plot_designer_context_payload_generation():
    result = {
        "final_test_label": "One-way ANOVA",
        "p_value": 0.013,
        "effect_size": 0.27,
        "effect_size_type": "eta2",
        "thresholds": [
            {"value": 1.5, "label": "LOD", "dash": "dot"},
            2.0,
        ],
        "raw_data": {
            "Control": [1.2, 1.1, 1.3, 1.25],
            "Treatment": [1.8, 1.9, 2.0, 2.1],
        },
        "pairwise_comparisons": [
            {
                "group1": "Control",
                "group2": "Treatment",
                "p_value": 0.01,
                "significant": True,
                "test": "Tukey",
            }
        ],
    }

    context = HTMLExporter._prepare_single_report_context(result)

    assert context.get("plot_designer_enabled") is True
    assert "Control" in context.get("plot_data_json", "")
    assert "Treatment" in context.get("plot_data_json", "")
    assert "mean" in context.get("stats_summary_json", "")
    assert "ci95_lower" in context.get("stats_summary_json", "")
    assert "ci95_upper" in context.get("stats_summary_json", "")
    assert "q1" in context.get("stats_summary_json", "")
    assert "median" in context.get("stats_summary_json", "")
    assert "q3" in context.get("stats_summary_json", "")
    assert "lower_fence" in context.get("stats_summary_json", "")
    assert "upper_fence" in context.get("stats_summary_json", "")
    assert "pairwise_data_json" in context
    assert "plot_reference_lines_json" in context
    assert "LOD" in context.get("plot_reference_lines_json", "")


def test_plot_designer_stats_summary_uses_full_data_before_downsampling():
    values = [float(i) for i in range(8000)]
    result = {
        "final_test_label": "One-way ANOVA",
        "raw_data": {
            "Control": values,
            "Treatment": [v + 10.0 for v in values],
        },
    }

    context = HTMLExporter._prepare_single_report_context(result)
    plot_data = json.loads(context["plot_data_json"])
    stats_summary = json.loads(context["stats_summary_json"])

    assert len(plot_data["Control"]) == 5000
    assert len(plot_data["Treatment"]) == 5000
    assert stats_summary["Control"]["n"] == 8000
    assert stats_summary["Treatment"]["n"] == 8000
    assert math.isclose(stats_summary["Control"]["mean"], sum(values) / len(values), rel_tol=1e-12)
    assert "ci95_lower" in stats_summary["Control"]
    assert "ci95_upper" in stats_summary["Control"]
    assert math.isclose(stats_summary["Control"]["q1"], 1999.75, rel_tol=1e-12)
    assert math.isclose(stats_summary["Control"]["median"], 3999.5, rel_tol=1e-12)
    assert math.isclose(stats_summary["Control"]["q3"], 5999.25, rel_tol=1e-12)
    assert stats_summary["Control"]["lower_fence"] <= stats_summary["Control"]["q1"]
    assert stats_summary["Control"]["upper_fence"] >= stats_summary["Control"]["q3"]


def test_rendered_html_contains_plot_designer_axis_and_font_controls():
    context = HTMLExporter._prepare_single_report_context(
        {
            "raw_data": {"A": [1.0, 2.0, 3.0], "B": [1.2, 2.2, 3.2]},
            "thresholds": [{"value": 1.75, "label": "Clinical cut-off"}],
        }
    )
    html = HTMLExporter._render_template(context, mode="single")

    assert 'id="pd-axis-thickness"' in html
    assert 'id="pd-tick-direction"' in html
    assert 'id="pd-x-tick-angle"' in html
    assert 'id="pd-log-x"' in html
    assert 'id="pd-minor-ticks"' in html
    assert 'id="pd-grid-style"' in html
    assert 'id="pd-grid-alpha"' in html
    assert 'id="pd-y-axis-format"' in html
    assert 'id="pd-y-min"' in html
    assert 'id="pd-y-max"' in html
    assert 'id="pd-ref-zero"' in html
    assert 'id="pd-ref-unit"' in html
    assert 'id="pd-ref-thresholds"' in html
    assert 'id="pd-ref-style"' in html
    assert 'id="pd-ref-width"' in html
    assert 'id="pd-data-reference-lines"' in html
    assert 'id="pd-font-warning"' in html
    assert "resolveFontFamilyStack" in html
    assert "buildReferenceLinesLayer" in html
    assert "gridStyle" in html
    assert "gridAlpha" in html
    assert "minorTicks" in html
    assert "yAxisFormat" in html


def test_plot_reference_line_payload_normalization():
    result = {
        "raw_data": {"A": [1.0, 2.0], "B": [1.2, 2.1]},
        "thresholds": [
            {"value": 1.25, "label": "LOD", "dash": "dot", "width": 2.4},
            {"threshold": 1.8, "name": "Clinical"},
            2.2,
            {"value": "not-a-number", "label": "invalid"},
        ],
    }

    context = HTMLExporter._prepare_single_report_context(result)
    lines = json.loads(context.get("plot_reference_lines_json", "[]"))

    assert len(lines) == 3
    assert lines[0]["label"] == "LOD"
    assert lines[0]["dash"] == "dot"
    assert math.isclose(lines[0]["width"], 2.4, rel_tol=1e-12)
    assert lines[1]["label"] == "Clinical"
    assert lines[2]["label"].startswith("Threshold")


def test_rendered_html_uses_dynamic_decision_tree_node_dimensions():
    context = HTMLExporter._prepare_single_report_context(
        {"raw_data": {"A": [1.0, 2.0, 3.0], "B": [1.2, 2.2, 3.2]}}
    )
    html = HTMLExporter._render_template(context, mode="single")

    assert "const nodeMetrics={};" in html
    assert "function edgeAnchor(" in html
    assert "const metric=nodeMetrics[n.id]" in html


def test_rendered_html_skips_math_runtime_without_latex_markers():
    context = HTMLExporter._prepare_single_report_context({"raw_data": {"A": [1.0, 2.0, 3.0]}})
    html = HTMLExporter._render_template(context, mode="single")

    assert context.get("math_render_enabled") is False
    assert "window.BioMedStatXMath" not in html


def test_rendered_html_injects_math_runtime_when_latex_detected():
    result = {
        "title": "Expression in $\\alpha$-units",
        "units": "$\\mu g/mL$",
        "raw_data": {"A": [1.0, 2.0, 3.0]},
    }
    context = HTMLExporter._prepare_single_report_context(result)
    html = HTMLExporter._render_template(context, mode="single")

    assert context.get("math_render_enabled") is True
    assert "window.BioMedStatXMath" in html
    assert "missing-local-runtime" in html or "loaded-katex" in html or "loaded-mathjax" in html


def test_group_names_with_dollar_signs_do_not_enable_math_runtime():
    result = {
        "title": "Regular title",
        "raw_data": {"$Control$": [1.0, 2.0, 3.0], "Treatment": [1.1, 2.1, 3.1]},
    }

    context = HTMLExporter._prepare_single_report_context(result)
    html = HTMLExporter._render_template(context, mode="single")

    assert context.get("math_render_enabled") is False
    assert "window.BioMedStatXMath" not in html


def test_katex_css_inlines_local_font_assets(tmp_path):
    font_path = tmp_path / "KaTeX_Main-Regular.woff2"
    font_path.write_bytes(b"dummy-font-data")
    css = "@font-face{font-family:KaTeX_Main;src:url('KaTeX_Main-Regular.woff2') format('woff2');}"

    inlined = HTMLExporter._inline_local_css_assets(css, tmp_path)

    assert "data:font/woff2;base64," in inlined
    assert "KaTeX_Main-Regular.woff2" not in inlined


def test_rendered_html_includes_pattern_symbol_designer_controls_and_mappings():
    result = {
        "title": "Pattern symbol validation",
        "raw_data": {
            "Group A": [1.0, 1.1, 1.2, 1.3],
            "Group B": [2.0, 2.1, 2.2, 2.3],
        },
    }
    context = HTMLExporter._prepare_single_report_context(result)
    html = HTMLExporter._render_template(context, mode="single")

    assert "pd-auto-pattern" in html
    assert "pd-pattern-controls" in html
    assert "pd-symbol-controls" in html
    assert "state.patterns" in html
    assert "state.symbols" in html
    assert "marker.pattern" in html
    assert "marker.symbol" in html
    assert "legendgroup" in html
    assert "pd-legend-orientation" in html
    assert "pd-legend-x" in html
    assert "pd-legend-y" in html
    assert "pd-legend-xanchor" in html
    assert "pd-legend-yanchor" in html
    assert "pd-legend-preset-inside-top-right" in html
    assert "pd-legend-preset-outside-right" in html
    assert "pd-legend-preset-bottom-horizontal" in html
    assert "state.legendX" in html
    assert "state.legendY" in html
    assert "state.legendXAnchor" in html
    assert "state.legendYAnchor" in html
    assert "applyLegendPreset" in html
    assert 'var defaultPatternCycle = ["x", "\\\\", "/", "-", "|", "+", "."]' in html
    assert 'var defaultSymbolCycle = ["diamond", "square", "circle", "cross", "triangle-up"]' in html
    assert '<option value="Bar" selected>Bar</option>' in html
    assert "plotType: \"Bar\"" in html
    assert "autoPatternsEnabled: false" in html
    assert 'state.patterns[group] = "";' in html
    assert "showLegend: true" in html
    assert "legendOrientation: \"v\"" in html
    assert "legendX: 1.02" in html
    assert "legendY: 1.0" in html
    assert "legendYAnchor: \"top\"" in html
    assert "legendOutsideRight" in html
    assert "legendBottom" in html


if __name__ == "__main__":
    import pytest as _pytest

    raise SystemExit(_pytest.main([__file__, "-v", "--tb=short"]))
