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
from test_all_paths import build_analysis_context
from stats_functions import AnalysisManager


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


def _run_analysis(design: dict, excel_path: str, out_excel_base: str):
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
    assert Path(export_result["excel_path"]).exists()
    assert export_result["html_path"] is not None

    html_output = Path(export_result["html_path"])
    assert html_output.exists()
    text = _read_text(html_output)
    assert "Overview Report" in text
    assert "Summary across exported analyses" in text
    for design in selected:
        assert design["name"] in text


if __name__ == "__main__":
    import pytest as _pytest

    raise SystemExit(_pytest.main([__file__, "-v", "--tb=short"]))
