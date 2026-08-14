"""
Advanced-pipeline half of the transformed-data gating fix (report bug 2026-08).

The advanced pipeline (`perform_advanced_test_pipeline`) gates
``raw_data_transformed`` on
``grouped_samples_changed(original_samples, transformed_samples)``. Unlike the
standard flow, its original ``if transformation_type and ...`` guard already
handled the ``None`` case; the residual risk here is a truthy-but-inert label
storing a no-op dict, and a key-format mismatch between the two engines that
build ``original_samples`` and ``transformed_samples`` (which would silently
drop a genuine transform).

Run:
    cd validation
    pytest test_advanced_transform_gating.py -v --tb=short
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in [str(ROOT), str(SRC), str(ROOT / "validation")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from statistical_testing.validators import grouped_samples_changed


def _run(design_name, tmp_path):
    from conftest import DESIGNS
    from test_all_paths import build_analysis_context
    from analysis.stats_functions import AnalysisManager

    design = next(d for d in DESIGNS if d["name"] == design_name)
    df = design["df_factory"]()
    xlsx = Path(tmp_path) / f"{design_name}.xlsx"
    df.to_excel(xlsx, index=False)
    context = build_analysis_context(design, design["group_labels"] or [])
    # Multi-factor designs must pass groups=[] so the engine auto-determines the
    # combined factor groups (mirrors test_all_paths.py).
    if design.get("factors", 1) >= 2:
        groups_arg = []
    else:
        groups_arg = design["group_labels"] or list(df[design["factor_columns"][0]].dropna().unique())
    return AnalysisManager.analyze(
        file_path=str(xlsx), group_col=design["factor_columns"][0], groups=groups_arg,
        sheet_name=0, value_cols=design["dv_columns"], dependent=design["dependent"],
        skip_plots=True, save_plot=False, error_type="sd",
        file_name=str(Path(tmp_path) / f"{design_name}_out"), analysis_context=context,
    )


@pytest.mark.parametrize("design_name", ["two_way_anova_parametric", "mixed_anova_parametric"])
def test_advanced_pipeline_no_transform_omits_dict(design_name, tmp_path):
    """Advanced pipeline (two-way / mixed ANOVA), no transformation: no crash,
    no no-op transformed dict."""
    res = _run(design_name, tmp_path)
    assert not res.get("error"), res.get("error")
    assert "raw_data_transformed" not in res


def test_advanced_engine_keys_match_and_change_detected():
    """The advanced pipeline gates raw_data_transformed on
    grouped_samples_changed(original_samples, transformed_samples). This asserts
    that the two engines that build those maps use identical group-label keys —
    otherwise a genuine transform would be silently dropped (empty key
    intersection). Exercises the real ExtractionEngine + TransformationEngine on
    a two-way design (the full advanced pipeline is not used here because its
    non-parametric branch may invoke R, which can hard-crash in this env)."""
    from conftest import DESIGNS
    from statistical_testing.engines.extraction import ExtractionEngine
    from statistical_testing.engines.transformation import TransformationEngine

    design = next(d for d in DESIGNS if d["name"] == "two_way_anova_nonparametric")
    df = design["df_factory"]()
    between = [c for c in df.columns if c != "Value"][:2]

    ext = ExtractionEngine().execute({
        "mode": "advanced_group_extraction", "df": df, "dv": "Value",
        "test": "two_way_anova", "between": between, "within": None, "subject": None,
    })
    ext_md = dict(ext.metadata or {})
    original_samples = ext_md.get("original_samples") or {}
    samples = ext_md.get("samples") or {}
    assert original_samples, ext_md.get("error")

    def _transform(label):
        tr = TransformationEngine().execute({
            "mode": "advanced_transformation", "df": df, "dv": "Value",
            "test": "two_way_anova", "between": between, "within": None,
            "test_info": {"transformation": label},
            "transformed_samples": {k: list(v) for k, v in samples.items()},
        })
        return dict(tr.metadata or {}).get("transformed_samples") or {}

    log_tr = _transform("log10")
    assert set(original_samples) == set(log_tr), "engines disagree on group-label keys"
    assert grouped_samples_changed(original_samples, log_tr) is True

    noop = _transform(None)
    assert grouped_samples_changed(original_samples, noop) is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--tb=short"]))
