"""
Regression tests for the transformed-data gating fix (report bug 2026-08).

Symptom: an untransformed independent t-test rendered a "Transformed value"
column identical to the "Raw value" column in the HTML report.

Root cause: the standard-flow producers in analysis_core.py gated
``raw_data_transformed`` / ``transformed_data`` on
``results.get('transformation', 'None') != 'None'``. With no transformation the
value is Python ``None`` (not the string ``'None'``), so ``None != 'None'`` was
always True and a no-op copy of the raw data was stored on every analysis. The
advanced pipeline used a truthiness gate (already immune to the ``None`` case)
but could still store a no-op dict for a truthy-but-inert transform label.

Fix: gate on an actual value change via
``statistical_testing.validators.grouped_samples_changed``.

Run:
    cd validation
    pytest test_transformed_data_gating.py -v --tb=short
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in [str(ROOT), str(SRC), str(ROOT / "validation")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from statistical_testing.validators import grouped_samples_changed


# --------------------------------------------------------------------------- #
# Unit tests for the gating helper
# --------------------------------------------------------------------------- #

def test_identity_copy_is_not_a_change():
    raw = {"WT": [1.0, 2.0, 3.0], "KO": [4.0, 5.0]}
    same = {g: list(v) for g, v in raw.items()}
    assert grouped_samples_changed(raw, same) is False


def test_real_change_is_detected():
    raw = {"WT": [1.0, 10.0, 100.0]}
    transformed = {"WT": [np.log10(v) for v in raw["WT"]]}
    assert grouped_samples_changed(raw, transformed) is True


def test_nan_in_same_slot_counts_as_unchanged():
    raw = {"WT": [1.0, float("nan"), 3.0]}
    same = {"WT": [1.0, float("nan"), 3.0]}
    # Plain `raw != same` would be True (NaN != NaN); the helper must treat a
    # NaN copied to the same position as unchanged.
    assert grouped_samples_changed(raw, same) is False


def test_ndarray_containers_do_not_raise():
    # A plain `a == b` on numpy arrays raises the ambiguous-truth ValueError;
    # the helper must be safe against ndarray element containers.
    raw = {"WT": np.array([1.0, 2.0, 3.0])}
    same = {"WT": np.array([1.0, 2.0, 3.0])}
    changed = {"WT": np.array([1.0, 2.0, 4.0])}
    assert grouped_samples_changed(raw, same) is False
    assert grouped_samples_changed(raw, changed) is True


def test_length_or_key_mismatch_counts_as_change():
    assert grouped_samples_changed({"WT": [1.0, 2.0]}, {"WT": [1.0, 2.0, 3.0]}) is True
    assert grouped_samples_changed({"WT": [1.0]}, {"KO": [1.0]}) is False  # no shared key
    assert grouped_samples_changed({"WT": [1.0]}, {"WT": [1.0], "KO": [2.0]},
                                   keys=["WT"]) is False


def test_non_mapping_inputs_are_safe():
    assert grouped_samples_changed(None, {"WT": [1.0]}) is False
    assert grouped_samples_changed({"WT": [1.0]}, None) is False


# --------------------------------------------------------------------------- #
# End-to-end pipeline tests through the real analysis flow
# --------------------------------------------------------------------------- #

def _run(design_name, tmp_path, transform=None):
    from conftest import DESIGNS
    from test_all_paths import build_analysis_context
    from analysis.stats_functions import AnalysisManager

    import contextlib
    from unittest.mock import patch

    design = next(d for d in DESIGNS if d["name"] == design_name)
    df = design["df_factory"]()
    xlsx = Path(tmp_path) / f"{design_name}.xlsx"
    df.to_excel(xlsx, index=False)
    group_labels = design["group_labels"] or []
    context = build_analysis_context(design, group_labels)
    # Multi-factor designs must pass groups=[] so the engine auto-determines the
    # combined factor groups (mirrors test_all_paths.py).
    if design.get("factors", 1) >= 2:
        groups_arg = []
    else:
        groups_arg = group_labels or list(df[design["factor_columns"][0]].dropna().unique())

    with contextlib.ExitStack() as stack:
        if transform is not None:
            stack.enter_context(patch(
                "analysis.statisticaltester.UIDialogManager.select_transformation_dialog",
                staticmethod(lambda *a, **k: transform),
            ))
        return AnalysisManager.analyze(
            file_path=str(xlsx), group_col=design["factor_columns"][0], groups=groups_arg,
            sheet_name=0, value_cols=design["dv_columns"], dependent=design["dependent"],
            skip_plots=True, save_plot=False, error_type="sd",
            file_name=str(Path(tmp_path) / f"{design_name}_out"), analysis_context=context,
        )


def _equal_grouped(a, b):
    if not isinstance(a, dict) or not isinstance(b, dict) or set(a) != set(b):
        return False
    return all(list(a[k]) == list(b[k]) for k in a)


def test_no_transform_omits_transformed_data(tmp_path):
    """An already-normal analysis must not carry a raw==transformed no-op dict."""
    res = _run("indep_ttest_parametric", tmp_path)
    assert not res.get("error"), res.get("error")
    assert "raw_data_transformed" not in res
    assert "transformed_data" not in res


def test_real_transform_keeps_and_differs(tmp_path):
    """A genuinely applied transform must keep the dict AND it must differ from raw."""
    res = _run("skewed_lognormal_boxcox", tmp_path, transform="log10")
    assert not res.get("error"), res.get("error")
    assert "raw_data_transformed" in res, "genuine transform dropped the transformed data"
    assert not _equal_grouped(res["raw_data"], res["raw_data_transformed"]), \
        "transformed data must not mirror the raw data"


def test_inert_transform_label_omits_dict(tmp_path):
    """A transform label that maps to no actual transform branch (here the
    'box_cox' vs 'boxcox' mismatch) leaves data untouched → no no-op dict."""
    res = _run("skewed_lognormal_boxcox", tmp_path, transform="box_cox")
    assert not res.get("error"), res.get("error")
    assert "raw_data_transformed" not in res
    assert "transformed_data" not in res


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--tb=short"]))
