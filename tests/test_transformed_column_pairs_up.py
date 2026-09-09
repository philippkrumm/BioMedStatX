"""The Transformed column may only be printed when it pairs with the raw one.

The two columns come from different extractions, and they disagree about missing
values: one drops a NaN row, the other keeps it. On a frame with scattered NaNs
a cell held 5 raw values against 8 transformed ones, so every printed row after
the first gap named the wrong measurement and three had no partner at all.

The same difference also fooled the change-gate that decides whether to show the
column: the two dicts differ, but by NaN handling rather than by a
transformation, so the column was emitted for a run whose declared
transformation was ``None``.

Dropped rather than realigned, for the same reason the subject labels are:
guessing which value belongs to which is what produced the defect.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd

# Root conftest.py puts src/ on sys.path and forces headless Qt.
from analysis.analysis_core import AnalysisManager


def _frame_with_gaps():
    """A 2x2 layout with one cell never run and NaNs scattered through two others.

    Reproduced from fuzz seed 1983 -- written out rather than referenced, so the
    test does not depend on the generator continuing to draw that combination.
    """
    rng = np.random.default_rng(1983)
    rows = []
    for factor_a, factor_b, gaps in (("A0", "B0", 3), ("A0", "B1", 3), ("A1", "B1", 0)):
        values = [float(rng.normal(0, 1.0)) for _ in range(8)]
        for index in range(gaps):
            values[index * 2 + 1] = float("nan")
        for value in values:
            rows.append({"FacA": factor_a, "FacB": factor_b, "Val": value})
    return pd.DataFrame(rows)


def _analyse(df, transformation=None, force_parametric=True):
    from analysis.statisticaltester import UIDialogManager
    original = UIDialogManager.select_transformation_dialog
    UIDialogManager.select_transformation_dialog = staticmethod(
        lambda *a, **k: transformation or "skip")
    try:
        cells = sorted({f"FacA={r['FacA']}, FacB={r['FacB']}" for _, r in df.iterrows()})
        with tempfile.TemporaryDirectory() as tmp:
            dummy = os.path.join(tmp, "dummy.xlsx")
            pd.DataFrame({"a": [1]}).to_excel(dummy, index=False)
            return AnalysisManager.analyze(
                file_path=dummy, file_name=os.path.join(tmp, "out"),
                group_col="__AUTO_GROUP__", groups=cells, value_cols=["Val"],
                dependent=False, plot_type="Bar", force_parametric=force_parametric,
                analysis_context={
                    "factor_columns": ["FacA", "FacB"], "dv_columns": ["Val"],
                    "group_labels": cells, "display_group_col": "__AUTO_GROUP__",
                    "selected_group_column": "FacA", "selected_groups": [],
                    "mode": "single", "inferred_test": "two_way_anova",
                    "injected_df": df,
                })
    finally:
        UIDialogManager.select_transformation_dialog = original


def test_a_column_that_does_not_line_up_is_not_printed():
    result = _analyse(_frame_with_gaps())
    raw = result.get("raw_data") or {}
    assert raw, result.get("error")

    # Guard the fixture: the two extractions must still disagree, or this test
    # is not exercising anything.
    assert any(len(values) < 8 for values in raw.values()), (
        "the raw extraction no longer drops the missing rows"
    )

    assert result.get("raw_data_transformed") is None, result.get("raw_data_transformed")
    assert result.get("transformed_data") is None, result.get("transformed_data")


def test_no_transformation_means_no_transformed_column():
    """The change-gate must not read a NaN-handling difference as a transformation."""
    result = _analyse(_frame_with_gaps())
    assert result.get("transformation") in (None, "None", "skip")
    assert "raw_data_transformed" not in result or result["raw_data_transformed"] is None


def test_a_clean_frame_still_shows_its_transformed_column():
    """The guard must not take the column away from the runs that earn it."""
    # Strongly lognormal, so the normality check fails and a transformation is
    # actually offered -- a mild skew at n=6 does not reach the dialog, and a
    # test that never opens it would prove nothing about the positive case.
    rng = np.random.default_rng(4)
    rows = [{"FacA": a, "FacB": b, "Val": float(np.exp(rng.normal(i + j, 1.6)))}
            for i, a in enumerate(("A0", "A1")) for j, b in enumerate(("B0", "B1"))
            for _ in range(14)]
    result = _analyse(pd.DataFrame(rows), transformation="log10",
                      force_parametric=False)
    assert result.get("transformation") == "log10", result.get("error")
    transformed = result.get("raw_data_transformed") or {}
    raw = result.get("raw_data") or {}
    assert transformed, "a clean log10 run lost its Transformed column"
    for group, values in transformed.items():
        assert len(values) == len(raw[group])
        # And it really is the transformation, row by row.
        for raw_value, transformed_value in zip(raw[group], values):
            assert np.isclose(float(transformed_value), np.log10(float(raw_value)),
                              rtol=1e-6, atol=1e-9)
