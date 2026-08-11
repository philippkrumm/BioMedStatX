"""Wave-4b BLOCKER 2 repair: group labels must be whitespace-normalized.

analysis_core stringified the group column with `.astype(str)` and the group
list with `[str(g) ...]`, neither stripped. So "A", "A " and " A" -- the same
group with a stray space from a dirty sheet -- counted as three distinct
groups. That both invents phantom groups and starves the real one: here the
four A-rows scatter across three labels, leaving the matched "A" group with
n=2, which the minimum-n gate then blocks -- a real analysis refused because
of invisible whitespace.

Fix: strip both the group column and the group-name list at their definition
points, consistently, so the split still matches. Case-folding is deliberately
NOT done here -- whether "DMSO"/"dmso" should merge is a separate design
question; this patch only removes whitespace, so " a" (case-variant) stays its
own group by design.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import tempfile
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module", autouse=True)
def _qt_and_dialogs():
    try:
        from PyQt5.QtWidgets import QApplication, QDialog
    except Exception:
        yield
        return
    app = QApplication.instance() or QApplication([])
    QDialog.exec_ = lambda self, *a, **k: 0
    QDialog.exec = lambda self, *a, **k: 0
    try:
        from analysis.statisticaltester import UIDialogManager
        for name in ("select_transformation_dialog",
                     "select_nonparametric_posthoc_dialog", "select_control_group_dialog",
                     "select_custom_pairs_dialog"):
            setattr(UIDialogManager, name, staticmethod(lambda *a, **k: None))
        # This suite only checks group-label normalization, not post-hoc. Pick a
        # real method so the analysis COMPLETES and the raw_data these tests
        # inspect is present. None would mean the user cancelled the post-hoc
        # dialog, which now aborts the whole analysis and discards raw_data.
        UIDialogManager.select_posthoc_test_dialog = staticmethod(lambda *a, **k: "games_howell")
    except Exception:
        pass
    yield app


def _run(long_df, groups):
    from analysis.analysis_core import AnalysisManager
    d = tempfile.mkdtemp()
    p = os.path.join(d, "x.xlsx")
    pd.DataFrame({"a": [1]}).to_excel(p, index=False)
    ctx = {
        "injected_df": long_df.reset_index(drop=True),
        "factor_columns": ["Group"], "between_factors": [], "dv_columns": ["Value"],
        "group_labels": groups, "mode": "single", "dependent": False,
    }
    return AnalysisManager.analyze(
        file_path=p, group_col="Group", groups=groups, value_cols=["Value"],
        save_plot=False, skip_plots=True, dependent=False,
        file_name=os.path.join(d, "out"), analysis_context=ctx,
    )


def _fixture():
    # group A split across three whitespace variants (4 rows total), B clean (4)
    rows = [("A", 10.0), ("A ", 10.4), ("A", 9.6), (" A", 10.2),
            ("B", 5.0), ("B", 5.3), ("B", 4.7), ("B", 5.1)]
    long = pd.DataFrame({"Group": [r[0] for r in rows], "Value": [r[1] for r in rows]})
    dirty_groups = ["A", "A ", " A", "B"]   # as a unique()-built UI list would produce
    return long, dirty_groups


def _group_set(result):
    for key in ("raw_data", "descriptive"):
        d = result.get(key) or {}
        if d:
            return set(d.keys())
    return set()


def test_whitespace_variants_collapse_to_one_group():
    long, dirty = _fixture()
    result = _run(long, dirty)

    assert not result.get("blocked"), (
        f"analysis blocked by phantom whitespace groups: {result.get('error')}"
    )
    assert result.get("error") is None, result.get("error")

    groups = _group_set(result)
    assert groups == {"A", "B"}, f"expected 2 groups after strip, got {sorted(groups)}"

    # the real group A must have all 4 of its rows back, not the n=2 the split left
    descr = result.get("descriptive") or {}
    if "A" in descr and isinstance(descr["A"], dict) and "n" in descr["A"]:
        assert descr["A"]["n"] == 4, f"group A lost rows to whitespace split: n={descr['A']['n']}"


def test_case_variants_stay_separate_by_design():
    """Deferred design decision: strip removes whitespace but NOT case. 'A' and
    'a' remain distinct groups until case-folding is decided separately."""
    rows = [("A", 10.0), ("A ", 10.4), ("A", 9.6),
            ("a", 1.0), ("a", 1.2), ("a", 0.9),
            ("B", 5.0), ("B", 5.3), ("B", 4.7)]
    long = pd.DataFrame({"Group": [r[0] for r in rows], "Value": [r[1] for r in rows]})
    result = _run(long, ["A", "A ", "a", "B"])
    groups = _group_set(result)
    # "A"/"A " merged (whitespace), but "a" stays separate (case not folded)
    assert groups == {"A", "a", "B"}, f"case must stay separate; got {sorted(groups)}"
