"""Wave-4b SHOULD-FIX 3+4: preprocessing data loss must be surfaced in the
persistent report, not dropped silently.

Two silent losses at the sample-building chokepoint (analysis_core
``_prepare_contextual_inputs``):

  SF3  a row whose GROUP LABEL is missing (NaN/blank) matches no group and is
       excluded from every sample -- and, when a selected-groups filter is
       active, the ``.isin()`` drops it even earlier -- with no warning.
  SF4  a VALUE cell holding text a number cannot be parsed from ("N/A", a stray
       unit, a German "1,5") is coerced to NaN and dropped without a trace.

The report renders ``results["data_health"]["warnings"]`` (report_summaries
``_build_data_health_warnings`` -> ``_build_assumption_summary``, built for
every result by html_exporter). So the fix routes both diagnostics into that
existing channel; asserting on that field is asserting what the report shows.
The third test is the positive control: clean data must invent no warning.
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
        for name in ("select_transformation_dialog", "select_posthoc_test_dialog",
                     "select_nonparametric_posthoc_dialog", "select_control_group_dialog",
                     "select_custom_pairs_dialog"):
            setattr(UIDialogManager, name, staticmethod(lambda *a, **k: None))
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


def _health_warnings(result):
    dh = result.get("data_health") or {}
    return [str(w) for w in (dh.get("warnings") or [])]


def test_missing_group_label_surfaces_diagnostic():
    rows = [("A", 10.0), ("A", 10.4), ("A", 9.6),
            ("B", 5.0), ("B", 5.3), ("B", 4.7),
            (np.nan, 7.0), (np.nan, 7.2)]   # 2 rows: group label missing
    long = pd.DataFrame({"Group": [r[0] for r in rows], "Value": [r[1] for r in rows]})
    result = _run(long, ["A", "B"])
    assert not result.get("blocked"), result.get("error")

    warns = _health_warnings(result)
    assert any("label" in w.lower() and ("missing" in w.lower() or "blank" in w.lower())
               for w in warns), f"missing-label rows dropped without a diagnostic; warnings={warns}"
    # the count of dropped rows is surfaced, not just "some were dropped"
    assert any("2 row" in w.lower() for w in warns), f"row count not surfaced; warnings={warns}"


def test_nonnumeric_value_surfaces_diagnostic():
    rows = [("A", "10.0"), ("A", "N/A"), ("A", "9.6"), ("A", "10.2"),
            ("B", "5.0"), ("B", "5.3"), ("B", "4.7")]
    long = pd.DataFrame({"Group": [r[0] for r in rows], "Value": [r[1] for r in rows]})
    result = _run(long, ["A", "B"])
    assert not result.get("blocked"), result.get("error")

    warns = _health_warnings(result)
    assert any(("non-numeric" in w.lower() or "number" in w.lower()) and "'a'" in w.lower()
               for w in warns), f"coerced value dropped without a diagnostic naming group A; warnings={warns}"


def test_clean_data_adds_no_preprocessing_warning():
    """Positive control: clean data must NOT invent a preprocessing warning."""
    rows = [("A", 10.0), ("A", 10.4), ("A", 9.6),
            ("B", 5.0), ("B", 5.3), ("B", 4.7)]
    long = pd.DataFrame({"Group": [r[0] for r in rows], "Value": [r[1] for r in rows]})
    result = _run(long, ["A", "B"])
    warns = " ".join(_health_warnings(result)).lower()
    assert "non-numeric" not in warns and "missing or blank" not in warns, (
        f"clean data produced a spurious preprocessing warning: {_health_warnings(result)}"
    )
