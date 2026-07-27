"""Wave-5 #3: outlier-detection default method + Modified-Z-Score small-n guard.

Measured on clean N(0,1) data, iterate=True: the Modified Z-Score flags a
phantom outlier in ~29% of n=3 samples (DLR triplicate — the common
molecular-biology group size), falling to ~12% at n=10, while Grubbs stays at
its nominal ~5% across all n. ModZ was the DEFAULT (on) with Grubbs off, i.e.
the worst-calibrated method was the default for exactly the small n this app is
used with. Since detection is flag-only, the data loss rides on the user
trusting that default.

Fix, two parts:
  (a) the default method flips to Grubbs (nominal across n=3..30);
  (b) if a user deliberately keeps Modified Z-Score, it warns for any group
      with n < 8, where the median/MAD scale is unstable — no silent trust.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module", autouse=True)
def _qt():
    try:
        from PyQt5.QtWidgets import QApplication
    except Exception:
        yield
        return
    yield QApplication.instance() or QApplication([])


# ---- (a) default method flip ----

def test_default_method_is_grubbs_not_modz():
    from ui.dialogs.statistical_analyzer_dialogs import OutlierDetectionDialog
    df = pd.DataFrame({"Group": ["a", "a", "a", "b", "b", "b"],
                       "Value": [1.0, 2, 3, 4, 5, 6]})
    dlg = OutlierDetectionDialog(df, None)
    assert dlg.grubbs_check.isChecked() is True, \
        "Grubbs must be the default (nominal false-positive rate across all n)"
    assert dlg.modz_check.isChecked() is False, \
        "Modified Z-Score must not be default-on (~29% false positive at n=3)"


# ---- (b) Modified-Z-Score small-n guard ----

def _detector(vals, group="a"):
    from analysis.outlier_core import OutlierDetector
    df = pd.DataFrame({"G": [group] * len(vals), "V": [float(v) for v in vals]})
    return OutlierDetector(df, "G", "V")


def test_modz_warns_below_n8():
    det = _detector([10.0, 10.4, 9.6, 10.2, 9.9])  # n=5
    det.run_mod_z_score(threshold=3.5, iterate=True)
    warns = getattr(det, "warnings", [])
    assert any("Modified Z-Score" in w and "n=5" in w for w in warns), \
        f"expected a small-n Modified Z-Score warning naming n=5, got {warns}"


def test_modz_no_warning_at_n10():
    det = _detector(list(10.0 + np.arange(10) * 0.1))  # n=10
    det.run_mod_z_score(threshold=3.5, iterate=True)
    warns = getattr(det, "warnings", [])
    assert not any("unstable" in w.lower() for w in warns), \
        f"n>=8 must not emit the small-n warning, got {warns}"


def test_grubbs_does_not_emit_modz_warning():
    """The guard is Modified-Z-Score-specific: Grubbs at small n is nominal."""
    det = _detector([10.0, 10.4, 9.6, 10.2, 9.9])  # n=5
    det.run_grubbs(alpha=0.05, iterate=True)
    warns = getattr(det, "warnings", [])
    assert not any("Modified Z-Score" in w for w in warns), \
        f"Grubbs must not emit a Modified Z-Score warning, got {warns}"


def test_smalln_warning_reaches_report():
    from export.outlier_html_exporter import export_single
    det = _detector([10.0, 10.4, 9.6, 10.2, 9.9])  # n=5
    det.run_mod_z_score(threshold=3.5, iterate=True)
    d = tempfile.mkdtemp()
    p = os.path.join(d, "o.html")
    export_single(det, p)
    html_text = open(p, encoding="utf-8").read()
    assert "unstable" in html_text.lower() and "n=5" in html_text, \
        "the small-n warning must be visible in the exported report, not just logged"


# ---- (c) Modified-Z-Score small-n warning is also visible up front in the UI ----
# The report warning (above) only appears after detection has run. SCHRITT 3 also
# asks for a warning at selection time, so a user is cautioned before they commit
# to Modified Z-Score on small groups -- not only afterwards in the export.

def _outlier_dialog(group_sizes):
    from ui.dialogs.statistical_analyzer_dialogs import OutlierDetectionDialog
    rows = []
    for gi, n in enumerate(group_sizes):
        for k in range(n):
            rows.append({"Group": f"g{gi}", "Value": float(gi + k * 0.1)})
    df = pd.DataFrame(rows)
    dlg = OutlierDetectionDialog(df, None)
    dlg.group_col_combo.setCurrentText("Group")
    dlg.value_col_combo.setCurrentText("Value")
    return dlg


def test_ui_smalln_warning_visible_for_modz_small_n():
    dlg = _outlier_dialog([5, 6])  # min n = 5 < 8
    dlg.modz_check.setChecked(True)
    assert dlg.modz_smalln_warning.isHidden() is False, \
        "selecting Modified Z-Score on small groups must show a visible UI warning"


def test_ui_smalln_warning_hidden_for_grubbs_default():
    dlg = _outlier_dialog([5, 6])  # small n, but Grubbs is the default
    assert dlg.modz_check.isChecked() is False
    assert dlg.modz_smalln_warning.isHidden() is True, \
        "Grubbs (the default) must not raise the Modified Z-Score small-n warning"


def test_ui_smalln_warning_hidden_for_modz_large_n():
    dlg = _outlier_dialog([10, 12])  # min n = 10 >= 8
    dlg.modz_check.setChecked(True)
    assert dlg.modz_smalln_warning.isHidden() is True, \
        "Modified Z-Score on n>=8 groups is adequately calibrated -- no warning"
