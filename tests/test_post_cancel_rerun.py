"""A cancelled analysis must not corrupt state for the next run.

Cancelling raises AnalysisCancelledError deep in the engine (a BaseException) and
unwinds through many layers. This guards that the unwind leaves no poisoned
process/module state and no dirty working directory, so a subsequent analysis --
including two cancels back to back with no normal run between them (which would
otherwise mask any per-abort accumulation) -- runs cleanly. Also guards that the
result cockpit recovers from the blanked "cancelled" state to a normal result.
"""
import glob
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _qt():
    from PyQt5.QtWidgets import QApplication
    QApplication.instance() or QApplication([])


def _normal_three_group():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "Group": ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
        "Value": (list(rng.normal(10, 1, 10)) + list(rng.normal(14, 3, 10))
                  + list(rng.normal(18, 5, 10))),
    })


def _nonnormal_three_group():
    rng = np.random.default_rng(3)
    return pd.DataFrame({
        "Group": ["A"] * 12 + ["B"] * 12 + ["C"] * 12,
        "Value": (list(5 + rng.lognormal(1, 1.2, 12)) + list(20 + rng.lognormal(1, 1.2, 12))
                  + list(40 + rng.lognormal(1, 1.2, 12))),
    })


def _run(tmp, tag, df, *, transform, posthoc, nonparam=None):
    from analysis.statisticaltester import UIDialogManager
    from analysis.analysis_core import AnalysisManager
    mp = pytest.MonkeyPatch()
    mp.setattr(UIDialogManager, "select_transformation_dialog", staticmethod(lambda *a, **k: transform), raising=False)
    mp.setattr(UIDialogManager, "select_posthoc_test_dialog", staticmethod(lambda *a, **k: posthoc), raising=False)
    mp.setattr(UIDialogManager, "select_nonparametric_posthoc_dialog", staticmethod(lambda *a, **k: nonparam), raising=False)
    mp.setattr(UIDialogManager, "select_control_group_dialog", staticmethod(lambda *a, **k: None), raising=False)
    try:
        dummy = os.path.join(tmp, "x.xlsx")
        if not os.path.exists(dummy):
            pd.DataFrame({"a": [1]}).to_excel(dummy, index=False)
        out = os.path.join(tmp, f"out_{tag}")
        groups = sorted(df["Group"].unique().tolist())
        ctx = {"injected_df": df.copy(), "factor_columns": ["Group"], "between_factors": ["Group"],
               "dv_columns": ["Value"], "group_labels": groups, "mode": "single", "dependent": False}
        cwd_before = os.getcwd()
        result = AnalysisManager.analyze(
            file_path=dummy, group_col="Group", groups=groups, value_cols=["Value"],
            save_plot=False, skip_plots=True, dependent=False, file_name=out, analysis_context=ctx)
        return result, glob.glob(out + "*.html"), cwd_before == os.getcwd()
    finally:
        mp.undo()


def test_two_cancels_back_to_back_do_not_break_next_run():
    """Cancel -> Cancel -> Normal: no normal run masks the two aborts."""
    tmp = tempfile.mkdtemp()
    # abort #1: post-hoc cancel on normal data
    r1, rep1, cwd1 = _run(tmp, "c1", _normal_three_group(), transform="skip", posthoc=None)
    assert r1.get("cancelled") is True and rep1 == [] and cwd1
    # abort #2: transformation cancel on non-normal data (no normal run between)
    r2, rep2, cwd2 = _run(tmp, "c2", _nonnormal_three_group(), transform=None, posthoc=None)
    assert r2.get("cancelled") is True and rep2 == [] and cwd2
    # now a full run must succeed after the two consecutive aborts
    r3, rep3, cwd3 = _run(tmp, "n3", _normal_three_group(), transform="skip", posthoc="games_howell")
    assert not r3.get("cancelled"), f"run after two cancels was itself cancelled: {list(r3)}"
    assert r3.get("test") and r3.get("posthoc_test"), "run after two cancels produced no result"
    assert rep3 and cwd3, "run after two cancels wrote no report / dirtied cwd"


def test_alternating_cancel_and_normal_stay_clean():
    """Cancel -> Normal -> Cancel -> Normal: interleaving does not accumulate."""
    tmp = tempfile.mkdtemp()
    seq = [
        ("a", _normal_three_group(), dict(transform="skip", posthoc=None), True),
        ("b", _normal_three_group(), dict(transform="skip", posthoc="games_howell"), False),
        ("c", _nonnormal_three_group(), dict(transform=None, posthoc=None), True),
        ("d", _normal_three_group(), dict(transform="skip", posthoc="games_howell"), False),
    ]
    for tag, df, kw, expect_cancel in seq:
        r, rep, cwd = _run(tmp, tag, df, **kw)
        assert cwd, f"{tag}: cwd dirtied"
        if expect_cancel:
            assert r.get("cancelled") is True and rep == [], f"{tag}: expected clean cancel"
        else:
            assert not r.get("cancelled") and r.get("test") and rep, f"{tag}: expected full result"


# NOTE: the UI-side recovery (ResultCockpitWidget.show_cancelled -> set_summary
# repopulates the blanked cards) was verified manually -- after show_cancelled the
# metric cards read "—" and a subsequent set_summary restores real values with the
# output button re-enabled. It is NOT committed as a test here: constructing a bare
# top-level ResultCockpitWidget in pytest segfaults on this macOS/Qt build at widget
# teardown (the same Qt-in-tests fragility tracked under the QDialog.exec_ hygiene
# ticket). The engine-level guards above cover the actual state-corruption risk the
# BaseException unwind could introduce.
