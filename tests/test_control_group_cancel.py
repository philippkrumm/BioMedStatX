"""Cancelling the control-group dialog must return None, not a silent groups[0].

Old bug: select_control_group_dialog returned groups[0] on Cancel. Every caller
guards `if control_group is None` (Dunnett -> Games-Howell fallback; ANCOVA/LMM ->
no designated control), so returning the first group made all of those guards
dead code -- Dunnett ran against an arbitrary control the user never chose. The
awakened Games-Howell fallback is exercised end-to-end in
test_oneway_posthoc_fallback_is_welch_consistent (control_returns=None); this
pins the dialog's own cancel contract that now feeds it.
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QDialog

from analysis.stats_functions import UIDialogManager

_app = QApplication.instance() or QApplication([])


def test_control_group_dialog_cancel_returns_none(monkeypatch):
    # Simulate the user rejecting/closing the dialog. This is the whole point of
    # the fix: cancel must yield None (so callers fall back), never a silent
    # groups[0]. The accept path is unchanged by this fix and is exercised by
    # every Dunnett post-hoc test that picks a control; a dedicated accept test
    # here would only fight the suite-wide global `QDialog.exec_ = lambda: 0`
    # stubbing that other modules leave installed.
    monkeypatch.setattr(QDialog, "exec_", lambda self: QDialog.Rejected)
    result = UIDialogManager.select_control_group_dialog(["A", "B", "C"], parent=None)
    assert result is None, f"cancel must yield None, got {result!r} (silent groups[0] regression)"
