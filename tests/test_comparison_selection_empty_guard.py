"""ComparisonSelectionDialog blocks an all-unchecked OK (GD6, round-3 audit).

Without the accept() guard the dialog would accept an empty selection, and the
caller (_custom_pairs_cb, `chosen if chosen else all_pairs`) would silently
substitute "all pairs" — the user's empty choice vanishing with no signal. The
guard warns and keeps the dialog open instead; Cancel still aborts.

Mirrors the working dialog-construction pattern in
test_plot_aesthetics_filename_sanitization.py: a module-level QApplication and
an explicit dialog.close() in a finally (a headless QDialog left open across
teardown aborts the pytest process under the offscreen platform).
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from PyQt5.QtWidgets import QApplication, QDialog

app = QApplication.instance() or QApplication([])

from ui.dialogs import comparison_selection_dialog
from ui.dialogs.comparison_selection_dialog import ComparisonSelectionDialog


def test_accept_blocks_empty_then_allows_nonempty(monkeypatch):
    warned = []
    monkeypatch.setattr(comparison_selection_dialog.QMessageBox, "warning",
                        lambda *a, **k: warned.append(True))

    dialog = ComparisonSelectionDialog([("A", "B"), ("A", "C")], checked_by_default=False)
    try:
        # Nothing checked -> OK is blocked and a warning is shown.
        dialog.accept()
        assert dialog.result() != QDialog.Accepted, "an empty selection must not accept"
        assert warned == [True], "a warning must be shown on empty accept"

        # Check one -> OK proceeds, no further warning.
        warned.clear()
        dialog.checkboxes[0].setChecked(True)
        dialog.accept()
        assert dialog.result() == QDialog.Accepted, "a non-empty selection must accept"
        assert warned == [], "no warning once a comparison is selected"
        assert dialog.get_selected_comparisons() == [("A", "B")]
    finally:
        dialog.close()
