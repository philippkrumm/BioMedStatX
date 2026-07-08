"""get_config()'s invalid-filename branch shows a blocking QMessageBox.warning and returns a
config dict missing file_name/create_plot/dependent - and because get_config() also runs on
every live-preview tick (not just final dialog acceptance), this modal can fire repeatedly for
as long as an invalid character sits in the field. Fix: sanitize inline, no modal, ever.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from PyQt5.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

from ui.dialogs import plot_aesthetics_dialog
from ui.dialogs.plot_aesthetics_dialog import PlotAestheticsDialog


def test_invalid_filename_is_sanitized_not_dropped(monkeypatch):
    warned = []
    monkeypatch.setattr(plot_aesthetics_dialog.QMessageBox, "warning",
                         lambda *a, **k: warned.append(True))

    groups = ["A", "B"]
    samples = {"A": [1.0, 2.0], "B": [3.0, 4.0]}
    dialog = PlotAestheticsDialog(groups=groups, samples=samples, show_export_controls=True)
    try:
        dialog.file_name_edit.setText("bad<>name")
        config = dialog.get_config()

        assert warned == [], "no modal should ever be shown for an invalid filename"
        assert config["file_name"] == "bad__name"
        assert dialog.file_name_edit.text() == "bad__name", (
            "the field should visibly reflect what will actually be used, not silently differ"
        )
        assert config["create_plot"] is True
        assert config["dependent"] is False
    finally:
        dialog.close()


def test_valid_filename_passes_through_unchanged(monkeypatch):
    warned = []
    monkeypatch.setattr(plot_aesthetics_dialog.QMessageBox, "warning",
                         lambda *a, **k: warned.append(True))

    groups = ["A", "B"]
    samples = {"A": [1.0, 2.0], "B": [3.0, 4.0]}
    dialog = PlotAestheticsDialog(groups=groups, samples=samples, show_export_controls=True)
    try:
        dialog.file_name_edit.setText("good_name-1")
        config = dialog.get_config()

        assert warned == []
        assert config["file_name"] == "good_name-1"
        assert dialog.file_name_edit.text() == "good_name-1"
    finally:
        dialog.close()
