"""PlotAestheticsDialog already receives the real `samples` dict at
construction time (src/ui/dialogs/plot_aesthetics_dialog.py:1512,
self.samples = samples or {}). "Log Y" must be disabled up front when that
data contains values <= 0, since log(<=0) is undefined and matplotlib would
otherwise silently drop those points with no visible indication.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from PyQt5.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

from ui.dialogs.plot_aesthetics_dialog import PlotAestheticsDialog


def test_logy_checkbox_disabled_when_data_has_nonpositive_values():
    groups = ["A", "B"]
    samples = {"A": [1.0, 2.0, -0.5], "B": [3.0, 4.0, 5.0]}
    dialog = PlotAestheticsDialog(groups=groups, samples=samples, show_export_controls=False)
    try:
        assert dialog.style_tab.logy_check.isEnabled() is False
        assert dialog.style_tab.logy_check.isChecked() is False
        assert "≤ 0" in dialog.style_tab.logy_check.toolTip()
    finally:
        dialog.close()


def test_logy_checkbox_enabled_when_data_all_positive():
    groups = ["A", "B"]
    samples = {"A": [1.0, 2.0, 3.0], "B": [3.0, 4.0, 5.0]}
    dialog = PlotAestheticsDialog(groups=groups, samples=samples, show_export_controls=False)
    try:
        assert dialog.style_tab.logy_check.isEnabled() is True
    finally:
        dialog.close()
