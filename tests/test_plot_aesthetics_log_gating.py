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


def test_logy_checkbox_stays_enabled_with_symlog_tooltip_when_data_has_nonpositive_values():
    # Sprint 2: the checkbox is no longer hard-disabled — _format_axes now
    # auto-adapts to symlog for this data instead of dropping points, so the
    # user keeps the ability to toggle log scaling on. Only the tooltip changes.
    groups = ["A", "B"]
    samples = {"A": [1.0, 2.0, -0.5], "B": [3.0, 4.0, 5.0]}
    dialog = PlotAestheticsDialog(groups=groups, samples=samples, show_export_controls=False)
    try:
        assert dialog.style_tab.logy_check.isEnabled() is True
        assert "symlog" in dialog.style_tab.logy_check.toolTip().lower()
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
