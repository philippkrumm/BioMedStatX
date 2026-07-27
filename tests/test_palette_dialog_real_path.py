"""Palette wiring must hold on the REAL app path, not just the direct
DataVisualizer.plot_* API.

Plots in the app are only ever produced through PlotAestheticsDialog
(the "Configure Plot" button -> _ap_configure_plot_from_result), which emits
an explicit per-group ``config['colors']`` dict. That dict -- not the
CURATED_PALETTES default inside plot_bar -- is what actually reaches the
exported figure. These tests exercise that dict, so a regression in the
dialog's colour logic (a divergent second palette source, a hard-coded
default that ignores the palette dropdown, modulo recycling that collides two
groups on one colour) is caught at the path the user really drives.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from PyQt5.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

from ui.dialogs.plot_aesthetics_dialog import PlotAestheticsDialog
from visualization.datavisualizer import DataVisualizer


def _dialog(groups):
    samples = {g: [1.0, 2.0, 3.0] for g in groups}
    return PlotAestheticsDialog(
        groups=groups, samples=samples, show_export_controls=False)


def test_default_export_colors_are_the_nature_palette():
    """No user picks -> the exported colours are Nature (the selected default),
    not the old hard-coded #0f766e... list."""
    groups = ["A", "B", "C"]
    dialog = _dialog(groups)
    try:
        colors = dialog.get_config()["colors"]
        nature = DataVisualizer.CURATED_PALETTES["Nature"]
        got = [colors[g].lower() for g in groups]
        want = [nature[i].lower() for i in range(len(groups))]
        assert got == want, f"default export colours {got} != Nature {want}"
    finally:
        dialog.close()


def test_every_curated_palette_applies_without_crashing():
    """Selecting any of the six curated palettes in the dropdown must fill the
    colour buttons with that palette's exact hex -- never raise."""
    groups = ["A", "B", "C"]
    for name in ["Nature", "Okabe-Ito", "Grayscale HC",
                 "Muted Pastel", "Deep", "Turbo"]:
        dialog = _dialog(groups)
        try:
            ct = dialog.colors_tab
            ct.palette_combo.setCurrentText(name)
            ct.on_seaborn_settings_changed()  # the dropdown handler
            expected = DataVisualizer.CURATED_PALETTES[name]
            got = [ct.color_buttons[g].get_color().lower() for g in groups]
            want = [expected[i].lower() for i in range(len(groups))]
            assert got == want, f"{name}: buttons {got} != palette {want}"
        finally:
            dialog.close()


def test_more_groups_than_palette_length_get_distinct_colors():
    """10 groups (> the 8-colour Nature palette) must still export 10 DISTINCT
    colours -- no modulo recycling that paints two conditions the same."""
    groups = [f"G{i}" for i in range(10)]
    dialog = _dialog(groups)
    try:
        colors = dialog.get_config()["colors"]
        vals = [colors[g].lower() for g in groups]
        assert len(set(vals)) == len(groups), (
            f"only {len(set(vals))} distinct colours for {len(groups)} groups: {vals}")
    finally:
        dialog.close()
