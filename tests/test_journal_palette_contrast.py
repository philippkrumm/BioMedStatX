"""The journal presets (Nature/Science/NEJM/Lancet) must clear WCAG's 3:1
non-text contrast floor against a white plot background - 11 of 29 bundled hex
values once failed it, some badly (#FFDC91 at 1.32:1), despite being presented
as publication-ready presets.

Palette tables now live in a single source (DataVisualizer.CURATED_PALETTES),
so this reads the four journal palettes from there instead of a second copy on
ColorsTab. The floor is asserted only for those four saturated journal sets;
the other curated palettes (Okabe-Ito, Grayscale HC, Muted Pastel, Deep, Turbo)
deliberately include light / colorblind-safe hues that do not aim at 3:1 on
white, so holding them to this floor would be wrong.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from PyQt5.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

from visualization.datavisualizer import DataVisualizer

JOURNAL_PALETTES = ["Nature", "Science", "NEJM", "Lancet"]


def _relative_luminance(hex_color):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    def lin(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = lin(r), lin(g), lin(b)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast_ratio(hex1, hex2):
    l1, l2 = _relative_luminance(hex1), _relative_luminance(hex2)
    l1, l2 = max(l1, l2), min(l1, l2)
    return (l1 + 0.05) / (l2 + 0.05)


def test_journal_palettes_all_clear_wcag_non_text_floor_against_white():
    failing = [
        (journal, c, round(_contrast_ratio(c, "#FFFFFF"), 2))
        for journal in JOURNAL_PALETTES
        for c in DataVisualizer.CURATED_PALETTES[journal]
        if _contrast_ratio(c, "#FFFFFF") < 3.0
    ]
    assert not failing, f"colors failing WCAG's 3:1 non-text floor against white: {failing}"
