"""DataVisualizer.DEFAULT_COLORS must clear WCAG's 3:1 non-text contrast floor against a white
plot background - #33FF57 and #33FFEC (1.35:1 and 1.26:1, computed via the standard sRGB
relative-luminance formula) currently fail it.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from visualization.datavisualizer import DataVisualizer


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


def test_default_colors_all_clear_wcag_non_text_floor_against_white():
    failing = [
        (c, round(_contrast_ratio(c, "#FFFFFF"), 2))
        for c in DataVisualizer.DEFAULT_COLORS
        if _contrast_ratio(c, "#FFFFFF") < 3.0
    ]
    assert not failing, f"colors failing WCAG's 3:1 non-text floor against white: {failing}"
