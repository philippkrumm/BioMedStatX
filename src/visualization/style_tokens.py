"""Single source of truth for cross-system plot style tokens.

Consumed by the two HTML-report rendering systems:

* System C — ``export/report_charts.py::_build_group_comparison_chart`` imports
  these constants directly (server-side Plotly).
* System B — ``templates/plot_designer.js`` cannot import Python at runtime, so
  ``export/html_exporter.py`` serialises :func:`as_json_dict` into the report as
  a ``pd-data-style`` JSON blob; the designer initialises its defaults from it
  (with hardcoded fallbacks, so an older report still renders).

The curated journal palettes below mirror the desktop app
(``visualization/datavisualizer.py`` / ``ui/dialogs/plot_aesthetics_dialog.py``)
so the HTML report and the desktop app agree on colour. This module is the one
place a palette or marker rule is defined; both systems read from here.

Leaf module (imports nothing from the app) — safe to import from any package.
"""

# --- Curated journal palettes (fixed lists) ----------------------------------
PALETTES = {
    "Nature":  ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7"],
    "Science": ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00", "#999999"],
    "NEJM":    ["#BC3C29", "#0072B5", "#E18727", "#20854E", "#7876B1", "#6F99AD", "#FFDC91"],
    "Lancet":  ["#00468B", "#ED0000", "#42B540", "#0099B4", "#925E9F", "#FDAF91", "#AD002A"],
    "Tab10":   ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B",
                "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF"],
}
DEFAULT_PALETTE_NAME = "grayscale"

# --- Grayscale ramp -----------------------------------------------------------
# Default palette: evenly spaced greys generated for the group count (2 for a
# t-test, more for an ANOVA). The darkest step is deliberately NOT pure black so
# the always-black data points (below) stay legible against the darkest
# bar/box/raincloud segment. GRAYSCALE_FLOOR is a starting value — tune by eye
# against real black points during the visual check.
GRAYSCALE_FLOOR = "#404040"
GRAYSCALE_CEIL = "#ffffff"


def _gray_channel(hex_color: str) -> int:
    return int(str(hex_color).lstrip("#")[0:2], 16)


def grayscale_ramp(n: int) -> list[str]:
    """Evenly spaced greys from GRAYSCALE_FLOOR (darkest) to white, for n groups."""
    floor = _gray_channel(GRAYSCALE_FLOOR)
    ceil = _gray_channel(GRAYSCALE_CEIL)
    if n <= 1:
        return ["#%02x%02x%02x" % (floor, floor, floor)]
    out = []
    for i in range(n):
        v = round(floor + (ceil - floor) * i / (n - 1))
        out.append("#%02x%02x%02x" % (v, v, v))
    return out


def resolve_palette(name: str, n: int) -> list[str]:
    """grayscale is generated per group count; every other name is a fixed list."""
    if name == "grayscale":
        return grayscale_ramp(n)
    return PALETTES.get(name, grayscale_ramp(n))


# --- Data-point marker — ALWAYS black, independent of the active palette -------
# One palette-independent rule: fill and edge are both #000000 for every plot
# type and every palette. Contrast against the darkest bar comes from the
# grayscale floor above, not from a conditional (white) edge.
POINT_FILL_COLOR = "#000000"
POINT_EDGE_COLOR = "#000000"
POINT_EDGE_WIDTH = 1
POINT_SIZE = 6

# --- Filled-shape outline (bar / box / violin / raincloud body) ---------------
SHAPE_OUTLINE_COLOR = "#000000"
SHAPE_OUTLINE_WIDTH = 2

# --- Frame / axis line --------------------------------------------------------
INK = "#16313a"
FRAME_COLOR = "rgba(22,49,58,0.75)"   # INK at 75% alpha
FRAME_LINEWIDTH = 0.7

# --- Font ---------------------------------------------------------------------
FONT_FAMILY = "Arial"
FONT_FAMILY_STACK = 'Arial, "Helvetica Neue", Helvetica, sans-serif'
TITLE_SIZE = 16
AXIS_SIZE = 12

# --- Export / background ------------------------------------------------------
PAPER_BGCOLOR = "rgba(0,0,0,0)"
PLOT_BGCOLOR = "#fffdf8"


def as_json_dict() -> dict:
    """Serialisable view consumed by ``plot_designer.js`` (System B)."""
    return {
        "palettes": {k: list(v) for k, v in PALETTES.items()},
        "default_palette": DEFAULT_PALETTE_NAME,
        "grayscale_floor": GRAYSCALE_FLOOR,
        "grayscale_ceil": GRAYSCALE_CEIL,
        "point_fill_color": POINT_FILL_COLOR,
        "point_edge_color": POINT_EDGE_COLOR,
        "point_edge_width": POINT_EDGE_WIDTH,
        "point_size": POINT_SIZE,
        "shape_outline_color": SHAPE_OUTLINE_COLOR,
        "shape_outline_width": SHAPE_OUTLINE_WIDTH,
        "frame_color": FRAME_COLOR,
        "frame_linewidth": FRAME_LINEWIDTH,
        "font_family": FONT_FAMILY,
        "title_size": TITLE_SIZE,
        "axis_size": AXIS_SIZE,
        "paper_bgcolor": PAPER_BGCOLOR,
        "plot_bgcolor": PLOT_BGCOLOR,
    }
