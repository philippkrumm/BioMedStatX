"""Every report chart must go through the shared layout builder.

The x-label clipping bug came back five or six times because each chart carried
its own copy of the layout/axis block: a fix applied to one renderer or one plot
type left the others untouched. These tests pin the two properties that keep
that from happening again — the builder always frames and automargins every
axis, and no chart is allowed to hand-roll its own layout instead.
"""
import re
from pathlib import Path

import pytest

from export.report_charts import _ChartsMixin
from visualization import style_tokens

SRC = Path(__file__).resolve().parents[1] / "src"


def test_base_layout_always_frames_both_axes():
    layout = _ChartsMixin._base_layout()
    for axis in ("xaxis", "yaxis"):
        assert layout[axis]["automargin"] is True
        assert layout[axis]["showline"] is True
        assert layout[axis]["linecolor"] == style_tokens.FRAME_COLOR


def test_base_layout_uses_the_shared_tokens():
    layout = _ChartsMixin._base_layout()
    assert layout["font"]["family"] == style_tokens.FONT_FAMILY_STACK
    assert layout["font"]["color"] == style_tokens.INK
    assert layout["paper_bgcolor"] == style_tokens.PAPER_BGCOLOR
    assert layout["plot_bgcolor"] == style_tokens.PLOT_BGCOLOR


def test_axis_title_shorthand_still_gets_automargin():
    """A chart that only passes `xaxis_title=` must not lose the frame/automargin."""
    layout = _ChartsMixin._base_layout(xaxis_title="Coefficient")
    assert "xaxis_title" not in layout          # folded into the axis dict
    assert layout["xaxis"]["title"] == "Coefficient"
    assert layout["xaxis"]["automargin"] is True


def test_extra_axes_are_framed_too():
    layout = _ChartsMixin._base_layout(yaxis2=dict(domain=[0.0, 0.22]))
    assert layout["yaxis2"]["automargin"] is True
    assert layout["yaxis2"]["domain"] == [0.0, 0.22]


def test_chart_overrides_win_over_defaults():
    layout = _ChartsMixin._base_layout(xaxis=dict(automargin=False, range=[0, 1]))
    assert layout["xaxis"]["automargin"] is False   # explicit opt-out honoured
    assert layout["xaxis"]["range"] == [0, 1]
    assert layout["xaxis"]["showline"] is True      # rest of the frame still applied


@pytest.mark.parametrize("module", ["export/report_charts.py", "export/report_summaries.py"])
def test_no_chart_hand_rolls_its_own_layout(module):
    """Guard: every update_layout call routes through _base_layout.

    Without this, adding a chart with its own template/font block silently
    reintroduces the drift (that is how one chart ended up in Arial while ten
    others stayed on a hardcoded Segoe UI, and how four charts lost automargin).
    """
    source = (SRC / module).read_text(encoding="utf-8")
    calls = list(re.finditer(r"\w+\.update_layout\(", source))
    assert calls, f"no charts found in {module} — did the file move?"
    for call in calls:
        tail = source[call.end():call.end() + 60]
        assert "_base_layout" in tail, (
            f"{module}: an update_layout call near offset {call.start()} does not "
            "use _ChartsMixin._base_layout()"
        )
