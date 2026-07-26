"""Wave-6 plot polish: significance letters must clear the group's real top
element (data points, box whisker, violin body) with a fixed gap, and the axis
must expand so no letter is clipped.

Before, letters were placed at mean+error, which dropped them inside the point
cloud (bar/box) or the violin body. The fix lifts each letter above whatever is
actually drawn at that column (DataVisualizer._max_drawn_y_near) plus a constant
clearance, and raises ylim to fit them.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
import itertools

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from visualization.datavisualizer import DataVisualizer as DV

GROUPS = ["WT", "KO", "Rescue", "Vehicle", "Drug"]


def _samples():
    rng = np.random.default_rng(7)
    return {g: list(rng.normal(m, 2.5, 8)) for g, m in zip(GROUPS, [10, 12, 15, 11, 18])}


def _pw():
    out = []
    for a, b in itertools.combinations(GROUPS, 2):
        p = 0.001 if (GROUPS.index(b) - GROUPS.index(a)) % 2 else 0.6
        out.append({"group1": a, "group2": b, "p_value": p, "test": "Tukey HSD",
                    "significant": p < 0.05})
    return out


def _letters_by_group(ax):
    picks = []
    for t in ax.texts:
        s = t.get_text().strip()
        if s and all(ch.islower() and ch.isalpha() for ch in s):
            x, y = t.get_position()
            picks.append((x, y))
    picks.sort(key=lambda p: p[0])
    return picks  # [(x, y), ...] in column order


@pytest.mark.parametrize("kind", ["bar", "box", "violin"])
def test_letters_clear_drawn_top_and_are_not_clipped(kind):
    fn = {"bar": DV.plot_bar, "box": DV.plot_box, "violin": DV.plot_violin}[kind]
    s = _samples()
    fig, ax = plt.subplots()
    fn(GROUPS, s, ax=ax, save_plot=False, show_points=True,
       pairwise_results=_pw(), posthoc_method="Tukey HSD")

    picks = _letters_by_group(ax)
    assert len(picks) == len(GROUPS), f"{kind}: expected {len(GROUPS)} letters, got {len(picks)}"

    y_top = ax.get_ylim()[1]
    for i, (x, y_letter) in enumerate(picks):
        drawn_top = DV._max_drawn_y_near(ax, round(x))
        assert drawn_top is not None, f"{kind}: no drawn content near column {i}"
        # letter sits ABOVE the real top element (points / whisker / violin body)
        assert y_letter > drawn_top, (
            f"{kind} group {i}: letter y={y_letter:.2f} not above drawn top {drawn_top:.2f}"
        )
        # and it is inside the (expanded) axis, i.e. not clipped
        assert y_letter <= y_top, f"{kind} group {i}: letter y={y_letter:.2f} clipped above ylim {y_top:.2f}"
    plt.close(fig)
