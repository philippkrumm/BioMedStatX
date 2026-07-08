"""_grouped_bracket_positions and _calculate_bracket_positions silently skip any pairwise
comparison whose group(s) don't resolve to a known bar/position, with no count or notice drawn
on the figure - a scientist has no way to tell some comparisons are simply missing from the
plot rather than "not significant."
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from visualization.datavisualizer import DataVisualizer


def test_grouped_bracket_positions_notices_dropped_comparisons():
    fig, ax = plt.subplots()
    centers = {"A": 0.0, "B": 1.0}
    label_map = {"grpA": "A", "grpB": "B"}
    pairwise_results = [
        {"group1": "grpA", "group2": "grpB", "p_value": 0.01},
        {"group1": "grpA", "group2": "unknown_group", "p_value": 0.02},  # unresolvable -> dropped
    ]

    brackets = DataVisualizer._grouped_bracket_positions(
        ax, centers, label_map, pairwise_results, y_max=10.0, line_height=0.05
    )

    assert len(brackets) == 1, "the resolvable comparison must still produce a bracket"
    notice_texts = [t.get_text() for t in ax.texts if "Notice" in t.get_text() or "notice" in t.get_text().lower()]
    assert notice_texts, f"expected a dropped-comparison notice, got: {[t.get_text() for t in ax.texts]}"
    assert "1" in notice_texts[0]
    plt.close(fig)


def test_calculate_bracket_positions_notices_dropped_comparisons():
    fig, ax = plt.subplots()
    ax.bar([1, 2], [1.0, 2.0])  # so _detect_plot_type has something to detect
    groups = ["A", "B"]
    compare = ["A", "B"]
    pairwise_results = [
        {"group1": "A", "group2": "B", "p_value": 0.01},
        {"group1": "A", "group2": "unknown_group", "p_value": 0.02},  # unresolvable -> dropped
    ]

    brackets = DataVisualizer._calculate_bracket_positions(
        ax, groups, compare, pairwise_results, y_max=10.0, line_height=0.05
    )

    assert len(brackets) == 1
    notice_texts = [t.get_text() for t in ax.texts if "notice" in t.get_text().lower()]
    assert notice_texts, f"expected a dropped-comparison notice, got: {[t.get_text() for t in ax.texts]}"
    plt.close(fig)
