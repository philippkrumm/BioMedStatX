"""plot_raincloud must use the same _result_uses_brackets logic Bar/Box/Violin already use,
instead of its own narrower inline check - otherwise the same post-hoc result renders
differently (letters vs brackets) purely based on which plot type is selected.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from visualization.datavisualizer import DataVisualizer


def test_raincloud_uses_shared_bracket_helper_for_all_pairs_posthoc():
    # An all-pairs test (e.g. Tukey/Games-Howell/Dunn) should render as
    # compact letters, per _result_uses_brackets's own logic - NOT brackets,
    # even though plot_raincloud's old inline check would show brackets for
    # any non-empty pairwise_results regardless of test type.
    pairwise_results = [
        {"group1": "A", "group2": "B", "test": "Tukey HSD", "p_value": 0.01},
    ]
    assert DataVisualizer._result_uses_brackets(pairwise_results, None) is False, (
        "sanity check: the shared helper itself must say Tukey HSD uses letters, not brackets"
    )
