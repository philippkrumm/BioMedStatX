"""Pure-RM EMM/mvt plot rendering (P3).

A pure repeated-measures design has no between factor, so its EMM result has
bare within-level group labels (no "between:within") and lands on the flat
plot_bar path. The level-vs-baseline result must render significance BRACKETS,
not a compact-letter display (which would imply an all-pairs grouping).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from visualization.datavisualizer import DataVisualizer


def _emm_pairwise():
    # Level-vs-baseline family as emitted by rm_dunnett_emm_mvt -> add_comparison.
    return [
        {"group1": "T0", "group2": "T1", "test": "EMM + multivariate-t",
         "p_value": 0.001, "significant": True},
        {"group1": "T0", "group2": "T2", "test": "EMM + multivariate-t",
         "p_value": 0.20, "significant": False},
        {"group1": "T0", "group2": "T3", "test": "EMM + multivariate-t",
         "p_value": 0.04, "significant": True},
    ]


def test_emm_result_selects_brackets_over_letters():
    assert DataVisualizer._result_uses_brackets(_emm_pairwise(), None) is True
    assert DataVisualizer._result_uses_brackets(
        None, "EMM + multivariate-t (level vs baseline)") is True


def test_tukey_and_anova_still_use_letters():
    # Regression guard: existing all-pairs families keep the letter display.
    assert DataVisualizer._result_uses_brackets(
        [{"test": "Tukey HSD", "group1": "A", "group2": "B", "p_value": 0.01}], None) is False
    assert DataVisualizer._result_uses_brackets(None, "tukey") is False
    assert DataVisualizer._result_uses_brackets(None, "games_howell") is False


def test_pairwise_ttest_still_uses_brackets():
    assert DataVisualizer._result_uses_brackets(
        [{"test": "Paired t-test (Holm-Bonferroni)", "group1": "A",
          "group2": "B", "p_value": 0.01}], None) is True


def test_flat_plot_bar_draws_brackets_for_emm_result():
    fig, ax = plt.subplots()
    rng = np.random.default_rng(0)
    samples = {lvl: list(rng.normal(m, 1.0, 12))
               for lvl, m in {"T0": 0.0, "T1": 2.0, "T2": 0.5, "T3": 1.5}.items()}
    groups = ["T0", "T1", "T2", "T3"]
    n_lines_before = len(ax.lines)
    DataVisualizer.plot_bar(
        groups, samples, ax=ax, save_plot=False,
        show_error_bars=False, pairwise_results=_emm_pairwise(),
    )
    # Each bracket = 3 ax.plot segments; two significant comparisons -> >=6 lines.
    assert len(ax.lines) - n_lines_before >= 6
    # And star annotations were added.
    star_texts = [t.get_text() for t in ax.texts if "*" in t.get_text()]
    assert len(star_texts) >= 2
    plt.close(fig)
