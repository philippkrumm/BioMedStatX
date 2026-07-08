"""_add_significance_letters and its raincloud variant catch their own exceptions and only
log.error() + traceback.print_exc() - unlike two other fallback paths in the same file
(grouped-EMM fallback, log-axis fallback) that already call _draw_warning_annotation so a
figure export doesn't silently ship with no significance annotations and no visible indication
why.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from visualization.datavisualizer import DataVisualizer


def test_add_significance_letters_failure_draws_visible_warning(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("synthetic significance-letters failure")
    monkeypatch.setattr(DataVisualizer, "get_significance_letters", staticmethod(_boom))

    fig, ax = plt.subplots()
    df = pd.DataFrame({"Value": [1.0, 2.0, 3.0, 4.0]})
    groups = ["A", "B"]
    samples = {"A": [1.0, 2.0], "B": [3.0, 4.0]}

    DataVisualizer._add_significance_letters(
        ax, df, groups, samples, test_recommendation="anova",
        height_offset=0.05, font_size=10, error_type="sd", pairwise_results=None
    )

    warning_texts = [t.get_text() for t in ax.texts if "Warning" in t.get_text()]
    assert warning_texts, f"expected a visible warning annotation, got: {[t.get_text() for t in ax.texts]}"
    plt.close(fig)


def test_add_significance_letters_raincloud_failure_draws_visible_warning(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("synthetic significance-letters failure")
    monkeypatch.setattr(DataVisualizer, "get_significance_letters", staticmethod(_boom))

    fig, ax = plt.subplots()
    groups = ["A", "B"]
    samples = {"A": [1.0, 2.0], "B": [3.0, 4.0]}

    DataVisualizer._add_significance_letters_raincloud(
        ax, groups, samples, test_recommendation="anova",
        height_offset=0.05, font_size=10, positions=None, pairwise_results=None
    )

    warning_texts = [t.get_text() for t in ax.texts if "Warning" in t.get_text()]
    assert warning_texts, f"expected a visible warning annotation, got: {[t.get_text() for t in ax.texts]}"
    plt.close(fig)
