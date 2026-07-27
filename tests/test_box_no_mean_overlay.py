"""T4: a box plot shows only its own median/IQR/whiskers -- never a mean±error
bar overlaid on top.

The box summarises the distribution by median and quartiles; a mean±SD/SE/CI
error bar is a different statistic about a different centre. Overlaying the two
reads as one summary and, on skewed data, the mean bar and the median box can
visibly contradict each other. A user who wants mean±CI has the bar or violin
plot for that. So plot_box draws no error-bar overlay; the bar plot keeps its
error bars (there the bar height IS the mean, so mean±error is coherent).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.container import ErrorbarContainer
import numpy as np

from visualization.datavisualizer import DataVisualizer


def _samples():
    rng = np.random.default_rng(0)
    return {g: list(rng.normal(m, 1.0, 12))
            for g, m in {"A": 0.0, "B": 2.0, "C": 1.0}.items()}


def _errorbar_containers(ax):
    return [c for c in ax.containers if isinstance(c, ErrorbarContainer)]


def test_box_has_no_mean_error_overlay_even_when_requested():
    for etype in ("sd", "se", "ci"):
        fig, ax = plt.subplots()
        DataVisualizer.plot_box(["A", "B", "C"], _samples(), ax=ax,
                                save_plot=False, show_error_bars=True,
                                error_type=etype)
        assert len(_errorbar_containers(ax)) == 0, \
            f"box must not overlay a mean±{etype} error bar on the median/IQR box"
        # the box itself must still be drawn (patches present)
        assert len(ax.patches) >= 3, "the box plot boxes must still be drawn"
        plt.close(fig)

# The bar plot draws its mean±error bars through seaborn's errorbar callback
# (line artists, not an ax.errorbar ErrorbarContainer), so it is unaffected by
# this box-only change and stays covered by test_wave6_ci_error_bars.
