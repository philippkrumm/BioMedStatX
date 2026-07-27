"""T1: non-significant significance brackets are hidden by default.

Showing a bracket for every tested pair (including n.s.) stacks 10+ brackets
on an all-pairs design and crushes the bars into the bottom quarter of the
plot. Nature/Cell/Science practice is to annotate the significant differences
on the figure and carry full disclosure ("all pairwise comparisons were
tested") through the methods/legend and the separate statistics table, which
this app already exports. So the plot hides n.s. brackets by default; a user
who wants them can re-enable via show_ns_brackets.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from visualization.datavisualizer import DataVisualizer


def _pw():
    return [
        {"group1": "A", "group2": "B", "test": "Mann-Whitney",
         "p_value": 0.001, "significant": True},
        {"group1": "A", "group2": "C", "test": "Mann-Whitney",
         "p_value": 0.30, "significant": False},
        {"group1": "B", "group2": "C", "test": "Mann-Whitney",
         "p_value": 0.60, "significant": False},
    ]


def _samples():
    rng = np.random.default_rng(0)
    return {g: list(rng.normal(m, 1.0, 10))
            for g, m in {"A": 0.0, "B": 3.0, "C": 0.2}.items()}


def _ns_texts(ax):
    return [t for t in ax.texts if t.get_text().strip() == "n.s."]


def _star_texts(ax):
    return [t for t in ax.texts if t.get_text().strip()
            and set(t.get_text().strip()) == {"*"}]


def test_ns_brackets_hidden_by_default():
    fig, ax = plt.subplots()
    DataVisualizer.plot_bar(["A", "B", "C"], _samples(), ax=ax, save_plot=False,
                            show_error_bars=False, pairwise_results=_pw())
    assert len(_ns_texts(ax)) == 0, \
        "non-significant brackets must be hidden by default"
    assert len(_star_texts(ax)) == 1, \
        "the one significant pair must still show its bracket"
    plt.close(fig)


def test_ns_brackets_shown_when_enabled():
    fig, ax = plt.subplots()
    DataVisualizer.plot_bar(["A", "B", "C"], _samples(), ax=ax, save_plot=False,
                            show_error_bars=False, pairwise_results=_pw(),
                            show_ns_brackets=True)
    assert len(_ns_texts(ax)) == 2, \
        "both n.s. pairs must show when the option is explicitly enabled"
    plt.close(fig)


# ---- real dispatch path (plot_from_config, used by preview + export) ----

def _cfg(**extra):
    c = {"plot_type": "Bar",
         "colors": {g: "#4E79A7" for g in ["A", "B", "C"]}}
    c.update(extra)
    return c


def test_plot_from_config_hides_ns_by_default():
    fig, ax = plt.subplots()
    DataVisualizer.plot_from_config(ax, ["A", "B", "C"], _samples(), _cfg(),
                                    pairwise_results=_pw())
    assert len(_ns_texts(ax)) == 0
    assert len(_star_texts(ax)) == 1
    plt.close(fig)


def test_plot_from_config_shows_ns_when_config_enables_it():
    fig, ax = plt.subplots()
    DataVisualizer.plot_from_config(ax, ["A", "B", "C"], _samples(),
                                    _cfg(show_ns_brackets=True),
                                    pairwise_results=_pw())
    assert len(_ns_texts(ax)) == 2
    plt.close(fig)

# The dialog-emits-the-flag test lives in test_plot_aesthetics_log_gating.py,
# which is pyplot-free: mixing pyplot's Agg figures with the dialog's own
# Qt FigureCanvas in one module aborts the interpreter.
