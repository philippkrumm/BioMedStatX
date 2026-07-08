"""_format_axes's logx branch always warns-and-drops non-positive values, unlike logy which
auto-adapts to a lossless symlog scale when a usable linthresh can be derived. Mirrors the
existing logy tests in test_visualization_warning_annotations.py.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from visualization.datavisualizer import DataVisualizer


def test_logx_with_nonpositive_data_uses_symlog_not_plain_log():
    fig, ax = plt.subplots()
    groups = ["A", "B"]
    samples = {"A": [1.0, 2.0, -0.5, 0.0], "B": [3.0, 4.0]}

    DataVisualizer.plot_bar(
        groups, samples, ax=ax, save_plot=False, logx=True, show_error_bars=False
    )

    assert ax.get_xscale() == "symlog"
    notice_texts = [t.get_text() for t in ax.texts if "Data Notice" in t.get_text()]
    assert len(notice_texts) == 1
    assert "symlog" in notice_texts[0]
    warning_texts = [t.get_text() for t in ax.texts if "Data Warning" in t.get_text()]
    assert len(warning_texts) == 0, "lossless symlog path must not show the red data-loss warning"
    plt.close(fig)


def test_logx_with_all_zero_data_falls_back_to_plain_log_with_warning():
    fig, ax = plt.subplots()
    groups = ["A"]
    samples = {"A": [0.0, 0.0, 0.0]}

    DataVisualizer.plot_bar(
        groups, samples, ax=ax, save_plot=False, logx=True, show_error_bars=False
    )

    assert ax.get_xscale() == "log"
    warning_texts = [t.get_text() for t in ax.texts if "Data Warning" in t.get_text()]
    assert len(warning_texts) == 1
    plt.close(fig)


def test_logx_with_all_positive_data_draws_no_warning():
    fig, ax = plt.subplots()
    groups = ["A", "B"]
    samples = {"A": [1.0, 2.0, 3.0], "B": [3.0, 4.0, 5.0]}

    DataVisualizer.plot_bar(
        groups, samples, ax=ax, save_plot=False, logx=True, show_error_bars=False
    )

    warning_texts = [t.get_text() for t in ax.texts if "Data Warning" in t.get_text()]
    assert len(warning_texts) == 0
    plt.close(fig)
