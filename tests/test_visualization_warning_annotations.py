"""Silent-fallback plot paths must draw a visible, export-persistent warning
on the axes, not just log a warning — per the "scientific transparency over
silent degradation" paradigm (docs/superpowers/specs/2026-07-03-visualization-error-transparency-design.md).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visualization.datavisualizer import DataVisualizer


def test_linthresh_uses_5th_percentile_not_min():
    # One artifact-tiny value (0.00001) alongside a real noise band (~1.0-1.3)
    # and real signal (50-200). A min-based threshold would collapse to
    # ~0.000005; the 5th-percentile estimator should sit near the noise band.
    groups = ["A"]
    samples = {"A": [0.00001, 1.0, 1.2, 1.1, 1.3, 1.05, 50.0, 100.0, 200.0]}
    count, thresh = DataVisualizer._analyze_nonpositive_values(groups, samples)
    assert count == 0  # no values <= 0 in this sample
    assert thresh is not None
    assert thresh > 0.01, f"linthresh collapsed toward the single artifact value: {thresh}"


def test_analyze_counts_nonpositive_and_returns_none_thresh_when_all_zero():
    groups = ["A"]
    samples = {"A": [0.0, 0.0, 0.0]}
    count, thresh = DataVisualizer._analyze_nonpositive_values(groups, samples)
    assert count == 3
    assert thresh is None


def test_analyze_handles_single_nonzero_value_without_crash():
    groups = ["A"]
    samples = {"A": [0.0, 0.0, 5.0]}
    count, thresh = DataVisualizer._analyze_nonpositive_values(groups, samples)
    assert count == 2
    assert thresh == 5.0


def test_notice_annotation_uses_neutral_style_distinct_from_warning():
    fig, ax = plt.subplots()
    DataVisualizer._draw_notice_annotation(ax, "Data Notice: test")
    DataVisualizer._draw_warning_annotation(ax, "Data Warning: test")
    notice_text = next(t for t in ax.texts if "Data Notice" in t.get_text())
    warning_text = next(t for t in ax.texts if "Data Warning" in t.get_text())
    assert (
        notice_text.get_bbox_patch().get_facecolor()
        != warning_text.get_bbox_patch().get_facecolor()
    ), "notice and warning annotations must be visually distinct"
    plt.close(fig)


def _emm_grouped_pairwise():
    return [
        {"group1": "ctrl:T0", "group2": "drug:T0", "test": "EMM + multivariate-t",
         "p_value": 0.01, "significant": True},
    ]


def test_grouped_emm_failure_draws_visible_warning(monkeypatch):
    def _boom(samples, sep=":"):
        raise RuntimeError("malformed group labels")

    monkeypatch.setattr(DataVisualizer, "grouped_inputs_from_samples", staticmethod(_boom))

    fig, ax = plt.subplots()
    groups = ["ctrl:T0", "ctrl:T1", "drug:T0", "drug:T1"]
    samples = {g: [1.0, 2.0, 3.0] for g in groups}
    config = {"plot_type": "Bar", "show_error_bars": False}

    DataVisualizer.plot_from_config(
        ax, groups, samples, config, pairwise_results=_emm_grouped_pairwise()
    )

    warning_texts = [t.get_text() for t in ax.texts if "Structural Warning" in t.get_text()]
    assert len(warning_texts) == 1, (
        "grouped-EMM fallback must draw an on-canvas warning, not just log one"
    )
    assert "flat pooling" in warning_texts[0]
    plt.close(fig)


def test_logscale_with_nonpositive_data_uses_symlog_not_plain_log():
    # Sprint 2: this scenario now auto-adapts to symlog (lossless) instead of
    # dropping points under a plain log scale with a red warning (Sprint 1
    # behavior) — group A has a real noise/signal band to derive linthresh from.
    fig, ax = plt.subplots()
    groups = ["A", "B"]
    samples = {"A": [1.0, 2.0, -0.5, 0.0], "B": [3.0, 4.0]}

    DataVisualizer.plot_bar(
        groups, samples, ax=ax, save_plot=False, logy=True, show_error_bars=False
    )

    assert ax.get_yscale() == "symlog"
    notice_texts = [t.get_text() for t in ax.texts if "Data Notice" in t.get_text()]
    assert len(notice_texts) == 1
    assert "symlog" in notice_texts[0]
    warning_texts = [t.get_text() for t in ax.texts if "Data Warning" in t.get_text()]
    assert len(warning_texts) == 0, (
        "lossless symlog path must not show the red data-loss warning"
    )
    plt.close(fig)


def test_logscale_with_all_zero_data_falls_back_to_plain_log_with_warning():
    # Degenerate case: no non-zero magnitude anywhere means there's nothing to
    # derive a linthresh from — must fall back to Sprint 1's honest warning
    # rather than inventing an arbitrary threshold.
    fig, ax = plt.subplots()
    groups = ["A"]
    samples = {"A": [0.0, 0.0, 0.0]}

    DataVisualizer.plot_bar(
        groups, samples, ax=ax, save_plot=False, logy=True, show_error_bars=False
    )

    assert ax.get_yscale() == "log"
    warning_texts = [t.get_text() for t in ax.texts if "Data Warning" in t.get_text()]
    assert len(warning_texts) == 1, (
        "degenerate all-zero data has no usable linthresh — must fall back safely"
    )
    plt.close(fig)


def test_logscale_with_all_positive_data_draws_no_warning():
    fig, ax = plt.subplots()
    groups = ["A", "B"]
    samples = {"A": [1.0, 2.0, 3.0], "B": [3.0, 4.0, 5.0]}

    DataVisualizer.plot_bar(
        groups, samples, ax=ax, save_plot=False, logy=True, show_error_bars=False
    )

    warning_texts = [t.get_text() for t in ax.texts if "Data Warning" in t.get_text()]
    assert len(warning_texts) == 0
    plt.close(fig)
