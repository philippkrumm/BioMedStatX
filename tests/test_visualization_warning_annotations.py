"""Silent-fallback plot paths must draw a visible, export-persistent warning
on the axes, not just log a warning — per the "scientific transparency over
silent degradation" paradigm (docs/superpowers/specs/2026-07-03-visualization-error-transparency-design.md).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visualization.datavisualizer import DataVisualizer


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
