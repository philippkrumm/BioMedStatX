"""Pre-2.0 audit: the one-way post-hoc fallbacks still named "tukey".

For >2 independent groups `select_comparison_test` returns `welch_anova`
unconditionally when the residuals are normal (and `kruskal_wallis` otherwise) --
`is_homoscedastic` is not even consulted. Classic equal-variance one-way ANOVA is
not reachable, so Tukey HSD, which assumes homoscedasticity, must not appear in
this branch at all: falling back to it after a Welch omnibus contradicts the very
assumption the Welch path exists for.

Two leftovers did exactly that:
  * `posthoc_choice = "tukey"` when Dunnett was chosen but no control group was
    picked (or the control dialog raised) -- reachable by cancelling one dialog;
  * `default_method = ... else "tukey"`, which additionally matched no radio
    button in the one-way option list, so the intended pre-selection silently
    fell through to the first entry.

Games-Howell is the heteroscedasticity-robust all-pairs test and is what the
dialog already offers and pre-selects.
"""
import numpy as np
import pytest

import statistical_testing.posthoc_fallback as pf


def _samples():
    rng = np.random.default_rng(11)
    return {"Ctrl": list(rng.normal(10, 1, 9)),
            "Low": list(rng.normal(12, 3, 9)),
            "High": list(rng.normal(15, 6, 9))}


def _run(monkeypatch, *, dialog_returns, control_returns=None, control_raises=False):
    class _UI:
        @staticmethod
        def select_posthoc_test_dialog(**kwargs):
            _UI.seen_default = kwargs.get("default_method")
            return dialog_returns

        @staticmethod
        def select_control_group_dialog(groups):
            if control_raises:
                raise RuntimeError("control dialog exploded")
            return control_returns

        @staticmethod
        def select_custom_pairs_dialog(groups):
            return [("Ctrl", "Low")]

    monkeypatch.setattr(pf, "_get_ui_dialog_manager", lambda: _UI)
    res = pf.PosthocFallbackEngine.perform_refactored_posthoc_testing(
        list(_samples()), _samples(), "welch", alpha=0.05,
        posthoc_choice=None, control_group=None, is_dependent=False,
    )
    return res, getattr(_UI, "seen_default", None)


def test_dunnett_without_control_group_falls_back_to_games_howell(monkeypatch):
    res, _ = _run(monkeypatch, dialog_returns="dunnett", control_returns=None)
    label = res.get("posthoc_test") or ""
    assert "Tukey" not in label, (
        f"fell back to Tukey after a Welch omnibus: {label!r}. Tukey assumes the "
        "variance homogeneity the Welch path exists because it is not given."
    )
    assert "Games-Howell" in label, label
    for comp in res["pairwise_comparisons"]:
        assert "Tukey" not in comp["test"], comp["test"]


def test_dunnett_control_dialog_failure_falls_back_to_games_howell(monkeypatch):
    res, _ = _run(monkeypatch, dialog_returns="dunnett", control_raises=True)
    label = res.get("posthoc_test") or ""
    assert "Tukey" not in label, label
    assert "Games-Howell" in label, label


def test_cancelled_post_hoc_dialog_aborts_analysis(monkeypatch):
    """Cancelling the one-way post-hoc dialog aborts the whole analysis
    (AnalysisCancelledError propagates up) rather than silently running a default
    method. (Product decision: post-hoc cancel = abort.)"""
    import pytest
    from statistical_testing.validators import AnalysisCancelledError

    with pytest.raises(AnalysisCancelledError):
        _run(monkeypatch, dialog_returns=None)


def test_default_method_offered_to_dialog_is_a_valid_one_way_option(monkeypatch):
    """default_method is echoed into the dialog for pre-selection, so it must be a
    real one-way option (games_howell), never tukey after a Welch omnibus."""
    res, seen_default = _run(monkeypatch, dialog_returns="games_howell")
    assert seen_default in {"games_howell", "dunnett", "paired_custom"}, seen_default
    assert seen_default != "tukey"
    label = res.get("posthoc_test") or ""
    assert "Tukey" not in label, label
