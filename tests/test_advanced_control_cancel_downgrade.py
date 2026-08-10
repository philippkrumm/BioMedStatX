"""Caller 3 (advanced RM/mixed/two-way path): cancelling the control-group dialog
during a Dunnett/emm_mvt selection must downgrade to the all-pairs default, not
run against a silently-picked first group.

Before the fix the control dialog returned groups[0] on cancel, so the advanced
engine ran emm_mvt/Dunnett against an arbitrary control. Now the callback returns
None and the engine falls back to its default method (paired_custom for RM).
Captures the method actually handed to the post-hoc computation.
"""
import pandas as pd

from statistical_testing.engines.advanced_posthoc import AdvancedPostHocEngine


def _rm_payload(method_cb, control_cb):
    df = pd.DataFrame({
        "dv": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "Time": ["T1", "T2", "T3", "T1", "T2", "T3"],
        "subj": [1, 1, 1, 2, 2, 2],
    })
    return {
        "test": "repeated_measures_anova",
        "df_transformed": df,
        "dv": "dv",
        "subject": "subj",
        "within": ["Time"],
        "between": None,
        "alpha": 0.05,
        "posthoc_method_callback": method_cb,
        "control_group_callback": control_cb,
    }


def _patch_capture(monkeypatch, captured):
    import analysis.stats_functions as sf

    def _capture(mode, **kwargs):
        captured["method"] = kwargs.get("method")
        captured["control"] = kwargs.get("control_group", "<absent>")
        return {"posthoc_test": kwargs.get("method"),
                "pairwise_comparisons": [{"group1": "T1", "group2": "T2", "p_value": 0.1}]}

    monkeypatch.setattr(sf.PostHocFactory, "perform_posthoc_for_anova",
                        staticmethod(_capture))


def test_control_cancel_downgrades_emm_mvt_to_default(monkeypatch):
    captured = {}
    _patch_capture(monkeypatch, captured)
    AdvancedPostHocEngine()._run_advanced_parametric_posthoc(_rm_payload(
        method_cb=lambda test, dv, default: "emm_mvt",
        control_cb=lambda groups: None,          # user cancels control-group dialog
    ))
    assert captured["method"] == "paired_custom", \
        f"cancel must downgrade to the all-pairs default, got {captured['method']!r}"
    assert captured["control"] == "<absent>", \
        "no control_group may be passed after the selection was cancelled"


def test_control_selected_keeps_emm_mvt(monkeypatch):
    captured = {}
    _patch_capture(monkeypatch, captured)
    AdvancedPostHocEngine()._run_advanced_parametric_posthoc(_rm_payload(
        method_cb=lambda test, dv, default: "emm_mvt",
        control_cb=lambda groups: "T1",          # user picks a control
    ))
    assert captured["method"] == "emm_mvt"
    assert captured["control"] == "T1"
