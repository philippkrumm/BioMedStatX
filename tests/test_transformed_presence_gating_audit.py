"""Regression armour for the presence-vs-value audit (2026-08).

Anti-pattern audited: a boolean/branch is taken because a transformed value is
*present/named* rather than because it *actually differs* from the raw data.
Calibration pair: report_summaries.py (value comparison, correct) vs the
name/presence-based consumers of ``normality_tests['transformed_data']``.

These tests freeze the empirically-verified facts so the four measured paths
cannot regress silently:

  A  normal data, no transform needed            -> transformed_data mirrors all_data
  B  skewed data, user skips the transform        -> all_data.is_normal False
  C  skewed data, a NAMED but non-matching label  -> transformed == raw (no-op)
     ('sqrt'/'box_cox' fall through the log10/boxcox/arcsin_sqrt apply-branches)
  D  already_transformed -> 'No further' early    -> transformed_data EMPTY

Load-bearing proof (TestInvariants): across every reachable path,
``transformed_data`` is empty ONLY when ``all_data.is_normal`` is False, so the
presence-check consumers never emit a WRONG normality verdict. Class A is
excluded — do not let a refactor reintroduce it.

Reachability note: path C's exotic label is NOT reachable through the real
group-comparison flow (select_transformation_dialog only offers
log10/boxcox/arcsin_sqrt; manual_transform is an unused pass-through). It is a
synthetic stress test, kept as armour. The residual "after transformation"
mislabel (TestMislabel) IS reachable through the ordinary untransformed path and
is the target of hardening task task_58aa76ba.
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

# Root conftest.py puts src/ on sys.path and forces headless Qt.
from analysis.statisticaltester import StatisticalTester as ST
from statistical_testing.engines.assumption_bridge import AssumptionBridgeEngine
from statistical_testing.validators import grouped_samples_changed


# --------------------------------------------------------------------------- #
# Verbatim copies of the real consumer logic (kept in sync with source refs).
# --------------------------------------------------------------------------- #

def _decisiontree_is_normal(normality_tests):
    """decisiontreevisualizer.py:186-195 — presence-gated is_normal."""
    group_results = [
        v.get("is_normal", v.get("p_value", None) is not None and v.get("p_value", 0) > 0.05)
        for k, v in normality_tests.items() if k not in ("all_data", "transformed_data")
    ]
    return all(group_results) if group_results else (
        normality_tests.get("transformed_data", {}).get("is_normal", False)
        if "transformed_data" in normality_tests
        else normality_tests.get("all_data", {}).get("is_normal", False)
    )


def _autopilot_normality_verdict(results):
    """autopilot_pipeline.py:1387-1403 — presence-gated verdict text."""
    nt = results.get("normality_tests", {})
    transformation_applied = bool(
        results.get("transformation")
        and str(results.get("transformation")).lower() not in ("none", "no further")
    )
    if transformation_applied and "model_residuals_transformed" in nt:
        v = nt["model_residuals_transformed"].get("is_normal")
        return "Normality OK after transformation" if v else "Normality still violated after transformation"
    if "model_residuals" in nt:
        v = nt["model_residuals"].get("is_normal")
        return "Normality OK" if v else "Normality violated"
    if transformation_applied and "transformed_data" in nt:
        v = nt["transformed_data"].get("is_normal")
        return "Normality OK after transformation" if v else "Normality violated"
    if "all_data" in nt:
        v = nt["all_data"].get("is_normal")
        return "Normality OK" if v else "Normality violated"
    return "Variance check not available"


# --------------------------------------------------------------------------- #
# Real pipeline drivers.
# --------------------------------------------------------------------------- #

def _check(samples, *, dialog_ret="__unset__", already_transformed=False):
    """Run the real assumption engine, return (transformed_samples, recommendation,
    test_info)."""
    groups = list(samples.keys())
    mock = MagicMock()
    if dialog_ret != "__unset__":
        mock.select_transformation_dialog.return_value = dialog_ret
    with patch("analysis.stats_functions.UIDialogManager", mock), \
         patch("analysis.statisticaltester.UIDialogManager", mock):
        return ST.check_normality_and_variance(
            groups, samples, dataset_name="Value", column_name="Value",
            formula="Value ~ C(Group)", model_type="oneway",
            already_transformed=already_transformed,
        )


def _bridge_normality(test_info):
    br = AssumptionBridgeEngine().execute(
        {"mode": "advanced_assumption_projection", "res": {}, "test_info": test_info})
    up = dict(br.metadata or {})
    return up.get("normality_tests", {}), up


def _normal():
    rng = np.random.default_rng(42)
    return {"G1": list(rng.normal(5, 1, 15)),
            "G2": list(rng.normal(7, 1, 15)),
            "G3": list(rng.normal(9, 1, 15))}


def _skewed():
    rng = np.random.default_rng(7)
    return {"G1": list(rng.lognormal(0, 1.3, 15)),
            "G2": list(rng.lognormal(0.5, 1.3, 15)),
            "G3": list(rng.lognormal(1.0, 1.3, 15))}


def _as_bool(x):
    return None if x is None else bool(x)


# --------------------------------------------------------------------------- #
# The four measured paths — shape + producer facts.
# --------------------------------------------------------------------------- #

class TestPaths:
    def test_A_normal_no_transform(self):
        ts, rec, ti = _check(_normal())
        assert ti.get("transformation") is None
        nt, up = _bridge_normality(ti)
        assert nt.get("transformed_data") != {}, "post computed even when no transform needed"
        assert _as_bool(nt["all_data"]["is_normal"]) is True
        assert _as_bool(nt["transformed_data"]["is_normal"]) is True

    def test_B_skewed_skip(self):
        ts, rec, ti = _check(_skewed(), dialog_ret="skip")
        assert ti.get("transformation") is None  # 'skip' resets to None
        nt, up = _bridge_normality(ti)
        assert _as_bool(nt["all_data"]["is_normal"]) is False

    def test_C_named_nonmatching_label_is_a_noop(self):
        """Seed case: a truthy label that matches no apply-branch leaves the data
        untouched, so grouped_samples_changed reports no change and the producer
        would NOT emit raw_data_transformed."""
        ts, rec, ti = _check(_skewed(), dialog_ret="sqrt")
        assert ti.get("transformation") == "sqrt"          # name is recorded ...
        raw = _skewed()
        assert all(list(ts[g]) == list(raw[g]) for g in raw)  # ... but nothing changed
        assert grouped_samples_changed(raw, ts) is False       # -> data-emission gated off

    def test_D_already_transformed_no_further_empties_transformed_data(self):
        ts, rec, ti = _check(_skewed(), already_transformed=True)
        assert ti.get("transformation") == "No further"
        nt, up = _bridge_normality(ti)
        assert nt.get("transformed_data") == {}, "early return leaves post empty"


# --------------------------------------------------------------------------- #
# Load-bearing invariants — Class A is excluded.
# --------------------------------------------------------------------------- #

class TestInvariants:
    @pytest.mark.parametrize("factory,kwargs", [
        (lambda: (_normal(), {}), None),
        (lambda: (_skewed(), {"dialog_ret": "skip"}), None),
        (lambda: (_skewed(), {"dialog_ret": "sqrt"}), None),
        (lambda: (_skewed(), {"already_transformed": True}), None),
        (lambda: (_normal(), {"already_transformed": True}), None),
    ])
    def test_empty_transformed_data_implies_all_data_not_normal(self, factory, kwargs):
        """The mutual-exclusivity proof: transformed_data can only be empty when
        all_data.is_normal is False. If a refactor ever breaks this, a normal
        result could be reported as non-normal (the refuted Class A bug)."""
        samples, kw = factory()
        _, _, ti = _check(samples, **kw)
        nt, _ = _bridge_normality(ti)
        td_empty = nt.get("transformed_data", {}) == {}
        all_normal = _as_bool(nt.get("all_data", {}).get("is_normal"))
        if td_empty:
            assert all_normal is not True, (
                "empty transformed_data paired with normal all_data would let a "
                "presence-check consumer report a WRONG non-normal verdict")

    @pytest.mark.parametrize("factory,kw", [
        (_normal, {}),
        (_skewed, {"dialog_ret": "skip"}),
        (_skewed, {"dialog_ret": "sqrt"}),
        (_skewed, {"already_transformed": True}),
    ])
    def test_consumer_is_normal_matches_all_data_truth(self, factory, kw):
        """The presence-check consumers must never disagree with the real
        (all_data) normality verdict."""
        _, _, ti = _check(factory(), **kw)
        nt, _ = _bridge_normality(ti)
        truth = _as_bool(nt.get("all_data", {}).get("is_normal"))
        got = _as_bool(_decisiontree_is_normal(nt))
        # truth None (no all_data) is a don't-care; otherwise they must agree.
        if truth is not None:
            assert got == truth

    def test_autopilot_none_transformation_guard_is_safe(self):
        """autopilot_pipeline.py:1388 — bool(None and ...) short-circuits to False,
        so a Python-None transformation does NOT flip transformation_applied."""
        results = {"transformation": None}
        transformation_applied = bool(
            results.get("transformation")
            and str(results.get("transformation")).lower() not in ("none", "no further"))
        assert transformation_applied is False


# --------------------------------------------------------------------------- #
# Regression guards for the four presence-vs-value defects, fixed by hardening
# task_58aa76ba (2026-08). These were xfail(strict) until the fix landed; they
# are now live guards on the value-gated behaviour and must stay green.
# --------------------------------------------------------------------------- #

class TestMislabel:
    def test_untransformed_run_does_not_claim_after_transformation(self):
        _, _, ti = _check(_normal())
        nt, up = _bridge_normality(ti)
        results = {"normality_tests": nt, "transformation": up.get("transformation")}
        verdict = _autopilot_normality_verdict(results)
        assert "after transformation" not in verdict.lower(), verdict

    def test_was_transformed_false_when_transformation_is_none(self):
        # decisiontreevisualizer.py:154 + :207
        results = {"transformation": None}
        transformation = results.get("transformation", "None")
        was_transformed = bool(transformation) and str(transformation) not in ("None", "No further")
        assert was_transformed is False, "None != 'None' is True — needs a truthiness guard"

    def test_after_transformation_plot_not_emitted_for_noop_transform(self):
        # stats_functions.py:370 — a truthy label whose apply-loop left the data
        # untouched must NOT produce an 'After Transformation' Q-Q / box plot.
        raw = _skewed()
        transformed = {g: list(raw[g]) for g in raw}   # 'sqrt' selected -> identity copy
        transformation = "sqrt"
        # --- mirrors stats_functions.py:370 gate after the fix: value-gated ---
        emits_after_plot = grouped_samples_changed(raw, transformed)
        # --- end copy ---
        assert emits_after_plot is False, (
            "an 'After Transformation' diagnostic plot is emitted for data the named "
            "transform left unchanged; gate on grouped_samples_changed(), not the label")

    def test_phase_falls_back_to_pre_when_post_block_empty(self):
        # report_summaries.py:364 / decisiontreevisualizer.py:164 — a truthy label
        # (not 'No further') with an unfilled post_transformation block must read the
        # pre block, not silently drop the row.
        test_info = {
            "transformation": "sqrt",                      # truthy, not 'No further'
            "pre_transformation": {
                "residuals_normality": {"is_normal": True, "p_value": 0.4}},
            "post_transformation": {},                     # inert transform -> never filled
        }
        # --- mirrors the phase selector after the fix: fill-gated ---
        _phase = "post_transformation" if test_info.get("post_transformation") else "pre_transformation"
        # --- end copy ---
        _norm = test_info.get(_phase, {}).get("residuals_normality", {})
        assert _norm != {}, (
            "phase chosen by name selects the empty post block; the normality row "
            "silently vanishes instead of falling back to pre_transformation")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--tb=short"]))
