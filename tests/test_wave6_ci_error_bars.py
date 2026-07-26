"""Wave-6 BLOCKER B1: the "ci" error bar must be a t-based 95% CI, computed by
one shared helper, consistent across every plot type.

Diagnosis (Monte-Carlo, mu=0 sigma=1): "ci" was three divergent copies --
bar/violin used seaborn's bootstrap CI, box and the annotation letter-height
used the z-approximation 1.96*SD/sqrt(n). Both massively undercover at the
small n this app runs on (n=3: box ~82%, bootstrap ~75%, correct t-CI 95%), and
bar vs box differed ~15% in width on identical data. sd/se were already correct
(Wave-6 CHECK, bar==box==hand) and must stay untouched.

Fix: a single t-based helper (t_{n-1} * s/sqrt(n)) called by all three sites.

These tests encode: (1) the helper's coverage is ~95% across n=3-6; (2) each
real plot type's rendered "ci" half-width equals the helper; (3) the types
agree with each other on identical data. All RED before the fix (helper absent;
box/annotation use 1.96; bar bootstrap differs).
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.container import ErrorbarContainer
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from visualization.datavisualizer import DataVisualizer as DV


# ---------- (1) helper coverage ~95% across small n ----------

@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_ci_helper_coverage_is_95_percent(n):
    rng = np.random.default_rng(20240726 + n)
    trials = 20000
    hits = 0
    for _ in range(trials):
        x = rng.normal(0.0, 1.0, n)
        hw = DV._ci_half_width(x)              # helper under test (t-based)
        if abs(x.mean() - 0.0) <= hw:
            hits += 1
    cov = hits / trials
    assert 0.93 <= cov <= 0.965, f"n={n}: coverage {cov:.3f}, expected ~0.95 (t-based)"


def test_ci_helper_matches_t_formula():
    x = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    n = len(x)
    expect = stats.t.ppf(0.975, n - 1) * np.std(x, ddof=1) / np.sqrt(n)
    assert abs(DV._ci_half_width(x) - expect) < 1e-9


# ---------- (2) each rendered plot type's "ci" == helper ----------

_GROUPS = ["A", "B", "C"]


def _samples():
    rng = np.random.default_rng(11)
    return {"A": list(rng.normal(10, 3, 4)),
            "B": list(rng.normal(20, 4, 4)),
            "C": list(rng.normal(30, 2, 4))}


def _expected_ci(samples):
    return sorted(round(float(DV._ci_half_width(np.asarray(samples[g], float))), 3) for g in _GROUPS)


def _box_halfwidths(ax):
    hw = []
    for c in ax.containers:
        if isinstance(c, ErrorbarContainer):
            _, _, bars = c
            for seg in bars:
                for (x0, y0), (x1, y1) in seg.get_segments():
                    hw.append(abs(y1 - y0) / 2)
    return sorted(round(v, 3) for v in hw)


def _bar_halfwidths(ax):
    hw = []
    for ln in ax.lines:
        yd = np.asarray(ln.get_ydata(), float)
        if np.isfinite(yd).sum() >= 2:
            hw.append(round((np.nanmax(yd) - np.nanmin(yd)) / 2, 3))
    return sorted(hw)


def test_box_ci_equals_helper():
    s = _samples()
    fig, ax = plt.subplots()
    DV.plot_box(_GROUPS, s, ax=ax, save_plot=False, show_points=False,
                show_significance_letters=False, show_error_bars=True, error_type="ci")
    got = _box_halfwidths(ax)
    plt.close(fig)
    assert got == _expected_ci(s), f"box ci {got} != helper {_expected_ci(s)}"


def test_bar_ci_equals_helper():
    s = _samples()
    fig, ax = plt.subplots()
    DV.plot_bar(_GROUPS, s, ax=ax, save_plot=False, show_points=False,
                show_significance_letters=False, error_type="ci")
    got = _bar_halfwidths(ax)
    plt.close(fig)
    assert got == _expected_ci(s), f"bar ci {got} != helper {_expected_ci(s)}"


# ---------- (3) cross-type consistency on identical data ----------

def test_ci_cross_type_consistency_bar_box():
    s = _samples()
    fig, ax = plt.subplots()
    DV.plot_bar(_GROUPS, s, ax=ax, save_plot=False, show_points=False,
                show_significance_letters=False, error_type="ci")
    bar = _bar_halfwidths(ax)
    plt.close(fig)
    fig, ax = plt.subplots()
    DV.plot_box(_GROUPS, s, ax=ax, save_plot=False, show_points=False,
                show_significance_letters=False, show_error_bars=True, error_type="ci")
    box = _box_halfwidths(ax)
    plt.close(fig)
    assert bar == box, f"same data, same 'ci' -> bar {bar} must equal box {box}"


# ---------- sd/se must remain untouched (regression guard) ----------

@pytest.mark.parametrize("et", ["sd", "se"])
def test_sd_se_unchanged(et):
    s = _samples()
    def hand(g):
        v = np.asarray(s[g], float)
        return np.std(v, ddof=1) if et == "sd" else np.std(v, ddof=1) / np.sqrt(len(v))
    want = sorted(round(hand(g), 3) for g in _GROUPS)
    fig, ax = plt.subplots()
    DV.plot_box(_GROUPS, s, ax=ax, save_plot=False, show_points=False,
                show_significance_letters=False, show_error_bars=True, error_type=et)
    got = _box_halfwidths(ax)
    plt.close(fig)
    assert got == want, f"{et} must be unchanged: {got} != {want}"
