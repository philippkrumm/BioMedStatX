"""
test_posthoc_correctness.py — Regression tests for post-hoc p-value correctness.

Validates GamesHowellTest and DunnettTest in analysis.posthoc_core against
independent reference implementations:

  - Games-Howell : pingouin.pairwise_gameshowell (studentized-range based)
  - Dunnett      : scipy.stats.dunnett (exact multivariate-t joint distribution)

Background (audit 2026-06): GamesHowellTest returned raw Welch t-test p-values
(missing the studentized-range step, ~3-4x anti-conservative), and DunnettTest
applied Holm-Šidák on top of already FWER-adjusted Dunnett p-values
(double correction, conservative).
"""

import numpy as np
import pandas as pd
import pytest

from analysis.posthoc_core import GamesHowellTest, DunnettTest


def _make_groups(seed, sizes):
    """Synthetic groups with heterogeneous means/variances (GH's target case)."""
    rng = np.random.default_rng(seed)
    means = [0.0, 1.2, 2.0, 0.5]
    sds = [1.0, 3.0, 0.4, 1.5]
    names = ["A", "B", "C", "D"]
    return {
        name: rng.normal(m, s, n)
        for name, m, s, n in zip(names, means, sds, sizes)
    }


def _pairs_dict(result):
    """Map (group1, group2) -> p_value from a PostHocAnalyzer result dict."""
    comps = result.get("pairwise_comparisons") or []
    assert comps, f"No comparisons returned: {result.get('error')}"
    return {(c["group1"], c["group2"]): c["p_value"] for c in comps}


def _lookup(pmap, a, b):
    p = pmap.get((a, b), pmap.get((b, a)))
    assert p is not None, f"Missing comparison {a} vs {b}"
    return p


@pytest.mark.parametrize(
    "seed,sizes",
    [
        (3, [12, 12, 12, 12]),   # balanced
        (42, [8, 15, 10, 12]),   # unbalanced
    ],
)
def test_games_howell_matches_pingouin(seed, sizes):
    """GH p-values must match pingouin's studentized-range implementation."""
    pg = pytest.importorskip("pingouin")

    groups = _make_groups(seed, sizes)
    samples = {k: list(v) for k, v in groups.items()}
    names = list(samples)

    result = GamesHowellTest.perform_test(names, samples, alpha=0.05)
    app = _pairs_dict(result)

    long = pd.DataFrame(
        {
            "v": np.concatenate([groups[g] for g in names]),
            "g": np.concatenate([[g] * len(groups[g]) for g in names]),
        }
    )
    ref = pg.pairwise_gameshowell(data=long, dv="v", between="g")

    for _, row in ref.iterrows():
        a, b = str(row["A"]), str(row["B"])
        p_app = _lookup(app, a, b)
        assert p_app == pytest.approx(row["pval"], abs=1e-4), (
            f"{a} vs {b}: app p={p_app:.6f}, pingouin GH p={row['pval']:.6f}"
        )


@pytest.mark.parametrize("seed", [7, 99])
def test_dunnett_matches_scipy(seed):
    """Dunnett p-values must match scipy's joint-distribution reference (no double correction)."""
    from scipy import stats as scipy_stats

    rng = np.random.default_rng(seed)
    groups = {
        "Ctrl": rng.normal(0.0, 1.0, 10),
        "T1": rng.normal(0.9, 1.0, 14),
        "T2": rng.normal(1.3, 1.0, 8),
        "T3": rng.normal(0.2, 1.0, 12),
    }
    samples = {k: list(v) for k, v in groups.items()}
    names = list(samples)
    treatment_names = [g for g in names if g != "Ctrl"]

    result = DunnettTest.perform_test(names, samples, "Ctrl", alpha=0.05)
    app = _pairs_dict(result)

    ref = scipy_stats.dunnett(
        *[groups[g] for g in treatment_names], control=groups["Ctrl"]
    )

    for i, name in enumerate(treatment_names):
        p_app = _lookup(app, name, "Ctrl")
        assert p_app == pytest.approx(ref.pvalue[i], abs=1e-3), (
            f"{name} vs Ctrl: app p={p_app:.6f}, scipy Dunnett p={ref.pvalue[i]:.6f}"
        )


def test_games_howell_two_groups_reduces_sensibly():
    """k=2 edge case: GH with two groups must still produce a valid p in (0, 1]."""
    rng = np.random.default_rng(5)
    samples = {"A": list(rng.normal(0, 1, 10)), "B": list(rng.normal(1, 2, 12))}
    result = GamesHowellTest.perform_test(["A", "B"], samples, alpha=0.05)
    pmap = _pairs_dict(result)
    p = _lookup(pmap, "A", "B")
    assert 0.0 < p <= 1.0


def _dunnett_comps(seed):
    rng = np.random.default_rng(seed)
    groups = {
        "Ctrl": rng.normal(0.0, 1.0, 10),
        "T1": rng.normal(0.9, 1.0, 14),
        "T2": rng.normal(1.4, 1.0, 8),
        "T3": rng.normal(0.2, 1.0, 12),
    }
    samples = {k: list(v) for k, v in groups.items()}
    result = DunnettTest.perform_test(list(samples), samples, "Ctrl", alpha=0.05)
    comps = {c["group1"]: c for c in result["pairwise_comparisons"]}
    return groups, comps


@pytest.mark.parametrize("seed", [7, 99, 21])
def test_dunnett_ci_matches_scipy_simultaneous(seed):
    """Dunnett CI must be the joint simultaneous interval from scipy, not an ad-hoc per-pair one."""
    from scipy import stats as scipy_stats

    groups, comps = _dunnett_comps(seed)
    treatment_names = [g for g in groups if g != "Ctrl"]
    ref = scipy_stats.dunnett(*[groups[g] for g in treatment_names], control=groups["Ctrl"])
    ci = ref.confidence_interval(confidence_level=0.95)

    for i, name in enumerate(treatment_names):
        lo, hi = comps[name]["confidence_interval"]
        assert lo == pytest.approx(ci.low[i], abs=1e-3), f"{name}: lo {lo} != scipy {ci.low[i]}"
        assert hi == pytest.approx(ci.high[i], abs=1e-3), f"{name}: hi {hi} != scipy {ci.high[i]}"


@pytest.mark.parametrize("seed", [7, 99, 21, 5])
def test_dunnett_ci_significance_consistent(seed):
    """Significant comparison (p < alpha) must have a CI that excludes 0, and vice versa."""
    _, comps = _dunnett_comps(seed)
    for name, c in comps.items():
        lo, hi = c["confidence_interval"]
        excludes_zero = (lo > 0) or (hi < 0)
        significant = c["p_value"] < 0.05
        assert excludes_zero == significant, (
            f"{name}: p={c['p_value']:.4f} sig={significant} but CI=({lo:.3f},{hi:.3f})"
        )
