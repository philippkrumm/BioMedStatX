"""
test_mwu_power.py — Regression test for Mann-Whitney-U power reporting.

Background (audit 2026-06): the MWU branch fed the rank-based effect size
r = |Z|/sqrt(N) into statsmodels TTestIndPower, which expects Cohen's d.
The `r * 0.955` "correction" (3/pi ARE) does not convert r to d; it produced
a dimensionally meaningless number. There is no clean closed-form post-hoc
power for the Wilcoxon/MWU test, so the value is dropped (set to None) while
the valid rank-biserial effect size r is still reported.

The parametric independent t-test power (computed from a genuine Cohen's d)
must remain unaffected.
"""

import numpy as np
import pytest

from analysis.statisticaltester import StatisticalTester


def _mwu_result():
    rng = np.random.default_rng(1)
    # Skewed, clearly different groups -> MWU path with a real effect
    data1 = list(rng.lognormal(0.0, 0.5, 18))
    data2 = list(rng.lognormal(0.8, 0.5, 20))
    results = {}
    return StatisticalTester._mannwhitney_test(results, "A", "B", data1, data2, alpha=0.05)


def test_mwu_reports_valid_effect_size_r():
    res = _mwu_result()
    comp = res["pairwise_comparisons"][0]
    assert comp["effect_size_type"] == "r"
    assert 0.0 <= comp["effect_size"] <= 1.0


def test_mwu_power_is_none_not_fabricated():
    res = _mwu_result()
    assert res.get("power") is None
    assert res["pairwise_comparisons"][0].get("power") is None


def test_parametric_ttest_power_still_computed():
    """Guard: the legitimate Cohen's-d power path must keep returning a value."""
    rng = np.random.default_rng(2)
    data1 = list(rng.normal(5.0, 1.0, 20))
    data2 = list(rng.normal(7.0, 1.0, 20))
    results = {}
    res = StatisticalTester._independent_ttest(results, "A", "B", data1, data2, alpha=0.05)
    assert res.get("power") is not None
    assert 0.0 <= res["power"] <= 1.0
