"""Numeric coverage for the Welch-branch Hedges' g.

Pre-2.0 audit gap: Hedges' g had only label-canonicalization and
small/medium/large bucket tests (test_effect_sizes.py) — no test asserted a
computed g value against an external reference. This locks the J-corrected
Welch effect size to pingouin.compute_effsize(eftype='hedges'), the same
convention the app documents (J = 1 - 3/(4*(n1+n2)-9) applied to the
pooled-SD Cohen's d).
"""
import numpy as np
import pytest

pg = pytest.importorskip("pingouin")

from analysis.statisticaltester import StatisticalTester  # noqa: E402


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 13])
def test_welch_branch_hedges_g_matches_pingouin(seed):
    rng = np.random.default_rng(seed)
    # Unequal n + unequal spread -> the Welch (equal_var=False) branch, which is
    # the one that reports Hedges' g.
    x = rng.normal(5.0, 1.0, 12)
    y = rng.normal(6.0, 1.3, 15)

    results = {}
    StatisticalTester._independent_ttest(
        results, "A", "B", x, y, alpha=0.05, equal_var=False
    )

    # The Welch branch must label the effect Hedges' g (not Cohen's d).
    assert results["effect_size_type"] == "hedges_g"

    expected = pg.compute_effsize(x, y, eftype="hedges")
    assert results["effect_size"] == pytest.approx(expected, abs=1e-9)


def test_pooled_branch_reports_cohen_d_not_hedges():
    # Equal-variance branch is Cohen's d (no J correction, honest label).
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 1.0, 12)
    y = rng.normal(6.0, 1.0, 12)

    results = {}
    StatisticalTester._independent_ttest(
        results, "A", "B", x, y, alpha=0.05, equal_var=True
    )
    assert results["effect_size_type"] == "cohen_d"
