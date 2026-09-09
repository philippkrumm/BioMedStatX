"""method='exact' with zeros/ties returns the p-value of no defined test.

scipy's exact null distribution for Wilcoxon and Mann-Whitney assumes no
zero differences and no ties. The app hardcoded method='exact' from sample
size alone (Wilcoxon: len<=25; MWU: n1+n2<20), with no zero/tie guard, and
scipy raises no warning. So a paired design with unchanged subjects, or an
ordinal scale with ties, shipped an exact p-value that is simply wrong -- the
audit measured it 4x conservative to 24-64x anti-conservative depending on the
data.

scipy's own method='auto' downgrades to the asymptotic p-value exactly when
zeros/ties are present, and keeps exact otherwise. The guard here reproduces
that downgrade explicitly while preserving the existing size thresholds for
clean data, so every currently-correct result is byte-identical.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from scipy import stats as ss

from analysis.statisticaltester import StatisticalTester


# ---- Wilcoxon ----

def _wilcoxon_zero_fixture():
    """Audit fixture: 14 paired ordinal scores, 5 subjects unchanged."""
    before = [4, 3, 5, 4, 2, 5, 3, 4, 5, 3, 4, 2, 5, 4]
    after = [2, 3, 3, 2, 2, 4, 1, 4, 3, 3, 2, 2, 4, 1]
    return before, after


def test_premise_exact_would_be_chosen_and_is_wrong():
    """Guard on the guard: confirm the exact branch is what fires today, and
    that it differs from the asymptotic p-value. If these ever coincide the
    fixture has stopped exercising the bug."""
    before, after = _wilcoxon_zero_fixture()
    a, b = np.array(before, float), np.array(after, float)
    assert len(a) <= 25, "fixture must fall in the exact-by-size range"
    assert int(np.sum(a - b == 0)) > 0, "fixture must contain zero differences"
    p_exact = ss.wilcoxon(a, b, zero_method="pratt", method="exact").pvalue
    p_approx = ss.wilcoxon(a, b, zero_method="pratt", method="approx").pvalue
    assert not np.isclose(p_exact, p_approx), "fixture no longer distinguishes the branches"


def test_wilcoxon_with_zeros_downgrades_to_asymptotic():
    before, after = _wilcoxon_zero_fixture()
    res = {}
    out = StatisticalTester._wilcoxon_test(res, "Before", "After", before, after, 0.05)

    a, b = np.array(before, float), np.array(after, float)
    p_approx = ss.wilcoxon(a, b, zero_method="pratt", method="approx").pvalue
    p_exact = ss.wilcoxon(a, b, zero_method="pratt", method="exact").pvalue

    assert out["p_value"] == pytest.approx(float(p_approx), rel=1e-12)
    assert not np.isclose(out["p_value"], p_exact), "still using the invalid exact p-value"
    # the downgrade is surfaced, not silent
    assert any("exact" in str(w).lower() for w in (out.get("warnings") or [])), (
        f"no note explaining the method downgrade: {out.get('warnings')}"
    )


def test_wilcoxon_clean_small_sample_still_exact():
    """Positive control: no zeros, all |differences| distinct, small n -> exact,
    byte-identical to before."""
    a = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    b = [1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0, 36.0]  # diffs 9,17,24,30,35,39,42,44 — all distinct
    diffs = np.array(a) - np.array(b)
    assert np.all(diffs != 0) and len(np.unique(np.abs(diffs))) == len(diffs)  # fixture is clean
    res = {}
    out = StatisticalTester._wilcoxon_test(res, "A", "B", a, b, 0.05)
    p_exact = ss.wilcoxon(np.array(a), np.array(b), zero_method="pratt", method="exact").pvalue
    assert out["p_value"] == pytest.approx(float(p_exact), rel=1e-12)
    assert not (out.get("warnings") or []), "clean data should not trigger a downgrade note"


# ---- Mann-Whitney ----

def _mwu_tie_fixture():
    """Small samples with ties in the pooled data."""
    a = [1, 2, 3, 4, 5, 5, 6]
    b = [2, 3, 4, 5, 6, 7, 8]
    return a, b


def test_premise_mwu_exact_would_be_chosen_and_is_wrong():
    a, b = _mwu_tie_fixture()
    from statistical_testing.validators import MIN_N_SMALL
    assert len(a) + len(b) < MIN_N_SMALL, "fixture must fall in the exact-by-size range"
    pooled = np.array(a + b)
    assert len(pooled) != len(np.unique(pooled)), "fixture must contain ties"
    p_exact = ss.mannwhitneyu(a, b, alternative="two-sided", method="exact").pvalue
    p_asym = ss.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic").pvalue
    assert not np.isclose(p_exact, p_asym), "fixture no longer distinguishes the branches"


def test_mwu_with_ties_downgrades_to_asymptotic():
    a, b = _mwu_tie_fixture()
    res = {}
    out = StatisticalTester._mannwhitney_test(res, "A", "B", a, b, 0.05)
    p_asym = ss.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic").pvalue
    p_exact = ss.mannwhitneyu(a, b, alternative="two-sided", method="exact").pvalue
    assert out["p_value"] == pytest.approx(float(p_asym), rel=1e-12)
    assert not np.isclose(out["p_value"], p_exact)
    assert "asymptotic" in out["test"].lower(), f"label still claims exact: {out['test']!r}"


def test_mwu_clean_small_sample_still_exact():
    """Positive control: no ties, small n -> exact, unchanged."""
    a = [1.1, 2.2, 3.3, 4.4]
    b = [5.5, 6.6, 7.7, 8.8]
    res = {}
    out = StatisticalTester._mannwhitney_test(res, "A", "B", a, b, 0.05)
    p_exact = ss.mannwhitneyu(np.array(a), np.array(b), alternative="two-sided", method="exact").pvalue
    assert out["p_value"] == pytest.approx(float(p_exact), rel=1e-12)
    assert "exact" in out["test"].lower()
