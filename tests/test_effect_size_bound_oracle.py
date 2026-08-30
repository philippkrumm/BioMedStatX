"""The |d| > 50 smell applies to independent groups, not to a paired d_z.

Cohen's d has no upper bound. For independent groups a value past 50 is a
useful smell for a scaling bug; for a paired design the reported quantity is
d_z, which divides by the SD of the DIFFERENCES, so pairs that track each other
make it arbitrarily large with nothing wrong. A real seed paired three values
against three that were 13.1 higher, with the difference varying by 0.23:
d_z = -55.93 was exactly right, and the check called it a finding.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from fuzzing.oracles import check_result


def _result(test, es, es_type="cohen_d"):
    return {"test": test, "statistic": -96.87, "p_value": 0.0001,
            "effect_size": es, "effect_size_type": es_type}


@pytest.mark.parametrize("test", [
    "Paired t-test",
    "Wilcoxon signed-rank test",
    "Pairwise Wilcoxon / Mann-Whitney U (within / between simple effects)",
])
def test_a_paired_design_may_report_a_huge_d(test):
    assert check_result(_result(test, -55.93)) == []


def test_an_independent_design_still_reports_one():
    """Or the check stops catching the scaling bug it exists for."""
    violations = check_result(_result("Independent t-test", -55.93))
    assert any("implausibly large" in v for v in violations)


def test_an_ordinary_paired_effect_is_still_clean():
    assert check_result(_result("Paired t-test", -1.2)) == []


def test_the_bounded_effect_sizes_are_untouched_by_the_exemption():
    """The exemption returns early, so it must not skip a later check.

    partial eta squared is judged in the branch above this one; a paired label
    with an out-of-range eta must still be caught.
    """
    violations = check_result(_result("Paired t-test", 1.4, "partial_eta_squared"))
    assert any("outside [0, 1]" in v for v in violations)
