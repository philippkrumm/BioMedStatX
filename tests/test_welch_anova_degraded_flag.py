"""_perform_welch_anova's manual F/df computation silently falls back to the standard,
non-robust f_oneway result on any exception (e.g. ZeroDivisionError from a zero-variance
group), relabeled under the same welch_f_statistic/welch_p_value keys - a caller has no way to
tell the "Welch" result isn't actually variance-robust. Fix: add an explicit
welch_calculation_degraded flag on that fallback path.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from statistical_testing.mixed_assumptions import MixedAnovaAssumptionEngine


def test_welch_anova_flags_degraded_fallback_on_zero_variance_group():
    # Group "A" has zero variance -> ZeroDivisionError in the manual Welch
    # weight calculation (weights = [n/var for n, var in ...]) -> falls back
    # to f_oneway, currently with no indication of degradation.
    group_data = [
        [1.0, 1.0, 1.0, 1.0],
        [2.0, 4.0, 6.0, 8.0],
        [1.0, 3.0, 5.0, 7.0],
    ]
    group_labels = ["A", "B", "C"]

    result = MixedAnovaAssumptionEngine._perform_welch_anova(group_data, group_labels, "Value", "Group")

    assert result.get("welch_calculation_degraded") is True, (
        f"expected the degraded-fallback flag to be set, got keys: {sorted(result.keys())}"
    )
    assert result["welch_f_statistic"] == pytest.approx(result["standard_f_statistic"]), (
        "the degraded fallback IS the standard f_oneway result - both fields should match "
        "exactly when this flag is True"
    )


def test_welch_anova_does_not_flag_degraded_on_normal_data():
    group_data = [
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 4.0, 6.0, 8.0],
        [1.0, 3.0, 5.0, 7.0],
    ]
    group_labels = ["A", "B", "C"]

    result = MixedAnovaAssumptionEngine._perform_welch_anova(group_data, group_labels, "Value", "Group")

    assert result.get("welch_calculation_degraded", False) is False
