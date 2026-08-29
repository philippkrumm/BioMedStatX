"""A number that cannot exist is not a weak result.

F is a ratio of mean squares and both are non-negative, so F < 0 cannot happen;
partial eta squared is a ratio of sums of squares and lies in [0, 1]. A factorial
design with an empty cell leaves the interaction unestimable, and pingouin
returns a negative sum of squares for it: measured at F = -3.07 with partial eta
squared = -0.28 on a 2x2 layout missing one cell, delivered to the report with
p = 1.0 printed beside it.

Both values are finite, which is why the existing non-finite safety net let them
through. "Not a number" and "a number that cannot be" are different failures and
only the first was caught.

The design that produces it is ordinary, not exotic -- a factorial experiment
where one combination was never run is routine, so this is a path a real user
walks rather than a fuzzer curiosity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Root conftest.py puts src/ on sys.path and forces headless Qt.
from analysis.statisticaltester import StatisticalTester as T


def _incomplete_factorial():
    """A 2x2 layout with one cell never run. Reproduced from fuzz seed 1338."""
    rng = np.random.default_rng(1338)
    rows = []
    for factor_a, factor_b, mean in (("A0", "B1", 2.0), ("A1", "B0", -0.3), ("A1", "B1", 2.2)):
        for _ in range(6):
            rows.append({"FacA": factor_a, "FacB": factor_b,
                         "Val": float(mean + rng.normal(0, 1.0))})
    return pd.DataFrame(rows)


def test_the_engine_still_produces_the_impossible_number():
    """Guard the fixture: if this stops holding, the tests below prove nothing."""
    result = T._run_two_way_anova(_incomplete_factorial(), dv="Val",
                                  between=["FacA", "FacB"], alpha=0.05)
    interaction = (result.get("interactions") or [None])[0]
    assert interaction is not None, "the fixture no longer reaches the interaction"
    assert float(interaction["F"]) < 0, (
        f"the fixture no longer produces a negative F (got {interaction['F']}), "
        f"so it is not exercising the guard"
    )
    assert np.isfinite(float(interaction["F"])), (
        "a non-finite F would be caught by the older net, not by this one"
    )


def test_an_unestimable_interaction_is_blocked():
    result = T._run_two_way_anova(_incomplete_factorial(), dv="Val",
                                  between=["FacA", "FacB"], alpha=0.05)
    blocked = T.nonfinite_block(result)
    assert blocked is not None, "the impossible interaction was let through"
    assert blocked["block_code"] == "NOT_ESTIMABLE"
    assert blocked["statistic"] is None and blocked["p_value"] is None
    reason = blocked["block_reason"]
    assert "negative F statistic" in reason, reason
    assert "empty cell" in reason, reason
    assert "mixed model" in reason, reason


def test_the_defect_is_reported_once_not_once_per_copy():
    """The headline is a copy of the interaction row; naming it twice is noise."""
    result = T._run_two_way_anova(_incomplete_factorial(), dv="Val",
                                  between=["FacA", "FacB"], alpha=0.05)
    quantities = T._impossible_quantities(result)
    assert len(quantities) == len(set(quantities)), quantities
    assert sum("negative F" in q for q in quantities) == 1, quantities


def test_a_complete_design_is_left_alone():
    rng = np.random.default_rng(7)
    rows = [{"FacA": a, "FacB": b, "Val": float(rng.normal(i + j, 1.0))}
            for i, a in enumerate(("A0", "A1")) for j, b in enumerate(("B0", "B1"))
            for _ in range(6)]
    result = T._run_two_way_anova(pd.DataFrame(rows), dv="Val",
                                  between=["FacA", "FacB"], alpha=0.05)
    assert T._impossible_quantities(result) == []
    assert T.nonfinite_block(result) is None


@pytest.mark.parametrize("kind,value", [
    # Signed by definition -- a negative value is the ordinary case and must not
    # be mistaken for an impossible one.
    ("cohen_d", -1.4),
    ("hedges_g", -0.8),
    ("r", -0.62),
    ("rank_biserial_r", -0.3),
    # Bias-corrected, so legitimately negative in small samples.
    ("omega_squared", -0.05),
])
def test_a_signed_effect_size_is_not_called_impossible(kind, value):
    assert T._impossible_quantities(
        {"test": "t-test", "effect_size": value, "effect_size_type": kind}) == []


@pytest.mark.parametrize("value", [-0.28, 1.4])
def test_a_ratio_effect_size_outside_its_range_is_impossible(value):
    quantities = T._impossible_quantities(
        {"test": "Two-Way ANOVA", "effect_size": value,
         "effect_size_type": "partial_eta_squared"})
    assert len(quantities) == 1 and "outside [0, 1]" in quantities[0]


def test_a_negative_t_statistic_is_not_blocked():
    """The headline statistic's sign is never judged on its own."""
    assert T.nonfinite_block(
        {"test": "Paired t-test", "statistic": -3.4, "p_value": 0.004}) is None


def test_the_older_non_finite_net_still_answers_first():
    blocked = T.nonfinite_block(
        {"test": "Linear Mixed Model", "statistic": float("-inf"), "p_value": None})
    assert blocked is not None and blocked["block_code"] == "NON_FINITE_RESULT"
