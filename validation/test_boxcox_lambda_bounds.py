"""
test_boxcox_lambda_bounds.py — Box-Cox lambda divergence guard.

Background (audit 2026-06): on extremely right-skewed assay data (e.g.
luciferase signal-to-background), boxcox_normmax can return |lambda| >> 1
(optimizer divergence). Forcing that lambda potentiates variance instead of
stabilizing it, destroying homoscedasticity for downstream parametric tests
and producing transformed values ~1e16.

Correct behaviour: validate the ML lambda against a strict interval [-3, 3].
If it falls outside, REJECT it and hard-fall-back to lambda = 0 (natural log,
the established standard for such data), and raise a transform_warning flag.
Clamping to the boundary (lambda = 3) is methodologically invalid and must NOT
happen.
"""

import numpy as np
import pytest

from statistical_testing.validators import bounded_boxcox_lambda


def _extreme_right_skew(n=300, seed=0):
    """Left-skewed-with-ceiling data that drives boxcox_normmax past lambda=3
    (optimizer divergence), the same pathology luciferase assays trigger."""
    rng = np.random.default_rng(seed)
    return rng.beta(8, 1.2, n) * 100 + 1


def test_divergent_lambda_falls_back_to_log_with_warning():
    data = _extreme_right_skew()
    # Sanity: unbounded ML lambda really is out of [-3, 3] for this data
    from scipy.stats import boxcox_normmax
    raw = float(boxcox_normmax(data[data > 0]))
    assert abs(raw) > 3, f"fixture not extreme enough, raw lambda={raw}"

    lam, reverted = bounded_boxcox_lambda(data)
    assert reverted is True
    assert lam == 0.0, "out-of-bounds lambda must hard-fall-back to 0 (log), not clamp to a boundary"


def test_well_behaved_lambda_is_kept():
    rng = np.random.default_rng(3)
    data = rng.normal(50.0, 5.0, 200)  # roughly symmetric -> lambda near 1, in bounds
    lam, reverted = bounded_boxcox_lambda(data)
    assert reverted is False
    assert -3.0 <= lam <= 3.0


def test_never_clamps_to_boundary():
    """A divergent fit must never silently produce lambda == ±3."""
    data = _extreme_right_skew(seed=11)
    lam, reverted = bounded_boxcox_lambda(data)
    assert lam not in (3.0, -3.0)


def test_degenerate_input_falls_back_safely():
    lam, reverted = bounded_boxcox_lambda(np.array([1.0, 1.0, 1.0]))
    assert reverted is True
    assert lam == 0.0


def test_custom_bounds_respected():
    data = _extreme_right_skew(seed=7)
    from scipy.stats import boxcox_normmax
    raw = float(boxcox_normmax(data[data > 0]))
    # widen bounds beyond the raw value -> should now be accepted
    lam, reverted = bounded_boxcox_lambda(data, bounds=(-abs(raw) - 1, abs(raw) + 1))
    assert reverted is False
    assert lam == pytest.approx(raw, abs=1e-6)


def test_report_surfaces_transform_warning():
    """report_summaries must pass a transform_warning into context.assumptions,
    from either results-level or nested test_info."""
    from export.report_summaries import _SummariesMixin

    msg = "Box-Cox lambda diverged; fell back to log."
    top = _SummariesMixin._build_assumption_summary({"transform_warning": msg})
    assert top["transform_warning"] == msg

    nested = _SummariesMixin._build_assumption_summary({"test_info": {"transform_warning": msg}})
    assert nested["transform_warning"] == msg

    absent = _SummariesMixin._build_assumption_summary({})
    assert absent["transform_warning"] is None
