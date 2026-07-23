"""Wave-4 BLOCKER 2 (S7) repair: the regression Box-Cox optimizer must obey the
same contract as the shared ``bounded_boxcox_lambda``.

``_optimize_boxcox_for_regression`` diverged from the shared helper on three
counts, all of which this test pins:

1. bounds ±2 instead of ±3, and it CLAMPED to the boundary — the shared helper's
   own docstring calls boundary clamping "methodologically invalid and is never
   done". A genuine optimum in (2, 3] was pinned to 2.0.
2. an out-of-range optimum must be REJECTED and fall back to lambda=0 (natural
   log), never clamped.
3. the failure / no-valid-data fallback was lambda=1 (identity, no transform);
   the shared contract falls back to lambda=0 (log). Opposite behaviour on the
   same failed input.

Named sub-decision for this fix: unify boundary + failure semantics onto the
shared contract (reject-out-of-[-3,3] -> log; failure -> log, not identity).
The RSS-of-OLS-residuals objective (correct for a regression Box-Cox) is kept.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from analysis.correlation_models import _optimize_boxcox_for_regression


def _design(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(1.0, 5.0, n)
    X = np.column_stack([np.ones(n), x])
    return rng, x, X


def _y_with_optimum(x, rng, inv_lambda):
    """y such that boxcox(y, inv_lambda) is linear in x -> RSS-optimal lambda≈inv_lambda."""
    lin = 3.0 + 2.0 * x + rng.normal(0.0, 0.02, x.size)
    return np.power(np.clip(lin, 1e-9, None), 1.0 / inv_lambda)


def test_optimum_in_2_to_3_is_accepted_not_clamped_to_2():
    rng, x, X = _design()
    y = _y_with_optimum(x, rng, inv_lambda=2.5)   # true optimum ≈ 2.5, was clamped to 2.0
    lam = _optimize_boxcox_for_regression(y, X)
    assert lam > 2.05, f"lambda still pinned near the old ±2 boundary: {lam}"
    assert lam <= 3.0 + 1e-6, f"lambda outside the [-3,3] validity interval: {lam}"
    assert lam == pytest.approx(2.5, abs=0.3), f"lambda not near the true optimum 2.5: {lam}"


def test_optimum_beyond_3_is_rejected_to_log_not_clamped():
    rng, x, X = _design(seed=1)
    y = _y_with_optimum(x, rng, inv_lambda=5.0)   # true optimum ≈ 5 > 3 -> reject -> log
    lam = _optimize_boxcox_for_regression(y, X)
    assert lam == 0.0, f"out-of-bounds optimum must fall back to log (0.0), got {lam}"


def test_no_valid_data_falls_back_to_log_not_identity():
    _, _, X = _design(seed=2)
    y = np.array([-1.0, -2.0, np.nan, 0.0] + [-3.0] * 196)  # nothing > 0
    lam = _optimize_boxcox_for_regression(y, X)
    assert lam == 0.0, f"failure fallback must be log (0.0), not identity (1.0); got {lam}"
