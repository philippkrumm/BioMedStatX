"""The permutation loop must not rebuild its models 30000 times.

Freedman-Lane permutes the RESIDUALS, so the two design matrices are identical
on every permutation and only ``y`` changes. The loop nonetheless rebuilt both
from their formula strings each time -- a patsy parse and a categorical
re-encode of the whole frame, twice, 5000 times, for each of three effects.

Measured on a 28-row two-way design: 91 seconds. The analysis runs on the UI
thread, so that is the window frozen, and it stayed invisible for as long as
two-way designs could not reach this fallback at all.

Two properties are pinned here, because speed alone is not the point:
the loop must not fit a model per permutation, and the numbers must not move.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Root conftest.py puts src/ on sys.path and forces headless Qt.
from analysis import nonparametricanovas as npa

# Fixed design, fixed seed: this test is about arithmetic, not about sampling.
#
# The p-value is stored as its NUMERATOR over n_permutations + 1, because that is
# what it is -- (#{F_perm >= F_obs} + 1) / 5001 -- and an integer count cannot be
# lost to transcription the way a rounded decimal can. Recorded from the
# implementation that refit a model per permutation.
PERMUTATIONS_PLUS_ONE = 5001
EXPECTED = {
    "FacA": (1.527435449610, 1173),
    "FacB": (4.647441547405, 167),
    "FacA x FacB": (0.581246183050, 2261),
}


def _frame():
    rng = np.random.default_rng(11)
    return pd.DataFrame([
        {"FacA": a, "FacB": b, "Val": float(np.exp(rng.normal(i + j, 0.8)))}
        for i, a in enumerate(("A0", "A1"))
        for j, b in enumerate(("B0", "B1"))
        for _ in range(7)
    ])


def _effects(result):
    found = {f["factor"]: (f["F"], f["p_value"]) for f in result.get("factors") or []}
    for interaction in result.get("interactions") or []:
        found[" x ".join(interaction["factors"])] = (
            interaction["F"], interaction["p_value"])
    return found


def test_the_numbers_are_exactly_what_the_formula_loop_produced():
    """Recorded from the implementation that refit per permutation.

    The rewrite reuses the same seeded permutations in the same order, so this
    is an equality check rather than a tolerance one: a change here means the
    answer moved, not that the sampling did.
    """
    found = _effects(npa.perform_freedman_lane_test(
        _frame(), dv="Val", factor_a="FacA", factor_b="FacB"))
    assert set(found) == set(EXPECTED), found
    for effect, (f_value, numerator) in EXPECTED.items():
        assert found[effect][0] == pytest.approx(f_value, abs=1e-9), effect
        assert found[effect][1] == pytest.approx(
            numerator / PERMUTATIONS_PLUS_ONE, abs=1e-15), effect


def test_no_model_is_fitted_inside_the_permutation_loop(monkeypatch):
    """Six fits: the observed full and reduced model for each of three effects.

    Counted rather than timed, so the check does not depend on how busy the
    machine is. Before the rewrite this was 6 + 2 * 5000 * 3 = 30006.
    """
    import statsmodels.formula.api as smf

    calls = {"n": 0}
    original = smf.ols

    def counting_ols(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(npa.smf, "ols", counting_ols)
    npa.perform_freedman_lane_test(_frame(), dv="Val",
                                   factor_a="FacA", factor_b="FacB")
    assert calls["n"] <= 8, (
        f"{calls['n']} models were fitted; the permutation loop is building one "
        f"per iteration again"
    )


def test_the_projection_spans_a_rank_deficient_design():
    """An empty cell is exactly the case this fallback exists to survive.

    QR would not do here: its Q spans the column space only at full column rank,
    and a factorial layout with a missing cell loses rank by construction. The
    basis has to come from the singular values.
    """
    design = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    ])  # third column duplicates the second
    basis = npa._column_space(design)
    assert basis.shape[1] == 2, basis.shape

    # Projection reproduces ordinary least squares on the same design.
    y = np.array([1.0, 4.0, 2.0, 5.0])
    rss_projected = float(y @ y) - float(np.square(basis.T @ y).sum())
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    rss_direct = float(np.square(y - design @ coefficients).sum())
    assert rss_projected == pytest.approx(rss_direct, abs=1e-12)
