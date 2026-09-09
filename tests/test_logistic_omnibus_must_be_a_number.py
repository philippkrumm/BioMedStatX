"""A test whose own statistic is not a number has produced no result.

The logistic path already guards identification, but it reads the coefficient
standard errors. Those can come back finite while the omnibus does not: a Firth
fit on quasi-separated data with a collinear covariate overflowed in the link
function and returned ``statistic = nan``, ``p_value = nan`` with finite
standard errors -- and was reported as ``converged = True``.

Firth is the penalised method that exists to survive separation. When even it
returns no number, the design is not weakly supported, it is unidentified, and
the result has to say so rather than let the report describe it as an outcome.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Root conftest.py puts src/ on sys.path and forces headless Qt.
from analysis.clinical_models import LogisticRegressionModel
from analysis.statisticaltester import StatisticalTester


def _separated_frame():
    """The frame that raised this, written out rather than referenced.

    Perfect separation -- every group A row is one outcome level and every group
    B row the other -- a covariate that is the group indicator again, so the
    design is rank-deficient, and missing outcomes scattered through it. Twelve
    usable rows once the NaNs drop. Reproduced from fuzz seed 20 and kept here
    verbatim so this test never depends on the fuzzer or on a lucky draw.

    The outsized outcome level is what the seed produced and is left as it is:
    the model maps the two distinct levels to 0/1 either way, and the point of
    the fixture is the separation, not the magnitude.
    """
    return pd.DataFrame({
        "Grp": ["A"] * 8 + ["B"] * 8,
        "Cov": [0.0] * 8 + [1.0] * 8,
        "Outcome": [1e160, 1e160, 1e160, np.nan, 1e160, 1e160, 1e160, 0.0,
                    np.nan, 0.0, 0.0, 0.0, 0.0, np.nan, 0.0, np.nan],
    })


def _fit(df):
    """Driven the way the pipeline drives it (statisticaltester._run_logistic_regression)."""
    model = LogisticRegressionModel()
    model.fit(df, dv="Outcome", predictors=["Grp"], covariates=["Cov"])
    return model.as_results_dict()


def test_a_non_finite_omnibus_is_not_a_converged_fit():
    """Whatever the numbers do, these two may never disagree."""
    result = _fit(_separated_frame())
    statistic = result.get("statistic")
    p_value = result.get("p_value")
    usable = all(
        isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)
        for v in (statistic, p_value)
    )
    assert not usable, (
        "the fixture no longer produces a non-finite omnibus, so this test is "
        "not exercising the guard any more"
    )

    assert result.get("converged") is False, (
        f"statistic={statistic!r} p_value={p_value!r} were reported as a "
        f"converged fit"
    )
    warnings = " ".join(str(w) for w in (result.get("warnings") or []))
    assert "no usable statistic" in warnings, warnings
    assert "separation" in warnings, warnings


def test_the_warning_does_not_recommend_the_method_already_in_use():
    """Firth cannot be the remedy for a Firth fit.

    The data-health layer says "Consider Firth regression as an alternative" on
    separated data, which is sound advice in general and useless on the path
    that is already penalised. The convergence warning has to know which model
    it is describing.
    """
    result = _fit(_separated_frame())
    assert result.get("converged") is False
    assert result.get("model_variant") == "Firth Penalized Likelihood", (
        "this fixture is meant to reach the penalised path"
    )
    warnings = " ".join(str(w) for w in (result.get("warnings") or []))
    assert "already applied" in warnings, warnings


def test_a_well_identified_fit_is_left_alone():
    """The guard must not start failing ordinary logistic regressions."""
    rng = np.random.default_rng(11)
    n = 80
    cov = rng.normal(0, 1, size=n)
    group = ["A" if v > 0 else "B" for v in rng.normal(0, 1, size=n)]
    logit = 0.8 * cov + np.array([0.6 if g == "A" else -0.6 for g in group])
    outcome = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({"Grp": group, "Cov": cov, "Outcome": outcome})

    result = _fit(df)
    assert result.get("converged") is True, result.get("warnings")
    assert np.isfinite(float(result["p_value"]))
    assert np.isfinite(float(result["statistic"]))
    warnings = " ".join(str(w) for w in (result.get("warnings") or []))
    assert "no usable statistic" not in warnings, warnings


def test_an_unidentified_fit_is_blocked_not_reported():
    """No test means no result, and no AUC either.

    The report used to show "No result" beside AUC 0.9167, a ROC curve and a
    calibration plot, all computed from the same unidentified model -- a
    quotable number from a fit that produced no test. Blocking stops at the
    data-quality gate the way the rest of the pipeline does.
    """
    df = _separated_frame()
    blocked = StatisticalTester._run_logistic_regression(
        df, dv="Outcome", between=["Grp"], covariates=["Cov"])

    assert blocked.get("blocked") is True, blocked.get("test")
    assert blocked.get("block_code") == "LOGISTIC_UNIDENTIFIED"
    assert blocked.get("p_value") is None
    assert blocked.get("statistic") is None
    assert "effect_size" not in blocked or blocked.get("effect_size") is None
    reason = str(blocked.get("block_reason") or "")
    assert "separation" in reason and "Firth" in reason, reason
    assert "Check the group sizes" in reason, reason


def test_a_usable_fit_is_not_blocked():
    rng = np.random.default_rng(11)
    n = 80
    cov = rng.normal(0, 1, size=n)
    group = ["A" if v > 0 else "B" for v in rng.normal(0, 1, size=n)]
    logit = 0.8 * cov + np.array([0.6 if g == "A" else -0.6 for g in group])
    outcome = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({"Grp": group, "Cov": cov, "Outcome": outcome})

    result = StatisticalTester._run_logistic_regression(
        df, dv="Outcome", between=["Grp"], covariates=["Cov"])
    assert not result.get("blocked")
    assert np.isfinite(float(result["p_value"]))


def test_both_entry_points_block_through_the_same_helper():
    """Two call sites, one decision.

    ``analysis_core`` runs the clinical branch and ``statisticaltester`` has its
    own entry; a guard added to one of them and not the other is the shape of
    half-applied fix this codebase keeps rediscovering, so assert that neither
    builds its own block.
    """
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent / "src" / "analysis"
    for name in ("analysis_core.py", "statisticaltester.py"):
        source = (root / name).read_text(encoding="utf-8")
        assert "blocked_unidentified_logistic" in source, (
            f"{name} does not go through the shared logistic block"
        )
    # And the reason exists in exactly one place.
    tester = (root / "statisticaltester.py").read_text(encoding="utf-8")
    assert tester.count("LOGISTIC_UNIDENTIFIED") == 1, (
        "the block code is spelled out more than once"
    )
