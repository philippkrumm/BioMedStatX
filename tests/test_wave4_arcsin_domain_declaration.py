"""Wave-4 follow-up: arcsin_sqrt requires an explicit data-domain declaration
(proportion 0-1 vs percent 0-100), and hard-rejects data that does not match.

arcsin(sqrt(p)) is variance-stabilizing ONLY for true proportions
(Var(p_hat)=p(1-p)/n). A pure value-range guess would still wave through data
that happens to land in range but is not a proportion — the same silent
misapplication class as the original per-group-rescale bug. So the user DECLARES
the domain, and values that violate it are rejected outright: an error, no
transform, no silent fallback.

A single central `validate_arcsin_domain` is called by both application sites
(classic assumption_checks and the advanced TransformationEngine). This is new
functionality, so it must not be two diverging copies.

Validation is gated on a declared type being present; programmatic callers that
don't declare one keep the prior behaviour (so the BLOCKER-1 global-rescale
tests, which use undeclared data, stay green unchanged).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from statistical_testing.validators import validate_arcsin_domain, GroupValidationError
from statistical_testing.engines.transformation import TransformationEngine


# ---- unit: the central validator ----

def test_proportion_declaration_rejects_values_above_one():
    with pytest.raises(GroupValidationError):
        validate_arcsin_domain([0.2, 0.5, 1.5], "proportion")


def test_percent_declaration_rejects_values_above_hundred():
    with pytest.raises(GroupValidationError):
        validate_arcsin_domain([10.0, 50.0, 150.0], "percent")


def test_matching_declaration_passes():
    # true proportions and true percents in range -> no raise
    assert validate_arcsin_domain([0.0, 0.5, 1.0], "proportion") is None
    assert validate_arcsin_domain([0.0, 50.0, 100.0], "percent") is None


def test_negative_values_rejected_for_both():
    with pytest.raises(GroupValidationError):
        validate_arcsin_domain([-0.1, 0.5], "proportion")


# ---- integration: the advanced TransformationEngine hard-rejects ----

def _run_engine(values, declared):
    df = pd.DataFrame({"Value": values, "Cond": ["a", "b"] * (len(values) // 2)})
    payload = {
        "mode": "advanced_transformation",
        "df": df, "dv": "Value", "test": "repeated_measures_anova",
        "between": None, "within": ["Cond"],
        "transformed_samples": {},
        "test_info": {"transformation": "arcsin_sqrt", "arcsin_declared_type": declared},
    }
    return TransformationEngine().execute(payload)


def test_engine_hard_rejects_proportion_with_out_of_range_values():
    res = _run_engine([0.2, 0.5, 1.5, 2.0], "proportion")
    err = (res.metadata or {}).get("error")
    assert err, "expected a hard reject error, got none"
    assert "arcsin" in err.lower() and ("proportion" in err.lower() or "outside" in err.lower())


def test_engine_allows_declared_proportion_in_range():
    res = _run_engine([0.1, 0.4, 0.6, 0.9], "proportion")
    assert not (res.metadata or {}).get("error"), (res.metadata or {}).get("error")


# ---- integration: the classic assumption_checks path hard-rejects ----

def test_classic_path_hard_rejects_out_of_range_declaration(monkeypatch):
    """assumption_checks must call the same validator and raise (no transform,
    no silent fallback) when the declaration does not match the data."""
    import statistical_testing.assumption_checks as ac

    class _Stub:
        @staticmethod
        def select_transformation_dialog(*a, **k):
            return "arcsin_sqrt"
        @staticmethod
        def select_arcsin_domain_type(*a, **k):
            return "proportion"       # declared proportion, but data are > 1
    monkeypatch.setattr(ac, "_get_ui_dialog_manager", lambda: _Stub())

    rng = np.random.default_rng(3)
    # strongly non-normal, all > 1 -> transform is offered, declared proportion,
    # values violate [0,1] -> hard reject
    samples = {g: list(c + rng.lognormal(1.0, 1.2, 20)) for g, c in [("A", 5.0), ("B", 30.0)]}
    with pytest.raises(GroupValidationError):
        ac.AssumptionCheckEngine.check_normality_and_variance(
            ["A", "B"], samples, model_type="oneway", formula="Value ~ C(Group)"
        )


def test_cancel_domain_dialog_aborts_analysis(monkeypatch):
    """Cancelling the arcsin domain dialog aborts the whole analysis, consistent
    with the transformation and post-hoc dialogs (Cancel == abort everywhere).
    Previously this was the reference "drop + continue non-parametric" path; it
    was aligned when the main transformation dialog gained an explicit "continue
    without transformation" option and Cancel became a hard abort."""
    from statistical_testing.validators import AnalysisCancelledError
    import statistical_testing.assumption_checks as ac

    class _Stub:
        @staticmethod
        def select_transformation_dialog(*a, **k):
            return "arcsin_sqrt"
        @staticmethod
        def select_arcsin_domain_type(*a, **k):
            return None       # user cancels the domain declaration
    monkeypatch.setattr(ac, "_get_ui_dialog_manager", lambda: _Stub())

    rng = np.random.default_rng(3)
    samples = {g: list(c + rng.lognormal(1.0, 1.2, 20)) for g, c in [("A", 5.0), ("B", 30.0)]}
    with pytest.raises(AnalysisCancelledError):
        ac.AssumptionCheckEngine.check_normality_and_variance(
            ["A", "B"], samples, model_type="oneway", formula="Value ~ C(Group)"
        )


@pytest.mark.parametrize("model_type", ["oneway", "rm", "twoway"])
def test_cancel_domain_aborts_on_classic_and_advanced_paths(monkeypatch, model_type):
    """The abort must hold on the classic (oneway) AND the advanced (rm, twoway)
    routes -- all consume the same assumption_checks writer, so each is pinned
    rather than assumed architecturally."""
    from statistical_testing.validators import AnalysisCancelledError
    import statistical_testing.assumption_checks as ac

    class _Stub:
        @staticmethod
        def select_transformation_dialog(*a, **k):
            return "arcsin_sqrt"
        @staticmethod
        def select_arcsin_domain_type(*a, **k):
            return None
    monkeypatch.setattr(ac, "_get_ui_dialog_manager", lambda: _Stub())

    rng = np.random.default_rng(3)
    if model_type == "twoway":
        groups = [f"FA={a}, FB={b}" for a in ("1", "2") for b in ("x", "y")]
        formula = "Value ~ C(FA)*C(FB)"
    else:
        groups = ["T1", "T2", "T3"]
        formula = "Value ~ C(Group)"
    samples = {g: list(20.0 + rng.lognormal(1.0, 1.2, 20)) for g in groups}

    with pytest.raises(AnalysisCancelledError):
        ac.AssumptionCheckEngine.check_normality_and_variance(
            groups, samples, model_type=model_type, formula=formula
        )
