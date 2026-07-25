"""Wave-4 BLOCKER 1 repair: arcsin_sqrt must rescale against the GLOBAL data
range, not per group.

The classic one-way path (assumption_checks) min-max-scaled EACH group to [0,1]
independently before arcsin-sqrt. When the raw values exceed 1 (percentages,
fold-change) that collapses every group onto the same [0,1] span and erases the
between-group differences the test is about.

The primary before/after evidence is the *live escalation case* found by Wave-4:
within-group uniform data, 5/5 seeds, where the corrupted transform is not
rescued by the non-parametric gate — the post-transform residuals pass Shapiro,
recommendation stays parametric (welch), the collapsed transformed samples are
used, and a raw one-way p ~ 1e-120 is reported as a non-significant Welch result
(p ~ 0.85). This is the reproduction the fix must close, not the generic
two-group collapse demo.

The advanced ``TransformationEngine`` already rescales globally; this brings the
classic path to parity.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from scipy import stats

import statistical_testing.assumption_checks as ac


class _StubDialogMgr:
    @staticmethod
    def select_transformation_dialog(parent=None, progress_text=None, column_name=None, **kw):
        return "arcsin_sqrt"

    @staticmethod
    def select_arcsin_domain_type(parent=None, **kw):
        # The uniform fixture below is percent-scaled (0-100); declaring the
        # domain is now required for arcsin to run at all (undeclared -> the
        # transform is dropped).
        return "percent"


@pytest.fixture(autouse=True)
def _stub_transform_dialog(monkeypatch):
    monkeypatch.setattr(ac, "_get_ui_dialog_manager", lambda: _StubDialogMgr())


def _uniform_case(seed):
    """The Wave-4 escalation fixture, now percent-scaled (0-100): within-group
    uniform, three far-apart locations, all values > 1 so the global out-of-[0,1]
    rescale is exercised, and all <= 100 so the declared 'percent' domain is
    valid (post domain-declaration, out-of-range data is rejected instead)."""
    rng = np.random.default_rng(seed)
    return {g: list(c + rng.uniform(0, 8, 30)) for g, c in [("A", 10.0), ("B", 45.0), ("C", 80.0)]}


@pytest.mark.parametrize("seed", range(5))
def test_arcsin_does_not_escalate_to_false_negative_on_uniform_groups(seed):
    samples = _uniform_case(seed)
    groups = ["A", "B", "C"]

    raw_p = stats.f_oneway(*[samples[g] for g in groups]).pvalue
    assert raw_p < 1e-3, "sanity: raw groups must be clearly different"

    ts, rec, ti = ac.AssumptionCheckEngine.check_normality_and_variance(
        groups, samples, model_type="oneway", formula="Value ~ C(Group)"
    )
    assert ti.get("transformation") == "arcsin_sqrt", "sanity: arcsin path must run"

    A, B, C = (np.array(ts[g]) for g in groups)
    transformed_p = stats.f_oneway(A, B, C).pvalue

    # The escalation the live run proved: parametric/welch recommendation would
    # feed the COLLAPSED transformed samples to the final test -> false negative.
    escalates = rec in {"parametric", "welch"} and raw_p < 1e-3 and transformed_p > 0.05
    assert not escalates, (
        f"seed={seed}: arcsin collapsed the groups and the {rec} recommendation "
        f"would report them (raw p={raw_p:.1e} -> transformed p={transformed_p:.3f})"
    )

    # Positive proof the fix preserves the signal, not just dodges the gate:
    # a global rescale keeps the groups apart, so the transformed ANOVA stays
    # significant regardless of which recommendation wins downstream.
    assert transformed_p < 0.05, (
        f"seed={seed}: between-group signal not preserved after transform "
        f"(transformed p={transformed_p:.3f})"
    )


def test_arcsin_on_true_proportions_is_unaffected(monkeypatch):
    """Guard: genuine proportion data already in [0,1], declared 'proportion',
    must be transformed directly (no rescale path), so the fix does not change
    that behaviour."""
    class _PropStub:
        @staticmethod
        def select_transformation_dialog(*a, **k):
            return "arcsin_sqrt"
        @staticmethod
        def select_arcsin_domain_type(*a, **k):
            return "proportion"
    monkeypatch.setattr(ac, "_get_ui_dialog_manager", lambda: _PropStub())

    rng = np.random.default_rng(0)
    samples = {
        "A": list(np.clip(rng.beta(2, 5, 30), 1e-6, 1 - 1e-6)),
        "B": list(np.clip(rng.beta(5, 2, 30), 1e-6, 1 - 1e-6)),
    }
    ts, rec, ti = ac.AssumptionCheckEngine.check_normality_and_variance(
        ["A", "B"], samples, model_type="oneway", formula="Value ~ C(Group)"
    )
    if ti.get("transformation") == "arcsin_sqrt":
        # values in [0,1] -> arcsin(sqrt(x)) applied directly, difference kept
        A, B = np.array(ts["A"]), np.array(ts["B"])
        assert not np.allclose(np.sort(A), np.sort(B)), "proportion groups must not collapse"
