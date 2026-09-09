"""Transformation dialog: Cancel aborts; "skip" continues non-parametric.

The dialog offers an explicit "Continue without transformation (use non-parametric
test)" option ("skip"), distinct from Cancel:
  * Cancel (None) aborts the whole analysis (AnalysisCancelledError), consistent
    with the post-hoc dialog -- and never silently applies a log10 the user did
    not pick nor mislabels the report as "Transformation: log10".
  * "skip" drops the transform, keeps the raw data, and routes non-normal
    residuals to the non-parametric test.
"""
import numpy as np
import pytest

from statistical_testing.validators import AnalysisCancelledError


def _nonnormal_two_groups():
    # Strongly right-skewed (lognormal) -> residuals non-normal -> the
    # transformation dialog is offered.
    rng = np.random.default_rng(3)
    return {g: list(c + rng.lognormal(1.0, 1.2, 20)) for g, c in [("A", 5.0), ("B", 30.0)]}


def _formula(model_type):
    return "Value ~ C(FactorA) * C(FactorB)" if model_type == "twoway" else "Value ~ C(Group)"


def _stub(monkeypatch, returns):
    import statistical_testing.assumption_checks as ac

    class _Stub:
        @staticmethod
        def select_transformation_dialog(*a, **k):
            return returns
    monkeypatch.setattr(ac, "_get_ui_dialog_manager", lambda: _Stub())
    return ac


@pytest.mark.parametrize("model_type", ["oneway", "rm", "twoway"])
def test_cancel_transform_aborts(monkeypatch, model_type):
    ac = _stub(monkeypatch, None)  # user cancels
    with pytest.raises(AnalysisCancelledError):
        ac.AssumptionCheckEngine.check_normality_and_variance(
            ["A", "B"], _nonnormal_two_groups(), model_type=model_type,
            formula=_formula(model_type),
        )


@pytest.mark.parametrize("model_type", ["oneway", "rm", "twoway"])
def test_skip_transform_routes_nonparametric(monkeypatch, model_type):
    ac = _stub(monkeypatch, "skip")  # explicit "continue without transformation"
    samples = _nonnormal_two_groups()
    ts, rec, ti = ac.AssumptionCheckEngine.check_normality_and_variance(
        ["A", "B"], samples, model_type=model_type, formula=_formula(model_type),
    )
    assert ti.get("transformation") is None, f"false transform label: {ti.get('transformation')!r}"
    assert rec == "non_parametric", f"skip must route nonparametric, got {rec!r}"
    assert np.allclose(np.sort(ts["A"]), np.sort(samples["A"]))
    assert np.allclose(np.sort(ts["B"]), np.sort(samples["B"]))


def test_skip_transform_records_a_note(monkeypatch):
    ac = _stub(monkeypatch, "skip")
    _, _, ti = ac.AssumptionCheckEngine.check_normality_and_variance(
        ["A", "B"], _nonnormal_two_groups(), model_type="oneway", formula="Value ~ C(Group)",
    )
    notes = " ".join(str(n) for n in (ti.get("validation_notes") or []))
    assert "without" in notes.lower() or "no transformation" in notes.lower(), \
        f"no skip note recorded; notes={ti.get('validation_notes')!r}"
