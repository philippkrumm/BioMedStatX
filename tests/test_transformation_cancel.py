"""Cancelling the transformation dialog must not silently apply log10.

Old bug: select_transformation_dialog returning None (Cancel) fell through to
`transformation_type = "log10"`, so the analysis force-applied a log10 transform
the user declined AND the report then read "Transformation: log10" -- a false
label on the actually-computed result. Correct behaviour (mirrors the arcsin
domain-cancel path): drop the transform, keep the raw data, and let non-normal
residuals route to the non-parametric test.
"""
import numpy as np
import pytest


def _nonnormal_two_groups():
    # Strongly right-skewed (lognormal) -> residuals non-normal -> the
    # transformation dialog is offered.
    rng = np.random.default_rng(3)
    return {g: list(c + rng.lognormal(1.0, 1.2, 20)) for g, c in [("A", 5.0), ("B", 30.0)]}


@pytest.mark.parametrize("model_type", ["oneway", "rm", "twoway"])
def test_cancel_transform_drops_it_and_routes_nonparametric(monkeypatch, model_type):
    import statistical_testing.assumption_checks as ac

    class _Stub:
        @staticmethod
        def select_transformation_dialog(*a, **k):
            return None  # user cancels the transformation dialog
    monkeypatch.setattr(ac, "_get_ui_dialog_manager", lambda: _Stub())

    samples = _nonnormal_two_groups()
    if model_type == "twoway":
        formula = "Value ~ C(FactorA) * C(FactorB)"
    else:
        formula = "Value ~ C(Group)"

    ts, rec, ti = ac.AssumptionCheckEngine.check_normality_and_variance(
        ["A", "B"], samples, model_type=model_type, formula=formula
    )

    # No stale / false transform label -- must be None, never "log10".
    assert ti.get("transformation") is None, f"false transform label: {ti.get('transformation')!r}"
    # Non-normal data with the transform declined -> non-parametric route.
    assert rec == "non_parametric", f"cancelled transform must route nonparametric, got {rec!r}"
    # The raw data was passed through unchanged (no log10 applied).
    assert np.allclose(np.sort(ts["A"]), np.sort(samples["A"]))
    assert np.allclose(np.sort(ts["B"]), np.sort(samples["B"]))


def test_cancel_transform_records_a_note(monkeypatch):
    import statistical_testing.assumption_checks as ac

    class _Stub:
        @staticmethod
        def select_transformation_dialog(*a, **k):
            return None
    monkeypatch.setattr(ac, "_get_ui_dialog_manager", lambda: _Stub())

    samples = _nonnormal_two_groups()
    _, _, ti = ac.AssumptionCheckEngine.check_normality_and_variance(
        ["A", "B"], samples, model_type="oneway", formula="Value ~ C(Group)"
    )
    notes = " ".join(str(n) for n in (ti.get("validation_notes") or []))
    assert "cancel" in notes.lower(), f"no cancellation note recorded; notes={ti.get('validation_notes')!r}"
