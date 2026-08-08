"""End-to-end routing test for the ambiguous binary-outcome confirmation.

Covers the footgun and its fix together:

  * A binary outcome coded 1/2 (not 0/1) with a single continuous predictor used
    to fall through to a Pearson correlation with no warning -- a
    plausible-but-wrong result. It is now flagged "maybe_binary" and the user is
    asked. Confirm -> logistic_regression; decline -> correlation.

  * The step-4 "continuous primary factor -> correlation" upgrade must NOT revert
    a confirmed binary outcome back to correlation. Logistic regression of a
    binary DV on a continuous predictor is not a correlation. Without the
    outcome_type guard the confirmation would be silently undone.

Exercises the real ``_ap_build_analysis_context`` with a minimal fake app,
monkeypatching only the modal confirmation prompt.
"""
import numpy as np
import pandas as pd

from autopilot.statistical_analyzer_autopilot_pipeline import _ap_build_analysis_context


class _FakeBucket:
    def __init__(self, columns=None):
        self._columns = list(columns or [])

    def get_assigned_columns(self):
        return list(self._columns)


class _FakeToggle:
    def __init__(self, checked=False):
        self._checked = checked

    def isChecked(self):
        return self._checked


class _FakeApp:
    """Minimal stand-in exposing only what _ap_build_analysis_context touches."""

    def __init__(self, df, dv, factor, confirm_returns):
        self.df = df
        self.dv_bucket = _FakeBucket([dv])
        self.factor1_bucket = _FakeBucket([factor])
        self.factor2_bucket = _FakeBucket([])
        self.subject_bucket = _FakeBucket([])
        self.covariates_bucket = _FakeBucket([])
        self.multi_mode_button = _FakeToggle(False)
        self.analysis_selected_groups = []
        self._confirm_calls = []
        self._confirm_returns = confirm_returns

    def _confirm_binary_outcome(self, dv_col, values):
        self._confirm_calls.append((dv_col, list(values)))
        return self._confirm_returns


def _make_df():
    # 20 rows: continuous predictor Age, binary outcome coded 1/2 (NOT 0/1).
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "Age": rng.integers(30, 80, size=20).astype(float) + rng.random(20),
        "Responder": [1, 2] * 10,
    })


def test_maybe_binary_confirmed_routes_to_logistic_not_correlation():
    app = _FakeApp(_make_df(), dv="Responder", factor="Age", confirm_returns=True)
    ctx = _ap_build_analysis_context(app)
    # The prompt fired for the ambiguous 1/2 outcome ...
    assert app._confirm_calls == [("Responder", [1, 2])]
    # ... and confirming binary routes to logistic, surviving the step-4
    # continuous-factor upgrade (which would otherwise force correlation).
    assert ctx["outcome_type"] == "binary"
    assert ctx["inferred_test"] == "logistic_regression"


def test_maybe_binary_declined_routes_to_correlation():
    app = _FakeApp(_make_df(), dv="Responder", factor="Age", confirm_returns=False)
    ctx = _ap_build_analysis_context(app)
    assert app._confirm_calls == [("Responder", [1, 2])]
    # Declining keeps it continuous -> the continuous Age predictor makes it a
    # correlation, exactly as before (no silent logistic).
    assert ctx.get("outcome_type") != "binary"
    assert ctx["inferred_test"] == "correlation"


def test_unambiguous_01_outcome_does_not_prompt():
    df = pd.DataFrame({
        "Age": np.linspace(30, 80, 20),
        "Died": [0, 1] * 10,
    })
    app = _FakeApp(df, dv="Died", factor="Age", confirm_returns=True)
    ctx = _ap_build_analysis_context(app)
    # 0/1 is unambiguous -> no prompt, straight to logistic.
    assert app._confirm_calls == []
    assert ctx["outcome_type"] == "binary"
    assert ctx["inferred_test"] == "logistic_regression"
