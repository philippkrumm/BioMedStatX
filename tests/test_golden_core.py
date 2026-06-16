"""Gated golden-reference correctness suite.

Runs the APP's statistical tests against frozen reference values (computed once
by validation/generate_golden.py via scipy/pingouin) and asserts they match
within tolerance. This freezes the expected statistics/p-values/df so any
regression in the app's test routing, corrections, or degrees of freedom fails
CI — the core trust guarantee for a scientific tool.

Reference values live in tests/golden/references.json (source-agnostic schema:
every numeric field is optional and only compared when present, so later R
dumps for nparLD/emmeans slot in unchanged).
"""
import json
import math
import os

import pytest

from analysis.statisticaltester import StatisticalTester

_REF = os.path.join(os.path.dirname(__file__), "golden", "references.json")
with open(_REF) as _fh:
    _DATA = json.load(_fh)
_CASES = _DATA["cases"]


@pytest.fixture(scope="module", autouse=True)
def _qt_and_dialogs():
    """A significant multi-group result triggers the post-hoc path, which can
    construct Qt dialogs. Provide a QApplication and neutralize every modal so
    the gated run never blocks or aborts."""
    try:
        from PyQt5.QtWidgets import QApplication, QDialog
    except Exception:
        yield
        return
    app = QApplication.instance() or QApplication([])
    QDialog.exec_ = lambda self, *a, **k: 0
    QDialog.exec = lambda self, *a, **k: 0
    try:
        from analysis.statisticaltester import UIDialogManager
        UIDialogManager.select_transformation_dialog = staticmethod(lambda *a, **k: "log10")
        UIDialogManager.select_posthoc_test_dialog = staticmethod(lambda *a, **k: "tukey")
        for name in ("select_nonparametric_posthoc_dialog", "select_control_group_dialog",
                     "select_custom_pairs_dialog"):
            setattr(UIDialogManager, name, staticmethod(lambda *a, **k: None))
    except Exception:
        pass
    yield app


def _assert_close(label, actual, expected, tol):
    assert actual is not None, f"{label}: app returned None, expected {expected}"
    assert math.isfinite(actual), f"{label}: app value not finite ({actual})"
    assert abs(actual - expected) <= tol, (
        f"{label}: app={actual!r} vs reference={expected!r} (tol={tol})"
    )


@pytest.mark.parametrize("case", _CASES, ids=[c["id"] for c in _CASES])
def test_golden_reference(case):
    samples = {k: list(v) for k, v in case["samples"].items()}
    groups = list(samples.keys())
    inv = case["invocation"]
    result = StatisticalTester.perform_statistical_test(
        groups, samples, samples,
        dependent=inv["dependent"],
        test_recommendation=inv["test_recommendation"],
        test_info=None,
    )

    test_name = str(result.get("test", "")).lower()
    assert case["expected_test_contains"] in test_name, (
        f"{case['id']}: expected a '{case['expected_test_contains']}' test, got '{result.get('test')}'"
    )

    exp = case["expected"]
    tol = exp.get("tol", {})
    _assert_close("statistic", result.get("statistic"), exp["statistic"], tol.get("statistic", 1e-6))
    _assert_close("p_value", result.get("p_value"), exp["p_value"], tol.get("p_value", 1e-6))
    if "df_num" in exp:
        _assert_close("df_num", result.get("df1"), exp["df_num"], 1e-6)
    if "df_den" in exp:
        _assert_close("df_den", result.get("df2"), exp["df_den"], 1e-6)
