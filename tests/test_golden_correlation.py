"""Golden-reference correctness suite for correlation (Pearson / Spearman).

Runs the app's CorrelationModel against frozen R cor.test values
(tests/golden/references_correlation.json, produced by
validation/generate_golden_correlation.py). The oracle uses exact=FALSE so R's
Spearman p-value is the large-sample t-approximation -- the same method the app
computes via scipy.spearmanr. R's default exact Spearman would diverge from the
app (proven: |dp| 4e-3 at n=12), so the golden must pin the app's actual method.

Complementary to test_correlation_ci.py, which validates the Fisher-z /
Bonett-Wright confidence-interval SE -- a quantity R's cor.test cannot even
produce for Spearman. This file validates r and p; that file validates the CI.
"""
import json
import math
import os

import pandas as pd
import pytest

from analysis.correlation_models import CorrelationModel

_REF = os.path.join(os.path.dirname(__file__), "golden", "references_correlation.json")
with open(_REF) as _fh:
    _CASES = json.load(_fh)["cases"]


def _assert_close(label, actual, expected, tol):
    assert actual is not None, f"{label}: app returned None, expected {expected}"
    assert math.isfinite(actual), f"{label}: app value not finite ({actual})"
    assert abs(actual - expected) <= tol, (
        f"{label}: app={actual!r} vs R reference={expected!r} (tol={tol})"
    )


@pytest.mark.parametrize("case", _CASES, ids=[c["id"] for c in _CASES])
def test_golden_correlation(case):
    df = pd.DataFrame({"X": case["x"], "Y": case["y"]})
    out = CorrelationModel().fit(df, "X", "Y", method=case["method"]).as_results_dict()

    assert case["method"] in str(out.get("method", "")).lower(), (
        f"{case['id']}: expected a '{case['method']}' fit, got '{out.get('method')}'"
    )

    exp = case["expected"]
    tol = exp.get("tol", {})
    _assert_close("r", out.get("r"), exp["r"], tol.get("r", 1e-6))
    _assert_close("p_value", out.get("p_value"), exp["p_value"], tol.get("p_value", 1e-6))
