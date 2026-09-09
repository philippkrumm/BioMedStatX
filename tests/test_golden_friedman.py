"""Golden-reference correctness suite for the Friedman test.

Runs the app's perform_friedman_test against frozen R friedman.test values
(tests/golden/references_friedman.json, produced by
validation/generate_golden_friedman.py) and asserts the Chi2 statistic and
p-value match within tolerance.

R friedman.test and scipy.friedmanchisquare (which the app calls) apply the same
tie correction -- measured, not assumed: |d chi2| = 0.0 (continuous), 2.3e-08
(tied), 2.3e-07 (heavy ties). So both tied and continuous cases validate against
R at 1e-4.

This file also absorbs the one complementary assertion from the now-removed
standalone validation/validate_friedman.py: the post-hoc comparison-count
structure (k*(k-1)/2 pairwise comparisons on a significant omnibus) and the
Friedman model_class. Its numeric chi2/p check (formerly vs scipy) is superseded
here by the stronger R oracle.
"""
import json
import math
import os

import pandas as pd
import pytest

from analysis.nonparametricanovas import perform_friedman_test

_REF = os.path.join(os.path.dirname(__file__), "golden", "references_friedman.json")
with open(_REF) as _fh:
    _CASES = json.load(_fh)["cases"]


def _long_df(matrix):
    rows = []
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            rows.append({"subject": f"S{i + 1}", "time": f"T{j + 1}", "score": float(val)})
    return pd.DataFrame(rows)


def _assert_close(label, actual, expected, tol):
    assert actual is not None, f"{label}: app returned None, expected {expected}"
    assert math.isfinite(actual), f"{label}: app value not finite ({actual})"
    assert abs(actual - expected) <= tol, (
        f"{label}: app={actual!r} vs R reference={expected!r} (tol={tol})"
    )


@pytest.mark.parametrize("case", _CASES, ids=[c["id"] for c in _CASES])
def test_golden_friedman(case):
    res = perform_friedman_test(_long_df(case["matrix"]), dv="score",
                                within_factor="time", subject_col="subject", alpha=0.05)

    assert res.get("model_class") == "Friedman", f"wrong model_class: {res.get('model_class')}"
    assert not res.get("error"), f"error: {res.get('error')}"

    tbl = res["anova_table"]
    chi2_col = next(c for c in tbl.columns if c.lower() in ("chi2", "statistic", "f", "wald_chi2"))
    p_col = next(c for c in tbl.columns if c.lower().replace("-", "_") in ("p_unc", "p_value", "pval"))
    chi2 = float(tbl[chi2_col].iloc[0])
    p_val = float(tbl[p_col].iloc[0])

    exp = case["expected"]
    tol = exp.get("tol", {})
    _assert_close("statistic", chi2, exp["statistic"], tol.get("statistic", 1e-4))
    _assert_close("p_value", p_val, exp["p_value"], tol.get("p_value", 1e-4))

    # structure absorbed from validate_friedman.py: post-hoc count on a
    # significant omnibus must be every unordered pair of time points.
    if p_val < 0.05:
        posthoc = res.get("pairwise_comparisons", [])
        assert len(posthoc) == exp["expected_comparisons"], (
            f"{case['id']}: significant omnibus -> expected {exp['expected_comparisons']} "
            f"comparisons, got {len(posthoc)}"
        )
