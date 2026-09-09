"""Gated golden test: Brunner-Langer ATS vs R nparLD (frozen).

The ANOVA-Type Statistic has no reliable Python reference, so R's nparLD is the
canonical oracle. References (dataset + per-effect ATS/df/p) were frozen once by
validation/generate_golden_nparld.py; this test runs the APP on the same data and
asserts a match — no R at test time.

ATS and df1 are checked for every effect; p only where compare_p is true. The
between (whole-plot) p is excluded because the app intentionally uses a
Satterthwaite finite df2 (F-test) there instead of nparLD's chi-square.
"""
import json
import math
import os

import pandas as pd
import pytest

from analysis.nonparametricanovas import perform_brunner_langer_ats

_REF = os.path.join(os.path.dirname(__file__), "golden", "references_nparld.json")
with open(_REF) as _fh:
    _DATA = json.load(_fh)


@pytest.fixture(scope="module")
def app_table():
    df = pd.DataFrame(_DATA["data"])
    d = _DATA["design"]
    result = perform_brunner_langer_ats(df, d["dv"], d["between"], d["within"], d["subject"], alpha=0.05)
    tab = result.get("anova_table")
    assert tab is not None, "app returned no anova_table"
    return {str(row["Source"]): row for _, row in tab.iterrows()}


@pytest.mark.parametrize("effect", _DATA["effects"], ids=[e["source"] for e in _DATA["effects"]])
def test_nparld_ats_matches(effect, app_table):
    src = effect["source"]
    assert src in app_table, f"effect '{src}' missing from app result ({list(app_table)})"
    row = app_table[src]
    tol = effect.get("tol", {})

    def close(label, actual, expected, t):
        assert actual is not None and math.isfinite(actual), f"{src}.{label}: bad value {actual}"
        assert abs(float(actual) - expected) <= t, f"{src}.{label}: app={actual} vs nparLD={expected} (tol={t})"

    close("ATS", row.get("ATS"), effect["ATS"], tol.get("ATS", 1e-3))
    close("df1", row.get("df1"), effect["df1"], tol.get("df1", 1e-3))
    if effect.get("compare_p"):
        close("p", row.get("p-value"), effect["p"], tol.get("p", 1e-3))
