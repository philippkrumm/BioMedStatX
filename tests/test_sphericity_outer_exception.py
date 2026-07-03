"""When BOTH the primary sphericity test (pg.sphericity) and its inner
fallback (_extract_sphericity_from_anova_table) fail, execution falls to the
outer except in _perform_comprehensive_sphericity_test. Per CHANGELOG.md:
"When sphericity cannot be formally tested ... the Greenhouse-Geisser
correction is now applied by default." The inner fallback already honors
this; the outer except did not — it used the uncorrected p-value instead,
silently reintroducing the pre-v2.0 behavior the changelog says was fixed.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import pytest

import analysis.statisticaltester as st_module
from analysis.statisticaltester import StatisticalTester


def test_outer_exception_still_applies_gg_correction_by_default(monkeypatch):
    def _boom_pg_module():
        class _Boom:
            @staticmethod
            def sphericity(*a, **kw):
                raise RuntimeError("pg.sphericity boom")
        return _Boom()

    def _boom_extract(*a, **kw):
        raise RuntimeError("anova table extraction boom")

    monkeypatch.setattr(st_module, "get_pingouin_module", _boom_pg_module)
    monkeypatch.setattr(StatisticalTester, "_extract_sphericity_from_anova_table",
                         staticmethod(_boom_extract))

    # 3 factor levels needed: with k<=2, the function short-circuits to
    # "sphericity always met" before ever reaching pg.sphericity or the
    # outer except this test targets.
    df = pd.DataFrame({
        "dv": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5, 1.2, 2.2, 3.2],
        "subject": ["s1", "s1", "s1", "s2", "s2", "s2", "s3", "s3", "s3"],
        "factor": ["a", "b", "c", "a", "b", "c", "a", "b", "c"],
    })
    row = pd.Series({"DF": 2.0, "eps": 0.75, "p_GG_corr": 0.03, "F": 5.2})
    error_row = pd.Series({"DF": 18.0})

    result = StatisticalTester._perform_comprehensive_sphericity_test(
        df, "dv", "subject", "factor", aov=None, row=row, error_row=error_row
    )

    assert result["correction_used"] == "Greenhouse-Geisser (ε = 0.750)", (
        f"outer exception must still apply the documented conservative GG "
        f"default, got correction_used={result.get('correction_used')!r}"
    )
    assert result["corrected_p_value"] == 0.03
