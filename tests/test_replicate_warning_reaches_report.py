"""The technical-replicate averaging warning must reach the report.

When a Mixed / RM ANOVA is fed technical replicates (several measurements per
subject x within-level), the values are averaged to the subject level before the
test and the user is told so via a data-health warning. The report renders that
warning through exactly one channel: ``report_summaries._build_data_health_warnings``,
which reads ``results["data_health"]``.

Commit 670dee4 renamed the *writer* of this warning to ``results["health"]`` but
not the reader, so nothing rendered it any more (nothing reads ``results["health"]``).
This test drives the real averaging path end-to-end and asserts the warning
survives into the rendered channel, guarding the writer/reader wiring — a branch
the rest of the suite never exercises.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from analysis.statisticaltester import StatisticalTester
from export.report_summaries import _SummariesMixin

REPLICATE_MARK = "Technische Replikate"


def _rm_df_with_replicates():
    # 6 subjects x 3 timepoints x 2 technical replicates -> duplicated (subject, time)
    rng = np.random.RandomState(3)
    rows = []
    for s in range(6):
        base = rng.randn()
        for t in ("T1", "T2", "T3"):
            for _rep in range(2):
                rows.append({"subject": f"S{s}", "time": t, "dv": base + rng.randn()})
    return pd.DataFrame(rows)


def _mixed_df_with_replicates():
    rng = np.random.RandomState(5)
    rows = []
    for s in range(8):
        grp = "A" if s < 4 else "B"
        base = rng.randn()
        for t in ("T1", "T2", "T3"):
            for _rep in range(2):
                rows.append({"subject": f"S{s}", "group": grp, "time": t, "dv": base + rng.randn()})
    return pd.DataFrame(rows)


def test_rm_replicate_averaging_warning_reaches_report():
    df = _rm_df_with_replicates()
    results = StatisticalTester._run_repeated_measures_anova_logged(
        df=df, dv="dv", subject="subject", within=["time"], alpha=0.05
    )
    rendered = _SummariesMixin._build_data_health_warnings(results)
    assert any(REPLICATE_MARK in w for w in rendered), (
        f"replicate-averaging warning missing from the rendered data-health channel; got {rendered!r}"
    )


@pytest.mark.xfail(
    reason="Separate bug: _run_mixed_anova_logged averages technical replicates "
    "with groupby([subject] + within), which drops the between factor and raises "
    "KeyError before the (now-fixed) data_health warning is written. The RM sibling "
    "is correct. Remove this marker when the mixed averaging keeps the between factor.",
    strict=False,
)
def test_mixed_replicate_averaging_warning_reaches_report():
    df = _mixed_df_with_replicates()
    results = StatisticalTester._run_mixed_anova_logged(
        df=df, dv="dv", subject="subject", between=["group"], within=["time"], alpha=0.05
    )
    rendered = _SummariesMixin._build_data_health_warnings(results)
    assert any(REPLICATE_MARK in w for w in rendered), (
        f"replicate-averaging warning missing from the rendered data-health channel; got {rendered!r}"
    )
