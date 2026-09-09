"""A constant predictor produced r=nan, which the report dressed up as a result.

The audit fed a constant X column (all values identical) into the correlation
path. scipy returns r=nan for a zero-variance input, and the pipeline shipped
it as three mutually contradictory statements in one report:

  * hero: "Correlation (Spearman) did not show evidence against the null
    hypothesis"  (p=nan read as not-significant)
  * interpretation: "Negative, very strong (|r| = nan)"  (nan fell through the
    _interpret cascade into the strongest bucket)
  * assumptions: the constant column certified "Normality ... Passed"

A constant column makes correlation genuinely undefined. The fix gates it at
the same data-quality pre-flight that already blocks a constant outcome or
covariate, so the user gets one honest "undefined" message instead of three
contradictory ones. _interpret is additionally hardened so a nan r can never
render as "very strong" anywhere.
"""
import math
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from analysis.analysis_core import AnalysisManager
from analysis.correlation_models import CorrelationModel


@pytest.fixture
def dummy_file(tmp_path):
    path = tmp_path / "dummy.xlsx"
    pd.DataFrame({"a": [1]}).to_excel(path, index=False)
    return str(path)


def _constant_x_df():
    """Audit fixture: seed 4, n=30, X held at a single value."""
    rng = np.random.default_rng(4)
    n = 30
    return pd.DataFrame({"X": np.full(n, 5.0), "Y": rng.normal(0, 1, n)})


def _healthy_df():
    rng = np.random.default_rng(4)
    n = 30
    x = rng.normal(50, 10, n)
    return pd.DataFrame({"X": x, "Y": 2.0 * x + rng.normal(0, 8, n)})


def _run(df, dummy_file, tmp_path, tag):
    out = str(tmp_path / tag)
    ctx = {
        "injected_df": df, "x_variable": "X", "factor_columns": ["X"],
        "dv_columns": ["Y"], "mode": "single",
    }
    results = AnalysisManager.analyze(
        file_path=dummy_file, group_col="X", groups=[], value_cols=["Y"],
        save_plot=False, skip_plots=True, file_name=out,
        analysis_context=ctx, test="correlation",
    )
    return results, out + "_results.html"


def _visible_text(path):
    doc = open(path, errors="ignore").read()
    doc = re.sub(r"<script.*?</script>", "", doc, flags=re.S | re.I)
    doc = re.sub(r"<style.*?</style>", "", doc, flags=re.S | re.I)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", doc))


# --- unit-level: the _interpret cascade must not call nan "very strong" ---

def test_interpret_rejects_nan():
    label = CorrelationModel._interpret(float("nan"))
    assert "very strong" not in label
    assert "nan" not in label.lower()
    assert "not computable" in label.lower() or "undefined" in label.lower()


def test_interpret_still_labels_real_values():
    """Positive control: the cascade still works for genuine r."""
    assert "very strong" in CorrelationModel._interpret(0.95)
    assert "negligible" in CorrelationModel._interpret(0.05)
    assert "Positive" in CorrelationModel._interpret(0.5)
    assert "Negative" in CorrelationModel._interpret(-0.5)


# --- pipeline-level: the degenerate run is blocked, not dressed up ---

def test_constant_predictor_is_blocked_not_reported(dummy_file, tmp_path):
    results, _ = _run(_constant_x_df(), dummy_file, tmp_path, "constant")
    assert results.get("blocked") is True, "constant predictor should block, not fit"
    reason = str(results.get("block_reason") or results.get("error") or "").lower()
    assert "variance" in reason or "identical" in reason or "undefined" in reason
    # no fabricated statistic survived
    assert results.get("statistic") is None
    r = results.get("r")
    assert r is None or (isinstance(r, float) and not math.isnan(r)) is False or math.isnan(r) is False


def test_constant_predictor_report_has_no_contradictions(dummy_file, tmp_path):
    """A blocked run returns early with no full report — so the three
    contradictory statements can never render. The result dict is the single
    honest surface; assert none of the fabricated strings survive in it."""
    results, report = _run(_constant_x_df(), dummy_file, tmp_path, "constant_html")

    # the contradictory report is never produced at all
    assert not os.path.exists(report), "a blocked analysis must not emit a full report"

    blob = repr(results).lower()
    assert "very strong" not in blob
    assert "|r| = nan" not in blob
    assert "did not show evidence" not in blob

    # the honest message is present exactly once, as the block reason
    assert results.get("test") == "Not performed"
    assert results.get("block_reason")


def test_healthy_correlation_is_not_blocked(dummy_file, tmp_path):
    """Positive control: an ordinary correlation still runs and reports."""
    results, report = _run(_healthy_df(), dummy_file, tmp_path, "healthy")
    assert not results.get("blocked")
    assert results.get("statistic") is not None
    assert not math.isnan(results.get("statistic"))
    text = _visible_text(report)
    assert "Correlation" in text
    assert "|r| = nan" not in text
