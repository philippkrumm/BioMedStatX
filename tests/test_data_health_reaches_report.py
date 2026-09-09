"""DataHealthScanner ran on every clinical analysis and its output never reached
the user.

`analysis_core` attaches the scanner's report as `results["data_health"]`, but
that key had exactly one occurrence outside the producer — the line that writes
it. Nothing narrowed the dict at any layer boundary; the consumer simply never
existed. The audit generated a report for a dataset with VIF = 1.06e10 and two
covariate outliers and found none of the five checks anywhere in the 5 MB
document.

This is the end-to-end test: real analysis, real HTML export, search the
rendered file.
"""
import os
import pathlib
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from analysis.analysis_core import AnalysisManager


@pytest.fixture
def dummy_file(tmp_path):
    path = tmp_path / "dummy.xlsx"
    pd.DataFrame({"a": [1]}).to_excel(path, index=False)
    return str(path)


def _unhealthy_df():
    """Trips covariate outliers (MAD) and multicollinearity (VIF)."""
    rng = np.random.default_rng(2026)
    n = 30
    df = pd.DataFrame({
        "Group": ["ctrl"] * n + ["low"] * n + ["high"] * n,
        "Cov": rng.normal(10, 2, 3 * n),
    })
    df.loc[0, "Cov"] = 900.0
    df.loc[1, "Cov"] = -700.0
    df["Cov2"] = df["Cov"] * 1.0001 + rng.normal(0, 1e-3, 3 * n)
    df["Value"] = (2 + 0.05 * df["Cov"]
                   + df["Group"].map({"ctrl": 0.0, "low": 1.5, "high": 3.0})
                   + rng.normal(0, 1, 3 * n))
    return df


def _healthy_df():
    rng = np.random.default_rng(7)
    n = 30
    df = pd.DataFrame({
        "Group": ["ctrl"] * n + ["low"] * n + ["high"] * n,
        "Cov": rng.normal(10, 2, 3 * n),
    })
    df["Value"] = (2 + 0.8 * df["Cov"]
                   + df["Group"].map({"ctrl": 0.0, "low": 1.5, "high": 3.0})
                   + rng.normal(0, 1, 3 * n))
    return df


def _run(df, dummy_file, tmp_path, tag, covariates):
    out = str(tmp_path / tag)
    ctx = {
        "injected_df": df, "factor_columns": ["Group"], "between_factors": ["Group"],
        "dv_columns": ["Value"], "group_labels": ["ctrl", "low", "high"],
        "mode": "single", "covariates": covariates,
    }
    results = AnalysisManager.analyze(
        file_path=dummy_file, group_col="Group", groups=["ctrl", "low", "high"],
        value_cols=["Value"], save_plot=False, skip_plots=True, file_name=out,
        analysis_context=ctx, test="ancova", covariates=covariates,
    )
    report = pathlib.Path(out + "_results.html")
    return results, report


def test_scanner_produces_warnings_for_this_dataset(dummy_file, tmp_path):
    """Premise guard: the producer side has to actually fire."""
    results, _ = _run(_unhealthy_df(), dummy_file, tmp_path, "premise", ["Cov", "Cov2"])
    warnings_list = (results.get("data_health") or {}).get("warnings") or []
    assert len(warnings_list) >= 2
    assert any("Multicollinearity" in w for w in warnings_list)
    assert any("mod. Z-score" in w for w in warnings_list)


def test_health_warnings_appear_in_the_rendered_report(dummy_file, tmp_path):
    results, report = _run(_unhealthy_df(), dummy_file, tmp_path, "unhealthy",
                           ["Cov", "Cov2"])
    assert report.exists(), "HTML report was not generated"
    document = report.read_text(errors="ignore")

    # positive control: the report really did render engine content, so an
    # "absent" verdict below cannot be blamed on a broken document
    assert "Slope Homogeneity" in document
    assert "ANCOVA" in document

    assert "Multicollinearity" in document, (
        "VIF warning still does not reach the report"
    )
    assert "mod. Z-score" in document, (
        "covariate outlier warning still does not reach the report"
    )


def test_clean_dataset_renders_no_empty_health_section(dummy_file, tmp_path):
    """Positive control on the other side: a healthy dataset must not produce a
    dangling or empty warning block."""
    results, report = _run(_healthy_df(), dummy_file, tmp_path, "healthy", ["Cov"])
    warnings_list = (results.get("data_health") or {}).get("warnings") or []
    assert warnings_list == [], "fixture is supposed to be clean"

    document = report.read_text(errors="ignore")
    assert "Multicollinearity" not in document
    assert "mod. Z-score" not in document
    assert "Data quality" not in document
