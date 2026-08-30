"""A clinical model that produced no number must be blocked, not reported.

`nonfinite_block` calls itself the safety net for "LMM, RM/Mixed/Two-Way ANOVA,
ANCOVA" in its own docstring. LMM and ANCOVA go through the clinical branch,
which returns before the line that consults it -- so the net named them and sat
on a path they never take. Found by the fuzzer on five separate seeds, each an
LMM whose fit overflowed: statistic nan, p_value nan, `blocked` unset, and a
4.8 MB report with five figures written from it.

The clinical branch gates its INPUT (validate_outcome, before the fit). Nothing
looked at what came back out.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from analysis.analysis_core import AnalysisManager
from analysis import clinical_models


def _lmm_frame():
    """The shape the fuzzer builds: subjects measured at two timepoints."""
    rows = []
    rng = np.random.default_rng(7)
    for subject in range(8):
        offset = float(rng.normal(0, 1))
        for level in ("T0", "T1"):
            rows.append({"Subject": f"S{subject}", "Time": level,
                         "Between": f"B{subject % 2}",
                         "Val": float(rng.normal(0, 1) + offset)})
    return pd.DataFrame(rows)


def _analyze(tmp_path, df):
    # analyze() insists on a real path even though the frame is injected --
    # the documented single-source-of-truth path is analysis_context.
    dummy = tmp_path / "unused.xlsx"
    pd.DataFrame({"a": [1]}).to_excel(dummy, index=False)
    context = {
        "factor_columns": ["Between"], "dv_columns": ["Val"],
        "group_labels": ["B0", "B1"], "subject_column": "Subject",
        "between_factors": ["Between"], "within_factors": ["Time"],
        "inferred_test": "lmm", "mode": "single", "injected_df": df,
    }
    return AnalysisManager.analyze(
        file_path=str(tmp_path / "unused.xlsx"),
        file_name=str(tmp_path / "out"),
        group_col="Between", groups=["B0", "B1"], value_cols=["Val"],
        dependent=True, subject_column="Subject",
        analysis_context=context, plot_type="Bar",
    )


@pytest.mark.parametrize("statistic,p_value", [
    (float("nan"), float("nan")),   # what the five fuzz seeds produced
    (float("inf"), 0.01),           # a statistic that overflowed
    (1.5, float("nan")),            # the fit ran, the test did not
])
def test_a_clinical_fit_without_a_number_is_blocked(tmp_path, monkeypatch,
                                                    statistic, p_value):
    def _no_number(self):
        return {"test": "Linear Mixed Model", "statistic": statistic,
                "p_value": p_value, "aic": float("nan"), "bic": float("nan")}

    monkeypatch.setattr(clinical_models.LinearMixedModel, "as_results_dict",
                        _no_number, raising=True)
    monkeypatch.setattr(clinical_models.LinearMixedModel, "fit",
                        lambda self, *a, **kw: None, raising=True)

    result = _analyze(tmp_path, _lmm_frame())

    assert result.get("blocked") is True, (
        "an LMM with statistic=%r p=%r was reported as a result" % (statistic, p_value))
    assert result.get("block_code") == "NON_FINITE_RESULT"
    # Blocked before the export, or the 4.8 MB report is written anyway.
    assert not list(tmp_path.glob("*results.html"))


def test_a_clinical_fit_that_produced_numbers_is_not_blocked(tmp_path, monkeypatch):
    """The counterpart, so the guard cannot pass by blocking everything."""
    def _fine(self):
        return {"test": "Linear Mixed Model", "statistic": 4.2, "p_value": 0.03}

    monkeypatch.setattr(clinical_models.LinearMixedModel, "as_results_dict",
                        _fine, raising=True)
    monkeypatch.setattr(clinical_models.LinearMixedModel, "fit",
                        lambda self, *a, **kw: None, raising=True)

    result = _analyze(tmp_path, _lmm_frame())
    assert not result.get("blocked")
    assert result.get("p_value") == 0.03
