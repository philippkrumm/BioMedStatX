"""The report's assumption summary showed no sphericity row for Mixed ANOVA.

`_build_assumption_summary` read only `results["sphericity_test"]`. The RM path
writes that key; the Mixed path writes the same information under a `within_`
prefix (`within_sphericity_test`, `within_correction_used`,
`within_sphericity_corrections`) because a mixed design has two effect families
and the within verdict has to be qualified.

Consequence, measured: a Mixed ANOVA with Mauchly p = 1.6e-22 rendered an EMPTY
assumption summary. The correction was applied and appears in the factor table
(report_stat_rows handles both schemas explicitly), but the section that tells
the reader "sphericity was tested and violated" stayed silent.

Reporting only -- `_build_assumption_summary` builds display rows; it gates no
computation.

The fallback must key on PRESENCE, not truthiness: an RM result whose
`sphericity_test` is an empty-but-present dict must stay on the RM branch rather
than silently borrowing the mixed field.
"""
import numpy as np
import pandas as pd
import pytest

from export.report_summaries import _SummariesMixin


def _frame(mixed):
    rng = np.random.default_rng(7)
    rows = []
    for g in (("ctrl", "trt") if mixed else ("only",)):
        for s in range(15):
            b = rng.normal(0, 1)
            off = 0.5 if g == "trt" else 0.0
            vals = {"T1": b + off + rng.normal(0, 0.3),
                    "T2": b + off + 0.6 + rng.normal(0, 0.3),
                    "T3": b + off + 1.0 + rng.normal(0, 3.5),
                    "T4": b + off + 1.4 + rng.normal(0, 3.5)}
            for t, y in vals.items():
                rows.append({"subj": f"{g}{s}", "grp": g, "time": t, "y": y})
    return pd.DataFrame(rows)


def _rows(results):
    summary = _SummariesMixin._build_assumption_summary(results)
    out = summary.get("rows") if isinstance(summary, dict) else summary
    return out or []


def _names(results):
    return [r.get("name") for r in _rows(results)]


@pytest.fixture(scope="module")
def rm_result():
    from analysis.statisticaltester import StatisticalTester as ST
    return ST._run_repeated_measures_anova(_frame(False), dv="y", subject="subj",
                                           within=["time"], alpha=0.05)


@pytest.fixture(scope="module")
def mixed_result():
    from analysis.statisticaltester import StatisticalTester as ST
    return ST._run_mixed_anova(_frame(True), dv="y", subject="subj",
                               between=["grp"], within=["time"], alpha=0.05)


def test_rm_still_reports_its_sphericity_row(rm_result):
    assert any("Sphericity" in str(n) for n in _names(rm_result)), _names(rm_result)


def test_mixed_reports_a_sphericity_row(mixed_result):
    """The defect: this list was empty."""
    assert any("Sphericity" in str(n) for n in _names(mixed_result)), _names(mixed_result)


def test_mixed_sphericity_row_carries_the_real_mauchly_numbers(mixed_result):
    row = next(r for r in _rows(mixed_result) if "Sphericity" in str(r.get("name")))
    src = mixed_result["within_sphericity_test"]
    assert str(round(float(src["W"]), 4)) in str(row["statistic"]), (row, src["W"])
    assert "Flagged" in str(row["status_label"]) or "✗" in str(row["status_label"]), row


def test_mixed_correction_note_names_greenhouse_geisser(mixed_result):
    summary = _SummariesMixin._build_assumption_summary(mixed_result)
    blob = str(summary)
    assert "Greenhouse-Geisser" in blob, blob[:400]


def test_empty_but_present_rm_field_does_not_borrow_the_mixed_field():
    """Presence, not truthiness. An RM payload whose sphericity_test happens to
    be an empty dict must not fall through to within_sphericity_test."""
    results = {
        "test": "Repeated Measures ANOVA",
        "sphericity_test": {},                     # present, empty
        "within_sphericity_test": {                # must NOT be used
            "W": 0.0173, "p_value": 1.5e-22, "sphericity_assumed": False,
        },
        "within_correction_used": "Greenhouse-Geisser (ε = 0.632)",
    }
    assert not any("Sphericity" in str(n) for n in _names(results)), _names(results)


def test_absent_rm_field_does_fall_back():
    results = {
        "test": "Mixed ANOVA",
        "within_sphericity_test": {
            "W": 0.0173, "p_value": 1.5e-22, "sphericity_assumed": False,
        },
        "within_correction_used": "Greenhouse-Geisser (ε = 0.632)",
    }
    assert any("Sphericity" in str(n) for n in _names(results)), _names(results)
