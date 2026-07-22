"""Heteroscedastic OLS regression died completely instead of switching to HC3.

``SimpleLinearRegressionModel.fit`` calls ``get_robustcov_results(cov_type='HC3')``
when Breusch-Pagan flags heteroscedasticity (n >= 20). That call returns a bare
``OLSResults``, not the ``RegressionResultsWrapper`` the rest of the class
assumes: ``.resid``/``.fittedvalues`` are ndarrays without ``.values``,
``.params`` has no ``.index``, and ``conf_int()`` is an ndarray rather than a
DataFrame. The first of those blew up in ``diagnostics()`` and the whole
analysis came back as ``test="Not performed"`` with every field None.

The audit's isolation: same X, same n=80, same seed 0 -- the homoscedastic
variant returned a full result, the heteroscedastic one returned nothing. So
the branch written to handle heteroscedasticity was the branch that killed it.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan

from analysis.analysis_core import AnalysisManager
from analysis.correlation_models import SimpleLinearRegressionModel


@pytest.fixture
def dummy_file(tmp_path):
    path = tmp_path / "dummy.xlsx"
    pd.DataFrame({"a": [1]}).to_excel(path, index=False)
    return str(path)


def _datasets():
    """Audit fixture: seed 0, n=80, identical X, two error structures."""
    rng = np.random.default_rng(0)
    n = 80
    x = rng.uniform(1, 10, n)
    homoscedastic = 2.0 + 0.8 * x + rng.normal(0, 3, n)
    # The heteroscedastic variant reuses the same X but draws its errors from a
    # fresh stream, exactly as the audit did.
    rng2 = np.random.default_rng(0)
    heteroscedastic = 2.0 + 0.8 * x + rng2.normal(0, 1, n) * x * 1.2
    return x, homoscedastic, heteroscedastic


def _run(df, dummy_file, tmp_path, tag):
    ctx = {
        "injected_df": df, "x_variable": "X", "factor_columns": ["X"],
        "dv_columns": ["Y"], "mode": "single",
    }
    return AnalysisManager.analyze(
        file_path=dummy_file, group_col="X", groups=[], value_cols=["Y"],
        save_plot=False, skip_plots=True, file_name=str(tmp_path / tag),
        analysis_context=ctx, test="linear_regression",
    )


def test_premise_the_fixture_really_is_heteroscedastic():
    """Guard: the HC3 branch only fires when Breusch-Pagan is significant."""
    x, homo, hetero = _datasets()
    for values, expect_significant in ((homo, False), (hetero, True)):
        model = smf.ols("Y ~ X", data=pd.DataFrame({"X": x, "Y": values})).fit()
        bp_p = het_breuschpagan(model.resid, model.model.exog)[1]
        assert bool(bp_p < 0.05) is expect_significant, f"BP p={bp_p}"


def test_homoscedastic_control_still_returns_a_result(dummy_file, tmp_path):
    """Positive control: the non-HC3 path was never broken."""
    x, homo, _ = _datasets()
    results = _run(pd.DataFrame({"X": x, "Y": homo}), dummy_file, tmp_path, "homo")
    assert results.get("test") == "Linear Regression (OLS)"
    assert results.get("cov_type") == "nonrobust"
    assert results.get("p_value") is not None


def test_heteroscedastic_regression_returns_hc3_results(dummy_file, tmp_path):
    x, _, hetero = _datasets()
    results = _run(pd.DataFrame({"X": x, "Y": hetero}), dummy_file, tmp_path, "hetero")

    assert results.get("error") is None, (
        f"analysis errored instead of applying HC3: {results.get('error')}"
    )
    assert results.get("test") == "Linear Regression (OLS)"
    assert results.get("cov_type") == "HC3", (
        "heteroscedasticity was detected but the robust covariance never took effect"
    )
    assert results.get("p_value") is not None
    assert results.get("beta") is not None
    assert results.get("r_squared") is not None


def test_hc3_coefficient_table_carries_real_names_and_intervals(dummy_file, tmp_path):
    """params.index / conf_int().loc were the other two casualties of the wrapper loss."""
    x, _, hetero = _datasets()
    results = _run(pd.DataFrame({"X": x, "Y": hetero}), dummy_file, tmp_path, "hetero_tbl")

    table = results.get("coefficient_table") or []
    assert [row["parameter"] for row in table] == ["Intercept", "X"]
    for row in table:
        assert row["ci_lower"] is not None and row["ci_upper"] is not None
        assert row["ci_lower"] < row["coefficient"] < row["ci_upper"]

    assert results.get("residuals"), "residuals list went missing"
    assert results.get("fitted_values"), "fitted values list went missing"


def test_hc3_standard_errors_actually_differ_from_ols(dummy_file, tmp_path):
    """Discriminating check: HC3 must change the inference, not just the label."""
    x, _, hetero = _datasets()
    df = pd.DataFrame({"X": x, "Y": hetero})
    results = _run(df, dummy_file, tmp_path, "hetero_se")

    plain = smf.ols("Y ~ X", data=df).fit()
    robust = plain.get_robustcov_results(cov_type="HC3")

    reported = {row["parameter"]: row["std_err"] for row in results["coefficient_table"]}
    assert reported["X"] == pytest.approx(float(robust.bse[1]), rel=1e-9)
    assert reported["X"] != pytest.approx(float(plain.bse["X"]), rel=1e-6)


def test_model_object_exposes_hc3_after_fit():
    """Unit-level: the model itself, without the analysis pipeline around it."""
    x, _, hetero = _datasets()
    model = SimpleLinearRegressionModel()
    model.fit(pd.DataFrame({"X": x, "Y": hetero}), x_col="X", y_col="Y")
    assert model._cov_type == "HC3"
    payload = model.as_results_dict()
    assert payload.get("error") is None
    assert payload["cov_type"] == "HC3"


def test_decision_tree_hc3_branch_becomes_reachable(dummy_file, tmp_path):
    """Check 6 flagged COV_HC3 as unreachable purely because of this crash."""
    from visualization.flowchartvisualizer import FlowchartVisualizer

    x, _, hetero = _datasets()
    results = _run(pd.DataFrame({"X": x, "Y": hetero}), dummy_file, tmp_path, "hetero_tree")
    tree = FlowchartVisualizer.get_tree_json(results)
    assert tree is not None

    active = {(e["source"], e["target"]) for e in tree["edges"] if e["isActive"]}
    alternatives = {(e["source"], e["target"]) for e in tree["edges"] if e["isAlternative"]}
    assert ("ROBUST_BRANCH", "COV_HC3") in active
    assert ("COV_HC3", "COEFFICIENTS") in active
    assert ("ROBUST_BRANCH", "COV_NONROBUST") in alternatives
