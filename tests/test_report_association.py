"""Linear regression's coefficient_table (correlation_models.py:848,
SimpleLinearRegressionModel.as_results_dict) was computed but had zero
readers anywhere in the export layer. This wires it into the HTML report,
mirroring the existing _build_beta_coefficient_table_html pattern but reading
the correct key (coefficient_table, not coefficients) and using a t-column
(OLS) instead of z-column (GLM, beta regression's own case).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from export.report_association import _AssociationMixin


def _linear_regression_results():
    return {
        "model_type": "LinearRegression",
        "coefficient_table": [
            {"parameter": "Intercept", "coefficient": 1.5, "std_err": 0.3,
             "t_value": 5.0, "p_value": 0.001, "ci_lower": 0.9, "ci_upper": 2.1},
            {"parameter": "x", "coefficient": 0.8, "std_err": 0.2,
             "t_value": 4.0, "p_value": 0.02, "ci_lower": 0.4, "ci_upper": 1.2},
        ],
    }


def test_linear_regression_coefficient_table_renders_html():
    block = _AssociationMixin._build_linear_regression_coefficient_table_html(
        _linear_regression_results()
    )
    assert block is not None
    assert "Intercept" in block["html"]
    assert "<th>t</th>" in block["html"]
    assert "0.001" in block["html"] or "&lt;0.001" in block["html"] or "0.0010" in block["html"]


def test_linear_regression_coefficient_table_returns_none_when_empty():
    block = _AssociationMixin._build_linear_regression_coefficient_table_html(
        {"model_type": "LinearRegression", "coefficient_table": []}
    )
    assert block is None
