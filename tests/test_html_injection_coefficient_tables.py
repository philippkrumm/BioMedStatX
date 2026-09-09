"""RE1: report_association.py's coefficient-table builders interpolate a fitted model's raw
parameter name into an HTML f-string with no escaping. Reproduces this end-to-end against a
real statsmodels logit fit with a malicious category value, matching how round 1's audit
originally demonstrated the injection - not a hand-crafted dict, a real patsy-encoded parameter
name.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import pytest
import statsmodels.formula.api as smf

from export.report_association import _AssociationMixin

_MALICIOUS_GROUP = "Zebra<script>alert(1)</script>"


def _fit_logit_with_malicious_group_name():
    # pd.Categorical with an explicit category order forces patsy to treat
    # "ctrl" as the reference level - without it, patsy sorts levels
    # alphabetically and "Zebra..." (uppercase Z) sorts BEFORE "ctrl"
    # (lowercase c) in ASCII, making the malicious group the reference
    # instead of the one that ends up encoded into the parameter name.
    df = pd.DataFrame({
        "y": [0, 0, 0, 1, 1, 1, 0, 1],
        "group": pd.Categorical(
            ["ctrl"] * 4 + [_MALICIOUS_GROUP] * 4,
            categories=["ctrl", _MALICIOUS_GROUP],
        ),
    })
    return smf.logit("y ~ C(group)", data=df).fit(disp=0)


def test_patsy_really_does_leak_the_raw_group_name_into_the_parameter():
    model = _fit_logit_with_malicious_group_name()
    param_names = list(model.params.index)
    assert any(_MALICIOUS_GROUP in name for name in param_names), (
        f"sanity check failed - patsy's parameter names don't contain the raw group name: "
        f"{param_names}"
    )


def test_or_table_html_escapes_the_malicious_parameter_name():
    model = _fit_logit_with_malicious_group_name()
    or_table = [
        {
            "parameter": str(name),
            "odds_ratio": 1.5,
            "ci_lower": 1.0,
            "ci_upper": 2.0,
            "z_value": 1.0,
            "p_value": 0.1,
        }
        for name in model.params.index
        if name != "Intercept"
    ]

    bundle = _AssociationMixin._build_or_table_html({"odds_ratios": or_table})

    assert "<script>alert(1)</script>" not in bundle["html"], (
        "raw <script> tag from the group name survived unescaped into the exported HTML"
    )
    assert "&lt;script&gt;" in bundle["html"], "the escaped form must still be present"


def test_linear_regression_coefficient_table_html_escapes_the_malicious_parameter_name():
    coef_table = [
        {
            "parameter": _MALICIOUS_GROUP,
            "coefficient": 0.5,
            "std_err": 0.1,
            "t_value": 5.0,
            "p_value": 0.001,
            "ci_lower": 0.3,
            "ci_upper": 0.7,
        }
    ]
    bundle = _AssociationMixin._build_linear_regression_coefficient_table_html({"coefficient_table": coef_table})
    assert "<script>alert(1)</script>" not in bundle["html"]
