"""M1 (round-3 audit): _build_ancova_chart interpolates raw ANCOVA group-level names
(adjusted_means[factor][level]) directly into a Plotly name=f"{label} (n={n})" with no
escaping - the same injection class RE1/RE2 fixed, in the same model type (ANCOVA) RE1's own
reproduction proved exploitable via patsy's C(group)[T.<value>] encoding. This function wasn't
in RE2's original enumerated list and wasn't touched by B6.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from export.report_charts import _ChartsMixin

_MALICIOUS_GROUP = "Zebra<script>alert(1)</script>"


def test_ancova_chart_escapes_malicious_group_level_name():
    results = {
        "model_type": "ANCOVA",
        "adjusted_means": {
            "Group": {
                "ctrl": {"adjusted_mean": 1.0, "raw_sd": 0.5, "n": 10},
                _MALICIOUS_GROUP: {"adjusted_mean": 2.0, "raw_sd": 0.6, "n": 10},
            }
        },
        "covariates_used": [],
    }

    bundle = _ChartsMixin._build_ancova_chart(results)

    assert bundle is not None, "expected a chart bundle for valid ANCOVA adjusted_means"
    name_field_start = bundle["html"].find('"name":"Zebra')
    assert name_field_start != -1, "expected the malicious group's trace name field in the html"
    name_field = bundle["html"][name_field_start:name_field_start + 80]
    assert "&lt;script&gt;" in name_field, (
        f"the trace name must be HTML-escaped before Plotly's own JSON serialization "
        f"double-encodes it, got: {name_field!r}"
    )
