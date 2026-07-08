"""RE2: report_charts.py's own group-comparison chart already escapes group_name via
html.escape() before using it as a Plotly name= - 5 other sites in report_charts.py and
report_summaries.py take factor-level strings straight from data without escaping. Applies the
shared _FormattingMixin._esc() helper (added in RE5) at all 5 sites for consistency.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from export.report_formatting import _FormattingMixin

_MALICIOUS = "<script>alert(1)</script>"


def test_esc_helper_is_what_report_charts_and_summaries_should_use():
    # This pins down the exact helper the fix must route through - the real
    # regression coverage is the full-suite run in Step 5, since these 5
    # sites are deep inside large Plotly-figure-building functions that would
    # need a disproportionately large harness to unit-test each in isolation.
    assert _FormattingMixin._esc(_MALICIOUS) == "&lt;script&gt;alert(1)&lt;/script&gt;"
