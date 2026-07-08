"""_FormattingMixin has no shared HTML-escaping helper, unlike outlier_html_exporter.py's
existing _esc(). RE1's fix needs one reusable helper instead of inlining html.escape() three
times.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from export.report_formatting import _FormattingMixin


def test_esc_escapes_html_special_characters():
    assert _FormattingMixin._esc("<script>alert(1)</script>") == "&lt;script&gt;alert(1)&lt;/script&gt;"


def test_esc_handles_non_string_input():
    assert _FormattingMixin._esc(42) == "42"
    assert _FormattingMixin._esc(None) == "None"
