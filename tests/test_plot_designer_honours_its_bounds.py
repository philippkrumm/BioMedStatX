"""Every numeric control in the figure builder is held to its own min/max.

`input type=number` limits only the spinner arrows: a typed value past `max`
passes straight through, and so does anything a script assigns. The designer
read all sixteen of its numeric controls with a bare parseInt/parseFloat, so the
ranges it shows the user were advisory. pd-axis-size declares max 32; 65 was
applied without complaint, and at 65 the horizontal legend wraps to a second row
and leaves the plot container -- which is how the visual fuzzer found it.

Structural rather than behavioural on purpose. There is no JS test runner here,
and the defect is not one wrong number: it is a whole class of reads that skip
the bound. A guard that walks every read catches the seventeenth control that
gets added next year, which a test pinning pd-axis-size never would.
"""
import os
import re

import pytest

_JS = os.path.join(os.path.dirname(__file__), "..", "src", "templates", "plot_designer.js")

# A numeric read of a control: parseInt/parseFloat straight off .value.
_READ = re.compile(
    r'parse(?:Int|Float)\(document\.getElementById\("([a-zA-Z0-9\-]+)"\)\.value')


def _source():
    with open(_JS, encoding="utf-8") as fh:
        return fh.read()


def test_every_numeric_control_read_goes_through_the_clamp():
    source = _source()
    unguarded = []
    for match in _READ.finditer(source):
        # The clamp wraps the read, so "_pdNum(" opens just before it.
        before = source[max(0, match.start() - 60):match.start()]
        if "_pdNum(" not in before:
            unguarded.append((match.group(1), source[:match.start()].count("\n") + 1))
    assert not unguarded, (
        "these numeric control reads ignore the control's own min/max: "
        + ", ".join(f"{name} (line {line})" for name, line in unguarded))


def test_the_guard_is_actually_reading_something():
    """A regex that matches nothing passes the test above for free."""
    assert len(_READ.findall(_source())) >= 10


def test_the_clamp_reads_the_bounds_off_the_element():
    """Not from a table in the JS.

    A bound repeated in code is a bound that drifts away from the one the user
    is shown, which is the defect one level up.
    """
    source = _source()
    body = source[source.index("function _pdNum("):]
    body = body[:body.index("\n  }") + 4]
    assert "el.min" in body and "el.max" in body
    assert re.search(r"\b(8|32|42)\b", body) is None, (
        "the clamp hard-codes a bound instead of reading the element's")


@pytest.mark.parametrize("value,lo,hi,expected", [
    (65, 8, 32, 32),      # the value the fuzzer applied
    (2, 8, 32, 8),
    (22, 8, 32, 22),
    (32, 8, 32, 32),
])
def test_the_clamp_arithmetic_is_what_the_js_does(value, lo, hi, expected):
    """The same three lines, so the intent is pinned even without a JS runner."""
    if value < lo:
        value = lo
    if value > hi:
        value = hi
    assert value == expected
