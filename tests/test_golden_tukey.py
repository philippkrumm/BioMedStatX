"""Golden-reference correctness suite for Tukey HSD.

Runs the app's TukeyHSD (statsmodels.pairwise_tukeyhsd) against frozen Base R
stats::TukeyHSD(aov(...)) values -- the canonical implementation statsmodels was
built to reproduce. References in tests/golden/references_tukey.json, produced by
validation/generate_golden_posthoc.py.

Matched per unordered pair: the Tukey-adjusted p-value and the |mean difference|
(the app normalises the pair, so the sign of the difference is compared on its
magnitude). The p-value tolerance is 1e-4 because statsmodels reports the p-adj
rounded to four decimals; the raw agreement is far tighter (probe |dp| 1.7e-5).

KNOWN LIMITATION: the observed p margin (max |dp| 4.5e-5 vs the 1e-4 tolerance)
is only ~2x, and it is bounded by statsmodels' 4-decimal p-adj rounding (max
5e-5), NOT by any methodological gap -- app and Base R compute the identical
pooled studentized-range HSD. A future statsmodels or R minor version that
changes that rounding could make this margin flicker without a real regression.
If it does, widen the tolerance to 1.5e-4 (still far below any true divergence)
rather than treating it as a correctness failure. Environment when frozen:
statsmodels + R 4.5.3.
"""
import json
import math
import os

import pytest

from analysis.posthoc_core import TukeyHSD

_REF = os.path.join(os.path.dirname(__file__), "golden", "references_tukey.json")
with open(_REF) as _fh:
    _DATA = json.load(_fh)
_CASES = _DATA["cases"]
_TOL = _DATA["tol"]


def _by_pair(comparisons):
    def _key(c):
        return frozenset(c["groups"]) if "groups" in c else frozenset([c["group1"], c["group2"]])
    return {_key(c): c for c in comparisons}


def _assert_close(label, actual, expected, tol):
    assert actual is not None, f"{label}: app returned None, expected {expected}"
    assert math.isfinite(actual), f"{label}: app value not finite ({actual})"
    assert abs(actual - expected) <= tol, (
        f"{label}: app={actual!r} vs R reference={expected!r} (tol={tol})"
    )


@pytest.mark.parametrize("case", _CASES, ids=[c["id"] for c in _CASES])
def test_golden_tukey(case):
    samples = {g: list(v) for g, v in case["samples"].items()}
    groups = list(case["groups"])
    result = TukeyHSD.perform_test(groups, samples, alpha=0.05)
    assert not result.get("error"), f"{case['id']}: {result.get('error')}"

    app = _by_pair(result["pairwise_comparisons"])
    ref = _by_pair(case["comparisons"])
    assert set(app.keys()) == set(ref.keys()), (
        f"{case['id']}: pair set mismatch app={set(app)} ref={set(ref)}"
    )

    for pair, r in ref.items():
        a = app[pair]
        _assert_close(f"{case['id']} {sorted(pair)} p", a.get("p_value"),
                      r["p_value"], _TOL["p_value"])
        _assert_close(f"{case['id']} {sorted(pair)} |diff|", abs(a.get("statistic")),
                      abs(r["diff"]), _TOL["diff"])
