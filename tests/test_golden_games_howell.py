"""Golden-reference correctness suite for the Games-Howell post-hoc test.

Runs the app's hand-rolled GamesHowellTest against frozen
PMCMRplus::gamesHowellTest p-values (tests/golden/references_games_howell.json,
produced by validation/generate_golden_posthoc.py). Both compute the same
sqrt(2)*|t| vs studentized-range-with-Welch-df statistic; agreement is near
exact (probe |dp| 4e-9).

Every golden dataset has n>=2 in all groups, so the app's studentized-range
nmeans (k = number of groups with n>=2) equals R's total group count. The n<2
exclusion path is not a numeric golden case -- reproducing the app's exact
group-exclusion logic in R would itself be a new error source -- so it is checked
by a separate reachability guard test below.
"""
import json
import math
import os

import numpy as np
import pytest

from analysis.posthoc_core import GamesHowellTest

_REF = os.path.join(os.path.dirname(__file__), "golden", "references_games_howell.json")
with open(_REF) as _fh:
    _DATA = json.load(_fh)
_CASES = _DATA["cases"]
_TOL = _DATA["tol"]


def _by_pair(comparisons):
    def _key(c):
        return frozenset(c["groups"]) if "groups" in c else frozenset([c["group1"], c["group2"]])
    return {_key(c): c for c in comparisons}


@pytest.mark.parametrize("case", _CASES, ids=[c["id"] for c in _CASES])
def test_golden_games_howell(case):
    samples = {g: list(v) for g, v in case["samples"].items()}
    groups = list(case["groups"])
    result = GamesHowellTest.perform_test(groups, samples, alpha=0.05)
    assert not result.get("error"), f"{case['id']}: {result.get('error')}"

    app = _by_pair(result["pairwise_comparisons"])
    ref = _by_pair(case["comparisons"])
    assert set(app.keys()) == set(ref.keys()), (
        f"{case['id']}: pair set mismatch app={set(app)} ref={set(ref)}"
    )

    for pair, r in ref.items():
        a = app[pair]["p_value"]
        assert a is not None and math.isfinite(a), f"{case['id']} {sorted(pair)}: app p not finite"
        assert abs(a - r["p_value"]) <= _TOL["p_value"], (
            f"{case['id']} {sorted(pair)} p: app={a!r} vs R={r['p_value']!r} (tol={_TOL['p_value']})"
        )


def test_games_howell_excludes_singleton_group():
    """Reachability guard for the n<2 exclusion path (not a numeric golden case).

    A group with a single observation cannot enter a Welch/studentized-range
    comparison; the app must silently drop every pair that involves it and still
    compute the remaining pairs. This pins that the exclusion path is reached and
    leaves the rest of the family intact."""
    rng = np.random.default_rng(99)
    samples = {"A": list(rng.normal(10, 2, 8)),
               "B": list(rng.normal(13, 2, 9)),
               "C": list(rng.normal(16, 2, 7)),
               "D": [12.0]}  # singleton -> must be excluded
    result = GamesHowellTest.perform_test(["A", "B", "C", "D"], samples, alpha=0.05)
    assert not result.get("error")

    pairs = {frozenset([c["group1"], c["group2"]]) for c in result["pairwise_comparisons"]}
    assert all("D" not in p for p in pairs), f"singleton group D leaked into {pairs}"
    # the three A/B/C pairs still present
    assert pairs == {frozenset(["A", "B"]), frozenset(["A", "C"]), frozenset(["B", "C"])}
