"""Golden-reference correctness suite for Dunn's test.

Deliberately decoupled (see validation/r_templates/dunn.R). The R oracle
validates only the hard part -- the rank-based, tie-corrected raw p-value -- and
the multiplicity correction is checked as a library unit, because no R package
offers a bit-identical Holm-Sidak (validating it against R would introduce a
second-oracle assumption). Two decoupled halves alone do not prove the seam
between them is wired correctly, so a third test pushes the R-validated raw
p-values through the app's own Holm-Sidak stage and checks each adjusted value
lands on the right pair -- exactly the positional-mismatch class of bug this
project has hit before (RM post-hoc, two-group pairing, group labels).

References in tests/golden/references_dunn.json (raw p per pair from
PMCMRplus::kwAllPairsDunnTest(p.adjust="none")), produced by
validation/generate_golden_posthoc.py.
"""
import json
import math
import os

import numpy as np
import pytest

from analysis.posthoc_core import DunnTest
from core.lazy_imports import get_scikit_posthocs, get_statsmodels_multitest

_REF = os.path.join(os.path.dirname(__file__), "golden", "references_dunn.json")
with open(_REF) as _fh:
    _DATA = json.load(_fh)
_CASES = _DATA["cases"]
_TOL = _DATA["tol"]


def _ref_by_pair(case):
    return {frozenset(c["groups"]): c["raw_p"] for c in case["comparisons"]}


# ---------------- part 1: raw rank statistic vs R ----------------

@pytest.mark.parametrize("case", _CASES, ids=[c["id"] for c in _CASES])
def test_golden_dunn_raw(case):
    """The app's raw Dunn layer (scikit_posthocs.posthoc_dunn, p_adjust=None --
    the exact call in DunnTest.perform_test) must match PMCMRplus raw p."""
    groups = list(case["groups"])
    samples = {g: list(v) for g, v in case["samples"].items()}
    sp = get_scikit_posthocs()
    raw = sp.posthoc_dunn([samples[g] for g in groups], p_adjust=None)

    ref = _ref_by_pair(case)
    for i, g1 in enumerate(groups):
        for j, g2 in enumerate(groups):
            if i < j:
                got = float(raw.iloc[i, j])
                exp = ref[frozenset([g1, g2])]
                assert math.isfinite(got), f"{case['id']} {g1}-{g2}: raw p not finite"
                assert abs(got - exp) <= _TOL["raw_p"], (
                    f"{case['id']} {g1}-{g2} raw p: scikit={got!r} vs R={exp!r} (tol={_TOL['raw_p']})"
                )


# ---------------- part 2: Holm-Sidak seam / pair wiring ----------------

@pytest.mark.parametrize("case", _CASES, ids=[c["id"] for c in _CASES])
def test_golden_dunn_holm_sidak_seam(case):
    """Push the R-validated raw p-values through the app's own Holm-Sidak stage
    and assert each adjusted p arrives on the correct pair -- i.e. the full app
    pipeline's adjusted p for pair X equals Holm-Sidak of the R raw p that truly
    belongs to X, not to some other pair."""
    groups = list(case["groups"])
    samples = {g: list(v) for g, v in case["samples"].items()}
    ref = _ref_by_pair(case)

    # Reconstruct the app's pair enumeration order (i<j over valid_groups) and
    # feed the R-validated raw p-values through the identical Holm-Sidak call.
    pairs, raws = [], []
    for i, g1 in enumerate(groups):
        for j, g2 in enumerate(groups):
            if i < j:
                pairs.append(frozenset([g1, g2]))
                raws.append(ref[frozenset([g1, g2])])
    multipletests = get_statsmodels_multitest()
    _, expected_adj, _, _ = multipletests(raws, alpha=0.05, method="holm-sidak")
    expected = {p: a for p, a in zip(pairs, expected_adj)}

    result = DunnTest.perform_test(groups, samples, alpha=0.05)
    assert not result.get("error"), f"{case['id']}: {result.get('error')}"
    app = {frozenset([c["group1"], c["group2"]]): c["p_value"] for c in result["pairwise_comparisons"]}

    assert set(app.keys()) == set(expected.keys()), (
        f"{case['id']}: pair set mismatch app={set(app)} expected={set(expected)}"
    )
    # Tolerance 1e-5, not machine-epsilon: the app adjusts its own scikit raw p
    # while `expected` adjusts the frozen R raw p, and those differ by <=1e-6
    # (test_golden_dunn_raw), propagating ~1:1 through Holm-Sidak. This test's
    # job is the pair wiring: a positional mismatch would attach a different
    # pair's adjusted p (off by O(0.1) here), which 1e-5 catches with huge margin.
    for pair in pairs:
        assert app[pair] == pytest.approx(expected[pair], abs=1e-5), (
            f"{case['id']} {sorted(pair)}: app adjusted p={app[pair]!r} != "
            f"Holm-Sidak of the R raw p for this pair={expected[pair]!r} "
            f"(raw->adjusted seam / pair wiring)"
        )


@pytest.mark.parametrize("case", _CASES, ids=[c["id"] for c in _CASES])
def test_dunn_seam_positive_control(case):
    """Positive control: prove the seam test above has teeth.

    Deliberately introduce the exact bug it guards against -- swap two pairs'
    raw p-values before the Holm-Sidak stage -- and confirm the resulting
    adjusted p-values diverge from the app's correct output by O(0.1), five
    orders of magnitude above the 1e-5 seam tolerance (and above the ~1e-7
    scikit-vs-R raw noise). This is the executed counterpart to the reasoned
    claim that a positional mismatch would be O(0.1), not rounding noise."""
    groups = list(case["groups"])
    samples = {g: list(v) for g, v in case["samples"].items()}
    ref = _ref_by_pair(case)

    pairs, raws = [], []
    for i, g1 in enumerate(groups):
        for j, g2 in enumerate(groups):
            if i < j:
                pairs.append(frozenset([g1, g2]))
                raws.append(ref[frozenset([g1, g2])])

    result = DunnTest.perform_test(groups, samples, alpha=0.05)
    app = {frozenset([c["group1"], c["group2"]]): c["p_value"] for c in result["pairwise_comparisons"]}

    # swap the smallest- and largest-raw-p positions, then re-run Holm-Sidak
    lo, hi = raws.index(min(raws)), raws.index(max(raws))
    mutated = list(raws)
    mutated[lo], mutated[hi] = mutated[hi], mutated[lo]
    multipletests = get_statsmodels_multitest()
    _, mutated_adj, _, _ = multipletests(mutated, alpha=0.05, method="holm-sidak")

    max_div = max(abs(mutated_adj[k] - app[pairs[k]]) for k in range(len(pairs)))
    assert max_div > 0.05, (
        f"{case['id']}: a swapped-pair mismatch moved the adjusted p by only "
        f"{max_div:.2e} -- the seam test (tol 1e-5) could not distinguish it from "
        f"noise, so it would not catch a real positional bug"
    )
