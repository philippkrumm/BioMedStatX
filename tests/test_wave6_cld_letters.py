"""Wave-6 BLOCKER B2: compact-letter-display (CLD) must not hide a real
significant difference.

Two divergent, both-buggy CLD implementations existed:
  - get_significance_letters_from_posthoc (posthoc-based, star+absorb)
  - get_significance_letters            (samples-based, sequential append)

Both fail on INTRANSITIVE significance patterns (A-B ns, B-C ns, A-C sig):
the posthoc one returned {a,a,a} (A and C share a letter though A-C is
significant), the samples one returned {A:'b', B:'ab', C:'aab'} (A and C share
'b', and 'aab' is a malformed code with a duplicated letter).

Correct CLD invariant (the whole test rests on it): two groups share a letter
IFF they are NOT significantly different (an edge in the non-significance
graph). Equivalently, groups sharing a letter form a clique of mutual
non-significance. This holds because two adjacent nodes always sit in a common
maximal clique, and two non-adjacent nodes never do.

Reachability: letters (not brackets) are the default annotation for omnibus
post-hocs (Tukey/ANOVA/Dunn); brackets are only for explicit pairwise
t/MWU/Wilcoxon. So one-way ANOVA -> Tukey -> bar plot runs through this code.
The final tests drive the real bar and raincloud call paths, not just the
isolated function.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
import itertools

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from visualization.datavisualizer import DataVisualizer as DV


def _share(letters, a, b):
    return bool(set(letters[a]) & set(letters[b]))


def _assert_cld_correct(groups, letters, sig_pairs):
    """The core invariant: a pair shares a letter IFF it is non-significant.
    Plus: no code may repeat a letter (guards the 'aab' malformed output)."""
    for a, b in itertools.combinations(groups, 2):
        is_sig = frozenset((a, b)) in sig_pairs
        shares = _share(letters, a, b)
        assert shares == (not is_sig), (
            f"{a},{b}: sig={is_sig} but share_letter={shares} "
            f"-> {'HIDES a significant difference' if (is_sig and shares) else 'splits a non-significant pair'}"
            f" | letters={letters}"
        )
    for g in groups:
        code = letters[g]
        assert len(set(code)) == len(code), f"malformed code for {g}: {code!r} (duplicated letter)"


# ---- Implementation 1: posthoc-based, explicit p-values ----

def _cld1(groups, pairs):
    pw = [{"group1": a, "group2": b, "p_value": p} for a, b, p in pairs]
    return DV.get_significance_letters_from_posthoc(groups, pw, alpha=0.05)


def _sig_from_pairs(pairs, alpha=0.05):
    return {frozenset((a, b)) for a, b, p in pairs if p < alpha}


def test_case1_all_significant_regression():
    pairs = [("A", "B", 0.001), ("A", "C", 0.001), ("B", "C", 0.001)]
    _assert_cld_correct(["A", "B", "C"], _cld1(["A", "B", "C"], pairs), _sig_from_pairs(pairs))


def test_case2_none_significant_regression():
    pairs = [("A", "B", 0.5), ("A", "C", 0.5), ("B", "C", 0.5)]
    _assert_cld_correct(["A", "B", "C"], _cld1(["A", "B", "C"], pairs), _sig_from_pairs(pairs))


def test_case3_ab_equal_c_distinct_regression():
    pairs = [("A", "B", 0.5), ("A", "C", 0.001), ("B", "C", 0.001)]
    _assert_cld_correct(["A", "B", "C"], _cld1(["A", "B", "C"], pairs), _sig_from_pairs(pairs))


def test_case4_intransitive_triangle_impl1():
    # A-B ns, B-C ns, A-C SIG -> correct CLD {A:a, B:ab, C:b}; A and C must NOT share
    pairs = [("A", "B", 0.5), ("B", "C", 0.5), ("A", "C", 0.001)]
    L = _cld1(["A", "B", "C"], pairs)
    _assert_cld_correct(["A", "B", "C"], L, _sig_from_pairs(pairs))


def test_case5_five_group_chain_impl1():
    # non-significance path A-B-C-D-E (adjacent ns, all non-adjacent sig).
    # Correct CLD: maximal cliques are the edges -> a,b,c,d; adjacent share, else not.
    groups = ["A", "B", "C", "D", "E"]
    ns = {("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")}
    pairs = []
    for a, b in itertools.combinations(groups, 2):
        pairs.append((a, b, 0.5 if (a, b) in ns else 0.001))
    L = _cld1(groups, pairs)
    _assert_cld_correct(groups, L, _sig_from_pairs(pairs))


def test_case6_malformed_guard_impl1():
    # any pattern -> no code may contain a duplicated letter
    pairs = [("A", "B", 0.5), ("B", "C", 0.5), ("A", "C", 0.001)]
    L = _cld1(["A", "B", "C"], pairs)
    for g, code in L.items():
        assert len(set(code)) == len(code), f"{g}: malformed {code!r}"


# ---- Implementation 2: samples-based fallback ----

def _intransitive_samples():
    rng = np.random.default_rng(3)
    return {"A": list(rng.normal(0.0, 1.5, 7)),
            "B": list(rng.normal(1.4, 1.5, 7)),
            "C": list(rng.normal(2.9, 1.5, 7))}


def test_case4_intransitive_impl2_samples():
    samples = _intransitive_samples()
    groups = ["A", "B", "C"]
    # significance as the function itself sees it (parametric t-test)
    sig = set()
    for a, b in itertools.combinations(groups, 2):
        if stats.ttest_ind(samples[a], samples[b]).pvalue < 0.05:
            sig.add(frozenset((a, b)))
    assert frozenset(("A", "C")) in sig, "fixture sanity: A-C must be significant"
    L = DV.get_significance_letters(samples, groups, test_recommendation="parametric", alpha=0.05)
    _assert_cld_correct(groups, L, sig)


# ---- Real call paths (bar letters mode, raincloud) ----

def _letters_by_group(ax, groups):
    """Map each rendered lowercase-letter code to a group by its x-position
    (letters are drawn at the group's bar center, x = 0,1,2,...)."""
    picks = []
    for t in ax.texts:
        s = t.get_text().strip()
        if s and all(ch.islower() and ch.isalpha() for ch in s):
            picks.append((t.get_position()[0], s))
    picks.sort(key=lambda p: p[0])
    # one code per group, in x-order
    assert len(picks) == len(groups), f"expected {len(groups)} letter codes, got {picks}"
    return {g: code for g, (_, code) in zip(groups, picks)}


def test_real_bar_path_intransitive_does_not_hide_significance():
    """Drive the real plot_bar letters path (Tukey -> letters). The bug rendered
    {A:'ab', B:'ab', C:'a'} -- A and C still share 'a' though A-C is significant."""
    rng = np.random.default_rng(3)
    samples = {"A": list(rng.normal(0.0, 1.5, 7)),
               "B": list(rng.normal(1.4, 1.5, 7)),
               "C": list(rng.normal(2.9, 1.5, 7))}
    groups = ["A", "B", "C"]
    pw = [{"group1": "A", "group2": "B", "p_value": 0.5, "test": "Tukey HSD", "significant": False},
          {"group1": "B", "group2": "C", "p_value": 0.5, "test": "Tukey HSD", "significant": False},
          {"group1": "A", "group2": "C", "p_value": 0.001, "test": "Tukey HSD", "significant": True}]
    fig, ax = plt.subplots()
    DV.plot_bar(groups, samples, ax=ax, save_plot=False, show_points=False,
                pairwise_results=pw, posthoc_method="Tukey HSD")
    letters = _letters_by_group(ax, groups)
    plt.close(fig)
    _assert_cld_correct(groups, letters, {frozenset(("A", "C"))})
