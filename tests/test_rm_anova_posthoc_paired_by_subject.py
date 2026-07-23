"""RM-ANOVA inline post-hoc must pair subjects by ID, not by row position.

``StatisticalTester._run_repeated_measures_anova`` built its paired-t post-hoc
samples as ``df[df[factor] == level][dv].tolist()`` -- one flat list per within
level, in whatever order the rows happened to arrive -- and handed them to
``perform_dependent_posthoc_tests`` -> ``DependentPostHoc`` -> ``ttest_rel``.
That pairs row *i* of level A with row *i* of level B, so reordering the rows
within a level (a user sorting the sheet, or a block-per-timepoint export)
silently changes which subject is paired with which.

The fix routes the samples through ``_build_rm_aligned_samples`` (pivot +
``sort_values(by=subject)`` + complete-case filter), the same subject-aligned
builder the modern non-parametric fallback already uses.

These tests assert row-order invariance (same post-hoc regardless of sheet
order), correctness against a subject-aligned scipy+Holm-Šidák oracle, and a
positive control proving the pipeline actually reads the subject column.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest
from itertools import combinations
from scipy import stats as sp_stats
from statsmodels.stats.multitest import multipletests

from analysis.statisticaltester import StatisticalTester

LEVELS = ["T1", "T2", "T3"]
N = 8
ALPHA = 0.05


@pytest.fixture(scope="module", autouse=True)
def _qt_and_dialogs():
    try:
        from PyQt5.QtWidgets import QApplication, QDialog
    except Exception:
        yield
        return
    app = QApplication.instance() or QApplication([])
    QDialog.exec_ = lambda self, *a, **k: 0
    QDialog.exec = lambda self, *a, **k: 0
    yield app


def _fixture():
    """8 subjects x 3 timepoints, block-per-timepoint, subjects ascending."""
    rng = np.random.default_rng(303)
    subj = [f"S{i}" for i in range(N)]
    base = rng.normal(10.0, 3.0, N)          # subject baseline (repeated-measures signal)
    effect = {"T1": 0.0, "T2": 2.0, "T3": 4.0}
    rows = []
    values = {}
    for lvl in LEVELS:
        v = base + effect[lvl] + rng.normal(0.0, 0.5, N)
        values[lvl] = dict(zip(subj, v))
        for s in subj:
            rows.append({"subject": s, "time": lvl, "value": values[lvl][s]})
    return pd.DataFrame(rows), values


def _permutations(long):
    rng = np.random.default_rng(7)
    per_level = []
    for lvl in LEVELS:
        block = long[long["time"] == lvl].iloc[::-1]   # reverse subject order within each level
        per_level.append(block)
    return {
        "identity": long.copy(),
        "reversed": long.iloc[::-1].copy(),
        "random": long.iloc[rng.permutation(len(long))].copy(),
        # the permutation that exposes the positional zip: each level block in a
        # different subject order, subject IDs and values otherwise untouched.
        "per_level_reversed": pd.concat(per_level).copy(),
    }


def _posthoc_p_by_pair(result):
    out = {}
    for c in result.get("pairwise_comparisons", []) or []:
        out[tuple(sorted((str(c["group1"]), str(c["group2"]))))] = c["p_value"]
    return out


def _run(df):
    return StatisticalTester._run_repeated_measures_anova(
        df.reset_index(drop=True), "value", "subject", ["time"], alpha=ALPHA
    )


def _oracle(values):
    """Subject-aligned paired-t per pair, Holm-Šidák corrected -- exactly what
    DependentPostHoc does, but on subjects aligned by ID rather than row order."""
    wide = {lvl: np.array([values[lvl][f"S{i}"] for i in range(N)]) for lvl in LEVELS}
    pairs, raw = [], []
    for a, b in combinations(LEVELS, 2):
        pairs.append(tuple(sorted((a, b))))
        raw.append(sp_stats.ttest_rel(wide[a], wide[b]).pvalue)
    _, p_adj, _, _ = multipletests(raw, alpha=ALPHA, method="holm-sidak")
    return dict(zip(pairs, p_adj))


def test_rm_posthoc_is_row_order_invariant():
    long, _ = _fixture()
    results = {name: _posthoc_p_by_pair(_run(perm)) for name, perm in _permutations(long).items()}
    ref = results["identity"]
    assert ref, "no pairwise comparisons produced"
    for name, byp in results.items():
        assert set(byp) == set(ref), f"{name}: different pairs"
        for pair, p in byp.items():
            assert p == pytest.approx(ref[pair], rel=0, abs=1e-12), (
                f"{name}: pair {pair} p={p} != identity {ref[pair]} "
                "-- post-hoc pairing depends on row order"
            )


def test_rm_posthoc_matches_subject_aligned_oracle():
    long, values = _fixture()
    got = _posthoc_p_by_pair(_run(long))
    oracle = _oracle(values)
    assert set(got) == set(oracle)
    for pair, p in oracle.items():
        assert got[pair] == pytest.approx(float(p), rel=1e-9), f"pair {pair}: {got[pair]} != oracle {p}"


def test_positive_control_reassigning_subjects_moves_the_result():
    """Reassign which subject each T3 value belongs to (IDs stay put, values are
    permuted across subjects within T3). A subject-aware post-hoc must now
    compute a different answer for the T3 pairs -- otherwise invariance is vacuous."""
    long, _ = _fixture()
    ref = _posthoc_p_by_pair(_run(long))

    scrambled = long.copy().reset_index(drop=True)
    mask = scrambled["time"] == "T3"
    vals = scrambled.loc[mask, "value"].to_numpy()
    scrambled.loc[mask, "value"] = vals[np.random.default_rng(555).permutation(N)]
    broken = _posthoc_p_by_pair(_run(scrambled))

    moved = any(
        broken.get(pair) != pytest.approx(ref.get(pair), rel=1e-6)
        for pair in ref if "T3" in pair
    )
    assert moved, "reassigning subjects within T3 did not change the post-hoc -- test has no power"
