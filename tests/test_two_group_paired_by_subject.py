"""Paired t-test / Wilcoxon paired by row position, not by subject ID.

``_prepare_contextual_inputs`` builds ``samples[group]`` in row order, and
``validate_paired_data`` then zips the two groups positionally. So the pairing
was correct only when both groups happened to list their subjects in the same
order. Reorder the sheet -- a user sorting by the measured value, say -- and
subject i in group A gets paired with a different subject in group B.

Audit demonstration (seed 101, N=16): the ``B_block_reversed`` permutation
returned t=-1.35 / p=0.196, byte-identical to ``ttest_rel(va, vb[::-1])`` --
the positional-zip signature -- instead of the correct t=-5.90 / p=2.94e-05.

These tests assert BOTH properties the HC3 fix insisted on: row-order
invariance (same answer regardless of sheet order) AND correctness against an
independent oracle (scipy on subject-aligned arrays), with a positive control
that destroying the pairing actually moves the result.
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp_stats

from analysis.analysis_core import AnalysisManager


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
    try:
        from analysis.statisticaltester import UIDialogManager
        # "skip" = continue without transformation (non-parametric); None now
        # means the user cancelled the dialog, which aborts the whole analysis.
        UIDialogManager.select_transformation_dialog = staticmethod(lambda *a, **k: "skip")
        for name in ("select_posthoc_test_dialog", "select_nonparametric_posthoc_dialog",
                     "select_control_group_dialog", "select_custom_pairs_dialog"):
            setattr(UIDialogManager, name, staticmethod(lambda *a, **k: None))
    except Exception:
        pass
    yield app


@pytest.fixture
def dummy_file(tmp_path):
    path = tmp_path / "dummy.xlsx"
    pd.DataFrame({"a": [1]}).to_excel(path, index=False)
    return str(path)


def _run(long_df, dummy_file, tmp_path, tag):
    ctx = {
        "injected_df": long_df.reset_index(drop=True), "factor_columns": ["Group"],
        "between_factors": [], "dv_columns": ["Value"], "group_labels": ["A", "B"],
        "mode": "single", "dependent": True, "subject_column": "Subject",
    }
    return AnalysisManager.analyze(
        file_path=dummy_file, group_col="Group", groups=["A", "B"],
        value_cols=["Value"], save_plot=False, skip_plots=True, dependent=True,
        file_name=str(tmp_path / tag), analysis_context=ctx, subject_column="Subject",
    )


def _ttest_fixture():
    """Audit fixture: seed 101, N=16."""
    N = 16
    rng = np.random.default_rng(101)
    subj = np.arange(N)
    base = rng.normal(10, 3, N)
    va = base + rng.normal(0, 0.7, N)
    vb = base + 1.4 + rng.normal(0, 0.7, N)
    long = pd.DataFrame({
        "Subject": np.concatenate([subj, subj]),
        "Group": ["A"] * N + ["B"] * N,
        "Value": np.concatenate([va, vb]),
    })
    return long, va, vb


def _wilcoxon_fixture():
    """Audit fixture: seed 202, N=18, heavy tails so the rank test is chosen."""
    N = 18
    rng = np.random.default_rng(202)
    subj = np.arange(N)
    base = rng.lognormal(1.0, 1.1, N)
    va = base * rng.lognormal(0, 0.25, N)
    vb = base * 2.2 * rng.lognormal(0, 0.25, N)
    va[0], vb[1] = 900.0, -700.0
    long = pd.DataFrame({
        "Subject": np.concatenate([subj, subj]),
        "Group": ["A"] * N + ["B"] * N,
        "Value": np.concatenate([va, vb]),
    })
    return long, va, vb


def _permutations(long, N):
    rs = np.random.default_rng(7).permutation(len(long))
    inter_order = np.argsort(np.concatenate([np.arange(N) * 2, np.arange(N) * 2 + 1]),
                             kind="stable")
    return {
        "identity": long.copy(),
        "reversed": long.iloc[::-1].copy(),
        "random": long.iloc[rs].copy(),
        "interleaved": long.iloc[inter_order].copy(),
        # the permutation the audit used to expose the positional zip
        "B_block_reversed": pd.concat([long.iloc[:N], long.iloc[N:].iloc[::-1]]).copy(),
    }


def test_paired_ttest_is_row_order_invariant_and_correct(dummy_file, tmp_path):
    long, va, vb = _ttest_fixture()
    N = 16
    oracle_t, oracle_p = sp_stats.ttest_rel(va, vb)  # subjects are aligned by construction

    results = {}
    for name, permuted in _permutations(long, N).items():
        r = _run(permuted, dummy_file, tmp_path, f"pt_{name}")
        results[name] = (r.get("statistic"), r.get("p_value"))

    ref = results["identity"]
    for name, (stat, p) in results.items():
        assert stat == pytest.approx(ref[0], rel=0, abs=1e-12), f"{name} not row-order invariant"
        assert p == pytest.approx(ref[1], rel=0, abs=1e-12), f"{name} not row-order invariant"

    # correctness, not just invariance
    assert ref[0] == pytest.approx(float(oracle_t), rel=1e-9)
    assert ref[1] == pytest.approx(float(oracle_p), rel=1e-9)


def test_wilcoxon_is_row_order_invariant_and_correct(dummy_file, tmp_path):
    long, va, vb = _wilcoxon_fixture()
    N = 18
    oracle_stat, oracle_p = sp_stats.wilcoxon(va, vb, zero_method="pratt", method="exact")

    results = {}
    for name, permuted in _permutations(long, N).items():
        r = _run(permuted, dummy_file, tmp_path, f"wx_{name}")
        results[name] = (r.get("test"), r.get("statistic"), r.get("p_value"))

    ref = results["identity"]
    assert "Wilcoxon" in str(ref[0])
    for name, (_test, stat, p) in results.items():
        assert stat == pytest.approx(ref[1], rel=0, abs=1e-12), f"{name} not invariant"
        assert p == pytest.approx(ref[2], rel=0, abs=1e-12), f"{name} not invariant"

    assert ref[1] == pytest.approx(float(oracle_stat), rel=1e-9)
    assert ref[2] == pytest.approx(float(oracle_p), rel=1e-9)


def test_positive_control_destroying_pairing_moves_the_result(dummy_file, tmp_path):
    """Scramble which subject each B value belongs to; the IDs stay put. A
    subject-aware pipeline must now compute a different (wrong-on-purpose)
    answer -- otherwise the invariance test above proves nothing."""
    long, va, vb = _ttest_fixture()
    N = 16
    ref = _run(long, dummy_file, tmp_path, "pc_intact")

    scrambled = long.copy()
    b_mask = scrambled["Group"] == "B"
    vb_scrambled = vb[np.random.default_rng(555).permutation(N)]
    scrambled.loc[b_mask, "Value"] = vb_scrambled
    broken = _run(scrambled, dummy_file, tmp_path, "pc_broken")

    assert broken.get("p_value") != pytest.approx(ref.get("p_value"), rel=1e-6), (
        "destroying the pairing did not change the result — the test has no power"
    )
    # and the intact run still matches the oracle
    assert ref.get("p_value") == pytest.approx(
        float(sp_stats.ttest_rel(va, vb).pvalue), rel=1e-9)
