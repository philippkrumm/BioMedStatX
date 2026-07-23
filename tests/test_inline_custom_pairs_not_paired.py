"""F-B1: a second, unguarded copy of the one-way custom-pairs blocker.

The pre-2.0 fix routed the custom-pairs post-hoc by design
(posthoc_fallback.py: paired = bool(is_dependent)), so an independent one-way
design runs an independent t-test, not a paired one. But
_analyze_single_dataset handles posthoc_choice=="paired_custom" in a SEPARATE
inline block ("directly here to avoid double dialog") that never went through
that engine and still calls scipy.stats.ttest_rel unconditionally. The prior
regression test exercises the engine directly, so it never covered this copy.

Reached through AnalysisManager.analyze: a significant parametric >2-group
result whose post-hoc dialog returns "paired_custom". For independent groups
that pairs observation i of A with observation i of B -- order-dependent, and
raising on unequal n -- exactly the blocker the engine copy already fixed.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest
from scipy import stats as ss

from analysis.analysis_core import AnalysisManager


@pytest.fixture(scope="module", autouse=True)
def _qt_app():
    try:
        from PyQt5.QtWidgets import QApplication, QDialog
    except Exception:
        yield
        return
    app = QApplication.instance() or QApplication([])
    QDialog.exec_ = lambda self, *a, **k: 0
    QDialog.exec = lambda self, *a, **k: 0
    yield app


@pytest.fixture
def dummy_file(tmp_path):
    path = tmp_path / "dummy.xlsx"
    pd.DataFrame({"a": [1]}).to_excel(path, index=False)
    return str(path)


def _stub_custom_pairs(pairs):
    """Force the parametric post-hoc dialog down the inline paired_custom block."""
    from analysis.statisticaltester import UIDialogManager
    UIDialogManager.select_transformation_dialog = staticmethod(lambda *a, **k: None)
    UIDialogManager.select_posthoc_test_dialog = staticmethod(lambda *a, **k: "paired_custom")
    UIDialogManager.select_custom_pairs_dialog = staticmethod(lambda *a, **k: pairs)
    UIDialogManager.select_control_group_dialog = staticmethod(lambda *a, **k: None)


def _three_group_df(b_values):
    """Independent one-way design, clearly significant so post-hoc runs."""
    rng = np.random.default_rng(0)
    a = list(rng.normal(10, 2, 8))
    c = list(rng.normal(16, 2, 8))
    return pd.DataFrame({
        "Group": ["A"] * 8 + ["B"] * 8 + ["C"] * 8,
        "Value": a + list(b_values) + c,
    }), a


def _run(df, dummy_file, tmp_path, tag):
    ctx = {
        "injected_df": df, "factor_columns": ["Group"], "between_factors": ["Group"],
        "dv_columns": ["Value"], "group_labels": ["A", "B", "C"], "mode": "single",
        "dependent": False,
    }
    return AnalysisManager.analyze(
        file_path=dummy_file, group_col="Group", groups=["A", "B", "C"],
        value_cols=["Value"], save_plot=False, skip_plots=True, dependent=False,
        file_name=str(tmp_path / tag), analysis_context=ctx,
    )


def _ab_comparison(results):
    for comp in results.get("pairwise_comparisons") or []:
        pair = {str(comp.get("group1")), str(comp.get("group2"))}
        if pair == {"A", "B"}:
            return comp
    return None


def test_inline_custom_pairs_is_row_order_invariant(dummy_file, tmp_path):
    rng = np.random.default_rng(0)
    b = list(rng.normal(13, 2, 8))
    _stub_custom_pairs([("A", "B")])

    df1, _ = _three_group_df(b)
    r1 = _run(df1, dummy_file, tmp_path, "inline_ident")
    df2, _ = _three_group_df(list(reversed(b)))
    r2 = _run(df2, dummy_file, tmp_path, "inline_rev")

    c1, c2 = _ab_comparison(r1), _ab_comparison(r2)
    assert c1 and c2, "A-vs-B comparison missing"
    assert c1["p_value"] == pytest.approx(c2["p_value"], abs=1e-12), (
        f"p changed when group B was reordered ({c1['p_value']} -> {c2['p_value']}): "
        "computed as if paired"
    )


def test_inline_custom_pairs_matches_independent_ttest(dummy_file, tmp_path):
    rng = np.random.default_rng(0)
    b = list(rng.normal(13, 2, 8))
    _stub_custom_pairs([("A", "B")])
    df, a = _three_group_df(b)
    r = _run(df, dummy_file, tmp_path, "inline_ind")
    comp = _ab_comparison(r)
    assert comp is not None

    ref_ind = ss.ttest_ind(a, b, equal_var=True)
    ref_rel = ss.ttest_rel(a, b)
    assert comp["statistic"] == pytest.approx(ref_ind.statistic, abs=1e-9), (
        f"stat {comp['statistic']} is not the independent t-test {ref_ind.statistic}"
    )
    assert comp["statistic"] != pytest.approx(ref_rel.statistic, abs=1e-9), (
        "still computing the paired t-test on independent groups"
    )


def test_inline_custom_pairs_label_does_not_claim_pairing(dummy_file, tmp_path):
    rng = np.random.default_rng(0)
    b = list(rng.normal(13, 2, 8))
    _stub_custom_pairs([("A", "B")])
    df, _ = _three_group_df(b)
    r = _run(df, dummy_file, tmp_path, "inline_label")
    comp = _ab_comparison(r)
    assert "Paired" not in str(comp.get("test")), comp.get("test")
    assert (comp.get("effect_size_type") or "") == "cohen_d"


def test_inline_custom_pairs_survives_unequal_group_sizes(dummy_file, tmp_path):
    """ttest_rel raises on unequal n; an independent comparison must not."""
    rng = np.random.default_rng(1)
    b = list(rng.normal(13, 2, 11))  # 11 vs 8 -> ttest_rel would raise
    _stub_custom_pairs([("A", "B")])
    df = pd.DataFrame({
        "Group": ["A"] * 8 + ["B"] * 11 + ["C"] * 8,
        "Value": list(rng.normal(10, 2, 8)) + b + list(rng.normal(16, 2, 8)),
    })
    ctx = {
        "injected_df": df, "factor_columns": ["Group"], "between_factors": ["Group"],
        "dv_columns": ["Value"], "group_labels": ["A", "B", "C"], "mode": "single",
        "dependent": False,
    }
    r = AnalysisManager.analyze(
        file_path=dummy_file, group_col="Group", groups=["A", "B", "C"],
        value_cols=["Value"], save_plot=False, skip_plots=True, dependent=False,
        file_name=str(tmp_path / "inline_unequal"), analysis_context=ctx,
    )
    comp = _ab_comparison(r)
    assert comp is not None and comp.get("p_value") is not None, r.get("error")
