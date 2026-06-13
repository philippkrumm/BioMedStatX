"""
test_welch_posthoc_dialog.py — Welch one-way ANOVA must offer a post-hoc choice.

Background (audit 2026-06): after a significant Welch's ANOVA the pipeline
auto-filled pairwise_comparisons with Games-Howell, so the
`if not pairwise_comparisons` guard in analysis_core skipped
select_posthoc_test_dialog entirely. The user could never reach Dunnett or
custom pairs — an asymmetry vs the Kruskal-Wallis path, which does offer a
choice. Games-Howell must remain the DEFAULT, but the dialog choice must be
honoured when made.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "src"), str(ROOT / "validation")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _run_welch_oneway(posthoc_choice, control_group=None):
    from conftest import DESIGNS
    from test_all_paths import build_analysis_context

    mgr = MagicMock()
    mgr.select_posthoc_test_dialog.return_value = posthoc_choice
    mgr.select_nonparametric_posthoc_dialog.return_value = "dunn"
    mgr.select_control_group_dialog.return_value = control_group
    mgr.select_custom_pairs_dialog.return_value = []
    mgr.select_transformation_dialog.return_value = None

    with patch("analysis.stats_functions.UIDialogManager", mgr), \
         patch("analysis.statisticaltester.UIDialogManager", mgr):
        from analysis.stats_functions import AnalysisManager
        rng = np.random.default_rng(7)
        df = pd.DataFrame({
            "Group": ["A"] * 12 + ["B"] * 12 + ["C"] * 12,
            "Value": np.concatenate([rng.normal(5, 1, 12), rng.normal(8, 1, 12), rng.normal(6.5, 1, 12)]),
        })
        xlsx = ROOT / "validation" / "_tmp_welch.xlsx"
        df.to_excel(xlsx, index=False)
        d = next(x for x in DESIGNS if x["name"] == "one_way_anova_parametric").copy()
        ctx = build_analysis_context(d, ["A", "B", "C"])
        res = AnalysisManager.analyze(
            file_path=str(xlsx), group_col="Group", groups=["A", "B", "C"], sheet_name=0,
            value_cols=["Value"], dependent=False, skip_plots=True, save_plot=False,
            error_type="sd", file_name=str(ROOT / "validation" / "_tmp_welch_out"),
            analysis_context=ctx,
        )
        xlsx.unlink(missing_ok=True)
        return res, mgr


def test_welch_oneway_offers_dialog():
    res, mgr = _run_welch_oneway("games_howell")
    assert res.get("test") == "Welch's ANOVA"
    assert mgr.select_posthoc_test_dialog.called, "parametric post-hoc dialog must be shown for Welch one-way"


def test_welch_oneway_dunnett_choice_honoured():
    res, _ = _run_welch_oneway("dunnett", control_group="A")
    assert "Dunnett" in (res.get("posthoc_test") or ""), res.get("posthoc_test")


def test_welch_oneway_cancel_means_no_posthoc():
    """Cancelling the dialog declines post-hoc — no fallback is forced."""
    res, _ = _run_welch_oneway(None)  # user cancels the dialog
    assert res.get("pairwise_comparisons", []) == [], res.get("pairwise_comparisons")
    assert "declined" in (res.get("posthoc_test") or "").lower(), res.get("posthoc_test")
