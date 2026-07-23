"""The micro-sample correlation flowchart highlighted a disconnected path.

For n < MIN_N_SMALL the builder highlights START->TIER_MICRO and
TIER_MICRO->SPEARMAN, then unconditionally highlights RESULT->CI and
RESULT->EFFECT -- but it never highlights SPEARMAN->RESULT. So RESULT (and the
CI/EFFECT leaves hanging off it) formed a second, disconnected component: the
user saw a path that stopped at SPEARMAN plus a floating result fragment.

The clinical and asymptotic tiers were fine only because they DO highlight
PEARSON->RESULT / SPEARMAN->RESULT, which is what makes RESULT reachable and
the unconditional RESULT->CI/EFFECT lines correct. The micro branch was simply
missing that one edge.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from analysis.analysis_core import AnalysisManager
from visualization.flowchartvisualizer import FlowchartVisualizer


@pytest.fixture
def dummy_file(tmp_path):
    path = tmp_path / "dummy.xlsx"
    pd.DataFrame({"a": [1]}).to_excel(path, index=False)
    return str(path)


def _run_correlation(n, dummy_file, tmp_path, tag):
    """Audit fixture: seed 5, X ~ N(50,10), Y = 2X + noise."""
    rng = np.random.default_rng(5)
    x = rng.normal(50, 10, n)
    y = 2.0 * x + rng.normal(0, 8, n)
    ctx = {
        "injected_df": pd.DataFrame({"X": x, "Y": y}), "x_variable": "X",
        "factor_columns": ["X"], "dv_columns": ["Y"], "mode": "single",
    }
    return AnalysisManager.analyze(
        file_path=dummy_file, group_col="X", groups=[], value_cols=["Y"],
        save_plot=False, skip_plots=True, file_name=str(tmp_path / tag),
        analysis_context=ctx, test="correlation",
    )


def _active_edges(results):
    tree = FlowchartVisualizer.get_tree_json(results)
    assert tree is not None
    return {(e["source"], e["target"]) for e in tree["edges"] if e["isActive"]}


def _disconnected_nodes(active):
    """Nodes touched by an active edge but unreachable from START."""
    reachable = set()
    frontier = ["START"]
    while frontier:
        cur = frontier.pop()
        if cur in reachable:
            continue
        reachable.add(cur)
        frontier += [v for u, v in active if u == cur]
    touched = {n for edge in active for n in edge}
    return touched - reachable


@pytest.mark.parametrize("n,tag", [(12, "micro"), (60, "clinical"), (150, "asymptotic")])
def test_active_path_is_connected_in_every_tier(n, tag, dummy_file, tmp_path):
    active = _active_edges(_run_correlation(n, dummy_file, tmp_path, tag))
    orphans = _disconnected_nodes(active)
    assert not orphans, f"{tag} tier (n={n}) has a disconnected active path: {sorted(orphans)}"
    # the leaves must actually be reachable, not just non-orphaned by accident
    reachable = {n_ for edge in active for n_ in edge} - orphans
    assert {"RESULT", "CI", "EFFECT"} <= reachable


def test_micro_tier_highlights_spearman_to_result(dummy_file, tmp_path):
    """The specific missing edge."""
    active = _active_edges(_run_correlation(12, dummy_file, tmp_path, "micro_edge"))
    assert ("START", "TIER_MICRO") in active
    assert ("TIER_MICRO", "SPEARMAN") in active
    assert ("SPEARMAN", "RESULT") in active, "the edge that made RESULT reachable is missing"
    assert ("RESULT", "CI") in active
    assert ("RESULT", "EFFECT") in active
