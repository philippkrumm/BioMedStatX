"""F2/F3: the correlation 'both_normal' flag lied, and the flowchart believed it.

CorrelationModel.fit builds normality_check with the key 'both_normal' twice in
one dict literal: first as (self._method_used == 'pearson') -- the honest
"did we take the pearson branch" -- then again as both_normal_sw, the raw
Shapiro verdict, which clobbers it (later key wins). The Shapiro value is
already carried separately as 'shapiro_both_normal', so the second write is
pure clobber.

Downstream, flowchartvisualizer computed
    used_pearson = (method == "pearson") or (both_normal is True)
so whenever the engine chose Spearman on skew/kurtosis while Shapiro happened
to pass, the tree highlighted the PEARSON leaf next to a "Correlation
(Spearman)" headline.

Seed 7, n=25: skew_x ~ 1.03 -> engine picks Spearman, but Shapiro passes on
both variables -> the two verdicts disagree, exposing the bug.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from analysis.correlation_models import CorrelationModel
from visualization.flowchartvisualizer import FlowchartVisualizer


def _disagreement_fixture():
    """Engine chooses Spearman (skew>1) while Shapiro says both normal."""
    rng = np.random.default_rng(7)
    n = 25
    return pd.DataFrame({"X": rng.lognormal(0, 0.55, n), "Y": rng.normal(0, 1, n)})


def _fit():
    model = CorrelationModel()
    model.fit(_disagreement_fixture(), x_col="X", y_col="Y", method="auto")
    return model.as_results_dict()


def test_premise_method_and_shapiro_disagree():
    res = _fit()
    nc = res["normality_check"]
    assert res["method"] == "spearman", "fixture no longer drives the engine to Spearman"
    assert nc["shapiro_both_normal"] is True, "fixture no longer has Shapiro passing"


def test_both_normal_reflects_the_method_decision_not_shapiro():
    res = _fit()
    nc = res["normality_check"]
    # both_normal must describe the branch actually taken (Spearman -> not the
    # both-normal/pearson branch), not the raw Shapiro verdict.
    assert nc["both_normal"] is False, (
        "both_normal still carries the Shapiro verdict that clobbers the method decision"
    )
    # the Shapiro verdict is still available under its own key
    assert nc["shapiro_both_normal"] is True


def test_flowchart_highlights_the_method_that_ran():
    res = _fit()
    tree = FlowchartVisualizer.get_tree_json(res)
    active = {(e["source"], e["target"]) for e in tree["edges"] if e["isActive"]}
    assert ("SKEW_KURT_CHECK", "SPEARMAN") in active, "tree does not highlight the Spearman leaf"
    assert ("SKEW_KURT_CHECK", "PEARSON") not in active, "tree still highlights PEARSON while Spearman ran"


def test_flowchart_pearson_case_still_highlights_pearson():
    """Positive control: a genuine Pearson result must still light the Pearson leaf."""
    rng = np.random.default_rng(11)
    n = 60
    x = rng.normal(50, 10, n)
    df = pd.DataFrame({"X": x, "Y": 2 * x + rng.normal(0, 8, n)})
    model = CorrelationModel()
    model.fit(df, x_col="X", y_col="Y", method="auto")
    res = model.as_results_dict()
    assert res["method"] == "pearson"
    tree = FlowchartVisualizer.get_tree_json(res)
    active = {(e["source"], e["target"]) for e in tree["edges"] if e["isActive"]}
    assert ("SKEW_KURT_CHECK", "PEARSON") in active
    assert ("SKEW_KURT_CHECK", "SPEARMAN") not in active
