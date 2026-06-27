import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest
import seaborn as sns
from visualization.datavisualizer import DataVisualizer


def _grouped_ax():
    df = pd.DataFrame({
        "Time":  ["T1", "T2", "T3"] * 3,
        "Group": ["Ctrl"] * 3 + ["TrtA"] * 3 + ["TrtB"] * 3,
        "Value": [1.0, 2.0, 3.0, 1.5, 2.5, 4.0, 0.9, 1.8, 2.2],
    })
    fig, ax = plt.subplots()
    sns.barplot(data=df, x="Time", y="Value", hue="Group",
                order=["T1", "T2", "T3"], hue_order=["Ctrl", "TrtA", "TrtB"],
                errorbar=None, ax=ax)
    return ax


def test_grouped_bar_centers_maps_each_cell_to_its_dodged_center():
    ax = _grouped_ax()
    centers = DataVisualizer._grouped_bar_centers(
        ax, between_order=["Ctrl", "TrtA", "TrtB"], within_order=["T1", "T2", "T3"]
    )
    assert len(centers) == 9
    assert centers[("Ctrl", "T1")] < centers[("TrtA", "T1")] < centers[("TrtB", "T1")]
    assert centers[("TrtA", "T1")] == pytest.approx(0.0, abs=1e-6)
    off = centers[("Ctrl", "T1")] - centers[("TrtA", "T1")]
    assert centers[("Ctrl", "T2")] - centers[("TrtA", "T2")] == pytest.approx(off, abs=1e-6)
    assert centers[("TrtA", "T2")] == pytest.approx(1.0, abs=1e-6)


def test_grouped_bar_centers_robust_to_dropout_no_misassignment():
    import numpy as np
    df = pd.DataFrame({
        "Time":  ["T1", "T2", "T3"] * 3,
        "Group": ["Ctrl"] * 3 + ["TrtA"] * 3 + ["TrtB"] * 3,
        "Value": [1.0, 2.0, 3.0, 1.5, 2.5, 4.0, 0.9, 1.8, 2.2],
    })
    df.loc[(df.Time == "T2") & (df.Group == "TrtA"), "Value"] = np.nan  # dropout
    fig, ax = plt.subplots()
    sns.barplot(data=df, x="Time", y="Value", hue="Group",
                order=["T1", "T2", "T3"], hue_order=["Ctrl", "TrtA", "TrtB"],
                errorbar=None, ax=ax)
    centers = DataVisualizer._grouped_bar_centers(
        ax, between_order=["Ctrl", "TrtA", "TrtB"], within_order=["T1", "T2", "T3"])
    assert ("TrtA", "T2") not in centers
    assert len(centers) == 8
    assert centers[("TrtA", "T3")] == pytest.approx(2.0, abs=1e-6)
    assert centers[("Ctrl", "T2")] == pytest.approx(0.733, abs=1e-2)
    assert centers[("TrtB", "T2")] == pytest.approx(1.267, abs=1e-2)


def test_grouped_bracket_positions_only_within_stratum_and_stacked():
    centers = {
        ("Ctrl", "T1"): -0.27, ("TrtA", "T1"): 0.0, ("TrtB", "T1"): 0.27,
        ("Ctrl", "T2"): 0.73, ("TrtA", "T2"): 1.0, ("TrtB", "T2"): 1.27,
    }
    label_map = {
        "Ctrl:T1": ("Ctrl", "T1"), "TrtA:T1": ("TrtA", "T1"), "TrtB:T1": ("TrtB", "T1"),
        "Ctrl:T2": ("Ctrl", "T2"), "TrtA:T2": ("TrtA", "T2"), "TrtB:T2": ("TrtB", "T2"),
    }
    pairwise = [
        {"group1": "Ctrl:T1", "group2": "TrtA:T1", "p_value": 0.2,  "significant": False},
        {"group1": "Ctrl:T1", "group2": "TrtB:T1", "p_value": 0.001, "significant": True},
        {"group1": "Ctrl:T2", "group2": "TrtA:T2", "p_value": 0.04, "significant": True},
    ]
    brackets = DataVisualizer._grouped_bracket_positions(
        centers, label_map, pairwise, y_max=10.0, line_height=0.08
    )
    assert len(brackets) == 3
    for b in brackets:
        assert b["x1"] < b["x2"]
        assert {round(b["x1"], 2), round(b["x2"], 2)} <= {-0.27, 0.0, 0.27, 0.73, 1.0, 1.27}
    t1 = sorted([b["height"] for b in brackets if b["x1"] < 0.5])
    assert t1[0] < t1[1]
