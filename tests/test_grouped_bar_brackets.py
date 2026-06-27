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
