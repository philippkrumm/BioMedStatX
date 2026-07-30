import pandas as pd
import numpy as np
from analysis.analysis_core import AnalysisManager
from PyQt5.QtWidgets import QApplication
import sys

app = QApplication(sys.argv)

def test_mixed_anova_autopilot_flow():
    # Simulate the exact context created by autopilot for a Mixed ANOVA
    np.random.seed(42)
    subjects = [f"S{i:02d}" for i in range(1, 21)] * 2
    groups = ["KO"] * 10 + ["WT"] * 10
    groups = groups * 2
    timepoints = ["0h"] * 20 + ["2h"] * 20
    values = np.random.randn(40) + 10

    df = pd.DataFrame({"Subject ID": subjects, "Factor 2": groups, "Factor 1": timepoints, "Dependent Variable": values})
    df.to_excel("temp_mixed_bug.xlsx", index=False)

    analysis_context = {
        "dv_columns": ["Dependent Variable"],
        "factor_columns": ["Factor 1", "Factor 2"],
        "subject_column": "Subject ID",
        # Simulating the bug: Autopilot somehow inferred two_way_anova
        "inferred_test": "two_way_anova",
        "selected_group_column": None,
        "selected_groups": [],
        "dependent": False,
        "display_group_col": "__AUTO_GROUP__"
    }

    # Simulate how autopilot calls AnalysisManager
    res = AnalysisManager.analyze(
        file_path="temp_mixed_bug.xlsx",
        group_col="__AUTO_GROUP__", 
        groups=[],
        sheet_name="Sheet1",
        value_cols=["Dependent Variable"],
        combine_columns=False,
        dependent=False, compare=False,
        colors=[], hatches=[], title="", x_label="", y_label="",
        file_name="", save_plot=False, skip_plots=True,
        error_type="se", dataset_name="",
        # Crucially, autopilot does NOT pass additional_factors directly in kwargs!
        test="two_way_anova",
        analysis_context=analysis_context,
        subject_column="Subject ID"
    )

    # Check what test was ACTUALLY run by inspecting the results
    assert res.get("test") == "Mixed ANOVA", f"Expected Mixed ANOVA, got {res.get('test')}"
    
    # Ensure factors are labeled 'within' and 'between'
    types = [f.get("type") for f in res.get("factors", [])]
    assert "within" in types and "between" in types, f"Expected within and between factors, got {types}"

if __name__ == "__main__":
    test_mixed_anova_autopilot_flow()
    print("Test passed!")
