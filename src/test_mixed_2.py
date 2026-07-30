import pandas as pd
import sys
import traceback

class MockDialogManager:
    @staticmethod
    def show_error_message(msg):
        print(f"MOCKED ERROR DIALOG: {msg}")

    @staticmethod
    def select_posthoc_test_dialog(parent, progress_text, column_name, default_method):
        return "pairwise_ttests"

import analysis.stats_functions
analysis.stats_functions.UIDialogManager = MockDialogManager

from analysis.analysis_core import AnalysisManager

file_path = "../assets/BioMedStatX_Excel_Template.xlsx"
sheet_name = "Mixed ANOVA"

analysis_context = {
    "dv_columns": ["Dependent Variable"],
    "factor_columns": ["Factor 1", "Factor 2"],
    "subject_column": "Subject ID",
    "inferred_test": "mixed_anova",
    "between_factors": ["Factor 2"],
    "within_factors": ["Factor 1"],
    "selected_group_column": None,
    "selected_groups": [],
    "dependent": True,
}

df = pd.read_excel(file_path, sheet_name=sheet_name)
df.rename(columns={"Subject": "Subject ID", "BetweenGrp": "Factor 2", "Timepoint": "Factor 1", "Value": "Dependent Variable"}, inplace=True)
df.to_excel("temp_mixed.xlsx", index=False)

res = AnalysisManager.analyze(
    file_path="temp_mixed.xlsx",
    group_col="__AUTO_GROUP__", 
    groups=[],
    sheet_name="Sheet1",
    value_cols=["Dependent Variable"],
    combine_columns=False,
    width=6, height=4,
    dependent=True, compare=False,
    colors=[], hatches=[], title="", x_label="", y_label="",
    file_name="", save_plot=False, skip_plots=True,
    error_type="se", dataset_name="",
    additional_factors=["Factor 1", "Factor 2"],
    show_individual_lines=False,
    test="mixed_anova",
    analysis_context=analysis_context,
    subject_column="Subject ID",
)
print("TEST TYPE:", res.get("test"))
print("ERROR:", res.get("error"))
import pprint
pprint.pprint(res.get("factors"))
