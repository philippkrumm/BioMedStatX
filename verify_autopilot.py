import pandas as pd
from src.analysis.analysis_core import AnalysisManager

file_path = "assets/BioMedStatX_Excel_Template.xlsx"
sheet_name = "Mixed ANOVA"

analysis_context = {
    "dv_columns": ["Value"],
    "factor_columns": ["BetweenGrp", "Timepoint"],
    "subject_col": "Subject",
}

# The autopilot is a method on StatisticalAnalyzer, let's just use the AnalysisManager's auto_pilot wrapper
try:
    results = AnalysisManager.auto_pilot(
        file_path=file_path,
        sheet_name=sheet_name,
        group_col="Subject",
        value_cols=["Value"],
        groups=[],
        analysis_context=analysis_context
    )
    print("Auto-pilot result test:", results["results"][0].get("test"))
    print("Inferred test:", results["results"][0].get("model_type"))
except Exception as e:
    import traceback
    traceback.print_exc()
