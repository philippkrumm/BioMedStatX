import pandas as pd
import json
from src.analysis.analysis_core import AnalysisManager

file_path = "assets/BioMedStatX_Excel_Template.xlsx"
sheet_name = "Mixed ANOVA"
df = pd.read_excel(file_path, sheet_name=sheet_name)
print(df.head())
print("Columns:", df.columns.tolist())

analysis_context = {
    "dv_columns": ["Dependent Variable"],
    "factor_columns": ["Factor 1", "Factor 2"],
    "subject_col": "Subject ID", # Wait, what does it expect?
}

results = AnalysisManager.auto_pilot(
    file_path=file_path,
    sheet_name=sheet_name,
    group_col="Subject ID", # wait, let's see how the GUI passes this.
    value_cols=["Dependent Variable"],
    groups=[],
    analysis_context=analysis_context
)
print("TEST RESULT:", results["results"][0].get("test", "NO TEST KEY"))
print("ERROR:", results["results"][0].get("error"))
for res in results.get("results", []):
    print(res.get("test"))
    print(res.get("effect_table"))
    print(res.get("descriptives_table"))

