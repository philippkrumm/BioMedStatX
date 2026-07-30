import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import pandas as pd
from analysis.analysis_core import AnalysisManager

overlapping = {
    "Responder": [0, 1]*4,
    "Age": [1, 1, 2, 2, 3, 3, 4, 4],
    "Biomarker": [1.1, 2.1, 1.2, 2.2, 1.3, 2.3, 1.4, 2.4]
}

non_overlapping = {
    "Responder": [1]*10 + [0]*12,
    "Age": [5, 5, 6, 6, 7, 7, 8, 8, 9, 9] + list(range(10, 22)),
    "Biomarker": [3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0] + [5.0 + i*0.1 for i in range(12)]
}

df_over = pd.DataFrame(overlapping)
df_non = pd.DataFrame(non_overlapping)
df = pd.concat([df_over, df_non], ignore_index=True)
df.to_excel("dummy.xlsx", index=False)

config = {
    "inferred_test": "lmm",
    "group_col": "Responder",
    "value_cols": ["Biomarker"],
    "subject_column": "Age",
    "dependent": True,
    "group_labels": [0, 1],
    "factor_columns": ["Responder"],
    "between_factors": ["Responder"],
    "within_factors": [],
    "mode": "single",
    "injected_df": df
}

result = AnalysisManager.analyze(
    file_path="dummy.xlsx",
    group_col=config["group_col"],
    groups=config["group_labels"],
    value_cols=config["value_cols"],
    dependent=config["dependent"],
    subject_column=config["subject_column"],
    test=config["inferred_test"],
    analysis_context=config,
    save_plot=False,
    skip_plots=True,
    file_name="dummy_out"
)

print(f"Total Subjects: {df['Age'].nunique()}")
print(f"Total Observations: {len(df)}")

model_obs = result.get("n_observations")
model_subj = result.get("n_subjects")
print(f"Model N-observations: {model_obs}, Model N-subjects: {model_subj}")

print("Raw Data Vault (Plot Ns):")
total_plot_n = 0
for group, vals in result.get("raw_data", {}).items():
    print(f"  Group {group}: N={len(vals)}")
    total_plot_n += len(vals)
print(f"Total Plot Observations: {total_plot_n}")
