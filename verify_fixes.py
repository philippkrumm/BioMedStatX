import sys
import os
sys.path.insert(0, os.path.abspath('src'))
import pandas as pd
import numpy as np
from analysis.analysis_core import AnalysisManager
from export.html_exporter import HTMLExporter

np.random.seed(42)
df = pd.DataFrame({
    'Group': ['A']*20 + ['B']*20,
    'Score': np.random.randn(40) * 2 + 10,
    'Age': np.random.randn(40) * 5 + 40
})

analyzer = AnalysisManager(df)

# Regression (for §4.3 and §4.4 if we add transform)
results_reg = analyzer.run_analysis(
    value_cols=['Score'],
    group_cols=[],
    factor_columns=[],
    covariates=['Age'],
    clinical_test='linear_regression',
    dataset_name="Test Reg",
    analysis_context={"y_transform": "log"}
)

exporter = HTMLExporter({"Test Reg": results_reg}, "Test Report")
html = exporter.generate_report()

print("=== REGRESSION ASSUMPTIONS ===")
lines = html.split('\n')
for i, line in enumerate(lines):
    if "Normality of residuals" in line or "Homoscedasticity" in line or "Linearity (Ramsey RESET)" in line:
        print(line.strip())

print("\n=== TRANSFORMED TEXT DESCRIPTIVE ===")
for line in lines:
    if "Transformed-scale means" in line:
        print(line.strip())

# T-Test (for §3.3 Effect size format and §4.1 Methods text)
results_t = analyzer.run_analysis(
    value_cols=['Score'],
    group_cols=['Group'],
    factor_columns=['Group'],
    clinical_test='auto',
    dataset_name="Test T"
)
exporter_t = HTMLExporter({"Test T": results_t}, "Test Report T")
html_t = exporter_t.generate_report()

print("\n=== EFFECT SIZE FORMAT (3.3) ===")
for line in html_t.split('\n'):
    if "Effect size" in line or "Cohen's d" in line or "Hedges" in line:
        # Just grab the row
        if "<td" in line:
            print(line.strip())

print("\n=== METHODS TEXT EFFECT SIZE (4.1) ===")
for line in html_t.split('\n'):
    if "Hedges" in line and "small" in line:
        print(line.strip())

# 2-level RM-ANOVA (for §4.2 Mauchly test N/A)
df_rm = pd.DataFrame({
    'Subject': [f'S{i}' for i in range(20)] * 2,
    'Time': ['T1']*20 + ['T2']*20,
    'Value': np.random.randn(40)
})
analyzer_rm = AnalysisManager(df_rm)
results_rm = analyzer_rm.run_analysis(
    value_cols=['Value'],
    group_cols=['Time'],
    factor_columns=['Time'],
    subject_col='Subject',
    dependent=True,
    dataset_name="Test RM"
)
exporter_rm = HTMLExporter({"Test RM": results_rm}, "Test Report RM")
html_rm = exporter_rm.generate_report()

print("\n=== MAUCHLY TEST 2-LEVEL (4.2) ===")
for line in html_rm.split('\n'):
    if "Mauchly's test" in line:
        print(line.strip())

