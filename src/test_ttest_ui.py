import sys
from PyQt6.QtWidgets import QApplication
from analysis.analysis_core import BioMedStatXAnalyzer
import pandas as pd
import numpy as np

app = QApplication(sys.argv)
np.random.seed(42)
df = pd.DataFrame({
    'Group': ['A']*20 + ['B']*20,
    'Value': np.random.randn(40) + np.repeat([0, 4.6], 20)
})
analyzer = BioMedStatXAnalyzer()
analyzer.load_data(df, "test_df")
results = analyzer.run_analysis(
    dataset_name="test_df",
    dependent_var="Value",
    factors=["Group"],
    design_type="Independent Groups",
    force_non_parametric=False
)

from export.html_exporter import HTMLExporter
exporter = HTMLExporter()
html = exporter.generate_report(results.model_dump(), "test_df", "Single")
with open("test_report.html", "w") as f:
    f.write(html)
print("Report generated: test_report.html")
