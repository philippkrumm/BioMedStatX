import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication
from analysis.statistical_analyzer import StatisticalAnalyzerApp

app = QApplication(sys.argv)
df = pd.read_excel("../assets/BioMedStatX_Excel_Template.xlsx", sheet_name="Mixed ANOVA")
df.rename(columns={"Subject": "Subject ID", "BetweenGrp": "Factor 2", "Timepoint": "Factor 1", "Value": "Dependent Variable"}, inplace=True)

class MockMain:
    def get_data(self): return df
    def add_tab(self, *args, **kwargs): pass

analyzer = StatisticalAnalyzerApp(MockMain())

class DummyBucket:
    def __init__(self, cols): self.cols = cols
    def get_assigned_columns(self): return self.cols
    def get_assigned_kinds(self): return ["numeric"] * len(self.cols)

analyzer.dv_bucket = DummyBucket(["Dependent Variable"])
analyzer.factor1_bucket = DummyBucket(["Factor 1"])
analyzer.factor2_bucket = DummyBucket(["Factor 2"])
analyzer.subject_bucket = DummyBucket(["Subject ID"])
analyzer.covariates_bucket = DummyBucket([])

class DummyMode:
    def isChecked(self): return False
analyzer.multi_mode_button = DummyMode()
analyzer.analysis_selected_groups = None

ctx = analyzer._ap_build_analysis_context()
print("Inferred test:", ctx.get("inferred_test"))
print("Between factors:", ctx.get("between_factors"))
print("Within factors:", ctx.get("within_factors"))
