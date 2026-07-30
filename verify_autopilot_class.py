import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication

# Mock classes to simulate UI components
class MockBucket:
    def __init__(self, cols): self.cols = cols
    def get_assigned_columns(self): return self.cols

class MockAnalyzer:
    def __init__(self):
        self.df = pd.read_excel("assets/BioMedStatX_Excel_Template.xlsx", sheet_name="Mixed ANOVA")
        self.dv_bucket = MockBucket(["Value"])
        self.factor1_bucket = MockBucket(["Timepoint"])
        self.factor2_bucket = MockBucket(["BetweenGrp"])
        self.subject_bucket = MockBucket(["Subject"])
        self.covariates_bucket = MockBucket([])

# Import the logic function (it's often a standalone function or mixed into the analyzer)
from src.autopilot.statistical_analyzer_autopilot_pipeline import _ap_build_context
