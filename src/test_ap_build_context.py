import pandas as pd
from autopilot.statistical_analyzer_autopilot_pipeline import _sorted_unique

# Mocking the UI class
class MockAutopilotUI:
    def __init__(self, df):
        self.df = df
        
        class MockBucket:
            def __init__(self, cols):
                self.cols = cols
            def get_assigned_columns(self):
                return self.cols
                
        self.dv_bucket = MockBucket(["Value"])
        self.factor1_bucket = MockBucket(["Timepoint"])
        self.factor2_bucket = MockBucket(["BetweenGrp"])
        self.subject_bucket = MockBucket(["Subject"])
        self.covariates_bucket = MockBucket([])
        self.analysis_selected_groups = None
        
        class MockButton:
            def isChecked(self):
                return False
        self.multi_mode_button = MockButton()

# We need to test the logic exactly as it is in `_ap_build_analysis_context`
def run_logic(analysis_df):
    factor_columns = ["Timepoint", "BetweenGrp"]
    subject_column = "Subject"
    
    role_by_factor = {}
    for factor in factor_columns:
        per_subject = analysis_df.groupby(subject_column)[factor].nunique(dropna=True)
        print(f"Factor '{factor}', per_subject.max(): {per_subject.max() if not per_subject.empty else 'Empty'}")
        role_by_factor[factor] = "between" if not per_subject.empty and per_subject.max() <= 1 else "within"
        
    print("Role by factor:", role_by_factor)

df = pd.read_excel("../assets/BioMedStatX_Excel_Template.xlsx", sheet_name="Mixed ANOVA")
run_logic(df)
