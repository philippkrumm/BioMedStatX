import pandas as pd
import numpy as np
from analysis.statisticaltester import StatisticalTester

# Create 40 rows dataset
# 20 subjects (10 KO, 10 WT)
# Each subject measured at 0h and 2h
np.random.seed(42)
subjects = [f"S{i:02d}" for i in range(1, 21)] * 2
groups = ["KO"] * 10 + ["WT"] * 10
groups = groups * 2
timepoints = ["0h"] * 20 + ["2h"] * 20
values = np.random.randn(40) + 10

df = pd.DataFrame({"Subject": subjects, "BetweenGrp": groups, "Timepoint": timepoints, "Value": values})

# Force fallback to statsmodels by temporarily deleting pingouin from sys.modules
import sys
sys.modules['pingouin'] = None

res = StatisticalTester._run_mixed_anova(df, "Value", "Subject", ["BetweenGrp"], ["Timepoint"])
import pprint
pprint.pprint(res)
