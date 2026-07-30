import pandas as pd
import numpy as np
from analysis.statisticaltester import StatisticalTester

np.random.seed(42)
subjects = [f"S{i:02d}" for i in range(1, 21)] * 2
groups = ["KO"] * 10 + ["WT"] * 10
groups = groups * 2
timepoints = ["0h"] * 20 + ["2h"] * 20
values = np.random.randn(40) + 10

df = pd.DataFrame({"Subject": subjects, "BetweenGrp": groups, "Timepoint": timepoints, "Value": values})

res = StatisticalTester._run_two_way_anova(df, "Value", ["BetweenGrp", "Timepoint"])
import pprint
pprint.pprint(res)
