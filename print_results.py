import sys
sys.path.insert(0, "src")
import json
import pandas as pd
from analysis.statisticaltester import StatisticalTester

_DATA = json.load(open("tests/golden/references_r_advanced.json"))
_DF = pd.DataFrame(_DATA["data"])

r1 = StatisticalTester.perform_advanced_test(_DF, test="two_way_anova", dv="y_car", subject=None, between=["groupA", "groupB"], within=[], force_parametric=True)
print("Two-Way ANOVA interactions:", r1.get("interactions"))

r2 = StatisticalTester.perform_advanced_test(_DF, test="mixed_anova", dv="y_mixed", subject="subj", between=["groupA"], within=["time"], force_parametric=True)
print("Mixed ANOVA interactions:", r2.get("interactions"))
