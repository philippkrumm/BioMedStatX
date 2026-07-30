import numpy as np
from analysis.statisticaltester import StatisticalTester
import pandas as pd

np.random.seed(42)
df = pd.DataFrame({
    'Group': ['A']*20 + ['B']*20,
    'Value': np.random.randn(40) + np.repeat([0, 4.6], 20)
})

tester = StatisticalTester()
res = tester.run_test(df, 'Value', 'Group')
print("Keys in results:", res.keys())
print("Confidence interval:", res.get("confidence_interval"))
print("Effect size:", res.get("effect_size"))
