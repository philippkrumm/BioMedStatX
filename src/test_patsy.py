import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

df = pd.DataFrame({
    'my-var': [1, 2, 1, 2, 1, 2],
    'group-a': ['A', 'A', 'B', 'B', 'A', 'B'],
    'group-b': ['X', 'Y', 'X', 'Y', 'Y', 'X']
})
model = ols("Q('my-var') ~ C(Q('group-a'), Sum) * C(Q('group-b'), Sum)", data=df).fit()
aov = sm.stats.anova_lm(model, typ=3)
print(aov.index)
