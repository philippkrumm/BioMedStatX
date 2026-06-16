import sys
sys.path.insert(0, "src")
import json
import pandas as pd
import pingouin as pg

_DATA = json.load(open("tests/golden/references_r_advanced.json"))
_DF = pd.DataFrame(_DATA["data"])

aov = pg.mixed_anova(data=_DF, dv="y_mixed", within="time", between="groupA", subject="subj")
print(aov)
