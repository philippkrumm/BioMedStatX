import pandas as pd
import pingouin as pg

subjects = [f"S{i:02d}" for i in range(1, 21)] * 2
groups = ["KO"] * 10 + ["WT"] * 10
groups = groups * 2
timepoints = ["0h"] * 20 + ["2h"] * 20
values = [1]*40

df = pd.DataFrame({"Subject": subjects, "BetweenGrp": groups, "Timepoint": timepoints, "Value": values})
df["Value"] = pd.to_numeric(df["Value"]) + pd.Series(range(40))/10

aov = pg.mixed_anova(data=df, dv="Value", within="Timepoint", between="BetweenGrp", subject="Subject")
print(aov)
