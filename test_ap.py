import pandas as pd

df = pd.read_excel("assets/BioMedStatX_Excel_Template.xlsx", sheet_name="Mixed ANOVA")
subject_column = "Subject"
factor_columns = ["Timepoint", "BetweenGrp"]

role_by_factor = {}
for factor in factor_columns:
    per_subject = df.groupby(subject_column)[factor].nunique(dropna=True)
    role_by_factor[factor] = "between" if not per_subject.empty and per_subject.max() <= 1 else "within"

print("Roles:", role_by_factor)

between_factors = [factor for factor, role in role_by_factor.items() if role == "between"]
within_factors = [factor for factor, role in role_by_factor.items() if role == "within"]

print("Between:", between_factors)
print("Within:", within_factors)
