# Advanced ANOVA Configuration Guide

## Overview

This guide explains how to properly configure different types of ANOVA tests in BioMedStatX. Choosing the correct ANOVA type and variable assignments is crucial for obtaining valid statistical results.

## 📊 ANOVA Types Available

### 1. Mixed ANOVA (Between + Within)
### 2. Repeated Measures ANOVA (Within only)  
### 3. Two-Way ANOVA (Between only)

---

## 🔬 Mixed ANOVA (Between + Within)

**Use when:** You have both repeated measurements over time/conditions AND different groups

### Configuration:
```
✅ Select test: Mixed ANOVA (Between + Within)
✅ Dependent variable: Value (your outcome measure)
✅ Subject/ID variable: SubjectID (unique identifier for each participant)
✅ Within factors: Timepoint, Condition, Session (repeated measurements)
✅ Between factors: Group, Treatment, Gender (independent groups)
```

### Example Scenarios:
- **Drug Study:** Measuring blood pressure (Value) before/after treatment (Timepoint) in placebo vs drug groups (Treatment)
- **Learning Study:** Testing performance (Value) across multiple sessions (Session) for different teaching methods (Method)
- **Clinical Trial:** Measuring symptoms (Value) at baseline/follow-up (Time) for different therapies (Therapy)

### Data Structure Example:
```
SubjectID | Group | Timepoint | Value
S001      | Drug  | Pre       | 120
S001      | Drug  | Post      | 110
S002      | Drug  | Pre       | 130
S002      | Drug  | Post      | 115
S003      | Placebo| Pre      | 125
S003      | Placebo| Post     | 123
```

---

## 🔄 Repeated Measures ANOVA (Within only)

**Use when:** You have repeated measurements but ALL participants are in the same group

### Configuration:
```
✅ Select test: Repeated Measures ANOVA (Within only)
✅ Dependent variable: Value (your outcome measure)
✅ Subject/ID variable: SubjectID (unique identifier)
✅ Within factors: Timepoint, Condition, Trial (repeated factor)
✅ Between factors: [LEAVE EMPTY!]
```

### Example Scenarios:
- **Learning Curve:** Testing the same group's performance over multiple weeks
- **Dose Response:** Testing different drug doses on the same participants
- **Cognitive Testing:** Measuring reaction time under different cognitive loads

### Data Structure Example:
```
SubjectID | Timepoint | Value
S001      | Week1     | 85
S001      | Week2     | 92
S001      | Week3     | 98
S002      | Week1     | 78
S002      | Week2     | 85
S002      | Week3     | 91
```

---

## 📈 Two-Way ANOVA (Between only)

**Use when:** You have two independent factors but NO repeated measurements

### Configuration:
```
✅ Select test: Two-Way ANOVA (Between only)
✅ Dependent variable: Value (your outcome measure)
✅ Subject/ID variable: [Can be empty or SubjectID]
✅ Within factors: [LEAVE EMPTY!]
✅ Between factors: Factor1, Factor2 (two independent factors)
```

### Example Scenarios:
- **Treatment × Gender:** Testing treatment effect across different genders (one measurement per person)
- **Diet × Exercise:** Testing weight loss for different diet/exercise combinations
- **Education × Socioeconomic:** Testing academic performance across education levels and socioeconomic status

### Data Structure Example:
```
SubjectID | Treatment | Gender | Value
S001      | DrugA     | Male   | 145
S002      | DrugA     | Female | 132
S003      | DrugB     | Male   | 158
S004      | DrugB     | Female | 141
S005      | Placebo   | Male   | 135
S006      | Placebo   | Female | 128
```

---

## 🎯 Decision Tree: Which ANOVA to Choose?

```
Do you have repeated measurements? 
├── YES → Do you also have different groups?
│   ├── YES → Mixed ANOVA (Between + Within)
│   └── NO → Repeated Measures ANOVA (Within only)
└── NO → Do you have two independent factors?
    ├── YES → Two-Way ANOVA (Between only)
    └── NO → One-Way ANOVA (not covered in advanced tests)
```

---

## ⚠️ Common Configuration Errors

### ❌ **Don't Do This:**

1. **Mixed ANOVA without Subject/ID:**
   - Will treat repeated measurements as independent
   - Violates assumptions and gives wrong results

2. **Repeated Measures with Between factors:**
   - Use Mixed ANOVA instead
   - RM ANOVA is for within-subjects only

3. **Two-Way ANOVA with repeated data:**
   - Will ignore the repeated nature of data
   - Use Mixed or RM ANOVA instead

4. **Wrong factor assignment:**
   - Don't put grouping variables in "Within factors"
   - Don't put time/condition variables in "Between factors"

---

## 📋 Variable Assignment Checklist

### Dependent Variable:
- ✅ Your outcome measure (continuous data)
- ✅ Examples: score, time, concentration, rating

### Subject/ID Variable:
- ✅ Unique identifier for each participant
- ✅ Required for any repeated measures design
- ✅ Examples: SubjectID, ParticipantNumber, PatientID

### Within Factors (Repeated):
- ✅ Variables measured multiple times on same subjects
- ✅ Examples: Time, Session, Condition, Trial, Dose
- ✅ Values: Pre/Post, Week1/Week2/Week3, Low/Medium/High

### Between Factors (Independent):
- ✅ Variables that differ between subjects
- ✅ Examples: Group, Treatment, Gender, Age_Category
- ✅ Values: Control/Treatment, Male/Female, Young/Old

---

## 🔍 Assumptions and Requirements

### All ANOVA Types:
- **Normality:** Residuals should be normally distributed
- **Independence:** Observations should be independent (except for repeated measures)
- **Homogeneity:** Equal variances across groups

### Additional for Repeated Measures:
- **Sphericity:** Equal variances of differences between all pairs of repeated measures
- Tested with Mauchly's test
- Corrected with Greenhouse-Geisser or Huynh-Feldt when violated

### Additional for Mixed ANOVA:
- **Between-factor homogeneity:** Equal variances across between-subjects groups
- **Within-factor sphericity:** For the repeated measures component
- **Compound symmetry:** More stringent assumption than sphericity

---

## 📊 Example Outputs to Expect

### Main Effects:
- **Significant main effect of Time:** Change occurs over time/conditions
- **Significant main effect of Group:** Groups differ overall
- **Non-significant main effect:** No overall difference for that factor

### Interactions:
- **Significant interaction:** The effect of one factor depends on the level of another
- **Non-significant interaction:** Factors act independently

### Post-hoc Tests:
- **Between-subjects:** Tukey HSD, Dunnett tests
- **Within-subjects:** Paired t-tests with correction
- **Mixed comparisons:** Combination of both approaches

---

## 🎯 Best Practices

1. **Plan your analysis before data collection**
2. **Check your data structure matches your ANOVA type**
3. **Verify all assumptions before interpreting results**
4. **Use appropriate post-hoc tests for significant effects**
5. **Report effect sizes along with p-values**
6. **Consider corrections for multiple comparisons**


*This guide is part of the BioMedStatX documentation. For technical issues or feature requests, please visit our GitHub repository.*