---
title: "Correlation & Regression Guide"
author: "BioMedStatX"
lang: en
geometry: "margin=1in"
mainfont: "DejaVu Serif"
fontsize: 11pt
---

# Correlation & Regression Guide

This guide explains how to configure and interpret **Correlation Analysis**, **Linear Regression (OLS)**, and the **Exploratory Correlation Matrix** in BioMedStatX. These analyses are triggered automatically by the Auto-pilot when a continuous variable is placed in the Factor 1 bucket.

> Example Excel template: `docs/StatisticalAnalyzer_Excel_Template.xlsx`

---

## Overview

| Analysis | Triggered when |
|---|---|
| **Correlation (Pearson/Spearman)** | Factor 1 = continuous, no Covariates |
| **Linear Regression (OLS)** | Factor 1 = continuous, Covariates bucket populated |
| **Exploratory Matrix** | Dedicated button in Auto-pilot panel |

**What makes a variable "continuous"?** BioMedStatX classifies a numeric column as continuous when it has more than 10 unique values. Columns with 10 or fewer unique values are treated as categorical (grouping factors).

---

## 1. When Pearson vs. Spearman?

BioMedStatX selects the correlation method automatically:

1. Shapiro-Wilk normality test is run on both variables (using the valid pairs after pairwise deletion).
2. If **both** pass normality (p > 0.05) → **Pearson r** (parametric, assumes bivariate normality).
3. If **at least one** fails normality → **Spearman ρ** (rank-based, assumption-free).

You can override this in the Exploratory Matrix dialog by selecting `Pearson` or `Spearman` explicitly.

### When to override auto-selection

- Small n (< 30): Shapiro-Wilk has low power. Prefer Spearman as a conservative choice.
- Known non-linear monotonic relationship: Use Spearman regardless of normality.
- Pre-registered analysis: Specify the method in advance and override auto-selection.

---

## 2. Bucket Configuration for Correlation

```
Dependent Variable:  Outcome column (numeric, e.g., NK cell count post-OP)
Factor 1:            Continuous predictor (e.g., miRNA-21 expression)
Covariates:          (empty — adding anything here triggers Regression instead)
Filter (optional):   Restrict to a subgroup (e.g., OP-Group = 1)
```

### Data structure example

```
PatientID | miRNA_21 | NK_cells_post | OP_Group
P001      | 2.34     | 1450          | 1
P002      | 1.87     | 1820          | 1
P003      | 3.12     | 1100          | 1
...
```

Drop `miRNA_21` → Factor 1, `NK_cells_post` → Dependent Variable.
Optionally drop `OP_Group` → Filter, select value `1`.

### Output (Correlation sheet in Excel)

| Column | Description |
|---|---|
| r / ρ | Correlation coefficient (−1 to +1) |
| p-value | Two-tailed significance |
| CI_lower / CI_upper | 95% confidence interval (Fisher z-transform) |
| n | Valid pairs after pairwise deletion |
| Method | Pearson or Spearman |
| Interpretation | Direction + strength label |

**Interpreting r / ρ:**

| |r| | Strength |
|---|---|
| 0.00 – 0.19 | Negligible |
| 0.20 – 0.39 | Weak |
| 0.40 – 0.59 | Moderate |
| 0.60 – 0.79 | Strong |
| 0.80 – 1.00 | Very strong |

---

## 3. Bucket Configuration for Linear Regression

Adding variables to the **Covariates** bucket switches the analysis from Correlation to OLS Regression.

```
Dependent Variable:  Outcome (e.g., NK cell count post-OP)
Factor 1:            Primary predictor (e.g., Pump time [min])
Covariates:          Confounders to control (e.g., Age, BMI, Baseline NK cells)
Filter (optional):   Subgroup restriction
```

**Simple regression** (Factor 1 only, no covariates): Set only Factor 1 — but note that the system will then choose Correlation, not Regression. To force Simple Regression, add at least one covariate.

**Multiple regression** (Factor 1 + covariates): All covariates enter the OLS model simultaneously as additional predictors.

### Data structure example

```
PatientID | Pump_time | Age | Baseline_NK | NK_cells_post | OP_Group
P001      | 87        | 62  | 1600        | 1450          | 1
P002      | 120       | 71  | 1400        | 1100          | 1
...
```

Drop `Pump_time` → Factor 1, `NK_cells_post` → Dependent Variable,
`Age` and `Baseline_NK` → Covariates, `OP_Group` → Filter (value = 1).

### Output (LinearRegression sheet in Excel)

**Model Summary:**

| Statistic | Description |
|---|---|
| R² | Proportion of variance explained |
| R² adjusted | R² penalised for number of predictors |
| F-statistic | Overall model significance |
| p(F) | p-value for F-test |
| AIC / BIC | Information criteria (lower = better fit, for model comparison) |
| n | Observations after listwise deletion |

**Coefficient Table:**

| Column | Description |
|---|---|
| Variable | Predictor name |
| Beta | Unstandardised regression coefficient |
| SE | Standard error of Beta |
| t | t-statistic (Beta / SE) |
| p | Two-tailed p-value |
| CI_lower / CI_upper | 95% confidence interval for Beta |

**Interpreting Beta:** A Beta of −2.5 for `Pump_time` means: for every additional minute of pump time, the outcome decreases by 2.5 units on average, holding all other predictors constant.

---

## 4. Residual Diagnostics

BioMedStatX automatically runs three diagnostic tests after every regression:

### 4a. Shapiro-Wilk on residuals (Normality)

Checks whether the model residuals follow a normal distribution. OLS inference (t-tests on coefficients, confidence intervals) is most reliable when residuals are normal.

- **Pass (p > 0.05):** Residuals are consistent with normality. Standard CIs and p-values are valid.
- **Fail (p ≤ 0.05):** Non-normal residuals. Consider: log-transforming the outcome, checking for outliers, or using bootstrapped CIs.

### 4b. Breusch-Pagan Test (Homoscedasticity)

Checks whether residual variance is constant across fitted values (homoscedasticity). Heteroscedasticity inflates or deflates standard errors.

- **Pass (p > 0.05):** Variance is homoscedastic. Standard errors are reliable.
- **Fail (p ≤ 0.05):** Heteroscedasticity detected. Consider: robust (HC3) standard errors, or transforming the outcome.

### 4c. Ramsey RESET Test (Linearity)

Checks whether the linear functional form is correctly specified, by testing whether higher-order terms (fitted²) improve the model.

- **Pass (p > 0.05):** Linear model is well-specified.
- **Fail (p ≤ 0.05):** Non-linearity detected. Consider: adding polynomial terms, log-transforming a predictor, or using a different model.

### Overall diagnostic interpretation

| Normality | Homoscedasticity | Linearity | Recommendation |
|---|---|---|---|
| Pass | Pass | Pass | Results reliable — interpret coefficients directly |
| Fail | Pass | Pass | Check for outliers; consider outcome transformation |
| Pass | Fail | Pass | Use robust standard errors (HC3) |
| Pass | Pass | Fail | Add polynomial terms or transform predictors |
| Multiple fail | — | — | Review model specification carefully before reporting |

---

## 5. Exploratory Correlation Matrix

The matrix dialog computes all pairwise correlations among a set of numeric variables and corrects for multiple testing.

### When to use

- Hypothesis generation: find unexpected associations in a dataset before confirmatory testing.
- Data quality checks: identify highly correlated predictors (multicollinearity) before regression.
- Overview: quickly visualise the correlation structure of a multi-variable dataset.

### Options explained

**Pairwise vs. Listwise deletion:**

| Mode | Behaviour | When to prefer |
|---|---|---|
| **Pairwise** | Each pair uses all rows where both columns are non-missing | More statistical power; n varies per pair |
| **Listwise** | Only rows complete for all selected variables | Comparable n across all pairs; loses data |

For datasets with scattered missing values (e.g., multiple biomarkers with different assay failures), **Pairwise** is almost always preferred. The n-matrix in the output makes data loss transparent.

**Multiple testing correction:**

| Method | Controls | When to use |
|---|---|---|
| **FDR (Benjamini-Hochberg)** | False Discovery Rate | Exploratory analyses — balanced power/specificity |
| **Bonferroni** | Family-wise Error Rate | Few pre-specified tests — strict Type I error control |
| **None** | — | Descriptive overview only; do not report as confirmatory |

For an exploratory matrix with many variables (e.g., 20 × 20 = 190 pairs), always use FDR or Bonferroni. Uncorrected p-values will produce many false positives by chance.

**Stratification:**

Dropping a categorical column into the Stratify-by field runs the full matrix separately per group. This doubles (or triples, etc.) the output sheets and makes group comparisons of the correlation structure possible.

### Output sheets

| Sheet | Content |
|---|---|
| `Corr_r` | Matrix of r / ρ values |
| `Corr_p_corrected` | Matrix of corrected p-values |
| `Corr_n` | Matrix of n (observations per pair) |

For stratified analyses, each group gets its own set of three sheets.

---

## 6. Filter Bucket in Combination with Regression / Correlation

The Filter bucket restricts the DataFrame **before** any assumption check or model fitting. This means:

- Shapiro-Wilk normality tests use only the filtered rows.
- The reported n reflects the filtered subset.
- The Excel sheet header notes the active filter (column + value).

**Typical workflow for Q4-type analyses** (e.g., "Does pump time predict NK cells in On-Pump patients only?"):

```
Filter:              OP_Group = 1          → restricts to 93 On-Pump rows
Factor 1:            Pump_time             → continuous → triggers Regression
Covariates:          Age, Baseline_NK      → multiple regression
Dependent Variable:  NK_cells_post
```

> **Warning:** Always check the filtered n in the bucket label. If n < 20 for a regression with multiple covariates, the model is underpowered and results should be interpreted cautiously.

---

## 7. Common Configuration Errors

| Error | Symptom | Fix |
|---|---|---|
| Categorical variable in Factor 1, expected continuous | Auto-pilot detects ANOVA instead of Correlation | Use a column with > 10 unique numeric values |
| Covariates bucket empty but Regression expected | Auto-pilot selects Correlation | Add at least one covariate to trigger OLS |
| Filter reduces n below 5 | Analysis aborts with warning | Choose a less restrictive filter value |
| All-missing column in Covariates | Model fails | Remove the column or impute missing values |
| Factor 1 = same column as Covariate | Perfect multicollinearity | Remove the duplicate from one of the buckets |

---

## 8. Expected Excel Output Structure

```
WorkbookName_Results.xlsx
├── Correlation          ← r, p, CI, n, method, interpretation
├── LinearRegression     ← model summary + coefficient table + diagnostics
│    (or)
├── Corr_r               ← exploratory r-matrix
├── Corr_p_corrected     ← FDR/Bonferroni corrected p-matrix
└── Corr_n               ← n-matrix (per pair)
```

For stratified exploratory matrices, sheets are named `Corr_r_GroupA`, `Corr_r_GroupB`, etc.

---

## Related Documentation

- [HowTo.md](./HowTo.md) — step-by-step GUI guide including Sections 15–18
- [ADVANCED_ANOVA_GUIDE.md](./ADVANCED_ANOVA_GUIDE.md) — ANOVA configuration reference
- [../validation/CLINICAL_MODELS_VALIDATION.md](../validation/CLINICAL_MODELS_VALIDATION.md) — numerical benchmarks for all model classes
