

# Correlation & Regression Guide

This guide covers Correlation, Linear Regression (OLS), ANCOVA, Linear Mixed Models, Logistic Regression, and the Exploratory Correlation Matrix. All analyses are selected based on the Smart Mapping configuration, not by manual test selection.

---

## Which Analysis Runs?

| Analysis | Trigger conditions |
|---|---|
| **Correlation (Pearson/Spearman)** | Factor 1 continuous; Covariates empty; Subject ID empty |
| **Simple Linear Regression** | Factor 1 continuous; Covariates empty; Regression toggle active |
| **Multiple Regression (OLS)** | Factor 1 continuous; Covariates populated |
| **ANCOVA** | Factor 1 categorical; Covariates populated |
| **Linear Mixed Model (LMM)** | Factor 1 continuous; Subject ID assigned |
| **Logistic Regression** | Dependent Variable has exactly 2 distinct values |
| **Exploratory Correlation Matrix** | Accessed via **Analysis → Exploratory Correlation Matrix** |

**Continuous vs. categorical:** A numeric column is classified as continuous when it contains more than 10 unique values. Ten or fewer unique values → categorical. This threshold drives test selection silently. If you see an unexpected design, check the mapping status line.

---

## 1. Correlation Analysis

### Pearson vs. Spearman: the decision

BioMedStatX runs Shapiro–Wilk on both variables using valid pairs after pairwise deletion:

- Both pass ($p > 0.05$): **Pearson $r$** (parametric, assumes bivariate normality).
- At least one fails: **Spearman $\rho$** (rank-based, no distributional assumption).

Pearson $r$ is defined as:

$$r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2 \sum_{i=1}^{n}(y_i - \bar{y})^2}}$$

With $n < 30$, Shapiro–Wilk has low power. Small samples rarely flag non-normality even when it is present. The conservative choice in that situation is Spearman $\rho$, and we recommend defaulting to it for small clinical datasets.

You can override the auto-selection in the Exploratory Correlation Matrix dialog. The main analysis respects the auto-selection unless the Regression toggle is used (see below).

### Smart Mapping configuration

| Bucket | Assign |
|---|---|
| **Dependent Variable** | Numeric outcome (e.g. NK cell count) |
| **Factor 1** | Continuous predictor (e.g. miRNA-21 expression) |
| **Covariates** | Leave empty: any entry here triggers OLS Regression |
| **Subject ID** | Leave empty: any entry here triggers LMM |
| **Filter** | Optional subgroup restriction |

### Correlation → Regression toggle

When Factor 1 is continuous and Covariates is empty, a checkbox appears:

**"Analyse as Linear Regression (Y = a + bX)"**

Unchecked (default) → Correlation. Checked → Simple OLS Regression with one predictor. Use the toggle when you want:
- The slope coefficient $\hat{\beta}_1$ and its 95% CI
- The full residual diagnostic battery (Shapiro–Wilk, Breusch–Pagan, Ramsey RESET)
- A scatter plot with the fitted regression line

### Data structure example

```
PatientID | miRNA_21 | NK_cells_post | OP_Group
P001      | 2.34     | 1450          | 1
P002      | 1.87     | 1820          | 1
P003      | 3.12     | 1100          | 1
```

Assign `miRNA_21` → Factor 1; `NK_cells_post` → Dependent Variable. To restrict to On-Pump patients: assign `OP_Group` → Filter and select `1`.

### HTML report output

| Statistic | Description |
|---|---|
| $r$ or $\rho$ | Correlation coefficient, range $[-1, 1]$ |
| $p$ | Two-tailed; based on $t = r\sqrt{(n-2)/(1-r^2)}$ on $n-2$ degrees of freedom |
| 95% CI | Fisher $z$-transformation: $z = \frac{1}{2}\ln\frac{1+r}{1-r}$; then back-transformed |
| $n$ | Valid pairs after pairwise deletion |
| Method | Pearson or Spearman |
| Interpretation | Strength label |

**Strength conventions** (Cohen, 1988):

| $|r|$ | Label |
|---|---|
| $0.00$–$0.19$ | Negligible |
| $0.20$–$0.39$ | Weak |
| $0.40$–$0.59$ | Moderate |
| $0.60$–$0.79$ | Strong |
| $0.80$–$1.00$ | Very strong |

---

## 2. Linear Regression (OLS)

The full OLS model:

$$Y_i = \beta_0 + \beta_1 X_{1i} + \beta_2 X_{2i} + \ldots + \beta_k X_{ki} + \varepsilon_i, \quad \varepsilon_i \overset{iid}{\sim} \mathcal{N}(0, \sigma^2)$$

**Simple regression** (one predictor): use the Regression toggle with Covariates empty.
**Multiple regression** (Factor 1 + covariates): all covariates enter the model simultaneously. Stepwise selection is not performed; all predictors are included.

### Smart Mapping configuration

| Bucket | Assign |
|---|---|
| **Dependent Variable** | Numeric outcome |
| **Factor 1** | Primary continuous predictor |
| **Covariates** | Additional predictors (continuous) |
| **Filter** | Optional subgroup restriction |

### Variable transformations

Available for X (Factor 1) and Y (Dependent Variable) in Regression mode:

| Transform | Formula | Use case |
|---|---|---|
| **log₁₀** | $X' = \log_{10}(X)$ | Right-skewed positive variables |
| **sqrt** | $X' = \sqrt{X}$ | Count data; moderate right skew |
| **Box–Cox** | $X'(\lambda) = \frac{X^\lambda - 1}{\lambda}$, $\lambda \neq 0$ | Automatic $\lambda$ optimisation via profile log-likelihood |

Transformations change the interpretation of $\hat{\beta}$. The HTML report states the correct interpretation when transforms are active. A few examples:

- log₁₀(Y), untransformed X: $\hat{\beta}_1 = 0.3$ means a one-unit increase in X multiplies Y by $10^{0.3} \approx 2.0$.
- log₁₀(X), untransformed Y: $\hat{\beta}_1 = 5.2$ means doubling X increases Y by $5.2 \cdot \log_{10}(2) \approx 1.57$ units.
- log₁₀(X) and log₁₀(Y): $\hat{\beta}_1$ is the elasticity: a 1% increase in X corresponds to a $\hat{\beta}_1 \%$ change in Y.

The warning label next to the transformation dropdowns activates when any transform is selected.

### Data structure example

```
PatientID | Pump_time | Age | Baseline_NK | NK_cells_post | OP_Group
P001      | 87        | 62  | 1600        | 1450          | 1
P002      | 120       | 71  | 1400        | 1100          | 1
```

Assign `Pump_time` → Factor 1; `NK_cells_post` → Dependent Variable; `Age`, `Baseline_NK` → Covariates; `OP_Group` → Filter (value = 1).

### HTML report output

**Model summary:**

| Statistic | Formula / Description |
|---|---|
| $R^2$ | $1 - SS_{\text{res}} / SS_{\text{tot}}$ |
| $R^2_{\text{adj}}$ | $1 - (1 - R^2)\frac{n-1}{n-k-1}$ (penalised for $k$ predictors) |
| $F(k, n-k-1)$ | Overall model test |
| $p(F)$ | Two-tailed |
| AIC | $2k - 2\ell$; BIC: $k\ln(n) - 2\ell$ (lower is better) |
| $n$ | Observations after listwise deletion |

**Coefficient table:**

| Column | Content |
|---|---|
| $\hat{\beta}_j$ | Unstandardised coefficient |
| SE$(\hat{\beta}_j)$ | Standard error |
| $t$ | $\hat{\beta}_j / \text{SE}(\hat{\beta}_j)$, evaluated on $t_{n-k-1}$ |
| $p$ | Two-tailed |
| 95% CI | $\hat{\beta}_j \pm t_{0.975, n-k-1} \cdot \text{SE}(\hat{\beta}_j)$ |

**Interpreting $\hat{\beta}_j$:** In a model without transformations, $\hat{\beta}_j$ is the expected change in $Y$ per one-unit increase in $X_j$, holding all other predictors constant. A $\hat{\beta}_1 = -2.5$ for `Pump_time` means: each additional minute of pump time is associated with a 2.5-unit decrease in the outcome, adjusted for the other covariates in the model.

---

## 3. Residual Diagnostics

Three diagnostic tests run after every regression. All three must pass for standard inference to be fully reliable.

### 3a. Shapiro–Wilk on residuals

Tests $H_0$: residuals $\hat{\varepsilon}_i$ are normally distributed. OLS coefficient tests and confidence intervals are exact only when residuals are normal.

- **Pass ($p > 0.05$):** Standard CIs and p-values are valid.
- **Fail ($p \leq 0.05$):** Consider log-transforming the outcome, checking for outliers, or bootstrap-based CIs.

With large $n$ ($> 100$), Shapiro–Wilk becomes sensitive to trivial departures. Inspect the Q–Q plot in the HTML report rather than relying on the p-value alone.

### 3b. Breusch–Pagan test

Tests $H_0$: residual variance is constant across fitted values (homoscedasticity). Regresses $\hat{\varepsilon}_i^2$ on the predictors.

- **Pass:** Standard errors are reliable.
- **Fail:** Heteroscedasticity inflates or deflates SEs. Consider HC3 robust standard errors or a variance-stabilising outcome transformation.

### 3c. Ramsey RESET test

Tests $H_0$: the linear functional form is correctly specified. Adds $\hat{Y}^2$ (and sometimes $\hat{Y}^3$) to the model and tests whether these terms are significant.

- **Pass:** Linear model well-specified.
- **Fail:** Non-linearity present. Add polynomial terms, transform a predictor, or consider a non-linear model.

### Diagnostic summary

| Normality | Homoscedasticity | Linearity | Action |
|---|---|---|---|
| Pass | Pass | Pass | Interpret coefficients directly |
| Fail | Pass | Pass | Check for outliers; consider outcome transformation |
| Pass | Fail | Pass | Report HC3 robust SEs |
| Pass | Pass | Fail | Add $X^2$ term or transform a predictor |
| Multiple fail | — | — | Review model specification before reporting |

---

## 4. ANCOVA (Categorical Factor 1 + Covariates)

When Factor 1 is categorical and the Covariates bucket is populated, the analysis switches to ANCOVA. The covariate is included as a linear term; its effect is partialled out before testing group differences.

For full ANCOVA documentation (adjusted means, slope homogeneity, Simple Slopes, Johnson–Neyman), see **Section 19 of [HowTo.md](./HowTo.md)**.

The key formula: the adjusted group mean for group $j$:

$$\hat{\mu}_j^* = \hat{\mu}_j - \hat{\beta}_{\text{cov}} \cdot (\bar{x}_{j,\text{cov}} - \bar{x}_{\text{cov}})$$

This is what ANCOVA tests: not the raw means, but the means after accounting for covariate imbalance between groups.

---

## 5. Linear Mixed Model (LMM)

When Factor 1 is continuous and Subject ID is assigned, the application fits an LMM instead of Correlation. This is the right choice for longitudinal data where subjects are measured at multiple values of a continuous predictor (e.g. days, pump duration in minutes).

The random-intercept model:

$$Y_{ij} = \underbrace{(\beta_0 + u_{0i})}_{\text{subject-specific intercept}} + \beta_1 X_{ij} + \varepsilon_{ij}$$

where $u_{0i} \sim \mathcal{N}(0, \sigma^2_u)$ and $\varepsilon_{ij} \sim \mathcal{N}(0, \sigma^2_\varepsilon)$ are assumed independent.

A random-slope extension adds $u_{1i} \sim \mathcal{N}(0, \sigma^2_{u_1})$:

$$Y_{ij} = (\beta_0 + u_{0i}) + (\beta_1 + u_{1i}) X_{ij} + \varepsilon_{ij}$$

The random-slope model is accepted when the Likelihood Ratio Test confirms it fits better:

$$\Lambda = -2\bigl(\ell_{\text{RI}} - \ell_{\text{RI+RS}}\bigr) \sim \chi^2(2)$$

If $\Lambda > \chi^2_{0.95}(2) = 5.99$ (i.e. $p < 0.05$), the RI + RS model is used. The HTML report states which structure was chosen and the LRT result.

### ICC

The Intraclass Correlation Coefficient quantifies how much of the total variance is attributable to between-subject differences:

$$\text{ICC} = \frac{\sigma^2_u}{\sigma^2_u + \sigma^2_\varepsilon}$$

| ICC | Interpretation |
|---|---|
| $< 0.10$ | Negligible clustering; OLS regression may be adequate |
| $0.10$–$0.30$ | Weak clustering |
| $0.30$–$0.60$ | Moderate clustering; LMM recommended |
| $> 0.60$ | Strong clustering; LMM indicated |

For full LMM documentation (Smart Mapping configuration, fixed effects table, convergence), see **Section 20 of [HowTo.md](./HowTo.md)**.

---

## 6. Logistic Regression

When the Dependent Variable contains exactly two distinct values, the application fits a logistic regression model:

$$\log\frac{P(Y=1\mid\mathbf{X})}{1 - P(Y=1\mid\mathbf{X})} = \beta_0 + \sum_{j=1}^{k} \beta_j X_j$$

The primary outputs are Odds Ratios $\text{OR}_j = \exp(\hat{\beta}_j)$ and the model discrimination via AUC.

McFadden's pseudo-$R^2$ summarises overall fit:

$$R^2_{\text{McFadden}} = 1 - \frac{\ell_{\text{full}}}{\ell_{\text{null}}}$$

Values $> 0.10$ suggest a useful model; $> 0.20$ suggests good fit. Unlike OLS $R^2$, McFadden's $R^2$ rarely approaches 1.0 in practice and should not be interpreted on the same scale.

When complete separation is detected (SE $> 5$ for any coefficient, or non-convergence), the application switches to Firth Penalized Likelihood. This regularisation method was developed specifically for small samples and rare events and produces finite, reliable estimates where standard ML fails.

For full Logistic Regression documentation (AUC, Brier score, calibration slope, OR table), see **Section 21 of [HowTo.md](./HowTo.md)**.

---

## 7. Filter Bucket with Regression and Correlation

The Filter restricts the dataset **before** any assumption check or model fitting. Shapiro–Wilk normality tests, reported $n$, and all model coefficients reflect only the filtered rows. The HTML report header identifies the active filter.

**Typical subgroup workflow:**

```
Filter:              OP_Group = 1      → 93 On-Pump rows
Factor 1:            Pump_time         → continuous → Regression
Covariates:          Age, Baseline_NK  → multiple regression
Dependent Variable:  NK_cells_post
```

With $n < 20$ and $k$ predictors, the model is underpowered and $R^2_{\text{adj}}$ will be unreliable. Check the filtered $n$ in the bucket label before running.

---

## 8. Common Errors

| Error | Symptom | Fix |
|---|---|---|
| Categorical column in Factor 1 | Application runs ANOVA instead of Correlation | Use a column with $> 10$ unique numeric values |
| Covariates empty when Regression was expected | Application runs Correlation | Add at least one covariate, or activate the Regression toggle |
| Filter reduces $n$ below 5 | Analysis aborts | Use a less restrictive filter value |
| All-missing column in Covariates | Model fails | Remove the column or impute missing values first |
| Factor 1 = same column as a Covariate | Perfect multicollinearity; model undefined | Remove the duplicate from one bucket |
| Subject ID assigned with continuous Factor 1 | Triggers LMM instead of Correlation | Remove Subject ID if a simple correlation is the goal |

---

## 9. Exploratory Correlation Matrix

**Analysis → Exploratory Correlation Matrix** computes all $\binom{m}{2}$ pairwise correlations for a user-selected set of $m$ numeric variables and corrects for multiple testing.

### Use cases

- Hypothesis generation: spot unexpected associations before confirmatory testing.
- Multicollinearity screening: identify highly correlated predictors before entering them into a regression model.
- Data quality: flag variables that are near-perfectly correlated (possible data entry errors or derived variables).

### Options

**Missing data handling:**

| Mode | Behaviour | Prefer when |
|---|---|---|
| **Pairwise deletion** | Each pair uses all rows where both variables are non-missing; $n$ varies per pair | Scattered missing data across many variables |
| **Listwise deletion** | Only rows complete for all selected variables; $n$ constant | Comparable sample sizes across all pairs are required |

**Multiple testing correction:**

| Method | Controls | When to use |
|---|---|---|
| **FDR (Benjamini–Hochberg)** | False Discovery Rate at level $q$ | Exploratory work; balances power and specificity |
| **Bonferroni** | Family-wise Error Rate at $\alpha$ | Few pre-specified hypotheses |
| **None** | — | Descriptive overview only; do not report as confirmatory |

With $m = 20$ variables, 190 simultaneous tests run. The expected number of false positives under no-correction is $0.05 \times 190 = 9.5$. Use FDR or Bonferroni.

**Stratification:** Assigns a categorical column to split the matrix by group. Each group receives its own set of three output matrices. Useful for comparing correlation structures between subgroups.

### HTML report output

- Matrix of $r$ / $\rho$ values
- Matrix of corrected $p$-values
- Matrix of $n$ per pair (inspect this when pairwise deletion is active; large variation in $n$ across pairs can distort the correlation structure)

---

## Related Documentation

- [HowTo.md](./HowTo.md): Sections 15–21 cover all analysis types with full configuration reference
- [ADVANCED_ANOVA_GUIDE.md](./ADVANCED_ANOVA_GUIDE.md): Factorial ANOVA (configuration, assumptions, sphericity, nonparametric fallbacks)
