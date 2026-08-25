

# BioMedStatX User Guide

BioMedStatX runs the entire statistical pipeline from assumption checks to HTML report generation. You supply the data and the mapping; the application selects the appropriate test. No code required.

---

## 0. First-Time Orientation

One concept determines everything in BioMedStatX: the division of labour between the application and the user.

| BioMedStatX decides | You decide |
|---|---|
| Which statistical test fits the design | Which columns map to which bucket |
| Which assumption checks to run | Whether to accept an offered transformation |
| Parametric vs. nonparametric route | Which post-hoc procedure to use when prompted |
| Which plots and tables to generate | Which number format your CSV uses (for import) |
| Whether the result meets $\alpha = 0.05$ | Which group subset to include in the analysis |

A complete analysis run follows this order:

1. Load the file; select the worksheet.
2. Assign columns to the Smart Mapping buckets.
3. Click **Start Auto Analysis**.
4. Respond to any prompts (transformation, post-hoc method).
5. Review the HTML report, opened in your browser when analysis completes.

---

## 1. Launching the Application

Double-click the **BioMedStatX** application icon. The main window opens.

> Developer note: if running from source, launcher scripts are available at the repository root. This has no bearing on normal usage.

---

## 2. Importing Data

Click **Load Data File**. Supported formats: Excel (`.xlsx`, `.xls`) and CSV (`.csv`).

Select the **Worksheet** from the dropdown (Excel files may contain multiple sheets). The **Table Preview** displays the first twelve rows so you can verify the import before proceeding.

For **CSV files** the application asks you to declare the number format instead of guessing it: the column **separator** (comma, semicolon, tab, or pipe), the **decimal mark** (dot or comma), and the **thousands separator** (none, dot, comma, or space). This is deliberate, since a wrong guess on a European file (for example reading `1.234,56` with a dot decimal) silently turns numbers into missing values. Set the three fields to match your file; the preview updates so you can confirm the columns parsed correctly.

### Minimum data requirements

| Requirement | Rationale |
|---|---|
| One row = one observation | Required for all supported workflows |
| At least one numeric measurement column | Needed for the Dependent Variable |
| At least one grouping or predictor column | Drives test selection |
| Subject ID column for repeated designs | Links multiple rows to the same individual |
| $\geq 2$ levels in the grouping factor | Required for any comparison |

Prepare your file with a single header row, unique column names, no merged cells, and consistent categorical labels. `WT` and `wt` are treated as different levels.

### Wide-format data

Wide-format files (one column per condition, e.g. `Value_Pre`, `Value_Post`) are detected and pivoted to long format before analysis. A notice confirms the pivot and lists the detected condition columns.

### Select Data Ranges

For raw Excel sheets not structured as a table, click **Select Data Ranges…** to open a spreadsheet viewer. Select cell ranges and assign them directly to groups. This workflow is limited to single-factor designs.

---

## 3. Smart Mapping

The center panel provides six mapping buckets. Drag column cards from the **Columns** list into the appropriate bucket. The application auto-detects an initial mapping, which you can override.

Each bucket carries an **ⓘ info button** describing what belongs there.

- **Dependent Variable**: The numeric outcome to be analysed: gene expression, cell count, weight, or any continuous measurement. Single analysis mode accepts one column; Multi-Dataset mode accepts several.
- **Factor 1**: The primary predictor. Categorical input (e.g. `Group` with levels `WT`, `KO`) triggers t-Test or ANOVA. Continuous input (e.g. `Pump time`) triggers Correlation or Regression. Only one column allowed here.
- **Factor 2** *(optional)*: A second grouping variable. Without Subject ID → Two-Way ANOVA. With Subject ID → Mixed ANOVA.
- **Subject ID** *(optional)*: The individual-level identifier for paired or repeated-measures designs. Assign this only when the same participant or experimental unit contributes more than one row.
- **Covariates** *(optional)*: Continuous confounders to control for (e.g. Age, BMI, Baseline). Categorical Factor 1 + Covariates → ANCOVA. Continuous Factor 1 + Covariates → Multiple Regression.
- **Filter** *(optional)*: Restricts the analysis to a subset of rows before any assumption checks or model fitting. See Section 15.

### Factor 1 vs. Subject ID: where most mistakes happen

A grouping variable and a subject identifier look identical in the data (both contain labels), but play opposite roles. `Group` with values `WT` and `KO` defines what you are *comparing*. `PatientID` with values `P001`, `P002` identifies *who was measured*. Getting this wrong produces an unpaired test where a paired design was intended, which inflates the error variance and reduces power.

### Mapping-to-design reference

| Design | Dep. Var. | Factor 1 | Factor 2 | Subject ID |
|---|---|---|---|---|
| Independent t-Test | Value | Group (WT / KO) | — | — |
| Paired t-Test | Value | Timepoint (Pre / Post) | — | SubjectID |
| One-Way ANOVA | Value | Group (≥ 3 levels) | — | — |
| Repeated Measures ANOVA | Value | Timepoint (≥ 3) | — | SubjectID |
| Two-Way ANOVA | Value | Group | Treatment | — |
| Mixed ANOVA | Value | Timepoint | Group | SubjectID |
| ANCOVA | Value | Group (categorical) | — | — | + Covariates |
| Correlation | Outcome | Predictor (continuous) | — | — |
| Linear Regression | Outcome | Predictor (continuous) | — | — | + Covariates |

The **mapping status line** below the buckets updates in real time and confirms which test will run. The messages are literal; the most common are (the app shows further messages for other incomplete states, e.g. a missing measurement column or too many subject-ID columns):

- *"Load a file to activate the mapping workflow."* — no file loaded yet.
- *"Assign at least one factor column."* — Dependent Variable is filled but Factor 1 is not.
- *"Auto-pilot currently supports at most two factor columns."* — too many factor columns mapped; reduce to one or two.
- *"Mapping looks valid. Start the analysis when you are ready."* — all required buckets are filled.
- *"Auto-pilot is analyzing the mapped design."* — analysis is running and the interface is locked.

After assigning Factor 1, use **Select Groups For Analysis** to restrict the analysis to a subset of factor levels. Leaving this empty runs all available groups.

---

## 4. Single vs. Multi-Dataset Analysis

Switch between modes with the radio buttons above the table preview.

**Single Analysis** runs one measurement column through the full pipeline. Use this for any single readout: one gene, one clinical parameter.

**Multi-Dataset Analysis** runs two or more measurement columns through the same factor mapping in sequence. The HTML report presents a summary card per column, with Benjamini–Hochberg FDR correction applied across all $m$ p-values. Restricted to ANOVA-capable designs.

---

## 5. Starting the Analysis

Click **Start Auto Analysis**. The pipeline executes in this order:

1. Apply the active data scope (Filter + group selection).
2. Normality check (Shapiro–Wilk on model residuals; bypassed if $N \ge 30$ per the Central Limit Theorem).
3. Variance homogeneity check (Levene's test).
4. Test selection (parametric, Welch, or nonparametric).
5. Main test.
6. Post-hoc comparisons (when $p < \alpha$ and $\geq 3$ groups).
7. Plot and HTML report generation.

Two interactive prompts may appear: transformation choice (Section 7) and post-hoc method selection (Section 8).

---

## 6. Export Settings

When you click **Start Auto Analysis**, a **Save Analysis Report** dialog opens first. Choose the folder and file name there: the location you pick becomes the output directory, and the base name is reused for all exports from that run (Excel file, HTML report, and plot image). The dialog suggests a name derived from your data file and measurement column; cancelling it aborts the run. There is no separate file-name field on the main window — the save dialog is where the output location is set.

---

## 7. Assumption Checks and Data Transformations

Shapiro–Wilk tests normality of model residuals (normality is assumed if $N \ge 30$ per the Central Limit Theorem); Levene's test checks variance homogeneity. When assumptions fail, the application prompts for a transformation.

| Transformation | Use case |
|---|---|
| **Log₁₀** | Right-skewed data; requires strictly positive values |
| **Box–Cox** | Automatic power transformation; $\lambda$ optimised by maximum likelihood |
| **Arcsin $\sqrt{x}$** | Proportion or percentage data; the scale is declared explicitly (see below) |

Skipping the transformation is always valid. The application takes the nonparametric route (Mann–Whitney U, Kruskal–Wallis, Friedman) without further prompting.

**Arcsin-square-root and the data domain.** Because arcsin($\sqrt{x}$) stabilises variance only for genuine proportion data, selecting it opens a prompt to declare the scale: **Proportion (0 – 1)** or **Percent (0 – 100)**. Values outside the declared range are rejected: the transformation is not applied and there is no silent fallback, so a percentage column mistakenly declared as a proportion stops with an error instead of corrupting the data. Cancelling the prompt (declaring no domain) skips arcsin and routes the analysis to the nonparametric test. The transform rescales against the global data range, not per group.

On very skewed data the Box–Cox $\lambda$ search can run away to a value so large it inflates the variance instead of taming it. The app guards against this: it checks the optimised $\lambda$ against the range $[-3, 3]$, and if $\lambda$ falls outside, it discards the estimate and uses a plain log transformation ($\lambda = 0$) instead. The report adds a note when this fallback happens.

---

## 8. Post-Hoc Comparisons

A significant main test with $\geq 3$ groups triggers a post-hoc selection prompt.

**Parametric options:**

| Test | When to use |
|---|---|
| **Tukey HSD** | All pairwise comparisons. Available for one-way designs and Two-Way ANOVA. Repeated-Measures and Mixed ANOVA do not offer Tukey; they use the Holm-Šidák pairwise option below, which stays coherent with the sphericity-corrected omnibus. |
| **Games-Howell** | All pairwise comparisons; does not assume equal variances. Default for One-Way Welch-ANOVA. |
| **Dunnett** | Each treatment group vs. one control; more power than Tukey when a reference group exists |
| **Holm-\u0160id\u00e1k corrected pairwise t-tests** | User-selected pairs; sequential \u0160id\u00e1k correction |
| **FDR-corrected pairwise t-tests** | User-selected pairs; Benjamini-Hochberg FDR (Available for Advanced ANOVAs) |

**Nonparametric path:**
- After **Kruskal–Wallis**, a prompt offers Dunn's test (all pairs, Holm-Šidák correction; the default) or pairwise Mann–Whitney U on pairs you pick.
- After **Friedman**, the app applies the **Conover-Iman** post-hoc (all pairs, Holm-corrected) directly, without a prompt; pairwise Wilcoxon signed-rank is used only as a fallback if the Conover-Iman routine is unavailable. The advanced nonparametric fallbacks (e.g. after Brunner–Langer) apply pairwise Wilcoxon/Mann–Whitney with Holm correction.

Cancelling a post-hoc prompt is a valid choice: the analysis keeps the main-test result and reports no pairwise comparisons. Pick it when only the overall effect matters.

The group-selection dialog and the pairwise-comparison dialog both provide **Select All** and **Deselect All** buttons to check or clear the whole list at once rather than ticking boxes individually.

Results appear on the plot as either significance brackets with stars or a compact letter display (see §9), and as a comparison table in the HTML report.

---

## 9. Plot Customisation

The HTML report includes an interactive **Plot Designer** section that rebuilds the plot in real time. Adjust settings and download publication-ready figures without re-running the analysis.

The Plot Designer has five tabs:

- **Plot**: chart type (Bar, Box, Violin, Raincloud, Forest, or Estimation), data point overlay (Jitter or Beeswarm), error bars (SD, SEM, 95% CI, IQR, or Range), central measure (mean or median), and a **Connect subjects** overlay. The overlay draws one line per subject across the levels — what a paired test actually analyses, and the only way to tell apart two datasets with identical boxes where one has every subject moving the same way and the other does not. It stays available only when the result carries subject identities, the level order is given by the data rather than guessed alphabetically, and no more than 30 subjects would be drawn; otherwise the checkbox is greyed out and says which condition failed. Raincloud places each group on its own row, so it is excluded there. Box plots show the median and interquartile range only; a mean ± error overlay is not drawn on a box (use the bar or violin plot for mean-based error bars).
- **Axes**: X/Y axis labels, Y-axis range and format, grid style, tick direction, legend position and orientation, and optional reference lines (y = 0 baseline, y = 1 fold-change, threshold lines from payload).
- **Style**: plot title, axis labels, font family and size, per-group colours (six curated palettes: Nature, Okabe-Ito, Grayscale HC, Muted Pastel, Deep, and Turbo, with Nature the default), bar fill patterns, and data point symbols.
- **Stats**: the **Significance display** selector (None, Brackets + stars, or Letters), line width, spacing, label size and offset, plus a checkbox per comparison. The selector opens on whichever form suits the result: letters when the post-hoc compared every pair and four or more groups are shown, brackets otherwise. Only significant pairs are drawn.
  - **Brackets + stars** annotate each comparison individually. Their checkboxes hide single brackets to declutter a crowded plot; the ones left standing each remain true on their own.
  - **Letters** label every group instead: groups sharing a letter are not significantly different. This stays available only when the post-hoc compared all pairs, because a comparison that was never run cannot honestly be shown as “not different” — with Dunnett or another control-referenced post-hoc the option is greyed out and the panel explains why. Letters are computed from the complete set of comparisons, so the per-pair checkboxes are disabled in this mode: dropping one comparison would merge two groups that do differ onto a shared letter and make the plot state the opposite of the result.
- **Export**: figure dimensions in inches, PNG scale (1x to 4x, up to ~400 DPI), SVG download, and PNG download.

---

## 10. Statistical Analyses: full reference

| Test | Triggered when |
|---|---|
| **Independent t-Test** (Welch's by default) | Factor 1 categorical, 2 groups, no Subject ID |
| **Paired t-Test** | Factor 1 categorical, 2 groups, Subject ID assigned |
| **Mann–Whitney U** | t-Test conditions; normality violated |
| **Wilcoxon signed-rank** | Paired t-Test conditions; normality violated |
| **Welch's ANOVA** | Factor 1 categorical, $\geq 3$ groups, no Subject ID. Default parametric One-Way ANOVA (robust to unequal variances) |
| **Kruskal–Wallis** | ANOVA conditions; normality violated |
| **Repeated Measures ANOVA** | Factor 1 categorical, Subject ID assigned, $\geq 3$ levels |
| **Friedman test** | RM-ANOVA conditions; normality violated |
| **Two-Way ANOVA** | Factor 1 + Factor 2 categorical, no Subject ID |
| **Freedman–Lane permutation** | Two-Way ANOVA conditions; normality violated |
| **Mixed ANOVA** | Factor 1 + Factor 2 + Subject ID assigned |
| **Brunner–Langer ATS** | Mixed ANOVA conditions; normality violated |
| **ANCOVA** | Factor 1 categorical + Covariates present |
| **Correlation (Pearson/Spearman)** | Factor 1 continuous, no Covariates, no Subject ID |
| **Simple/Multiple Regression (OLS)** | Factor 1 continuous + Covariates; or Regression toggle active |
| **Linear Mixed Model** | Factor 1 continuous + Subject ID |
| **Logistic Regression** | Dependent Variable contains exactly 2 distinct values |

Effect sizes reported per test family:

| Test family | Effect size |
|---|---|
| t-Test (independent) | Cohen's $d = \frac{\bar{x}_1 - \bar{x}_2}{s_p}$ |
| t-Test (Welch) | Hedges' $g$ |
| Wilcoxon / Mann–Whitney | Rank-biserial $r$ |
| ANOVA family | Partial $\eta^2_p$ |
| Correlation | Pearson $r$ or Spearman $\rho$ |
| Regression | $R^2$, adjusted $R^2$ |
| LMM | ICC $= \frac{\sigma^2_u}{\sigma^2_u + \sigma^2_\varepsilon}$ |
| Logistic Regression | AUC, McFadden $R^2$ |

---

## 11. Decision Tree Visualisation

The HTML report contains an interactive decision tree. The path actually taken is highlighted with animated arrows that replay in sequence. The initial view centres on the active path. Zoom, pan, and reset are available.

The decision-tree panel in the results area also has a **Maximize** button that opens the tree in a full-window overlay — useful on complex designs where the default panel size clips branches.

---

## 12. Reviewing Results: HTML Report

Analysis produces a single self-contained `.html` file that opens in your browser.

The report contains:

- **Header**: test name, $p$-value, significance label, effect size with magnitude badge (Small / Medium / Large by Cohen's conventions)
- **Statistical results**: statistic, degrees of freedom, $p$-value, effect size, 95% CI, power ($1 - \hat{\beta}$)
- **Assumption results**: normality, variance, and applied corrections
- **Data quality checks (pre-analysis)**: a warning table flagging any rows or values dropped during import (non-numeric cells, missing group labels), shown when the import found problems
- **Descriptive statistics**: $\bar{x}$, SD, SEM, median, $n$ per group
- **Pairwise comparison table**: post-hoc results with corrected $p$-values
- **Interactive decision tree**: full path with zoom and replay
- **Interactive plot**: main chart with optional plot designer
- **Raw data**: the filtered, analysis-ready dataset
- **Methods text**: a plain-language description of the pipeline, formatted for direct inclusion in a Methods section

---

## 13. Outlier Detection

**Analysis → Detect Outliers** offers:

- **Grubbs' test** (single or iterative). The default; holds its nominal false-positive rate down to small samples.
- **Modified Z-Score** (threshold at $|M_i| > 3.5$, where $M_i = \frac{0.6745(x_i - \tilde{x})}{MAD}$). Not the default: its median/MAD scale is unstable on small groups and flags a normal point as an outlier in about 29% of clean $n = 3$ samples, so the dialog shows a caution when it is selected on groups below $n = 8$.

Review flagged observations before proceeding. Removing outliers changes the analysis. Document this decision in your methods.

---

## 14. Quick-Reference Workflow

1. Launch the application.
2. Load file; select worksheet.
3. Assign: Dependent Variable, Factor 1, and (if needed) Factor 2, Subject ID, Covariates.
4. Choose Single or Multi-Dataset mode.
5. Optionally restrict the analysis to a subset of levels with **Select Groups For Analysis**.
6. Click **Start Auto Analysis**, then set the output folder and file name in the save dialog.
7. Respond to transformation and post-hoc prompts if they appear.
8. Open the HTML report from the output directory.

---

### Tips

- Group labels (WT, KO, Control) belong in **Factor 1**: they define experimental conditions, not individuals.
- Subject ID is needed only when one individual contributes more than one row.
- In paired designs, every subject must appear exactly once per condition. Imbalanced data triggers a warning.
- Copy the **Methods text** section from the HTML report directly into your manuscript draft.
- For skewed data, the Log₁₀ transformation is the safe first choice. Box–Cox is more aggressive and less interpretable after back-transformation.

---

## 15. Filter Bucket — Subgroup Analysis

The Filter bucket restricts the dataset **before** assumption checks and model fitting — the correct approach for a subgroup analysis, rather than filtering results after the fact.

**How to use:**

1. Drag a categorical column (e.g. `OP_Group`, `Sex`) into the Filter bucket.
2. Select the value to keep from its dropdown.
3. Click **Start Auto Analysis** — the whole pipeline runs on the restricted subset.

If the chosen subset leaves too few observations for the design, the analysis stops with a *"Too few observations after filter"* message rather than running on unstable data.

The ⓘ button on the bucket title explains its purpose at any time.

---

## 16. Correlation Analysis

**Trigger conditions:** Factor 1 continuous (> 10 unique numeric values), no Covariates, no Subject ID.

### Configuration

| Bucket | Assign |
|---|---|
| **Dependent Variable** | Numeric outcome |
| **Factor 1** | Continuous predictor |
| **Covariates** | Leave empty: any entry here switches to Regression |
| **Subject ID** | Leave empty: any entry here switches to LMM |
| **Filter** | Optional subgroup restriction |

### Pearson vs. Spearman: the decision rule

The choice is driven by sample size and distribution shape (skewness and excess kurtosis), not by a Shapiro–Wilk significance gate. A normality pre-test has too little power at the small $n$ typical of biomedical correlation and rejects trivial departures at large $n$; the app selects on shape tiers instead (the same reasoning behind the Welch-only $t$-test default). Shapiro–Wilk is still computed and shown in the report, but it does not decide the method.

- $n < 20$ → **Spearman $\rho$** (too few points to judge shape reliably).
- $20 \leq n < 100$ → **Pearson $r$** if **both** variables have $|\text{skew}| \leq 1.0$ and $|\text{excess kurtosis}| \leq 2.0$; otherwise **Spearman $\rho$**.
- $n \geq 100$ → **Pearson $r$** unless either variable has $|\text{skew}| > 2.0$ or $|\text{excess kurtosis}| > 4.0$ (extreme asymmetry → Spearman).

### Correlation→Regression toggle

When Factor 1 is continuous and Covariates is empty, a checkbox appears:

**"Analyze as Linear Regression (Y = a + bX)"**

Leaving it unchecked runs Correlation. Checking it runs Simple OLS Regression with one predictor: slope coefficient $\hat{\beta}_1$, 95% CI on $\hat{\beta}_1$, and the full residual diagnostic battery (Shapiro–Wilk, Breusch–Pagan, Ramsey RESET).

### What is reported (HTML report)

| Statistic | Description |
|---|---|
| $r$ or $\rho$ | Correlation coefficient, range $[-1, 1]$ |
| $p$ | Two-tailed significance |
| 95% CI | Fisher $z$-transformation interval |
| $n$ | Valid pairs after pairwise deletion |
| Method | Pearson or Spearman |
| Interpretation | Strength label (Negligible / Weak / Moderate / Strong / Very strong) |

Strength thresholds follow Cohen (1988): $|r| < 0.10$ negligible, $0.10$–$0.29$ weak, $0.30$–$0.49$ moderate, $0.50$–$0.69$ strong, $\geq 0.70$ very strong.

---

## 17. Linear Regression (OLS)

**Trigger conditions:** Factor 1 continuous + at least one Covariate assigned. Or: Regression toggle active (no Covariate needed for simple regression).

The fitted model:

$$Y_i = \beta_0 + \beta_1 X_{1i} + \beta_2 X_{2i} + \ldots + \beta_k X_{ki} + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, \sigma^2)$$

### Configuration

| Bucket | Assign |
|---|---|
| **Dependent Variable** | Numeric outcome |
| **Factor 1** | Primary continuous predictor |
| **Covariates** | Additional predictors to control for |
| **Filter** | Optional subgroup restriction |

### Variable transformations

Available for both X (Factor 1) and Y (Dependent Variable) when Regression mode is active:

| Transform | Formula | Use case |
|---|---|---|
| **log₁₀** | $X' = \log_{10}(X)$ | Right-skewed positive variables |
| **log₁₀(x+1)** | $X' = \log_{10}(X+1)$ | Count data with exact zeros |
| **sqrt** | $X' = \sqrt{X}$ | Count data; moderate skew |
| **Box–Cox** | $X' = \frac{X^\lambda - 1}{\lambda}$ | Automatic $\lambda$ optimisation |

When a transform is applied, the displayed $\hat{\beta}$ operates on the transformed scale. A $\hat{\beta}_1 = 0.3$ with log₁₀-transformed Y means a one-unit increase in X multiplies the original Y by $10^{0.3} \approx 2.0$. The HTML report states this interpretation explicitly.

If **both X and Y are log-transformed** (Log-Log model), $\beta$ represents an elasticity: a 1% increase in X is associated with a $\beta$% change in Y.

> [!WARNING]
> Transformations should be chosen based on theoretical grounds or to resolve assumption violations (e.g., heteroscedasticity or non-linearity), not purely because raw predictors "look skewed". Normality of predictors (X) is not an assumption of OLS regression.
> **Avoid P-Hacking:** Do not iteratively try different transformations just to drive the p-value below 0.05. This invalidates your statistical inference.

### What is reported (HTML report)

**Model summary:** $R^2$, adjusted $R^2 = 1 - \frac{(1-R^2)(n-1)}{n-k-1}$, $F(k, n-k-1)$, $p(F)$, AIC, BIC, $n$.

**Coefficient table:**

| Column | Content |
|---|---|
| $\hat{\beta}$ | Unstandardised regression coefficient |
| SE | Standard error of $\hat{\beta}$ |
| $t$ | $\hat{\beta} / \text{SE}$, evaluated on $t_{n-k-1}$ |
| $p$ | Two-tailed |
| 95% CI | $\hat{\beta} \pm t_{0.975, n-k-1} \cdot \text{SE}$ |

**Residual diagnostics:** Shapiro–Wilk (normality), Breusch–Pagan (homoscedasticity), Ramsey RESET (linearity). See [CORRELATION_REGRESSION_GUIDE.md](./CORRELATION_REGRESSION_GUIDE.md) for interpretation.

---

## 18. Exploratory Correlation Matrix

**Analysis → Exploratory Correlation Matrix** computes all pairwise correlations across selected numeric variables and corrects for multiple testing.

| Option | Description |
|---|---|
| Variable selection | Check/uncheck numeric columns |
| Method | Auto (Shapiro–Wilk per pair), Spearman, or Pearson |
| Missing data | Pairwise deletion ($n$ varies per pair) or Listwise (complete cases only) |
| Correction | FDR (Benjamini–Hochberg), Bonferroni, or None |
| Stratify by | Run the matrix separately per level of a categorical column |

For $m$ variables, the matrix contains $\binom{m}{2}$ tests. With $m = 20$, that is 190 simultaneous tests. Uncorrected p-values guarantee false positives. Use FDR or Bonferroni.

**HTML report output:**
- Matrix of $r$ / $\rho$ values
- Matrix of corrected $p$-values
- Matrix of $n$ per pair (essential when missing data is present)

---

## 19. ANCOVA / Two-Way ANCOVA

ANCOVA tests group mean differences while partitioning out the linear effect of one or more continuous covariates. The adjusted group mean for group $j$ estimates:

$$\hat{\mu}_j^* = \hat{\mu}_j - \hat{\beta}_{cov}(\bar{x}_{j,cov} - \bar{x}_{cov})$$

**Trigger conditions:** Factor 1 categorical + at least one Covariate assigned. Two-Way ANCOVA: Factor 1 + Factor 2 (both categorical) + Covariates.

### Configuration

| Bucket | Assign |
|---|---|
| **Dependent Variable** | Numeric outcome |
| **Factor 1** | Categorical grouping factor |
| **Factor 2** | Optional second categorical factor → Two-Way ANCOVA |
| **Covariates** | Continuous confounders |
| **Filter** | Optional subgroup restriction |

### What is reported (HTML report)

- **ANOVA table** (Type II SS): $F$, $df$, $p$, $\eta^2$ per source
- **Covariate effects**: $\hat{\beta}$, SE, $t$, $p$, 95% CI per covariate
- **Adjusted means**: estimated marginal means at the grand mean of each covariate
- **Regression slope homogeneity**: tests whether the covariate–outcome slope is equal across groups (the core ANCOVA assumption)
- **Simple Slopes and Johnson–Neyman interval**: reported when slopes are heterogeneous ($p < 0.05$ for the Factor × Covariate interaction)
- **Model fit**: $R^2$, adjusted $R^2$, AIC, $n$

### The slope homogeneity assumption

ANCOVA assumes parallel regression slopes. When this assumption fails, the adjusted means are misleading. The Johnson–Neyman (J-N) technique identifies the range of covariate values where the group difference is and is not statistically significant, which is more informative than simply flagging an assumption violation.

---

## 20. Linear Mixed Model (LMM)

LMM is the appropriate tool for longitudinal data where the same subjects are measured at multiple levels of a continuous factor (e.g. repeated timepoints, varying pump durations). Unlike simple regression, LMM accounts for the within-subject correlation between repeated measurements.

**Trigger conditions:** Factor 1 continuous + Subject ID assigned.

The model:

$$Y_{ij} = \beta_0 + \beta_1 X_{ij} + u_{0i} + \varepsilon_{ij}$$

where $u_{0i} \sim \mathcal{N}(0, \sigma^2_u)$ is the random intercept for subject $i$ and $\varepsilon_{ij} \sim \mathcal{N}(0, \sigma^2_\varepsilon)$ is the residual.

A random-intercept + random-slope model is also tested:

$$Y_{ij} = (\beta_0 + u_{0i}) + (\beta_1 + u_{1i}) X_{ij} + \varepsilon_{ij}$$

### Configuration

| Bucket | Assign |
|---|---|
| **Dependent Variable** | Numeric outcome |
| **Factor 1** | Continuous time or predictor variable |
| **Subject ID** | Subject/patient identifier |
| **Covariates** | Optional additional predictors |
| **Filter** | Optional subgroup restriction |

### Model selection: random intercept vs. random slope

The application fits both structures and compares them with a Likelihood Ratio Test:

$$\Lambda = -2\bigl(\ell_{\text{RI}} - \ell_{\text{RI+RS}}\bigr) \sim \chi^2(2)$$

If $\Lambda > \chi^2_{0.05}(2) = 5.99$ (i.e. $p < 0.05$), the random-intercept + slope model is retained. Otherwise the simpler random-intercept model is used. The structure chosen is noted in the HTML report.

### ICC: Intraclass Correlation Coefficient

$$\text{ICC} = \frac{\sigma^2_u}{\sigma^2_u + \sigma^2_\varepsilon}$$

| ICC | Interpretation |
|---|---|
| $< 0.10$ | Negligible clustering; standard regression may suffice |
| $0.10$–$0.30$ | Weak clustering |
| $0.30$–$0.60$ | Moderate clustering; LMM recommended |
| $> 0.60$ | Strong clustering; LMM strongly indicated |

### What is reported (HTML report)

Fixed effects table ($\hat{\beta}$, SE, $df$, $t/z$, $p$, 95% CI), random effects variances ($\sigma^2_u$, $\sigma^2_\varepsilon$), ICC, AIC, BIC, log-likelihood, LRT result, random structure chosen, convergence status.

---

## 21. Logistic Regression

**Trigger conditions:** Dependent Variable contains exactly 2 distinct values, either $\{0, 1\}$ or two unique string labels. The application encodes non-numeric outcomes as 0/1 internally.

The model:

$$\log\frac{P(Y=1)}{1 - P(Y=1)} = \beta_0 + \sum_{j=1}^{k} \beta_j X_j$$

### Configuration

| Bucket | Assign |
|---|---|
| **Dependent Variable** | Binary outcome (0/1 or two-level label) |
| **Factor 1** | Primary predictor |
| **Covariates** | Additional predictors |
| **Filter** | Optional subgroup restriction |

### What is reported (HTML report)

| Output | Description |
|---|---|
| **Odds Ratio (OR)** | $\exp(\hat{\beta}_j)$ with 95% CI: $\exp(\hat{\beta}_j \pm 1.96 \cdot \text{SE}_j)$ |
| $p$-value | Per predictor |
| **McFadden $R^2$** | $1 - \ell_{\text{full}} / \ell_{\text{null}}$; values $> 0.20$ indicate good fit |
| **AUC (ROC)** | Discrimination; $0.70$–$0.80$ acceptable, $0.80$–$0.90$ good, $> 0.90$ excellent |
| **Brier score** | Calibration; lower is better |
| **Calibration slope** | Ideal = 1.0; deviation indicates over- or underconfidence |
| AIC / BIC | Model comparison |
| $n$ | Observations after listwise deletion |
| **Model variant** | Standard ML or Firth Penalized Likelihood |

**Interpreting the Odds Ratio.** $\text{OR} > 1$: the predictor increases the probability of the event. $\text{OR} < 1$: it decreases the probability. $\text{OR} = 1$: no effect. An $\text{OR} = 2.5$ means the odds of the event are 2.5 times higher per unit increase in the predictor, holding everything else constant.

**Firth correction.** Complete separation (when one predictor perfectly predicts the outcome) causes standard maximum likelihood to diverge. The application detects this (large SEs, $> 5$) and switches to Firth Penalized Likelihood regression automatically. The report notes which variant was used.

Under Firth, the Wald confidence interval ($\hat{\beta} \pm 1.96\,\text{SE}$) is unreliable, because separation is the one case it cannot handle. The app instead reports penalized **profile-likelihood** intervals and penalized likelihood-ratio $p$-values, the same inference R's `logistf` uses. The odds-ratio table names the method it applied.

---

Happy analysing.
