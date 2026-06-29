

# Advanced ANOVA Configuration Guide

Three factorial ANOVA designs are available in BioMedStatX: Two-Way ANOVA, Repeated Measures ANOVA, and Mixed ANOVA. All three are triggered by what you assign to the Smart Mapping buckets. No manual test selection exists.

---

## Overview

| Design | Factor 1 | Factor 2 | Subject ID |
|---|---|---|---|
| **Two-Way ANOVA** | categorical | categorical | — |
| **Repeated Measures ANOVA** | categorical (within) | — | required |
| **Mixed ANOVA** | categorical (within) | categorical (between) | required |

Each design has a fully implemented nonparametric fallback. The parametric or nonparametric path is chosen based on Shapiro–Wilk residual normality (after any transformation attempt, not before).

---

## Mixed ANOVA (Between × Within)

Use this design when participants belong to different groups *and* are measured repeatedly across conditions or timepoints. Both sources of variation (between subjects and within subjects) enter the model simultaneously.

**Example:** Blood pressure measured at `Pre` and `Post` (within-subject factor: `Timepoint`) in patients randomised to `Drug` vs. `Placebo` (between-subject factor: `Group`). Each patient contributes two rows.

### Smart Mapping configuration

| Bucket | Assign |
|---|---|
| **Dependent Variable** | Numeric outcome (e.g. `Value`) |
| **Factor 1** | The **within-subject** factor: the repeated condition (e.g. `Timepoint`: `Pre`, `Post`) |
| **Factor 2** | The **between-subject** factor: the independent grouping (e.g. `Group`: `Drug`, `Placebo`) |
| **Subject ID** | Individual identifier (e.g. `PatientID`) |

Factor 1 carries the repeated-measures structure. Factor 2 carries the group structure. Swapping them produces a valid design, but with the between/within labels reversed in the output. Check the report carefully.

### Data structure

```
SubjectID | Group   | Timepoint | Value
S001      | Drug    | Pre       | 120
S001      | Drug    | Post      | 110
S002      | Drug    | Pre       | 130
S002      | Drug    | Post      | 115
S003      | Placebo | Pre       | 125
S003      | Placebo | Post      | 123
```

Each subject appears once per level of Factor 1. Missing cells (a subject with no `Post` value) cause that subject to be excluded.

---

## Repeated Measures ANOVA (Within only)

All participants are in the same group. The factor of interest is the within-subject factor: time, dose, condition, trial.

**Example:** Twelve mice tested at `Week1`, `Week2`, and `Week3`. The question is whether performance changes over time, not whether groups differ.

### Smart Mapping configuration

| Bucket | Assign |
|---|---|
| **Dependent Variable** | Numeric outcome |
| **Factor 1** | The repeated factor (e.g. `Timepoint`: `Week1`, `Week2`, `Week3`) |
| **Subject ID** | Individual identifier |
| **Factor 2** | **Leave empty**: assigning Factor 2 switches to Mixed ANOVA |

### Data structure

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

## Two-Way ANOVA (Between × Between)

Two independent categorical factors. No subject appears more than once. Each row is a distinct individual.

**Example:** Weight loss outcomes in a 3 × 2 design: three diets (`Diet`) crossed with two exercise levels (`Exercise`). One measurement per participant.

### Smart Mapping configuration

| Bucket | Assign |
|---|---|
| **Dependent Variable** | Numeric outcome |
| **Factor 1** | First independent grouping factor (e.g. `Diet`) |
| **Factor 2** | Second independent grouping factor (e.g. `Exercise`) |
| **Subject ID** | **Leave empty**: assigning Subject ID switches to Mixed ANOVA |

### Data structure

```
SubjectID | Treatment | Gender | Value
S001      | DrugA     | Male   | 145
S002      | DrugA     | Female | 132
S003      | DrugB     | Male   | 158
S004      | DrugB     | Female | 141
```

---

## Which Design to Use

```
Same subjects measured more than once?
│
├── YES: Is there also an independent groups factor?
│       ├── YES  →  Mixed ANOVA       (F1 = within, F2 = between, Subject ID required)
│       └── NO   →  RM-ANOVA          (F1 = within, Subject ID required)
│
└── NO: Two independent categorical factors?
        ├── YES  →  Two-Way ANOVA     (F1 + F2, no Subject ID)
        └── NO   →  One-Way / t-Test  (Factor 1 only, see HowTo.md Section 10)
```

---

## Common Configuration Errors

| Error | Consequence | Fix |
|---|---|---|
| Mixed ANOVA without Subject ID | Repeated measurements treated as independent; inflated Type I error | Assign the subject identifier |
| RM-ANOVA with Factor 2 assigned | Switches to Mixed ANOVA (intended?) | Remove Factor 2 if a pure within design was planned |
| Two-Way ANOVA with Subject ID | Switches to Mixed ANOVA | Remove Subject ID if all measurements are independent |
| Within-subject factor assigned to Factor 2 instead of Factor 1 | Between/within labels reversed in output | Verify which factor is repeated; assign it to Factor 1 |
| Imbalanced design (missing cells) | Subject excluded from analysis | Impute or verify data completeness before importing |

---

## Assumptions

### Shared by all three designs

**Normality**: Shapiro–Wilk on model residuals. The $F$-statistic:

$$F(df_1, df_2) = \frac{MS_{\text{effect}}}{MS_{\text{error}}}$$

is robust to mild normality violations when groups are balanced and $n$ is moderate ($\geq 10$ per cell). With small or unbalanced samples, violations matter more.

**Variance homogeneity**: Levene's test. Required for between-subjects comparisons.

### Additional for RM-ANOVA and Mixed ANOVA: Sphericity

Sphericity requires that the variances of all pairwise differences between within-subject factor levels are equal. Formally: for all pairs $(j, k)$,

$$\text{Var}(Y_{ij} - Y_{ik}) = \text{constant}$$

Mauchly's test ($W$) evaluates sphericity. When sphericity fails ($p \leq 0.05$), the nominal $F$-test degrees of freedom are too liberal and must be corrected.

BioMedStatX applies corrections based on the Greenhouse–Geisser epsilon estimate ($\hat{\varepsilon}_{GG}$):

| Condition | Correction applied |
|---|---|
| $\hat{\varepsilon}_{GG} < 0.75$ | **Greenhouse–Geisser**: more conservative; adjusted $df' = \hat{\varepsilon}_{GG} \cdot df_{\text{nominal}}$ |
| $\hat{\varepsilon}_{GG} \geq 0.75$ | **Huynh–Feldt**: less conservative, higher power; $\hat{\varepsilon}_{HF} \geq \hat{\varepsilon}_{GG}$ |

Both corrections reduce the effective degrees of freedom, which raises the critical $F$-value. The correction applied is stated in the HTML report.

**Practical guidance:** Sphericity is most likely to be violated when the within-subject factor has many levels (e.g. five or more timepoints) or when subjects show markedly different trajectories. With only two levels, sphericity is trivially satisfied. When sphericity cannot be formally tested (e.g. due to indeterminate or incomplete ANOVA table outputs), BioMedStatX conservatively assumes it is violated and defaults to applying the Greenhouse-Geisser correction.

---

## Nonparametric Fallbacks

When normality fails (after any transformation), the application switches automatically:

| Design | Nonparametric Test | Reference |
|---|---|---|
| RM-ANOVA | **Friedman test** | $\chi^2_F = \frac{12}{nk(k+1)}\sum_j R_j^2 - 3n(k+1)$ |
| Two-Way ANOVA | **Freedman–Lane permutation** | Permutation-based $F$; handles covariates under the full model |
| Mixed ANOVA | **Brunner–Langer ATS** | Ranks-based ANOVA-type statistic; robust to non-spherical covariance |

No configuration is required. All fallbacks produce post-hoc comparisons, effect sizes, and descriptive statistics in the same HTML report format as the parametric path.

### Welch's ANOVA: unequal variances, normal residuals

For One-Way ANOVA designs where normality holds but Levene's test flags unequal variances, the application runs **Welch's ANOVA** instead of the standard $F$-test. Welch's version adjusts the degrees of freedom using the Welch–Satterthwaite approximation:

$$df_{\text{Welch}} = \frac{\left(\sum_j \frac{s_j^2}{n_j}\right)^2}{\sum_j \frac{(s_j^2/n_j)^2}{n_j - 1}}$$

This produces a valid test even when group variances differ substantially.

---

## Interpreting the Output

### Main effects and interactions

A **main effect** of Factor 1 means the outcome differs across the levels of Factor 1, averaged over all levels of Factor 2. A main effect of Factor 2 is the converse.

An **interaction** (Factor 1 × Factor 2) means the effect of one factor depends on the level of the other. When the interaction is significant, the main effects are qualified: they describe marginal trends, not the full story.

Always interpret a significant interaction before the main effects. The interaction plot in the HTML report is the most direct way to see what is happening.

**Effect size** for factorial ANOVA: partial eta-squared,

$$\eta^2_p = \frac{SS_{\text{effect}}}{SS_{\text{effect}} + SS_{\text{error}}}$$

Cohen's benchmarks: $\eta^2_p = 0.01$ small, $0.06$ medium, $0.14$ large.

### Post-hoc tests

| Comparison type | Parametric test | Nonparametric test |
|---|---|---|
| Between-subjects (groups) | Tukey HSD or Dunnett (Control-only) | Dunn's test (Holm correction) |
| Within-subjects (timepoints) | Holm-corrected paired $t$-tests or Dunnett-RM (Control-only) | Wilcoxon signed-rank (Holm) |

Post-hoc tests are only run when the corresponding main effect or interaction is significant.

---

## Visualisations in the HTML Report

**Two-Way ANOVA (interaction plot):**
x-axis = Factor 1 levels; one line per Factor 2 level; points show cell means ± SE. When the interaction is significant ($p < 0.05$), the interaction plot is the primary figure. When not significant, it appears as a secondary figure below the main bar chart.

**RM-ANOVA (profile plot):**
x-axis = within-factor levels; the group mean ± SE as a bold line; individual subject trajectories as thin grey lines in the background. Trajectories are omitted when subject-level data is unavailable.

**Mixed ANOVA (mixed profile plot and interaction plot):**
One line per between-group over within-factor levels. When the interaction is significant, both the interaction plot and the profile plot are shown. When it is not, the profile plot is primary.

---

## Best Practices

1. Plan the design before data collection. Knowing whether you need RM-ANOVA or Mixed ANOVA changes your sample-size calculation.
2. Verify balance. In mixed designs, every subject must have one value per within-factor level. Missing cells exclude the whole subject.
3. Report $\eta^2_p$ alongside $p$-values. A $p$-value without an effect size cannot tell you whether the result is scientifically meaningful.
4. If the interaction is significant, do not interpret the main effects in isolation. Plot the cell means and look at the interaction plot first.
5. Document which post-hoc test was used and why. Tukey answers all-pairwise questions, whereas Dunnett specifically compares all treatments against a single control group.

---

*See also: [HowTo.md](./HowTo.md) for the general user guide and [CORRELATION_REGRESSION_GUIDE.md](./CORRELATION_REGRESSION_GUIDE.md) for regression and correlation.*
