# BioMedStatX — Clinical Models Numerical Validation

**Status:** 12/12 checks passed
**Date:** 2026-03-21
**Script:** `validation/benchmark_clinical_models.py`

---

## Summary

Three clinical model classes implemented in `Source_Code/clinical_models.py` were numerically validated against established reference datasets. The benchmark tests our wrapper classes — not statsmodels directly — to verify that parameter extraction, formula construction, and results serialization produce correct output end-to-end.

| Model | Dataset | Reference | Validation Status |
|-------|---------|-----------|-------------------|
| Linear Mixed Model | sleepstudy | Bates et al. (2015), JSS 67(1) | **Externally verified** |
| Logistic Regression | birthwt (MASS) | Hosmer & Lemeshow (2000), Table 2.1 | **Externally verified** |
| ANCOVA | Davis (carData) | Davis (1990), PubMed 2241138 | **Externally verified** (R 4.5.3, `drop1()` Type II SS) |

---

## Benchmark 1: Linear Mixed Model — Sleepstudy

**Reference:** Bates D, Mächler M, Bolker B, Walker S (2015). Fitting Linear Mixed-Effects Models Using lme4. *Journal of Statistical Software* 67(1):1–48.

**Model:** `Reaction ~ Days + (1 + Days | Subject)`, REML
**Dataset:** sleepstudy (lme4 package), 18 subjects × 10 days = 180 observations
**Implementation:** `LinearMixedModel` with `re_formula="~Days"` (correlated random intercept + slope)

### Results

| Parameter | Expected | Got | Tolerance | Status |
|-----------|----------|-----|-----------|--------|
| Intercept β₀ | 251.405 | 251.405 | ±0.01 | PASS |
| Slope β₁ (Days) | 10.467 | 10.467 | ±0.01 | PASS |
| SE(Days) | 1.502 | 1.546 | ±0.05 | PASS |
| Residual Variance σ² | 654.94 | 654.94 | ±2.0 | PASS |

### Implementation Notes

The canonical sleepstudy benchmark requires a **correlated random intercept + random slope** model, matching lme4's `(1 + Days | Subject)` syntax. This is implemented via `re_formula="~Days"` in statsmodels MixedLM.

A random-intercept-only model produces SE(Days) ≈ 0.80 (wrong). The correct value SE = 1.502 is achieved only with the full random effects structure, because between-subject variation in learning rate (heterogeneous slopes) inflates the uncertainty of the fixed slope estimate. SE(Days) therefore serves as a discriminator between the two model specifications.

`Days` must be passed as `covariates=["Days"]` (continuous fixed effect), not as `fixed_effects=["Days"]`. The latter triggers patsy's `C(Days)` treatment (10 dummy variables), which produces incorrect estimates.

---

## Benchmark 2: Logistic Regression — Low Birth Weight

**Reference:** Hosmer DW, Lemeshow S (2000). *Applied Logistic Regression*, 2nd ed. Wiley. Table 2.1.

**Model:** `low ~ age + lwt + ptl + C(smoke) + C(ht) + C(ui) + C(race)`
**Dataset:** birthwt (MASS package), n = 189
**Implementation:** `LogisticRegressionModel`

### Results

| Parameter | Expected OR | Got OR | Tolerance | Status |
|-----------|-------------|--------|-----------|--------|
| OR(smoke) | 2.518 | 2.518 | ±5% | PASS |
| OR(ht) | 6.445 | 6.257 | ±10% | PASS |
| OR(lwt) | 0.985 | 0.985 | ±5% | PASS |
| Hosmer-Lemeshow p | > 0.05 | 0.290 | boolean | PASS |

### Implementation Notes

**Variable classification** is critical for this benchmark. The `ptl` variable (count of prior preterm births, range 0–3) must be treated as a continuous covariate, not wrapped in `C()`. Wrapping `ptl` as categorical creates four dummy groups with cell sizes n=1, 5, which causes MLE convergence failure and astronomically wrong coefficients.

Correct split:
- `predictors` (→ C() wrapping): `smoke`, `ht`, `ui`, `race`
- `covariates` (→ continuous): `age`, `lwt`, `ptl`

**Corrections vs. initial Gemini-generated targets:**

| Parameter | Gemini claim | Correct value | Error source |
|-----------|-------------|---------------|-------------|
| OR(ht) | 5.67 (β = 1.735) | 6.445 (β = 1.863) | Wrong exponent in cited value |
| OR(smoke) bivariate | 1.91 | — | Bivariate model; full model gives 2.518 |

The Hosmer-Lemeshow goodness-of-fit test (p = 0.290 > 0.05) confirms the model fits adequately. No specific χ² target is benchmarked because the value depends on decile group boundaries, which vary slightly with dataset ordering.

---

## Benchmark 3: ANCOVA — Davis Body Measurements

**Reference:** Davis C (1990). Body image and weight preoccupation: a comparison between exercising and non-exercising women. *Appetite* 15(2):119–128. PubMed 2241138.
Documented in: Fox J, Weisberg S (2019). *An R Companion to Applied Regression*, 3rd ed. Sage. (carData package)

**Model:** `repwt ~ C(sex) + weight`, Type II SS
**Dataset:** Davis (carData package), n = 200; n = 182 after removing one documented data-entry error (observation 12: weight/height values transposed) and 17 missing `repwt` values
**Implementation:** `ANCOVAModel` with `between_factors=["sex"]`, `covariates=["weight"]`

### Results

| Parameter | Expected | Got | Tolerance | Status |
|-----------|----------|-----|-----------|--------|
| F(sex) Type II | 11.040 | 11.040 | ±0.05 | PASS |
| F(weight) Type II | 3114.42 | 3114.42 | ±5.0 | PASS |
| Slope homogeneity p-value present | — | 0.143 | boolean | PASS |
| Slope homogeneity assumption holds | True | True | boolean | PASS |

**Validation status: Externally verified.** F-values confirmed independently in R 4.5.3 using `drop1()` (base R Type II SS, equivalent to `car::Anova(type="II")`):

```r
library(carData)
Davis_clean <- Davis[Davis$weight < 150 & Davis$height > 100, ]
Davis_clean <- Davis_clean[complete.cases(Davis_clean[, c("repwt", "weight", "sex")]), ]
m <- lm(repwt ~ sex + weight, data = Davis_clean)
drop1(m, . ~ ., test = "F")
# sex    F = 11.04   p = 0.001081
# weight F = 3114.42 p < 2.2e-16
```

R output matches Python statsmodels to 6 significant figures. Slope homogeneity (sex:weight interaction, `drop1` on the interaction model) confirmed p = 0.1431.

**Citation scope note:** Davis (1990) is cited for dataset provenance only. The F-value targets (11.040, 3114.42) do not appear in the published paper — they are the result of fitting `repwt ~ sex + weight` to this dataset, verified independently in both Python and R.

### Implementation Notes

**Dataset selection rationale:** The initial benchmark used the anorexia dataset (Venables & Ripley 2002, MASS). That dataset was replaced for two reasons:

1. The slope homogeneity test yields p = 0.0067, meaning the fundamental ANCOVA assumption (parallel regression slopes across groups) is violated. Benchmarking ANCOVA on a dataset where ANCOVA is inappropriate is methodologically unsound.

2. The expected F-values (F(Treat) = 7.87, F(Prewt) = 7.27) were derived from our own `anova_lm(typ=2)` call with no independent source, making the test a consistency check rather than external validation.

The Davis dataset resolves both issues: slope homogeneity holds (p = 0.143), and the dataset is linked to a published paper with a PubMed record.

**On F(weight) = 3114:** This value is large by design — measured weight is a near-perfect linear predictor of reported weight, which is exactly what makes it an effective covariate. A large F(weight) confirms the covariate is providing strong adjustment, not that the test is trivial. The scientifically meaningful result is **F(sex) = 11.04** (p = 0.001): after controlling for actual weight, men and women differ significantly in how they report it. Sex differences in weight misreporting bias are a documented phenomenon in body-image research, and this F-value quantifies that effect. The ANCOVA benchmark tests both: that the covariate adjustment is correctly applied (F(weight) large), and that a real group difference survives it (F(sex) significant).

**Corrections vs. initial Gemini-generated targets (anorexia):**

| Claim | Gemini value | Correct value | Error source |
|-------|-------------|---------------|-------------|
| F(Prewt) | 40.44 | 7.27 | Mathematically impossible: Prewt-Postwt corr = 0.33 → max possible F ≈ 7.27. Likely hallucinated or from a different dataset. |
| F(Treat) | 8.984 | 7.87 | Gemini value corresponds to Type I SS; implementation correctly uses Type II. |

---

## Implementation Architecture

All three model classes reside in `Source_Code/clinical_models.py`. Each exposes:

- `fit(df, dv, ...)` — fits the model and stores results internally
- `as_results_dict()` — serializes results to a standardized dict used by the results exporter
- `check_regression_slope_homogeneity()` — ANCOVA only; tests the parallel slopes assumption

The `DataHealthScanner` class (same file) runs five pre-analysis checks before model fitting: MAD-based outlier detection on covariates, Little's MCAR test for missing data mechanism, VIF multicollinearity check, quasi-perfect separation detection (logistic), and minimum group size check (logistic).

---

## Reproducibility

```bash
cd /path/to/BioMedStatX
python validation/benchmark_clinical_models.py
# Exit 0 = all checks passed
```

All datasets are loaded at runtime via `statsmodels.datasets.get_rdataset()` from public CRAN mirrors. No local data files required. The benchmark output is deterministic (REML estimation converges to a unique solution for these datasets).
