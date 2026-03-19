# Nonparametric Alternatives for Advanced ANOVAs: Implementation Notes

This document describes what was implemented for robust/nonparametric alternatives to advanced ANOVA designs and what is statistically computed.

## 1) Where the new logic is wired in

- Entry into advanced-test flow:
  - `prepare_advanced_test(...)` in `Source_Code/statisticaltester.py`
  - `perform_advanced_test(...)` in `Source_Code/statisticaltester.py`
- If recommendation is `non_parametric`, the code routes into a modern-model fallback:
  - branch in `perform_advanced_test(...)`
  - call to `fallback_modern_models(...)` in `Source_Code/nonparametricanovas.py`

## 2) The three advanced fallback analyses

All three are handled in a unified fallback function:
- `fallback_modern_models(data, dependent_var, formula, design_type, subject_col=None, cov_struct_option=None, time_col=None)`

### A) Two-way design fallback
- Formula: `DV ~ C(FactorA) * C(FactorB)`
- Model class used in fallback:
  - `GLM` (generalized linear model)
- Intended interpretation:
  - robust alternative when parametric assumptions fail.

### B) Repeated-measures design fallback
- Formula: `DV ~ C(WithinFactor)`
- Model class used in fallback:
  - `GEE` with auto-selected covariance structure
- Default covariance behavior:
  - If an ordered time-like `time_col` is detected: `Autoregressive (AR1)`
  - Otherwise: `Exchangeable`
- Subject ID is used as clustering/grouping variable.

### C) Mixed design fallback
- Formula: `DV ~ C(BetweenFactor) * C(WithinFactor)`
- Model class used in fallback:
  - `GEE` with subject-level clustering
- Covariance behavior:
  - Same auto-rule as RM (`AR1` for detected time-like within factor, else `Exchangeable`)
- Supports mixed between/within structure in fallback mode.

## 3) Distribution family selection inside fallback

Family is selected from the dependent variable values with diagnostics:

1. Integer-like outcome (`allclose(y, round(y))`) and `y >= 0`:
- Initial fit: `Poisson` (log link)
- Model-based overdispersion check via Pearson phi:
  - `phi_hat = sum((y_i - mu_i)^2 / V(mu_i)) / (n - p)`
  - If `phi_hat > 1.2`: switch to `NegativeBinomial` (log link)
  - Otherwise: keep `Poisson` (log link)

2. Non-integer continuous outcome:
- If all values are strictly positive: `Gamma` (log link)
- If any value is `<= 0`: `Gaussian` (identity link) as explicit fallback with robust (sandwich/HC3) covariance in GLM fallback

3. Diagnostics and warnings:
- `family_diagnostics` is stored in results (`zero_fraction`, `overdispersion_ratio`, `pearson_phi`, selection reason)
- If `zero_fraction > 0.30`, a warning is added suggesting zero-inflated confirmatory models.

## 4) What test statistic is computed

After model fit, global effects are extracted from:
- `fitted.wald_test_terms()`

Per-term statistics are then stored in an ANOVA-like table with fields:
- Source
- F (in code naming)
- Wald_Chi2 (explicit alias of the same statistic)
- p-unc
- DF1
- DF2 (None)
- StatisticType = `Wald Chi-square`

Important: the reported `F` field actually contains Wald-type statistics (chi-square based), not classical ANOVA F-values.

## 5) Which p-value becomes the primary result

For compatibility with downstream app flow, a top-level `p_value` / `statistic` is still provided.

Current policy:
- evaluate all omnibus effects (main effects + interactions)
- use the minimum available omnibus p-value as gate value
- store policy metadata in `primary_effect_policy` and selected effect details in `primary_effect`
- store reporting hierarchy metadata (`interaction_significant`, `interpretation_order`) for text-report ordering

Interpretation guidance:
- scientific interpretation should use the full omnibus table (`anova_table`) rather than only the gate value.
- when interaction is significant, interaction effects are reported first; main effects are interpreted as averaged effects.

## 6) Post-hoc pipeline in fallback mode

Post-hoc is only attempted if fallback model is significant (`p < alpha`).

### Step 1: Try marginal effects pairwise comparisons

Wrapper function:
- `posthoc_marginaleffects(...)` in `Source_Code/nonparametricanovas.py`

Design-specific calls:
- Two-way: `_run_two_way_marginaleffects_posthoc(...)`
- RM: `_run_rm_marginaleffects_posthoc(...)`
- Mixed: `_run_mixed_marginaleffects_posthoc(...)` (between-pass and within-pass)

The comparison table is transformed into exporter format by:
- `_map_marginaleffects_to_exporter(...)`

Note:
- Mapped marginaleffects comparisons are now multiplicity-corrected with Holm-Sidak and exported as `corrected=True`.
- For mixed designs, correction is applied globally across both marginaleffects passes (between-at-within and within-at-between).

### Step 2: If marginaleffects fails or yields no usable comparisons

Fallback to robust pairwise tests:
- `_run_modern_fallback_posthoc(...)`

Used methods:
- Two-way:
  - pairwise nonparametric via `perform_refactored_posthoc_testing(..., posthoc_choice="dunn")`
- RM:
  - dependent pairwise via `perform_dependent_posthoc_tests(..., parametric=False)`
- Mixed:
  - between-level comparisons inside each within level (Dunn)
  - within-level dependent comparisons inside each between level (Wilcoxon-based dependent posthoc)

## 7) Multiple testing correction in fallback post-hoc tests

In the nonparametric/dependent fallback methods:
- Dunn and dependent pairwise tests use Holm-Sidak correction (`multipletests(..., method='holm-sidak')`).

In the marginaleffects standard path:
- pairwise comparisons now also use Holm-Sidak correction in exporter mapping.

Exporter transparency:
- a dedicated `Methodology` sheet summarizes model class/family, link, covariance structure, covariance estimator,
  Pearson phi, family-selection rationale, gate policy, post-hoc path, and multiplicity method.

This is implemented in:
- `DunnTest` (Holm-Sidak corrected pairwise p-values)
- `DependentPostHoc` (Holm-Sidak corrected paired comparisons)

## 8) Statistical interpretation summary

Conceptually, the robust branch is not a classic rank-ANOVA replacement (like pure aligned-rank transform ANOVA).
It is a model-based robust fallback pipeline:

- GLM/GEE modeling under non-normal / heteroscedastic conditions
- Wald-term inference for omnibus effects
- pairwise post-hoc from marginal effects, with nonparametric pairwise fallback when needed

This is a pragmatic robustness strategy with explicit assumption-triggered routing.

## 9) Quick checklist for academic review

1. Confirm whether Wald chi-square inference is acceptable for your target reporting standard.
2. Confirm that minimum-p omnibus gate policy is acceptable for your workflow and report full effect table.
3. Confirm whether Holm-Sidak is the desired multiplicity correction for both marginaleffects and fallback paths.
4. Confirm distribution-family auto-selection (Poisson vs NegativeBinomial vs Gamma vs Gaussian fallback) against your endpoint type.
5. Confirm whether zero-inflation warnings should trigger mandatory alternative model recommendations.

