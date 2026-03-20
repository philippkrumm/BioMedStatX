# Nonparametric Alternatives for Advanced ANOVAs: Implementation Notes

This document describes what was implemented for nonparametric alternatives to
advanced ANOVA designs and what is statistically computed. All claims were verified
against the current source code (`nonparametricanovas.py` as of the latest revision).

---

## 1) Where the new logic is wired in

Entry into the advanced-test flow:

- `StatisticalTester.prepare_advanced_test(...)` in `Source_Code/statisticaltester.py`
  Runs assumption checks (Shapiro–Wilk normality on residuals, Levene variance test)
  and attempts data transformations. Returns a `recommendation` flag.

- `StatisticalTester.perform_advanced_test(...)` in `Source_Code/statisticaltester.py`
  Dispatches on the `recommendation` flag. When `recommendation == 'non_parametric'`,
  it routes to **design-specific nonparametric tests** (new) or falls back to the
  legacy GLM/GEE pipeline for unknown design types.

- Three new functions in `Source_Code/nonparametricanovas.py`:
  - `perform_friedman_test()` — Repeated-measures fallback
  - `perform_freedman_lane_test()` — Two-way ANOVA fallback
  - `perform_brunner_langer_ats()` — Mixed ANOVA fallback

- `fallback_modern_models(...)` in `Source_Code/nonparametricanovas.py`
  **Retained as safety net** for any design type not handled by the three functions
  above. Not called for RM, Two-Way, or Mixed designs.

### Dispatch logic (statisticaltester.py, lines 1821–1858):

```
recommendation == 'non_parametric'
  ├── test == 'repeated_measures_anova' → perform_friedman_test()
  ├── test == 'two_way_anova'           → perform_freedman_lane_test()
  ├── test == 'mixed_anova'             → perform_brunner_langer_ats()
  └── else                              → fallback_modern_models()
```

---

## 2) The three nonparametric fallback analyses

### A) Repeated-measures fallback: Friedman test

- **Function:** `perform_friedman_test(data, dv, within_factor, subject_col, alpha)`
- **Statistic:** Friedman Chi-square via `scipy.stats.friedmanchisquare()`
- **`model_class`:** `"Friedman"`
- **`StatisticType`:** `"Friedman Chi-square"`
- **Prerequisites:** ≥2 within-levels, ≥3 complete subjects
- **Data handling:** Pivots to wide format (subjects × within-levels), drops incomplete cases
- **Degrees of freedom:** df1 = k − 1 (k = number of within-levels), df2 = None
- **Valid for small samples:** Yes (no asymptotic assumptions beyond k ≥ 2, n ≥ 3)

**Warnings emitted:**
- k = 2: suggests paired Wilcoxon instead
- n < 5: notes potential low power

### B) Two-way ANOVA fallback: Freedman-Lane permutation test

- **Function:** `perform_freedman_lane_test(data, dv, factor_a, factor_b, alpha, n_permutations, seed)`
- **Statistic:** Permutation F-statistic (Type III SS via reduced vs. full model comparison)
- **`model_class`:** `"Freedman-Lane Permutation"`
- **`StatisticType`:** `"Permutation F (Freedman-Lane)"`
- **Default permutations:** 5000
- **Random number generator:** Local `np.random.default_rng(seed)` — does not affect global state

**Algorithm (run independently for each of 3 effects: A, B, A×B):**

1. Fit the full OLS model: `DV ~ C(A) + C(B) + C(A):C(B)`
2. Fit the reduced OLS model (omitting only the tested effect, retaining all others):
   - For effect A: `DV ~ C(B) + C(A):C(B)`
   - For effect B: `DV ~ C(A) + C(A):C(B)`
   - For interaction A×B: `DV ~ C(A) + C(B)`
3. Compute observed F from the increase in residual SS:
   `F_obs = ((RSS_reduced − RSS_full) / df_effect) / (RSS_full / df_residual)`
4. Extract fitted values and residuals of the reduced model
5. Permute the reduced-model residuals 5000 times, reconstruct pseudo-outcomes
   (`y_perm = ŷ_reduced + permuted_residuals`), refit full model, record F
6. `p_perm = (#{F_perm ≥ F_obs} + 1) / (n_permutations + 1)`

**Note on reduced models:** Each reduced model retains all effects *except* the one
being tested. This ensures each F-statistic tests the marginal (Type III) contribution
of a single effect, which is critical for unbalanced designs.

Both the permutation p-value (`p-perm`) and the parametric reference p-value
(`p-parametric`, from `scipy.stats.f.sf`) are stored in `anova_table`.
Only `p-perm` is used for significance decisions.

**Warnings emitted:**
- Min cell n < 5: limited permutation resolution
- Total N < 12: very few unique permutations possible

### C) Mixed ANOVA fallback: Brunner-Langer ANOVA-Type Statistic (ATS)

- **Function:** `perform_brunner_langer_ats(data, dv, between_factor, within_factor, subject_col, alpha)`
- **Statistic:** ANOVA-Type Statistic (ATS) after Brunner, Domhof & Langer (2002)
- **`model_class`:** `"Brunner-Langer ATS"`
- **`StatisticType`:** `"ANOVA-Type Statistic (ATS)"`
- **Design:** F1-LD-F1 (1 between-factor × 1 within-factor)
- **Valid for small samples:** Yes — the ATS requires fewer assumptions on the
  covariance structure than Wald-type statistics and has superior small-sample
  performance (Brunner et al. 2002).

**Algorithm:**

1. **Global mid-ranks:** All observations ranked together using
   `scipy.stats.rankdata(method='average')`. Ties receive average ranks.

2. **Relative Treatment Effects (RTE):** Per cell (group × time):
   `RTE = (mean_rank_in_cell − 0.5) / N`
   where N = total number of observations.

3. **Per-group rank covariance:** For each between-group i with n_i subjects:
   - Pivot ranks to wide format (subjects × within-levels)
   - Compute sample covariance (Bessel-corrected, ddof=1) divided by N²:
     `V̂_i = cov(R_i^T, ddof=1) / N²`  — shape (t × t)

4. **Block-diagonal total covariance:**
   `V_N = block_diag(V̂_1/n_1, V̂_2/n_2, ..., V̂_a/n_a)`  — shape (at × at)

5. **Idempotent projection matrices** (at × at, applied directly — no pseudoinverse):
   ```
   T_between = kron(I_a − J_a/a,  J_t/t)       — rank a−1
   T_within  = kron(J_a/a,         I_t − J_t/t) — rank t−1
   T_inter   = kron(I_a − J_a/a,  I_t − J_t/t) — rank (a−1)(t−1)
   ```
   where I = identity matrix, J = matrix of ones.

6. **ATS computation per effect:**
   ```
   ATS = N · (p̂ᵀ T p̂) / tr(T V_N)
   ```
   where p̂ is the (at × 1) vector of RTEs.

7. **Box-approximation degrees of freedom:**
   ```
   df1 = tr(T V_N)² / tr(T V_N T V_N)
   ```

8. **p-values:**
   - **Within and Interaction:** From F(df1, ∞) ≡ Chi²(df1)/df1:
     `p = 1 − χ²_cdf(ATS · df1, df=df1)`
   - **Between:** Finite df2 via Satterthwaite marginal-covariance approximation:
     ```
     λ_i = (1_t^T V̂_i 1_t) / (t² · n_i)    — marginal variance per group
     df2 = (Σ λ_i)² / Σ(λ_i² / (n_i − 1))
     p = 1 − F_cdf(ATS, dfn=df1, dfd=df2)
     ```
     This reduces to the simple `Σ(n_i−1)/(a−1)` formula for balanced designs,
     but is more accurate for unbalanced designs (common in biological data where
     subjects/samples are lost).

**Extra output:** `results["RTE"]` — a DataFrame with columns
`[between_group, within_level, RTE, n]` for each cell.

**Warnings emitted:**
- Any group n < 3: covariance estimation unreliable
- min(n_i) × t < 6: very few observations per cell

> **Reference:** Brunner, E., Domhof, S., & Langer, F. (2002). *Nonparametric
> Analysis of Longitudinal Data in Factorial Experiments.* Wiley, New York.
> R implementation: `nparLD` package (Noguchi et al., 2012).

---

## 3) Primary p-value selection policy: interaction_first

**Changed from the legacy pipeline.** The old `fallback_modern_models()` used a
`minimum_p_across_omnibus_effects` policy (selecting the smallest p across all
effects as the gate value). This was statistically problematic because it inflated
Type I error without correction.

The new policy is **interaction_first**:

- If the interaction p < alpha → use the interaction as the primary/gate effect
- Otherwise → use the main effect with the smallest p-value

Metadata stored in `results["primary_effect_policy"]` = `"interaction_first"`

Reporting hierarchy stored in `results["interpretation_order"]`:
- If interaction significant: `["interaction", "main_effects_cautious"]`
- Otherwise: `["main_effects", "interaction"]`

**Interpretation guidance:** Scientific reporting should use the full omnibus table
(`results["anova_table"]`), not only the gate p-value. When an interaction is
significant, main effects represent averaged effects across levels of the other
factor and should be interpreted with caution.

---

## 4) Post-hoc pipeline

Post-hoc is only attempted when `p_value < alpha` and no model error occurred.

All post-hoc orchestration remains in `Source_Code/statisticaltester.py`.

### Cascade logic:

1. **Step 1 — Marginaleffects-based pairwise comparisons** are attempted first
   via `_run_*_marginaleffects_posthoc()`. These methods check `model_class`:
   - `model_class == "GEE"` or `"GLM"` → proceeds with marginaleffects
   - **New model classes** (`"Friedman"`, `"Freedman-Lane Permutation"`,
     `"Brunner-Langer ATS"`) → returns `None` (skipped gracefully)

2. **Step 2 — `_run_modern_fallback_posthoc()`** catches all cases where Step 1
   returns None or fails. This function already implements correct nonparametric
   pairwise tests:

   | Design | Post-hoc method |
   |---|---|
   | Two-way | Dunn test (Kruskal-Wallis based) via `perform_refactored_posthoc_testing()` |
   | RM | Pairwise Wilcoxon signed-rank tests via `perform_dependent_posthoc_tests(parametric=False)` |
   | Mixed | **Conditioned simple effects:** Dunn (between-comparisons within each within-level) + Wilcoxon (within-comparisons within each between-group) |

   For mixed designs, `_run_modern_fallback_posthoc` (lines 3675–3726 in
   `statisticaltester.py`) splits the data by conditioning factor before running
   pairwise tests — this preserves the simple-effects structure.

---

## 5) Multiple testing correction

All post-hoc comparisons use **Holm-Sidak correction** (`method='holm-sidak'` in
`statsmodels.stats.multitest.multipletests`).

For mixed designs, Holm-Sidak is applied globally across both between- and
within-comparisons.

---

## 6) Result-dict structure

All three functions return an identical dict schema, compatible with the
downstream exporter, interpreter, and plotting code. Key fields:

| Field | Content |
|---|---|
| `test` | Human-readable test name, e.g. `"Mixed ANOVA [Brunner-Langer ATS Fallback]"` |
| `model_class` | `"Friedman"` / `"Freedman-Lane Permutation"` / `"Brunner-Langer ATS"` |
| `StatisticType` | Description of the test statistic used |
| `anova_table` | DataFrame with per-effect statistics (Source, F/ATS/Chi2, p, df1, df2) |
| `factors` | List of dicts, one per main effect |
| `interactions` | List of dicts, one per interaction (empty for Friedman) |
| `primary_effect` | Dict identifying the gate effect (source, kind, p_value) |
| `primary_effect_policy` | `"interaction_first"` |
| `analysis_note` | Human-readable explanation suitable for publication methods section |
| `warnings` | List of strings with statistical caveats |
| `p_value` | Gate p-value (from primary effect) |
| `statistic` | Gate test statistic |
| `descriptive` | Per-cell descriptive statistics (n, mean, sd, stderr, median, min, max) |
| `RTE` | (Brunner-Langer only) DataFrame of Relative Treatment Effects per cell |
| `n_permutations` | (Freedman-Lane only) Number of permutations used |

The `F` and `Wald_Chi2` fields in `factors` and `interactions` dicts contain
the actual test statistic (Chi², permutation F, or ATS), aliased for downstream
compatibility. They are **not** classical ANOVA F-values.

---

## 7) What was retained from the legacy pipeline

| Component | Status | Reason |
|---|---|---|
| `fallback_modern_models()` | **Kept** | Safety net for unknown design types |
| `posthoc_marginaleffects()` | **Kept** | Used by parametric paths |
| `GLMMTwoWayANOVA`, `GEERMANOVA`, `GLMMMixedANOVA` | **Kept** | Used by parametric paths |
| `_run_modern_fallback_posthoc()` | **Kept and reused** | Provides pairwise tests for all new methods |
| All exporter / plotting / interpretation code | **Unchanged** | New result dicts are fully compatible |

---

## 8) Statistical interpretation summary

This is a **design-specific nonparametric fallback pipeline** optimized for the
small sample sizes typical of biological research (n = 3–18 per group):

- **Friedman** (RM): Classical rank-based test, no distributional assumptions,
  valid at n ≥ 3. Inference via Chi-square approximation.
- **Freedman-Lane** (Two-Way): Permutation test with residual exchange under the
  reduced model. Approximately exact for all sample sizes with sufficient
  permutations (5000 default). Makes no distributional assumptions; requires only
  exchangeability of residuals under H₀.
- **Brunner-Langer ATS** (Mixed): Rank-based test using Relative Treatment Effects
  and ANOVA-Type Statistics. Valid under heteroscedasticity and non-normality.
  Produces population-averaged (marginal) inference. Box-approximated degrees of
  freedom; Satterthwaite correction for between-effect df₂.

All three methods avoid the asymptotic limitations of the previous GLM/GEE pipeline,
which required n > 30 per subgroup for reliable Wald-test inference.

---

## 9) Checklist for academic review

1. **Friedman (RM):** Confirm that the Chi-square approximation is acceptable
   for your sample size. For very small n (≤5), consider reporting the exact
   Friedman p-value if available.

2. **Freedman-Lane (Two-Way):** Confirm that 5000 permutations provide sufficient
   p-value resolution for your alpha level. At α = 0.05, the 95% CI for a true
   p = 0.05 is approximately [0.044, 0.056] with 5000 permutations.

3. **Brunner-Langer (Mixed):** Note that the ATS uses **Relative Treatment Effects**
   (RTEs), not means or medians. An RTE of 0.5 indicates no treatment effect.
   Reviewers unfamiliar with RTEs may require explanation in the methods section.

4. **Between-effect df₂ (Brunner-Langer):** For unbalanced designs, the
   Satterthwaite-approximated df₂ may differ substantially from the simple
   Σ(n_i−1)/(a−1) formula. The exact value is reported in `anova_table`.

5. Confirm that **Holm-Sidak** is the desired multiplicity correction. Note that
   Holm-Sidak assumes test independence; if strong positive correlation between
   comparisons is expected, Bonferroni is more conservative.

6. For mixed designs: note that rank-based methods produce **population-averaged**
   effects. If subject-specific effects are of scientific interest, a Mixed Effects
   Model (LMM/GLMM) should be considered for confirmatory analysis.

7. **Reduced model formulas (Freedman-Lane):** Confirm that each reduced model
   retains all effects except the one being tested (Type III logic). This is
   critical for unbalanced designs where effects are non-orthogonal.

8. Report the `analysis_note` text (or a version of it) in your methods section.
   It includes the specific test used, key parameters, and sample sizes.