# Nonparametric Alternatives for Advanced ANOVAs: Implementation Notes

This document describes what was implemented for nonparametric alternatives to
advanced ANOVA designs, the statistical basis of each method, and the validation
carried out against reference implementations. All formulas and code references
were verified against `src/analysis/nonparametricanovas.py` (current revision).

---

## 1) Where the new logic is wired in

Entry into the advanced-test flow:

- `StatisticalTester.prepare_advanced_test(...)` in `src/analysis/statisticaltester.py` (line 1160)
  Runs assumption checks (Shapiro–Wilk normality on residuals, Levene variance test)
  and attempts data transformations. Returns a `recommendation` flag.

- `StatisticalTester.perform_advanced_test(...)` in `src/analysis/statisticaltester.py` (line 1243)
  Delegates to `perform_advanced_test_pipeline` in `src/statistical_testing/advanced_pipeline.py`,
  which dispatches on the `recommendation` flag. When `recommendation == 'non_parametric'`,
  it routes to the design-specific nonparametric test or returns an error dict for unknown designs.

- Three functions in `src/analysis/nonparametricanovas.py`:
  - `perform_friedman_test()`: Repeated-measures fallback
  - `perform_freedman_lane_test()`: Two-way ANOVA fallback
  - `perform_brunner_langer_ats()`: Mixed ANOVA fallback

- For unknown design types, the pipeline returns an error dict
  (`"No non-parametric fallback implemented for test type: {test}"`).
  RM, Two-Way, and Mixed all have explicit branches.

### Dispatch logic (advanced_pipeline.py, lines 281–316):

```
recommendation == 'non_parametric'
  |---- test == 'repeated_measures_anova' → perform_friedman_test()
  |---- test == 'two_way_anova'           → perform_freedman_lane_test()
  |---- test == 'mixed_anova'             → perform_brunner_langer_ats()
  `---- else                              → fallback_modern_models()
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

**Post-hoc tests (built-in, triggered automatically if p < alpha):**

Pairwise Wilcoxon signed-rank tests for all k·(k−1)/2 level pairs, with
Holm step-down correction applied across all comparisons simultaneously.

- **Effect size:** rank-biserial correlation r = |2W − n(n+1)/2| / (n(n+1)/2)
- **Output:** `pairwise_comparisons` list in result dict; `posthoc_test` set to
  `"Pairwise Wilcoxon Signed-Rank (Holm, n=... subjects)"`
- If p ≥ alpha, `pairwise_comparisons` is empty and `posthoc_test` is `None`.

**Warnings emitted:**
- k = 2: suggests paired Wilcoxon instead
- n < 5: notes potential low power

---

### B) Two-way ANOVA fallback: Freedman-Lane permutation test

- **Function:** `perform_freedman_lane_test(data, dv, factor_a, factor_b, alpha, n_permutations, seed)`
- **Statistic:** Permutation F-statistic (Type III SS via reduced vs. full model comparison)
- **`model_class`:** `"Freedman-Lane Permutation"`
- **`StatisticType`:** `"Permutation F (Freedman-Lane)"`
- **Default permutations:** 5000
- **Random number generator:** Local `np.random.default_rng(seed)` (does not affect global state)

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
   (`y_perm = y_hat_reduced + permuted_residuals`), refit full model, record F
6. `p_perm = (#{F_perm ≥ F_obs} + 1) / (n_permutations + 1)`

**Note on reduced models:** Each reduced model retains all effects *except* the one
being tested. This ensures each F-statistic tests the marginal (Type III) contribution
of a single effect, which is critical for unbalanced designs.

Both the permutation p-value (`p-perm`) and the parametric reference p-value
(`p-parametric`, from `scipy.stats.f.sf`) are stored in `anova_table`.
Only `p-perm` is used for significance decisions.

**Post-hoc tests (built-in, triggered per significant effect if p < alpha):**

Pairwise Mann-Whitney U (MWU) tests are run for each effect whose permutation
p-value falls below alpha. All comparisons from all triggered effects are pooled
and Holm-corrected together.

| Triggered effect | Comparisons performed |
|---|----|
| Factor A significant | All pairs of Factor A levels (data collapsed over Factor B) |
| Factor B significant | All pairs of Factor B levels (data collapsed over Factor A) |
| Interaction significant | All pairs of cells (Factor A level × Factor B level) |

- **Effect size:** rank-biserial correlation r = |2U − n_1n_2| / (n_1n_2)
- **Output:** `pairwise_comparisons` list; `posthoc_test` = `"Pairwise Mann-Whitney U (Holm-corrected)"`
- If no effect is significant, `pairwise_comparisons` is empty.

**Warnings emitted:**
- Min cell n < 5: limited permutation resolution
- Total N < 12: very few unique permutations possible

---

### C) Mixed ANOVA fallback: Brunner-Langer ANOVA-Type Statistic (ATS)

- **Function:** `perform_brunner_langer_ats(data, dv, between_factor, within_factor, subject_col, alpha)`
- **Statistic:** ANOVA-Type Statistic (ATS) after Brunner, Domhof & Langer (2002)
- **`model_class`:** `"Brunner-Langer ATS"`
- **`StatisticType`:** `"ANOVA-Type Statistic (ATS)"`
- **Design:** F1-LD-F1 (1 between-factor × 1 within-factor)
- **Valid for small samples:** Yes. The ATS requires fewer assumptions on the
  covariance structure than Wald-type statistics and has superior small-sample
  performance (Brunner et al. 2002).

#### Algorithm

**Step 1: Global mid-ranks**

All N observations are ranked together using `scipy.stats.rankdata(method='average')`.
Tied observations receive their average rank.

**Step 2: Relative Treatment Effects (RTEs)**

For each cell (group i, time point s):

```
p_hat_is = (R_bar_is − 0.5) / N
```

where R_bar_is is the mean rank of observations in cell (i, s) and N is the total
number of observations. RTEs lie in (0, 1); p_hat = 0.5 indicates no treatment effect
(the cell's distribution is stochastically equal to the marginal).

**Step 3: Per-group rank covariance matrix**

For between-group i with n_i subjects and t within-levels, pivot ranks to a wide
matrix R_i of shape (n_i × t). Compute the sample covariance scaled by N:

```
S_hat_i = cov(R_i, ddof=1) / N        shape (t × t)
```

This is the Brunner et al. (2002) notation: S_hat_i = (1/N) · (1/(n_i−1)) · R_i^T C_n R_i,
where C_n is the centering matrix. The factor 1/N (not 1/N^2) ensures that the total
covariance matrix V_hat_N defined in Step 4 has the correct asymptotic scale.

**Step 4: Block-diagonal total covariance**

```
V_hat_N = block_diag(S_hat_1/n_1,  S_hat_2/n_2,  ...,  S_hat_a/n_a)      shape (at × at)
```

This is equivalent to Brunner et al.'s V_hat_N = N · block_diag(S_hat_i/n_i) with
S_hat_i = cov(R_i)/N^2, which gives the same result. The implementation uses the
reformulation above to avoid numerical overflow with large N.

**Step 5: Idempotent projection matrices** (at × at, applied directly, no pseudoinverse):

```
T_between = kron(I_a − J_a/a,   J_t/t )     rank a−1
T_within  = kron(J_a/a,          I_t − J_t/t) rank t−1
T_inter   = kron(I_a − J_a/a,   I_t − J_t/t) rank (a−1)(t−1)
```

where I = identity matrix, J = matrix of ones (all elements = 1).

**Step 6: ATS per effect**

```
ATS = N · (p_hat^T T p_hat) / tr(T V_hat_N)
```

where p_hat is the (at × 1) vector of RTEs arranged row-major (group 1 time 1,
group 1 time 2, …, group a time t).

**Step 7: Box-approximation degrees of freedom (df1)**

```
df1 = tr(T V_hat_N)^2 / tr(T V_hat_N T V_hat_N)
```

The ATS is approximately F(df1, ∞)-distributed (Box approximation).

**Step 8: p-values**

- **Within and Interaction effects:** F(df1, ∞) ≡ χ^2(df1)/df1:

  ```
  p = 1 − χ^2_cdf(ATS · df1,  df = df1)
  ```

- **Between effect:** Finite df2 via Satterthwaite marginal-covariance approximation:

  ```
  λ_i = (1_t^T S_hat_i 1_t) / (t^2 · n_i)     marginal variance of group-mean RTE
  df2 = (Σ λ_i)^2 / Σ(λ_i^2 / (n_i − 1))
  p   = 1 − F_cdf(ATS,  dfn = df1,  dfd = df2)
  ```

  For balanced designs (all n_i equal), this simplifies to
  df2 = (a · n_i − a) / (a − 1) = n_i − 1, which matches textbook formulas.
  For unbalanced designs (common in biological data), the Satterthwaite
  approximation is more accurate.

**Extra output:** `results["RTE"]`: a DataFrame with columns
`[between_group, within_level, RTE, n]` for each cell.

**Post-hoc tests (built-in, triggered per significant ATS effect):**

All comparisons from all triggered effects are pooled and Holm-corrected together.

| Triggered effect | Comparisons performed | Test used |
|---|---|---|
| Between factor significant | All pairs of between-groups (collapsed over within levels) | Mann-Whitney U |
| Within factor significant | All pairs of within-levels (all subjects, paired by subject) | Wilcoxon signed-rank |
| Interaction significant | All between-group pairs at each individual within-level | Mann-Whitney U |

- **Effect size (MWU):** rank-biserial r = |2U − n_1n_2| / (n_1n_2)
- **Effect size (Wilcoxon):** rank-biserial r = |2W − n(n+1)/2| / (n(n+1)/2)
- **Output:** `pairwise_comparisons` list; `posthoc_test` = `"Pairwise Wilcoxon/MWU (Holm-corrected)"`
- If no effect is significant, `pairwise_comparisons` is empty.

**Warnings emitted:**
- Any group n < 3: covariance estimation unreliable
- min(n_i) × t < 6: very few observations per cell

> **Primary reference:** Brunner, E., Domhof, S., & Langer, F. (2002). *Nonparametric
> Analysis of Longitudinal Data in Factorial Experiments.* Wiley, New York.
>
> **Reference R implementation:** `nparLD` package (Noguchi, K., Gel, Y. R.,
> Brunner, E., & Konietschke, F., 2012. nparLD: An R Software Package for the
> Nonparametric Analysis of Longitudinal Data in Factorial Experiments. *Journal
> of Statistical Software*, 50(12), 1–23.)

---

## 3) Validation against R/nparLD

The Python implementation of `perform_brunner_langer_ats()` was validated against
R's `nparLD::f1.ld.f1()` (version 2.2) on the **Orthodont** dataset (27 subjects,
4 time points, F1-LD-F1 design: Sex as between-factor, Age as within-factor,
N = 108 total observations).

**Validation script:** `validation/validate_brunner_langer_orthodont.py`

### Results

All statistics matched to 2 decimal places (tolerance 0.005):

| Effect | ATS | df1 | df2 | p-value | Match? |
|--------|----:|----:|----:|--------:|--------|
| Sex (between) | 8.798 | 1.00 | 17.57 | 0.0084 | Yes |
| Age (within) | 46.191 | 2.56 | — | <0.001 | Yes |
| Sex × Age (interaction) | 1.872 | 2.56 | — | 0.141 | Yes |

All 8 cell RTEs also matched exactly (max absolute difference < 0.0001).

**Notes on p-value for the between-effect:**

R's `ANOVA.test` reports the raw chi-square p-value for the between-effect
(p = 0.003). The Python implementation and R's `ANOVA.test.mod.Box` both
use the Satterthwaite F-distribution with finite df2 (p = 0.0084). The latter
is more conservative and correct for small between-group sample sizes. Both
implementations agree.

### Covariance scaling: a note

During development, an error in the covariance scaling was identified and corrected.
The incorrect formula (`V_hat_i = cov(R_i)/N^2`) produced ATS values inflated by a
factor of N. The correct formula is:

```
S_hat_i = cov(R_i, ddof=1) / N      [nonparametricanovas.py, line 1187]
```

This is consistent with Brunner et al. (2002, eq. 3.1) and confirmed by the
R/nparLD comparison above.

---

## 4) Primary p-value selection policy: interaction_first

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

## 5) Post-hoc pipeline

Post-hoc tests are **built directly into each of the three nonparametric ANOVA
functions** (`src/analysis/nonparametricanovas.py`). They run automatically at the
end of each function; no external orchestration is needed.

### Design principle

Post-hoc comparisons are only computed when the relevant omnibus effect is
significant (p < alpha). All comparisons from all triggered effects within one
function call are pooled together before Holm correction is applied, so the
family-wise error rate is controlled across the entire set of comparisons
produced by that test.

### Implementation

Four shared helper functions (defined at module level, before `perform_friedman_test`):

| Helper | Role |
|---|---|
| `_holm_correct(p_values)` | Holm step-down correction; returns corrected p-values in original order |
| `_wilcoxon_posthoc_comp(arr1, arr2, label1, label2, alpha)` | Paired Wilcoxon signed-rank; returns raw-p comparison dict |
| `_mwu_posthoc_comp(arr1, arr2, label1, label2, alpha)` | Mann-Whitney U; returns raw-p comparison dict |
| `_apply_holm(raw_comps, alpha)` | In-place Holm correction + significance update on a list of comparison dicts |

### Per-test post-hoc logic

**Friedman**: triggered if overall p < alpha:
- All k·(k−1)/2 within-level pairs tested with Wilcoxon signed-rank (paired)
- Holm correction across all pairs simultaneously

**Freedman-Lane**: triggered per effect:
- Factor A significant → all Factor A level pairs (MWU, collapsed over Factor B)
- Factor B significant → all Factor B level pairs (MWU, collapsed over Factor A)
- Interaction significant → all cell pairs (MWU)
- Holm correction across all comparisons from all triggered effects

**Brunner-Langer ATS**: triggered per ATS effect:
- Between factor significant → all between-group pairs (MWU, collapsed over time)
- Within factor significant → all within-level pairs (Wilcoxon signed-rank, all subjects)
- Interaction significant → all between-group pairs at each individual within-level (MWU)
- Holm correction across all comparisons from all triggered effects

### Output fields in result dict

```
pairwise_comparisons : list of dicts (one per comparison; empty if nothing significant)
posthoc_test         : str describing the method, or None
```

Each comparison dict contains:

| Key | Content |
|---|---|
| `group1`, `group2` | Full group labels (e.g. `"Sex=Male, Age=8"`) |
| `test` | `"Wilcoxon Signed-Rank"` or `"Mann-Whitney U"` |
| `statistic` | Test statistic (W or U) |
| `p_value` | Holm-corrected p-value |
| `corrected` | `True` |
| `correction` | `"Holm"` |
| `significant` | `True` / `False` at the given alpha |
| `effect_size` | Rank-biserial correlation r (absolute value) |
| `effect_size_type` | `"rank_biserial_r"` |
| `confidence_interval` | `None` (not computed for rank-based tests) |

This format is directly consumed by `ResultsExporter.export_results_to_excel()`
to populate the **Pairwise Comparisons** sheet without any further transformation.

---

## 6) Multiple testing correction

All post-hoc comparisons use **Holm step-down correction**, implemented directly
via the `_holm_correct()` helper function in `src/analysis/nonparametricanovas.py`.

The Holm procedure orders all m raw p-values p_(1) ≤ p_(2) ≤ … ≤ p_(m) and
adjusts each as p_tilde_(k) = min(1, max_{j≤k}(m − j + 1) · p_(j)). This controls
the family-wise error rate (FWER) at level alpha without assuming independence
between tests, and is uniformly more powerful than Bonferroni.

**Scope of correction:** All comparisons produced within a single function call
(across all triggered effects) are corrected together as one family. For example,
if both the between and interaction effects are significant in Brunner-Langer,
their combined set of comparisons is Holm-corrected as a single family.

---

## 7) Result-dict structure

All three functions return an identical dict schema, compatible with the
downstream exporter, interpreter, and plotting code. Key fields:

| Field | Content |
|---|---|
| `test` | Human-readable test name, e.g. `"Mixed ANOVA [Brunner-Langer ATS Fallback]"` |
| `model_class` | `"Friedman"` / `"Freedman-Lane Permutation"` / `"Brunner-Langer ATS"` |
| `StatisticType` | Description of the test statistic used |
| `anova_table` | DataFrame with per-effect statistics (Source, ATS/F/Chi2, p, df1, df2) |
| `factors` | List of dicts, one per main effect |
| `interactions` | List of dicts, one per interaction (empty for Friedman) |
| `primary_effect` | Dict identifying the gate effect (source, kind, p_value) |
| `primary_effect_policy` | `"interaction_first"` |
| `analysis_note` | Human-readable explanation suitable for publication methods section |
| `warnings` | List of strings with statistical caveats |
| `p_value` | Gate p-value (from primary effect) |
| `statistic` | Gate test statistic |
| `descriptive` | Per-cell descriptive statistics (n, mean, sd, stderr, median, min, max) |
| `posthoc_test` | String describing the post-hoc method used, or `None` if not triggered |
| `pairwise_comparisons` | List of comparison dicts (see Section 5); empty list if not triggered |
| `RTE` | (Brunner-Langer only) DataFrame of Relative Treatment Effects per cell |
| `n_permutations` | (Freedman-Lane only) Number of permutations used |

The `F` and `Wald_Chi2` fields in `factors` and `interactions` dicts contain
the actual test statistic (Chi^2, permutation F, or ATS), aliased for downstream
compatibility. They are **not** classical ANOVA F-values.

---

## 8) Statistical properties and choice of methods

### Why not classical nonparametric alternatives?

| Scenario | Naive alternative | Why ATS/permutation is preferred |
|---|---|---|
| Mixed ANOVA with non-normal data | Separate Kruskal-Wallis + Friedman | Ignores the mixed design structure; no joint test for interaction |
| Two-way ANOVA with non-normal data | Scheirer-Ray-Hare test | Known to have inflated Type I error for unequal group sizes |
| Repeated measures | Friedman | Yes, standard and appropriate; used here |

### Relative Treatment Effects vs. means

The Brunner-Langer ATS is based on **Relative Treatment Effects (RTEs)**, not
means or medians. An RTE answers: *"What is the probability that a randomly
selected observation from this group/time combination is smaller than a randomly
selected observation from the overall population?"*

- RTE = 0.5: no stochastic ordering; group distributes identically to the population
- RTE > 0.5: group stochastically dominates the population
- RTE < 0.5: group is stochastically dominated by the population

RTEs are invariant to monotone transformations of the outcome and require no
specific distributional shape, making them appropriate for ordinal-like continuous
data (pain scales, grip strength, biomarker concentrations with floor/ceiling effects).

### Sample size considerations

| Method | Minimum recommended | Notes |
|---|---|---|
| Friedman | n ≥ 5 subjects | Chi-square approximation acceptable; exact test for n < 5 |
| Freedman-Lane | n ≥ 3 per cell | p-value resolution limited by permutation count at small n |
| Brunner-Langer ATS | n_i ≥ 3 per group | Covariance estimation requires ≥ 2 subjects; ≥ 3 recommended |

All three methods are explicitly designed for small-sample biological research and
avoid the asymptotic requirements (n > 30 per subgroup) of the legacy GLM/GEE pipeline.

---

## 9) Suggested methods-section wording

### For Friedman (repeated-measures):

> "Due to violation of the normality assumption (Shapiro–Wilk test, p < 0.05),
> a Friedman test was used to assess the effect of [within-factor] on [outcome]
> (k = [levels], n = [subjects]; Chi^2([df1]) = [statistic], p = [p-value]).
> Post-hoc pairwise comparisons were conducted using Wilcoxon signed-rank tests
> for all [k·(k−1)/2] level pairs, with Holm correction for multiple comparisons.
> Effect sizes are reported as rank-biserial correlations (r)."

### For Freedman-Lane (two-way):

> "Normality and/or variance homogeneity assumptions were not met; a
> Freedman-Lane permutation test (5000 permutations, Type III SS) was used to
> assess main effects of [factor A] and [factor B] and their interaction on
> [outcome] (F_[effect]([df1], [df2]) = [F], p_perm = [p]).
> For each significant effect, pairwise post-hoc comparisons were performed using
> Mann-Whitney U tests (between groups) or cell-level comparisons (interaction),
> with Holm correction applied across all comparisons simultaneously.
> Effect sizes are reported as rank-biserial correlations (r)."

### For Brunner-Langer ATS (mixed):

> "Due to violation of normality and/or sphericity assumptions, a nonparametric
> ANOVA-Type Statistic (ATS; Brunner, Domhof & Langer, 2002) was computed for
> the mixed design ([between-factor] as between-subjects factor,
> [within-factor] as within-subjects factor, F1-LD-F1 design,
> a = [groups], t = [time points], N = [total obs]).
> Global mid-ranks were used to compute Relative Treatment Effects (RTEs);
> degrees of freedom were Box-approximated for within/interaction effects
> and Satterthwaite-corrected for the between-subjects effect
> (df2 = [value]). Analysis was performed in Python using a validated
> implementation verified against the R package nparLD (Noguchi et al., 2012).
> Post-hoc pairwise comparisons were performed for each significant effect:
> Mann-Whitney U tests for between-group contrasts (including group-by-time-point
> simple effects for a significant interaction) and Wilcoxon signed-rank tests
> for within-level contrasts. All comparisons were Holm-corrected as a single
> family. Effect sizes are reported as rank-biserial correlations (r)."

---

## 10) Checklist for academic review

1. **Friedman (RM):** Confirm that the Chi-square approximation is acceptable
   for your sample size. For very small n (≤5), consider reporting the exact
   Friedman p-value if available.

2. **Freedman-Lane (Two-Way):** Confirm that 5000 permutations provide sufficient
   p-value resolution for your alpha level. At α = 0.05, the 95% CI for a true
   p = 0.05 is approximately [0.044, 0.056] with 5000 permutations.

3. **Brunner-Langer (Mixed):** Note that the ATS uses **Relative Treatment Effects**
   (RTEs), not means or medians. An RTE of 0.5 indicates no stochastic shift
   relative to the pooled population. Reviewers unfamiliar with RTEs may require
   explanation in the methods section (see Section 9 above).

4. **Between-effect df_2 (Brunner-Langer):** For unbalanced designs, the
   Satterthwaite-approximated df_2 may differ substantially from the simple
   Σ(n_i−1)/(a−1) formula. The exact value is reported in `anova_table`.

5. **p-value for the between effect (Brunner-Langer):** The implementation uses
   an F-distribution with Satterthwaite df_2 (equivalent to R's
   `ANOVA.test.mod.Box`). This is more conservative than the raw chi-square
   approximation and is preferred for between-subjects effects with small group
   sizes. Both values are verifiable via the R/nparLD comparison script.

6. Confirm that **Holm step-down correction** is acceptable for your reviewers.
   Holm controls the family-wise error rate without assuming independence and is
   uniformly more powerful than Bonferroni. If a reviewer requires a specific
   alternative (e.g., Bonferroni, Benjamini–Hochberg FDR), this can be changed
   in `_holm_correct()` in `src/analysis/nonparametricanovas.py` without touching
   the rest of the post-hoc logic.

7. For mixed designs: rank-based methods produce **population-averaged** effects.
   If subject-specific (random-effects) estimates are of scientific interest, a
   Linear Mixed Model (LMM) should be considered for confirmatory analysis once
   the distributional assumption is addressed (e.g., transformation, robust
   variance estimation).

8. **Reduced model formulas (Freedman-Lane):** Confirm that each reduced model
   retains all effects except the one being tested (Type III logic). This is
   critical for unbalanced designs where effects are non-orthogonal.

9. Report the `analysis_note` text (or a version of it) in your methods section.
   It includes the specific test used, key parameters, and sample sizes.

10. **Validation:** The Brunner-Langer implementation can be independently verified
    by running `validation/validate_brunner_langer_orthodont.py`, which compares
    ATS, df1, df2, p-values, and RTEs against R's nparLD package on the Orthodont
    reference dataset (see Section 3).

---

## 11) References

- Brunner, E., Domhof, S., & Langer, F. (2002). *Nonparametric Analysis of
  Longitudinal Data in Factorial Experiments.* Wiley, New York.

- Noguchi, K., Gel, Y. R., Brunner, E., & Konietschke, F. (2012). nparLD: An R
  Software Package for the Nonparametric Analysis of Longitudinal Data in
  Factorial Experiments. *Journal of Statistical Software*, 50(12), 1–23.

- Freedman, D., & Lane, D. (1983). A nonstochastic interpretation of reported
  significance levels. *Journal of Business and Economic Statistics*, 1(4), 292–298.

- Holm, S. (1979). A simple sequentially rejective multiple test procedure.
  *Scandinavian Journal of Statistics*, 6(2), 65–70.

- Box, G. E. P. (1954). Some theorems on quadratic forms applied in the study
  of analysis of variance problems. *Annals of Mathematical Statistics*, 25(2),
  290–302.
