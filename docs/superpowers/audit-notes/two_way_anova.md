# Audit note: `two_way_anova` recipe

Recipe location: `src/core/help_content.py:142` (`"id": "two_way_anova"`).

Ground-truth dispatch path (traced, not guessed):
`AnalysisManager.analyze` routes `test == "two_way_anova"` at
`src/analysis/analysis_core.py:938` -> `StatisticalTester.prepare_advanced_test`
(assumption checks) `src/analysis/statisticaltester.py:1029` ->
`StatisticalTester.perform_advanced_test` `:1112` ->
`perform_advanced_test_pipeline` `src/statistical_testing/advanced_pipeline.py:26`.
The parametric branch calls `StatisticalTester._run_two_way_anova_logged`
(`statisticaltester.py:1356`) -> `_run_two_way_anova` (`:2185`). Post-hoc for a
significant result runs through `AdvancedPostHocEngine.execute`
(`src/statistical_testing/engines/advanced_posthoc.py:14`) ->
`PostHocFactory.perform_posthoc_for_anova("two_way", ...)`
(`src/analysis/posthoc_core.py:1742`) ->
`TwoWayPostHocAnalyzer.perform_test` (`posthoc_core.py:88`). The non-parametric
branch calls `perform_freedman_lane_test`
(`src/analysis/nonparametricanovas.py:391`). Bucket-to-test routing lives in
`_ap_build_analysis_context`
(`src/autopilot/statistical_analyzer_autopilot_pipeline.py:1054`).

The citation anchor (symbol name or quoted string) is authoritative; the line
number is only a navigation hint.

## Claim table

| # | Claim (from title/html) | Verdict | Citation |
|---|-------------------------|---------|----------|
| 1 | Title "Two independent grouping factors (Two-Way ANOVA)" | correct | `src/core/help_content.py:144` (`"title"`); test label built as `"Two-Way ANOVA ({between[0]} * {between[1]})"` at `statisticaltester.py:2207` (`StatisticalTester._run_two_way_anova`) |
| 2 | Use when there are two separate ways of grouping (e.g. Treatment and Sex) and every combination is measured | correct | routing: two factor columns and no Subject ID -> `two_way_anova` at `_ap_build_analysis_context:1197` (`context["between_factors"] = factor_columns[:2]`, `:1198` `inferred_test = "two_way_anova"`); design validated as exactly two between factors at `validators.py:263` (`validate_test_design`, `"Two-Way ANOVA requires two between factors."`) |
| 3 | You want to know whether each factor has an effect and whether they interact | correct | main effects computed per factor into `results["factors"]` at `_run_two_way_anova:2262`-`2276` (each with F, p, partial eta squared); interaction computed into `results["interactions"]` at `:2287`-`2298`; top-level `results["p_value"]` is set to the interaction p-value at `:2299` |
| 4 | No subject appears more than once; if the same subjects are measured at multiple time points or conditions, use Mixed ANOVA | correct | two factors plus a Subject ID route to `mixed_anova`, not two-way, at `_ap_build_analysis_context:1182`-`1195` (`context["inferred_test"] = "mixed_anova"`); two-way is the no-subject branch at `:1196`-`1198` |
| 5 | Data layout: one row per measurement, both group columns present in every row alongside the measured value (long format) | correct | cells are read as factor_a x factor_b combinations from column values at `_run_two_way_anova:2226` (`df.groupby([factor_a, factor_b])`) and `_extract_raw_data_two_way_anova` (`statisticaltester.py:1418`, `df[(df[a]==a_val) & (df[b]==b_val)][dv]`); membership comes from column row values (long format), not headers. Note: this is a between-subjects design with replicate rows per cell, so "one row per measurement" is the accurate framing (the recipe previously said "one row per subject", corrected). |
| 6 | Common mistake: both factors hidden in column names; the app cannot separate them into Factor 1 and Factor 2 | correct | the two-factor long structure is required (claim 5); the paired/repeated auto-melt (`_ap_maybe_pivot` -> `_detect_wide_format`, pipeline `:1961`) only recovers a single value dimension from a subject-keyed wide table and cannot split one merged header into two crossed factors, so a `Control_Male / Control_Female / ...` layout is genuinely unusable |
| 7 | Bucket mapping: Dependent Variable = measurement column; Factor 1 and Factor 2 = the two group columns; Subject ID and Covariates left empty | correct | DV bucket at `_ap_init_ui` (`"Dependent Variable"`, `accepted_kinds={"numeric"}`); Factor 1 (`:263`) and Factor 2 (`:275`) hold the two grouping columns; two factors with empty Subject and empty Covariates keep the inference on the two-way branch at `_ap_build_analysis_context:1196`-`1198`. Subject ID filled -> `mixed_anova` (`:1195`); Covariates filled -> `two_way_ancova` (`:1242`) |
| 8 | Checklist: both group columns hold category labels; numeric group codes are fine, a continuous measurement is not | correct (rewritten) | groups are read from raw column values with no type check: `_run_two_way_anova:2226` (`df.groupby([factor_a, factor_b]).size()`) and `_extract_raw_data_two_way_anova` (`statisticaltester.py:1418`, `df[(df[a]==a_val) & (df[b]==b_val)][dv]`) key cells directly on the factor column's native values (numeric or string), no groupby failure either way. The non-parametric path explicitly casts factors to string before grouping: `perform_freedman_lane_test:424`-`425` (`df[safe_a].astype(str)`) and, in the statsmodels post-hoc fallback, `statisticaltester.py:2438` (`df['interaction_group'] = df[factor_a].astype(str) + "_" + df[factor_b].astype(str)`). So integer-coded groups work in every branch. The single-factor continuous redirect to correlation/regression only fires for `len(factor_columns) == 1` (`_ap_build_analysis_context:1251`), so with two factors a genuinely continuous numeric column is NOT redirected and would produce many singleton cells. The old wording "text labels, not numbers" was wrong; the new wording accepts numeric group codes and warns against a continuous column. |
| 9 | Checklist: every row has a value in both group columns | correct | rows with a missing factor value are dropped before analysis: `perform_freedman_lane_test:407` (`data[[dv, factor_a, factor_b]].dropna()`); cell extraction reads only present factor combinations at `_run_two_way_anova:2226` |
| 10 | Checklist: no subject is measured more than once (else Mixed ANOVA) | correct | same routing as claim 4; two-way is between-subjects with no Subject column |
| 11 | Checklist: each combination of the two factors has at least a few measurements | correct | pre-flight blocks a cell with fewer than `min_n_block=2` non-NaN values: `advanced_pipeline.py:95` (`validate_samples_for_test(..., min_n_block=2)`) -> `validators.py:377` (`if valid.size < min_n_block: return issue("N_BELOW_MIN", ...)`). The Brown-Forsythe variance check additionally needs n>=3 per group to run (`validate_levene_inputs(..., min_n_per_group=3)`, `assumption_checks.py:279`-`285`), and `_run_two_way_anova:2226`-`2238` warns on empty or unbalanced cells. "A few measurements" is deliberately vague per the recipe-economy rule; exact thresholds recorded here. |
| 12 | Checklist: no group name embedded in a column header | correct | same basis as claim 6 |
| 13 (new section) | "What the app checks and runs": app checks normality and equal spread, runs a two-way ANOVA on normal data reporting each main effect plus the interaction, tries a transformation then a permutation-based fallback if not normal, and adjusts post-hoc p-values for the number of comparisons | correct | assumption route below; parametric vs non-parametric selection below; post-hoc correction below |

## Assumption / correction control check

- Assumptions are checked by the SAME engine as the one-way path,
  `check_normality_and_variance` (`assumption_checks.py:37`), invoked with
  `model_type="twoway"` and formula `Value ~ C(fA) * C(fB)`
  (`prepare_advanced_test`, `statisticaltester.py:1065`-`1099`).
- **Normality:** Shapiro-Wilk on the residuals of the two-factor model
  `Value ~ C(f0) * C(f1)` (`assumption_checks.py:198`-`201` builds the two-way
  formula; `:215` `stats.shapiro(resid_raw)`; normal if `p > 0.05` at `:245`).
  This mirrors the one-way path but fits the full crossed model, not a
  single-factor model.
- **Equal spread:** Brown-Forsythe, i.e. median-centered Levene across the
  factor-combination cells (`assumption_checks.py:274` `test_name = "Brown-Forsythe"`,
  `:285` `stats.levene(*validated_levene_data, center='median')`, equal if
  `p > 0.05` at `:286`). Same test as the one-way path. Two-way is NOT in the
  `("rm", "mixed", "paired")` bypass branch (`:261`), so the variance test does run.
- **Parametric vs non-parametric selection:** driven by residual normality only.
  `need_transform = not ...residuals_normality...is_normal` (`assumption_checks.py:317`).
  Non-normal residuals trigger a transformation dialog (`:327`
  `select_transformation_dialog`; default `log10` if cancelled, `:331`/`:333`);
  if still non-normal after transforming, the engine returns recommendation
  `"non_parametric"` (`:323`). Equal/unequal variance does NOT switch the test
  (Welch-style robustness note at `:315`); only normality does. This matches the
  recipe's plain-language "checks normality and spread; if not normal, transform
  then fall back".

## Which exact test runs

- **Parametric (normal data):** two-way ANOVA via pingouin
  `pg.anova(data=df, dv=dv, between=between, detailed=True)`
  (`_run_two_way_anova:2249`); effect size is partial eta squared
  (`np2`, `:2274`/`:2295`, `effect_size_type = "partial_eta_squared"` at `:2216`).
  Unbalanced designs are handled with Type III sums of squares and a warning
  (`:2236`-`2238`); empty cells emit an LMM-recommendation warning (`:2233`-`2235`).
  A statsmodels fallback (`ols(... C(fA, Sum) * C(fB, Sum) ...)`, `typ=3`) is used
  only if pingouin is unavailable (`:2360`-`2426`).
- **Non-parametric fallback (residuals non-normal after transform):**
  `perform_freedman_lane_test` (`nonparametricanovas.py:391`), a Freedman-Lane
  permutation test with `n_permutations=5000` computing a permutation p-value for
  each of A, B, and A x B (`:399` `p_perm = (#{F_perm >= F_obs} + 1) / (n_perm + 1)`).
  Dispatched at `advanced_pipeline.py:321`-`328`.

## Post-hoc method and adjustment (exact)

- Post-hoc fires only when the omnibus is significant (`p_value < alpha`,
  `advanced_pipeline.py:237`), then `AdvancedPostHocEngine` mode
  `"advanced_parametric"` runs.
- **Default method for two-way is `paired_custom`** (`advanced_posthoc.py:85`
  `default_method = "paired_custom" if test == "two_way_anova" else "tukey"`).
  A method callback / dialog can override to `tukey` or `dunnett`; a custom-pairs
  callback lets the user pick which cell pairs to compare, defaulting to all
  pairs when headless or cancelled (`:106`-`114`).
- **`paired_custom` adjustment is Holm-Sidak** across the selected pairwise
  interaction-cell t-tests: `TwoWayPostHocAnalyzer.perform_test` runs
  `scipy.stats.ttest_ind(..., equal_var=True)` per pair (`posthoc_core.py:256`)
  then `multipletests(pvals, alpha=alpha, method='holm-sidak')`
  (`:268`-`269`, `correction_method = "Holm-Sidak"`). The alternative `paired_fdr`
  uses Benjamini-Hochberg (`:263`-`265`). Effect size per pair is Cohen's d
  (`:280`).
- If overridden to **Tukey HSD**: `pairwise_tukeyhsd(df[dv], interaction_group,
  alpha=alpha)` across all cells, family-wise studentized-range correction, no
  extra adjustment (`posthoc_core.py:157`-`187`). If overridden to **Dunnett**:
  `scipy.stats.dunnett(*samples, control=control_sample)` versus a chosen control
  cell (`:189`-`247`).
- Non-parametric fallback post-hoc (when the omnibus went non-parametric):
  pairwise Mann-Whitney U on marginal / cell simple effects with Holm correction
  (`advanced_posthoc.py:319`-`357`, `_apply_holm`,
  `"Pairwise Mann-Whitney U (marginal / cell simple effects, Holm-corrected)"`).

The recipe body says only "adjusts the p-values for the number of comparisons",
which is true for every branch above (Holm-Sidak, Tukey HSD, Dunnett, Holm). The
named procedures are recorded here per the recipe-economy rule and kept out of
the shipped text.

Note: the pairwise p-values themselves are always the correctly adjusted values
for whichever method ran. The `posthoc_test` *label* attached to the default
`paired_custom` run is a separate, buggy field; see "Unclear / possible code
bug" item 2 below.

## Alpha / adjustment control check

Default `alpha` is 0.05 end to end: `perform_advanced_test_pipeline(..., alpha=0.05)`
(`advanced_pipeline.py:35`), `analysis_core.py:917`-style call sites pass
`alpha=0.05`, `_run_two_way_anova(..., alpha=0.05)` (`statisticaltester.py:2185`),
and post-hoc `alpha = float(payload.get("alpha", 0.05))` (`advanced_posthoc.py:50`).
Significance gate is `p_value < alpha` (`advanced_pipeline.py:237`) and assumption
thresholds are `p > 0.05` (`assumption_checks.py:245`/`:286`). The recipe prints
no numeric alpha, so there is nothing to contradict.

## Data-structure control check

The recipe's two-factor long layout (one measured-value column plus two group-label
columns, both filled on every row) matches the parser: cells are read as
factor_a x factor_b combinations from column row values
(`_run_two_way_anova:2226`, `_extract_raw_data_two_way_anova` at
`statisticaltester.py:1418`, `perform_freedman_lane_test:407`). The "both factors
merged into column headers"
example is correctly flagged as failing: no auto-melt splits a single merged header
into two crossed factors (`_ap_maybe_pivot` handles only the paired/repeated
subject-keyed wide signature). The old "one row per subject" phrasing was corrected
to "one row per measurement" because replicate rows per cell are expected in a
between-subjects design.

## Unclear / possible code bug

None that contradict the recipe text itself. Two observations recorded for the
human, not acted on here:

1. **Numeric factor footgun (documented, not a bug).** With exactly two factor
   columns, the continuous-factor redirect to correlation/regression does not fire
   (`_ap_build_analysis_context:1251` guards on `len(factor_columns) == 1`). A
   continuous numeric second factor would therefore be cast to string cells and
   produce many singleton cells rather than being rejected or redirected. The
   rewritten checklist item now warns the user against using a continuous column as
   a factor. Worth a human decision on whether the code should also guard the
   two-factor case.

2. **Inline post-hoc label mismatch: a real, user-visible display bug, not
   cosmetic.** The inline pingouin post-hoc inside `_run_two_way_anova` uses
   `pg.pairwise_tests(..., padjust='holm')` but sets
   `results["posthoc_test"] = "Tukey HSD Test (Pingouin)"`
   (`statisticaltester.py:2315` the `pairwise_tests` call, `:2347`
   `"corrected": "Holm-Bonferroni"` on each comparison, `:2353`
   `results["posthoc_test"] = "Tukey HSD Test (Pingouin)"`). The label says Tukey
   while the correction actually applied is Holm.

   This is NOT overridden in the default path, contrary to what a first read of the
   override guard suggests. `AdvancedPostHocEngine`'s result replaces
   `res["posthoc_test"]` only when `should_override` is true
   (`advanced_pipeline.py:258`-`266`). One of the `should_override` branches is
   `"Pingouin" in str(current_posthoc) and new_posthoc and "Tukey" in str(new_posthoc)`
   (`:263`), i.e. it only fires when the *replacement* label also contains the
   substring "Tukey". But the default two-way post-hoc method is `paired_custom`
   (`advanced_posthoc.py:85`), whose label is
   `"Custom paired t-tests (Holm-Sidak)"` (`posthoc_core.py:317`,
   `method_name_map["paired_custom"]`), which does not contain "Tukey". None of the
   other three `should_override` conditions match either (`current_posthoc` is
   truthy and is not `"Two-Way ANOVA Post-hoc Tests"`, `"parametric paired
   t-tests"`, or `"pairwise paired t-tests"`). So for the default `paired_custom`
   two-way post-hoc, `should_override` evaluates `False`, `res["posthoc_test"]`
   keeps the inline value `"Tukey HSD Test (Pingouin)"`, and
   `pairwise_comparisons` is separately overwritten with the correct
   `paired_custom`/Holm-Sidak values (`advanced_pipeline.py:255`). The two fields
   go out of sync: the comparisons are Holm-Sidak but the test-name field still
   says Tukey.

   That mislabeled `posthoc_test` string then reaches the user-facing analysis
   log: `analysis_core.py:1362`-`1364` reads `results.get("posthoc_test", None)`
   and, because it contains `"Tukey HSD"`, sets `posthoc_display = "Tukey HSD
   Test"`, which is printed into the log as the post-hoc method used. This means
   the DEFAULT two-way ANOVA post-hoc run through the ordinary autopilot path logs
   "Post-hoc test: Tukey HSD Test" while the pairwise p-values it displays are
   actually Holm-Sidak corrected. The override only actually fires when the user
   explicitly picks the `tukey` method via the post-hoc dialog (then both the
   inline and replacement labels contain "Tukey" and the two happen to agree by
   coincidence, not by working override logic). This is a genuine, user-visible
   misreporting of the correction method for the default post-hoc path, not a
   dead/unreachable code path, and it is worth fixing at the code level (e.g. by
   removing the `"Tukey" in str(new_posthoc)` gate, or by having `_run_two_way_anova`
   not set an inline `posthoc_test` label at all since it is always meant to be
   replaced) even though this audit task does not touch code.
