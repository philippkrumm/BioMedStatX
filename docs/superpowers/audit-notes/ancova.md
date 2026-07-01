# Audit note: `ancova` recipe

Recipe location: `src/core/help_content.py:331` (`"id": "ancova"`).

Ground-truth dispatch path (traced, not guessed):
`AnalysisManager.analyze` (`src/analysis/analysis_core.py:143`) reaches the
clinical-model dispatch block gated at `analysis_core.py:540`
(`if kwargs.get('test') in ('ancova', 'two_way_ancova', 'lmm', ...)`). For
`test in ('ancova', 'two_way_ancova')` it instantiates the real model at
`analysis_core.py:595` (`model = ANCOVAModel()`), imported at `:542`, and calls
`model.fit(df, dv=value_cols[0], between_factors=between_factors, covariates=covariates)`
at `:597`. This block fits, then exports and returns from inside the same block
(export at `:783`), so ancova never reaches the parallel
`StatisticalTester._run_ancova_logged` path in
`src/statistical_testing/advanced_pipeline.py:187`. The actual statistics live in
`ANCOVAModel` (`src/analysis/clinical_models.py:90`): "ANCOVA via statsmodels OLS
with Type III SS (Sum contrasts)".

Bucket-to-test routing: two-plus factor columns with a filled Covariates bucket
and no Subject ID infer `ancova` in `_ap_build_analysis_context`
(`src/autopilot/statistical_analyzer_autopilot_pipeline.py:1241`,
`context["inferred_test"] = "ancova"`; single continuous factor with a covariate
routes to `linear_regression`, `:774`/`:786`).

The citation anchor (symbol name or quoted string) is authoritative; the line
number is only a navigation hint.

## What the model actually does (verified)

- **Model:** OLS via `smf.ols(formula, data=...)` where
  `formula = "{dv} ~ C(f, Sum) [* C(f2, Sum)] + {covariates}"`
  (`ANCOVAModel.fit`, `clinical_models.py:131`-`136`). Type III SS with
  Sum-to-zero contrasts: `anova_lm(model, typ=3)` (`:138`).
- **Omnibus p / F:** the main-effect row of the primary between factor, keyed
  `f"C({primary_factor}, Sum)"` (`as_results_dict`, `clinical_models.py:557`-`560`),
  exposed as top-level `p_value` / `statistic` (`:577`-`578`).
- **Effect size:** partial eta squared = `ss_factor / (ss_factor + ss_residual)`
  (`clinical_models.py:562`-`571`, `effect_size_type = "partial_eta_squared"` at
  `:580`).
- **Adjusted means (EMMs):** `adjusted_means` (`clinical_models.py:181`) predicts
  over a balanced reference grid with every other between-factor level weighted
  equally and covariates fixed at their grand mean (`:196` `cov_means`, `:207`-`214`
  grid). Each level reports `adjusted_mean`, `raw_mean`, `raw_sd`, `n`. Surfaced to
  the user in the descriptive summary table (raw mean and adjusted mean side by
  side) at `src/export/report_summaries.py:492`-`507`
  (`_build_descriptive_summary`, `model_type == "ANCOVA"`).
- **Slope-homogeneity assumption:** `check_regression_slope_homogeneity`
  (`clinical_models.py:141`) fits `{dv} ~ C(factor, Sum) * {cov}` per
  factor-covariate pair and reads the interaction row's F and p
  (`:160`-`172`); `assumption_holds = p_value > self._alpha` with
  `self._alpha = 0.05` (`:107`, `:172`). Surfaced to the user as
  "Slope Homogeneity: {factor}:{cov}" assumption rows at
  `report_summaries.py:306`-`320`.
- **Heterogeneous-slopes follow-up:** if any interaction p < alpha
  (`slopes_heterogeneous`, `clinical_models.py:522`), `run_simple_slopes_and_jn`
  (`:349`) runs Simple Slopes (pick-a-point at mean, mean +/- 1 SD) and a
  Johnson-Neyman interval (2-level factor), reported at
  `src/export/report_stat_rows.py:82`-`107` ("Simple Slopes & Johnson-Neyman").
- **Data cleaning:** `fit` drops rows with any NaN in
  `[dv] + between_factors + covariates` (`clinical_models.py:118`), so one row per
  subject with a missing outcome or covariate is silently excluded.

## Claim table

| # | Claim (title/summary/html) | Verdict | Citation |
|---|-----------------------------|---------|----------|
| 1 | Title "Comparing groups while correcting for a background variable (ANCOVA)" | correct | test label built as `"ANCOVA"` (one factor) / `"Two-Way ANCOVA"` (two) at `clinical_models.py:575` (`ANCOVAModel.as_results_dict`) |
| 2 | Summary "Like ANOVA, but you control for an additional numeric variable" | correct | OLS model adds covariate terms to the factor model: `formula = f"{dv} ~ {factor_terms} + {cov_terms}"` (`clinical_models.py:134`, `ANCOVAModel.fit`) |
| 3 | "ANCOVA adjusts the group averages mathematically to account for the background variable, allowing for a fairer comparison" | correct | covariate-adjusted EMMs computed with covariates fixed at grand mean: `adjusted_means` (`clinical_models.py:181`, `cov_means` at `:196`, predictions at `:216`); shown as "adjusted mean" vs "raw mean" in the report (`report_summaries.py:505`) |
| 4 | "The covariate must not be affected by the treatment itself ... overadjustment bias" | correct (domain advice, not code-checkable) | Not enforced or checked in code; standard ANCOVA design guidance. No code contradicts it. Recorded as valid statistical caution, not a code behavior. |
| 5 | "The correcting variable goes into the Covariates bucket. It must be a number (not a group label). You can add more than one." (added: multiple covariates) | correct | covariates enter the formula as raw numeric terms `cov_terms = " + ".join(self._covariates)` (`clinical_models.py:133`), not `C(...)` factors; a non-numeric covariate is blocked pre-fit by `validate_outcome(df[_cov])` (`analysis_core.py:561`-`566`, `validators.py:417` `validate_outcome`). Multiple covariates are supported: `covariates` is a list threaded through `fit(..., covariates)` (`clinical_models.py:114`) and each is added to the formula and to `covariate_effects` (`clinical_models.py:539`-`550`). |
| 6 | "What the app checks and runs": reports adjusted average next to raw average; checks equal-slopes assumption; if it fails, runs a follow-up over the covariate range; when significant, compares adjusted averages with p-value adjustment (NEW section) | correct | adjusted vs raw means (`clinical_models.py:181`, report `report_summaries.py:492`-`507`); equal-slopes check (`check_regression_slope_homogeneity`, `clinical_models.py:141`, report `report_summaries.py:306`-`320`); heterogeneous-slopes follow-up = Simple Slopes + Johnson-Neyman (`run_simple_slopes_and_jn`, `clinical_models.py:349`, report `report_stat_rows.py:82`-`107`); significant-effect post-hoc on adjusted means with correction (see "Post-hoc" section below) |
| 7 | Data layout: "One row per subject. One measurement column (outcome), one group column, and one or more numeric covariate columns." | correct | `fit` reads `dv`, `between_factors`, and `covariates` as columns and drops per-row NaN (`clinical_models.py:118`); one-row-per-subject between-subjects design (`design_type = DesignType.INDEPENDENT`, `:111`-`112`). Example table columns (Group, Score_post, Score_baseline, Age) match this shape. |
| 8 | Common mistake: "pre-computed group means" fail because ANCOVA needs raw individual values | correct | the OLS fit and the covariate-slope estimate need row-level (subject-level) data; `fit` regresses individual rows (`clinical_models.py:136`), and `check_regression_slope_homogeneity` estimates the factor x covariate interaction from rows (`:157`-`161`). Two-row group-mean input has n=2 and no within-group covariate variation, so the slope is unidentifiable. |
| 9 | Bucket mapping: DV = outcome; Factor 1 = group label; Covariates = numeric variable(s); Factor 2 empty (else Two-Way ANCOVA); Subject ID empty | correct | DV -> `value_cols[0]` (`analysis_core.py:597`); Factor 1 -> `between_factors[0]` from `factor_columns`/`between_factors` (`analysis_core.py:596`); a second factor makes `len(between_factors) == 2` -> label "Two-Way ANCOVA" (`clinical_models.py:575`); Covariates -> numeric `covariates` list (`analysis_core.py:597`); Subject ID left empty (independent design, `clinical_models.py:111`-`112`). Routing to `ancova` when a covariate is present and no subject: `_ap_build_analysis_context:1241`. |
| 10 | Checklist: one row per subject not per time point; covariates numeric; Factor 1 a group label; raw individual values not aggregated means | correct | one-row-per-subject independent design (claim 7); numeric covariates (claim 5); Factor 1 as a between factor (claim 9); raw values (claim 8) |

## Alpha / adjustment control check

Default `alpha` is 0.05 for ancova. The clinical dispatch calls
`model.fit(...)` at `analysis_core.py:597` without an `alpha` argument, so the
`ANCOVAModel.fit` default `alpha=0.05` (`clinical_models.py:114`) is used and
stored as `self._alpha` (`:119`). Every downstream threshold uses it:
slope-homogeneity `assumption_holds = p > self._alpha` (`:172`), post-hoc
`significant = p < self._alpha` (`:345`), heterogeneity gate `p_value < self._alpha`
(`:522`). The recipe prints no numeric alpha, so there is nothing to contradict.

## Post-hoc method and adjustment (exact) — and a code discrepancy

- Post-hoc contrasts run on the EMMs (covariate-adjusted marginal means), not on
  raw means: `emm_contrasts` builds a linear functional per level from the same
  balanced grid as `adjusted_means` and evaluates contrasts via `result.t_test`
  on the OLS fit (`clinical_models.py:279`-`347`).
- `as_results_dict` picks the family by whether a control group is known
  (`clinical_models.py:499`-`516`):
  - control group present -> `method="vs_control"`, treatment-vs-control family,
    single-step **multivariate-t** adjustment (`_mvt_pvalues`, `:259`-`277`;
    label `"EMM contrasts vs control '{ctrl}' (multivariate-t)"`).
  - otherwise -> `method="pairwise"`, all C(G,2) contrasts, **Holm-Bonferroni**
    (`multipletests(..., method="holm")`, `:332`; label
    `"EMM pairwise contrasts (Holm-Bonferroni)"`).
- **Live behavior for the ordinary autopilot ancova path: always the pairwise
  Holm-Bonferroni branch.** The clinical dispatch at `analysis_core.py:597` calls
  `model.fit(df, dv=..., between_factors=..., covariates=...)` and never passes
  `control_group`, so `ANCOVAModel._control_group` stays `None` (its `fit`
  default, `clinical_models.py:114`/`:122`), `ctrl` resolves to `None`
  (`:505`-`511`), and the vs-control branch is skipped. The recipe body deliberately
  says only "adjusts the p-values for the number of comparisons", which is true of
  both branches; the exact method name (Holm-Bonferroni) is recorded here per the
  recipe-economy rule and kept out of the shipped text.

## Data-structure control check

The recipe's layout (one measured-value column, one group-label column, one or
more numeric covariate columns, one row per subject) matches
`ANCOVAModel.fit`: columns are read directly and per-row NaN dropped
(`clinical_models.py:118`); the design is between-subjects
(`DesignType.INDEPENDENT`, `:111`-`112`); no auto-pivot applies (ancova is not a
paired/repeated wide signature handled by `_ap_maybe_pivot`). Covariates are
consumed as numeric regression terms, not factors (claim 5). The "pre-computed
group means" example is correctly flagged as failing (claim 8).

## Unclear / possible code bug

1. **vs-control EMM post-hoc is unreachable on the live ancova path (worth a
   human decision, not fixed here).** `ANCOVAModel` fully implements the
   treatment-vs-control multivariate-t family (`emm_contrasts(method="vs_control")`,
   `clinical_models.py:305`-`327`), and a parallel runner
   (`advanced_pipeline.py:175`-`190`) is written to prompt for a control group and
   pass it through `_run_ancova_logged(..., control_group=ancova_control)`. But the
   clinical dispatch in `analysis_core.py:594`-`598` intercepts `test='ancova'`
   first, fits `ANCOVAModel` without a `control_group`, and returns after export
   (`:783`), so `_run_ancova_logged` is never reached for ancova. Net effect: the
   ancova post-hoc is always the pairwise Holm-Bonferroni branch, and the
   vs-control multivariate-t code plus the advanced-pipeline control-group prompt
   are dead for this test. (LMM does pass `control_group` through the same clinical
   dispatch at `analysis_core.py:608`-`622`, so the omission looks specific to the
   ancova branch rather than intentional.) This does not contradict the recipe text
   (which names no post-hoc method), so no recipe change is needed; flagged for a
   human to decide whether the ancova dispatch should also wire up
   `control_group_callback`.

2. **Two-Way ANCOVA slope-homogeneity check tests only single factor-by-covariate
   interactions (documented, not acted on).** `check_regression_slope_homogeneity`
   loops over each `(covariate, factor)` pair and fits
   `{dv} ~ C(factor, Sum) * {cov}` in isolation (`clinical_models.py:153`-`161`),
   so for a two-factor design it never tests the joint or higher-order
   factor x factor x covariate interaction. For the one-factor ancova the recipe
   describes, this is exactly the equal-slopes assumption and the claim is correct;
   the note only matters for the Two-Way ANCOVA variant, which the recipe mentions
   but does not describe in detail. Recorded for completeness, no recipe or code
   change.
