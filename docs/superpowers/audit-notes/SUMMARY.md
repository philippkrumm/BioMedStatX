# Help Hub content audit: collated code findings

Date: 2026-07-02
Source: `docs/superpowers/audit-notes/*.md`, one file per recipe, produced while
auditing all 12 Help Hub recipes in `src/core/help_content.py` against the current
code (spec: `docs/superpowers/specs/2026-06-30-help-content-audit-design.md`).

This file collates every "Unclear / possible code bug" item from the 12 per-recipe
audit notes. None of these were fixed as part of the content audit (scope was recipe
text only); they are flagged here for a human to triage. Each item links back to its
source note for full citations and reasoning.

No recipe `id` or `category` was changed anywhere in this audit. Full test suite:
313 passed, 4 skipped, 0 failed (`pytest tests/`). `ruff check src/core/help_content.py`:
clean.

## Worth fixing (real, user-visible misbehavior)

1. **Post-hoc method name is mislabeled for the default path in Two-Way ANOVA, Mixed
   ANOVA, and Repeated Measures ANOVA (same bug, three places).** In all three designs,
   an inline `results["posthoc_test"]` label is set before the real post-hoc engine
   runs, and the override guard in `advanced_pipeline.py:258-266` only replaces that
   label when specific substrings match. The engine's actual result
   (`pairwise_comparisons`) is correct, but the label that reaches the user-facing
   analysis log is not:
   - Two-Way: label says "Tukey HSD Test (Pingouin)", correction actually applied is
     Holm-Sidak (`two_way_anova.md` item 2, most fully traced — confirms this reaches
     `analysis_core.py:1362-1364` and prints "Post-hoc test: Tukey HSD Test" in the log).
   - Mixed ANOVA: label says "Pairwise t-tests for interaction (Holm-Bonferroni)",
     engine actually used Tukey (`mixed_anova.md` item 2).
   - Repeated Measures: same pattern (`repeated_measures_anova.md` item 2).
   Fix direction: broaden the `should_override` condition in `advanced_pipeline.py`
   to always sync `posthoc_test` with the method the engine actually ran, rather than
   pattern-matching specific label substrings.

2. **Repeated Measures ANOVA: an outer-exception path skips the Greenhouse-Geisser
   correction, contradicting the documented conservative default.** `_perform_
   comprehensive_sphericity_test`'s inner "cannot test sphericity" fallback correctly
   defaults to GG (matches the CHANGELOG's stated behavior). But if the outer `try`
   throws before that fallback runs (e.g. `pg.sphericity` and the inner fallback both
   fail), the outer `except` at `statisticaltester.py:2690-2701` never sets
   `final_p_value`, so the uncorrected p-value from `pg.rm_anova` silently becomes the
   canonical result — the opposite of "conservative correction applied when sphericity
   can't be tested." Narrow failure mode, but it directly contradicts a documented v2.0
   behavioral guarantee. (`repeated_measures_anova.md` item 1.)

3. **Graph export dispatch has a dead branch and a silent wrong-plot fallback,
   inconsistent with the preview dispatch's error handling.** `AnalysisManager.analyze`'s
   export-time plot dispatch (`analysis_core.py:1469-1535`) has an unreachable
   `elif plot_type == "Strip":` branch (Strip was removed from the UI dropdown but this
   branch was never deleted) and a catch-all `else` that silently renders a Bar plot on
   any unrecognized `plot_type`, only logging a warning. The *preview* dispatch
   (`datavisualizer.py:plot_from_config`) raises `ValueError` for the same case. Today
   this is unreachable (the dropdown only offers Bar/Box/Violin/Raincloud), but the two
   dispatches disagreeing on failure behavior is a latent trap for future plot-type
   additions. (`graph_visualization.md` item 1.)

4. **Logistic regression: Factor 1 is always dummy-coded, including continuous
   predictors — a real usability trap.** `clinical_models.py:1080` unconditionally wraps
   every Factor 1 predictor in `C()`. A continuous predictor placed in Factor 1 (e.g. a
   dose in mg) silently produces one odds ratio per distinct value against a reference
   category, not a per-unit slope. This is a deliberate bucket-semantics convention
   (Factor 1 = grouping, Covariates = continuous), but nothing in the UI warns a user who
   puts a continuous variable in Factor 1 — this is exactly the mistake the recipe's own
   original worked example made before this audit caught it. (`logistic_regression.md`
   item 2.)

5. **Linear regression: `coefficient_table` (per-coefficient SE/t/p/95% CI) is computed
   but never rendered anywhere.** `as_results_dict` builds a full coefficient table
   (`correlation_models.py:780-790`), but no exporter reads that key for
   `model_type == "LinearRegression"` — only the primary predictor's p-value and a plain
   R² reach the user. Looks like an intended-but-unfinished report feature (would matter
   most for multi-covariate models, where the user currently can't see per-covariate
   stats at all). Worth a decision: wire it into the report, or remove the dead
   computation. (`linear_regression.md` item 2.)

6. **Untranslated German checkbox label in an otherwise English application.** The
   checkbox that activates simple linear regression reads
   `QCheckBox("Als Lineare Regression analysieren (Y = a + bX)")`
   (`statistical_analyzer_autopilot_pipeline.py:353`). No i18n system exists anywhere in
   this codebase, so this is a leftover untranslated string, not a deliberate
   localization choice. User-visible outside of Help. (`linear_regression.md` item 3.)

7. **ANCOVA: the vs-control multivariate-t post-hoc path is fully implemented but
   unreachable.** `ANCOVAModel.emm_contrasts(method="vs_control")` and a parallel
   `advanced_pipeline.py:175-190` runner that prompts for a control group both exist, but
   the clinical dispatch (`analysis_core.py:594-598`) fits `ANCOVAModel` without ever
   passing `control_group` and returns before that runner is reached. ANCOVA therefore
   always uses the pairwise Holm-Bonferroni post-hoc; the vs-control code path is dead.
   By contrast, LMM's dispatch three lines below (`:608-622`) does wire `control_group`
   through — the omission looks specific to ancova, not intentional. (`ancova.md` item 1.)

## Worth a defensive fix (low risk today, fragile)

8. **Binary-outcome detector has an operator-precedence bug.** In
   `_ap_build_analysis_context` (`pipeline:1119-1124`), `and` binds tighter than `or` in
   an unparenthesized boolean expression, so a numeric 2-value column is flagged as a
   binary outcome even if its values aren't 0/1 and even if the column name looks like a
   grouping variable. The model's own `ValueError` on non-2-level input is the real
   backstop today, so no user currently reaches a wrong result, but the logic doesn't do
   what its shape suggests. Confirmed independently in both `getting_started.md` and
   `logistic_regression.md`. Fix: add parentheses to make the intended grouping explicit.

9. **`validate_test_design` doesn't check for a subject column on the `mixed_anova`
   branch**, unlike its sibling `repeated_measures_anova` branch which does
   (`validators.py:255-262`). Harmless today because the autopilot only ever routes to
   `mixed_anova` with a Subject ID present, but a non-autopilot caller could dispatch
   `mixed_anova` without one and get a raw pingouin error instead of a clean
   `ModelDesignError`. (`mixed_anova.md` item 1.)

## Code documentation bug (not behavioral)

10. **`CorrelationModel`'s class docstring describes the wrong selection mechanism.**
    It says `method='auto'` "applies Shapiro-Wilk... uses Pearson when both are normally
    distributed (p > alpha)," but the actual `fit` logic is skew/excess-kurtosis
    N-tier gating that never reads the Shapiro p-value. Found during spec review of the
    `correlation` audit (`correlation.md` item 3). Cheap fix: update the docstring to
    match `fit`'s real logic.

## Follow-up items found after this audit (now fixed)

Two more structural gaps surfaced during quality review of the fixes for items 1-9
above (not part of the original 12-recipe pass, so not numbered with it). Both are
now fixed.

- **Help Hub's binary-outcome hint disagreed with real test routing on
  grouping-named columns.** `_ap_is_binary_outcome_for_help` lacked the same
  "column name doesn't look like a grouping variable" guard that the real routing
  function `_classify_binary_outcome` has (the fix for item 8 above), so a column
  like `Treatment_Arm` coded Yes/No could get suggested the wrong Help Hub recipe
  even though real analysis correctly treated it as non-binary. Fixed by making the
  hint delegate to `_classify_binary_outcome` directly (commit `cb45f39`).
- **`ANCOVAModel`, `LinearMixedModel`, and `LogisticRegressionModel.fit()` had no
  structural pre-flight checks** for empty `between_factors`/`covariates`/
  `fixed_effects`/`predictors` or a missing `random_intercept`, unlike the sibling
  `mixed_anova`/`repeated_measures_anova` subject-column check (item 9 above).
  Investigation traced the actual failure modes: `ANCOVAModel` and
  `LogisticRegressionModel` would raise a patsy `PatsyError` from a malformed
  formula, `LinearMixedModel`'s missing `random_intercept` would raise a pandas
  `KeyError`, and — the one silent case — an empty `fixed_effects` list would not
  raise at all, silently degrading to a meaningless intercept-only model. All of
  these are already caught by existing broad exception handlers in both dispatch
  paths (`analysis_core.py`'s clinical dispatch and `statisticaltester.py`'s
  `_run_ancova`/`_run_lmm`/`_run_logistic_regression`, reached via
  `advanced_pipeline.py`), so this was never a crash risk — only a message-quality
  gap (an opaque `PatsyError`/`KeyError` string reaching the user instead of a
  clear `ModelDesignError`). Confirmed via routing trace that today's autopilot UI
  never reaches these test types with empty structural fields, so — like item 9 —
  this is a defensive fix, not a fix for a currently-reachable UI bug. Fixed with
  five `ModelDesignError` pre-flight checks plus one `ValueError`→
  `ModelDesignError` conversion for consistency (spec:
  `docs/superpowers/specs/2026-07-03-clinical-model-preflight-validation-design.md`,
  plan: `docs/superpowers/plans/2026-07-03-clinical-model-preflight-validation.md`).

## Recorded for completeness, not bugs (deliberate design choices)

- **Welch's t-test/ANOVA is used whenever data is normal, regardless of the
  Brown-Forsythe variance-equality result.** `select_comparison_test` ignores its
  `is_homoscedastic` argument by design (in-code "A1 Fix" comment: Welch as the
  unconditional robust default). (`one_way_anova.md`.)
- **Dependent-samples pairing is purely positional (row-index based), with no
  subject-id key.** Matches the documented "matching order" contract, but a wrong row
  order is silently mispaired with no detection — a real data-integrity footgun a user
  could hit, even though it isn't a code defect relative to the contract. Worth
  considering an optional subject-id key for the two-group paired path, mirroring what
  the multi-group RM path already has. (`dependent_samples.md`.)
- **HC3 robust standard errors in linear regression surface only as a qualitative
  decision-tree branch, never a labeled "HC3" field.** Not a correctness bug (a
  first draft of this finding incorrectly said HC3 was never surfaced at all; corrected
  during spec review). (`linear_regression.md` item 1.)
- **Two-Way ANCOVA's slope-homogeneity check only tests single factor-by-covariate
  interactions**, not joint/higher-order interactions. Correct for the one-factor ANCOVA
  case the recipe documents; only a gap for the less-common Two-Way ANCOVA variant.
  (`ancova.md` item 2.)
- **Graph visualization: the live Qt dialog and the in-report Plot Designer use
  different error-metric vocabularies** (`sd/se/ci` vs `sd/sem/ci95/iqr/range`). Cosmetic
  UI inconsistency between two intentionally separate surfaces, not a correctness issue.
  (`graph_visualization.md`.)
- **Correlation: Shapiro-Wilk is computed but never drives the Pearson/Spearman
  decision** (skew/kurtosis gating is used instead, deliberately — Shapiro over-rejects
  at large n). Spearman's p-value uses a t-approximation only for `20 <= n < 100`.
  (`correlation.md` items 1-2.)

## Terminology consistency (already fixed within this audit, noted for the record)

The audit found and fixed two terminology mismatches between what the Help Hub text
said and what the app's UI/report actually calls things:
- "letters or bars" -> "letters or brackets" (the UI/report literally call these
  brackets; letters are a separate, plot-only feature) — fixed in both
  `graph_visualization` and `statistical_tests_html`.
- "Levene" -> "Brown-Forsythe" in `statistical_tests_html` (the variance-homogeneity
  test is Brown-Forsythe internally and everywhere it reaches a user; "Levene" only
  appeared in an internal, non-user-facing docstring).
A full terminology scan across all 12 recipes at the end of the audit found no further
stale occurrences of either term.
