# Audit note: `dependent_samples` recipe

Recipe location: `src/core/help_content.py:563` (`"id": "dependent_samples"`,
`"category": "Concepts"`).

This is a conceptual recipe ("when are samples dependent, and which tests
apply"), not a test-selection page. It was migrated verbatim from an older
inline QMessageBox during an earlier consolidation task, so it was audited with
the same staleness suspicion as a from-scratch recipe. It does not describe a
single dispatch path; instead it spans the two-group dependent path
(`statisticaltester.py`) and the multi-group repeated-measures path
(`advanced_pipeline.py`).

The citation anchor (symbol name or quoted string) is authoritative; the line
number is only a navigation hint.

## Traced ground-truth (two-group dependent path)

Autopilot is the single source of truth (injects the in-memory DataFrame). For
two paired groups, `AnalysisManager.analyze` sets `model_type = "paired"` and
runs the assumption check on within-pair differences before dispatching:

- `model_type = "paired" if dependent else "ttest"` with the comment "Paired
  t-test assumption is normality of within-pair differences, not of pooled OLS
  residuals (B1)" (`src/analysis/analysis_core.py:826`-`831`).
- `check_normality_and_variance` computes the differences and runs Shapiro-Wilk
  on them: `is_paired = model_type == "paired"` (`assumption_checks.py:78`);
  `_paired_diffs` returns `np.asarray(a) - np.asarray(b)` element-wise by
  position (`assumption_checks.py:80`-`87`); the paired branch runs
  `stats.shapiro(diffs)` under the comment "B1: paired t-test -> Shapiro-Wilk on
  within-pair differences" (`assumption_checks.py:166`-`172`). Requires equal
  length and n>=3, else the check is skipped with a note
  (`assumption_checks.py:85`-`86`, `:177`-`181`).
- The resulting `residuals_normal` flag drives the strategy:
  `select_comparison_test(..., is_paired=True, group_count=2)` returns
  `"paired_ttest" if is_normal else "wilcoxon"`
  (`src/statistical_testing/decision_logic.py:53`-`55`); dispatched from
  `_stat_test_two_groups` (`statisticaltester.py:337`-`361`).
- Default significance level is `alpha = 0.05` (`statisticaltester.py:38`
  `DEFAULT_ALPHA = 0.05`; two-group entry defaults `alpha=0.05` at `:235`).

Implementations of the two two-group dependent tests:

- Paired t-test `_paired_ttest` (`statisticaltester.py:454`-`506`): validates
  `validate_paired_data(...)`, effect size Cohen's d on differences
  (`diff = data1_arr - data2_arr`, `cohen_d = mean(diff)/std_diff`, `:464`-`470`).
- Wilcoxon `_wilcoxon_test` (`statisticaltester.py:509`-`559`):
  `stats.wilcoxon(data1_arr, data2_arr, zero_method='pratt', method='exact' if
  len<=25 else 'approx')` (`:520`-`523`); rank-biserial `r` as effect size
  (`:531`-`538`).
- For contrast, the INDEPENDENT-samples counterpart the recipe distinguishes
  itself from is `_mannwhitney_test` (`statisticaltester.py:633`-`666`),
  reached only when `is_paired=False` (`decision_logic.py:58`,
  `"mann_whitney_u"`). The recipe correctly does not list it.

## Traced ground-truth (multi-group dependent path)

- The simple pipeline refuses RM in-line: for `dependent and len(valid_groups) >
  2` the strategy is `"repeated_measures_required"` and it returns an error
  telling the caller to use the advanced path
  (`_stat_test_multi_groups`, `statisticaltester.py:684`-`685`, `:725`-`728`).
- The advanced RM path runs pingouin RM ANOVA on the parametric branch, and
  falls back to Friedman only when the recommendation is non-parametric:
  `if effective_recommendation == "non_parametric": ... if test ==
  "repeated_measures_anova": res = perform_friedman_test(...)`
  (`src/statistical_testing/advanced_pipeline.py:310`-`320`).
- `perform_friedman_test` uses `scipy.stats.friedmanchisquare`
  (`src/analysis/nonparametricanovas.py:189`, `:220`), effect size Kendall's W
  (`:226`), and emits a note "Only 2 time points: consider paired Wilcoxon
  instead of Friedman" for k==2 (`:214`-`215`) plus a near-zero-power warning
  for `n_subjects < MIN_N_HARD` (`:216`-`217`). The parametric-vs-Friedman
  choice is again driven by a normality check inside the advanced pipeline, so
  the recipe's "the app switches automatically" statement holds for the
  multi-group case too.

## Data-structure invariance control check

- Equal group sizes are enforced. Two-group: `validate_paired_data` raises
  `PairedDataError` when `arr_a.size != arr_b.size` (`validators.py:115`-`130`,
  "Paired tests require equal sample sizes"). Multi-group dependent validation:
  `validate_dependent_data` -> `ensure_equal_group_sizes`
  (`statisticaltester.py:1008`, `validators.py:133`-`158`, "Dependent tests
  require equal sample sizes across groups").
- Matched-order requirement is real but **only positional**. Pairing is by row
  index everywhere: `_paired_diffs` subtracts `a[i] - b[i]`
  (`assumption_checks.py:87`), `_paired_ttest`/`_wilcoxon_test` subtract the
  aligned arrays, and `_build_paired_subject_trajectories` pairs `data1[index]`
  with `data2[index]` and returns `[]` if lengths differ
  (`statisticaltester.py:376`-`396`). There is no subject-id key match in the
  two-group path, so a wrong row order silently pairs the wrong observations
  without any error. The rewritten recipe now states this explicitly. Minimum n
  to run at all is `MIN_N_BLOCK = 3` (`validators.py:8`).

## Alpha / adjustment control check

- `alpha = 0.05` throughout (`DEFAULT_ALPHA`, `statisticaltester.py:38`;
  advanced pipeline defaults). The recipe does not state a numeric alpha or a
  post-hoc correction, so nothing to reconcile there; per recipe-economy the
  correction detail is left to the RM/mixed recipes it now points to.
- Recipe-economy: the recipe says "the app checks whether the paired
  differences are normally distributed and switches automatically" and does not
  name Shapiro-Wilk, Wilcoxon's zero_method, Pratt correction, Kendall's W, or
  the exact/approx cutoff. This matches the `correlation` recipe's
  one-sentence "picks Pearson or Spearman automatically" benchmark. The full
  named-procedure detail is recorded here for the human record.

## Claim table

| # | Claim (from title/html) | Verdict | Citation |
|---|-------------------------|---------|----------|
| 1 | Title/summary: measurements paired or repeated on the same subjects; which tests apply | correct | dependent two-group path (`_stat_test_two_groups`, `statisticaltester.py:312`) and RM path (`advanced_pipeline.py:310`-`320`) both exist and are reachable via the autopilot |
| 2 | Same subject measured at different time points is dependent | correct | RM design uses Subject + within factor; two-group paired path uses `dependent=True` (`analysis_core.py:826`-`831`) |
| 3 | Naturally paired measurements (e.g. left/right eye) are dependent | correct (conceptual) | matches positional pairing model in `_paired_ttest`/`_wilcoxon_test`; no code contradicts it |
| 4 | Matched pairs / repeated measurements are dependent | correct (conceptual) | same; `dependent` branch routes to paired/RM tests |
| 5 | Each group must contain the same number of measurements | correct | `validate_paired_data` raises on size mismatch (`validators.py:125`-`129`); `ensure_equal_group_sizes` for multi-group (`validators.py:150`-`156`) |
| 6 | Measurements must be in matching order | correct, with caveat | pairing is purely positional: `_paired_diffs` `a[i]-b[i]` (`assumption_checks.py:87`), `_build_paired_subject_trajectories` pairs by index (`statisticaltester.py:381`-`396`). App does NOT verify order, so a wrong order is silently mispaired. Rewrite now says this. |
| 7 | Example: measurement 1 in A and measurement 1 in B must be the same subject | correct | restates the positional pairing (`assumption_checks.py:87`) |
| 8 | Two groups: paired t-test or Wilcoxon signed-rank test | correct | `select_comparison_test(is_paired=True, group_count=2)` -> `"paired_ttest" if is_normal else "wilcoxon"` (`decision_logic.py:55`); tests at `statisticaltester.py:454`, `:509` |
| 9 | More than two groups: Repeated Measures ANOVA or Friedman test | correct | RM parametric via pingouin; Friedman is the non-parametric fallback (`advanced_pipeline.py:310`-`320`); `perform_friedman_test` uses `scipy.stats.friedmanchisquare` (`nonparametricanovas.py:220`) |
| 10 (new) | The app checks whether the paired differences are normal and picks the parametric vs rank-based test automatically | correct (was MISSING) | Shapiro on within-pair differences (`assumption_checks.py:166`-`172`), feeding `select_comparison_test` (`decision_logic.py:55`); analogous normality gate in the advanced pipeline for RM-vs-Friedman (`advanced_pipeline.py:310`) |
| 11 (new) | Pointer to the Repeated Measures ANOVA and Mixed ANOVA recipes for the multi-group mechanics | correct (recipe-economy) | RM/mixed dispatch is a separate, already-audited path (`advanced_pipeline.py`); avoids duplicating those recipes here, matching `statistical_tests_html`'s "see the related recipes in this hub" convention (`help_content.py:658`, `:677`) |

## Changes applied to the recipe html

- Reworded "Dependent samples arise when" to "Samples are dependent when the
  values in one group are tied to the values in another" (plain `are` copula,
  states the actual criterion).
- Replaced the third "When are samples dependent?" bullet ("Experiments are
  conducted with repeated measurements", which duplicated the first bullet) with
  "Subjects are matched into pairs before the measurement", a distinct case.
- Merged the "Data structure" bullets into one paragraph that keeps the
  same-number and matching-order requirements and now states the positional
  pairing behavior and that a wrong row order is not detected (claim 6 caveat).
- Renamed "Available tests" to "Which test the app runs" and folded the
  parametric/non-parametric choice into each bullet (paired t vs Wilcoxon on
  non-normal differences; RM ANOVA vs Friedman on non-normal data).
- Added the recipe-economy sentence: the app checks normality of the paired
  differences and switches automatically (claim 10).
- Added a pointer to the Repeated Measures ANOVA and Mixed ANOVA recipes
  (claim 11).
- No `id` or `category` change. No emoji or typographic dashes introduced.

## Unclear / possible code bug

- **Silent mispairing on wrong row order (behavior, not a bug to fix here).**
  The two-group dependent path pairs strictly by row index and never checks a
  subject id, so if the user's two columns are not in the same subject order the
  app computes differences on mismatched subjects and reports a valid-looking
  but wrong result. Equal length is the only guard. This is a genuine
  data-integrity foot-gun, but it matches the documented "matching order"
  contract, so it is described in the recipe rather than flagged as a defect.
  Noted here in case a future task wants to add an optional subject-id key for
  the two-group paired path (the multi-group RM path already keys on a Subject
  column).
- **Friedman k==2 note.** `perform_friedman_test` warns and still runs when
  called with exactly two within-levels (`nonparametricanovas.py:214`-`215`),
  but the two-level dependent case is normally routed to `paired_ttest` before
  reaching Friedman (`_ap_build_analysis_context` sets `paired_ttest` for two
  levels, per the repeated_measures_anova audit note), so the warning is a
  defensive guard rather than a path a normal user hits. No action.
