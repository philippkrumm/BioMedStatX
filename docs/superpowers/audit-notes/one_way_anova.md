# Audit note: `one_way_anova` recipe

Recipe location: `src/core/help_content.py:78` (`"id": "one_way_anova"`).
Ground-truth modules: `src/analysis/analysis_core.py` (`AnalysisManager.analyze` test dispatch),
`src/statistical_testing/assumption_checks.py`
(`AssumptionCheckEngine.check_normality_and_variance`, the one-way/t-test decision engine),
`src/statistical_testing/decision_logic.py` (`select_comparison_test`, `strategy_to_recommendation`),
`src/analysis/statisticaltester.py`
(`_stat_test_two_groups`, `_stat_test_multi_groups`, `_welch_anova_test`),
`src/statistical_testing/engines/comparison.py` (`ComparisonEngine._run_kruskal_wallis`),
`src/statistical_testing/posthoc_fallback.py`
(`PosthocFallbackEngine.perform_refactored_posthoc_testing`, default post-hoc selection),
`src/analysis/posthoc_core.py` (`TukeyHSD.perform_test`),
`src/statistical_testing/validators.py` (`MIN_N_BLOCK`, `validate_minimum_n`),
`src/autopilot/statistical_analyzer_autopilot_pipeline.py` (`_ap_build_analysis_context` test inference).

The citation anchor (symbol name or quoted string) is authoritative; the line number is only a
navigation hint.

## Claim table

| # | Claim (from title/html) | Verdict | Citation |
|---|-------------------------|---------|----------|
| 1 | Title "Comparing groups (t-Test / One-Way ANOVA)" | correct | `src/core/help_content.py:80` (`"title"`) — descriptive heading; matches the two-group vs multi-group split the code implements |
| 2 | Use when there are two or more independent groups, each subject in exactly one group | correct | `_ap_build_analysis_context:1168` (`statistical_analyzer_autopilot_pipeline.py`, `"independent_ttest" if len(levels) == 2 else "one_way_anova"`) — a single categorical factor with 2+ levels routes here; independence (no repeated subject) is the between-subjects gate at `:1162` (subject-span check raises for repeated subjects) |
| 3 | t-Test automatically for exactly 2 groups, ANOVA for 3 or more, chosen automatically | correct | `_ap_build_analysis_context:1168` (2 levels -> `independent_ttest`, else `one_way_anova`); at compute time `_stat_test_two_groups:312` vs `_stat_test_multi_groups:669` (`statisticaltester.py`) branch on `len(valid_groups) == 2` at `:283`. The test family is chosen from the data, not by the user. See claim 12 for the exact members of each family. |
| 4 | Data layout: one row per measurement; a group-label column plus a measured-value column (long format) | correct | `_ap_build_analysis_context:1157` (`levels = _sorted_unique(analysis_df[factor].tolist())`) and `:1096` (`analysis_df[factor_columns[0]]`) read group membership from a column's row values, i.e. long format. Same convention as `getting_started`. |
| 5 | Common mistake: groups as column headers fails; group names belong as values in one column | correct | `_ap_build_analysis_context:1096` reads the group from a single column's values, not from headers. `_ap_maybe_pivot` (`:1961`) only auto-melts the paired/repeated wide signature (one subject column, no 2-level categorical, uniqueness >= 0.8 — `statistical_analyzer_autopilot_ui.py:128`, `_detect_wide_format`), which does not match group-comparison wide data. So groups-as-headers is genuinely rejected for this design. |
| 6 | Bucket mapping: Dependent Variable = measurement column | correct | `_ap_init_ui:250` (`MappingBucketWidget("Dependent Variable", ..., accepted_kinds={"numeric"})`) |
| 7 | Bucket mapping: Factor 1 = the group-label column | correct | `_ap_init_ui:263` (`"Factor 1"`); categorical Factor 1 routes to t-test/ANOVA at `_ap_build_analysis_context:1168` |
| 8 | Factor 2, Subject ID, Covariates left empty for this design | correct | leaving them empty keeps the inference on the single-factor branch (`_ap_build_analysis_context:1168`); Factor 2 filled -> two-way/mixed (`:1195`/`:1198`), Subject ID filled -> paired/RM (`:1166`), Covariates filled -> ANCOVA (`:1240`) |
| 9 | Checklist: group names must be spelled identically ("Control" != "control") | correct | groups are the distinct string values of the factor column via `_sorted_unique(analysis_df[factor].tolist())` (`_ap_build_analysis_context:1157`); distinct spellings become distinct levels |
| 10 | Checklist: measurement column must be numeric only | correct | DV bucket enforces `accepted_kinds={"numeric"}` (`_ap_init_ui:250`); values are cast with `float(...)` in the tester (e.g. `_welch_anova_test:864`) |
| 11 | Checklist: no subject appears more than once; if measured twice use Repeated Measures ANOVA | correct | repeated-subject data is caught by the subject-span gate at `_ap_build_analysis_context:1162` and routed to `repeated_measures_anova`/`paired_ttest` at `:1166`, not to this between-subjects path |
| 12 (new) | For normal data the app runs Welch's t-test (2 groups) or Welch's ANOVA (3+); these stay valid under unequal variance and are the default | correct | `check_normality_and_variance:582` calls `select_comparison_test(is_normal=post_norm, ...)`; `decision_logic.py:57` returns `welch_ttest` for a normal 2-group case and `:64` returns `welch_anova` for a normal 3+ case, both independent of homoscedasticity ("A1 Fix"). `strategy_to_recommendation` (`decision_logic.py:82`) maps both to `"welch"`. At compute time `_stat_test_two_groups:337` re-derives the strategy via `select_comparison_test(...group_count=2)` (-> `welch_ttest`), and `_stat_test_multi_groups:693` re-derives it (-> `welch_anova`) then runs `_welch_anova_test:830` (`pg.welch_anova`, `results["test"] = "Welch's ANOVA"` at `:899`). |
| 13 (new) | Assumptions checked before the test: Shapiro-Wilk on model residuals (normality) and Brown-Forsythe (equal spread) | correct | `check_normality_and_variance` fits `Value ~ C(Group)` and runs `stats.shapiro(resid_raw)` (`assumption_checks.py:215`, normal if p>0.05 at `:245`); variance via `stats.levene(*data, center='median')` labeled `"Brown-Forsythe"` (`:274`, `:285`, equal if p>0.05 at `:286`). Median-centered Levene is Brown-Forsythe. `analysis_core.py:1042` logs it as "Brown-Forsythe test". Note: the recipe path does NOT use `_perform_levene_test` (`statisticaltester.py:2889`, bound from `MixedAnovaAssumptionEngine`); that is the mixed/RM path only. |
| 14 (new) | If residuals are not normal the app offers a transformation first, then falls back to rank-based tests: Mann-Whitney U (2 groups) or Kruskal-Wallis (3+) | correct | non-normal residuals trigger `need_transform` (`assumption_checks.py:317`) and a transformation dialog (`:327`, `select_transformation_dialog`, log10/boxcox/arcsin-sqrt). If still not normal, `select_comparison_test(is_normal=False, ...)` returns `mann_whitney_u` (`decision_logic.py:58`) or `kruskal_wallis` (`:65`), family `"non_parametric"` (`:80`). Kruskal-Wallis runs via `ComparisonEngine._run_kruskal_wallis` (`comparison.py:207`, `stats.kruskal(...)`, `results["test"] = "Kruskal-Wallis test"` at `:208`). |
| 15 (new) | With 3+ groups a significant ANOVA triggers automatic pairwise post-hoc; default Tukey HSD (standard ANOVA) or Games-Howell (Welch path); Dunnett vs control selectable; p-values corrected for the number of comparisons | correct | post-hoc fires only when the main test is significant and groups >= 3 (`_stat_test_multi_groups:744`, `should_run_posthoc`) via `PostHocEngine().execute` (`:765`) -> `perform_refactored_posthoc_testing`. Default for the parametric/welch family: `default_method = "games_howell" if test_recommendation == "welch" else "tukey"` (`posthoc_fallback.py:545`); a dialog can override to Dunnett/others (`:557`), falling back to the default if cancelled (`:563`). Tukey computes via `pairwise_tukeyhsd(endog=all_data, groups=group_labels, alpha=alpha)` with `correction_method="Tukey HSD"` (`posthoc_core.py:1371`, `:1396`) — the Tukey HSD studentized-range adjustment across all pairs. |
| 16 (new) | Checklist: each group needs at least three measurements or the test is blocked | correct | `MIN_N_BLOCK = 3` (`validators.py:8`); enforced per group by `validate_minimum_n(..., min_n=MIN_N_BLOCK)` at `_stat_test_two_groups:319`/`:320` and `_stat_test_multi_groups:674`. Below three, the test returns an error result rather than running. |

## Alpha / adjustment control check

- Default `alpha` is 0.05. The results dict is seeded with `"alpha": alpha` (`statisticaltester.py:252`) and `alpha` defaults to 0.05 throughout (e.g. `PostHocEngine.execute:14` `float(payload.get("alpha", 0.05))`; assumption thresholds `p > 0.05` at `assumption_checks.py:245`/`:286`). The recipe does not print a numeric alpha, so there is nothing to contradict; the added post-hoc claim only says p-values are "corrected for the number of comparisons," which matches Tukey HSD (family-wise studentized-range correction, `posthoc_core.py:1371`) and Games-Howell.
- Post-hoc adjustment for the default parametric one-way path is **Tukey HSD** (studentized range, all pairs), not Bonferroni or Holm. For the Welch path the default is **Games-Howell**. Both are family-wise-correct across the pairwise set. The recipe now names Tukey HSD and Games-Howell explicitly and does not overstate a specific numeric threshold.

## Data-structure control check

The recipe's long-format layout (one row per measurement, group label in its own column, measured value
in another) matches what the analyzer reads: `_ap_build_analysis_context:1157`/`:1096` take group
membership from a column's row values. The wide "groups as headers" example is correctly flagged as a
failure: `_ap_maybe_pivot`/`_detect_wide_format` (`statistical_analyzer_autopilot_ui.py:128`) only
auto-melt the paired/repeated signature (a single subject-like column, no 2-level categorical column,
subject uniqueness >= 0.8), which a between-subjects group-comparison table does not satisfy. No change to
the data-layout guidance was required.

## Assumption / correction control check

The recipe previously described only the family-level split (t-Test vs ANOVA, automatic) and omitted the
assumption-driven routing. That omission was misleading because the app's real default for normal data is
the Welch variant (Welch's t-test / Welch's ANOVA), and non-normal data is routed to a transformation and
then to Mann-Whitney U / Kruskal-Wallis. The rewrite adds these as new `<h3>` sections, each backed by the
citations above. The assumption tests are named correctly for this path: Shapiro-Wilk on residuals and
Brown-Forsythe (median-centered Levene), not the mixed/RM `_perform_levene_test`.

## Unclear / possible code bug

None that contradict the recipe. One observation worth recording for the human, not acted on here:
`select_comparison_test` (`decision_logic.py:57`, `:64`) ignores the `is_homoscedastic` argument entirely —
it returns the Welch variant for any normal data regardless of the Brown-Forsythe result (the in-code "A1
Fix" comment says this is intentional: Welch is used as the unconditional robust default). This means the
Brown-Forsythe equal-variance verdict does not change the selected test for normal data; it is computed and
logged but not acted on in the selection. This is deliberate per the comment and the `test_info["note"]`
strings at `assumption_checks.py:591`/`:596`, so it is not a bug and the recipe does not claim otherwise.
The recipe describes Brown-Forsythe as an assumption the app "checks" (true) without claiming it switches
the test, which is the accurate framing.
