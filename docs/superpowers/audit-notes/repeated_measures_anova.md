# Audit note: `repeated_measures_anova` recipe

Recipe location: `src/core/help_content.py:205` (`"id": "repeated_measures_anova"`).

Ground-truth dispatch path (traced, not guessed). Autopilot is the single source
of truth: `_ap_build_analysis_context` injects the in-memory DataFrame and sets
`inferred_test`, so `AnalysisManager` never re-reads from disk.

Routing (autopilot): a Subject ID plus exactly one factor column, no Factor 2,
routes to `repeated_measures_anova` when the factor has 3+ levels
(`_ap_build_analysis_context`, `src/autopilot/statistical_analyzer_autopilot_pipeline.py:1161`-`1166`,
`context["inferred_test"] = "paired_ttest" if len(levels) == 2 else "repeated_measures_anova"`).
Two levels route to `paired_ttest` instead. Bucket-to-test routing preview also
lives at `_ap_infer_test_for_bucket` (`pipeline:768`-`784`,
`"repeated_measures_anova"`).

Dispatch (analysis core): `AnalysisManager.analyze` builds `local_kwargs["test"]
= "repeated_measures_anova"` (`src/analysis/analysis_core.py:266`-`269`) and
carries the subject column and within factors
(`analysis_core.py:270`-`284`, `resolved_additional_factors =
analysis_context.get("within_factors")`). The advanced path then calls
`perform_advanced_test_pipeline` (`src/statistical_testing/advanced_pipeline.py:26`).
The parametric branch dispatches
`StatisticalTester._run_repeated_measures_anova_logged`
(`advanced_pipeline.py:166`-`174`) ->
`StatisticalTester._run_repeated_measures_anova`
(`src/analysis/statisticaltester.py:1833`). Design validation:
`validate_test_design` requires a within factor and a subject column for
`repeated_measures_anova` (`src/statistical_testing/validators.py:258`-`262`).

Wide-format auto-melt: `_ap_maybe_pivot` runs on every file load and sheet switch
(`pipeline:1002`, `:1043`) and calls `_detect_wide_format` /
`_pivot_wide_to_long` (`src/autopilot/statistical_analyzer_autopilot_ui.py:128`,
`:176`).

The citation anchor (symbol name or quoted string) is authoritative; the line
number is only a navigation hint.

## Claim table

| # | Claim (from title/html) | Verdict | Citation |
|---|-------------------------|---------|----------|
| 1 | Title / summary: same subjects measured at several time points, one group only | correct | routing requires Subject ID + one within factor with 3+ levels and no second factor -> `repeated_measures_anova` (`_ap_build_analysis_context`, `pipeline:1161`-`1166`); test label built as `"Repeated Measures ANOVA ({factor})"` at `_run_repeated_measures_anova:1963` |
| 2 | Use when the same subjects are measured at multiple time points / conditions; if also split into groups, use Mixed ANOVA | correct | a Subject ID plus a second (between) factor routes to `mixed_anova`, not RM: `_ap_build_analysis_context:1182`-`1195` (`context["inferred_test"] = "mixed_anova"`). RM is the single-factor-plus-subject branch (`:1161`-`1166`) |
| 3 | Data layout: one row per measurement, three columns (subject id, timepoint label, value); each subject once per time point | correct | RM runs on long-format `df` with `dv`, `within[0]`, `subject`; pingouin `pg.rm_anova(data=df, dv=dv, within=factor, subject=subject, detailed=True, correction=True)` at `_run_repeated_measures_anova:1937`. Exactly one observation per subject x within-level is required by pingouin; duplicate SubjectID+Timepoint combinations make the design non-estimable (see claim 5) |
| 4 (rewritten) | "One column per time point also works": a wide layout (one subject id column, one numeric column per time point) is recognized and reshaped to long format for you | correct (was WRONG before) | `_detect_wide_format` matches exactly this signature: one subject-like column + 2-8 numeric value columns + no 2-level categorical + subject uniqueness ratio >= 0.8 (`autopilot_ui.py:128`-`173`, `unique_ratio = df[subject_col].nunique() / max(len(df), 1)` `:169`, `if unique_ratio < 0.8: return None` `:170`). `_pivot_wide_to_long` melts value columns into a `Condition` column (`autopilot_ui.py:176`-`188`). Runs on every load/sheet-switch via `_ap_maybe_pivot` (`pipeline:1002`, `:1043`). The recipe's example (SubjectID + Baseline/Week_4/Week_8, 5 unique subjects) satisfies the signature (3 value cols in 2..8, ratio 5/5 = 1.0), so it is auto-melted, not rejected. The prior recipe's "Why this fails: the app cannot map Timepoint to Factor 1" was factually wrong. |
| 5 | Bucket mapping: DV = measurement, Factor 1 = timepoint/condition, Subject ID = subject column, Factor 2 and Covariates empty | correct | Factor 1 holds the within factor, Subject ID drives the repeated-measures structure (`_ap_build_analysis_context:1161`-`1166`); a filled Factor 2 with a subject would route to `mixed_anova` (`:1182`-`1195`); a filled Covariates bucket upgrades non-clinical tests to ANCOVA variants but RM has no covariate upgrade branch, so covariates should stay empty (`:1240`-`1243` only touch `one_way`/`two_way`) |
| 6 | Checklist: each subject appears exactly once per time point; no duplicate SubjectID+Timepoint | correct | pingouin `rm_anova` (`:1937`) requires a single value per subject x level; the RM post-hoc also assumes one row per subject per level and aligns on common subjects (`RMAnovaPostHocAnalyzer.perform_test`, `src/analysis/posthoc_core.py:1086`-`1104`). Duplicate cells break the balanced within-subject matrix. |
| 7 (rewritten) | Checklist: if some measurements are missing, the app switches to a mixed model that uses every subject instead of dropping anyone with a gap | correct (was MISLEADING before) | see "Missing-data / LMM control check" below. The prior ">5% of subjects -> LMM" threshold is real only inside `_run_repeated_measures_anova:1862` and is largely pre-empted by the autopilot context builder, which switches to `lmm` on ANY missing subject x timepoint cell with no percentage threshold (`_ap_build_analysis_context:1216`-`1237`). The rewritten sentence drops the incorrect "5%" figure. |
| 8 (rewritten) | Checklist: the Timepoint column can hold text labels or numbers; both work | correct (was WRONG before) | within-factor levels are read via `df[factor].unique()` with no type constraint (`_run_repeated_measures_anova:1853`, `:1999`-`2002`); pingouin `rm_anova` accepts numeric within levels. The continuous-factor redirect that could reject a numeric column only fires for `len(factor_columns) == 1 and not subject_column` (`_ap_build_analysis_context:1251`); with a Subject ID present it never fires, so a numeric Timepoint still routes to RM. A properly long numeric Timepoint does not trip wide-format detection because the subject column repeats across rows (`unique_ratio < 0.8`, `autopilot_ui.py:170`). The prior "text labels, not numbers like 0, 4, 8" was wrong. |
| 9 | Checklist: Subject ID values repeat across rows | correct | long format has one row per subject per level, so the subject column repeats; this is exactly the low-uniqueness signal used to distinguish long from wide (`autopilot_ui.py:167`-`170`) |
| 10 (new section) | "What the app checks and runs": checks normality and the equal-footing (sphericity) assumption, runs RM ANOVA on normal data, applies a conservative correction when the assumption cannot be confirmed, falls back to a rank-based test when not normal, and adjusts post-hoc p-values for the number of comparisons | correct | assumption route, sphericity/GG route, non-parametric fallback, and post-hoc all cited below |

## Assumption / correction control check (sphericity -> Greenhouse-Geisser)

This is the recipe's high-stakes claim. The CHANGELOG (`CHANGELOG.md:12`) states:
"When sphericity cannot be formally tested (for example, with incomplete tables),
the Greenhouse-Geisser correction is now applied by default. Earlier versions
assumed sphericity was met." Verified against current code, not the changelog
description:

- **Normality gate (parametric vs non-parametric).** RM assumption checks run
  before the test; a non-normal recommendation routes to the non-parametric
  branch (`advanced_pipeline.py:163` gates on `effective_recommendation ==
  "parametric"` vs `:310` `"non_parametric"`).
- **Sphericity is tested inside** `_perform_comprehensive_sphericity_test`
  (`statisticaltester.py:2582`), called from `_run_repeated_measures_anova:1967`.
  The correction-selected p-value is written back to the canonical field:
  `if sphericity_results.get("final_p_value") is not None: results["p_value"] =
  sphericity_results["final_p_value"]` (`:1972`-`1974`).
- **k <= 2 (two levels):** sphericity is always met; no correction, uncorrected p
  used (`_perform_comprehensive_sphericity_test:2622`-`2636`,
  `"correction_used": "None (sphericity assumption met)"`). (RM only routes here
  for 3+ levels anyway; two levels are `paired_ttest`.)
- **Sphericity TESTED and MET** (pingouin `pg.sphericity` returns `spher=True`):
  `sphericity_violated = False` (`:2662`); `_apply_sphericity_corrections`
  returns the uncorrected p (`:2807`-`2816`, `"correction_used": "None
  (sphericity assumption met)"`, `final_p_value = uncorrected`).
- **Sphericity TESTED and VIOLATED** (`spher=False`): `sphericity_violated = True`
  (`:2662`); Greenhouse-Geisser is applied unconditionally
  (`_apply_sphericity_corrections:2855`-`2861`,
  `# Use Greenhouse-Geisser unconditionally (as requested by user / conservative
  default)`, `final_p_value = gg_p_value`, `correction_used =
  "Greenhouse-Geisser (eps = ...)"`). Huynh-Feldt is computed and stored for
  reference (`:2839`-`2850`) but is NOT selected when GG is available; it is used
  only if GG epsilon is missing but HF is present (`:2866`-`2869`).
- **Sphericity CANNOT be formally tested** (pingouin `pg.sphericity` raises, e.g.
  incomplete / singular tables): the fallback
  `_extract_sphericity_from_anova_table` returns `sphericity_assumed = False`
  when no sphericity columns exist in the ANOVA table
  (`:2764`-`2772`, `"sphericity_assumed": False,  # Conservative assumption
  (Apply GG)`, `"interpretation": "Indeterminate (Defaulting to GG
  correction)"`). That makes `sphericity_violated = True`
  (`:2680`, `not mauchly_results.get("sphericity_assumed", True)`), so GG is then
  applied via the same `_apply_sphericity_corrections` path. **This matches the
  CHANGELOG claim exactly against current code: when sphericity cannot be tested,
  GG is applied by default rather than assuming sphericity.**

The recipe body deliberately does not name Mauchly / Greenhouse-Geisser /
Huynh-Feldt (recipe-economy). It says the app "checks whether the differences
between time points are stable enough to compare them on equal footing" and, if
that cannot be confirmed, "applies a conservative correction rather than assuming
it holds." That plain-language framing is accurate for all three cases above.

Caveat recorded (not a recipe error): there is one narrow path where GG is NOT
applied even though sphericity is unresolved. If the OUTER try in
`_perform_comprehensive_sphericity_test` throws before corrections run, the
except block sets `corrected_p_value = uncorrected` and `correction_used = "None
(sphericity test failed)"` (`:2690`-`2701`) and does NOT set `final_p_value`, so
`_run_repeated_measures_anova:1973` leaves the uncorrected p in place. See
"Unclear / possible code bug" item 1.

## Which exact test runs

- **Parametric (normal data):** one-way RM ANOVA via pingouin
  `pg.rm_anova(data=df, dv=dv, within=factor, subject=subject, detailed=True,
  correction=True)` (`_run_repeated_measures_anova:1937`); effect size is
  generalized eta squared (`ng2`, `:1953`/`:1959`,
  `effect_size_type = "partial_eta_squared"`). The canonical `p_value` is the
  sphericity-correction-selected value (`:1972`-`1974`). Statsmodels is only a
  fallback if pingouin is unavailable (`:1923`-`1925`).
- **Non-parametric fallback (residuals non-normal):** Friedman test
  `perform_friedman_test(data=df_original, dv=dv, within_factor=within[0],
  subject_col=subject, alpha=alpha)` (`advanced_pipeline.py:313`-`320`,
  `src/analysis/nonparametricanovas.py:perform_friedman_test`). The recipe's
  "rank-based test for repeated measures" refers to this (name kept out of the
  body per recipe-economy).

## Post-hoc method and adjustment (exact)

Post-hoc fires only when the omnibus is significant (`res["p_value"] < alpha`,
`advanced_pipeline.py:237`).

- **Inline post-hoc** inside `_run_repeated_measures_anova` runs paired t-tests
  with a Holm-Bonferroni family (`perform_dependent_posthoc_tests(..., parametric
  =True)`, `:2005`-`2009`, default label `"Paired t-tests (Holm-Bonferroni)"`).
- **Engine post-hoc** then overrides it for the reported `pairwise_comparisons`.
  `AdvancedPostHocEngine` (`advanced_pipeline.py:238`-`266`) routes RM to
  `PostHocFactory.perform_posthoc_for_anova("rm", ...)`
  (`src/statistical_testing/engines/advanced_posthoc.py:137`-`140`) ->
  `RMAnovaPostHocAnalyzer.perform_test` (`posthoc_core.py:994`).
  The default method when no dialog callback is present is **`tukey`**
  (`advanced_posthoc.py:85`, `default_method = "paired_custom" if test ==
  "two_way_anova" else "tukey"`). For RM, `tukey` runs a paired t-test per level
  pair (`ttest_rel`, `posthoc_core.py:1104`) then converts each t to a Tukey q
  statistic and applies a studentized-range correction across levels
  (`:1143`-`1157`, `correction_method = "Tukey HSD (RM)"`; Bonferroni fallback if
  pingouin is unavailable, `:1160`-`1166`). A post-hoc dialog callback can
  override the method to `dunnett` (control-vs-level) or `emm_mvt` (EMM +
  multivariate-t, level-vs-baseline, `posthoc_core.py:1009`-`1033`); Dunnett does
  control-only comparisons (`advanced_posthoc.py:102`-`103`), consistent with
  `CHANGELOG.md:14`.
- Effect size per pair is Cohen's d for repeated measures on difference scores
  with `ddof=1` (`posthoc_core.py:1108`-`1112`), consistent with `CHANGELOG.md:25`.

The recipe body says only "compares the time points and adjusts the p-values for
the number of comparisons", which holds for every branch above. Named procedures
recorded here per recipe-economy, kept out of the shipped text.

## Alpha / adjustment control check

Default `alpha` is 0.05 end to end:
`perform_advanced_test_pipeline(..., alpha=0.05)` (`advanced_pipeline.py:35`);
`_run_repeated_measures_anova(df, dv, subject, within, alpha=0.05)`
(`statisticaltester.py:1833`); significance gate `res["p_value"] < alpha`
(`advanced_pipeline.py:237`); assumption thresholds `p > 0.05`. Sphericity
significance is judged by pingouin's `spher` boolean, not a hard-coded alpha in
this module. The recipe prints no numeric alpha, so there is nothing to
contradict.

## Data-structure / missing-data / LMM control check

Two independent LMM-redirect mechanisms exist. This is why the old ">5% of
subjects" checklist line was misleading:

1. **Autopilot context builder (the path users actually hit).**
   `_ap_build_analysis_context:1216`-`1237`: with a Subject ID and a within
   factor, it builds the subject x timepoint count matrix and switches
   `inferred_test` to `"lmm"` if `(counts == 0).any().any()` (a structurally
   absent cell) OR if any cell has all-NaN DV values. There is **no percentage
   threshold**: a single missing subject x timepoint cell flips the whole
   analysis to LMM before RM ANOVA runs. `lmm` then dispatches to
   `_run_lmm_logged` (`analysis_core.py:206`-`209`), not RM.
2. **RM function threshold.** `_run_repeated_measures_anova:1855`-`1899` computes
   `n_excluded / n_total` and redirects to a `LinearMixedModel` only if that
   ratio `> 0.05` (`:1862`); at or below 5% it does listwise deletion with a
   warning (`:1901`-`1918`). This branch is reached only when the context builder
   did NOT already flip to LMM, which given mechanism 1 is uncommon in the
   autopilot path.

So the user-facing behavior is: any missing repeated-measures cell tends to route
to a mixed model that keeps all subjects; strict complete cases run the ordinary
RM ANOVA. The rewritten checklist line states exactly that without citing a
threshold that does not govern the real path.

Long/wide invariance: the recipe's long layout matches the RM parser
(`_run_repeated_measures_anova:1937`), and the wide "one column per time point"
layout is auto-melted (`_detect_wide_format` / `_pivot_wide_to_long`,
`autopilot_ui.py:128`-`188`) rather than rejected. The old "common mistake"
framing was inverted relative to the code and has been corrected to "one column
per time point also works."

## Unclear / possible code bug

1. **Sphericity outer-exception path leaves the uncorrected p (possible bug,
   contradicts the GG-by-default intent, does NOT contradict the recipe body).**
   `_perform_comprehensive_sphericity_test` has two "cannot test sphericity"
   outcomes that behave differently:
   - If `pg.sphericity` raises but the ANOVA table has no sphericity columns, the
     INNER fallback `_extract_sphericity_from_anova_table` returns
     `sphericity_assumed = False` (`:2768`), so GG is applied (correct, matches
     CHANGELOG).
   - If any statement in the OUTER `try` throws before
     `_apply_sphericity_corrections` runs (e.g. `pg.sphericity` raises AND the
     `_extract_...` fallback itself throws, or `get_pingouin_module()` fails),
     the OUTER except at `:2690`-`2701` sets `correction_used = "None (sphericity
     test failed)"` and `corrected_p_value = uncorrected`, but never sets
     `final_p_value`. Back in `_run_repeated_measures_anova:1973`-`1974` the
     write-back is guarded by `if sphericity_results.get("final_p_value") is not
     None`, so the uncorrected p from `pg.rm_anova` remains the canonical
     `p_value`. In that narrow failure mode sphericity "cannot be formally
     tested" yet GG is NOT applied, contrary to the stated conservative default.
     This is inconsistent with the inner fallback and worth a human decision (e.g.
     have the outer except also default `final_p_value` to the GG-corrected
     value, or surface an explicit "correction indeterminate" state). Flagged,
     not fixed; the recipe text ("applies a conservative correction rather than
     assuming it holds") describes the intended and dominant behavior and is not
     rewritten to match this edge case.

2. **Inline `posthoc_test` label may be superseded but not always coherent (minor,
   observational).** `_run_repeated_measures_anova` sets an inline
   `results["posthoc_test"]` from `perform_dependent_posthoc_tests`
   (default `"Paired t-tests (Holm-Bonferroni)"`, `:2009`) while the engine's RM
   run uses method `tukey` and returns `"RM ANOVA Post-hoc Tests"` /
   `"Tukey HSD (RM)"`. The override guard in
   `advanced_pipeline.py:258`-`266` only replaces `posthoc_test` under specific
   conditions (`not current_posthoc`, or specific known labels, or a
   Pingouin/Tukey substring match). The inline label
   `"Paired t-tests (Holm-Bonferroni)"` does not match those conditions, so for a
   significant default RM run the reported `pairwise_comparisons` are the engine's
   Tukey-corrected values while the `posthoc_test` name field can retain the
   inline paired-t label. This mirrors the two-way label-vs-correction mismatch
   documented in `two_way_anova.md` item 2. It does not affect the recipe text
   (which names no procedure) but is recorded for the human. Flagged, not fixed.
