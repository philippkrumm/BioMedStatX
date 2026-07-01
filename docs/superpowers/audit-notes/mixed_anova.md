# Audit note: `mixed_anova` recipe

Recipe location: `src/core/help_content.py:269` (`"id": "mixed_anova"`).

Ground-truth dispatch path (traced, not guessed). Autopilot is the single source
of truth: `_ap_build_analysis_context` injects the in-memory DataFrame and sets
`inferred_test`, so `AnalysisManager` never re-reads from disk.

Routing (autopilot): a Subject ID plus exactly two factor columns routes to
`mixed_anova`, provided the two factors resolve to exactly one between-subject
and one within-subject role
(`_ap_build_analysis_context`,
`src/autopilot/statistical_analyzer_autopilot_pipeline.py:1182`-`1195`,
`context["inferred_test"] = "mixed_anova"`). Role is decided per factor by
`role_by_factor[factor] = "between" if ... per_subject.max() <= 1 else "within"`
(`:1185`-`1186`): a factor whose value never varies within a subject is the
between (group) factor; a factor that varies within a subject is the within
(repeated) factor. If the two factors do not split cleanly into one between and
one within, the builder raises
`"With Subject ID plus two factors, auto-pilot requires exactly one
between-subject factor and one within-subject factor."` (`:1189`-`1192`).

Dispatch (analysis core): `AnalysisManager.analyze` builds `local_kwargs["test"]
= "mixed_anova"` and carries the subject, between, and within factors, then the
advanced path calls `perform_advanced_test_pipeline`
(`src/statistical_testing/advanced_pipeline.py:26`). The parametric branch
dispatches `StatisticalTester._run_mixed_anova_logged`
(`advanced_pipeline.py:164`-`165`) ->
`StatisticalTester._run_mixed_anova` (`src/analysis/statisticaltester.py:1470`).

Design validation: `validate_test_design` requires both a between factor and a
within factor for `mixed_anova` but does NOT explicitly check the subject column
(`src/statistical_testing/validators.py:255`-`257`,
`"Mixed ANOVA requires between and within factor."`). The subject requirement is
enforced downstream by `pg.mixed_anova(..., subject=subject)`
(`statisticaltester.py:1510`) and, in practice, by the autopilot only ever
routing to `mixed_anova` when a Subject ID is present (`pipeline:1182`).

The citation anchor (symbol name or quoted string) is authoritative; the line
number is only a navigation hint.

## Claim table

| # | Claim (from title/html) | Verdict | Citation |
|---|-------------------------|---------|----------|
| 1 | Title / summary: two or more groups AND the same subjects within each group measured multiple times (between + within design) | correct | routing requires Subject ID + exactly one between factor + one within factor -> `mixed_anova` (`_ap_build_analysis_context`, `pipeline:1182`-`1195`); the test runs `pg.mixed_anova(data=df, dv=dv, within=rm_factor, between=between_factor, subject=subject)` (`_run_mixed_anova:1510`) |
| 2 (rewritten) | The interaction (does the change over time differ across groups) is the main result | correct (added) | top-level `p_value`/`statistic`/`effect_size` are set from the Interaction row of the pingouin table (`_run_mixed_anova:1548`-`1555`, `results.update({"p_value": interaction["p_value"], ...})`); the significance gate that fires post-hoc keys off this top-level p (`advanced_pipeline.py:237`) |
| 3 (new section) | "What the app checks and runs": checks normality, whether groups have similar spread, and the within-factor equal-footing (sphericity) assumption; runs mixed ANOVA separating group / time / interaction; applies a conservative correction when the within assumption cannot be confirmed; falls back to a rank-based test when not normal; adjusts post-hoc p-values for the number of comparisons | correct | normality gate, between-variance test, within-sphericity/GG, non-parametric fallback, and post-hoc all cited below |
| 4 | Data layout: one row per measurement, four columns (subject id, group, timepoint, value), long format | correct | mixed ANOVA runs on long-format `df` with `dv`, `between[0]`, `within[0]`, `subject` (`_run_mixed_anova:1492`-`1510`); pingouin `mixed_anova` requires one row per subject x within-level |
| 5 | A subject's Group value stays the same across all their rows | correct | role detection: the between (group) factor is the one whose value does not vary within a subject, `per_subject.max() <= 1` -> between (`pipeline:1185`-`1186`). If Group varied within a subject it would be classified within, giving two within factors and raising the "exactly one between-subject factor and one within-subject factor" error (`:1189`-`1192`) |
| 6 (rewritten) | Common mistake: time points as separate columns fails; the app does not auto-reshape a wide file that has a group column | correct (was under-explained before) | `_detect_wide_format` returns `None` when any categorical (non-subject, non-numeric) column has exactly 2 unique values, i.e. a group column: `for col in categorical_cols: if df[col].nunique() == 2: return None` (`autopilot_ui.py:161`-`165`). The recipe's wide example has a 2-level Group column ("Control"/"Treatment"), so it is not melted. Even the melt path drops non-value columns: `_pivot_wide_to_long` keeps only `[subject_col] + value_cols` (`autopilot_ui.py:176`-`188`), so a group column would be lost anyway. Auto-melt runs on every load/sheet-switch via `_ap_maybe_pivot` (`pipeline:1002`, `:1043`) but no-ops here. |
| 7 | Bucket mapping: DV = measurement, Factor 1 = within (timepoint), Factor 2 = between (group), Subject ID = subject column | correct | `_run_mixed_anova(df, dv, subject, between, within, ...)` maps `between[0]` to the group factor and `within[0]` to the repeated factor (`:1492`-`1493`, `:1510`); routing fills `context["between_factors"]`/`context["within_factors"]` from the role split (`pipeline:1193`-`1194`) |
| 8 | Bucket mapping: swapping Factor 1 and Factor 2 reverses the labels | correct (harmless caveat) | role is auto-detected from within-subject variance (`pipeline:1185`-`1186`), so which UI bucket a factor sits in does not change the statistical role as long as one factor is constant per subject and the other varies. The visible group-vs-time labels in the output do follow the bucket assignment (`_run_mixed_anova` uses `between[0]`/`within[0]` for factor typing and labels, `:1521`, `:1587`-`1588`). The recipe's caution is not wrong; it is a labeling nicety rather than a hard correctness requirement. Recorded, not changed. |
| 9 | Covariates: leave empty | correct-as-guidance (simplification) | the autopilot has no ANCOVA-style upgrade for mixed designs; covariates in a mixed/LMM design flow through as additional fixed effects rather than triggering a different test (`pipeline:1239`-`1246`, the ANCOVA upgrade only touches `one_way`/`two_way`, and the comment `"Covariates also flow into LMM/mixed_anova as additional fixed effects"`). For the triage purpose of this recipe, "leave empty" is safe guidance; it is not a hard parser requirement. Left as-is. |
| 10 (rewritten) | Checklist: each subject appears exactly once per time point (no duplicate SubjectID + Timepoint within a group) | correct | `pg.mixed_anova` requires a single value per subject x within-level (`:1510`); duplicate cells break the balanced within-subject matrix |
| 11 (rewritten) | Checklist: a subject's Group value is identical across all their rows, and this is how the app decides which factor is the group vs the repeated one; a subject under more than one group breaks the design | correct | same role-detection + error path as claim 5 (`pipeline:1185`-`1192`) |
| 12 (rewritten) | Checklist: if any subject is missing even one time point, the app switches to a mixed model that uses every subject instead of dropping anyone (was ">5% -> LMM") | correct (was WRONG before) | see "Missing-data / LMM control check" below. The LMM switch fires on ANY structurally absent or all-NaN subject x timepoint cell with no percentage threshold (`_ap_build_analysis_context:1216`-`1237`). The old ">5% of subjects -> Linear Mixed Model" figure does not govern the autopilot path. |
| 13 | Checklist: Subject ID, Factor 1 (within), and Factor 2 (between) all filled; this distinguishes Mixed ANOVA from Two-Way ANOVA | correct | with two factors and NO subject the builder routes to `two_way_anova` (`pipeline:1196`-`1199`); with two factors AND a subject it routes to `mixed_anova` (`:1182`-`1195`). The subject column is the discriminator. |

## Assumption / correction control check

The recipe's "What the app checks and runs" section makes four assumption/behavior
claims. Verified against current code:

- **Normality gate (parametric vs non-parametric).** The pipeline computes a
  `recommendation` upstream and gates the mixed branch on it:
  `if effective_recommendation == "parametric": ... test == "mixed_anova"`
  (`advanced_pipeline.py:163`-`165`) vs the non-parametric branch at `:310`
  (`if effective_recommendation == "non_parametric"`). A non-normal
  recommendation routes away from the parametric mixed ANOVA.
- **Between-group spread (homogeneity of variance).**
  `_test_mixed_anova_between_assumptions` runs Levene's test (median-centered),
  Brown-Forsythe, and Bartlett on the between-group data
  (`src/statistical_testing/mixed_assumptions.py:32`,
  `_perform_levene_test:126`-`142`, `levene(*group_data, center='median')`).
  Results are merged into the mixed-ANOVA result (`_run_mixed_anova:1560`-`1563`).
- **Within-factor equal footing (sphericity) + conservative correction.**
  `_test_mixed_anova_within_sphericity` tests sphericity for the within factor
  via pingouin (`pg.sphericity(df, dv=dv, subject=subject, within=within_factor)`,
  `mixed_assumptions.py:387`, `:443`); for `k <= 2` levels sphericity is always
  met and no correction is applied (`:421`-`436`). When violated,
  `_apply_mixed_anova_sphericity_corrections` computes Greenhouse-Geisser and
  Huynh-Feldt and recommends/uses Greenhouse-Geisser as the conservative default
  (`mixed_assumptions.py:570`, `_apply_specific_corrections:643`-`709`,
  `"recommended_correction" = "greenhouse_geisser"`,
  `"correction_used" = "Greenhouse-Geisser (...)"`). Called from
  `_run_mixed_anova` via the interaction-assumption path
  (`_test_mixed_anova_interaction_assumptions`, `statisticaltester.py:1566`-`1569`).
  The recipe body deliberately does not name Mauchly / Greenhouse-Geisser /
  Huynh-Feldt (recipe-economy); it says the app "checks whether the differences
  between time points are stable enough to compare on equal footing" and, if not,
  "applies a conservative correction rather than assuming it holds." Accurate.
- **Non-parametric fallback.** When the recommendation is non-parametric, mixed
  ANOVA falls back to the Brunner-Langer ATS (nonparametric ANOVA-type
  statistic), `perform_brunner_langer_ats(data=..., between_factor=between[0],
  within_factor=within[0], subject_col=subject, alpha=alpha)`
  (`advanced_pipeline.py:329`-`337`). The recipe's "rank-based test for this
  design" refers to this (name kept out of the body per recipe-economy).

## Which exact test runs

- **Parametric (normal data):** two-factor mixed ANOVA via pingouin
  `pg.mixed_anova(data=df, dv=dv, within=rm_factor, between=between_factor,
  subject=subject)` (`_run_mixed_anova:1510`). Both main effects and the
  interaction are reported with partial eta squared
  (`np2`, `effect_size_type = "partial_eta_squared"`, `:1526`-`1527`,
  `:1543`-`1544`). Top-level canonical fields come from the Interaction row
  (`:1548`-`1555`). Statsmodels is only a fallback if pingouin is unavailable
  (`:1499`-`1501`).
- **Non-parametric fallback (residuals non-normal):** Brunner-Langer ATS
  (`advanced_pipeline.py:330`-`337`, `model_class == "Brunner-Langer ATS"`).

## Post-hoc method and adjustment (exact)

Post-hoc fires only when the omnibus (top-level = interaction) p is significant
(`res["p_value"] < alpha`, `advanced_pipeline.py:237`).

- **Engine post-hoc.** `AdvancedPostHocEngine` builds the comparison universe from
  the between x within cell combinations for mixed designs
  (`src/statistical_testing/engines/advanced_posthoc.py:64`-`69`,
  `group_names = ["{b_factor}={b_val}, {w_factor}={w_val}", ...]`) and routes to
  `PostHocFactory.perform_posthoc_for_anova("mixed", ...)`
  (`advanced_posthoc.py:133`-`136`) ->
  `MixedAnovaPostHocAnalyzer` (`src/analysis/posthoc_core.py:328`,
  factory dispatch `:1735`-`1736`, `:1781`).
- **Default method** when no dialog callback is present is **`tukey`**
  (`advanced_posthoc.py:85`, `default_method = "paired_custom" if test ==
  "two_way_anova" else "tukey"`; mixed_anova is not two_way, so it defaults to
  `tukey`). For mixed, `tukey` produces `"Tukey HSD (Mixed)"`: it classifies each
  pair as within-subject, between-subject, or mixed, converts each t to a
  studentized-range q and applies the corresponding Tukey correction
  (`posthoc_core.py:496`-`514`); Bonferroni is the fallback when pingouin is
  unavailable (`:516`-`522`).
- **Callback overrides.** A post-hoc dialog callback can select `bonferroni`
  (`:524`-`526`), `dunnett` (control-vs-level, corrected with Holm-Bonferroni and
  labelled `"Dunnett-type (Holm-adjusted, mixed design)"` because mixed contrasts
  are not exchangeable, `:528`-`559`), or `emm_mvt` (EMM + multivariate-t, via
  `src/analysis/emm_posthoc.py`). Without a callback the default `tukey` path
  runs.
- **Inline pairwise (interaction significant).** Independently,
  `_run_mixed_anova` runs `pg.pairwise_tests(..., padjust="holm")` when the
  interaction is significant and sets `posthoc_test = "Pairwise t-tests for
  interaction (Holm-Bonferroni)"` (`statisticaltester.py:1576`-`1596`). See
  "Unclear / possible code bug" item 2 for the label-vs-correction mismatch this
  can produce.
- Post-hoc effect size per pair is Cohen's d for the mixed design
  (`effect_size_type = "cohen_d_mixed"`, `posthoc_core.py:579`).

The recipe body says only that the app "compares the group-and-time combinations
and adjusts the p-values for the number of comparisons," which holds for every
branch above. Named procedures recorded here per recipe-economy, kept out of the
shipped text.

## Alpha / adjustment control check

Default `alpha` is 0.05 end to end:
`perform_advanced_test_pipeline(..., alpha=0.05)` (`advanced_pipeline.py:35`);
`_run_mixed_anova(df, dv, subject, between, within, alpha=0.05)`
(`statisticaltester.py:1470`); significance gate `res["p_value"] < alpha`
(`advanced_pipeline.py:237`); assumption thresholds `p > 0.05`. Sphericity
significance is judged by pingouin's `spher` boolean, not a hard-coded alpha in
this module. The recipe prints no numeric alpha, so there is nothing to
contradict.

## Data-structure / missing-data / LMM control check

The old ">5% of subjects -> Linear Mixed Model" checklist line was wrong for the
autopilot path, the same way it was wrong for `repeated_measures_anova` (Task 4).
The LMM redirect that users actually hit has no percentage threshold and applies
to any Subject-ID-plus-within-factor design, which includes mixed ANOVA:

- **Autopilot context builder (the path users hit).**
  `_ap_build_analysis_context:1216`-`1237`. The redirect is guarded by
  `elif subject_column and context["within_factors"]:` (`:1217`), which is true
  for mixed ANOVA (routing fills `within_factors` at `:1194`). It builds the
  subject x within-factor count matrix and switches `inferred_test` to `"lmm"`
  if `(counts == 0).any().any()` (a structurally absent cell) OR if any cell has
  all-NaN DV values (`has_structural_missing or has_nan_missing`, `:1234`-`1235`).
  There is **no percentage threshold**: a single missing subject x timepoint cell
  flips the analysis to LMM before mixed ANOVA runs. `lmm` then dispatches to the
  LMM path, not `mixed_anova`.
- The `> 0.05` (5%) exclusion-ratio threshold documented for
  `repeated_measures_anova` lives inside `_run_repeated_measures_anova`, not in
  the mixed-ANOVA function; `_run_mixed_anova` has no such threshold. So there is
  no code location where ">5% of subjects -> LMM" governs mixed ANOVA at all.

So the user-facing behavior is: any missing repeated-measures cell tends to route
to a mixed model that keeps all subjects; strict complete cases run the ordinary
mixed ANOVA. The rewritten checklist line states exactly that without citing a
threshold that does not govern the real path.

Long/wide invariance: the recipe's long layout matches the mixed-ANOVA parser
(`_run_mixed_anova:1510`), and a wide layout that carries a group column is NOT
auto-melted (`_detect_wide_format` rejects a 2-level categorical group column,
`autopilot_ui.py:161`-`165`) and would lose the group column even if it were
(`_pivot_wide_to_long` keeps only subject + value columns, `:176`-`188`). The
recipe's "common mistake" framing is therefore correct for mixed ANOVA (unlike
the RM recipe, where a group-less wide table IS melted). The wording was sharpened
to explain why (no group column allowed for the auto-reshape) rather than the
vague prior "Timepoint must be a single column."

## Unclear / possible code bug

1. **`validate_test_design` does not check the subject column for `mixed_anova`
   (minor, defensive gap, does not contradict the recipe).**
   `validate_test_design` requires between + within for `mixed_anova` but omits
   the subject check that the sibling `repeated_measures_anova` branch enforces
   (`validators.py:255`-`262`). In the autopilot path this is harmless because
   `mixed_anova` is only ever routed with a Subject ID present
   (`pipeline:1182`), and `pg.mixed_anova(subject=...)` would fail later without
   one. But a non-autopilot caller could dispatch `mixed_anova` without a subject
   and get a pingouin error instead of a clean `ModelDesignError`. Flagged for a
   human decision (add a subject check to the mixed branch), not fixed. Does not
   affect the recipe text.

2. **Inline `posthoc_test` label may be superseded but not always coherent
   (minor, observational).** `_run_mixed_anova` sets an inline
   `results["posthoc_test"] = "Pairwise t-tests for interaction
   (Holm-Bonferroni)"` when the interaction is significant
   (`statisticaltester.py:1584`), while the engine's mixed run uses method
   `tukey` and returns `"Tukey HSD (Mixed)"` pairwise comparisons
   (`posthoc_core.py:496`-`498`). The override guard in
   `advanced_pipeline.py:258`-`266` only replaces `posthoc_test` under specific
   conditions (`not current_posthoc`; equals `"Two-Way ANOVA Post-hoc Tests"`;
   contains "parametric paired t-tests" / "pairwise paired t-tests"; or a
   Pingouin/Tukey substring match). The inline label
   `"Pairwise t-tests for interaction (Holm-Bonferroni)"` does not match those
   conditions, so for a significant default mixed run the reported
   `pairwise_comparisons` are the engine's Tukey-corrected values while the
   `posthoc_test` name field can retain the inline Holm-Bonferroni label. This
   mirrors the two-way and RM label-vs-correction mismatches documented in
   `two_way_anova.md` item 2 and `repeated_measures_anova.md` item 2. It does not
   affect the recipe text (which names no procedure) but is recorded for the
   human. Flagged, not fixed.
