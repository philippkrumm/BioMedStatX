# Audit note: `statistical_tests_html` recipe

Recipe location: `src/core/help_content.py:625` (`"id": "statistical_tests_html"`,
`"category": "Workflow & Output"`).

This is the last of the 12 Help Hub recipes. It is a "Workflow & Output" sibling
of `graph_visualization` and covers two distinct topics: (a) how the app selects
which statistical test to run, and (b) what the exported HTML report contains. The
citation anchor (symbol name or quoted string) is authoritative; the line number is
only a navigation hint.

The known issue carried in from the just-completed `graph_visualization` audit
(the code calls significance annotations "brackets", not "bars") is confirmed and
fixed here too, for terminology parity between the two Workflow & Output recipes.

## Recipe-economy decision (topic a: test selection)

The old recipe re-taught the entire test-selection taxonomy (t-Test vs
Mann-Whitney, paired-t vs Wilcoxon, ANOVA vs Kruskal-Wallis, Shapiro-Wilk, Levene)
in the recipe body. Eleven prior audits already verified that mechanics against
`statisticaltester.py`, and two shipped recipes own it in full: `one_way_anova`
("Comparing groups (t-Test / One-Way ANOVA)") and `dependent_samples`
("Dependent (paired) samples"). Per the spec's recipe-economy rule (design doc
lines 58-68), this triage page now gives a one-paragraph summary in the same
"the app checks whether your data is normally distributed ... and adjusts
automatically" register as the audited `one_way_anova` body
(`help_content.py:92`), and points the reader to those two recipes for the full
decision logic and data layout, rather than duplicating them. The full named
procedures live in this note (below), not in the shipped text.

## Ground truth: test selection (topic a)

All verified in `src/analysis/statisticaltester.py`, which routes through
`AssumptionCheckEngine.check_normality_and_variance`
(`src/statistical_testing/assumption_checks.py:37`):

- **Default alpha = 0.05.** `perform_statistical_test(..., alpha=0.05, ...)`
  (`statisticaltester.py:235`). Post-hoc engine invoked at `p < alpha`
  (`statisticaltester.py:744`-`765`, log string "Main {rec} test significant
  (p<{alpha})" at `:761`).
- **Two independent groups:** `stats.ttest_ind(..., equal_var=equal_var)`
  (`_independent_ttest`, `statisticaltester.py:562`-`566`);
  `stats.mannwhitneyu(..., alternative='two-sided', ...)` (`_mannwhitney_test`,
  `statisticaltester.py:633`-`640`). Note the code also offers a Welch t-test
  strategy when variances are unequal (`"welch_ttest"` at `:348`); the recipe folds
  this into "checks ... whether the groups have similar spread, then picks
  accordingly" rather than naming Welch (recipe-economy), consistent with how
  `one_way_anova` handles it.
- **Two paired groups:** paired-t / Wilcoxon selected via the dependent branch;
  Shapiro-Wilk runs on the within-pair differences
  (`assumption_checks.py:167`-`172`, comment "paired t-test → Shapiro-Wilk on
  within-pair differences"). Full detail is in the `dependent_samples` audit.
- **Three or more independent groups:** strategy dispatch chooses parametric
  one-way ANOVA / Welch ANOVA / Kruskal-Wallis from the assumption results:
  `strategy = "kruskal_wallis"` when not normal, `"welch_anova"` /`"oneway"`
  otherwise (`_stat_test_multi_groups`, `statisticaltester.py:669`,
  `:686`-`691`), keyed on `assumptions.residuals_normal` /
  `assumptions.equal_variance` (`:694`-`707`).
- **Assumption checks are Shapiro-Wilk + Brown-Forsythe, not Levene.** Corrected
  after spec review; the first version of this note and recipe wrongly called the
  variance check "Levene". `check_normality_and_variance` fits the model and
  tests residuals with `stats.shapiro(...)` (`assumption_checks.py:172`, `:215`),
  and tests variance homogeneity with `stats.levene(*validated_levene_data,
  center='median')` (`assumption_checks.py:285`), but the code explicitly labels
  the result `test_name = "Brown-Forsythe"` on every real path
  (`assumption_checks.py:274` pre-transformation, `:534` post-transformation; the
  only other assignments in that branch are the dependent-design bypass strings
  `"N/A (Paired)"` / `"N/A (Repeated Measures / Mixed)"`, never `"Levene"`). That
  string is what reaches the user: `report_summaries.py:387` renders
  `f"Variance homogeneity ({_var_name})"` where `_var_name =
  variance_test.get("test_name", "Levene")`; the `"Levene"` in that `.get()` is a
  fallback default that is dead code in practice because every populated
  `variance_test` dict already carries `"Brown-Forsythe"` from
  `assumption_checks.py`. So the exported HTML report's assumption table row
  literally reads "Variance homogeneity (Brown-Forsythe)". Every other
  user-facing surface agrees: `html_exporter.py:400` methods text says "Levene's
  test (Brown-Forsythe variant, center = median)", i.e. it names Brown-Forsythe
  as the variant actually run; `analysis_core.py:1042`, `:1045`, `:1064`, `:1067`
  log "Brown-Forsythe test"; `statisticaltester.py:1212`, `:1221` log the same.
  Only an internal module docstring (`assumption_checks.py:49`, never shown to a
  user) says "Levene test". **Precedent check, also corrected:** the first
  version of this note claimed this matched "the same nuance already accepted in
  the `one_way_anova` audit". That was wrong. `one_way_anova.md` claim 13
  explicitly labels the variance check "Brown-Forsythe", and the shipped
  `one_way_anova` recipe text never uses the word "Levene" at all (it stays
  generic: "the groups have similar spread"). A repo-wide grep of
  `help_content.py` confirms "Levene" appeared nowhere else; this audit's first
  draft was the sole, inaccurate introduction of the word. Fixed to
  "Brown-Forsythe for equal variance" in the recipe, matching the label the
  report actually displays.
- **Post-hoc is auto and adjusted.** When the omnibus test is significant, the
  app runs `PostHocEngine().execute(...)` and stores `pairwise_comparisons`
  (`statisticaltester.py:765`-`779`); adjustment for the number of comparisons is
  the post-hoc engine's job (Tukey/Games-Howell/Dunn etc., detailed in the
  per-family audits). The recipe says "compares each pair of groups and adjusts
  the p-values for the number of comparisons", matching `one_way_anova`'s audited
  phrasing (`help_content.py:93`) without re-listing the named methods.

## Ground truth: HTML report contents (topic b)

The report is rendered by `HTMLExporter` (`src/export/html_exporter.py`) into the
Jinja templates `src/templates/report_single.html.j2` (single analysis) and
`report_multi.html.j2` (companion overview). The single-report template has a
**fixed, generic section layout** (TOC at `report_single.html.j2:26`), not
sections named after the test:

| Section (id / kicker / heading) | What it shows | Anchor |
|---|---|---|
| Hero banner | "Selected Test", "p-value", effect size + magnitude badge, significance badge, summary note | `report_single.html.j2:25`; built by `_build_hero_context` (`html_exporter.py:273`, fields `test_name`/`p_value_display`/`effect_size_display`/`effect_label`/`significance_label` `:284`-`297`) |
| Decision Path (`#sec-decision`) "How BioMedStatX reached this decision" | animated decision-tree of the test choice | `report_single.html.j2:27` |
| Main Results (`#sec-results`, kicker "Statistical Engine") | Metric/Value table: Test, Model type, Statistic, p-value, Adjusted p-value, Effect size, Effect size type, Confidence interval, df1/df2, Transformation, Post-hoc test | `report_single.html.j2:28`; rows from `_build_statistical_rows` generic loop (`report_stat_rows.py:554`, list at `:585`-`598`) |
| Assumptions (`#sec-assumptions`) "Model validity checks" | Check / Statistic / p-value / Status table + Q-Q, distribution, residual plots | `report_single.html.j2:29` |
| Descriptives (`#sec-descriptive`) "Group-level summary" | Group / n / Mean / Median / SD / SEM / Min / Max | `report_single.html.j2:30`; rows from `_build_descriptive_summary` emitting mean/median/sd/sem per group (`report_summaries.py:487`, `:535`-`538`) |
| Pairwise (`#sec-pairwise`) "Post-hoc findings" | Comparison / Procedure / Statistic / p-value / Effect size / Interpretation, plus a per-row "Chart" bracket toggle | `report_single.html.j2:31` (rendered only `{% if context.pairwise_rows %}`) |
| Interactive Charts (`#sec-charts`) "Visual evidence" | Plotly charts | `report_single.html.j2:32` |
| Plot Designer | in-report figure builder, only `{% if context.plot_designer_enabled %}` | `report_single.html.j2:33` (see `graph_visualization` audit) |
| Raw Data Vault (`#sec-raw`) | searchable raw values, Copy / Download CSV | `report_single.html.j2:34` |
| Methods Snippet (`#sec-methods`) | reusable narrative text, Copy button | `report_single.html.j2:35` |

**Self-contained / offline:** the report bundles Plotly and renders charts "fully
offline inside this file" (`html_exporter.py:430`, "Interactive Plotly charts
rendered fully offline inside this file"); written with `open(output_path, "w",
encoding="utf-8")` (`html_exporter.py:59`, `:73`); template footer reads
"Generated by BioMedStatX as a fully offline HTML scientific report"
(`report_single.html.j2:39`).

**Significance markers = brackets with stars, not letters, not "bars".** In the
HTML report, significant pairs are annotated with brackets carrying star strings.
The pairwise table has a "Chart" column of checkboxes labelled
`Show bracket for {comparison}` (`report_single.html.j2:31`,
`class="bracket-toggle"`), and the report JS draws bracket line shapes plus a
`<b>{stars}</b>` annotation per checked pair (`report_single.html.j2:269`). The
static-figure equivalent is `_ChartsMixin._build_significance_brackets`
(`src/export/report_charts.py:1324`, docstring "Add significance bracket
annotations (*, **, ***)", star thresholds `p<0.001 → ***`, `p<0.01 → **`, else
`*` at `:1354`). Star thresholds also appear in
`report_formatting.py:188` and `report_stat_rows.py:721`. A repo-wide search for
significance **letters** finds them only in the matplotlib plotting path
(`DataVisualizer._add_significance_letters`, called from the plot renderers,
`datavisualizer.py:1026`, `:1220`, `:1426`), i.e. the `graph_visualization`
surface, never in the export layer or the report templates. So in the report the
markers are brackets/stars; compact letters belong to the plot dialog. The recipe
now says exactly that.

## Claim table

| # | Claim (from title/html) | Verdict | Citation |
|---|-------------------------|---------|----------|
| 1 | Test is auto-selected from group count and independent/paired structure | correct | `perform_statistical_test` dispatch two vs multi group, dependent flag (`statisticaltester.py:235`, `_stat_test_two_groups` `:312`, `_stat_test_multi_groups` `:669`) |
| 2 | Two independent groups: t-Test, or Mann-Whitney U when not normal | correct | `stats.ttest_ind` (`:566`), `stats.mannwhitneyu(alternative='two-sided')` (`:639`); MWU chosen when non-parametric recommendation |
| 3 | Two paired groups: paired t-Test, or Wilcoxon signed-rank when differences not normal | correct | dependent branch; Shapiro on within-pair differences (`assumption_checks.py:167`-`172`); full detail in `dependent_samples` audit |
| 4 | Three or more independent groups: one-way ANOVA, or Kruskal-Wallis when assumptions fail | correct | strategy `"kruskal_wallis"` vs ANOVA (`statisticaltester.py:686`-`691`) keyed on residual normality / equal variance (`:694`-`707`) |
| 5 | Decision uses Shapiro-Wilk (normality) and a median-centered Levene call, displayed as Brown-Forsythe (variance homogeneity) | correct (label fixed after spec review) | `stats.shapiro` on residuals (`assumption_checks.py:172`,`:215`); `stats.levene(..., center='median')` (`:285`) is the underlying scipy call, but the code assigns `test_name = "Brown-Forsythe"` on every real path (`:274`,`:534`), which is what reaches the report (`report_summaries.py:387`). First draft of this recipe and note wrongly said "Levene" and wrongly claimed precedent from `one_way_anova`; that recipe actually says "Brown-Forsythe" (its audit note claim 13) and never uses "Levene". Recipe now says "Brown-Forsythe for equal variance". |
| 6 | You don't choose parametric vs non-parametric; app switches automatically | correct | recommendation drives strategy without user input (`statisticaltester.py:281`, `:686`-`691`); register matches audited `one_way_anova` `help_content.py:92` |
| 7 | Post-hoc pairwise comparisons auto-added when the main test is significant, with p-values adjusted for the number of comparisons | correct | post-hoc gated on `p<alpha` (`statisticaltester.py:744`-`765`), `PostHocEngine().execute(...)` populates `pairwise_comparisons` (`:765`-`779`); adjustment is the engine's (named methods in per-family audits) |
| 8 (reworded) | Points the reader to the sibling recipes for full logic and data layout | correct (recipe-economy) | sibling recipes `one_way_anova` (`help_content.py:80`) and `dependent_samples` (`:565`) own the mechanics; titles cross-referenced verbatim |
| 9 | p-values are shown | correct | hero "p-value" card (`report_single.html.j2:25`, `p_value_display` `html_exporter.py:290`); "p-value" row in Main Results (`report_stat_rows.py:589`); p-value column in pairwise table (`report_single.html.j2:31`) |
| 10 (was "letters or bars") | Significance markers on report charts are brackets with stars (*, **, ***); compact letters are a plot-dialog option | correct, reworded | brackets: `bracket-toggle` + `<b>{stars}</b>` (`report_single.html.j2:31`,`:269`), `_build_significance_brackets` (`report_charts.py:1324`,`:1354`). Letters only in `DataVisualizer._add_significance_letters` (`datavisualizer.py:1026` etc.), i.e. the `graph_visualization` surface, never in the report. Old "bars" term corrected to "brackets" per code/hub usage. |
| 11 (was "means, SDs, test statistics clearly displayed") | Report shows test statistic, effect size, per-group means/medians/SD/SEM | correct, expanded | Main Results rows incl. Statistic + Effect size (`report_stat_rows.py:585`-`598`); descriptive table mean/median/sd/sem (`report_summaries.py:535`-`538`). Old wording omitted effect sizes, which are prominent (hero + Main Results + pairwise); now included. |
| 12 (was "self-contained HTML report") | One self-contained HTML file, opens offline in a browser, printable/shareable | correct | offline Plotly bundle (`html_exporter.py:430`), single file write (`:59`,`:73`), footer "fully offline HTML scientific report" (`report_single.html.j2:39`) |
| 13 (was "Sections reflect the test/plot type, e.g. 'ANOVA Results'") | Report has a fixed set of sections (banner, decision path, main results, assumptions, descriptives, pairwise, charts, raw data, methods) | wrong -> corrected | sections are generic and fixed: TOC `Decision Path / Main Results / Assumptions / Descriptives / Pairwise / Charts / Raw Data / Methods` (`report_single.html.j2:26`); headings "Main results" (`:28`), "Post-hoc findings" (`:31`) etc. No "ANOVA Results" heading exists; the test name appears as the hero "Selected Test" value (`:25`), not as a section title. Recipe rewritten to list the real sections. |
| 14 (was "Each section shows group names, means, test statistics, p-values, significance markers") | (folded into the per-section list in claim 13) | corrected | that flat claim conflated several tables; replaced by the accurate per-section breakdown above |
| 15 (new) | Report opens with a decision path tracing how the test was chosen | correct (was MISSING) | `#sec-decision` "How BioMedStatX reached this decision" + decision-tree (`report_single.html.j2:27`) |
| 16 (new) | Assumption checks appear with their own p-values and pass/fail status | correct (was MISSING) | Assumptions table Check/Statistic/p-value/Status (`report_single.html.j2:29`) |
| 17 (new) | Report includes searchable raw data and a ready-to-paste methods paragraph | correct (was MISSING) | Raw Data Vault with search + CSV (`report_single.html.j2:34`); Methods Snippet with copy button (`:35`, `methods_text`) |

## Changes applied to the recipe html

- **Topic a (test selection)** collapsed from a full taxonomy into one summary
  paragraph plus a three-bullet group-scenario list, matching the audited
  `one_way_anova` register, and now cross-references the two owning recipes by
  their exact titles (recipe-economy, claims 1-8). Dropped the standalone
  parametric/non-parametric labels on every bullet; kept the "when the data is
  not normal" trigger inline.
- **Topic b (report contents)** rewritten from four vague bullets to the actual
  fixed section list in render order (claims 13-17), adding the decision path,
  assumption table, effect sizes, raw-data vault, and methods snippet that the old
  text omitted, and correcting "sections reflect the test/plot type ('ANOVA
  Results')" to the real generic layout.
- **Significance markers:** "letters or bars" -> "brackets with stars", with the
  star thresholds and the per-pair chart toggle, and a note that compact letters
  are a plot-dialog feature covered in `graph_visualization` (claim 10). This is
  the known terminology fix carried over from the `graph_visualization` audit.
- **Key statistics** line expanded to include effect sizes (claim 11).
- Rewrote prose for AI-writing tells per the humanizer skill (copula avoidance,
  `-ing` padding, rule-of-three, filler). Kept the `<h2>` + `<ul>` +
  bold-label-list structure to match the sibling `graph_visualization` recipe and
  the Help Hub list-structure tests. No `id`/`category` change. No emoji or
  typographic dashes introduced (verified: one `<h2>`, zero `<h3>`, no forbidden
  dash codepoints).
- **Post-review fix (same bug class as the letters/brackets fix):** spec review
  caught that the first draft's assumption-check bullet said "Levene for equal
  variance" (`help_content.py:645`). Traced independently and confirmed wrong:
  `assumption_checks.py:274`/`:534` sets `test_name = "Brown-Forsythe"` on every
  real path, which flows through `report_summaries.py:387` into the exported
  report's assumption table, so the report literally reads "Variance homogeneity
  (Brown-Forsythe)". Fixed to "Brown-Forsythe for equal variance". The note's
  precedent claim ("matches the `one_way_anova` audit") was also wrong and is
  corrected in the claim-5 row and the ground-truth bullet above; `one_way_anova`
  actually used "Brown-Forsythe", never "Levene".

## Unclear / possible code bug

- **None affecting this recipe.** Every claim traced cleanly to a rendered
  surface (template + context-builder), all the way to the exported HTML, not just
  to a model that computes a value. No code looked wrong along the audited paths.
- **Observation (not a defect):** the report's "Main Results" table is titled with
  the generic kicker "Statistical Engine" / heading "Main results" regardless of
  the test, while the test name lives in the hero "Selected Test" card and the
  decision path. This is intentional (the layout is test-agnostic), and is the
  reason the old recipe's "sections named after the test (e.g. 'ANOVA Results')"
  claim was wrong. Recorded so a future reader does not reintroduce test-named
  section headings into the recipe.
- **Cross-note consistency:** the significance-letters-vs-brackets split confirmed
  here is the mirror image of the `graph_visualization` finding: letters and
  brackets are both real, but they live on different surfaces. Brackets/stars are
  the report-chart annotation (`report_charts.py:1324`); compact letters are the
  matplotlib plot-dialog annotation (`datavisualizer.py:_add_significance_letters`).
  Both Workflow & Output recipes now use "brackets", never "bars".
