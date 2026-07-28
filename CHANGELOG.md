# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Plots and visualisation

- Non-significant ("n.s.") significance brackets are hidden by default. Only
  significant pairs are bracketed; a new **Show non-significant brackets**
  checkbox on the Significance tab draws every tested pair for those who want it.
  Significance letters and significant-only brackets are unaffected.
- Box plots show the median and interquartile range only. The mean ± SD/SEM/CI
  overlay that used to sit on top of the box was removed — it layered a second
  statistic about a different centre on the same mark. Mean-based error bars stay
  on the bar and violin plots, where the height already represents the mean.
- Curated colour palettes are selectable — Nature (the default), Okabe-Ito,
  Grayscale HC, Muted Pastel, Deep, and Turbo — resolved through a single palette
  source so the dialog, live preview, and exported figure agree. Beyond a
  palette's length every group still gets a distinct colour instead of recycling.
- The raincloud layout was rebuilt to scale with group spacing (violin, box, and
  points no longer overlap or clip at the edges), significance letters are placed
  above each group's real top element, and significance brackets are aligned to
  the 0-based bar positions with clean corners and visible legs.
- Data points render as a single filled circle instead of cycling through
  per-index marker shapes.
- The live plot preview no longer clips long or rotated x-axis labels and the
  legend; the on-screen figure now fits its labels the way the exported file does.

### Testing / validation

- Added frozen R golden references for five statistical methods that previously
  had only targeted regression tests: correlation (Pearson/Spearman, 7 cases),
  Friedman (5 cases), Tukey HSD (3 cases / 12 pairs), Games-Howell (3 cases / 12
  pairs), and Dunn (3 cases / 12 pairs). Oracles match each method's actual
  implementation: `cor.test(exact=FALSE)` for the app's t-approximation Spearman
  p-value, `friedman.test`, Base R `TukeyHSD(aov)` for `statsmodels.pairwise_tukeyhsd`,
  and `PMCMRplus::gamesHowellTest` / `kwAllPairsDunnTest`. Dunn is validated in
  decoupled halves — the rank-based tie-corrected raw p-value against R, the
  Holm-Šidák multiplicity step as a statsmodels unit — with a wiring test (plus a
  positive control) proving the raw→adjusted seam attaches each adjusted p-value
  to the correct pair. The stale, never-collected `validation/validate_friedman.py`
  script (which read a since-renamed column and would crash) was removed; its one
  complementary structural check is absorbed into the new Friedman golden test.

## [2.0.0] - 2026-07-21

This release is the result of a multi-round statistical and release-readiness
audit. Many changes make the default behavior more conservative and correct
effect-size, p-value, and post-hoc computations. Several are behavioral changes
that can alter reported results, so read the breaking-changes section before
upgrading.

### Breaking changes (behavioral)

- Repeated-measures and mixed-ANOVA post-hoc now default to `paired_custom`
  (Holm-Šidák over type-correct per-pair tests: paired t for within-subject
  pairs, independent t for between-subject pairs). The previous default was a
  hand-rolled studentized-range "Tukey" that fed the paired-t degrees of freedom
  and a per-pair standard deviation into the studentized-range distribution,
  producing systematically too-conservative p-values — and it was incoherent
  with an omnibus that corrects for sphericity by default. The hand-rolled
  formula has been removed. Two-Way ANOVA keeps its correct `statsmodels`
  Tukey HSD. Non-normal designs continue to route to the nonparametric
  Friedman + Wilcoxon/Conover path automatically.
- **Mixed ANOVA with a non-significant interaction now uses the effect-driven
  post-hoc instead of a row-order-dependent one.** When the interaction was not
  significant — the ordinary situation when there is a real main effect — the
  follow-up came from an inline routine that paired observations by their
  position in the imported file rather than by subject. Reordering the rows of
  the same data set changed the result: in a measured example a timepoint
  contrast moved from p = 0.012 to p = 0.298, and comparison directions
  flipped. The contrasts were reported under the label "Paired t-tests
  (Holm-Bonferroni)", so nothing in the output looked wrong. The follow-up for
  this case now comes from the same effect-driven path described below, which
  pairs by subject. If you re-run a mixed analysis whose interaction was not
  significant, expect the pairwise p-values to change — the previous ones
  depended on the row order of your file.
- Mixed-ANOVA post-hoc is now effect-driven: the follow-up is chosen by which
  omnibus effects are significant. A significant interaction yields simple main
  effects (within-subject contrasts per group, plus between-group contrasts per
  within-level); otherwise the significant main effect's marginal-mean
  contrasts are reported. Holm-Šidák is applied per effect family. Earlier
  versions built every interaction-cell pair, including cross-cell pairs that
  differ in both factors at once and cannot be interpreted. The number and
  identity of reported comparisons will change for most mixed designs.
  The within-contrast error term is itself assumption-driven: a Levene test
  (Brown-Forsythe, `center='median'`) on the subject differences decides between
  a pooled error term and per-pair isolated tests, matching how the rest of the
  app routes Student vs. Welch. The Levene statistic and decision are carried in
  the output.
- Logistic regression now reports non-convergence for rank-deficient /
  unidentified designs (e.g. collinear predictors that yield non-finite
  standard errors) instead of presenting a misleading "converged" result with
  NaN standard errors. A warning is surfaced in the result's `warnings` field.
- When sphericity cannot be formally tested (for example, with incomplete
  tables), the Greenhouse-Geisser correction is now applied by default. Earlier
  versions assumed sphericity was met, which could inflate the Type-I error rate.
- The J-correction factor is now applied to every effect size labeled Hedges' g.
  Some Welch's test branches previously reported uncorrected Cohen's d under the
  Hedges label.
- Repeated-measures Dunnett now performs control-only comparisons. Earlier
  versions could fall through to all-pairwise comparisons, which Dunnett's test
  is not designed for.

### Removed

- **Beta Regression.** It was never a documented feature and had no entry point
  of its own: it was reached only through an auto-detection that overrode the
  inferred group-comparison test whenever the single dependent variable happened
  to be numeric, within [0, 1], and had more than five distinct values. Nothing
  announced it and nothing let you decline it. For this application's typical
  data a [0, 1] outcome is usually a ratio, a normalised signal or a
  fraction-of-control rather than a true proportion, so the substitution was
  methodologically wrong for the common case. When exact 0 or 1 values were
  present, the Smithson-Verkuilen step also rewrote the loaded data in place.
  A [0, 1]-valued dependent variable now runs the ordinary group comparison for
  its design, and the data is left untouched. If your outcome really is a
  proportion, the arcsin-square-root transformation remains available as an
  explicit choice in the transformation menu.
- **The Filter bucket.** It pinned one column to one value and silently narrowed
  the analysed data set to that subset. It duplicated what the group selection
  already does explicitly, and the report recorded only a single
  `Filter: column = value` line with no record of how many rows had been
  dropped — making it easy to leave one active and misread the result. Use the
  group selection instead.

### Statistical corrections

- Mixed/RM ANOVA: the Greenhouse-Geisser-corrected p-value is now wired into the
  canonical verdict field, read from the real `pingouin` sphericity columns and
  the correct backend keys, with a conservative GG default in the
  sphericity outer-exception path.
- ANCOVA: `control_group` is now wired through the primary clinical dispatch,
  and EMM contrast keys are aligned with the LMM schema.
- The LMM-vs-RM-ANOVA decision is recomputed per dependent-variable column in
  multi-DV mode instead of being decided once for all columns.
- Welch ANOVA now flags a silently degraded fallback (e.g. a zero-variance
  group) instead of returning a misleading result.
- The reported post-hoc test label is now always synced to the method actually
  applied.
- Post-hoc method labels no longer describe independent tests as paired. The
  Two-Way ANOVA post-hoc reported "Custom paired t-tests" while running
  independent t-tests over the interaction cells, and the selection dialog
  advised that "paired t-tests are often preferred" for every advanced ANOVA —
  false for Two-Way and misleading for mixed designs. The dialog now gives
  design-specific guidance: paired for repeated measures, independent for
  Two-Way, and automatic per-comparison pairing for mixed.
- Standard deviation computations use the sample estimator (`ddof=1`)
  consistently, including Cohen's d for repeated measures and bootstrap methods.
- Confidence intervals for bootstraps and effect sizes use the chosen `alpha`
  level instead of a hardcoded 95% (1.96) cutoff.
- Decimal parsing: US-formatted decimals are no longer misparsed as German
  thousands separators, which could silently corrupt imported values.
- The arcsin-square-root transformation now requires an explicit data-domain
  declaration — a proportion in [0, 1] or a percent in [0, 100] — and rejects
  data that violates the declared range instead of silently transforming it on
  the wrong scale (arcsin(√p) is variance-stabilizing only for true
  proportions). The classic one-way path also rescales out-of-range data against
  the global data range rather than per group, matching the advanced pipeline;
  the previous per-group min-max rescale collapsed the between-group differences
  the test is about. Cancelling the domain prompt drops the transform (the raw
  data routes to the nonparametric test) rather than applying an unchecked
  arcsin.
- Outlier detection now defaults to Grubbs' test rather than the Modified
  Z-Score. On clean data the Modified Z-Score flags a phantom outlier in about
  29% of n=3 samples (the common triplicate size), falling to ~12% at n=10,
  because its median/MAD scale is unstable at small n; Grubbs holds its nominal
  ~5% across all sizes. The Modified Z-Score remains available as an explicit
  choice, but now warns in the report for any group with n < 8. Detection
  continues to only flag rows — it never deletes data or feeds the analysis.
- Significance letters (compact-letter display) no longer hide a real
  difference. On an intransitive pattern — A not different from B, B not from C,
  but A different from C — the old assignment collapsed A, B and C onto one
  letter, displaying "no significant difference" where one existed. Letters now
  come from the maximal cliques of the non-significance graph: two groups share
  a letter only if they are mutually non-significant. Letters are the default
  annotation for omnibus post-hocs (Tukey/ANOVA/Dunn), so this affects the
  common one-way-ANOVA bar plot, and the violin and raincloud plots that offer
  no bracket alternative.
- "CI" error bars are now a t-based 95% interval (t(n-1)·s/√n) from one shared
  helper. Previously bar and grouped-bar drew a bootstrap CI while the box plot
  and the significance-letter height used the z-approximation 1.96·SD/√n; both
  understate uncertainty at the small n typical here (n=3 coverage ~75–82%
  instead of 95%) and disagreed by ~15% on identical data. SD and SEM error bars
  are unchanged.

### Data import and preprocessing

- CSV import now asks for the number format explicitly (International:
  `,`-separator with `.`-decimal, or German: `;`-separator with `,`-decimal and
  `.`-thousands) instead of trusting the pandas defaults, with a live preview of
  the parse before committing. A German-formatted export (`1,5`, or `1.234,56`
  with a thousands separator) previously read as garbage or silently became
  `NaN` with no error — the number was already wrong at read time and no
  downstream step could recover it. Cancelling loads nothing rather than
  importing corrupted data. Excel files are unaffected (numbers are stored as
  floats, not locale-formatted text).
- Group labels are whitespace-normalized. `"A"`, `"A "` and `" A"` — the same
  group with a stray space from a dirty sheet — counted as three distinct
  groups, which split a real group across phantom duplicates and could starve it
  below the minimum-n gate, blocking an otherwise valid analysis. Labels are now
  stripped consistently. Case is deliberately not folded (`"A"` and `"a"` stay
  separate, pending a separate decision).
- Silent row loss during preprocessing is now surfaced in the report. Rows
  dropped because their group label was missing or blank, and non-numeric value
  cells that could not be read as a number, previously vanished without a trace;
  they now appear as data-health warnings in the report, with counts, so a
  silently shrunk data set is visible rather than invisible.

### Model input validation (reject invalid designs instead of misleading output)

- Logistic regression: covariates-only models are allowed; models with no
  predictors are rejected; a wrong outcome-level count raises `ModelDesignError`;
  an operator-precedence bug in the binary-outcome classifier is fixed.
- LMM: models with no random-intercept column or no fixed effects are rejected;
  `mixed_anova` requires a subject column.
- ANCOVA: models with no covariates or no between-subjects factor are rejected.
- Advanced pipeline: a pre-flight data-quality gate guards logistic regression;
  the pre-flight gate no longer blocks text-labeled outcomes; wide-format data
  with missing subject IDs is rejected at detection time and before
  analysis-context building; all-NaN columns are excluded from wide-format
  value columns.

### Security / output hardening

- HTML export is escaped consistently through a shared `_FormattingMixin`
  helper, closing HTML-injection vectors: the parameter cell in coefficient-table
  builders, the remaining Plotly label sites, and ANCOVA chart group labels that
  a prior regex (RE2) missed are now escaped.

### Bug fixes and stability

- Crashes fixed: the global excepthook no longer crashes before showing its
  dialog; `posthoc_choice` is initialized to prevent a `NameError` in the
  non-parametric branch; a nonexistent `test_name` kwarg was dropped from
  `make_blocked_result` calls; the Help Hub no longer raises an `AttributeError`
  on a missing copy button.
- Silent failures made visible: a notice when comparison brackets are dropped;
  a visible warning when significance letters fail to compute; an on-canvas
  warning when a log-scale axis drops non-positive data; a visible warning when a
  grouped-EMM plot falls back to a flat bar; loud logging instead of silently
  blanking missing report-table row keys.
- Plots: an unrecognized `plot_type` now raises instead of silently falling back
  to a bar chart; invalid plot filenames are sanitized inline instead of being
  dropped behind a modal; raincloud plots route through the shared
  bracket-vs-letters helper.
- Rotated x-axis tick labels are no longer clipped. The figure now grows its
  bottom margin to fit them, on screen and in every exported format (including
  SVG), for all plot types.
- The decision-tree viewer shows the post-hoc options actually offered for the
  design rather than a fixed list.
- Log axes: `logx` now matches `logy`'s lossless symlog auto-adapt standard;
  symlog is auto-applied for log-axis data with values ≤ 0 (with a data-driven
  `linthresh`); Log Y is grey-toggled with an explanatory tooltip for
  non-positive data instead of silently mis-scaling.
- Detailed analysis logs are no longer discarded during standard exports; they
  appear again in the HTML reports. Standard export paths are wrapped in error
  handlers so a failed export no longer affects later datasets.
- Invalid p-values (negative numbers, NaN) are flagged as `invalid` rather than
  formatted as `< 0.001`.
- A leftover German checkbox label was translated to English.

### Accessibility

- Replaced 11 WCAG-failing journal-palette colors and 2 non-compliant
  `DEFAULT_COLORS` entries with legible, contrast-compliant variants.

### Performance

- Vectorized the Dunn-test bootstrap confidence interval (~13.5 s/pair to
  sub-second).

### UI & Help

- Added keyboard zoom (+/−/0) to the decision-tree viewer.
- Reorganized in-app help into a categorized Help Hub (recipes grouped by
  category, inline dialogs migrated in, redundant standalone menu items removed).
- The linear-regression coefficient table is now rendered in the HTML export.
- Factor levels are ordered the same way everywhere — plots, post-hoc labels,
  comparison dialogs and column previews — using a human-friendly ordering that
  no longer depends on the row order of the input file (so `Dose 2` sorts before
  `Dose 10`). This is presentational only; the levels and the statistics
  computed on them are unchanged.
- The comparison-selection dialog was restyled to match the rest of the
  interface.
