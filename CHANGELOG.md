# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]


## [2.0] - 2026-08-30

### Plots and visualisation

- The interactive figure builder's top margin grows with the title size. It was
  fixed at 58 px while every axis had `automargin`, so a title was clipped by the
  top edge of the figure at 42 pt — the largest size the control itself offers.
  The default (16 pt) is unchanged. Found by the new visual fuzzer on its first
  real batch, and pinned by a negative control that restores the old margin.
- Choosing a plot type that cannot be drawn no longer leaves the previous figure
  on screen. `buildPlot` warned ("No plottable data found.") and returned, so
  picking **Forest** on a design without per-comparison effect sizes left the
  bar chart standing — with its significance letters, at their old coordinates,
  under a significance control that had switched itself to "none". The canvas is
  now cleared, so the warning is the whole story.
- The raincloud layout no longer applies a log scale to values that include zero
  or less. It draws its violins along x, and a log x axis with a non-positive
  value produced SVG paths with a missing coordinate — Chromium reported
  `<path> attribute d: Expected number` and the shape landed thousands of pixels
  off-canvas. Log is now skipped for that data with an explicit note, matching
  the wording the significance layer and the reference lines already use; an
  all-positive raincloud still plots on log as before.
- Two-factor cells keep the control first. A two-factor design reaches the
  ranking as one joined label per cell (`Genotype=WT, Time=D0`). The numeric half
  survived that join — D7 still preceded D21 — but control-first did not, so
  `Genotype=KO, …` sorted ahead of `Genotype=WT, …` and the reference cell
  stopped leading the axis. Cells are now ranked by their components, which the
  same code already records, so every rule the single-factor path has applies to
  the major factor and then the minor one.

- Subject lines are drawn in a mixed design, inside each between group. They
  were refused there, but by accident rather than by rule: the subject labels
  were keyed differently from the values, so the "no subject identity" branch
  fired before anything considered the axis. A subject belongs to one between
  group for the whole study, so a line can never cross one; what remains is the
  path along the ordered within factor, which is the repeated-measures case
  drawn once per between group. The blocks are read off the subjects rather than
  the labels, so nothing depends on which factor was written first; the ordering
  requirement is asked of the part that varies inside a block, since whether
  `Site=Aachen` precedes `Site=Bonn` says nothing about the path a line asserts;
  and a block whose levels are not drawn side by side is still refused, because
  a line would then reach across another group's bars.
- Every factor of a cell label is ranked on its own. Stripping the `factor=`
  prefixes and rejoining what was left meant looking up `"M, WT"` as if it were
  a level, which matches no reference-term table, so the pair fell to the
  alphabet: `Sex=F, Geno=KO` sorted ahead of `Sex=M, Geno=WT` while the same two
  levels alone put WT first. Numbers hid it, since digits are found anywhere in
  a string, so a cell whose levels were T0/T1 came out right and looked like
  proof. The sort key is now rank and name per factor, interleaved, so the
  primary factor keeps deciding the grouping. The transparency note follows the
  same split: each adjacent pair is judged at the factor that separates them,
  which is what keeps an alphabetical guess in one factor declared even when the
  other is ranked.

### Import and mapping

- The wide-format notice names the design it built. `_detect_wide_format` accepts
  two to eight condition columns, but the line explaining the pivot read "Mapped
  as paired t-test design" for all of them — so a four-timepoint study was told
  it had been mapped as a paired t-test and then handed a repeated-measures
  ANOVA. It now reads "paired design (2 conditions)" or "repeated-measures
  design (N conditions)"; naming the design rather than a test stays true when
  the assumption checks pick Wilcoxon or Friedman instead.

### Fixed

- The cockpit's design card names the model that ran. `inferred_test` is chosen
  from the shape of the data, before the assumption checks look at the numbers,
  so on three groups with unequal spread the card said "Model: One-Way ANOVA"
  while the results, the post-hoc and the methodology trace all said Welch's
  ANOVA. The formatter was already handed the result and did not read it, while
  the two cards beside it and the post-hoc line one row down all do. A run that
  produced no test still shows the planned design, since a placeholder is worse
  than a plan.
- A design label no longer names a test this program does not run. The router
  picks Welch unconditionally for independent groups -- `select_comparison_test`
  does not even read the variance check on that branch -- so a classic one-way
  ANOVA and a Student t-test are never the outcome, and the decision tree, the
  report and the post-hoc all say Welch. Two labels still did not: the inferred
  design was announced as "One-Way ANOVA" and "Independent t-test", in the
  cockpit and in "Structure inferred as ...". They now name the layout ("One-way
  design (independent groups)", "Two independent groups"), so the design is
  described by the shape of the data and the test name comes only from what ran.
  The rename is narrow: two-way, repeated-measures, mixed, ANCOVA and LMM really
  are what runs under those names, and keep them.

- A logistic fit that produced no test statistic is now stopped at the
  data-quality gate instead of reported. The identification guard read the
  coefficient standard errors, which can come back finite while the omnibus does
  not: a Firth fit on quasi-separated data with a collinear predictor overflowed
  in the link function and returned `statistic = nan`, `p = nan` — and
  `converged` was hardcoded `True` on the Firth path, so it was reported as a
  converged result. The guard now also reads the omnibus, and an unidentified
  fit is blocked with a reason naming separation and collinear predictors, and
  saying that Firth penalisation was already applied so it cannot be the remedy.
  This matters beyond the p-value: the report carried AUC 0.9167, a ROC curve
  and a calibration plot from that same model, all quotable. Both entry points
  block through one shared helper.
- The "No result" note says why when the engine knows. A correct negation is
  still a dead end for the reader; where the fit recorded a failure to converge,
  the sentence names it and says what to check.
- A p-value that does not exist is no longer reported as a p-value above alpha.
  `isinstance(nan, float)` is True and `nan < 0.05` is False, so a model that
  produced no answer was filed under "produced a negative answer": a Firth
  logistic fit that overflowed on separated data with a collinear covariate
  returned `p = nan` and the report badged it **Not significant** and wrote that
  the test "did not show evidence against the null hypothesis" -- a claim about
  the data, drawn from a number that does not exist, and indistinguishable to
  the reader from a genuine null finding. The hero badge now has three states,
  the third reading **No result**, and the summary sentence makes no claim at
  all. The gate is a shared helper, so the four places that asked this question
  ask it the same way. Found by the fuzzer.
- A repeated-measures analysis no longer reports a transformation it did not
  perform. The gate deciding whether anything was transformed compared the two
  sample sets position by position, and the RM path hands the same measurements
  back in a different order -- so a permutation read as a change, and an
  untransformed run produced a "Transformed value" column, a transformed-scale
  means note and two "After transformation" diagnostic charts, all showing the
  raw numbers, beside a badge correctly reading "Transformation: None". The
  comparison is now between multisets: a transformation that leaves the values
  intact has altered nothing, which is the case the gate exists to suppress, and
  a real transformation still shows all three. Found by the new report check on
  its first run, not by hand.
- The raw data table pairs each measurement with its own transformed value. It
  prints `raw_data[g][i]` beside `raw_data_transformed[g][i]`, one row per
  index, which is a claim about a single measurement -- and the two halves came
  from two different extractions. Advanced designs get their transformed
  samples from `prepare_advanced_test`, which reads the frame itself, while the
  raw half was taken from `AnalysisManager`'s own separately-extracted copy;
  same values, same length, same groups, different order. On one repeated-
  measures run with a genuine log10, 24 of 28 printed rows showed one subject's
  raw value next to another subject's transformed value, including raw values
  below 1 printed beside positive base-10 logarithms. Every summary built from
  the columns (means, SD, Q-Q plot, distribution charts) was unaffected, which
  is why nothing caught it. The raw half is now taken from the dict the
  transformed one was derived from, whenever that dict describes the same
  groups; where it does not, the table drops the transformed column rather than
  pairing across extractions, as it already did.
- The report self-check reads the raw data vault's header row correctly. Its
  header pattern `<th[^>]*>` also matched `<thead>` -- "<th" plus "ead" plus
  ">" -- so the first header came back as `<tr><th>Group` and the Group column
  could never be found by name. The raw and transformed columns were located
  correctly anyway, by accident: the mangled entry occupied exactly the one slot
  "Group" would have.

- The mixed EMM/multivariate-t post-hoc could not be reached from the product at
  all. The engine matches its R reference and has its own tests, which call it
  the way it documents itself; the pipeline calls it across a seam, and all
  three things that crossed were wrong. `between`/`within` arrived as lists and
  went straight into a pandas column selection, which raises `unhashable type:
  'list'`. The control group arrived as a cell label where a between-factor
  level was wanted, so the engine refused it as "not present" and the run
  degraded to isolated t-tests with only a log line to say so. And the
  comparisons came back in a label vocabulary nothing else uses, so even a
  successful run drew no brackets. None of it was visible because the function's
  own error handler wrote into a variable assigned below the branch that raised:
  the handler itself failed, and "cannot access local variable 'result'" was
  recorded as the post-hoc's error.
- A value and the subject printed beside it now come from the same extraction.
  They were filled by two, and both ways of disagreeing were live. Without
  technical replicates the keys disagreed (`T0` against `Time=T0`), so the
  Subject column was absent from every repeated-measures report and the figure
  refused subject lines with "No subject was measured at more than one level" --
  false about every subject in the design. With replicates the keys agreed and
  the lengths did not, because the analysis averages replicates and the raw
  values do not: 24 values per level were labelled from a list of 8, and every
  printed row named the wrong subject. Both halves are now filled in one loop,
  and where they do not line up the labels are dropped rather than printed, on
  the ground that an absent column says nothing while a wrong one says something
  false.

- An impossible number is no longer reported as a weak result. A 2x2 design with
  one cell never run -- a factorial experiment where a combination was not
  performed, which is routine -- reported `F(1, 14) = -3.07`, partial eta squared
  `-0.28` and `p = 1.0`. F is a ratio of mean squares and both are non-negative;
  partial eta squared lies in [0, 1]; neither value can occur. An empty cell
  leaves the interaction unestimable and the negative sum of squares behind it
  was read straight through. The app had already warned that the design has
  empty cells and recommended a mixed model -- and then printed the number
  anyway, beside a p-value that reads as an ordinary null result. The existing
  safety net tested only for non-finite values, so "not a number" was caught and
  "a number that cannot be" was not; it now also rejects a negative F and an
  effect size outside the range its own name fixes, and the block names the
  incomplete layout as the cause. Deliberately narrow: signed statistics and
  signed effect sizes are the ordinary case, and omega squared is legitimately
  negative in small samples.
- The Transformed column is printed only where it pairs with the raw one. The
  two columns come from different extractions that disagree about missing
  values: one drops a NaN row, the other keeps it. On a 2x2 layout with
  scattered NaNs a cell held 5 raw values against 8 transformed ones, so every
  row after the first gap named the wrong measurement. The same difference
  fooled the gate deciding whether to show the column at all -- the two dicts
  differed, by NaN handling rather than by any transformation -- so a report
  declaring "Transformation: None" printed a transformed value for every row.
  The column is dropped rather than realigned, for the reason the defect exists:
  guessing which value belongs to which is what produced it. Both writers are
  guarded, since the table falls back to a second source and closing one door
  alone would have left the misaligned column reaching the page through it.
- A two-way analysis that falls back to permutation no longer freezes the
  window. Two designs took 91 seconds on 28 rows, all of it in the
  Freedman-Lane loop, which rebuilt both design matrices from their formula
  strings on every permutation -- a patsy parse and a categorical re-encode of
  the whole frame, twice, 5000 times, for each of three effects: 30000 model
  fits to answer a question about 28 numbers. Freedman-Lane permutes the
  residuals, so the designs are constant and only y changes; the residual sum of
  squares now comes from a projection onto an orthonormal basis of each design's
  column space, from an SVD rather than a QR because a rank-deficient layout is
  exactly the case this fallback exists to survive. Measured against the old
  implementation on the same fixture: 36.20 s to 0.07 s, with F and p identical
  to every digit. The analysis runs on the UI thread, so this was a minute and a
  half of frozen window on a design a user can build by accident.

- A two-way main effect at `p = 0.0002` was reported as not significant. The
  post-hoc after a significant interaction runs `pairwise_tests(padjust='holm')`,
  whose corrected-p column is present on every row and left empty for a family
  with only one comparison -- there is nothing to correct there. The code asked
  whether the column existed, which is always true, and took the empty value: on
  a 2x2 design that is both main effects, so a factor separating its groups at
  `p = 3.8e-16` was filed as `p = NaN`, "not significant", indistinguishable
  from a genuine negative result. The corrected value is used where one exists
  and the uncorrected one where no correction applied, and each row now says
  which, since both occur in the same table.
- That post-hoc is no longer called Tukey HSD. It is a set of pairwise
  t-tests with a Holm-Bonferroni adjustment; Tukey uses the studentized
  range and gives different p-values. The comparison rows had always
  recorded "Pairwise t-test" -- only the heading claimed otherwise, and a
  methods section is written from the heading. Found by the new post-hoc-name
  check on the first two-way seed it reached.
- A simple-effect row names the level it was measured at. Such a row compares
  two levels of one factor at one level of the other, and the conditioning level
  was dropped, so a 2x2 table printed "B0 vs B1" twice with different
  p-values and no way to tell the two comparisons apart. They now read
  `B0 (FacA=A0)` against `B1 (FacA=A0)`; main-effect rows, which are conditioned
  on nothing, are unchanged.

- The Transformed column is paired with the raw values that were kept. An
  earlier fix this cycle guarded the standard path's writer; there are four --
  the standard path, the tester's own, and both branches of the advanced
  pipeline -- and the raw half is chosen between two extractions after all of
  them have run. So the pairing was not broken by any write: every writer paired
  correctly at the time, and the choice then replaced the raw dict and left the
  transformed one from the first. A two-way design with an empty cell and
  scattered NaNs printed a Box-Cox column of 10 values against raw columns of 9,
  6 and 8, while the log reported the column dropped -- that was the guarded
  writer declining, after the unguarded one had already written. There is now
  one pairing rule in one place, used by all four writers, and the result is
  re-checked after the raw values are chosen, so a mismatch created downstream
  of every writer is caught too. Both keys are dropped rather than only the
  first, since the report falls back to the second when the first is absent.

### Testing

- A report may no longer show transformed values for a run that transformed
  nothing. Three separate things on a page claim a transformation -- the
  "Transformed value" column in the raw data vault, the transformed-scale means
  note, and the after-transformation diagnostic charts -- and each is gated
  differently, so a new check reads the finished page and requires all three to
  agree with the transformation the report declares, and requires the column to
  actually differ from the raw one. Built from a real defect that the existing
  tests could not have caught: they check the builders, and this is a
  disagreement between builders. Both spellings of "nothing was transformed"
  are covered, since the standard and correlation paths render different ones.

- A new check asks whether each printed row pairs a measurement with its own
  transformed value. It needs no knowledge of which transformation ran: log10,
  sqrt, Box-Cox at any lambda and arcsin-sqrt are all monotonically increasing,
  so within a group the ranking of the raw column must be the ranking of the
  transformed one. Where the badge names log10 there is a second check that
  reproduces the arithmetic: `log10` means `log10(v + shift)` with one shift
  across every group, so each row implies `10**t - raw` and one transformation
  means one implied shift. Rows that cannot be reproduced from the shift the
  rest of the table agrees on are the finding. On a real repeated-measures
  report carrying the pairing bug above, the two checks report 44 out-of-order
  pairs and 26 of 28 unreproducible rows.
- A printed tie is not a disagreement. The ordering check skipped ties on the
  raw side but not on the transformed one, and a compressing transform collapses
  distinct values onto a single six-digit cell -- a real Box-Cox report prints
  2.25381, 2.30519 and 27.2228 all as 30780, correctly ordered underneath. Ties
  are now skipped on both sides.
- The fuzzer draws every answer its dialogs offer, cancelling included. A sweep
  of the remaining stand-ins, prompted by the transformation-answer fix below,
  found four more: `mw_custom` (a real nonparametric post-hoc choice with its own
  Mann-Whitney branch) was never drawn; and the post-hoc, control-group and
  pair-selection dialogs could never be cancelled, so the guards behind them —
  including one whose own comment says "return None, never a silent groups[0].
  Every caller guards" — had nothing exercising them. Cancelling is drawn one
  time in six, so the paths behind the dialogs stay well covered too.
- The fuzzer answers the transformation dialog the way the dialog answers. It
  drew from `["log10", "sqrt", "box_cox", None]`, but the dialog offers `log10`,
  `boxcox` and `arcsin_sqrt` -- so two draws in four were values no user can
  produce, fell through every apply-branch and transformed nothing, and two of
  the three real transformations had never been exercised at all. Correcting the
  list immediately surfaced what had been hidden behind it: `arcsin_sqrt` opens
  a domain-declaration dialog that no stand-in answered, so the run built a
  QDialog with no QApplication and the process aborted -- six seeds in two
  hundred, on a path that had never once been taken. Both fixed; that dialog's
  three answers (proportion, percent, cancel) are now drawn per seed.
- A new `mild_skew` mutation makes data the transformation branch can actually
  repair. `heavy_skew` applies `exp(3v)`, which stays non-normal after log10, so
  the run falls to a rank test -- across 200 seeds not one `heavy_skew` case
  produced a transformed column. `exp(v)` is log-normal by construction.

- The report checks can now run against your own exports, not only generated
  ones. With `BIOMEDSTATX_SELFCHECK=1` set before launch, each exported report is
  read back and verified — every section rendered, the headline number on the
  page, compact letters drawn only from a complete comparison matrix and
  agreeing with the pairwise table, the group axis in ranked order, estimated
  p-values within their resolution, one font family throughout. Nothing appears
  when all of that holds; otherwise a `<report>_selfcheck.txt` lands beside the
  report with pass/fail/n-a and a count per check, carrying no values from the
  data. Off unless the variable is set, so an installed copy never runs it and
  pays nothing for it. One implementation
  (`src/export/report_selfcheck.py`), shared with the fuzzers.

- A third fuzzer covers the layer after export: `fuzzing/run_visual_fuzzer.py`
  opens each generated report in headless Chromium and uses the figure builder —
  switching plot types, changing controls, pressing presets, exporting SVG and
  PNG — with seven oracles on the rendered page plus the exported bytes. It
  catches what reading the HTML as text cannot: a script that fails to parse, a
  figure that draws no trace, a label outside its container, a Download that
  yields an empty file. `fuzzing/visual_selfcheck.py` proves each oracle can
  fail by breaking a real report in the way that oracle exists to catch.
- The import fuzzer writes wide-format files. Roughly one seed in three is now
  one subject per row and one column per condition, so the pivot path — the
  detection, the subject-name heuristic behind it and the melt — is exercised by
  a real file rather than only by its unit tests; every previous run reported
  `wide-pivoted 0`. Six oracles cover it, and three more mutations join the four:
  a BOM (what Excel writes whenever it saves a CSV), umlaut headers, and a blank
  subject cell, which has to be refused out loud.

- Two-factor designs are addressed by their cells, the way the window addresses
  them. The window builds the group labels as `FacA=A0, FacB=B0` and hands the
  analysis a group column of `__AUTO_GROUP__`; the fuzzer sent the first
  factor's levels and that factor as the group column, a shape no window can
  produce. The pipeline refuses it, and the refusal counted as success: every
  clean two-way seed came back blocked with "Group 'A0' has no usable numeric
  values", because a blocked result is a legitimate outcome for bad data. Two-way
  ANOVA therefore had no effective coverage for the whole life of this fuzzer
  while appearing in the coverage table as a design that ran, and mixed was
  covered over a partition the product never builds.
- The tiny-groups mutation deleted the column it grouped by. Grouping with
  `.apply` lets pandas consume the key into the index, and dropping the index
  then discards it, so on a two-factor design the mutation removed the first
  factor instead of shrinking the groups -- and what ran was a design whose
  context named a column the frame no longer had.
- The report self-check measures the log10 arithmetic against the digits the
  page actually printed. Five of eleven findings in a 2000-seed run were the
  check's own, not the product's: the tolerance was a fixed 1e-4 on the
  logarithm, as if every cell carried six decimals, but the report switches to
  four significant digits at large and small magnitudes -- so recomputing log10
  from `8.301e-05` and demanding agreement to 1e-4 measures the format rather
  than the arithmetic. The reconstructed shift carries the same error, divided
  by the shifted value, which is how a row whose shift nearly cancels its
  measurement produced a reported discrepancy of 8.69 that was entirely the
  truncated tail of the input. The tolerance is now derived per row from half a
  unit in the last printed place, propagated through the logarithm; a row whose
  tolerance exceeds 0.05 is counted undecidable rather than reported, because
  the page does not carry enough digits to say. Widened, not removed: the defect
  it exists for is a disagreement of whole orders of magnitude, and a test swaps
  two rows in the same coarse format and requires the check to still fire.
- A new check asks whether the post-hoc a report names is the one that ran.
  Every family the comparisons come from must appear in the name the page
  prints, and a control-referenced name may not sit on a complete all-pairs
  family -- Dunnett over k-1 comparisons and Tukey over k(k-1)/2 carry different
  corrections, so the name is a claim about which was applied. The heading is
  read from the finished page and the comparisons from the result, so the report
  is not compared with itself, and rows carrying no method of their own are not
  counted as evidence. It fires on 26 of 400 seeds. It immediately found two
  existing fixtures that built EMM/multivariate-t comparisons and rendered a
  Tukey heading over them -- self-contradictory, and never checked.
- The fuzzer grades a run against the effects it built the data with. The
  generator draws each design's effect size, zero included, and threw it away
  the moment the frame was built -- so the one dimension the fuzzer could have
  graded itself against was varied for its whole life and nothing looked. Two
  rates are now reported per run: how often a term built with no effect is
  called significant, which should be about alpha, and how often a real one is
  found. Counted per term and each term against the p-value for that same term:
  holding a built main effect against the headline measures the interaction for
  a mixed design, which those data are built without, and the first version of
  this reported 7% "power" that was really the interaction's type-I error.
  Mutated seeds, multi-dataset runs, blocked runs and designs that draw no
  effect are excluded rather than averaged in, and below 60 null terms the
  summary says the run is too small to tell rather than printing a number. One
  finding fell out of building it, recorded rather than acted on: LMM draws no
  effect at all, so every LMM seed is a pure null and that design's power has
  never been exercised.
- A run can skip the seeds it cannot learn from. The calibration only counts
  designs that draw an effect, so seven designs in ten were built, analysed,
  rendered and oracled before contributing nothing to it. `--designs` reads the
  design from the generator's first draw, without building anything, and never
  spawns a seed that cannot speak to the question. The peek and the build go
  through one draw, so the filter cannot end up selecting on a rule the
  generator has stopped following, and `--count` still counts seeds actually
  run, so every denominator in the summary keeps meaning what it says.
- The fuzz runner writes its report as the run goes and records every run. A
  2000-seed run killed near its end left nothing behind, having spent forty
  minutes on a report written only after the last seed; it is now flushed every
  hundred seeds through a temporary file, and carries a "complete" flag so a
  partial report cannot be read as a finished one. Each run also appends a line
  to a history file, with the finding rate beside the number of oracles and the
  design filter -- the easiest way to make a discovery rate fall is to stop
  looking, and a rate that drops while the oracle count drops with it says
  nothing about the product.
- A run can skip the designs it cannot learn from. The calibration only counts
  designs that draw an effect, so seven designs in ten were built, analysed,
  rendered and oracled before contributing nothing to it. `--designs` reads the
  design from the generator's first draw, without building anything, and never
  spawns a seed that cannot answer the question; the peek and the build go
  through one draw, so the filter cannot select on a rule the generator has
  stopped following.
- A repeated-measures case claims only the term it is analysed on. Surfaced by
  the calibration's own list of built-but-never-reported terms, which is printed
  for exactly this reason: the repeated-measures and mixed designs are built
  from one branch, so both carried a between-factor effect, and a
  repeated-measures analysis never sees it.
- The fuzz runner runs seeds side by side. Each seed already ran in its own
  process with its own generator, temporary directory and report, and shared
  nothing with the others -- the runner simply waited for each before starting
  the next, which measured out as one worker at 98% of a single core on a
  14-core machine, or 7% of it. `--jobs` defaults to cores minus two, capped at
  eight so every worker keeps a core of its own: the per-seed budget is wall
  clock, and oversubscribing would turn a queued seed into a TIMEOUT finding
  that says nothing about the product. A/B'd on the same twelve seeds rather
  than assumed -- 19.3 s to 4.2 s, with every field of every record identical.
- A main effect is not judged against alpha when the design carries an
  interaction. The generator records the coefficient it built with; the ANOVA
  tests the marginal means, which an interaction moves even where the
  coefficient is zero. Over 2500 seeds those terms were called significant 37%
  of the time against 6.8% on a purely null design, and that difference was the
  whole of what looked like an inflated two-way error rate. They are skipped in
  both directions, and the count is printed so the narrowing stays visible.

### Reporting

- Transformation display is gated on an actual change in the data, not on the
  presence of a transformation label. Four paths previously showed a
  "Transformed" column, transformed-data normality metrics, or a "transformed"
  decision-tree node whenever a transformation was *named* — even when the
  chosen transformation left the values unchanged (e.g. "None" / "No further") —
  and now compare values and surface the transformation only when it moved the
  data. Covers the assumption summary (`stats_functions.py`), the autopilot
  assumptions formatter (`statistical_analyzer_autopilot_pipeline.py`), the
  report summaries (`report_summaries.py`), and the decision-tree node label
  (`decisiontreevisualizer.py`). A frozen audit test pins the four paths.

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
- The report's interactive decision tree now fits the active path into the frame
  on load instead of opening zoomed into a corner, with a tighter layout, larger
  legible node labels, correctly sized arrowheads, and a slower path animation.
  Node widths are measured from the actual rendered text (Canvas / font metrics)
  rather than a character-count guess, so labels no longer overflow their boxes —
  including in Safari, where the previous offscreen-SVG measurement returned zero.
- The in-app Decision Tree Dashboard (Correlation, Regression, ANCOVA, LMM, and
  the other clinical trees) no longer renders with overlapping, unreadable nodes:
  spacing scales to the measured node width within a width budget, and the
  highlighted path no longer routes an arrow through the middle of a node.
- The post-analysis confetti burst is smoother. It now animates against real
  elapsed time instead of per frame, so it keeps a steady speed and a constant
  duration even on the first analysis, when the busy main thread previously made
  it crawl in slow motion; the burst is also deferred until the result render
  finishes so it no longer stutters on launch.

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
- Added parsing-core coverage for the cell-range selector, and a regression test
  that forces the statsmodels ANOVA fallback (`pingouin` made unavailable) so the
  degrees-of-freedom key fix cannot silently regress behind the primary engine.
- Ended global dialog-state leaks across the test suite (11+ files): bare
  `QDialog` / `UIDialogManager` patches that stayed live past their own test were
  converted to function-scoped `monkeypatch` fixtures with teardown, so a
  cancelled or suppressed dialog can no longer leak into an unrelated test. The
  suite is now order-independent, verified under `pytest-randomly` (seeds
  1 / 42 / 1337 produce identical results).
- Corrected the R validation oracle for the independent t-test effect size
  (`validation/r_templates/indep_ttest.R`). It used Student's t plus a classic
  Cohen's d; since the Welch default the app computes pooled-SD Hedges' g, so the
  oracle now uses `t.test(var.equal=FALSE)` with
  `effsize::cohen.d(hedges.correction=TRUE)` — an independent implementation of
  the same J correction. No production code changed; the app was already correct.
- Updated several stale test expectations to the current contracts — the
  post-hoc / transformation cancel contract (`{"cancelled": True}` abort with no
  report written), the plot symbol-cycle order (circle-first), and the
  Freedman-Lane permutation test name on the nonparametric two-way path. No
  production code changed.
- Made the R p-value cross-validation tolerance scale-aware
  (`validation/test_all_paths.py`): a single fixed absolute tolerance was
  meaningless across p-value magnitudes, so it now uses `numpy.isclose` semantics
  — a per-family absolute floor plus a per-family relative term. No production
  code changed.

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

### Dialog cancel behaviour (behavioral)

Cancelling a mid-analysis dialog now consistently aborts the whole analysis — no
results, no report file, no success animation, the app returns to the mapping
state — instead of silently substituting a value the user never chose. Several
dialogs previously turned Cancel into a hidden default:

- **Transformation dialog.** Cancel silently applied a log10 transform and then
  labelled the report "Transformation: log10" — a transform the user had
  declined, mislabelled on the actually-computed result. Cancel now aborts. A new
  explicit **Continue without transformation (use non-parametric test)** option
  provides the deliberate no-transform path that Cancel used to stand in for.
- **Arcsin data-domain dialog.** Cancel dropped the arcsin transform and
  continued; it now aborts, so Cancel means the same thing everywhere. (To run
  non-parametric without a transform, pick the explicit option above.)
- **Post-hoc selection dialog** (parametric, non-parametric, and the advanced
  RM/Mixed/Two-Way engine). Cancel ran the default method (Games-Howell or Dunn)
  and celebrated a result the user had just cancelled; it now aborts. The dialog
  only appears after a significant omnibus, but Cancel there discards that run
  rather than proceeding without the post-hoc.
- **Comparison-selection dialog** (advanced custom pairs). Cancel silently ran
  every pairwise comparison; it now aborts.
- **Control-group dialog.** Cancel silently ran Dunnett against the first group.
  It no longer does: cancelling the control selection falls back to Games-Howell
  (the heteroscedasticity-robust all-pairs test) rather than testing against an
  arbitrary control the user never chose. (Bringing this dialog fully onto the
  "Cancel aborts" line, via an explicit "no control group" option, is a tracked
  follow-up.)

A dialog that genuinely cannot be shown (an infrastructure failure, not a user
Cancel) still continues rather than aborting, but now logs a warning so it is not
invisible. A cancelled analysis leaves no corrupted state: the next run — even
after two cancels in a row — proceeds cleanly.

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
  the test is about. Cancelling the domain prompt now aborts the analysis
  (see **Dialog cancel behaviour** above) rather than applying an unchecked
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
- Mixed ANOVA is now selected when a subject column is mapped to a two-way
  design. The design previously ran as an independent Two-Way ANOVA even when the
  same subjects were measured across a factor; the upgrade to a mixed model is
  enforced, its trigger condition tightened, and a warning is shown when the
  chosen model changes as a result.
- A binary outcome coded as anything other than 0/1 (for example 1/2) is no
  longer silently analysed as a Pearson correlation. The autopilot now treats any
  two-value numeric outcome as a binary candidate and, when the coding is
  ambiguous, asks whether to run logistic regression or keep it continuous —
  showing the two actual values so a genuine two-point continuous measure is
  obvious — instead of shipping a plausible-but-wrong correlation. A two-value
  outcome coded 0/1 still routes straight to logistic with no prompt.
- A two-factor design whose subject-ID column is present in the sheet but left
  unmapped now warns that it is running as a between-subjects Two-Way ANOVA and
  points to the subject column to map for a Mixed ANOVA, instead of silently
  ignoring the repeated-measures structure and reporting between-subjects
  p-values for a within-subject factor.
- Linear mixed models and logistic regression no longer drop unbalanced subjects.
  These likelihood-based models were routed through the same two-group paired
  inner-join as the t-tests, which silently discarded any subject without a
  complete pair; they now keep every usable observation, since the model handles
  the imbalance directly.
- The statsmodels ANOVA fallback (used when `pingouin` is unavailable) read its
  degrees of freedom from a mistyped column key — `"d"` instead of `"df"` — and
  raised `KeyError` the moment it was reached, across the Repeated-Measures,
  Mixed, and Two-Way paths. All sites now read the correct column through one
  shared helper, so the key lives in a single place.
- Dunn's test: corrected the effect-size and test-statistic computation.

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
- The bundled data template now teaches subject-ID pairing and includes the
  previously missing designs, so repeated-measures and mixed layouts can be built
  from the template directly.

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
- A subject carrying two different between-group values (a data-entry error) is
  rejected with a clear "each subject must belong to one between group" error when
  a Mixed ANOVA averages technical replicates, instead of being silently split
  into two pseudo-subjects. This is the same check the post-hoc path already
  applied, now enforced at the earlier averaging step as well.

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
- Exported reports no longer leave templated fields blank: previously missing
  report text is filled in, and NA values are rendered explicitly instead of as
  empty cells.
- The technical-replicate notice (measurements averaged to the subject level
  before a Repeated-Measures or Mixed ANOVA) reaches the report again. The notice
  had been written to a result key nothing rendered; the Mixed path additionally
  dropped the between factor while averaging and raised before the notice was
  written. Both are fixed.
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
