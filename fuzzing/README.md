# Fuzzing BioMedStatX

Four hundred imaginary users, each with their own badly-shaped spreadsheet.
Every case is a pure function of an integer seed, so anything the fuzzer finds
is reproducible by re-running that one seed.

```bash
python -m fuzzing.run_fuzzer --count 200
```

```bash
python -m fuzzing._worker 142
```

## What a run does

1. `generators.build_case(seed)` picks one of ten designs (`oneway`, `ttest`,
   `rm_anova`, `two_way_anova`, `mixed_anova`, `ancova`, `correlation`,
   `regression`, `firth_logistic`, `lmm`) and layers zero to three of nineteen
   mutations on it — scattered NaNs, infinities, zero-variance groups, unicode
   and control characters in labels, comma decimals, dropped factor cells,
   missing subject×time observations, rank ties, collinearity, skew.
2. `_worker` runs that one case in its own subprocess. Isolation is the point: a
   C-level fault inside NumPy or SciPy kills the child and leaves the
   orchestrator to record the seed. Every interactive dialog is answered by a
   seed-determined choice drawn from the options the real dialog offers, so a
   fuzzed run stays a run a user could have had.
3. `oracles.check_result` inspects the result dict for statistically impossible
   answers — non-positive degrees of freedom, p outside [0, 1], an effect size
   outside its own range, a huge F with a large p.
4. `html_oracles` reads the **exported HTML** back off disk before the working
   directory is removed, and checks the invariants that only exist in the
   artefact. Every file the run wrote is checked, not just the first: a
   multi-dataset run writes one report per dataset plus a combined overview,
   and the overview has its own template and its own oracles.

Step 4 is the one that closes the loop. The report is the product; a result dict
that is perfectly sound can still reach the reader as a chart claiming something
the test never found.

## Where these checks live

The report/export half of them moved to `src/export/report_selfcheck.py`, and
`html_oracles` imports them from there rather than keeping a copy — two
implementations of one check drifting apart is the failure this repository has
paid for more than once.

The export path can run the same checks on the reports you export yourself:
set `BIOMEDSTATX_SELFCHECK=1` before launching and a `<report>_selfcheck.txt`
appears beside any report where one does not pass. It is off unless that
variable is set, and an installed copy never runs it — see BUILD.md,
"Developer switches". The point of switching it on is the data: the fuzzers only
ever invent theirs.

What stayed here is what only a fuzz run needs: `designer_when_plottable` and
`paired_line_gate` (both read the fuzzer's notion of a result), the
multi-dataset overview, and the violation plumbing the orchestrator reports on.
The other two fuzzers stay out of the export path entirely and on purpose — the
visual one needs a browser, which is not a per-export cost a researcher's
desktop app should carry, and the import one inspects live window state, which
would mean reaching into the import path rather than reading its result.

## The report oracles

| Oracle | What it refuses to accept |
| --- | --- |
| `payloads_parse` | a `pd-data-*` block that is not valid JSON, or one missing where the figure builder exists |
| `designer_when_plottable` | a result with drawable groups whose report carries no figure builder |
| `sections_present` | a missing section, or comparisons in the result with no pairwise table in the file |
| `result_number_rendered` | a finite p-value that reaches the reader as a placeholder |
| `p_precision_capped` | a simulated p-value printed with precision the method never had |
| `letters_gate_complete` | compact letters drawn from fewer than all k(k−1)/2 comparisons |
| `letters_match_pairs` | groups that differ sharing a letter, or equal groups not sharing one |
| `brackets_have_no_letters` | letters left on a chart whose mode says brackets |
| `paired_line_gate` | a subject-line verdict that drifted from the rule, a refusal without a reason, lines across the cells of a mixed design |
| `axis_order_ranked` | an axis in label order rather than ranked order |
| `one_plot_font` | a second font family in the figures |

For the combined overview of a multi-dataset run:

| Oracle | What it refuses to accept |
| --- | --- |
| `multi_lists_datasets` | an analysed dataset absent from the overview |
| `multi_count_matches` | a headline count that disagrees with the cards behind it |
| `multi_p_values_rendered` | a dataset p-value, raw or FDR-adjusted, that never reaches the reader |
| `multi_failures_surfaced` | a dataset that failed and vanished from the report |

`tests/test_fuzz_html_oracles.py` breaks each invariant in an otherwise valid
report and demands the matching oracle says so. An oracle that cannot fail
reports coverage it does not have.

## Reading the summary

The run prints outcomes **and** coverage:

```
  OK                 300
  reports written    160/300
--- coverage ---
  designs:   ancova=40, correlation=25, firth_logistic=29, lmm=33, ...
  mutations: all_constant=19, collinear_covariate=17, comma_decimals=21, ...
  oracles:   axis_order_ranked=113, letters_gate_complete=12, p_precision_capped=1, ...
  post-hoc:  EMM pairwise contrasts (Holm-Bonferroni)=52, Dunnett Test=2, ...
  results with an estimated p-value: 3
```

"300 OK" on its own means nothing: a run can be green because every seed took
the same branch. What matters is the rest — a design, mutation or oracle the
run never touched is called out by name under `untouched`, because an oracle
that never fired contributed only the appearance of safety.

`reports written` is low by design rather than by fault: most heavily mutated
cases are stopped at the data-quality gate before any report exists, which is
the correct behaviour and is why the count is reported next to the outcomes.

`with an actual figure` is the line to read next to it. A blocked run still
writes a report -- honestly, full of placeholders -- and almost every oracle
then has nothing to say, so it passes without being tested. Counting figures and
empty-state blocks separates "thin because it was rightly gated" from "thin
because the checks did not apply", which the firing count on its own cannot.

## Findings

Reports belonging to a finding are copied to `fuzzing/failures/seed_<n>.html`
(git-ignored) so the artefact can be opened, not only the seed re-run. The full
record, including the per-seed design, mutations and which oracles fired, is
written to `fuzzing/fuzz_report.json`.

---

# Fuzzing the import and mapping layers

The fuzzer above starts from a clean DataFrame already in memory:
`analysis_context["injected_df"]` short-circuits file reading, and the analysis
context it hands over is one it built itself, which makes a wrong mapping
impossible by construction. So it covers exactly one span — DataFrame in, HTML
out — and says nothing about the two layers on either side of it.

This second fuzzer covers the layer before: an actual `.csv` or `.xlsx` written
to disk, opened by the actual window.

```bash
python -m fuzzing.run_import_fuzzer --count 200
```

```bash
python -m fuzzing._import_worker 42
```

## What a run does

1. `import_generators.build_case(seed, dir)` writes a real file and records the
   ground truth of what it wrote — the values, the group label of every row, the
   shape. Roughly one seed in three is a **wide** file (one subject per row, one
   column per condition) rather than long, because the pivot path had no
   file-level coverage at all. Seven mutations, drawn from what a lab actually
   sends: **German number format** (`;` separated, comma decimal, dot thousands, and values large enough
   that the thousands separator is exercised at all), **notes above the header**
   (an export with an experiment title and an operator line before the table),
   **merged group cells** (a genuinely merged range, so the rows beneath the
   label are empty in the file), and **the wrong number format declared** — the
   realistic mistake, since there is deliberately no autodetect and the
   declaration is the user's. Then the second wave: a **BOM** (what Excel writes
   whenever it saves a CSV, landing on the first header cell), **umlaut
   headers** (`Größe`, `Behandlung`, `Präparat ID` — what a German lab names its
   columns), and, for wide files only, a **blank subject cell**. The first two
   deliberately do *not* relax `expect_faithful_read`: the app is expected to
   handle both, so a failure there is a finding rather than a legitimate
   refusal. The third must be refused out loud, because rows without an ID drop
   silently out of every groupby that decides repeated-measures structure.
2. `_import_worker` builds the real `StatisticalAnalyzerApp` headless and calls
   `load_file()` on it. Exactly two things are stubbed, and both are things a
   *user* answers rather than things the app computes: the CSV number-format
   dialog, whose fuzzed answer is the case's declared format, and the message
   boxes — which are recorded rather than swallowed, because an app that gives
   up on a file without telling anyone is the failure this layer hides best.
3. `import_oracles.check_import` compares the window's state against the ground
   truth. Each violation is tagged with the oracle that raised it.

## The oracles

The split that governs all of them is `expect_faithful_read`: header on the
first row, and, for a CSV, the declared format is the one the file was written
in. When it holds, the read must reproduce the file exactly. When it does not,
the app is *allowed* to misread — what it may not do is present a finished
mapping built on a misread.

| oracle | precondition | asserts |
|---|---|---|
| `file_loaded` | faithful read expected | the file loaded and has rows |
| `load_failure_is_announced` | nothing loaded | the user was actually told |
| `shape_survives` | faithful, not pivoted | row and column counts match the file |
| `dv_is_numeric` | faithful | a column of numbers arrived as numbers |
| `values_survive` | faithful, numeric, not pivoted | the numbers are the numbers that were written |
| `levels_survive` | faithful, not pivoted | the group labels are the labels that were written |
| `no_phantom_levels` | loaded | a blank cell did not become a group |
| `broken_import_fails_visibly` | faithful read **not** expected | a misread never reaches a ready-looking mapping |
| `dv_reaches_bucket` | faithful, numeric | the measurement column is the one the mapping chose |
| `factor_reaches_bucket` | faithful, ≥2 levels | the group column landed in Factor 1 |
| `measurement_is_not_a_subject` | faithful | a column of measurements was not filed as a subject ID |

Wide files -- one subject per row, one column per condition -- get their own
five, because the long ground truth does not describe the frame after a melt:

| oracle | precondition | asserts |
|---|---|---|
| `wide_is_pivoted` | loaded | a wide file was melted, **and** a long one was not |
| `pivot_keeps_every_value` | pivoted | the melt moved every measurement and changed none |
| `pivot_keeps_every_subject` | pivoted | every subject survived, once per condition |
| `conditions_are_the_columns` | pivoted | the Condition levels are the file's value-column headers |
| `wide_feedback_matches_design` | pivoted | the line explaining the pivot names the design that was built |
| `missing_subject_is_refused` | blank subject cell | the file was declined out loud, not read in part |

`wide_is_pivoted` checks both directions on purpose. A wide file left unpivoted
reaches the mapping as one column per condition, so the user is asked to pick a
"measurement" from four equally plausible ones; a long file wrongly pivoted
invents a subject structure the data never had.

`values_survive` is the one worth understanding. The failure it exists for is
not a crash and not a warning: a thousands separator read as a decimal point
divides every number by a thousand and leaves a column that is entirely
plausible and entirely wrong.

## Proving the oracles can fail

`tests/test_import_oracles.py` breaks one invariant per test against synthetic
state. That the state matches the real window was established by mutating real
`src/` code and re-running affected seeds — six deliberate breaks, each caught
by the oracle named for it:

| break | caught by |
|---|---|
| `read_csv_localized` drops `thousands` | `dv_is_numeric` |
| the loader quietly `dropna()`s the frame | `shape_survives`, `values_survive` |
| `"id"` goes back to a substring match | `dv_reaches_bucket` |
| a failed load no longer shows its message box | `load_failure_is_announced` |
| the mapping goes live with nothing assigned | `broken_import_fails_visibly` |
| blank labels stringified into the group `nan` | `levels_survive`, `no_phantom_levels` |
| `_detect_wide_format` always declines | `wide_is_pivoted` |
| the melt drops a row | `pivot_keeps_every_value` |
| the missing-subject-ID guard removed | `missing_subject_is_refused` |
| the pivot notice back to "paired t-test design" | `wide_feedback_matches_design` |

The fifth of those found a hole in the oracle rather than in the product: an
empty measurement bucket was being counted as a refusal, so an app claiming to
be ready with nothing assigned walked straight past. `file_loaded`,
`factor_reaches_bucket` and `measurement_is_not_a_subject` are covered by the
unit tests but have not been put in front of a real mutation yet.

## Reading the summary

```
=== IMPORT FUZZ SUMMARY ===
  OK                 250
  files the app must parse exactly : 180
  files it may only refuse visibly : 70
  loaded 221  |  measurement column numeric 126  |  mapping ready 172  |  wide-pivoted 46
```

The two middle lines are the ones that stop a green run from being mistaken for
a tested one. The value-level checks live entirely on the first group; a run made
only of files the app may legitimately refuse would be green without having
verified a single number. Seeds that fired **no** oracle at all are counted and
named — the first run had one, a file that failed to load so early that nothing
else applied, and it counted as OK.

`wide-pivoted` is the line that used to read 0. The generator now writes wide
files for roughly one seed in three, so the detector, the subject heuristic
behind it and the melt are exercised by the file rather than only by their unit
tests -- and a wide seed runs the whole chain, pivot included, into a real
repeated-measures analysis and its report.

---

# Fuzzing the rendered figure

Both fuzzers above stop where the HTML is written. Everything a reader then
does happens in a browser: open the file, look at the chart, switch it to a
violin, rename the axis, move the legend, press **Download**. None of that was
checked by anything, and it is where two of the four symptoms this work started
from live — "der Plot sieht komisch aus" and "die Formatierung vom Plot
funktioniert nicht".

Reading the file as text cannot close that gap. A report whose
`plot_designer.js` fails to parse still contains a perfectly valid Plotly
payload and a perfectly valid `<script>` tag; the figure builder is simply
dead. That shipped once, in `HEAD` and in `dist/`, and was found by clicking.

```bash
python -m fuzzing.run_visual_fuzzer --count 60
```

```bash
python -m fuzzing._visual_worker 42
```

Needs Playwright and its Chromium, which are **development-only** and
deliberately absent from `requirements.txt` — that file feeds the PyInstaller
build, and a browser toolchain has no business in a 294 MB app bundle:

```bash
pip install playwright && python -m playwright install chromium
```

## What a run does

1. The seed builds a **real report** through the same case generator the
   analysis fuzzer uses, so the figure under test is one the product actually
   produced rather than a fixture.
2. Headless Chromium opens it over `file://` — the Plotly bundle is inlined, so
   nothing is fetched and the page behaves exactly as it does on a reader's
   machine.
3. `visual_generators.build_plan(seed, surface)` decides what this user does.
   The control list is **discovered from the page**, not hardcoded: every input
   and select inside the designer panel, with its type, its option list, its
   bounds and the tab that owns it. A control added to `plot_designer.html` is
   fuzzed the day it appears. What the generator owns is the choice — which
   controls, and with what value: a label long enough to need a margin, an
   empty one, `α-Synuclein (µg·mL⁻¹)`, `$\Delta$F/F$_0$`, a y-minimum above the
   y-maximum, an out-of-range spin value.
4. Each action is performed the way a user must perform it — the owning tab is
   brought to the front first — and then waited on precisely: the worker counts
   `plotly_afterplot` events rather than sleeping.
5. After every stage the page is snapshotted and the oracles run. At the end
   both **Download** buttons are pressed and the bytes that come back are
   checked.

## The oracles

| Oracle | What it refuses to accept |
| --- | --- |
| `no_script_error` | any uncaught JS error or console error, at load or after any action |
| `figures_render` | a chart that declares data but draws no trace, no SVG, or has zero size |
| `designer_live_when_plottable` | a report with plot data whose figure builder is missing or empty |
| `labels_not_clipped` | a tick label, axis title, legend entry or annotation sticking out of the plot container |
| `significance_matches_mode` | brackets drawn while the control says letters, or either drawn while it says none |
| `designer_keeps_report_order` | a builder that re-sorts the ranked group order it was handed |
| `download_svg` / `download_png` | a download that fails, produces no file, is not really that format, or is an empty canvas |

`designer_keeps_report_order` has one deliberate exception. The builder offers
move-up / move-down per group, so once the user has pressed one, an axis that
differs from the report *is* the feature working; from that point the oracle
checks membership only — reordering must not lose a group. Without that
exception the oracle would have been asserting that a button it just pressed
should have had no effect, which is how it reported its first three findings.

## Proving the oracles can fail

```bash
python -m fuzzing.visual_selfcheck
```

Each oracle gets a negative control: a deliberate break in a real report that
it is supposed to catch. Two of them are only correct because a weaker version
was tried first and turned out to be no mutation at all:

* turning `automargin` off clips **nothing** — Plotly's own `autoexpand` still
  fits the tick labels — so the mutation restores the bug that actually
  shipped: a fixed pixel margin that cannot grow;
* scrambling the `pd-data-order` payload changes nothing observable, because
  the designer reads that payload. Both sides move together, which is the
  chained-to-its-own-source class exactly. The honest mutation makes the
  renderer re-sort while the payload stays ranked — and then the axis comes out
  `Dose1, Dose10, Dose2, Vehicle`, which is the alphabetical bug this repo
  fixed centrally a day earlier, now caught at the browser end too.

A third was a dead mutation: relaxing the bracket builder's own gate proves
nothing, because in letters mode the dispatcher returns before that builder is
ever called. Both the dispatch and the gate have to move.

## Reading the summary

```
=== VISUAL FUZZ SUMMARY ===
  OK                 120
  seeds that produced a report: 66/120   driven in the browser: 66   stages checked: 669
--- coverage ---
  plot types: Bar=13, Box=15, Estimation=13, Forest=14, Raincloud=19, Violin=15
  oracles:    designer_keeps_report_order=44, figures_render=57, no_script_error=66, ...
  significance on screen: brackets=10, brackets(empty)=29, letters=6, none=4
  actions:    download:svg=44, download:png=44, preset_poster=19, group_down=13, ...
  designer said:
        19x  Reference lines are disabled for Raincloud layout.
        19x  No plottable data found.
         1x  ... Log scale ignored: log requires positive values.
```

`designer said` is worth reading next to the outcomes: those are the product's
own refusals, counted. "No plottable data found." 19 times is a forest or
estimation plot correctly declining a design that has no effect sizes -- and
after the fix in this batch it also means the canvas was cleared rather than
left showing the previous chart.

`significance on screen` is the line that keeps a green run honest. Firing
counts cannot answer it: the mode oracle fires just as happily on a figure that
never left brackets, so "letters were never rendered" is precisely the kind of
gap a passing run hides. Modes are recorded as what actually reached the
canvas — `letters(empty)` when the select says letters and no letter was drawn.

## Not covered

Still nothing looks at **pixels**. Every check here reads the DOM and the Plotly
layout, so it can say a label overflows its container but not that two labels
overlap each other, that a colour is unreadable, or that a figure is ugly. No
screenshot is taken and none is compared.

The decision-tree viewer, the modal image viewer and the report's own toolbar
are loaded and error-checked but never driven — only the figure builder is.
