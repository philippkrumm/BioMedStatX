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
   shape. Four mutations, drawn from what a lab actually sends: **German number
   format** (`;` separated, comma decimal, dot thousands, and values large enough
   that the thousands separator is exercised at all), **notes above the header**
   (an export with an experiment title and an operator line before the table),
   **merged group cells** (a genuinely merged range, so the rows beneath the
   label are empty in the file), and **the wrong number format declared** — the
   realistic mistake, since there is deliberately no autodetect and the
   declaration is the user's.
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

The fifth of those found a hole in the oracle rather than in the product: an
empty measurement bucket was being counted as a refusal, so an app claiming to
be ready with nothing assigned walked straight past. `file_loaded`,
`factor_reaches_bucket` and `measurement_is_not_a_subject` are covered by the
unit tests but have not been put in front of a real mutation yet.

## Reading the summary

```
=== IMPORT FUZZ SUMMARY ===
  OK                 120
  files the app must parse exactly : 64
  files it may only refuse visibly : 56
  loaded 109  |  measurement column numeric 64  |  mapping ready 64  |  wide-pivoted 0
```

The two middle lines are the ones that stop a green run from being mistaken for
a tested one. The value-level checks live entirely on the first group; a run made
only of files the app may legitimately refuse would be green without having
verified a single number. Seeds that fired **no** oracle at all are counted and
named — the first run had one, a file that failed to load so early that nothing
else applied, and it counted as OK.

`wide-pivoted 0` is an honest gap, not a result: the generator does not yet
write wide-format files, so `_detect_wide_format` and the melt behind it are
untested here.

## Not covered

The layer after export — what the rendered page actually looks like — is
untouched by either fuzzer. No browser, no pixels. A clipped axis label or a
broken plot layout is invisible to both.
