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
4. `html_oracles.check_report` reads the **exported HTML** back off disk before
   the working directory is removed, and checks the invariants that only exist
   in the artefact.

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

## Findings

Reports belonging to a finding are copied to `fuzzing/failures/seed_<n>.html`
(git-ignored) so the artefact can be opened, not only the seed re-run. The full
record, including the per-seed design, mutations and which oracles fired, is
written to `fuzzing/fuzz_report.json`.
