# AUDIT: BioMedStatX @ b16cf24 — Specialized Statistical Models & Post-hoc / Outlier Engines

**Scope:** `src/analysis/correlation_models.py`, `posthoc_core.py`, `nonparametricanovas.py`,
`outlier_core.py`, `stats_functions.py`, `emm_posthoc.py`, `effect_sizes.py`.

**Verdict.** No live emergency. The core math (Games-Howell, Dunnett via `scipy.stats.dunnett`,
the hand-rolled Holm correction, EMM+multivariate-t Dunnett, effect-size canonicalization) is
correct and in several places demonstrably well-engineered (documented deviations from naive
approaches, honest relabeling of approximations). I found one genuine "computed-but-mislabeled"
correctness bug in the same family as the session's prior findings (a live post-hoc branch reports
"Tukey HSD" while actually running Holm-Šidák), one dead duplicate-dict-key bug that silently
discards a computed value, one substring-match control-group bug in dead code (safe only because
unreachable), and confirmed the `DataImporter` dead-code claim in this project's CLAUDE.md is
still accurate for the shipped autopilot path. No instance of the exact 3-times-seen "key written
under name A, read under name B" bug recurred in this batch as a *live* defect — the closest
analogue (`primary_effect` missing `F`/`df1`/`df2`) is a redundant/cosmetic summary line, not a
gating value, because the same numbers are correctly rendered from the `factors` list one branch
earlier in the same function.

## What I mechanically verified (not eyeballed)

| Check | Method | Result |
|---|---|---|
| All 7 files read in full (1163+1813+1025+525+782+228+184 = 5720 lines) | `Read` tool, full-file | Done |
| `DataImporter` dead-code claim | `git grep` for all callers of `.import_data(` and `DataImporter` | Confirmed dead in the shipped autopilot path (see SM6) |
| Hand-rolled Holm correction (`nonparametricanovas.py::_holm_correct`) vs `statsmodels.stats.multitest.multipletests(method='holm')` | 2000-trial randomized numeric comparison, `atol=1e-9` | Exact match, all trials |
| `effect_size_type` strings used by callers (`hedges_g`, `cohen_d`, `cohen_d_mixed`, `cohen_d_rm`, `rank_biserial_r`) vs `effect_sizes.py::canonicalize()` | Direct Python invocation | All canonicalize to the intended `EffectSizeKind`; none silently return `None` |
| Every dict key set in a literal `return {...}` across the 5 files with plain-dict result builders (regex extraction of `^"key":` patterns) | Python script, 178 unique keys extracted | Cross-checked against `git grep` in `src/export/*.py` |
| `primary_effect` / `interpretation_order` / `interaction_significant` read sites | `git grep` across `*.py` | `primary_effect` read once (report_stat_rows.py:523, cosmetic-only); `interpretation_order` and `interaction_significant` never read anywhere |
| `_perform_test_legacy` (Mixed ANOVA, substring control-group match) call sites | `git grep -n "_perform_test_legacy"` | Zero callers — confirmed dead |
| HC3 robust-covariance propagation in `SimpleLinearRegressionModel.fit` | Read `self.result` reassignment at line 589 + downstream use in `as_results_dict` | Correctly propagates — not a repeat of the Mixed-ANOVA sphericity gating bug |
| Games-Howell family size `k` | Read: computed once from `comparable` groups, reused for every pairwise `studentized_range` call | Correct FWER family size, no per-pair drift |
| `CorrelationModel` auto-method docstring claim ("branch never reads Shapiro-Wilk p-value") | Read `fit()` lines 274–342: threshold branch keys off skew/kurtosis only | Docstring accurate |

## Findings — severity ranked

### HIGH

**SM1 — Mixed-ANOVA post-hoc "Tukey" branch silently runs Holm-Šidák and reports it as Tukey HSD.**
`src/analysis/posthoc_core.py:866-869`, inside the **live** `MixedAnovaPostHocAnalyzer.perform_test`
(dispatched via `PostHocFactory.perform_posthoc_for_anova`, confirmed by `git grep` — this is not
the dead `_perform_test_legacy`):
```python
if method.lower() == 'tukey':
    # For Tukey, we'll use a different approach
    correction_method = "Tukey HSD"
    reject, pvals_corr, _, _ = multipletests(pvals, alpha=alpha, method='holm-sidak')  # Fallback
```
No actual Tukey studentized-range calculation happens here — `pvals_corr` is Holm-Šidák-corrected,
but `correction_method` (which flows into every `PostHocAnalyzer.add_comparison(..., correction_method=correction_method)`
call at line 962 and into the exported `test` / `correction` fields a user reads in the HTML report)
says "Tukey HSD". A user selecting "Tukey" for a Mixed ANOVA post-hoc gets Holm-Šidák p-values
under a Tukey label — the two methods produce different p-values and a different implied
correction strength, so this misrepresents which family-wise procedure actually gated the
reported significance. Contrast with the sibling `RMAnovaPostHocAnalyzer.perform_test` (same file,
~line 1143), which *does* attempt the real Tukey studentized-range calculation via
`_tukey_p_value` and only falls back to Bonferroni-with-relabeling on failure — so the codebase
knows how to do this correctly elsewhere and only this one branch skipped it.
**Impact:** every Mixed-ANOVA "Tukey" post-hoc result in the shipped app is actually Holm-Šidák,
mislabeled — a reproducibility/citation-correctness defect a domain reviewer would catch.
**Fix:** either implement the real Tukey studentized-range correction here (mirroring
`RMAnovaPostHocAnalyzer._tukey_p_value`, which already handles the mixed-design interaction-group
count), or rename `correction_method` to `"Holm-Šidák (Tukey unavailable for mixed interaction groups)"` matching the honest-relabeling pattern already used in the `dunnett` branches of the same file (e.g. line 538: `"Dunnett-type (Holm-adjusted, mixed design)"`).

### MEDIUM

**SM2 — Duplicate dict key silently discards a computed `both_normal` value in `CorrelationModel`.**
`src/analysis/correlation_models.py:329-342`, the non-transform branch of `_normality_check`:
```python
self._normality_check = {
    ...
    "shapiro_both_normal": both_normal_sw,
    "both_normal": self._method_used == 'pearson',   # line 338 — discarded
    x_col: {...}, y_col: {...},
    "both_normal": both_normal_sw,                    # line 341 — wins
}
```
Python dict literals keep only the last assignment for a repeated key, so line 338's value
(whether the auto-selection actually picked Pearson, i.e. the skew/kurtosis-threshold decision
the docstring says is authoritative) is dead code, silently replaced by the raw Shapiro-Wilk
"both p > alpha" result. Per the class's own docstring, method selection does **not** use
Shapiro-Wilk — but the exported `both_normal` field reports the Shapiro result anyway, which
could read as "why Pearson/Spearman was chosen" to a report consumer even though it isn't the
actual criterion. Currently latent: `git grep` confirms no exporter reads the top-level
`both_normal` key (only the per-variable `x_col`/`y_col` sub-dicts at `report_summaries.py:166`
are read), so no visible report is wrong today — but this is exactly the write/read key-mismatch
shape the task asked to hunt for, one edit away from becoming visible (e.g. if a future export
change reads `both_normal` expecting the method-selection criterion).
**Impact:** dead computation, confusing to a future maintainer, latent misreport risk if the key
is ever consumed.
**Fix:** delete the line-338 assignment (or rename it, e.g. `"pearson_selected"`) so both facts
are preserved instead of one silently overwriting the other.

**SM3 — `primary_effect` dict never carries `F`/`df1`/`df2`, so the "Main effect: X" summary line
in the ANOVA effects report is always silently skipped for Friedman / Freedman-Lane / Brunner-Langer.**
`src/analysis/nonparametricanovas.py:268-274, 558-564, 883-889` build `primary_effect` with only
`source`, `kind`, `policy`, `p_value`, `wald_chi2` — never `F`, `df1`, `df2`. The reader,
`src/export/report_stat_rows.py:523-536`, requires all of `F`, `p_value`, `df1`, `df2` to be
non-`None` before emitting the `"Main effect: {factor}"` row:
```python
F_primary = primary_effect.get("F")            # never set -> always None
df1_p = primary_effect.get("df1")              # never set -> always None
df2_p = primary_effect.get("df2")               # never set -> always None
if primary_factor and F_primary is not None and p_primary is not None and df1_p is not None and df2_p is not None:
```
So this branch is dead for every nonparametric-fallback result and the summary row never renders.
**This is lower-severity than the sphericity-gating bug found in the parallel core audit** because
the same F/df1/df2/p-value numbers are correctly rendered one code block earlier via the
`factors`/`interactions` lists (`report_stat_rows.py:402-461`), which the nonparametric functions
populate correctly (verified: `factors` entries in `nonparametricanovas.py` do carry `F`, `df1`,
`df2`, `effect_size`). No decision is gated on the missing `primary_effect` fields; it is a
redundant cosmetic summary line that silently never fires.
**Impact:** cosmetic only — the ANOVA effects table already shows the correct numbers; the
one-line "Main effect: X" recap is simply always absent for these three test types.
**Fix:** either add `"F": <chi2_stat or ATS>, "df1": ..., "df2": ...` to the three `primary_effect`
dict literals to match what the report reader expects, or drop the dead branch's requirement on
`F`/`df1`/`df2` (fall back to just `p_value` when they're absent) since `factors` already covers
the detailed table.

### LOW

**SM4 — Dead-code substring control-group match in `MixedAnovaPostHocAnalyzer._perform_test_legacy`.**
`src/analysis/posthoc_core.py:543`: `if control_group in comp["group1"] or control_group in comp["group2"]:`
uses Python's `in` substring containment instead of exact equality — if one group's interaction
label is a substring of another's (e.g. control group `"A"` vs a level named `"AB"`, or more
realistically `"Control"` vs `"Control_2"` if such labels ever occur), this would incorrectly
include/exclude comparisons from the Dunnett-style correction family, changing the FWER-controlled
p-values for an arbitrary subset. **Confirmed via `git grep` that `_perform_test_legacy` has zero
callers** — it is entirely superseded by `perform_test` (line 734), which correctly uses exact
match (`g1 == control_group or g2 == control_group`, line 879). Not exploitable today.
**Impact:** none currently (unreachable code), but dead code carrying a known-wrong pattern is a
maintenance hazard if anyone ever re-wires a caller to it.
**Fix:** delete `_perform_test_legacy` (58 lines, `posthoc_core.py:331-629`) as part of general
cleanup — it duplicates logic now done correctly and more simply in the live `perform_test`.

**SM5 — `RMAnovaPostHocAnalyzer`/`MixedAnovaPostHocAnalyzer` "Dunnett" labeled as exact Dunnett is
actually Holm-Bonferroni** — this is *already self-documented* in the code
(`posthoc_core.py:528-538`, `1172-1181`) with an accurate comment explaining why the exact
multivariate-t Dunnett doesn't apply to dependent/mixed contrasts, and the exported
`correction_method` string honestly says `"Dunnett-type (Holm-adjusted, ...)"` rather than
claiming exact Dunnett. Flagging only so the reviewer can confirm this is intentional
(it is — well-reasoned) and not confuse it with SM1's mislabeling, which has no such disclosure.

**SM6 — `DataImporter` (stats_functions.py:407) dead-code claim: confirmed still accurate, with one caveat.**
`git grep` shows `AnalysisManager._prepare_contextual_inputs` (`analysis_core.py:186-194`) only
calls `DataImporter.import_data(...)` when `analysis_context` is falsy. The autopilot pipeline
(`statistical_analyzer_autopilot_pipeline.py:1365, 1894`) always passes
`analysis_context["injected_df"] = self.df` per this project's CLAUDE.md, so `analysis_context` is
always truthy in the shipped UI path — `DataImporter.import_data` is never reached from the app.
**Caveat:** it is exercised by `fuzzing/_worker.py:93` and several files under `tests/` and
`validation/` calling `AnalysisManager.analyze(**kwargs)` without `analysis_context` — so the code
path is live for test/fuzz harnesses, just not for the shipped desktop app. The CLAUDE.md
"dead-code-in-the-autopilot-path" phrasing is precisely correct and should not be broadened to
"fully dead."

## Strengths (verified)

- **Games-Howell (`posthoc_core.py:1415-1489`)** — correctly computes the family size `k` once
  from groups with n≥2 and reuses it for every pairwise `studentized_range` call (both p-value and
  simultaneous CI), matching the standard Tukey-Kramer/Games-Howell procedure. Welch-Satterthwaite
  df and Hedges' g bias correction (`1 - 3/(4·df-1)`) are the textbook formulas.
- **Dunnett via `scipy.stats.dunnett` (`posthoc_core.py:1492-1557`)** — p-values and simultaneous
  CIs are drawn from the same joint multivariate-t fit (documented in-code at lines 1509-1513),
  guaranteeing p-value/CI consistency — a subtle correctness property many hand-rolled
  implementations get wrong.
- **EMM + multivariate-t Dunnett (`emm_posthoc.py`)** — closed-form reproduction of R's
  `emmeans(...) |> contrast("trt.vs.ctrl", adjust="mvt")`, with explicit `UnsupportedDesignError`
  fallback for unbalanced/incomplete designs rather than silently applying an invalid formula.
  Verified the equicorrelation-0.5 assumption is derived and commented, not assumed blindly.
- **Hand-rolled Holm correction (`nonparametricanovas.py::_holm_correct`)** — numerically verified
  identical to `statsmodels.stats.multitest.multipletests(method='holm')` across 2000 randomized
  trials; the step-down running-max logic is textbook-correct.
- **`effect_sizes.py`** — order-sensitive pattern matching explicitly engineered to avoid the
  "cohen_f matches cohen_d substring" trap (documented in-code), fails closed (`None`, not a wrong
  guess) on unrecognized labels, and every effect-size-type string actually produced by
  `posthoc_core.py`/`nonparametricanovas.py` was verified to canonicalize correctly.
- **`SimpleLinearRegressionModel` HC3 auto-switch (`correlation_models.py:581-592`)** — when
  Breusch-Pagan flags heteroscedasticity, `self.result` is reassigned to the robust-covariance
  result object, and every downstream statistic (`main_p`, `main_t`, `coefficient_table`, F-test)
  is read from that reassigned object — the corrected values correctly reach the fields that gate
  significance, unlike the Mixed-ANOVA sphericity bug found in the parallel core audit.
- **Freedman-Lane / Brunner-Langer ATS (`nonparametricanovas.py`)** — both include detailed
  in-code derivation comments (e.g. the reduced-model formula choice, the Satterthwaite df2
  approximation for the between-effect), and both report RTE tables / partial-η² alongside
  primary results with an honest "no standardized Cohen thresholds apply to RTE" caveat rather
  than force-fitting a Cohen's-d-style magnitude label onto a rank-based statistic.
- **`GamesHowellTest`/`DunnettTest` control-group exact matching** in the live (non-legacy) code
  paths of `posthoc_core.py` (lines 197, 879, 907, 1188, 1217) all use `==`/`!=`, not substring
  `in` — the one substring-match instance (SM4) is confirmed dead.

## Recommended remediation order

1. **SM1** (HIGH) — fix or honestly relabel the Mixed-ANOVA "Tukey" branch; cheapest fix is the
   relabel (one line), full fix (implement real Tukey via the existing `_tukey_p_value` helper)
   is also small since the RM sibling already has the pattern.
2. **SM2** (MEDIUM) — delete/rename the shadowed `both_normal` dict-literal assignment; a 1-line
   diff, purely preventive since nothing reads it today.
3. **SM3** (MEDIUM) — either populate `F`/`df1`/`df2` in the three `primary_effect` dicts or relax
   the reader's `and` condition; low risk either way since the `factors` table already renders the
   correct numbers.
4. **SM4** (LOW) — delete dead `_perform_test_legacy` (58 lines) as general cleanup; zero
   behavioral risk since it has no callers.
5. **SM5 / SM6** — no action needed; both are correctly implemented/documented as-is. Confirm SM6's
   scope (test/fuzz-only reachability) stays true if `AnalysisManager.analyze()`'s calling
   convention ever changes.
