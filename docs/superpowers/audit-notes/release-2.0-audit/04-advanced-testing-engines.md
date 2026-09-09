# AUDIT: BioMedStatX — Advanced Statistical Testing Engines @ 3fd4796

**Scope:** `src/statistical_testing/mixed_assumptions.py`, `posthoc_fallback.py`,
`assumption_checks.py`, `advanced_pipeline.py`, `validators.py`, `decision_logic.py`,
`engines/{advanced_posthoc,comparison,transformation,extraction,assumption_bridge,reporting}.py`
(5,013 lines total, all read in full). Round 2 of a 7-subsystem pre-2.0 audit.

**Verdict.** Not a live emergency (single-user desktop app, no external attack surface), but
there is a **confirmed, still-open correctness bug**: the Mixed ANOVA within-factor sphericity
correction in `mixed_assumptions.py` is computed, stored under its own key, and even rendered in
the decision-tree diagram — but never overwrites the canonical `p_value` field that gates the
significance verdict and post-hoc dispatch in `_run_mixed_anova` (`src/analysis/statisticaltester.py`).
This is the same bug class the prior audit round found and partially fixed (commit `d0e5394`) —
but that fix landed only in the sibling RM-ANOVA path (`_perform_comprehensive_sphericity_test`),
never in the Mixed-ANOVA twin owned by this batch's `MixedAnovaAssumptionEngine`. Worse: live
verification against the installed pingouin 0.6.1 shows the Mixed-ANOVA correction *computation*
itself is dead code on the real data shape — `mixed_anova()` never returns the `GG-eps`/`p-GG`/
`W-spher`/`p-spher` columns this engine looks for, so even a correct wiring fix would still not
apply a real GG/HF correction without also computing epsilon independently. Outside that cluster,
the batch is solid: `validators.py` is exemplary defensive-programming, and the Holm/Holm-Šidák/
FDR corrections in `posthoc_fallback.py` were checked against their actual statsmodels method
kwargs and are correctly labeled.

## What I mechanically verified (not eyeballed)

| Check | Command / method | Result |
|---|---|---|
| Prior-round fix location | `git show d0e5394` | Fix touched only `src/analysis/statisticaltester.py` (RM-ANOVA `_perform_comprehensive_sphericity_test`); `mixed_assumptions.py` untouched |
| Mixed-ANOVA sphericity call sites | `git grep -n "_test_mixed_anova_within_sphericity\|within_corrected_p_value\|within_correction_used"` across `*.py` | Only written in `mixed_assumptions.py`; only read by `src/visualization/decisiontreevisualizer.py:283` (display-only) |
| Where `results["p_value"]` is set for Mixed ANOVA | Read `src/analysis/statisticaltester.py:1470-1706` (`_run_mixed_anova`) | Set once from the raw interaction `p_col` at line 1548-1549, **before** the within-sphericity block (lines 1701-1706) runs; never reassigned afterward |
| Contrast: RM-ANOVA does wire it | Read `src/analysis/statisticaltester.py:1966-1996` | Line 1973-1974: `if sphericity_results.get("final_p_value") is not None: results["p_value"] = sphericity_results["final_p_value"]` — correct pattern, absent in the Mixed-ANOVA function |
| Regression test coverage for the Mixed-ANOVA path | `git grep -ln "within_corrected_p_value\|within_correction_used\|MixedAnovaAssumptionEngine\|_test_mixed_anova_within_sphericity" tests/*.py` | 0 matches (the RM-ANOVA sibling has `tests/test_sphericity_outer_exception.py`; Mixed ANOVA has nothing) |
| Real pingouin `mixed_anova()` schema | Live `python3` run: `pg.mixed_anova(data=df, dv='dv', within='within', between='between', subject='subject')`, pingouin 0.6.1 | Columns: `['Source','SS','DF1','DF2','MS','F','p_unc','np2','eps']` — Source values are exactly `between`/`within`/`Interaction`; **no** `GG-eps`, `p-GG`, `HF-eps`, `p-HF`, `W-spher`, `p-spher`, or `sphericity` columns |
| Contrast: real `rm_anova()` schema | Same script, `pg.rm_anova(..., correction=True)` | Columns include `p_GG_corr`, `sphericity`, `W_spher`, `p_spher` — the RM path's columns genuinely exist, unlike Mixed ANOVA's |
| Interaction-row string filter reachability | `aov["Source"].str.contains(within_factor, regex=False) & aov["Source"].str.contains("*", regex=False)` (mixed_assumptions.py:615-616) tested against real Source values | Always `False` — pingouin's `Interaction` label contains no `*` character; block at lines 618-625 is unreachable dead code |
| Bare `except:` | `git grep -n "^\s*except:"` across the 12 files | 1 hit: `mixed_assumptions.py:1235` (Box's M log-determinant on a singular matrix; falls back to `np.log(1e-10)`, non-blocking) |
| Broad `except Exception` count per file | `git grep -c "except Exception"` per file | mixed_assumptions.py 22, posthoc_fallback.py 14, assumption_checks.py 8, advanced_pipeline.py 3, validators.py 1, comparison.py 4, advanced_posthoc.py 5, transformation.py 1, decision_logic.py/extraction.py/assumption_bridge.py/reporting.py 0 |
| Every `raise ...Error(...)` in-batch | `git grep -n "raise .*Error("` across the 12 files | 25 call sites, all in `validators.py`, `mixed_assumptions.py`, `comparison.py`; each spot-checked against its guard condition — messages accurate to the firing condition |
| Holm / Holm-Šidák / FDR correctness in `posthoc_fallback.py` | Read lines 611-693; verified `method='holm-sidak'`/`'fdr_bh'`/`'holm'` kwargs match the human-readable labels attached ("Holm-Šidák", "FDR (Benjamini-Hochberg)", "Holm-Bonferroni") | Correct — labels match `statsmodels.stats.multitest.multipletests` method semantics |
| Line counts match expected scope | `wc -l` on all 12 files | 5,013 total, matches the assigned batch exactly |

## Findings — severity ranked

### HIGH

**AT1 — Mixed ANOVA within-factor sphericity correction computed but never gates the
significance verdict or post-hoc dispatch (same bug class as the RM-ANOVA fix in `d0e5394`,
still open here).**
`src/statistical_testing/mixed_assumptions.py:414-499` (`_test_mixed_anova_within_sphericity`)
computes `within_corrected_p_value` / `within_correction_used` and returns them as extra dict
keys. The caller, `src/analysis/statisticaltester.py:1701-1706`, merges them into `results` via
`results.update(within_sphericity_results)` — but `results["p_value"]` was already set at
`statisticaltester.py:1548-1549` from the raw, uncorrected `interaction["p_value"]`, and is never
reassigned afterward. The within-factor and interaction post-hoc dispatch gates at
`statisticaltester.py:1573-1650` (`int_row[p_col]`, `within_row[p_col]`) also read the raw
`p_col`, not any corrected value. **Impact:** a Mixed ANOVA with within-factor sphericity
violation reports (and gates post-hoc on / off, and the top-level significant/non-significant
verdict) using the *uncorrected* p-value, silently reintroducing inflated Type-I error exactly
like the bug the prior round fixed for RM-ANOVA — except here it was never fixed at all. The
`within_correction_used` string does get rendered in the decision-tree diagram
(`src/visualization/decisiontreevisualizer.py:283`), so a user sees "Greenhouse-Geisser applied"
next to a verdict that was actually computed from the uncorrected number — a directly misleading
report. **Fix:** in `_run_mixed_anova` (`statisticaltester.py`), after
`results.update(within_sphericity_results)`, mirror the RM-ANOVA pattern at line 1973-1974:
`if within_sphericity_results.get("within_corrected_p_value") is not None: results["p_value"] = within_sphericity_results["within_corrected_p_value"]` — but only do this once AT2 (below) is
also fixed, since currently the "corrected" value is frequently just the uncorrected p-value
relabeled (see AT2). Add a regression test mirroring
`tests/test_sphericity_outer_exception.py` but targeting `MixedAnovaAssumptionEngine` directly.

**AT2 — The GG/HF correction lookup in `_apply_corrections_to_effect_row` checks column names
(`GG-eps`, `p-GG`, `HF-eps`, `p-HF`) that pingouin's `mixed_anova()` never returns; the
"correction" silently degrades to the uncorrected p-value on every real run.**
`src/statistical_testing/mixed_assumptions.py:641-725`. Verified live against the installed
pingouin 0.6.1: `pg.mixed_anova(...)` returns columns
`['Source','SS','DF1','DF2','MS','F','p_unc','np2','eps']` — a single `eps` column, no
`GG-eps`/`p-GG`/`HF-eps`/`p-HF` (those column names are RM-ANOVA-table-specific, from
`pg.rm_anova(..., correction=True)`, which does return `p_GG_corr`). Because line 661
(`if 'GG-eps' in effect_row and 'p-GG' in effect_row:`) and line 678 always fail to find these
keys on a real Mixed-ANOVA row, `gg_epsilon = None` (line 674) and `hf_epsilon = None` (line 691)
on every real call, so the function always falls to the `else` branch at lines 714-718:
`"correction_used": "None (corrections not available)"`, with `final_p_value` equal to the raw
uncorrected p. The same blind spot affects the fallback extractor,
`_extract_mixed_sphericity_from_anova_table` (lines 502-567): it checks for `'W-spher'`,
`'p-spher'`, `'sphericity'` columns (also confirmed absent from the real table), so it always
hits the `else` at lines 542-553 and reports `"Indeterminate (Defaulting to GG correction)"` — a
correction that, per this finding, can never actually be computed from the data pingouin gives
it. **Impact:** even after fixing AT1's wiring, the "corrected" p-value routed into
`results["p_value"]` would still be numerically identical to the uncorrected one — the sphericity
correction for Mixed ANOVA is a no-op end to end, not merely disconnected. **Fix:** compute
epsilon explicitly (pingouin does return a plain `eps` column on the `mixed_anova()` table, or
compute GG epsilon manually from the covariance matrix the way `pg.epsilon()` does) and apply the
correction to `DF1`/`DF2` and re-derive the p-value from `scipy.stats.f.sf`, rather than expecting
pre-computed `GG-eps`/`p-GG` columns that this pingouin version's Mixed-ANOVA table does not
provide.

**AT3 — Interaction-effect sphericity-correction block is unreachable dead code: the row filter
requires a literal `*` in `Source`, which pingouin's Mixed ANOVA table never contains.**
`src/statistical_testing/mixed_assumptions.py:614-616`:
```python
interaction_rows = aov[aov["Source"].str.contains(within_factor, regex=False) &
                      aov["Source"].str.contains("*", regex=False)]
```
Verified live: pingouin 0.6.1's Mixed-ANOVA `Source` column contains exactly the strings
`between`, `within` (the raw column names passed in) and the literal word `Interaction` — never a
`factor_a * factor_b`-style label with an asterisk. `"Interaction".__contains__("*")` is `False`.
This mask is therefore always empty, so `corrections["within_sphericity_corrections"]["interactions"]`
is never populated by `_apply_mixed_anova_sphericity_corrections` (lines 614-625), even though the
sibling function `_test_interaction_sphericity` (line 951: `if 'GG-eps' in interaction_row and
'HF-eps' in interaction_row:`) clearly expects this data to exist somewhere. **Impact:** low on
its own (the whole GG/HF-column premise is broken per AT2 regardless), but it means the
interaction-level correction path was written and never exercised even in a hypothetical
column-fixed future — the recommendation text "Interaction effects also require sphericity
corrections" (`_generate_within_factor_recommendations`, line 765) can never actually fire.
**Fix:** once AT2's column-availability issue is resolved, match interaction rows against the
literal pingouin label `"Interaction"` (or, for a design with a named third factor, the
`f"{a} * {b}"` pattern pingouin does use in `anova()`/`rm_anova()` for two-way — check the actual
label per pingouin function, don't assume the same delimiter everywhere).

### MEDIUM

**AT4 — Sample-quality pre-flight (`validate_samples_for_test`) is skipped for
`logistic_regression` in the shared advanced pipeline, with no equivalent gate substituted.**
`src/statistical_testing/advanced_pipeline.py:93`: `if test not in ["logistic_regression"]:` wraps
the entire `validate_samples_for_test(...)` call. `ExtractionEngine` (engines/extraction.py:66-73)
does build per-level samples for `logistic_regression` but nothing validates them for the
zero-variance / Inf / n-below-minimum conditions the shared gate exists to catch (per
`validators.py`'s own docstring: "Run before any statistical test so that pathological inputs
become a clean labeled block instead of a crash or a silently-wrong result"). **Impact:**
scoped to logistic regression only — a perfectly-separated or zero-variance predictor level can
reach `LogisticRegressionModel.fit()` (in `clinical_models.py`, outside this batch) without the
same pre-flight net every other advanced test gets, risking either an opaque statsmodels
convergence warning or a silently wrong coefficient rather than the clean `BLOCK_MESSAGES`-style
error. **Fix:** either route `logistic_regression` through `validate_samples_for_test` with
`dependent=False` (it already has per-level `samples`/`groups` from `ExtractionEngine`), or add an
explicit comment plus a `validate_outcome`-style single-vector check (already exists in
`validators.py:419-438`) scoped to logistic regression's binary outcome column.

**AT5 — `_perform_welch_anova`'s manual Welch F/df computation silently falls back to the
non-robust `f_oneway` result on ANY exception, including a benign one, without setting a flag the
caller can detect.**
`src/statistical_testing/mixed_assumptions.py:263-294`. The `try` block (manual Welch calculation)
catches bare `except Exception:` (line 291) and falls back to
`welch_f, p_val_welch = f_oneway(*group_data)` (line 293) — i.e., silently returns the *standard*
ANOVA result mislabeled as `"welch_f_statistic"`/`"welch_p_value"` in the returned dict (keys at
lines 298-299). **Impact:** the whole point of this function is to give a variance-robust
alternative when Levene's test flags heterogeneity; if the manual computation throws (e.g. a
`ZeroDivisionError` from a group with `var=0`, or a numerically degenerate `weights` sum), the
caller receives a result dict that *looks* like a Welch correction (same keys, same shape) but is
actually numerically identical to the standard F-test — an assumption-check contradicting itself
downstream in the recommendation text (`_generate_between_assumption_recommendations` compares
`levene_met`/`brown_met` against a "robust_alternatives.welch_anova" that, under this fallback, is
not actually robust). **Fix:** on the inner exception, set an explicit
`"welch_calculation_degraded": True` flag (or reuse the existing `error` key pattern seen
elsewhere in this file) so `_generate_between_assumption_recommendations` and any HTML export
consumer can detect and disclose the degradation instead of silently presenting a fallback as if
it were the requested statistic.

### LOW

**AT6 — `_perform_box_m_test`'s p-value is a documented, non-standard approximation
(`min(0.5, exp(-|boxM|/10))`), and its own bare `except:` at line 1235 masks a singular-covariance
failure with an arbitrary `1e-10` floor rather than reporting the singularity.**
`src/statistical_testing/mixed_assumptions.py:1169-1278`. The docstring is honest about the
approximation ("This is an approximation — consider using specialized software for exact test",
line 1260), which is good practice; but the bare `except:` at line 1235
(`log_det_sum += n_i * np.log(1e-10)`) silently absorbs a `LinAlgError` from a singular per-group
covariance matrix (e.g., from a between-group cell with fewer subjects than within-levels, or
perfectly collinear repeated measures) and folds it into the same statistic as a well-conditioned
case, rather than surfacing "covariance matrix singular for group X" the way `validators.py`'s
`DataQualityError` pattern does elsewhere in this codebase. **Impact:** low — this is an
assumption-check side-channel (Box's M is informational, not gating), and the function is already
labeled experimental. **Fix:** catch `np.linalg.LinAlgError` specifically, and surface a
`"note": "Covariance matrix singular for one or more groups; Box's M approximation unreliable"`
rather than folding a synthetic floor value into the aggregate statistic silently.

**AT7 — `_generate_between_assumption_recommendations` and related recommendation-builders return
plain lists of free-text strings (some prefixed `"⚠️"`/`"✅"`/`"INFO:"` inconsistently) rather than
a small structured type, making them brittle to consume for anything beyond direct display.**
`src/statistical_testing/mixed_assumptions.py:328-384`, `728-778`, `1359-1439`. Not a correctness
bug — purely a maintainability note. Emoji/prefix conventions are inconsistent (`"⚠️"` vs
`"INFO:"` vs no prefix at all, e.g. line 363 vs line 344), which makes it hard for any downstream
consumer (HTML export, a future non-English UI) to filter/style these by severity without
string-parsing. **Fix (optional, low priority):** introduce a
`RecommendationItem(level: Literal["ok","warn","info"], text: str)`-style TypedDict/dataclass:
low urgency for a single-user desktop tool, but worth doing opportunistically if this code is
touched again for AT1-AT3.

## Strengths (verified)

- **`validators.py` is genuinely strong defensive-programming.** `validate_samples_for_test`
  (lines 348-416) is a single, well-documented pre-flight chokepoint that catches Inf values,
  empty groups, below-minimum-n, numeric overflow (via a correctly-derived
  `sqrt(float64_max/n)` bound, not a naive global constant — see the docstring at lines 22-27),
  zero-variance groups, too-few-groups, and — for dependent designs — constant *pairwise*
  differences across every pair (not just adjacent ones), which is exactly what an RM-ANOVA
  covariance matrix needs to stay non-singular. No bare excepts; every raised exception is a
  named `ValidationError` subclass with a clear, template-driven message
  (`BLOCK_MESSAGES`, lines 315-324) shared between the UI and the HTML report.
- **`bounded_boxcox_lambda`** (`validators.py:166-201`) correctly guards the exact Box-Cox
  "blow-up" pattern named in the audit brief: it rejects an ML lambda estimate outside `[-3, 3]`
  and hard-falls-back to log rather than clamping to the boundary (which the docstring correctly
  identifies as "methodologically invalid"). This same helper is reused consistently in both
  `assumption_checks.py:369` and `engines/transformation.py:70`, so the guard isn't duplicated or
  drifted between the two call sites.
- **The significance-gated candidate-comparison logic** the Freedman-Lane / Brunner-Langer
  non-parametric fallback methods use (`engines/advanced_posthoc.py:256-499`) is unusually
  careful: post-hoc candidate pairs are
  significance-gated per-effect (marginal pairs offered only when that main effect is significant,
  cell/interaction pairs only when the interaction is significant), consistently reusing the same
  `_mwu_posthoc_comp`/`_wilcoxon_posthoc_comp`/`_apply_holm` helpers so the reported statistics
  match the all-pairs code path exactly — a real effort to avoid multiple-testing inflation from
  an uncontrolled "test everything" post-hoc sweep.
- **Holm / Holm-Šidák / FDR labels in `posthoc_fallback.py` match their actual statsmodels
  method kwargs** (verified at lines 625-628, 663-668): `'holm-sidak'` labeled "Holm-Šidák",
  `'fdr_bh'` labeled "FDR (Benjamini-Hochberg)", `'holm'` labeled "Holm-Bonferroni" — no
  mislabeled-correction-method bug in this file, unlike prior findings elsewhere in the codebase
  (the Games-Howell/Dunnett fixes noted in project memory).
- **`decision_logic.py` is a clean, small, testable pure-function module** — `DecisionInput`/
  `AssumptionState` are frozen dataclasses, `select_comparison_test` is a straightforward
  branch table with no hidden state, and `extract_assumption_state` explicitly documents its
  backward-compatibility fallback path for legacy `test_info` shapes rather than silently
  guessing.
- **The engines/ layer (`comparison.py`, `transformation.py`, `extraction.py`,
  `assumption_bridge.py`, `reporting.py`) is a consistent, well-factored `execute(payload) ->
  StatisticalResult` pattern** with an explicit unsupported-mode error branch in every single
  `execute()` method (never a silent no-op default) — a good structural guard against the
  string-coupled-mode-drift bug class named in the audit brief.
- **`assumption_checks.py`'s transformation-selection logic correctly separates the normality
  concern from the variance concern** (comment at line 315-317: "Welch-ANOVA (and RM corrections)
  handles variance heteroscedasticity. Transformation is strictly for correcting non-normality.")
  and guards every transformation branch (log10, Box-Cox, arcsin-sqrt) against the zero-variance
  edge case with an explicit fallback and a logged warning (lines 414-434), rather than letting
  `NaN`/`Inf` propagate silently into the downstream model fit.

## Recommended remediation order

1. **AT2 first, then AT1** — fixing the wiring (AT1) before the underlying computation (AT2) would
   just relabel the uncorrected p-value as "corrected," which is worse than the status quo
   (false confidence). Compute a real epsilon/correction from the actual `eps` column (or
   `pg.epsilon()`) for Mixed ANOVA, verify it against a golden-R Mixed ANOVA fixture the way
   `tests/test_golden_r_advanced.py` already does for other designs, *then* wire
   `results["p_value"]` the same way the RM-ANOVA path does at `statisticaltester.py:1973-1974`.
   Add a regression test mirroring `tests/test_sphericity_outer_exception.py` scoped to
   `MixedAnovaAssumptionEngine`.
2. **AT3** — trivial one-line fix (match the literal `"Interaction"` label) once AT2 lands;
   otherwise defer, since it's currently moot (no columns to correct with anyway).
3. **AT4** — quick, bounded fix: route `logistic_regression` through the existing
   `validate_samples_for_test`/`validate_outcome` gate; low effort, closes a real gap.
4. **AT5** — add the degraded-fallback flag; cheap, improves honesty of an already-existing
   code path without changing its numeric behavior.
5. **AT6, AT7** — low priority; bundle into the same PR as AT1-AT3 if that file is being touched
   anyway, otherwise defer indefinitely (neither gates a verdict).
