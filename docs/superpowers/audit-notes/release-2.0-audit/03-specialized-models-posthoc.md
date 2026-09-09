# AUDIT: BioMedStatX — Specialized Models & Post-hoc/Outlier Engines @ 3fd4796

**Scope:** `src/analysis/correlation_models.py`, `posthoc_core.py`, `nonparametricanovas.py`,
`outlier_core.py`, `stats_functions.py`, `emm_posthoc.py`, `effect_sizes.py` (5,719 lines total,
all read in full). Second independent audit pass (round 2) of this subsystem — a prior audit
(2026-06-10/11/27) found and fixed real bugs here (Games-Howell, Dunnett, Box-Cox, mixed/RM
Dunnett `p*k*0.8` crutch). This pass re-verifies those fixes against source and hunts for new
issues.

**Verdict.** Posture is good and the prior fixes hold under re-derivation — no live emergency.
Games-Howell, Dunnett (both independent-groups and mixed/RM Holm-Bonferroni variants), the
bounded Box-Cox lambda guard, and the MWU power removal are all still correctly implemented as
described in the prior audit's memory notes. One new MEDIUM-severity defensive-validation gap
(an unconditional decimal-format assumption in outlier ingestion that can silently corrupt
US/international-formatted numeric strings) and one MEDIUM performance/usability issue
(Dunn-test bootstrap CI blocking the UI thread for tens of seconds on realistic group sizes)
are the top new findings. Everything else is LOW/cosmetic.

## What I mechanically verified (not eyeballed)

| Check | Command / method | Result |
|---|---|---|
| Line counts match task brief | `wc -l` on all 7 files | 1163/1812/1025/525/782/228/184 = 5,719 total, all read start-to-end |
| Grubbs' critical-value formula | Recomputed `G_crit` for n=10, α=0.05 two-sided in Python against the canonical published table value (2.290) | `2.289954...` — matches to 3 dp |
| Hedges' g bias-correction factor | Recomputed `J(df) = 1 - 3/(4·df - 1)` (Hedges & Olkin 1985) for df=20, compared to code's `1 - 3/(4*(n1+n2-2)-1)` | Identical formula, `0.9620...` |
| Cohen f² → R² threshold conversion | Recomputed `R² = f²/(1+f²)` for Cohen's canonical f² thresholds (0.02/0.15/0.35) | `0.0196/0.1304/0.2593` vs. code's `(0.02, 0.13, 0.26)` — see SM6 |
| `bounded_boxcox_lambda` guard | Read `src/statistical_testing/validators.py:166-201` in full | Confirmed: rejects `\|λ\|>3` or non-finite, hard-falls-back to λ=0, never clamps — matches prior-audit fix description exactly |
| Mixed/RM Dunnett correction method | Read `posthoc_core.py:528-559` (Two-Way… actually Mixed) and `:1172-1204` (RM) | Both use `multipletests(..., method='holm')` (plain Holm-Bonferroni, FWER-valid under arbitrary dependence), labelled `"Dunnett-type (Holm-adjusted, ...)"` — the `p*k*0.8` crutch is gone, replaced as claimed |
| MWU power field | `git grep -in "power" -- <all 7 files>` | Zero hits — `_mwu_posthoc_comp`/`_wilcoxon_posthoc_comp` in `nonparametricanovas.py` only ever set `effect_size`/`effect_size_type` (rank-biserial r), no power field exists to be miscomputed |
| `_convert_values_to_float` decimal-format bug | Reproduced in Python: `pd.Series(['1.5','2.75']).astype(str).str.replace('.','',regex=False).str.replace(',','.',regex=False).astype(float)` | `['1.5','2.75']` → `[15.0, 275.0]` — confirms silent 10-1000x corruption on US-formatted numeric strings stored as text (SM1) |
| DunnTest bootstrap CI cost | Timed the exact nested-loop pattern (`for u in b1 for v in b2`) at n=200/group (1000 boots) and n=500/group | 2.16s and 13.5s respectively; extrapolated to ~81s for 4 groups (6 pairs) at n=500/group (SM2) |
| Effect-size-type writer/reader contract | `python3 -c` round-tripped every `effect_size_type=` string literal found via `git grep` (`cohen_d`, `cohen_d_mixed`, `cohen_d_rm`, `hedges_g`, `"Cohen's d (RM)"`, `r`, `rank_biserial_r`) through `effect_sizes.canonicalize()` | All 7 resolve to the correct `EffectSizeKind` — no drift between what post-hoc code writes and what the report-layer classifier reads (Strength, not a finding) |
| Exception-handler density | `git grep -c "except "` per file | correlation_models.py:12, posthoc_core.py:15, nonparametricanovas.py:7, outlier_core.py:4, stats_functions.py:6, effect_sizes.py:1, emm_posthoc.py:0 |
| Bare `except:` (no type) | `git grep -n "except:"` across the 7 files | 4 hits, all in `posthoc_core.py` (lines 519, 726, 1289, 1300) — all four are Tukey/studentized-range p-value fallbacks that degrade to a documented t-approximation, not silent data loss (see Strengths) |

## Findings — severity ranked

### MEDIUM

**SM1 — Outlier ingestion silently corrupts US/international-formatted decimal strings.**
`src/analysis/outlier_core.py:96-107` (`OutlierDetector._convert_values_to_float`). When the
value column is not already numeric dtype (e.g. an Excel cell formatted as text, or a CSV column
containing a stray non-numeric character in one row that forces pandas to infer `object` dtype
for the whole column), the code unconditionally treats `.` as a thousands separator and `,` as
the decimal point:
```python
self.df[self.value_col] = (
    self.df[self.value_col].astype(str)
        .str.replace('.', '', regex=False)   # Remove thousand separators
        .str.replace(',', '.', regex=False)  # Comma → Period
        .astype(float)
)
```
There is no format detection, no locale parameter, and no plausibility check afterward. Verified
in Python: `"1.5"` → `15.0`, `"2.75"` → `275.0`, `"100.25"` → `10025.0` — silent 10×-1000×
inflation, not an error. This is the "German decimal" assumption baked in as the *only* path;
any US/international-formatted numeric string column that fails the `is_numeric_dtype` fast-path
for any reason (one bad cell, mixed formatting, `NA` strings, leading/trailing whitespace) gets
silently multiplied by up to 1000 before outlier detection runs — and outlier detection would
then "correctly" flag the *now-corrupted* extreme values, producing a plausible-looking but
completely wrong report with no diagnostic trail beyond the (non-user-facing) debug log.
**Impact:** A user whose Excel export stores the value column as text with plain decimal points
(common when copy-pasting from another tool, or a column with one stray text cell) gets outlier
results computed on data inflated by 10-1000x, with no error and no visible warning — this is
exactly the "silently produces a wrong-but-plausible result" failure mode called out in the audit
brief. Grubbs/Mod-Z will report different (and wrong) outliers than the true data would.
**Fix:** Only apply the comma→period substitution when the column actually contains commas as
apparent decimal separators (e.g., regex-detect `^\d{1,3}(\.\d{3})*,\d+$` or simply: if a value
parses cleanly as float as-is, leave it alone; only run the German-format substitution on values
that fail a direct `float()` parse AND match a comma-decimal pattern). At minimum, add a
post-conversion sanity check (e.g., compare magnitude distribution before/after, or require the
caller to declare the input locale) and surface a warning in `debug_log` / the exported report
when the fallback path fires, not just a silent debug-log-only "before/after" value dump.

**SM2 — Dunn-test post-hoc bootstrap CI is O(n_boot × n₁ × n₂) pure-Python and blocks the UI thread.**
`src/analysis/posthoc_core.py:1596-1603` (`DunnTest.perform_test`). The confidence interval for
each pairwise median difference is computed via 1000 bootstrap resamples, each building a full
`n1 × n2` pairwise-difference list in pure Python:
```python
for _ in range(n_boot):
    b1 = np.random.choice(x, n1, replace=True)
    b2 = np.random.choice(y, n2, replace=True)
    boots.append(np.median([u - v for u in b1 for v in b2]))
```
Measured directly: n1=n2=200 → 2.16s per pair; n1=n2=500 → 13.5s per pair. For a realistic
4-group comparison (6 pairs) at n=500/group this is ~80 seconds of blocking computation, and
nothing in `posthoc_core.py`/`stats_functions.py` runs this off the main Qt thread (no
`QThread`/worker dispatch visible in this call path) — the desktop app UI would freeze for that
duration with no progress indication beyond the one "Running outlier detection..." /
generic-analysis modal already in place elsewhere. **Impact:** on biomedical high-throughput
datasets (n in the hundreds per group is common for plate-reader / flow-cytometry exports), Dunn
test becomes impractically slow or reads as a hang. **Fix:** vectorize with
`b1[:, None] - b2[None, :]` (or `np.subtract.outer`) — this turns the O(n1·n2) Python-level loop
into a single vectorized numpy operation per bootstrap iteration, expected to cut runtime by
1-2 orders of magnitude at these sizes; if still slow at very large n, subsample the
cross-difference matrix for the median estimate (already a common bootstrap-CI approximation)
rather than materializing the full outer product.

### LOW

**SM3 — `ExploratoryCorrelationMatrix._compute_matrix` swallows per-cell computation failures with a bare `except Exception: pass`.**
`src/analysis/correlation_models.py:993-994`. If `pearsonr`/`spearmanr` raises for a specific
column pair (e.g., zero-variance column within one stratum), the cell is silently left at its
`np.nan` initialization value with no warning surfaced to the caller or the exported report.
**Impact:** low — NaN is already the documented "no data" sentinel for this matrix and is
correctly rendered as blank/None downstream (`_ndarray_to_nested_dict` maps NaN→None), so this
does not produce a wrong-but-plausible value, only a silently-missing one with no diagnostic
trail explaining *why* that specific cell is missing. **Fix:** log a warning (or accumulate a
per-cell error list surfaced in `as_results_dict()`) naming the failed pair and the exception,
so a user staring at an unexpectedly blank cell in an otherwise-complete matrix has somewhere to
look.

**SM4 — Four bare `except:` (no exception type) in `posthoc_core.py`.**
Lines 519, 726, 1289, 1300 — all in Tukey/studentized-range p-value fallback paths (e.g.
`TwoWayPostHocAnalyzer.perform_test`'s Tukey branch, `MixedAnovaPostHocAnalyzer._tukey_p_value`,
`RMAnovaPostHocAnalyzer._get_tukey_critical_value`/`_tukey_p_value`). Each catches literally
everything, including `KeyboardInterrupt`/`SystemExit`, and falls back to either a Bonferroni
correction or a t-distribution approximation. **Impact:** low in practice — the fallback is a
documented, methodologically defensible substitute (Bonferroni is conservative, not silently
wrong), not a data-corrupting default — but a bare `except:` is still bad practice: it would also
swallow a genuine `KeyboardInterrupt` mid-computation, and it can mask an unrelated bug (e.g. a
typo introduced in a future edit to the try block) as if it were "distribution unavailable."
**Fix:** narrow to `except Exception:` at minimum (already done at 3 of the ~15 similar sites
elsewhere in this same file, e.g. line 1163 `except Exception as err:` in the twin RM-ANOVA
Tukey path) so `KeyboardInterrupt`/`SystemExit` propagate.

**SM5 — `RegressionHealthScanner` VIF and outlier-scan blocks (`correlation_models.py:1140-1161`) catch bare `Exception` and only record `{"error": str(exc)}`.**
This is a pre-flight advisory check (not a hard gate), so failure here is correctly non-blocking
by design — but the caught exception is never logged (no `logger.warning`), only stashed in the
returned dict's `checks["vif"]["error"]` key. If nothing downstream reads that key on the failure
path, the user has no visible signal that multicollinearity was never actually checked.
**Impact:** low — advisory-only path, but a silent skip of a documented "the value" check is
still worth a log line. **Fix:** add `logger.warning(f"VIF check failed: {exc}")` alongside the
existing `checks["vif"] = {"error": str(exc)}` so it at least surfaces in the debug console even
if the export layer doesn't render it.

**SM6 — `effect_sizes.py` R²-threshold values labeled "Cohen f²-derived" don't match the literal f²→R² conversion.**
`src/analysis/effect_sizes.py:120`: `R_SQUARED: (0.02, 0.13, 0.26)`. Re-deriving via
`R² = f²/(1+f²)` on Cohen's canonical f² thresholds (0.02 small / 0.15 medium / 0.35 large,
Cohen 1988) gives `(0.0196, 0.1304, 0.2593)`. The code's small-threshold (`0.02`) is actually the
raw f² value, not the converted R² value (`0.0196`); medium/large are correctly rounded
conversions. **Impact:** negligible in practice — the discrepancy is <0.001 in every band, far
below the precision at which a magnitude label ("small"/"medium"/"large") would ever flip for a
real R² value — but the docstring/comment claims a derivation that the small-threshold constant
doesn't quite follow. **Fix:** either change `0.02` → `0.0196` for internal consistency, or amend
the comment to say "small threshold kept at Cohen's raw f² convention (0.02) for round-number
simplicity; medium/large use the f²→R² conversion" so the two conventions aren't silently mixed
under one citation.

**SM7 — `GamesHowellTest`/Dunnett effect-size CI and Games-Howell `k` definition worth a docs note, not a bug.**
`src/analysis/posthoc_core.py:1430-1433`: `k = len(comparable)` counts only groups with `n>=2`
that appear anywhere in `valid_groups`, computed once outside the pairwise loop — correct per
Games-Howell's definition (k = number of groups in the whole family, not just the current pair),
confirmed by re-deriving the studentized-range formula (`q = √2·|t|`, `p = sf(q, k, df_welch)`)
against the pingouin implementation description already cross-validated in the prior audit
(memory: "Matches pingouin to 1e-4"). No new issue found; flagging only because a future editor
might be tempted to move `k` inside the loop and inadvertently break FWER control by
recalculating it per-comparable-pair-only.

## Strengths (verified)

- **Games-Howell is genuinely Games-Howell.** `posthoc_core.py:1415-1489` — Welch-Satterthwaite
  df, `q = √2·|t|` against the studentized-range distribution, Hedges' g with the exact
  Hedges & Olkin (1985) small-sample correction `1 - 3/(4·df - 1)` (re-derived and matched to the
  textbook formula in this pass), and a simultaneous CI built from the same q-distribution
  (`q_crit`). The prior-audit fix holds.
- **Dunnett is a single coherent multivariate-t fit, not double-corrected.** `posthoc_core.py:1502-1547`
  (independent-groups) uses `scipy.stats.dunnett(...)` once and derives both p-values and CIs from
  the same joint fit — the code comment explicitly documents why this guarantees CI/p-value
  consistency, and it does (no separate Šidák pass layered on top).
- **Mixed/RM Dunnett honestly labelled, not exact-Dunnett-by-assumption.** `posthoc_core.py:528-559`,
  `:1172-1204` — both correctly recognize that within-subject/mixed contrasts violate the
  independence/equicorrelation assumptions `scipy.stats.dunnett` requires, and fall back to plain
  Holm-Bonferroni (not Holm-Šidák — the comments correctly explain *why* the Bonferroni variant is
  required for FWER validity under arbitrary/negative dependence), labelling the result
  `"Dunnett-type (Holm-adjusted, ...)"` rather than claiming exact Dunnett. This is the fix
  described in the prior-audit memory and it is intact.
- **Box-Cox divergence guard is correctly wired at every call site found in this batch.**
  `correlation_models.py:128-141` (`_apply_transform`) calls `bounded_boxcox_lambda` (in
  `statistical_testing/validators.py`), which rejects `|λ|>3` or non-finite estimates and
  hard-falls-back to λ=0 (natural log) rather than clamping to a boundary — re-read in full and
  confirmed to match the described fix exactly, including the "never clamp" invariant.
  `_optimize_boxcox_for_regression` (correlation_models.py:143-191) independently bounds its
  `minimize_scalar` search to `[-2, 2]`, a second, consistent guard for the Y-on-X regression path.
- **MWU power was correctly *removed*, not just hidden.** No `power` computation exists anywhere
  in `nonparametricanovas.py`'s `_mwu_posthoc_comp`/`_wilcoxon_posthoc_comp` — confirmed via
  `git grep` returning zero hits for "power" across all 7 files in this batch. The fix wasn't a
  patch that could regress; the invalid code path was deleted outright.
- **EMM + multivariate-t Dunnett (`emm_posthoc.py`) is dimensionally and statistically careful.**
  The split-plot/RM variance-component derivations (`variance_components`, `contrast_se_df`) and
  the shared-equicorrelation multivariate-t adjustment (`_mvt_adjusted_p`, R=0.5 off-diagonal,
  matching the documented "Var=2·MS_res/n, Cov=MS_res/n ⇒ ρ=0.5" derivation) are internally
  consistent, and the module correctly refuses (`UnsupportedDesignError`) rather than silently
  approximating when the design is unbalanced or incomplete — every caller (`posthoc_core.py:739-762`,
  `:1009-1033`) catches that specific exception and falls back to the isolated-t-test path, never
  swallowing it as a generic exception.
- **Effect-size-type strings have zero writer/reader drift.** Every `effect_size_type=` literal
  written by the post-hoc engines (`cohen_d`, `cohen_d_mixed`, `cohen_d_rm`, `hedges_g`, `r`,
  `"Cohen's d (RM)"`, `rank_biserial_r`) round-trips correctly through
  `effect_sizes.canonicalize()` to the intended `EffectSizeKind` — verified by direct execution,
  not just static reading. The enum-dispatch design (replacing ambiguous substring matching)
  described in the module's own docstring is doing its job.
- **Brunner-Langer ATS and Freedman-Lane permutation implementations are heavily
  self-documenting about design choices** (e.g. the comment block at
  `nonparametricanovas.py:443-446` explicitly explains why a naive reduced-model choice would be
  wrong and what the correct Freedman-Lane reduced model is) — this is exactly the kind of
  "why, not just what" comment that makes a re-audit tractable, and it made cross-checking the
  df1/df2 bookkeeping straightforward in this pass.
- **Grubbs' test matches the canonical formula exactly.** `outlier_core.py:165-192` — re-derived
  independently and matched the standard published critical value (n=10, α=0.05, two-sided:
  2.290) to 3 decimal places.

## Recommended remediation order

1. **SM1 (outlier decimal-format corruption)** — cheapest fix with the highest silent-corruption
   risk: gate the comma-decimal substitution behind a format-detection check or a "does this
   already parse as float" short-circuit. This is the one finding in this batch that matches the
   audit brief's core concern (silently produces a wrong-but-plausible result).
2. **SM2 (Dunn-test bootstrap performance)** — vectorize the outer-product bootstrap; quick,
   isolated, high user-visible payoff (turns a ~80s freeze into a sub-second wait at realistic
   group sizes).
3. **SM4 (bare `except:` → `except Exception:`)** — four one-line changes, zero behavior change,
   removes the `KeyboardInterrupt`/`SystemExit`-swallowing risk.
4. **SM3 / SM5 (silent per-cell / per-check failure logging)** — add `logger.warning` calls;
   no behavior change, pure observability improvement.
5. **SM6 (R² threshold constant vs. its own citation)** — cosmetic; fix the constant or amend the
   comment, whichever better reflects intent.
6. **SM7** — no action needed; documented here only as a "don't refactor this into a bug" note
   for future editors.
