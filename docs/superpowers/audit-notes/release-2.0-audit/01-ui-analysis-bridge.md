# AUDIT: BioMedStatX @ b16cf24 — UI-to-Analysis Bridge / Entry Point

**Scope.** `src/analysis/statistical_analyzer.py` (628 lines), `src/autopilot/statistical_analyzer_autopilot_ui.py`
(2453 lines), `src/autopilot/statistical_analyzer_autopilot_pipeline.py` (2341 lines). All three read in full,
line 1 to EOF, no excerpting. Environment: local single-user PyQt5 desktop app, no CI, no multi-tenancy —
correctness bar is statistical/data validity, not web-security posture (`references/my-environment.md`).

**Verdict.** The bridge layer is generally well-structured (clear mixin boundary, a real pre-flight
validation helper for subject IDs, several previously-fixed footguns visible in git history), but the
subject-ID NaN guard added this session (`_reject_missing_subject_ids`, commits 3771884/5723f525/c9a0349)
does **not** cover the auto-mapping heuristic path, and I found one new, mechanically-confirmed **live bug**:
`_ap_build_analysis_context` permanently mutates `self.df` in place with a compounding boundary transform
every time it runs — and it runs on *every mapping change*, not just on "Start Analysis." Nothing here is a
security emergency (single-user local tool), but two findings (P1, P2) produce silently-wrong statistical
results, which is the correctness bar that matters for this app.

## What I mechanically verified (not eyeballed)

| Check | Command | Result |
|---|---|---|
| File sizes | `wc -l` on all 3 files | 628 / 2453 / 2341 lines — all read in full |
| `except Exception` count | `grep -n "except Exception"` per file | `statistical_analyzer.py`: 11; `..._ui.py`: 4; `..._pipeline.py`: 15 |
| Bare `except:` count | `grep -n "except:"` per file | `statistical_analyzer.py`: 1 (line 617); others: 0 |
| `groupby(...)` call sites | `grep -n "groupby("` across all 3 files | 5 hits, all in `..._pipeline.py`: lines 601, 1166, 1189, 1226, 1235 |
| `raise ValueError` sites | `grep -n "raise ValueError"` across all 3 files | 14 hits — 2 in `..._ui.py` (141, 1024), 12 in `..._pipeline.py` |
| `self.df[...] =` in-place mutation | `grep -n 'self\.df\[.*\]\s*='` in pipeline | 1 hit: line 1150 (see P1) |
| `self.df =` reassignment sites | `grep -n 'self\.df\s*='` in pipeline | 9 hits, all in load/pivot/range-extraction paths |
| Call order: `_maybe_pivot()` vs `_apply_mapping_heuristics()` | `grep -n` both symbols in `_ap_load_file`/`_ap_load_sheet` | pivot (line 997/1038) always precedes heuristics (line 1009/1042) |
| `_reject_missing_subject_ids` call sites | `grep -n` across `..._ui.py` + `..._pipeline.py` | 2 real call sites: inside `_detect_wide_format` (ui.py:175) and `_ap_build_analysis_context` (pipeline.py:1108) — **not** in `_ap_apply_mapping_heuristics` |
| `model_type ==` string-dispatch count | `grep -n 'model_type =='` in pipeline | 11 occurrences across 4 formatting functions, all matching bare string literals `"LogisticRegression"`/`"LMM"`/`"ANCOVA"`/`"BetaRegression"` |
| Help-recipe ID drift check | `grep -n '"id":'` in `core/help_content.py` vs literals in pipeline | All currently match (no live drift), but coupling is by string literal, not a shared constant/enum |
| Blocked-result handling coverage | `grep -n "blocked"` in pipeline | Only `result` (single mode) and `lead_result` (multi mode's first DV) are checked — other multi-mode DVs' blocked status is never surfaced |
| `group_hints` substring false-positive | `python3 -c` reproduction (below) | Confirmed: `"ArmLength_mm"`, `"BatchNumber"`, `"Grouping_Note"` all false-positive as "grouping name" via substring match |

Reproduction for the substring false-positive:
```
group_hints = {'group','arm','treatment','condition','sex','gender','cohort','batch','grp'}
'ArmLength_mm' -> True   (contains "arm")
'BatchNumber'  -> True   (contains "batch")
'Grouping_Note'-> True   (contains "group" - intended here, but shows the mechanism)
```

## Findings — severity ranked

### HIGH

**P1 — `_ap_build_analysis_context` permanently mutates `self.df` with a compounding transform, and it runs on every mapping change.** `src/autopilot/statistical_analyzer_autopilot_pipeline.py:1148-1151`:
```python
if _has_boundary:
    # Apply Smithson-Verkuilen transformation to push boundary values inside (0,1)
    self.df[dv_col] = (self.df[dv_col] * (_n - 1) + 0.5) / _n
    context["beta_sv_transformed"] = True
```
This is the Smithson-Verkuilen boundary transform for beta-regression-eligible DVs (proportions with 0/1
boundary values). The bug: `_ap_build_analysis_context` is not a one-shot "commit" step — it is called from
`_ap_on_mapping_changed` (pipeline.py:895, inside a try/except purely to read `context["is_corr_family"]`)
on **every single drag/drop or bucket change**, and again from `_ap_determine_and_run_test` (pipeline.py:1746)
on every "Start Auto Analysis" click, and again from `_ap_configure_plot_from_result`'s context rebuild.
Each call re-reads `self.df[dv_col]`'s *current* min/max, and if a boundary value (exactly 0.0 or 1.0) is
still present it rewrites `self.df[dv_col]` again. Because the transform pulls values toward 0.5, repeated
application over several mapping-change events compounds monotonically toward 0.5, silently degrading the
outcome variable the user thinks they mapped. There is no "already transformed" guard — `context.get("beta_sv_transformed")`
is set on the *context* dict (discarded every call), never checked against `self.df` state before mutating.
**Impact:** A user who maps a proportion DV, tweaks any other bucket (e.g. adds a covariate, changes Factor 2),
and then runs the analysis gets a DV that has been SV-transformed 2+ times instead of once — silently wrong
beta-regression inputs, no error, no warning. This is exactly the "silently produces a wrong-but-plausible
result instead of erroring" anti-pattern class this session has been hunting. **Fix:** never mutate `self.df`
inside a context-building/preview function. Either (a) apply the SV transform on a copy (`analysis_df`, which
the function already builds at line 1102, and pass that copy through `injected_df`) and leave `self.df`
untouched, or (b) if in-place mutation is intentional for downstream consistency, guard it with an idempotency
flag stored on `self` (e.g. `self._sv_transformed_columns: set`) checked before mutating again, and only ever
apply it from the actual "run analysis" path, never from `on_mapping_changed`'s preview call.

### MEDIUM

**P2 — Subject-ID NaN guard added this session does not cover the auto-mapping heuristic path.** `src/autopilot/statistical_analyzer_autopilot_pipeline.py:596-605`:
```python
if subject_column and factor_candidates:
    factor1_col = factor_candidates[0]
    try:
        subject_span = self.df.groupby(subject_column)[factor1_col].nunique(dropna=True)
        if subject_span.max() > 1:
            self.subject_bucket.assign_column(subject_column, _infer_column_kind(self.df[subject_column]))
    except Exception:
        pass  # silently skip if validation fails
```
`_reject_missing_subject_ids` (added in commits 3771884/5723f52/c9a0349) is wired into exactly two call
sites: inside `_detect_wide_format` (`ui.py:175`, only fires when data happens to match the wide-format
shape signature) and inside `_ap_build_analysis_context` (`pipeline.py:1108`, only fires when the user
clicks "Start Auto Analysis"). `_ap_apply_mapping_heuristics` — which runs unconditionally on every file
load (`pipeline.py:1009`) and sheet switch (`pipeline.py:1042`), *before* any analysis is requested — has
its own unguarded `groupby(subject_column)[factor1_col].nunique(dropna=True)` at line 601, wrapped in a bare
`except Exception: pass`. For the common case of **already-long-format** data (most real-world uploads: one
row per measurement, a Subject-ID column, a Group column) with missing subject IDs, `_detect_wide_format`
returns `None` immediately (the long-format shape doesn't match the wide-format signature), so
`_reject_missing_subject_ids` never fires on load. pandas' `groupby` then silently drops the NaN-keyed rows
before computing `nunique`, so `subject_span.max()` is computed from an incomplete subject set — exactly the
footgun this session already fixed twice elsewhere in this same file, just not here. The result: whether
Subject ID auto-assigns to the bucket (and thus whether the auto-pilot even offers RM-ANOVA/LMM routing)
silently depends on a `nunique` computed over rows missing their subject ID, with any failure swallowed by
`except Exception: pass` and no user-visible warning either way.
**Impact:** the auto-mapping heuristic can silently fail to detect a legitimate repeated-measures design (or
succeed with corrupted span counts) when the raw file has incomplete subject IDs — the user only learns
something is wrong if they later happen to click "Start Analysis" and hit the *other* guard's hard error,
by which point they may already have accepted whatever bucket auto-assignment resulted. **Fix:** call
`_reject_missing_subject_ids(self.df, subject_column)` (or a softer warning variant, since this is only a
heuristic, not a hard analysis gate) before line 601's `groupby`, and don't blanket-swallow the resulting
exception — at minimum log it or set a UI hint, rather than `pass`.

**P3 — LMM-upgrade heuristic uses unfiltered `self.df` instead of the filter-applied `analysis_df`, silently ignoring the active row filter, and swallows all errors.** `src/autopilot/statistical_analyzer_autopilot_pipeline.py:1220-1241`:
```python
elif subject_column and context["within_factors"]:
    within_factor = context["within_factors"][0]
    dv_col_for_balance = dv_columns[0] if dv_columns else None
    try:
        counts = self.df.groupby([subject_column, within_factor]).size().unstack(fill_value=0)
        has_structural_missing = (counts == 0).any().any()
        ...
    except Exception:
        pass
```
Every other computation in `_ap_build_analysis_context` (levels, group counts, binary/proportion detection,
between/within role assignment at lines 1166 and 1189) operates on `analysis_df` — the copy with the active
`FilterBucketWidget` row-filter applied (built at pipeline.py:1102-1106). This one block reverts to the raw,
unfiltered `self.df`. If a user has restricted the analysis to a subset of rows via the Filter bucket (e.g.
"OP-Group = 1"), the LMM-vs-RM-ANOVA balance check silently evaluates missingness over the *whole* dataset,
not the filtered subset — it can flag "structural missing" (and upgrade to LMM) or fail to flag it based on
rows the user explicitly excluded from the analysis. Combined with the bare `except Exception: pass`, any
failure here (e.g. a `KeyError` if `within_factor` isn't in `self.df`, which can happen after a filter drops
a level entirely) is invisible — the test silently stays at its prior (possibly wrong) `inferred_test`.
**Impact:** wrong model family selected (RM-ANOVA vs LMM) for filtered analyses with missing visits; user
gets no indication the check even ran into trouble. **Fix:** replace `self.df` with `analysis_df` in this
block (matching the rest of the function), and split the `except Exception: pass` into a narrower catch that
at minimum logs via `logger.debug`/`logger.warning` so a real bug doesn't silently look like "no missingness
detected."

**P4 — Multi-dataset mode never surfaces a blocked result for any non-lead DV column.** `src/autopilot/statistical_analyzer_autopilot_pipeline.py:1786-1816`:
```python
all_results = {}
for dv_column in context["dv_columns"]:
    ...
    all_results[dv_column] = self._execute_single_analysis(per_dv_context, dv_column, output_dir, skip_plots=True)
...
lead_result = all_results[lead_dv]
if lead_result.get("blocked"):
    self._handle_blocked_result(lead_result)
    ...
    return
self._render_result_summary(...)
```
`_handle_blocked_result` — the codepath that surfaces a data-quality block (zero variance, all-NaN group,
too few observations) to the user — is only ever called for the single-mode `result` (line 1776) or the
**first** DV column (`lead_result`, line 1805) in multi mode. If DV column #2 through #N in a multi-dataset
run comes back `blocked` (e.g. one gene column in a panel happens to be constant across all samples), that
blocked dict is stored into `all_results[dv_column]` and passed straight to
`ExportDispatcher.export_multi_dataset_results` and into `self.current_multi_results` with no user-facing
warning at all — silently mixed in among the successful per-DV results. **Impact:** a multi-gene/multi-marker
batch analysis can silently include one or more "blocked" (i.e. non-existent, data-quality-refused) results
that the user has no way to distinguish from a real completed analysis in the UI, only by opening the
combined HTML report and noticing a missing test statistic. **Fix:** after the loop, scan `all_results.values()`
for any `blocked` entries (not just the lead one) and surface them — e.g. append their reasons to the
success/subtitle message, or route through a small non-blocking warning dialog listing which DV columns were
skipped and why, without preventing export of the DVs that did succeed.

### LOW

**P5 — Binary-outcome "grouping name" guard uses unanchored substring matching, producing false positives on plausible clinical column names.** `src/autopilot/statistical_analyzer_autopilot_pipeline.py:1049-1063`:
```python
group_hints = {"group", "arm", "treatment", "condition", "sex",
               "gender", "cohort", "batch", "grp"}
name_is_grouping = any(h in dv_col_name.lower() for h in group_hints)
```
Confirmed by direct execution: `"ArmLength_mm"` and `"BatchNumber"` both match `name_is_grouping = True`
purely because they contain "arm"/"batch" as substrings, not because they're actually a treatment-arm or
batch-id column. A genuine binary clinical measurement named e.g. `"ArmLength_Category"` (Short/Long) would
be excluded from binary-outcome (and thus logistic-regression) auto-detection for the wrong reason.
**Impact:** low — this only affects an auto-detection heuristic (Help Hub recipe suggestion and
logistic-regression auto-routing), not a hard validation gate; a user can still map columns manually. But it
is a silent misclassification with a concrete falsifiable counterexample. **Fix:** switch to word-boundary
matching (e.g. split on non-alphanumeric and check whole tokens) or a stricter regex (`\bgroup\b` etc.)
instead of raw substring containment.

**P6 — Help-Hub recipe IDs are free-floating string literals duplicated between the pipeline and `core/help_content.py`, with no shared constant.** `src/autopilot/statistical_analyzer_autopilot_pipeline.py:268,280,293,309,761-782` vs `src/core/help_content.py` (`"id": "one_way_anova"`, etc.).
Currently all IDs match (verified via `grep` on both files), so this is not a live bug. But the coupling is
by bare string literal in two independently-edited files with no enum/constant and no test asserting the two
lists stay in sync — a future rename of a recipe ID in `help_content.py` would silently break
`_ap_resolve_help_recipe_for_bucket`'s routing (the "i" info-button dialog would just fall back to showing
plain text with no "Suggested recipe" line and no working "Open in Help Hub" button — `ui.py:670-685` — no
exception, no visible error, just a quietly degraded UI). Matches the project's known "String-Coupling &
State Desync" anti-pattern class. **Fix:** define the recipe IDs as a shared enum or module-level constant
list in `core/help_content.py`, imported by the pipeline instead of re-typed as literals; optionally add a
cheap unit test asserting every `help_recipe_id=` / return value in the pipeline exists in `HELP_RECIPES`.

**P7 — `_ap_apply_mapping_heuristics`'s Subject-ID auto-assignment silently no-ops on any exception, including ones unrelated to "validation."** `src/autopilot/statistical_analyzer_autopilot_pipeline.py:604-605`: the comment says "silently skip if validation fails," but the bare `except Exception` also catches genuine bugs (e.g. a `KeyError` from a column-name typo, a `TypeError` from an unexpected dtype) with identical silent behavior. This is a narrower instance of the general fault-swallowing pattern seen throughout the file (15 `except Exception` sites in the pipeline file, see mechanized count above); flagged separately from P2 because the fix is trivial (log at debug level) and doesn't require restructuring the guard logic.
**Fix:** at minimum, `logger.debug(f"Subject-ID auto-assignment heuristic skipped: {exc}")` instead of a bare `pass`, so a real bug shows up in the log file the app already writes (`biomedstatx.log`, surfaced via "Report a Problem" in `statistical_analyzer.py:313-333`) instead of vanishing entirely.

## Strengths (verified)

- **`_reject_missing_subject_ids` is a well-designed, correctly-worded guard where it is wired in.** `ui.py:128-145` has an accurate docstring naming the exact pandas footgun it defends against (`groupby`/`nunique` silently dropping NaN keys), and its `ValueError` message (`"Subject ID column '{subject_col}' has {n_missing} missing value(s)..."`) is precise and matches the condition that triggers it — verified by reading both call sites (`ui.py:175` inside `_detect_wide_format`, `pipeline.py:1108` inside `_ap_build_analysis_context`) and confirming the raised message would actually fire under real missing-ID data.
- **The wide-format detector (`_detect_wide_format`, `ui.py:148-197`) is conservative and well-reasoned**: it requires exactly one subject-like column, 2-8 numeric value columns with `notna().any()` (explicitly guarding against all-NaN columns reaching analysis as an empty group — a comment at `ui.py:178-180` correctly explains why), no 2-level categorical column (to avoid misreading long-format Group columns as wide conditions), and a high subject-uniqueness ratio. Each guard has a comment explaining the discriminating signal, and the function fails closed (`return None`) rather than guessing.
- **The mixin binding architecture (`AutopilotMixin`, `pipeline.py:2274-2327`) is clean and exactly matches what the project's `CLAUDE.md` documents** — module-level `_ap_*` functions bound as class attributes at definition time, restoring MRO/`super()`/static-analysis support versus the deprecated `attach_autopilot_methods` monkey-patch (kept only as a documented, `DeprecationWarning`-emitting shim at `pipeline.py:2330-2341`). No legacy fallback code was reintroduced.
- **Pre-flight bounds validation for transform functions is genuinely good.** `_check_bounds` (`pipeline.py:1290-1300`) checks `log10`/`boxcox` against `min_val <= 0`, `log10(x+1)` against `min_val <= -1`, and `sqrt` against negative values — each with an accurate, actionable `ValueError` message naming the offending column and suggesting a concrete alternative (e.g. "Consider using log10(x+1) instead"). This is exactly the "defensive validation" pattern the project is trying to establish everywhere.
- **The "Single source of truth" `injected_df` pattern is correctly and consistently applied** at both call sites that need it (`_ap_execute_single_analysis` at `pipeline.py:1362-1363` and `_ap_configure_plot_from_result` at `pipeline.py:1886-1887`), each with the same explanatory comment about why re-reading from disk would silently diverge from on-screen state — good self-documentation of a design decision that matters.
- **The data-quality "blocked" result path (`show_block`/`_handle_blocked_result`) is a real, deliberate UX control**, not an afterthought: `ResultCockpitWidget.show_block` (`ui.py:1422-1441`) explicitly blanks every metric card to `"—"` rather than leaving stale/misleading numbers on screen, with a clear rationale in its docstring. (Its one gap — not covering every multi-mode DV — is P4 above, not a design flaw in the mechanism itself.)
- **Operator-precedence and string-coupling bugs already found this session (git history: commits ea940a7, cb45f39, c9a0349) were verified fixed** — `_classify_binary_outcome` (`pipeline.py:1049-1063`) is now a single extracted, directly-testable function used consistently by both the real-routing call site (`pipeline.py:1128`) and the Help-Hub-hint call site (`pipeline.py:731`), eliminating the duplication that caused the earlier inconsistency.

## Recommended remediation order

1. **P1 (self.df mutation bug)** — highest value, cheapest fix: stop mutating `self.df` from a function called on every mapping-change tick; operate on the local `analysis_df` copy instead and pass it through `injected_df` like every other consumer of the context already does. This is a pure correctness fix with no UI/architecture change.
2. **P2 (subject-ID guard gap in auto-mapping heuristic)** — same guard function already exists (`_reject_missing_subject_ids`); this is a one-line call-site addition plus deciding whether to hard-fail or soft-warn from a heuristic (recommend soft-warn, since this path runs before the user has committed to an analysis).
3. **P3 (LMM heuristic uses wrong df)** — one-line fix (`self.df` → `analysis_df`), plus narrowing the `except Exception: pass` to at least log.
4. **P4 (multi-mode blocked-result visibility)** — moderate UI work: extend the post-loop summary to enumerate blocked DVs; no architecture change needed, `_handle_blocked_result`'s reason-extraction logic can be reused per-DV.
5. **P7 (bare except logging)** — trivial, bundle with P2/P3 since they're the same function neighborhood.
6. **P5 (substring-match false positive) and P6 (recipe-ID string coupling)** — lowest urgency; both are heuristic-quality issues with no silent-corruption risk, good candidates for a later cleanup pass rather than a dedicated fix cycle.
