# AUDIT: BioMedStatX @ 3fd4796 — UI-to-Analysis Bridge / Entry Point (Round 2)

**Scope.** `src/analysis/statistical_analyzer.py` (628 lines), `src/autopilot/statistical_analyzer_autopilot_ui.py`
(2453 lines), `src/autopilot/statistical_analyzer_autopilot_pipeline.py` (2341 lines). All three read in full,
line 1 to EOF. This is a **second independent pass** over the same subsystem round 1 audited at `b16cf24`
(`docs/superpowers/audit-notes/release-2.0-audit-round1/01-ui-analysis-bridge.md`). Environment: local
single-user PyQt5 desktop app, no CI, no multi-tenancy — correctness bar is statistical/data validity, not
web-security posture (`references/my-environment.md`).

**Verdict.** No code changed in these three files since round 1 (`git log` shows the last touches are
subject-ID-guard commits `c9a0349`/`5723f55`/`3771884`, all predating and consistent with round 1's report) —
**all four of round 1's findings (P1–P4) are confirmed still open**, verified by direct re-inspection of the
same line ranges. This pass corrects one factual imprecision in round 1 (the P1 "compounding" claim overstates
the mechanism — measured below, it self-limits after one application) and adds **two new, independently
verified findings**: the global crash handler itself throws before it can show the user-facing error dialog
(U1), and multi-DV batch mode computes the RM-ANOVA-vs-LMM decision from only the first DV column and silently
reuses it for every other column in the batch (U2). Nothing here is a live security emergency; the bar that
matters is silently-wrong statistical output and a crash path that fails exactly when it's needed.

## What I mechanically verified (not eyeballed)

| Check | Command | Result |
|---|---|---|
| File sizes | `wc -l` on all 3 files | 628 / 2453 / 2341 lines — all read in full, confirmed unchanged from round-1's line counts |
| Git history on these 3 files | `git log --oneline -3 -- <files>` | Last 3 commits touching them: `c9a0349`, `5723f55`, `3771884` (subject-ID guard work) — nothing since round 1's audit commit |
| Round-1 P1 status (self.df mutation) | Re-read `pipeline.py:1145-1155` verbatim | Byte-identical to round 1's quote — **unfixed** |
| Round-1 P2 status (heuristic subject-ID guard gap) | Re-read `pipeline.py:596-606` verbatim | Byte-identical — **unfixed** |
| Round-1 P3 status (LMM check uses unfiltered `self.df`) | Re-read `pipeline.py:1220-1241` verbatim | Byte-identical — **unfixed** |
| Round-1 P4 status (multi-mode blocked-result visibility) | Re-read `pipeline.py:1786-1817` verbatim | Byte-identical — **unfixed** |
| P1 mechanism check — is the SV transform actually unbounded/compounding? | `python3` numeric simulation, 3 repeated calls of `(x*(n-1)+0.5)/n` starting from boundary values 0.0/1.0 | **Correction to round 1**: after exactly 1 application, `_min`/`_max` are strictly interior (e.g. 0.0625/0.9375 for n=8), so `_has_boundary` (`== 0.0 or == 1.0`) is `False` on every subsequent call — the mutation is idempotent-after-first-hit, not unboundedly compounding. Still a real, undesired, silent side effect from a "build context" function — see U-corrected-P1 below. |
| `injected_df` filter re-application (traced end to end) | Read `analysis_core.py:208-233` | `filter_spec` and `selected_groups` ARE correctly re-applied server-side against the raw `injected_df` before the real test runs (`analysis_core.py:224-233`) — this specific concern is **not** a bug; see note under U2 |
| `inferred_test` dispatch fidelity | Read `analysis_core.py:264-269` | `analysis_context.get("inferred_test")` is passed through verbatim as `local_kwargs["test"]` — no re-derivation at the dispatch layer, confirming any upstream misclassification (P3, U2) reaches the actual model dispatch, not just a UI label |
| Global exception hook — reproduced the crash | `python3` repro of `logger.info("%s %s", msg, file=sys.stderr)` against stdlib `logging` | Raises `TypeError: Logger._log() got an unexpected keyword argument 'file'` every time; confirmed `src/core/logger_config.py` uses vanilla `logging.StreamHandler`/`RotatingFileHandler`, no custom `Logger` subclass that would tolerate `file=` |
| Bare `except:` count | `git grep -n "except:"` per file | `statistical_analyzer.py`: 1 (line 617); other two: 0 (matches round 1) |
| `except Exception` count | `git grep -c "except Exception"` per file | `statistical_analyzer.py`: 11; `..._ui.py`: 4; `..._pipeline.py`: 15 (matches round 1) |
| `self.df[...] =` in-place mutation sites | `git grep -n 'self\.df\[.*\] *='` in pipeline+ui | 1 hit: `pipeline.py:1150` (the P1 mutation) — no other in-place column mutation exists in either file |
| `dict(context)` shallow-copy sites | `git grep -n "dict(context)"` in pipeline | 4 sites: `1356` (single-DV), `1789` (per-DV in multi-mode loop), `1802` (lead DV), `1880` (plot reconfigure) — `1789` is the site of U2: `inferred_test` is never recomputed per DV, only shallow-copied from the once-computed `context` |
| Multi-mode `inferred_test` recomputation check | Read `pipeline.py:1786-1793`, confirm `dv_col_for_balance = dv_columns[0]` at `pipeline.py:1223` | Confirmed: LMM-vs-RM-ANOVA balance check (which decides `inferred_test`) runs exactly once per "Start Analysis" click, keyed to `dv_columns[0]` only; the loop at `1788-1793` reuses that single `inferred_test` string for every subsequent DV column via `dict(context)` |
| `_reject_missing_subject_ids` call-site count | `git grep -n "_reject_missing_subject_ids"` across both files | 2 real call sites — `ui.py:175` (`_detect_wide_format`) and `pipeline.py:1108` (`_ap_build_analysis_context`) — matches round 1; still not called from `_ap_apply_mapping_heuristics` (P2 still open) |
| Window geometry: `resize`/`move` then unconditional `setGeometry` | Read `statistical_analyzer.py:109-115` | `self.resize(width, height)` + `self.move(...)` (screen-relative, 72% of primary screen, centered) computed then immediately overwritten by unconditional `self.setGeometry(100, 50, 1600, 1300)` two lines later — dead computation; masked in practice by `window.showMaximized()` at startup (`statistical_analyzer.py:624`), but visible if the user un-maximizes |

## Findings — severity ranked

### HIGH

**U1 — The global uncaught-exception handler itself raises a `TypeError`, so the user-facing crash dialog never appears.** `src/analysis/statistical_analyzer.py:501`, inside `_install_global_excepthook`'s `_excepthook`, which is assigned directly to `sys.excepthook` at line 514 (the last line of defense for anything not caught elsewhere):
```python
logger.info("%s %s", msg, file=sys.stderr)
```
Reproduced directly against this app's actual logging setup (`src/core/logger_config.py` — a vanilla `logging.getLogger()` root logger with `StreamHandler`/`RotatingFileHandler`, no custom `Logger` subclass):
```
TypeError: Logger._log() got an unexpected keyword argument 'file'
```
`Logger.info(msg, *args, **kwargs)` forwards `**kwargs` straight into `Logger._log()`, which has no `file` parameter — that kwarg belongs to `print()`, not `logging`. This line is not wrapped in its own `try/except` (unlike the file-write two lines above it and the dialog-show block below it, both of which do have their own `except Exception: pass`), so the `TypeError` propagates out of `_excepthook` itself. When an exception hook raises, CPython's runtime prints a separate "Error in sys.excepthook" traceback to the real stderr and swallows the rest of the handler — meaning **line 503's `QMessageBox.critical(...)` never executes for any uncaught exception, ever.**
**Impact:** the crash-log file write at lines 495-500 does still succeed (it's self-protected), so `crash_log.txt` retains a record — but the entire point of this handler, showing the user a "Ein Fehler ist aufgetreten..." dialog with next steps, is dead code that has never fired since this was written. In a PyInstaller-frozen build with no attached console (the normal end-user launch path), an uncaught exception anywhere outside Qt's event loop (e.g. during `StatisticalAnalyzerApp.__init__`, or in any non-Qt thread) currently just makes the app vanish with zero visible explanation to the user, who has no console to see the "Error in sys.excepthook" fallback either.
**Fix:** delete the stray `file=sys.stderr` kwarg and collapse the format string to one `%s` (`logger.info("%s", msg)`), or switch to `logger.error(msg)` (arguably the correct level for a crash anyway — `info` under-signals severity for an uncaught exception in the log file too). Add a regression test that calls `_excepthook` with a synthetic exception and asserts `QMessageBox.critical` was invoked (mock the dialog) — this exact class of bug (an unguarded statement between two guarded ones) is easy to reintroduce silently.

### MEDIUM

**U2 — Multi-dataset mode computes the LMM-vs-RM-ANOVA test decision once from `dv_columns[0]` only, then silently reuses it for every other DV column in the batch.** `src/autopilot/statistical_analyzer_autopilot_pipeline.py:1223` (decision) and `:1788-1793` (reuse):
```python
# inside _ap_build_analysis_context, runs ONCE per "Start Analysis" click:
dv_col_for_balance = dv_columns[0] if dv_columns else None
...
if has_structural_missing or has_nan_missing:
    context["inferred_test"] = "lmm"
...
# inside _ap_determine_and_run_test, multi-mode loop:
for dv_column in context["dv_columns"]:
    per_dv_context = dict(context)          # shallow copy — inferred_test carried over verbatim
    per_dv_context["dv_columns"] = [dv_column]
    per_dv_context["current_dv"] = dv_column
    all_results[dv_column] = self._execute_single_analysis(per_dv_context, dv_column, output_dir, skip_plots=True)
```
`_ap_build_analysis_context` is called exactly once (`pipeline.py:1746`, before the multi-mode loop starts), and its Case-1/Case-2 missingness check at lines 1223-1236 only ever inspects `dv_columns[0]`'s NaN pattern (`dv_col_for_balance = dv_columns[0]`). The resulting `context["inferred_test"]` (`"lmm"` or whatever it was before) is then baked into every `per_dv_context` via `dict(context)` for the rest of the multi-DV loop — there is no per-DV re-evaluation. Confirmed this string reaches the actual model dispatch unmodified: `analysis_core.py:265-269` takes `analysis_context.get("inferred_test")` verbatim as `local_kwargs["test"]`, with no re-derivation at that layer either.
**Impact:** in a legitimate multi-gene/multi-marker repeated-measures panel (Subject ID + a within-factor, multiple DV columns), if column 1 (e.g. Gene_A) happens to have complete visits for every subject but column 3 (e.g. Gene_C) has some missing measurements (or vice versa), the whole batch is forced through whichever test column 1's missingness pattern happened to select — an RM-ANOVA run on a DV that actually has missing visits (biased/less-efficient handling of the imbalance), or an unnecessary LMM run on a perfectly balanced DV. This is silent: nothing in the UI or export indicates the decision was made from a different column than the one being analyzed.
**Fix:** move the missingness check inside the per-DV loop (or into `_ap_execute_single_analysis`, which already receives `dv_column` and rebuilds `single_context`), keyed to the DV column actually being analyzed in that iteration, rather than deciding once from `dv_columns[0]` for the whole batch.

### LOW

**U3 (correction to round-1 P1) — the Smithson-Verkuilen mutation is a one-shot silent side effect, not an unboundedly compounding one, but the underlying design flaw (mutating `self.df` from a "build context" function invoked on every mapping-change tick) is still real and still open.** `src/autopilot/statistical_analyzer_autopilot_pipeline.py:1148-1151`. Round 1's P1 characterized this as compounding "monotonically toward 0.5" across repeated mapping-change events; direct numerical simulation here (3 repeated calls of `(x*(n-1)+0.5)/n` starting from an array containing exact 0.0/1.0 boundary values) shows `_has_boundary` (`_min == 0.0 or _max == 1.0`) evaluates `False` after exactly one application, because the transformed boundary values are now strictly interior — so the mutation does not keep compounding across dozens of mapping-change ticks as the original framing suggested. It is still a real bug: `_ap_build_analysis_context` — called from `on_mapping_changed` (fired on every bucket drag/drop, covariate change, or toggle — `pipeline.py:895`) as well as from the actual "Start Analysis" click (`pipeline.py:1746`) — silently rewrites `self.df[dv_col]` in place on the *first* occasion a boundary-containing proportion DV is mapped, with no user confirmation and no "already transformed" flag checked against `self.df` itself (only against the discarded-per-call `context` dict). A user who maps a proportion DV, notices nothing, then unmaps it and remaps a different column as DV would find the first column has already been silently permanently altered in the in-memory table (and in the live preview) the moment mapping merely *looked* like beta-regression-eligible — before "Start Analysis" was ever clicked.
**Impact:** same root cause as round-1 P1, just a corrected magnitude — one silent, irreversible (within the session; only a file reload undoes it) mutation of the user's data from what should be a pure preview/heuristic path, not a progressively worsening one. Still worth the same fix.
**Fix:** unchanged from round 1 — operate on `analysis_df` (already built at `pipeline.py:1102`) and thread the transformed copy through `injected_df`, leaving `self.df` untouched until an actual "Start Analysis" commit; or gate the mutation behind an explicit self-state flag checked before writing, and only trigger it from `_ap_determine_and_run_test`, never from `on_mapping_changed`'s preview call.

**U4 — `resize`/`move` screen-relative window sizing is dead code, immediately overwritten by a fixed 1600×1300 `setGeometry`.** `src/analysis/statistical_analyzer.py:103-115`:
```python
width = int(_sw * 0.72)
height = int(_sh * 0.72)
self.resize(width, height)
self.move((_sw - width) // 2, (_sh - height) // 2)
self.setWindowTitle(...)
self.setGeometry(100, 50, 1600, 1300)   # overwrites the resize/move above unconditionally
```
The screen-relative sizing (72% of the primary screen, centered) is computed and applied, then two lines later unconditionally discarded by a fixed `setGeometry(100, 50, 1600, 1300)`. In practice this is masked because `__main__` calls `window.showMaximized()` immediately after construction (`statistical_analyzer.py:624`), so most users never see the intermediate un-maximized geometry. But if the user un-maximizes the window later in the session, it reverts to the fixed 1600×1300 at (100, 50) rather than the screen-relative size — on a laptop display smaller than 1700×1350 (with taskbar/menu-bar chrome), part of the window can render off-screen or require manual repositioning.
**Impact:** low — cosmetic, recoverable by the user (drag/resize), and mostly masked by the immediate maximize. Not a data-correctness issue.
**Fix:** delete the `setGeometry(100, 50, 1600, 1300)` call (the `resize`/`move` above it already does the intended job), or make it conditional (`if not screen-relative computation succeeded`).

**U5 — Bare `except:` around stylesheet loading swallows all exceptions including `KeyboardInterrupt`.** `src/analysis/statistical_analyzer.py:614-619`:
```python
try:
    stylesheet = _load_auto_pilot_stylesheet()
    logger.info(...)
except:
    stylesheet = ""
    logger.info("No stylesheet found")
```
Confirmed via `git grep -n "except:"` — this is the only bare `except:` across all three files (matches round 1's count). Bare `except:` catches `BaseException`, including `KeyboardInterrupt`/`SystemExit`, not just stylesheet-loading errors. Worst-case impact here is cosmetic (an unstyled but functional app), so this is LOW severity in this environment, but it's a two-line fix while already touching the neighboring U1 crash-handler code.
**Fix:** narrow to `except Exception:`.

## Strengths (verified)

- **Round 1's confirmed-fixed subject-ID work is holding.** `_reject_missing_subject_ids` (`ui.py:128-145`) remains correctly wired at both its call sites (`ui.py:175` inside `_detect_wide_format`, `pipeline.py:1108` inside `_ap_build_analysis_context`), and its `ValueError` message is still accurate and specific. No regression since round 1.
- **`analysis_core.py`'s server-side re-application of `filter_spec` and `selected_groups` against `injected_df` is correct and thorough** (`analysis_core.py:208-233`) — verified by tracing the full path from `single_context["injected_df"] = self.df` (raw, unfiltered) at `pipeline.py:1363` through to `working_df` re-filtering at the analysis-core layer. This closes what could otherwise have been a serious filter-bypass bug: the actual statistical computation always sees the correctly filtered/subset data, even though the raw `self.df` is what gets injected. Only the *test-family decision* (P3/U2), not the *computation itself*, is affected by unfiltered-data bugs in this subsystem.
- **The mixin binding architecture (`AutopilotMixin`, `pipeline.py:2274-2327`) remains clean and matches `CLAUDE.md`'s documented architecture exactly** — module-level `_ap_*` functions bound as class attributes at definition time, the deprecated `attach_autopilot_methods` monkey-patch kept only as a `DeprecationWarning`-emitting shim (`pipeline.py:2330-2341`). No legacy fallback code has been reintroduced since round 1.
- **`_detect_wide_format`'s guards remain well-reasoned and unchanged** (`ui.py:148-197`): exactly one subject-like column, 2-8 numeric value columns with an explicit `notna().any()` guard against all-NaN columns reaching analysis as a silently-empty group, a 2-level-categorical exclusion to avoid misreading long-format Group columns as wide conditions, and a high-uniqueness-ratio discriminator. Fails closed (`return None`) rather than guessing.
- **Pre-flight bounds validation for variable transforms is genuinely solid** (`_check_bounds`, `pipeline.py:1290-1300`): accurate per-transform domain checks (`log10`/`boxcox` vs. `<= 0`, `log10(x+1)` vs. `<= -1`, `sqrt` vs. negative) with actionable error messages naming the offending column and a concrete alternative. Verified these fire under exactly the condition each message claims.
- **The range-selection dialog's cross-design validation (`_on_apply`, `ui.py:2403-2450`) is thorough and correctly scoped per design mode** — paired-design block-height mismatch, bivariate X/Y count mismatch, and the "no assignment at all" guard are each checked with an accurate, specific warning dialog before `accept()` is allowed, preventing several classes of malformed coordinate-extraction input from ever reaching `extract_from_coordinates`/`extract_paired_from_coordinates`.
- **The dispatch-layer `inferred_test` pass-through is simple and traceable** — `analysis_core.py:265-269` takes the upstream string verbatim with no silent re-interpretation, which made it possible to verify with certainty that P3/U2's upstream misclassifications genuinely reach the real dispatch rather than being caught and corrected downstream. A more defensive design might have hidden this; this one is at least honestly traceable end to end.

## Recommended remediation order

1. **U1 (excepthook itself crashes)** — highest value, cheapest fix (one line: drop `file=sys.stderr`, fix the format string). This silently disables the app's entire crash-visibility mechanism for end users; fix before anything else in this batch.
2. **Round-1 P1 / U3 (self.df mutation in build-context)** — same fix round 1 already specified: operate on `analysis_df`/`injected_df`, never mutate `self.df` from a function invoked on every mapping-change tick. Still the highest-value *data-correctness* fix outstanding from round 1.
3. **Round-1 P3 (LMM heuristic uses unfiltered `self.df`)** — one-line fix (`self.df` → `analysis_df` at `pipeline.py:1226/1232`), bundle with U2 since both are in the same missingness-detection neighborhood and both concern the LMM-vs-RM-ANOVA decision being computed from the wrong scope of data.
4. **U2 (multi-DV mode reuses one DV's test decision for all DVs)** — move the missingness check into the per-DV loop; natural to fix alongside P3 since it's the same block, just needs to run once per DV instead of once per batch.
5. **Round-1 P2 (subject-ID guard gap in heuristic path) and P4 (multi-mode blocked-result visibility)** — both still open, both already have a clear fix specified in round 1's report; no new information changes that guidance.
6. **U4 (dead geometry code) and U5 (bare except)** — trivial, bundle into the same PR as U1 since both are in the immediately surrounding code in `statistical_analyzer.py`.
7. **Round-1 P5/P6/P7 (substring false-positive, help-recipe string coupling, heuristic except-swallow)** — lowest urgency, unchanged from round 1's assessment; good candidates for a later cleanup pass.
