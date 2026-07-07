# AUDIT: BioMedStatX — Statistical Core Dispatch @ 3fd4796

**Scope:** `src/analysis/analysis_core.py` (1,592 lines), `src/analysis/statisticaltester.py`
(2,915 lines), `src/analysis/clinical_models.py` (2,040 lines) — 6,547 lines total, all read in
full, sequential chunks. Second independent audit pass (round 2) of this subsystem — a prior
pass added 6 `ModelDesignError` pre-flight checks to `clinical_models.py` and fixed
`control_group` wiring through ANCOVA. This pass re-verifies those fixes end-to-end (both
dispatch paths) and hunts for new issues / recurrences of the known bug classes (defensive-
validation deficits, fault swallowing, string-coupling, implicit-logic violations, writer/reader
key-contract drift, "computed but never wired to the decision path").

**Verdict.** The two prior fixes hold: all 6 `ModelDesignError` pre-flight checks fire correctly
(verified by direct repro, not just inspection), and `control_group` reaches the vs-control EMM
post-hoc path from **both** dispatch entry points (`analysis_core.py` direct dispatch and
`advanced_pipeline.py` → `statisticaltester.py`). However, this pass found one **live, 100%
reproducible crash-message bug** reachable from the primary UI flow (SC1 — a broken
`make_blocked_result()` call signature that fires whenever Mixed/Two-Way/RM-ANOVA hits an
invalid-design or prep-error path, silently replacing the intended user-facing message with a
Python `TypeError` string), a **recurrence of the known Mixed-ANOVA sphericity-correction bug**
in a fresh location (`statisticaltester.py`'s pingouin Mixed ANOVA branch — GG/HF-corrected
p-value computed and nested in the results dict, never overwrites the canonical `p_value` that
gates the verdict), and a **writer/reader key-contract drift** between ANCOVA's and LMM's EMM
post-hoc dict shapes that leaves ANCOVA's t-statistic column silently blank in the HTML report.
None of these are exploitable/security issues (single-user desktop app) — they are correctness
and error-surfacing bugs that produce a misleading report or a confusing message instead of the
intended one. No live data-loss or crash-that-loses-work was found.

## What I mechanically verified (not eyeballed)

| Check | Command / method | Result |
|---|---|---|
| Line counts match task brief | `wc -l` on all 3 files | 1592/2915/2040 = 6,547 total, all read start-to-end |
| All 6 `ModelDesignError` pre-flight checks fire | Direct Python repro: `ANCOVAModel.fit()` with empty `between_factors`/`covariates`; `LinearMixedModel.fit()` with empty `fixed_effects`/`random_intercept=None`; `LogisticRegressionModel.fit()` with empty `predictors`/>2 outcome levels | All 6 raise `ModelDesignError` with the exact claimed message (script output quoted below) |
| `control_group` reaches vs-control EMM from **both** dispatch paths | `git grep -n "control_group="` on `analysis_core.py` and `advanced_pipeline.py` | Path 1 (`analysis_core.py:611,636`) and Path 2 (`advanced_pipeline.py:189,208`) both resolve `primary_levels` via callback and pass `control_group=` into `.fit()` — both reach `ANCOVAModel.emm_contrasts(method="vs_control", ...)` / `LinearMixedModel.emm_contrasts(...)` |
| `make_blocked_result()` actual signature vs. call sites | `git grep -n "make_blocked_result("` across `analysis_core.py`; compared to `def make_blocked_result(reason, *, code, details=None, warnings=None)` at `statisticaltester.py:45` | 4 of 8 call sites in `analysis_core.py` (lines 914, 921, 960, 1005) pass a `test_name=` kwarg that does not exist on the function → confirmed `TypeError` at every one of those 4 sites (SC1) |
| SC1 live reproduction through the public `AnalysisManager.analyze()` entry point | End-to-end repro: `test="mixed_anova"`, `additional_factors=[]` (triggers the "requires two factors" block at line 914) | `result["block_code"] == "UNHANDLED_EXCEPTION"`, `result["block_reason"] == "StatisticalTester.make_blocked_result() got an unexpected keyword argument 'test_name'"` — the intended message ("Mixed ANOVA requires two factors...") and code (`INVALID_DESIGN`) never reach the user |
| Mixed-ANOVA sphericity correction wiring (recurrence check) | Read `statisticaltester.py:1510-1706` (pingouin Mixed ANOVA branch) in full; traced `results["p_value"]` and `results["factors"][i]["p_value"]` assignments | Both are set from `row[p_col]` (`p_unc`, **uncorrected**) at lines 1523 and 1548; the GG/HF `final_p_value` computed at line ~1706 via `_test_mixed_anova_within_sphericity` only reaches a nested `results["within_sphericity_corrections"]["main_effect"]["final_p_value"]` key — never overwrites `p_value` (SC2, same bug class as the previously-documented Mixed ANOVA case, confirmed present a second time in this file) |
| Contrast this with plain RM-ANOVA (claimed-fixed sibling path) | Read `statisticaltester.py:1966-1974` | RM-ANOVA **does** have the fix: `if sphericity_results.get("final_p_value") is not None: results["p_value"] = sphericity_results["final_p_value"]` (comment tags it "E1"). Confirms the fix was applied to RM-ANOVA specifically but not replicated to the Mixed-ANOVA sibling code path |
| ANCOVA vs. LMM EMM contrast dict key parity | Read `clinical_models.py:342-354` (ANCOVA `emm_contrasts`) vs. `:994-1009` (LMM `emm_contrasts`); cross-checked against `report_stat_rows.py:744` (`comp.get("statistic")`) | ANCOVA writes `"t"`/`"se"`; LMM writes `"statistic"`/`"std_err"`. Export layer reads `comp.get("statistic")` — resolves for LMM, returns `None`/blank for ANCOVA (SC3) |
| Whether `_standardize_results` back-fills the missing `"statistic"` key on the primary ANCOVA path | `git grep -n "_standardize_results"` in `analysis_core.py` | Zero hits — the primary clinical-model dispatch (`analysis_core.py:594-612`) calls `model.as_results_dict()` directly and never routes through `_standardize_results`, so the key is simply absent, not defaulted; the secondary path (`statisticaltester.py:1444`) does call `_standardize_results`, which would backfill `None` — same net effect (blank statistic) via a different mechanism |
| `posthoc_choice` variable-scoping repro | Isolated the exact control-flow shape of `analysis_core.py:1085-1196` in a standalone script, driving the non-parametric branch (line 1104) with an already-empty `pairwise_comparisons` on `test_results` | `NameError: cannot access local variable 'posthoc_choice' where it is not associated with a value` — confirmed reachable when the non-parametric post-hoc re-entry branch runs and a `dict`-shaped `posthoc_results` comes back (SC4) |
| `except Exception`/bare `except:` density | `git grep -c "except Exception"` per file | `analysis_core.py`:15, `statisticaltester.py`:27, `clinical_models.py`:27. Bare `except Exception:` (no bound variable): 4 / 8 / 15 respectively — all inspected; all are bounded-blast-radius diagnostic/display fallbacks (ICC, Hosmer-Lemeshow, calibration curve, VIF, CI) that degrade to `None`/empty rather than crashing, no silent data-loss found among them |
| Second `AdvancedPostHocEngine` call for ANCOVA/LMM/Logistic Regression (via `advanced_pipeline.py`) | Read `advanced_posthoc.py:43-153` (`_run_advanced_parametric_posthoc`) | `test` is only branched for `two_way_anova`/`mixed_anova`/`repeated_measures_anova`; for `ancova`/`lmm`/`logistic_regression` it falls to `group_names = []` → the call is a no-op (returns empty `pairwise_comparisons`), and the caller's `if advanced_posthoc_updates.get("pairwise_comparisons")` guard (line 254 of `advanced_pipeline.py`) prevents it from overwriting the model's own EMM contrasts — wasted work, not a correctness bug (SC5, LOW) |
| `ancova`/`lmm`/`logistic_regression` reachability via `advanced_pipeline.py` from the UI | `git grep -n "perform_advanced_test\b"` (all call sites) | Only called from `analysis_core.py` for `mixed_anova`/`two_way_anova`/`repeated_measures_anova` (lines 924, 963, 1008). The `ancova`/`lmm`/`logistic_regression` branches in `advanced_pipeline.py` (lines 175-213) are unreachable from the current autopilot UI (those tests are dispatched directly inside `analysis_core.py`'s clinical-model block, which always `return`s before reaching `perform_advanced_test`) — but they **are** reachable via the public `StatisticalTester.perform_advanced_test()` API and are covered by `tests/test_golden_r_advanced.py::test_golden_lme4_lmm` (SC6, contract-drift bug in dead-for-UI-but-live-for-API code) |
| Error-shape contract drift on that dead-for-UI path | Read `statisticaltester.py:1438-1467` (`_run_ancova`/`_run_lmm`/`_run_logistic_regression`) and `advanced_pipeline.py:453-459` (outer `except ValidationError`) | Both catch `ModelDesignError`/`ValidationError` and return a bare `{"error": ..., "test": ...}` dict with no `blocked=True`/`block_code` — contrasts with the primary path's `make_blocked_result()` contract that the UI's `result.get("blocked")` check (`statistical_analyzer_autopilot_pipeline.py:1776,1805`) depends on (SC6) |

## Findings — severity ranked

### HIGH

**SC1 — `make_blocked_result()` called with a nonexistent `test_name=` kwarg; crashes with a
`TypeError` that is caught and replaces the intended user-facing message.**
`src/analysis/analysis_core.py:914,921,960,1005`. The function is defined as
`make_blocked_result(reason, *, code, details=None, warnings=None)` (`statisticaltester.py:45`)
— there is no `test_name` parameter. All 4 call sites pass `test_name=...`:
```python
return StatisticalTester.make_blocked_result(code="INVALID_DESIGN", reason="Mixed ANOVA requires two factors (between and within)", test_name="mixed_anova")
...
return StatisticalTester.make_blocked_result("PREP_ERROR", prep["error"], test_name=kwargs.get('test', 'unknown_test'))
```
This is inside the same outer `try` block that ends in `except Exception as e:` at line 1564, so
the `TypeError` is caught and converted into a *different* blocked result
(`code="UNHANDLED_EXCEPTION"`), but the `block_reason` becomes the Python error message itself.
**Verified end-to-end**: calling `AnalysisManager.analyze(test="mixed_anova", additional_factors=[])`
returns `block_code="UNHANDLED_EXCEPTION"`, `block_reason="StatisticalTester.make_blocked_result()
got an unexpected keyword argument 'test_name'"` — the user never sees "Mixed ANOVA requires two
factors (between and within)" or the `INVALID_DESIGN` code. Same failure mode hits any
`prep["error"]` from `prepare_advanced_test` for `mixed_anova`/`two_way_anova`/
`repeated_measures_anova` (lines 921/960/1005), so a legitimate prep-time validation message
(e.g. an unbalanced RM design) is also replaced by this confusing internal error.
**Impact:** every "invalid design" and "prep error" block for the three advanced ANOVA types
loses its actual diagnostic message and reports a Python signature error instead — a real
regression in error-message quality on a path a user can trigger by simply mis-configuring
factors in the UI.
**Fix:** drop the `test_name=` kwarg from all 4 call sites (the function has no use for it —
`test_name` isn't part of the blocked-result schema at all); if the test name should be recorded,
add it via `details={"test_name": ...}` instead, matching the function's actual signature.

### MEDIUM

**SC2 — Mixed ANOVA's Greenhouse-Geisser/Huynh-Feldt sphericity correction is computed but never
overwrites the canonical `p_value`/`factors[i]["p_value"]` that gates the significance verdict —
recurrence of the documented bug, found in a second location.**
`src/analysis/statisticaltester.py:1510-1706` (pingouin Mixed ANOVA branch, `_run_mixed_anova`).
The within-factor and interaction rows in `results["factors"]`/`results["interactions"]` are
populated from `row[p_col]` where `p_col` is `p_unc`/`p-unc` — the **uncorrected** p-value
(lines 1519-1528, 1546-1548). `results["p_value"]` (the field `analysis_core.py:1087`
`test_results['p_value'] < 0.05` reads to decide significance and trigger post-hoc) is likewise
set from the **interaction's uncorrected** p-value at line 1548. Only afterward, at line
1703-1706, does `_test_mixed_anova_within_sphericity` compute the GG/HF-corrected
`final_p_value` — but it is nested three levels deep at
`results["within_sphericity_corrections"]["main_effect"]["final_p_value"]` and is never written
back to `results["p_value"]` or `results["factors"][i]["p_value"]`.
Contrast: plain RM-ANOVA (`statisticaltester.py:1972-1974`) has the explicit fix — `if
sphericity_results.get("final_p_value") is not None: results["p_value"] = sphericity_results
["final_p_value"]` (tagged "E1" in a comment) — proving the fix pattern is known and applied to
one sibling path but not propagated to Mixed ANOVA's pingouin branch.
**Impact:** when within-subject sphericity is violated in a Mixed ANOVA (common with ≥3
repeated-measures levels), the significance verdict, the post-hoc trigger
(`analysis_core.py:1087`), and the displayed p-value in `results["factors"]`/interactions can all
be **anti-conservative** (using the inflated-Type-I-error uncorrected p-value) even though the
correctly-corrected p-value sits unused in the same dict. This matches the exact "computed,
displayed, never gates the decision" bug class already documented for this codebase, now
independently reconfirmed via full re-read rather than assumed from memory.
**Fix:** apply the same pattern used in RM-ANOVA (line 1972-1974) to the Mixed ANOVA branch:
after `within_sphericity_results` is merged into `results`, if a `final_p_value` is present for
the within-factor (and/or interaction, if it also involves the within factor), overwrite the
corresponding `results["factors"][i]["p_value"]` entry and, if that factor/interaction is the one
driving `results["p_value"]`, overwrite the top-level field too.

**SC3 — ANCOVA's EMM pairwise-comparison dicts use `"t"`/`"se"` while LMM's use `"statistic"`/
`"std_err"`; the HTML export layer reads `"statistic"`, silently blanking ANCOVA's t-column.**
`src/analysis/clinical_models.py:342-354` (`ANCOVAModel.emm_contrasts`) vs. `:994-1009`
(`LinearMixedModel.emm_contrasts`). ANCOVA's contrast dicts have keys `group1, group2, estimate,
se, t, df, p_value, significant` — no `"statistic"`, `"test"`, `"corrected"`, or `"correction"`
key. LMM's contrast dicts have `group1, group2, estimate, std_err, statistic, p_value,
significant, test, df, corrected, correction`. The export layer
(`src/export/report_stat_rows.py:744`) reads `comp.get("statistic")` to render the pairwise-
comparison table's statistic column. Verified the primary ANCOVA dispatch path
(`analysis_core.py:594-612`) never calls `_standardize_results` (which would otherwise backfill
`comp["statistic"] = None`) — `git grep -n "_standardize_results" src/analysis/analysis_core.py`
returns zero hits — so the key is simply absent on that path; the secondary path
(`statisticaltester.py:1444`) does call `_standardize_results`, producing the same net blank
value via backfill instead of absence.
**Impact:** the HTML report's pairwise-comparisons table shows a blank/N-A statistic column for
every ANCOVA vs-control or pairwise EMM contrast, while the identical table for LMM correctly
shows the t-statistic. No error or warning surfaces anywhere — a silent report-layer gap that a
user reviewing the ANCOVA output would have no way to notice unless they already expected a
number there.
**Fix:** rename ANCOVA's `"t"` → `"statistic"` and `"se"` → `"std_err"` in
`clinical_models.py:347-349` to match LMM's (and the export layer's) key contract; add the
missing `"test"`/`"corrected"`/`"correction"` keys for consistency (LMM already sets
`correction="multivariate-t" if method=="vs_control" else "Holm-Bonferroni"` — ANCOVA's
`posthoc_label` string encodes the same information but not as a structured field per comparison).

**SC4 — `posthoc_choice` referenced before assignment when the non-parametric post-hoc
re-entry branch runs; raises `NameError` and aborts the whole analysis.**
`src/analysis/analysis_core.py:1085-1196`. `posthoc_choice` is only ever assigned at line 1116,
inside the `else:` branch (parametric dialog path) of the `if 'kruskal' in test_name or
'friedman' in test_name or test_recommendation == 'non_parametric':` conditional at line 1104.
Line 1196, `if posthoc_choice == "dunnett" and "control_group" in posthoc_results:`, sits at the
same indentation as both branches (a sibling `if posthoc_results:` block, confirmed by column
offset: line 1102's `if not test_results.get('pairwise_comparisons'):` opens the parent scope
that contains both the `if`/`else` at 1104/1114 and the `if posthoc_results:` at 1195). Because
`posthoc_choice` is assigned *somewhere* in the enclosing function, Python treats it as a local
for the whole function body — so reading it before the `if` branch (non-parametric) runs raises
`NameError: cannot access local variable 'posthoc_choice' where it is not associated with a
value` (confirmed by isolating the exact control-flow shape in a standalone repro).
This is reachable specifically when: (a) the main test is non-parametric with a significant
result and ≥3 groups, (b) `test_results.get('pairwise_comparisons')` is empty at the point
`analysis_core.py` reaches its own post-hoc block (this happens whenever the earlier, separate
post-hoc dispatch inside `perform_statistical_test`/`_stat_test_multi_groups`
(`statisticaltester.py:744-795`) didn't populate it — e.g. the user cancels that first dialog, or
it errors), and (c) the *second* `perform_refactored_posthoc_testing` call inside
`analysis_core.py`'s own non-parametric branch (line 1106) succeeds and returns a non-empty dict.
**Impact:** a hard crash (uncaught inside this specific block, though caught by the file-level
`except Exception` at line 1564 and converted to an `UNHANDLED_EXCEPTION` block with a confusing
`NameError` message) instead of the intended silent no-op (the `if posthoc_choice ==
"dunnett"` check was clearly meant to be a no-op for the non-parametric branch, where Dunnett's
control-group key never applies).
**Fix:** initialize `posthoc_choice = None` alongside `posthoc_results = None` at the top of the
block (line 1085), or move the `if posthoc_choice == "dunnett"` check inside the `else:` branch
where `posthoc_choice` is actually meaningful (the non-parametric branch has no concept of
"dunnett" as a `posthoc_choice` value in the first place, since that variable name is exclusive
to the parametric dialog flow).

### LOW

**SC5 — `AdvancedPostHocEngine` is invoked a second, no-op time for ANCOVA/LMM/Logistic
Regression via the (currently UI-unreachable) `advanced_pipeline.py` path.**
`src/statistical_testing/advanced_pipeline.py:237-262` calls `AdvancedPostHocEngine().execute(
mode="advanced_parametric", test=test, ...)` whenever `res["p_value"] < alpha`, for *any* test
type including `ancova`/`lmm`/`logistic_regression` — but
`advanced_posthoc.py:_run_advanced_parametric_posthoc` (lines 58-74) only builds `group_names`
for `two_way_anova`/`mixed_anova`/`repeated_measures_anova`; the `else: group_names = []` branch
means the call is a guaranteed no-op for the clinical models (empty `all_comparisons`, `posthoc`
stays `None`, returns `{"posthoc_test": "No post-hoc tests performed", "pairwise_comparisons":
[]}`). The caller's `if advanced_posthoc_updates.get("pairwise_comparisons"):` guard at line 254
prevents this from overwriting the model's own EMM contrasts computed inside
`as_results_dict()`, so there is no correctness impact — just a wasted call and a slightly
misleading code path that looks like it does post-hoc dispatch for these models but never can.
**Impact:** none currently (this whole branch of `advanced_pipeline.py` is unreachable from the
UI for `ancova`/`lmm`/`logistic_regression` per SC6's reachability check) — purely a
maintainability/dead-code-shape issue that would become confusing if this path is ever wired up.
**Fix:** either special-case `ancova`/`lmm`/`logistic_regression` to skip the
`AdvancedPostHocEngine` call entirely (their EMM contrasts are already final), or remove them
from consideration in the `res.get("p_value") is not None and res["p_value"] < alpha` gate at
`advanced_pipeline.py:237`.

**SC6 — Contract drift on a live-for-API, dead-for-UI path: `ModelDesignError`/`ValidationError`
raised inside `_run_ancova`/`_run_lmm`/`_run_logistic_regression` (statisticaltester.py) produces
a bare `{"error": ..., "test": ...}` dict with no `blocked=True`, unlike the primary dispatch
path's `make_blocked_result()` contract.**
`src/analysis/statisticaltester.py:1438-1467` and `src/statistical_testing/advanced_pipeline.py
:453-459`. Both catch model-fit exceptions (including `ModelDesignError`, a `ValidationError`
subclass) and return `{"error": str(e), "test": "ANCOVA"}` / `{"error": str(e), "test": f"{test}
(failed)", "p_value": None, "statistic": None}` — no `blocked`, no `block_code`. The UI's
`_ap_determine_and_run_test` checks `result.get("blocked")` (`statistical_analyzer_autopilot_
pipeline.py:1776,1805`) to decide whether to route to `_handle_blocked_result()` (a distinct
"analysis was blocked" UI treatment) vs. `_render_result_summary()` (the success cockpit, with
confetti). A dict shaped like `{"error": ..., "test": ...}` has `blocked` absent/falsy, so it
would fall through to the success-path renderer — which does not check `results.get("error")` at
all (`_ap_render_result_summary`, `statistical_analyzer_autopilot_pipeline.py:1715-1737`) — and
would attempt to format a `p_value`/`effect_size`/etc that don't exist in the dict, likely
throwing further inside the formatting helpers rather than showing the clean "blocked" dialog.
**Verified reachability:** `perform_advanced_test(test="ancova"/"lmm"/"logistic_regression", ...)`
is NOT called from the current autopilot UI flow — `analysis_core.py`'s clinical-model dispatch
(lines 594-810) handles these three tests directly and always `return`s before reaching
`perform_advanced_test` (only `mixed_anova`/`two_way_anova`/`repeated_measures_anova` reach it,
per `git grep` — lines 924/963/1008). However, `StatisticalTester.perform_advanced_test()` is a
public API, exercised directly by `tests/test_golden_r_advanced.py::test_golden_lme4_lmm`, so the
drift is live for any future caller (a new UI wiring, a CLI entry point, or a test) that hits a
`ModelDesignError` through this path.
**Impact:** currently none via the shipped UI (dead code for the invalid-input case specifically
— the happy path IS tested and works); becomes a real "success cockpit shown for a validation
failure" bug the moment anything wires the UI to call `perform_advanced_test` with these test
names, or the moment a `ModelDesignError` is somehow triggered on this path.
**Fix:** have `_run_ancova`/`_run_lmm`/`_run_logistic_regression` (and the outer `except
ValidationError`/`except Exception` in `advanced_pipeline.py`) return
`StatisticalTester.make_blocked_result(...)` instead of a bare error dict, matching the primary
path's contract, so `result.get("blocked")` is reliable regardless of which dispatch path a
future caller takes.

**SC7 — `DataHealthScanner._check_group_sizes` emits a German-language warning string; every
other user-facing string in the file is English.**
`src/analysis/clinical_models.py:2035-2038`: `f"Kleine Gruppenbesetzung: {...}. Logistische
Regression instabil bei n < 10 pro Outcome-Kategorie."` — the only non-English string among the
~40+ user-facing warning/error messages in this file (all others, e.g. the MCAR/VIF/separation
warnings a few lines above, are English). Cosmetic/i18n-consistency issue, not a correctness bug.
**Fix:** translate to English to match the rest of the health-report warnings.

## Strengths (verified)

- **All 6 `ModelDesignError` pre-flight checks fire exactly as claimed, verified by direct
  repro, not inspection.** `ANCOVAModel.fit()` (empty `between_factors`, empty `covariates`),
  `LinearMixedModel.fit()` (empty `fixed_effects`, `random_intercept=None`),
  `LogisticRegressionModel.fit()` (empty `predictors`, >2 outcome levels) all raise
  `ModelDesignError` with the exact stated message, confirmed via a standalone Python script
  exercising all 6 conditions.
- **`control_group` wiring for ANCOVA/LMM vs-control EMM post-hoc genuinely reaches both dispatch
  entry points**, not just one. `analysis_core.py`'s direct clinical dispatch (lines 594-637) and
  `advanced_pipeline.py`'s secondary path (lines 175-209) both independently resolve
  `primary_levels` from the data, call the same `control_group_callback`, and pass the result
  into `.fit(..., control_group=...)`, which correctly routes to `emm_contrasts(method=
  "vs_control", ...)` in both `ANCOVAModel` and `LinearMixedModel`. This was the subject of a
  prior fix and it holds under a fresh, independent trace of both code paths.
- **The universal non-finite-result safety net (`StatisticalTester.nonfinite_block`,
  `statisticaltester.py:64-94`) is well-designed and correctly wired.** It runs
  unconditionally after every test path in `analysis_core.py` (line 1297), catches any
  `inf`/`nan`/out-of-`[0,1]` p-value or non-finite statistic from *any* engine (LMM, RM/Mixed/
  Two-Way ANOVA, ANCOVA), and converts it into a clean, correctly-shaped blocked result via
  `make_blocked_result` (the *correct* call signature, positional `reason` + keyword `code`) —
  this is the same pattern SC1/SC6 should have used and didn't.
- **The clinical-model pre-flight data-quality gate (`analysis_core.py:551-573`) correctly
  covers the continuous DV and every covariate**, using `validate_outcome` to block constant/
  empty/Inf/overflow-risk data before it reaches a model fit that would otherwise either error
  opaquely or silently produce a singular/meaningless result — exactly the kind of defensive
  check this bug class is about, done right.
- **`DataHealthScanner` is a genuinely useful, well-scoped non-blocking diagnostic layer**
  (Little's MCAR test, MAD-based outlier detection, VIF, quasi-perfect-separation check, minimum
  group size) that degrades gracefully (`except Exception: pass` around the scanner
  instantiation at `analysis_core.py:591` falls back to an empty report) rather than blocking a
  valid analysis on a diagnostic-only failure.
- **The Firth penalized-likelihood logistic-regression fallback
  (`clinical_models.py:1121-1145`) is a genuinely careful implementation** — it correctly
  triggers on both non-convergence and large standard errors (separation proxy), uses penalized
  likelihood-ratio p-values (matching R's `logistf` default) rather than an unreliable Wald test
  under separation, and falls back to a documented Wald CI only when the profile-likelihood root
  search itself fails.
- **The RM-ANOVA sphericity-correction write-back (line 1972-1974, tagged "E1") is correctly
  implemented** and stands in useful contrast to SC2 — proof the fix pattern for "computed but
  not wired to the decision path" is known and works when applied; it simply wasn't propagated
  to the Mixed-ANOVA sibling path.

## Recommended remediation order

1. **SC1** (HIGH, cheapest fix) — drop the invalid `test_name=` kwarg from the 4 call sites in
   `analysis_core.py`; a 4-line change that restores the intended error messages for Mixed/
   Two-Way/RM-ANOVA invalid-design and prep-error blocks. No design decision needed.
2. **SC4** (MEDIUM, cheap) — initialize `posthoc_choice = None` at the top of the post-hoc block
   in `analysis_core.py`; a 1-line change that eliminates a crash path.
3. **SC3** (MEDIUM, small, needs a decision on the canonical EMM comparison schema) — rename
   ANCOVA's `"t"`/`"se"` to `"statistic"`/`"std_err"` to match LMM and the export layer; add the
   missing `"test"`/`"corrected"`/`"correction"` keys for full parity. Low risk since these are
   report-display-only fields, not decision-gating ones.
4. **SC2** (MEDIUM, requires the most care) — port the RM-ANOVA sphericity write-back pattern
   (line 1972-1974) to the Mixed ANOVA branch; needs a decision on whether to correct just the
   within-factor row, the interaction row, or both, and how that interacts with
   `results["p_value"]` when the interaction (not the within-factor main effect) is what's
   currently driving the top-level field.
5. **SC6** (LOW today, becomes real the moment this path is wired to a UI) — route
   `_run_ancova`/`_run_lmm`/`_run_logistic_regression`'s exception handlers through
   `make_blocked_result()` instead of a bare error dict, for contract parity with the primary
   dispatch path.
6. **SC5** (LOW, cosmetic/dead-code) — skip or special-case the no-op `AdvancedPostHocEngine`
   call for `ancova`/`lmm`/`logistic_regression` in `advanced_pipeline.py`.
7. **SC7** (LOW, cosmetic) — translate the one German warning string in `DataHealthScanner`.
