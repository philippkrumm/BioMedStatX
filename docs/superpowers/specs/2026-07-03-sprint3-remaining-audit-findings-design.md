# Spec: Sprint 3 — remaining 7 audit findings

Date: 2026-07-03
Sources: `docs/superpowers/audit-notes/SUMMARY_PROACTIVE.md` (A2, A3, B3, B4) and
`docs/superpowers/audit-notes/SUMMARY.md` (items 2, 5, 7 — the three items
listed as still-open in `HANDOFF.md`'s "Next Steps"; item 6, the German label,
was already fixed and needs no action).

All two genuine product decisions in this batch were resolved via
`AskUserQuestion` before this spec was written (recorded below per item, not
re-litigated here). The other five are mechanical — a well-defined target
state with no design ambiguity.

## 1. RTE table blank-label fragility (A2) — mechanical

`src/export/report_stat_rows.py:670-671` (`_build_statistical_rows`,
BrunnerLangerATS branch):
```python
between = rte_row.get("between_group", "")
within = rte_row.get("within_level", "")
```
Silent `""` fallback on a missing key. Fix: look up the keys explicitly and
log loudly (not silently substitute blanks) if either is missing — matches
this session's established "surface the failure, don't degrade silently"
paradigm from Sprint 1/2. Since the keys currently match in practice, this
change has no behavioral effect today; it only changes what happens if the
Brunner-Langer/ATS engine (`nonparametricanovas.py`) ever renames them.

## 2. All-NaN value columns pass wide-format detection (A3) — mechanical

`src/autopilot/statistical_analyzer_autopilot_ui.py:128-173` (`_detect_wide_format`).
`is_numeric_dtype` is `True` for an all-NaN float64 column; the function never
checks a value column has any non-null data. Fix: reject a candidate value
column that is entirely NaN before returning the wide-format signature, so the
failure surfaces as a clear "column X has no data" condition rather than a
downstream empty-group error inside a test function
(`src/analysis/analysis_core.py:262`).

## 3. NaN/missing subject IDs (B3 + B4) — **decided: hard reject at load time**

Two call sites, same root question:
- `src/autopilot/statistical_analyzer_autopilot_ui.py:169-171` (`_detect_wide_format`):
  `df[subject_col].nunique()` silently excludes NaN, biasing the uniqueness
  ratio used to decide wide-vs-long format.
- `src/autopilot/statistical_analyzer_autopilot_pipeline.py:1223` (`_ap_build_analysis_context`):
  `groupby([subject_column, within_factor])` silently drops NaN subject keys,
  so incomplete subjects are invisible to the balanced-vs-unbalanced check
  that decides RM-ANOVA vs. LMM routing.

**Decision (user, via AskUserQuestion):** hard reject — raise a clear error
("N rows have no subject ID") at load time rather than silently dropping or
bucketing those rows. Consistent with Sprint 1/2's established philosophy
(scientific transparency over silent degradation) applied to data loading
instead of visualization.

**Correction found while tracing (do not assume one guard covers both):**
these two call sites are reachable via two *different* paths, so they need
two *independent* guards, not one:
- `_detect_wide_format` only ever runs for auto-detected **wide-format**
  files, at load time (`_ap_maybe_pivot`, called from `_ap_load_file`/
  `_ap_load_sheet`, both already wrap their body in
  `try/except Exception as exc: QMessageBox.critical(...)` — confirmed by
  reading `statistical_analyzer_autopilot_pipeline.py:942-1021` — so a raised
  `ValueError` here surfaces as a clean dialog, not a crash).
- Data that's already **long-format** (the common case — user drags columns
  into the Subject/Factor/DV buckets manually) never goes through
  `_detect_wide_format` at all. The only point every path — wide-pivoted or
  manually-mapped — passes through before any test runs is
  `_ap_build_analysis_context` (`statistical_analyzer_autopilot_pipeline.py:1065`),
  which already raises plain `ValueError` directly (not swallowed by any
  broad except) for other structural problems at lines 1079, 1082, 1111,
  1160, 1165, 1192 — established precedent to follow.
- The existing balance-detection groupby at line 1223 is itself wrapped in a
  local `try: ... except Exception: pass` (lines 1221-1238) for an unrelated
  reason (heuristic LMM-routing fallback) — a guard placed *inside* that
  block would be silently swallowed. The fix must guard *before* that block,
  not inside it.

Two independent fixes: B3 raises in `_detect_wide_format` (earliest possible
feedback for the auto-pivot path); B4's fix is not "inside the groupby" but a
new guard near the top of `_ap_build_analysis_context` (after the filtered
`analysis_df` is built, before any factor/subject logic runs) — this is the
true single choke point that covers both the wide-pivoted and the
manually-mapped path, since every analysis run passes through it regardless
of how the data got its current shape.

## 4. RM-ANOVA sphericity outer-exception skips the documented GG default (SUMMARY.md item 2) — mechanical, matches an existing documented guarantee

`src/analysis/statisticaltester.py:2617-2703` (`_perform_comprehensive_sphericity_test`).
The **inner** exception path (`except Exception:` at line 2677, when
`pg.sphericity` itself fails) correctly falls through to
`_apply_sphericity_corrections`, which — per
`src/analysis/statisticaltester.py:2856-2861` — applies Greenhouse-Geisser
"unconditionally" when available. This matches the documented guarantee in
`CHANGELOG.md:12`:
> "When sphericity cannot be formally tested (for example, with incomplete
> tables), the Greenhouse-Geisser correction is now applied by default.
> Earlier versions assumed sphericity was met, which could inflate the
> Type-I error rate."

But the **outer** exception (`except Exception as e:` at line 2690-2701 —
catches failures from `_apply_sphericity_corrections` itself, or anything
else in the outer `try`) does not call `_apply_sphericity_corrections` at
all:
```python
except Exception as e:
    results["sphericity_test"] = {..., "sphericity_assumed": None, ...}
    results["corrected_p_value"] = StatisticalTester._pingouin_p_value(row)
    results["correction_used"] = "None (sphericity test failed)"
```
This uses the **uncorrected** p-value — functionally the pre-v2.0 "assume met"
behavior the changelog says was fixed, just reachable through a narrower
failure path (`_apply_sphericity_corrections` throwing) that the inner fix
didn't cover.

Fix: in the outer `except`, attempt the same conservative default the
changelog promises — call `_apply_sphericity_corrections(row, error_row,
sphericity_violated=True, aov)` (forcing "assume violated, apply GG by
default", matching the stated philosophy) inside its own nested
try/except (defensive: `row`/`aov` could in principle also be unusable),
falling back to the current uncorrected behavior only if that also fails.

## 5. ANCOVA vs-control post-hoc never wired through primary dispatch (SUMMARY.md item 7) — mechanical, mirrors an existing working pattern

`src/analysis/analysis_core.py:594-598` (ANCOVA branch):
```python
if clinical_test in ('ancova', 'two_way_ancova'):
    model = ANCOVAModel()
    between_factors = analysis_context.get('between_factors') or analysis_context.get('factor_columns', [])
    model.fit(df, dv=value_cols[0], between_factors=between_factors, covariates=covariates)
    test_results = model.as_results_dict()
```
No `control_group` is ever passed, even though `ANCOVAModel.fit` accepts it
(`clinical_models.py:116`) and uses it to auto-run a `vs_control`
multivariate-t EMM post-hoc (`clinical_models.py:512-519`) — fully
implemented, just never triggered from this path.

The **LMM branch immediately below it** (lines 600-623, same function, same
`kwargs`) already does this correctly:
```python
elif clinical_test == 'lmm':
    ...
    lmm_control = None
    primary_factor = fixed_effects[0] if fixed_effects else None
    _control_cb = kwargs.get('control_group_callback')
    if _control_cb and primary_factor:
        try:
            primary_levels = sorted(str(v) for v in df[primary_factor].dropna().unique())
            lmm_control = _control_cb(primary_levels)
        except Exception as exc:
            logger.warning("LMM control-group selection failed in core: %s", exc)
    model.fit(..., control_group=lmm_control)
```
Fix: mirror this exact pattern for ANCOVA — same `control_group_callback`
lookup, same primary-factor derivation (from `between_factors` instead of
`fixed_effects`), same try/except-and-log-on-failure, passed into
`ANCOVAModel.fit(..., control_group=ancova_control)`.

Note: `src/statistical_testing/advanced_pipeline.py` (the second dispatch
path noted in `HANDOFF.md`) already wires `control_group_callback` through
for ANCOVA (lines 179-189) — only the `analysis_core.py` path was missing it.

## 6. Linear regression `coefficient_table` computed but never rendered (SUMMARY.md item 5) — **decided: wire it in**

`src/analysis/correlation_models.py:848` (`SimpleLinearRegressionModel.as_results_dict`)
computes `coefficient_table` (rows: `parameter, coefficient, std_err,
t_value, p_value, ci_lower, ci_upper` — built at lines 782-794). Zero readers
anywhere in the codebase (confirmed by grep across `src/`).

**Decision (user, via AskUserQuestion):** wire it in, following the existing
`BetaRegression` pattern exactly:
- `src/export/report_association.py:71-113` has
  `_build_beta_coefficient_table_html`, reading `results.get("coefficients")`
  (a *different* key, Beta Regression's own) and rendering a `z`-column table
  (Beta Regression is a GLM, uses z-statistics).
- `src/export/report_charts.py:613-617` gates that call on
  `model_type == "BetaRegression"`.

New: `_build_linear_regression_coefficient_table_html(results)` — same HTML
shape as the beta version, reading `results.get("coefficient_table")` (the
correct key for this model), with a `t`-column instead of `z` (OLS uses
t-statistics, reading `row.get("t_value")` instead of `row.get("z_value")`),
no "logit scale" subtitle (that's beta-regression-specific — linear
regression coefficients are on the original response scale). Gated on
`model_type == "LinearRegression"` in `report_charts.py`, added as a new
`elif` branch alongside the existing `BetaRegression`/`CorrelationMatrix`
branches (`report_charts.py:604-676`). Note: like the sibling
`LogisticRegression`/`BetaRegression` branches, this naturally replaces the
generic group-comparison boxplot that `LinearRegression` currently falls
through to via the catch-all `else` (a boxplot has no meaning for a
regression with no groups — same reasoning already applied to the other two
special-cased model types, not a new design choice).

## Non-goals

- No other items from `SUMMARY_PROACTIVE.md`/`SUMMARY.md` are in scope beyond
  the 7 listed above.
- Item 6 in `SUMMARY.md` (German label) needs no action — already fixed
  (commit `27da427`, per `HANDOFF.md`).
