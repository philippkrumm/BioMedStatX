# Clinical model pre-flight validation (ANCOVA / LMM / Logistic Regression)

## Background

The Help Hub content audit (`docs/superpowers/audit-notes/SUMMARY.md`) flagged that
`ancova`, `two_way_ancova`, `lmm`, and `logistic_regression` reach their model
classes' `.fit()` methods with no structural pre-flight check that the required
design fields (`between_factors`, `covariates`, `fixed_effects`,
`random_intercept`, `predictors`) are non-empty/non-None — unlike
`mixed_anova`/`repeated_measures_anova`, which got explicit `ModelDesignError`
checks for their subject-column requirement (`validators.py`, commit `f9cb42d`).

This mirrors the earlier `validate_test_design` fix (Task 5 of
`plans/2026-07-02-audit-code-bug-fixes.md`), but for a separate code path: the
"clinical dispatch" block in `analysis_core.py` (~lines 540-796), which calls
`ANCOVAModel`, `LinearMixedModel`, and `LogisticRegressionModel` directly and
never goes through `validate_test_design` at all. (The
`elif test_name in ("ancova", "two_way_ancova", "lmm", "logistic_regression"): pass`
branch inside `validate_test_design` is dead code for these four test types —
confirmed during investigation.)

## Investigation findings

Traced what actually happens today with a missing/empty structural field, for
each of the three model classes:

- **`ANCOVAModel.fit`** (`src/analysis/clinical_models.py:114`): an empty
  `between_factors` list does not crash inside `fit()` itself — patsy tolerates
  the resulting formula (`"dv ~  + cov1"`). The crash instead happens later, in
  `as_results_dict()`, at `self._between_factors[0]` (lines ~500, ~555) with an
  `IndexError: list index out of range`. An empty `covariates` list produces a
  formula with a trailing operator (`"dv ~ factor + "`), a foreseeable patsy
  break.
- **`LinearMixedModel.fit`** (`src/analysis/clinical_models.py:643`): an empty
  `fixed_effects` list does not crash — the formula silently degrades to
  `"dv ~ 1"` (intercept-only model), which fits without error but is not an
  LMM in any meaningful sense; the result would be silently wrong rather than
  visibly blocked. A `None` `random_intercept` breaks at
  `self._df[self._random_intercept]` with a `KeyError`.
- **`LogisticRegressionModel.fit`** (`src/analysis/clinical_models.py:1060`):
  an empty `predictors` list produces `formula = f"{dv} ~ "` (empty
  right-hand side), which raises a patsy `PatsyError` at the `smf.glm(...)`
  call. The existing outcome-level-count check
  (`if len(unique_vals) != 2: raise ValueError(...)`) already produces a
  reasonable message, but as a generic `ValueError` rather than the
  codebase's `ModelDesignError` convention.

**Crash-risk reassessment.** The entire clinical dispatch block in
`analysis_core.py` sits inside `_analyze_single_dataset`'s outer
`try/except Exception` (closing around line 1550), which converts any
exception — `IndexError`, `KeyError`, `PatsyError`, `ValueError`, or a future
`ModelDesignError` alike — into a blocked result via
`StatisticalTester.make_blocked_result(str(e), code="UNHANDLED_EXCEPTION", ...)`.
So, exactly as with Task 5's fix, **no raw crash is possible today** — the
value of this fix is entirely in what `str(e)` says to the user
(`"list index out of range"` vs. `"ANCOVA requires at least one
between-subjects factor."`), plus `error_type` becoming a meaningful
`ModelDesignError` instead of an incidental `IndexError`/`KeyError`.

**Reachability from the real UI.** Traced the autopilot routing logic
(`statistical_analyzer_autopilot_pipeline.py:1239-1242`): `ancova`/
`two_way_ancova` are only ever inferred when `covariate_columns` is
non-empty AND the starting test was `independent_ttest`/`one_way_anova`/
`two_way_anova` (all of which already require a non-empty between-factor by
construction). So, like Task 5's mixed_anova/RM-ANOVA subject check, this is
a **defensive/robustness fix**, not a fix for a currently-reachable UI bug —
it protects against a malformed `analysis_context` reaching
`AnalysisManager.analyze()` directly (e.g. from a future UI change, a
scripting entry point, or a test), same category as the earlier fix.

## Fix

Add five pre-flight checks, each raising `ModelDesignError` (imported from
`src.statistical_testing.validators` — verified no circular import, since
`validators.py` has no dependency on `analysis/`), placed at the top of the
relevant `fit()` method before any data manipulation:

1. `ANCOVAModel.fit`: `if not between_factors: raise ModelDesignError("ANCOVA requires at least one between-subjects factor.")`
2. `ANCOVAModel.fit`: `if not covariates: raise ModelDesignError("ANCOVA requires at least one covariate.")`
3. `LinearMixedModel.fit`: `if not fixed_effects: raise ModelDesignError("Linear Mixed Model requires at least one fixed effect.")`
4. `LinearMixedModel.fit`: `if random_intercept is None: raise ModelDesignError("Linear Mixed Model requires a subject/ID column for the random intercept.")`
5. `LogisticRegressionModel.fit`: `if not predictors: raise ModelDesignError("Logistic regression requires at least one predictor.")`

Additionally, change the existing outcome-level-count check in
`LogisticRegressionModel.fit` from `ValueError` to `ModelDesignError`, for
consistency with the rest of the codebase's model-design-error convention:

```python
if len(unique_vals) != 2:
    raise ModelDesignError(f"Logistic regression requires exactly 2 outcome levels, found {len(unique_vals)}")
```

No changes to `validators.py` — the dead `pass` branch there is left as-is,
since these four test types never reach `validate_test_design`.

## Testing

One test module, `tests/test_clinical_model_preflight.py`, calling each
`fit()` method directly (plain model classes, no Qt/UI harness needed) with
the offending empty/None argument and a minimal valid DataFrame for the rest,
asserting `pytest.raises(ModelDesignError, match=...)` with the specific
message. Mirrors the pattern in `tests/test_validators_mixed_anova_subject.py`.
Six tests: one per check above (5), plus one for the `ValueError`→
`ModelDesignError` conversion on wrong outcome-level count.

## Out of scope

- No change to `analysis_core.py`'s clinical dispatch block — the fix is
  entirely inside the three `fit()` methods.
- No change to `validate_test_design`'s dead `pass` branch.
- No change to how `analysis_core.py`'s outer exception handler formats
  blocked results (`code` stays `"UNHANDLED_EXCEPTION"` for these — same as
  Task 5's fix, which also didn't introduce a dedicated block code).
