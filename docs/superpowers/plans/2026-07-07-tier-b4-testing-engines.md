# Tier B4: Testing Engines Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give logistic regression the same data-quality pre-flight gate every other advanced
test already gets (AT4), and stop `_perform_welch_anova`'s exception fallback from silently
presenting a non-robust standard F-test as if it were the requested Welch correction (AT5).

**Architecture:** Two independent, single-file-each fixes.
`src/statistical_testing/mixed_assumptions.py` was already modified by the A1 (Mixed-ANOVA
sphericity) fix on this branch — `_perform_welch_anova` is a different function than the ones A1
touched, but **read the file fresh before editing**; do not assume pre-A1 line numbers.

**Tech Stack:** Python, pytest.

---

### Task 1: AT4 — give logistic regression a pre-flight data-quality gate

**Files:**
- Modify: `src/statistical_testing/advanced_pipeline.py:16-20` (imports),
  `src/statistical_testing/advanced_pipeline.py:93-107` (the pre-flight block)
- Test: `tests/test_advanced_pipeline_logistic_preflight.py`

`advanced_pipeline.py:93` wraps the shared `validate_samples_for_test` pre-flight in
`if test not in ["logistic_regression"]:`, excluding logistic regression entirely with no
substitute. `validate_samples_for_test`'s group-based shape doesn't fit logistic regression
(binary outcome + predictors, not comparison groups) — the existing single-vector gate
`validate_outcome` (`validators.py:419`, already used elsewhere for regression-style models per
its own docstring: "Single-vector degeneracy gate for regression-style models... whose data
shape doesn't fit the group-based gate") is the right tool, scoped to the outcome column.

- [ ] **Step 1: Write the failing test**

```python
"""advanced_pipeline.py explicitly excludes logistic_regression from the shared
validate_samples_for_test pre-flight gate (`if test not in ["logistic_regression"]:`) with no
substitute - a constant (all-0 or all-1) binary outcome reaches LogisticRegressionModel.fit()
with no pre-flight net, unlike every other advanced test.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import pytest

from statistical_testing.advanced_pipeline import perform_advanced_test_pipeline


def test_constant_binary_outcome_is_blocked_before_fitting():
    df = pd.DataFrame({
        "Outcome": [0, 0, 0, 0, 0, 0],  # zero variance - logistic regression is meaningless
        "Predictor": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })

    result = perform_advanced_test_pipeline(
        df=df,
        test="logistic_regression",
        dv="Outcome",
        subject=None,
        between=["Predictor"],
        within=None,
    )

    assert result.get("blocked") is True, (
        f"expected a blocked result for a constant outcome, got: {result}"
    )
    assert result.get("block_code") == "VAR_ZERO"
```

**Note:** confirm `perform_advanced_test_pipeline`'s exact required kwargs by reading its
signature first (`grep -n "^def perform_advanced_test_pipeline" src/statistical_testing/advanced_pipeline.py`)
— adjust the call above if any required parameter is missing (e.g. `covariates`,
`transformed_samples`, `recommendation`, `test_info`); the goal is to reach the pre-flight block
at line ~93 with a `test="logistic_regression"` dispatch and a constant `dv` column.

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/test_advanced_pipeline_logistic_preflight.py -v`
Expected: FAIL — `result.get("blocked")` is not `True` (the constant outcome reaches whatever
comes after the pre-flight block unguarded, likely a downstream statsmodels convergence warning
or error instead of a clean blocked result).

- [ ] **Step 3: Add the import**

Read the current import block fresh
(`grep -n "from .validators import" src/statistical_testing/advanced_pipeline.py`), then change:

```python
from .validators import (
    ValidationError,
    validate_samples_for_test,
    validate_test_design,
)
```

to:

```python
from .validators import (
    ValidationError,
    validate_outcome,
    validate_samples_for_test,
    validate_test_design,
)
```

- [ ] **Step 4: Route logistic_regression through `validate_outcome` instead of skipping the gate**

Read the current block fresh (`grep -n 'if test not in \["logistic_regression"\]:' src/statistical_testing/advanced_pipeline.py`),
then change:

```python
        valid_groups = [g for g in groups if g in transformed_samples and len(transformed_samples[g]) > 0]
        # Data-quality pre-flight on the extracted cells
        if test not in ["logistic_regression"]:
            _quality = validate_samples_for_test(
                transformed_samples, valid_groups, dependent=False, min_n_block=2,
            )
            if _quality.blocking_issue is not None:
                issue = _quality.blocking_issue
                logger.warning("Advanced pre-flight blocked: %s", issue.message)
                blocked = StatisticalTester.make_blocked_result(
                    issue.message, code=issue.code,
                    details={"groups": [str(g) for g in valid_groups], "test": test},
                    warnings=_quality.warnings,
                )
                blocked["test_info"] = test_info
                blocked["recommendation"] = recommendation
                return blocked
```

to:

```python
        valid_groups = [g for g in groups if g in transformed_samples and len(transformed_samples[g]) > 0]
        # Data-quality pre-flight on the extracted cells
        if test == "logistic_regression":
            # logistic_regression's shape (binary outcome + predictors) doesn't
            # fit validate_samples_for_test's group-based gate - use the
            # single-vector degeneracy gate on the outcome column instead.
            # Previously this test had NO pre-flight gate at all (AT4).
            _outcome_issue = validate_outcome(df[dv], label=dv, min_n_block=2)
            if _outcome_issue is not None:
                logger.warning("Advanced pre-flight blocked: %s", _outcome_issue.message)
                blocked = StatisticalTester.make_blocked_result(
                    _outcome_issue.message, code=_outcome_issue.code,
                    details={"test": test},
                )
                blocked["test_info"] = test_info
                blocked["recommendation"] = recommendation
                return blocked
        else:
            _quality = validate_samples_for_test(
                transformed_samples, valid_groups, dependent=False, min_n_block=2,
            )
            if _quality.blocking_issue is not None:
                issue = _quality.blocking_issue
                logger.warning("Advanced pre-flight blocked: %s", issue.message)
                blocked = StatisticalTester.make_blocked_result(
                    issue.message, code=issue.code,
                    details={"groups": [str(g) for g in valid_groups], "test": test},
                    warnings=_quality.warnings,
                )
                blocked["test_info"] = test_info
                blocked["recommendation"] = recommendation
                return blocked
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/test_advanced_pipeline_logistic_preflight.py -v`
Expected: PASS.

- [ ] **Step 6: Confirm a normal, non-degenerate logistic regression still runs (no regression)**

Add this test to the same file:

```python
def test_normal_binary_outcome_is_not_blocked():
    import numpy as np
    rng = np.random.RandomState(0)
    n = 60
    predictor = rng.randn(n)
    df = pd.DataFrame({
        "Outcome": (predictor + rng.randn(n) * 0.5 > 0).astype(int),
        "Predictor": predictor,
    })

    result = perform_advanced_test_pipeline(
        df=df,
        test="logistic_regression",
        dv="Outcome",
        subject=None,
        between=["Predictor"],
        within=None,
    )

    assert result.get("blocked") is not True, f"a normal binary outcome should not be blocked: {result}"
```

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/test_advanced_pipeline_logistic_preflight.py -v`
Expected: both tests PASS.

- [ ] **Step 7: Commit**

```bash
git add tests/test_advanced_pipeline_logistic_preflight.py src/statistical_testing/advanced_pipeline.py
git commit -m "fix(advanced-pipeline): add a pre-flight data-quality gate for logistic regression"
```

---

### Task 2: AT5 — flag `_perform_welch_anova`'s silent degraded fallback

**Files:**
- Modify: `src/statistical_testing/mixed_assumptions.py` (inside `_perform_welch_anova` —
  confirm current line with `grep -n "def _perform_welch_anova" src/statistical_testing/mixed_assumptions.py`,
  was line 251 as of A1 landing on this branch)
- Test: `tests/test_welch_anova_degraded_flag.py`

The manual Welch F/df computation is wrapped in `try/except Exception:`; on any exception
(e.g. a `ZeroDivisionError` from a zero-variance group) it silently falls back to
`f_oneway(*group_data)` — the **standard**, non-robust ANOVA — relabeled under the same
`"welch_f_statistic"`/`"welch_p_value"` keys with no indication the result isn't actually a
Welch correction.

- [ ] **Step 1: Write the failing test**

```python
"""_perform_welch_anova's manual F/df computation silently falls back to the standard,
non-robust f_oneway result on any exception (e.g. ZeroDivisionError from a zero-variance
group), relabeled under the same welch_f_statistic/welch_p_value keys - a caller has no way to
tell the "Welch" result isn't actually variance-robust. Fix: add an explicit
welch_calculation_degraded flag on that fallback path.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from statistical_testing.mixed_assumptions import MixedAnovaAssumptionEngine


def test_welch_anova_flags_degraded_fallback_on_zero_variance_group():
    # Group "A" has zero variance -> ZeroDivisionError in the manual Welch
    # weight calculation (weights = [n/var for n, var in ...]) -> falls back
    # to f_oneway, currently with no indication of degradation.
    group_data = [
        [1.0, 1.0, 1.0, 1.0],
        [2.0, 4.0, 6.0, 8.0],
        [1.0, 3.0, 5.0, 7.0],
    ]
    group_labels = ["A", "B", "C"]

    result = MixedAnovaAssumptionEngine._perform_welch_anova(group_data, group_labels, "Value", "Group")

    assert result.get("welch_calculation_degraded") is True, (
        f"expected the degraded-fallback flag to be set, got keys: {sorted(result.keys())}"
    )
    assert result["welch_f_statistic"] == pytest.approx(result["standard_f_statistic"]), (
        "the degraded fallback IS the standard f_oneway result - both fields should match "
        "exactly when this flag is True"
    )


def test_welch_anova_does_not_flag_degraded_on_normal_data():
    group_data = [
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 4.0, 6.0, 8.0],
        [1.0, 3.0, 5.0, 7.0],
    ]
    group_labels = ["A", "B", "C"]

    result = MixedAnovaAssumptionEngine._perform_welch_anova(group_data, group_labels, "Value", "Group")

    assert result.get("welch_calculation_degraded", False) is False
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/test_welch_anova_degraded_flag.py -v`
Expected: FAIL on `test_welch_anova_flags_degraded_fallback_on_zero_variance_group` —
`result.get("welch_calculation_degraded")` is `None`, not `True` (the key doesn't exist yet).

- [ ] **Step 3: Fix — set the flag on the fallback path**

Read `mixed_assumptions.py` around `_perform_welch_anova` fresh
(`grep -n "except Exception:" src/statistical_testing/mixed_assumptions.py | head -5` to locate
the inner except inside this specific function — cross-check against
`grep -n "def _perform_welch_anova\|def _generate_between_assumption_recommendations"` to
confirm you're editing the except block between those two lines, not a different function's).
Change:

```python
            except Exception:
                # Fallback to scipy's implementation if available
                welch_f, p_val_welch = f_oneway(*group_data)
                df1, df2 = len(group_data) - 1, sum(group_sizes) - len(group_data)
            
            return {
                "test_name": "Welch's ANOVA (Unequal Variances)",
                "welch_f_statistic": float(welch_f),
                "welch_p_value": float(p_val_welch),
                "standard_f_statistic": float(f_stat_standard),
```

to:

```python
            except Exception:
                # Fallback to scipy's implementation if available - this is
                # NOT a variance-robust result, it's the standard f_oneway
                # relabeled. welch_calculation_degraded (below) tells the
                # caller so it isn't presented as if it were actually robust.
                welch_f, p_val_welch = f_stat_standard, None
                welch_calculation_degraded = True
                df1, df2 = len(group_data) - 1, sum(group_sizes) - len(group_data)
            else:
                welch_calculation_degraded = False

            if welch_calculation_degraded and p_val_welch is None:
                p_val_welch = p_val_standard

            return {
                "test_name": "Welch's ANOVA (Unequal Variances)",
                "welch_f_statistic": float(welch_f),
                "welch_p_value": float(p_val_welch),
                "welch_calculation_degraded": welch_calculation_degraded,
                "standard_f_statistic": float(f_stat_standard),
```

**Note:** the original fallback line was `welch_f, p_val_welch = f_oneway(*group_data)` — this
re-runs `f_oneway`, which is already computed once earlier in the function as
`f_stat_standard, p_val_standard = f_oneway(*group_data)` (confirm this exact line still exists
a few lines above via `grep -n "f_stat_standard, p_val_standard = f_oneway" src/statistical_testing/mixed_assumptions.py`
first). The rewrite above reuses that existing result instead of calling `f_oneway` a second
time — same values, one fewer redundant computation. If the surrounding variable names differ
from what's shown here (re-verify by reading the function fresh, since exact wording may have
drifted), adapt the edit to match the real variable names rather than copy-pasting blindly.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/test_welch_anova_degraded_flag.py -v`
Expected: PASS (both cases).

- [ ] **Step 5: Run the full test suite**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/ -q --tb=no`
Expected: same pass count as before this task plus the 4 new tests across both files (the 1
pre-existing unrelated `test_convergence.py::test_convergence_keys` failure is expected — see
the A1 plan's Task 4 for how this was confirmed pre-existing). Since this task and A1 both touch
`mixed_assumptions.py`, pay particular attention to `tests/test_mixed_anova_sphericity_correction.py`
and `tests/test_golden_r_advanced.py::test_golden_afex_mixed_anova` (A1's tests) still passing.

- [ ] **Step 6: Commit**

```bash
git add tests/test_welch_anova_degraded_flag.py src/statistical_testing/mixed_assumptions.py
git commit -m "fix(mixed-assumptions): flag Welch ANOVA's silent degraded fallback"
```

---

## Self-review notes

- **Spec coverage:** AT4 (Task 1), AT5 (Task 2) — both findings assigned to this package are
  covered.
- **AT4 reuses `validate_outcome`, an already-existing function** whose own docstring explicitly
  names it as the right tool for "regression-style models... whose data shape doesn't fit the
  group-based gate" — not a new validation concept invented for this fix.
- **AT5's fix avoids the double `f_oneway` call** the naive version of this fix would introduce,
  by reusing the already-computed `f_stat_standard`/`p_val_standard` values from earlier in the
  same function.
- **File-overlap risk with A1 called out explicitly** at the top of this plan and again in
  Task 2's Step 5 — `mixed_assumptions.py` was already modified by A1 on this branch before this
  plan runs.
