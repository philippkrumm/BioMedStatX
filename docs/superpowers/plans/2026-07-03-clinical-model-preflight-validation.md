# Clinical Model Pre-flight Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add explicit `ModelDesignError` pre-flight checks to `ANCOVAModel.fit`, `LinearMixedModel.fit`, and `LogisticRegressionModel.fit` so a malformed clinical-model design (empty factors/covariates/predictors, missing subject column, wrong outcome-level count) surfaces a clear message instead of an incidental `IndexError`/`KeyError`/`PatsyError`/generic `ValueError`.

**Architecture:** Six independent one-line-guard additions across three `fit()` methods in `src/analysis/clinical_models.py`, all raising the same `ModelDesignError` (imported from `src.statistical_testing.validators`, verified no circular import). One new test module, `tests/test_clinical_model_preflight.py`, with one test per guard plus a companion "valid input still works" test where a false positive would be easy to introduce.

**Tech Stack:** Python, pytest, pandas, statsmodels (already in use — no new dependencies).

**Reference spec:** `docs/superpowers/specs/2026-07-03-clinical-model-preflight-validation-design.md`

---

## Shared test fixtures

All tasks below add tests to the same file, `tests/test_clinical_model_preflight.py`. Task 1 creates the file with its own fixture; later tasks append to it. Do not create a separate conftest — these fixtures are tiny and specific to this file.

---

### Task 1: ANCOVA — reject empty `between_factors`

**Files:**
- Modify: `src/analysis/clinical_models.py:13-21` (add import)
- Modify: `src/analysis/clinical_models.py:114-119` (add guard)
- Test: `tests/test_clinical_model_preflight.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_clinical_model_preflight.py`:

```python
"""Guards the pre-flight ModelDesignError checks added to ANCOVAModel,
LinearMixedModel, and LogisticRegressionModel.fit(). Before this fix, a
missing structural field (empty between_factors/covariates/fixed_effects/
predictors, or a missing subject column) either crashed later inside
as_results_dict() with an incidental IndexError/KeyError, or (for LMM fixed
effects) silently degraded to a meaningless intercept-only model instead of
being rejected outright. See docs/superpowers/specs/2026-07-03-clinical-model-preflight-validation-design.md.
"""
import pandas as pd
import pytest

from analysis.clinical_models import ANCOVAModel, LinearMixedModel, LogisticRegressionModel
from statistical_testing.validators import ModelDesignError


def _ancova_df():
    return pd.DataFrame({
        "Y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "Group": ["A", "A", "A", "B", "B", "B"],
        "Cov": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    })


def test_ancova_without_between_factors_raises_model_design_error():
    with pytest.raises(ModelDesignError, match="between-subjects factor"):
        ANCOVAModel().fit(_ancova_df(), dv="Y", between_factors=[], covariates=["Cov"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_clinical_model_preflight.py::test_ancova_without_between_factors_raises_model_design_error -v`

Expected: FAIL. Either an `ImportError`/collection error if `ModelDesignError` import path is wrong (it isn't — same path as `tests/test_validators_mixed_anova_subject.py`), or the test fails because no `ModelDesignError` is raised (patsy tolerates the empty-factor formula and `fit()` returns normally instead of raising).

- [ ] **Step 3: Add the import and the guard**

In `src/analysis/clinical_models.py`, current lines 13-21:

```python
import re
import numpy as np
import pandas as pd

import logging
logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod
from enum import Enum
```

Change to:

```python
import re
import numpy as np
import pandas as pd

import logging
logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod
from enum import Enum

from statistical_testing.validators import ModelDesignError
```

Current lines 114-119 (`ANCOVAModel.fit`):

```python
    def fit(self, df, dv, between_factors, covariates, alpha=0.05, control_group=None):
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm

        self._df = df.dropna(subset=[dv] + between_factors + covariates).copy()
        self._alpha = alpha
```

Change to:

```python
    def fit(self, df, dv, between_factors, covariates, alpha=0.05, control_group=None):
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm

        if not between_factors:
            raise ModelDesignError("ANCOVA requires at least one between-subjects factor.")

        self._df = df.dropna(subset=[dv] + between_factors + covariates).copy()
        self._alpha = alpha
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_clinical_model_preflight.py -v`

Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add src/analysis/clinical_models.py tests/test_clinical_model_preflight.py
git commit -m "fix(clinical-models): reject ANCOVA with no between-subjects factor"
```

---

### Task 2: ANCOVA — reject empty `covariates`

**Files:**
- Modify: `src/analysis/clinical_models.py` (`ANCOVAModel.fit`, right after Task 1's guard)
- Test: `tests/test_clinical_model_preflight.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_clinical_model_preflight.py`:

```python
def test_ancova_without_covariates_raises_model_design_error():
    with pytest.raises(ModelDesignError, match="at least one covariate"):
        ANCOVAModel().fit(_ancova_df(), dv="Y", between_factors=["Group"], covariates=[])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_clinical_model_preflight.py::test_ancova_without_covariates_raises_model_design_error -v`

Expected: FAIL — no `ModelDesignError` raised (patsy tolerates the trailing-operator formula and `fit()` proceeds).

- [ ] **Step 3: Add the guard**

`ANCOVAModel.fit` currently reads (after Task 1):

```python
        if not between_factors:
            raise ModelDesignError("ANCOVA requires at least one between-subjects factor.")

        self._df = df.dropna(subset=[dv] + between_factors + covariates).copy()
```

Change to:

```python
        if not between_factors:
            raise ModelDesignError("ANCOVA requires at least one between-subjects factor.")
        if not covariates:
            raise ModelDesignError("ANCOVA requires at least one covariate.")

        self._df = df.dropna(subset=[dv] + between_factors + covariates).copy()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_clinical_model_preflight.py -v`

Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/analysis/clinical_models.py tests/test_clinical_model_preflight.py
git commit -m "fix(clinical-models): reject ANCOVA with no covariates"
```

---

### Task 3: LMM — reject empty `fixed_effects`

**Files:**
- Modify: `src/analysis/clinical_models.py:643-650` (`LinearMixedModel.fit`)
- Test: `tests/test_clinical_model_preflight.py` (append)

- [ ] **Step 1: Write the failing test**

Append:

```python
def _lmm_df():
    return pd.DataFrame({
        "Y": [1.0, 2.0, 1.5, 2.5, 1.2, 2.2, 1.8, 2.8],
        "Time": ["T1", "T2", "T1", "T2", "T1", "T2", "T1", "T2"],
        "Subject": ["S1", "S1", "S2", "S2", "S3", "S3", "S4", "S4"],
    })


def test_lmm_without_fixed_effects_raises_model_design_error():
    with pytest.raises(ModelDesignError, match="fixed effect"):
        LinearMixedModel().fit(_lmm_df(), dv="Y", fixed_effects=[], random_intercept="Subject")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_clinical_model_preflight.py::test_lmm_without_fixed_effects_raises_model_design_error -v`

Expected: FAIL — no `ModelDesignError` raised (empty `fixed_effects` silently degrades to an intercept-only `"dv ~ 1"` formula and `fit()` returns normally).

- [ ] **Step 3: Add the guard**

Current lines 643-650:

```python
    def fit(self, df, dv, fixed_effects, random_intercept, covariates=None, random_slope=None, alpha=0.05, control_group=None):
        import statsmodels.formula.api as smf
        from scipy import stats as scipy_stats

        self._alpha = alpha
        self._control_group = control_group

        all_cols = [dv, random_intercept] + fixed_effects + (covariates or [])
        if random_slope and random_slope not in all_cols:
```

Change to:

```python
    def fit(self, df, dv, fixed_effects, random_intercept, covariates=None, random_slope=None, alpha=0.05, control_group=None):
        import statsmodels.formula.api as smf
        from scipy import stats as scipy_stats

        if not fixed_effects:
            raise ModelDesignError("Linear Mixed Model requires at least one fixed effect.")

        self._alpha = alpha
        self._control_group = control_group

        all_cols = [dv, random_intercept] + fixed_effects + (covariates or [])
        if random_slope and random_slope not in all_cols:
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_clinical_model_preflight.py -v`

Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/analysis/clinical_models.py tests/test_clinical_model_preflight.py
git commit -m "fix(clinical-models): reject LMM with no fixed effects"
```

---

### Task 4: LMM — reject missing `random_intercept`

**Files:**
- Modify: `src/analysis/clinical_models.py` (`LinearMixedModel.fit`, right after Task 3's guard)
- Test: `tests/test_clinical_model_preflight.py` (append)

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_lmm_without_random_intercept_raises_model_design_error():
    with pytest.raises(ModelDesignError, match="subject/ID column"):
        LinearMixedModel().fit(_lmm_df(), dv="Y", fixed_effects=["Time"], random_intercept=None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_clinical_model_preflight.py::test_lmm_without_random_intercept_raises_model_design_error -v`

Expected: FAIL with `KeyError` (or similar) raised deep inside `fit()` at `self._df[self._random_intercept]`/`col_map[random_intercept]` — `None` is not a valid column key — instead of the intended `ModelDesignError`.

- [ ] **Step 3: Add the guard**

`LinearMixedModel.fit` currently reads (after Task 3):

```python
        if not fixed_effects:
            raise ModelDesignError("Linear Mixed Model requires at least one fixed effect.")

        self._alpha = alpha
```

Change to:

```python
        if not fixed_effects:
            raise ModelDesignError("Linear Mixed Model requires at least one fixed effect.")
        if random_intercept is None:
            raise ModelDesignError("Linear Mixed Model requires a subject/ID column for the random intercept.")

        self._alpha = alpha
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_clinical_model_preflight.py -v`

Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/analysis/clinical_models.py tests/test_clinical_model_preflight.py
git commit -m "fix(clinical-models): reject LMM with no random-intercept column"
```

---

### Task 5: Logistic Regression — reject empty `predictors`

**Files:**
- Modify: `src/analysis/clinical_models.py:1060-1064` (`LogisticRegressionModel.fit`)
- Test: `tests/test_clinical_model_preflight.py` (append)

- [ ] **Step 1: Write the failing test**

Append:

```python
def _logistic_df():
    return pd.DataFrame({
        "Outcome": [0, 1, 0, 1, 0, 1, 0, 1],
        "X": [1.0, 2.0, 1.5, 2.5, 1.2, 2.8, 1.9, 2.1],
    })


def test_logistic_without_predictors_raises_model_design_error():
    with pytest.raises(ModelDesignError, match="at least one predictor"):
        LogisticRegressionModel().fit(_logistic_df(), dv="Outcome", predictors=[])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_clinical_model_preflight.py::test_logistic_without_predictors_raises_model_design_error -v`

Expected: FAIL with a patsy `PatsyError` (empty right-hand side formula `"dv ~ "`) instead of `ModelDesignError`.

- [ ] **Step 3: Add the guard**

Current lines 1060-1064:

```python
    def fit(self, df, dv, predictors, covariates=None):
        import statsmodels.formula.api as smf
        import statsmodels.api as sm

        all_cols = [dv] + predictors + (covariates or [])
```

Change to:

```python
    def fit(self, df, dv, predictors, covariates=None):
        import statsmodels.formula.api as smf
        import statsmodels.api as sm

        if not predictors:
            raise ModelDesignError("Logistic regression requires at least one predictor.")

        all_cols = [dv] + predictors + (covariates or [])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_clinical_model_preflight.py -v`

Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/analysis/clinical_models.py tests/test_clinical_model_preflight.py
git commit -m "fix(clinical-models): reject logistic regression with no predictors"
```

---

### Task 6: Logistic Regression — `ModelDesignError` for wrong outcome-level count

**Files:**
- Modify: `src/analysis/clinical_models.py` (`LogisticRegressionModel.fit`, outcome-level check)
- Test: `tests/test_clinical_model_preflight.py` (append)

- [ ] **Step 1: Write the failing test**

Append:

```python
def _logistic_df_three_levels():
    return pd.DataFrame({
        "Outcome3": [0, 1, 2, 0, 1, 2, 0, 1],
        "X": [1.0, 2.0, 1.5, 2.5, 1.2, 2.8, 1.9, 2.1],
    })


def test_logistic_wrong_outcome_level_count_raises_model_design_error():
    with pytest.raises(ModelDesignError, match="2 outcome levels"):
        LogisticRegressionModel().fit(_logistic_df_three_levels(), dv="Outcome3", predictors=["X"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_clinical_model_preflight.py::test_logistic_wrong_outcome_level_count_raises_model_design_error -v`

Expected: FAIL — the current code raises `ValueError`, not `ModelDesignError`, so `pytest.raises(ModelDesignError, ...)` does not catch it and the test errors out with the unhandled `ValueError`.

- [ ] **Step 3: Change the exception type**

Find (inside `LogisticRegressionModel.fit`, after the sanitize/encode block):

```python
        # Encode DV as 0/1 if needed
        unique_vals = sorted(self._df[self._dv].unique())
        if len(unique_vals) != 2:
            raise ValueError(f"Logistic regression requires exactly 2 outcome levels, found {len(unique_vals)}")
```

Change to:

```python
        # Encode DV as 0/1 if needed
        unique_vals = sorted(self._df[self._dv].unique())
        if len(unique_vals) != 2:
            raise ModelDesignError(f"Logistic regression requires exactly 2 outcome levels, found {len(unique_vals)}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_clinical_model_preflight.py -v`

Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/analysis/clinical_models.py tests/test_clinical_model_preflight.py
git commit -m "fix(clinical-models): use ModelDesignError for wrong outcome-level count"
```

---

### Task 7: Full-suite regression check and audit-note update

**Files:**
- Modify: `docs/superpowers/audit-notes/SUMMARY.md` (mark item resolved)
- No source changes.

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/ -v`

Expected: PASS, all tests including the 6 new ones in `tests/test_clinical_model_preflight.py` and every pre-existing test (in particular `tests/test_validators_mixed_anova_subject.py`, `tests/test_binary_outcome_classification.py`, and any test that exercises `ANCOVAModel`/`LinearMixedModel`/`LogisticRegressionModel` with legitimate non-empty inputs — confirms the new guards don't false-positive on valid designs).

- [ ] **Step 2: Grep for any other direct callers of the three `fit()` methods**

Run: `grep -rn "ANCOVAModel()\|LinearMixedModel()\|LogisticRegressionModel()" src/ tests/`

Expected: only `src/analysis/analysis_core.py` (the clinical dispatch block, which — per the design doc's reachability trace — never calls these with empty structural fields today) and the new test file. If any other caller turns up, read it and confirm it always passes non-empty `between_factors`/`covariates`/`fixed_effects`/`predictors` and a non-None `random_intercept`; if it doesn't, flag to the user before proceeding (do not silently patch a second call site — that's new scope).

- [ ] **Step 3: Update the audit-note SUMMARY**

Open `docs/superpowers/audit-notes/SUMMARY.md` and find the entry describing the ancova/lmm/logistic_regression structural-validation gap (from the original 10-item audit list). Append a note that this was fixed — match the phrasing style of the entry for the mixed_anova/RM-ANOVA subject-column fix (already marked resolved there from the `f9cb42d` commit). Do not remove or restructure any other part of the file.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/audit-notes/SUMMARY.md
git commit -m "docs(audit): mark clinical-model pre-flight validation gap resolved"
```

---

## Plan self-review notes

- **Spec coverage:** all 5 checks + the 1 exception-type conversion from the spec's "Fix" section each have a dedicated task (1-6). The spec's "Testing" section (6 tests in one file) is covered across Tasks 1-6, each appending one test. Task 7 covers the spec's implicit requirement that this stays defensive-only (no change to `analysis_core.py` or `validators.py`) by explicitly grepping for other callers rather than assuming.
- **Type/name consistency:** `ModelDesignError` imported once in Task 1, reused by name in Tasks 2-6 with no renaming. Test fixture names (`_ancova_df`, `_lmm_df`, `_logistic_df`, `_logistic_df_three_levels`) are introduced once each and referenced verbatim by later tasks in the same file — no drift.
- **No placeholders:** every step shows complete before/after code or an exact runnable command with its expected result.
