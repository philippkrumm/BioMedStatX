# Tier B2: UI Bridge Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix two independent bugs: the app's global crash handler itself throws before it can
show the user-facing dialog (U1), and multi-DV batch mode reuses one DV column's LMM-vs-RM-ANOVA
decision for every other DV column in the batch (U2).

**Architecture:** U1 is a one-line fix. U2 extracts the existing inline missingness-check logic
(currently duplicated conceptually but only ever run once) into a small pure function, following
this codebase's established pattern of extracting pure functions out of the large
`StatisticalAnalyzerApp`/autopilot-mixin methods for testability (see
`tests/test_analysis_context_subject_guard.py`'s docstring and HANDOFF.md's "What Worked" notes)
rather than building a full Qt harness.

**Tech Stack:** Python, PyQt5, pytest.

---

### Task 1: U1 — fix the global excepthook's own crash

**Files:**
- Modify: `src/analysis/statistical_analyzer.py:501`
- Test: `tests/test_global_excepthook.py`

`_install_global_excepthook`'s inner `_excepthook` (statistical_analyzer.py:493-514) calls
`logger.info("%s %s", msg, file=sys.stderr)` — `file=` is not a valid `logging.Logger.info`
keyword (that belongs to `print()`). This line is not wrapped in its own `try/except`, unlike
the file-write two lines above and the dialog-show block below, so the resulting `TypeError`
propagates out of `_excepthook` itself. When an exception hook raises, CPython prints a separate
"Error in sys.excepthook" traceback and abandons the rest of the handler — the
`QMessageBox.critical(...)` call at line 505 never executes for any uncaught exception.

- [ ] **Step 1: Write the failing test**

```python
"""_install_global_excepthook's inner _excepthook calls logger.info(msg, file=sys.stderr) -
file= isn't a valid logging kwarg, so this raises TypeError before the QMessageBox.critical
dialog call ever runs. The crash dialog has never fired for any uncaught exception.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest


@pytest.fixture(autouse=True)
def _qapp():
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_global_excepthook_shows_the_crash_dialog(monkeypatch, tmp_path):
    import analysis.statistical_analyzer as sa_module

    shown = []
    monkeypatch.setattr(
        sa_module.QMessageBox, "critical",
        staticmethod(lambda *a, **k: shown.append((a, k))), raising=False
    )
    # Avoid writing to the repo's real crash_log.txt during the test - the
    # write is already wrapped in its own try/except that silently passes on
    # any failure, so forcing it to fail here is a safe, realistic way to
    # isolate the test from disk state without changing what's under test.
    monkeypatch.setattr("builtins.open", lambda *a, **k: (_ for _ in ()).throw(OSError("no log in test")))

    sa_module._install_global_excepthook()
    hook = sys.excepthook
    assert hook is not sys.__excepthook__, "excepthook was not installed"

    try:
        raise ValueError("synthetic test exception")
    except ValueError:
        exc_type, exc_value, exc_tb = sys.exc_info()
        hook(exc_type, exc_value, exc_tb)

    assert len(shown) == 1, (
        "QMessageBox.critical was never called - the excepthook itself likely "
        "raised (e.g. the file=sys.stderr TypeError) before reaching it"
    )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/test_global_excepthook.py -v`
Expected: FAIL — `shown` is empty because `_excepthook` raises `TypeError` at the `logger.info`
call before reaching `QMessageBox.critical`. (The `TypeError` itself surfaces as a second,
separate "Error in sys.excepthook while handling..." print from CPython's runtime, not as a
pytest-visible exception — the test fails on the `assert len(shown) == 1` line.)

- [ ] **Step 3: Fix the logging call**

Read `statistical_analyzer.py` around line 501 fresh
(`grep -n 'logger.info."%s %s"' src/analysis/statistical_analyzer.py`) to confirm the current
line, then change:

```python
        logger.info("%s %s", msg, file=sys.stderr)
```
to:
```python
        logger.error("%s", msg)
```

(Switched to `logger.error` — an uncaught exception reaching the last-resort global hook
warrants error level, not info; `msg` already contains the full formatted traceback from
`_tb.format_exception`, so a single `%s` placeholder is correct.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/test_global_excepthook.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_global_excepthook.py src/analysis/statistical_analyzer.py
git commit -m "fix(ui): stop the global excepthook from crashing before showing the dialog"
```

---

### Task 2: U2 — recompute the LMM-vs-RM-ANOVA decision per DV column in multi-DV mode

**Files:**
- Modify: `src/autopilot/statistical_analyzer_autopilot_pipeline.py` (new function near
  `_ap_build_analysis_context`, plus two call-site edits — exact current line numbers below,
  confirm via `grep` since this file changes often)
- Test: `tests/test_lmm_vs_rmanova_per_dv.py`

`_ap_build_analysis_context` computes whether the batch needs LMM (vs. RM-ANOVA/paired-ttest)
from only `dv_columns[0]`'s missingness pattern, once per "Start Analysis" click. The multi-DV
loop in `_ap_determine_and_run_test` (currently ~line 1788) then reuses that single decision for
every other DV column via a shallow `dict(context)` copy — a DV column with a different
missingness pattern than the first one silently gets the wrong test family.

- [ ] **Step 1: Write the failing test for the extracted pure function (doesn't exist yet)**

```python
"""U2: multi-DV batch mode must decide LMM-vs-RM-ANOVA/paired-ttest per DV column, not once
from dv_columns[0] for the whole batch - each DV column can have its own missingness pattern.
_ap_lmm_vs_rmanova_needed is the extracted pure function this decision should go through,
callable both from _ap_build_analysis_context (the original single-shot site) and from the
multi-DV loop in _ap_determine_and_run_test (per column).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from autopilot.statistical_analyzer_autopilot_pipeline import _ap_lmm_vs_rmanova_needed


def test_complete_data_does_not_need_lmm():
    df = pd.DataFrame({
        "Subject": ["S1", "S1", "S2", "S2"],
        "Time": ["T1", "T2", "T1", "T2"],
        "Gene_A": [1.0, 2.0, 3.0, 4.0],
    })
    assert _ap_lmm_vs_rmanova_needed(df, "Subject", "Time", "Gene_A") is False


def test_structurally_missing_timepoint_needs_lmm():
    # S2 has no T2 row at all.
    df = pd.DataFrame({
        "Subject": ["S1", "S1", "S2"],
        "Time": ["T1", "T2", "T1"],
        "Gene_A": [1.0, 2.0, 3.0],
    })
    assert _ap_lmm_vs_rmanova_needed(df, "Subject", "Time", "Gene_A") is True


def test_nan_measurement_needs_lmm():
    # S2 has a T2 row, but the measurement itself is NaN.
    df = pd.DataFrame({
        "Subject": ["S1", "S1", "S2", "S2"],
        "Time": ["T1", "T2", "T1", "T2"],
        "Gene_A": [1.0, 2.0, 3.0, np.nan],
    })
    assert _ap_lmm_vs_rmanova_needed(df, "Subject", "Time", "Gene_A") is True


def test_different_dv_columns_can_disagree():
    # Gene_A is complete; Gene_B is missing S2's T2 measurement. A multi-DV
    # batch over both columns must NOT force Gene_A through the same
    # decision Gene_B needs.
    df = pd.DataFrame({
        "Subject": ["S1", "S1", "S2", "S2"],
        "Time": ["T1", "T2", "T1", "T2"],
        "Gene_A": [1.0, 2.0, 3.0, 4.0],
        "Gene_B": [1.0, 2.0, 3.0, np.nan],
    })
    assert _ap_lmm_vs_rmanova_needed(df, "Subject", "Time", "Gene_A") is False
    assert _ap_lmm_vs_rmanova_needed(df, "Subject", "Time", "Gene_B") is True
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/test_lmm_vs_rmanova_per_dv.py -v`
Expected: FAIL with `ImportError: cannot import name '_ap_lmm_vs_rmanova_needed'` — the function
doesn't exist yet.

- [ ] **Step 3: Extract the pure function**

Find `_ap_build_analysis_context`'s current line
(`grep -n "^def _ap_build_analysis_context" src/autopilot/statistical_analyzer_autopilot_pipeline.py`)
and add this new module-level function immediately before it:

```python
def _ap_lmm_vs_rmanova_needed(df, subject_column, within_factor, dv_column):
    """True if this DV column has structural or NaN-driven missingness across
    the within-factor's levels for this subject_column, requiring an LMM
    instead of RM-ANOVA/paired-ttest for THIS specific column. Each DV column
    in a multi-DV batch can have a different missingness pattern (round-2
    audit finding U2) - callers must invoke this per DV column, not just
    once for the whole batch.
    """
    try:
        counts = df.groupby([subject_column, within_factor]).size().unstack(fill_value=0)
        if (counts == 0).any().any():
            return True
        if dv_column:
            valid = df[[subject_column, within_factor, dv_column]].dropna(subset=[dv_column])
            valid_counts = valid.groupby([subject_column, within_factor]).size().unstack(fill_value=0)
            if (valid_counts == 0).any().any():
                return True
        return False
    except Exception:
        return False
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/test_lmm_vs_rmanova_per_dv.py -v`
Expected: PASS (all 4 cases).

- [ ] **Step 5: Wire the pure function into `_ap_build_analysis_context`, preserving the pre-upgrade test choice**

Find the exact current block via
`grep -n 'elif subject_column and context\["within_factors"\]:' src/autopilot/statistical_analyzer_autopilot_pipeline.py`.
Replace:

```python
    elif subject_column and context["within_factors"]:
        within_factor = context["within_factors"][0]
        dv_col_for_balance = dv_columns[0] if dv_columns else None
        try:
            # Case 1: structural missingness (whole Subject×Timepoint combos absent)
            counts = self.df.groupby([subject_column, within_factor]).size().unstack(fill_value=0)
            has_structural_missing = (counts == 0).any().any()

            # Case 2: row exists but DV is NaN (patient present at visit but no measurement)
            has_nan_missing = False
            if dv_col_for_balance and not has_structural_missing:
                valid = self.df[[subject_column, within_factor, dv_col_for_balance]].dropna(
                    subset=[dv_col_for_balance]
                )
                valid_counts = valid.groupby([subject_column, within_factor]).size().unstack(fill_value=0)
                has_nan_missing = (valid_counts == 0).any().any()

            if has_structural_missing or has_nan_missing:
                context["inferred_test"] = "lmm"
        except Exception:
            pass
```

with:

```python
    elif subject_column and context["within_factors"]:
        within_factor = context["within_factors"][0]
        dv_col_for_balance = dv_columns[0] if dv_columns else None
        # Snapshot the test choice as it stood before the LMM upgrade check,
        # so the multi-DV loop (_ap_determine_and_run_test) can fall back to
        # it for any DV column that doesn't itself need LMM (U2 fix).
        context["_test_before_lmm_upgrade"] = context["inferred_test"]
        if _ap_lmm_vs_rmanova_needed(self.df, subject_column, within_factor, dv_col_for_balance):
            context["inferred_test"] = "lmm"
```

- [ ] **Step 6: Add a regression test confirming `_ap_build_analysis_context` still upgrades correctly and stores the snapshot**

Reuses the `_FakeApp` harness already established in
`tests/test_analysis_context_subject_guard.py` (read that file first to confirm the fixture's
current shape hasn't drifted). Add this test to `tests/test_lmm_vs_rmanova_per_dv.py`:

```python
from autopilot.statistical_analyzer_autopilot_pipeline import _ap_build_analysis_context


class _FakeBucket:
    def __init__(self, columns=None):
        self._columns = columns or []

    def get_assigned_columns(self):
        return list(self._columns)


class _FakeFilterBucket(_FakeBucket):
    def get_filter(self):
        return None


class _FakeCheckbox:
    def __init__(self, checked=False):
        self._checked = checked

    def isChecked(self):
        return self._checked


class _FakeApp:
    def __init__(self, df, subject_col, dv_col="Value"):
        self.df = df
        self.dv_bucket = _FakeBucket([dv_col])
        self.factor1_bucket = _FakeBucket(["Time"])
        self.factor2_bucket = _FakeBucket([])
        self.subject_bucket = _FakeBucket([subject_col] if subject_col else [])
        self.covariates_bucket = _FakeBucket([])
        self.filter_bucket = _FakeFilterBucket([])
        self.multi_mode_button = _FakeCheckbox(False)
        self.analysis_selected_groups = []


def test_build_analysis_context_upgrades_to_lmm_and_stores_pre_upgrade_choice():
    df = pd.DataFrame({
        "Subject": ["S1", "S1", "S2"],
        "Time": ["T1", "T2", "T1"],
        "Value": [1.0, 2.0, 3.0],
    })
    fake_self = _FakeApp(df, subject_col="Subject")
    context = _ap_build_analysis_context(fake_self)
    assert context["inferred_test"] == "lmm"
    assert context["_test_before_lmm_upgrade"] == "paired_ttest"


def test_build_analysis_context_leaves_complete_data_as_paired_ttest():
    df = pd.DataFrame({
        "Subject": ["S1", "S1", "S2", "S2"],
        "Time": ["T1", "T2", "T1", "T2"],
        "Value": [1.0, 2.0, 3.0, 4.0],
    })
    fake_self = _FakeApp(df, subject_col="Subject")
    context = _ap_build_analysis_context(fake_self)
    assert context["inferred_test"] == "paired_ttest"
    assert context["_test_before_lmm_upgrade"] == "paired_ttest"
```

- [ ] **Step 7: Run the test to verify it passes**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/test_lmm_vs_rmanova_per_dv.py -v`
Expected: PASS (all 6 cases now).

- [ ] **Step 8: Wire the pure function into the multi-DV loop**

Find the exact current loop via
`grep -n 'for dv_column in context\["dv_columns"\]:' src/autopilot/statistical_analyzer_autopilot_pipeline.py`.
Replace:

```python
            for dv_column in context["dv_columns"]:
                per_dv_context = dict(context)
                per_dv_context["dv_columns"] = [dv_column]
                per_dv_context["current_dv"] = dv_column
                QApplication.processEvents()
                all_results[dv_column] = self._execute_single_analysis(per_dv_context, dv_column, output_dir, skip_plots=True)
```

with:

```python
            for dv_column in context["dv_columns"]:
                per_dv_context = dict(context)
                per_dv_context["dv_columns"] = [dv_column]
                per_dv_context["current_dv"] = dv_column
                # U2 fix: re-derive the LMM-vs-base-test decision for THIS
                # column instead of reusing the once-computed batch decision.
                subject_column = context.get("subject_column")
                within_factors = context.get("within_factors")
                base_test = context.get("_test_before_lmm_upgrade")
                if subject_column and within_factors and base_test is not None:
                    per_dv_context["inferred_test"] = (
                        "lmm"
                        if _ap_lmm_vs_rmanova_needed(self.df, subject_column, within_factors[0], dv_column)
                        else base_test
                    )
                QApplication.processEvents()
                all_results[dv_column] = self._execute_single_analysis(per_dv_context, dv_column, output_dir, skip_plots=True)
```

- [ ] **Step 9: Run the full test suite**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/ -q --tb=no`
Expected: same pass count as before this task plus the 6 new tests (the 1 pre-existing unrelated
`test_convergence.py::test_convergence_keys` failure is expected — see the A1 plan's Task 4 for
how this was confirmed pre-existing and untouched by this session's work).

- [ ] **Step 10: Commit**

```bash
git add tests/test_lmm_vs_rmanova_per_dv.py src/autopilot/statistical_analyzer_autopilot_pipeline.py
git commit -m "fix(autopilot): recompute LMM-vs-RM-ANOVA decision per DV column in multi-DV mode"
```

---

## Self-review notes

- **Spec coverage:** U1 (Task 1), U2 (Task 2) — both findings assigned to this package are
  covered.
- **U2's design choice (extract a pure function) matches this codebase's own established
  pattern**, confirmed by reading `tests/test_analysis_context_subject_guard.py`'s docstring
  during planning, which explicitly documents preferring pure-function extraction over building
  a full Qt harness for exactly this class of bug.
- **U2 does not attempt to test the multi-DV loop inside `_ap_determine_and_run_test` end to
  end** — that function is deeply tied to the full `StatisticalAnalyzerApp` (dialogs, exports,
  progress UI). The pure function extracted in Step 3 carries all the actual decision logic and
  is fully covered; the loop wiring in Step 8 is a small, direct call to that already-tested
  function.
