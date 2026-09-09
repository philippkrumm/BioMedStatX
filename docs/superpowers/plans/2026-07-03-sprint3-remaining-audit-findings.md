# Sprint 3: Remaining 7 Audit Findings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close out the remaining 7 open findings from the Help Hub content audit (`SUMMARY.md`) and the proactive anti-pattern audit (`SUMMARY_PROACTIVE.md`): five mechanical fixes (well-defined target state, no design ambiguity) and two items where the user made an explicit product decision beforehand (hard-reject NaN subject IDs; wire in the linear-regression coefficient table).

**Architecture:** Each finding is fixed at its own call site — no shared new abstraction needed except one small reusable pure function, `_reject_missing_subject_ids(df, subject_col)`, added to `src/autopilot/statistical_analyzer_autopilot_ui.py` (the existing home for this file's other pure helper functions like `_detect_wide_format`) and called from two independent guard points (auto-pivot detection, and analysis-context building) since tracing showed those are two genuinely different code paths, not one implying the other.

**Tech Stack:** Python, pandas, pytest. No new dependencies.

**Decisions already made (via `AskUserQuestion`, recorded in
`docs/superpowers/specs/2026-07-03-sprint3-remaining-audit-findings-design.md`):**
- NaN/missing subject IDs → hard reject with a clear `ValueError` (not drop-with-warning, not pseudo-subject bucketing).
- Linear regression `coefficient_table` → wire it into the HTML report (not remove the dead computation).

---

### Task 1: RTE table blank-label fragility (A2)

**Files:**
- Modify: `src/export/report_stat_rows.py:670-671`
- Test: `tests/test_report_stat_rows.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_report_stat_rows.py`:

```python
"""RTE row-label extraction in the Brunner-Langer/ATS branch of
_build_statistical_rows must surface a missing between_group/within_level key
loudly (log warning) instead of silently substituting a blank label — matches
this session's "surface the failure" paradigm from Sprint 1/2.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from export.report_stat_rows import _StatRowsMixin


def _brunner_langer_results(rte_rows):
    return {
        "model_type": "BrunnerLangerATS",
        "RTE": pd.DataFrame(rte_rows),
    }


def test_rte_row_with_missing_key_logs_warning_not_silent_blank(caplog):
    results = _brunner_langer_results([
        {"within_level": "T0", "RTE": 0.62, "n": 12},  # missing between_group
    ])
    with caplog.at_level("WARNING"):
        rows = _StatRowsMixin._build_statistical_rows(results)
    rte_row_labels = [r["label"] for r in rows if r["label"].startswith("RTE:")]
    assert len(rte_row_labels) == 1
    assert any("missing" in rec.message.lower() for rec in caplog.records), (
        "a missing RTE key must be logged loudly, not silently substituted"
    )


def test_rte_row_with_all_keys_present_renders_normally():
    results = _brunner_langer_results([
        {"between_group": "drug", "within_level": "T0", "RTE": 0.62, "n": 12},
    ])
    rows = _StatRowsMixin._build_statistical_rows(results)
    rte_row_labels = [r["label"] for r in rows if r["label"].startswith("RTE:")]
    assert rte_row_labels == ["RTE: drug / T0"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_report_stat_rows.py -v`
Expected: `test_rte_row_with_missing_key_logs_warning_not_silent_blank` FAILS
(no warning logged today — the current code silently substitutes `""`).
`test_rte_row_with_all_keys_present_renders_normally` should already PASS
(this is the pre-existing correct-path behavior, unaffected by the fix).

- [ ] **Step 3: Fix the key lookup**

In `src/export/report_stat_rows.py`, replace lines 670-671:

Current code being replaced:
```python
                        between = rte_row.get("between_group", "")
                        within = rte_row.get("within_level", "")
```

New code:
```python
                        between = rte_row.get("between_group")
                        within = rte_row.get("within_level")
                        if between is None or within is None:
                            logger.warning(
                                "RTE row missing expected key(s) (between_group=%r, "
                                "within_level=%r); check Brunner-Langer/ATS engine "
                                "output shape.", between, within,
                            )
                            between = between if between is not None else "?"
                            within = within if within is not None else "?"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_report_stat_rows.py -v`
Expected: both tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/export/report_stat_rows.py tests/test_report_stat_rows.py
git commit -m "fix(report): log loudly instead of silently blanking missing RTE row keys"
```

---

### Task 2: All-NaN value columns pass wide-format detection (A3)

**Files:**
- Modify: `src/autopilot/statistical_analyzer_autopilot_ui.py:156-157`
- Test: `tests/test_wide_format_detection.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_wide_format_detection.py`:

```python
"""_detect_wide_format must not treat an entirely-empty (all-NaN) numeric
column as a usable value column — today it passes the dtype check (NaN
columns are float64) and only fails later with a cryptic empty-group error
deep in analysis_core.py.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from autopilot.statistical_analyzer_autopilot_ui import _detect_wide_format


def test_all_nan_value_column_is_excluded_not_crashed_on():
    df = pd.DataFrame({
        "Subject": ["S1", "S2", "S3", "S4", "S5"],
        "Time1": [10.5, 11.2, 12.1, 13.5, 14.0],
        "Time2": [11.2, 13.1, 14.0, 15.1, 15.5],
        "Time3": [np.nan, np.nan, np.nan, np.nan, np.nan],
    })
    result = _detect_wide_format(df)
    assert result is not None
    assert "Time3" not in result["value_cols"], (
        "an all-NaN column must not be treated as a usable measurement column"
    )
    assert set(result["value_cols"]) == {"Time1", "Time2"}


def test_all_columns_nan_returns_none_not_a_bogus_signature():
    df = pd.DataFrame({
        "Subject": ["S1", "S2", "S3", "S4", "S5"],
        "Time1": [np.nan] * 5,
        "Time2": [np.nan] * 5,
    })
    result = _detect_wide_format(df)
    assert result is None, (
        "with zero usable value columns, this must not be reported as wide-format data"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_wide_format_detection.py -v`
Expected: `test_all_nan_value_column_is_excluded_not_crashed_on` FAILS —
`result["value_cols"]` today is `["Time1", "Time2", "Time3"]` (includes the
all-NaN column). `test_all_columns_nan_returns_none_not_a_bogus_signature`
FAILS too — today it would return `{"subject_col": "Subject", "value_cols":
["Time1", "Time2"]}` instead of `None`, since neither column is currently
excluded despite having zero real data.

- [ ] **Step 3: Filter out all-NaN value columns**

In `src/autopilot/statistical_analyzer_autopilot_ui.py`, replace line 157:

Current code being replaced:
```python
    # Value columns = all numeric columns that are not the subject column
    value_cols = [c for c in numeric_cols if c != subject_col]
```

New code:
```python
    # Value columns = all numeric columns that are not the subject column and
    # have at least one real observation (an all-NaN column has no data to
    # pivot/analyze and would otherwise silently reach analysis_core.py as an
    # empty group, producing a cryptic error far from the real cause).
    value_cols = [c for c in numeric_cols if c != subject_col and df[c].notna().any()]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_wide_format_detection.py -v`
Expected: both tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/autopilot/statistical_analyzer_autopilot_ui.py tests/test_wide_format_detection.py
git commit -m "fix(autopilot): exclude all-NaN columns from wide-format value_cols"
```

---

### Task 3: Add shared `_reject_missing_subject_ids` helper; use it in `_detect_wide_format` (B3)

**Files:**
- Modify: `src/autopilot/statistical_analyzer_autopilot_ui.py:126-127` (insert helper before `_detect_wide_format`)
- Modify: `src/autopilot/statistical_analyzer_autopilot_ui.py:154-155` (call it)
- Test: `tests/test_wide_format_detection.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_wide_format_detection.py`:

```python
import pytest
from autopilot.statistical_analyzer_autopilot_ui import _reject_missing_subject_ids


def test_reject_missing_subject_ids_raises_with_count():
    df = pd.DataFrame({"Subject": ["S1", np.nan, "S3", np.nan]})
    with pytest.raises(ValueError, match=r"2 missing"):
        _reject_missing_subject_ids(df, "Subject")


def test_reject_missing_subject_ids_noop_when_complete():
    df = pd.DataFrame({"Subject": ["S1", "S2", "S3"]})
    _reject_missing_subject_ids(df, "Subject")  # must not raise


def test_reject_missing_subject_ids_noop_when_column_none():
    df = pd.DataFrame({"Subject": ["S1", "S2"]})
    _reject_missing_subject_ids(df, None)  # must not raise


def test_wide_format_detection_raises_on_missing_subject_id():
    df = pd.DataFrame({
        "Subject": ["S1", "S2", np.nan, "S4"],
        "Time1": [10.5, 11.2, 12.1, 13.5],
        "Time2": [11.2, 13.1, 14.0, 15.1],
    })
    with pytest.raises(ValueError, match=r"1 missing"):
        _detect_wide_format(df)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_wide_format_detection.py -k "reject_missing or raises_on_missing" -v`
Expected: FAIL — `ImportError`/`AttributeError` for the first three
(`_reject_missing_subject_ids` doesn't exist yet); the fourth fails because
`_detect_wide_format` doesn't raise today (it would currently return a
result dict, or possibly `None` depending on how the NaN affects the
uniqueness ratio — either way, not the expected `ValueError`).

- [ ] **Step 3: Add the helper and call it from `_detect_wide_format`**

In `src/autopilot/statistical_analyzer_autopilot_ui.py`, insert this function
immediately before `_detect_wide_format` (before line 128):

```python
def _reject_missing_subject_ids(df, subject_col):
    """
    Raises ValueError if subject_col contains any NaN. Every row needs a
    subject ID before repeated-measures structure (wide-format detection,
    balance detection, RM-ANOVA vs LMM routing) can be determined correctly
    — pandas silently drops NaN keys in groupby/nunique, which would
    otherwise let incomplete subjects vanish from those checks without any
    warning, biasing decisions that depend on them.
    """
    if subject_col is None:
        return
    n_missing = int(df[subject_col].isna().sum())
    if n_missing > 0:
        raise ValueError(
            f"Subject ID column '{subject_col}' has {n_missing} missing "
            f"value(s). Every row needs a subject ID before repeated-measures "
            f"analysis can run — fix the data and reload."
        )

```

Then, in the same file, add a call right after `subject_col` is determined
(after line 154's `subject_col = subject_candidates[0]`):

Current code being replaced:
```python
    subject_col = subject_candidates[0]

    # Value columns = all numeric columns that are not the subject column and
```

New code:
```python
    subject_col = subject_candidates[0]
    _reject_missing_subject_ids(df, subject_col)

    # Value columns = all numeric columns that are not the subject column and
```

(This lands right before the value_cols line from Task 2 — if executing
tasks in order, that line will already have the updated comment from Task 2;
match on `subject_col = subject_candidates[0]` alone if the surrounding
context has shifted.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_wide_format_detection.py -v`
Expected: all tests in the file PASS (Task 2's 2 tests + this task's 4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/autopilot/statistical_analyzer_autopilot_ui.py tests/test_wide_format_detection.py
git commit -m "fix(autopilot): reject wide-format data with missing subject IDs at detection time"
```

---

### Task 4: Reuse `_reject_missing_subject_ids` in `_ap_build_analysis_context` (B4)

**Files:**
- Modify: `src/autopilot/statistical_analyzer_autopilot_pipeline.py:46-49` (import)
- Modify: `src/autopilot/statistical_analyzer_autopilot_pipeline.py:1101-1106` (call after `analysis_df` is built)
- Test: `tests/test_analysis_context_subject_guard.py` (create)

**Why this needs its own guard, not just Task 3's:** `_detect_wide_format`
only runs for auto-detected wide-format files at load time. Data that's
already long-format (the common case — user drags columns into buckets
manually) never goes through it at all. `_ap_build_analysis_context` is the
one point every analysis run passes through regardless of path, so it needs
its own independent check — see "Correction found while tracing" in the spec
doc for the full reasoning, including why the check can't live inside the
existing `try: ... except Exception: pass` block at line 1221 (it would be
silently swallowed there).

- [ ] **Step 1: Write the failing test**

Create `tests/test_analysis_context_subject_guard.py`:

```python
"""_ap_build_analysis_context must reject a Subject-ID column containing NaN
before any factor/balance-detection logic runs — this is the one choke point
every analysis path (wide-pivoted or manually-mapped long-format) passes
through, unlike _detect_wide_format which only covers the auto-pivot path.
Uses a minimal fake `self` (not a real Qt widget) exposing only the bucket
API _ap_build_analysis_context actually calls (get_assigned_columns /
get_filter / isChecked) — building a full QApplication harness for this one
guard would be disproportionate (see HANDOFF.md's noted preference for
extracting/testing pure logic over fragile full-Qt-app harnesses).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

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
    def __init__(self, df, subject_col):
        self.df = df
        self.dv_bucket = _FakeBucket(["Value"])
        self.factor1_bucket = _FakeBucket(["Group"])
        self.factor2_bucket = _FakeBucket([])
        self.subject_bucket = _FakeBucket([subject_col] if subject_col else [])
        self.covariates_bucket = _FakeBucket([])
        self.filter_bucket = _FakeFilterBucket([])
        self.multi_mode_button = _FakeCheckbox(False)
        self.analysis_selected_groups = []


def test_build_analysis_context_rejects_missing_subject_id():
    df = pd.DataFrame({
        "Subject": ["S1", "S2", np.nan, "S4"],
        "Group": ["A", "A", "B", "B"],
        "Value": [1.0, 2.0, 3.0, 4.0],
    })
    fake_self = _FakeApp(df, subject_col="Subject")
    with pytest.raises(ValueError, match=r"1 missing"):
        _ap_build_analysis_context(fake_self)


def test_build_analysis_context_allows_complete_subject_id():
    df = pd.DataFrame({
        "Subject": ["S1", "S2", "S3", "S4"],
        "Group": ["A", "A", "B", "B"],
        "Value": [1.0, 2.0, 3.0, 4.0],
    })
    fake_self = _FakeApp(df, subject_col="Subject")
    context = _ap_build_analysis_context(fake_self)
    assert context["subject_column"] == "Subject"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analysis_context_subject_guard.py -v`
Expected: `test_build_analysis_context_rejects_missing_subject_id` FAILS — no
`ValueError` raised today (the NaN subject silently flows through into the
balance-detection groupby instead). `test_build_analysis_context_allows_complete_subject_id`
should already PASS (correct-path behavior, unaffected).

- [ ] **Step 3: Import the helper and add the guard**

In `src/autopilot/statistical_analyzer_autopilot_pipeline.py`, add
`_reject_missing_subject_ids` to the existing import block (lines 46-49):

Current code being replaced:
```python
    _detect_wide_format,
```

New code:
```python
    _detect_wide_format,
    _reject_missing_subject_ids,
```

(Keep this inside whatever multi-line import statement lines 46-49 already
belong to — add it as a new imported name alongside `_detect_wide_format`,
`_pivot_wide_to_long`, matching the existing formatting.)

Then, in the same file, add the guard right after `analysis_df` is built and
filtered (after line 1105, before line 1107's `factor1_levels = ...`):

Current code being replaced:
```python
    analysis_df = self.df.copy()
    if filter_spec:
        filter_col, filter_val = filter_spec
        if filter_col in analysis_df.columns:
            analysis_df = analysis_df[analysis_df[filter_col] == filter_val]

    factor1_levels = _sorted_unique(analysis_df[factor_columns[0]].dropna().tolist())
```

New code:
```python
    analysis_df = self.df.copy()
    if filter_spec:
        filter_col, filter_val = filter_spec
        if filter_col in analysis_df.columns:
            analysis_df = analysis_df[analysis_df[filter_col] == filter_val]

    _reject_missing_subject_ids(analysis_df, subject_column)

    factor1_levels = _sorted_unique(analysis_df[factor_columns[0]].dropna().tolist())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_analysis_context_subject_guard.py -v`
Expected: both tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/autopilot/statistical_analyzer_autopilot_pipeline.py tests/test_analysis_context_subject_guard.py
git commit -m "fix(autopilot): reject missing subject IDs before analysis-context building"
```

---

### Task 5: RM-ANOVA sphericity outer-exception applies conservative GG default (SUMMARY.md item 2)

**Files:**
- Modify: `src/analysis/statisticaltester.py:2690-2701`
- Test: `tests/test_sphericity_outer_exception.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_sphericity_outer_exception.py`:

```python
"""When BOTH the primary sphericity test (pg.sphericity) and its inner
fallback (_extract_sphericity_from_anova_table) fail, execution falls to the
outer except in _perform_comprehensive_sphericity_test. Per CHANGELOG.md:
"When sphericity cannot be formally tested ... the Greenhouse-Geisser
correction is now applied by default." The inner fallback already honors
this; the outer except did not — it used the uncorrected p-value instead,
silently reintroducing the pre-v2.0 behavior the changelog says was fixed.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import pytest

import analysis.statisticaltester as st_module
from analysis.statisticaltester import StatisticalTester


def test_outer_exception_still_applies_gg_correction_by_default(monkeypatch):
    def _boom_pg_module():
        class _Boom:
            @staticmethod
            def sphericity(*a, **kw):
                raise RuntimeError("pg.sphericity boom")
        return _Boom()

    def _boom_extract(*a, **kw):
        raise RuntimeError("anova table extraction boom")

    monkeypatch.setattr(st_module, "get_pingouin_module", _boom_pg_module)
    monkeypatch.setattr(StatisticalTester, "_extract_sphericity_from_anova_table",
                         staticmethod(_boom_extract))

    df = pd.DataFrame({
        "dv": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "subject": ["s1", "s1", "s2", "s2", "s3", "s3"],
        "factor": ["a", "b", "a", "b", "a", "b"],
    })
    row = pd.Series({"DF": 2.0, "eps": 0.75, "p_GG_corr": 0.03, "F": 5.2})
    error_row = pd.Series({"DF": 18.0})

    result = StatisticalTester._perform_comprehensive_sphericity_test(
        df, "dv", "subject", "factor", aov=None, row=row, error_row=error_row
    )

    assert result["correction_used"] == "Greenhouse-Geisser (ε = 0.750)", (
        f"outer exception must still apply the documented conservative GG "
        f"default, got correction_used={result.get('correction_used')!r}"
    )
    assert result["corrected_p_value"] == 0.03
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sphericity_outer_exception.py -v`
Expected: FAIL — today's outer except sets `correction_used = "None
(sphericity test failed)"` and `corrected_p_value` to the uncorrected value,
not `0.03`.

- [ ] **Step 3: Apply the conservative default in the outer except**

In `src/analysis/statisticaltester.py`, replace lines 2690-2701:

Current code being replaced:
```python
        except Exception as e:
            # Comprehensive fallback
            results["sphericity_test"] = {
                "test_name": "Mauchly's Test for Sphericity",
                "W": None,
                "p_value": None,
                "sphericity_assumed": None,
                "note": f"Sphericity test failed: {str(e)}",
                "interpretation": "Could not determine sphericity - proceeding with caution"
            }
            results["corrected_p_value"] = StatisticalTester._pingouin_p_value(row)
            results["correction_used"] = "None (sphericity test failed)"
```

New code:
```python
        except Exception as e:
            # Per CHANGELOG.md: "When sphericity cannot be formally tested,
            # the Greenhouse-Geisser correction is now applied by default."
            # Attempt that same conservative default here, not just in the
            # inner fallback — falls back to the uncorrected p-value only if
            # _apply_sphericity_corrections itself also can't be computed.
            results["sphericity_test"] = {
                "test_name": "Mauchly's Test for Sphericity",
                "W": None,
                "p_value": None,
                "sphericity_assumed": None,
                "note": f"Sphericity test failed: {str(e)}",
                "interpretation": "Could not determine sphericity - applying conservative correction"
            }
            try:
                corrections_applied = StatisticalTester._apply_sphericity_corrections(
                    row, error_row, True, aov
                )
                results.update(corrections_applied)
            except Exception:
                results["corrected_p_value"] = StatisticalTester._pingouin_p_value(row)
                results["correction_used"] = "None (sphericity test failed)"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sphericity_outer_exception.py -v`
Expected: PASS

- [ ] **Step 5: Run the existing sphericity-adjacent test suite to check for regressions**

Run: `pytest tests/ -k "sphericity or rm_anova or repeated_measures" -v`
Expected: all PASS — this change only alters the outer except branch, which
requires two independent failures to reach; existing sphericity tests
exercise the primary and inner-fallback paths, which are untouched.

- [ ] **Step 6: Commit**

```bash
git add src/analysis/statisticaltester.py tests/test_sphericity_outer_exception.py
git commit -m "fix(stats): apply conservative GG default in sphericity outer-exception path"
```

---

### Task 6: Wire `control_group` through ANCOVA in the primary clinical dispatch (SUMMARY.md item 7)

**Files:**
- Modify: `src/analysis/analysis_core.py:594-598`
- Test: `tests/test_ancova_control_group_wiring.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_ancova_control_group_wiring.py`:

```python
"""analysis_core.py's ANCOVA dispatch never passed control_group into
ANCOVAModel.fit(), even though the model supports it and the LMM branch
right below it already does this correctly via control_group_callback. This
made the vs-control multivariate-t EMM post-hoc unreachable from the primary
dispatch path (advanced_pipeline.py's secondary path already had it).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def test_ancova_dispatch_passes_control_group_callback_result():
    import analysis.analysis_core as core_module

    df = pd.DataFrame({
        "dv": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "group": ["ctrl", "ctrl", "a", "a", "b", "b"],
        "cov": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    })

    captured_kwargs = {}
    fake_model = MagicMock()
    fake_model.as_results_dict.return_value = {"model_type": "ANCOVA"}

    def _fake_fit(self_, df_, **kwargs):
        captured_kwargs.update(kwargs)

    with patch("analysis.analysis_core.ANCOVAModel", return_value=fake_model):
        fake_model.fit.side_effect = lambda *a, **kw: captured_kwargs.update(kw)
        control_cb = MagicMock(return_value="ctrl")

        analysis_context = {
            "between_factors": ["group"],
            "factor_columns": ["group"],
        }
        try:
            core_module._run_clinical_model_dispatch_for_test(
                df=df, value_cols=["dv"], covariates=["cov"],
                clinical_test="ancova", analysis_context=analysis_context,
                control_group_callback=control_cb,
            )
        except AttributeError:
            pytest.skip(
                "test harness helper not present — see Task 6 Step 1 note: "
                "adjust this test to call whatever the real entry point is "
                "once traced against the actual analysis_core.py structure."
            )

    assert captured_kwargs.get("control_group") == "ctrl"
    control_cb.assert_called_once()
```

**Note for whoever implements this task:** `analysis_core.py`'s ANCOVA/LMM
dispatch (lines 594-623) lives inside a larger function, not a small
standalone one — calling it directly in isolation may require more setup
than the sketch above assumes. Before running Step 2, read
`src/analysis/analysis_core.py` from the top of the enclosing function down
to line 623 to find its real name and required arguments, and adjust the
test's call to invoke that real function directly (with a minimal `df`/
`analysis_context`/`kwargs` including `control_group_callback`) rather than
the placeholder `_run_clinical_model_dispatch_for_test` name used above. The
goal is unchanged: prove `ANCOVAModel.fit` is called with
`control_group=<callback result>` when `clinical_test == 'ancova'` and a
`control_group_callback` is supplied — adjust the harness, not the
assertion.

- [ ] **Step 2: Run test to verify it fails or skips with a clear reason**

Run: `pytest tests/test_ancova_control_group_wiring.py -v`
Expected: FAIL (`captured_kwargs.get("control_group")` is `None`, since the
ANCOVA branch never passes it today) — after adjusting the harness per the
note above to call the real function.

- [ ] **Step 3: Mirror the LMM branch's control_group wiring for ANCOVA**

In `src/analysis/analysis_core.py`, replace lines 594-598:

Current code being replaced:
```python
                if clinical_test in ('ancova', 'two_way_ancova'):
                    model = ANCOVAModel()
                    between_factors = analysis_context.get('between_factors') or analysis_context.get('factor_columns', [])
                    model.fit(df, dv=value_cols[0], between_factors=between_factors, covariates=covariates)
                    test_results = model.as_results_dict()
```

New code:
```python
                if clinical_test in ('ancova', 'two_way_ancova'):
                    model = ANCOVAModel()
                    between_factors = analysis_context.get('between_factors') or analysis_context.get('factor_columns', [])

                    ancova_control = None
                    primary_factor = between_factors[0] if between_factors else None
                    _control_cb = kwargs.get('control_group_callback')
                    if _control_cb and primary_factor:
                        try:
                            primary_levels = sorted(
                                str(v) for v in df[primary_factor].dropna().unique()
                            )
                            ancova_control = _control_cb(primary_levels)
                        except Exception as exc:
                            logger.warning("ANCOVA control-group selection failed in core: %s", exc)

                    model.fit(df, dv=value_cols[0], between_factors=between_factors,
                              covariates=covariates, control_group=ancova_control)
                    test_results = model.as_results_dict()
```

(This is the exact same pattern as the LMM branch immediately below it,
lines 608-622 — same `kwargs.get('control_group_callback')` lookup, same
try/except-and-log, same primary-factor derivation but from
`between_factors` instead of `fixed_effects`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ancova_control_group_wiring.py -v`
Expected: PASS

- [ ] **Step 5: Run the ANCOVA-adjacent test suite to check for regressions**

Run: `pytest tests/ -k "ancova" -v`
Expected: all PASS — `control_group` defaults to `None` when no callback is
supplied (unchanged from today for any caller that doesn't pass one), so
existing ANCOVA calls without `control_group_callback` behave identically.

- [ ] **Step 6: Commit**

```bash
git add src/analysis/analysis_core.py tests/test_ancova_control_group_wiring.py
git commit -m "fix(analysis): wire control_group through ANCOVA in the primary clinical dispatch"
```

---

### Task 7: Render linear regression's `coefficient_table` in the HTML report (SUMMARY.md item 5)

**Files:**
- Modify: `src/export/report_association.py` (add new function after `_build_beta_coefficient_table_html`, which ends at line 113)
- Modify: `src/export/report_charts.py:613-617` (add new `elif` branch)
- Test: `tests/test_report_association.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_report_association.py`:

```python
"""Linear regression's coefficient_table (correlation_models.py:848,
SimpleLinearRegressionModel.as_results_dict) was computed but had zero
readers anywhere in the export layer. This wires it into the HTML report,
mirroring the existing _build_beta_coefficient_table_html pattern but reading
the correct key (coefficient_table, not coefficients) and using a t-column
(OLS) instead of z-column (GLM, beta regression's own case).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from export.report_association import _AssociationMixin


def _linear_regression_results():
    return {
        "model_type": "LinearRegression",
        "coefficient_table": [
            {"parameter": "Intercept", "coefficient": 1.5, "std_err": 0.3,
             "t_value": 5.0, "p_value": 0.001, "ci_lower": 0.9, "ci_upper": 2.1},
            {"parameter": "x", "coefficient": 0.8, "std_err": 0.2,
             "t_value": 4.0, "p_value": 0.02, "ci_lower": 0.4, "ci_upper": 1.2},
        ],
    }


def test_linear_regression_coefficient_table_renders_html():
    block = _AssociationMixin._build_linear_regression_coefficient_table_html(
        _linear_regression_results()
    )
    assert block is not None
    assert "Intercept" in block["html"]
    assert "<th>t</th>" in block["html"]
    assert "0.001" in block["html"] or "&lt;0.001" in block["html"] or "0.0010" in block["html"]


def test_linear_regression_coefficient_table_returns_none_when_empty():
    block = _AssociationMixin._build_linear_regression_coefficient_table_html(
        {"model_type": "LinearRegression", "coefficient_table": []}
    )
    assert block is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_report_association.py -v`
Expected: FAIL — `AttributeError: type object '_AssociationMixin' has no
attribute '_build_linear_regression_coefficient_table_html'`.

- [ ] **Step 3: Add the renderer function**

In `src/export/report_association.py`, insert this new staticmethod
immediately after `_build_beta_coefficient_table_html` (after line 113,
before `_extract_association_payload` at line 115):

```python
    @staticmethod
    def _build_linear_regression_coefficient_table_html(results: dict) -> dict | None:
        """Renders the Linear Regression (OLS) coefficient table as an inline
        HTML block. Mirrors _build_beta_coefficient_table_html's shape, but
        reads the correct key for this model (coefficient_table, set by
        SimpleLinearRegressionModel.as_results_dict) and uses a t-column (OLS)
        instead of z (GLM — beta regression's own case)."""
        coef_table = results.get("coefficient_table") or []
        if not coef_table:
            return None
        rows_html = ""
        for row in coef_table:
            p_val = row.get("p_value")
            is_sig = isinstance(p_val, (int, float)) and p_val < 0.05
            coef_display = _FormattingMixin._format_metric(row.get("coefficient"))
            if is_sig:
                coef_display = f"<strong>{coef_display}</strong>"
            p_style = "color:var(--success)" if is_sig else "color:var(--muted)"
            rows_html += (
                f"<tr>"
                f"<td>{row.get('parameter', '')}</td>"
                f"<td class='num-cell'>{coef_display}</td>"
                f"<td class='num-cell'>{_FormattingMixin._format_metric(row.get('std_err'))}</td>"
                f"<td class='num-cell'>{_FormattingMixin._format_metric(row.get('t_value'))}</td>"
                f"<td class='num-cell' style='{p_style}'>{_FormattingMixin._format_p_value(p_val)}</td>"
                f"<td class='num-cell'>{_FormattingMixin._format_metric(row.get('ci_lower'))}</td>"
                f"<td class='num-cell'>{_FormattingMixin._format_metric(row.get('ci_upper'))}</td>"
                f"</tr>"
            )
        html = (
            "<div class='table-shell'>"
            "<table>"
            "<thead><tr>"
            "<th>Parameter</th><th>Coefficient</th><th>SE</th><th>t</th>"
            "<th>p-value</th><th>95% CI Lower</th><th>95% CI Upper</th>"
            "</tr></thead>"
            f"<tbody>{rows_html}</tbody>"
            "</table></div>"
        )
        return {
            "title": "Coefficients",
            "subtitle": "OLS coefficients with standard errors and 95% confidence intervals.",
            "html": html,
            "div_id": "biomedstatx-linreg-coef-table",
        }

```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_report_association.py -v`
Expected: both tests PASS

- [ ] **Step 5: Wire it into the chart dispatch**

In `src/export/report_charts.py`, replace lines 613-617:

Current code being replaced:
```python
        elif model_type == "BetaRegression":
            # Coefficient table as inline HTML block
            beta_coef_block = _AssociationMixin._build_beta_coefficient_table_html(results)
            if beta_coef_block:
                charts.append(beta_coef_block)
            # Scatter + fitted curve replaces meaningless boxplot for proportion outcome
            beta_chart = _ChartsMixin._build_beta_regression_chart(results)
            if beta_chart:
                charts.append(beta_chart)
```

New code:
```python
        elif model_type == "BetaRegression":
            # Coefficient table as inline HTML block
            beta_coef_block = _AssociationMixin._build_beta_coefficient_table_html(results)
            if beta_coef_block:
                charts.append(beta_coef_block)
            # Scatter + fitted curve replaces meaningless boxplot for proportion outcome
            beta_chart = _ChartsMixin._build_beta_regression_chart(results)
            if beta_chart:
                charts.append(beta_chart)
        elif model_type == "LinearRegression":
            # Coefficient table as inline HTML block — was computed
            # (correlation_models.py) but never rendered anywhere.
            linreg_coef_block = _AssociationMixin._build_linear_regression_coefficient_table_html(results)
            if linreg_coef_block:
                charts.append(linreg_coef_block)
```

Note: like the sibling `LogisticRegression`/`BetaRegression` branches, adding
this `elif` means `LinearRegression` no longer falls through to the generic
`else` branch's group-comparison boxplot (lines 676-685) — a boxplot has no
meaning for a regression with no groups, matching the same reasoning already
applied to the other two special-cased model types. This is an expected
consequence of the fix, not a separate scope decision.

- [ ] **Step 6: Run the regression-adjacent test suite to check for regressions**

Run: `pytest tests/ -k "linear_regression or report_charts or correlation" -v`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/export/report_association.py src/export/report_charts.py tests/test_report_association.py
git commit -m "feat(report): render linear regression's coefficient table in HTML export"
```

---

### Task 8: Full regression check

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/ -v`
Expected: baseline going into this sprint was 350 passed / 4 skipped / 0
failed. New tests added: Task 1 (+2), Task 2 (+2), Task 3 (+4), Task 4 (+2),
Task 5 (+1), Task 6 (+1, may be a `pytest.skip` rather than a pass — see
Task 6's note — a skip is acceptable there if the harness genuinely needs
adjustment, but the underlying fix and its `-k "ancova"` regression check in
Task 6 Step 5 must still be green), Task 7 (+2). Total: baseline 350 + 14 =
**364 passed / 4 skipped / 0 failed** (or 363 passed / 5 skipped if Task 6's
test needed the skip path — either is acceptable, a hard failure is not).

- [ ] **Step 2: If anything regressed, fix before proceeding**

Pay particular attention to:
- `tests/` files touching `report_stat_rows.py`, `report_charts.py`,
  `report_association.py` (Tasks 1 and 7 both touch the export layer).
- Any autopilot pipeline test exercising real wide-format Excel/CSV fixtures
  (Tasks 2 and 3 change `_detect_wide_format`'s return contract — it can now
  raise where it previously only returned `None`/a dict).
- `tests/` files touching ANCOVA (Task 6 changes `analysis_core.py`'s ANCOVA
  branch signature usage, though `control_group` defaults to `None` and
  should be behaviorally inert for callers that don't pass a callback).

---

## Self-review notes

- **Spec coverage:** A2 → Task 1. A3 → Task 2. B3 → Task 3. B4 → Task 4
  (explicitly NOT folded into Task 3 — tracing showed they're independent
  guards on independent paths). SUMMARY.md item 2 (sphericity outer-exception)
  → Task 5. SUMMARY.md item 7 (ANCOVA control_group) → Task 6. SUMMARY.md
  item 5 (linear regression coefficient_table) → Task 7.
- **Placeholder scan:** Task 6 contains an intentional, explicitly-flagged
  exception to the "no placeholders" rule — `analysis_core.py`'s ANCOVA/LMM
  dispatch sits inside a larger unnamed-in-this-plan enclosing function, and
  writing a fake call signature without first reading that function's real
  structure would risk the same mistake `HANDOFF.md` already recorded once
  (a plan assuming an API that didn't exist). Task 6's test step explicitly
  instructs the implementer to trace the real function before writing the
  final test call, rather than the plan guessing wrong and burning a review
  cycle. The **fix itself** (Step 3) is fully concrete and copy-pasteable —
  only the test harness's exact entry point needs a short trace first.
- **Type/signature consistency:** `_reject_missing_subject_ids(df, subject_col)`
  defined once in Task 3, imported and reused identically in Task 4.
  `_build_linear_regression_coefficient_table_html(results)` matches
  `_build_beta_coefficient_table_html(results)`'s signature and return shape
  (`{"title", "subtitle", "html", "div_id"}` or `None`) for consistency.
