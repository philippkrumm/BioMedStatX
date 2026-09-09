# Audit-Flagged Code Bug Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the 6 mechanical, single-correct-answer code bugs flagged by the Help Hub content audit (`docs/superpowers/audit-notes/SUMMARY.md`), each with a TDD test that reproduces the bug before the fix.

**Architecture:** One task per fix, each touching one file (Fix 3 also touches one recipe string in `help_content.py` as a direct consequence). No shared infrastructure changes. Tests use targeted monkeypatching to isolate the exact logic under test without invoking expensive full pipelines where avoidable.

**Tech Stack:** Python 3.12, pytest 7.4 (headless Qt via root `conftest.py`, `QT_QPA_PLATFORM=offscreen`, `src/` on `sys.path`). Spec: `docs/superpowers/specs/2026-07-02-audit-code-bug-fixes-design.md`.

---

### Task 1: Post-hoc label desync (Two-Way / Mixed / RM ANOVA)

**Files:**
- Modify: `src/statistical_testing/advanced_pipeline.py:256-266`
- Test: `tests/test_advanced_pipeline_posthoc_label.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_advanced_pipeline_posthoc_label.py`:

```python
"""Guards the fix for the post-hoc-label desync bug found in the Help Hub content
audit: `posthoc_test` must always be resynced to the actually-applied method once
`pairwise_comparisons` is replaced by the advanced post-hoc engine, regardless of
what the inline (pre-advanced-engine) label happened to say.
"""
import pandas as pd
import pytest

from statistical_testing.advanced_pipeline import perform_advanced_test_pipeline
from statistical_testing.engines.advanced_posthoc import AdvancedPostHocEngine
from statistical_testing.models import StatisticalResult


def _canned_posthoc_result(new_label):
    return StatisticalResult(
        test_name=new_label,
        statistic_value=None,
        p_value=None,
        metadata={
            "pairwise_comparisons": [{"group1": "A", "group2": "B", "p_value": 0.01}],
            "posthoc_test": new_label,
        },
    )


@pytest.mark.parametrize(
    "test_name,stale_inline_label,real_method_name",
    [
        ("two_way_anova", "Tukey HSD Test (Pingouin)", "Custom paired t-tests (Holm-Sidak)"),
        ("mixed_anova", "Pairwise t-tests for interaction (Holm-Bonferroni)", "Tukey HSD (Mixed)"),
        ("repeated_measures_anova", "Paired t-tests (Holm-Bonferroni)", "Tukey HSD (RM)"),
    ],
)
def test_posthoc_label_synced_to_real_method(
    monkeypatch, test_name, stale_inline_label, real_method_name
):
    from analysis.statisticaltester import StatisticalTester

    def _canned_initial_result(*args, **kwargs):
        return {
            "p_value": 0.001,
            "posthoc_test": stale_inline_label,
            "test_info": None,
        }

    monkeypatch.setattr(
        StatisticalTester, "_run_two_way_anova_logged",
        staticmethod(_canned_initial_result), raising=False,
    )
    monkeypatch.setattr(
        StatisticalTester, "_run_mixed_anova_logged",
        staticmethod(_canned_initial_result), raising=False,
    )
    monkeypatch.setattr(
        StatisticalTester, "_run_repeated_measures_anova_logged",
        staticmethod(_canned_initial_result), raising=False,
    )
    monkeypatch.setattr(
        AdvancedPostHocEngine, "execute",
        lambda self, data: _canned_posthoc_result(real_method_name),
    )

    df = pd.DataFrame({"dv": [1, 2, 3, 4], "grp": ["A", "A", "B", "B"]})
    result = perform_advanced_test_pipeline(
        df=df,
        test=test_name,
        dv="dv",
        subject=None,
        between=["grp"] if test_name != "repeated_measures_anova" else None,
        within=["grp"] if test_name != "two_way_anova" else None,
        force_parametric=True,
        alpha=0.05,
    )

    assert result["posthoc_test"] == real_method_name
    assert result["posthoc_test"] != stale_inline_label
```

- [ ] **Step 2: Run test to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_advanced_pipeline_posthoc_label.py -v`
Expected: FAIL for the `mixed_anova` and `repeated_measures_anova` cases (their stale
labels don't match any of the five old `should_override` patterns, so
`result["posthoc_test"]` stays the stale label). The `two_way_anova` case may also
fail depending on exact string matching — confirm which of the three fail before
proceeding.

- [ ] **Step 3: Fix the override logic**

In `src/statistical_testing/advanced_pipeline.py`, replace lines 256-266:

```python
                    current_posthoc = res.get("posthoc_test", "")
                    new_posthoc = advanced_posthoc_updates.get("posthoc_test") or advanced_posthoc_result.test_name
                    should_override = (
                        not current_posthoc
                        or current_posthoc == "Two-Way ANOVA Post-hoc Tests"
                        or "parametric paired t-tests" in current_posthoc.lower()
                        or "pairwise paired t-tests" in current_posthoc.lower()
                        or ("Pingouin" in str(current_posthoc) and new_posthoc and "Tukey" in str(new_posthoc))
                    )
                    if should_override:
                        res["posthoc_test"] = new_posthoc
```

with:

```python
                    new_posthoc = advanced_posthoc_updates.get("posthoc_test") or advanced_posthoc_result.test_name
                    res["posthoc_test"] = new_posthoc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_advanced_pipeline_posthoc_label.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git -C /Users/philippkrumm/Documents/BioMedStatX add src/statistical_testing/advanced_pipeline.py tests/test_advanced_pipeline_posthoc_label.py
git -C /Users/philippkrumm/Documents/BioMedStatX commit -m "fix(stats): always sync posthoc_test label to the method actually applied

Removes should_override's fragile string-pattern-matching guard. The
pairwise_comparisons are unconditionally replaced by the advanced
post-hoc engine's output right above this block; the reported method
name must always match, not just when the stale inline label happens
to match one of five hardcoded patterns. Fixes a real, user-visible
mislabel in Two-Way ANOVA, Mixed ANOVA, and Repeated Measures ANOVA
default post-hoc runs (flagged in the Help Hub content audit).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Dead Strip branch and silent plot-type fallback

**Files:**
- Modify: `src/analysis/analysis_core.py:1500-1535`
- Test: `tests/test_plot_type_dispatch.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_plot_type_dispatch.py`:

```python
"""Guards the fix for the dead 'Strip' plot-type branch and the silent
wrong-plot-type fallback in the export dispatch, found in the Help Hub content
audit. An unrecognized plot_type must raise, matching the preview dispatch's
behavior, instead of silently rendering a Bar plot.
"""
import pandas as pd
import pytest

from analysis.analysis_core import AnalysisManager


@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch):
    from PyQt5.QtWidgets import QDialog
    monkeypatch.setattr(QDialog, "exec_", lambda self, *a, **k: 0, raising=False)
    monkeypatch.setattr(QDialog, "exec", lambda self, *a, **k: 0, raising=False)
    from analysis.statisticaltester import UIDialogManager
    monkeypatch.setattr(UIDialogManager, "select_transformation_dialog",
                        staticmethod(lambda *a, **k: "log10"), raising=False)
    monkeypatch.setattr(UIDialogManager, "select_posthoc_test_dialog",
                        staticmethod(lambda *a, **k: "tukey"), raising=False)
    for name in ("select_nonparametric_posthoc_dialog",
                 "select_control_group_dialog", "select_custom_pairs_dialog"):
        monkeypatch.setattr(UIDialogManager, name,
                            staticmethod(lambda *a, **k: None), raising=False)


@pytest.fixture
def dummy_file(tmp_path):
    path = tmp_path / "dummy.xlsx"
    pd.DataFrame({"a": [1]}).to_excel(path, index=False)
    return str(path)


def _run(dummy_file, tmp_path, plot_type):
    df = pd.DataFrame({
        "Grp": ["Control", "Control", "Control", "Treatment", "Treatment", "Treatment"],
        "Val": [1.0, 2.0, 1.5, 5.0, 6.0, 5.5],
    })
    ctx = {
        "injected_df": df,
        "factor_columns": ["Grp"],
        "dv_columns": ["Val"],
        "group_labels": ["Control", "Treatment"],
        "mode": "single",
    }
    return AnalysisManager.analyze(
        file_path=dummy_file,
        group_col="Grp",
        groups=["Control", "Treatment"],
        value_cols=["Val"],
        save_plot=False,
        skip_plots=False,
        plot_type=plot_type,
        file_name=str(tmp_path / "out"),
        analysis_context=ctx,
    )


def test_unrecognized_plot_type_raises(dummy_file, tmp_path):
    with pytest.raises(ValueError, match="Unknown plot type"):
        _run(dummy_file, tmp_path, plot_type="NotARealPlotType")


def test_strip_is_no_longer_a_special_case(dummy_file, tmp_path):
    with pytest.raises(ValueError, match="Unknown plot type"):
        _run(dummy_file, tmp_path, plot_type="Strip")


def test_bar_still_renders(dummy_file, tmp_path):
    result = _run(dummy_file, tmp_path, plot_type="Bar")
    assert result is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_plot_type_dispatch.py -v`
Expected: `test_unrecognized_plot_type_raises` and `test_strip_is_no_longer_a_special_case`
FAIL (no `ValueError` raised today — both currently succeed silently with a Bar-plot
fallback). `test_bar_still_renders` should already PASS (baseline behavior check).

- [ ] **Step 3: Fix the dispatch**

In `src/analysis/analysis_core.py`, delete the dead `elif plot_type == "Strip":` branch
(lines 1500-1512):

```python
                elif plot_type == "Strip":
                    # Strip plot doesn't exist, fall back to box plot with points
                    plot_kwargs['show_points'] = plot_kwargs.get('show_points', True)
                    plot_kwargs['point_size'] = plot_kwargs.get('point_size', 80)
                    plot_kwargs['point_alpha'] = plot_kwargs.get('point_alpha', 0.8)
                    fig, ax = DataVisualizer.plot_box(
                        groups, filtered_samples, width=width, height=height,
                        colors=colors, hatches=hatches,
                        test_recommendation=test_recommendation,
                        x_label=x_label, y_label=y_label,
                        title=title, save_plot=save_plot,
                        pairwise_results=pairwise_comparisons,
                        file_name=file_base, legend_colors=colors, **plot_kwargs)
```

Replace the catch-all `else` (lines 1522-1535):

```python
                else:
                    # Fallback to bar plot for unknown plot types
                    logger.warning(f"WARNING: Unknown plot type '{plot_type}', falling back to Bar plot")
                    plot_kwargs['show_points'] = plot_kwargs.get('show_points', True)
                    plot_kwargs['point_size'] = plot_kwargs.get('point_size', 80)
                    plot_kwargs['point_alpha'] = plot_kwargs.get('point_alpha', 0.8)
                    fig, ax = DataVisualizer.plot_bar(
                        groups, filtered_samples, width=width, height=height,
                        colors=colors, hatches=hatches, compare=compare,
                        test_recommendation=test_recommendation,
                        x_label=x_label, y_label=y_label,
                        title=title, save_plot=save_plot, error_type=error_type,
                        pairwise_results=pairwise_comparisons,
                        file_name=file_base, legend_colors=colors, **plot_kwargs)
```

with:

```python
                else:
                    raise ValueError(f"Unknown plot type: {plot_type!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_plot_type_dispatch.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git -C /Users/philippkrumm/Documents/BioMedStatX add src/analysis/analysis_core.py tests/test_plot_type_dispatch.py
git -C /Users/philippkrumm/Documents/BioMedStatX commit -m "fix(plots): raise on unrecognized plot_type instead of silent Bar fallback

Deletes the dead 'Strip' branch (unreachable since the dropdown was
pinned to Bar/Box/Violin/Raincloud) and matches the preview dispatch's
ValueError behavior for the export dispatch. Flagged in the Help Hub
content audit as a latent trap for future plot-type additions.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: German checkbox label

**Files:**
- Modify: `src/autopilot/statistical_analyzer_autopilot_pipeline.py:353`
- Modify: `src/core/help_content.py` (recipe `linear_regression`, the line quoting this label)
- Test: none (pure string change; existing `tests/test_help_hub.py` guards the recipe text)

- [ ] **Step 1: Change the checkbox label**

In `src/autopilot/statistical_analyzer_autopilot_pipeline.py:353`, change:

```python
        self.corr_regression_toggle = QCheckBox("Als Lineare Regression analysieren (Y = a + bX)")
```

to:

```python
        self.corr_regression_toggle = QCheckBox("Analyze as Linear Regression (Y = a + bX)")
```

- [ ] **Step 2: Find and update the recipe quote**

Run: `grep -n "Als Lineare Regression" /Users/philippkrumm/Documents/BioMedStatX/src/core/help_content.py`

This shows the exact line in the `linear_regression` recipe that quotes the German
label (added in commit `578b8e1` during the content audit, specifically to match the
live UI at the time). Update that quoted string to the new English label
`"Analyze as Linear Regression (Y = a + bX)"`, keeping the rest of the sentence
unchanged.

- [ ] **Step 3: Run the recipe test suite to confirm nothing broke**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_help_hub.py -v`
Expected: PASS (all tests, no regression — this file only enforces structural
invariants, not string content, so a wording change inside a recipe cannot fail it).

- [ ] **Step 4: Commit**

```bash
git -C /Users/philippkrumm/Documents/BioMedStatX add src/autopilot/statistical_analyzer_autopilot_pipeline.py src/core/help_content.py
git -C /Users/philippkrumm/Documents/BioMedStatX commit -m "fix(i18n): translate leftover German checkbox label to English

No i18n system exists anywhere in this codebase, so the German string
was an untranslated leftover, not a deliberate localization choice.
Updates the linear_regression Help Hub recipe's quote of this label to
match, since it was written during the content audit specifically to
describe the live (German) UI text.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Binary-outcome detector operator precedence

**Files:**
- Modify: `src/autopilot/statistical_analyzer_autopilot_pipeline.py:1054-1124` (adds a
  new pure function immediately above `_ap_build_analysis_context`, and changes the
  call site inside it)
- Test: `tests/test_binary_outcome_classification.py` (create)

`_ap_build_analysis_context` requires a live `self.df`, populated bucket widgets, and
a valid 1-or-2-column factor selection before it reaches the binary-detection block
(it raises on empty/invalid factor columns earlier in the function) — too much setup
for a focused unit test, and no existing test in this repo drives
`StatisticalAnalyzerApp`'s bucket widgets or calls this method directly (confirmed:
`grep -rn "dv_bucket\|_ap_build_analysis_context" tests/` finds no matches). Rather
than build fragile, first-of-its-kind Qt/bucket test scaffolding for one boolean
expression, extract that expression into a small pure function next to
`_ap_build_analysis_context` and unit test the pure function directly. The call site
inside `_ap_build_analysis_context` is updated to use it, preserving the existing
`is_binary` local variable name so nothing else in that function needs to change.

- [ ] **Step 1: Write the failing test**

Create `tests/test_binary_outcome_classification.py`:

```python
"""Guards the operator-precedence fix in the binary-outcome classifier found in the
Help Hub content audit. The intended semantics (per the original code's own
comment): exactly 2 values that are 0/1 (or two strings), AND the column name does
not hint at a grouping variable. Because `and` binds tighter than `or`, the
un-parenthesized original expression let a numeric 2-value column bypass the
grouping-name guard, and let an all-string column bypass the len==2 guard entirely.
"""
from autopilot.statistical_analyzer_autopilot_pipeline import _classify_binary_outcome


def test_numeric_two_value_grouping_named_column_is_not_binary():
    # Regression case: a numeric 2-value column named like a grouping variable
    # (e.g. "Treatment_Group" coded 1/2) must NOT be classified as binary, even
    # though it has exactly 2 values. Before the fix, the numeric branch of the
    # buggy `or`-split expression bypassed the "not grouping" guard entirely.
    assert _classify_binary_outcome([1, 2], "Treatment_Group") is False


def test_string_column_with_more_than_two_values_is_not_binary():
    # Regression case: an all-string column with more than 2 unique values must
    # NOT be classified as binary. Before the fix, the string branch of the buggy
    # expression had no len==2 gate at all.
    assert _classify_binary_outcome(["Low", "Medium", "High"], "Outcome") is False


def test_numeric_01_column_is_binary():
    assert _classify_binary_outcome([0, 1], "Died") is True


def test_numeric_two_value_non_01_column_is_binary_if_not_grouping_named():
    # A numeric 2-value column that ISN'T 0/1 and ISN'T grouping-named should still
    # be rejected per the stated "0/1 (or two strings)" contract -- only is_01 or
    # is_str values count, not any 2-value numeric column.
    assert _classify_binary_outcome([5, 12], "ScoreCode") is False


def test_yes_no_string_column_is_binary():
    assert _classify_binary_outcome(["Yes", "No"], "Survived") is True


def test_yes_no_grouping_named_column_is_not_binary():
    assert _classify_binary_outcome(["Yes", "No"], "Treatment_Arm") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_binary_outcome_classification.py -v`
Expected: FAIL with `ImportError: cannot import name '_classify_binary_outcome'`
(the function doesn't exist yet).

- [ ] **Step 3: Extract and fix**

In `src/autopilot/statistical_analyzer_autopilot_pipeline.py`, add this function
immediately before `def _ap_build_analysis_context(self):` (currently line 1054):

```python
def _classify_binary_outcome(unique_values, dv_col_name):
    """Whether unique_values (from a single DV column) represent a binary outcome:
    exactly 2 values that are 0/1 (or two strings), and the column name does not
    hint at a grouping variable.
    """
    is_01 = set(unique_values) <= {0, 1, 0.0, 1.0}
    is_str = all(isinstance(v, str) for v in unique_values)
    group_hints = {"group", "arm", "treatment", "condition", "sex",
                   "gender", "cohort", "batch", "grp"}
    name_is_grouping = any(h in dv_col_name.lower() for h in group_hints)
    return (
        len(unique_values) == 2
        and (is_01 or is_str)
        and not name_is_grouping
    )


```

Then, inside `_ap_build_analysis_context`, replace lines 1112-1124 (the comment plus
the buggy computation):

```python
        # Conservative check: exactly 2 values that are 0/1 (or two strings),
        # AND column name does not hint at a grouping variable.
        _is_01 = set(_unique) <= {0, 1, 0.0, 1.0}
        _is_str = all(isinstance(v, str) for v in _unique)
        _group_hints = {"group", "arm", "treatment", "condition", "sex",
                        "gender", "cohort", "batch", "grp"}
        _name_is_grouping = any(h in dv_col.lower() for h in _group_hints)
        is_binary = (
            len(_unique) == 2
            and pd.api.types.is_numeric_dtype(self.df[dv_col]) or _is_str
            and (_is_01 or _is_str)
            and not _name_is_grouping
        )
```

with:

```python
        is_binary = _classify_binary_outcome(_unique, dv_col)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_binary_outcome_classification.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git -C /Users/philippkrumm/Documents/BioMedStatX add src/autopilot/statistical_analyzer_autopilot_pipeline.py tests/test_binary_outcome_classification.py
git -C /Users/philippkrumm/Documents/BioMedStatX commit -m "fix(autopilot): fix operator-precedence bug in binary-outcome classifier

`and` binds tighter than `or` in Python, so the unparenthesized
expression parsed as (len==2 and is_numeric) or (is_str and ...),
letting a numeric 2-value grouping-named column bypass the
not-grouping guard and an all-string column bypass the len==2 gate
entirely. Extracted into a standalone _classify_binary_outcome
function so the fix is directly unit-testable without driving the
full Qt app and bucket widgets (no existing test did either). The
is_numeric_dtype check is dropped as redundant: is_01 and is_str
already establish numeric-vs-string from the actual unique values.
Matches the comment's stated intent and the existing ground-truth
reference helper _ap_is_binary_outcome_for_help. Flagged in the Help
Hub content audit.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Missing subject check for mixed_anova

**Files:**
- Modify: `src/statistical_testing/validators.py:254-260`
- Test: `tests/test_validators_mixed_anova_subject.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_validators_mixed_anova_subject.py`:

```python
"""Guards the fix for the missing subject-column check on the mixed_anova branch
of validate_test_design, found in the Help Hub content audit. Without a subject,
pg.mixed_anova(subject=...) fails with a raw pingouin exception instead of the
app's own clean ModelDesignError.
"""
import pytest

from statistical_testing.validators import ModelDesignError, validate_test_design


def test_mixed_anova_without_subject_raises_model_design_error():
    with pytest.raises(ModelDesignError, match="subject"):
        validate_test_design(
            test_name="mixed_anova",
            between=["Group"],
            within=["Time"],
            subject=None,
        )


def test_mixed_anova_with_subject_does_not_raise():
    validate_test_design(
        test_name="mixed_anova",
        between=["Group"],
        within=["Time"],
        subject="SubjectID",
    )


def test_repeated_measures_anova_subject_check_unchanged():
    # Regression guard: this task must not touch the sibling RM branch.
    with pytest.raises(ModelDesignError, match="subject"):
        validate_test_design(
            test_name="repeated_measures_anova",
            within=["Time"],
            subject=None,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_validators_mixed_anova_subject.py -v`
Expected: `test_mixed_anova_without_subject_raises_model_design_error` FAILS (no
exception is raised today for `mixed_anova` without a subject). The other two should
already PASS.

- [ ] **Step 3: Add the subject check**

In `src/statistical_testing/validators.py`, replace lines 254-256:

```python
    if test_name == "mixed_anova":
        if not between or not within:
            raise ModelDesignError("Mixed ANOVA requires between and within factor.")
```

with:

```python
    if test_name == "mixed_anova":
        if not between or not within:
            raise ModelDesignError("Mixed ANOVA requires between and within factor.")
        if subject is None:
            raise ModelDesignError("Mixed ANOVA requires subject column.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_validators_mixed_anova_subject.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git -C /Users/philippkrumm/Documents/BioMedStatX add src/statistical_testing/validators.py tests/test_validators_mixed_anova_subject.py
git -C /Users/philippkrumm/Documents/BioMedStatX commit -m "fix(validators): require subject column for mixed_anova

Mirrors the sibling repeated_measures_anova branch's existing subject
check. Harmless in the autopilot path (which never routes to
mixed_anova without a Subject ID present), but a non-autopilot caller
would previously hit a raw pingouin exception instead of a clean
ModelDesignError. Flagged in the Help Hub content audit.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Stale CorrelationModel docstring

**Files:**
- Modify: `src/analysis/correlation_models.py:199-203`
- Test: none (docstring-only change)

- [ ] **Step 1: Rewrite the docstring**

In `src/analysis/correlation_models.py`, replace lines 199-203:

```python
class CorrelationModel:
    """Pearson or Spearman correlation with 95 % CI (Fisher z-transform).

    method='auto' applies Shapiro-Wilk to both variables and uses Pearson when
    both are normally distributed (p > alpha), otherwise Spearman.
    Pairwise deletion: only rows without NaN in x_col or y_col are used.
    """
```

with:

```python
class CorrelationModel:
    """Pearson or Spearman correlation with 95 % CI (Fisher z-transform).

    method='auto' picks Pearson or Spearman based on sample size and shape, not
    on the Shapiro-Wilk p-value (Shapiro-Wilk is computed and reported as a
    diagnostic, but the branch never reads it): n < 20 always uses Spearman;
    20 <= n < 100 uses Pearson only if both variables have |skewness| <= 1.0
    and |excess kurtosis| <= 2.0; n >= 100 uses Pearson unless either variable
    has |skewness| > 2.0 or |excess kurtosis| > 4.0.
    Pairwise deletion: only rows without NaN in x_col or y_col are used.
    """
```

- [ ] **Step 2: Verify the file still imports cleanly**

Run: `python -c "from analysis.correlation_models import CorrelationModel; print(CorrelationModel.__doc__)"`
Expected: prints the new docstring, no import error.

- [ ] **Step 3: Commit**

```bash
git -C /Users/philippkrumm/Documents/BioMedStatX add src/analysis/correlation_models.py
git -C /Users/philippkrumm/Documents/BioMedStatX commit -m "docs(correlation): fix CorrelationModel docstring to match real fit() logic

The docstring described Shapiro-Wilk-p-value-driven selection; the
actual fit() logic is skew/excess-kurtosis N-tier gating that never
reads the Shapiro p-value. Found during spec review of the Help Hub
content audit's correlation recipe.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Full verification

- [ ] **Step 1: Run the whole suite**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/ -q`
Expected: PASS, no regressions. New test count: baseline 317 + the new tests added
in Tasks 1, 2, 4, 5 (3 + 3 + 6 + 3 = 15 new tests) = 332 total collected (adjust
expectation if any task's step 2 required a different test count than planned).

- [ ] **Step 2: Lint**

Run: `python -m ruff check src/statistical_testing/advanced_pipeline.py src/analysis/analysis_core.py src/autopilot/statistical_analyzer_autopilot_pipeline.py src/statistical_testing/validators.py src/analysis/correlation_models.py src/core/help_content.py`
Expected: no errors.

- [ ] **Step 3: Manual smoke test**

Run: `python src/analysis/statistical_analyzer.py`
Check: app launches without error. If time permits, run a real Two-Way ANOVA or
Mixed ANOVA analysis with a significant result and confirm the analysis log's
"Post-hoc test:" line names the same method as the pairwise comparison table
below it (this is the exact user-visible symptom Task 1 fixes).

---

## Self-review

- **Spec coverage:** all 6 fixes from the spec have a task (Tasks 1-6); Task 7 covers
  the spec's testing section's "full suite + ruff" requirement.
- **Placeholder scan:** no TBD/TODO; every code block is exact current-state-to-
  target-state, taken from direct file reads during planning, not guessed.
- **Type/API consistency:** `ModelDesignError`, `StatisticalResult`, `AdvancedPostHocEngine`,
  `perform_advanced_test_pipeline`'s parameter names are used identically to their
  actual definitions (verified by reading the source, not assumed) across Tasks 1 and 5.
- **Task 4 revised during planning:** the original draft assumed a `StatisticalAnalyzerApp`
  + bucket-widget test harness (`set_assigned_columns`), but that method doesn't exist
  on `MappingBucketWidget` (confirmed: the real API is `assign_column(name, kind)`,
  `get_assigned_columns()`, `get_assigned_kinds()`) and no existing test drives buckets
  or calls `_ap_build_analysis_context` directly (confirmed via
  `grep -rn "dv_bucket\|_ap_build_analysis_context" tests/`, no matches). Revised to
  extract the buggy expression into a standalone `_classify_binary_outcome` function,
  independently confirmed against the surrounding code (`_ap_build_analysis_context`
  lines 1054-1153) to ensure `is_binary`, `_unique`, and `_series` remain correctly
  wired to the rest of that function after the extraction.
