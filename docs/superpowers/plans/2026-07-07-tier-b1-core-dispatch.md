# Tier B1: Core Dispatch Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three independent, fully-diagnosed bugs in `src/analysis/analysis_core.py` and
`src/analysis/clinical_models.py`: a crash-message bug that replaces a real diagnostic with a
Python `TypeError` string (SC1), a writer/reader key mismatch that silently blanks ANCOVA's
t-statistic in the HTML report (SC3), and a `NameError` crash path in the non-parametric post-hoc
branch (SC4).

**Architecture:** Three small, independent mechanical fixes in the same two files. No design
decisions — each fix is fully specified by the round-2 audit
(`docs/superpowers/audit-notes/release-2.0-audit/02-statistical-core-dispatch.md`, findings
SC1/SC3/SC4) and confirmed against current source during planning.

**Tech Stack:** Python, pytest.

---

### Task 1: SC1 — drop the nonexistent `test_name=` kwarg from `make_blocked_result()` calls

**Files:**
- Modify: `src/analysis/analysis_core.py:914,921,960,1005`
- Test: `tests/test_analysis_core_blocked_result_kwarg.py`

`StatisticalTester.make_blocked_result` is defined as
`def make_blocked_result(reason, *, code, details=None, warnings=None)` — there is no
`test_name` parameter. All 4 call sites below pass `test_name=...`, which raises `TypeError`,
caught by the outer `except Exception as e:` (`analysis_core.py:1564`) and converted into a
**different** blocked result (`code="UNHANDLED_EXCEPTION"`) whose `block_reason` is the Python
error message itself — the intended diagnostic message is lost.

- [ ] **Step 1: Write the failing test**

```python
"""make_blocked_result() has signature (reason, *, code, details=None, warnings=None) - no
test_name parameter. analysis_core.py's Mixed ANOVA invalid-design check passes test_name=
anyway, so the TypeError this raises gets caught by the outer except-Exception in
AnalysisManager._analyze_single_dataset and replaces the intended "Mixed ANOVA requires two
factors" message with a confusing Python signature error instead.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import pytest

from analysis.analysis_core import AnalysisManager


@pytest.fixture
def dummy_file(tmp_path):
    path = tmp_path / "dummy.xlsx"
    pd.DataFrame({"a": [1]}).to_excel(path, index=False)
    return str(path)


def test_mixed_anova_invalid_design_reports_the_real_message_not_a_typeerror(dummy_file, tmp_path):
    df = pd.DataFrame({
        "Group": ["ctrl", "ctrl", "a", "a"],
        "Time": ["T1", "T2", "T1", "T2"],
        "Value": [1.0, 2.0, 3.0, 4.0],
    })
    ctx = {
        "injected_df": df,
        "factor_columns": ["Group"],
        "between_factors": ["Group"],
        "dv_columns": ["Value"],
        "group_labels": ["ctrl", "a"],
        "mode": "single",
    }

    result = AnalysisManager.analyze(
        file_path=dummy_file,
        group_col="Group",
        groups=["ctrl", "a"],
        value_cols=["Value"],
        save_plot=False,
        skip_plots=True,
        file_name=str(tmp_path / "out"),
        analysis_context=ctx,
        test="mixed_anova",
        additional_factors=["Time"],  # only 1 factor: triggers the "requires two factors" block
    )

    assert result.get("blocked") is True
    assert result.get("block_code") == "INVALID_DESIGN", (
        f"expected INVALID_DESIGN, got {result.get('block_code')!r} "
        f"(reason={result.get('block_reason')!r}) - the TypeError from the bad "
        f"test_name= kwarg is likely being caught and relabeled UNHANDLED_EXCEPTION"
    )
    assert result.get("block_reason") == "Mixed ANOVA requires two factors (between and within)"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/test_analysis_core_blocked_result_kwarg.py -v`
Expected: FAIL — `result.get("block_code")` is `"UNHANDLED_EXCEPTION"`, not `"INVALID_DESIGN"`,
because the `test_name=` `TypeError` was caught by the outer handler.

- [ ] **Step 3: Fix all 4 call sites**

Read the current file first (line numbers may have drifted) via
`grep -n 'make_blocked_result(code="INVALID_DESIGN"\|make_blocked_result("PREP_ERROR"' src/analysis/analysis_core.py`.
For each of the 4 matches, remove the trailing `, test_name=...` (or `test_name=...,` if it's not
the last argument) from the call. The 4 exact current lines to change:

```python
                    return StatisticalTester.make_blocked_result(code="INVALID_DESIGN", reason="Mixed ANOVA requires two factors (between and within)", test_name="mixed_anova")
```
becomes:
```python
                    return StatisticalTester.make_blocked_result(code="INVALID_DESIGN", reason="Mixed ANOVA requires two factors (between and within)")
```

and all 3 occurrences of:
```python
                    return StatisticalTester.make_blocked_result("PREP_ERROR", prep["error"], test_name=kwargs.get('test', 'unknown_test'))
```
become:
```python
                    return StatisticalTester.make_blocked_result("PREP_ERROR", prep["error"])
```

(The 3 `"PREP_ERROR"` occurrences are for `mixed_anova`, `two_way_anova`, and
`repeated_measures_anova` respectively — same fix, 3 separate call sites.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/test_analysis_core_blocked_result_kwarg.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_analysis_core_blocked_result_kwarg.py src/analysis/analysis_core.py
git commit -m "fix(core): drop nonexistent test_name kwarg from make_blocked_result calls"
```

---

### Task 2: SC4 — initialize `posthoc_choice` to avoid a `NameError` crash

**Files:**
- Modify: `src/analysis/analysis_core.py:1085` (and surrounding block through line ~1196)
- Test: `tests/test_analysis_core_posthoc_choice_scoping.py`

`posthoc_choice` is only ever assigned inside the parametric-dialog `else:` branch (currently
line 1116, `posthoc_choice = UIDialogManager.select_posthoc_test_dialog(...)`), but it is read
at line 1196 (`if posthoc_choice == "dunnett" and ...`) which sits at the same indentation as
BOTH the `if 'kruskal' in test_name or ...` branch (non-parametric, never assigns
`posthoc_choice`) and the parametric `else:` branch. Because Python treats a name assigned
anywhere in a function as local to the whole function, reading it before the non-parametric
branch assigns anything raises `NameError: cannot access local variable 'posthoc_choice' where
it is not associated with a value`.

- [ ] **Step 1: Write the failing test**

```python
"""analysis_core.py's post-hoc block only assigns posthoc_choice inside the parametric
dialog's else-branch (~line 1116), but reads it afterwards (~line 1196) regardless of which
branch ran. When the significant result comes from a non-parametric test (Kruskal-Wallis /
Friedman), posthoc_choice is never assigned, so the read at line 1196 raises NameError -
caught by the outer except-Exception and surfaced as a confusing UNHANDLED_EXCEPTION block
instead of the intended silent no-op (Dunnett's control-group key never applies to the
non-parametric branch in the first place).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import pytest

from analysis.analysis_core import AnalysisManager


@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch):
    from PyQt5.QtWidgets import QDialog
    monkeypatch.setattr(QDialog, "exec_", lambda self, *a, **k: 0, raising=False)
    monkeypatch.setattr(QDialog, "exec", lambda self, *a, **k: 0, raising=False)


@pytest.fixture
def dummy_file(tmp_path):
    path = tmp_path / "dummy.xlsx"
    pd.DataFrame({"a": [1]}).to_excel(path, index=False)
    return str(path)


def test_kruskal_significant_result_does_not_crash_with_nameerror(dummy_file, tmp_path, monkeypatch):
    # Force the non-parametric post-hoc re-entry branch (analysis_core.py's own
    # perform_refactored_posthoc_testing call, NOT the earlier dialog-driven one)
    # to actually run by making test_results.get('pairwise_comparisons') empty
    # and test_recommendation == 'non_parametric'.
    from analysis.analysis_core import StatisticalTester

    def _fake_perform_refactored_posthoc_testing(*args, **kwargs):
        return {"posthoc_test": "Dunn (Holm-Sidak)", "pairwise_comparisons": [
            {"group1": "a", "group2": "b", "p_value": 0.01, "significant": True}
        ], "error": None}

    monkeypatch.setattr(
        StatisticalTester, "perform_refactored_posthoc_testing",
        staticmethod(_fake_perform_refactored_posthoc_testing), raising=False
    )

    df = pd.DataFrame({
        "Group": ["a"] * 5 + ["b"] * 5 + ["c"] * 5,
        "Value": [1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24],
    })
    ctx = {
        "injected_df": df,
        "factor_columns": ["Group"],
        "between_factors": ["Group"],
        "dv_columns": ["Value"],
        "group_labels": ["a", "b", "c"],
        "mode": "single",
    }

    result = AnalysisManager.analyze(
        file_path=dummy_file,
        group_col="Group",
        groups=["a", "b", "c"],
        value_cols=["Value"],
        save_plot=False,
        skip_plots=True,
        file_name=str(tmp_path / "out"),
        analysis_context=ctx,
        test="kruskal_wallis",
    )

    assert result.get("block_code") != "UNHANDLED_EXCEPTION", (
        f"got block_reason={result.get('block_reason')!r} - looks like the "
        f"posthoc_choice NameError fired"
    )
    assert "posthoc_choice" not in str(result.get("block_reason", ""))
```

**Note on this test's robustness:** the exact test name (`"kruskal_wallis"`) and whether this
specific dispatch path reaches the vulnerable code depends on `analysis_core.py`'s test-name
dispatch logic upstream of line 1085 — if this test doesn't reach the vulnerable branch as
written (check by temporarily adding a `print` or running with `-s` and confirming the mock's
`_fake_perform_refactored_posthoc_testing` was actually called), adjust the `test=` value or
`test_recommendation`-driving inputs until it does, following the branch condition literally
at `analysis_core.py:1104`: `if 'kruskal' in test_name or 'friedman' in test_name or
test_recommendation == 'non_parametric':`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/test_analysis_core_posthoc_choice_scoping.py -v`
Expected: FAIL with `block_code == "UNHANDLED_EXCEPTION"` and `"posthoc_choice"` present in the
block reason (the literal `NameError` message).

- [ ] **Step 3: Fix — initialize `posthoc_choice` at the top of the block**

Read `analysis_core.py` around line 1085 fresh (`grep -n "posthoc_results = None" src/analysis/analysis_core.py`)
to confirm the exact current line, then change:

```python
            posthoc_results = None
```
to:
```python
            posthoc_results = None
            posthoc_choice = None
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/test_analysis_core_posthoc_choice_scoping.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_analysis_core_posthoc_choice_scoping.py src/analysis/analysis_core.py
git commit -m "fix(core): initialize posthoc_choice to prevent NameError in non-parametric branch"
```

---

### Task 3: SC3 — align ANCOVA's EMM contrast keys with LMM's and the export layer's contract

**Files:**
- Modify: `src/analysis/clinical_models.py:344-353` (inside `ANCOVAModel.emm_contrasts`)
- Test: `tests/test_ancova_emm_contrast_keys.py`

`ANCOVAModel.emm_contrasts` (clinical_models.py:286-354) builds each comparison dict with keys
`group1, group2, estimate, se, t, df, p_value, significant`. `LinearMixedModel.emm_contrasts`
(clinical_models.py:898-1009) builds `group1, group2, estimate, std_err, statistic, p_value,
significant, test, df, corrected, correction`. The HTML export layer
(`src/export/report_stat_rows.py:744`, confirmed via `git grep -n '"statistic"' src/export/report_stat_rows.py`)
reads `comp.get("statistic")` — this resolves for LMM but returns `None`/blank for ANCOVA,
silently blanking the pairwise-comparison table's statistic column for every ANCOVA report.

- [ ] **Step 1: Write the failing test**

```python
"""ANCOVAModel.emm_contrasts() writes "t"/"se" while LinearMixedModel.emm_contrasts() writes
"statistic"/"std_err" for the exact same kind of comparison dict - the HTML export layer
(report_stat_rows.py) reads "statistic", so ANCOVA's t-column silently renders blank. Fix:
ANCOVA's dict shape must match LMM's.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from analysis.clinical_models import ANCOVAModel


def test_ancova_emm_contrasts_use_the_same_keys_as_lmm():
    rng = np.random.RandomState(0)
    n = 60
    df = pd.DataFrame({
        "Group": np.repeat(["ctrl", "a", "b"], n // 3),
        "Cov": rng.randn(n),
    })
    df["Value"] = (
        df["Cov"] * 1.5
        + df["Group"].map({"ctrl": 0.0, "a": 2.0, "b": 4.0})
        + rng.randn(n) * 0.5
    )

    model = ANCOVAModel()
    model.fit(df, dv="Value", between_factors=["Group"], covariates=["Cov"])
    contrasts = model.emm_contrasts(method="pairwise")

    assert contrasts, "expected at least one pairwise contrast"
    for c in contrasts:
        assert "statistic" in c, f"missing 'statistic' key (found: {sorted(c.keys())})"
        assert "std_err" in c, f"missing 'std_err' key (found: {sorted(c.keys())})"
        assert "t" not in c, "stale 't' key should have been renamed to 'statistic'"
        assert "se" not in c, "stale 'se' key should have been renamed to 'std_err'"
        assert c.get("test") == "ANCOVA EMM Contrast"
        assert c.get("corrected") is True
        assert c.get("correction") == "Holm-Bonferroni"


def test_ancova_emm_contrasts_vs_control_correction_label():
    rng = np.random.RandomState(1)
    n = 60
    df = pd.DataFrame({
        "Group": np.repeat(["ctrl", "a", "b"], n // 3),
        "Cov": rng.randn(n),
    })
    df["Value"] = (
        df["Cov"] * 1.5
        + df["Group"].map({"ctrl": 0.0, "a": 2.0, "b": 4.0})
        + rng.randn(n) * 0.5
    )

    model = ANCOVAModel()
    model.fit(df, dv="Value", between_factors=["Group"], covariates=["Cov"])
    contrasts = model.emm_contrasts(method="vs_control", control_group="ctrl")

    assert contrasts
    assert all(c.get("correction") == "multivariate-t" for c in contrasts)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/test_ancova_emm_contrast_keys.py -v`
Expected: FAIL on `assert "statistic" in c` (`KeyError`-style assertion failure — the dict has
`"t"` instead).

- [ ] **Step 3: Fix — rename keys and add the missing fields**

Read `clinical_models.py` around line 342 fresh
(`grep -n "contrasts.append" src/analysis/clinical_models.py` — the ANCOVA one is the first
match) to confirm current line numbers, then change:

```python
        contrasts = []
        for (a, b), e, s, tv, p in zip(pairs, est, se, t_values, p_adj):
            contrasts.append({
                "group1": str(a),
                "group2": str(b),
                "estimate": float(e),
                "se": float(s),
                "t": float(tv),
                "df": ddf,
                "p_value": float(p),
                "significant": bool(p < self._alpha),
            })
        return contrasts
```

to:

```python
        contrasts = []
        for (a, b), e, s, tv, p in zip(pairs, est, se, t_values, p_adj):
            contrasts.append({
                "group1": str(a),
                "group2": str(b),
                "estimate": float(e),
                "std_err": float(s),
                "statistic": float(tv),
                "df": ddf,
                "p_value": float(p),
                "significant": bool(p < self._alpha),
                "test": "ANCOVA EMM Contrast",
                "corrected": True,
                "correction": "multivariate-t" if method == "vs_control" else "Holm-Bonferroni",
            })
        return contrasts
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/test_ancova_emm_contrast_keys.py -v`
Expected: PASS.

- [ ] **Step 5: Check for any other reader of the old `"t"`/`"se"` keys before committing**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && git grep -n '\.get("t")\|\["t"\]\|\.get("se")\|\["se"\]' -- src/export/ src/analysis/ src/statistical_testing/`
Expected: no hits reading an EMM-contrast dict's `"t"`/`"se"` keys specifically (both are common
short names, so eyeball each hit — none should be from a `pairwise_comparisons`/`emm_contrasts`
consumer). If a real reader of the old keys is found, update it to read `"statistic"`/`"std_err"`
too, in this same task.

- [ ] **Step 6: Run the full test suite**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/ -q --tb=no`
Expected: same pass count as before this task plus the 2 new tests (1 pre-existing unrelated
failure, `tests/test_convergence.py::test_convergence_keys`, is expected and untouched by this
work — see `docs/superpowers/plans/2026-07-07-mixed-anova-sphericity-fix.md`'s Task 4 for how
this was confirmed pre-existing).

- [ ] **Step 7: Commit**

```bash
git add tests/test_ancova_emm_contrast_keys.py src/analysis/clinical_models.py
git commit -m "fix(clinical-models): align ANCOVA EMM contrast keys with LMM's schema"
```

---

## Self-review notes

- **Spec coverage:** SC1 (Task 1), SC4 (Task 2), SC3 (Task 3) — all 3 findings assigned to this
  package in `docs/superpowers/specs/2026-07-07-audit-fix-clustering-design.md` are covered.
- **Test pattern precedent:** Tasks 1 and 2 follow the exact `AnalysisManager.analyze(file_path=,
  ..., analysis_context={"injected_df": ...})` fixture pattern already established in
  `tests/test_ancova_control_group_wiring.py`, confirmed by reading that file during planning —
  not invented from scratch.
- **SC3's key rename is checked for other readers** (Task 3, Step 5) before committing, since a
  rename is exactly the kind of change that can silently break an unrelated caller.
