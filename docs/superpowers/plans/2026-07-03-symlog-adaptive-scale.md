# Sprint 2: Symlog Adaptive Scale Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Sprint 1's hard gate on "Log Y" (disable + drop non-positive points) with a symmetric-log (`symlog`) auto-adaptation that preserves all data points, so datasets with legitimate zero/negative readings (e.g. background-subtracted assays) stay fully visualizable at high dynamic range.

**Architecture:** One new pure-data helper `DataVisualizer._analyze_nonpositive_values(groups, samples)` (single pass: count of ≤0 values + a data-driven `linthresh` from the 5th percentile of non-zero magnitudes, or `None` if no non-zero values exist). One new neutral-styled `DataVisualizer._draw_notice_annotation` alongside the existing red `_draw_warning_annotation`. `_format_axes` routes to `symlog` (with explicit `SymmetricalLogLocator`/`LogFormatterMathtext`) when a usable `linthresh` exists, falling back to Sprint 1's plain-log-plus-red-warning only in the degenerate all-zero case. `PlotAestheticsDialog._apply_log_scale_gating` becomes a tooltip-only "smart toggle" — no more disabling the checkbox.

**Tech Stack:** Python, matplotlib 3.10 (`symlog` scale, `SymmetricalLogLocator`, `LogFormatterMathtext` — API verified directly against the installed version before writing this plan), numpy (`percentile`), PyQt5, pytest.

**Verified before writing this plan** (ad-hoc `python -c` check against the installed matplotlib 3.10.8):
- `SymmetricalLogLocator(base=10, linthresh=...)` and `LogFormatterMathtext(base=10, linthresh=...)` construct and apply without error — no `transform=` object needed.
- `np.percentile([5.0], 5)` returns `5.0` — works fine on a single-element array, no special-casing needed for tiny non-empty samples.
- `np.percentile([], 5)` raises `IndexError` — this is the one real degenerate case, handled by returning `None` from `_analyze_nonpositive_values` when there are zero non-zero values to sample from.

---

## Important: this sprint changes behavior Sprint 1 tests currently assert

Sprint 1 added a test (`tests/test_visualization_warning_annotations.py::test_logscale_with_nonpositive_data_draws_warning`) that asserts mixed positive/non-positive data produces a **red "Data Warning"** annotation and plain `'log'` scale. Sprint 2 changes that exact scenario's behavior to **symlog + neutral "Data Notice"** (no warning, no dropped points) — that's the entire point of this sprint. Task 2 below explicitly updates that test's assertions rather than leaving it to fail as a surprise during the final regression run.

Similarly, `tests/test_plot_aesthetics_log_gating.py::test_logy_checkbox_disabled_when_data_has_nonpositive_values` asserted the checkbox gets disabled — Task 3 updates it to assert the checkbox stays enabled with an updated tooltip instead.

---

### Task 1: Add `_analyze_nonpositive_values` helper and `_draw_notice_annotation`

**Files:**
- Modify: `src/visualization/datavisualizer.py:2508-2519` (insert both new staticmethods between `annotate_box_medians` and `_draw_warning_annotation`)
- Test: `tests/test_visualization_warning_annotations.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_visualization_warning_annotations.py`:

```python
def test_linthresh_uses_5th_percentile_not_min():
    # One artifact-tiny value (0.00001) alongside a real noise band (~1.0-1.3)
    # and real signal (50-200). A min-based threshold would collapse to
    # ~0.000005; the 5th-percentile estimator should sit near the noise band.
    groups = ["A"]
    samples = {"A": [0.00001, 1.0, 1.2, 1.1, 1.3, 1.05, 50.0, 100.0, 200.0]}
    count, thresh = DataVisualizer._analyze_nonpositive_values(groups, samples)
    assert count == 0  # no values <= 0 in this sample
    assert thresh is not None
    assert thresh > 0.01, f"linthresh collapsed toward the single artifact value: {thresh}"


def test_analyze_counts_nonpositive_and_returns_none_thresh_when_all_zero():
    groups = ["A"]
    samples = {"A": [0.0, 0.0, 0.0]}
    count, thresh = DataVisualizer._analyze_nonpositive_values(groups, samples)
    assert count == 3
    assert thresh is None


def test_analyze_handles_single_nonzero_value_without_crash():
    groups = ["A"]
    samples = {"A": [0.0, 0.0, 5.0]}
    count, thresh = DataVisualizer._analyze_nonpositive_values(groups, samples)
    assert count == 2
    assert thresh == 5.0


def test_notice_annotation_uses_neutral_style_distinct_from_warning():
    fig, ax = plt.subplots()
    DataVisualizer._draw_notice_annotation(ax, "Data Notice: test")
    DataVisualizer._draw_warning_annotation(ax, "Data Warning: test")
    notice_text = next(t for t in ax.texts if "Data Notice" in t.get_text())
    warning_text = next(t for t in ax.texts if "Data Warning" in t.get_text())
    assert (
        notice_text.get_bbox_patch().get_facecolor()
        != warning_text.get_bbox_patch().get_facecolor()
    ), "notice and warning annotations must be visually distinct"
    plt.close(fig)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_visualization_warning_annotations.py -k "analyze_nonpositive or linthresh or notice_annotation" -v`
Expected: FAIL — `AttributeError: type object 'DataVisualizer' has no attribute '_analyze_nonpositive_values'`
(and similarly for `_draw_notice_annotation`), since neither exists yet.

- [ ] **Step 3: Add the two staticmethods**

In `src/visualization/datavisualizer.py`, insert between `annotate_box_medians`
(ends line 2508 — wait, confirm: `annotate_box_medians` body ends right before
`_draw_warning_annotation` at line 2519) and the existing `_draw_warning_annotation`:

```python
    @staticmethod
    def _analyze_nonpositive_values(groups, samples):
        """
        Single pass over samples: returns (count_nonpositive, linthresh).
        `count_nonpositive` is how many values are <= 0 or NaN (i.e. would be
        dropped/undefined on a plain log axis). `linthresh` is the 5th
        percentile of |v| over all non-zero values — used as matplotlib's
        symlog linear-region threshold — or None if there are no non-zero
        values to derive a threshold from (e.g. all values are exactly 0),
        in which case callers must fall back to the plain-log-plus-warning
        path since symlog has nothing to anchor on.

        5th percentile (not min) is used deliberately: a single technical
        artifact reading near zero (e.g. one pipetting-error value in a
        background-subtracted assay) would collapse a min-based threshold to
        near-zero, forcing the real noise floor into the log domain and
        distorting it. The percentile is robust to that single-point failure
        mode.
        """
        keys = groups if groups else list((samples or {}).keys())
        count_nonpositive = 0
        abs_vals = []
        for g in keys:
            for v in (samples or {}).get(g, []) or []:
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    continue
                if v != v:  # NaN
                    count_nonpositive += 1
                    continue
                if v <= 0:
                    count_nonpositive += 1
                if v != 0.0:
                    abs_vals.append(abs(v))
        linthresh = float(np.percentile(abs_vals, 5)) if abs_vals else None
        return count_nonpositive, linthresh

    @staticmethod
    def _draw_notice_annotation(ax, text):
        """
        Draw a neutral, low-severity annotation for auto-adapted-but-lossless
        plot behavior (e.g. symlog auto-selected) — visually distinct from
        _draw_warning_annotation's red styling, which is reserved for cases
        where data was actually dropped or degraded.
        """
        ax.text(
            0.5, 1.02, text,
            transform=ax.transAxes,
            ha='center', va='bottom',
            fontsize=8, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.4', fc='#4A5568', ec='none', alpha=0.9),
            zorder=1000, clip_on=False,
        )

```

Insert this block immediately before the existing:
```python
    @staticmethod
    def _draw_warning_annotation(ax, text):
```
(at line 2519 currently — inserting before it will push it down, that's expected
and fine, nothing else references it by line number).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_visualization_warning_annotations.py -k "analyze_nonpositive or linthresh or notice_annotation" -v`
Expected: all 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/visualization/datavisualizer.py tests/test_visualization_warning_annotations.py
git commit -m "feat(viz): add data-driven linthresh helper and neutral notice annotation"
```

---

### Task 2: Route `_format_axes` to symlog for non-positive Y data

**Files:**
- Modify: `src/visualization/datavisualizer.py:1875-1917` (`_format_axes`)
- Modify: `src/visualization/datavisualizer.py:10` (ticker import)
- Modify (update outdated test): `tests/test_visualization_warning_annotations.py::test_logscale_with_nonpositive_data_draws_warning`
- Test: `tests/test_visualization_warning_annotations.py` (append new tests)

- [ ] **Step 1: Add the new ticker imports**

In `src/visualization/datavisualizer.py`, line 10:

Current code being replaced:
```python
from matplotlib.ticker import ScalarFormatter, FuncFormatter
```

New code:
```python
from matplotlib.ticker import ScalarFormatter, FuncFormatter, SymmetricalLogLocator, LogFormatterMathtext
```

- [ ] **Step 2: Update the now-outdated Sprint 1 test**

In `tests/test_visualization_warning_annotations.py`, replace the existing test
(this scenario's expected behavior changes from "plain log + red warning" to
"symlog + neutral notice" — that's the point of this sprint):

Current code being replaced:
```python
def test_logscale_with_nonpositive_data_draws_warning():
    fig, ax = plt.subplots()
    groups = ["A", "B"]
    # Group A has two non-positive values (-0.5, 0.0); group B is all positive.
    samples = {"A": [1.0, 2.0, -0.5, 0.0], "B": [3.0, 4.0]}

    DataVisualizer.plot_bar(
        groups, samples, ax=ax, save_plot=False, logy=True, show_error_bars=False
    )

    warning_texts = [t.get_text() for t in ax.texts if "Data Warning" in t.get_text()]
    assert len(warning_texts) == 1, (
        "log-scale axis with non-positive data must draw an on-canvas warning"
    )
    assert "2 values" in warning_texts[0]
    plt.close(fig)
```

New code:
```python
def test_logscale_with_nonpositive_data_uses_symlog_not_plain_log():
    # Sprint 2: this scenario now auto-adapts to symlog (lossless) instead of
    # dropping points under a plain log scale with a red warning (Sprint 1
    # behavior) — group A has a real noise/signal band to derive linthresh from.
    fig, ax = plt.subplots()
    groups = ["A", "B"]
    samples = {"A": [1.0, 2.0, -0.5, 0.0], "B": [3.0, 4.0]}

    DataVisualizer.plot_bar(
        groups, samples, ax=ax, save_plot=False, logy=True, show_error_bars=False
    )

    assert ax.get_yscale() == "symlog"
    notice_texts = [t.get_text() for t in ax.texts if "Data Notice" in t.get_text()]
    assert len(notice_texts) == 1
    assert "symlog" in notice_texts[0]
    warning_texts = [t.get_text() for t in ax.texts if "Data Warning" in t.get_text()]
    assert len(warning_texts) == 0, (
        "lossless symlog path must not show the red data-loss warning"
    )
    plt.close(fig)


def test_logscale_with_all_zero_data_falls_back_to_plain_log_with_warning():
    # Degenerate case: no non-zero magnitude anywhere means there's nothing to
    # derive a linthresh from — must fall back to Sprint 1's honest warning
    # rather than inventing an arbitrary threshold.
    fig, ax = plt.subplots()
    groups = ["A"]
    samples = {"A": [0.0, 0.0, 0.0]}

    DataVisualizer.plot_bar(
        groups, samples, ax=ax, save_plot=False, logy=True, show_error_bars=False
    )

    assert ax.get_yscale() == "log"
    warning_texts = [t.get_text() for t in ax.texts if "Data Warning" in t.get_text()]
    assert len(warning_texts) == 1, (
        "degenerate all-zero data has no usable linthresh — must fall back safely"
    )
    plt.close(fig)
```

- [ ] **Step 3: Run tests to verify they fail with the predicted symptom**

Run: `pytest tests/test_visualization_warning_annotations.py -k "symlog or all_zero_data" -v`
Expected: FAIL — `test_logscale_with_nonpositive_data_uses_symlog_not_plain_log` fails
on `assert ax.get_yscale() == "symlog"` (today it's still `"log"`);
`test_logscale_with_all_zero_data_falls_back_to_plain_log_with_warning` fails
because today's code doesn't distinguish the all-zero case at all — it
happens to already produce a warning today by coincidence of the old logic,
so double check this one specifically shows the *old* undifferentiated
`omitted` counting, not the new linthresh-aware branch, before proceeding.

- [ ] **Step 4: Replace the log-scale block in `_format_axes`**

In `src/visualization/datavisualizer.py`, replace lines 1898-1917:

Current code being replaced:
```python
        omitted = 0
        if (logx or logy) and samples:
            keys = groups if groups else list(samples.keys())
            for g in keys:
                for v in samples.get(g, []) or []:
                    try:
                        v = float(v)
                    except (TypeError, ValueError):
                        continue
                    if v != v or v <= 0:  # v != v excludes NaN
                        omitted += 1

        if logx:
            ax.set_xscale('log', base=10)
        if logy:
            ax.set_yscale('log', base=10)

        if omitted > 0:
            DataVisualizer._draw_warning_annotation(
                ax, f"Data Warning: {omitted} values ≤ 0 omitted from log-scale axis.")
```

New code:
```python
        if logx:
            ax.set_xscale('log', base=10)
            if samples:
                count_x, _ = DataVisualizer._analyze_nonpositive_values(groups, samples)
                if count_x > 0:
                    DataVisualizer._draw_warning_annotation(
                        ax, f"Data Warning: {count_x} values ≤ 0 omitted from log-scale axis.")

        if logy:
            count_y, linthresh = (
                DataVisualizer._analyze_nonpositive_values(groups, samples) if samples else (0, None)
            )
            if count_y > 0 and linthresh is not None:
                # Lossless path: symlog preserves near-zero/negative readings
                # (e.g. background-subtracted assay data) instead of dropping them.
                ax.set_yscale('symlog', linthresh=linthresh)
                ax.yaxis.set_major_locator(SymmetricalLogLocator(base=10, linthresh=linthresh))
                ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10, linthresh=linthresh))
                DataVisualizer._draw_notice_annotation(
                    ax, f"Data Notice: Values ≤ 0 detected. Auto-applied symlog scale "
                    f"(linthresh = {linthresh:.4g}).")
            elif count_y > 0:
                # No non-zero magnitude anywhere (e.g. all values are exactly 0)
                # — nothing to derive a threshold from; fall back to the plain
                # log scale and report the omission honestly.
                ax.set_yscale('log', base=10)
                DataVisualizer._draw_warning_annotation(
                    ax, f"Data Warning: {count_y} values ≤ 0 omitted from log-scale axis.")
            else:
                ax.set_yscale('log', base=10)
```

Note: `logx`'s path deliberately keeps the old plain-log-plus-warning behavior
unchanged (Log X is out of scope for symlog per the spec — it's always a
categorical axis for every plot type reachable through the dialog, so this
branch is effectively dead in practice today; it's preserved as-is rather than
deleted, matching Sprint 1's scope decision).

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_visualization_warning_annotations.py -v`
Expected: all tests in the file PASS (Sprint 1's `test_grouped_emm_failure_draws_visible_warning`
and `test_logscale_with_all_positive_data_draws_no_warning` are untouched and
still pass; the updated/new tests from this task pass; Task 1's 4 new tests
still pass).

- [ ] **Step 6: Commit**

```bash
git add src/visualization/datavisualizer.py tests/test_visualization_warning_annotations.py
git commit -m "feat(viz): auto-apply symlog scale for log-axis data with values <= 0"
```

---

### Task 3: UI smart toggle — stop disabling "Log Y", update tooltip instead

**Files:**
- Modify: `src/ui/dialogs/plot_aesthetics_dialog.py:1570-1596` (`_apply_log_scale_gating`)
- Modify (update outdated test): `tests/test_plot_aesthetics_log_gating.py`

- [ ] **Step 1: Update the now-outdated Sprint 1 test**

In `tests/test_plot_aesthetics_log_gating.py`, replace:

Current code being replaced:
```python
def test_logy_checkbox_disabled_when_data_has_nonpositive_values():
    groups = ["A", "B"]
    samples = {"A": [1.0, 2.0, -0.5], "B": [3.0, 4.0, 5.0]}
    dialog = PlotAestheticsDialog(groups=groups, samples=samples, show_export_controls=False)
    try:
        assert dialog.style_tab.logy_check.isEnabled() is False
        assert dialog.style_tab.logy_check.isChecked() is False
        assert "≤ 0" in dialog.style_tab.logy_check.toolTip()
    finally:
        dialog.close()
```

New code:
```python
def test_logy_checkbox_stays_enabled_with_symlog_tooltip_when_data_has_nonpositive_values():
    # Sprint 2: the checkbox is no longer hard-disabled — _format_axes now
    # auto-adapts to symlog for this data instead of dropping points, so the
    # user keeps the ability to toggle log scaling on. Only the tooltip changes.
    groups = ["A", "B"]
    samples = {"A": [1.0, 2.0, -0.5], "B": [3.0, 4.0, 5.0]}
    dialog = PlotAestheticsDialog(groups=groups, samples=samples, show_export_controls=False)
    try:
        assert dialog.style_tab.logy_check.isEnabled() is True
        assert "symlog" in dialog.style_tab.logy_check.toolTip().lower()
    finally:
        dialog.close()
```

- [ ] **Step 2: Run test to verify it fails with the predicted symptom**

Run: `pytest tests/test_plot_aesthetics_log_gating.py::test_logy_checkbox_stays_enabled_with_symlog_tooltip_when_data_has_nonpositive_values -v`
Expected: FAIL — `assert dialog.style_tab.logy_check.isEnabled() is True` fails
(today it's still `False`, Sprint 1's hard gate).

- [ ] **Step 3: Replace `_apply_log_scale_gating`**

In `src/ui/dialogs/plot_aesthetics_dialog.py`, replace lines 1570-1596:

Current code being replaced:
```python
    def _apply_log_scale_gating(self):
        """
        Disable "Log Y" (on self.style_tab) when self.samples contains
        non-positive values. log(<=0) is undefined; matplotlib silently drops
        those points on a log-scale axis with no visible warning, so gate it
        here instead. StyleTab itself only receives `config`, not `samples`,
        so this reaches into the already-built tab's checkbox directly.
        """
        has_nonpositive = False
        for values in self.samples.values():
            for v in (values or []):
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    continue
                if v == v and v <= 0:  # v == v excludes NaN
                    has_nonpositive = True
                    break
            if has_nonpositive:
                break

        if has_nonpositive:
            self.style_tab.logy_check.setChecked(False)
            self.style_tab.logy_check.setEnabled(False)
            self.style_tab.logy_check.setToolTip("Log scale unavailable: data contains values ≤ 0.")
        else:
            self.style_tab.logy_check.setEnabled(True)
            self.style_tab.logy_check.setToolTip("")
```

New code:
```python
    def _apply_log_scale_gating(self):
        """
        Update the "Log Y" tooltip (on self.style_tab) when self.samples
        contains non-positive values. Sprint 1 hard-disabled the checkbox
        here; Sprint 2 replaces that with a "smart toggle" — _format_axes
        (datavisualizer.py) auto-switches to a symlog scale for this data
        instead of dropping points, so the checkbox stays usable and only the
        tooltip changes. StyleTab itself only receives `config`, not
        `samples`, so this reaches into the already-built tab's checkbox
        directly.
        """
        has_nonpositive = False
        for values in self.samples.values():
            for v in (values or []):
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    continue
                if v == v and v <= 0:  # v == v excludes NaN
                    has_nonpositive = True
                    break
            if has_nonpositive:
                break

        self.style_tab.logy_check.setEnabled(True)
        if has_nonpositive:
            self.style_tab.logy_check.setToolTip(
                "Values ≤ 0 detected. Symmetric log scale (symlog) will be used "
                "automatically for lossless display.")
        else:
            self.style_tab.logy_check.setToolTip("")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_plot_aesthetics_log_gating.py -v`
Expected: both tests PASS (`test_logy_checkbox_stays_enabled_with_symlog_tooltip_when_data_has_nonpositive_values`
and the unchanged `test_logy_checkbox_enabled_when_data_all_positive`).

- [ ] **Step 5: Commit**

```bash
git add src/ui/dialogs/plot_aesthetics_dialog.py tests/test_plot_aesthetics_log_gating.py
git commit -m "feat(ui): smart-toggle Log Y tooltip instead of hard-disabling for symlog data"
```

---

### Task 4: Full regression check

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/ -v`
Expected: baseline going into this sprint was 345 passed / 4 skipped / 0 failed.
Test count changes: Task 1 adds 4 new tests (+4). Task 2 replaces 1 existing
test 1-for-1 (net 0) and adds 1 new all-zero-case test (+1). Task 3 replaces 1
existing test 1-for-1 (net 0). Total: +5 → **350 passed / 4 skipped / 0
failed**, no regressions elsewhere.

- [ ] **Step 2: If anything regressed, fix before proceeding**

In particular, re-check `tests/test_plot_type_dispatch.py`,
`tests/test_rm_emm_plot_render.py`, and `tests/test_decision_tree_graphics.py`
(the same regression set Sprint 1 checked) — the `_format_axes` signature is
unchanged from Sprint 1 (still `groups=None, samples=None` as optional
keywords), so no caller outside the 3 already-threaded call sites should be
affected, but verify rather than assume.

Run: `pytest tests/test_plot_type_dispatch.py tests/test_rm_emm_plot_render.py tests/test_decision_tree_graphics.py -v`
Expected: all PASS.

---

## Self-review notes

- **Spec coverage:** linthresh (5th percentile, degenerate-case fallback) →
  Task 1. `_format_axes` symlog routing + explicit locator/formatter → Task 2.
  Neutral notice annotation, distinct from warning → Task 1 + used in Task 2.
  UI smart-toggle (no more hard-disable) → Task 3. Log X out of scope →
  explicitly preserved as Sprint-1 behavior in Task 2, not touched.
- **Outdated-test handling:** both Sprint 1 tests whose asserted behavior this
  sprint deliberately changes are explicitly updated in Task 2 and Task 3,
  not left to surface as surprise failures in Task 4's regression run.
- **Type/signature consistency:** `_analyze_nonpositive_values(groups, samples)`
  returns `(count: int, linthresh: float | None)` — used identically in Task 2's
  `_format_axes` changes for both the `logx` and `logy` branches. `_draw_notice_annotation(ax, text)`
  signature matches the existing `_draw_warning_annotation(ax, text)` for
  consistency.
- **No placeholders:** every step has literal code, exact line numbers
  (re-verified against the actual post-Sprint-1 file state immediately before
  writing this plan), and exact pytest commands with expected output.
- **API verified, not assumed:** `SymmetricalLogLocator`/`LogFormatterMathtext`
  construction and `np.percentile` edge-case behavior were checked against the
  actual installed matplotlib/numpy before this plan was written (see header).
