# Sprint 1: Sphericity Report Fix + Visualization Error Transparency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the sphericity-correction epsilon key mismatch in HTML report export, and make two silent plot-degradation paths (grouped-EMM fallback, log-scale non-positive data loss) visibly self-documenting on the exported figure instead of only logging to a backend log file.

**Architecture:** One mechanical key-path fix in `report_summaries.py` (no new abstractions). One new shared static helper `DataVisualizer._draw_warning_annotation(ax, text)` in `datavisualizer.py`, reused at two call sites (grouped-EMM except-block, log-scale point-omission check inside `_format_axes`). One new gating method on `PlotAestheticsDialog` that inspects the `samples` dict already passed into its constructor to disable "Log Y" pre-emptively.

**Tech Stack:** Python, matplotlib (Agg backend in tests), PyQt5, pytest. No new dependencies.

---

## Scope note found while tracing (resolves spec's open question 1)

`PlotAestheticsDialog.__init__(self, groups=None, samples=None, ...)` (`src/ui/dialogs/plot_aesthetics_dialog.py:1512`) already receives the actual `samples` dict (group name → list of values) at construction time — confirmed both from its signature and from its real caller
(`statistical_analyzer_autopilot_pipeline.py:1851`, which passes `samples=self.samples or {}`). No new data plumbing is needed; the gating method reads `self.samples` directly.

**Scope correction:** the spec asked about gating both "Log X" and "Log Y". Tracing `plot_bar`/`plot_box`/`plot_violin` (the only plot types reachable through this dialog) shows the X axis is always the categorical `Group` column (`sns.barplot(x='Group', y='Value', ...)` at `datavisualizer.py:955`) — there is no numeric x-data in `samples` to test for non-positive values. "Log X" on a categorical axis isn't the same hazard the audit flagged (that was specifically about numeric measurement data on a log-transformed axis silently dropping points). This plan scopes the gating fix to **Log Y only** (Task 4). "Log X" is left as-is — flagging this explicitly rather than silently narrowing scope.

## Resolves spec's open question 2

Adding one shared `DataVisualizer._draw_warning_annotation(ax, text)` helper (Task 2), reused by both the grouped-EMM fallback (Task 2) and the log-scale omission warning (Task 3), per the spec's recommendation — same visual language for both warning types.

---

### Task 1: Fix sphericity correction/epsilon key mismatch in report export

**Files:**
- Modify: `src/export/report_summaries.py:450-464`
- Test: `tests/test_report_assumption_summary.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_report_assumption_summary.py`:

```python
"""Sphericity-correction note in the HTML assumption summary must read the
epsilon value from where statisticaltester.py actually writes it
(results["correction_used"] top-level, results["sphericity_corrections"][...]
["epsilon"] nested) — not from results["sphericity_test"], which never
contains a correction/epsilon key.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from export.report_summaries import _SummariesMixin


def _rm_anova_results_with_sphericity_violation():
    """Shaped exactly like StatisticalTester._perform_comprehensive_sphericity_test
    + _apply_sphericity_corrections actually produce it (statisticaltester.py:2624-2874).
    """
    return {
        "model_type": "RMANOVA",
        "sphericity_test": {
            "test_name": "Mauchly's Test for Sphericity",
            "W": 0.72,
            "p_value": 0.01,
            "sphericity_assumed": False,
            "d": 2,
            "interpretation": "Sphericity violated",
        },
        "sphericity_corrections": {
            "needed": True,
            "greenhouse_geisser": {
                "epsilon": 0.6543,
                "corrected_df1": 1.31,
                "corrected_df2": 13.1,
                "p_value": 0.02,
                "conservative": True,
                "description": "Conservative correction for sphericity violation",
            },
        },
        "corrected_p_value": 0.02,
        "correction_used": "Greenhouse-Geisser (ε = 0.654)",
        "final_p_value": 0.02,
    }


def test_sphericity_note_includes_epsilon_from_real_backend_shape():
    result = _SummariesMixin._build_assumption_summary(
        _rm_anova_results_with_sphericity_violation()
    )
    note = result["sphericity_correction_note"]
    assert note is not None
    assert "Greenhouse-Geisser" in note
    assert "0.6543" in note, (
        f"epsilon missing from note (key-path bug not fixed): {note!r}"
    )
```

- [ ] **Step 2: Run test to verify it fails with the predicted symptom**

Run: `pytest tests/test_report_assumption_summary.py -v`

Expected: FAIL — `note` is `"Sphericity violated → Greenhouse-Geisser correction applied"`
(no `"(ε = ...)"` suffix at all), so `assert "0.6543" in note` fails. This confirms the
bug is the *missing epsilon*, not a crash — matching the audit finding.

- [ ] **Step 3: Fix the key lookup**

In `src/export/report_summaries.py`, replace lines 450-464 (the `if status_value is False:`
block inside `_build_assumption_summary`):

Current code being replaced:
```python
            if status_value is False:
                corr = (sphericity.get("correction") or sphericity.get("correction_applied") or "").lower()
                gg_eps = sphericity.get("greenhouse_geisser") or sphericity.get("gg_epsilon") or sphericity.get("epsilon_gg")
                hf_eps = sphericity.get("huynh_feldt") or sphericity.get("hf_epsilon") or sphericity.get("epsilon_hf")
                if "huynh" in corr or "hf" in corr:
                    label = "Huynh-Feldt"
                    eps = hf_eps or gg_eps
                elif gg_eps or "greenhouse" in corr or "gg" in corr:
                    label = "Greenhouse-Geisser"
                    eps = gg_eps
                else:
                    label, eps = "Greenhouse-Geisser", gg_eps
                if label:
                    eps_str = f" (ε = {_FormattingMixin._format_metric(eps)})" if eps else ""
                    sphericity_correction_note = f"Sphericity violated → {label} correction applied{eps_str}"
```

New code:
```python
            if status_value is False:
                # Primary source: statisticaltester.py writes the correction label
                # to the top-level "correction_used" key and the epsilon values
                # nested under "sphericity_corrections", NOT into the
                # "sphericity_test" sub-dict this function reads for W/p_value.
                top_correction = str(results.get("correction_used") or "")
                sph_corrections = results.get("sphericity_corrections") or {}
                gg_block = sph_corrections.get("greenhouse_geisser") or {}
                hf_block = sph_corrections.get("huynh_feldt") or {}
                gg_eps = gg_block.get("epsilon") if isinstance(gg_block, dict) else None
                hf_eps = hf_block.get("epsilon") if isinstance(hf_block, dict) else None
                corr = top_correction.lower()
                if not corr:
                    # Fallback for older/serialized payloads that only populated
                    # the sphericity_test sub-dict directly.
                    corr = (sphericity.get("correction") or sphericity.get("correction_applied") or "").lower()
                    if gg_eps is None:
                        gg_eps = sphericity.get("greenhouse_geisser") or sphericity.get("gg_epsilon") or sphericity.get("epsilon_gg")
                    if hf_eps is None:
                        hf_eps = sphericity.get("huynh_feldt") or sphericity.get("hf_epsilon") or sphericity.get("epsilon_hf")
                if "huynh" in corr or "hf" in corr:
                    label = "Huynh-Feldt"
                    eps = hf_eps if hf_eps is not None else gg_eps
                elif gg_eps is not None or "greenhouse" in corr or "gg" in corr:
                    label = "Greenhouse-Geisser"
                    eps = gg_eps
                else:
                    label, eps = "Greenhouse-Geisser", gg_eps
                if label:
                    eps_str = f" (ε = {_FormattingMixin._format_metric(eps)})" if eps is not None else ""
                    sphericity_correction_note = f"Sphericity violated → {label} correction applied{eps_str}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_report_assumption_summary.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/export/report_summaries.py tests/test_report_assumption_summary.py
git commit -m "fix(report): read sphericity correction/epsilon from correct backend keys"
```

---

### Task 2: Add shared warning-annotation helper; use it for grouped-EMM plot fallback

**Files:**
- Modify: `src/visualization/datavisualizer.py:2496` (insert new helper after `annotate_box_medians`)
- Modify: `src/visualization/datavisualizer.py:2844-2846` (grouped-EMM except-block)
- Test: `tests/test_visualization_warning_annotations.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_visualization_warning_annotations.py`:

```python
"""Silent-fallback plot paths must draw a visible, export-persistent warning
on the axes, not just log a warning — per the "scientific transparency over
silent degradation" paradigm (docs/superpowers/specs/2026-07-03-visualization-error-transparency-design.md).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visualization.datavisualizer import DataVisualizer


def _emm_grouped_pairwise():
    return [
        {"group1": "ctrl:T0", "group2": "drug:T0", "test": "EMM + multivariate-t",
         "p_value": 0.01, "significant": True},
    ]


def test_grouped_emm_failure_draws_visible_warning(monkeypatch):
    def _boom(samples, sep=":"):
        raise RuntimeError("malformed group labels")

    monkeypatch.setattr(DataVisualizer, "grouped_inputs_from_samples", staticmethod(_boom))

    fig, ax = plt.subplots()
    groups = ["ctrl:T0", "ctrl:T1", "drug:T0", "drug:T1"]
    samples = {g: [1.0, 2.0, 3.0] for g in groups}
    config = {"plot_type": "Bar", "show_error_bars": False}

    DataVisualizer.plot_from_config(
        ax, groups, samples, config, pairwise_results=_emm_grouped_pairwise()
    )

    warning_texts = [t.get_text() for t in ax.texts if "Structural Warning" in t.get_text()]
    assert len(warning_texts) == 1, (
        "grouped-EMM fallback must draw an on-canvas warning, not just log one"
    )
    assert "flat pooling" in warning_texts[0]
    plt.close(fig)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_visualization_warning_annotations.py::test_grouped_emm_failure_draws_visible_warning -v`
Expected: FAIL — `AssertionError: grouped-EMM fallback must draw an on-canvas warning...`
(0 matching texts found; today the fallback only calls `logger.warning`).

- [ ] **Step 3: Add the shared helper and use it in the except-block**

In `src/visualization/datavisualizer.py`, insert this new staticmethod after
`annotate_box_medians` (after line 2496, before `add_reference_line` at line 2498):

```python
    @staticmethod
    def _draw_warning_annotation(ax, text):
        """
        Draw a high-contrast warning box directly on the axes so it survives
        figure export (PNG/SVG) instead of only appearing in the backend log.
        Used when a plot silently degrades or loses data rather than erroring.
        """
        ax.text(
            0.5, 1.02, text,
            transform=ax.transAxes,
            ha='center', va='bottom',
            fontsize=8, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.4', fc='#CC3300', ec='none', alpha=0.9),
            zorder=1000, clip_on=False,
        )
```

Then replace the except-block at lines 2844-2846:

Current code being replaced:
```python
                except Exception as exc:
                    logger.warning("grouped EMM plot failed (%s); using flat plot_bar", exc)
                    DataVisualizer.plot_bar(groups, samples, **bar_kwargs)
```

New code:
```python
                except Exception as exc:
                    logger.warning("grouped EMM plot failed (%s); using flat plot_bar", exc)
                    DataVisualizer.plot_bar(groups, samples, **bar_kwargs)
                    DataVisualizer._draw_warning_annotation(
                        ax, "Structural Warning: Within-Between interaction split "
                        "failed. Showing flat pooling.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_visualization_warning_annotations.py::test_grouped_emm_failure_draws_visible_warning -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/visualization/datavisualizer.py tests/test_visualization_warning_annotations.py
git commit -m "feat(viz): draw visible warning when grouped-EMM plot falls back to flat bar"
```

---

### Task 3: Defensive in-canvas warning for log-scale non-positive data loss

**Files:**
- Modify: `src/visualization/datavisualizer.py:1872-1897` (`_format_axes`)
- Modify: `src/visualization/datavisualizer.py:990, 1186, 1392` (the 3 call sites of `_format_axes`)
- Test: `tests/test_visualization_warning_annotations.py` (append to file created in Task 2)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_visualization_warning_annotations.py`:

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


def test_logscale_with_all_positive_data_draws_no_warning():
    fig, ax = plt.subplots()
    groups = ["A", "B"]
    samples = {"A": [1.0, 2.0, 3.0], "B": [3.0, 4.0, 5.0]}

    DataVisualizer.plot_bar(
        groups, samples, ax=ax, save_plot=False, logy=True, show_error_bars=False
    )

    warning_texts = [t.get_text() for t in ax.texts if "Data Warning" in t.get_text()]
    assert len(warning_texts) == 0
    plt.close(fig)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_visualization_warning_annotations.py::test_logscale_with_nonpositive_data_draws_warning -v`
Expected: FAIL — 0 matching warning texts found (today `_format_axes` sets the log
scale unconditionally with no point-omission check at all).

- [ ] **Step 3: Add omission counting + warning to `_format_axes`, thread `groups`/`samples` through its 3 call sites**

In `src/visualization/datavisualizer.py`, replace the `_format_axes` signature and
body (lines 1872-1897):

Current code being replaced:
```python
    def _format_axes(ax, y_format, y_limits, x_limits, grid_style, grid_alpha, spine_style,
                     tick_direction='out', offset_axes=False, axis_offset_points=10,
                     logx=False, logy=False, axis_break_enabled=False,
                     axis_break_start=20.0, axis_break_end=80.0):
        """Format axes according to specifications"""
        # Y-axis formatting
        if y_format == 'scientific':
            formatter = ScalarFormatter(useOffset=False, useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((0, 0))
            ax.yaxis.set_major_formatter(formatter)
        elif y_format == 'percentage':
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.1%}'))
        elif y_format == 'decimal':
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.2f}'))

        # Set limits
        if y_limits:
            ax.set_ylim(y_limits)
        if x_limits:
            ax.set_xlim(x_limits)

        if logx:
            ax.set_xscale('log', base=10)
        if logy:
            ax.set_yscale('log', base=10)
```

New code:
```python
    def _format_axes(ax, y_format, y_limits, x_limits, grid_style, grid_alpha, spine_style,
                     tick_direction='out', offset_axes=False, axis_offset_points=10,
                     logx=False, logy=False, axis_break_enabled=False,
                     axis_break_start=20.0, axis_break_end=80.0,
                     groups=None, samples=None):
        """Format axes according to specifications"""
        # Y-axis formatting
        if y_format == 'scientific':
            formatter = ScalarFormatter(useOffset=False, useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((0, 0))
            ax.yaxis.set_major_formatter(formatter)
        elif y_format == 'percentage':
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.1%}'))
        elif y_format == 'decimal':
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.2f}'))

        # Set limits
        if y_limits:
            ax.set_ylim(y_limits)
        if x_limits:
            ax.set_xlim(x_limits)

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

Then thread `groups`/`samples` through all 3 call sites. All 3 occurrences of this
exact block are byte-identical (confirmed by inspection at lines 990, 1186, 1392),
so a single `replace_all` edit covers all three:

Current code being replaced (matches 3 times):
```python
        DataVisualizer._format_axes(
            ax, y_axis_format, y_limits, x_limits,
            grid_style, grid_alpha, spine_style, tick_direction, offset_axes, axis_offset_points,
            logx, logy, axis_break_enabled, axis_break_start, axis_break_end
        )
```

New code (use `replace_all=true`):
```python
        DataVisualizer._format_axes(
            ax, y_axis_format, y_limits, x_limits,
            grid_style, grid_alpha, spine_style, tick_direction, offset_axes, axis_offset_points,
            logx, logy, axis_break_enabled, axis_break_start, axis_break_end,
            groups=groups, samples=samples,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_visualization_warning_annotations.py -v`
Expected: all 3 tests in the file PASS (Task 2's test + both Task 3 tests).

- [ ] **Step 5: Run the full visualization test suite to check for regressions**

Run: `pytest tests/test_plot_type_dispatch.py tests/test_rm_emm_plot_render.py tests/test_decision_tree_graphics.py -v`
Expected: all PASS (the `_format_axes` signature change is additive — new
keyword-only-by-convention params with defaults — so no existing caller breaks).

- [ ] **Step 6: Commit**

```bash
git add src/visualization/datavisualizer.py tests/test_visualization_warning_annotations.py
git commit -m "feat(viz): warn on-canvas when log-scale axis silently drops non-positive data"
```

---

### Task 4: UI-gating — disable "Log Y" when current data contains values ≤ 0

**Correction found during execution:** the plan as originally written assumed
`logy_check` lived directly on `PlotAestheticsDialog`. It actually lives on the
child `StyleTab` widget (`plot_aesthetics_dialog.py:620`, instantiated as
`self.style_tab = StyleTab(self.config)` at line 1624) — `StyleTab` only
receives `config`, not `samples`. `_apply_log_scale_gating` was implemented on
`PlotAestheticsDialog` (which does have `self.samples`) and reaches into
`self.style_tab.logy_check` after the tab is constructed, rather than
operating on `self.logy_check` directly. Same net behavior, corrected
attribute path.

**Files:**
- Modify: `src/ui/dialogs/plot_aesthetics_dialog.py:1568` (insert gating method after `__init__`)
- Modify: `src/ui/dialogs/plot_aesthetics_dialog.py:828-831` (call gating after checkbox creation)
- Test: `tests/test_plot_aesthetics_log_gating.py` (create)

Scope: "Log Y" only. See "Scope note found while tracing" at the top of this plan
for why "Log X" is out of scope (categorical group axis, no numeric x-data exists
in `samples` to gate on).

- [ ] **Step 1: Write the failing test**

Create `tests/test_plot_aesthetics_log_gating.py`:

```python
"""PlotAestheticsDialog already receives the real `samples` dict at
construction time (src/ui/dialogs/plot_aesthetics_dialog.py:1512,
self.samples = samples or {}). "Log Y" must be disabled up front when that
data contains values <= 0, since log(<=0) is undefined and matplotlib would
otherwise silently drop those points with no visible indication.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from PyQt5.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

from ui.dialogs.plot_aesthetics_dialog import PlotAestheticsDialog


def test_logy_checkbox_disabled_when_data_has_nonpositive_values():
    groups = ["A", "B"]
    samples = {"A": [1.0, 2.0, -0.5], "B": [3.0, 4.0, 5.0]}
    dialog = PlotAestheticsDialog(groups=groups, samples=samples, show_export_controls=False)
    try:
        assert dialog.logy_check.isEnabled() is False
        assert dialog.logy_check.isChecked() is False
        assert "≤ 0" in dialog.logy_check.toolTip()
    finally:
        dialog.close()


def test_logy_checkbox_enabled_when_data_all_positive():
    groups = ["A", "B"]
    samples = {"A": [1.0, 2.0, 3.0], "B": [3.0, 4.0, 5.0]}
    dialog = PlotAestheticsDialog(groups=groups, samples=samples, show_export_controls=False)
    try:
        assert dialog.logy_check.isEnabled() is True
    finally:
        dialog.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_plot_aesthetics_log_gating.py -v`
Expected: FAIL — `dialog.logy_check.isEnabled()` is `True` (checkbox has no gating
logic today; both tests' setup runs fine but the first assertion fails).

- [ ] **Step 3: Add the gating method and call it after checkbox creation**

In `src/ui/dialogs/plot_aesthetics_dialog.py`, insert this new method right after
`__init__` ends (after line 1568, before `def init_ui(self):` at line 1570):

```python
    def _apply_log_scale_gating(self):
        """
        Disable "Log Y" when self.samples contains non-positive values.
        log(<=0) is undefined; matplotlib silently drops those points on a
        log-scale axis with no visible warning, so gate it here instead.
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
            self.logy_check.setChecked(False)
            self.logy_check.setEnabled(False)
            self.logy_check.setToolTip("Log scale unavailable: data contains values ≤ 0.")
        else:
            self.logy_check.setEnabled(True)
            self.logy_check.setToolTip("")
```

Then call it right after the "Log Y" checkbox is created (after line 831):

Current code being replaced:
```python
        self.logy_check = QCheckBox("Log Y (base 10)")
        self.logy_check.setChecked(self.config.get('logy', False))
        self.logy_check.toggled.connect(self.settingsChanged)
        dynamics_layout.addWidget(self.logy_check, 0, 1)
```

New code:
```python
        self.logy_check = QCheckBox("Log Y (base 10)")
        self.logy_check.setChecked(self.config.get('logy', False))
        self.logy_check.toggled.connect(self.settingsChanged)
        dynamics_layout.addWidget(self.logy_check, 0, 1)
        self._apply_log_scale_gating()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_plot_aesthetics_log_gating.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ui/dialogs/plot_aesthetics_dialog.py tests/test_plot_aesthetics_log_gating.py
git commit -m "feat(ui): grey out Log Y when current data contains values <= 0"
```

---

### Task 5: Full regression check

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/ -v`
Expected: all previously-passing tests still pass (339 passed / 4 skipped baseline
per HANDOFF.md, now +7 new tests from Tasks 1-4: 346 passed / 4 skipped / 0 failed),
no new failures.

- [ ] **Step 2: If anything regressed, fix before proceeding**

If a call site of `_format_axes` outside the 3 already covered breaks, or any
other test depending on exact `ax.texts` contents (e.g. significance-letter tests)
now sees the extra warning annotation unexpectedly, investigate and fix — do not
suppress the new warning to make an unrelated test pass without understanding why
that test's `ax.texts` assumption changed.

---

## Self-review notes

- **Spec coverage:** A1 → Task 1. B1 → Task 2. B2(a) UI-gating → Task 4. B2(b)
  defensive in-canvas warning → Task 3. Both open questions from the spec resolved
  above (dialog already has `samples`; shared helper added in Task 2, reused in
  Task 3).
- **Type/signature consistency:** `_draw_warning_annotation(ax, text)` defined once
  in Task 2, called identically in Task 2 and Task 3. `_format_axes(..., groups=None,
  samples=None)` — new params are optional keyword args with safe defaults, so the
  3 threaded call sites (Task 3) and any call site this plan didn't touch (there are
  none — all 3 are covered) stay compatible.
- **No placeholders:** every step has literal code, exact line numbers, and exact
  pytest commands with expected output.
