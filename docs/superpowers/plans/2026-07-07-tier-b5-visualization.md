# Tier B5: Visualization Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix five independent, fully-diagnosed bugs in `src/visualization/datavisualizer.py`:
significance letters silently disappearing on any exception (VZ1), silently-dropped comparison
brackets with no notice (VZ2), `logx`'s non-positive-data handling lagging behind `logy`'s
symlog auto-adapt (VZ3), two WCAG-failing default colors (VZ4), and raincloud plots using a
narrower inline bracket-vs-letters check instead of the shared helper (VZ8).

**Architecture:** All five fixes are in one file and mirror patterns that already exist
elsewhere in the same file (the grouped-EMM fallback's `_draw_warning_annotation` call for VZ1,
`logy`'s symlog handling for VZ3, `_result_uses_brackets` for VZ8) — no new design, just
consistency fixes. Do all 5 as separate commits within one file to keep each diff reviewable,
but they can be implemented in one working session since they don't conflict with each other
(different functions).

**Tech Stack:** Python, matplotlib, pytest.

---

### Task 1: VZ1 — significance letters must show a visible warning on failure, not vanish silently

**Files:**
- Modify: `src/visualization/datavisualizer.py:2383-2386` (`_add_significance_letters`'s
  except block), `src/visualization/datavisualizer.py:2448-2449`
  (`_add_significance_letters_raincloud`'s except block)
- Test: `tests/test_significance_letters_warning.py`

Both functions catch their annotation logic's exceptions and only log — unlike two other
fallback paths in this same file (grouped-EMM fallback, log-axis fallback) which already call
`_draw_warning_annotation(ax, ...)` so the failure survives figure export, not just the backend
log.

- [ ] **Step 1: Write the failing test**

```python
"""_add_significance_letters and its raincloud variant catch their own exceptions and only
log.error() + traceback.print_exc() - unlike two other fallback paths in the same file
(grouped-EMM fallback, log-axis fallback) that already call _draw_warning_annotation so a
figure export doesn't silently ship with no significance annotations and no visible indication
why.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from visualization.datavisualizer import DataVisualizer


def test_add_significance_letters_failure_draws_visible_warning(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("synthetic significance-letters failure")
    monkeypatch.setattr(DataVisualizer, "get_significance_letters", staticmethod(_boom))

    fig, ax = plt.subplots()
    df = pd.DataFrame({"Value": [1.0, 2.0, 3.0, 4.0]})
    groups = ["A", "B"]
    samples = {"A": [1.0, 2.0], "B": [3.0, 4.0]}

    DataVisualizer._add_significance_letters(
        ax, df, groups, samples, test_recommendation="anova",
        height_offset=0.05, font_size=10, error_type="sd", pairwise_results=None
    )

    warning_texts = [t.get_text() for t in ax.texts if "Warning" in t.get_text()]
    assert warning_texts, f"expected a visible warning annotation, got: {[t.get_text() for t in ax.texts]}"
    plt.close(fig)


def test_add_significance_letters_raincloud_failure_draws_visible_warning(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("synthetic significance-letters failure")
    monkeypatch.setattr(DataVisualizer, "get_significance_letters", staticmethod(_boom))

    fig, ax = plt.subplots()
    groups = ["A", "B"]
    samples = {"A": [1.0, 2.0], "B": [3.0, 4.0]}

    DataVisualizer._add_significance_letters_raincloud(
        ax, groups, samples, test_recommendation="anova",
        height_offset=0.05, font_size=10, positions=None, pairwise_results=None
    )

    warning_texts = [t.get_text() for t in ax.texts if "Warning" in t.get_text()]
    assert warning_texts, f"expected a visible warning annotation, got: {[t.get_text() for t in ax.texts]}"
    plt.close(fig)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/test_significance_letters_warning.py -v`
Expected: FAIL — `ax.texts` is empty (both functions log to the backend logger only).

- [ ] **Step 3: Fix `_add_significance_letters`**

Read the current except block fresh
(`grep -n "Error adding significance letters" src/visualization/datavisualizer.py`), then change:

```python
        except Exception as e:
            logger.error(f"Error adding significance letters: {str(e)}")
            import traceback
            traceback.print_exc()
```

to:

```python
        except Exception as e:
            logger.error(f"Error adding significance letters: {str(e)}")
            import traceback
            traceback.print_exc()
            DataVisualizer._draw_warning_annotation(
                ax, "Warning: significance letters could not be computed and are not shown."
            )
```

- [ ] **Step 4: Fix `_add_significance_letters_raincloud`**

Read the current except block fresh
(`grep -n "Error adding raincloud significance letters" src/visualization/datavisualizer.py`),
then change:

```python
        except Exception as e:
            logger.error(f"Error adding raincloud significance letters: {str(e)}")
```

to:

```python
        except Exception as e:
            logger.error(f"Error adding raincloud significance letters: {str(e)}")
            DataVisualizer._draw_warning_annotation(
                ax, "Warning: significance letters could not be computed and are not shown."
            )
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/test_significance_letters_warning.py -v`
Expected: PASS (both cases).

- [ ] **Step 6: Commit**

```bash
git add tests/test_significance_letters_warning.py src/visualization/datavisualizer.py
git commit -m "fix(viz): show a visible warning when significance letters fail to compute"
```

---

### Task 2: VZ2 — surface a notice when comparison brackets get silently dropped

**Files:**
- Modify: `src/visualization/datavisualizer.py:394` (call site),
  `src/visualization/datavisualizer.py:497-533` (`_grouped_bracket_positions`),
  `src/visualization/datavisualizer.py:576-577` (call site),
  `src/visualization/datavisualizer.py:647-696ish` (`_calculate_bracket_positions`)
- Test: `tests/test_bracket_drop_notice.py`

Both bracket-position builders silently skip any comparison whose groups don't resolve to a
known position, with no count or notice. Add a dropped-count tally and a
`_draw_notice_annotation` call when any comparisons were dropped.

- [ ] **Step 1: Write the failing test**

```python
"""_grouped_bracket_positions and _calculate_bracket_positions silently skip any pairwise
comparison whose group(s) don't resolve to a known bar/position, with no count or notice drawn
on the figure - a scientist has no way to tell some comparisons are simply missing from the
plot rather than "not significant."
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from visualization.datavisualizer import DataVisualizer


def test_grouped_bracket_positions_notices_dropped_comparisons():
    fig, ax = plt.subplots()
    centers = {"A": 0.0, "B": 1.0}
    label_map = {"grpA": "A", "grpB": "B"}
    pairwise_results = [
        {"group1": "grpA", "group2": "grpB", "p_value": 0.01},
        {"group1": "grpA", "group2": "unknown_group", "p_value": 0.02},  # unresolvable -> dropped
    ]

    brackets = DataVisualizer._grouped_bracket_positions(
        ax, centers, label_map, pairwise_results, y_max=10.0, line_height=0.05
    )

    assert len(brackets) == 1, "the resolvable comparison must still produce a bracket"
    notice_texts = [t.get_text() for t in ax.texts if "Notice" in t.get_text() or "notice" in t.get_text().lower()]
    assert notice_texts, f"expected a dropped-comparison notice, got: {[t.get_text() for t in ax.texts]}"
    assert "1" in notice_texts[0]
    plt.close(fig)


def test_calculate_bracket_positions_notices_dropped_comparisons():
    fig, ax = plt.subplots()
    ax.bar([1, 2], [1.0, 2.0])  # so _detect_plot_type has something to detect
    groups = ["A", "B"]
    compare = ["A", "B"]
    pairwise_results = [
        {"group1": "A", "group2": "B", "p_value": 0.01},
        {"group1": "A", "group2": "unknown_group", "p_value": 0.02},  # unresolvable -> dropped
    ]

    brackets = DataVisualizer._calculate_bracket_positions(
        ax, groups, compare, pairwise_results, y_max=10.0, line_height=0.05
    )

    assert len(brackets) == 1
    notice_texts = [t.get_text() for t in ax.texts if "notice" in t.get_text().lower()]
    assert notice_texts, f"expected a dropped-comparison notice, got: {[t.get_text() for t in ax.texts]}"
    plt.close(fig)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/test_bracket_drop_notice.py -v`
Expected: FAIL on the first test with a `TypeError` (`_grouped_bracket_positions` doesn't take
`ax` as its first argument yet), and the second test's notice assertion fails (no notice drawn).

- [ ] **Step 3: Fix `_grouped_bracket_positions` — add `ax` param and a dropped-count notice**

Read the current function fresh
(`grep -n "def _grouped_bracket_positions" src/visualization/datavisualizer.py`), then change:

```python
    def _grouped_bracket_positions(centers, label_map, pairwise_results,
                                   y_max, line_height):
        """Build bracket dicts for treatment-vs-control comparisons using bar
        patch centers. Each comparison's two groups resolve (via label_map) to
        (between, within) cells; their center x become x1/x2. Heights are
        stacked with the existing x-overlap collision check. Comparisons whose
        groups are not both resolvable/keyed are skipped (defensive).
        """
        base_height = y_max * 1.05
        step = y_max * line_height
        prepared = []
        for comp in pairwise_results:
            g1, g2 = comp.get("group1"), comp.get("group2")
            c1, c2 = label_map.get(g1), label_map.get(g2)
            if c1 is None or c2 is None or c1 not in centers or c2 not in centers:
                continue
            x1, x2 = centers[c1], centers[c2]
            if x1 > x2:
                x1, x2 = x2, x1
            prepared.append({"comp": comp, "x1": x1, "x2": x2,
                             "distance": abs(x2 - x1)})

        prepared.sort(key=lambda d: d["distance"])
        used = []
        brackets = []
        for d in prepared:
            x1, x2 = d["x1"], d["x2"]
            height = base_height
            level = 0
            while DataVisualizer._brackets_collide(x1, x2, height, used):
                level += 1
                height = base_height + step * level * 1.2
            brackets.append({"x1": x1, "x2": x2, "height": height,
                             "p_value": d["comp"].get("p_value"),
                             "comp": d["comp"]})
            used.append((x1, x2, height))
        return brackets
```

to:

```python
    def _grouped_bracket_positions(ax, centers, label_map, pairwise_results,
                                   y_max, line_height):
        """Build bracket dicts for treatment-vs-control comparisons using bar
        patch centers. Each comparison's two groups resolve (via label_map) to
        (between, within) cells; their center x become x1/x2. Heights are
        stacked with the existing x-overlap collision check. Comparisons whose
        groups are not both resolvable/keyed are skipped (defensive) - a
        dropped-count notice is drawn on `ax` when this happens (VZ2).
        """
        base_height = y_max * 1.05
        step = y_max * line_height
        prepared = []
        dropped = 0
        for comp in pairwise_results:
            g1, g2 = comp.get("group1"), comp.get("group2")
            c1, c2 = label_map.get(g1), label_map.get(g2)
            if c1 is None or c2 is None or c1 not in centers or c2 not in centers:
                dropped += 1
                continue
            x1, x2 = centers[c1], centers[c2]
            if x1 > x2:
                x1, x2 = x2, x1
            prepared.append({"comp": comp, "x1": x1, "x2": x2,
                             "distance": abs(x2 - x1)})

        prepared.sort(key=lambda d: d["distance"])
        used = []
        brackets = []
        for d in prepared:
            x1, x2 = d["x1"], d["x2"]
            height = base_height
            level = 0
            while DataVisualizer._brackets_collide(x1, x2, height, used):
                level += 1
                height = base_height + step * level * 1.2
            brackets.append({"x1": x1, "x2": x2, "height": height,
                             "p_value": d["comp"].get("p_value"),
                             "comp": d["comp"]})
            used.append((x1, x2, height))

        if dropped > 0:
            DataVisualizer._draw_notice_annotation(
                ax, f"Notice: {dropped} comparison(s) could not be positioned and were omitted."
            )
        return brackets
```

Update the one call site (`grep -n "_grouped_bracket_positions(" src/visualization/datavisualizer.py`):

```python
            brackets = DataVisualizer._grouped_bracket_positions(
                centers, label_map, pairwise_results, y_max, comparison_line_height)
```
to:
```python
            brackets = DataVisualizer._grouped_bracket_positions(
                ax, centers, label_map, pairwise_results, y_max, comparison_line_height)
```

- [ ] **Step 4: Fix `_calculate_bracket_positions` — add a dropped-count notice**

Read the current function fresh (it already takes `ax` as its first parameter, so no signature
change needed here). Find where `comparisons` finishes being built
(`grep -n "comparisons.sort(key=lambda x: x\['distance'\])" src/visualization/datavisualizer.py`)
and the point right after `return brackets` is reached at the end of the function. Add, right
before the final `return brackets` of this function:

```python
        if len(comparisons) < len(pairwise_results):
            dropped = len(pairwise_results) - len(comparisons)
            DataVisualizer._draw_notice_annotation(
                ax, f"Notice: {dropped} comparison(s) could not be positioned and were omitted."
            )
        return brackets
```

(Insert this immediately before the function's existing `return brackets` line — read the
function fully first via `sed -n '647,735p' src/visualization/datavisualizer.py` to find its
exact current end, since the intervening code wasn't fully re-quoted here.)

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/test_bracket_drop_notice.py -v`
Expected: PASS (both cases).

- [ ] **Step 6: Run the full test suite (this task changes a function signature)**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/ -q --tb=no`
Expected: same pass count as before this task plus the 2 new tests (the 1 pre-existing unrelated
`test_convergence.py::test_convergence_keys` failure is expected). Since
`_grouped_bracket_positions`'s signature changed (added `ax` as the first param), specifically
check for any other test file calling it directly with the old signature —
`grep -rn "_grouped_bracket_positions(" tests/` — and update any hit found.

- [ ] **Step 7: Commit**

```bash
git add tests/test_bracket_drop_notice.py src/visualization/datavisualizer.py
git commit -m "fix(viz): surface a notice when comparison brackets are silently dropped"
```

---

### Task 3: VZ3 — bring `logx` up to `logy`'s symlog auto-adapt standard

**Files:**
- Modify: `src/visualization/datavisualizer.py:1898-1904` (inside `_format_axes`)
- Test: `tests/test_logx_symlog_parity.py`

`logy`'s handling (lines 1906-1927) auto-adapts to `symlog` scaling when non-positive values are
present and a usable linthresh can be derived (lossless), only falling back to a plain log +
warning when no usable threshold exists. `logx` (lines 1898-1904) always warns-and-drops,
regardless of whether a lossless symlog adaptation was possible.

- [ ] **Step 1: Write the failing test**

Mirrors the existing `logy` tests in `tests/test_visualization_warning_annotations.py`
(`test_logscale_with_nonpositive_data_uses_symlog_not_plain_log`,
`test_logscale_with_all_zero_data_falls_back_to_plain_log_with_warning`) — read that file first
to confirm its current exact assertions, then add the `logx` equivalents:

```python
"""_format_axes's logx branch always warns-and-drops non-positive values, unlike logy which
auto-adapts to a lossless symlog scale when a usable linthresh can be derived. Mirrors the
existing logy tests in test_visualization_warning_annotations.py.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from visualization.datavisualizer import DataVisualizer


def test_logx_with_nonpositive_data_uses_symlog_not_plain_log():
    fig, ax = plt.subplots()
    groups = ["A", "B"]
    samples = {"A": [1.0, 2.0, -0.5, 0.0], "B": [3.0, 4.0]}

    DataVisualizer.plot_bar(
        groups, samples, ax=ax, save_plot=False, logx=True, show_error_bars=False
    )

    assert ax.get_xscale() == "symlog"
    notice_texts = [t.get_text() for t in ax.texts if "Data Notice" in t.get_text()]
    assert len(notice_texts) == 1
    assert "symlog" in notice_texts[0]
    warning_texts = [t.get_text() for t in ax.texts if "Data Warning" in t.get_text()]
    assert len(warning_texts) == 0, "lossless symlog path must not show the red data-loss warning"
    plt.close(fig)


def test_logx_with_all_zero_data_falls_back_to_plain_log_with_warning():
    fig, ax = plt.subplots()
    groups = ["A"]
    samples = {"A": [0.0, 0.0, 0.0]}

    DataVisualizer.plot_bar(
        groups, samples, ax=ax, save_plot=False, logx=True, show_error_bars=False
    )

    assert ax.get_xscale() == "log"
    warning_texts = [t.get_text() for t in ax.texts if "Data Warning" in t.get_text()]
    assert len(warning_texts) == 1
    plt.close(fig)


def test_logx_with_all_positive_data_draws_no_warning():
    fig, ax = plt.subplots()
    groups = ["A", "B"]
    samples = {"A": [1.0, 2.0, 3.0], "B": [3.0, 4.0, 5.0]}

    DataVisualizer.plot_bar(
        groups, samples, ax=ax, save_plot=False, logx=True, show_error_bars=False
    )

    warning_texts = [t.get_text() for t in ax.texts if "Data Warning" in t.get_text()]
    assert len(warning_texts) == 0
    plt.close(fig)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/test_logx_symlog_parity.py -v`
Expected: FAIL on `test_logx_with_nonpositive_data_uses_symlog_not_plain_log` —
`ax.get_xscale()` is `"log"`, not `"symlog"`.

- [ ] **Step 3: Fix — mirror `logy`'s symlog auto-adapt for `logx`**

Read `_format_axes` fresh (`grep -n "if logx:" src/visualization/datavisualizer.py`), then
change:

```python
        if logx:
            ax.set_xscale('log', base=10)
            if samples:
                count_x, _ = DataVisualizer._analyze_nonpositive_values(groups, samples)
                if count_x > 0:
                    DataVisualizer._draw_warning_annotation(
                        ax, f"Data Warning: {count_x} values ≤ 0 omitted from log-scale axis.")
```

to:

```python
        if logx:
            count_x, linthresh_x = (
                DataVisualizer._analyze_nonpositive_values(groups, samples) if samples else (0, None)
            )
            if count_x > 0 and linthresh_x is not None:
                # Lossless path: symlog preserves near-zero/negative readings
                # instead of dropping them - mirrors the logy branch below.
                ax.set_xscale('symlog', linthresh=linthresh_x)
                ax.xaxis.set_major_locator(SymmetricalLogLocator(base=10, linthresh=linthresh_x))
                ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10, linthresh=linthresh_x))
                DataVisualizer._draw_notice_annotation(
                    ax, f"Data Notice: Values ≤ 0 detected. Auto-applied symlog scale "
                    f"(linthresh = {linthresh_x:.4g}).")
            elif count_x > 0:
                ax.set_xscale('log', base=10)
                DataVisualizer._draw_warning_annotation(
                    ax, f"Data Warning: {count_x} values ≤ 0 omitted from log-scale axis.")
            else:
                ax.set_xscale('log', base=10)
```

(`SymmetricalLogLocator` and `LogFormatterMathtext` are already imported at module level — the
`logy` branch a few lines below already uses both.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/test_logx_symlog_parity.py -v`
Expected: PASS (all 3 cases).

- [ ] **Step 5: Commit**

```bash
git add tests/test_logx_symlog_parity.py src/visualization/datavisualizer.py
git commit -m "fix(viz): bring logx up to logy's lossless symlog auto-adapt standard"
```

---

### Task 4: VZ4 — swap 2 WCAG-failing default colors

**Files:**
- Modify: `src/visualization/datavisualizer.py:798`
- Test: `tests/test_default_colors_contrast.py`

`DataVisualizer.DEFAULT_COLORS` contains `#33FF57` (bright green, 1.35:1 contrast against white)
and `#33FFEC` (bright cyan, 1.26:1) — both fail WCAG's 3:1 non-text floor. Verified during
planning (standard sRGB relative-luminance formula): `#2E7D32` (darker green) computes to
5.13:1, and `#00897B` (teal) computes to 4.32:1 — both comfortably clear the floor while staying
in the same hue family as the originals.

- [ ] **Step 1: Write the failing test**

```python
"""DataVisualizer.DEFAULT_COLORS must clear WCAG's 3:1 non-text contrast floor against a white
plot background - #33FF57 and #33FFEC (1.35:1 and 1.26:1, computed via the standard sRGB
relative-luminance formula) currently fail it.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from visualization.datavisualizer import DataVisualizer


def _relative_luminance(hex_color):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    def lin(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = lin(r), lin(g), lin(b)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast_ratio(hex1, hex2):
    l1, l2 = _relative_luminance(hex1), _relative_luminance(hex2)
    l1, l2 = max(l1, l2), min(l1, l2)
    return (l1 + 0.05) / (l2 + 0.05)


def test_default_colors_all_clear_wcag_non_text_floor_against_white():
    failing = [
        (c, round(_contrast_ratio(c, "#FFFFFF"), 2))
        for c in DataVisualizer.DEFAULT_COLORS
        if _contrast_ratio(c, "#FFFFFF") < 3.0
    ]
    assert not failing, f"colors failing WCAG's 3:1 non-text floor against white: {failing}"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/test_default_colors_contrast.py -v`
Expected: FAIL — `failing` contains `('#33FF57', 1.35)` and `('#33FFEC', 1.26)`.

- [ ] **Step 3: Fix — swap the two failing colors**

Read the current line fresh (`grep -n "DEFAULT_COLORS = " src/visualization/datavisualizer.py`),
then change:

```python
    DEFAULT_COLORS = ['#3357FF', '#FF5733', '#33FF57', '#F033FF', '#FF3366', '#33FFEC']
```

to:

```python
    DEFAULT_COLORS = ['#3357FF', '#FF5733', '#2E7D32', '#F033FF', '#FF3366', '#00897B']
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/test_default_colors_contrast.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_default_colors_contrast.py src/visualization/datavisualizer.py
git commit -m "fix(viz): swap 2 WCAG-failing DEFAULT_COLORS entries for compliant alternatives"
```

---

### Task 5: VZ8 — route raincloud plots through the shared bracket-vs-letters helper

**Files:**
- Modify: `src/visualization/datavisualizer.py:1707-1717` (inside `plot_raincloud`)
- Test: `tests/test_raincloud_bracket_parity.py`

`plot_raincloud` uses its own narrower inline check instead of the shared
`_result_uses_brackets` helper `plot_bar`/`plot_violin`/`plot_box` all use — the same
statistical result (e.g. a Tukey HSD all-pairs post-hoc) renders as compact letters on
Bar/Box/Violin but as brackets on Raincloud, purely because of which plot type was picked.

- [ ] **Step 1: Write the failing test**

```python
"""plot_raincloud must use the same _result_uses_brackets logic Bar/Box/Violin already use,
instead of its own narrower inline check - otherwise the same post-hoc result renders
differently (letters vs brackets) purely based on which plot type is selected.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from visualization.datavisualizer import DataVisualizer


def test_raincloud_uses_shared_bracket_helper_for_all_pairs_posthoc():
    # An all-pairs test (e.g. Tukey/Games-Howell/Dunn) should render as
    # compact letters, per _result_uses_brackets's own logic - NOT brackets,
    # even though plot_raincloud's old inline check would show brackets for
    # any non-empty pairwise_results regardless of test type.
    pairwise_results = [
        {"group1": "A", "group2": "B", "test": "Tukey HSD", "p_value": 0.01},
    ]
    assert DataVisualizer._result_uses_brackets(pairwise_results, None) is False, (
        "sanity check: the shared helper itself must say Tukey HSD uses letters, not brackets"
    )
```

**Note:** this test only pins down `_result_uses_brackets`'s own behavior (already correct,
unchanged) as a sanity check for Step 3 below — the actual regression proof is structural
(Step 3 replaces `plot_raincloud`'s inline logic with a call to this exact function), verified
by Step 4's full-suite run rather than a new integration test of `plot_raincloud` itself (which
would need a full raincloud-plot rendering harness disproportionate to a parity fix).

- [ ] **Step 2: Run the test to verify it passes already (this is a sanity check, not a RED step)**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/test_raincloud_bracket_parity.py -v`
Expected: PASS already — `_result_uses_brackets` itself is not being changed by this task, only
`plot_raincloud`'s use of it.

- [ ] **Step 3: Fix — replace `plot_raincloud`'s inline logic with the shared helper**

Read `plot_raincloud` fresh (`grep -n "Entscheide, ob Buchstaben oder Bars" src/visualization/datavisualizer.py`),
then change:

```python
        # Entscheide, ob Buchstaben oder Bars angezeigt werden sollen
        show_letters = True
        show_bars = False
        if posthoc_method is not None and isinstance(posthoc_method, str):
            method_lower = posthoc_method.lower()
            if method_lower in ["pairwise t-test", "pairwise t test", "pairwise mann-whitney", "pairwise mann whitney", "pairwise_mannwhitney", "pairwise_ttest"]:
                show_letters = False
                show_bars = True
        elif pairwise_results is not None and len(pairwise_results) > 0:
            show_letters = False
            show_bars = True
```

to:

```python
        # Entscheide, ob Buchstaben oder Bars angezeigt werden sollen - via
        # the shared helper Bar/Box/Violin already use, for parity (VZ8).
        show_bars = DataVisualizer._result_uses_brackets(pairwise_results, posthoc_method)
        show_letters = not show_bars
```

- [ ] **Step 4: Run the full test suite**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/ -q --tb=no`
Expected: same pass count as before this task plus the 1 new sanity-check test (the 1
pre-existing unrelated `test_convergence.py::test_convergence_keys` failure is expected). Pay
particular attention to any existing raincloud-plot tests
(`grep -rln "plot_raincloud" tests/`) — run those explicitly with `-v` to confirm no visual
regression in their assertions.

- [ ] **Step 5: Commit**

```bash
git add tests/test_raincloud_bracket_parity.py src/visualization/datavisualizer.py
git commit -m "fix(viz): route raincloud plots through the shared bracket-vs-letters helper"
```

---

## Self-review notes

- **Spec coverage:** VZ1 (Task 1), VZ2 (Task 2), VZ3 (Task 3), VZ4 (Task 4), VZ8 (Task 5) — all
  5 findings assigned to this package are covered.
- **VZ4's replacement colors are computed, not eyeballed** — `#2E7D32` (5.13:1) and `#00897B`
  (4.32:1), both verified against the standard sRGB relative-luminance formula during planning,
  matching the rigor the audit itself used to find the original failures.
- **VZ2's `_grouped_bracket_positions` signature change (`ax` added as first param) is called
  out explicitly** with a full-suite run in Step 6 to catch any other caller of the old
  signature, since a positional-arg signature change is exactly the kind of edit that silently
  breaks a caller elsewhere.
- **VZ8 deliberately does not build a full `plot_raincloud` rendering test** — the fix is a
  structural one-line replacement of already-tested logic (`_result_uses_brackets`), and a full
  raincloud rendering harness would be disproportionate; the full-suite run in Step 4 is the
  real regression check.
