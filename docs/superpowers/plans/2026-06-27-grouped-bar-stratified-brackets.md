# Grouped Bars + Stratified Simple-Main-Effect Brackets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render two-factor (Mixed / Two-Way) designs as true grouped bar plots (x = within/first factor, hue = between/second factor) and draw treatment-vs-control significance brackets *within each stratum*, using bar-patch geometry for exact x-coordinates.

**Architecture:** A new, isolated `plot_grouped_bar` function (the existing flat `plot_bar` stays untouched, so one-way/2-group plots cannot regress). It builds a long-form DataFrame, calls `sns.barplot(x=within, hue=between)`, extracts each bar's center x from the rendered matplotlib patches, and reuses the existing bracket drawing/collision machinery (`_draw_single_bracket`, `_brackets_collide`) fed with patch-center coordinates. Routing detects the control-referenced EMM/Dunnett mixed result and dispatches to the new function.

**Tech Stack:** seaborn (grouped barplot via `hue`), matplotlib (patch geometry, manual brackets), numpy/pandas, pytest with `matplotlib.use("Agg")` for headless geometry assertions.

---

## Verified facts (established before writing this plan — do not re-derive)

1. **seaborn grouped-bar patch ordering is hue-major.** For `sns.barplot(x=within, hue=between, order=within_order, hue_order=between_order)`, the non-zero-width patches come out in the order `between_order[0]×(all within), between_order[1]×(all within), ...`. So `patch_index = between_idx * len(within_order) + within_idx`. (Spiked: 3 groups × 3 times → 9 patches; Ctrl@T1,T2,T3 then TrtA@T1,T2,T3 then TrtB@…).
2. **Bar center x** = `patch.get_x() + patch.get_width()/2`. Dodge offsets are automatic (e.g. for 3 hues at tick 0: centers ≈ -0.267, 0.0, +0.267; width ≈ 0.267).
3. **Bracket drawing is proprietary matplotlib**, no statannotations. `_draw_single_bracket(ax, bracket, line_width, font_size, bracket_color, vertical_fraction, p_value_style)` consumes a dict `{'x1','x2','height','p_value'}` and draws two verticals + a horizontal + a centered label — purely from x-coordinates (datavisualizer.py:585-615).
4. **Collision stacking** `_brackets_collide(x1, x2, height, used)` (datavisualizer.py:558-568) is x-overlap based and can be reused directly with patch-center coordinates (unlike `_brackets_collide_improved`, which is group-index based).
5. **The EMM result only ever contains within-stratum control-vs-treatment pairs** (`group1="Ctrl:T1"`, `group2="TrtA:T1"`, …). No cross-stratum pairs exist, so no cross-stratum brackets can be produced — the "Treatment:T2 vs Ctrl:T1" risk is structurally absent.
6. **Existing `plot_bar`** (datavisualizer.py:664) takes flat `groups`/`samples` (no factor identity) and renders `sns.barplot(x='Group', order=groups)`. It is NOT modified by this plan.
7. **Bars-vs-letters dispatch** lives at datavisualizer.py ~840 inside `plot_bar`; it routes everything that is not a flat pairwise/MWU test to letters. Irrelevant once the grouped path owns its own annotation, but the routing task (Task 5) makes the controller call `plot_grouped_bar` for the EMM mixed result.

---

## File Structure

- Modify `src/visualization/datavisualizer.py`: add three new static methods — `_grouped_bar_centers`, `_grouped_bracket_positions`, `plot_grouped_bar` — and reuse the existing `_draw_single_bracket`, `_brackets_collide`, `_get_plot_max_height_robust`, `_format_p_value_label`.
- Create `tests/test_grouped_bar_brackets.py`: headless geometry + bracket unit tests.
- Modify the plot caller (the autopilot `configure_plot_from_result` path — exact location resolved in Task 5) to detect the control-referenced Mixed result and dispatch to `plot_grouped_bar` with the factor structure.

RM-only (within-only, no between factor) grouped plotting is **out of scope** (no hue factor); it keeps the existing rendering until its own EMM milestone.

---

## Task 1: Patch-center extraction for grouped bars

**Files:**
- Modify: `src/visualization/datavisualizer.py` (add `_grouped_bar_centers` static method, e.g. next to `_get_group_extents`)
- Test: `tests/test_grouped_bar_brackets.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_grouped_bar_brackets.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest
import seaborn as sns
from visualization.datavisualizer import DataVisualizer


def _grouped_ax():
    df = pd.DataFrame({
        "Time":  ["T1", "T2", "T3"] * 3,
        "Group": ["Ctrl"] * 3 + ["TrtA"] * 3 + ["TrtB"] * 3,
        "Value": [1.0, 2.0, 3.0, 1.5, 2.5, 4.0, 0.9, 1.8, 2.2],
    })
    fig, ax = plt.subplots()
    sns.barplot(data=df, x="Time", y="Value", hue="Group",
                order=["T1", "T2", "T3"], hue_order=["Ctrl", "TrtA", "TrtB"],
                errorbar=None, ax=ax)
    return ax


def test_grouped_bar_centers_maps_each_cell_to_its_dodged_center():
    ax = _grouped_ax()
    centers = DataVisualizer._grouped_bar_centers(
        ax, between_order=["Ctrl", "TrtA", "TrtB"], within_order=["T1", "T2", "T3"]
    )
    # 9 cells present
    assert len(centers) == 9
    # Within tick 0 (T1): Ctrl left, TrtA center (~0.0), TrtB right
    assert centers[("Ctrl", "T1")] < centers[("TrtA", "T1")] < centers[("TrtB", "T1")]
    assert centers[("TrtA", "T1")] == pytest.approx(0.0, abs=1e-6)
    # Same dodge pattern repeats at every tick (constant offset)
    off = centers[("Ctrl", "T1")] - centers[("TrtA", "T1")]
    assert centers[("Ctrl", "T2")] - centers[("TrtA", "T2")] == pytest.approx(off, abs=1e-6)
    assert centers[("TrtA", "T2")] == pytest.approx(1.0, abs=1e-6)
```

- [ ] **Step 2: Run it, verify it FAILS**

Run: `python -m pytest tests/test_grouped_bar_brackets.py::test_grouped_bar_centers_maps_each_cell_to_its_dodged_center -v`
Expected: FAIL — `AttributeError: ... has no attribute '_grouped_bar_centers'`

- [ ] **Step 3: Implement `_grouped_bar_centers`**

```python
    @staticmethod
    def _grouped_bar_centers(ax, between_order, within_order):
        """Map each (between_level, within_level) cell to its rendered bar's
        center x. seaborn lays grouped bars out hue-major: for each hue level
        (between_order), all x positions (within_order) in order.
        """
        bars = [p for p in ax.patches
                if hasattr(p, "get_width") and (p.get_width() or 0) > 0]
        n_w = len(within_order)
        expected = len(between_order) * n_w
        if len(bars) < expected:
            raise ValueError(
                f"grouped barplot has {len(bars)} bars, expected {expected}")
        centers = {}
        for h_idx, b_lev in enumerate(between_order):
            for w_idx, w_lev in enumerate(within_order):
                p = bars[h_idx * n_w + w_idx]
                centers[(b_lev, w_lev)] = p.get_x() + p.get_width() / 2.0
        return centers
```

- [ ] **Step 4: Run it, verify it PASSES**

Run: `python -m pytest tests/test_grouped_bar_brackets.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/visualization/datavisualizer.py tests/test_grouped_bar_brackets.py
git commit -m "feat(viz): patch-center extraction for grouped bar plots"
```

---

## Task 2: Stratified bracket positions from patch centers

**Files:**
- Modify: `src/visualization/datavisualizer.py` (add `_grouped_bracket_positions`)
- Test: `tests/test_grouped_bar_brackets.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_grouped_bar_brackets.py
def test_grouped_bracket_positions_only_within_stratum_and_stacked():
    centers = {
        ("Ctrl", "T1"): -0.27, ("TrtA", "T1"): 0.0, ("TrtB", "T1"): 0.27,
        ("Ctrl", "T2"): 0.73, ("TrtA", "T2"): 1.0, ("TrtB", "T2"): 1.27,
    }
    # cell-label -> (between, within); labels match pairwise group1/group2
    label_map = {
        "Ctrl:T1": ("Ctrl", "T1"), "TrtA:T1": ("TrtA", "T1"), "TrtB:T1": ("TrtB", "T1"),
        "Ctrl:T2": ("Ctrl", "T2"), "TrtA:T2": ("TrtA", "T2"), "TrtB:T2": ("TrtB", "T2"),
    }
    pairwise = [
        {"group1": "Ctrl:T1", "group2": "TrtA:T1", "p_value": 0.2,  "significant": False},
        {"group1": "Ctrl:T1", "group2": "TrtB:T1", "p_value": 0.001, "significant": True},
        {"group1": "Ctrl:T2", "group2": "TrtA:T2", "p_value": 0.04, "significant": True},
    ]
    brackets = DataVisualizer._grouped_bracket_positions(
        centers, label_map, pairwise, y_max=10.0, line_height=0.08
    )
    assert len(brackets) == 3
    # every bracket spans cells of ONE stratum only (x within that tick's cluster)
    for b in brackets:
        assert b["x1"] < b["x2"]
        assert {round(b["x1"], 2), round(b["x2"], 2)} <= {-0.27, 0.0, 0.27, 0.73, 1.0, 1.27}
    # the two T1 brackets (overlapping x) get different heights (collision stacking)
    t1 = sorted([b["height"] for b in brackets if b["x1"] < 0.5])
    assert t1[0] < t1[1]
```

- [ ] **Step 2: Run it, verify it FAILS**

Run: `python -m pytest tests/test_grouped_bar_brackets.py::test_grouped_bracket_positions_only_within_stratum_and_stacked -v`
Expected: FAIL — no `_grouped_bracket_positions`.

- [ ] **Step 3: Implement `_grouped_bracket_positions`**

```python
    @staticmethod
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

        prepared.sort(key=lambda d: d["distance"])     # shorter brackets first
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

- [ ] **Step 4: Run it, verify it PASSES**

Run: `python -m pytest tests/test_grouped_bar_brackets.py -v`
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add src/visualization/datavisualizer.py tests/test_grouped_bar_brackets.py
git commit -m "feat(viz): stratified bracket positions from bar patch centers"
```

---

## Task 3: `plot_grouped_bar` — render grouped bars and draw the brackets

**Files:**
- Modify: `src/visualization/datavisualizer.py` (add `plot_grouped_bar`)
- Test: `tests/test_grouped_bar_brackets.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_grouped_bar_brackets.py
def test_plot_grouped_bar_renders_bars_and_significant_brackets():
    long_df = pd.DataFrame({
        "within":  (["T1", "T2", "T3"] * 3) * 6,
        "between": (["Ctrl"] * 9 + ["TrtA"] * 9 + ["TrtB"] * 9) * 1,
        "value":   None,
    })
    # build deterministic values per cell so T3 TrtA differs strongly from Ctrl
    import numpy as np
    rng = np.random.default_rng(0)
    base = {("Ctrl", "T1"): 10, ("Ctrl", "T2"): 10, ("Ctrl", "T3"): 10,
            ("TrtA", "T1"): 10, ("TrtA", "T2"): 11, ("TrtA", "T3"): 16,
            ("TrtB", "T1"): 10, ("TrtB", "T2"): 10, ("TrtB", "T3"): 10}
    long_df["value"] = [base[(b, w)] + rng.normal(0, 0.3)
                        for b, w in zip(long_df["between"], long_df["within"])]

    pairwise = [
        {"group1": "Ctrl:T3", "group2": "TrtA:T3", "p_value": 0.0001, "significant": True},
        {"group1": "Ctrl:T3", "group2": "TrtB:T3", "p_value": 0.9,    "significant": False},
    ]
    label_map = {"Ctrl:T3": ("Ctrl", "T3"), "TrtA:T3": ("TrtA", "T3"),
                 "TrtB:T3": ("TrtB", "T3")}

    ax = DataVisualizer.plot_grouped_bar(
        long_df=long_df, within="within", between="between", value="value",
        within_order=["T1", "T2", "T3"], between_order=["Ctrl", "TrtA", "TrtB"],
        pairwise_results=pairwise, label_map=label_map, ax=None, save_plot=False,
    )
    # 9 bars rendered
    bars = [p for p in ax.patches if (p.get_width() or 0) > 0]
    assert len(bars) == 9
    # at least one bracket line drawn (Line2D added beyond the bars)
    assert len(ax.lines) >= 3   # one significant bracket = 3 line segments
```

- [ ] **Step 2: Run it, verify it FAILS**

Run: `python -m pytest tests/test_grouped_bar_brackets.py::test_plot_grouped_bar_renders_bars_and_significant_brackets -v`
Expected: FAIL — no `plot_grouped_bar`.

- [ ] **Step 3: Implement `plot_grouped_bar`**

```python
    @staticmethod
    def plot_grouped_bar(long_df, within, between, value,
                         within_order, between_order,
                         pairwise_results=None, label_map=None,
                         width=8, height=6, dpi=300,
                         color_palette="Greys", error_type="sd",
                         show_error_bars=True, p_value_style="Fixed stars",
                         bracket_line_width=1.5, bracket_vertical_fraction=0.05,
                         bracket_color="#000000", comparison_line_height=0.1,
                         comparison_font_size=14,
                         x_label=None, y_label=None, title=None,
                         save_plot=True, file_name=None, file_formats=("png", "svg"),
                         ax=None):
        """Grouped bar plot (x=within factor, hue=between factor) with
        within-stratum treatment-vs-control significance brackets.

        long_df: long-form data with columns [within, between, value].
        label_map: {"<between>:<within>": (between_level, within_level)} so the
        pairwise comparison group labels resolve to grouped-bar cells.
        Returns the matplotlib Axes.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        if ax is None:
            fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
        else:
            fig = ax.figure

        colors = sns.color_palette(color_palette, len(between_order))
        sns.barplot(
            data=long_df, x=within, y=value, hue=between,
            order=within_order, hue_order=between_order,
            errorbar=(error_type if show_error_bars else None),
            palette=colors, ax=ax,
        )

        if pairwise_results and label_map:
            centers = DataVisualizer._grouped_bar_centers(
                ax, between_order=between_order, within_order=within_order)
            y_max = DataVisualizer._get_plot_max_height_robust(ax, None)
            brackets = DataVisualizer._grouped_bracket_positions(
                centers, label_map, pairwise_results, y_max, comparison_line_height)
            for bracket in brackets:
                DataVisualizer._draw_single_bracket(
                    ax, bracket, bracket_line_width, comparison_font_size,
                    bracket_color, bracket_vertical_fraction, p_value_style)

        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)
        if title:
            ax.set_title(title)

        if save_plot and file_name:
            for ext in file_formats:
                fig.savefig(f"{file_name}.{ext}", dpi=dpi, bbox_inches="tight")
        return ax
```

- [ ] **Step 4: Run it, verify it PASSES**

Run: `python -m pytest tests/test_grouped_bar_brackets.py -v`
Expected: PASS (all three tests)

- [ ] **Step 5: Commit**

```bash
git add src/visualization/datavisualizer.py tests/test_grouped_bar_brackets.py
git commit -m "feat(viz): plot_grouped_bar with stratified vs-control brackets"
```

---

## Task 4: Build the long-form data + label map from an EMM result (adapter)

**Files:**
- Modify: `src/visualization/datavisualizer.py` (add `grouped_inputs_from_samples` helper)
- Test: `tests/test_grouped_bar_brackets.py`

This converts the flat `groups`/`samples` (interaction-cell labels like `"Ctrl:T1"`) that the plot pipeline already produces into the long-form df + orders + label_map that `plot_grouped_bar` needs. The cell label convention is `"<between>:<within>"` (confirmed in MixedAnovaPostHocAnalyzer, which builds `f"{between_level}:{within_level}"`).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_grouped_bar_brackets.py
def test_grouped_inputs_from_samples_splits_interaction_labels():
    samples = {
        "Ctrl:T1": [1.0, 1.2], "TrtA:T1": [1.4, 1.6],
        "Ctrl:T2": [2.0, 2.2], "TrtA:T2": [2.6, 2.8],
    }
    long_df, within_order, between_order, label_map = \
        DataVisualizer.grouped_inputs_from_samples(samples, sep=":")
    assert set(long_df.columns) == {"within", "between", "value"}
    assert len(long_df) == 8
    assert between_order == ["Ctrl", "TrtA"]   # first-seen order
    assert within_order == ["T1", "T2"]
    assert label_map["Ctrl:T2"] == ("Ctrl", "T2")
```

- [ ] **Step 2: Run it, verify it FAILS**

Run: `python -m pytest tests/test_grouped_bar_brackets.py::test_grouped_inputs_from_samples_splits_interaction_labels -v`
Expected: FAIL — no `grouped_inputs_from_samples`.

- [ ] **Step 3: Implement `grouped_inputs_from_samples`**

```python
    @staticmethod
    def grouped_inputs_from_samples(samples, sep=":"):
        """Split flat interaction-cell samples ({"<between><sep><within>": [vals]})
        into the long-form inputs plot_grouped_bar needs. Preserves first-seen
        order for both factors. Raises ValueError if any key lacks the separator.
        """
        import pandas as pd

        rows = []
        between_order, within_order, label_map = [], [], {}
        for label, vals in samples.items():
            if sep not in label:
                raise ValueError(f"cell label {label!r} has no {sep!r} separator")
            b_lev, w_lev = label.split(sep, 1)
            label_map[label] = (b_lev, w_lev)
            if b_lev not in between_order:
                between_order.append(b_lev)
            if w_lev not in within_order:
                within_order.append(w_lev)
            for v in vals:
                rows.append({"within": w_lev, "between": b_lev, "value": v})
        return pd.DataFrame(rows, columns=["within", "between", "value"]), \
            within_order, between_order, label_map
```

- [ ] **Step 4: Run it, verify it PASSES**

Run: `python -m pytest tests/test_grouped_bar_brackets.py -v`
Expected: PASS (all four tests)

- [ ] **Step 5: Commit**

```bash
git add src/visualization/datavisualizer.py tests/test_grouped_bar_brackets.py
git commit -m "feat(viz): adapter from interaction-cell samples to grouped-bar inputs"
```

---

## Task 5: Route the control-referenced Mixed result to the grouped plot

**Files:**
- Modify: the plot caller in the autopilot pipeline. FIRST locate it:
  run `grep -rn "plot_bar(" src/ | grep -vi test` and `grep -rn "configure_plot_from_result\|def .*configure_plot" src/`. The caller that turns a result + `samples`/`groups` into a `plot_bar(...)` call is the integration point.
- Test: none automated (UI plot dispatch; covered by Task 1-4 units + manual QA). Keep the change minimal and guard-railed.

- [ ] **Step 1: Identify the dispatch and the design signal**

Read the located caller. Establish: (a) how it currently calls `plot_bar`, (b) whether the result/context exposes that this is a Mixed design with a control group and the post-hoc is `emm_mvt` / `posthoc_test` contains "EMM" / "Dunnett-type", and (c) where `samples` (interaction-cell keyed) is available.

- [ ] **Step 2: Add the grouped dispatch (minimal, guarded)**

At the dispatch, before the existing `plot_bar(...)` call, add a guarded branch. Use the real variable names found in Step 1; the shape is:

```python
        _ph = str(result.get("posthoc_test") or "")
        _is_mixed_vs_control = (
            "mixed" in str(result.get("test", "")).lower()
            and ("emm" in _ph.lower() or "dunnett-type" in _ph.lower())
            and all(":" in g for g in groups)          # interaction-cell labels
        )
        if _is_mixed_vs_control:
            try:
                long_df, w_order, b_order, label_map = \
                    DataVisualizer.grouped_inputs_from_samples(samples, sep=":")
                DataVisualizer.plot_grouped_bar(
                    long_df=long_df, within="within", between="between", value="value",
                    within_order=w_order, between_order=b_order,
                    pairwise_results=pairwise_results, label_map=label_map,
                    x_label=x_label, y_label=y_label, title=title,
                    file_name=file_name, save_plot=save_plot, ax=ax,
                )
                return   # grouped path handled the plot
            except Exception as exc:
                logger.warning("grouped plot failed (%s); using flat plot_bar", exc)
        # ... existing flat plot_bar(...) call stays as the fallback
```

Match the surrounding function's actual parameter names (`x_label`, `file_name`, `save_plot`, `ax`, `pairwise_results`, `groups`, `samples`, `result`) — adjust to what Step 1 found. Do not remove the existing `plot_bar` call; it remains the fallback.

- [ ] **Step 3: Verify import + full suite**

Run: `python -c "import sys; sys.path.insert(0,'src'); import visualization.datavisualizer"`  → exit 0
Run: `python -m pytest tests/ validation/ -q`  → all pass.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat(viz): dispatch Mixed EMM/Dunnett result to grouped bar plot"
```

---

## Manual QA (run the app: `python src/analysis/statistical_analyzer.py`)

- [ ] Run a balanced Mixed ANOVA, significant, pick "Dunnett vs control, each timepoint (EMM + multivariate-t)".
- [ ] The plot shows **grouped bars** — x axis is the within factor (e.g. Time), bars dodged by treatment group (hue), with a legend.
- [ ] Significance brackets appear **within each timepoint cluster**, only between a treatment and the control bar of the SAME timepoint. No bracket spans two different timepoints.
- [ ] Non-significant contrasts draw no bracket (or "n.s." per the chosen p-value style); multiple significant treatments in one stratum stack without overlap.
- [ ] A one-way ANOVA plot is unchanged (still flat `plot_bar`) — confirms no regression.

---

## Out of scope (future)

- RM-only (within-only, no between factor) grouped rendering — no hue factor; separate milestone.
- Per-hue hatches/point-overlay (jitter/swarm) and full legend styling parity with flat `plot_bar` — add only if requested; the core deliverable is grouped bars + correct brackets.
- Two-Way ANOVA grouped dispatch — the same `plot_grouped_bar` applies once a two-way EMM/Dunnett result exists (tracked with the two-way EMM follow-up).
