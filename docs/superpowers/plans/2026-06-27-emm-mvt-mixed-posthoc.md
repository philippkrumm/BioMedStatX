# EMM + multivariate-t Post-hoc for Mixed ANOVA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a peer-review-grade post-hoc test for the parametric Mixed ANOVA path: estimated-marginal-means treatment-vs-control contrasts at each within level, with a pooled error term and a simultaneous multivariate-t correction, matching R `afex::aov_ez` + `emmeans(adjust="mvt")`.

**Architecture:** A new, dependency-light numeric module (`src/analysis/emm_posthoc.py`) computes the contrasts from the balanced split-plot ANOVA error strata in closed form (no model refit, no R at runtime). It is validated against the frozen golden reference. A thin adapter then exposes it through the existing post-hoc result structure, a new dialog option, and the advanced post-hoc routing, while the existing isolated-t-test analyzer stays as the fallback for designs the closed form does not support (unbalanced / missing cells).

**Tech Stack:** Python, numpy, pandas, scipy.stats (`multivariate_t`, `t`, `studentized_range` not needed here), PyQt5 (dialog only). Validation ground truth: R 4.5.3 / afex 1.5.1 / emmeans 2.0.3 (already committed under `tests/golden/`).

---

## Background: the verified math (already reproduced against the golden)

For a **balanced** split-plot design (one between factor with `G` groups, one within factor with `W` levels, `n` subjects per group, complete cells), the emmeans `model="univariate"` contrast "treatment group minus control group at a fixed within level" has these closed forms. All were verified against `tests/golden/references_mixed_dunnett_emmeans.json` to: estimate exact, SE exact, df exact, mvt p ≤ 2.5e-4.

Error strata (classical split-plot):
- Between-subjects error `MS_sg` = SS over subjects of `W * (subject_mean - group_mean)^2`, divided by `df_sg = G*(n-1)`.
- Within-subjects (residual) error `MS_res` = SS over rows of `(value - subject_mean - cell_mean + group_mean)^2`, divided by `df_res = G*(n-1)*(W-1)`.

Variance components: `sig_w2 = MS_res`, `sig_s2 = (MS_sg - MS_res) / W`.

Per-observation variance at a fixed within level: `var_t = sig_s2 + sig_w2 = (MS_sg + (W-1)*MS_res) / W`.

Contrast (treatment - control at one within level):
- `estimate = cell_mean[treatment, level] - cell_mean[control, level]`
- `SE = sqrt(2 * var_t / n)` (constant across all contrasts)
- Satterthwaite `df = (c1*MS_sg + c2*MS_res)^2 / ((c1*MS_sg)^2/df_sg + (c2*MS_res)^2/df_res)` with `c1 = 1/W`, `c2 = (W-1)/W`.
- `t = estimate / SE`.

Simultaneous mvt p-value: the adjustment is applied **within each within level** (the `by = within` family), so the family size is `k = G-1` (one contrast per treatment, all sharing the control). The contrast correlation for a balanced design is `0.5` between any two contrasts (they share the control cell, treatments are mutually independent with equal variance). For `k >= 2`:
`p_adj = 1 - P(all |T_j| <= |t_i|)`, `T ~ multivariate_t(loc=0, shape=R, df=df)`, computed with `scipy.stats.multivariate_t(...).cdf([c]*k, lower_limit=[-c]*k)` where `c = |t_i|`.
For `k == 1` (single treatment, G==2) there is no multiplicity, so `p_adj = 2 * scipy.stats.t.sf(|t|, df)`.

---

## File Structure

- Create `src/analysis/emm_posthoc.py` — the numeric engine. Responsibilities: split-plot strata, variance components, contrast estimate/SE/df, mvt adjustment, and one public entry `mixed_dunnett_emm_mvt(...)` returning a list of contrast dicts. No PyQt, no posthoc_core imports (keeps it unit-testable and reusable).
- Create `tests/test_emm_mvt_posthoc.py` — validation against the golden JSON plus unit tests for the building blocks and the balance guard.
- Modify `src/analysis/posthoc_core.py` — add a `MixedAnovaPostHocAnalyzer` branch that calls the engine and maps to the standard result structure, used when `method == "emm_mvt"`; fall back to the existing isolated-t path if the engine raises `UnsupportedDesignError`.
- Modify `src/analysis/stats_functions.py` — add the "Dunnett (vs control, EMM + multivariate-t)" radio option to `select_posthoc_test_dialog` for the advanced-ANOVA branch.
- Modify `src/statistical_testing/engines/advanced_posthoc.py` — when `posthoc_method == "emm_mvt"`, resolve the control group and pass it through (mirror the existing `dunnett` handling).

The RM-only (no between factor) design uses a different contrast family (level-vs-baseline) and a different error structure; it is **out of scope for this plan** and will get its own golden + plan.

---

## Task 1: Engine scaffold and the balance/strata computation

**Files:**
- Create: `src/analysis/emm_posthoc.py`
- Test: `tests/test_emm_mvt_posthoc.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_emm_mvt_posthoc.py
import json
import os
import pandas as pd
import pytest
from analysis.emm_posthoc import split_plot_strata, UnsupportedDesignError

_DATA = os.path.join("tests", "golden", "mixed_dunnett_emmeans_dataset.csv")


def _df():
    return pd.read_csv(_DATA)


def test_split_plot_strata_matches_classical_values():
    s = split_plot_strata(_df(), dv="Value", subject="Subject",
                          between="Group", within="Time")
    assert s.n_per_group == 8
    assert s.W == 3 and s.G == 3
    assert s.df_sg == 21 and s.df_res == 42
    assert s.ms_sg == pytest.approx(4.559091, abs=1e-4)
    assert s.ms_res == pytest.approx(0.714811, abs=1e-4)


def test_unbalanced_design_raises():
    df = _df()
    df = df[~((df.Subject == "S01") & (df.Time == "T3"))]  # missing cell
    with pytest.raises(UnsupportedDesignError):
        split_plot_strata(df, dv="Value", subject="Subject",
                          between="Group", within="Time")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_emm_mvt_posthoc.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'analysis.emm_posthoc'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/analysis/emm_posthoc.py
"""Estimated-marginal-means + multivariate-t post-hoc for balanced split-plot
(Mixed ANOVA) designs. Closed-form reproduction of R afex::aov_ez +
emmeans(model="univariate") + contrast(trt.vs.ctrl, adjust="mvt").

Pure numeric (numpy/pandas/scipy); no PyQt, no model refit, no R at runtime.
Only balanced, complete designs are supported; callers must fall back to the
isolated-t-test path on UnsupportedDesignError.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class UnsupportedDesignError(ValueError):
    """Raised when the design is not a balanced, complete split-plot."""


@dataclass
class SplitPlotStrata:
    G: int
    W: int
    n_per_group: int
    df_sg: int
    df_res: int
    ms_sg: float
    ms_res: float
    cell_means: pd.DataFrame          # index=between level, columns=within level
    group_levels: list
    within_levels: list


def split_plot_strata(df: pd.DataFrame, dv: str, subject: str,
                      between: str, within: str) -> SplitPlotStrata:
    data = df[[subject, between, within, dv]].dropna()
    group_levels = list(pd.unique(data[between]))
    within_levels = list(pd.unique(data[within]))
    G, W = len(group_levels), len(within_levels)

    # Each subject must appear exactly once per within level, in one group.
    counts = data.groupby([subject, within]).size()
    if not (counts == 1).all():
        raise UnsupportedDesignError("subjects must have exactly one row per within level")
    subj_group = data.groupby(subject)[between].nunique()
    if not (subj_group == 1).all():
        raise UnsupportedDesignError("each subject must belong to one between group")

    subjects_per_group = data.groupby(between)[subject].nunique()
    if subjects_per_group.nunique() != 1:
        raise UnsupportedDesignError("unbalanced: groups differ in subject count")
    n = int(subjects_per_group.iloc[0])

    if len(data) != G * W * n:
        raise UnsupportedDesignError("incomplete cells in the design")

    grand = data[dv].mean()
    group_mean = data.groupby(between)[dv].mean()
    subj_mean = data.groupby(subject)[dv].mean()
    subj_to_group = data.groupby(subject)[between].first()
    cell_means = data.groupby([between, within])[dv].mean().unstack()

    ss_sg = 0.0
    for subj, m in subj_mean.items():
        ss_sg += W * (m - group_mean[subj_to_group[subj]]) ** 2
    df_sg = G * (n - 1)
    ms_sg = ss_sg / df_sg

    ss_res = 0.0
    for _, r in data.iterrows():
        e = (r[dv] - subj_mean[r[subject]]
             - cell_means.loc[r[between], r[within]] + group_mean[r[between]])
        ss_res += e * e
    df_res = G * (n - 1) * (W - 1)
    ms_res = ss_res / df_res

    return SplitPlotStrata(G=G, W=W, n_per_group=n, df_sg=df_sg, df_res=df_res,
                           ms_sg=ms_sg, ms_res=ms_res, cell_means=cell_means,
                           group_levels=group_levels, within_levels=within_levels)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_emm_mvt_posthoc.py -v`
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add src/analysis/emm_posthoc.py tests/test_emm_mvt_posthoc.py
git commit -m "feat(stats/emm): split-plot strata with balance guard"
```

---

## Task 2: Variance components, contrast SE, and Satterthwaite df

**Files:**
- Modify: `src/analysis/emm_posthoc.py`
- Test: `tests/test_emm_mvt_posthoc.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_emm_mvt_posthoc.py
from analysis.emm_posthoc import contrast_se_df


def test_contrast_se_and_satterthwaite_df_match_golden():
    s = split_plot_strata(_df(), dv="Value", subject="Subject",
                          between="Group", within="Time")
    se, df = contrast_se_df(s)
    assert se == pytest.approx(0.706441, abs=1e-5)
    assert df == pytest.approx(34.5371, abs=1e-3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_emm_mvt_posthoc.py::test_contrast_se_and_satterthwaite_df_match_golden -v`
Expected: FAIL with `ImportError: cannot import name 'contrast_se_df'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/analysis/emm_posthoc.py

def variance_components(s: SplitPlotStrata) -> tuple[float, float]:
    """Return (sig_s2, sig_w2): between-subject and within-subject variances."""
    sig_w2 = s.ms_res
    sig_s2 = (s.ms_sg - s.ms_res) / s.W
    return sig_s2, sig_w2


def contrast_se_df(s: SplitPlotStrata) -> tuple[float, float]:
    """Constant contrast SE and the Satterthwaite df for a between-group
    contrast at a fixed within level."""
    sig_s2, sig_w2 = variance_components(s)
    var_t = sig_s2 + sig_w2                      # per-obs variance at a fixed level
    se = float(np.sqrt(2.0 * var_t / s.n_per_group))

    c1 = 1.0 / s.W
    c2 = (s.W - 1.0) / s.W
    num = (c1 * s.ms_sg + c2 * s.ms_res) ** 2
    den = (c1 * s.ms_sg) ** 2 / s.df_sg + (c2 * s.ms_res) ** 2 / s.df_res
    df = float(num / den)
    return se, df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_emm_mvt_posthoc.py::test_contrast_se_and_satterthwaite_df_match_golden -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/analysis/emm_posthoc.py tests/test_emm_mvt_posthoc.py
git commit -m "feat(stats/emm): pooled contrast SE and Satterthwaite df"
```

---

## Task 3: Public entry — full contrast table with mvt p-values matching the golden

**Files:**
- Modify: `src/analysis/emm_posthoc.py`
- Test: `tests/test_emm_mvt_posthoc.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_emm_mvt_posthoc.py
from analysis.emm_posthoc import mixed_dunnett_emm_mvt

_GOLD = os.path.join("tests", "golden", "references_mixed_dunnett_emmeans.json")


def test_mixed_dunnett_emm_mvt_matches_emmeans_golden():
    gold = {(c["Time"], c["contrast"]): c for c in json.load(open(_GOLD))["contrasts"]}
    out = mixed_dunnett_emm_mvt(_df(), dv="Value", subject="Subject",
                                between="Group", within="Time",
                                control_group="Ctrl", alpha=0.05)
    assert len(out) == len(gold)
    for c in out:
        key = (c["within_level"], f"{c['treatment']} - Ctrl")
        g = gold[key]
        assert c["estimate"] == pytest.approx(g["estimate"], abs=1e-4)
        assert c["se"] == pytest.approx(g["SE"], abs=1e-4)
        assert c["df"] == pytest.approx(g["df"], abs=1e-2)
        assert c["t"] == pytest.approx(g["t_ratio"], abs=1e-3)
        assert c["p_value"] == pytest.approx(g["p_value"], abs=2e-3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_emm_mvt_posthoc.py::test_mixed_dunnett_emm_mvt_matches_emmeans_golden -v`
Expected: FAIL with `ImportError: cannot import name 'mixed_dunnett_emm_mvt'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/analysis/emm_posthoc.py
from scipy.stats import multivariate_t, t as student_t


def _mvt_adjusted_p(t_values: list[float], df: float) -> list[float]:
    """Single-step two-sided multivariate-t adjusted p-values for a family that
    shares one control (balanced => equicorrelation 0.5). Family size k = len."""
    k = len(t_values)
    if k == 1:
        return [float(2.0 * student_t.sf(abs(t_values[0]), df))]
    R = np.full((k, k), 0.5)
    np.fill_diagonal(R, 1.0)
    rv = multivariate_t(loc=np.zeros(k), shape=R, df=df, allow_singular=True)
    out = []
    for tv in t_values:
        c = abs(float(tv))
        p_all = float(rv.cdf(np.full(k, c), lower_limit=np.full(k, -c)))
        out.append(float(min(1.0, max(0.0, 1.0 - p_all))))
    return out


def mixed_dunnett_emm_mvt(df: pd.DataFrame, dv: str, subject: str, between: str,
                          within: str, control_group, alpha: float = 0.05) -> list[dict]:
    """Treatment-vs-control EMM contrasts at each within level, mvt-adjusted
    within each within level. Raises UnsupportedDesignError for designs the
    closed form does not cover (callers must then fall back to isolated t-tests).
    """
    s = split_plot_strata(df, dv=dv, subject=subject, between=between, within=within)
    if control_group not in s.group_levels:
        raise UnsupportedDesignError(f"control group {control_group!r} not present")

    se, ddf = contrast_se_df(s)
    treatments = [g for g in s.group_levels if g != control_group]

    results: list[dict] = []
    for level in s.within_levels:
        t_values, rows = [], []
        for trt in treatments:
            est = float(s.cell_means.loc[trt, level] - s.cell_means.loc[control_group, level])
            tval = est / se
            t_values.append(tval)
            rows.append({"within_level": level, "treatment": trt,
                         "control": control_group, "estimate": est,
                         "se": se, "df": ddf, "t": tval})
        p_adj = _mvt_adjusted_p(t_values, ddf)
        for row, p in zip(rows, p_adj):
            row["p_value"] = p
            row["significant"] = bool(p < alpha)
            results.append(row)
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_emm_mvt_posthoc.py::test_mixed_dunnett_emm_mvt_matches_emmeans_golden -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/analysis/emm_posthoc.py tests/test_emm_mvt_posthoc.py
git commit -m "feat(stats/emm): EMM treatment-vs-control with mvt p-values (matches emmeans golden)"
```

---

## Task 4: Map the engine into the standard post-hoc result structure

**Files:**
- Modify: `src/analysis/posthoc_core.py` (the `MixedAnovaPostHocAnalyzer.perform_test` correction dispatch — the `elif method.lower() == 'dunnett'` block region, ~lines 528-548)
- Test: `tests/test_emm_mvt_posthoc.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_emm_mvt_posthoc.py
from analysis.posthoc_core import MixedAnovaPostHocAnalyzer


def test_analyzer_emm_mvt_method_produces_significant_t3_trta():
    analyzer = MixedAnovaPostHocAnalyzer()
    result = analyzer.perform_test(
        _df(), dv="Value", between_factor="Group", within_factor="Time",
        subject="Subject", alpha=0.05, method="emm_mvt", control_group="Ctrl",
    )
    pairs = result["pairwise_comparisons"]
    hit = [c for c in pairs if "TrtA" in (c["group1"] + c["group2"])
           and "T3" in (c["group1"] + c["group2"])]
    assert hit and hit[0]["significant"] is True
    assert "EMM" in result["posthoc_test"] or "multivariate-t" in result["posthoc_test"]
```

Note: confirm the exact `perform_test` signature in `posthoc_core.py` before writing the call (parameter names for between/within/subject may differ — match them). If the signature differs, adapt the test call and the implementation to the real names.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_emm_mvt_posthoc.py::test_analyzer_emm_mvt_method_produces_significant_t3_trta -v`
Expected: FAIL (the `emm_mvt` method is not handled; result lacks the EMM contrasts)

- [ ] **Step 3: Write minimal implementation**

In `MixedAnovaPostHocAnalyzer.perform_test`, before the existing comparison loop builds `comparisons`, add an early dispatch for the EMM method. Insert near the top of the method body:

```python
# inside MixedAnovaPostHocAnalyzer.perform_test, after parsing dv/between/within/subject/alpha
if method.lower() == "emm_mvt":
    from analysis.emm_posthoc import mixed_dunnett_emm_mvt, UnsupportedDesignError
    try:
        contrasts = mixed_dunnett_emm_mvt(
            df, dv=dv, subject=subject, between=between_factor,
            within=within_factor, control_group=control_group, alpha=alpha,
        )
    except UnsupportedDesignError as exc:
        logger.warning("EMM/mvt unavailable (%s); falling back to isolated t-tests", exc)
    else:
        result = {
            "posthoc_test": "Dunnett-type (EMM + multivariate-t, Mixed)",
            "pairwise_comparisons": [],
            "test_type": "EMM contrast (pooled error, mvt-adjusted)",
        }
        for c in contrasts:
            PostHocAnalyzer.add_comparison(
                result,
                group1=f"{c['control']}, {c['within_level']}",
                group2=f"{c['treatment']}, {c['within_level']}",
                p_value=c["p_value"],
                test_statistic=c["t"],
                effect_size=None,
                ci_lower=None,
                ci_upper=None,
                significant=c["significant"],
            )
        return result
# else: fall through to the existing isolated-t-test paths
```

Confirm `PostHocAnalyzer.add_comparison`'s real keyword names in `posthoc_core.py` and adjust the call to match (the names above mirror the existing usage in this file; if they differ, use the real ones). Keep the existing `dunnett`/`tukey`/`paired_custom` branches untouched.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_emm_mvt_posthoc.py::test_analyzer_emm_mvt_method_produces_significant_t3_trta -v`
Expected: PASS

- [ ] **Step 5: Run the whole post-hoc suite to confirm no regression**

Run: `python -m pytest tests/test_posthoc_correctness.py tests/test_emm_mvt_posthoc.py tests/test_golden_r_advanced.py -q`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/analysis/posthoc_core.py tests/test_emm_mvt_posthoc.py
git commit -m "feat(stats/posthoc): EMM/mvt method in MixedAnovaPostHocAnalyzer with isolated-t fallback"
```

---

## Task 5: Offer the EMM option in the post-hoc dialog (advanced ANOVA branch)

**Files:**
- Modify: `src/analysis/stats_functions.py` (the advanced-ANOVA options list in `select_posthoc_test_dialog`, ~lines 506-510)
- Test: none (UI dialog; covered by manual QA below)

- [ ] **Step 1: Add the radio option**

In `select_posthoc_test_dialog`, the advanced branch currently is:

```python
        if progress_text and ("two_way_anova" in progress_text or "mixed_anova" in progress_text or "repeated_measures_anova" in progress_text):
            options = [
                ("Tukey-HSD Test (all pairs, strict FWER control)", "tukey"),
                ("Specific comparisons – strict correction (Holm-Šidák)", "paired_custom"),
            ]
```

Change it to offer EMM/mvt for mixed designs only (the closed form is split-plot specific):

```python
        if progress_text and ("two_way_anova" in progress_text or "mixed_anova" in progress_text or "repeated_measures_anova" in progress_text):
            options = [
                ("Tukey-HSD Test (all pairs, strict FWER control)", "tukey"),
                ("Specific comparisons – strict correction (Holm-Šidák)", "paired_custom"),
            ]
            if "mixed_anova" in progress_text:
                options.append(
                    ("Dunnett vs control, each timepoint (EMM + multivariate-t)", "emm_mvt")
                )
```

- [ ] **Step 2: Verify the module still imports**

Run: `python -c "import sys; sys.path.insert(0,'src'); import analysis.stats_functions"`
Expected: no error (exit 0)

- [ ] **Step 3: Commit**

```bash
git add src/analysis/stats_functions.py
git commit -m "feat(ui): offer EMM+multivariate-t Dunnett option for Mixed ANOVA post-hoc"
```

---

## Task 6: Route the `emm_mvt` method and control group through the advanced engine

**Files:**
- Modify: `src/statistical_testing/engines/advanced_posthoc.py` (~lines 93-103, the method/control/selected-comparisons block)
- Test: none (integration covered by Task 4 unit + manual QA)

- [ ] **Step 1: Resolve the control group for `emm_mvt`**

The block that resolves the control group only triggers for `posthoc_method == "dunnett"`. Extend it so `emm_mvt` also resolves a control group, and so its selected comparisons are control-vs-treatment:

Change:

```python
                if posthoc_method == "dunnett":
                    if control_group_callback:
                        control_group = control_group_callback(group_names)
                    elif group_names:
                        control_group = group_names[0]
```

to:

```python
                if posthoc_method in ("dunnett", "emm_mvt"):
                    if control_group_callback:
                        control_group = control_group_callback(group_names)
                    elif group_names:
                        control_group = group_names[0]
```

And in the selected-comparisons block, change:

```python
            if posthoc_method == "dunnett" and control_group:
                selected_comparisons = [(control_group, group) for group in group_names if group != control_group]
```

to:

```python
            if posthoc_method in ("dunnett", "emm_mvt") and control_group:
                selected_comparisons = [(control_group, group) for group in group_names if group != control_group]
```

Note: the EMM engine derives its own contrasts from the raw data (it does not consume `selected_comparisons`), but resolving `control_group` here is what makes the analyzer call in Task 4 receive a control group via `posthoc_kwargs`.

- [ ] **Step 2: Verify the module still imports**

Run: `python -c "import sys; sys.path.insert(0,'src'); import statistical_testing.engines.advanced_posthoc"`
Expected: no error (exit 0)

- [ ] **Step 3: Run the full suite**

Run: `python -m pytest tests/ validation/ -q`
Expected: all PASS (the 398 existing + the new EMM tests)

- [ ] **Step 4: Commit**

```bash
git add src/statistical_testing/engines/advanced_posthoc.py
git commit -m "feat(stats/posthoc): route emm_mvt method + control group through advanced engine"
```

---

## Task 7: Documentation note

**Files:**
- Modify: the post-hoc section of the user-facing help (search for the existing post-hoc help text: `grep -rin "Games-Howell\|post-hoc" src/analysis/*.py src/**/help*` to find where advanced post-hoc methods are described)

- [ ] **Step 1: Add a short description**

Add one paragraph where the other advanced post-hoc methods are documented, stating: for Mixed ANOVA, the "Dunnett vs control (EMM + multivariate-t)" option compares each treatment group to the control group at each within level, using the pooled split-plot error term and a simultaneous multivariate-t correction within each within level; it matches R's `emmeans`/`afex`. It requires a balanced, complete design; otherwise the tool falls back to corrected pairwise t-tests.

- [ ] **Step 2: Commit**

```bash
git add -A
git commit -m "docs: describe EMM+multivariate-t Mixed ANOVA post-hoc option"
```

---

## Manual QA (run the app: `python src/analysis/statistical_analyzer.py`)

- [ ] Load a balanced mixed dataset (one between factor incl. a control level, one within factor). Map it, run the Mixed ANOVA, make it significant.
- [ ] In the post-hoc dialog the new option "Dunnett vs control, each timepoint (EMM + multivariate-t)" appears. Selecting it asks for the control group.
- [ ] Results show one row per treatment per within level, with a constant SE and the mvt-adjusted p-values; the report labels the test "EMM + multivariate-t".
- [ ] Load an unbalanced / missing-cell mixed dataset; pick the EMM option; confirm the tool falls back to the corrected pairwise t-tests without crashing (a warning is logged).

---

## Out of scope (future plans)

- Repeated-measures (within-only) EMM Dunnett (level-vs-baseline) — needs its own golden and within-subject error structure.
- Two-way (between-only) EMM Dunnett — `scipy.stats.dunnett` already covers the one-way independent case; a two-way EMM extension is separate.
- Unbalanced / missing-cell EMM via a fitted mixed model with Kenward-Roger df — large, separate effort; the balance guard + isolated-t fallback covers these safely for now.
