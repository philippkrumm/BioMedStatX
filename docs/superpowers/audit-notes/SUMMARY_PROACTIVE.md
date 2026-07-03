# Proactive anti-pattern audit: collated findings

Date: 2026-07-03
Scope: Data-Parser (wide-format detection + pivot, `src/autopilot/statistical_analyzer_autopilot_ui.py`,
`src/autopilot/statistical_analyzer_autopilot_pipeline.py`), Visualization
(`src/visualization/datavisualizer.py`), Report generation
(`src/export/report_summaries.py`, `src/export/report_stat_rows.py`).

Trigger: after the 2026-07-02/03 Help Hub content audit surfaced several code bugs
(missing pre-flight validation, string-coupled dispatch, silent fallbacks — see
`SUMMARY.md`), the user classified those bugs into four systemic anti-pattern
classes and asked for a proactive hunt for the same classes in three modules that
share the "bridge between UI and math core" role. Three parallel Explore-agent
audits ran (data-parser, visualization, report generation); this document collates
their 7 surviving findings after independent verification of every citation
against current source (agent-reported line numbers were re-read, not trusted
as-is — one finding's severity was corrected during verification, see item 1).

None of these were fixed as part of this audit; scope was detection + triage only.

## Anti-pattern classes (for reference)

1. **Defensive Validation Deficits** — missing pre-flight checks; downstream
   library (patsy/statsmodels/pandas) throws an opaque error instead.
2. **Fault Swallowing & Silent Fallbacks** — broad `except`/`else` masks a failure
   and substitutes a plausible-but-wrong default instead of surfacing it.
3. **String-Coupling & State Desync** — backend logic gated on volatile string
   labels/dict keys instead of stable identifiers; renames break things silently.
4. **Implicit Logic Violations** — syntactically valid code that breaks the
   statistical/mathematical intent (precedence, missing type-gating).

---

## Kategorie A — Mechanische Fixes (eindeutiger Zielzustand, keine Product Decision)

### A1. Sphericity correction note never shows epsilon; export reads the wrong dict entirely — **verified currently active, not a rename risk**

Class: 3 (String-Coupling / key mismatch), also 2 (silently produces an incomplete
result instead of erroring).

Backend writes correction info to **two places**, neither of which is where the
export looks:

- `results["correction_used"]` — top-level key, e.g. `"Greenhouse-Geisser (ε = 0.456)"`.
  GG is applied **unconditionally** when available (comment says so explicitly):

  ```python
  # src/analysis/statisticaltester.py:2856-2861 (_apply_sphericity_corrections)
  if gg_epsilon is not None and hf_epsilon is not None:
      # Use Greenhouse-Geisser unconditionally (as requested by user / conservative default)
      corrections["corrected_p_value"] = gg_p_value
      corrections["correction_used"] = f"Greenhouse-Geisser (ε = {gg_epsilon:.3f})"
  ```

- Epsilon value itself lives nested under `results["sphericity_corrections"]["greenhouse_geisser"]["epsilon"]`:

  ```python
  # src/analysis/statisticaltester.py:2826-2833
  corrections["sphericity_corrections"]["greenhouse_geisser"] = {
      "epsilon": gg_epsilon,
      "corrected_df1": float(row["DF"]) * gg_epsilon,
      ...
  }
  ```

- `results["sphericity_test"]` (built at `statisticaltester.py:2624-2703`) contains
  only `{test_name, W, chi_square, d, p_value, sphericity_assumed, interpretation}` —
  **no `correction`, `correction_applied`, `greenhouse_geisser`, `gg_epsilon`, or
  `epsilon_gg` key is ever written into this sub-dict.**

Export reads exclusively from that empty sub-dict:

```python
# src/export/report_summaries.py:437, 450-464 (_build_assumption_summary)
sphericity = results.get("sphericity_test", {}) or {}
...
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

**Runtime consequence, every single time:** `corr` is always `""`, `gg_eps`/`hf_eps`
are always `None`. Falls through to the `else` branch → label happens to read
"Greenhouse-Geisser" (correct, purely because GG is the hardcoded backend default,
not because the export found real evidence of it) — but `eps` is always `None`, so
`eps_str` is always empty. **The exported sphericity-violation note has never shown
an epsilon value, for any RM-ANOVA or Mixed-ANOVA report with a sphericity
violation.** This was reported by the audit agent as a rename-risk; verification
against the current backend confirms it is not hypothetical — the keys the export
looks for do not exist today.

Target state: point the export at the correct locations — `results["correction_used"]`
for the label, `results["sphericity_corrections"]["greenhouse_geisser"]["epsilon"]` /
`["huynh_feldt"]["epsilon"]` for the value. No ambiguity in what "correct" means;
purely a key-path correction.

### A2. RTE table renders blank group labels on any key rename — `report_stat_rows.py`

Class: 3 (String-Coupling).

```python
# src/export/report_stat_rows.py:670-671 (_build_statistical_rows, BrunnerLangerATS branch)
between = rte_row.get("between_group", "")
within = rte_row.get("within_level", "")
```

Silent `""` fallback on missing key — if the Brunner-Langer/ATS engine (in
`nonparametricanovas.py`) ever renames these two dict keys, the RTE table renders
with blank group/level labels (`"RTE:  / "`) instead of erroring. Currently the
keys match (no active bug), but the fix — replace the silent `.get(..., "")` with
an explicit lookup that raises or logs loudly on a missing key — is mechanical: no
design choice about *what* the correct keys are, only about *how loudly* a
mismatch should surface. Reachability: medium (Brunner-Langer/ATS is a
specialized/less-used test path).

### A3. All-NaN value columns pass wide-format detection silently, crash downstream with an opaque error

Class: 1 (Defensive Validation Deficit).

```python
# src/autopilot/statistical_analyzer_autopilot_ui.py:128-173 (_detect_wide_format)
numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
...
value_cols = [c for c in numeric_cols if c != subject_col]
if not (2 <= len(value_cols) <= 8):
    return None
...
unique_ratio = df[subject_col].nunique() / max(len(df), 1)
if unique_ratio < 0.8:
    return None
return {"subject_col": subject_col, "value_cols": value_cols}
```

`is_numeric_dtype` is `True` for an all-`NaN` float64 column — the function never
checks that a value column has *any* non-null data. A wide-format sheet where an
entire measurement column is empty (e.g. equipment failure at one timepoint) is
accepted, pivoted, and only fails later:

```python
# src/analysis/analysis_core.py:262
samples[group_name] = pd.to_numeric(subset[primary_dv], errors="coerce").dropna().tolist()
```

— which silently produces an **empty list** for that group, surfacing as a generic
"empty group" error deep in a test function instead of "column X has no data" at
load time. Target state: add a non-null check in `_detect_wide_format` (e.g.
`df[c].notna().any()`) and reject/flag columns that are entirely `NaN` before they
reach the pivot. No product decision — an all-empty column is never valid input;
the only question is where the error message is raised, which is a mechanical
placement.

---

## Kategorie B — Strategische Fixes / Product Decisions

### B1. Grouped-EMM plot silently degrades to a flat bar plot on any internal failure

Class: 2 (Fault Swallowing).

```python
# src/visualization/datavisualizer.py:2830-2846 (plot_from_config)
try:
    long_df, w_order, b_order, label_map = \
        DataVisualizer.grouped_inputs_from_samples(samples, sep=":")
    DataVisualizer.plot_grouped_bar(
        long_df=long_df, within="within", between="between", value="value",
        ...
    )
except Exception as exc:
    logger.warning("grouped EMM plot failed (%s); using flat plot_bar", exc)
    DataVisualizer.plot_bar(groups, samples, **bar_kwargs)
```

Triggered whenever Mixed-ANOVA EMM/multivariate-t post-hoc results carry
interaction-structured group labels (`"treatment:baseline"` etc., `sep=":"`) and
`grouped_inputs_from_samples`/`plot_grouped_bar` throws for any reason (malformed
label, dimension mismatch, unsupported param). The chart still renders — as a flat
bar plot — with no visible indication that the within/between structure was lost.
A user reading the exported figure has no way to know the grouped comparison
failed; only a `logger.warning` (not surfaced in the UI) records it.

**Design decision needed:** should a failed grouped-EMM render (a) raise and block
the plot with a clear message, (b) fall back to the flat plot but stamp a visible
"structure could not be rendered" annotation on the canvas, or (c) attempt a
narrower fallback (e.g. retry without the interaction split) before giving up?
Same category of decision as B2 below — both are "silent fallback on plot
failure," and a single design decision about how this codebase wants failed plots
to communicate could resolve both at once.

### B2. Log-scale axes silently drop non-positive data points with no warning

Class: 2 (Fault Swallowing) / 4 (Implicit Logic Violation — log of ≤0 is undefined,
but nothing gates on it).

```python
# src/visualization/datavisualizer.py:1894-1897 (_format_axes)
if logx:
    ax.set_xscale('log', base=10)
if logy:
    ax.set_yscale('log', base=10)
```

Exposed directly in the UI:

```python
# src/ui/dialogs/plot_aesthetics_dialog.py:828-831
self.logy_check = QCheckBox("Log Y (base 10)")
self.logy_check.setChecked(self.config.get('logy', False))
self.logy_check.toggled.connect(self.settingsChanged)
```

If the plotted data contains zeros, negative values, or the log-transform produces
NaN, matplotlib silently drops those points from the rendered axis — no exception,
no visible warning. The chart still looks complete and normal. Consequence: any
statistical annotation drawn on top (significance brackets, group n counts) can
reference a data count that no longer matches what's visible, because points were
silently removed by the axis transform, not by the statistical test.

**Design decision needed (user's own framing):** should the UI block/disable "Log
Y" when the current data range includes values ≤ 0, or should an explicit
in-canvas message ("N points omitted: log scale requires positive values") appear
when it happens? Both are reasonable; requires a UX call, not just a code fix.

### B3. NaN in the subject-ID column biases wide-format auto-detection

Class: 1 (Defensive Validation Deficit) / 4 (Implicit Logic Violation — `nunique()`
silently drops NaN before the ratio is computed, changing what the ratio means).

```python
# src/autopilot/statistical_analyzer_autopilot_ui.py:169-171 (_detect_wide_format)
unique_ratio = df[subject_col].nunique() / max(len(df), 1)
if unique_ratio < 0.8:
    return None
```

`Series.nunique()` excludes `NaN` by default. A wide-format sheet with one missing
subject ID (e.g. 3 valid IDs out of 4 rows) computes `3/4 = 0.75 < 0.8` and is
**rejected** as wide-format even though it clearly is — the user falls back to
manual column mapping for what should have been auto-detected, with no indication
that a missing subject ID was the cause.

### B4. NaN subjects silently excluded from repeated-measures balance detection

Class: 1 / 2 — same root cause as B3 (incomplete subject IDs), different failure
point downstream.

```python
# src/autopilot/statistical_analyzer_autopilot_pipeline.py:1223 (_ap_build_analysis_context)
counts = self.df.groupby([subject_column, within_factor]).size().unstack(fill_value=0)
has_structural_missing = (counts == 0).any().any()
```

`groupby()` drops `NaN` keys by default (pandas). If `subject_column` contains
`NaN` after the wide-to-long pivot, those rows are invisible to the
balanced-vs-unbalanced check that decides RM-ANOVA vs. LMM routing — a real
subject's incomplete data silently doesn't count toward "this design is
unbalanced," potentially routing to the wrong model.

**Design decision needed for B3 + B4 together (one decision, two call sites):** how
should the pipeline handle rows with a missing/NaN subject identifier — reject the
row at load time with a clear error ("N rows have no subject ID"), drop them with a
visible warning, or treat a missing ID as its own subject bucket? Whatever is
chosen should apply consistently at both the detection stage (B3) and the balance
computation stage (B4), since both stem from the same underlying question.

---

## Triage summary

| # | Finding | Class | Category | Confidence |
|---|---------|-------|----------|------------|
| A1 | Sphericity correction/epsilon key mismatch | 3, 2 | Mechanical | Verified active |
| A2 | RTE table blank-label fragility | 3 | Mechanical | Medium (latent) |
| A3 | All-NaN value columns pass detection | 1 | Mechanical | High |
| B1 | Grouped-EMM plot silent flat-bar fallback | 2 | Strategic | High |
| B2 | Log-scale silent data loss | 2, 4 | Strategic | High |
| B3 | NaN subject ID biases wide-format detection | 1, 4 | Strategic | Medium-High |
| B4 | NaN subjects dropped from balance detection | 1, 2 | Strategic | Medium |

Next step per this branch's established workflow (brainstorm → spec →
plan → subagent-driven-development / executing-plans → finishing-a-development-branch)
for anything in Kategorie B; Kategorie A items can go straight to a mechanical-fix
plan similar to `docs/superpowers/plans/2026-07-02-audit-code-bug-fixes.md`.
