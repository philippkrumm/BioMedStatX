# Spec: Symlog adaptive scale for log-axis data with values ≤ 0 (Sprint 2 preview)

Date: 2026-07-03
Status: **not started** — captured during Sprint 1 wrap-up (see
`docs/superpowers/plans/2026-07-03-visualization-error-transparency.md`) as the
follow-up to the hard-gate shipped there. Not scheduled unless asked.
Scope: `src/visualization/datavisualizer.py` (`_format_axes`), `src/ui/dialogs/plot_aesthetics_dialog.py`
(`StyleTab`, `PlotAestheticsDialog._apply_log_scale_gating`).

## Motivation

Sprint 1 shipped a hard gate: `PlotAestheticsDialog._apply_log_scale_gating`
disables "Log Y" outright whenever `self.samples` contains any value ≤ 0
(`plot_aesthetics_dialog.py`, method added next to `self.style_tab = StyleTab(self.config)`
at line 1624). That's a correct fail-safe against silent data loss, but it's a
blunt instrument: background-subtracted assay data (e.g. dual-luciferase after
blank subtraction) legitimately produces near-zero and negative readings, and
those datasets still need high-dynamic-range visualization. Hard-blocking log
scale for them removes a real analysis capability, not just an error case.

## Design

`matplotlib`'s `symlog` scale splits the axis into three domains around a
threshold `linthresh`:
- `y > linthresh` → standard log10.
- `-linthresh ≤ y ≤ linthresh` → linear (avoids the singularity at zero, shows
  baseline noise/near-zero readings without loss).
- `y < -linthresh` → mirrored log10 (`-log10(-y)`).

### 1. Data-driven `linthresh` — **decided: 5th percentile, not min**

A fixed constant (e.g. `0.1`) doesn't generalize across assay types with
wildly different magnitudes. Compute it from the actual data:

```
Y_abs = {|y| : y in samples, y != 0}
linthresh = percentile(Y_abs, 5)
```

**`min(Y_abs) * 0.5` was rejected during review:** a single technical-artifact
value near zero (e.g. one pipetting-error reading at 0.00001 RLU in a
background-subtracted assay) would collapse `linthresh` to near-zero, forcing
the entire real noise floor into the log domain and artificially inflating its
visual spread. The 5th-percentile estimator is robust to that single-point
failure mode — it characterizes the noise band from multiple low readings
instead of the single smallest one.

Goal: baseline noise/near-zero readings fall inside the linear zone, while the
biological signal spreads out in log space above it.

### 2. Backend (`_format_axes` in `datavisualizer.py`)

Currently (post-Sprint-1): when `logy=True` and non-positive values are
present, `_format_axes` still calls `ax.set_yscale('log', base=10)` and draws
`_draw_warning_annotation` reporting the omitted count (see
`_format_axes`, the `omitted > 0` branch).

New behavior: when `logy=True` and non-positive values are present, route to
`ax.set_yscale('symlog', linthresh=calculated_thresh)` instead of the plain
log scale + omission warning. No points are dropped, so the "N values omitted"
warning no longer applies in this path.

**Decided: new `_draw_notice_annotation(ax, text)` helper, distinct from
`_draw_warning_annotation`.** Reusing the red/orange "Data Warning" styling for
a successful, lossless symlog transform would misrepresent the state of the
data — nothing was dropped, only the display was adapted. `_draw_notice_annotation`
uses a neutral background (slate gray / muted blue) and reads e.g.
`"Data Notice: Values ≤ 0 detected. Auto-applied symlog scale (linthresh = X.XX)."`
Keeps transparency without implying a structural failure.

**Ticks:** matplotlib's default tick placement on `symlog` is coarse/uneven.
Needs explicit `matplotlib.ticker.SymmetricalLogLocator` (with the same
`linthresh`) and a matching formatter (e.g. `LogFormatterMathtext`) applied to
the y-axis, not just `set_yscale`.

### 3. UI (`plot_aesthetics_dialog.py`)

`PlotAestheticsDialog._apply_log_scale_gating` (Sprint 1) changes from a hard
gate to a "smart toggle":
- No longer calls `self.style_tab.logy_check.setEnabled(False)`.
- Still detects values ≤ 0 in `self.samples`.
- Updates the tooltip instead of disabling: e.g. `"Values ≤ 0 detected.
  Symmetric log scale (symlog) will be used automatically for lossless
  display."`
- **Decided: no new config flag.** `_format_axes` already recomputes
  omitted/non-positive counts from `samples` at render time (Sprint 1); it can
  equally decide symlog-vs-plain-log from that same data independently. The
  UI only owns the tooltip/awareness update, not the routing decision — this
  avoids a state-mismatch between the UI config dict and the actual data
  payload at render time (e.g. stale config loaded before current data was
  known, same class of problem Sprint 1's gating already had to consider).

## Resolved (was open questions, now decided during spec review)

- `linthresh`: 5th percentile of `|Y_abs|`, not `0.5 * min`. See rationale above.
- Annotation styling: dedicated `_draw_notice_annotation` helper (neutral
  color), not a reuse of `_draw_warning_annotation` (red/orange).
- Log X: confirmed out of scope — `plot_bar`/`plot_box`/`plot_violin` all map
  the independent variable to a categorical axis position; a continuous
  transform like symlog doesn't apply.
- Routing: runtime inference inside `_format_axes`, no new config flag.

## Remaining open question for the eventual plan doc

- Percentile implementation detail: `numpy.percentile` default interpolation
  method is fine for this use case (no need for a specific interpolation
  scheme), but confirm behavior on small samples (e.g. n < 5 non-zero values)
  doesn't produce a degenerate `linthresh` — worth a dedicated test case with
  a tiny sample during implementation.
