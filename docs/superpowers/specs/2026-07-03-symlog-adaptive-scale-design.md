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

### 1. Data-driven `linthresh`

A fixed constant (e.g. `0.1`) doesn't generalize across assay types with
wildly different magnitudes. Compute it from the actual data:

```
Y_abs = {|y| : y in samples, y != 0}
linthresh = min(Y_abs) * 0.5   # or a low percentile, e.g. 5th percentile of Y_abs
```

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
warning no longer applies in this path — replace it with a lower-severity
informational annotation noting symlog was auto-selected and the `linthresh`
used (still visible/export-persistent, reusing `_draw_warning_annotation`'s
styling but arguably a distinct, less alarming visual — decide during
implementation whether it needs its own helper or an `ax.text` variant with a
neutral color instead of the red/orange "Data Warning" box).

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
- Some signal needs to reach `plot_from_config`/`_format_axes` indicating
  "this config's logy should route through symlog, not plain log" — likely a
  new config flag or just re-deriving "has non-positive values" at render time
  the same way `_format_axes` already does (it already recomputes `omitted`
  from `samples` today, so it could equally decide symlog-vs-log from the same
  data without a new flag — decide during implementation which is cleaner).

## Open questions for the eventual plan doc

- Exact `linthresh` formula (fixed 0.5× min-abs vs. percentile-based) — needs
  a decision, possibly informed by a couple of real assay datasets if
  available, rather than picked arbitrarily.
- Whether the "symlog auto-selected" annotation reuses
  `_draw_warning_annotation` as-is (red/orange, same as an actual data-loss
  warning) or needs a visually distinct "informational" style — using the
  alarming red box for a case where nothing was actually lost would misrepresent
  severity.
- Whether "Log X" ever needs the same treatment — Sprint 1 found X is always
  categorical (`Group`) for every plot type reachable through this dialog, so
  presumably still out of scope, but worth confirming this still holds before
  assuming it.
