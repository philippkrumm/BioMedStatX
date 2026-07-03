# Spec: Sprint 1 — Sphericity report fix + visualization error transparency

Date: 2026-07-03
Source: `docs/superpowers/audit-notes/SUMMARY_PROACTIVE.md`, items A1, B1, B2.
Scope: `src/export/report_summaries.py` (A1), `src/visualization/datavisualizer.py`
(B1, B2), `src/ui/dialogs/plot_aesthetics_dialog.py` (B2).

## Paradigm

Scientific transparency over silent degradation. No exported plot or report may
present a degraded, incomplete, or data-lossy result without an unmistakable,
export-persistent indication of what happened. Decided in conversation (not a
separate brainstorming pass — the tradeoffs were already worked through: Hard
Block / In-Canvas Warning / Strict UI-Gating, see conversation for the
three-option analysis). Per-item decision below.

## A1 — Sphericity correction/epsilon key mismatch (mechanical, no design choice)

**Current bug** (verified against source, not just agent-reported):
`report_summaries.py:437-464` (`_build_assumption_summary`) reads
`sphericity = results.get("sphericity_test", {})` and looks for
`sphericity.get("correction")` / `.get("correction_applied")` /
`.get("greenhouse_geisser")` / `.get("gg_epsilon")` / `.get("epsilon_gg")` etc.
None of these keys are ever written into `results["sphericity_test"]`
(`statisticaltester.py:2624-2703` — that dict only ever contains
`test_name/W/chi_square/d/p_value/sphericity_assumed/interpretation`). The real
data lives at:
- `results["correction_used"]` (top-level string, e.g. `"Greenhouse-Geisser (ε = 0.456)"`)
  — set at `statisticaltester.py:2857-2874`.
- `results["sphericity_corrections"]["greenhouse_geisser"]["epsilon"]` /
  `["huynh_feldt"]["epsilon"]` (nested floats) — set at
  `statisticaltester.py:2826-2833` and `2843-2850`.

Consequence: exported sphericity-violation note has never shown an epsilon value.

**Fix:**
1. In `_build_assumption_summary`, read the correction label from
   `results.get("correction_used", "")` instead of the `sphericity_test` sub-dict.
2. Parse whether it names Greenhouse-Geisser or Huynh-Feldt (same substring logic
   as today: `"huynh"/"hf"` vs `"greenhouse"/"gg"`), then pull the matching epsilon
   from `results.get("sphericity_corrections", {}).get("greenhouse_geisser", {}).get("epsilon")`
   or the `huynh_feldt` equivalent.
3. Keep the existing `sphericity.get(...)` lookups as a secondary fallback only
   (in case older cached/serialized result payloads lack the top-level keys) — do
   not delete them outright, just reorder so the correct path is tried first.
4. `eps_str` construction and the final `sphericity_correction_note` f-string stay
   as-is; only the two lookups feeding `corr`/`gg_eps`/`hf_eps` change.

**Acceptance:** an RM-ANOVA or Mixed-ANOVA result with a sphericity violation and a
known epsilon produces an exported report note containing the actual epsilon value
used for correction (matching `sphericity_corrections`), not an empty `eps_str`.

**Test:** construct a `results` dict shaped like real `statisticaltester.py` output
(top-level `correction_used` + nested `sphericity_corrections`, `sphericity_test`
sub-dict WITHOUT the (nonexistent) correction keys — mirroring reality) and assert
`_build_assumption_summary`'s note contains the epsilon string. Negative control:
run against current code first, confirm the note is missing epsilon, then apply
fix and confirm it appears.

## B1 — Grouped-EMM plot silent flat-bar fallback → in-canvas warning

**Current:** `datavisualizer.py:2830-2846` (`plot_from_config`) — on any exception
from `grouped_inputs_from_samples`/`plot_grouped_bar`, falls back to
`DataVisualizer.plot_bar(groups, samples, **bar_kwargs)` with only a
`logger.warning`, no visible indication in the rendered/exported figure.

**Decision: In-Canvas Warning Annotation** (Hard Block rejected — degrades a
workflow the user may already be aware of; Strict UI-Gating rejected — the failure
only manifests deep inside the post-hoc engine during axis construction, not
detectable ahead of time at a UI gate).

**Fix:** in the `except Exception as exc:` branch, after rendering the flat
`plot_bar` fallback, draw a visible warning annotation directly on `ax` before the
figure is returned/saved — must be part of the raster/vector output, not a
UI-only overlay, so it survives PNG/SVG export and can't be cropped out
accidentally. Suggested text: `"Structural Warning: Within-Between interaction
split failed. Showing flat pooling."` Placement: top of axes, high-contrast
(e.g. red/orange background box), non-obstructive to data (small font, corner or
top strip). If `datavisualizer.py` already has an annotation/badge helper for
other warnings, reuse its styling for consistency; otherwise define one minimal
helper.

**Acceptance:** deliberately triggering the except branch (e.g. malformed
`samples` dict that fails `grouped_inputs_from_samples`) produces a saved figure
(check via `ax.texts` or pixel inspection) containing the warning text — not just
a log line.

## B2 — Log-scale silent data loss → UI-gating + defensive in-canvas warning

**Current:** `plot_aesthetics_dialog.py:823-831` — "Log X"/"Log Y" checkboxes
always enabled regardless of data. `datavisualizer.py:1894-1897` (`_format_axes`)
sets `ax.set_xscale('log')`/`ax.set_yscale('log')` unconditionally; matplotlib
silently drops non-positive points with no warning.

**Decision: combination, per user's recommendation** — UI-gating as the primary
prevention, in-canvas warning as the defensive fallback for cases the gate can't
cover (config loaded from a saved preset before current data was known, or data
reloaded/changed after the checkbox was already checked).

**Fix, part (a) — UI-gating:**
When `plot_aesthetics_dialog` is opened or the underlying data changes, compute
`min()` over the relevant axis's data (y-data for `logy_check`, x-data for
`logx_check` — dialog needs access to the current plot's data or value range;
trace how config/data is threaded into the dialog today before deciding the exact
hook point). If `min <= 0`, disable (`setEnabled(False)`) the corresponding
checkbox and set a tooltip: `"Log scale unavailable: data contains values ≤ 0."`
If the checkbox was already checked when data changes to include non-positive
values, uncheck it as part of disabling (don't leave a disabled-but-checked
control whose state silently activates a stale log transform).

**Fix, part (b) — defensive in-canvas warning:**
In `_format_axes` (or wherever `logx`/`logy` are actually applied), before
setting log scale, count points that will be omitted (`<= 0` or non-finite after
transform). If `logx`/`logy` is `True` and the omitted count > 0 (meaning the gate
in (a) didn't catch this — e.g. stale config), draw the same style of warning
annotation as B1: `"Data Warning: N values ≤ 0 omitted from log-scale axis."`

**Acceptance:**
(a) opening the aesthetics dialog with data containing values ≤ 0 disables the
relevant checkbox with the tooltip present.
(b) forcing the warning path (e.g. programmatically calling the plot function with
`logy=True` on data containing ≤0 values, bypassing the dialog gate) produces a
saved figure containing the omitted-count warning text.

## Non-goals (explicitly out of scope for this sprint)

- B3/B4 (NaN subject-ID handling in wide-format detection / balance detection) —
  separate design decision, separate sprint.
- A2 (RTE table key fragility) / A3 (all-NaN value columns) — separate mechanical
  sprint.
- **`symlog` as an alternative to hard-gating "Log Y" for values ≤ 0** (flagged
  during plan review): values ≤ 0 aren't always garbage — background-subtracted
  luminescence assays (e.g. dual-luciferase) can legitimately produce zero/negative
  readings near baseline. `matplotlib`'s `symlog` scale (linear near zero via
  `linthresh`, log beyond it) would preserve those points instead of hard-blocking
  the log transform. This sprint still hard-gates (Task 4 in the plan) — revisit
  `symlog` as a follow-up once the hard-gate is in and real usage patterns for
  ≤0 data are better understood.

## Open questions for the plan doc

- Exact hook point for computing "current data range" inside
  `plot_aesthetics_dialog.py` — does the dialog already receive the data/series,
  or only a config dict? (Needs tracing before task breakdown.)
- Whether a shared `_draw_warning_annotation(ax, text)` helper should be added to
  `datavisualizer.py` and reused by both B1 and B2, vs. two separate call sites
  with duplicated styling. Recommendation: shared helper — same visual language
  for both warning types is part of the paradigm ("consistent UX safety
  philosophy" per this sprint's framing).
