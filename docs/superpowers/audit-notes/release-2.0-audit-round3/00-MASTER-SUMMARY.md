# AUDIT: BioMedStatX — GUI Dialogs + Documentation Parity (Round 3) @ ac2ff93

**Date:** 2026-08-16 · **Branch:** `feature/advanced-stats-automation` · **HEAD:** `ac2ff93`
**Prior rounds:** Round 1 @ `b16cf24`, Round 2 @ `3fd4796` (2026-07-07).
**Method:** `senior-engineering-partner` AUDIT mode — report-first, mechanized, `file:line` + the
verify command per claim. Every Round-2 finding was re-checked against **current source**, not the
prior write-up. **No code was changed in this pass.**

## Verdict

The premise that motivated Round 3 — "five weeks passed, nothing was fixed" — is **falsified by
the git record**. Round 1 → Round 2 saw zero commits to the batch (Round 2 correctly reported all
GD1–GD7 still open). But **after** Round 2's commit (`3fd4796`) a substantial remediation wave
landed (the "post-tag remediation round folded into 2.0"). Re-verifying each finding from source:
**8 of the 13 Round-2 findings are now FIXED — including the sole HIGH (GD1)** — verified from the
code/doc itself, not from the fix commits' messages. **Three LOW items remain open**; a fourth (GD4)
proved on re-read **not to be a defect**, and GD5 is moot because the doc it lived in left the
repository. **No new HIGH/MEDIUM regressions**, and a
fresh full pass of `help_content.py` against current behavior found **no new staleness**. Nothing
here is a release blocker.

## Status of every Round-2 finding (re-verified at `ac2ff93`)

| ID | Round-2 severity | Round-3 verdict | Proof (command / read) |
|---|---|---|---|
| **GD1** mixed_anova recipe overstated sphericity correction | HIGH | **FIXED** | `statisticaltester.py:1683-1684,1694-1695` reassigns canonical `p_value` to the GG-corrected value for the within **and** interaction rows (mirrors RM-ANOVA); recipe reworded at `help_content.py:282` (`6588803`). Verified by reading the E1 block, not the commit msg. |
| **GD2** dead `ColumnSelectionDialog` | MEDIUM | **FIXED** | `git grep -n ColumnSelectionDialog src/ tests/` → **0 hits** (`162e645`). |
| **GD3** no keyboard zoom in decision-tree viewer | MEDIUM | **FIXED** | `decision_tree_view.py:455-475` `keyPressEvent` handles `Key_Plus/Equal/Minus` (+ reset) (`c949e4c`). |
| **GD8** journal palettes fail WCAG 3:1 | MEDIUM | **FIXED** | Recomputed relative-luminance contrast on live `DataVisualizer.CURATED_PALETTES`: Nature/Science/NEJM/Lancet now **0 entries < 3:1**; palette tables also consolidated to one source (`b55d638`). |
| **GD9** linear_regression recipe undersells coefficient table | MEDIUM | **FIXED** | `help_content.py:491` now describes the full per-predictor table (param, coef, SE, t, p, 95% CI) (`b83e700`). |
| **GD10** README launcher-name self-contradiction | MEDIUM | **FIXED** | `README.md:73` and `:158-159` now both say `start.sh` / `run.bat`; `ls *.sh *.bat` matches (`12ed201`). |
| **GD11** `HelpHubDialog.copy_button` AttributeError branch | MEDIUM | **FIXED** | `git grep -n copy_button statistical_analyzer_dialogs.py` → **0 hits** (`a930ba4`). |
| **GD12** `get_config()` drops filename on invalid input | MEDIUM | **FIXED** | `plot_aesthetics_dialog.py:1886-1889` now sanitizes inline (`_re.sub(...,'_')`, updates field, continues); no early `return` before keys set (`8ed0a93`). |
| **GD4** MultiSelection list "no hint" | LOW | **NOT A DEFECT** (re-read) | Round-2 cited `:931-939`, a dialog since **removed** (`33ef11d`). The only surviving MultiSelection list — `ExploratoryMatrixDialog:703-711` — uses `QListWidget.MultiSelection` (a plain click **toggles**, no Ctrl/Space needed), **pre-selects all** items (`:709-710`), and is labeled (`:697`). No gesture to hint. Earlier keyword-grep verdict corrected by reading the region. |
| **GD5** CLAUDE.md `_ap_*` line-number drift | LOW | **MOOT** | `git ls-files --error-unmatch CLAUDE.md` → *not tracked* (removed in `2e67a98`). No tracked doc inherited the `pipeline:822/832/875` refs (actual defs now `947/957/1052`). Local-only dev file. |
| **GD6** `ComparisonSelectionDialog` empty-selection has no guard | LOW | **STILL OPEN** | `comparison_selection_dialog.py:90-95` `get_selected_comparisons` returns the (possibly empty) list, no `QMessageBox.warning`. |
| **GD7** tooltip-vs-checked-state comment | LOW | **N/A** (not a bug) | Unchanged; advisory only. |
| **GD13** dead `create_plot_check` guard | LOW | **STILL OPEN** | `autopilot_pipeline.py:2020-2021` `if hasattr(dialog,'create_plot_check')` — attribute never exists; documented no-op. |
| **GD14** German strings in `_ap_detected_test_label` | LOW | **STILL OPEN** | `autopilot_pipeline.py:1483-1484` `"Korrelationsanalyse (Spearman/Pearson)"`, `"Lineare Regression (OLS)"` among 10 English siblings. |

Round-1 fix still holding: the `"Als Lineare Regression analysieren…"` German checkbox label is
**absent** (`git grep` → 0), confirming it stays fixed.

## `decisiontreevisualizer.py` coverage question (Round-3 task 2)

**Answer: it was audited — but in batch `05-visualization`, never in the GUI/Docs-Parity batch
(`07`).** `ctx_search` across the audit-notes shows the file's findings live in Round 1/2's
`05-visualization.md` (VZ5 NaN→"p = nan", VZ6 flowchart `else` catch-all, VZ9 dead
`create_association_tree`/`FlowchartVisualizer`). So it is **not** an audit blind spot, but the
*parity lens* (does the rendered node match the docs?) had never been applied to it.

Applying that lens now to **today's** changes (`0b610a7`): the value-based `was_transformed` guard
(`decisiontreevisualizer.py:212`), the `_phase` selection (`:168`), and the `"No further"` handling
introduce **no new documentation drift**. The `help_content.py` recipes describe the flow
generically ("first tries a transformation, then falls back" — `:92,:156,:219`), which remains
accurate; `tutorial_overlay.py` describes no transform node; and the new `CHANGELOG.md [Unreleased]`
entry documents the behavior correctly. Parity holds.

## New findings from the fresh pass (tasks 3–5)

- **None.** A fresh read of every `help_content.py` recipe against current behavior (Welch default,
  transformation-then-rank fallback, two-way **permutation** fallback, mixed **GG-corrected** gating,
  linear-regression coefficient table) found **no code-vs-recipe drift** beyond the already-fixed
  GD1/GD9. The `README`/`CHANGELOG` contradiction check (GD10 method), including the freshly-added
  `[Unreleased]` lines, surfaced **no internal inconsistency**.
- **False positive avoided (recorded so it isn't re-raised):** four non-journal palettes still carry
  sub-3:1-vs-white entries — Okabe-Ito (`#E69F00/#56B4E9/#F0E442`), Grayscale HC, Muted Pastel (8/8),
  Turbo. These are **by design**: Okabe-Ito's are the canonical colorblind-safe values (optimized for
  series discriminability, not white-background contrast), Grayscale HC is intentionally a light-to-
  dark ramp, Muted Pastel is pastel by name, Turbo is a perceptual colormap. "Fixing" them would
  destroy the palette's identity. Round-2's GD8 was correctly scoped to the *journal* presets, which
  are now fixed.

## Recommended remediation order (the 4 open LOWs — all optional, none blocking)

1. **GD14** — translate the two German labels to match the 10 English siblings (`autopilot_pipeline.py:1483-1484`); trivial, user-visible on the main-window rationale.
2. **GD13** — delete the dead `create_plot_check` guard (`:2020-2021`); bundle with any dead-code sweep.
3. **GD6** — add the sibling `if not selected: QMessageBox.warning(...)` guard to `get_selected_comparisons`; one line, mirrors `GroupSelectionDialog`.
4. **GD5** (housekeeping, not code) — if the local `CLAUDE.md` is ever promoted into a *tracked*
   architecture doc, cite `_ap_*` functions **by name**, not line number, to stop the drift recurring.

See `07-gui-docs-parity.md` for the full per-finding detail and the mechanized-check table.
