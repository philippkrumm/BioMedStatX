# Design: Clustering round-2 audit findings into fixable work packages

Date: 2026-07-07. Branch: `feature/advanced-stats-automation`. Source: 7-batch pre-2.0
release audit round 2, collated in
`docs/superpowers/audit-notes/release-2.0-audit/00-MASTER-SUMMARY.md` (commit `bd021a6`).

## Problem

The round-2 audit produced ~40 findings across 7 subsystems. Fixing them one at a time in
whatever order they were reported wastes context-switching (jumping between unrelated files)
and conflates two very different kinds of work: findings with an unambiguous, already-specified
fix vs. findings that need a genuine design/product decision before any code should be written.
This doc defines how the findings are grouped into work packages, which packages need their own
brainstorm→spec cycle before implementation, and the order to work through them in.

## Scope

**In scope:** all HIGH and MEDIUM findings from the master summary (~20 findings).
**Out of scope (this pass):** LOW-severity findings — tracked in the master summary, no
release-blocking urgency per the audit's own assessment. Revisit after the HIGH/MEDIUM pass.

## Grouping axis

Fix-type first, not file or severity:

- **Tier A — needs a design decision.** The fix approach itself is an open question (which
  algorithm, which UX behavior, which product tradeoff). Each item gets its own
  brainstorm → spec → plan → TDD cycle before any code is written, per this project's
  established workflow (see `HANDOFF.md`).
- **Tier B — mechanical.** The master summary already specifies the exact fix (rename this
  key, add this one call, drop this kwarg, vectorize this loop). No open design question — goes
  straight to TDD via `subagent-driven-development`, grouped by file/subsystem to keep each
  package's diff contained to files no other in-flight package touches.

## Tier A — design-decision packages (sequential, own brainstorm each)

| # | Findings | File(s) | Why it needs a design decision |
|---|---|---|---|
| **A1** | SC2/AT1/AT2/AT3/GD1 (Mixed-ANOVA sphericity — found independently by 3 batches) | `src/analysis/statisticaltester.py`, `src/statistical_testing/mixed_assumptions.py`, `src/core/help_content.py` | pingouin 0.6.1 doesn't return the GG/HF columns the code expects — a real epsilon computation has to be designed (derive from `eps`, or call `pg.epsilon()`), plus the interaction-row fix (AT3) and a golden-R Mixed ANOVA test fixture. Highest-value fix in the whole audit. Fixes GD1's Help Hub claim for free once landed — no separate doc edit. |
| **A2** | GD8 (journal palette contrast) | `src/ui/dialogs/plot_aesthetics_dialog.py` | Product decision: swap the lowest-contrast preset colors for darker/more-saturated variants of the same journal house style, vs. add an in-app low-legibility indicator on the swatch. |
| **A3** | GD12 (`get_config()` silently drops filename/config keys) | `src/ui/dialogs/plot_aesthetics_dialog.py` | UX decision: sanitize the invalid filename and continue vs. block acceptance and re-prompt the user. |

**Sequencing:** A1 first — highest value, and B4 (below) touches the same file
(`mixed_assumptions.py`) so the whole Tier B batch waits for A1 to be committed to avoid a
merge conflict. A2 and A3 are small and independent of A1 (different functions, no shared file
with any Tier A or Tier B package) — can be brainstormed in parallel with A1 or right after.
Once decided, A2+A3 implementation is small enough to fold into one follow-up mechanical
package (B8) rather than needing their own `subagent-driven-development` dispatch.

## Tier B — mechanical packages (parallel, `subagent-driven-development`, dispatched after A1 is committed)

| Package | Findings | File(s) |
|---|---|---|
| B1 Core Dispatch | SC1 (`make_blocked_result` bad kwarg, 4 sites), SC3 (ANCOVA/LMM EMM key rename to match LMM's `statistic`/`std_err`), SC4 (`posthoc_choice` NameError) | `src/analysis/analysis_core.py`, `src/analysis/clinical_models.py` |
| B2 UI Bridge | U1 (excepthook `TypeError`), U2 (multi-DV batch reuses one DV's test decision) | `src/analysis/statistical_analyzer.py`, `src/autopilot/statistical_analyzer_autopilot_pipeline.py` |
| B3 Specialized Models | SM1 (outlier decimal-format corruption — gate the comma-substitution behind a "does it already parse as float" check), SM2 (Dunn-test bootstrap CI — vectorize with `np.subtract.outer`) | `src/analysis/outlier_core.py`, `src/analysis/posthoc_core.py` |
| B4 Testing Engines | AT4 (logistic regression skips the shared pre-flight gate), AT5 (Welch ANOVA silent fallback needs a degraded-flag) | `src/statistical_testing/advanced_pipeline.py`, `src/statistical_testing/mixed_assumptions.py` *(same file as A1 — dispatch after A1 lands)* |
| B5 Visualization | VZ1 (sig-letters swallow exceptions), VZ2 (silent bracket-drop), VZ3 (logx not brought up to logy's symlog fix), VZ4 (2 WCAG-failing default colors), VZ8 (raincloud plots skip shared bracket/letter helper) | `src/visualization/datavisualizer.py` |
| B6 Report/Export | RE1 (HTML injection, 3 sites), RE2 (inconsistent escaping, 4 sites), RE5 (shared `_esc()` helper — bundled in since RE1 needs it; zero extra cost, not scope creep) | `src/export/report_association.py`, `src/export/report_charts.py`, `src/export/report_summaries.py`, `src/export/report_formatting.py` |
| B7-GUI | GD2 (delete dead `ColumnSelectionDialog`), GD3 (add keyboard zoom binding), GD11 (delete unreachable `copy_button` reference — `git blame` first; default to delete unless it shows a removed copy-to-clipboard feature) | `src/ui/dialogs/statistical_analyzer_dialogs.py`, `src/ui/components/decision_tree_view.py` |
| B7-Docs | GD9 (stale `linear_regression` Help Hub recipe), GD10 (README launcher-script self-contradiction) | `src/core/help_content.py`, `README.md` |

8 packages, no file overlap with each other (only B4 overlaps with A1, hence the ordering
constraint above). Each package: one `subagent-driven-development` dispatch (implementer →
spec-compliance reviewer → code-quality reviewer), all 8 running in parallel once A1 is in.

## Out of scope follow-up

Tier A's A2/A3 outcomes become package **B8** (small, mechanical once the decision is made) —
not planned in detail here since the decision hasn't been made yet.

## Testing

Every Tier B fix gets a negative-control test (revert → confirm the exact bug symptom
reproduces → restore), per this project's established practice (`HANDOFF.md`, "What Worked").
Tier A fixes get whatever test design comes out of their own spec (A1 in particular needs a
new golden-R fixture, not just a unit test).

## Next steps

1. Brainstorm A1 (Mixed-ANOVA sphericity) as its own session — highest value, do first.
2. Once A1 is spec'd/planned/implemented and committed: invoke `writing-plans` for the Tier B
   batch (this doc already fully specifies each package — no further brainstorming needed for
   Tier B).
3. Brainstorm A2 and A3 (can happen in parallel with A1, since no file/logic overlap).
