# Pre-2.0 Release Audit — Master Summary

Date: 2026-07-06/07. Repo state audited: `b16cf24` (branch `feature/advanced-stats-automation`).
Method: 7 parallel subsystem audits, each using the `senior-engineering-partner` skill's
`AUDIT:` mode (report-first, `file:line` evidence, severity-ranked, mechanized checks —
see `~/.claude/skills/senior-engineering-partner/references/audit-report-format.md`).
Total: ~38,000 lines across 57 `src/` files, plus root docs (CHANGELOG.md, README.md,
CLAUDE.md) and Help Hub content.

**Process note (transparency):** the first dispatch of all 7 batches hit a session usage
limit mid-run; only 2 of 7 (statistical core, advanced testing engines) produced a report.
Those 2 were independently spot-verified against the actual source before being trusted.
The remaining 5 were re-run after the limit reset and completed cleanly. Every batch's
top finding was independently re-verified by direct code inspection (grep/read/reproduction
script) before being included here — not accepted on the sub-agent's word alone. All 7 raw
reports are in this directory (`01-`–`07-*.md`); this file is the cross-batch synthesis.

**Overall verdict.** No CRITICAL findings. No live security emergency (this is a local
single-user desktop app — see `references/my-environment.md`). But there is a real,
tri-independently-confirmed **statistical-correctness** bug (Mixed ANOVA sphericity
correction never gates the significance verdict it's supposed to), a reproducible **HTML/
script-injection** path in the exported report, and a handful of other HIGH findings that
are worth fixing before shipping 2.0. The codebase's overall engineering discipline is
good — several previously-fixed bug classes from earlier this session were checked and
confirmed to hold up under re-audit, and every batch found more strengths than it did
LOW-severity issues.

---

## The one finding to fix first: Mixed ANOVA sphericity correction doesn't gate the verdict

Found independently by **3 separate audits** (statistical core, advanced testing engines,
and — as a documentation consequence — GUI/docs parity), all converging on the same root
cause from different files:

- `src/analysis/statisticaltester.py:1548-1555` sets the canonical `results["p_value"]`
  from the **uncorrected** pingouin interaction row.
- `_test_mixed_anova_within_sphericity` (`src/statistical_testing/mixed_assumptions.py:387-499`)
  correctly computes the Greenhouse-Geisser/Huynh-Feldt correction — but nothing ever
  overwrites `results["p_value"]` with it (contrast: RM-ANOVA's sibling function has this
  exact rewrite at `statisticaltester.py:1972-1974`, commented "E1: write the
  correction-selected p-value back to the canonical field").
- `src/statistical_testing/advanced_pipeline.py:237` gates post-hoc dispatch on this same
  stale, uncorrected `p_value`.
- `src/export/report_stat_rows.py:427-437` **does** apply the correction when rendering
  the HTML effects table (with a "(GG)"/"(HF)" suffix) — so the report can show a
  non-significant *corrected* p-value in one section while post-hoc comparisons (triggered
  by the uncorrected value) appear a few paragraphs later for the same effect.
- `src/core/help_content.py:282` (the Mixed ANOVA Help Hub recipe) tells the user the app
  "applies a conservative correction rather than assuming [sphericity] holds" — true for
  RM-ANOVA, not true for Mixed ANOVA, as of this audit.

**Fix** (small, well-scoped, has a working template in the same file): in `_run_mixed_anova`,
right after `results.update(within_sphericity_results)` (`statisticaltester.py:1706`),
mirror the RM-ANOVA pattern — extract the corrected p-value for whichever row drives
`results["p_value"]` and overwrite it before returning. Add a regression test with
synthetic sphericity-violating data asserting the correction reaches the canonical field.
This one fix also makes the Help Hub recipe text true again.

---

## HIGH findings (fix before 2.0)

| ID | Batch | One-line | file:line |
|---|---|---|---|
| **SC1/AT1** | Core, Testing Engines | Mixed ANOVA sphericity correction never reaches canonical `p_value` (see above) | `statisticaltester.py:1548-1555`, `mixed_assumptions.py:386-499` |
| **SC2** | Core | Logistic Regression reports first-predictor p-value as headline result; Beta Regression got the omnibus-LR fix, Logistic never did, though `llf`/`llnull` are already in scope | `clinical_models.py:1526-1530` |
| **P1** | UI Bridge | `_ap_build_analysis_context` mutates `self.df` in place with a Smithson-Verkuilen boundary transform, and runs on *every* mapping-change tick (not just "Start Analysis") — compounds toward 0.5 with repeated calls | `statistical_analyzer_autopilot_pipeline.py:1148-1151` |
| **SM1** | Specialized Models | Mixed-ANOVA "Tukey" post-hoc branch silently runs Holm-Šidák and reports it as "Tukey HSD" — reproducibility/citation-correctness defect | `posthoc_core.py:866-869` |
| **VZ1** | Visualization | `_add_significance_letters`/raincloud variant swallow ALL letters on any exception, with zero on-canvas indication — a publication figure can silently ship with no significance annotations | `datavisualizer.py:2314-2449` |
| **EX1** | Report Export | Real, reproduced HTML/script injection: unescaped `parameter` cell in 3 coefficient-table builders, rendered via `{{ chart.html \| safe }}` (deliberately bypasses Jinja's `autoescape=True`) — verified end-to-end with a real statsmodels fit | `report_association.py:34,87,135` |
| **GD1** | GUI/Docs | Help Hub `mixed_anova` recipe asserts a guarantee that doesn't hold (documentation-side consequence of SC1/AT1) | `help_content.py:282` |

Note: SC1/AT1/GD1 are one underlying bug found three ways — fixing the code (SC1/AT1) also
fixes the doc claim (GD1) with no separate edit needed.

## MEDIUM findings (worth doing, none urgent)

- **P2** — this session's `_reject_missing_subject_ids` NaN-subject guard covers 2 of 3
  call sites; `_ap_apply_mapping_heuristics` (`pipeline.py:596-605`) has its own unguarded
  `groupby(...).nunique()` — the exact footgun already fixed twice, missed at a third site.
- **P3** — LMM-vs-RM-ANOVA balance heuristic reads unfiltered `self.df` instead of the
  filter-applied `analysis_df` used everywhere else in the same function; wrapped in a bare
  `except Exception: pass`.
- **P4** — multi-dataset mode only surfaces a blocked (data-quality-refused) result for the
  *first* DV column; a blocked result for DV #2+ rides silently into the combined report.
- **SC3** — Mixed ANOVA's between-factor variance-homogeneity tests are computed but purely
  informational; the main test never switches to a robust alternative even when clearly
  violated (needs a product decision, not a pure bugfix).
- **SM2** — a duplicate dict-key literal in `CorrelationModel._normality_check` silently
  discards the actual method-selection criterion in favor of a raw Shapiro-Wilk result
  (currently latent — nothing reads the top-level key yet).
- **SM3** — `primary_effect` dicts (Friedman/Freedman-Lane/Brunner-Langer) never carry
  `F`/`df1`/`df2`, so a cosmetic "Main effect: X" summary line silently never renders (the
  detailed table one section earlier already has the correct numbers — not gating anything).
- **AT2/AT3** — advanced-model Dunnett/EMM-MVT post-hoc silently substitutes an arbitrary
  "control" group when no selection callback is provided (headless/batch invocation only);
  a cancelled control-group dialog leaves `posthoc_method`/`selected_comparisons`
  inconsistent. The simpler `posthoc_fallback.py` sibling already handles this safely
  (downgrades to Tukey) — same fix should be mirrored here.
- **VZ2** — two independent silent-drop paths can under-count significance brackets vs.
  `pairwise_results` with no on-canvas indication.
- **VZ3** — `logx`'s non-positive-data handling was never brought up to the `logy` fix's
  standard (still warns-and-drops instead of auto-adapting to symlog).
- **VZ4** — `DataVisualizer.DEFAULT_COLORS` fails WCAG's 3:1 non-text-contrast floor for 2
  of 6 colors (`#33FF57` = 1.35:1, `#33FFEC` = 1.26:1 against white) — mechanically verified,
  reachable as the actual rendered palette for any direct/programmatic plot call.
- **EX2** — inconsistent escaping of group/factor-level labels in Plotly trace names (one
  function escapes, four siblings don't; lower-confidence than EX1, not reproduced as
  exploitable).
- **GD2** — `ColumnSelectionDialog`'s "multi-dataset" checkbox is dead code that directly
  contradicts CLAUDE.md's own claim that this was "already removed."
- **GD3** — decision-tree viewer has no keyboard-accessible zoom/reset, only mouse-wheel.
- **GD4** — a multi-select list's keyboard interaction (Space-to-toggle) is undiscoverable
  with no on-screen hint.

## LOW findings (track, no urgency)

P5 (substring false-positives in binary-outcome grouping-name heuristic — e.g. "ArmLength_mm"
false-positives as a grouping column), P6 (Help Hub recipe IDs duplicated as string literals
with no shared constant — currently in sync, but no test enforces it), P7 (bare except
swallows more than "validation failures" as its comment claims), SC4 (LMM df-method
mislabeled "Kenward-Roger/Satterthwaite" when it's a simpler heuristic), SC5 (`.get("F", 0)`
would silently substitute F=0 on a future statsmodels column rename), SM4 (dead substring-match
control-group code, confirmed unreachable), AT4 (Box's M test uses a hand-rolled, honestly-labeled
approximate p-value), AT5/AT6 (two orphaned dead-code engine scaffolds, one of which could
confusingly be wired in by mistake), AT7 (logistic regression's validation chokepoint is split
across two files with no cross-reference comment), VZ5 (NaN p-values render as literal "nan"
strings instead of an explicit "N/A"/flag), VZ6 (unrecognized `model_type` silently renders a
CorrelationMatrix-shaped diagram — deliberate catch-all, low blast radius), VZ7 (a census of
~15 cosmetic-only except-and-degrade sites, individually reviewed, none affect statistical
correctness), EX3 (sphericity status reads a key no writer sets, `sphericity_met` vs.
`sphericity_assumed`, masked by a fallback that's incomplete for 2-level designs), EX4 (two
inert dead-code fallback keys), GD5 (CLAUDE.md line-number citations drifted ~100 lines),
GD6 (one dialog missing an empty-selection warning its siblings have), GD7 (a tooltip updates
regardless of checkbox state — intentional, undocumented as such).

---

## Strengths confirmed across all 7 audits

- **Every previously-fixed bug from this session's earlier work was re-verified and holds
  up.** RM-ANOVA's sphericity→p_value rewrite, the grouped-EMM/log-axis on-canvas warning
  fixes, ANCOVA/LMM `control_group` wiring, the linear-regression coefficient table's
  dispatch wiring, the operator-precedence and Help-Hub-label fixes — all confirmed still
  correct, not regressed.
- **The writer/reader key-contract problem (found 3× earlier this session) is now
  overwhelmingly clean.** Of ~140 distinct result-dict keys cross-referenced against every
  writer in `src/analysis/`/`src/statistical_testing/`, only one live gap remained (EX3,
  LOW) and every `_build_*` report function is now reachable from its dispatch chain — zero
  orphaned builders.
- **Core statistical machinery is correct where checked**: DF/p-value pairing, Cohen's
  d/Hedges' g/Welch's f/partial η² formulas, Games-Howell and `scipy.stats.dunnett`'s joint
  p/CI fit, the hand-rolled Holm correction (numerically matched to statsmodels across 2000
  trials), HC3 robust-covariance propagation, and the data-quality validation chokepoint
  (`validate_samples_for_test`) all verified correct and genuinely gating every dispatch path.
- **No recurrence of the "string-coupling" anti-pattern** in the advanced testing-engines
  package — an exhaustive grep confirmed every dispatch decision there is an exact match
  against internal constants, not fuzzy label matching.
- **`TutorialOverlay`** is a genuinely well-built accessible component (cross-platform
  `prefers-reduced-motion` probing, full keyboard nav, focus-stealing prevention).
- **CHANGELOG.md is well-maintained** — 7 of 7 spot-checked behavioral claims verified true
  against current code.
- **Jinja autoescaping is correctly configured** and un-bypassed for the vast majority of
  report fields; the injection (EX1) is narrowly scoped to 3 chart-table builders, not a
  blanket bypass.

---

## Recommended order (across all 7 batches)

1. **Mixed ANOVA sphericity → p_value fix** (SC1/AT1/GD1) — highest statistical-correctness
   value, smallest diff, has a working template in the same file (RM-ANOVA's existing fix).
2. **EX1 (HTML injection)** — `html.escape()` around 3 f-string sites; cheapest fix, closes
   the only reproducible injection path found.
3. **P1 (self.df compounding mutation)** — stop mutating shared state from a function called
   on every UI tick; operate on the existing local copy instead.
4. **SM1 (Tukey/Holm-Šidák mislabel) + SC2 (Logistic omnibus p-value)** — both are
   citation/reproducibility-correctness fixes with existing in-file templates to mirror.
5. **VZ1 (significance letters silent-drop)** — same fix mechanism already used twice this
   session (`_draw_warning_annotation`), just needs a third call site.
6. **P2 + AT2/AT3 (remaining guard/fallback gaps)** — mechanical, mirror existing patterns.
7. **VZ4 (color contrast), GD2 (dead dialog), EX3 (key rename)** — cheap, low-risk, bundle
   into a cleanup pass.
8. Everything else in the MEDIUM/LOW tables — track, no urgency before 2.0; SC3 and GD3
   need a product/design decision rather than a pure bugfix, same as this session's earlier
   B3/B4/coefficient-table findings.

Per the skill's `AUDIT:` mode discipline: **this document changes nothing**. Next step is
picking which findings to fix now vs. defer, then following this project's established
workflow (brainstorm for anything needing a design decision → spec → plan → TDD → verify).
