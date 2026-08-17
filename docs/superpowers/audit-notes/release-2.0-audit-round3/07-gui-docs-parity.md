# AUDIT: BioMedStatX — GUI Dialogs + Documentation Parity (Round 3) @ ac2ff93

Scope (Round 1/2 parity set, plus the two Round-3 additions):
`src/ui/dialogs/plot_aesthetics_dialog.py` (1934L), `statistical_analyzer_dialogs.py` (815L),
`comparison_selection_dialog.py` (95L), `src/ui/components/decision_tree_view.py` (480L),
`tutorial_overlay.py` (426L), `src/core/help_content.py` (655L),
`src/export/report_tooltips.py` (101L, newly folded in), `CHANGELOG.md` (415L), `README.md` (246L),
`CLAUDE.md` (untracked — see GD5). **Added for Round 3:** `src/visualization/decisiontreevisualizer.py`
(642L, touched today by `0b610a7`).

**Verdict.** A post-Round-2 remediation wave fixed 8 of 13 findings, the sole HIGH included, each
verified from current source. Three LOW items remain; GD4 proved **not a defect** on re-read; GD5 is
moot (its doc left the repo). No new HIGH/MEDIUM, no new recipe/README drift. Not a blocker.

## What I mechanically verified (not eyeballed)

| Check | Command / method | Result |
|---|---|---|
| Commits to the parity scope since Round 2 (`3fd4796`) | `git log --oneline 3fd4796..HEAD -- <11 scope paths>` | **41 commits** — a real remediation round landed after Round 2 (contrast: Round 1→2 was 0). This is why "nothing changed" is false. |
| GD1 mixed p-value now GG-corrected on the *gated* fields | Read `statisticaltester.py:1664-1697` | `f["p_value"] = main_effect_corr["final_p_value"]` (`:1684`) and `inter["p_value"] = inter_p_corr` (`:1695`) — canonical p reassigned for within **and** interaction, `p_unc` preserved. Recipe `help_content.py:282` reworded to match. **FIXED.** |
| GD2 dead `ColumnSelectionDialog` | `git grep -n "ColumnSelectionDialog" src/ tests/` | 0 hits. **FIXED.** |
| GD3 keyboard zoom | `git grep -nE "keyPressEvent\|Key_Plus\|Key_Minus\|Key_Equal" decision_tree_view.py` | `keyPressEvent` at `:455`, `Key_Plus/Equal` `:456`, `Key_Minus` `:463`. **FIXED.** |
| GD8 journal-palette contrast | Recomputed WCAG relative-luminance contrast-vs-white in Python for every hex in `DataVisualizer.CURATED_PALETTES` | Nature 0/8, Science 0/7, NEJM 0/7, Lancet 0/7 below 3:1 — the four flagged journal presets all pass now. Palettes also single-sourced (`datavisualizer.py:876`). **FIXED.** |
| GD9 linear_regression recipe | Read `help_content.py:489-491` | `:491` now lists the full per-predictor coefficient table (param, coef, SE, t, p, 95% CI, "not just the primary one"). **FIXED.** |
| GD10 README launcher names | `grep -nE "start\.sh\|run\.bat\|start\.bat\|Start_BioMedStatX" README.md` + `ls *.sh *.bat` | `:73` and `:158-159` both `start.sh`/`run.bat`; disk has exactly `start.sh`, `run.bat`. **FIXED.** |
| GD11 `copy_button` crash branch | `git grep -n "copy_button" statistical_analyzer_dialogs.py` | 0 hits. **FIXED.** |
| GD12 `get_config()` invalid-filename | Read `plot_aesthetics_dialog.py:1830-1893` | `:1886-1889` sanitizes (`_re.sub(r'[<>:"/\\\|?*]','_',raw_name)`), rewrites the field, and continues; the early `return config` is gone. **FIXED.** |
| GD4 MultiSelection "hint" | Read `statistical_analyzer_dialogs.py:688-724` (the widget region — **not** a keyword grep) | Round-2's `:931-939` dialog was **removed** (`33ef11d`). The surviving list `ExploratoryMatrixDialog:703-711` uses `MultiSelection` (a click **toggles**, no modifier), **pre-selects all** (`:709-710`), and is labeled (`:697`). **NOT A DEFECT.** |
| GD5 CLAUDE.md tracking | `git ls-files --error-unmatch CLAUDE.md`; `git grep -nE "pipeline:822\|_ap_browse_file" -- 'docs/**' '*.md'` | CLAUDE.md untracked (removed `2e67a98`); no tracked doc carries the drift. **MOOT.** |
| GD6 empty-selection guard | Read `comparison_selection_dialog.py:90-95` | Returns `selected` list with no `if not selected: QMessageBox.warning`. **OPEN.** |
| GD13 dead `create_plot_check` | `git grep -n "create_plot_check" src/` | Only `autopilot_pipeline.py:2020-2021`, behind `hasattr` — attribute never defined. **OPEN.** |
| GD14 German rationale strings | `git grep -nE "Korrelationsanalyse\|Lineare Regression \(OLS\)" src/` | `autopilot_pipeline.py:1483-1484`. **OPEN.** |
| Round-1 German-checkbox fix holding | `git grep -n "Als Lineare Regression analysieren" src/` | 0 hits — still fixed. |
| `decisiontreevisualizer.py` prior coverage | `ctx_search "decisiontreevisualizer" docs/superpowers/audit-notes` | Findings in `05-visualization.md` (VZ5/6/9), none in `07-gui-docs-parity.md`. Audited, but never through the parity lens. |
| Today's transform-node change vs docs | `git grep -nEi "transformed\|transformation" help_content.py tutorial_overlay.py` | Recipes describe the flow generically and still accurately; no node-label doc to go stale. **Parity holds.** |

## Findings — severity ranked

### LOW (all optional; none blocks release)

**GD14 — German strings in an otherwise-English rationale.** `autopilot_pipeline.py:1483-1484`:
`"correlation": "Korrelationsanalyse (Spearman/Pearson)"` and
`"linear_regression": "Lineare Regression (OLS)"` sit among ten English siblings in
`_ap_detected_test_label`, which feeds the main-window "Structure inferred as …" rationale. A user
running Correlation or Linear Regression sees a German clause. **Impact:** cosmetic i18n
inconsistency on a user-visible label. **Fix:** translate the two values (e.g. "Correlation
(Spearman/Pearson)", "Linear Regression (OLS)"). Same bug class as the already-fixed checkbox label.

**GD6 — `ComparisonSelectionDialog` accepts an empty selection silently.**
`comparison_selection_dialog.py:90-95`. `get_selected_comparisons()` returns whatever is checked —
possibly `[]` — with no `QMessageBox.warning`, unlike its sibling `GroupSelectionDialog`. **Impact:**
a user who unchecks everything and clicks OK gets a silent fall-through (the caller
`_custom_pairs_cb` then defaults to all-pairs), with no signal that their selection was ignored.
**Fix:** add the sibling guard: `if not selected: QMessageBox.warning(self, "No comparisons",
"Select at least one pair."); return None` and have the caller treat `None` as cancel.

**GD13 — dead `create_plot_check` guard.** `autopilot_pipeline.py:2020-2021`:
`if hasattr(dialog, 'create_plot_check'): dialog.create_plot_check.setChecked(False)`.
`PlotAestheticsDialog` has no such attribute (`git grep` finds only these two lines), so the branch
is a permanent no-op. **Impact:** none at runtime; dead defensive code pointing at a removed
attribute, same family as GD2. **Fix:** delete both lines.

### RECLASSIFIED — not a defect (re-read)

**GD4 — multi-select list "has no discoverability hint".** Round 2 cited
`statistical_analyzer_dialogs.py:931-939`; that location no longer exists — the dialog was removed
(`33ef11d`, "remove PairwiseComparisonDialog and TwoWayAnovaDialog"), and the file is now 815 lines.
The only surviving `MultiSelection` list is `ExploratoryMatrixDialog._build_ui` (`:703-711`), and it
does not carry the flagged problem: `QListWidget.MultiSelection` mode means a **plain single click
toggles** an item (no Ctrl/Space gesture to discover — that was Round 2's premise), every item is
**pre-selected by default** (`:709-710`), and the list is introduced by a label ("Select variables
(numeric columns):", `:697`). There is no hidden gesture, so there is nothing to hint. **This was
first carried as OPEN on the strength of a narrow keyword grep (`MultiSelection|Space|hint|Hold`)
returning no hint keyword — corrected here by reading the actual widget region.** No fix needed.

### MOOT

**GD5 — CLAUDE.md `_ap_*` line-number drift.** The cited `pipeline:822/832/875` no longer match the
actual defs (`947/957/1052`), but `CLAUDE.md` is **no longer tracked** (removed in `2e67a98`), and no
tracked doc inherited those refs. The stale copy exists only as a local, git-ignored developer file.
**Recommendation (housekeeping):** if it is ever promoted into a tracked architecture doc, reference
`_ap_*` functions **by name**, not line, so the drift cannot recur.

## Strengths (verified)

- **The sole HIGH is genuinely closed in code, not papered over.** Mixed ANOVA now wires the
  Greenhouse-Geisser-corrected p-value into the canonical fields the verdict and post-hoc dispatch
  read, for both the within factor and the interaction (`statisticaltester.py:1681-1697`), mirroring
  the RM-ANOVA fix — and the recipe was updated in lockstep. Round 2's central concern is resolved.
- **The remediation wave was disciplined.** Each fix is a small, single-purpose commit with a
  Conventional-Commit message that names the finding it closes (`162e645`, `c949e4c`, `a930ba4`,
  `8ed0a93`, `b55d638`, `b83e700`, `12ed201`, `6588803`). Re-verifying from source confirmed the
  messages were accurate — no "fixed" claim was hollow.
- **The palette fix also removed a duplication class.** Beyond swapping the low-contrast journal
  colors, the palette tables were consolidated into one source of truth
  (`DataVisualizer.CURATED_PALETTES`, `datavisualizer.py:876`), with the dialog resolving through it
  (`plot_aesthetics_dialog.py:419-420`) — the drift that produced GD8 can't reappear per-copy.
- **`help_content.py` is currently in parity with the engine.** A fresh recipe-by-recipe read against
  live behavior (Welch default `:92`, transformation-then-rank `:92/:156/:219`, two-way permutation
  fallback `:156`, mixed GG-corrected gating `:282`, linear coefficient table `:491`) found no
  code-vs-doc drift.
- **Today's value-based transform change is documentation-consistent.** The `was_transformed` guard
  (`decisiontreevisualizer.py:212`) and `_phase` selection introduce no stale recipe or tutorial
  text, and are correctly recorded in `CHANGELOG.md [Unreleased]`.

## report_tooltips.py — deferred parity check (Strang B)

`src/export/report_tooltips.py` was named in the Round-3 scope but never examined in Round 1/2. It
holds `STAT_ROW_DESCRIPTIONS` — 24 layperson tooltips keyed by stat-row **label** — surfaced via
`stat_row_info(label)`, attached to each rendered row in `html_exporter.py:93,205`
(`_r["info"] = _stat_row_info(_r.get("label",""))`). Parity here means two things: (a) each key
matches a label a report row actually emits, and (b) the copy accurately describes current behavior.

### What I mechanically verified (not eyeballed)

| Check | Command / method | Result |
|---|---|---|
| Tooltip consumers | `git grep stat_row_info -- src/` | Only `html_exporter.py:93,205`, keyed by each row's `label`. No other path. |
| Riskiest content claim — "Assumptions (Linear Regression)" names Shapiro-Wilk + Breusch-Pagan + Ramsey RESET | Read `correlation_models.py:805-842` | All three genuinely execute: `scipy_stats.shapiro` (:807), `het_breuschpagan` (:821), `linear_reset(power=2)` (:834). **Copy is accurate.** (A scoped grep of `analysis/`+`statistical_testing/` alone returned zero — a false-negative; the broad tree grep found them. GD4 lesson applied.) |
| Which of the 24 keys map to an emitted row label | `git grep -F '"<key>"' -- src/` per key, then broad case-insensitive re-check on the zero-hit ones | 21 keys map to real emitted labels (`report_stat_rows.py`: "Test" :321, "Model type" :322, "Statistic" :506, "p-value" :507, "Transformation" :389, "Post-hoc test" :390, "Sample size (n)" :645, "Correlation method" :648, "Interpretation" :651, etc.). **3 keys emit nowhere** — orphans (below). |
| Are the 3 zero-hit keys built dynamically / under another case? | `git grep -niE` bare-phrase, plus read of `report_association.py` (253L) | No dynamic construction; `report_association.py` renders LR assumptions and β-interpretation as custom HTML and never calls `stat_row_info`. Confirmed orphan, not false-negative. |

### Findings

**GD15 — three orphan tooltip keys (dead copy).** `report_tooltips.py:45,72-75,93-95` (and the
`"Effect size type"` key at `:53`). No stat row is ever labeled `"Assumptions (Linear Regression)"`,
`"β Interpretation"`, or `"Effect size type"`, so `stat_row_info` can never return them:
- LR assumptions render as bespoke HTML in `report_association.py`, not as a labelled stat row.
- The β-interpretation is emitted as inline **value** text (`correlation_models.py:670,711`), not a
  row whose label is `"β Interpretation"`.
- The effect-size row is labelled by the metric's **name** (`html_exporter.py:286`,
  `effect_label = str(results.get("effect_size_type") or "Effect size")`), so the literal
  `"Effect size type"` is never a label.
**Impact:** none at runtime — unused, and the copy is accurate if a matching row were ever added.
Pure maintenance dead-weight, same class as GD2/GD13. **Severity: LOW.** **Fix:** delete the three
keys, or (better, if the intent is to document those outputs) wire the copy to the labels the report
actually emits.

**GD16 — effect-size row loses its tooltip when labelled by metric name.** `html_exporter.py:286`
labels the effect-size row with the metric name (e.g. "Cohen's d", "η²") and only falls back to the
literal "Effect size" when `effect_size_type` is empty. `stat_row_info("Cohen's d")` returns `""`, so
the row that carries a specific effect-size metric shows **no** tooltip, while the generic-labelled
fallback does. **Impact:** minor, cosmetic — the most informative rows are the ones missing the
help text. **Severity: LOW.** **Fix:** look the tooltip up by a stable key (e.g. always keep an
`"Effect size"` info regardless of the displayed metric name), or add per-metric entries.

### Strengths (verified)

- **The copy is technically sound and genuinely layperson-readable.** Spot-checks held up: the
  p-value tooltip correctly states it is *not* P(H₀ true); df₂ is correctly "N − k, not the number of
  groups"; the adjusted-p note names BH and correctly flags its independence/positive-dependence
  assumption. No content inaccuracy found across the 24 entries.
- **The riskiest, most specific claim is true.** The "Assumptions (Linear Regression)" tooltip names
  three exact tests and the code runs all three — no overstatement (contrast the R2 GD1/GD9 pattern
  where docs drifted from code).

> Note: the three open items from the Round-3 body (GD6, GD13, GD14) were fixed after this report was
> written — commits `5eb4984` (GD14), `445af4b` (GD6), `fce568d` (GD13). The remediation order below
> is retained as the as-audited snapshot; GD15/GD16 (this section) remain open.

## Recommended remediation order

1. **GD14** (two-string translation) — trivial, user-visible.
2. **GD13** (delete dead guard) — trivial; bundle with any dead-code sweep.
3. **GD6** (empty-selection guard) — one line, mirrors an existing pattern.

GD4 needs no fix (reclassified above). GD5 needs no code change. GD7 remains a non-bug. All three
open items are LOW; none gates the release.
