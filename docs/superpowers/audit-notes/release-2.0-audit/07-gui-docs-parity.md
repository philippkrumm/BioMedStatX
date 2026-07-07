# AUDIT: BioMedStatX — GUI Dialogs + Documentation Parity (round 2) @ 3fd4796

**Scope:** `src/ui/dialogs/plot_aesthetics_dialog.py` (1920 lines), `src/ui/dialogs/statistical_analyzer_dialogs.py`
(1043 lines), `src/ui/dialogs/comparison_selection_dialog.py` (86 lines), `src/ui/components/decision_tree_view.py`
(424 lines), `src/ui/components/tutorial_overlay.py` (373 lines), `src/core/help_content.py` (654 lines),
root `CHANGELOG.md`, `README.md`, `CLAUDE.md`.

**Verdict.** No CRITICAL or live-emergency findings. `git log --oneline b16cf24..HEAD -- <every file in this
batch>` returns **zero commits** — this batch's code is byte-identical to what the round-1 audit
(`docs/superpowers/audit-notes/release-2.0-audit-round1/07-gui-docs-parity.md`) reviewed. I re-verified
round 1's seven findings (GD1–GD7) directly against current source rather than trusting the prior write-up;
all seven are confirmed still present, unchanged, at the same locations. Beyond that re-verification, this
pass found **two new, previously unreported bugs** with concrete reachability: an `AttributeError` on a
dead code path in `HelpHubDialog` (`self.copy_button` referenced but never defined), and a `get_config()`
early-return in `PlotAestheticsDialog` that silently drops `file_name`/`create_plot`/`dependent` from the
returned config on an invalid-filename warning instead of blocking accept or fully populating the dict. I
also found a **new recurrence of the established "untranslated German string" bug class**, one hop outside
this batch's files but directly reached from it, plus a **new CHANGELOG-adjacent Help Hub staleness** item
(the `linear_regression` recipe now undersells what the report actually shows, following a same-session
CHANGELOG-driven feature commit), and a **README internal self-contradiction** on launcher script filenames
that round 1's CHANGELOG spot-check didn't surface. No untranslated label was found inside any widget text
in this batch's own dialog/component files — the only recurrence is adjacent, not inside the audited files.

## What I mechanically verified (not eyeballed)

| Check | Command / method | Result |
|---|---|---|
| Any commits touching this batch since round 1's commit (`b16cf24`) | `git log --oneline b16cf24..HEAD -- <9 files>` | **0 commits.** Code is byte-identical to round 1; this is a genuine independent re-read, not a re-audit of changed code. |
| Round-1 GD1 (mixed_anova recipe "conservative correction" overstatement) still present | Read `help_content.py:282` | Verbatim identical to round 1's citation. Still present. |
| Round-1 GD2 (`ColumnSelectionDialog` dead code contradicting CLAUDE.md) still present | `git grep -n "ColumnSelectionDialog" src/ tests/` | Only 2 hits: class def (`statistical_analyzer_dialogs.py:390`) + its own `ValueError` string. Zero instantiation sites. Confirmed still dead. |
| Round-1 GD3 (no keyboard zoom in decision tree) still present | `git grep -nE "keyPressEvent\|Key_Plus\|Key_Minus" src/ui/components/decision_tree_view.py` | No matches — confirmed no keyboard zoom/reset binding exists. |
| Round-1 GD6 (`ComparisonSelectionDialog` allows empty-selection OK with no warning) still present | Read `comparison_selection_dialog.py:81-86` in full | `get_selected_comparisons()` has no `if not selected: QMessageBox.warning(...)` guard, unlike its siblings `GroupSelectionDialog`/`ColumnSelectionDialog`. Confirmed. |
| Round-1 GD5 (CLAUDE.md `_ap_*` line-number drift) still present | `git grep -n "^def _ap_browse_file\|^def _ap_load_file\|^def _ap_load_sheet" src/autopilot/statistical_analyzer_autopilot_pipeline.py` | Actual lines 930/940/1025 vs. CLAUDE.md's cited 822/832/875. Confirmed unchanged from round 1. |
| `ComparisonSelectionDialog` → `advanced_pipeline.py` "paired_custom" reachability claim in CLAUDE.md | `git grep -n "ComparisonSelectionDialog\|custom_pairs_callback" src/` | Import + instantiation only at `analysis_core.py:893,897` inside `_custom_pairs_cb`; the callback itself is threaded through `perform_advanced_test` and consumed by `advanced_posthoc.py:108-114`. **Zero references to `ComparisonSelectionDialog` or `custom_pairs_callback` anywhere in `advanced_pipeline.py`.** CLAUDE.md's claim that the dialog is reached "via `advanced_pipeline.py`" is imprecise — the actual chain is `analysis_core.py` → callback → `advanced_posthoc.py`. |
| Dialog return-shape vs. consumer expectation for `ComparisonSelectionDialog` | Read `comparison_selection_dialog.py:81-86` (`get_selected_comparisons` returns `list[tuple[str,str]]`) against `advanced_posthoc.py:118-123` (`normalize_pair`/`set` wrapping) | Shapes match: caller normalizes into a sorted-tuple `set` downstream, so the dialog itself doesn't need to pre-sort. No writer/reader drift. |
| Hardcoded palette WCAG contrast ratios (journal palettes, default colors, decision-tree text) | Computed relative luminance + contrast ratio in Python for every hex value in `plot_aesthetics_dialog.py`'s `journal_palettes` dict and both `default_colors` lists, and `decision_tree_view.py`'s dark/light node/text colors | See **GD8** and **Strengths** below for the full numeric breakdown. |
| Untranslated non-English strings in this batch's actual UI-facing text (labels/tooltips/titles/button text) | `git grep -nE '"[^"]*[äöüÄÖÜß][^"]*"'` and a German-keyword heuristic across all 6 files | Only docstrings/comments matched (e.g. `"""Tab für Farbeinstellungen"""`, `# Sicherstellen, dass Farben immer gesetzt sind`) — **zero hits inside an actual `QLabel`/`setText`/`setWindowTitle`/`setToolTip`/button-text string** in this batch. The previously-fixed German checkbox label bug class does **not** recur inside these 6 files. |
| Adjacent recurrence of the German-label bug class (one hop outside this batch) | Traced `_ap_detected_test_label` (`statistical_analyzer_autopilot_pipeline.py:1311-1326`), reached via `_ap_format_rationale` → `mapping_feedback_label` | `"correlation": "Korrelationsanalyse (Spearman/Pearson)"` and `"linear_regression": "Lineare Regression (OLS)"` are German, while every other of the 10 sibling dict entries is English. Confirmed user-facing (feeds the "Structure inferred as ..." rationale text shown on the main window). File is outside this batch's assigned scope (belongs to the autopilot-pipeline batch), flagged here only because it's the exact bug class this audit was asked to hunt for and was found one hop from `statistical_analyzer_dialogs.py`. |
| `HelpHubDialog.copy_button` reference | `git grep -n "copy_button" src/ui/dialogs/statistical_analyzer_dialogs.py` | Single hit: `self.copy_button.setEnabled(False)` at line 382, inside `_update_recipe_view`'s `if not recipe:` branch. **No `copy_button` is ever constructed anywhere in the class.** Confirmed `AttributeError` on that branch. |
| Reachability of the `copy_button` crash branch today | Cross-referenced `_populate_recipe_list` (sets `Qt.UserRole` to `recipe["id"]` for all 12 `HELP_RECIPES` entries) against `_recipe_by_id` (built from the same list, `{recipe["id"]: recipe for recipe in self._recipes}`) | Currently unreachable — every populated list item's `id` is guaranteed present in `_recipe_by_id` given today's 12-recipe data. Confirmed dead-but-latent: a future recipe with a duplicate/missing `id`, or any future edit to `_recipe_by_id`'s construction, makes this a live crash. |
| `PlotAestheticsDialog.get_config()` early-return on invalid filename | Read `plot_aesthetics_dialog.py:1866-1879` in full | `if raw_name and _re.search(r'[<>:"/\\\|?*]', raw_name): QMessageBox.warning(...); return config` — returns before `config['file_name']`, `config['create_plot']`, `config['dependent']` are set. Confirmed by reading the full function body. |
| Actual caller-side blast radius of that early return | Read both call sites: `statistical_analyzer.py:370` and `statistical_analyzer_autopilot_pipeline.py:1868-1917` | Autopilot caller uses `.get()` with safe defaults for every affected key (`plot_config.get("create_plot", True)`, `plot_config.get("file_name")` falls back to an auto-generated name, `context.get("dependent", False)` — not read from `plot_config` at all). **Net effect: not a crash** — the dialog silently discards the exact filename the user just saw flagged as invalid and substitutes the auto-generated name instead of blocking OK or re-prompting. Confirmed MEDIUM, not HIGH. |
| Test coverage for the two new bugs and `comparison_selection_dialog.py` | `git grep -l "ComparisonSelectionDialog\|get_config" tests/` plus per-file `test_` counts | Zero tests reference `ComparisonSelectionDialog` in `tests/`, and no test drives `PlotAestheticsDialog.get_config()`'s invalid-filename branch. Existing coverage: `test_help_hub.py` (14 tests), `test_decision_tree_graphics.py` (5), `test_decision_tree_posthoc_mapping.py` (4), `test_tutorial_overlay.py` (7), `test_tutorial_onboarding_app.py` (5), `test_plot_aesthetics_log_gating.py` (2). |
| `except Exception`/bare `except:` census across this batch | `git grep -nE "except Exception\|except:"` per file | 4 total: `plot_aesthetics_dialog.py` (2, incl. one bare `except:` at line 229), `statistical_analyzer_dialogs.py` (1), `tutorial_overlay.py` (1). `comparison_selection_dialog.py` and `decision_tree_view.py`: 0. All four fall back to a clearly-labeled safe default or log a warning — no fault-swallowing that masks a real failure with a silently-wrong result. |
| `raise ValueError` census | `git grep -nE "raise ValueError\|raise [A-Za-z_]*Error\("` per file | 4, all in `statistical_analyzer_dialogs.py` (`GroupSelectionDialog`, `ColumnSelectionDialog`, `PairwiseComparisonDialog`, `TwoWayAnovaDialog` constructors — "too few groups/columns" guards). Only `GroupSelectionDialog` is ever instantiated (`autopilot_pipeline.py:686`), and its caller pre-checks the count, so the guard is defense-in-depth, not a live crash path. |
| CHANGELOG claim spot-check (linear regression coefficient rendering, new since round 1's CHANGELOG pass) | `git show b16cf24 --stat` + read `report_association.py`'s new `_build_linear_regression_coefficient_table_html` | Confirmed: a full per-coefficient table (parameter, coefficient, SE, t, p, 95% CI for every predictor/covariate, not just the primary one) was added to the HTML export in the same session. See **GD9**. |
| README launcher-script filename consistency | `git grep -n "start\.sh\|start\.bat\|run\.bat\|Start_BioMedStatX" README.md docs/SCRIPTS.md` + `ls *.sh *.bat` | Actual files on disk: `start.sh`, `run.bat`. `README.md:76` correctly names both. `README.md:162-163` ("Repository Structure" tree) instead lists `Start_BioMedStatX_on_Linux.sh` and `start.bat` — **neither exists**, and the section contradicts the correct names 86 lines earlier in the same file. See **GD10**. |

## Findings — severity ranked

### HIGH

**GD1 — (round 1, reconfirmed unchanged) The `mixed_anova` Help Hub recipe asserts a "conservative correction" guarantee that does not hold for Mixed ANOVA.**
`src/core/help_content.py:282`. Identical citation and reasoning as round 1's GD1: a parallel audit
(`02-statistical-core-dispatch.md`, finding SC1) confirmed Mixed ANOVA's sphericity correction is computed
and shown in the effects table but never rewrites the canonical `results["p_value"]` that gates the
significance verdict and post-hoc dispatch — unlike RM-ANOVA, which does get that rewrite. This recipe
paragraph is word-for-word the RM-ANOVA claim, applied to a design where it isn't true yet.
**Impact:** unchanged from round 1 — a user is told the app applies a conservative correction for Mixed
ANOVA specifically, when in the one case that matters, the significance decision is based on the
uncorrected p-value.
**Fix:** unchanged from round 1 — mirror RM-ANOVA's `results["p_value"]` rewrite for Mixed ANOVA (the
actual code fix, scoped in `02-statistical-core-dispatch.md`), or soften `help_content.py:282` as an interim
measure.

### MEDIUM

**GD2 — (round 1, reconfirmed unchanged) `ColumnSelectionDialog`'s "multi-dataset" checkbox is dead code contradicting CLAUDE.md's own "already removed" claim.**
`src/ui/dialogs/statistical_analyzer_dialogs.py:390-449`. Still zero instantiation sites in `src/` or
`tests/`. Unchanged from round 1.
**Fix:** unchanged — delete the class and its orphaned `get_selected_columns`/`multi_dataset_check`, or
document the surviving caller in CLAUDE.md if one is intended.

**GD3 — (round 1, reconfirmed unchanged) `InteractiveDecisionTreeWidget` has no keyboard-accessible zoom or reset-view control.**
`src/ui/components/decision_tree_view.py:404-415`. Zoom remains wheel-only; no `keyPressEvent` override
exists anywhere in the class. Unchanged from round 1.
**Fix:** unchanged — bind `+`/`-`/`0` to zoom-in/zoom-out/refit when focused.

**GD8 — Several "journal palette" preset colors fail even the 3:1 non-text contrast floor against a white plot background, despite being presented as publication-safe defaults.**
`src/ui/dialogs/plot_aesthetics_dialog.py:388-393` (`ColorsTab.journal_palettes`). Computed WCAG relative-luminance
contrast ratios against `#ffffff` for all 29 hex values across the four bundled journal palettes:
- **Nature**: 4 of 8 colors fail the 3:1 floor outright (`#F28E2B`→2.42, `#76B7B2`→2.29, `#EDC948`→1.61,
  `#FF9DA7`→1.98); only 1 of 8 (`#4E79A7`→4.55) clears the 4.5:1 body-text floor.
- **Science**: 3 of 7 fail 3:1 (`#56B4E9`→2.31, `#E69F00`→2.25, `#999999`→2.85).
- **NEJM**: 2 of 7 fail 3:1 (`#E18727`→2.73, `#FFDC91`→1.32).
- **Lancet**: 2 of 7 fail 3:1 (`#42B540`→2.65, `#FDAF91`→1.79).
These are plot data-series colors (bar/box/violin fills), not app-chrome text, so WCAG's SC 1.4.3/1.4.11 do
not strictly bind them the way they would an interactive control. But the dialog explicitly brands these as
named-journal presets a user picks *for* publication legibility (`ColorsTab.__init__`, "Professional palettes
- excluding rainbow/childish ones"), and several of the bundled defaults (`#EDC948`, `#FFDC91`, `#FDAF91`) are
low-contrast pastel/yellow tones that are genuinely hard to read as small swatches, legend chips, or thin
lines on a white figure background — independent of the WCAG technicality, this undermines the feature's
own stated purpose.
**Impact:** a user relying on a "Nature"/"NEJM" preset for a publication figure can end up with a
hard-to-distinguish color (e.g., pale yellow `#FFDC91` at 1.32:1) with no in-app warning, silently
undermining the "publication-ready plots" claim in README.md.
**Fix:** either swap the lowest-contrast entries in each palette for a same-hue darker/more saturated
variant that still matches the named journal's real house style, or add a small on-swatch contrast
indicator (a la a "low legibility" hint) when a selected color falls under ~3:1 against the current plot
background.

**GD9 — The `linear_regression` Help Hub recipe is now stale: it undersells the coefficient table the report gained this session.**
`src/core/help_content.py:488-490` ("Reading the result") states only: *"The main output is the coefficient
(β) for the primary predictor, with its p-value... The report also shows R²."* Commit `b16cf24` (same
session, landed after this recipe was last edited) added `_build_linear_regression_coefficient_table_html`
(`src/export/report_association.py`), which renders a full multi-row table — parameter, coefficient, SE, t,
p-value, 95% CI lower/upper — for **every** predictor and covariate, not just the primary one. This directly
closes SUMMARY.md's pre-2.0 item 5 ("coefficient_table computed but never rendered") but the Help Hub text
was never revisited afterward, so it now describes yesterday's, more limited report.
**Impact:** a user running multi-covariate regression (the exact case this feature was built for) now gets
more information in the report than the in-app Help Hub tells them to expect — a documentation-lags-code
gap in the good direction, but still inaccurate and worth fixing before 2.0 ships alongside the feature that
caused it.
**Fix:** add one sentence to the "Reading the result" section: the exported report also lists a full
coefficient table (SE, t, 95% CI) for every predictor and covariate, not only the primary one.

**GD10 — README.md contradicts itself on the launcher script filenames, in a way round 1's CHANGELOG-only doc check didn't catch.**
`README.md:76` correctly links `` [`start.sh`](./start.sh) `` and `` [`run.bat`](./run.bat) ``, matching both
the files actually on disk and `docs/SCRIPTS.md`'s own naming. But the "Repository Structure" tree 86 lines
later, `README.md:162-163`, lists:
```
├─ Start_BioMedStatX_on_Linux.sh  # Launcher for Linux/macOS source/binary startup
├─ start.bat                      # Launcher for Windows source/binary startup
```
Neither `Start_BioMedStatX_on_Linux.sh` nor `start.bat` exists anywhere in the repository (`ls *.sh *.bat`
shows only `start.sh` and `run.bat`). This is not merely stale relative to code — it's an internal
contradiction within the same document.
**Impact:** a developer/contributor skimming the Repository Structure section (a natural first stop) for the
launcher script name will look for a file that doesn't exist, and the correct name is not visually
next to the wrong one to self-correct.
**Fix:** update the tree at `README.md:162-163` to `start.sh` and `run.bat`, matching line 76 and the real
files on disk.

**GD11 — `HelpHubDialog._update_recipe_view`'s error branch throws `AttributeError` on `self.copy_button`, which is never defined anywhere in the class.**
`src/ui/dialogs/statistical_analyzer_dialogs.py:378-383`:
```python
recipe = self._recipe_by_id.get(recipe_id)
if not recipe:
    self._current_recipe = None
    self.recipe_title.setText("Recipe not found")
    self.recipe_browser.setHtml("<p>Recipe content is unavailable.</p>")
    self.copy_button.setEnabled(False)
    return
```
`git grep -n "copy_button" src/ui/dialogs/statistical_analyzer_dialogs.py` returns exactly this one line —
`copy_button` is never constructed in `__init__`/`_populate_recipe_list`/anywhere else in `HelpHubDialog`.
Currently unreachable: every list item's `Qt.UserRole` is set to a real `recipe["id"]` drawn from the same
12-entry `HELP_RECIPES` list that builds `_recipe_by_id`, so `recipe` is never falsy today. But this is
exactly the kind of defensive branch meant to catch a data problem (a bad/missing recipe id) — and the
defensive code itself is broken, so the moment it's ever needed (a future recipe entry with a typo'd or
duplicate `id`, or any refactor of `_recipe_by_id`'s construction), the fallback path crashes instead of
degrading gracefully.
**Impact:** currently zero (dead code), but it inverts the intent of a defensive branch — the safety net has
a hole in exactly the spot it exists to catch.
**Fix:** either remove the line (the two lines above it already handle the "not found" UI state acceptably),
or add the `copy_button` if a copy-to-clipboard feature was intended and got dropped mid-refactor — `git
blame` the line to check for a removed sibling feature before deciding which.

**GD12 — `PlotAestheticsDialog.get_config()`'s invalid-filename path returns a config dict silently missing `file_name`, `create_plot`, and `dependent`, instead of blocking the OK action or fully populating a corrected dict.**
`src/ui/dialogs/plot_aesthetics_dialog.py:1868-1879`:
```python
if hasattr(self, 'file_name_edit') and self.file_name_edit is not None:
    raw_name = self.file_name_edit.text().strip()
    if raw_name and _re.search(r'[<>:"/\\|?*]', raw_name):
        QMessageBox.warning(self, "Invalid filename", ...)
        return config
    config['file_name'] = raw_name or None
config['create_plot'] = True
config['dependent'] = self.dependent
return config
```
The dialog already accepted (`dialog.exec_() == Accepted`) by the time `get_config()` runs
(`statistical_analyzer_autopilot_pipeline.py:1865-1868`), so the warning box fires but the workflow
continues anyway with a config dict that's missing three keys the rest of `get_config()` always sets on the
valid path. The only caller (`_ap_configure_plot_from_result` in the autopilot pipeline) happens to `.get()`
every affected key with safe fallbacks (auto-generated filename, `create_plot` defaulting `True`, `dependent`
read from `context` rather than `plot_config` at all) — so this is not a crash today, and not silent data
loss for `dependent`. But the user's exact typed filename, the one they just saw flagged as invalid, is
silently discarded and replaced with an auto-generated name with no further prompt, chance to fix it, or
indication that their choice was dropped.
**Impact:** a validation warning that doesn't actually block progress or offer a corrected value reads to
the user as "the app is complaining but doing something anyway" — the warning's only visible effect is that
their filename choice silently vanishes.
**Fix:** either (a) don't return early — sanitize/strip the invalid characters and continue with a corrected
`file_name`, or (b) if blocking is intended, don't call `accept()`-equivalent completion — re-show the
dialog or focus/select the offending field instead of returning a partial dict the caller has to
`.get()`-shield against.

### LOW

**GD4 — (round 1, reconfirmed unchanged) `ExploratoryMatrixDialog`'s multi-select variable list has an undiscoverable keyboard interaction.**
`src/ui/dialogs/statistical_analyzer_dialogs.py:931-939`. Unchanged from round 1 — `QListWidget.MultiSelection`
requires Space-per-item with no on-screen hint.
**Fix:** unchanged — add a one-line hint, or switch to `ExtendedSelection` with visible checkboxes.

**GD5 — (round 1, reconfirmed unchanged) CLAUDE.md's `_ap_*` pipeline line-number citations have drifted, and the "paired_custom → advanced_pipeline.py" reachability claim is imprecise.**
`_ap_browse_file`/`_ap_load_file`/`_ap_load_sheet` are cited at pipeline:822/832/875; actual current lines
are 930/940/1025 (unchanged from round 1). This pass additionally re-confirmed CLAUDE.md's separate claim
("The 'paired_custom' post-hoc branch reaches `src/ui/dialogs/comparison_selection_dialog.py` via a UI
dialog... control flows through `advanced_pipeline.py`") is loosely worded: `advanced_pipeline.py` itself
never references `ComparisonSelectionDialog` or the `custom_pairs_callback` that carries its result — the
actual chain is `analysis_core.py` (imports + instantiates the dialog, defines the callback) →
`StatisticalTester.perform_advanced_test` (threads the callback through) → `advanced_posthoc.py` (invokes
the callback and normalizes its return value). The architectural substance (mixin binding, method list) is
still accurate; only the specific line numbers and this one reachability sentence have drifted.
**Fix:** unchanged recommendation — drop line numbers from prose (name functions, not lines), and reword the
paired_custom sentence to name `analysis_core.py` as the actual dialog call site.

**GD6 — (round 1, reconfirmed unchanged) `ComparisonSelectionDialog.get_selected_comparisons()` allows an empty confirm with no warning.**
`src/ui/dialogs/comparison_selection_dialog.py:81-86`. Still no `QMessageBox.warning` guard for the
all-unchecked case, unlike `GroupSelectionDialog`/`ColumnSelectionDialog`. Unchanged from round 1. Also
still has zero test coverage (`git grep -l "ComparisonSelectionDialog" tests/` → no hits).
**Fix:** unchanged — add the same empty-selection guard used by sibling dialogs.

**GD7 — (round 1, reconfirmed unchanged) `_apply_log_scale_gating`'s tooltip updates regardless of whether "Log Y" is checked.**
`src/ui/dialogs/plot_aesthetics_dialog.py:1580-1600`. Not a bug (advisory tooltip framing is correct either
way), just worth a comment. Unchanged from round 1.

**GD13 — `PlotAestheticsDialog`'s autopilot caller guards a `create_plot_check` attribute that doesn't exist anywhere on the dialog.**
`src/autopilot/statistical_analyzer_autopilot_pipeline.py:1862-1863`: `if hasattr(dialog, 'create_plot_check'): dialog.create_plot_check.setChecked(False)`. `git grep -n "create_plot_check" src/` returns only these two lines — `PlotAestheticsDialog` has no such attribute (its actual plot-creation toggle, if any, is `config['create_plot']`, set unconditionally `True` in `get_config()`). Harmless today because of the `hasattr` guard (a documented no-op), but it's a piece of dead defensive code pointing at a feature/attribute that no longer exists, in the same file family as GD2's dead `ColumnSelectionDialog`.
**Fix:** delete the dead `hasattr` guard, or wire an actual `create_plot_check` checkbox into `PlotAestheticsDialog` if a "preview only, don't render" toggle was intended.

**GD14 — Adjacent recurrence (outside this batch's files) of the previously-fixed "untranslated German string" bug class.**
`src/autopilot/statistical_analyzer_autopilot_pipeline.py:1324-1325`, inside `_ap_detected_test_label`'s
`labels` dict: `"correlation": "Korrelationsanalyse (Spearman/Pearson)"` and `"linear_regression": "Lineare
Regression (OLS)"` are German, while the other 10 entries in the same dict (`"Independent t-test"`, `"One-Way
ANOVA"`, `"Linear Mixed Model (handles missing visits)"`, etc.) are English. This function feeds
`_ap_format_rationale`'s `"Structure inferred as {label}."` string, rendered on the main window's
`mapping_feedback_label` — so any user running a Correlation or Linear Regression analysis sees a
German clause in an otherwise all-English rationale. This is the identical bug class the prior audit round
already found and fixed once (the `"Als Lineare Regression analysieren..."` checkbox, confirmed still fixed
in this pass — see the mechanized check above). Flagged here at LOW/informational because the responsible
file (`statistical_analyzer_autopilot_pipeline.py`) belongs to a different subsystem's audit batch, not this
one — surfaced only because the audit brief specifically asked to hunt for recurrences of this bug class,
and this one was found one hop from `statistical_analyzer_dialogs.py` while tracing the Help Hub/rationale
UI.
**Fix:** translate both entries to English (`"Correlation (Spearman/Pearson)"`, `"Linear Regression (OLS)"`),
matching the other 10 entries' style — trivial, same fix shape as the already-fixed checkbox label.

## Strengths (verified)

- **This batch's own 6 files are fully clean of the untranslated-string bug class.** A regex sweep for
  quoted strings containing German umlauts/ß, plus a German-keyword heuristic, across all 6 files found
  matches **only** in docstrings and comments (e.g. `"""Tab für Farbeinstellungen"""`,
  `# Sicherstellen, dass Farben immer gesetzt sind`) — zero hits inside any `QLabel` text, `setToolTip`,
  `setWindowTitle`, or button label. The one recurrence found (GD14) is one file outside this batch.
- **`decision_tree_view.py`'s text/background contrast passes WCAG 2.2 AA comfortably in both themes,
  computed exactly, not eyeballed.** Dark mode: active node text `#2dd4bf` on node bg `#133835` → 6.86:1;
  inactive text `#8ba4ac` on node bg `#162428` → 6.08:1; both well above the 4.5:1 body-text floor. Light
  mode: active text `#0f766e` on white → 5.47:1; inactive text `rgba(22,49,58,0.68)` flattened over white →
  4.96:1. Both themes clear AA with margin.
- **Every dialog in both dialog files uses `QDialogButtonBox(Ok | Cancel)`**, giving correct
  Enter-accepts/Escape-cancels keyboard behavior for free — verified across all 7 dialog classes in
  `statistical_analyzer_dialogs.py` plus `PlotAestheticsDialog` and `ComparisonSelectionDialog`.
- **`TutorialOverlay` remains a genuinely well-built accessible component** (unchanged from round 1):
  correct native OS reduced-motion probing on macOS/Windows/Linux with an explicit env override and a
  fail-safe `except Exception: return False`; full keyboard support (Escape/Enter/Space/arrows); a keyboard
  grab plus an `eventFilter` that swallows stray background key presses so a focused button behind the
  overlay can't be accidentally triggered.
- **The dialog return-shape contract between `comparison_selection_dialog.py` and its consumers is
  correct end-to-end**, confirmed by reading both sides: `get_selected_comparisons()` returns
  `list[tuple[str, str]]` in whatever order the user's checkboxes happen to be in;
  `advanced_posthoc.py`'s `normalize_pair`/`set(...)` wrapping normalizes ordering downstream, so the
  dialog doesn't need to pre-sort. No writer/reader key-contract drift found on this path.
- **CHANGELOG claim re-spot-checked in this pass (Beta regression's omnibus LR p-value) still holds
  exactly as round 1 found it**: `clinical_models.py:1706-1707` uses `self.result.llr_pvalue` as `main_p`.
  Combined with round 1's 6 other verified CHANGELOG claims, this is a well-maintained CHANGELOG for a 2.0
  release — the one Help Hub-adjacent gap that exists now (GD9) is new drift from a same-session feature
  commit, not a pre-existing inaccuracy.
- **`help_content.py`'s `graph_visualization` recipe text matches the live `plot_aesthetics_dialog.py`
  dropdown contents exactly**, checked item-by-item: plot types (Bar/Box/Violin/Raincloud), error metrics
  (sd/se/ci), error styles (caps/line), and point layouts (Jitter/Beeswarm/Strip) all match the recipe's
  prose one-for-one.
- **The `CorrelationModel` docstring fix from the prior Help Hub content audit (`eaaf8e4`) is correctly
  landed** — the docstring now accurately describes the real skew/excess-kurtosis N-tier gating instead of
  the old (wrong) Shapiro-Wilk-p-value claim, confirmed by reading both the docstring and `fit()`'s actual
  logic side by side.

## Recommended remediation order

1. **GD1 (mixed_anova recipe overstatement, HIGH, unchanged)** — still the highest-value fix; same
   recommendation as round 1: land the small code fix that mirrors RM-ANOVA's `results["p_value"]` rewrite,
   which fixes the code and this doc claim together.
2. **GD12 (`get_config()` silently drops the user's filename)** — cheap, well-scoped fix (stop the early
   `return` before all keys are set, or sanitize-and-continue); currently masked by defensive `.get()` calls
   in the only caller, but fragile if a second caller is ever added without the same defaults.
3. **GD11 (`copy_button` `AttributeError` on a dead defensive branch)** — trivial one-line fix (delete the
   stray line or wire the intended feature); zero current risk but cheap to close before it becomes live.
4. **GD10 (README launcher filename self-contradiction)** — trivial doc fix, two lines, removes an internal
   inconsistency a contributor could hit on day one.
5. **GD9 (linear_regression recipe undersells the new coefficient table)** — one added sentence, keeps docs
   in lockstep with the same-session `b16cf24` feature.
6. **GD2 / GD13 (dead `ColumnSelectionDialog` + dead `create_plot_check` guard)** — bundle both as one
   cleanup pass; same root cause (leftover surface from a removed workflow).
7. **GD6 (`ComparisonSelectionDialog` empty-selection guard)** — one-line, mirrors an existing pattern.
8. **GD5 (CLAUDE.md line-number + reachability wording drift)** — cheap doc fix; consider naming functions
   instead of lines to stop this recurring every release.
9. **GD14 (adjacent German-string recurrence)** — trivial two-string translation; hand off to whichever
   batch owns `statistical_analyzer_autopilot_pipeline.py`, since it's outside this batch's file scope.
10. **GD8 (journal palette low-contrast entries)** — needs a small design decision (swap colors vs. add a
    legibility hint), not just a doc fix; scope as a short follow-up.
11. **GD4 / GD3 (keyboard-discoverability items)** — UX polish, no correctness risk; bundle into any future
    accessibility pass.
12. **GD7 (tooltip-vs-checked-state comment)** — trivial, opportunistic.
