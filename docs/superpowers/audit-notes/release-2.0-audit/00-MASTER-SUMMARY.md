# Pre-2.0 Release Audit — Master Summary (Round 2)

Date: 2026-07-07. Repo state audited: `3fd4796` (branch `feature/advanced-stats-automation`).
Method: 7 parallel subsystem audits, each using the `senior-engineering-partner` skill's
`AUDIT:` mode (report-first, `file:line` evidence, severity-ranked, mechanized checks — see
`~/.claude/skills/senior-engineering-partner/references/audit-report-format.md`), per the
reusable template at `docs/superpowers/audit-notes/senior-engineering-partner-audit-prompt-template.md`.
Total: ~38,000 lines across 57 `src/` files, plus root docs. This is a **second, fully
independent** pass over the same subsystems audited in round 1
(`docs/superpowers/audit-notes/release-2.0-audit-round1/`, commit `b16cf24`) — every batch
re-derived its findings from live source rather than trusting the prior write-up, and
explicitly re-verified round-1's own findings for regressions/fixes along the way.

**Process note (transparency):** the first dispatch of all 7 batches hit a Claude usage
limit mid-run and produced zero report files (confirmed by directory listing before
redispatch). All 7 were redispatched cleanly after the limit reset; one of the seven
(report/export layer) died a second time to a transient API-overload error before writing
anything and was redispatched a third time successfully. Every batch's top finding(s) were
independently spot-verified against live source (grep/read, quoted below) by the orchestrating
session before being included here — not accepted on any sub-agent's word alone.

**Overall verdict.** No CRITICAL findings. No live security emergency (local single-user
desktop app — `references/my-environment.md`). Five of seven batches (UI bridge, specialized
models, visualization, report/export, GUI/docs) found **zero commits** touching their files
since round 1, meaning round 1's recommended fixes have not yet been applied — this round is
a genuine independent re-derivation, not a check of new code, and every one of round 1's
HIGH/MEDIUM findings in those five batches is confirmed still open, unregressed. The other two
batches (statistical core, advanced testing engines) sit downstream of code that *did* change
between rounds (5 new `ModelDesignError` pre-flight checks, ANCOVA `control_group` wiring) —
both fixes hold under fresh, independent re-verification (direct repro, not just inspection).
The single most important finding, below, is a **statistical-correctness bug independently
found by three separate audits from three different files** — the same bug class round 1
found and partially fixed for RM-ANOVA, never propagated to its Mixed-ANOVA sibling. A
reproducible HTML/script-injection path in the exported report (round 1's EX1) is also still
open. New this round: the app's global crash handler itself throws before it can show the
user the crash dialog.

---

## The one finding to fix first: Mixed ANOVA sphericity correction still doesn't gate the verdict — and the correction itself is currently unbuildable from this pingouin version's output

Found independently by **three separate batches** (statistical core dispatch, advanced testing
engines, GUI/docs parity), converging on the same root cause from three different files, plus
a fourth angle that makes the fix non-trivial:

- `src/analysis/statisticaltester.py:1548-1549` sets the canonical `results["p_value"]` from
  the **uncorrected** pingouin interaction row (`p_unc`), for both the top-level verdict and
  every within-factor/interaction row in `results["factors"]`/`results["interactions"]`.
- `_test_mixed_anova_within_sphericity` (`src/statistical_testing/mixed_assumptions.py:414-499`)
  computes a GG/HF-corrected `final_p_value` — but it is merged into `results` as a nested key
  (`within_sphericity_corrections.main_effect.final_p_value`) and **never** overwrites
  `results["p_value"]`. Contrast: RM-ANOVA's sibling code (`statisticaltester.py:1973-1974`,
  spot-verified this session) has the exact rewrite, tagged `# E1: write the correction-selected
  p-value back to the canonical field` — proving the fix pattern is known and works, just never
  propagated to Mixed ANOVA.
- **New this round (AT2, batch 4):** live-verified against the installed pingouin 0.6.1 that
  `mixed_anova()` never returns the `GG-eps`/`p-GG`/`HF-eps`/`p-HF`/`W-spher`/`p-spher` columns
  `_apply_corrections_to_effect_row`/`_extract_mixed_sphericity_from_anova_table` look for
  (only RM-ANOVA's table has those). So even wiring `results["p_value"]` today would just
  relabel the *uncorrected* p-value as "corrected" — the correction has to be computed from
  scratch (pingouin does return a plain `eps` column; GG epsilon can be derived from it or from
  `pg.epsilon()` directly) before the wiring fix is meaningful. **Fix the computation before the
  wiring**, or the wiring fix creates false confidence.
- **New this round (AT3, batch 4):** the interaction-row correction path is separately dead —
  its row filter requires a literal `*` character in pingouin's `Source` column, which pingouin
  actually labels `"Interaction"` (no asterisk). Live-verified: the filter always returns empty.
- `src/core/help_content.py:282` (GD1, batch 7, reconfirmed unchanged from round 1) tells the
  user the app "applies a conservative correction rather than assuming [sphericity] holds" for
  Mixed ANOVA specifically — true for RM-ANOVA, not true here.

**Fix (scoped across AT1/AT2/AT3/SC2/GD1):** (1) compute a real GG/HF epsilon for Mixed ANOVA
from the `eps` column pingouin actually provides (or via `pg.epsilon()`), correcting `DF1`/`DF2`
and re-deriving the p-value with `scipy.stats.f.sf`; (2) fix the interaction-row filter to match
pingouin's real `"Interaction"` label; (3) only then mirror RM-ANOVA's write-back
(`statisticaltester.py:1973-1974`) into the Mixed-ANOVA branch; (4) add a regression test
mirroring `tests/test_sphericity_outer_exception.py`, scoped to `MixedAnovaAssumptionEngine`,
against a golden-R Mixed ANOVA fixture. Step (3) alone (without 1-2) would be worse than the
status quo — false confidence in a number that's still uncorrected. This also makes the Help
Hub recipe (GD1) true again with no separate doc edit needed once the code is fixed.

---

## HIGH findings (fix before 2.0)

| ID | Batch | One-line | file:line |
|---|---|---|---|
| **SC2/AT1/AT2/AT3/GD1** | Core, Testing Engines, GUI/Docs | Mixed ANOVA sphericity correction never reaches canonical `p_value`, and the correction itself can't currently be computed from this pingouin version's output (see above) | `statisticaltester.py:1548-1549`, `mixed_assumptions.py:414-499,614-725` |
| **U1** | UI Bridge | Global uncaught-exception handler itself raises `TypeError` (`logger.info(msg, file=sys.stderr)` — `file=` isn't a `logging` kwarg) before it can show the crash dialog; confirmed by direct repro against the app's real logger config — the crash dialog has never fired | `statistical_analyzer.py:501` |
| **SC1** | Core | `make_blocked_result()` called with a nonexistent `test_name=` kwarg at 4 sites; raises `TypeError` that's caught and replaces the intended "Mixed ANOVA requires two factors..." message with the Python signature error — verified end-to-end via `AnalysisManager.analyze()` | `analysis_core.py:914,921,960,1005` |
| **RE1** | Report Export | Real, reproduced HTML/script injection: unescaped `parameter` cell in 3 coefficient-table builders, rendered via `{{ chart.html \| safe }}` (deliberately bypasses Jinja's `autoescape=True`) — round 1's EX1, confirmed still unfixed (zero commits to `src/export/` since round 1) | `report_association.py:34,87,135` |
| **VZ1** | Visualization | `_add_significance_letters`/raincloud variant swallow ALL letters on any exception with zero on-canvas indication — round 1's VZ1, confirmed still unfixed | `datavisualizer.py:2314-2450` |
| **GD1** | GUI/Docs | Help Hub `mixed_anova` recipe asserts a guarantee that doesn't hold — documentation-side consequence of the finding above, reconfirmed unchanged from round 1 | `help_content.py:282` |

Note: SC2/AT1/AT2/AT3/GD1 are one underlying bug found four ways across three batches —
fixing the code also fixes the doc claim with no separate edit needed. RE1 and VZ1 are round
1's top two findings, both confirmed completely unfixed (zero commits to their files since
round 1) — not regressions, just not yet actioned.

## MEDIUM findings (worth doing, none urgent)

- **SC3** — ANCOVA's EMM contrast dicts use `"t"`/`"se"`; LMM's use `"statistic"`/`"std_err"`;
  the HTML export reads `"statistic"` — silently blanks ANCOVA's t-statistic column in every
  report.
- **SC4** — `posthoc_choice` referenced before assignment when the non-parametric post-hoc
  re-entry branch runs; raises `NameError` and aborts the analysis (confirmed by isolated
  repro of the exact control-flow shape).
- **U2** — Multi-DV batch mode computes the LMM-vs-RM-ANOVA test-family decision once from only
  `dv_columns[0]`'s missingness pattern, then silently reuses it for every other DV column via
  a shallow `dict(context)` copy — confirmed to reach the actual model dispatch unmodified.
- **SM1** — Outlier ingestion (`OutlierDetector._convert_values_to_float`) unconditionally
  treats `.` as a thousands separator whenever a value column isn't already numeric dtype —
  reproduced in Python: silently inflates plain US/international-formatted decimal strings by
  10-1000× with no error or warning. The one finding this round that most directly matches the
  audit brief's "silently produces a wrong-but-plausible result" bug class.
- **SM2** — Dunn-test bootstrap CI is a pure-Python O(n1×n2) nested loop; timed at ~13.5s per
  pair at n=500/group, extrapolating to ~80s for a realistic 4-group comparison, with no
  progress indicator and nothing off the Qt main thread.
- **AT4** — `logistic_regression` is explicitly excluded from the shared
  `validate_samples_for_test` pre-flight gate in `advanced_pipeline.py`, with no equivalent
  substitute — a perfectly-separated or zero-variance predictor level skips the safety net
  every other advanced test gets.
- **AT5** — `_perform_welch_anova`'s manual computation silently falls back to the non-robust
  standard F-test on ANY exception, relabeling it as `"welch_f_statistic"`/`"welch_p_value"`
  with no degradation flag the caller can detect.
- **VZ8** — Raincloud plots always render significance brackets instead of compact-letter
  display whenever any pairwise result exists, unlike Bar/Box/Violin (which correctly
  discriminate via `_result_uses_brackets`) — the same statistical result gets a different
  annotation style depending only on which plot type is picked in the dropdown.
- **VZ2/VZ3/VZ4** — round 1 carryovers, all confirmed still unfixed: silent bracket-count
  drops with no on-canvas notice; `logx`'s non-positive-data handling never brought up to
  `logy`'s symlog-auto-adapt standard; `DataVisualizer.DEFAULT_COLORS` still fails WCAG's 3:1
  floor for 2 of 6 colors.
- **RE2** — Inconsistent escaping of group/factor-level labels in Plotly `name=`/`hovertext`
  (one function escapes, four adjacent ones don't) — round 1's EX2, still unfixed.
- **GD8** — Several bundled "journal palette" preset colors (Nature/Science/NEJM/Lancet) fail
  even the 3:1 non-text contrast floor against white, undermining their branded
  "publication-ready" purpose — computed exactly (e.g. `#EDC948` → 1.61:1), not eyeballed.
- **GD9** — The `linear_regression` Help Hub recipe is now stale: it undersells the full
  per-predictor coefficient table the report gained this same session (the coefficient-table
  dead-code finding from round 1's SUMMARY.md item 5, now fixed in code but not in docs).
- **GD10** — README.md contradicts itself on launcher script filenames: line 76 correctly
  names `start.sh`/`run.bat`; the "Repository Structure" tree 86 lines later lists nonexistent
  `Start_BioMedStatX_on_Linux.sh`/`start.bat` — an internal contradiction, not just doc drift.
- **GD11** — `HelpHubDialog._update_recipe_view`'s error branch throws `AttributeError` on
  `self.copy_button`, which is never constructed anywhere in the class — a defensive branch
  whose own safety net is broken (currently dead, since every recipe ID is valid today).
- **GD12** — `PlotAestheticsDialog.get_config()`'s invalid-filename path returns early, silently
  omitting `file_name`/`create_plot`/`dependent` from the config dict instead of blocking or
  fully repopulating it — masked today only because the one caller defensively `.get()`s every
  affected key.
- **GD2/GD3** — round 1 carryovers, confirmed still unfixed: `ColumnSelectionDialog`'s
  "multi-dataset" checkbox is dead code contradicting CLAUDE.md's own "already removed" claim;
  the decision-tree viewer has no keyboard-accessible zoom/reset (wheel-only).

## LOW findings (track, no urgency)

U3 (correction to round-1 P1: the `self.df` Smithson-Verkuilen mutation self-limits after one
application rather than compounding, but the underlying silent-mutation-from-preview-path
design flaw is still real), U4 (dead window-geometry code, masked by startup maximize), U5
(bare `except:` around stylesheet loading), SC5 (`AdvancedPostHocEngine` invoked a second,
no-op time for ANCOVA/LMM/Logistic Regression via a currently UI-unreachable path), SC6
(contract drift on that same dead-for-UI-but-live-for-API path — bare error dict instead of
`make_blocked_result()`), SC7 (one German warning string in `DataHealthScanner`, ~40 other
strings in the same file are English), SM3 (per-cell correlation-matrix failures silently
swallowed, masked by NaN already being the documented sentinel), SM4 (4 bare `except:` in
Tukey/studentized-range fallback paths, all degrade to a documented conservative substitute),
SM5 (VIF/outlier-scan exceptions stashed in a dict key but never logged), SM6 (an R² threshold
constant labeled "Cohen f²-derived" is actually the raw f² value, off by <0.001), AT6 (Box's M
test's own bare `except:` masks a singular-covariance failure with an arbitrary floor — the
test is already labeled experimental/approximate), AT7 (recommendation-builder strings use
inconsistent emoji/prefix conventions, brittle for non-English UI), VZ5 (no NaN guards at
~8 p-value format sites), VZ9/VZ10 (substantial newly-confirmed dead code — `create_association_tree`,
`FlowchartVisualizer.visualize`/`generate_and_save`, 22 unreferenced `DataVisualizer` utility
methods including matplotlib ROC/forest-plot duplicates of the live Plotly renderer — round 1
had characterized two of these as "production call paths," which this round's `git grep` shows
is not accurate), VZ11 (`apply_custom_colormap` divides by zero for n≤1, zero callers today),
VZ12 (figure watermark renders at 1.82:1 contrast, dormant/unwired today), RE3 (sphericity
status reads `sphericity_met`, no writer ever sets it, only `sphericity_assumed` — masked by an
incomplete p-value fallback), RE4 (two dead reader-side fallback keys, no writer, inert), RE5
(no shared HTML-escaping helper in `_FormattingMixin` — the structural root cause of RE1/RE2),
GD4 (undiscoverable multi-select keyboard interaction), GD5 (CLAUDE.md `_ap_*` line-number
citations drifted, and one reachability sentence is imprecise), GD6 (one dialog missing an
empty-selection warning its siblings have), GD7 (a tooltip updates regardless of checkbox
state — intentional, undocumented as such), GD13 (dead `hasattr(dialog, 'create_plot_check')`
guard pointing at an attribute that no longer exists), GD14 (adjacent recurrence, one file
outside its batch's scope, of the previously-fixed "untranslated German string" bug class in
`_ap_detected_test_label`'s `correlation`/`linear_regression` entries).

---

## Strengths confirmed across all 7 audits

- **Every fix made between round 1 and round 2 holds under fresh, independent re-verification.**
  All 6 `ModelDesignError` pre-flight checks (ANCOVA/LMM/Logistic Regression empty-factor and
  design-invariant guards) fire exactly as claimed, verified by direct Python repro, not just
  inspection. ANCOVA's `control_group` wiring for vs-control EMM post-hoc genuinely reaches
  both dispatch entry points (`analysis_core.py` direct, and `advanced_pipeline.py` →
  `statisticaltester.py`), independently traced end to end.
- **Every prior-session statistical-correctness fix named in project memory still holds.**
  Games-Howell (Welch-Satterthwaite df, exact Hedges & Olkin small-sample correction),
  `scipy.stats.dunnett`'s single coherent joint fit, the mixed/RM Dunnett Holm-Bonferroni
  replacement of the old `p*k*0.8` crutch, the bounded Box-Cox lambda guard (rejects `|λ|>3`,
  never clamps), and the complete removal (not just hiding) of MWU power — all re-derived from
  first principles this round, not assumed from memory.
- **The plot-type dispatch dead-branch fix and the linear-regression coefficient-table wiring
  fix both hold and are confirmed end-to-end.** No orphaned `plot_type` string reaches
  `datavisualizer.py`; `coefficient_table` (written `correlation_models.py:848`) is read and
  dispatched all the way to the HTML export for every `LinearRegression` model.
- **`validators.py` is exemplary defensive programming** — a single, well-documented
  pre-flight chokepoint (`validate_samples_for_test`) with a correctly-derived
  `sqrt(float64_max/n)` overflow bound (not a naive constant), zero-variance/below-minimum-n/
  Inf checks, and dependent-design pairwise-constant-difference checks, all raising named
  `ValidationError` subclasses with accurate messages — no bare excepts anywhere in the file.
- **Jinja autoescaping is correctly configured** (`autoescape=True`) and un-bypassed for the
  large majority of report fields; RE1's injection remains narrowly scoped to 3 chart-table
  builders plus 4 unescaped Plotly label sites, not a blanket bypass. No silent exception
  swallowing was found anywhere in the report/export layer — every `except Exception` logs
  before falling back.
- **Writer/reader key-contract drift is rare and narrowly scoped.** Of the dozen-plus result
  dict shapes cross-checked this round (`roc_data`, `xy_data`, `adjusted_means`,
  `association_points`, effect-size-type strings, etc.), only SC3/RE3 showed drift — everything
  else matched writer-to-reader exactly, flat vs. nested shape included.
- **The mixin-binding architecture (`AutopilotMixin`) remains clean and matches CLAUDE.md's
  documented architecture exactly** — no legacy fallback code reintroduced since round 1.
- **`TutorialOverlay` and `decision_tree_view.py`'s theme contrast** are both genuinely
  well-built — cross-platform reduced-motion probing with a fail-safe fallback, full keyboard
  support, and computed WCAG contrast ratios clearing AA with real margin in both light and
  dark themes.

---

## Recommended order (across all 7 batches)

1. **Mixed ANOVA sphericity fix** (SC2/AT1/AT2/AT3/GD1) — highest statistical-correctness
   value. Requires computing a real epsilon first (AT2), not just wiring the existing (currently
   inert) correction — a wiring-only fix would create false confidence. Fixes the Help Hub
   claim (GD1) for free.
2. **RE1 (HTML injection)** — `html.escape()` around the 3 `parameter`-cell f-string sites in
   `report_association.py`; cheapest fix, closes the only reproducible injection path found,
   still open since round 1.
3. **U1 (excepthook itself crashes)** — one-line fix (`logger.info("%s", msg)`, drop the stray
   `file=` kwarg); restores the app's entire crash-visibility mechanism for end users.
4. **SC1 (`make_blocked_result` bad kwarg)** — drop `test_name=` from 4 call sites; restores the
   intended error messages for Mixed/Two-Way/RM-ANOVA invalid-design and prep-error blocks.
5. **VZ1 (significance letters silent-drop)** — same fix mechanism already used twice elsewhere
   in the same file (`_draw_warning_annotation`); still open since round 1.
6. **SC4 (`posthoc_choice` NameError) + SM1 (outlier decimal corruption)** — both one-line-class
   fixes that close a live crash path and a live silent-data-corruption path respectively.
7. **U2, AT4, AT5, SM2, VZ8, RE2** — MEDIUM findings with clear, scoped, low-risk fixes; bundle
   opportunistically with whichever HIGH fix touches the same file.
8. **GD9, GD10, GD11, GD12** — cheap documentation/dialog fixes, no design decision needed.
9. Everything else in the MEDIUM/LOW lists above — track, no urgency before 2.0. GD8 (journal
   palette colors) and SC3 (EMM key-schema rename) each benefit from a short product decision
   on the target schema/palette rather than being pure mechanical fixes.

Per the skill's `AUDIT:` mode discipline: **this document changes nothing**. All 7 raw batch
reports are in this directory (`01-`–`07-*.md`); this file is the cross-batch synthesis. Next
step is picking which findings to fix now vs. defer, then following this project's established
workflow (brainstorm for anything needing a design decision → spec → plan → TDD → verify).
