# Audit note: `graph_visualization` recipe

Recipe location: `src/core/help_content.py:589` (`"id": "graph_visualization"`,
`"category": "Workflow & Output"`).

This is a "Workflow & Output" recipe describing how to configure and export a
plot from a finished analysis result. It was migrated verbatim from an older
inline dialog. Its plot-type drift ("Bar, box, violin, strip") is the finding
that originally triggered this whole content-vs-code audit, so it was checked
with maximum suspicion and every claim re-verified against current code rather
than trusting the spec's seed finding.

The citation anchor (symbol name or quoted string) is authoritative; the line
number is only a navigation hint.

## Two distinct plot surfaces exist

The recipe conflated two things. There are two separate figure-building
surfaces in the app, with different plot-type lists:

1. **Live Qt dialog `PlotAestheticsDialog`** (`src/ui/dialogs/plot_aesthetics_dialog.py`).
   This is the "configure a plot from an analysis result" surface the recipe's
   title/summary is about. Reached from a finished analysis via
   `_ap_configure_plot_from_result`
   (`src/autopilot/statistical_analyzer_autopilot_pipeline.py:1833`), which
   constructs `PlotAestheticsDialog(...)` at `:1850`; the button is wired at
   `configure_plot_requested.connect(self.configure_plot_from_result)`
   (`...pipeline.py:439`). Its plot-type dropdown is
   `self.plot_type_combo.addItems(['Bar', 'Box', 'Violin', 'Raincloud'])`
   (`plot_aesthetics_dialog.py:646`). On accept, it re-runs
   `AnalysisManager.analyze(..., save_plot=True)` and produces `xlsx`, `pdf`,
   and `png` files (`...pipeline.py:1890`-`1920`, the `for ext in ("xlsx",
   "pdf", "png")` loop at `:1916`).

2. **In-report interactive Plot Designer** (`src/templates/plot_designer.html`
   plus `src/templates/plot_designer.js`). This is embedded in the exported HTML
   report, not the live app. Its type dropdown is
   `Bar, Box, Violin, Raincloud, Forest, Estimation`
   (`plot_designer.html:107`-`112`).

### Reachability of `plot_designer.html` (the spec's required check)

`plot_designer.html` is **reachable and functional**, but only inside the
exported HTML report, not as a live dialog. The full chain:

- The normal analysis workflow (`_ap_determine_and_run_test`,
  `...pipeline.py:1740`) prompts "Save Analysis Report" as `.html`
  (`...pipeline.py:1753`-`1760`) and runs `_execute_single_analysis`, which
  renders the report through `HTMLExporter`.
- `HTMLExporter._prepare_single_report_context` sets
  `plot_designer_enabled = bool(plot_data)` (`src/export/html_exporter.py:149`,
  returned at `:174`).
- The report template includes the designer when that flag is set:
  `{% if context.plot_designer_enabled %}{% include "plot_designer.html" %}{% endif %}`
  (`src/templates/report_single.html.j2:33`).
- `plot_designer.html` in turn pulls in its logic:
  `{% include "plot_designer.js" %}` (`plot_designer.html:413`).
- `_render_template` sets `plotly_enabled` (and thus bundles Plotly) partly on
  `bool(context.get("plot_designer_enabled"))` (`html_exporter.py:346`), so the
  designer actually runs in the browser.

Its own header describes it as an "Interactive figure builder ... Adjust style
and export publication-ready SVG/PNG directly from the report"
(`plot_designer.html`, `<h2>Interactive figure builder</h2>` and the muted
subtitle). So it is a real user-facing surface, not dead code, and Forest /
Estimation are legitimately reachable. The recipe now mentions it as a separate
"redesign in the report" surface rather than listing Forest/Estimation as
live-dialog types.

**Forest / Estimation gating.** These two are not general "regression" plots.
`plot_designer.js` only renders them when post-hoc pairwise comparisons carry an
effect size and confidence interval:
- Forest: filters `pairwiseData` to `p.effect_size != null`; if none, warns
  "Forest plot requires effect size and confidence intervals from post-hoc
  tests" (`plot_designer.js:1458`-`1461`).
- Estimation: same effect-size/CI requirement (`plot_designer.js:1490`-`1493`)
  and additionally "requires a control-referenced design (e.g., Dunnett's).
  All-pairwise (Tukey) is not supported in this view" (`plot_designer.js:1508`).
The recipe therefore says Forest/Estimation appear "when post-hoc comparisons
supply effect sizes and confidence intervals", not "for regression".

## "Strip" is a point-overlay layout, not a plot type

Confirmed the seed finding independently. The four plot types are dispatched
in `DataVisualizer` with an explicit `raise ValueError(f"Unbekannter plot_type:
{plot_type}")` for anything else: `if plot_type == "Bar"` / `elif ... "Box"` /
`elif ... "Violin"` / `elif ... "Raincloud"` / `else: raise`
(`src/visualization/datavisualizer.py:2803`-`2905`). There is no "Strip" plot
type.

"Strip" is one of three point-overlay layouts applied on top of a plot:
`if style == 'jitter'` / `elif style == 'strip'` / `elif style == 'swarm'` in
`_add_data_points` (`datavisualizer.py:1828`, `:1858`, `:1865`). In the live
dialog these are exposed as the "Point Layout" dropdown
`self.point_style_combo.addItems(['Jitter', 'Beeswarm', 'Strip'])`
(`plot_aesthetics_dialog.py:770`), mapped `Jitter->jitter, Beeswarm->swarm,
Strip->strip` (`:948`-`951`).

## Overlay / points control check

- "Show individual data points" is a single global toggle in the Style tab,
  `self.points_check = QCheckBox("Show Individual Points")`
  (`plot_aesthetics_dialog.py`, in `StyleTab.init_ui`, near `:741`), producing
  `show_points` in the config (`:974`). It is not limited to box/violin/strip.
  Every plot family honors it: Bar, Violin, Box each call
  `DataVisualizer._add_data_points(...)` guarded by `if show_points`
  (`datavisualizer.py:982`, `:1179`, `:1385`), and Raincloud shows points by
  construction. So the old "on box, violin, or strip plots" phrasing was wrong
  on both counts (there is no strip plot, and points work on bar plots too).

## Error bar control check

- **Error metric.** The live dialog offers three, not two:
  `self.error_type_combo.addItems(['sd', 'se', 'ci'])`, default `'sd'`
  (`plot_aesthetics_dialog.py:1283`-`1284`, in `ErrorBarsTab`). The recipe
  previously listed only "SD or SEM"; it now says "SD, SE, or CI". (The
  in-report designer offers a wider set: `sd, sem, ci95, iqr, range`,
  `plot_designer.html:140`-`146`; not enumerated in the recipe to avoid
  conflating the two surfaces.)
- **Error style ("caps or line only").** Real, verified (the spec had it as
  unverified). `self.error_style_combo.addItems(['caps', 'line'])`, default
  `'caps'` (`plot_aesthetics_dialog.py:1292`-`1293`). The renderer honors it:
  `eb_caps = capsize if error_style == "caps" else 0` (`datavisualizer.py:1378`)
  and `'capsize': 0 if error_style == 'line' else ...` for Bar/Box
  (`datavisualizer.py:2808`, `:2855`).

## Significance-annotation control check

- The live dialog's `SignificanceTab` (`plot_aesthetics_dialog.py:1319`) has two
  groups: "Significance Letters"
  (`self.show_letters_check = QCheckBox("Show Significance Letters")`,
  `:1341`-`1346`, config key `show_significance_letters`) and "Significance
  Brackets" (`brackets_group = QGroupBox("Significance Brackets")`, `:1372`).
  So "letters or brackets" is correct. The old recipe called the second option
  "bars (significance lines)"; the code and the rest of the hub (e.g. the
  one_way_anova recipe) call them brackets, so the recipe now uses "brackets"
  and describes letters as grouping non-differing conditions and brackets as
  marking each significant pair, matching the compact-letter-display vs
  pairwise-bracket distinction established in the one_way_anova audit.

## Claim table

| # | Claim (from title/html) | Verdict | Citation |
|---|-------------------------|---------|----------|
| 1 | Title/summary: configure and export plots from an analysis result | correct | `_ap_configure_plot_from_result` builds `PlotAestheticsDialog` from `current_analysis_result` (`...pipeline.py:1833`, `:1850`); exports xlsx/pdf/png (`:1916`) |
| 2 | Plot types are "Bar, box, violin, and strip" | wrong | live types are `Bar, Box, Violin, Raincloud` (`plot_aesthetics_dialog.py:646`); dispatch raises on anything else (`datavisualizer.py:2803`-`2905`). "Strip" is not a plot type. |
| 3 | Bar shows group means with error bars | correct | Bar branch draws mean + error bars (`datavisualizer.py:2803`-`2848`) |
| 4 | Box displays medians, quartiles, and outliers | correct | Box branch (`datavisualizer.py:2850`-`2865`); standard boxplot semantics |
| 5 | Violin combines boxplot with a KDE | correct | Violin branch (`datavisualizer.py:2867`-`2878`), density + inner box |
| 6 | Strip shows all individual data points as dots (as a plot type) | wrong | "strip" is a point-overlay layout, not a plot type: `elif style == 'strip'` in `_add_data_points` (`datavisualizer.py:1858`); dropdown "Point Layout" (`plot_aesthetics_dialog.py:770`) |
| 7 (new) | Raincloud is a half violin + box + points | correct (was MISSING) | Raincloud branch (`datavisualizer.py:2879`-`2905`, `_create_raincloud_plot` at `:1492`); listed in dialog (`plot_aesthetics_dialog.py:646`) |
| 8 | Switch plot types in the configuration/appearance dialog | correct | `self.plot_type_combo` in `StyleTab` (`plot_aesthetics_dialog.py:644`-`648`) |
| 9 | Change colors and hatches per group | correct | palettes (`plot_aesthetics_dialog.py:430`), per-group hatch combos (`update_hatch_selectors`, `:458`-`501`) |
| 10 | Error bar type: SD or SEM | wrong (incomplete) | dropdown is `['sd', 'se', 'ci']` (`plot_aesthetics_dialog.py:1283`); three options, not two. Recipe now says SD/SE/CI. |
| 11 | Error bar style: with caps or line only | correct (was flagged unverified) | `['caps', 'line']` (`plot_aesthetics_dialog.py:1292`); renderer zeroes capsize for `'line'` (`datavisualizer.py:1378`, `:2808`) |
| 12 | Customize fonts, axes, and grid lines | correct | `TypographyTab` (`plot_aesthetics_dialog.py:176`), axes/grid in `StyleTab` (`grid_check` `:679`, `AxesTab` tick controls `:725`) |
| 13 (new) | Export DPI is adjustable | correct (was MISSING) | `self.dpi_spin` range 72-600, default 300 (`plot_aesthetics_dialog.py:119`-`124`); passed to `analyze(dpi=...)` (`...pipeline.py:1910`) |
| 14 | Show individual data points on box, violin, or strip plots | wrong | single global `show_points` toggle (`plot_aesthetics_dialog.py:741`) honored by Bar/Violin/Box/Raincloud (`datavisualizer.py:982`, `:1179`, `:1385`); not limited to box/violin, and no strip plot exists |
| 15 (new) | Point layout is jitter, beeswarm, or strip | correct (was MISSING) | `point_style_combo.addItems(['Jitter', 'Beeswarm', 'Strip'])` (`plot_aesthetics_dialog.py:770`); `_add_data_points` styles (`datavisualizer.py:1828`, `:1858`, `:1865`) |
| 16 | Statistical annotations: letters (grouping) or bars (significance lines) | correct, reworded | `SignificanceTab` "Significance Letters" (`:1341`) + "Significance Brackets" (`:1372`). Code/hub term is "brackets", not "bars"; recipe reworded. |
| 17 (new) | Exported HTML report embeds an interactive plot designer with SVG/PNG export | correct (was MISSING) | `{% include "plot_designer.html" %}` gated on `plot_designer_enabled` (`report_single.html.j2:33`, `html_exporter.py:149`); "Interactive figure builder" header + SVG/PNG download bar in `plot_designer.html` |
| 18 (new) | Report designer adds Forest and Estimation when post-hocs supply effect sizes + CIs | correct (was MISSING) | extra options `Forest, Estimation` (`plot_designer.html:111`-`112`); JS requires effect size + CI (`plot_designer.js:1458`-`1461`, `:1490`-`1493`, `:1508`) |

## Changes applied to the recipe html

- Added a one-line lead sentence stating the surface and that it exports PNG,
  PDF, and Excel (claims 1, 13).
- Plot-type list: `Bar, box, violin, strip` -> `Bar, Box, Violin, Raincloud`.
  Replaced the "Strip: shows all individual points" bullet with a Raincloud
  bullet (claims 2, 6, 7). Removed the separate "Switching plot types" bullet as
  redundant (the Type dropdown is now named in the lead of the plot-type item).
- Error metric: "SD or SEM" -> "SD, SE, or CI" (claim 10). Kept the caps/line
  style line (claim 11) and folded DPI into the fonts/axes/grid line (claim 13).
- Overlay: replaced "on box, violin, or strip plots" with a global points toggle
  plus the jitter/beeswarm/strip layout choice (claims 14, 15). Reworded "bars
  (significance lines)" to "brackets", and described letters as grouping
  non-differing conditions vs brackets marking each significant pair (claim 16).
- Added a "Redesign in the report" bullet for the in-report interactive designer,
  including the Forest/Estimation availability condition (claims 17, 18).
- Rewrote prose for AI-writing tells (copula avoidance, `-ing` padding,
  rule-of-three) per the humanizer skill. Kept the existing `<ul>` /
  bold-label list structure to match sibling Workflow & Output recipes.
- No `id` or `category` change. No emoji or typographic dashes introduced.

## Unclear / possible code bug

- **None affecting the recipe.** The two surfaces are intentional (a native Qt
  export dialog plus a browser-side re-plotter in the report), not a bug.
- **Observation, not a defect to fix here:** the live dialog's error-metric
  values are the lowercase codes `sd/se/ci` while the in-report designer uses
  `sd/sem/ci95/iqr/range`. The two lists diverge (the report designer is a
  superset with different labels). This is a UI-consistency wrinkle, not a
  correctness bug, and the recipe deliberately describes them as two separate
  surfaces so the mismatch does not mislead. Noted for any future task that
  wants to unify the two error-metric vocabularies.
