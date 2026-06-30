# Help Hub content-vs-code audit design

Date: 2026-06-30
Status: approved

## Problem

The Help Hub recipes in `src/core/help_content.py` were partly migrated from older
inline dialogs and partly written by hand. Their content has drifted from what the
application actually does. A spot check of `graph_visualization` found it lists plot
types "Bar, box, violin, strip" while the live plot dialog offers Bar, Box, Violin,
Raincloud (and a separate plot designer adds more), and "strip" is a point-overlay
style, not a top-level plot type. Other recipes may contain similar drift.

## Goal

Every recipe's factual content matches current code behavior. Wrong statements are
corrected and relevant missing features and statistical specifics are added. The code
is the source of truth. Every correction is backed by a concrete code citation
so reviewers can verify it rather than trust prose.

## Citation format

A bare `file:line` goes stale after any refactor or formatter run. Every citation must
carry a stable functional anchor; the line number is only a navigation hint:

`path/to/file.py:NNN (ClassName.method_name)` — or, when not inside a class/method,
`path/to/file.py:NNN ("unique signature or string literal")`.

The anchor (symbol name or quoted string) is authoritative; if the line is stale at
review time the reviewer locates the anchor by name. A citation with no verifiable
anchor is treated as a defect.

## Scope

In scope: the `title` and `html` of all 12 recipes in `HELP_RECIPES`. Both
user-visible features (plot types, dialog options, test-selection logic, export) and
statistical specifics (test names, decision thresholds, post-hoc methods, effect-size
definitions, required data structure). For statistical claims the audit verifies the
*exact* implementation, not just the family: the precise p-value adjustment method
(e.g. Tukey HSD vs Bonferroni vs Holm), one- vs two-sided tests, the default `alpha`,
and the key call parameters of the underlying function (e.g. the arguments passed to
`statsmodels.stats.multicomp.pairwise_tukeyhsd` or the relevant `scipy.stats` call).

Out of scope: recipe `id` and `category` values (deep-link callers depend on ids);
the Help Hub UI; the Interactive Tour; any code change to fix a discrepancy (if the
code is wrong rather than the recipe, that is reported, not fixed here).

## Correction policy

- Fix any statement that is factually wrong against the code.
- Add relevant features the recipe omits (e.g. the Raincloud plot type, additional
  post-hoc options) when their omission would mislead a user.
- Do not add exhaustive internal detail that no user-facing surface exposes.
- If the recipe and the code disagree and it is unclear which is intended (recipe
  describes desired behavior, code does something else), flag it in the audit notes
  for human decision rather than silently rewriting.

## Invariants (enforced by existing tests in tests/test_help_hub.py)

- No emoji, no typographic dashes (— – ― ‒ ‐ ‑) in recipe text.
- Each recipe opens with exactly one sentence-case `<h2>`, then `<h3>` sections.
- Recipe `id` and `category` values unchanged; deep-link ids
  (`one_way_anova`, `two_way_anova`, `repeated_measures_anova`, `ancova`) preserved.
- All of `pytest tests/` stays green.

## Authoritative code sources per recipe

| Recipe | Ground-truth modules |
|--------|----------------------|
| `getting_started` | autopilot pipeline: `_ap_load_file`, `_ap_load_sheet`, `_ap_maybe_pivot`, `_ap_build_analysis_context`; `export_example_template` |
| `one_way_anova` | `statisticaltester.py` (t-test/ANOVA selection, Shapiro, Levene), `analysis_core.py`, `validators.py` |
| `two_way_anova` | `advanced_pipeline.py`, `engines/advanced_posthoc.py`, `validators.py`, `analysis_core.py` |
| `repeated_measures_anova` | `advanced_pipeline.py`, sphericity/Greenhouse-Geisser logic, LMM fallback note, `validators.py` |
| `mixed_anova` | `advanced_pipeline.py`, `emm_posthoc.py`, between/within factor handling, `validators.py` |
| `ancova` | `advanced_pipeline.py`, ANCOVA path, `validators.py` |
| `correlation` | `correlation_models.py` (Pearson/Spearman, Shapiro on inputs) |
| `linear_regression` | regression path, `effect_sizes.py`, statsmodels usage |
| `logistic_regression` | logistic path, convergence/Firth fallback, OR/AUC reporting, `effect_sizes.py` |
| `dependent_samples` | `statisticaltester.py` (`_wilcoxon_test`, paired t, Friedman), RM path |
| `graph_visualization` | `ui/dialogs/plot_aesthetics_dialog.py`, `visualization/datavisualizer.py`, `templates/plot_designer.html` |
| `statistical_tests_html` | test-decision logic in `statisticaltester.py`/`analysis_core.py`, `export/report_methods.py` |

## Known seed findings (from initial spot check)

- `graph_visualization`: live plot dialog is `PlotAestheticsDialog`
  (`plot_aesthetics_dialog.py:646` → `addItems(['Bar', 'Box', 'Violin', 'Raincloud'])`),
  reached via `_ap_configure_plot_from_result`
  (`statistical_analyzer_autopilot_pipeline.py:1833,1850`). `plot_designer.html`
  additionally lists `Forest` and `Estimation` (`plot_designer.html:107-112`) and point
  layouts `Jitter`/`Beeswarm` — before citing Forest/Estimation the auditor must prove
  `plot_designer.html` is reachable: cite the instantiation/entry path (a menu action,
  a call site, or a route/registration that loads the template), not just the
  template's existence. A grep hit on the template is not proof. If no live entry point
  exists, declare it dead code in the audit note and do NOT add Forest/Estimation.
  "Strip" in the recipe
  is a point-display style (`datavisualizer.py:1858`), not a plot type. Error bars
  SD/SEM are real (`plot_designer.html:140-141`); the "caps or line only" claim is
  unverified and must be checked.
- `statistical_tests_html`: spot check found its claims accurate (t-test/Mann-Whitney,
  paired-t/Wilcoxon, One-Way ANOVA/Kruskal-Wallis, Shapiro-Wilk, Levene, auto post-hoc,
  letters/brackets all present in code). Still re-verify and add citations.

## Method (per recipe)

1. Read the recipe `html`; list every factual claim.
2. For each claim, locate the authoritative code and record evidence in the citation
   format above (anchor + line). For a statistical claim, open the actual function and
   confirm the exact adjustment method, sided-ness, `alpha`, and call parameters — not
   just that a test of that family exists.
3. Classify each claim: correct / wrong / missing-relevant-feature / unclear.
4. Rewrite `html` to fix wrong claims and add missing relevant features, preserving the
   humanizer invariants and the existing table/badge HTML structure.
5. Produce an audit note per recipe: the claim list with verdicts and citations, plus
   any "unclear / possible code bug" items flagged for the human.

### Mandatory control checks (every applicable recipe)

- **Alpha / adjustment match:** does the recipe's stated significance level and p-value
  adjustment match the code's default `alpha` and the actual correction used by the
  post-hoc call? Cite the parameter.
- **Data-structure invariance:** if the recipe tells the user to arrange data in wide
  or long format, confirm it matches what the parser actually expects. For paired /
  repeated-measures / mixed designs check against `advanced_pipeline.py` and the
  auto-pivot logic in `_ap_maybe_pivot`; cite the parsing/validation site.
- **Assumption/correction claims:** where a recipe describes an assumption handling
  (e.g. sphericity → Greenhouse-Geisser, normality → non-parametric fallback), confirm
  the code applies it as described. If the code's behavior contradicts the recipe and
  the code looks wrong, flag as "unclear / possible code bug" rather than rewriting.

## Verification

- `pytest tests/` green (emoji/dash/heading/id invariants plus the rest of the suite).
- For each recipe, the spec-compliance reviewer independently re-checks the citations:
  locate each cited anchor (symbol or quoted string) and confirm it supports the recipe
  statement, even if the line number drifted. For statistical claims the reviewer
  confirms the exact adjustment method / sided-ness / alpha at the anchor. A claim
  without a verifiable anchor is treated as a defect.
- Final pass: confirm no recipe `id`/`category` changed across the whole diff.

## Risks

- A subagent could write a plausible but wrong statistical statement. Mitigation: every
  factual claim must carry a `file:line` citation; the reviewer verifies citations
  against code, not prose.
- `plot_designer.html` vs `PlotAestheticsDialog` ambiguity could lead to listing plot
  types users cannot actually reach. Mitigation: the auditor must confirm reachability
  before listing Forest/Estimation, and cite the entry point.
- Session limit / model availability could interrupt a multi-recipe run. Mitigation:
  one recipe per task with its own commit, so progress is durable and resumable.
