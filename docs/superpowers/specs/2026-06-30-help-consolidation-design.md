# Help consolidation design

Date: 2026-06-30
Status: approved

## Problem

The Help menu has six entries that all serve as help content: Interactive Tour,
Getting Started, Help Hub (Recipes), Dependent Samples, Graph Visualization, and
Statistical Tests & HTML Report. Four of these are standalone dialogs with ~270
lines of HTML hardcoded inline in `src/analysis/statistical_analyzer.py`. One of
them (Getting Started) duplicates a recipe that already exists in the Help Hub.
The content is scattered and partly redundant.

## Goal

Two help surfaces remain: the Interactive Tour and the Help Hub. All textual help
lives in the Help Hub, sourced from `src/core/help_content.py`. Every recipe text
passes a humanizer review and contains no emojis.

## Scope

In scope:
- Migrate three inline help dialogs into Help Hub recipes.
- Drop the Getting Started menu item (its recipe already exists).
- Add category grouping to the Help Hub navigation.
- Humanizer pass over all recipes (11 existing plus 3 migrated).
- Remove all emojis from recipe titles and bodies.

Out of scope:
- Interactive Tour content.
- Non-help menu items (Save Example Template, Check for Updates, Report a Problem,
  Confetti).
- Unrelated refactoring.

## Current state

- Recipe data: `src/core/help_content.py`, `HELP_RECIPES` list, 521 lines, 11
  recipes. Each recipe is a dict with keys `id`, `title`, `summary`, `keywords`,
  `html`. There is no `category` key today.
- Help Hub UI: `HelpHubDialog` in `src/ui/dialogs/statistical_analyzer_dialogs.py`.
  A flat `QListWidget` (`recipe_list`) populated by `_populate_recipe_list`,
  filtered by `_filter_recipe_list`, rendered by `_update_recipe_view`.
  `navigate_to(recipe_id)` deep-links to a recipe.
- Menu: `create_menu` in `src/analysis/statistical_analyzer.py` (lines 167+) adds
  the four standalone help actions at lines 193-216, wired to
  `show_getting_started_help`, `show_dependent_samples_help`,
  `show_graph_visualization_help`, `show_statistical_tests_html_help`.

## Design

### 1. Data model (`src/core/help_content.py`)

Add a `"category"` key to every recipe. Categories and membership:

- Start here: `getting_started`
- Choosing a test: `one_way_anova`, `two_way_anova`, `repeated_measures_anova`,
  `mixed_anova`, `ancova`, `correlation`, `linear_regression`,
  `logistic_regression`
- Concepts: `dependent_samples`
- Workflow & Output: `graph_visualization`, `statistical_tests_html`

Add three recipes migrated from the inline dialogs:
- `dependent_samples` (from `show_dependent_samples_help`)
- `graph_visualization` (from `show_graph_visualization_help`)
- `statistical_tests_html` (from `show_statistical_tests_html_help`)

Each new recipe gets `id`, `title`, `category`, `summary`, `keywords`, and `html`.
The `html` is the migrated dialog body; `summary` and `keywords` are written so the
recipe is findable by the existing search.

Category order is fixed and defined by a module-level list so the UI can render
groups in a stable order.

### 2. Help Hub UI (`HelpHubDialog`)

- `_populate_recipe_list`: iterate categories in defined order. For each category
  insert a non-selectable header `QListWidgetItem` (no `Qt.ItemIsSelectable`,
  `Qt.UserRole` data is `None`), then the recipes in that category. Select the
  first real recipe, not a header.
- `_filter_recipe_list`: skip header items when matching. After filtering, hide a
  category header if all its recipes are hidden.
- `_update_recipe_view`: if the current item is a header (no `recipe_id`), do
  nothing (guard already handles `None` recipe lookup, but headers must not be
  selectable so this should not fire).
- `navigate_to`: unchanged in contract; it already iterates by `Qt.UserRole`, so
  headers (data `None`) are skipped naturally.

### 3. Menu cleanup (`src/analysis/statistical_analyzer.py`)

Remove from `create_menu`: `getting_started_action`, `dependent_help_action`,
`graph_vis_action`, `stats_html_action`, and the now-redundant separators.
Remove the four methods `show_getting_started_help`, `show_dependent_samples_help`,
`show_graph_visualization_help`, `show_statistical_tests_html_help`.

### 4. Humanizer pass

Run the humanizer process (draft, audit, final) over all 14 recipe HTML bodies and
titles. Remove emojis (including the `▶` in the `getting_started` title), em and en
dashes, promotional tone, and title-case headings. Preserve all technical content,
tables, and the `.badge` markup used by the recipe stylesheet.

### 5. Verification

- App launches without error.
- Help menu shows only: Interactive Tour, Save Example Template, Help Hub, Check
  for Updates, Report a Problem, Confetti.
- Help Hub opens, categories render in order with headers, headers are not
  selectable.
- Search filters within categories and hides empty category headers.
- `navigate_to` deep-links still work, including any in-code callers of
  `show_help_hub(recipe_id=...)`.
- No emoji or em/en dash remains in `help_content.py`.

## Risks

- Header items in the list could be accidentally selectable, breaking navigation.
  Mitigation: clear the selectable flag and assert the first selected row is a
  recipe.
- In-code deep-link callers reference recipe ids by string. Confirmed callers:
  `help_recipe_id="one_way_anova" | "two_way_anova" | "repeated_measures_anova" |
  "ancova"` in `src/autopilot/statistical_analyzer_autopilot_pipeline.py` (lines
  267, 279, 292, 308), resolved through `_ap_resolve_help_recipe_for_bucket` and
  passed to `show_help_hub(recipe_id)` in
  `src/autopilot/statistical_analyzer_autopilot_ui.py:661`. The humanizer pass must
  change only `title` and `html`, never the `id` of any recipe. The three new
  recipes only add ids, so existing links stay valid.
