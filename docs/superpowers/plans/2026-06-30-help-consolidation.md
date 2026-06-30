# Help Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the Help menu to Interactive Tour + Help Hub by migrating three inline help dialogs into categorized Help Hub recipes, dropping the duplicate Getting Started menu item, and running a humanizer pass (no emojis, no em/en dashes) over all recipe text.

**Architecture:** All help content lives in `src/core/help_content.py` as `HELP_RECIPES` (list of dicts). The Help Hub UI (`HelpHubDialog` in `src/ui/dialogs/statistical_analyzer_dialogs.py`) renders that list. We add a `category` field plus a fixed `CATEGORY_ORDER`, group the nav list under non-selectable category headers, delete the four standalone help dialog methods and their menu actions in `src/analysis/statistical_analyzer.py`, then humanize the recipe text. Recipe `id` values are referenced by deep-link callers and must never change.

**Tech Stack:** Python 3.12, PyQt5, pytest 7.4 (headless Qt via root `conftest.py`, `QT_QPA_PLATFORM=offscreen`, `src/` on `sys.path`).

---

## Invariants (do not break)

- Recipe `id` values are stable. Confirmed deep-link callers:
  `help_recipe_id="one_way_anova" | "two_way_anova" | "repeated_measures_anova" | "ancova"`
  in `src/autopilot/statistical_analyzer_autopilot_pipeline.py` (lines 267, 279, 292, 308).
- The 11 existing recipe ids: `getting_started`, `one_way_anova`, `two_way_anova`,
  `repeated_measures_anova`, `mixed_anova`, `ancova`, `correlation`,
  `linear_regression`, `logistic_regression`. (Two of the 11 share the file; full set
  is whatever `HELP_RECIPES` currently contains — the test in Task 1 snapshots them.)
- Recipe dict keys in use: `id`, `title`, `summary`, `keywords`, `html`. We add `category`.

## File structure

- `src/core/help_content.py` — add `CATEGORY_ORDER`, add `category` to every recipe, append 3 migrated recipes.
- `src/ui/dialogs/statistical_analyzer_dialogs.py` — category grouping in `HelpHubDialog`.
- `src/analysis/statistical_analyzer.py` — remove 4 menu actions + 4 dialog methods.
- `tests/test_help_hub.py` — new test file for data-model and UI invariants.

---

### Task 1: Category data model + invariants test

**Files:**
- Test: `tests/test_help_hub.py` (create)
- Modify: `src/core/help_content.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_help_hub.py`:

```python
from core.help_content import HELP_RECIPES, CATEGORY_ORDER

# Ids that external code deep-links to; must never change.
DEEPLINK_IDS = {
    "one_way_anova", "two_way_anova", "repeated_measures_anova", "ancova",
}

def _by_id():
    return {r["id"]: r for r in HELP_RECIPES}

def test_every_recipe_has_known_category():
    for r in HELP_RECIPES:
        assert "category" in r, f"{r['id']} missing category"
        assert r["category"] in CATEGORY_ORDER, f"{r['id']} has unknown category {r['category']!r}"

def test_required_keys_present():
    for r in HELP_RECIPES:
        for key in ("id", "title", "summary", "keywords", "html", "category"):
            assert key in r and r[key] is not None, f"{r['id']} missing {key}"

def test_ids_unique():
    ids = [r["id"] for r in HELP_RECIPES]
    assert len(ids) == len(set(ids))

def test_deeplink_ids_preserved():
    ids = set(_by_id())
    assert DEEPLINK_IDS <= ids

def test_migrated_recipes_present():
    ids = set(_by_id())
    assert {"dependent_samples", "graph_visualization", "statistical_tests_html"} <= ids
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_help_hub.py -v`
Expected: FAIL — `ImportError: cannot import name 'CATEGORY_ORDER'`.

- [ ] **Step 3: Add CATEGORY_ORDER and category to existing recipes**

At the top of `src/core/help_content.py`, above `HELP_RECIPES`, add:

```python
CATEGORY_ORDER = [
    "Start here",
    "Choosing a test",
    "Concepts",
    "Workflow & Output",
]
```

Add a `"category"` key to each existing recipe dict using this mapping:

- `getting_started` -> `"Start here"`
- `one_way_anova`, `two_way_anova`, `repeated_measures_anova`, `mixed_anova`,
  `ancova`, `correlation`, `linear_regression`, `logistic_regression` -> `"Choosing a test"`

Place `"category"` next to the existing `"id"` line in each dict, e.g.:

```python
{
    "id": "one_way_anova",
    "category": "Choosing a test",
    "title": "Comparing groups (t-Test / One-Way ANOVA)",
    ...
}
```

- [ ] **Step 4: Append the three migrated recipes**

Append these to `HELP_RECIPES` (after the last existing recipe). Copy each `html`
body **verbatim** from the inline dialog in `src/analysis/statistical_analyzer.py`
at the cited line range (the HTML string passed to `setHtml` / `QMessageBox`):

```python
{
    "id": "dependent_samples",
    "category": "Concepts",
    "title": "Dependent (paired) samples",
    "summary": "When measurements are paired or repeated on the same subjects, and which tests apply.",
    "keywords": ["paired", "dependent", "repeated", "wilcoxon", "friedman", "matched"],
    "html": (
        # verbatim body from statistical_analyzer.py:439-457 (the QMessageBox text)
        "<h3>When are samples dependent?</h3>"
        # ... paste remaining lines exactly ...
    ),
},
{
    "id": "graph_visualization",
    "category": "Workflow & Output",
    "title": "Graph visualization",
    "summary": "How to configure and export plots from an analysis result.",
    "keywords": ["plot", "graph", "chart", "figure", "visualization", "export", "bar", "box"],
    "html": (
        # verbatim body from statistical_analyzer.py:307-339 (the html string set via setHtml)
        "<h3>Graph Visualization</h3>"
        # ... paste remaining lines exactly ...
    ),
},
{
    "id": "statistical_tests_html",
    "category": "Workflow & Output",
    "title": "Statistical tests and HTML report",
    "summary": "How tests are chosen and what the exported HTML report contains.",
    "keywords": ["report", "html", "export", "results", "tests", "output"],
    "html": (
        # verbatim body from statistical_analyzer.py:350-394 (the html string set via setHtml)
        "<h3>Statistical Tests &amp; HTML Report</h3>"
        # ... paste remaining lines exactly ...
    ),
},
```

Note: the inline references like "open the Help Hub (Recipes) from the Help menu"
inside the migrated bodies are now self-referential. Leave them for this task; the
humanizer pass in Task 4 rewrites them (e.g. "see the other recipes in this hub").

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_help_hub.py -v`
Expected: PASS (all five tests).

- [ ] **Step 6: Commit**

```bash
git -C . add src/core/help_content.py tests/test_help_hub.py
git -C . commit -m "feat(help): categorize recipes and migrate 3 inline dialogs into Help Hub"
```

---

### Task 2: Category grouping in HelpHubDialog

**Files:**
- Modify: `src/ui/dialogs/statistical_analyzer_dialogs.py` (`HelpHubDialog._populate_recipe_list`, `_filter_recipe_list`, `_update_recipe_view`)
- Test: `tests/test_help_hub.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_help_hub.py`:

```python
import pytest
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])

def _header_items(dlg):
    items = [dlg.recipe_list.item(i) for i in range(dlg.recipe_list.count())]
    return [it for it in items if it.data(Qt.UserRole) is None]

def test_headers_render_and_are_not_selectable(qapp):
    from ui.dialogs.statistical_analyzer_dialogs import HelpHubDialog
    dlg = HelpHubDialog()
    try:
        headers = _header_items(dlg)
        # one header per category that has at least one recipe
        assert {h.text() for h in headers} <= set(CATEGORY_ORDER)
        assert len(headers) >= 2
        for h in headers:
            assert not (h.flags() & Qt.ItemIsSelectable)
        # the initially selected row is a real recipe, not a header
        current = dlg.recipe_list.currentItem()
        assert current is not None and current.data(Qt.UserRole) is not None
    finally:
        dlg.deleteLater()

def test_navigate_to_still_selects_recipe(qapp):
    from ui.dialogs.statistical_analyzer_dialogs import HelpHubDialog
    dlg = HelpHubDialog()
    try:
        dlg.navigate_to("ancova")
        current = dlg.recipe_list.currentItem()
        assert current.data(Qt.UserRole) == "ancova"
    finally:
        dlg.deleteLater()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_help_hub.py -k "headers or navigate" -v`
Expected: FAIL — no header items exist (flat list), `_header_items` returns `[]`.

- [ ] **Step 3: Rewrite `_populate_recipe_list` to group by category**

Replace the body of `_populate_recipe_list` (currently statistical_analyzer_dialogs.py:253-262) with:

```python
def _populate_recipe_list(self):
    from core.help_content import CATEGORY_ORDER
    self.recipe_list.clear()
    by_category = {cat: [] for cat in CATEGORY_ORDER}
    for recipe in self._recipes:
        by_category.setdefault(recipe.get("category", CATEGORY_ORDER[-1]), []).append(recipe)

    first_recipe_row = None
    for category in CATEGORY_ORDER:
        recipes = by_category.get(category) or []
        if not recipes:
            continue
        header = QListWidgetItem(category)
        header.setData(Qt.UserRole, None)
        header.setFlags(Qt.ItemIsEnabled)  # enabled for display, NOT selectable
        font = header.font()
        font.setBold(True)
        header.setFont(font)
        self.recipe_list.addItem(header)
        for recipe in recipes:
            item = QListWidgetItem(recipe["title"])
            item.setData(Qt.UserRole, recipe["id"])
            item.setToolTip(recipe.get("summary", ""))
            self.recipe_list.addItem(item)
            if first_recipe_row is None:
                first_recipe_row = self.recipe_list.row(item)

    if first_recipe_row is not None:
        self.recipe_list.setCurrentRow(first_recipe_row)
```

(Imports: `QListWidgetItem` and `Qt` are already imported at the top of this file —
verify they are; the existing flat code already uses `QListWidgetItem`.)

- [ ] **Step 4: Guard `_filter_recipe_list` against headers**

In `_filter_recipe_list` (statistical_analyzer_dialogs.py:264-289), skip header rows
and hide empty category headers. Replace the per-item loop body so a `None`
`recipe_id` is treated as a header. Insert at the start of the loop body, right after
`recipe_id = item.data(Qt.UserRole)`:

```python
        if recipe_id is None:
            # category header: visibility decided after recipe pass
            continue
```

Then, after the existing recipe loop and before the `current = ...` block, add a
second pass to hide headers whose following recipes are all hidden:

```python
        # Hide a category header when every recipe under it is hidden.
        count = self.recipe_list.count()
        for index in range(count):
            item = self.recipe_list.item(index)
            if item.data(Qt.UserRole) is not None:
                continue
            any_visible = False
            for j in range(index + 1, count):
                nxt = self.recipe_list.item(j)
                if nxt.data(Qt.UserRole) is None:
                    break  # reached next header
                if not nxt.isHidden():
                    any_visible = True
                    break
            item.setHidden(not any_visible)
```

- [ ] **Step 5: Guard `_update_recipe_view` against headers**

In `_update_recipe_view` (statistical_analyzer_dialogs.py:291+), after the
`if current is None: return` line, add:

```python
        if current.data(Qt.UserRole) is None:
            return  # header row, nothing to render
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_help_hub.py -v`
Expected: PASS (all tests, including Task 1's).

- [ ] **Step 7: Commit**

```bash
git -C . add src/ui/dialogs/statistical_analyzer_dialogs.py tests/test_help_hub.py
git -C . commit -m "feat(help): group Help Hub nav list by category"
```

---

### Task 3: Remove standalone help menu items and dialogs

**Files:**
- Modify: `src/analysis/statistical_analyzer.py` (`create_menu` lines 193-216; methods at 297-458, 500-562)
- Test: `tests/test_help_hub.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_help_hub.py`:

```python
def test_help_menu_has_only_kept_actions(qapp):
    from analysis.statistical_analyzer import StatisticalAnalyzerApp
    app = StatisticalAnalyzerApp()
    try:
        texts = {a.text() for a in app.help_menu.actions() if a.text()}
        # removed standalone help entries:
        assert "Getting Started" not in texts
        assert "Dependent Samples" not in texts
        assert "Graph Visualization" not in texts
        # Qt doubles '&' for accelerators; the action text contains '&&'
        assert not any("Statistical Tests" in t for t in texts)
        # kept entries still present:
        assert "Interactive Tour" in texts
        assert "Help Hub (Recipes)" in texts
    finally:
        app.close()

def test_removed_dialog_methods_gone():
    from analysis.statistical_analyzer import StatisticalAnalyzerApp
    for name in (
        "show_getting_started_help",
        "show_dependent_samples_help",
        "show_graph_visualization_help",
        "show_statistical_tests_html_help",
    ):
        assert not hasattr(StatisticalAnalyzerApp, name), f"{name} should be removed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_help_hub.py -k "menu or removed_dialog" -v`
Expected: FAIL — actions and methods still exist.

- [ ] **Step 3: Remove the four menu actions**

In `create_menu` (statistical_analyzer.py:193-216), delete these blocks:
- `getting_started_action` (lines 193-196)
- `dependent_help_action` (lines 204-206)
- `graph_vis_action` (lines 208-211)
- `stats_html_action` (lines 213-216)

Keep `help_hub_action`. Remove the now-orphaned `help_menu.addSeparator()` at line
202 and 218 only if they produce a double/leading separator; keep exactly one
separator between the Tour/Template group, the Help Hub, and the Updates group.
Resulting Help menu order: Interactive Tour, Save Example Template, separator,
Help Hub (Recipes), separator, Check for Updates, Report a Problem, ... Confetti.

- [ ] **Step 4: Remove the four dialog methods**

Delete these methods entirely from `statistical_analyzer.py`:
- `show_graph_visualization_help` (297-339)
- `show_statistical_tests_html_help` (340-434)
- `show_dependent_samples_help` (435-458)
- `show_getting_started_help` (500-562)

Verify nothing else calls them: run
`grep -rn "show_getting_started_help\|show_dependent_samples_help\|show_graph_visualization_help\|show_statistical_tests_html_help" src`
Expected: no matches after deletion.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_help_hub.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git -C . add src/analysis/statistical_analyzer.py tests/test_help_hub.py
git -C . commit -m "refactor(help): drop standalone help menu items and inline dialogs"
```

---

### Task 4: Humanizer pass + emoji/dash invariants

**Files:**
- Modify: `src/core/help_content.py` (all recipe `title` and `html`)
- Test: `tests/test_help_hub.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_help_hub.py`:

```python
import re

# Emoji + dingbat ranges + the two known decorative chars (▶, ✓, etc.)
_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF←-⇿⬀-⯿️]"
)

def test_no_emoji_in_recipe_text():
    for r in HELP_RECIPES:
        blob = r["title"] + r["html"]
        found = _EMOJI.findall(blob)
        assert not found, f"{r['id']} contains emoji/symbol: {found}"

def test_no_em_or_en_dash_in_recipe_text():
    for r in HELP_RECIPES:
        blob = r["title"] + r["html"]
        assert "—" not in blob, f"{r['id']} contains em dash"
        assert "–" not in blob, f"{r['id']} contains en dash"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_help_hub.py -k "emoji or dash" -v`
Expected: FAIL — at least `getting_started` title contains `▶`; other recipes
likely contain em dashes and decorative symbols.

- [ ] **Step 3: Run the humanizer pass**

For each of the 14 recipes, apply the `anthropic-skills:humanizer` process to its
`title` and `html`:
- Remove emojis and decorative symbols (including the `▶` in the `getting_started`
  title and any badges using emoji; the `.badge-good`/`.badge-bad` HTML spans are
  styling, not emoji — keep them).
- Replace em/en dashes per humanizer rule 14 (period, comma, colon, or restructure).
- Flatten promotional tone and title-case headings to sentence case.
- Fix the self-referential "open the Help Hub from the Help menu" lines in the three
  migrated recipes to read naturally now that they live inside the hub.
- **Do not change any recipe `id` or `category`.** Do not alter table structure or
  the `.badge` spans.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_help_hub.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git -C . add src/core/help_content.py tests/test_help_hub.py
git -C . commit -m "docs(help): humanizer pass over all recipes, remove emojis and dashes"
```

---

### Task 5: Full verification

- [ ] **Step 1: Run the whole suite**

Run: `pytest tests/ -q`
Expected: PASS, no regressions (existing tutorial/onboarding tests still pass).

- [ ] **Step 2: Lint**

Run: `ruff check src/core/help_content.py src/ui/dialogs/statistical_analyzer_dialogs.py src/analysis/statistical_analyzer.py`
Expected: no errors.

- [ ] **Step 3: Manual smoke test**

Run: `python src/analysis/statistical_analyzer.py`
Check:
- Help menu shows only: Interactive Tour, Save Example Template, Help Hub (Recipes),
  Check for Updates, Report a Problem, Confetti.
- Help Hub opens; nav list shows bold category headers (Start here, Choosing a test,
  Concepts, Workflow & Output) with recipes underneath; headers are not clickable.
- Typing in search filters recipes and hides empty category headers.
- The three migrated recipes (Dependent samples, Graph visualization, Statistical
  tests and HTML report) open and render.
- No emoji visible anywhere in the hub.

---

## Self-review

- **Spec coverage:** data model + categories (Task 1), UI grouping (Task 2), menu
  cleanup (Task 3), humanizer + emoji removal (Task 4), verification incl. deep-link
  check (Tasks 2/5). All spec sections covered.
- **Id stability:** enforced by `test_deeplink_ids_preserved` (Task 1) and the
  explicit "do not change id/category" instruction in Task 4.
- **Placeholders:** the only deferred content is the verbatim HTML migration in
  Task 1 Step 4, which points to exact source line ranges to copy rather than
  re-typing ~100 lines of existing markup.
