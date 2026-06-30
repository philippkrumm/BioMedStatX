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
            assert key in r and r[key], f"{r['id']} has empty or missing {key}"

def test_ids_unique():
    ids = [r["id"] for r in HELP_RECIPES]
    assert len(ids) == len(set(ids))

def test_deeplink_ids_preserved():
    ids = set(_by_id())
    assert DEEPLINK_IDS <= ids

def test_migrated_recipes_present():
    ids = set(_by_id())
    assert {"dependent_samples", "graph_visualization", "statistical_tests_html"} <= ids

def test_category_order_is_valid():
    assert len(CATEGORY_ORDER) > 0
    assert len(CATEGORY_ORDER) == len(set(CATEGORY_ORDER))


# ---------------------------------------------------------------------------
# Task 2: Category grouping in HelpHubDialog
# ---------------------------------------------------------------------------
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
        assert {h.text() for h in headers} <= set(CATEGORY_ORDER)
        assert len(headers) >= 2
        for h in headers:
            assert not (h.flags() & Qt.ItemIsSelectable)
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


def test_filter_reselects_visible_recipe_when_current_hidden(qapp):
    from ui.dialogs.statistical_analyzer_dialogs import HelpHubDialog
    dlg = HelpHubDialog()
    try:
        dlg.navigate_to("ancova")
        assert dlg.recipe_list.currentItem().data(Qt.UserRole) == "ancova"
        dlg.search_input.setText("logistic")
        current = dlg.recipe_list.currentItem()
        assert current is not None
        assert not current.isHidden()
        assert current.data(Qt.UserRole) is not None
        assert current.data(Qt.UserRole) == "logistic_regression"
    finally:
        dlg.deleteLater()


def test_keyboard_down_skips_category_header(qapp):
    from PyQt5.QtGui import QKeyEvent
    from PyQt5.QtCore import QEvent
    from ui.dialogs.statistical_analyzer_dialogs import HelpHubDialog
    dlg = HelpHubDialog()
    try:
        dlg.recipe_list.setCurrentRow(
            next(i for i in range(dlg.recipe_list.count())
                 if dlg.recipe_list.item(i).data(Qt.UserRole) is not None)
        )
        for _ in range(dlg.recipe_list.count()):
            ev = QKeyEvent(QEvent.KeyPress, Qt.Key_Down, Qt.NoModifier)
            QApplication.sendEvent(dlg.recipe_list, ev)
            cur = dlg.recipe_list.currentItem()
            assert cur is None or cur.data(Qt.UserRole) is not None, "landed on header"
    finally:
        dlg.deleteLater()
