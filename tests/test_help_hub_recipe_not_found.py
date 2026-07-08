"""HelpHubDialog._update_recipe_view's "not found" branch calls
self.copy_button.setEnabled(False), but copy_button is never constructed anywhere in the class -
an AttributeError on a branch meant to degrade gracefully for a bad/missing recipe id.
"""
import pytest
from PyQt5.QtWidgets import QApplication, QListWidgetItem
from PyQt5.QtCore import Qt

from ui.dialogs.statistical_analyzer_dialogs import HelpHubDialog


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_recipe_not_found_branch_does_not_crash(qapp):
    dialog = HelpHubDialog()
    item = QListWidgetItem("Bad Recipe")
    item.setData(Qt.UserRole, "this-recipe-id-does-not-exist")
    # Should not raise AttributeError on a missing copy_button.
    dialog._update_recipe_view(item, None)
    assert dialog._current_recipe is None
    assert "unavailable" in dialog.recipe_browser.toHtml().lower() or "not found" in dialog.recipe_title.text().lower()
