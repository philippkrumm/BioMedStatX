"""InteractiveDecisionTreeWidget has no keyboard-accessible zoom or reset-view control,
wheel-only. Adds +/-/0 bindings mirroring the existing wheelEvent's zoom logic and the existing
refit_view() reset.
"""
import pytest
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtGui import QKeyEvent

from ui.components.decision_tree_view import InteractiveDecisionTreeWidget


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def _key_event(key):
    return QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier)


def test_plus_key_zooms_in(qapp):
    widget = InteractiveDecisionTreeWidget()
    initial_scale = widget.transform().m11()
    widget.keyPressEvent(_key_event(Qt.Key_Plus))
    assert widget.transform().m11() > initial_scale


def test_equals_key_also_zooms_in(qapp):
    # '+' on most keyboards is Shift+'=' - bind the bare '=' too for reachability.
    widget = InteractiveDecisionTreeWidget()
    initial_scale = widget.transform().m11()
    widget.keyPressEvent(_key_event(Qt.Key_Equal))
    assert widget.transform().m11() > initial_scale


def test_minus_key_zooms_out(qapp):
    widget = InteractiveDecisionTreeWidget()
    initial_scale = widget.transform().m11()
    widget.keyPressEvent(_key_event(Qt.Key_Minus))
    assert widget.transform().m11() < initial_scale


def test_zero_key_calls_refit_view(qapp, monkeypatch):
    widget = InteractiveDecisionTreeWidget()
    called = []
    monkeypatch.setattr(widget, "refit_view", lambda: called.append(True))
    widget.keyPressEvent(_key_event(Qt.Key_0))
    assert called == [True]


def test_unhandled_key_falls_through_to_default_behavior(qapp):
    widget = InteractiveDecisionTreeWidget()
    # Should not raise for an unrelated key - falls through to QGraphicsView's
    # own keyPressEvent instead of being silently swallowed.
    widget.keyPressEvent(_key_event(Qt.Key_A))
