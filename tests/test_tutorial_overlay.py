# tests/test_tutorial_overlay.py
import pytest
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton
from PyQt5.QtCore import QRect
from ui.components.tutorial_overlay import (
    TourStep, resolve_union_rect, from_widgets,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_union_rect_skips_hidden_and_none(qapp):
    host = QWidget()
    host.resize(400, 300)
    a = QPushButton(host); a.setGeometry(10, 10, 100, 40); a.show()
    b = QPushButton(host); b.setGeometry(200, 100, 50, 20); b.hide()
    host.show()
    qapp.processEvents()
    rect_a_only = resolve_union_rect([a, b, None], host)
    assert rect_a_only is not None
    assert rect_a_only.contains(QRect(10, 10, 100, 40))
    assert not rect_a_only.contains(QRect(200, 100, 50, 20))


def test_union_rect_none_when_all_missing(qapp):
    host = QWidget(); host.resize(100, 100)
    hidden = QPushButton(host)  # not shown
    assert resolve_union_rect([None, hidden], host) is None


def test_tourstep_holds_copy(qapp):
    step = TourStep(title="T", body="B", tip="ti")
    assert step.title == "T" and step.tip == "ti"
    assert step.placement == "auto"
