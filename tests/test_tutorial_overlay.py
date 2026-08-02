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


from ui.components.tutorial_overlay import TutorialOverlay


def _three_steps():
    return [
        TourStep(title="One", body="b1"),
        TourStep(title="Two", body="b2"),
        TourStep(title="Three", body="b3"),
    ]


def test_navigation_clamps_and_closes(qapp):
    host = QWidget(); host.resize(800, 600); host.show()
    closed = {"v": False}
    ov = TutorialOverlay(host, _three_steps())
    ov.closed_callback = lambda: closed.__setitem__("v", True)
    ov.start()
    assert ov.is_first and not ov.is_last
    ov.prev_step()                 # clamped, stays at 0
    assert ov.current_index == 0
    ov.next_step(); ov.next_step()  # -> index 2 (last)
    assert ov.is_last
    ov.next_step()                  # past last -> closes
    assert closed["v"] is True


def test_missing_target_uses_centered_bubble(qapp):
    host = QWidget(); host.resize(800, 600); host.show()
    step = TourStep(title="x", body="y", resolve_rect=lambda: None)
    ov = TutorialOverlay(host, [step])
    ov.start()
    # No spotlight rect resolved -> overlay records None, no crash
    assert ov.current_spotlight is None
    ov.close_tour()


def test_pulse_active_only_on_pulse_step(qapp):
    host = QWidget(); host.resize(800, 600); host.show()
    plain = TourStep(title="p", body="b", resolve_rect=lambda: QRect(10, 10, 80, 30))
    pulsing = TourStep(title="q", body="b", resolve_rect=lambda: QRect(10, 10, 80, 30),
                       pulse=True)
    ov = TutorialOverlay(host, [plain, pulsing])
    ov.start(); qapp.processEvents()
    assert ov._pulse_active is False          # step 1 not pulsing
    ov.next_step(); qapp.processEvents()
    assert ov._pulse_active is True           # step 2 pulses
    ov.close_tour()
    assert ov._pulse_anim.state() == ov._pulse_anim.Stopped


def test_step_scrolls_below_fold_target_into_view(qapp):
    """A tour target scrolled out of a QScrollArea must be scrolled into view
    before the spotlight is drawn — otherwise the highlight lands off-screen on
    small windows, because a scrolled-out widget is still isVisible()==True."""
    from PyQt5.QtWidgets import QScrollArea, QVBoxLayout, QLabel
    host = QWidget()
    host.resize(400, 300)
    outer = QVBoxLayout(host)
    area = QScrollArea()
    area.setWidgetResizable(True)
    inner = QWidget()
    ilay = QVBoxLayout(inner)
    filler = QLabel("filler")
    filler.setFixedHeight(1200)              # pushes the target far below the fold
    ilay.addWidget(filler)
    target = QLabel("target")
    target.setFixedHeight(30)
    ilay.addWidget(target)
    area.setWidget(inner)
    outer.addWidget(area)
    host.show()
    qapp.processEvents()

    sb = area.verticalScrollBar()
    assert sb.maximum() > 0 and sb.value() == 0     # target starts below the fold

    step = TourStep(title="T", body="B", resolve_rect=from_widgets(host, target))
    ov = TutorialOverlay(host, [step])
    ov.start()
    qapp.processEvents()                            # run the deferred geometry/scroll

    assert sb.value() > 0, "tour did not scroll the below-fold target into view"
    ov.close_tour()


def test_spotlight_clips_to_scroll_viewport(qapp):
    """A target taller than its scroll viewport is spotlighted only over its
    visible portion, not its full content height (which otherwise spills the
    spotlight across neighbouring panels — e.g. the cockpit over the tree box)."""
    from PyQt5.QtWidgets import QScrollArea, QVBoxLayout, QLabel
    host = QWidget()
    host.resize(400, 300)
    outer = QVBoxLayout(host)
    area = QScrollArea()
    area.setWidgetResizable(True)
    inner = QWidget()
    ilay = QVBoxLayout(inner)
    tall = QLabel("tall target")
    tall.setFixedHeight(1500)          # far taller than the ~300px viewport
    ilay.addWidget(tall)
    area.setWidget(inner)
    outer.addWidget(area)
    host.show()
    qapp.processEvents()

    rect = resolve_union_rect([tall], host)
    assert rect is not None
    assert rect.height() <= area.viewport().height() + 2, (
        f"spotlight not clipped to viewport: rect {rect.height()}px vs viewport "
        f"{area.viewport().height()}px"
    )


from autopilot.statistical_analyzer_autopilot_pipeline import should_offer_tour


def test_should_offer_tour_logic():
    assert should_offer_tour("", "2.0.0") is True          # fresh machine
    assert should_offer_tour("1.9.0", "2.0.0") is True      # older version seen
    assert should_offer_tour("2.0.0", "2.0.0") is False     # already seen this version
    assert should_offer_tour(None, "2.0.0") is True         # missing key
