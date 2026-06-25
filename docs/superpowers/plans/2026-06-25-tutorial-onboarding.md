# In-App Onboarding Tour Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a passive 7-step guided-tour overlay to the BioMedStatX main window, a first-run welcome gate, a Help-menu re-entry, and an example-template export action.

**Architecture:** A new `TutorialOverlay` QWidget dims the live window and punches an antialiased spotlight around each target; a bubble carries Back/Next/Skip. Step targets resolve to rects via pure helper functions (unit-testable headless). Controller methods live as `_ap_*` functions bound on `AutopilotMixin`; the first-run gate and menu live in `StatisticalAnalyzerApp`.

**Tech Stack:** Python, PyQt5, pytest (headless via root `conftest.py` which sets `QT_QPA_PLATFORM=offscreen` and adds `src/` to `sys.path`).

---

## Reference facts (verified against the codebase)

- Main window class: `StatisticalAnalyzerApp(AutopilotMixin, QMainWindow)` in `src/analysis/statistical_analyzer.py`.
- `AutopilotMixin` (`src/autopilot/statistical_analyzer_autopilot_pipeline.py:2091`) binds methods as class attributes, e.g. `init_ui = _ap_init_ui`. New methods follow the same pattern.
- `_ap_init_ui` builds a 3-column `QSplitter` inside a `QScrollArea`. Relevant widgets already stored as `self.*`: `auto_sheet_combo`, `range_select_btn`, `preview_table`, `header_cards_widget`, `dv_bucket`, `factor1_bucket`, `factor2_bucket`, `subject_bucket`, `covariates_bucket`, `filter_bucket`, `analysis_group_button`, `start_analysis_button`, `decision_tree_panel`, `result_cockpit`.
- The Load Data button is a **local** `browse_button` (pipeline ~line 148); must be promoted to `self.browse_button`.
- The six buckets are added directly to the center layout in a loop (pipeline ~line 313); we add a wrapping container `self.mapping_panel` for one clean spotlight.
- `create_menu` builds a local `help_menu` (`src/analysis/statistical_analyzer.py:191`); must be stored as `self.help_menu`.
- Existing resource helper: `_resource_path(...)` in the pipeline module handles `sys._MEIPASS`.
- Bundled asset already present: `assets/BioMedStatX_Excel_Template.xlsx`. Build globs `assets/` (no `.spec` change).
- Test precedent for headless Qt widget tests: `tests/test_decision_tree_graphics.py` (session-scoped `qapp` fixture creating `QApplication`).

---

## File structure

- Create: `src/ui/components/tutorial_overlay.py` — overlay widget + `TourStep` + rect resolvers. One responsibility: drawing/navigating the tour.
- Create: `tests/test_tutorial_overlay.py` — unit tests for resolvers, navigation, first-run decision.
- Modify: `src/autopilot/statistical_analyzer_autopilot_pipeline.py` — widget promotions, `_ap_build_tour_steps`, `_ap_start_tutorial`, `_ap_export_example_template`, `should_offer_tour`, mixin bindings.
- Modify: `src/analysis/statistical_analyzer.py` — menu actions, `self.help_menu`, first-run gate in `__init__`.

---

## Task 1: Pure rect-resolver helpers + TourStep

**Files:**
- Create: `src/ui/components/tutorial_overlay.py`
- Test: `tests/test_tutorial_overlay.py`

- [ ] **Step 1: Write the failing test**

```python
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
    b = QPushButton(host); b.setGeometry(200, 100, 50, 20)  # never shown
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tutorial_overlay.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ui.components.tutorial_overlay'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ui/components/tutorial_overlay.py
"""Passive guided-tour overlay for the BioMedStatX main window.

A full-window translucent overlay dims the app, punches an antialiased
spotlight around the current target, and shows a bubble with Back/Next/Skip.
The tour is linear and passive: it never triggers the real widgets, so it
cannot corrupt analysis state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from PyQt5.QtCore import Qt, QRect, QPoint
from PyQt5.QtWidgets import QWidget

RectResolver = Callable[[], Optional[QRect]]


def resolve_union_rect(widgets: Sequence, host: QWidget) -> Optional[QRect]:
    """Union of widgets' rects in host coordinates.

    Skips widgets that are None, not visible, or zero-sized. Returns None if
    nothing visible remains, so the caller can show a centered bubble.
    """
    rect: Optional[QRect] = None
    for w in widgets:
        if w is None or not w.isVisible():
            continue
        size = w.size()
        if size.isEmpty():
            continue
        top_left = w.mapTo(host, QPoint(0, 0))
        r = QRect(top_left, size)
        rect = r if rect is None else rect.united(r)
    return rect


@dataclass
class TourStep:
    title: str
    body: str
    tip: Optional[str] = None
    resolve_rect: Optional[RectResolver] = None
    placement: str = "auto"          # auto|above|below|left|right
    pre_show: Optional[Callable[[], None]] = None
    pulse: bool = False              # animate the spotlight ring (final step)


def from_widgets(host: QWidget, *widgets) -> RectResolver:
    return lambda: resolve_union_rect(widgets, host)


def from_menu_action(menubar, action) -> RectResolver:
    def _resolve() -> Optional[QRect]:
        geo = menubar.actionGeometry(action)
        if geo.isEmpty():
            return None
        top_left = menubar.mapTo(menubar.window(), geo.topLeft())
        return QRect(top_left, geo.size())
    return _resolve
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tutorial_overlay.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ui/components/tutorial_overlay.py tests/test_tutorial_overlay.py
git commit -m "feat(tour): rect resolvers and TourStep for onboarding overlay"
```

---

## Task 2: TutorialOverlay widget (navigation + paint + input capture)

**Files:**
- Modify: `src/ui/components/tutorial_overlay.py`
- Test: `tests/test_tutorial_overlay.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_tutorial_overlay.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tutorial_overlay.py -v`
Expected: FAIL with `ImportError: cannot import name 'TutorialOverlay'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/ui/components/tutorial_overlay.py`:

```python
from PyQt5.QtCore import QTimer, QEvent, QVariantAnimation
from PyQt5.QtGui import QPainter, QColor, QPainterPath, QPen
from PyQt5.QtWidgets import (
    QLabel, QPushButton, QFrame, QVBoxLayout, QHBoxLayout,
)


class TutorialOverlay(QWidget):
    SPOTLIGHT_PAD = 8
    SPOTLIGHT_RADIUS = 12
    DIM_COLOR = QColor(15, 23, 32, 200)
    # Spotlight accent ring. These constants live here (not QSS) because a
    # custom paintEvent cannot be styled or animated by QSS. frontend-design
    # tunes these values in Task 7; the teal default suits the dark palette.
    RING_COLOR = QColor(45, 212, 191)        # teal accent
    RING_WIDTH = 2
    PULSE_MIN_ALPHA = 70
    PULSE_MAX_ALPHA = 255
    PULSE_PERIOD_MS = 1100

    def __init__(self, host_window: QWidget, steps):
        super().__init__(host_window)
        self._host = host_window
        self._steps = list(steps)
        self._index = 0
        self.current_spotlight: Optional[QRect] = None
        self.closed_callback: Optional[Callable[[], None]] = None
        self.setObjectName("tutorialOverlay")
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self._build_bubble()
        self._ring_alpha = self.PULSE_MAX_ALPHA
        self._pulse_active = False
        self._pulse_anim = QVariantAnimation(self)
        self._pulse_anim.setStartValue(self.PULSE_MAX_ALPHA)
        self._pulse_anim.setKeyValueAt(0.5, self.PULSE_MIN_ALPHA)
        self._pulse_anim.setEndValue(self.PULSE_MAX_ALPHA)
        self._pulse_anim.setDuration(self.PULSE_PERIOD_MS)
        self._pulse_anim.setLoopCount(-1)
        self._pulse_anim.valueChanged.connect(self._on_pulse_value)
        self.setGeometry(host_window.rect())
        host_window.installEventFilter(self)

    # ---- bubble UI ----
    def _build_bubble(self):
        self.bubble = QFrame(self)
        self.bubble.setObjectName("tutorialBubble")
        lay = QVBoxLayout(self.bubble)
        lay.setContentsMargins(18, 16, 18, 14)
        lay.setSpacing(8)
        self._title = QLabel(self.bubble); self._title.setObjectName("tutorialTitle")
        self._title.setWordWrap(True)
        self._body = QLabel(self.bubble); self._body.setObjectName("tutorialBody")
        self._body.setWordWrap(True)
        self._tip = QLabel(self.bubble); self._tip.setObjectName("tutorialTip")
        self._tip.setWordWrap(True)
        self._progress = QLabel(self.bubble); self._progress.setObjectName("tutorialProgress")
        lay.addWidget(self._title)
        lay.addWidget(self._body)
        lay.addWidget(self._tip)
        row = QHBoxLayout(); row.setSpacing(8)
        self._skip = QPushButton("Skip tour", self.bubble); self._skip.setObjectName("tutorialSkip")
        self._back = QPushButton("Back", self.bubble); self._back.setObjectName("tutorialBack")
        self._next = QPushButton("Next", self.bubble); self._next.setObjectName("tutorialNext")
        self._skip.clicked.connect(self.close_tour)
        self._back.clicked.connect(self.prev_step)
        self._next.clicked.connect(self.next_step)
        row.addWidget(self._progress)
        row.addStretch(1)
        row.addWidget(self._skip)
        row.addWidget(self._back)
        row.addWidget(self._next)
        lay.addLayout(row)
        self.bubble.adjustSize()

    # ---- lifecycle ----
    def start(self):
        self.setGeometry(self._host.rect())
        self.show(); self.raise_(); self.setFocus(); self.grabKeyboard()
        self._index = 0
        self._refresh()

    def close_tour(self):
        try:
            self.releaseKeyboard()
            self._host.removeEventFilter(self)
        finally:
            self._pulse_anim.stop()
            self.hide()
            if self.closed_callback:
                self.closed_callback()
            self.deleteLater()

    # ---- navigation ----
    @property
    def current_index(self) -> int:
        return self._index

    @property
    def is_first(self) -> bool:
        return self._index == 0

    @property
    def is_last(self) -> bool:
        return self._index == len(self._steps) - 1

    def next_step(self):
        if self.is_last:
            self.close_tour()
            return
        self._index += 1
        self._refresh()

    def prev_step(self):
        if self.is_first:
            return
        self._index -= 1
        self._refresh()

    # ---- per-step refresh ----
    def _refresh(self):
        step = self._steps[self._index]
        self._title.setText(step.title)
        self._body.setText(step.body)
        self._tip.setText(f"Tip: {step.tip}" if step.tip else "")
        self._tip.setVisible(bool(step.tip))
        self._progress.setText(f"{self._index + 1} / {len(self._steps)}")
        self._back.setEnabled(not self.is_first)
        self._next.setText("Done" if self.is_last else "Next")
        if step.pre_show:
            step.pre_show()
        # Defer geometry read so layout/scroll/paint settle first.
        QTimer.singleShot(0, lambda s=step: self._apply_step_geometry(s))

    def _apply_step_geometry(self, step):
        self.setGeometry(self._host.rect())
        self.current_spotlight = step.resolve_rect() if step.resolve_rect else None
        self._position_bubble(step)
        self._set_pulse(getattr(step, "pulse", False) and self.current_spotlight is not None)
        self.update()

    def _set_pulse(self, active: bool):
        self._pulse_active = active
        if active:
            self._pulse_anim.start()
        else:
            self._pulse_anim.stop()
            self._ring_alpha = self.PULSE_MAX_ALPHA
            self.update()

    def _on_pulse_value(self, value):
        self._ring_alpha = int(value)
        self.update()

    def _position_bubble(self, step):
        self.bubble.adjustSize()
        bw, bh = self.bubble.width(), self.bubble.height()
        host_rect = self.rect()
        spot = self.current_spotlight
        if spot is None:
            x = (host_rect.width() - bw) // 2
            y = (host_rect.height() - bh) // 2
            self.bubble.move(max(0, x), max(0, y))
            return
        # Prefer below, flip above if it would clip.
        x = min(max(8, spot.left()), host_rect.width() - bw - 8)
        if spot.bottom() + 12 + bh <= host_rect.height():
            y = spot.bottom() + 12
        else:
            y = max(8, spot.top() - 12 - bh)
        self.bubble.move(x, y)

    # ---- painting ----
    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), self.DIM_COLOR)
        spot = self.current_spotlight
        if spot is not None:
            padded = spot.adjusted(-self.SPOTLIGHT_PAD, -self.SPOTLIGHT_PAD,
                                   self.SPOTLIGHT_PAD, self.SPOTLIGHT_PAD)
            path = QPainterPath()
            path.addRoundedRect(
                float(padded.x()), float(padded.y()),
                float(padded.width()), float(padded.height()),
                self.SPOTLIGHT_RADIUS, self.SPOTLIGHT_RADIUS,
            )
            painter.setCompositionMode(QPainter.CompositionMode_DestinationOut)
            painter.fillPath(path, QColor(0, 0, 0, 255))
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
            # Accent ring around the spotlight (alpha pulses on a pulse step).
            ring_color = QColor(self.RING_COLOR)
            ring_color.setAlpha(self._ring_alpha if self._pulse_active
                                else self.PULSE_MAX_ALPHA)
            painter.setPen(QPen(ring_color, self.RING_WIDTH))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(path)
        painter.end()

    # ---- input capture ----
    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Escape:
            self.close_tour()
        elif key in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space, Qt.Key_Right):
            self.next_step()
        elif key == Qt.Key_Left:
            self.prev_step()
        else:
            event.accept()

    def eventFilter(self, obj, event):
        # Swallow background key presses so a stray Space/Enter cannot trigger
        # a focused widget (e.g. start_analysis_button) while the tour runs.
        if event.type() == QEvent.KeyPress and self.isVisible():
            self.keyPressEvent(event)
            return True
        return super().eventFilter(obj, event)

    def resizeEvent(self, _event):
        if self.isVisible():
            self._apply_step_geometry(self._steps[self._index])
```

Note: keep the `from PyQt5...` imports at the top of the file by moving the second import line up next to the Task 1 imports during implementation (do not leave a mid-file import). The block above lists them for clarity.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tutorial_overlay.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ui/components/tutorial_overlay.py tests/test_tutorial_overlay.py
git commit -m "feat(tour): TutorialOverlay widget with spotlight ring, pulse, bubble, input capture"
```

---

## Task 3: First-run decision helper + QSettings gate logic

**Files:**
- Modify: `src/autopilot/statistical_analyzer_autopilot_pipeline.py`
- Test: `tests/test_tutorial_overlay.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_tutorial_overlay.py
from autopilot.statistical_analyzer_autopilot_pipeline import should_offer_tour


def test_should_offer_tour_logic():
    assert should_offer_tour("", "2.0.0") is True          # fresh machine
    assert should_offer_tour("1.9.0", "2.0.0") is True      # older version seen
    assert should_offer_tour("2.0.0", "2.0.0") is False     # already seen this version
    assert should_offer_tour(None, "2.0.0") is True         # missing key
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tutorial_overlay.py::test_should_offer_tour_logic -v`
Expected: FAIL with `ImportError: cannot import name 'should_offer_tour'`.

- [ ] **Step 3: Write minimal implementation**

Add near the top of `src/autopilot/statistical_analyzer_autopilot_pipeline.py` (module level, after imports):

```python
def should_offer_tour(stored_version, current_version: str) -> bool:
    """First-run gate: offer the tour when the stored completed-version does
    not match the current app version (covers empty/None on a fresh machine)."""
    return (stored_version or "") != (current_version or "")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tutorial_overlay.py::test_should_offer_tour_logic -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/autopilot/statistical_analyzer_autopilot_pipeline.py tests/test_tutorial_overlay.py
git commit -m "feat(tour): first-run decision helper should_offer_tour"
```

---

## Task 4: Pipeline wiring — widget promotions, tour steps, start_tutorial

**Files:**
- Modify: `src/autopilot/statistical_analyzer_autopilot_pipeline.py`
- Test: `tests/test_tutorial_onboarding_app.py` (new)

- [ ] **Step 1: Promote the Load button and wrap the buckets**

In `_ap_init_ui`, change the local Load button to an attribute. Find:

```python
    browse_button = QPushButton("Load Data File")
    browse_button.clicked.connect(self.browse_file)
    file_row.addWidget(browse_button)
```

Replace with:

```python
    self.browse_button = QPushButton("Load Data File")
    self.browse_button.clicked.connect(self.browse_file)
    file_row.addWidget(self.browse_button)
```

Then find the bucket loop:

```python
    for bucket in (self.dv_bucket, self.factor1_bucket, self.factor2_bucket,
                   self.subject_bucket, self.covariates_bucket, self.filter_bucket):
        bucket.changed.connect(self.on_mapping_changed)
        center_layout.addWidget(bucket)
```

Replace with a wrapping container so the tour can spotlight one tidy rect:

```python
    self.mapping_panel = QWidget()
    self.mapping_panel.setObjectName("mappingPanel")
    _mapping_layout = QVBoxLayout(self.mapping_panel)
    _mapping_layout.setContentsMargins(0, 0, 0, 0)
    _mapping_layout.setSpacing(8)
    for bucket in (self.dv_bucket, self.factor1_bucket, self.factor2_bucket,
                   self.subject_bucket, self.covariates_bucket, self.filter_bucket):
        bucket.changed.connect(self.on_mapping_changed)
        _mapping_layout.addWidget(bucket)
    center_layout.addWidget(self.mapping_panel)
```

(`QWidget` and `QVBoxLayout` are already imported in this module.)

- [ ] **Step 2: Write the failing test**

```python
# tests/test_tutorial_onboarding_app.py
import pytest
from PyQt5.QtWidgets import QApplication
from analysis.statistical_analyzer import StatisticalAnalyzerApp


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def app(qapp):
    w = StatisticalAnalyzerApp()
    yield w
    w.close()


def test_tour_steps_titles_and_count(app):
    steps = app._build_tour_steps()
    titles = [s.title for s in steps]
    assert titles == [
        "Bring In Your Data",
        "Meet Your Variables",
        "Shape Your Analysis",
        "Define and Compute",
        "Full Statistical Transparency",
        "Your Results at a Glance",
        "Help Is Always One Click Away",
    ]


def test_start_tutorial_is_reentrant(app):
    app.start_tutorial()
    assert app._tour_active is True
    first = app._tour_overlay
    app.start_tutorial()             # second call must not create a 2nd overlay
    assert app._tour_overlay is first
    app._tour_overlay.close_tour()
    assert app._tour_active is False
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_tutorial_onboarding_app.py -v`
Expected: FAIL with `AttributeError: 'StatisticalAnalyzerApp' object has no attribute '_build_tour_steps'`.

- [ ] **Step 4: Add the module-level functions**

Add to `src/autopilot/statistical_analyzer_autopilot_pipeline.py` (module level). Ensure the imports `from ui.components.tutorial_overlay import TourStep, TutorialOverlay, from_widgets, from_menu_action` are added near the other UI imports:

```python
def _ap_build_tour_steps(self):
    """Return the ordered 7-step tour. Targets resolve lazily via closures so
    hidden/empty widgets on first run degrade to a centered bubble."""
    from ui.components.tutorial_overlay import TourStep, from_widgets, from_menu_action

    central = self  # host window for coordinate mapping
    help_action = getattr(self, "help_menu", None)
    help_resolver = None
    if help_action is not None:
        help_resolver = from_menu_action(self.menuBar(), self.help_menu.menuAction())

    return [
        TourStep(
            title="Bring In Your Data",
            body=("Start by loading your experimental or clinical data. Open any Excel "
                  "or CSV file here, then pick the relevant sheet from the worksheet "
                  "dropdown. If a sheet is cluttered, use Select Data Ranges to capture "
                  "only the cells you need."),
            tip=("New to the layout? A ready-made Excel template ships with BioMedStatX, "
                 "with one tab per design (t-test, ANOVA, repeated measures, and more), "
                 "each already in the long format the app expects."),
            resolve_rect=from_widgets(central, self.browse_button,
                                      self.auto_sheet_combo, self.range_select_btn),
        ),
        TourStep(
            title="Meet Your Variables",
            body=("A live preview of your table appears as soon as the file loads, and "
                  "each column header becomes a draggable card here. These cards are the "
                  "building blocks you route into your analysis."),
            tip=("Every card shows its detected type (Numeric, Categorical, or Datetime) "
                 "right on the card, so you can check each column at a glance."),
            resolve_rect=from_widgets(central, self.preview_table, self.header_cards_widget),
        ),
        TourStep(
            title="Shape Your Analysis",
            body=("This is where you define your study design. Drag the variable cards "
                  "into the buckets to assign their roles: Dependent Variable, Factor 1 "
                  "and 2, an optional Subject ID for repeated measures, and Covariates. "
                  "As you map, the status line below the buckets shows what each "
                  "assignment means and what is still missing, and the Start button "
                  "activates once your mapping is complete and consistent."),
            tip=("Placed a card in the wrong bucket? Click the small x on the chip to "
                 "return it."),
            resolve_rect=from_widgets(central, self.mapping_panel),
        ),
        TourStep(
            title="Define and Compute",
            body=("Use the group selector to set your comparison cohorts, such as "
                  "Treatment versus Control, then start the analysis. BioMedStatX checks "
                  "normality and distribution to select the appropriate statistical test "
                  "for your data."),
            tip=("Assumption testing runs automatically in the background, so the method "
                 "comes from your data rather than a guess."),
            resolve_rect=from_widgets(central, self.analysis_group_button,
                                      self.start_analysis_button),
        ),
        TourStep(
            title="Full Statistical Transparency",
            body=("BioMedStatX is not a black box. The decision tree traces every step "
                  "the engine took, so you can see why a given test (an ANOVA, a "
                  "Mann-Whitney U, and so on) was selected for your dataset."),
            tip=("Hover over any node to read its full label, or use Maximize to study "
                 "the whole path full-screen."),
            resolve_rect=from_widgets(central, self.decision_tree_panel),
        ),
        TourStep(
            title="Your Results at a Glance",
            body=("The cockpit gathers your test statistics, p-values, effect sizes, and "
                  "a written summary you can drop straight into a manuscript, all in one "
                  "view."),
            tip=("Use Open Output Folder for the full HTML report, including tables, "
                 "plots, and the complete method trace."),
            resolve_rect=from_widgets(central, self.result_cockpit),
        ),
        TourStep(
            title="Help Is Always One Click Away",
            body=("That is the whole workflow, from loading a file to a finished result. "
                  "You can restart this tour anytime from the Help menu, which also holds "
                  "the Getting Started guide and recipe-based help for every design."),
            tip=("The Help menu links to dedicated guides for paired samples, advanced "
                 "ANOVA, and correlation and regression."),
            resolve_rect=help_resolver,
            pulse=True,
        ),
    ]


def _ap_start_tutorial(self):
    """Launch (or no-op if already running) the guided tour overlay."""
    from ui.components.tutorial_overlay import TutorialOverlay
    if getattr(self, "_tour_active", False):
        return
    self._tour_active = True
    overlay = TutorialOverlay(self, self._build_tour_steps())

    def _on_closed():
        self._tour_active = False
        self._tour_overlay = None
        self._mark_tour_seen()

    overlay.closed_callback = _on_closed
    self._tour_overlay = overlay
    overlay.start()


def _mark_tour_seen_impl(self):
    from PyQt5.QtCore import QSettings
    QSettings("BioMedStatX", "BioMedStatX").setValue(
        "onboarding/completed_version", _current_app_version())
```

Add a version accessor (module level) that reuses the updater's version if available, else a constant:

```python
def _current_app_version() -> str:
    try:
        from core.updater import CURRENT_VERSION  # verified: updater.py:11
        return str(CURRENT_VERSION)
    except Exception:
        return "2.0"
```

Add the mixin bindings inside `class AutopilotMixin:` (alongside the existing `name = _ap_name` lines):

```python
    _build_tour_steps = _ap_build_tour_steps
    start_tutorial = _ap_start_tutorial
    _mark_tour_seen = _mark_tour_seen_impl
```

Initialize the flag: in `_ap_init_ui`, near the other `self.current_* = None` assignments at the end, add:

```python
    self._tour_active = False
    self._tour_overlay = None
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_tutorial_onboarding_app.py -v`
Expected: PASS (2 passed). (`help_menu` may be absent until Task 6; the step still builds with `resolve_rect=None`, which the overlay handles.)

- [ ] **Step 6: Commit**

```bash
git add src/autopilot/statistical_analyzer_autopilot_pipeline.py tests/test_tutorial_onboarding_app.py
git commit -m "feat(tour): build 7 tour steps and re-entrant start_tutorial"
```

---

## Task 5: Example-template export action

**Files:**
- Modify: `src/autopilot/statistical_analyzer_autopilot_pipeline.py`
- Test: `tests/test_tutorial_onboarding_app.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_tutorial_onboarding_app.py
def test_export_template_copies_file(app, tmp_path, monkeypatch):
    from PyQt5.QtWidgets import QFileDialog
    from PyQt5.QtGui import QDesktopServices
    target = tmp_path / "out.xlsx"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(target), "")))
    opened = {"url": None}
    monkeypatch.setattr(QDesktopServices, "openUrl",
                        staticmethod(lambda url: opened.__setitem__("url", url)))
    app.export_example_template()
    assert target.exists() and target.stat().st_size > 0
    assert opened["url"] is not None


def test_export_template_cancelled_is_noop(app, tmp_path, monkeypatch):
    from PyQt5.QtWidgets import QFileDialog
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    app.export_example_template()  # no exception, nothing written
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tutorial_onboarding_app.py::test_export_template_copies_file -v`
Expected: FAIL with `AttributeError: ... has no attribute 'export_example_template'`.

- [ ] **Step 3: Write minimal implementation**

Add module-level function (the module already imports `os`, `shutil`; add `from PyQt5.QtCore import QUrl` and `from PyQt5.QtGui import QDesktopServices` and `from PyQt5.QtWidgets import QFileDialog, QMessageBox` if not present):

```python
def _ap_export_example_template(self):
    """Copy the bundled Excel template to a user-chosen location and reveal it."""
    from PyQt5.QtCore import QUrl
    from PyQt5.QtGui import QDesktopServices
    from PyQt5.QtWidgets import QFileDialog, QMessageBox
    src = _resource_path("assets/BioMedStatX_Excel_Template.xlsx")
    if not os.path.exists(src):
        QMessageBox.critical(self, "Error", "Internal template asset missing.")
        return
    default = os.path.join(os.path.expanduser("~"), "Desktop",
                           "BioMedStatX_Template.xlsx")
    target, _ = QFileDialog.getSaveFileName(
        self, "Save Example Template As", default, "Excel Worksheets (*.xlsx)")
    if not target:
        return
    try:
        shutil.copy2(src, target)
        QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(target)))
    except OSError as exc:
        QMessageBox.critical(self, "Export Error", f"Could not write file:\n{exc}")
```

Add the mixin binding inside `class AutopilotMixin:`:

```python
    export_example_template = _ap_export_example_template
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tutorial_onboarding_app.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/autopilot/statistical_analyzer_autopilot_pipeline.py tests/test_tutorial_onboarding_app.py
git commit -m "feat(tour): export bundled example template via native save dialog"
```

---

## Task 6: Menu entries, help_menu reference, and first-run gate

**Files:**
- Modify: `src/analysis/statistical_analyzer.py`
- Test: `tests/test_tutorial_onboarding_app.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_tutorial_onboarding_app.py
def test_help_menu_has_tour_and_template_actions(app):
    assert hasattr(app, "help_menu")
    texts = [a.text() for a in app.help_menu.actions()]
    assert any("Interactive Tour" in t for t in texts)
    assert any("Example Template" in t for t in texts)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tutorial_onboarding_app.py::test_help_menu_has_tour_and_template_actions -v`
Expected: FAIL with `AttributeError: ... has no attribute 'help_menu'`.

- [ ] **Step 3: Edit create_menu**

In `src/analysis/statistical_analyzer.py`, find:

```python
        # Help menu
        help_menu = menubar.addMenu('&Help')

        # Getting Started should be first
        getting_started_action = QAction('Getting Started', self)
```

Replace the first two lines and insert the new actions:

```python
        # Help menu
        self.help_menu = menubar.addMenu('&Help')
        help_menu = self.help_menu

        tour_action = QAction('Interactive Tour', self)
        tour_action.triggered.connect(self.start_tutorial)
        help_menu.addAction(tour_action)

        template_action = QAction('Save Example Template...', self)
        template_action.triggered.connect(self.export_example_template)
        help_menu.addAction(template_action)

        help_menu.addSeparator()

        # Getting Started should be first
        getting_started_action = QAction('Getting Started', self)
```

- [ ] **Step 4: Add the first-run gate in `__init__`**

In `StatisticalAnalyzerApp.__init__`, after the UI is built and the window is shown (after `self.init_ui()` / `self.show()` equivalent, at the end of `__init__`), add:

```python
        from PyQt5.QtCore import QTimer, QSettings
        from autopilot.statistical_analyzer_autopilot_pipeline import (
            should_offer_tour, _current_app_version,
        )
        _stored = QSettings("BioMedStatX", "BioMedStatX").value(
            "onboarding/completed_version", "")
        if should_offer_tour(_stored, _current_app_version()):
            QTimer.singleShot(400, self._maybe_offer_tour)
```

Add the `_maybe_offer_tour` method to `StatisticalAnalyzerApp` (this class, not the mixin, since the welcome dialog lives with the other `show_*` dialogs):

```python
    def _maybe_offer_tour(self):
        from PyQt5.QtWidgets import QMessageBox
        box = QMessageBox(self)
        box.setWindowTitle("Welcome to BioMedStatX")
        box.setText("New here? Take a 60-second tour of the workflow.")
        start_btn = box.addButton("Start tour", QMessageBox.AcceptRole)
        box.addButton("Maybe later", QMessageBox.RejectRole)
        box.exec_()
        if box.clickedButton() is start_btn:
            self.start_tutorial()
        else:
            self._mark_tour_seen()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_tutorial_onboarding_app.py -v`
Expected: PASS (5 passed).

- [ ] **Step 6: Run the full suite**

Run: `pytest tests/test_tutorial_overlay.py tests/test_tutorial_onboarding_app.py -v`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/analysis/statistical_analyzer.py tests/test_tutorial_onboarding_app.py
git commit -m "feat(tour): Help-menu entries and first-run welcome gate"
```

---

## Task 7: Visual styling pass (frontend-design) + manual QA

**Files:**
- Modify: `assets/BioMedStatX_2_0.qss`
- Modify: `src/ui/components/tutorial_overlay.py` (apply elevation/animation only)

- [ ] **Step 1: Invoke the frontend-design skill (two parts)**

1. QSS (bubble widgets only): style the objectNames already set — `tutorialBubble`, `tutorialTitle`, `tutorialBody`, `tutorialTip`, `tutorialProgress`, `tutorialNext`, `tutorialBack`, `tutorialSkip` — in `assets/BioMedStatX_2_0.qss`, matching the dark dashboard palette. Reuse `_apply_elevation()` for the bubble shadow.
2. paintEvent constants (spotlight ring + pulse): the ring and pulse animation are already implemented in `tutorial_overlay.py` (Task 2). QSS cannot reach a custom paintEvent, so tune the look by editing the class constants `RING_COLOR`, `RING_WIDTH`, `PULSE_MIN_ALPHA`, `PULSE_MAX_ALPHA`, `PULSE_PERIOD_MS` directly. Do not try to style the ring via QSS.

- [ ] **Step 2: Manual QA checklist (run the app)**

Run: `./start.sh` (macOS/Linux) and verify:
- First launch on a clean profile shows the welcome dialog; "Start tour" runs the tour; "Maybe later" does not, and neither re-appears on the next launch.
- `Help -> Interactive Tour` restarts the tour anytime.
- All 7 steps render; the spotlight tracks the right widget; the bubble never clips off-screen.
- On an empty app (no file loaded), steps whose targets are hidden (preview table, cockpit) fall back to a centered bubble without crashing.
- Resize the window and drag the splitter mid-tour: the spotlight re-anchors.
- Pressing Space/Enter advances the tour and does NOT start an analysis.
- `Help -> Save Example Template...` writes the file and opens the containing folder (Finder/Explorer).

- [ ] **Step 3: Commit**

```bash
git add assets/BioMedStatX_2_0.qss src/ui/components/tutorial_overlay.py
git commit -m "style(tour): dark-palette styling for overlay bubble and spotlight"
```

---

## Self-review notes

- **Spec coverage:** Section 3 UX flow → Tasks 4/6. Section 4 content → Task 4 (`_build_tour_steps`). Section 5 architecture → Tasks 1/2. Section 6 targeting + promotions → Task 4 Step 1. Section 7 edge guards: singleShot(0) (Task 2 `_refresh`), keyboard capture (Task 2 `eventFilter`/`grabKeyboard`), DestinationOut paint plus teal accent ring and pulse animation (Task 2 `paintEvent` + `QVariantAnimation`, tuned in Task 7), missing/hidden fallback (Task 1 `resolve_union_rect` + Task 2 test), re-entrancy (Task 4 `_tour_active`). Section 8 QSettings → Tasks 3/6. Section 9 template export → Task 5 (no `.spec` change, reuses `_resource_path`). Section 10 styling → Task 7.
- **No placeholders:** every code step shows complete code; commands have expected output.
- **Type consistency:** `resolve_rect`/`from_widgets`/`from_menu_action`, `closed_callback`, `current_index`/`is_first`/`is_last`, `current_spotlight`, `start_tutorial`/`_build_tour_steps`/`export_example_template`/`_tour_active`/`_tour_overlay`/`_mark_tour_seen`/`should_offer_tour`/`_current_app_version` are used consistently across tasks.
- **Version source confirmed:** `_current_app_version` imports `CURRENT_VERSION` from `core/updater.py:11` (value `"2.0"`); the literal fallback matches.
