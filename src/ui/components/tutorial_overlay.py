"""Passive guided-tour overlay for the BioMedStatX main window.

A full-window translucent overlay dims the app, punches an antialiased
spotlight around the current target, and shows a bubble with Back/Next/Skip.
The tour is linear and passive: it never triggers the real widgets, so it
cannot corrupt analysis state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from PyQt5.QtCore import Qt, QRect, QPoint, QTimer, QEvent, QVariantAnimation
from PyQt5.QtGui import QPainter, QColor, QPainterPath, QPen
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QFrame, QVBoxLayout, QHBoxLayout,
)

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


class TutorialOverlay(QWidget):
    SPOTLIGHT_PAD = 10
    SPOTLIGHT_RADIUS = 14
    DIM_COLOR = QColor(22, 49, 58, 170)      # blue-grey, softer on light UI
    # Spotlight accent ring — custom paintEvent, not QSS.
    RING_COLOR = QColor(15, 118, 110)         # teal accent (#0f766e)
    RING_WIDTH = 2.5
    PULSE_MIN_ALPHA = 60
    PULSE_MAX_ALPHA = 220
    PULSE_PERIOD_MS = 1200

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
        # Drop shadow for elevation (matches _apply_elevation style)
        from PyQt5.QtWidgets import QGraphicsDropShadowEffect
        shadow = QGraphicsDropShadowEffect(self.bubble)
        shadow.setBlurRadius(24)
        shadow.setXOffset(0)
        shadow.setYOffset(6)
        shadow.setColor(QColor(0, 0, 0, 46))
        self.bubble.setGraphicsEffect(shadow)

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
