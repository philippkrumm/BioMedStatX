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
