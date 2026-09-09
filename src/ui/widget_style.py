"""Shared Qt widget styling helpers.

`apply_elevation` and `configure_dialog` were each copy-pasted across the UI
modules (elevation twice plus an inline third copy in the tutorial overlay,
dialog configuration three times — one of which had quietly dropped the
`remove_context_help` flag). They are defined once here so the windows keep
picking up the same QSS rules and the same shadow.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QDialog, QGraphicsDropShadowEffect


def apply_elevation(widget, radius=18, x_offset=0, y_offset=4, opacity=0.18):
    """Apply a drop shadow to give a widget visual elevation. QSS cannot do this."""
    shadow = QGraphicsDropShadowEffect(widget)
    shadow.setBlurRadius(radius)
    shadow.setOffset(x_offset, y_offset)
    shadow.setColor(QColor(0, 0, 0, int(255 * opacity)))
    widget.setGraphicsEffect(shadow)


def configure_dialog(dialog, object_name=None, remove_context_help=True):
    """Apply common dialog defaults so all windows pick up the same QSS rules."""
    if object_name:
        dialog.setObjectName(object_name)
    if remove_context_help and isinstance(dialog, QDialog):
        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
