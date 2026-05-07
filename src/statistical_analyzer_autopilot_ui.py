import os
import re
import string
import tempfile
import time

import numpy as np
import pandas as pd
from PyQt5.QtCore import (
    QEasingCurve,
    QMimeData,
    QPoint,
    QPropertyAnimation,
    QSequentialAnimationGroup,
    QTimer,
    Qt,
    QUrl,
    pyqtSignal,
)
from PyQt5.QtGui import QColor, QDesktopServices, QDrag, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGraphicsDropShadowEffect,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from decisiontreevisualizer import DecisionTreeVisualizer

try:
    from help_content import HELP_RECIPES
except ImportError:
    HELP_RECIPES = []

AUTO_PILOT_STEP_ORDER = ["load", "map", "analyze", "results"]

def _safe_file_slug(text):
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text))
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or "analysis"


def _infer_column_kind(series):
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "categorical"


def _looks_like_subject(column_name, series=None):
    lowered = str(column_name).strip().lower()
    strong_keywords = ("subject", "subjekt", "patient", "participant", "animal", "mouse", "id")
    weak_keywords = ("sample",)

    strong_hit = any(keyword in lowered for keyword in strong_keywords)
    weak_hit = any(keyword in lowered for keyword in weak_keywords)
    if not (strong_hit or weak_hit):
        return False

    if series is None:
        return strong_hit

    non_null = series.dropna()
    if non_null.empty:
        return False

    unique_count = int(non_null.nunique())
    unique_ratio = unique_count / max(len(non_null), 1)

    # Estimate whether values look like true IDs (e.g. S01, WT_03, Mouse12).
    unique_values = [str(value).strip().lower() for value in non_null.unique().tolist()[:80]]
    id_like = 0
    for value in unique_values:
        if not value:
            continue
        has_letters = any(ch.isalpha() for ch in value)
        has_digits = any(ch.isdigit() for ch in value)
        if (value.startswith("s") and value[1:].isdigit()) or (has_letters and has_digits):
            id_like += 1
    id_like_ratio = id_like / max(len(unique_values), 1)

    if strong_hit:
        return unique_ratio >= 0.25 or unique_count >= 8 or id_like_ratio >= 0.40

    # For "sample" columns be stricter to avoid misclassifying group labels like WT/KO.
    return unique_ratio >= 0.60 or unique_count >= 8 or id_like_ratio >= 0.40


def _detect_wide_format(df):
    """
    Returns {"subject_col": str, "value_cols": list[str]} if df looks like
    a wide-format paired/repeated design, otherwise returns None.

    Signature: 1 subject-like column + 2–8 numeric columns + no categorical
    column that looks like a group factor (2 unique values) + the subject column
    has high uniqueness (each row is a distinct subject, not repeated).
    """
    if len(df) < 3:
        return None

    # Find all numeric columns
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    # Find exactly one subject-like column among non-numeric columns first,
    # then also check numeric columns (subject IDs can be integers like 1,2,3)
    subject_candidates = []
    for col in df.columns:
        if _looks_like_subject(col, df[col]):
            subject_candidates.append(col)

    if len(subject_candidates) != 1:
        return None

    subject_col = subject_candidates[0]

    # Value columns = all numeric columns that are not the subject column
    value_cols = [c for c in numeric_cols if c != subject_col]
    if not (2 <= len(value_cols) <= 8):
        return None

    # Guard: no categorical column with exactly 2 unique values (would indicate long format with a Group col)
    categorical_cols = [c for c in non_numeric_cols if c != subject_col]
    for col in categorical_cols:
        if df[col].nunique() == 2:
            return None

    # Key discriminator: in wide format every row IS a subject, so uniqueness ratio is high.
    # In long format the subject repeats across rows (once per condition), so ratio is low.
    unique_ratio = df[subject_col].nunique() / max(len(df), 1)
    if unique_ratio < 0.8:
        return None

    return {"subject_col": subject_col, "value_cols": value_cols}


def _pivot_wide_to_long(df, subject_col, value_cols):
    """
    Melts a wide-format DataFrame into long format.
    Returns a new DataFrame with columns [subject_col, 'Condition', 'Value'].
    """
    var_name = "_Condition" if "Condition" in df.columns else "Condition"
    return pd.melt(
        df,
        id_vars=[subject_col],
        value_vars=value_cols,
        var_name=var_name,
        value_name="Value",
    )


def _to_display_coords(r, c):
    col_letter = chr(ord("A") + c) if c < 26 else f"A{chr(ord('A') + c - 26)}"
    return f"Row {r + 1}, Col {col_letter}"


def _selected_indexes_to_ranges(indexes):
    """
    Convert QTableWidget.selectedIndexes() to a single bounding-box range dict.
    Gaps within the bounding box are fine — blanks coerce to NaN and are dropped later.
    For non-contiguous blocks call this per selection and accumulate into the group list.
    """
    if not indexes:
        return []
    rows = [idx.row() for idx in indexes]
    cols = [idx.column() for idx in indexes]
    return [{"rows": (min(rows), max(rows)), "cols": (min(cols), max(cols))}]


def extract_from_coordinates(df_raw, selection_map, replicate_type="biological",
                              value_col="Value", group_col="Group"):
    """
    df_raw: raw DataFrame loaded with header=None, dtype=str.
            All row/col coordinates are 0-based iloc indices.
    selection_map: {
        "Group_A": [{"rows": (3, 10), "cols": (1, 3)}],
        "Group_B": [{"rows": (3, 10), "cols": (5, 7)}],
    }
    replicate_type: "biological" → each cell = 1 N; "technical" → mean per block.
    Returns (result_df, nan_report).
    nan_report: {group_name: total NaN count across all its ranges}.
    n_replicates column is float64 throughout (NaN for biological rows).
    """
    frames = []
    nan_report = {}

    for group_name, ranges in selection_map.items():
        nan_report.setdefault(group_name, 0)
        for rng in ranges:
            r1, r2 = rng["rows"]
            c1, c2 = rng["cols"]
            block = df_raw.iloc[r1 : r2 + 1, c1 : c2 + 1]
            vals_raw = block.values.flatten()
            vals = pd.to_numeric(vals_raw, errors="coerce")
            n_nan = int(np.isnan(vals).sum())
            nan_report[group_name] += n_nan
            range_label = f"r{r1}:{r2}|c{c1}:{c2}"
            n_valid = int(np.sum(~np.isnan(vals)))

            if replicate_type == "technical":
                mean_val = float(np.nanmean(vals)) if n_valid > 0 else np.nan
                frames.append(pd.DataFrame({
                    group_col: [group_name],
                    value_col: [mean_val],
                    "Source_Range": [range_label],
                    "n_replicates": [float(n_valid)],
                }))
            else:
                n = len(vals)
                frames.append(pd.DataFrame({
                    group_col: [group_name] * n,
                    value_col: vals,
                    "Source_Range": [range_label] * n,
                    "n_replicates": [np.nan] * n,
                }))

    result = pd.concat(frames, ignore_index=True)
    return result.dropna(subset=[value_col]), nan_report


def _sorted_unique(values):
    unique_values = []
    for value in values:
        if pd.isna(value):
            continue
        if value not in unique_values:
            unique_values.append(value)
    return sorted(unique_values, key=lambda item: str(item))


class PipelineStepChip(QFrame):
    def __init__(self, icon_text, title, parent=None):
        super().__init__(parent)
        self.setObjectName("pipelineStepChip")
        self.setProperty("state", "idle")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)

        self.icon_label = QLabel(icon_text)
        self.icon_label.setAlignment(Qt.AlignCenter)
        self.icon_label.setObjectName("pipelineStepIcon")
        layout.addWidget(self.icon_label)

        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setObjectName("pipelineStepTitle")
        layout.addWidget(self.title_label)

    def set_state(self, state):
        self.setProperty("state", state)
        self.style().unpolish(self)
        self.style().polish(self)


class PipelineTrackerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.steps = {}
        self._pulse_effect = None
        self._pulse_animation = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        step_specs = [
            ("load", "Load", "L"),
            ("map", "Map", "M"),
            ("analyze", "Analyze", "A"),
            ("results", "Results", "R"),
        ]

        for index, (step_key, title, icon_text) in enumerate(step_specs):
            chip = PipelineStepChip(icon_text, title)
            self.steps[step_key] = chip
            layout.addWidget(chip, 1)
            if index < len(step_specs) - 1:
                connector = QLabel("···")
                connector.setAlignment(Qt.AlignCenter)
                connector.setObjectName("pipelineConnector")
                layout.addWidget(connector)

        self._pulse_effect = QGraphicsOpacityEffect(self.steps["analyze"])
        self.steps["analyze"].setGraphicsEffect(self._pulse_effect)
        self._pulse_animation = QPropertyAnimation(self._pulse_effect, b"opacity", self)
        self._pulse_animation.setDuration(850)
        self._pulse_animation.setStartValue(1.0)
        self._pulse_animation.setEndValue(0.45)
        self._pulse_animation.setEasingCurve(QEasingCurve.InOutQuad)
        self._pulse_animation.setLoopCount(-1)

        self.set_stage("load", running=False)

    def set_stage(self, stage, running=False):
        current_index = AUTO_PILOT_STEP_ORDER.index(stage)
        for index, step_key in enumerate(AUTO_PILOT_STEP_ORDER):
            if index < current_index:
                state = "completed"
            elif index == current_index:
                state = "running" if running else "active"
            else:
                state = "idle"
            self.steps[step_key].set_state(state)

        if stage == "analyze" and running:
            self._pulse_animation.start()
        else:
            self._pulse_animation.stop()
            self._pulse_effect.setOpacity(1.0)


class DraggableColumnCard(QFrame):
    def __init__(self, column_name, column_kind, preview_text, parent=None):
        super().__init__(parent)
        self.column_name = column_name
        self.column_kind = column_kind
        self._drag_start_position = QPoint()
        self.setObjectName("columnCard")
        self.setCursor(Qt.OpenHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(3)

        title = QLabel(str(column_name))
        title.setObjectName("columnCardTitle")
        layout.addWidget(title)

        meta = QLabel(f"{column_kind.title()} column")
        meta.setObjectName("columnCardMeta")
        layout.addWidget(meta)

        preview = QLabel(preview_text)
        preview.setObjectName("columnCardPreview")
        preview.setWordWrap(True)
        layout.addWidget(preview)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_start_position = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not (event.buttons() & Qt.LeftButton):
            return
        if (event.pos() - self._drag_start_position).manhattanLength() < QApplication.startDragDistance():
            return

        drag = QDrag(self)
        mime_data = QMimeData()
        mime_data.setText(f"{self.column_name}\t{self.column_kind}")
        drag.setMimeData(mime_data)
        drag.setPixmap(self.grab())
        drag.exec_(Qt.CopyAction)


class BucketChip(QFrame):
    removed = pyqtSignal(str)

    def __init__(self, column_name, parent=None):
        super().__init__(parent)
        self.column_name = column_name
        self.setObjectName("bucketChip")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(8)

        label = QLabel(str(column_name))
        label.setObjectName("bucketChipLabel")
        layout.addWidget(label)

        remove_button = QToolButton()
        remove_button.setText("x")
        remove_button.setObjectName("bucketChipRemove")
        remove_button.clicked.connect(lambda: self.removed.emit(self.column_name))
        layout.addWidget(remove_button)


class MappingBucketWidget(QFrame):
    changed = pyqtSignal()

    def __init__(self, title, placeholder, accepted_kinds=None, allow_multiple=False,
                 info_text="", help_recipe_id=None, parent=None):
        super().__init__(parent)
        self.setObjectName("mappingBucket")
        self.setAcceptDrops(True)
        self.accepted_kinds = set(accepted_kinds or [])
        self.allow_multiple = allow_multiple
        self._assignments = []
        self.help_recipe_id = help_recipe_id

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("mappingBucketTitle")
        if info_text:
            title_row = QHBoxLayout()
            title_row.setContentsMargins(0, 0, 0, 0)
            title_row.setSpacing(4)
            title_row.addWidget(self.title_label)
            title_row.addStretch()
            _info_btn = QPushButton("i")
            _info_btn.setObjectName("bucketInfoButton")
            _info_btn.setFlat(True)
            _info_btn.setFocusPolicy(Qt.NoFocus)
            _captured_text = info_text
            _captured_title = title
            _info_btn.clicked.connect(
                lambda: self._show_info_dialog(_captured_title, _captured_text))
            title_row.addWidget(_info_btn)
            layout.addLayout(title_row)
        else:
            layout.addWidget(self.title_label)

        self.placeholder_label = QLabel(placeholder)
        self.placeholder_label.setObjectName("mappingBucketPlaceholder")
        self.placeholder_label.setWordWrap(True)
        layout.addWidget(self.placeholder_label)

        self.chip_container = QWidget()
        self.chip_layout = QVBoxLayout(self.chip_container)
        self.chip_layout.setContentsMargins(0, 0, 0, 0)
        self.chip_layout.setSpacing(6)
        layout.addWidget(self.chip_container)

    def set_allow_multiple(self, allow_multiple):
        self.allow_multiple = allow_multiple
        if not allow_multiple and len(self._assignments) > 1:
            first_assignment = self._assignments[0]
            self.clear_assignments()
            self.assign_column(first_assignment[0], first_assignment[1])

    def _can_accept_kind(self, column_kind):
        return not self.accepted_kinds or column_kind in self.accepted_kinds

    def _show_info_dialog(self, title, text):
        recipe_id = self.help_recipe_id
        main_window = self.window()
        if main_window is not None and hasattr(main_window, "_resolve_help_recipe_for_bucket"):
            try:
                resolved_recipe = main_window._resolve_help_recipe_for_bucket(self, recipe_id)
                if resolved_recipe:
                    recipe_id = resolved_recipe
            except Exception:
                pass

        if not recipe_id:
            QMessageBox.information(self, title, text)
            return

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle(title)
        msg.setText(text)

        recipe_title = None
        if recipe_id:
            for recipe in HELP_RECIPES:
                if recipe.get("id") == recipe_id:
                    recipe_title = recipe.get("title")
                    break
        if recipe_title:
            msg.setInformativeText(f"Suggested recipe: {recipe_title}")

        open_help_button = msg.addButton("Open in Help Hub", QMessageBox.ActionRole)
        msg.addButton(QMessageBox.Ok)
        msg.exec_()

        if msg.clickedButton() == open_help_button:
            if main_window is not None and hasattr(main_window, "show_help_hub"):
                main_window.show_help_hub(recipe_id)

    def dragEnterEvent(self, event):
        payload = event.mimeData().text().split("\t")
        if len(payload) == 2 and self._can_accept_kind(payload[1]):
            self.setProperty("dragHover", True)
            self.style().unpolish(self)
            self.style().polish(self)
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.setProperty("dragHover", False)
        self.style().unpolish(self)
        self.style().polish(self)
        super().dragLeaveEvent(event)

    def dropEvent(self, event):
        self.setProperty("dragHover", False)
        self.style().unpolish(self)
        self.style().polish(self)

        payload = event.mimeData().text().split("\t")
        if len(payload) != 2:
            event.ignore()
            return
        column_name, column_kind = payload
        if self.assign_column(column_name, column_kind):
            event.acceptProposedAction()
        else:
            event.ignore()

    def assign_column(self, column_name, column_kind):
        if not self._can_accept_kind(column_kind):
            return False
        if not self.allow_multiple:
            self.clear_assignments()
        elif any(existing_name == column_name for existing_name, _, _ in self._assignments):
            return True

        chip = BucketChip(column_name)
        chip.removed.connect(self.remove_column)
        self.chip_layout.addWidget(chip)
        self._assignments.append((column_name, column_kind, chip))
        self._refresh_placeholder()
        self.changed.emit()
        return True

    def remove_column(self, column_name):
        remaining = []
        for existing_name, existing_kind, chip in self._assignments:
            if existing_name == column_name:
                chip.setParent(None)
                chip.deleteLater()
                continue
            remaining.append((existing_name, existing_kind, chip))
        self._assignments = remaining
        self._refresh_placeholder()
        self.changed.emit()

    def clear_assignments(self):
        for _, _, chip in self._assignments:
            chip.setParent(None)
            chip.deleteLater()
        self._assignments = []
        self._refresh_placeholder()
        self.changed.emit()

    def get_assigned_columns(self):
        return [column_name for column_name, _, _ in self._assignments]

    def get_assigned_kinds(self):
        return [column_kind for _, column_kind, _ in self._assignments]

    def _refresh_placeholder(self):
        self.placeholder_label.setVisible(len(self._assignments) == 0)


class FilterBucketWidget(MappingBucketWidget):
    """Bucket that accepts any column and shows a value-picker dropdown after drop.

    Lets the user restrict the analysis to a subset of rows before running any
    statistical model (e.g. 'OP-Group = 1' → only On-Pump patients).

    Usage:
        bucket = FilterBucketWidget(get_df=lambda: self.df)
        bucket.get_filter()  # → ('OP-Group ...', 1)  or  None
    """

    def __init__(self, get_df, parent=None):
        super().__init__(
            title="Filter (optional)",
            placeholder="Drop any column here to restrict the analysis to a subset of rows.",
            accepted_kinds={"numeric", "categorical", "datetime"},
            allow_multiple=False,
            info_text=(
                "Restricts the analysis to a subset of rows. Drop a categorical column here, "
                "then select a value from the dropdown — the analysis will run only on the "
                "filtered rows. Useful for subgroup analyses (e.g. only On-Pump patients). "
                "The row count (n) is shown after selection."
            ),
            parent=parent,
        )
        self._get_df = get_df
        self._filter_col = None
        self._filter_val = None

        # Value picker — hidden until a column is dropped
        self._value_combo = QComboBox()
        self._value_combo.setVisible(False)
        self._value_combo.currentIndexChanged.connect(self._on_value_changed)
        self.layout().addWidget(self._value_combo)

        self._filter_label = QLabel("")
        self._filter_label.setObjectName("panelDescription")
        self._filter_label.setWordWrap(True)
        self.layout().addWidget(self._filter_label)

    def assign_column(self, column_name, column_kind):
        accepted = super().assign_column(column_name, column_kind)
        if accepted:
            self._filter_col = column_name
            self._populate_values(column_name)
        return accepted

    def remove_column(self, column_name):
        super().remove_column(column_name)
        self._filter_col = None
        self._filter_val = None
        self._value_combo.setVisible(False)
        self._value_combo.clear()
        self._filter_label.setText("")

    def clear_assignments(self):
        super().clear_assignments()
        self._filter_col = None
        self._filter_val = None
        self._value_combo.setVisible(False)
        self._value_combo.clear()
        self._filter_label.setText("")

    def _populate_values(self, column_name):
        df = self._get_df()
        if df is None or column_name not in df.columns:
            return
        unique_vals = sorted(df[column_name].dropna().unique(), key=lambda v: str(v))
        self._value_combo.blockSignals(True)
        self._value_combo.clear()
        for v in unique_vals:
            self._value_combo.addItem(str(v), userData=v)
        self._value_combo.blockSignals(False)
        self._value_combo.setVisible(True)
        # Set default
        if unique_vals:
            self._filter_val = unique_vals[0]
            n = (df[column_name] == unique_vals[0]).sum()
            self._filter_label.setText(f"Analysis restricted to n={n} rows.")
        self.changed.emit()

    def _on_value_changed(self, index):
        val = self._value_combo.itemData(index)
        self._filter_val = val
        df = self._get_df()
        if df is not None and self._filter_col in df.columns and val is not None:
            n = (df[self._filter_col] == val).sum()
            self._filter_label.setText(f"Analysis restricted to n={n} rows.")
        self.changed.emit()

    def get_filter(self):
        """Return (col, val) tuple or None if no filter is set."""
        if self._filter_col and self._filter_val is not None:
            return (self._filter_col, self._filter_val)
        return None


class DecisionTreeOverlayWidget(QFrame):
    closed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("decisionTreeOverlay")
        self.setVisible(False)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAttribute(Qt.WA_DeleteOnClose, False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        header = QHBoxLayout()
        title = QLabel("Decision Tree (Expanded)")
        title.setObjectName("panelTitle")
        header.addWidget(title)
        header.addStretch()

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.hide_overlay)
        header.addWidget(self.close_button)
        layout.addLayout(header)

        self.overlay_status = QLabel("Run an analysis to render the decision tree.")
        self.overlay_status.setObjectName("panelDescription")
        self.overlay_status.setWordWrap(True)
        layout.addWidget(self.overlay_status)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setObjectName("decisionTreeOverlayScroll")
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.overlay_image_label = QLabel("Decision tree preview")
        self.overlay_image_label.setObjectName("decisionTreeOverlayImage")
        self.overlay_image_label.setAlignment(Qt.AlignCenter)
        self.overlay_image_label.setMinimumSize(1400, 900)
        self.overlay_image_label.setWordWrap(True)
        self.scroll_area.setWidget(self.overlay_image_label)
        layout.addWidget(self.scroll_area, 1)

        self._render_path = None
        self._source_pixmap = QPixmap()

    def _target_render_size(self):
        parent = self.parentWidget()
        parent_width = parent.width() if parent is not None else 1400
        parent_height = parent.height() if parent is not None else 1000

        # Keep margins and header space to avoid clipping while maximizing usable area.
        target_width = max(1200, parent_width - 40)
        target_height = max(760, parent_height - 170)
        return target_width, target_height

    def show_overlay(self):
        if self.parentWidget() is not None:
            self.setGeometry(self.parentWidget().rect())
        self.show()
        self.raise_()
        self.activateWindow()
        self.setFocus(Qt.ActiveWindowFocusReason)

    def hide_overlay(self):
        self.hide()
        self.closed.emit()

    def closeEvent(self, event):
        # If the overlay gets a window-close event, treat it like "hide" instead
        # of destroying the hosting UI hierarchy.
        self.hide_overlay()
        event.ignore()

    def set_placeholder(self, text):
        self.overlay_status.setText(text)
        self._render_path = None
        self._source_pixmap = QPixmap()
        self.overlay_image_label.setPixmap(QPixmap())
        self.overlay_image_label.setText("Decision tree preview")

    def set_render_path(self, render_path, status_text=None):
        if render_path != self._render_path:
            pixmap = QPixmap(render_path)
            if pixmap.isNull():
                self.set_placeholder("Decision tree image could not be loaded.")
                return
            self._render_path = render_path
            self._source_pixmap = pixmap

        if self._source_pixmap.isNull():
            self.set_placeholder("Decision tree image could not be loaded.")
            return

        if status_text:
            self.overlay_status.setText(status_text)
        self.overlay_image_label.setText("")
        target_width, target_height = self._target_render_size()
        scaled = self._source_pixmap.scaled(target_width, target_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.overlay_image_label.setMinimumSize(scaled.width(), scaled.height())
        self.overlay_image_label.setPixmap(scaled)

    def refresh_scaled_render(self):
        if self._source_pixmap.isNull():
            return

        target_width, target_height = self._target_render_size()
        refreshed = self._source_pixmap.scaled(target_width, target_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.overlay_image_label.setMinimumSize(refreshed.width(), refreshed.height())
        self.overlay_image_label.setPixmap(refreshed)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.hide_overlay()
            event.accept()
            return
        super().keyPressEvent(event)

    def cleanup(self):
        self.hide()
        self._render_path = None
        self._source_pixmap = QPixmap()
        self.overlay_image_label.clear()


class DecisionTreePanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("decisionTreePanel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        title_row = QHBoxLayout()
        title = QLabel("Decision Tree Dashboard")
        title.setObjectName("panelTitle")
        title_row.addWidget(title)
        title_row.addStretch()

        self.maximize_button = QPushButton("Maximize")
        self.maximize_button.setObjectName("decisionTreeMaximizeButton")
        self.maximize_button.setEnabled(False)
        self.maximize_button.clicked.connect(self.show_overlay)
        title_row.addWidget(self.maximize_button)
        layout.addLayout(title_row)

        self.status_label = QLabel("The statistical decision path will appear here after the analysis.")
        self.status_label.setObjectName("panelDescription")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.image_label = QLabel("Decision tree preview")
        self.image_label.setObjectName("decisionTreeImage")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(320)
        # Prevent pixmap size-hint feedback loops that can make split layouts "wobble".
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_label.setWordWrap(True)
        layout.addWidget(self.image_label, 1)

        self.last_render_path = None
        self.overlay = None
        self._source_pixmap = QPixmap()
        self._last_scaled_size = None
        self._resize_scale_timer = QTimer(self)
        self._resize_scale_timer.setSingleShot(True)
        self._resize_scale_timer.setInterval(60)
        self._resize_scale_timer.timeout.connect(self._refresh_scaled_preview)

    def show_placeholder(self, text):
        self.status_label.setText(text)
        self.last_render_path = None
        self._source_pixmap = QPixmap()
        self._last_scaled_size = None
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText("Decision tree preview")
        self.maximize_button.setEnabled(False)
        self._sync_overlay_placeholder(text)

    def _refresh_scaled_preview(self, force=False):
        if self._source_pixmap.isNull():
            return

        target_size = self.image_label.size()
        if target_size.width() < 10 or target_size.height() < 10:
            return

        size_tuple = (target_size.width(), target_size.height())
        if not force and self._last_scaled_size == size_tuple:
            return

        scaled = self._source_pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)
        self._last_scaled_size = size_tuple

    def _ensure_overlay(self):
        host_window = self.window()
        if host_window is None:
            return None

        if self.overlay is None or self.overlay.parentWidget() is not host_window:
            self.overlay = DecisionTreeOverlayWidget(host_window)
        self.overlay.setGeometry(host_window.rect())
        return self.overlay

    def _sync_overlay_placeholder(self, text):
        if self.overlay is not None:
            self.overlay.set_placeholder(text)

    def _sync_overlay_render(self):
        if self.overlay is None:
            return
        overlay = self.overlay
        if self.last_render_path and os.path.exists(self.last_render_path):
            overlay.set_render_path(self.last_render_path, self.status_label.text())
        else:
            overlay.set_placeholder(self.status_label.text())

    def show_overlay(self):
        overlay = self._ensure_overlay()
        if overlay is None:
            return
        self._sync_overlay_render()
        overlay.show_overlay()

    def update_results(self, results):
        try:
            import tempfile

            output_base = os.path.join(tempfile.gettempdir(), f"biomedstatx_tree_{int(time.time() * 1000)}")
            rendered_path = DecisionTreeVisualizer.visualize(results, output_path=output_base)
            pixmap = QPixmap(rendered_path)
            if pixmap.isNull():
                raise ValueError("Decision tree image could not be loaded.")
            self.last_render_path = rendered_path
            self._source_pixmap = pixmap
            self._last_scaled_size = None
            self.status_label.setText(results.get("tested_against", results.get("test", "Decision path ready.")))
            self.image_label.setText("")
            self._refresh_scaled_preview(force=True)
            self.maximize_button.setEnabled(True)
            self._sync_overlay_render()
        except Exception as exc:
            self.show_placeholder(f"Decision tree could not be rendered: {exc}")

    def resizeEvent(self, event):
        if self.overlay is not None and self.overlay.isVisible():
            self.overlay.setGeometry(self.overlay.parentWidget().rect())
            self.overlay.refresh_scaled_render()

        if not self._source_pixmap.isNull():
            self._resize_scale_timer.start()
        super().resizeEvent(event)

    def cleanup(self):
        self._resize_scale_timer.stop()
        self._source_pixmap = QPixmap()
        self.last_render_path = None
        self._last_scaled_size = None
        self.image_label.clear()

        if self.overlay is not None:
            try:
                self.overlay.cleanup()
            except Exception:
                pass
            self.overlay.setParent(None)
            self.overlay.deleteLater()
            self.overlay = None


class GlowFrame(QFrame):
    """QFrame that softly glows on mouse hover via an animated drop shadow."""

    def __init__(self, parent=None, glow_color=None, radius=22):
        super().__init__(parent)
        if glow_color is None:
            glow_color = QColor(20, 112, 160, 70)
        self._glow_radius = radius
        self._shadow = QGraphicsDropShadowEffect(self)
        self._shadow.setBlurRadius(0)
        self._shadow.setOffset(0, 0)
        self._shadow.setColor(glow_color)
        self.setGraphicsEffect(self._shadow)
        self._anim = QPropertyAnimation(self._shadow, b"blurRadius", self)
        self._anim.setDuration(200)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self.setMouseTracking(True)

    def enterEvent(self, event):
        self._anim.stop()
        self._anim.setStartValue(self._shadow.blurRadius())
        self._anim.setEndValue(self._glow_radius)
        self._anim.start()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._anim.stop()
        self._anim.setStartValue(self._shadow.blurRadius())
        self._anim.setEndValue(0)
        self._anim.start()
        super().leaveEvent(event)


class ConfettiOverlay(QWidget):
    """Full-window confetti burst overlay. Self-destructs after ~2 s."""

    _COLORS = ["#0f766e", "#1f7a5a", "#b7791f", "#9f3a38", "#1d4ed8", "#7c3aed", "#2dd4bf"]

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.resize(parent.size())
        self.raise_()
        self.show()

        import random
        self._particles = []
        for _ in range(90):
            self._particles.append({
                "x": random.uniform(0.05, 0.95),
                "y": random.uniform(-0.25, 0.0),
                "vx": random.uniform(-0.0025, 0.0025),
                "vy": random.uniform(0.004, 0.010),
                "size": random.randint(6, 13),
                "color": QColor(random.choice(self._COLORS)),
                "angle": random.uniform(0, 360),
                "spin": random.uniform(-5, 5),
            })

        self._opacity = 1.0
        self._tick = 0
        self._total_ticks = 120  # ~2 s at 60 fps

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update)
        self._timer.start(16)

    def _update(self):
        self._tick += 1
        for p in self._particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["angle"] += p["spin"]
        fade_start = self._total_ticks * 0.6
        if self._tick > fade_start:
            self._opacity = max(0.0, 1.0 - (self._tick - fade_start) / (self._total_ticks * 0.4))
        if self._tick >= self._total_ticks:
            self._timer.stop()
            self.setParent(None)
            self.deleteLater()
            return
        self.update()

    def paintEvent(self, event):
        from PyQt5.QtGui import QPainter
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        for p in self._particles:
            if p["y"] > 1.1:
                continue
            color = QColor(p["color"])
            color.setAlphaF(self._opacity * 0.92)
            painter.save()
            painter.translate(p["x"] * w, p["y"] * h)
            painter.rotate(p["angle"])
            sz = p["size"]
            painter.fillRect(-sz // 2, -sz // 4, sz, sz // 2, color)
            painter.restore()
        painter.end()


class ResultCardWidget(QFrame):
    def __init__(self, title, info_text="", parent=None):
        super().__init__(parent)
        self.setObjectName("resultCard")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.setMinimumHeight(120)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("resultCardTitle")
        if info_text:
            title_row = QHBoxLayout()
            title_row.setContentsMargins(0, 0, 0, 0)
            title_row.setSpacing(4)
            title_row.addWidget(self.title_label)
            title_row.addStretch()
            _info_btn = QPushButton("i")
            _info_btn.setObjectName("bucketInfoButton")
            _info_btn.setFlat(True)
            _info_btn.setFocusPolicy(Qt.NoFocus)
            _captured_text = info_text
            _captured_title = title
            _info_btn.clicked.connect(
                lambda: QMessageBox.information(self, _captured_title, _captured_text))
            title_row.addWidget(_info_btn)
            layout.addLayout(title_row)
        else:
            layout.addWidget(self.title_label)

        self.value_label = QLabel("Waiting for analysis")
        self.value_label.setObjectName("resultCardValue")
        self.value_label.setWordWrap(True)
        self.value_label.setMinimumHeight(44)
        self.value_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(self.value_label)

    def set_value(self, text):
        self.value_label.setText(text)


class ResultCockpitWidget(QFrame):
    configure_plot_requested = pyqtSignal()
    open_output_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("resultCockpit")
        self._animation_group = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("Result Cockpit")
        title.setObjectName("panelTitle")
        layout.addWidget(title)

        self.subtitle = QLabel("Run an analysis to populate the cockpit.")
        self.subtitle.setObjectName("panelDescription")
        self.subtitle.setWordWrap(True)
        layout.addWidget(self.subtitle)

        metrics_title = QLabel("Validity Checks")
        metrics_title.setObjectName("sectionLabel")
        layout.addWidget(metrics_title)

        self.metric_cards = {
            "metric_normality": ResultCardWidget("Normality"),
            "metric_variance": ResultCardWidget("Variance Homogeneity"),
        }
        self.metric_grid_widget = QWidget()
        self.metric_grid = QGridLayout(self.metric_grid_widget)
        self.metric_grid.setContentsMargins(0, 0, 0, 0)
        self.metric_grid.setHorizontalSpacing(12)
        self.metric_grid.setVerticalSpacing(12)
        self.metric_grid.setColumnStretch(0, 1)
        self.metric_grid.setColumnStretch(1, 1)
        cards_in_order = [
            self.metric_cards["metric_normality"],
            self.metric_cards["metric_variance"],
        ]
        for index, card in enumerate(cards_in_order):
            self.metric_grid.addWidget(card, index // 2, index % 2)
        layout.addWidget(self.metric_grid_widget)

        inference_title = QLabel("Inference")
        inference_title.setObjectName("sectionLabel")
        layout.addWidget(inference_title)

        self.inference_cards = {
            "inference_main_test": ResultCardWidget("Main Test + p-value"),
            "inference_effect_size": ResultCardWidget("Effect Size"),
        }
        self.inference_grid_widget = QWidget()
        self.inference_grid = QGridLayout(self.inference_grid_widget)
        self.inference_grid.setContentsMargins(0, 0, 0, 0)
        self.inference_grid.setHorizontalSpacing(12)
        self.inference_grid.setVerticalSpacing(12)
        self.inference_grid.setColumnStretch(0, 1)
        self.inference_grid.setColumnStretch(1, 1)
        inference_cards_in_order = [
            self.inference_cards["inference_main_test"],
            self.inference_cards["inference_effect_size"],
        ]
        for index, card in enumerate(inference_cards_in_order):
            self.inference_grid.addWidget(card, index // 2, index % 2)
        layout.addWidget(self.inference_grid_widget)

        context_title = QLabel("Context")
        context_title.setObjectName("sectionLabel")
        layout.addWidget(context_title)

        self.context_cards = {
            "context_design": ResultCardWidget(
                "Design",
                info_text=(
                    "Study design metadata shown here summarize the inferred model setup: "
                    "selected design family, mapped factors, and repeated-measures context."
                ),
            ),
            "context_sample_overview": ResultCardWidget("Sample Overview"),
            "context_analysis_scope": ResultCardWidget("Analysis Scope"),
        }
        self.context_grid_widget = QWidget()
        self.context_grid = QGridLayout(self.context_grid_widget)
        self.context_grid.setContentsMargins(0, 0, 0, 0)
        self.context_grid.setHorizontalSpacing(12)
        self.context_grid.setVerticalSpacing(12)
        self.context_grid.setColumnStretch(0, 1)
        self.context_grid.setColumnStretch(1, 1)
        context_cards_in_order = [
            self.context_cards["context_design"],
            self.context_cards["context_sample_overview"],
            self.context_cards["context_analysis_scope"],
        ]
        for index, card in enumerate(context_cards_in_order):
            self.context_grid.addWidget(card, index // 2, index % 2)
        layout.addWidget(self.context_grid_widget)

        buttons_row = QHBoxLayout()
        self.configure_plot_button = QPushButton("Configure Plot...")
        self.configure_plot_button.clicked.connect(self.configure_plot_requested.emit)
        self.configure_plot_button.setEnabled(False)
        buttons_row.addWidget(self.configure_plot_button)

        self.open_output_button = QPushButton("Open Output Folder")
        self.open_output_button.clicked.connect(self.open_output_requested.emit)
        self.open_output_button.setEnabled(False)
        buttons_row.addWidget(self.open_output_button)
        layout.addLayout(buttons_row)

        self.clear()


    def clear(self):
        self.subtitle.setText("Run an analysis to populate the cockpit.")
        metric_defaults = {
            "metric_normality": "Will be calculated after analysis.",
            "metric_variance": "Will be calculated after analysis.",
        }
        inference_defaults = {
            "inference_main_test": "Will be calculated after analysis.",
            "inference_effect_size": "Will be calculated after analysis.",
        }
        context_defaults = {
            "context_design": "The inferred design metadata will appear here.",
            "context_sample_overview": "Sample and group counts will appear here.",
            "context_analysis_scope": "Filters, covariates, and post-hoc scope will appear here.",
        }
        for key, text in metric_defaults.items():
            self.metric_cards[key].set_value(text)
        for key, text in inference_defaults.items():
            self.inference_cards[key].set_value(text)
        for key, text in context_defaults.items():
            self.context_cards[key].set_value(text)
        self.configure_plot_button.setEnabled(False)
        self.open_output_button.setEnabled(False)

    def set_summary(self, summary, enable_plot=False, enable_output=False):
        self.subtitle.setText(summary.get("subtitle", "Analysis complete."))
        for key, card in self.metric_cards.items():
            card.set_value(summary.get(key, "N/A"))
        self.inference_cards["inference_main_test"].set_value(
            summary.get("inference_main_test", summary.get("metric_main_test", "N/A"))
        )
        self.inference_cards["inference_effect_size"].set_value(
            summary.get("inference_effect_size", summary.get("metric_effect_size", "N/A"))
        )
        self.context_cards["context_design"].set_value(
            summary.get("context_design", summary.get("detected_test", "N/A"))
        )
        self.context_cards["context_sample_overview"].set_value(
            summary.get("context_sample_overview", summary.get("rationale", "N/A"))
        )
        self.context_cards["context_analysis_scope"].set_value(
            summary.get("context_analysis_scope", summary.get("posthoc", "N/A"))
        )

        self.configure_plot_button.setEnabled(enable_plot)
        self.open_output_button.setEnabled(enable_output)
        self._animate_cards()

    def _animate_cards(self):
        # Animation disabled: QSequentialAnimationGroup crashes on macOS 26 Tahoe
        # with dual Qt binaries (Homebrew + conda). Cards appear instantly instead.
        self._clear_card_effects()

    def _clear_card_effects(self):
        cards = []
        if hasattr(self, "metric_cards") and isinstance(self.metric_cards, dict):
            cards.extend(self.metric_cards.values())
        if hasattr(self, "inference_cards") and isinstance(self.inference_cards, dict):
            cards.extend(self.inference_cards.values())
        if hasattr(self, "context_cards") and isinstance(self.context_cards, dict):
            cards.extend(self.context_cards.values())
        for card in cards:
            effect = card.graphicsEffect()
            if effect is not None:
                # Restore the GlowFrame's own shadow instead of removing all effects
                if isinstance(card, GlowFrame):
                    card.setGraphicsEffect(card._shadow)
                else:
                    card.setGraphicsEffect(None)


_GROUP_COLORS = [
    "#2563eb",
    "#ea580c",
    "#7c3aed",
    "#16a34a",
    "#dc2626",
    "#ca8a04",
    "#0891b2",
    "#be185d",
]


class SheetSelectionDialog(QDialog):
    def __init__(self, df_raw, initial_sheet=None, available_sheets=None,
                 source_path=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Data Ranges")
        self.resize(1050, 680)

        self._df_raw = df_raw
        self._current_sheet = initial_sheet
        self._available_sheets = available_sheets
        self._source_path = source_path
        self._selection_map = {}   # {group_name: [{"rows": (r1,r2), "cols": (c1,c2)}]}
        self._group_colors = {}    # {group_name: color_hex}
        self._color_index = 0
        self._replicate_type = "biological"
        self._last_indexes = []    # cached table selection (survives focus loss on button click)

        self._build_ui()
        self._add_group_internal("Group_A")
        self._add_group_internal("Group_B")
        self._populate_table(df_raw)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(10)

        hint = QLabel(
            "Select cells in the sheet, then right-click (or use Assign) to assign them to "
            "a group.  Single-factor only — for multi-factor designs use a pre-formatted "
            "long-format file."
        )
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)
        main_layout.addWidget(hint)

        if self._available_sheets and len(self._available_sheets) > 1:
            sheet_row = QHBoxLayout()
            sheet_row.addWidget(QLabel("Sheet:"))
            self._sheet_combo = QComboBox()
            self._sheet_combo.addItems(self._available_sheets)
            if self._current_sheet and self._current_sheet in self._available_sheets:
                self._sheet_combo.setCurrentText(self._current_sheet)
            self._sheet_combo.currentTextChanged.connect(self._on_sheet_changed)
            self._apply_combo_arrow_style(self._sheet_combo)
            sheet_row.addWidget(self._sheet_combo, 1)
            sheet_row.addStretch()
            main_layout.addLayout(sheet_row)
        else:
            self._sheet_combo = None

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # Left: raw table
        self._table = QTableWidget()
        self._table.setObjectName("sheetTable")
        self._table.setSelectionMode(QTableWidget.ExtendedSelection)
        self._table.setSelectionBehavior(QTableWidget.SelectItems)
        self._table.setContextMenuPolicy(Qt.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._on_context_menu)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        splitter.addWidget(self._table)

        # Right: group panel — #objectName selector styles only this widget, not children
        right_panel = QWidget()
        right_panel.setObjectName("rangeDialogPanel")
        right_panel.setStyleSheet(
            "#rangeDialogPanel { background: #ffffff; border: 1px solid #c4daea; border-radius: 8px; }"
        )
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setSpacing(8)

        groups_title = QLabel("Groups")
        groups_title.setObjectName("columnCardTitle")
        right_layout.addWidget(groups_title)

        self._groups_container = QVBoxLayout()
        self._groups_container.setSpacing(4)
        right_layout.addLayout(self._groups_container)

        add_group_btn = QPushButton("+ Add Group")
        add_group_btn.setObjectName("secondaryButton")
        add_group_btn.clicked.connect(self._on_add_group)
        right_layout.addWidget(add_group_btn)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        right_layout.addWidget(sep)

        rep_title = QLabel("Replicate type")
        rep_title.setObjectName("columnCardTitle")
        right_layout.addWidget(rep_title)

        self._bio_radio = QRadioButton("Biological  (each cell = 1 N)")
        self._bio_radio.setChecked(True)
        self._bio_radio.toggled.connect(self._on_replicate_changed)
        right_layout.addWidget(self._bio_radio)

        self._tech_radio = QRadioButton("Technical  (mean per block)")
        right_layout.addWidget(self._tech_radio)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        right_layout.addWidget(sep2)

        self._status_label = QLabel("No cells selected")
        self._status_label.setObjectName("columnCardMeta")
        self._status_label.setWordWrap(True)
        right_layout.addWidget(self._status_label)

        assign_row = QHBoxLayout()
        self._assign_btn = QPushButton("Assign to:")
        self._assign_btn.setObjectName("secondaryButton")
        self._assign_btn.clicked.connect(self._on_assign_clicked)
        self._assign_combo = QComboBox()
        self._apply_combo_arrow_style(self._assign_combo)
        assign_row.addWidget(self._assign_btn)
        assign_row.addWidget(self._assign_combo, 1)
        right_layout.addLayout(assign_row)

        right_layout.addStretch()

        splitter.addWidget(right_panel)
        splitter.setSizes([700, 330])
        main_layout.addWidget(splitter, 1)

        self._nan_warning = QLabel("")
        self._nan_warning.setObjectName("hintLabel")
        self._nan_warning.setVisible(False)
        main_layout.addWidget(self._nan_warning)

        self._preview_label = QLabel("")
        self._preview_label.setObjectName("columnCardMeta")
        main_layout.addWidget(self._preview_label)

        btn_box = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Apply)
        btn_box.button(QDialogButtonBox.Apply).setObjectName("primaryButton")
        btn_box.button(QDialogButtonBox.Cancel).setObjectName("secondaryButton")
        btn_box.button(QDialogButtonBox.Apply).clicked.connect(self._on_apply)
        btn_box.rejected.connect(self.reject)
        main_layout.addWidget(btn_box)

    def _apply_combo_arrow_style(self, combo):
        try:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            arrow = os.path.join(base, "assets", "icons", "chevron-down.svg").replace("\\", "/")
            combo.setStyleSheet(
                f"QComboBox::drop-down {{ border-left: 1px solid #c4daea; width: 22px; "
                f"background: #eef8f6; border-top-right-radius: 7px; "
                f"border-bottom-right-radius: 7px; }}"
                f"QComboBox::down-arrow {{ image: url('{arrow}'); width: 10px; height: 6px; }}"
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Table population
    # ------------------------------------------------------------------

    def _populate_table(self, df_raw):
        self._df_raw = df_raw
        self._table.blockSignals(True)
        self._table.clear()
        self._table.setRowCount(len(df_raw))
        self._table.setColumnCount(len(df_raw.columns))

        flat = pd.to_numeric(df_raw.values.flatten(), errors="coerce")
        is_nonnumeric = np.isnan(flat).reshape(df_raw.shape)

        muted = QColor("#6b7c84")
        for r in range(len(df_raw)):
            for c in range(len(df_raw.columns)):
                raw_val = df_raw.iloc[r, c]
                text = "" if pd.isna(raw_val) else str(raw_val)
                item = QTableWidgetItem(text)
                if is_nonnumeric[r, c] and text:
                    item.setForeground(muted)
                    font = item.font()
                    font.setItalic(True)
                    item.setFont(font)
                self._table.setItem(r, c, item)

        self._table.blockSignals(False)
        self._table.resizeColumnsToContents()
        for c in range(self._table.columnCount()):
            if self._table.columnWidth(c) > 150:
                self._table.setColumnWidth(c, 150)

        self._recolor_table()

    def _recolor_table(self):
        transparent = QColor(0, 0, 0, 0)
        for r in range(self._table.rowCount()):
            for c in range(self._table.columnCount()):
                item = self._table.item(r, c)
                if item:
                    item.setBackground(transparent)

        for group_name, ranges in self._selection_map.items():
            hex_color = self._group_colors.get(group_name, "#2563eb")
            color = QColor(hex_color)
            color.setAlpha(55)
            for rng in ranges:
                r1, r2 = rng["rows"]
                c1, c2 = rng["cols"]
                for r in range(r1, min(r2 + 1, self._table.rowCount())):
                    for c in range(c1, min(c2 + 1, self._table.columnCount())):
                        item = self._table.item(r, c)
                        if item:
                            item.setBackground(color)

    # ------------------------------------------------------------------
    # Group management
    # ------------------------------------------------------------------

    def _add_group_internal(self, name):
        if name in self._group_colors:
            return
        color = _GROUP_COLORS[self._color_index % len(_GROUP_COLORS)]
        self._color_index += 1
        self._group_colors[name] = color
        self._selection_map[name] = []
        self._assign_combo.addItem(name)
        self._rebuild_group_list()

    def _rebuild_group_list(self):
        while self._groups_container.count():
            item = self._groups_container.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)  # immediate detach+hide; deleteLater alone is async
                w.deleteLater()

        for group_name in list(self._group_colors.keys()):
            row_w = QWidget()
            row_layout = QHBoxLayout(row_w)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)

            swatch = QFrame()
            swatch.setObjectName("groupColorSwatch")
            swatch.setFixedSize(12, 12)
            swatch.setStyleSheet(
                f"background: {self._group_colors[group_name]}; border-radius: 3px;"
            )
            row_layout.addWidget(swatch)

            name_lbl = QLabel(group_name)
            name_lbl.setObjectName("columnCardMeta")
            row_layout.addWidget(name_lbl, 1)

            n_ranges = len(self._selection_map.get(group_name, []))
            count_lbl = QLabel(f"({n_ranges})")
            count_lbl.setObjectName("columnCardMeta")
            row_layout.addWidget(count_lbl)

            rm_btn = QToolButton()
            rm_btn.setText("×")
            rm_btn.setObjectName("hintDismiss")
            rm_btn.clicked.connect(lambda _checked, g=group_name: self._on_remove_group(g))
            row_layout.addWidget(rm_btn)

            self._groups_container.addWidget(row_w)

    def _on_add_group(self):
        name, ok = QInputDialog.getText(self, "Add Group", "Group name:")
        if ok and name.strip():
            name = name.strip()
            if name in self._group_colors:
                QMessageBox.warning(self, "Duplicate", f"Group '{name}' already exists.")
                return
            self._add_group_internal(name)

    def _on_remove_group(self, group_name):
        del self._group_colors[group_name]
        del self._selection_map[group_name]
        idx = self._assign_combo.findText(group_name)
        if idx >= 0:
            self._assign_combo.removeItem(idx)
        self._rebuild_group_list()
        self._recolor_table()
        self._update_preview()

    # ------------------------------------------------------------------
    # Selection + assignment
    # ------------------------------------------------------------------

    def _on_context_menu(self, pos):
        indexes = self._table.selectedIndexes()
        if not indexes:
            return
        menu = QMenu(self)
        for group_name in self._group_colors:
            action = menu.addAction(f"Assign to: {group_name}")
            action.triggered.connect(
                lambda _checked, g=group_name: self._assign_selection(g)
            )
        menu.addSeparator()
        clear_act = menu.addAction("Clear assignment")
        clear_act.triggered.connect(self._clear_selection_assignment)
        menu.exec_(self._table.viewport().mapToGlobal(pos))

    def _on_assign_clicked(self):
        group_name = self._assign_combo.currentText()
        if group_name:
            self._assign_selection(group_name)

    def _assign_selection(self, group_name):
        indexes = self._last_indexes or self._table.selectedIndexes()
        if not indexes:
            return
        new_ranges = _selected_indexes_to_ranges(indexes)
        self._selection_map[group_name].extend(new_ranges)
        self._rebuild_group_list()
        self._recolor_table()
        self._update_preview()

    def _clear_selection_assignment(self):
        indexes = self._last_indexes or self._table.selectedIndexes()
        if not indexes:
            return
        sel_rows = {idx.row() for idx in indexes}
        sel_cols = {idx.column() for idx in indexes}
        for group_name in self._selection_map:
            kept = []
            for rng in self._selection_map[group_name]:
                r1, r2 = rng["rows"]
                c1, c2 = rng["cols"]
                rng_rows = set(range(r1, r2 + 1))
                rng_cols = set(range(c1, c2 + 1))
                if not (rng_rows & sel_rows and rng_cols & sel_cols):
                    kept.append(rng)
            self._selection_map[group_name] = kept
        self._rebuild_group_list()
        self._recolor_table()
        self._update_preview()

    def _on_selection_changed(self):
        indexes = self._table.selectedIndexes()
        self._last_indexes = indexes   # cache so Assign button works after focus loss
        if not indexes:
            self._status_label.setText("No cells selected")
            return
        rows = [idx.row() for idx in indexes]
        cols = [idx.column() for idx in indexes]
        r1, r2 = min(rows), max(rows)
        c1, c2 = min(cols), max(cols)
        n = len(indexes)
        self._status_label.setText(
            f"{_to_display_coords(r1, c1)} – {_to_display_coords(r2, c2)}\n{n} cells"
        )

    # ------------------------------------------------------------------
    # Sheet switching
    # ------------------------------------------------------------------

    def _on_sheet_changed(self, sheet_name):
        has_assignments = any(len(v) > 0 for v in self._selection_map.values())
        if has_assignments:
            reply = QMessageBox.question(
                self, "Switch Sheet",
                "Switching sheets will clear current selections. Continue?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                if self._sheet_combo:
                    self._sheet_combo.blockSignals(True)
                    self._sheet_combo.setCurrentText(self._current_sheet or "")
                    self._sheet_combo.blockSignals(False)
                return
        self._current_sheet = sheet_name
        for g in self._selection_map:
            self._selection_map[g] = []
        if self._source_path:
            df_raw = pd.read_excel(
                self._source_path, sheet_name=sheet_name, header=None, dtype=str
            )
            self._populate_table(df_raw)
        self._rebuild_group_list()
        self._update_preview()

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def _on_replicate_changed(self, _checked):
        self._replicate_type = "biological" if self._bio_radio.isChecked() else "technical"

    def _update_preview(self):
        parts = []
        total = 0
        for group_name, ranges in self._selection_map.items():
            n = sum(
                (r["rows"][1] - r["rows"][0] + 1) * (r["cols"][1] - r["cols"][0] + 1)
                for r in ranges
            )
            total += n
            parts.append(f"{group_name}: ~{n}")
        if total:
            self._preview_label.setText("→ ~" + str(total) + " cells | " + ", ".join(parts))
        else:
            self._preview_label.setText("")

    def _on_apply(self):
        if not any(len(v) > 0 for v in self._selection_map.values()):
            QMessageBox.warning(
                self, "No Selection", "Assign at least one group before applying."
            )
            return
        self.accept()

    def get_result(self):
        return self._selection_map, self._replicate_type, self._current_sheet
