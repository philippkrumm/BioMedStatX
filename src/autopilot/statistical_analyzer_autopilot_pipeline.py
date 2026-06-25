# pyright: reportAttributeAccessIssue=false
# pyright: reportOptionalMemberAccess=false
import os
import sys

import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QColor, QDesktopServices
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from visualization.datavisualizer import DataVisualizer
from export.export_dispatcher import ExportDispatcher
from ui.dialogs.plot_aesthetics_dialog import PlotAestheticsDialog
from ui.dialogs.statistical_analyzer_dialogs import ExploratoryMatrixDialog, GroupSelectionDialog
from autopilot.statistical_analyzer_autopilot_ui import (
    ConfettiOverlay,
    DecisionTreePanel,
    DraggableColumnCard,
    FilterBucketWidget,
    MappingBucketWidget,
    PipelineTrackerWidget,
    ResultCockpitWidget,
    SheetSelectionDialog,
    _detect_wide_format,
    _infer_column_kind,
    _looks_like_subject,
    _pivot_wide_to_long,
    _safe_file_slug,
    _sorted_unique,
    extract_bivariate_from_coordinates,
    extract_from_coordinates,
    extract_paired_from_coordinates,
)
from analysis.statisticaltester import StatisticalTester
from analysis.stats_functions import AnalysisManager

import logging
logger = logging.getLogger(__name__)

DEFAULT_COLORS = ["#0f766e", "#1f7a5a", "#b7791f", "#9f3a38", "#1d4ed8", "#7c3aed"]
DEFAULT_HATCHES = ["/", "\\", "|", "-", "+", "x", "o", ".", "*", ""]


def _resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller."""
    try:
        base_path = sys._MEIPASS
    except Exception:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.dirname(os.path.dirname(script_dir))
    return os.path.join(base_path, relative_path)


def _apply_elevation(widget, radius=18, x_offset=0, y_offset=4, opacity=0.18):
    """Apply a drop shadow to give a widget visual elevation. QSS cannot do this."""
    shadow = QGraphicsDropShadowEffect(widget)
    shadow.setBlurRadius(radius)
    shadow.setOffset(x_offset, y_offset)
    shadow.setColor(QColor(0, 0, 0, int(255 * opacity)))
    widget.setGraphicsEffect(shadow)

def _load_auto_pilot_stylesheet():
    stylesheet_paths = [
        _resource_path("assets/BioMedStatX_2_0.qss"),
        _resource_path("assets/StyleSheet.qss"),
    ]
    for path in stylesheet_paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                qss = handle.read()
                arrow_path = _resource_path("assets/icons/chevron-down.png").replace("\\", "/")
                return qss.replace("{CHEVRON_DOWN_PATH}", arrow_path)
    return ""


def _ap_init_ui(self):
    central_scroll = QScrollArea()
    central_scroll.setWidgetResizable(True)
    central_scroll.setFrameShape(QFrame.NoFrame)

    central_widget = QWidget()
    central_widget.setObjectName("autoPilotRoot")
    root_layout = QVBoxLayout(central_widget)
    root_layout.setContentsMargins(24, 18, 24, 18)
    root_layout.setSpacing(16)

    self.pipeline_tracker = PipelineTrackerWidget()
    root_layout.addWidget(self.pipeline_tracker)

    headline_row = QHBoxLayout()
    headline_row.setSpacing(12)
    title_block = QVBoxLayout()
    title = QLabel("BioMedStatX 2.0")
    title.setObjectName("heroTitle")
    title_block.addWidget(title)
    subtitle = QLabel("Auto-pilot statistical analysis with guided mapping and transparent decisions.")
    subtitle.setObjectName("heroSubtitle")
    subtitle.setWordWrap(True)
    title_block.addWidget(subtitle)
    headline_row.addLayout(title_block, 1)

    self.analysis_status_badge = QLabel("Waiting for a dataset")
    self.analysis_status_badge.setObjectName("analysisStatusBadge")
    headline_row.addWidget(self.analysis_status_badge, 0, Qt.AlignTop)
    root_layout.addLayout(headline_row)

    splitter = QSplitter(Qt.Horizontal)
    splitter.setChildrenCollapsible(False)
    root_layout.addWidget(splitter, 1)

    # Left panel
    left_panel = QFrame()
    left_panel.setObjectName("dashboardPanel")
    left_layout = QVBoxLayout(left_panel)
    left_layout.setContentsMargins(18, 18, 18, 18)
    left_layout.setSpacing(14)

    left_title = QLabel("Data Source")
    left_title.setObjectName("panelTitle")
    left_layout.addWidget(left_title)

    file_row = QHBoxLayout()
    self.auto_file_label = QLabel("No file selected")
    self.auto_file_label.setObjectName("filePathLabel")
    file_row.addWidget(self.auto_file_label, 1)
    browse_button = QPushButton("Load Data File")
    browse_button.clicked.connect(self.browse_file)
    file_row.addWidget(browse_button)
    left_layout.addLayout(file_row)

    mode_row = QHBoxLayout()
    self.single_mode_button = QRadioButton("Single Analysis")
    self.multi_mode_button = QRadioButton("Multi-Dataset Analysis")
    self.single_mode_button.setChecked(True)
    self.single_mode_button.toggled.connect(self.update_mode_constraints)
    self.multi_mode_button.toggled.connect(self.update_mode_constraints)
    mode_row.addWidget(self.single_mode_button)
    mode_row.addWidget(self.multi_mode_button)
    left_layout.addLayout(mode_row)

    self.auto_mode_hint = QLabel("Single mode expects exactly one measurement column (for example one gene).")
    self.auto_mode_hint.setObjectName("panelDescription")
    self.auto_mode_hint.setWordWrap(True)
    left_layout.addWidget(self.auto_mode_hint)

    sheet_label = QLabel("Worksheet")
    sheet_label.setObjectName("sectionLabel")
    left_layout.addWidget(sheet_label)
    self.auto_sheet_combo = QComboBox()
    self.auto_sheet_combo.currentIndexChanged.connect(self.load_sheet)
    left_layout.addWidget(self.auto_sheet_combo)
    self.sheet_combo = self.auto_sheet_combo

    self.range_select_btn = QPushButton("Select Data Ranges...")
    self.range_select_btn.setObjectName("secondaryButton")
    self.range_select_btn.setToolTip(
        "Handles one grouping at a time. For two-factor designs, covariates, or mixed models, "
        "arrange your data as a table with one row per measurement and load that instead."
    )
    self.range_select_btn.setVisible(False)
    self.range_select_btn.clicked.connect(self._ap_open_range_selector)
    left_layout.addWidget(self.range_select_btn)

    self._hint_dismissed = False

    self._range_groups_label = QLabel("")
    self._range_groups_label.setObjectName("hintLabel")
    self._range_groups_label.setWordWrap(True)
    self._range_groups_label.setVisible(False)
    left_layout.addWidget(self._range_groups_label)

    preview_label = QLabel("Table Preview")
    preview_label.setObjectName("sectionLabel")
    left_layout.addWidget(preview_label)
    self.preview_hint = QLabel("Table preview appears after loading a file.")
    self.preview_hint.setObjectName("lblEmptyState")
    self.preview_hint.setWordWrap(True)
    left_layout.addWidget(self.preview_hint)
    self.preview_table = QTableWidget(0, 0)
    self.preview_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    self.preview_table.setSelectionMode(QAbstractItemView.NoSelection)
    self.preview_table.setAlternatingRowColors(True)
    self.preview_table.setMinimumHeight(160)
    self.preview_table.setVisible(False)
    left_layout.addWidget(self.preview_table, 2)

    cards_label = QLabel("Columns")
    cards_label.setObjectName("sectionLabel")
    left_layout.addWidget(cards_label)
    cards_scroll = QScrollArea()
    cards_scroll.setWidgetResizable(True)
    cards_scroll.setObjectName("headerCardScroll")
    cards_scroll.setMinimumHeight(120)
    self.header_cards_widget = QWidget()
    self.header_cards_widget.setObjectName("headerCardsInner")
    self.header_cards_layout = QVBoxLayout(self.header_cards_widget)
    self.header_cards_layout.setContentsMargins(0, 0, 0, 0)
    self.header_cards_layout.setSpacing(10)
    self.header_cards_layout.addStretch()
    cards_scroll.setWidget(self.header_cards_widget)
    left_layout.addWidget(cards_scroll, 1)

    _apply_elevation(left_panel, radius=14, y_offset=3, opacity=0.13)
    splitter.addWidget(left_panel)

    # Center panel
    center_panel = QFrame()
    center_panel.setObjectName("dashboardPanel")
    center_layout = QVBoxLayout(center_panel)
    center_layout.setContentsMargins(18, 18, 18, 18)
    center_layout.setSpacing(14)

    center_title = QLabel("Smart Mapping")
    center_title.setObjectName("panelTitle")
    center_layout.addWidget(center_title)

    center_description = QLabel("Drag header cards into the buckets. The auto-pilot will infer the test design from this mapping.")
    center_description.setObjectName("panelDescription")
    center_description.setWordWrap(True)
    center_layout.addWidget(center_description)

    self.dv_bucket = MappingBucketWidget(
        "Dependent Variable",
        "Drop numeric measurement columns here. Single mode uses one column; multi mode uses multiple columns (for example several genes).",
        accepted_kinds={"numeric"},
        allow_multiple=False,
        info_text=(
            "The dependent variable is the measurement outcome to be analyzed "
            "(e.g. cell count, miRNA expression, protein concentration). "
            "Must be numeric.\n\n"
            "Single mode: one column.\n"
            "Multi mode: multiple columns for simultaneous analysis (e.g. several genes)."
        ),
    )
    self.factor1_bucket = MappingBucketWidget(
        "Factor 1",
        "Drop the primary grouping factor here.",
        accepted_kinds={"numeric", "categorical", "datetime"},
        help_recipe_id="one_way_anova",
        info_text=(
            "The primary predictor — determines the statistical test to be used.\n\n"
            "Categorical (e.g. treatment group, sex) → t-Test or ANOVA.\n"
            "Continuous (e.g. pump duration, BMI, age) → Correlation or Regression.\n\n"
            "Only one variable allowed. For multiple predictors use the Covariates bucket."
        ),
    )
    self.factor2_bucket = MappingBucketWidget(
        "Factor 2",
        "Optional: drop a second factor here for Two-Way or Mixed ANOVA.",
        accepted_kinds={"numeric", "categorical", "datetime"},
        help_recipe_id="two_way_anova",
        info_text=(
            "Second factor for Two-Way ANOVA or Mixed ANOVA.\n\n"
            "Only needed when interaction effects are of interest "
            "(e.g. treatment × time, genotype × condition).\n\n"
            "Without Subject ID: Two-Way ANOVA (between-subjects).\n"
            "With Subject ID: Mixed ANOVA (between + within)."
        ),
    )
    self.subject_bucket = MappingBucketWidget(
        "Subject ID",
        "Optional: drop a subject identifier here for paired / repeated-measures designs.",
        accepted_kinds={"numeric", "categorical", "datetime"},
        help_recipe_id="repeated_measures_anova",
        info_text=(
            "Patient or subject identifier for paired/repeated-measures designs.\n\n"
            "Required for:\n"
            "  • Paired t-Test (two time points per subject)\n"
            "  • Repeated-Measures ANOVA (≥3 time points)\n"
            "  • Linear Mixed Models (LMM)\n\n"
            "If the ID is missing for longitudinal data, an unpaired test will be used — "
            "this increases error and reduces statistical power."
        ),
    )
    self.covariates_bucket = MappingBucketWidget(
        "Covariates (optional)",
        "Drop continuous confounders here (e.g., Age, BMI, Baseline).",
        accepted_kinds={"numeric"},
        allow_multiple=True,
        help_recipe_id="ancova",
        info_text=(
            "Continuous confounders to be statistically controlled for "
            "(e.g. Age, BMI, Baseline value, Comorbidity score).\n\n"
            "Categorical factor in Factor 1 → ANCOVA.\n"
            "Continuous factor in Factor 1 → Multiple Regression (OLS).\n\n"
            "Multiple variables can be added simultaneously."
        ),
    )
    self.filter_bucket = FilterBucketWidget(get_df=lambda: self.df)

    for bucket in (self.dv_bucket, self.factor1_bucket, self.factor2_bucket,
                   self.subject_bucket, self.covariates_bucket, self.filter_bucket):
        bucket.changed.connect(self.on_mapping_changed)
        center_layout.addWidget(bucket)

    self.analysis_group_button = QPushButton("Select Groups For Analysis")
    self.analysis_group_button.setObjectName("secondaryButton")
    self.analysis_group_button.clicked.connect(self.open_analysis_group_selector)
    self.analysis_group_button.setEnabled(False)
    center_layout.addWidget(self.analysis_group_button)

    self.analysis_group_label = QLabel("No group subset selected. Default: all groups in Factor 1.")
    self.analysis_group_label.setObjectName("panelDescription")
    self.analysis_group_label.setWordWrap(True)
    center_layout.addWidget(self.analysis_group_label)

    self.mapping_feedback_label = QLabel("Load a file to activate the mapping workflow.")
    self.mapping_feedback_label.setObjectName("panelDescription")
    self.mapping_feedback_label.setWordWrap(True)
    center_layout.addWidget(self.mapping_feedback_label)

    # --- Correlation / Regression widget (hidden until continuous-factor design detected) ---
    self.corr_transform_widget = QWidget()
    corr_tr_layout = QVBoxLayout(self.corr_transform_widget)
    corr_tr_layout.setContentsMargins(0, 4, 0, 0)
    corr_tr_layout.setSpacing(6)

    # Mode toggle: Correlation (default) vs. Simple Linear Regression
    self.corr_regression_toggle = QCheckBox("Als Lineare Regression analysieren (Y = a + bX)")
    self.corr_regression_toggle.setObjectName("panelDescription")
    self.corr_regression_toggle.setChecked(False)
    self.corr_regression_toggle.stateChanged.connect(self.on_mapping_changed)
    corr_tr_layout.addWidget(self.corr_regression_toggle)

    # Transform controls — only meaningful for regression (theoretical choice before fit).
    # For correlation the transform is chosen reactively after Shapiro-Wilk, so no UI needed.
    self.corr_reg_transform_container = QWidget()
    reg_tr_inner = QVBoxLayout(self.corr_reg_transform_container)
    reg_tr_inner.setContentsMargins(0, 4, 0, 0)
    reg_tr_inner.setSpacing(4)

    corr_tr_title = QLabel("Variable Transformations (optional)")
    corr_tr_title.setObjectName("panelDescription")
    reg_tr_inner.addWidget(corr_tr_title)

    _TRANSFORMS = ["none", "log10", "log10(x+1)", "sqrt", "boxcox"]

    x_row = QHBoxLayout()
    x_row.addWidget(QLabel("X transform:"))
    self.corr_x_transform_combo = QComboBox()
    self.corr_x_transform_combo.addItems(_TRANSFORMS)
    x_row.addWidget(self.corr_x_transform_combo)
    reg_tr_inner.addLayout(x_row)

    y_row = QHBoxLayout()
    y_row.addWidget(QLabel("Y transform:"))
    self.corr_y_transform_combo = QComboBox()
    self.corr_y_transform_combo.addItems(_TRANSFORMS)
    y_row.addWidget(self.corr_y_transform_combo)
    reg_tr_inner.addLayout(y_row)

    # Warning: shown only when regression mode is active AND a transform is selected.
    self.corr_transform_warning = QLabel(
        "Warning: Transformed variables require exponentiation or elasticity formulas to interpret β correctly. See manual for exact equations."
    )
    self.corr_transform_warning.setObjectName("warningLabel")
    self.corr_transform_warning.setWordWrap(True)
    self.corr_transform_warning.setVisible(False)
    reg_tr_inner.addWidget(self.corr_transform_warning)

    self.corr_reg_transform_container.setVisible(False)
    corr_tr_layout.addWidget(self.corr_reg_transform_container)

    def _update_transform_warning():
        if not hasattr(self, 'corr_regression_toggle') or not hasattr(self, 'corr_x_transform_combo'):
            return
        is_regression = self.corr_regression_toggle.isChecked()
        # Show transform controls only in regression mode
        self.corr_reg_transform_container.setVisible(is_regression)
        has_transform = (
            self.corr_x_transform_combo.currentText() not in ('none', '')
            or self.corr_y_transform_combo.currentText() not in ('none', '')
        )
        self.corr_transform_warning.setVisible(is_regression and has_transform)

    self.corr_x_transform_combo.currentIndexChanged.connect(_update_transform_warning)
    self.corr_y_transform_combo.currentIndexChanged.connect(_update_transform_warning)
    self.corr_regression_toggle.stateChanged.connect(_update_transform_warning)

    self.corr_transform_widget.setVisible(False)
    center_layout.addWidget(self.corr_transform_widget)

    self.start_analysis_button = QPushButton("Start Auto Analysis")
    self.start_analysis_button.setObjectName("primaryButton")
    self.start_analysis_button.clicked.connect(self.determine_and_run_test)
    self.start_analysis_button.setEnabled(False)
    center_layout.addWidget(self.start_analysis_button)

    center_layout.addStretch()

    _apply_elevation(center_panel, radius=14, y_offset=3, opacity=0.13)
    splitter.addWidget(center_panel)

    # Right panel
    right_panel = QFrame()
    right_panel.setObjectName("dashboardPanel")
    right_layout = QVBoxLayout(right_panel)
    right_layout.setContentsMargins(18, 18, 18, 18)
    right_layout.setSpacing(14)

    self.decision_tree_panel = DecisionTreePanel()
    right_layout.addWidget(self.decision_tree_panel, 1)

    self.result_cockpit = ResultCockpitWidget()
    self.result_cockpit.configure_plot_requested.connect(self.configure_plot_from_result)
    self.result_cockpit.open_output_requested.connect(self.open_current_output_folder)
    right_layout.addWidget(self.result_cockpit, 1)
    _apply_elevation(self.decision_tree_panel)
    _apply_elevation(self.result_cockpit)

    splitter.addWidget(right_panel)
    splitter.setSizes([450, 430, 520])

    central_scroll.setWidget(central_widget)
    self.setCentralWidget(central_scroll)

    self._refresh_preview_table()
    self._rebuild_column_cards()

    self.file_path_label = self.auto_file_label
    self.current_analysis_context = None
    self.current_analysis_result = None
    self.current_output_dir = None
    self.current_multi_results = {}
    self.current_rendered_dataset = None
    self.analysis_selected_groups = []


def _ap_refresh_preview_table(self):
    if self.df is None:
        self.preview_table.clear()
        self.preview_table.setRowCount(0)
        self.preview_table.setColumnCount(0)
        self.preview_table.setVisible(False)
        self.preview_hint.setVisible(True)
        return

    self.preview_hint.setVisible(False)
    self.preview_table.setVisible(True)
    preview_df = self.df.head(12).copy()
    self.preview_table.clear()
    self.preview_table.setColumnCount(len(preview_df.columns))
    self.preview_table.setHorizontalHeaderLabels([str(column) for column in preview_df.columns])
    self.preview_table.setRowCount(len(preview_df.index))
    for row_index in range(len(preview_df.index)):
        for column_index, column_name in enumerate(preview_df.columns):
            value = preview_df.iloc[row_index, column_index]
            item = QTableWidgetItem("" if pd.isna(value) else str(value))
            self.preview_table.setItem(row_index, column_index, item)
    self.preview_table.resizeColumnsToContents()


def _ap_rebuild_column_cards(self):
    while self.header_cards_layout.count():
        item = self.header_cards_layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()

    self._column_cards = {}

    if self.df is None:
        empty = QLabel("Load a file to see its columns here.")
        empty.setObjectName("lblEmptyState")
        empty.setWordWrap(True)
        empty.setAlignment(Qt.AlignCenter)
        self.header_cards_layout.addWidget(empty)
        self.header_cards_layout.addStretch()
        return

    for column_name in self.df.columns:
        series = self.df[column_name]
        column_kind = _infer_column_kind(series)
        # For categorical/text columns: show unique distinct values instead of
        # first 3 rows (which are often duplicates like "WT, WT, WT").
        if column_kind != "numeric":
            uniques = series.dropna().astype(str).unique().tolist()
            preview_values = uniques[:5]
            suffix = "" if len(uniques) <= 5 else f"  (+{len(uniques)-5} more)"
            preview_text = "Levels: " + (", ".join(preview_values) if preview_values else "—") + suffix
        else:
            preview_values = [str(value) for value in series.dropna().head(3).tolist()]
            preview_text = "Preview: " + (", ".join(preview_values) if preview_values else "No preview values")
        card = DraggableColumnCard(column_name, column_kind, preview_text)
        self._column_cards[column_name] = card
        self.header_cards_layout.addWidget(card)
    self.header_cards_layout.addStretch()


def _ap_apply_mapping_heuristics(self):
    for bucket in (self.dv_bucket, self.factor1_bucket, self.factor2_bucket, self.subject_bucket):
        bucket.clear_assignments()

    if self.df is None:
        return

    # Bivariate range extraction: X = predictor → Factor 1, Y = outcome → DV
    if getattr(self, "_range_design_mode", None) == "bivariate":
        if "X" in self.df.columns and "Y" in self.df.columns:
            self.dv_bucket.assign_column("Y", _infer_column_kind(self.df["Y"]))
            self.factor1_bucket.assign_column("X", _infer_column_kind(self.df["X"]))
            return

    columns = list(self.df.columns)
    numeric_all = [column for column in columns if pd.api.types.is_numeric_dtype(self.df[column])]

    factor_candidates = []
    for column in columns:
        if column in numeric_all:
            continue
        unique_count = self.df[column].dropna().nunique()
        if 1 < unique_count < max(len(self.df) * 0.75, 5):
            factor_candidates.append(column)

    if not factor_candidates:
        factor_candidates = [
            column for column in columns
            if column not in numeric_all
        ]

    subject_candidates = [
        column for column in columns
        if _looks_like_subject(column, self.df[column])
    ]
    subject_column = None
    for candidate in subject_candidates:
        # Keep the only available factor-like column for Factor 1 if possible.
        if candidate in factor_candidates and len(factor_candidates) <= 1:
            continue
        subject_column = candidate
        break
    if subject_column is None and subject_candidates:
        subject_column = subject_candidates[0]

    if subject_column in factor_candidates:
        factor_candidates = [column for column in factor_candidates if column != subject_column]

    numeric_columns = [column for column in numeric_all if column != subject_column]

    if numeric_columns:
        if self.multi_mode_button.isChecked() and len(numeric_columns) > 1:
            for column in numeric_columns:
                self.dv_bucket.assign_column(column, _infer_column_kind(self.df[column]))
        else:
            self.dv_bucket.assign_column(numeric_columns[0], _infer_column_kind(self.df[numeric_columns[0]]))

    if factor_candidates:
        self.factor1_bucket.assign_column(factor_candidates[0], _infer_column_kind(self.df[factor_candidates[0]]))
    if len(factor_candidates) > 1:
        self.factor2_bucket.assign_column(factor_candidates[1], _infer_column_kind(self.df[factor_candidates[1]]))

    if subject_column and factor_candidates:
        # Only assign Subject ID when it actually creates a repeated-measures structure:
        # each subject must appear under more than one level of Factor 1.
        factor1_col = factor_candidates[0]
        try:
            subject_span = self.df.groupby(subject_column)[factor1_col].nunique(dropna=True)
            if subject_span.max() > 1:
                self.subject_bucket.assign_column(subject_column, _infer_column_kind(self.df[subject_column]))
        except Exception:
            pass  # silently skip if validation fails


def _ap_update_mode_constraints(self):
    allow_multiple_dv = self.multi_mode_button.isChecked()
    self.dv_bucket.set_allow_multiple(allow_multiple_dv)
    if allow_multiple_dv:
        self.auto_mode_hint.setText("Multi mode analyzes multiple measurement columns (for example several genes) with the same factor mapping. It remains restricted to ANOVA-capable designs.")
    else:
        self.auto_mode_hint.setText("Single mode expects exactly one measurement column (for example one gene).")
    # Re-run heuristics so the DV bucket is populated correctly for the new mode
    if hasattr(self, 'df') and self.df is not None:
        self._apply_mapping_heuristics()
    else:
        self.on_mapping_changed()


def _ap_get_available_analysis_groups(self):
    if self.df is None:
        return []
    factor_columns = self.factor1_bucket.get_assigned_columns()
    if not factor_columns:
        return []
    factor_col = factor_columns[0]
    if factor_col not in self.df.columns:
        return []

    working_df = self.df.copy()
    active_filter = getattr(self, 'filter_bucket', None)
    filter_spec = active_filter.get_filter() if active_filter else None
    if filter_spec:
        filter_col, filter_val = filter_spec
        if filter_col in working_df.columns:
            working_df = working_df[working_df[filter_col] == filter_val]
    return _sorted_unique(working_df[factor_col].dropna().tolist())


def _ap_update_analysis_group_selection_ui(self):
    available_groups = self._ap_get_available_analysis_groups()
    selected_set = set(self.analysis_selected_groups or [])
    self.analysis_selected_groups = [group for group in available_groups if group in selected_set]

    has_factor1 = bool(self.factor1_bucket.get_assigned_columns())

    # Continuous Factor 1 → correlation/regression path, group selection not applicable
    if has_factor1 and self._is_continuous_factor1_for_help():
        self.analysis_group_button.setEnabled(False)
        self.analysis_group_label.setText(
            "Factor 1 is continuous \u2192 Correlation / Regression (no group selection needed)."
        )
        return

    self.analysis_group_button.setEnabled(has_factor1 and len(available_groups) >= 2)

    if not has_factor1:
        self.analysis_group_label.setText("Assign Factor 1 to enable explicit group selection for analysis.")
        return
    if not available_groups:
        self.analysis_group_label.setText("No groups are currently available in Factor 1.")
        return
    if self.analysis_selected_groups:
        selected_text = ", ".join(map(str, self.analysis_selected_groups))
        self.analysis_group_label.setText(
            f"Selected groups for analysis ({len(self.analysis_selected_groups)}/{len(available_groups)}): {selected_text}"
        )
    else:
        self.analysis_group_label.setText(
            f"No group subset selected. Default: all {len(available_groups)} groups in Factor 1."
        )


def _ap_open_analysis_group_selector(self):
    available_groups = self._ap_get_available_analysis_groups()
    if len(available_groups) < 2:
        QMessageBox.information(
            self,
            "Group Selection",
            "At least two groups in Factor 1 are required to define an analysis subset."
        )
        return

    dialog = GroupSelectionDialog(
        available_groups,
        self,
        window_title="Select Groups For Analysis",
        description="Select the Factor 1 groups to include in the analysis run:"
    )
    preselected = set(self.analysis_selected_groups or available_groups)
    for group, checkbox in dialog.group_checks.items():
        checkbox.setChecked(group in preselected)

    if dialog.exec_() != QDialog.Accepted:
        return

    selected_groups = dialog.get_selected_groups()
    if len(selected_groups) < 2:
        QMessageBox.warning(
            self,
            "Group Selection",
            "Please select at least two groups for the analysis."
        )
        return

    self.analysis_selected_groups = selected_groups
    self._ap_update_analysis_group_selection_ui()
    self.on_mapping_changed()


def _ap_set_workflow_state(self, stage, message, running=False):
    self.pipeline_tracker.set_stage(stage, running=running)
    self.analysis_status_badge.setText(message)


def _ap_is_binary_outcome_for_help(self):
    dv_columns = self.dv_bucket.get_assigned_columns()
    if len(dv_columns) != 1 or self.df is None:
        return False

    dv_col = dv_columns[0]
    if dv_col not in self.df.columns:
        return False

    series = self.df[dv_col].dropna()
    if series.empty:
        return False

    unique_values = series.unique()
    if len(unique_values) != 2:
        return False

    is_01 = set(unique_values) <= {0, 1, 0.0, 1.0}
    is_str = all(isinstance(value, str) for value in unique_values)
    return bool(is_01 or is_str)


def _ap_is_continuous_factor1_for_help(self):
    factor_columns = self.factor1_bucket.get_assigned_columns()
    if not factor_columns or self.df is None:
        return False

    factor_col = factor_columns[0]
    if factor_col not in self.df.columns:
        return False

    try:
        from analysis.correlation_models import _is_continuous as _corr_is_continuous
        return bool(_corr_is_continuous(self.df, factor_col))
    except Exception:
        factor_kinds = self.factor1_bucket.get_assigned_kinds()
        if factor_kinds and factor_kinds[0] != "numeric":
            return False
        return bool(self.df[factor_col].dropna().nunique() > 10)


def _ap_resolve_help_recipe_for_bucket(self, bucket_widget, fallback_recipe_id=None):
    has_factor2 = bool(self.factor2_bucket.get_assigned_columns())
    has_subject = bool(self.subject_bucket.get_assigned_columns())
    has_covariates = bool(self.covariates_bucket.get_assigned_columns())
    has_binary_outcome = self._is_binary_outcome_for_help()
    factor1_continuous = self._is_continuous_factor1_for_help()

    if bucket_widget is self.factor2_bucket:
        return "mixed_anova" if has_subject else "two_way_anova"

    if bucket_widget is self.subject_bucket:
        return "mixed_anova" if has_factor2 else "repeated_measures_anova"

    if bucket_widget is self.covariates_bucket:
        if has_binary_outcome:
            return "logistic_regression"
        return "linear_regression" if factor1_continuous else "ancova"

    if bucket_widget is self.factor1_bucket:
        if has_binary_outcome:
            return "logistic_regression"
        if has_factor2 and has_subject:
            return "mixed_anova"
        if has_factor2:
            return "two_way_anova"
        if has_subject:
            return "repeated_measures_anova"
        if has_covariates:
            return "linear_regression" if factor1_continuous else "ancova"
        return "correlation" if factor1_continuous else "one_way_anova"

    return fallback_recipe_id


def _ap_on_mapping_changed(self):
    # Sync column-card "assigned" highlight with current bucket state
    if hasattr(self, '_column_cards') and self._column_cards:
        assigned = set(
            col
            for bucket in (self.dv_bucket, self.factor1_bucket, self.factor2_bucket,
                           self.subject_bucket, self.covariates_bucket)
            for col in bucket.get_assigned_columns()
        )
        for col_name, card in self._column_cards.items():
            card.set_assigned(col_name in assigned)

    # Hide the corr/regression widget on every mapping change;
    # it will be re-shown below if is_corr_family is detected.
    if hasattr(self, 'corr_transform_widget'):
        self.corr_transform_widget.setVisible(False)
    if self.df is None:
        self.start_analysis_button.setEnabled(False)
        self.mapping_feedback_label.setText("Load a file to activate the mapping workflow.")
        self._ap_update_analysis_group_selection_ui()
        return

    dv_columns = self.dv_bucket.get_assigned_columns()
    factor_columns = [column for column in [
        *self.factor1_bucket.get_assigned_columns(),
        *self.factor2_bucket.get_assigned_columns(),
    ] if column]
    subject_columns = self.subject_bucket.get_assigned_columns()
    covariate_columns = self.covariates_bucket.get_assigned_columns()
    self._ap_update_analysis_group_selection_ui()

    if not dv_columns:
        self.mapping_feedback_label.setText("Assign at least one measurement column.")
        self.start_analysis_button.setEnabled(False)
        return
    if not factor_columns:
        if subject_columns:
            self.mapping_feedback_label.setText("Assign Factor 1 (group column, for example WT/KO). Subject ID alone is not sufficient.")
        else:
            self.mapping_feedback_label.setText("Assign at least one factor column.")
        self.start_analysis_button.setEnabled(False)
        return
    if not self.multi_mode_button.isChecked() and len(dv_columns) != 1:
        self.mapping_feedback_label.setText("Single mode requires exactly one measurement column.")
        self.start_analysis_button.setEnabled(False)
        return
    if self.multi_mode_button.isChecked() and len(dv_columns) < 2:
        self.mapping_feedback_label.setText("Multi mode requires at least two measurement columns (for example two or more genes).")
        self.start_analysis_button.setEnabled(False)
        return
    if len(factor_columns) > 2:
        self.mapping_feedback_label.setText("Auto-pilot currently supports at most two factor columns.")
        self.start_analysis_button.setEnabled(False)
        return
    if len(subject_columns) > 1:
        self.mapping_feedback_label.setText("Only one subject-ID column is supported.")
        self.start_analysis_button.setEnabled(False)
        return

    wide_info = getattr(self, '_wide_format_info', None)
    if wide_info:
        cond_labels = ', '.join(f'"{c}"' for c in wide_info['value_cols'])
        self.mapping_feedback_label.setText(
            f"Wide format detected \u2192 pivoted to long format. "
            f"Conditions: {cond_labels}. Mapped as paired t-test design."
        )
        self.start_analysis_button.setEnabled(True)
        return

    range_meta = getattr(self, '_range_selection_metadata', None)
    if range_meta:
        selections = range_meta.get('selections', [])
        group_labels = ', '.join(f'"{s["group"]}"' for s in selections)
        n_groups = len(selections)
        self.mapping_feedback_label.setText(
            f"Range selection \u2192 {n_groups} group{'s' if n_groups != 1 else ''} "
            f"({group_labels}). Ready to analyze."
        )
        self.start_analysis_button.setEnabled(True)
        return

    # Check for overlapping assignments (covariates must not overlap with DV/factors/subject)
    all_assigned = dv_columns + factor_columns + subject_columns + covariate_columns
    if len(set(all_assigned)) != len(all_assigned):
        self.mapping_feedback_label.setText("Each mapped role must use a distinct column. Remove duplicates.")
        self.start_analysis_button.setEnabled(False)
        return

    # Covariate cardinality check — warn if a "numeric" covariate looks categorical
    _cov_warnings = []
    for _cov_col in covariate_columns:
        if _cov_col in self.df.columns:
            _n_unique = self.df[_cov_col].nunique()
            _n_total  = len(self.df[_cov_col].dropna())
            if _n_unique == 2:
                _cov_warnings.append(f"'{_cov_col}' has only 2 values → treated as dummy covariate (0/1).")
            elif _n_unique <= 5 and _n_total > 20:
                _cov_warnings.append(f"'{_cov_col}' has few unique values — consider mapping as Factor instead.")

    if _cov_warnings:
        self.mapping_feedback_label.setText(
            "Mapping valid. Note: " + " | ".join(_cov_warnings)
        )
    else:
        self.mapping_feedback_label.setText("Mapping looks valid. Start the analysis when you are ready.")
    self.start_analysis_button.setEnabled(True)

    try:
        context = self._build_analysis_context()
        is_corr_family = context.get("is_corr_family", False)
    except Exception as _ctx_err:
        import traceback
        logger.debug(f"DEBUG _ap_on_mapping_changed context error: {_ctx_err}")
        traceback.print_exc()
        # Fallback: check directly whether the single factor looks continuous
        try:
            factor_columns = [c for c in [
                *self.factor1_bucket.get_assigned_columns(),
                *self.factor2_bucket.get_assigned_columns(),
            ] if c]
            subject_columns = self.subject_bucket.get_assigned_columns()
            if len(factor_columns) == 1 and not subject_columns and self.df is not None:
                from analysis.correlation_models import _is_continuous as _corr_is_continuous
                is_corr_family = _corr_is_continuous(self.df, factor_columns[0])
            else:
                is_corr_family = False
        except Exception:
            is_corr_family = False
    if hasattr(self, 'corr_transform_widget'):
        self.corr_transform_widget.setVisible(is_corr_family)
        if not is_corr_family:
            self.corr_x_transform_combo.setCurrentIndex(0)
            self.corr_y_transform_combo.setCurrentIndex(0)
            if hasattr(self, 'corr_regression_toggle'):
                self.corr_regression_toggle.setChecked(False)
        else:
            # When switching away from regression mode, reset transform combos so that
            # no stale regression transform is passed to the correlation model.
            if hasattr(self, 'corr_regression_toggle') and not self.corr_regression_toggle.isChecked():
                self.corr_x_transform_combo.setCurrentIndex(0)
                self.corr_y_transform_combo.setCurrentIndex(0)


def _ap_browse_file(self):
    file_path, _ = QFileDialog.getOpenFileName(
        self, "Open Excel, CSV or HTML report", "",
        "Excel files (*.xlsx *.xls);;CSV files (*.csv);;HTML reports (*.html);;All files (*.*)"
    )
    if file_path:
        self.file_path = file_path
        self.load_file()


def _ap_load_file(self):
    if not self.file_path:
        return
    try:
        path_lower = self.file_path.lower()

        # HTML report reload: extract embedded tidy data
        if path_lower.endswith(".html"):
            import base64
            import io
            import re as _re
            with open(self.file_path, "r", encoding="utf-8") as _fh:
                html_src = _fh.read()
            match = _re.search(
                r'<script[^>]+id="biomedstatx-tidy-data"[^>]*>(.*?)</script>',
                html_src, _re.DOTALL
            )
            if not match:
                QMessageBox.warning(
                    self, "No tidy data",
                    "This HTML file does not contain embedded BioMedStatX tidy data."
                )
                return
            csv_str = base64.b64decode(match.group(1).strip()).decode("utf-8")
            self.df = pd.read_csv(io.StringIO(csv_str))
            self.sheet_names = ["HTML"]
            self.auto_sheet_combo.clear()
            self.auto_sheet_combo.addItem("HTML")
            self.auto_sheet_combo.setEnabled(False)
        elif path_lower.endswith(".csv"):
            self.df = pd.read_csv(self.file_path)
            self.sheet_names = ["CSV"]
            self.auto_sheet_combo.clear()
            self.auto_sheet_combo.addItem("CSV")
            self.auto_sheet_combo.setEnabled(False)
        else:
            excel = pd.ExcelFile(self.file_path)
            self.sheet_names = excel.sheet_names
            self.auto_sheet_combo.blockSignals(True)
            self.auto_sheet_combo.clear()
            self.auto_sheet_combo.addItems(self.sheet_names)
            self.auto_sheet_combo.blockSignals(False)
            self.auto_sheet_combo.setEnabled(True)
            first_sheet = self.sheet_names[0] if self.sheet_names else 0
            self.df = pd.read_excel(self.file_path, sheet_name=first_sheet)

        self.auto_file_label.setText(os.path.basename(self.file_path))
        self.selected_columns = []
        self.combine_columns = False
        self.analysis_selected_groups = []
        self._range_selection_metadata = None
        self._hint_dismissed = False
        self._maybe_pivot()
        self.numeric_columns = [
            column for column in self.df.columns
            if pd.api.types.is_numeric_dtype(self.df[column])
        ]

        # Show the range-selector button once a file is loaded
        if hasattr(self, "range_select_btn"):
            self.range_select_btn.setVisible(True)

        self._refresh_preview_table()
        self._rebuild_column_cards()
        self._apply_mapping_heuristics()
        self._set_workflow_state("map", "Dataset loaded")
        self.result_cockpit.clear()
        self.decision_tree_panel.show_placeholder(
            "Map the columns, then run the auto-pilot analysis."
        )
        self.current_analysis_context = None
        self.current_analysis_result = None
        self.current_multi_results = {}
        self.current_output_dir = None
        self.on_mapping_changed()
    except Exception as exc:
        self.df = None
        QMessageBox.critical(self, "Error", f"Error loading file: {exc}")


def _ap_load_sheet(self, index):
    if self.file_path is None or self.file_path.endswith(".csv"):
        return
    if index < 0:
        return
    try:
        self.df = pd.read_excel(self.file_path, sheet_name=self.auto_sheet_combo.itemText(index))
        self.analysis_selected_groups = []
        self._maybe_pivot()
        self.numeric_columns = [column for column in self.df.columns if pd.api.types.is_numeric_dtype(self.df[column])]
        self._refresh_preview_table()
        self._rebuild_column_cards()
        self._apply_mapping_heuristics()
        self._set_workflow_state("map", "Worksheet loaded")
        self.on_mapping_changed()
    except Exception as exc:
        QMessageBox.critical(self, "Error", f"Error loading worksheet: {exc}")


def _ap_build_analysis_context(self):
    dv_columns = self.dv_bucket.get_assigned_columns()
    factor_columns = [column for column in [
        *self.factor1_bucket.get_assigned_columns(),
        *self.factor2_bucket.get_assigned_columns(),
    ] if column]
    subject_columns = self.subject_bucket.get_assigned_columns()
    subject_column = subject_columns[0] if subject_columns else None
    covariate_columns = self.covariates_bucket.get_assigned_columns()
    active_filter = getattr(self, 'filter_bucket', None)
    filter_spec = active_filter.get_filter() if active_filter else None

    all_assigned = dv_columns + factor_columns + ([subject_column] if subject_column else []) + covariate_columns
    if len(set(all_assigned)) != len(all_assigned):
        raise ValueError("Each mapped role must use a distinct column.")

    if len(factor_columns) == 0 or len(factor_columns) > 2:
        raise ValueError("Auto-pilot currently supports one or two factor columns.")

    context = {
        "mode": "multi" if self.multi_mode_button.isChecked() else "single",
        "dv_columns": dv_columns,
        "factor_columns": factor_columns,
        "subject_column": subject_column,
        "covariates": covariate_columns,
        "between_factors": [],
        "within_factors": [],
        "dependent": bool(subject_column),
        "group_labels": [],
        "display_group_col": factor_columns[0],
        "inferred_test": None,
        "filter": filter_spec,
        "selected_groups": list(self.analysis_selected_groups or []),
        "selected_group_column": factor_columns[0],
    }

    analysis_df = self.df.copy()
    if filter_spec:
        filter_col, filter_val = filter_spec
        if filter_col in analysis_df.columns:
            analysis_df = analysis_df[analysis_df[filter_col] == filter_val]

    factor1_levels = _sorted_unique(analysis_df[factor_columns[0]].dropna().tolist())
    selected_factor1_groups = [group for group in context["selected_groups"] if group in factor1_levels]
    if selected_factor1_groups:
        if len(selected_factor1_groups) < 2:
            raise ValueError("At least two selected groups are required for the analysis.")
        context["selected_groups"] = selected_factor1_groups
        analysis_df = analysis_df[analysis_df[factor_columns[0]].isin(selected_factor1_groups)]
    else:
        context["selected_groups"] = []

    # --- Binary DV detection: Logistic Regression ---
    # --- Proportion DV detection: Beta Regression ---
    if len(dv_columns) == 1:
        dv_col = dv_columns[0]
        _series = analysis_df[dv_col].dropna()
        _unique = _series.unique()
        # Conservative check: exactly 2 values that are 0/1 (or two strings),
        # AND column name does not hint at a grouping variable.
        _is_01 = set(_unique) <= {0, 1, 0.0, 1.0}
        _is_str = all(isinstance(v, str) for v in _unique)
        _group_hints = {"group", "arm", "treatment", "condition", "sex",
                        "gender", "cohort", "batch", "grp"}
        _name_is_grouping = any(h in dv_col.lower() for h in _group_hints)
        is_binary = (
            len(_unique) == 2
            and pd.api.types.is_numeric_dtype(self.df[dv_col]) or _is_str
            and (_is_01 or _is_str)
            and not _name_is_grouping
        )
        if is_binary:
            context["outcome_type"] = "binary"

        # Beta regression: outcome is a continuous proportion in [0, 1]
        # Detection is purely data-driven (no column name assumptions).
        # Guard: >5 unique values rules out discrete encoded scales (e.g. 0/0.25/0.5/0.75/1).
        if (
            not is_binary
            and pd.api.types.is_numeric_dtype(_series)
            and len(_unique) > 5
        ):
            _min, _max = float(_series.min()), float(_series.max())
            _in_unit_interval = _min >= 0.0 and _max <= 1.0
            if _in_unit_interval:
                _n = int(_series.count())
                _n_predictors = max(1, len(covariate_columns) + 1)
                _epv = _n / _n_predictors
                _has_boundary = _min == 0.0 or _max == 1.0

                if _has_boundary:
                    # Apply Smithson-Verkuilen transformation to push boundary values inside (0,1)
                    self.df[dv_col] = (self.df[dv_col] * (_n - 1) + 0.5) / _n
                    context["beta_sv_transformed"] = True

                context["outcome_type"] = "proportion"
                context["beta_epv"] = _epv
                context["beta_n"] = _n
                context["beta_n_predictors"] = _n_predictors
                context["beta_bias_corrected"] = _epv < 10

    if len(factor_columns) == 1:
        factor = factor_columns[0]
        levels = _sorted_unique(analysis_df[factor].tolist())
        if len(levels) < 2:
            raise ValueError(f"Factor '{factor}' needs at least two levels.")
        context["group_labels"] = levels
        if subject_column:
            subject_span = analysis_df.groupby(subject_column)[factor].nunique(dropna=True)
            if not subject_span.empty and subject_span.max() <= 1:
                raise ValueError("The selected Subject ID does not create a repeated-measures structure for Factor 1.")
            context["within_factors"] = [factor]
            context["inferred_test"] = "paired_ttest" if len(levels) == 2 else "repeated_measures_anova"
        else:
            context["inferred_test"] = "independent_ttest" if len(levels) == 2 else "one_way_anova"
    else:
        factor_a, factor_b = factor_columns
        combinations = []
        seen_pairs = set()
        for _, row in analysis_df[[factor_a, factor_b]].dropna().iterrows():
            pair = (row[factor_a], row[factor_b])
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            combinations.append(f"{factor_a}={row[factor_a]}, {factor_b}={row[factor_b]}")
        context["group_labels"] = sorted(combinations, key=lambda item: str(item))
        context["display_group_col"] = "__AUTO_GROUP__"

        if subject_column:
            role_by_factor = {}
            for factor in factor_columns:
                per_subject = analysis_df.groupby(subject_column)[factor].nunique(dropna=True)
                role_by_factor[factor] = "between" if not per_subject.empty and per_subject.max() <= 1 else "within"
            between_factors = [factor for factor, role in role_by_factor.items() if role == "between"]
            within_factors = [factor for factor, role in role_by_factor.items() if role == "within"]
            if len(between_factors) != 1 or len(within_factors) != 1:
                raise ValueError(
                    "With Subject ID plus two factors, auto-pilot requires exactly one between-subject factor and one within-subject factor."
                )
            context["between_factors"] = between_factors
            context["within_factors"] = within_factors
            context["inferred_test"] = "mixed_anova"
        else:
            context["between_factors"] = factor_columns[:2]
            context["inferred_test"] = "two_way_anova"
            context["dependent"] = False

    # --- Clinical test upgrades ---

    # 1. Binary DV → Logistic Regression (overrides everything)
    if context.get("outcome_type") == "binary":
        context["inferred_test"] = "logistic_regression"

    # 1b. Proportion DV → Beta Regression (overrides group-comparison tests)
    elif context.get("outcome_type") == "proportion":
        context["inferred_test"] = "beta_regression"
        # Pass EPV info into covariates/analysis metadata for dispatch
        if context.get("beta_bias_corrected"):
            context["beta_regression_variant"] = "bias_corrected"
        else:
            context["beta_regression_variant"] = "standard"

    # 2. Unbalanced repeated-measures → LMM (when Subject ID + within-factor present)
    elif subject_column and context["within_factors"]:
        within_factor = context["within_factors"][0]
        dv_col_for_balance = dv_columns[0] if dv_columns else None
        try:
            # Case 1: structural missingness (whole Subject×Timepoint combos absent)
            counts = self.df.groupby([subject_column, within_factor]).size().unstack(fill_value=0)
            has_structural_missing = (counts == 0).any().any()

            # Case 2: row exists but DV is NaN (patient present at visit but no measurement)
            has_nan_missing = False
            if dv_col_for_balance and not has_structural_missing:
                valid = self.df[[subject_column, within_factor, dv_col_for_balance]].dropna(
                    subset=[dv_col_for_balance]
                )
                valid_counts = valid.groupby([subject_column, within_factor]).size().unstack(fill_value=0)
                has_nan_missing = (valid_counts == 0).any().any()

            if has_structural_missing or has_nan_missing:
                context["inferred_test"] = "lmm"
        except Exception:
            pass

    # 3. Covariates present → ANCOVA upgrade (only for non-clinical tests)
    if covariate_columns and context["inferred_test"] in ("independent_ttest", "one_way_anova"):
        context["inferred_test"] = "ancova"
    elif covariate_columns and context["inferred_test"] == "two_way_anova":
        context["inferred_test"] = "two_way_ancova"

    # Covariates also flow into LMM/mixed_anova as additional fixed effects
    # (handled in dispatch, no test upgrade needed)

    # 4. Continuous primary factor → Correlation or Linear Regression
    #    Applied AFTER all other upgrades so it takes precedence over ANCOVA/t-test
    #    inferences when the factor is not a grouping variable.
    if len(factor_columns) == 1 and not subject_column:
        try:
            from analysis.correlation_models import _is_continuous as _corr_is_continuous
            if _corr_is_continuous(analysis_df, factor_columns[0]):
                if covariate_columns:
                    context["inferred_test"] = "linear_regression"
                else:
                    # Mark as corr-family so the mode widget stays visible regardless of toggle.
                    context["is_corr_family"] = True
                    _use_regression = (
                        hasattr(self, "corr_regression_toggle")
                        and self.corr_regression_toggle.isChecked()
                    )
                    context["inferred_test"] = "linear_regression" if _use_regression else "correlation"
                context["x_variable"] = factor_columns[0]
        except Exception:
            pass  # correlation_models not available — skip silently

    if context["mode"] == "single" and len(dv_columns) != 1:
        raise ValueError("Single mode requires exactly one measurement column.")
    if context["mode"] == "multi" and len(dv_columns) < 2:
        raise ValueError("Multi mode requires at least two measurement columns (for example two or more genes).")
    if context["mode"] == "multi" and context["inferred_test"] in {"independent_ttest", "paired_ttest", "logistic_regression", "beta_regression"}:
        raise ValueError("Multi mode is restricted to ANOVA-capable designs.")

    # Variable transforms (meaningful for both correlation and user-selected linear regression)
    if hasattr(self, 'corr_x_transform_combo'):
        context['x_transform'] = self.corr_x_transform_combo.currentText() or 'none'
        context['y_transform'] = self.corr_y_transform_combo.currentText() or 'none'
    else:
        context['x_transform'] = 'none'
        context['y_transform'] = 'none'

    # Pre-flight data validation for regression/correlation transformations
    if context['x_transform'] != 'none' or context['y_transform'] != 'none':
        def _check_bounds(col, transform_name):
            if transform_name == 'none': return
            series = pd.to_numeric(analysis_df[col], errors='coerce').dropna()
            if len(series) == 0: return
            min_val = series.min()
            if transform_name in ('log10', 'boxcox') and min_val <= 0:
                raise ValueError(f"Dataset contains zero or negative values in '{col}'. Standard {transform_name} transformation is mathematically undefined. Consider using log10(x+1) instead, or filter out non-positive values.")
            if transform_name == 'log10(x+1)' and min_val <= -1:
                raise ValueError(f"Dataset contains values <= -1 in '{col}'. log10(x+1) transformation is mathematically undefined.")
            if transform_name == 'sqrt' and min_val < 0:
                raise ValueError(f"Dataset contains negative values in '{col}'. Square root transformation is mathematically undefined.")

        if context.get("inferred_test") in ("correlation", "linear_regression", "lmm"):
            if context["factor_columns"]:
                _check_bounds(context["factor_columns"][0], context['x_transform'])
            for dv in context["dv_columns"]:
                _check_bounds(dv, context['y_transform'])

    return context


def _ap_detected_test_label(self, context):
    labels = {
        "independent_ttest": "Independent t-test",
        "paired_ttest": "Paired t-test",
        "one_way_anova": "One-Way ANOVA",
        "repeated_measures_anova": "Repeated Measures ANOVA",
        "two_way_anova": "Two-Way ANOVA",
        "mixed_anova": "Mixed ANOVA",
        "ancova": "ANCOVA (One-Way + Covariates)",
        "two_way_ancova": "Two-Way ANCOVA",
        "lmm": "Linear Mixed Model (handles missing visits)",
        "logistic_regression": "Logistic Regression (Binary Outcome)",
        "beta_regression": "Beta Regression (Proportion Outcome)",
        "correlation": "Korrelationsanalyse (Spearman/Pearson)",
        "linear_regression": "Lineare Regression (OLS)",
    }
    label = labels.get(context["inferred_test"], context["inferred_test"])
    if context.get("inferred_test") == "beta_regression":
        epv = context.get("beta_epv")
        sv = context.get("beta_sv_transformed", False)
        bias = context.get("beta_bias_corrected", False)
        parts = []
        if sv:
            parts.append("Smithson-Verkuilen transformation applied (boundary values present)")
        if bias and epv is not None:
            parts.append(f"EPV = {epv:.1f} — small sample, bias correction applied")
        elif epv is not None:
            parts.append(f"EPV = {epv:.1f} — adequate sample size")
        if parts:
            label += "\n  " + "\n  ".join(parts)
    return label


def _ap_execute_single_analysis(self, context, dv_column, output_dir, skip_plots=True, title_suffix=None, file_base_override=None):
    if not self.file_path:
        raise ValueError("No input file selected.")

    base_name = os.path.splitext(os.path.basename(self.file_path))[0]
    if file_base_override is not None:
        file_base = file_base_override
    else:
        file_base = os.path.join(output_dir, f"{_safe_file_slug(base_name)}_{_safe_file_slug(dv_column)}")
    group_labels = context.get("group_labels", [])
    colors = [DEFAULT_COLORS[index % len(DEFAULT_COLORS)] for index, _ in enumerate(group_labels)]
    hatches = [DEFAULT_HATCHES[index % len(DEFAULT_HATCHES)] for index, _ in enumerate(group_labels)]
    single_context = dict(context)
    single_context["dv_columns"] = [dv_column]
    single_context["current_dv"] = dv_column
    # Single source of truth: always inject the in-memory df. UI may have
    # applied pivot, range-selection, or outlier removal — re-reading the
    # file would silently diverge from what the user sees on screen.
    if getattr(self, "df", None) is not None:
        single_context["injected_df"] = self.df

    result = AnalysisManager.analyze(
        file_path=self.file_path,
        group_col=context.get("display_group_col", context["factor_columns"][0]),
        groups=group_labels,
        sheet_name=self.auto_sheet_combo.currentText() if self.auto_sheet_combo.isEnabled() else 0,
        value_cols=[dv_column],
        dependent=context.get("dependent", False),
        compare=None,
        colors=colors,
        hatches=hatches,
        title=title_suffix or dv_column,
        x_label=", ".join(context["factor_columns"]),
        y_label=dv_column,
        file_name=file_base,
        save_plot=True,
        skip_plots=skip_plots,
        error_type="sd",
        analysis_context=single_context,
        subject_column=context.get("subject_column"),
        test=single_context.get("inferred_test", ""),
    )

    # Inject provenance metadata into the HTML report if range selection was used
    if (
        getattr(self, "_range_selection_metadata", None) is not None
        and self.df is not None
    ):
        html_path = file_base + ".html"
        if os.path.isfile(html_path):
            try:
                import base64
                import json
                with open(html_path, "r", encoding="utf-8") as _fh:
                    html_content = _fh.read()
                meta_json = json.dumps(self._range_selection_metadata)
                tidy_b64 = base64.b64encode(
                    self.df.to_csv(index=False).encode()
                ).decode()
                inject = (
                    f'\n<script id="biomedstatx-provenance" type="application/json">'
                    f'{meta_json}</script>\n'
                    f'<script id="biomedstatx-tidy-data" type="text/plain">'
                    f'{tidy_b64}</script>\n'
                )
                html_content = html_content.replace("</body>", inject + "</body>")
                with open(html_path, "w", encoding="utf-8") as _fh:
                    _fh.write(html_content)
            except Exception as _exc:
                logger.warning(f"WARNING: Could not inject provenance into HTML: {_exc}")

    return result


def _ap_format_assumptions(self, results):
    normality = "Normality not available"
    variance = "Variance check not available"

    normality_tests = results.get("normality_tests", {})
    transformation_applied = bool(
        results.get("transformation")
        and str(results.get("transformation")).lower() not in ("none", "no further")
    )
    if transformation_applied and "model_residuals_transformed" in normality_tests:
        is_normal = normality_tests["model_residuals_transformed"].get("is_normal")
        normality = "Normality OK after transformation" if is_normal else "Normality still violated after transformation"
    elif "model_residuals" in normality_tests:
        is_normal = normality_tests["model_residuals"].get("is_normal")
        normality = "Normality OK" if is_normal else "Normality violated"
    elif "transformed_data" in normality_tests:
        is_normal = normality_tests["transformed_data"].get("is_normal")
        normality = "Normality OK after transformation" if is_normal else "Normality violated"
    elif "all_data" in normality_tests:
        is_normal = normality_tests["all_data"].get("is_normal")
        normality = "Normality OK" if is_normal else "Normality violated"

    variance_test = results.get("variance_test", {})
    variance_source = variance_test.get("transformed", variance_test) if transformation_applied else variance_test
    if variance_source and isinstance(variance_source, dict) and variance_source.get("equal_variance") is not None:
        variance = "Variance homogeneous" if variance_source.get("equal_variance") else "Variance heterogeneous"

    return f"{normality}. {variance}."


def _ap_extract_normality_metric(self, results):
    # Clinical models: override card semantics
    model_type = results.get("model_type")
    if model_type == "LogisticRegression":
        hl = results.get("hosmer_lemeshow", {})
        p = hl.get("p_value")
        if p is not None:
            return f"Hosmer-Lemeshow p = {p:.4f}" + (" (Good fit)" if p > 0.05 else " (Poor fit)")
        return "Hosmer-Lemeshow: N/A"
    if model_type == "LMM":
        converged = results.get("converged")
        if converged is True:
            return "Model converged"
        elif converged is False:
            return "Model did NOT converge"
        return "Convergence: N/A"
    if model_type == "ANCOVA":
        slope_hom = results.get("slope_homogeneity", {})
        violations = [k for k, v in slope_hom.items() if v.get("assumption_holds") is False]
        if violations:
            return f"Slope homogeneity violated ({', '.join(violations)})"
        elif slope_hom:
            return "Slope homogeneity OK"

    normality_tests = results.get("normality_tests", {})
    transformation_applied = bool(
        results.get("transformation")
        and str(results.get("transformation")).lower() not in ("none", "no further")
    )
    if transformation_applied and "model_residuals_transformed" in normality_tests:
        is_normal = normality_tests["model_residuals_transformed"].get("is_normal")
        return "OK (after transformation)" if is_normal else "Violated (after transformation)"
    if "model_residuals" in normality_tests:
        is_normal = normality_tests["model_residuals"].get("is_normal")
        return "OK" if is_normal else "Violated"
    if "transformed_data" in normality_tests:
        is_normal = normality_tests["transformed_data"].get("is_normal")
        return "OK (after transformation)" if is_normal else "Violated"
    if "all_data" in normality_tests:
        is_normal = normality_tests["all_data"].get("is_normal")
        return "OK" if is_normal else "Violated"
    return "Not available"


def _ap_extract_variance_metric(self, results):
    # Clinical models: override card semantics
    model_type = results.get("model_type")
    if model_type == "LogisticRegression":
        roc = results.get("roc_data", {})
        auc = roc.get("auc")
        if auc is not None:
            return f"AUC = {auc:.3f}"
        return "AUC: N/A"
    if model_type == "LMM":
        icc = results.get("icc")
        if icc is not None:
            return f"ICC = {icc:.3f}"
        return "ICC: N/A"
    if model_type == "ANCOVA":
        r2 = results.get("r_squared_adj")
        if r2 is not None:
            return f"Adj. R² = {r2:.3f}"

    transformation_applied = bool(
        results.get("transformation")
        and str(results.get("transformation")).lower() not in ("none", "no further")
    )
    variance_test = results.get("variance_test", {})
    variance_source = variance_test.get("transformed", variance_test) if transformation_applied else variance_test
    if variance_source and isinstance(variance_source, dict):
        equal_variance = variance_source.get("equal_variance")
        if equal_variance is not None:
            try:
                return "Homogeneous" if bool(equal_variance) else "Heterogeneous"
            except Exception:
                pass
    return "Not available"


def _ap_format_main_test_metric(self, results):
    tested_against = results.get("tested_against") or results.get("final_test_label") or results.get("test") or "Not available"
    p_value = results.get("p_value")
    if p_value is None:
        return f"{tested_against}; p = N/A"
    if p_value < 0.0001:
        p_text = "< 0.0001"
    else:
        p_text = f"= {p_value:.4f}"
    return f"{tested_against}; p {p_text}"


def _ap_format_effect_size_metric(self, results):
    model_type = results.get("model_type")
    effect_size = results.get("effect_size")
    effect_size_type = results.get("effect_size_type")

    # For logistic regression, show primary OR
    if model_type == "LogisticRegression":
        or_table = results.get("odds_ratios", [])
        if or_table:
            primary = or_table[0]
            return f"OR = {primary['odds_ratio']:.2f} [{primary['ci_lower']:.2f}-{primary['ci_upper']:.2f}]"
        return "OR: N/A"

    if effect_size is None:
        return "Not available"

    labels = {
        "cohen_d": "Cohen's d",
        "hedges_g": "Hedges' g",
        "cohen_f": "Cohen's f",
        "cohen's f": "Cohen's f",
        "r": "r (rank correlation)",
        "eta_squared": "Eta-squared",
        "partial_eta_squared": "Partial eta-squared",
        "epsilon_squared": "Epsilon-squared",
        "kendall_w": "Kendall's W",
        "rank_biserial_r": "Rank-biserial r",
        "icc": "ICC",
        "auc": "AUC",
    }
    # Match case-insensitively so plain-text types (e.g. "Cohen's f") resolve to
    # a clean label instead of being mangled by .title() ("Cohen'S F").
    type_label = labels.get(
        str(effect_size_type).lower() if effect_size_type else None,
        effect_size_type if effect_size_type else "Effect size",
    )
    return f"{type_label} = {effect_size:.4f}"


def _ap_is_ttest_result(self, context, results):
    if context.get("inferred_test") in {"independent_ttest", "paired_ttest"}:
        return True

    fields_to_scan = [
        results.get("tested_against"),
        results.get("final_test_label"),
        results.get("test"),
    ]
    for field in fields_to_scan:
        if not field:
            continue
        lowered = str(field).lower()
        if "t-test" in lowered or "ttest" in lowered:
            return True
    return False


def _ap_format_rationale(self, context, results):
    reasons = [f"Structure inferred as {self._detected_test_label(context)}."]
    if context.get("subject_column"):
        reasons.append(f"Subject ID detected via '{context['subject_column']}'.")
    if context.get("covariates"):
        reasons.append(f"Covariates: {', '.join(context['covariates'])}.")

    model_type = results.get("model_type")
    if model_type == "ANCOVA":
        reasons.append("Treatment effects are adjusted for covariates (Type II SS).")
    elif model_type == "LMM":
        reasons.append("Linear Mixed Model uses all available data (ML estimation) — missing visits do not cause patient dropout.")
    elif model_type == "LogisticRegression":
        reasons.append("Binary outcome detected. Logistic regression provides odds ratios and AUC.")
    elif model_type == "BetaRegression":
        epv = context.get("beta_epv")
        sv = context.get("beta_sv_transformed", False)
        bias = context.get("beta_bias_corrected", False)
        if sv:
            reasons.append("Boundary values (0 or 1) were present — Smithson-Verkuilen transformation applied before fitting.")
        if bias and epv is not None:
            reasons.append(
                f"Proportion outcome detected (EPV = {epv:.1f} < 10 — small sample). "
                f"Bias-corrected Beta Regression applied with bootstrapped standard errors."
            )
        elif epv is not None:
            reasons.append(
                f"Proportion outcome detected (EPV = {epv:.1f} ≥ 10 — adequate sample). "
                f"Standard Beta Regression applied."
            )
        else:
            reasons.append("Proportion outcome detected. Beta Regression applied.")
    elif results.get("transformation"):
        reasons.append(f"Transformation chosen by user: {results['transformation']}.")

    if results.get("analysis_note"):
        reasons.append(results["analysis_note"])
    elif results.get("note"):
        reasons.append(results["note"])
    elif results.get("recommendation") == "non_parametric":
        reasons.append("Parametric assumptions failed, so a robust fallback model path was used.")
    elif model_type not in ("ANCOVA", "LMM", "LogisticRegression"):
        reasons.append("Auto-pilot stayed on the default supported path for this design.")
    return " ".join(reasons)


def _ap_format_posthoc_status(self, context, results):
    if self._is_ttest_result(context, results):
        return "No post-hoc applicable for t-tests (two groups only)."

    if results.get("posthoc_test"):
        return f"{results['posthoc_test']}."
    if results.get("p_value") is not None and results["p_value"] < 0.05 and len(results.get("groups", [])) > 2:
        return "Significant omnibus result, but no post-hoc result was stored."
    if results.get("p_value") is not None and results["p_value"] >= 0.05:
        return "No post-hoc required because the omnibus test was not significant."
    return "No post-hoc performed."


def _ap_format_context_design(self, context, results):
    model_label = self._detected_test_label(context)
    factor_columns = context.get("factor_columns") or []
    factor_text = ", ".join(map(str, factor_columns)) if factor_columns else "Not specified"
    subject_column = context.get("subject_column") or "None"
    return (
        f"Model: {model_label}\n"
        f"Factors: {factor_text}\n"
        f"Subject ID: {subject_column}"
    )


def _ap_format_context_sample_overview(self, context, results):
    selected_groups = results.get("selected_groups") or context.get("selected_groups") or results.get("groups") or []
    selected_groups = [str(group) for group in selected_groups]
    group_column = results.get("group_column") or context.get("display_group_col") or "Not specified"

    n_total = results.get("n_total")
    if n_total is None:
        n_total = results.get("n")
    if n_total is None:
        raw_data = results.get("raw_data") or {}
        if isinstance(raw_data, dict):
            n_total = sum(len(values) for values in raw_data.values() if hasattr(values, "__len__"))

    if selected_groups:
        if len(selected_groups) > 6:
            group_text = ", ".join(selected_groups[:6]) + f" (+{len(selected_groups) - 6} more)"
        else:
            group_text = ", ".join(selected_groups)
    else:
        group_text = "All available groups"

    n_display = str(n_total) if n_total is not None else "N/A"
    return (
        f"Sample size (N): {n_display}\n"
        f"Grouping column: {group_column}\n"
        f"Groups: {group_text}"
    )


def _ap_format_context_analysis_scope(self, context, results):
    covariates = results.get("covariates") or context.get("covariates") or []
    covariate_text = ", ".join(map(str, covariates)) if covariates else "None"

    filter_text = results.get("filter_applied")
    if not filter_text and context.get("filter"):
        filter_col, filter_val = context["filter"]
        filter_text = f"{filter_col} = {filter_val}"
    if not filter_text:
        filter_text = "None"

    posthoc_text = self._format_posthoc_status(context, results)
    return (
        f"Filter: {filter_text}\n"
        f"Covariates: {covariate_text}\n"
        f"Post-hoc: {posthoc_text}"
    )


def _ap_render_result_summary(self, context, results, output_dir, subtitle):
    summary = {
        "subtitle": subtitle,
        "metric_normality": self._extract_normality_metric(results),
        "metric_variance": self._extract_variance_metric(results),
        "inference_main_test": self._format_main_test_metric(results),
        "inference_effect_size": self._format_effect_size_metric(results),
        "context_design": self._format_context_design(context, results),
        "context_sample_overview": self._format_context_sample_overview(context, results),
        "context_analysis_scope": self._format_context_analysis_scope(context, results),
    }
    self.result_cockpit.set_summary(summary, enable_plot=False, enable_output=bool(output_dir))
    ConfettiOverlay(self)
    self.decision_tree_panel.update_results(results)
    self._set_workflow_state("results", "Results ready")
    self.current_output_dir = output_dir
    self.current_analysis_context = context
    self.current_analysis_result = results
    self.current_rendered_dataset = context.get("current_dv") or context["dv_columns"][0]
    self.samples = results.get("raw_data") or results.get("samples")
    self.available_groups = list((self.samples or {}).keys())


def _ap_determine_and_run_test(self):
    if self.df is None:
        QMessageBox.warning(self, "Error", "Please load a dataset first.")
        return

    try:
        context = self._build_analysis_context()
    except Exception as exc:
        QMessageBox.warning(self, "Invalid Mapping", str(exc))
        return

    _ap_base_name = _safe_file_slug(os.path.splitext(os.path.basename(self.file_path))[0])
    if context["mode"] == "single":
        _ap_suggested = f"{_ap_base_name}_{_safe_file_slug(context['dv_columns'][0])}.html"
    else:
        _ap_suggested = f"{_ap_base_name}_multi_dataset_results.html"

    ap_file_path, _ = QFileDialog.getSaveFileName(
        self, "Save Analysis Report", _ap_suggested,
        "HTML Report (*.html);;All Files (*)"
    )
    if not ap_file_path:
        return
    if not ap_file_path.lower().endswith('.html'):
        ap_file_path += '.html'
    output_dir = os.path.dirname(ap_file_path) or os.getcwd()

    self._set_workflow_state("analyze", "Running analysis", running=True)
    self.mapping_feedback_label.setText("Auto-pilot is analyzing the mapped design.")
    self.decision_tree_panel.show_placeholder("Analyzing data and tracing the statistical decision path...")
    QApplication.processEvents()

    try:
        if context["mode"] == "single":
            file_base_override = os.path.splitext(ap_file_path)[0]
            result = self._execute_single_analysis(context, context["dv_columns"][0], output_dir, skip_plots=True, file_base_override=file_base_override)
            if result.get("blocked"):
                self._handle_blocked_result(result)
                return
            self._render_result_summary(
                context,
                result,
                output_dir,
                subtitle=f"Analysis completed for '{context['dv_columns'][0]}'."
            )
            self.current_multi_results = {}
        else:
            all_results = {}
            for dv_column in context["dv_columns"]:
                per_dv_context = dict(context)
                per_dv_context["dv_columns"] = [dv_column]
                per_dv_context["current_dv"] = dv_column
                QApplication.processEvents()
                all_results[dv_column] = self._execute_single_analysis(per_dv_context, dv_column, output_dir, skip_plots=True)

            combined_report = ap_file_path
            export_result = ExportDispatcher.export_multi_dataset_results(all_results, combined_report)
            if export_result.get("warning"):
                logger.warning(f"WARNING: {export_result['warning']}")

            lead_dv = context["dv_columns"][0]
            lead_result = all_results[lead_dv]
            lead_context = dict(context)
            lead_context["dv_columns"] = [lead_dv]
            lead_context["current_dv"] = lead_dv
            if lead_result.get("blocked"):
                self._handle_blocked_result(lead_result)
                self.current_multi_results = all_results
                return
            self._render_result_summary(
                lead_context,
                lead_result,
                output_dir,
                subtitle=f"Multi-dataset analysis completed for {len(all_results)} dependent variables. Combined report: {os.path.basename(combined_report)}"
            )
            self.current_multi_results = all_results
            self.current_analysis_result["combined_report"] = combined_report

    except Exception as exc:
        self._set_workflow_state("map", "Analysis failed")
        self.decision_tree_panel.show_placeholder(f"Analysis failed: {exc}")
        QMessageBox.critical(self, "Analysis Error", str(exc))


def _ap_handle_blocked_result(self, result):
    """Surface a data-quality block: clear cockpit cards, show the reason in the
    cockpit + decision-tree panel, and stop short of rendering a (non-existent)
    result. No confetti, no Open-Output."""
    reason = result.get("block_reason") or result.get("error") or "Analysis could not be performed."
    warnings = result.get("warnings") or []
    self.result_cockpit.show_block(reason, warnings)
    self.decision_tree_panel.show_placeholder(reason)
    self._set_workflow_state("map", "Analysis blocked")
    self.current_analysis_result = result
    self.current_multi_results = getattr(self, "current_multi_results", {}) or {}


def _ap_configure_plot_from_result(self):
    if not self.current_analysis_result or not self.current_analysis_context:
        QMessageBox.information(self, "No Result", "Run an analysis before configuring a plot.")
        return

    groups = list((self.current_analysis_result.get("raw_data") or self.current_analysis_result.get("samples") or {}).keys())
    if not groups:
        QMessageBox.warning(self, "Plot Configuration", "No group data are available for plot configuration.")
        return

    self.samples = self.current_analysis_result.get("raw_data") or self.current_analysis_result.get("samples")
    self.available_groups = groups

    default_filename = os.path.splitext(os.path.basename(self.file_path))[0]
    if self.current_rendered_dataset:
        default_filename = f"{default_filename}_{_safe_file_slug(self.current_rendered_dataset)}"

    dialog = PlotAestheticsDialog(
        groups=groups,
        samples=self.samples or {},
        analysis_result=self.current_analysis_result,
        parent=self,
        default_filename=default_filename,
        dependent=self.current_analysis_context.get("dependent", False),
    )
    if hasattr(dialog, 'create_plot_check'):
        dialog.create_plot_check.setChecked(False)

    if dialog.exec_() != QDialog.Accepted:
        return

    plot_config = dialog.get_config()
    if not plot_config:
        return

    output_dir = self.current_output_dir or QFileDialog.getExistingDirectory(self, "Select output directory for plot export")
    if not output_dir:
        return

    try:
        self._set_workflow_state("analyze", "Rendering plot", running=True)
        QApplication.processEvents()

        context = dict(self.current_analysis_context)
        if self.current_rendered_dataset:
            context["dv_columns"] = [self.current_rendered_dataset]
            context["current_dv"] = self.current_rendered_dataset
        context["group_labels"] = plot_config["groups"]
        # Single source of truth: re-inject current in-memory df (see _execute_single_analysis).
        if getattr(self, "df", None) is not None and context.get("injected_df") is None:
            context["injected_df"] = self.df

        file_base = os.path.join(
            output_dir,
            plot_config.get("file_name") or f"{_safe_file_slug(os.path.splitext(os.path.basename(self.file_path))[0])}_{_safe_file_slug(context['dv_columns'][0])}_plot"
        )
        appearance = plot_config.get("appearance_settings", {})
        plot_result = AnalysisManager.analyze(
            file_path=self.file_path,
            group_col=context.get("display_group_col", context["factor_columns"][0]),
            groups=plot_config["groups"],
            sheet_name=self.auto_sheet_combo.currentText() if self.auto_sheet_combo.isEnabled() else 0,
            value_cols=context["dv_columns"],
            dependent=context.get("dependent", False),
            compare=None,
            colors=[plot_config["colors"].get(group, DEFAULT_COLORS[index % len(DEFAULT_COLORS)]) for index, group in enumerate(plot_config["groups"])],
            hatches=[plot_config["hatches"].get(group, "") for group in plot_config["groups"]],
            title=plot_config.get("title") or context["dv_columns"][0],
            x_label=plot_config.get("x_label"),
            y_label=plot_config.get("y_label"),
            file_name=file_base,
            save_plot=True,
            skip_plots=not plot_config.get("create_plot", True),
            error_type=plot_config.get("error_type", "sd"),
            analysis_context=context,
            subject_column=context.get("subject_column"),
            plot_type=appearance.get("plot_type", "Bar"),
            dpi=appearance.get("dpi", 300),
            colors_override=plot_config["colors"],
            test=context.get("inferred_test", ""),
        )

        files = []
        for ext in ("xlsx", "pdf", "png"):
            candidate = f"{file_base}.{ext}"
            if os.path.exists(candidate):
                files.append(candidate)
        self.show_analysis_success_dialog("Configured plot export", files, output_dir)
        self.current_analysis_result = plot_result
        self._set_workflow_state("results", "Results ready")
    except Exception as exc:
        self._set_workflow_state("results", "Plot export failed")
        QMessageBox.critical(self, "Plot Error", str(exc))


def _ap_open_current_output_folder(self):
    if self.current_output_dir and os.path.isdir(self.current_output_dir):
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.current_output_dir))


def _ap_reset_application_state(self):
    self.current_analysis_context = None
    self.current_analysis_result = None
    self.current_multi_results = {}
    self.current_output_dir = None
    self.current_rendered_dataset = None
    self.analysis_selected_groups = []
    self.temp_plot_appearance_settings = None
    self._wide_format_info = None
    self._range_selection_metadata = None
    self._hint_dismissed = False
    if hasattr(self, "_range_groups_label"):
        self._range_groups_label.setVisible(False)
    self.result_cockpit.clear()
    self.decision_tree_panel.show_placeholder("Map the columns, then run the auto-pilot analysis.")
    self._set_workflow_state("map" if self.df is not None else "load", "Ready for the next analysis")
    if self.df is not None:
        self._apply_mapping_heuristics()
        self.on_mapping_changed()


def _ap_maybe_pivot(self):
    """
    Detects wide-format paired/repeated data and, if found, replaces self.df
    with the melted long-format version. Records the transformation in
    self._wide_format_info for downstream use and UI feedback. No-op for
    long-format data.
    """
    result = _detect_wide_format(self.df)
    self._wide_format_info = None
    if result is None:
        return
    self.df = _pivot_wide_to_long(self.df, result["subject_col"], result["value_cols"])
    self._wide_format_info = result


def _ap_open_exploratory_matrix_dialog(self):
    """Open the ExploratoryMatrixDialog with the currently loaded DataFrame."""
    if self.df is None or self.df.empty:
        QMessageBox.warning(self, "No data", "Please load a file first.")
        return
    output_dir = getattr(self, 'current_output_dir', None) or os.path.dirname(
        getattr(self, 'file_path', '') or ''
    ) or os.getcwd()
    dlg = ExploratoryMatrixDialog(self.df, output_dir=output_dir, parent=self)
    dlg.exec_()




def _ap_open_range_selector(self):
    if getattr(self, "current_analysis_result", None) is not None:
        reply = QMessageBox.question(
            self, "Redefine Data Selection",
            "Redefining selection will clear existing results. Continue?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self._ap_reset_result_area()

    path = getattr(self, "file_path", None)
    if not path:
        QMessageBox.warning(self, "No file", "Load a file first.")
        return

    if path.lower().endswith(".csv"):
        df_raw = pd.read_csv(path, header=None, dtype=str)
        available_sheets, initial_sheet = None, None
    else:
        xf = pd.ExcelFile(path)
        available_sheets = xf.sheet_names
        initial_sheet = (
            self.auto_sheet_combo.currentText()
            if self.auto_sheet_combo.isEnabled()
            else (available_sheets[0] if available_sheets else None)
        )
        df_raw = pd.read_excel(path, sheet_name=initial_sheet, header=None, dtype=str)

    # Restore previous range selection (if any) so reopening the dialog
    # keeps prior group assignments instead of starting from scratch.
    prior = getattr(self, "_range_selection_metadata", None) or {}
    prior_selection_map = None
    prior_replicate_type = None
    prior_design_mode = None
    if prior:
        sels = prior.get("selections") or []
        if sels:
            prior_selection_map = {s["group"]: s.get("ranges", []) for s in sels}
            prior_replicate_type = sels[0].get("replicate_type")
        prior_design_mode = prior.get("design_mode")
        # If prior selection was made on a different sheet, reopen that sheet
        prior_sheet = prior.get("sheet")
        if prior_sheet and available_sheets and prior_sheet in available_sheets \
                and prior_sheet != initial_sheet:
            initial_sheet = prior_sheet
            df_raw = pd.read_excel(path, sheet_name=initial_sheet, header=None, dtype=str)

    dlg = SheetSelectionDialog(
        df_raw,
        initial_sheet=initial_sheet,
        available_sheets=available_sheets,
        source_path=path,
        parent=self,
        initial_selection_map=prior_selection_map,
        initial_replicate_type=prior_replicate_type,
        initial_design_mode=prior_design_mode,
    )
    if dlg.exec_() != QDialog.Accepted:
        return
    selection_map, replicate_type, sheet_name, design_mode = dlg.get_result()
    if not selection_map:
        return

    if sheet_name and sheet_name != initial_sheet and not path.lower().endswith(".csv"):
        df_raw = pd.read_excel(path, sheet_name=sheet_name, header=None, dtype=str)

    # Route to correct extractor based on design mode
    if design_mode == "paired":
        result_df, nan_report = extract_paired_from_coordinates(df_raw, selection_map)
        self.df = result_df
    elif design_mode == "bivariate":
        self.df = extract_bivariate_from_coordinates(df_raw, selection_map)
        nan_report = {}
    else:
        result_df, nan_report = extract_from_coordinates(
            df_raw, selection_map, replicate_type=replicate_type
        )
        self.df = result_df.drop(columns=["Source_Range", "n_replicates"], errors="ignore")

    self._range_design_mode = design_mode
    self._range_selection_metadata = {
        "source_file": path,
        "sheet": sheet_name,
        "design_mode": design_mode,
        "selections": [
            {"group": g, "ranges": r, "replicate_type": replicate_type}
            for g, r in selection_map.items()
        ],
    }

    total_nan = sum(nan_report.values()) if nan_report else 0
    if total_nan > 0:
        self.statusBar().showMessage(
            f"{total_nan} non-numeric value(s) dropped during import.", 8000
        )

    # Summary label — wording varies by design mode
    if hasattr(self, "_range_groups_label"):
        if design_mode == "bivariate":
            x_n = len(self.df["X"].dropna()) if "X" in self.df.columns else 0
            y_n = len(self.df["Y"].dropna()) if "Y" in self.df.columns else 0
            summary = f"X = {x_n} values  |  Y = {y_n} values"
            self._range_groups_label.setText(f"Imported variables — {summary}")
        elif design_mode == "paired":
            group_counts = []
            for g in selection_map:
                n = int((self.df["Group"] == g).sum()) if "Group" in self.df.columns else 0
                group_counts.append(f"{g}: n={n}")
            self._range_groups_label.setText(
                "Imported conditions — " + "  |  ".join(group_counts)
            )
        else:
            group_counts = []
            for g in selection_map:
                n = int((self.df["Group"] == g).sum()) if "Group" in self.df.columns else 0
                group_counts.append(f"{g}: n={n}")
            self._range_groups_label.setText(
                "Imported groups — " + "  |  ".join(group_counts)
            )
        self._range_groups_label.setVisible(True)

    self._wide_format_info = None
    self._ap_reset_result_area()
    self._refresh_preview_table()
    self._rebuild_column_cards()
    self._apply_mapping_heuristics()
    self._set_workflow_state("map", "Range selection imported — assign columns and run the analysis.")
    self.on_mapping_changed()


def _ap_reset_result_area(self):
    self.current_analysis_context = None
    self.current_analysis_result = None
    self.current_multi_results = {}
    self.result_cockpit.clear()
    self.decision_tree_panel.show_placeholder(
        "Map the columns, then run the auto-pilot analysis."
    )


class AutopilotMixin:
    """Bundles the autopilot pipeline methods for ``StatisticalAnalyzerApp``.

    Replaces the legacy ``attach_autopilot_methods`` runtime monkey-patch:
    binding methods at class-definition time restores MRO discoverability,
    static analysis (mypy/pyright), and ``super()`` chains. Module-level
    ``_ap_*`` functions are assigned as class attributes — Python treats them
    as unbound methods identically to the previous ``setattr`` path.
    """

    init_ui = _ap_init_ui
    browse_file = _ap_browse_file
    load_file = _ap_load_file
    load_sheet = _ap_load_sheet
    _refresh_preview_table = _ap_refresh_preview_table
    _rebuild_column_cards = _ap_rebuild_column_cards
    _apply_mapping_heuristics = _ap_apply_mapping_heuristics
    update_mode_constraints = _ap_update_mode_constraints
    on_mapping_changed = _ap_on_mapping_changed
    _set_workflow_state = _ap_set_workflow_state
    _is_binary_outcome_for_help = _ap_is_binary_outcome_for_help
    _is_continuous_factor1_for_help = _ap_is_continuous_factor1_for_help
    _resolve_help_recipe_for_bucket = _ap_resolve_help_recipe_for_bucket
    _ap_get_available_analysis_groups = _ap_get_available_analysis_groups
    _ap_update_analysis_group_selection_ui = _ap_update_analysis_group_selection_ui
    open_analysis_group_selector = _ap_open_analysis_group_selector
    _build_analysis_context = _ap_build_analysis_context
    _detected_test_label = _ap_detected_test_label
    _execute_single_analysis = _ap_execute_single_analysis
    _format_assumptions = _ap_format_assumptions
    _extract_normality_metric = _ap_extract_normality_metric
    _extract_variance_metric = _ap_extract_variance_metric
    _format_main_test_metric = _ap_format_main_test_metric
    _format_effect_size_metric = _ap_format_effect_size_metric
    _is_ttest_result = _ap_is_ttest_result
    _format_rationale = _ap_format_rationale
    _format_posthoc_status = _ap_format_posthoc_status
    _format_context_design = _ap_format_context_design
    _format_context_sample_overview = _ap_format_context_sample_overview
    _format_context_analysis_scope = _ap_format_context_analysis_scope
    _render_result_summary = _ap_render_result_summary
    _handle_blocked_result = _ap_handle_blocked_result
    determine_and_run_test = _ap_determine_and_run_test
    configure_plot_from_result = _ap_configure_plot_from_result
    open_current_output_folder = _ap_open_current_output_folder
    reset_application_state = _ap_reset_application_state
    _maybe_pivot = _ap_maybe_pivot
    open_exploratory_matrix_dialog = _ap_open_exploratory_matrix_dialog
    _ap_open_range_selector = _ap_open_range_selector
    _ap_reset_result_area = _ap_reset_result_area


def attach_autopilot_methods(app_cls):  # pragma: no cover — legacy shim
    """Deprecated. Use ``class App(AutopilotMixin, QMainWindow)`` instead."""
    import warnings
    warnings.warn(
        "attach_autopilot_methods() is deprecated; inherit from AutopilotMixin.",
        DeprecationWarning,
        stacklevel=2,
    )
    for name, attr in vars(AutopilotMixin).items():
        if name.startswith("__"):
            continue
        setattr(app_cls, name, attr)
