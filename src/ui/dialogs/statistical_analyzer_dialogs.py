import os
import sys

import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt, QTimer, QObject, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QTextBrowser,
    QTextEdit,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from analysis.stats_functions import OUTLIER_IMPORTS_AVAILABLE

import logging
logger = logging.getLogger(__name__)

try:
    from core.help_content import HELP_RECIPES, CATEGORY_ORDER
except ImportError as e:
    HELP_RECIPES = []
    CATEGORY_ORDER = []
    logger.info(f"Warning: help content not available: {e}")


def _configure_dialog(dialog, object_name=None, remove_context_help=True):
    """Apply common dialog defaults so all windows pick up the same QSS rules."""
    if object_name:
        dialog.setObjectName(object_name)
    if remove_context_help and isinstance(dialog, QDialog):
        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)

class GroupSelectionDialog(QDialog):
    """Dialog for selecting groups for a plot or analysis."""
    def __init__(self, available_groups, parent=None, window_title="Select Groups",
                 description="Select the groups to be displayed in the plot:"):
        if not available_groups:
            QMessageBox.critical(parent, "Error", "No groups available! Dialog will not open.")
            raise ValueError("No groups passed to GroupSelectionDialog.")
        super().__init__(parent)
        _configure_dialog(self, object_name="groupSelectionDialog")
        self.setWindowTitle(window_title)
        self.resize(300, 400)
        
        layout = QVBoxLayout(self)
        layout.setObjectName("lyoGroupSelection")
        
        # Explanation
        label = QLabel(description)
        label.setObjectName("lblGroupSelectionHelp")
        layout.addWidget(label)
        
        # Checkboxes for each group
        self.group_checks = {}
        group_container = QWidget()
        group_container.setObjectName("widGroupCheckboxes")
        group_layout = QVBoxLayout(group_container)
        group_layout.setObjectName("lyoGroupCheckboxes")
        
        for group in available_groups:
            check = QCheckBox(str(group))
            check.setObjectName(f"chkGroup_{str(group).replace(' ', '_')}")
            self.group_checks[group] = check
            group_layout.addWidget(check)
        
        layout.addWidget(group_container)
        
        # Select/Deselect All buttons
        button_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All")
        self.deselect_all_btn = QPushButton("Deselect All")
        self.select_all_btn.setObjectName("btnSelectAll")
        self.deselect_all_btn.setObjectName("btnDeselectAll")
        self.select_all_btn.clicked.connect(self._select_all_groups)
        self.deselect_all_btn.clicked.connect(self._deselect_all_groups)
        button_layout.addWidget(self.select_all_btn)
        button_layout.addWidget(self.deselect_all_btn)
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.setObjectName("btnDialogButtons")
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def _select_all_groups(self):
        """Select all group checkboxes"""
        for check in self.group_checks.values():
            check.setChecked(True)
    
    def _deselect_all_groups(self):
        """Deselect all group checkboxes"""
        for check in self.group_checks.values():
            check.setChecked(False)
    
    def get_selected_groups(self):
        selected = [group for group, check in self.group_checks.items() if check.isChecked()]
        if not selected:
            QMessageBox.warning(self, "Warning", "Please select at least one group!")
        return selected


class HelpHubDialog(QDialog):
    """Static in-app help hub with searchable analysis recipes."""

    def __init__(self, parent=None):
        super().__init__(parent)
        _configure_dialog(self, object_name="helpHubDialog")
        self.setWindowTitle("BioMedStatX Help Hub")
        self.setModal(False)
        self.resize(1120, 760)

        self._recipes = list(HELP_RECIPES)
        self._recipe_by_id = {recipe["id"]: recipe for recipe in self._recipes}
        self._current_recipe = None

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

        header = QLabel("Recipe-based Help: If you want X, map Y like this")
        header.setObjectName("helpHubHeader")
        root_layout.addWidget(header)

        body_splitter = QSplitter(Qt.Horizontal)
        body_splitter.setChildrenCollapsible(False)
        root_layout.addWidget(body_splitter, 1)

        nav_panel = QFrame()
        nav_panel.setObjectName("helpNavPanel")
        nav_layout = QVBoxLayout(nav_panel)
        nav_layout.setContentsMargins(10, 10, 10, 10)
        nav_layout.setSpacing(8)

        self.search_input = QLineEdit()
        self.search_input.setObjectName("helpNavSearch")
        self.search_input.setPlaceholderText("Search recipes...")
        self.search_input.textChanged.connect(self._filter_recipe_list)
        nav_layout.addWidget(self.search_input)

        self.recipe_list = QListWidget()
        self.recipe_list.setObjectName("helpNavList")
        self._skipping_header = False
        self.recipe_list.currentItemChanged.connect(self._skip_header_item)
        self.recipe_list.currentItemChanged.connect(self._update_recipe_view)
        nav_layout.addWidget(self.recipe_list, 1)

        body_splitter.addWidget(nav_panel)

        content_panel = QFrame()
        content_panel.setObjectName("helpContentPanel")
        content_layout = QVBoxLayout(content_panel)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(8)

        self.recipe_title = QLabel("")
        self.recipe_title.setObjectName("helpRecipeTitle")
        content_layout.addWidget(self.recipe_title)

        self.recipe_browser = QTextBrowser()
        self.recipe_browser.setObjectName("helpRecipeBrowser")
        self.recipe_browser.setOpenExternalLinks(True)
        self.recipe_browser.document().setDefaultStyleSheet("""
            body { line-height: 1.45; }
            h3 { margin-top: 14px; color: #16313a; }
            table {
                border-collapse: collapse;
                width: 100%;
                margin: 8px 0 12px 0;
                display: block;
                overflow-x: auto;
            }
            th, td {
                border: 1px solid #cdd9e0;
                padding: 6px 8px;
                text-align: left;
                white-space: nowrap;
            }
            th { background: #eaf1f6; font-weight: 700; }
            tr:nth-child(even) td { background: #f8fbfd; }
            table:nth-of-type(1) th { background: #e4f4ec; }
            table:nth-of-type(1) { border-left: 4px solid #1f7a5a; }
            table:nth-of-type(2) th { background: #f8e7e7; }
            table:nth-of-type(2) { border-left: 4px solid #9f3a38; }
            .badge {
                display: inline-block;
                margin-left: 8px;
                padding: 1px 8px;
                border-radius: 999px;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 0.2px;
            }
            .badge-good {
                color: #0f5132;
                background: #d8f3e5;
                border: 1px solid #9fd9bc;
            }
            .badge-bad {
                color: #7d2e2c;
                background: #fbe3e2;
                border: 1px solid #efb1ad;
            }
        """)
        content_layout.addWidget(self.recipe_browser, 1)

        actions_layout = QHBoxLayout()
        actions_layout.addStretch()

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        actions_layout.addWidget(close_button)

        content_layout.addLayout(actions_layout)

        body_splitter.addWidget(content_panel)
        body_splitter.setStretchFactor(0, 0)
        body_splitter.setStretchFactor(1, 1)
        body_splitter.setSizes([320, 780])

        self._populate_recipe_list()

    def navigate_to(self, recipe_id):
        """Select and display a specific recipe by its ID."""
        self.search_input.clear()  # Clear search so all items are visible
        for index in range(self.recipe_list.count()):
            item = self.recipe_list.item(index)
            if item.data(Qt.UserRole) == recipe_id:
                self.recipe_list.setCurrentItem(item)
                break

    def _skip_header_item(self, current, previous):
        """If keyboard navigation lands on a category header, jump to the nearest recipe."""
        if self._skipping_header or current is None:
            return
        if current.data(Qt.UserRole) is not None:
            return  # already on a real recipe — nothing to do
        self._skipping_header = True
        try:
            current_row = self.recipe_list.row(current)
            previous_row = self.recipe_list.row(previous) if previous is not None else -1
            # Determine direction: moving down (or no prior) → search forward; moving up → search backward
            moving_down = previous_row < current_row
            count = self.recipe_list.count()
            if moving_down:
                search_range = range(current_row + 1, count)
            else:
                search_range = range(current_row - 1, -1, -1)
            for idx in search_range:
                candidate = self.recipe_list.item(idx)
                if candidate is not None and not candidate.isHidden() and candidate.data(Qt.UserRole) is not None:
                    self.recipe_list.setCurrentItem(candidate)
                    return
            # No candidate found in that direction: prefer staying on the previous recipe.
            if previous is not None and not previous.isHidden() and previous.data(Qt.UserRole) is not None:
                self.recipe_list.setCurrentItem(previous)
                return
            # Boundary case (e.g. Up from the first recipe lands on the header above with no
            # usable previous): fall back to the first visible non-header recipe so selection
            # never gets stuck on a header.
            for idx in range(self.recipe_list.count()):
                candidate = self.recipe_list.item(idx)
                if candidate is not None and not candidate.isHidden() and candidate.data(Qt.UserRole) is not None:
                    self.recipe_list.setCurrentItem(candidate)
                    return
        finally:
            self._skipping_header = False

    def _populate_recipe_list(self):
        self.recipe_list.clear()
        by_category = {cat: [] for cat in CATEGORY_ORDER}
        for recipe in self._recipes:
            by_category.setdefault(recipe.get("category", CATEGORY_ORDER[-1]), []).append(recipe)

        first_recipe_row = None
        for category in CATEGORY_ORDER:
            recipes = by_category.get(category) or []
            if not recipes:
                continue
            header = QListWidgetItem(category)
            header.setData(Qt.UserRole, None)
            header.setFlags(Qt.ItemIsEnabled)  # enabled for display, NOT selectable
            font = header.font()
            font.setBold(True)
            header.setFont(font)
            self.recipe_list.addItem(header)
            for recipe in recipes:
                item = QListWidgetItem(recipe["title"])
                item.setData(Qt.UserRole, recipe["id"])
                item.setToolTip(recipe.get("summary", ""))
                self.recipe_list.addItem(item)
                if first_recipe_row is None:
                    first_recipe_row = self.recipe_list.row(item)

        if first_recipe_row is not None:
            self.recipe_list.setCurrentRow(first_recipe_row)

    def _filter_recipe_list(self, text):
        query = str(text or "").strip().lower()
        first_visible = None

        for index in range(self.recipe_list.count()):
            item = self.recipe_list.item(index)
            recipe_id = item.data(Qt.UserRole)
            if recipe_id is None:
                continue  # category header: handled in second pass below
            recipe = self._recipe_by_id.get(recipe_id, {})
            haystack = " ".join([
                recipe.get("title", ""),
                recipe.get("summary", ""),
                " ".join(recipe.get("keywords", [])),
            ]).lower()
            visible = not query or query in haystack
            item.setHidden(not visible)
            if visible and first_visible is None:
                first_visible = item

        # Hide category headers whose recipes are all hidden. This pass must run even for an
        # empty query: with no query every recipe is visible, so each header's forward scan
        # finds a visible recipe and un-hides the header again. Skipping this pass when the
        # query is empty would leave headers hidden after a search is cleared.
        count = self.recipe_list.count()
        for index in range(count):
            item = self.recipe_list.item(index)
            if item.data(Qt.UserRole) is not None:
                continue
            any_visible = False
            for j in range(index + 1, count):
                nxt = self.recipe_list.item(j)
                if nxt.data(Qt.UserRole) is None:
                    break
                if not nxt.isHidden():
                    any_visible = True
                    break
            item.setHidden(not any_visible)

        current = self.recipe_list.currentItem()
        if current is None or current.isHidden():
            if first_visible is not None:
                self.recipe_list.setCurrentItem(first_visible)
            else:
                self._current_recipe = None
                self.recipe_title.setText("No matching recipe")
                self.recipe_browser.setHtml("<p>No recipe matches your search query.</p>")

    def _update_recipe_view(self, current, _previous):
        if current is None:
            return
        if current.data(Qt.UserRole) is None:
            return  # header row, nothing to render

        recipe_id = current.data(Qt.UserRole)
        recipe = self._recipe_by_id.get(recipe_id)
        if not recipe:
            self._current_recipe = None
            self.recipe_title.setText("Recipe not found")
            self.recipe_browser.setHtml("<p>Recipe content is unavailable.</p>")
            return

        self._current_recipe = recipe
        self.recipe_title.setText(recipe.get("title", ""))
        self.recipe_browser.setHtml(recipe.get("html", "<p>No content available.</p>"))


        

class OutlierDetectionDialog(QDialog):
    def __init__(self, df, parent=None):
        super().__init__(parent)
        self.df = df
        _configure_dialog(self, object_name="outlierDetectionDialog")
        self.setWindowTitle("Outlier Detection")
        self.setup_ui()
                
    def setup_ui(self):
        """Set up the dialog UI"""
        layout = QVBoxLayout(self)
        
        # Check if outlier detection is available
        if not OUTLIER_IMPORTS_AVAILABLE:
            warning_label = QLabel("Warning: Outlier detection is not available.\n"
                                "Please install required packages: outliers, pingouin, openpyxl")
            warning_label.setObjectName("lblCriticalWarning")
            layout.addWidget(warning_label)
            
            button_box = QDialogButtonBox(QDialogButtonBox.Ok)
            button_box.accepted.connect(self.reject)
            layout.addWidget(button_box)
            return
        
        # Info section
        info_label = QLabel("Select data columns and parameters for outlier detection:")
        info_label.setObjectName("lblSectionHeading")
        layout.addWidget(info_label)
        
        # Data selection section
        data_group = QGroupBox("Data Selection")
        data_layout = QFormLayout(data_group)
        
        # Group column selection
        self.group_col_combo = QComboBox()
        self.group_col_combo.addItems(self.df.columns)
        data_layout.addRow("Group Column:", self.group_col_combo)
        
        # Analysis mode selection
        mode_group = QGroupBox("Analysis Mode")
        mode_layout = QVBoxLayout(mode_group)
        
        self.single_dataset_radio = QRadioButton("Single dataset analysis")
        self.multi_dataset_radio = QRadioButton("Multi-dataset analysis")
        self.single_dataset_radio.setChecked(True)
        
        mode_layout.addWidget(self.single_dataset_radio)
        mode_layout.addWidget(self.multi_dataset_radio)
        
        # Connect radio buttons to update UI
        self.single_dataset_radio.toggled.connect(self.update_dataset_selection)
        self.multi_dataset_radio.toggled.connect(self.update_dataset_selection)
        
        layout.addWidget(mode_group)
        
        # Single dataset selection
        self.single_dataset_group = QGroupBox("Single Dataset")
        single_layout = QFormLayout(self.single_dataset_group)
        
        self.value_col_combo = QComboBox()
        numeric_columns = [col for col in self.df.columns if pd.api.types.is_numeric_dtype(self.df[col])]
        self.value_col_combo.addItems(numeric_columns)
        single_layout.addRow("Value Column:", self.value_col_combo)
        
        layout.addWidget(self.single_dataset_group)
        
        # Multi-dataset selection
        self.multi_dataset_group = QGroupBox("Multiple Datasets")
        multi_layout = QVBoxLayout(self.multi_dataset_group)
        
        multi_info = QLabel("Select all value columns to analyze for outliers:")
        multi_layout.addWidget(multi_info)
        
        # Scrollable list for dataset selection
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        self.dataset_checkboxes = {}
        # REMOVED: CNRQ-specific filtering - now shows ALL numeric columns
        for col in numeric_columns:
            checkbox = QCheckBox(col)
            checkbox.setChecked(False)  # REMOVED: Auto-select based on CNRQ
            self.dataset_checkboxes[col] = checkbox
            scroll_layout.addWidget(checkbox)
        
        scroll_area.setWidget(scroll_widget)
        scroll_area.setMaximumHeight(150)
        multi_layout.addWidget(scroll_area)
        
        # SIMPLIFIED: Select all/none buttons only (removed CNRQ button)
        button_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_none_btn = QPushButton("Select None")
        
        select_all_btn.clicked.connect(lambda: self.set_all_datasets(True))
        select_none_btn.clicked.connect(lambda: self.set_all_datasets(False))
        # REMOVED: select_cnrq_btn and its connection
        
        button_layout.addWidget(select_all_btn)
        button_layout.addWidget(select_none_btn)
        # REMOVED: button_layout.addWidget(select_cnrq_btn)
        multi_layout.addLayout(button_layout)
        
        layout.addWidget(self.multi_dataset_group)
        layout.addWidget(data_group)
        
        # Test parameters section (unchanged)
        params_group = QGroupBox("Test Parameters")
        params_layout = QVBoxLayout(params_group)
        
        # Test mode
        mode_layout = QVBoxLayout()
        mode_label = QLabel("Test Mode:")
        mode_label.setObjectName("lblSectionHeading")
        mode_layout.addWidget(mode_label)
        
        self.single_mode = QRadioButton("Single-pass detection (detect one outlier per test)")
        self.iterative_mode = QRadioButton("Iterative detection (remove outliers until none found)")
        self.iterative_mode.setChecked(True)  # Default to iterative
        
        mode_layout.addWidget(self.single_mode)
        mode_layout.addWidget(self.iterative_mode)
        
        params_layout.addLayout(mode_layout)
        
        # Test types
        test_layout = QVBoxLayout()
        test_label = QLabel("Tests to Perform:")
        test_label.setObjectName("lblSectionHeading")
        test_layout.addWidget(test_label)

        self.modz_check = QCheckBox("Modified Z-Score Test (robust detection using median)")
        self.grubbs_check = QCheckBox("Grubbs' Test (for normally distributed data)")
        # Grubbs is the default: it holds its nominal ~5% false-positive rate
        # across n=3..30, whereas the Modified Z-Score's median/MAD scale is
        # unstable at small n and flags a phantom outlier in ~29% of clean n=3
        # samples (the common triplicate size). ModZ stays available as a
        # deliberate choice, but is no longer the default.
        self.modz_check.setChecked(False)
        self.grubbs_check.setChecked(True)

        test_layout.addWidget(self.modz_check)
        test_layout.addWidget(self.grubbs_check)
        
        params_layout.addLayout(test_layout)
        
        layout.addWidget(params_group)
        
        # Output section
        output_group = QGroupBox("Output Options")
        output_layout = QVBoxLayout(output_group)
        
        # File path selection
        file_layout = QHBoxLayout()
        file_label = QLabel("Output File:")
        self.file_path_label = QLabel("outlier_analysis_results.html")
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_output_file)
        
        file_layout.addWidget(file_label)
        file_layout.addWidget(self.file_path_label, 1)
        file_layout.addWidget(browse_button)
        
        output_layout.addLayout(file_layout)
        layout.addWidget(output_group)
        
        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        # Initial UI update
        self.update_dataset_selection()
        
    def update_dataset_selection(self):
        """Update UI based on selected analysis mode"""
        is_single = self.single_dataset_radio.isChecked()
        self.single_dataset_group.setVisible(is_single)
        self.multi_dataset_group.setVisible(not is_single)
        
    def set_all_datasets(self, checked):
        """Select or deselect all dataset checkboxes"""
        for checkbox in self.dataset_checkboxes.values():
            checkbox.setChecked(checked)
    
    def select_cnrq_datasets(self):
        """Select only CNRQ columns"""
        for col, checkbox in self.dataset_checkboxes.items():
            checkbox.setChecked('CNRQ' in col.upper())
    
    def browse_output_file(self):
        """Browse for output file location"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Outlier Analysis Results",
            "outlier_analysis_results.html",
            "HTML Report (*.html);;All Files (*)"
        )
        if file_path:
            if not file_path.lower().endswith(".html"):
                file_path += ".html"
            self.file_path_label.setText(file_path)
    
    def get_config(self):
        """Get the configuration from the dialog"""
        if not OUTLIER_IMPORTS_AVAILABLE:
            return None
            
        config = {
            'group_column': self.group_col_combo.currentText(),
            'iterate': self.iterative_mode.isChecked(),
            'run_modz': self.modz_check.isChecked(),
            'run_grubbs': self.grubbs_check.isChecked(),
            'output_file': self.file_path_label.text(),
            'is_multi_dataset': self.multi_dataset_radio.isChecked()
        }
        
        if config['is_multi_dataset']:
            # Get selected datasets
            selected_datasets = [col for col, checkbox in self.dataset_checkboxes.items() 
                            if checkbox.isChecked()]
            if not selected_datasets:
                QMessageBox.warning(self, "No Datasets Selected", 
                                "Please select at least one dataset column for analysis.")
                return None
            config['dataset_columns'] = selected_datasets
        else:
            config['value_column'] = self.value_col_combo.currentText()
            
        return config


import numpy as np


# ---------------------------------------------------------------------------
# Exploratory Correlation Matrix Dialog
# ---------------------------------------------------------------------------

class ExploratoryMatrixDialog(QDialog):
    """Dialog for running an exploratory pairwise correlation matrix.

    Lets the user select variables, choose correlation method and missing-data
    handling, apply multiple-testing correction, and optionally stratify by a
    categorical column.  Results are exported as an HTML report.
    """

    def __init__(self, df, output_dir=None, parent=None):
        super().__init__(parent)
        self.df = df
        self.output_dir = output_dir or os.getcwd()
        _configure_dialog(self, object_name="exploratoryMatrixDialog")
        self.setWindowTitle("Exploratory Correlation Matrix")
        self.setMinimumWidth(520)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Variable selection
        var_label = QLabel("Select variables (numeric columns):")
        var_label.setObjectName("panelDescription")
        layout.addWidget(var_label)

        numeric_cols = [c for c in self.df.columns
                        if pd.api.types.is_numeric_dtype(self.df[c])]
        self._var_list = QListWidget()
        self._var_list.setSelectionMode(QListWidget.MultiSelection)
        self._var_list.setMaximumHeight(200)
        for col in numeric_cols:
            self._var_list.addItem(col)
        # Select all by default
        for i in range(self._var_list.count()):
            self._var_list.item(i).setSelected(True)
        layout.addWidget(self._var_list)

        # Method
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        from PyQt5.QtWidgets import QComboBox
        self._method_combo = QComboBox()
        self._method_combo.addItems(["spearman", "pearson", "auto"])
        method_row.addWidget(self._method_combo)
        layout.addLayout(method_row)

        # Missing data handling
        missing_row = QHBoxLayout()
        missing_row.addWidget(QLabel("Missing values:"))
        from PyQt5.QtWidgets import QRadioButton, QButtonGroup
        self._pairwise_radio = QRadioButton("Pairwise Deletion (recommended)")
        self._listwise_radio = QRadioButton("Listwise Deletion")
        self._pairwise_radio.setChecked(True)
        missing_grp = QButtonGroup(self)
        missing_grp.addButton(self._pairwise_radio)
        missing_grp.addButton(self._listwise_radio)
        missing_row.addWidget(self._pairwise_radio)
        missing_row.addWidget(self._listwise_radio)
        layout.addLayout(missing_row)

        # Correction
        corr_row = QHBoxLayout()
        corr_row.addWidget(QLabel("Multiple testing correction:"))
        self._corr_combo = QComboBox()
        self._corr_combo.addItems(["fdr_bh (Benjamini-Hochberg)", "bonferroni", "none"])
        corr_row.addWidget(self._corr_combo)
        layout.addLayout(corr_row)

        # Stratify
        strat_row = QHBoxLayout()
        strat_row.addWidget(QLabel("Stratify by (optional):"))
        self._strat_combo = QComboBox()
        cat_cols = ["— none —"] + [
            c for c in self.df.columns
            if not pd.api.types.is_numeric_dtype(self.df[c])
            or self.df[c].nunique() <= 10
        ]
        self._strat_combo.addItems(cat_cols)
        strat_row.addWidget(self._strat_combo)
        layout.addLayout(strat_row)

        # Buttons
        btn_row = QHBoxLayout()
        run_btn = QPushButton("Start analysis")
        run_btn.setObjectName("primaryButton")
        run_btn.clicked.connect(self._run)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(run_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

    def _run(self):
        selected_items = [item.text() for item in self._var_list.selectedItems()]
        if len(selected_items) < 2:
            self._status_label.setText("Please select at least 2 variables.")
            return

        method = self._method_combo.currentText()
        pairwise = self._pairwise_radio.isChecked()

        corr_raw = self._corr_combo.currentText()
        if corr_raw.startswith("fdr_bh"):
            correction = "fdr_bh"
        elif corr_raw == "bonferroni":
            correction = "bonferroni"
        else:
            correction = None

        strat_raw = self._strat_combo.currentText()
        stratify_by = None if strat_raw == "— none —" else strat_raw

        try:
            from analysis.correlation_models import ExploratoryCorrelationMatrix
            matrix_model = ExploratoryCorrelationMatrix()
            matrix_model.fit(self.df, selected_items, method=method,
                             correction=correction, pairwise=pairwise,
                             stratify_by=stratify_by)
            results = matrix_model.as_results_dict()

            # Export as HTML report
            import os
            from export.export_dispatcher import ExportDispatcher
            out_file = os.path.join(self.output_dir,
                                    "exploratory_correlation_matrix.html")
            results["pairwise_comparisons"] = []
            export_result = ExportDispatcher.export_analysis_results(results, out_file)
            if export_result.get("warning"):
                logger.warning(f"WARNING: {export_result['warning']}")

            self._status_label.setText(
                f"Done! Saved to:\n{out_file}"
            )
        except Exception as exc:
            self._status_label.setText(f"Error: {exc}")

