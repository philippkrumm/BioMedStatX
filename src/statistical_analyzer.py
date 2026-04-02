import sys
import time
import os
if sys.platform == "darwin":
    # Keep Qt environment clean so PyQt uses one runtime only.
    os.environ.pop("DYLD_FRAMEWORK_PATH", None)
    os.environ.pop("DYLD_LIBRARY_PATH", None)
    os.environ.pop("QT_PLUGIN_PATH", None)
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
from PyQt5.QtWidgets import QDesktopWidget
# Core imports - always needed
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QWidget, QVBoxLayout,
                           QHBoxLayout, QLabel, QComboBox, QPushButton, QListWidget, 
                           QTabWidget, QGroupBox, QCheckBox, QSpinBox, QColorDialog, 
                           QMessageBox, QScrollArea, QListWidgetItem, QDialog, QDialogButtonBox,
                           QGridLayout, QLineEdit, QRadioButton, QAction, QFormLayout, QAbstractItemView, QDoubleSpinBox, QButtonGroup,
                           QFrame, QTableWidget, QTableWidgetItem, QSplitter, QToolButton, QGraphicsOpacityEffect, QGraphicsDropShadowEffect, QSizePolicy, QTextEdit, QTextBrowser)
from PyQt5.QtGui import QColor, QIcon, QPixmap, QDrag, QDesktopServices
from PyQt5.QtCore import Qt, QMimeData, QPoint, pyqtSignal, QObject, QPropertyAnimation, QEasingCurve, QSequentialAnimationGroup, QTimer, QUrl

# Initialize lazy loading system
from lazy_imports import preload_critical_modules, get_matplotlib_pyplot as get_matplotlib
preload_critical_modules()

from stats_functions import (
    DataImporter, AnalysisManager, 
    UIDialogManager, OutlierDetector, OUTLIER_IMPORTS_AVAILABLE
)
from export_dispatcher import ExportDispatcher
from datavisualizer import DataVisualizer
from decisiontreevisualizer import DecisionTreeVisualizer
from statisticaltester import StatisticalTester
# Import updater for auto-update functionality
try:
    from updater import AutoUpdater
    UPDATE_AVAILABLE = True
except ImportError:
    UPDATE_AVAILABLE = False
    print("Warning: Updater module not available")
# Import the new PlotAestheticsDialog for advanced plot appearance configuration
try:
    from plot_aesthetics_dialog import PlotAestheticsDialog
    PLOT_MODULES_AVAILABLE = True
    print(f"SUCCESS: Imported PlotAestheticsDialog from plot_aesthetics_dialog.py")
    print(f"DEBUG: PlotAestheticsDialog class: {PlotAestheticsDialog}")
except ImportError as e:
    print(f"WARNING: Could not import new plot modules: {e}")
    PlotAestheticsDialog = None
    PLOT_MODULES_AVAILABLE = False

try:
    from help_content import HELP_RECIPES
except ImportError as e:
    HELP_RECIPES = []
    print(f"Warning: help content not available: {e}")

import traceback
print(f"DEBUG: RUNNING FILE VERSION FROM {time.time()} - {os.path.abspath(__file__)}")

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # When running from Python directly, get the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to project root
        base_path = os.path.dirname(script_dir)
    
    return os.path.join(base_path, relative_path)


def _apply_elevation(widget, radius=18, x_offset=0, y_offset=4, opacity=0.18):
    """Apply a drop shadow to give a widget visual elevation. QSS cannot do this."""
    shadow = QGraphicsDropShadowEffect(widget)
    shadow.setBlurRadius(radius)
    shadow.setOffset(x_offset, y_offset)
    shadow.setColor(QColor(0, 0, 0, int(255 * opacity)))
    widget.setGraphicsEffect(shadow)
    
def _configure_dialog(dialog, object_name=None, remove_context_help=True):
    """Apply common dialog defaults so all windows pick up the same QSS rules."""
    if object_name:
        dialog.setObjectName(object_name)
    if remove_context_help and isinstance(dialog, QDialog):
        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)


DEFAULT_COLORS = ['#0f766e', '#1f7a5a', '#b7791f', '#9f3a38', '#1d4ed8', '#7c3aed']  # Teal, DarkGreen, Amber, DuskyRed, Indigo, Violet
DEFAULT_HATCHES = ['/', '\\', '|', '-', '+', 'x', 'o', '.', '*', '']

def dict_to_long_format(samples, groups):
    """
    Converts a dictionary with groups and measurements into a DataFrame in long format.
    
    Parameters:
    -----------
    samples : dict
        Dictionary with group names as keys and lists of measurements as values
    groups : list
        List of groups to analyze
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame in long format with columns 'subject', 'group', 'value'
    """
    data = []
    for group in groups:
        if group in samples:
            values = samples[group]
            for i, value in enumerate(values):
                data.append({
                    'subject': i,  # Use a unique identifier here
                    'group': group,
                    'value': value
                })
    return pd.DataFrame(data)

def no_transform(df, dv):
    """No transformation - returns the DataFrame unchanged."""
    return df

def log_transform(df, dv):
    """Logarithmic transformation of the dependent variable."""
    df2 = df.copy()
    # Min + 1 to avoid negative or zero values
    min_val = df2[dv].min()
    offset = abs(min_val) + 1 if min_val <= 0 else 0
    df2[dv] = np.log(df2[dv] + offset)
    return df2

def boxcox_transform(df, dv):
    """Box-Cox transformation of the dependent variable."""
    from scipy import stats
    df2 = df.copy()
    # For Box-Cox, all values must be positive
    min_val = df2[dv].min()
    offset = abs(min_val) + 1 if min_val <= 0 else 0
    
    # Perform Box-Cox transformation
    try:
        transformed_data, lambda_val = stats.boxcox(df2[dv] + offset)
        df2[dv] = transformed_data
        print(f"Box-Cox transformation performed with Lambda={lambda_val:.4f}")
    except Exception as e:
        print(f"Box-Cox transformation failed: {str(e)}")
        # Fallback to log transformation
        df2[dv] = np.log(df2[dv] + offset)
        print("Fallback: Logarithmic transformation was used instead")
    
    return df2

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

        self.copy_button = QPushButton("Copy Example Data")
        self.copy_button.setObjectName("btnCopyExampleData")
        self.copy_button.clicked.connect(self._copy_current_example_data)
        actions_layout.addWidget(self.copy_button)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        actions_layout.addWidget(close_button)

        content_layout.addLayout(actions_layout)

        self.copy_feedback = QLabel("")
        self.copy_feedback.setObjectName("helpCopyFeedback")
        content_layout.addWidget(self.copy_feedback)

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

    def _populate_recipe_list(self):
        self.recipe_list.clear()
        for recipe in self._recipes:
            item = QListWidgetItem(recipe["title"])
            item.setData(Qt.UserRole, recipe["id"])
            item.setToolTip(recipe.get("summary", ""))
            self.recipe_list.addItem(item)

        if self.recipe_list.count():
            self.recipe_list.setCurrentRow(0)

    def _filter_recipe_list(self, text):
        query = str(text or "").strip().lower()
        first_visible = None

        for index in range(self.recipe_list.count()):
            item = self.recipe_list.item(index)
            recipe_id = item.data(Qt.UserRole)
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

        current = self.recipe_list.currentItem()
        if current is None or current.isHidden():
            if first_visible is not None:
                self.recipe_list.setCurrentItem(first_visible)
            else:
                self._current_recipe = None
                self.recipe_title.setText("No matching recipe")
                self.recipe_browser.setHtml("<p>No recipe matches your search query.</p>")
                self.copy_button.setEnabled(False)

    def _update_recipe_view(self, current, _previous):
        if current is None:
            return

        recipe_id = current.data(Qt.UserRole)
        recipe = self._recipe_by_id.get(recipe_id)
        if not recipe:
            self._current_recipe = None
            self.recipe_title.setText("Recipe not found")
            self.recipe_browser.setHtml("<p>Recipe content is unavailable.</p>")
            self.copy_button.setEnabled(False)
            return

        self._current_recipe = recipe
        self.recipe_title.setText(recipe.get("title", ""))
        self.recipe_browser.setHtml(recipe.get("html", "<p>No content available.</p>"))

        has_example = bool(recipe.get("example_tsv"))
        self.copy_button.setEnabled(has_example)
        self.copy_feedback.setText("")

    def _copy_current_example_data(self):
        if not self._current_recipe:
            return

        sample_data = self._current_recipe.get("example_tsv", "")
        if not sample_data:
            return

        QApplication.clipboard().setText(sample_data)
        self.copy_feedback.setText("Example data copied to clipboard (tab-separated, Excel-ready).")
        QTimer.singleShot(2500, lambda: self.copy_feedback.setText(""))


class ColumnSelectionDialog(QDialog):
    """Dialog for selecting measurement columns for a dataset"""
    def __init__(self, available_columns, parent=None):
        if not available_columns:
            QMessageBox.critical(parent, "Error", "No measurement columns available! Dialog will not open.")
            raise ValueError("No measurement columns passed to ColumnSelectionDialog.")
        super().__init__(parent)
        _configure_dialog(self, object_name="columnSelectionDialog")
        self.setWindowTitle("Select Measurement Columns")
        self.resize(400, 500)
        
        layout = QVBoxLayout(self)
        layout.setObjectName("lyoColumnSelection")
        
        # Explanation
        label = QLabel("Select the columns to be used for analysis:")
        layout.addWidget(label)
        
        # NEW OPTION: Multi-dataset analysis
        self.multi_dataset_check = QCheckBox("Separate analysis per dataset with shared Excel file")
        self.multi_dataset_check.setToolTip("Analyzes each dataset separately, but combines all results in a shared Excel file")
        layout.addWidget(self.multi_dataset_check)
        
        # Checkboxes for each column
        scroll_area = QScrollArea()
        scroll_area.setObjectName("scrollColumns")
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_content.setObjectName("widColumnContainer")
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setObjectName("lyoColumnCheckboxes")
        
        self.column_checks = {}
        for column in available_columns:
            check = QCheckBox(str(column))
            check.setObjectName(f"chkColumn_{str(column).replace(' ', '_')}")
            self.column_checks[column] = check
            scroll_layout.addWidget(check)
        
        scroll_area.setWidget(scroll_content)
        # Limit height for many columns
        scroll_area.setMaximumHeight(300)
        layout.addWidget(scroll_area)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.setObjectName("btnDialogButtons")
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def get_selected_columns(self):
        selected = [column for column, check in self.column_checks.items() if check.isChecked()]
        if not selected:
            QMessageBox.warning(self, "Warning", "Please select at least one measurement column!")
        return {
            "columns": selected,
            "multi_dataset": self.multi_dataset_check.isChecked(),
            "combine": False
        }


class PairwiseComparisonDialog(QDialog):
    """Dialog for selecting groups for pairwise comparisons"""
    def __init__(self, available_groups, parent=None):
        if not available_groups or len(available_groups) < 2:
            QMessageBox.critical(parent, "Error", "At least 2 groups are required for pairwise comparisons!")
            raise ValueError("Too few groups passed to PairwiseComparisonDialog.")
        super().__init__(parent)
        _configure_dialog(self, object_name="pairwiseComparisonDialog")
        self.setWindowTitle("Pairwise Comparisons")
        self.resize(400, 300)

        main_layout = QVBoxLayout(self)
        main_layout.setObjectName("lyoPairwiseComparison")

        # --- Scrollable content widget ---
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)

        # Explanation
        label = QLabel("Select two groups between which a significance line should be displayed:")
        label.setObjectName("lblComparisonHelp")
        content_layout.addWidget(label)

        # Selection for group 1
        group1_layout = QHBoxLayout()
        group1_layout.setObjectName("lyoGroup1Selection")
        group1_label = QLabel("Group 1:")
        group1_label.setObjectName("lblGroup1")
        group1_layout.addWidget(group1_label)
        self.group1_combo = QComboBox()
        self.group1_combo.setObjectName("cboGroup1")
        self.group1_combo.addItems([str(g) for g in available_groups])
        group1_layout.addWidget(self.group1_combo)
        content_layout.addLayout(group1_layout)

        # Selection for group 2
        group2_layout = QHBoxLayout()
        group2_layout.setObjectName("lyoGroup2Selection")
        group2_label = QLabel("Group 2:")
        group2_label.setObjectName("lblGroup2")
        group2_layout.addWidget(group2_label)
        self.group2_combo = QComboBox()
        self.group2_combo.setObjectName("cboGroup2")
        self.group2_combo.addItems([str(g) for g in available_groups])
        if len(available_groups) > 1:
            self.group2_combo.setCurrentIndex(1)
        group2_layout.addWidget(self.group2_combo)
        content_layout.addLayout(group2_layout)

        # Hint text for explanation
        hint_label = QLabel("Note: Significance is automatically taken from the post-hoc tests.")
        hint_label.setObjectName("lblSignificanceHint")
        content_layout.addWidget(hint_label)

        # Dependent samples with better description
        dependent_layout = QHBoxLayout()
        dependent_layout.setObjectName("lyoDependentOption")

        self.dependent_check = QCheckBox("Dependent samples (paired test)")
        self.dependent_check.setObjectName("chkDependentSamples")
        dependent_layout.addWidget(self.dependent_check)

        # Info button
        dependent_info = QPushButton("?")
        dependent_info.setObjectName("btnPairwiseDependentInfo")
        dependent_info.setMaximumWidth(20)
        dependent_info.clicked.connect(self.show_dependent_info)
        dependent_layout.addWidget(dependent_info)
        dependent_layout.addStretch()

        content_layout.addLayout(dependent_layout)

        # --- QScrollArea setup ---
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)

        # --- Dialog buttons (not inside scroll area) ---
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.setObjectName("btnDialogButtons")
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)
    
    def get_comparison(self):
        g1 = self.group1_combo.currentText()
        g2 = self.group2_combo.currentText()
        if g1 == g2:
            QMessageBox.warning(self, "Warning", "The two groups must be different!")
            return None
        return {
            'group1': g1,
            'group2': g2,
            'dependent': self.dependent_check.isChecked()
        }
    def show_dependent_info(self):
        QMessageBox.information(
            self, "Dependent samples for pairwise comparisons",
            "Select this option if the groups to be compared are dependent samples "
            "(e.g. measurements on the same subject at different time points).\n\n"
            "For dependent samples, a paired t-test (parametric) or "
            "a Wilcoxon signed-rank test (non-parametric) is performed.\n\n"
            "Note: The groups must have the same number of measurements and "
            "the order of measurements must match."
        )
        
class TwoWayAnovaDialog(QDialog):
    """Dialog for configuring a Two-Way ANOVA"""
    def __init__(self, groups, parent=None):
        if not groups or len(groups) < 2:
            QMessageBox.critical(parent, "Error", "At least 2 groups are required for a Two-Way ANOVA!")
            raise ValueError("Too few groups passed to TwoWayAnovaDialog.")
        super().__init__(parent)
        _configure_dialog(self, object_name="twoWayAnovaDialog")
        self.setWindowTitle("Configure Two-Way ANOVA")
        self.resize(500, 400)
        self.groups = groups
        
        layout = QVBoxLayout(self)
        layout.setObjectName("lyoTwoWayAnova")
        
        # Explanation
        label = QLabel("Define additional factors for the Two-Way ANOVA:")
        label.setObjectName("lblTwoWayAnovaHelp")
        layout.addWidget(label)
        
        # Factor definition
        factor_group = QGroupBox("Factor definition")
        factor_group.setObjectName("grpFactorDefinition")
        factor_layout = QGridLayout(factor_group)
        factor_layout.setObjectName("lyoFactorDefinition")
        
        # Factor name
        factor_label = QLabel("Factor name:")
        factor_label.setObjectName("lblFactorName")
        factor_layout.addWidget(factor_label, 0, 0)
        
        self.factor_name = QLineEdit()
        self.factor_name.setObjectName("edtFactorName")
        self.factor_name.setPlaceholderText("e.g. Treatment, Gender, etc.")
        factor_layout.addWidget(self.factor_name, 0, 1)
        
        layout.addWidget(factor_group)
        
        # Factor values per group
        value_group = QGroupBox("Factor values per group")
        value_group.setObjectName("grpFactorValues")
        value_layout = QGridLayout(value_group)
        value_layout.setObjectName("lyoFactorValues")
        
        # Header
        group_header = QLabel("Group")
        group_header.setObjectName("lblGroupHeader")
        value_layout.addWidget(group_header, 0, 0)
        
        factor_header = QLabel("Factor value")
        factor_header.setObjectName("lblFactorHeader")
        value_layout.addWidget(factor_header, 0, 1)
        
        # Input fields for each group
        self.factor_values = {}
        for i, group in enumerate(groups):
            group_label = QLabel(str(group))
            group_label.setObjectName(f"lblGroup_{str(group).replace(' ', '_')}")
            value_layout.addWidget(group_label, i+1, 0)
            
            value_field = QLineEdit()
            value_field.setObjectName(f"edtFactorValue_{str(group).replace(' ', '_')}")
            value_field.setPlaceholderText("Value for this group")
            self.factor_values[group] = value_field
            value_layout.addWidget(value_field, i+1, 1)
        
        layout.addWidget(value_group)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.setObjectName("btnDialogButtons")
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def get_factor_data(self):
        import re
        factor_name = self.factor_name.text().strip()
        if not factor_name:
            QMessageBox.warning(self, "Warning", "Please specify a factor name!")
            return None
        if len(factor_name) > 50:
            QMessageBox.warning(self, "Warning", "Factor name must be 50 characters or fewer.")
            return None
        if not re.match(r'^[A-Za-z0-9_\- ]+$', factor_name):
            QMessageBox.warning(self, "Warning",
                "Factor name may only contain letters, digits, spaces, hyphens, and underscores.")
            return None

        factor_data = {}
        for group, field in self.factor_values.items():
            value = field.text().strip()
            if value:  # Only add values if a value was entered
                try:
                    # Try to convert the value to a number if possible
                    numeric_value = float(value)
                    if numeric_value.is_integer():
                        numeric_value = int(numeric_value)
                    factor_data[group] = {factor_name: numeric_value}
                except ValueError:
                    # If not a number, use the string value
                    factor_data[group] = {factor_name: value}
        if not factor_data:
            return None            
        
        return factor_data

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
        self.modz_check.setChecked(True)
        self.grubbs_check.setChecked(False)

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
        self.file_path_label = QLabel("outlier_analysis_results.xlsx")
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
            "outlier_analysis_results.xlsx",
            "Excel Files (*.xlsx);;All Files (*)"
        )
        if file_path:
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
    categorical column.  Results are exported to Excel.
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
            from correlation_models import ExploratoryCorrelationMatrix
            matrix_model = ExploratoryCorrelationMatrix()
            matrix_model.fit(self.df, selected_items, method=method,
                             correction=correction, pairwise=pairwise,
                             stratify_by=stratify_by)
            results = matrix_model.as_results_dict()

            # Export to Excel
            import os
            from export_dispatcher import ExportDispatcher
            out_file = os.path.join(self.output_dir,
                                    "exploratory_correlation_matrix.xlsx")
            results["pairwise_comparisons"] = []
            export_result = ExportDispatcher.export_analysis_results(results, out_file)
            if export_result.get("warning"):
                print(f"WARNING: {export_result['warning']}")

            self._status_label.setText(
                f"Done! Saved to:\n{out_file}"
            )
        except Exception as exc:
            self._status_label.setText(f"Error: {exc}")


# ---------------------------------------------------------------------------
# Debug Console — redirects sys.stdout/stderr to a floating Qt window
# ---------------------------------------------------------------------------
class _DebugStream(QObject):
    """Wraps sys.stdout/stderr and emits each write() call as a Qt signal."""
    text_written = pyqtSignal(str)

    def write(self, text):
        if text:
            self.text_written.emit(text)

    def flush(self):
        pass


class DebugConsoleWindow(QWidget):
    """
    Floating log window that captures all print() / DEBUG: output.
    Color-coded: DEBUG=blue, ERROR=red, WARNING=orange, rest=default.
    Toggle with Ctrl+D or View > Debug Console.
    """

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Window)
        self.setObjectName("debugLogWindow")
        self.setWindowTitle("Debug Console")
        self.resize(700, 400)
        self.setWindowFlags(
            Qt.Window |
            Qt.WindowStaysOnTopHint |
            Qt.WindowCloseButtonHint |
            Qt.WindowMinimizeButtonHint
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.text_edit = QTextEdit()
        self.text_edit.setObjectName("debugConsoleText")
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(__import__('PyQt5.QtGui', fromlist=['QFont']).QFont("Courier", 10))
        layout.addWidget(self.text_edit)

        btn_row = QHBoxLayout()
        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(70)
        clear_btn.clicked.connect(self.text_edit.clear)
        btn_row.addWidget(clear_btn)
        btn_row.addStretch()

        self._line_count_label = QLabel("0 lines")
        btn_row.addWidget(self._line_count_label)
        layout.addLayout(btn_row)

        self._line_count = 0

        # Redirect stdout and stderr
        self._stdout_stream = _DebugStream()
        self._stderr_stream = _DebugStream()
        self._stdout_stream.text_written.connect(self._append)
        self._stderr_stream.text_written.connect(self._append)
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = self._stdout_stream
        sys.stderr = self._stderr_stream

    def _append(self, text):
        if not text.strip():
            return
        color = "#d4d4d4"
        lower = text.lower()
        if "error" in lower or "traceback" in lower or "exception" in lower:
            color = "#f48771"   # red
        elif "warning" in lower or "warn" in lower:
            color = "#ce9178"   # orange
        elif lower.startswith("debug") or "debug:" in lower:
            color = "#9cdcfe"   # blue
        elif lower.startswith("success") or "success" in lower:
            color = "#4ec9b0"   # teal

        import html
        safe = html.escape(text.rstrip())
        self.text_edit.append(f'<span style="color:{color};">{safe}</span>')

        # Auto-scroll
        sb = self.text_edit.verticalScrollBar()
        sb.setValue(sb.maximum())

        self._line_count += 1
        self._line_count_label.setText(f"{self._line_count} lines")

    def closeEvent(self, event):
        # Restore streams on close so the process doesn't lose output
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
        super().closeEvent(event)

    def toggle(self):
        if self.isVisible():
            self.hide()
        else:
            self.show()
            self.raise_()


class StatisticalAnalyzerApp(QMainWindow):
    """Main application for statistical analysis of data from Excel/CSV files."""
    
    def __init__(self):
        """Initializes the application with all UI elements."""
        super().__init__()
        _primary = QApplication.instance().primaryScreen() if QApplication.instance() else None
        screen = _primary.geometry() if _primary else None
        _sw = screen.width()  if screen else 1920
        _sh = screen.height() if screen else 1080
        width = int(_sw * 0.72)
        height = int(_sh * 0.72)
        self.resize(width, height)
        self.move(
            (_sw - width) // 2,
            (_sh - height) // 2
        )
        self.setWindowTitle("BioMedStatX v2.0 - Comprehensive Statistical Analysis Tool")
        self.setGeometry(100, 50, 1600, 1300)
        
        # Set window icon
        try:
            icon_candidates = [
                resource_path("assets/Institutslogo.ico"),
                resource_path("assets/Institutslogo.png"),
            ]
            icon_path = next((path for path in icon_candidates if os.path.exists(path)), None)

            if icon_path:
                self.setWindowIcon(QIcon(icon_path))
                print(f"SUCCESS: Window icon set from {icon_path}")
            else:
                print("WARNING: Icon file not found (checked .ico and .png variants)")
        except Exception as e:
            print(f"ERROR: Could not set window icon: {e}")
        
        # Data attributes
        self.file_path = None
        self.df = None
        self.samples = None
        self.sheet_names = []
        self.available_groups = []
        self.numeric_columns = []
        self.plot_configs = []
        
        # Temporäre Plot-Appearance-Einstellungen (bleiben bis Programm geschlossen wird)
        self.temp_plot_appearance_settings = None
        
        # Initialize UI elements
        self.init_ui()
        
        # Status for combined columns
        self.selected_columns = []
        self.combine_columns = False
        
        # Add menu bar
        self.create_menu()

        # Initialize updater
        self.setup_updater()

        # Debug console — starts alongside the main window
        self.debug_console = DebugConsoleWindow()
        self._position_debug_console()
        self.debug_console.show()
               
    def _position_debug_console(self):
        """Position the debug console to the right of the main window, or below if no space."""
        _primary = QApplication.instance().primaryScreen() if QApplication.instance() else None
        screen = _primary.geometry() if _primary else None
        main_geo = self.geometry()
        console_w = 700
        console_h = 400
        right_x = main_geo.right() + 8
        if right_x + console_w <= screen.width():
            self.debug_console.setGeometry(right_x, main_geo.top(), console_w, console_h)
        else:
            # Fall back to bottom of main window
            bottom_y = main_geo.bottom() + 8
            self.debug_console.setGeometry(main_geo.left(), bottom_y, console_w, console_h)

    def create_menu(self):
        """Creates the menu bar with help options"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu('&File')
        
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = menubar.addMenu('&Help')

        # Getting Started should be first
        getting_started_action = QAction('Getting Started', self)
        getting_started_action.triggered.connect(self.show_getting_started_help)
        help_menu.addAction(getting_started_action)

        help_hub_action = QAction('Help Hub (Recipes)', self)
        help_hub_action.triggered.connect(self.show_help_hub)
        help_menu.addAction(help_hub_action)
        
        help_menu.addSeparator()

        dependent_help_action = QAction('Dependent Samples', self)
        dependent_help_action.triggered.connect(self.show_dependent_samples_help)
        help_menu.addAction(dependent_help_action)

        # New: Graph Visualization help
        graph_vis_action = QAction('Graph Visualization', self)
        graph_vis_action.triggered.connect(self.show_graph_visualization_help)
        help_menu.addAction(graph_vis_action)

        # New: Statistical Tests & Excel Export help
        stats_excel_action = QAction('Statistical Tests && Excel Export', self)
        stats_excel_action.triggered.connect(self.show_statistical_tests_excel_help)
        help_menu.addAction(stats_excel_action)

        help_menu.addSeparator()
        
        # Check for updates
        update_action = QAction('Check for Updates...', self)  
        update_action.triggered.connect(self.check_for_updates)
        help_menu.addAction(update_action)
        
        # About should be last
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

        # View menu
        view_menu = menubar.addMenu('&View')
        debug_action = QAction('Debug Console', self)
        debug_action.setShortcut('Ctrl+D')
        debug_action.setCheckable(True)
        debug_action.setChecked(True)
        debug_action.triggered.connect(lambda checked: self.debug_console.show() if checked else self.debug_console.hide())
        view_menu.addAction(debug_action)
        self._debug_menu_action = debug_action

        # Analysis menu (create new or use existing)
        analysis_menu = menubar.addMenu('&Analysis')

        # Action for outlier detection
        outlier_action = QAction('Detect Outliers', self)
        outlier_action.triggered.connect(self.run_outlier_detection)
        analysis_menu.addAction(outlier_action)

        exploratory_matrix_action = QAction('Exploratory Correlation Matrix', self)
        exploratory_matrix_action.setToolTip(
            'Optional screening tool: explore pairwise correlations before confirmatory hypothesis tests.'
        )
        exploratory_matrix_action.triggered.connect(self.open_exploratory_matrix_dialog)
        analysis_menu.addAction(exploratory_matrix_action)

        
    def show_about(self):
        QMessageBox.information(
            self,
            "About BioMedStatX",
            """
            <h2>BioMedStatX</h2>
            <p>Version 1.0</p>
            <p>An application for statistical analysis and visualization of data.</p>
            <p>© 2025 Philipp Krumm &lt;philipp.krumm@rwth-aachen.de&gt;<br>
            Uniklinik RWTH Aachen<br>
            Department of Anatomy and Cell Biology</p>
            """
        )

    def show_help_hub(self, recipe_id=None):
        """Open the static Help Hub in a non-modal window."""
        # Action triggered signals pass a boolean 'checked', so we catch that
        if isinstance(recipe_id, bool):
            recipe_id = None
            
        if hasattr(self, '_help_hub_dialog') and self._help_hub_dialog is not None:
            if self._help_hub_dialog.isVisible():
                if recipe_id:
                    self._help_hub_dialog.navigate_to(recipe_id)
                self._help_hub_dialog.raise_()
                self._help_hub_dialog.activateWindow()
                return

        self._help_hub_dialog = HelpHubDialog(self)
        self._help_hub_dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        self._help_hub_dialog.destroyed.connect(lambda: setattr(self, '_help_hub_dialog', None))
        self._help_hub_dialog.show()
        if recipe_id:
            self._help_hub_dialog.navigate_to(recipe_id)
        self._help_hub_dialog.raise_()
        self._help_hub_dialog.activateWindow()

    def show_graph_visualization_help(self):
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextBrowser, QPushButton
        dlg = QDialog(self)
        _configure_dialog(dlg, object_name="graphVisualizationHelpDialog")
        dlg.setWindowTitle("Graph Visualization")
        dlg.resize(800, 600)
        layout = QVBoxLayout(dlg)
        browser = QTextBrowser()
        browser.setObjectName("helpDialogBrowser")
        browser.setHtml("""
            <h3>Graph Visualization</h3>
            <ul>
                <li><b>Plot types:</b> Bar, box, violin, and strip plots are generated from your data. Each type visualizes group distributions differently:
                    <ul>
                        <li><b>Bar:</b> Shows group means with error bars.</li>
                        <li><b>Box:</b> Displays medians, quartiles, and outliers.</li>
                        <li><b>Violin:</b> Combines boxplot with a kernel density estimate.</li>
                        <li><b>Strip:</b> Shows all individual data points as dots.</li>
                    </ul>
                </li>
                <li><b>Switching plot types:</b> Use the plot configuration or appearance dialog to select your preferred plot type.</li>
                <li><b>Appearance adjustments:</b>
                    <ul>
                        <li>Change <b>colors</b> and <b>hatches</b> for each group.</li>
                        <li>Choose <b>error bar type</b>: Standard deviation (SD) or standard error (SEM).</li>
                        <li>Set <b>error bar style</b>: With caps or line only.</li>
                        <li>Customize <b>fonts</b>, <b>axes</b>, and <b>grid lines</b> for clarity.</li>
                    </ul>
                </li>
                <li><b>Overlay features:</b>
                    <ul>
                        <li>Show <b>individual data points</b> on box, violin, or strip plots.</li>
                        <li>Add <b>statistical annotations</b>: Letters (grouping) or bars (significance lines) to highlight significant differences.</li>
                    </ul>
                </li>
            </ul>
        """)
        layout.addWidget(browser)
        btn = QPushButton("OK")
        btn.clicked.connect(dlg.accept)
        layout.addWidget(btn)
        dlg.exec_()

    def show_statistical_tests_excel_help(self):
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextBrowser, QPushButton
        dlg = QDialog(self)
        _configure_dialog(dlg, object_name="statsExcelHelpDialog")
        dlg.setWindowTitle("Statistical Tests & Excel Export")
        dlg.resize(900, 600)
        layout = QVBoxLayout(dlg)
        browser = QTextBrowser()
        browser.setObjectName("helpDialogBrowser")
        browser.setHtml("""
            <h3>Statistical Tests & Excel Export</h3>
            <ul>
                <li><b>How does the program select the test?</b>
                    <ul>
                        <li>The program automatically detects the appropriate test based on group count and data structure.</li>
                        <li><b>Two independent groups:</b>
                            <ul>
                                <li><b>t-Test</b> (parametric): Used when data is normally distributed and variances are comparable.</li>
                                <li><b>Mann-Whitney-U Test</b> (non-parametric): Used when assumptions for t-test are not met.</li>
                            </ul>
                        </li>
                        <li><b>Two dependent groups (e.g. paired measurements):</b>
                            <ul>
                                <li><b>Paired t-Test</b> (parametric): For normally distributed differences.</li>
                                <li><b>Wilcoxon signed-rank test</b> (non-parametric): For non-normally distributed differences.</li>
                            </ul>
                        </li>
                        <li><b>More than two independent groups:</b>
                            <ul>
                                <li><b>One-Way ANOVA</b> (parametric): For normally distributed data with equal variances.</li>
                                <li><b>Kruskal-Wallis Test</b> (non-parametric): When ANOVA assumptions are violated.</li>
                            </ul>
                        </li>
                        <li>The decision is based on normality tests (Shapiro-Wilk) and variance homogeneity (Levene test). When assumptions are violated, a non-parametric test is automatically selected.</li>
                        <li>Post-hoc tests (e.g. pairwise comparisons) are automatically added when significant differences are found.</li>
                        <li><i>Note: For detailed data templates (including long-format examples), open the Help Hub (Recipes) from the Help menu.</i></li>
                    </ul>
                </li>
                <li><b>Interpreting Results:</b>
                    <ul>
                        <li><b>p-values</b> indicate the probability that observed differences are due to chance.</li>
                        <li><b>Significance indicators</b> (letters or bars) show which groups differ significantly.</li>
                        <li>Key statistics (means, standard deviations, test statistics) are clearly displayed.</li>
                    </ul>
                </li>
                <li><b>Excel Export:</b>
                    <ul>
                        <li>Results are written to an Excel workbook with separate worksheets for each analysis.</li>
                        <li>Sheet names reflect the test or plot type (e.g. "ANOVA Results", "Pairwise Comparisons").</li>
                        <li>Each sheet contains clear columns: group names, means, test statistics, p-values, and significance markers.</li>
                        <li>Open the exported file in Excel to review, print, or share results. Use the tabs to switch between analyses.</li>
                    </ul>
                </li>
            </ul>
            <p style='color:gray; font-size:90%'>Note: Use Help -> Help Hub (Recipes) for detailed long-format templates for advanced and basic models.</p>
        """)
        layout.addWidget(browser)
        btn = QPushButton("OK")
        btn.clicked.connect(dlg.accept)
        layout.addWidget(btn)
        dlg.exec_()
        
    def init_ui(self):
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        """Initializes all UI elements of the application."""
        # Main widget and layout
        central_widget = QWidget()
        central_widget.setObjectName("widMainContainer")
        main_layout = QVBoxLayout(central_widget)
        main_layout.setObjectName("lyoMainLayout")
        main_layout.setContentsMargins(16, 8, 16, 16)
        main_layout.setSpacing(12)
        
        # File selection
        file_section = QGroupBox("Data Source")
        file_section.setObjectName("grpDataSource")
        file_layout = QHBoxLayout(file_section)
        file_layout.setObjectName("lyoFileSection")
        file_layout.setContentsMargins(0, 0, 0, 0)
        file_layout.setSpacing(8)
        
        file_label = QLabel("Excel file:")
        file_label.setObjectName("lblFileLabel")
        self.file_path_label = QLabel("No file selected")
        self.file_path_label.setObjectName("lblFilePath")
        browse_button = QPushButton("Browse...")
        browse_button.setObjectName("btnBrowse")
        browse_button.clicked.connect(self.browse_file)
        
        file_layout.addWidget(file_label)
        file_layout.addWidget(self.file_path_label, 1)
        file_layout.addWidget(browse_button)
        
        main_layout.addWidget(file_section)
        
        # Excel sheet and column selection
        data_section = QGroupBox("Data Configuration")
        data_section.setObjectName("grpDataConfig")
        data_layout = QVBoxLayout(data_section)
        data_layout.setObjectName("lyoDataSection")
        data_layout.setContentsMargins(0, 0, 0, 0)
        data_layout.setSpacing(8)
        
        # Excel sheet
        sheet_layout = QHBoxLayout()
        sheet_layout.setObjectName("lyoSheetSection")
        sheet_label = QLabel("Worksheet:")
        sheet_label.setObjectName("lblSheetLabel")
        self.sheet_combo = QComboBox()
        self.sheet_combo.setObjectName("cboWorksheet")
        self.sheet_combo.currentIndexChanged.connect(self.load_sheet)
        sheet_layout.addWidget(sheet_label)
        sheet_layout.addWidget(self.sheet_combo, 1)
        data_layout.addLayout(sheet_layout)
        
        # Group column
        group_col_layout = QHBoxLayout()
        group_col_layout.setObjectName("lyoGroupColSection")
        group_col_label = QLabel("Group column:")
        group_col_label.setObjectName("lblGroupColLabel")
        self.group_col_combo = QComboBox()
        self.group_col_combo.setObjectName("cboGroupColumn")
        self.group_col_combo.currentIndexChanged.connect(self.update_available_groups)
        group_col_layout.addWidget(group_col_label)
        group_col_layout.addWidget(self.group_col_combo, 1)
        data_layout.addLayout(group_col_layout)
        
        # Value column(s)
        value_cols_layout = QHBoxLayout()
        value_cols_layout.setObjectName("lyoValueColSection")
        value_cols_label = QLabel("Value column(s):")
        value_cols_label.setObjectName("lblValueColLabel")
        self.value_cols_combo = QComboBox()
        self.value_cols_combo.setObjectName("cboValueColumn")
        self.value_cols_combo.currentIndexChanged.connect(self.update_samples)
        value_cols_layout.addWidget(value_cols_label)
        value_cols_layout.addWidget(self.value_cols_combo, 1)
        
        # Button to select multiple columns
        self.select_columns_button = QPushButton("Multiple columns...")
        self.select_columns_button.setObjectName("btnSelectColumns")
        self.select_columns_button.clicked.connect(self.select_multiple_columns)
        value_cols_layout.addWidget(self.select_columns_button)
        
        data_layout.addLayout(value_cols_layout)
        
        # Mark for combined columns
        self.combine_columns_label = QLabel("No combined columns selected")
        self.combine_columns_label.setObjectName("lblCombineStatus")
        data_layout.addWidget(self.combine_columns_label)
        
        main_layout.addWidget(data_section)
        
        # Available groups and plot management
        groups_and_plots = QHBoxLayout()
        groups_and_plots.setObjectName("lyoGroupsAndPlots")
        groups_and_plots.setSpacing(12)
        
        # Available groups
        groups_section = QGroupBox("Available Groups")
        groups_section.setObjectName("grpAvailableGroups")
        groups_layout = QVBoxLayout(groups_section)
        groups_layout.setObjectName("lyoGroupsSection")
        
        self.groups_list = QListWidget()
        self.groups_list.setObjectName("lstAvailableGroups")
        # Enable multi-selection for comparing groups
        self.groups_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.groups_list.setToolTip("Select groups to preview (Ctrl+Click for multiple selection)")
        # Limit height for many groups
        self.groups_list.setMaximumHeight(200)
        # Connection for automatic preview updates when group selection changes
        self.groups_list.itemSelectionChanged.connect(self.update_preview_on_selection_change)
        groups_layout.addWidget(self.groups_list)

        self.groups_empty_label = QLabel("Load a file and\nselect a group column")
        self.groups_empty_label.setObjectName("lblEmptyState")
        self.groups_empty_label.setAlignment(Qt.AlignCenter)
        self.groups_empty_label.setMinimumHeight(64)
        groups_layout.addWidget(self.groups_empty_label)
        self.groups_list.hide()
        
        # Buttons for group selection
        group_buttons = QHBoxLayout()
        group_buttons.setObjectName("lyoGroupButtons")
        select_groups_button = QPushButton("Select groups for analysis")
        select_groups_button.setObjectName("btnSelectGroups")
        select_groups_button.clicked.connect(self.select_groups_for_plot)
        
        group_buttons.addWidget(select_groups_button)
        groups_layout.addLayout(group_buttons)
        
        groups_and_plots.addWidget(groups_section)
        
        # Plot configurations
        plots_section = QGroupBox("Plot Configurations")
        plots_section.setObjectName("grpPlotConfigs")
        plots_layout = QVBoxLayout(plots_section)
        plots_layout.setObjectName("lyoPlotsSection")
        
        self.plots_list = QListWidget()
        self.plots_list.setObjectName("lstPlotConfigurations")
        # Limit height for many plot configurations
        self.plots_list.setMaximumHeight(200)
        self.plots_list.itemDoubleClicked.connect(self.edit_plot_config)
        plots_layout.addWidget(self.plots_list)

        self.plots_empty_label = QLabel("No plots configured yet.\nSelect groups to add a plot.")
        self.plots_empty_label.setObjectName("lblEmptyState")
        self.plots_empty_label.setAlignment(Qt.AlignCenter)
        self.plots_empty_label.setMinimumHeight(64)
        plots_layout.addWidget(self.plots_empty_label)
        self.plots_list.hide()
        
        # Plot buttons
        plot_buttons = QHBoxLayout()
        plot_buttons.setObjectName("lyoPlotButtons")
        remove_plot_button = QPushButton("Remove plot")
        remove_plot_button.setObjectName("btnRemovePlot")
        remove_plot_button.clicked.connect(self.remove_plot)
        preview_plot_button = QPushButton("Plot preview")
        preview_plot_button.setObjectName("btnPreviewPlot")
        preview_plot_button.clicked.connect(self.preview_selected_plot)
        
        plot_buttons.addWidget(remove_plot_button)
        plot_buttons.addWidget(preview_plot_button)
        plots_layout.addLayout(plot_buttons)
        
        groups_and_plots.addWidget(plots_section)
        main_layout.addLayout(groups_and_plots)
        
        # Plot preview - use new PlotPreviewWidget
        preview_section = QGroupBox("Live Plot Preview")
        preview_section.setObjectName("grpPlotPreview")
        preview_section.setToolTip("Shows preview of selected groups. Updates automatically when data changes.")
        preview_layout = QVBoxLayout(preview_section)
        preview_layout.setObjectName("lyoPreviewSection")
        
        # Try to import and use the new PlotPreviewWidget
        try:
            from plot_preview import PlotPreviewWidget
            self.plot_preview_widget = PlotPreviewWidget()
            self.plot_preview_widget.setObjectName("widgetPlotPreview")
            preview_layout.addWidget(self.plot_preview_widget)
        except ImportError:
            # Fallback to old matplotlib canvas
            self.figure = Figure(figsize=(8, 6), dpi=100)
            self.canvas = FigureCanvas(self.figure)
            self.canvas.setObjectName("canvasPlotPreview")
            preview_layout.addWidget(self.canvas)
            self.plot_preview_widget = None
        
        main_layout.addWidget(preview_section)

        # Action buttons
        actions_layout = QHBoxLayout()
        actions_layout.setObjectName("lyoActionButtons")
        actions_layout.setSpacing(8)
        analyze_button = QPushButton("Start all analyses")
        analyze_button.setObjectName("btnAnalyzeAll")
        analyze_button.clicked.connect(self.run_all_analyses)
        analyze_selected_button = QPushButton("Start selected analysis")
        analyze_selected_button.setObjectName("btnAnalyzeSelected")
        analyze_selected_button.clicked.connect(self.run_selected_analysis)
        multi_analyze_button = QPushButton("Start multi-dataset analysis")
        multi_analyze_button.setObjectName("btnMultiDatasetAnalyze")
        multi_analyze_button.clicked.connect(self.run_multi_dataset_analysis)
        actions_layout.addWidget(analyze_button)
        actions_layout.addWidget(analyze_selected_button)
        actions_layout.addWidget(multi_analyze_button)
        
        main_layout.addLayout(actions_layout)
        
        self.setCentralWidget(central_widget)
    
    def browse_file(self):
        """Opens a dialog to select an Excel or CSV file."""
        options = QFileDialog.Options()
        if sys.platform == "darwin":
            # Native macOS dialogs can become non-selectable for some users/setups.
            # The Qt dialog is more reliable for extension-filtered selection.
            options |= QFileDialog.DontUseNativeDialog

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Excel/CSV file",
            "",
            "All supported files (*.xlsx *.xls *.csv);;Excel files (*.xlsx *.xls);;CSV files (*.csv);;All files (*)",
            options=options,
        )
        
        if file_path:
            # Check if file exists and has a supported format
            if not os.path.exists(file_path):
                QMessageBox.critical(self, "Error", f"The file {file_path} does not exist.")
                return
                
            if not file_path.lower().endswith((".xlsx", ".xls", ".csv")):
                QMessageBox.critical(self, "Error", "Only Excel and CSV files are supported.")
                return
                
            self.file_path = file_path
            self.file_path_label.setText(os.path.basename(file_path))
            self.load_file()
    
    def load_file(self):
        """Loads the selected file and prepares it for analysis."""
        if not self.file_path:
            return
            
        try:
            if self.file_path.lower().endswith('.csv'):
                # CSV file
                self.df = pd.read_csv(self.file_path)
                self.sheet_combo.clear()
                self.sheet_combo.setEnabled(False)
            else:
                # Excel file
                excel = pd.ExcelFile(self.file_path)
                self.sheet_names = excel.sheet_names
                
                self.sheet_combo.clear()
                self.sheet_combo.addItems(self.sheet_names)
                self.sheet_combo.setEnabled(True)
                
                # Load first worksheet by default
                if self.sheet_names:
                    self.df = pd.read_excel(self.file_path, sheet_name=self.sheet_names[0])
            
            # Reset selection
            self.selected_columns = []
            self.combine_columns = False
            self.combine_columns_label.setText("No combined columns selected")
            
            self.update_column_lists()
            
        except Exception as e:
            self.df = None
            QMessageBox.critical(self, "Error", f"Error loading file: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def load_sheet(self, index):
        """Loads a specific worksheet from an Excel file."""
        if index < 0 or not self.file_path or not self.file_path.lower().endswith((".xlsx", ".xls")):
            return
            
        try:
            sheet_name = self.sheet_combo.itemText(index)
            self.df = pd.read_excel(self.file_path, sheet_name=sheet_name)
            self.update_column_lists()
        except Exception as e:
            self.df = None
            QMessageBox.critical(self, "Error", f"Error loading worksheet: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def update_column_lists(self):
        """Updates the combo boxes for group and value columns."""
        if self.df is None:
            return
            
        try:
            # Group columns
            self.group_col_combo.clear()
            self.group_col_combo.addItems(self.df.columns)
            
            # Value columns (only numeric columns)
            self.value_cols_combo.clear()
            self.numeric_columns = []
            for col in self.df.columns:
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    self.numeric_columns.append(col)
            
            self.value_cols_combo.addItems(self.numeric_columns)
            
            # Reset selected columns when new data is loaded
            self.selected_columns = []
            
            # Update available groups after loading new columns
            self.update_available_groups()
            
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Error updating column lists: {str(e)}")
    
    def select_multiple_columns(self):
        """Opens a dialog to select multiple value columns."""
        if self.df is None:
            QMessageBox.warning(self, "Warning", "No data loaded.")
            return
            
        if not self.numeric_columns:
            QMessageBox.warning(self, "Warning", "No numeric columns available.")
            return
            
        dialog = ColumnSelectionDialog(self.numeric_columns, self)
        if dialog.exec_() == QDialog.Accepted:
            result = dialog.get_selected_columns()
            selected_columns = result["columns"]
            multi_dataset = result["multi_dataset"]
            
            if not selected_columns:
                QMessageBox.warning(self, "Warning", "No columns selected.")
                return
                
            self.selected_columns = selected_columns
            self.combine_columns = False  # Always set to False
            self.multi_dataset_analysis = multi_dataset
            
            # Update display
            if len(selected_columns) == 1:
                # If only one column, select it in the combobox
                index = self.value_cols_combo.findText(selected_columns[0])
                if index >= 0:
                    self.value_cols_combo.setCurrentIndex(index)
                self.combine_columns_label.setText("No combined columns selected")
            else:
                # For multiple columns, show info
                if multi_dataset:
                    self.combine_columns_label.setText(f"Multi-dataset analysis: {', '.join(selected_columns)}")
                    # ... rest of code ...
                else:
                    self.combine_columns_label.setText(f"Selected: {', '.join(selected_columns)}")
            
            # Update available groups
            self.update_available_groups()
    
    def update_available_groups(self):
        """Updates the list of available groups based on the selected group column."""
        if self.df is None:
            return
            
        if not self.group_col_combo.currentText():
            return
            
        group_col = self.group_col_combo.currentText()
        
        try:
            # Extract groups from the column
            self.available_groups = sorted(self.df[group_col].unique())
            
            # Show list of available groups
            self.groups_list.clear()
            for group in self.available_groups:
                self.groups_list.addItem(str(group))

            has_groups = self.groups_list.count() > 0
            self.groups_list.setVisible(has_groups)
            self.groups_empty_label.setVisible(not has_groups)

            # Reload data
            self.update_samples()
            
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Error loading groups: {str(e)}")
    
    def update_samples(self):
        """Updates the samples based on the current selection of group column and value columns."""
        if self.df is None:
            # No data loaded
            self.samples = None
            return
            
        if not self.group_col_combo.currentText():
            # No group column selected
            QMessageBox.warning(self, "Warning", "Please select a group column.")
            self.samples = None
            return
            
        group_col = self.group_col_combo.currentText()
        
        try:
            # Determine columns to use
            if len(self.selected_columns) > 1:
                value_cols = self.selected_columns
                # Check if all columns actually exist
                for col in value_cols:
                    if col not in self.df.columns:
                        QMessageBox.warning(self, "Warning", 
                            f"The value column '{col}' was not found. Please select valid columns.")
                        self.samples = None
                        return
            else:
                # If no explicit selection, use current combobox selection
                current_value_col = self.value_cols_combo.currentText()
                if not current_value_col:
                    QMessageBox.warning(self, "Warning", "Please select at least one value column.")
                    self.samples = None
                    return
                    
                value_cols = [current_value_col]
            
            # Import data with the DataImporter class
            sheet_name = self.sheet_combo.currentText() if self.sheet_combo.isEnabled() else 0
            self.samples, _ = DataImporter.import_data(
                self.file_path,
                sheet_name=sheet_name,
                group_col=group_col,
                value_cols=value_cols,
                combine_columns=self.combine_columns
            )
            
            # New validation for possible dependent samples
            self.validate_dependent_samples_possibility()
            
            # Automatically create preview when samples are loaded
            self.auto_generate_preview()
            
        except Exception as e:
            self.samples = None
            QMessageBox.warning(self, "Warning", f"Error importing data: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def validate_dependent_samples_possibility(self):
        """Checks if the loaded data might be suitable for dependent tests"""
        if not self.samples or len(self.samples) < 2:
            return
            
        # Check if all groups have the same number of measurements
        group_sizes = [len(values) for values in self.samples.values()]
        equal_sizes = len(set(group_sizes)) == 1
        
        if not equal_sizes:
            # Discreet hint message at the bottom of the screen or as status
            self.statusBar().showMessage(
                "Note: Groups have different sizes - dependent tests may be unsuitable", 
                10000  # Show for 10 seconds
            )        
    
    def detect_data_format(self, df):
        """Detects the data format based on column names"""
        columns = set(df.columns)
        columns_lower = {col.lower() for col in columns}  # Convert to lowercase for more robust detection
        format_type = "unknown"
        
        # Check for Subject column - required for both RM and Mixed ANOVA
        has_subject = 'subject' in columns_lower or any(col.startswith('s') and col[1:].isdigit() for col in columns_lower)
        
        # Check for Timepoint column - indicates a within-subjects factor
        has_timepoint = 'timepoint' in columns_lower or 'time' in columns_lower or 'zeit' in columns_lower
        
        # Check for obvious between-subjects factor columns
        has_between = 'group' in columns_lower or 'gruppe' in columns_lower or 'treatment' in columns_lower or 'condition' in columns_lower
        
        # Repeated Measures ANOVA: Subject + Timepoint without obvious between factors
        if has_subject and has_timepoint and not has_between:
            format_type = "repeated_measures_anova"
        # Mixed ANOVA: Subject + Timepoint + Between factors
        elif has_subject and has_timepoint and has_between:
            format_type = "mixed_anova"
        # Two-Way ANOVA format (FactorA, FactorB, Value)
        elif ('factora' in columns_lower and 'factorb' in columns_lower) or \
            any('factor' in col.lower() for col in columns):
            format_type = "two_way_anova"
        
        print(f"Detected data format: {format_type}")
        return format_type
    
    def select_groups_for_plot(self):
        """Opens a dialog to select groups for a plot."""
        if not self.available_groups:
            QMessageBox.warning(self, "Error", "No groups available. Please load data first.")
            return
        
        dialog = GroupSelectionDialog(self.available_groups, self)
        if dialog.exec_() == QDialog.Accepted:
            selected_groups = dialog.get_selected_groups()
            if selected_groups:
                self.configure_plot(selected_groups)
            else:
                QMessageBox.warning(self, "Error", "No groups selected.")
    
    def configure_plot(self, groups):
        """Opens a dialog to configure a plot with the selected groups."""
        if not groups:
            QMessageBox.warning(self, "Error", "No groups provided for plot configuration.")
            return
        default_filename = None
        if hasattr(self, 'file_path') and self.file_path:
            base_filename = os.path.splitext(os.path.basename(self.file_path))[0]
            default_filename = f"{base_filename}_analyzed"

        samples_for_dlg = {}
        if hasattr(self, 'samples') and self.samples:
            samples_for_dlg = {g: self.samples[g] for g in groups if g in (self.samples or {})}

        dlg = PlotAestheticsDialog(
            groups=groups,
            samples=samples_for_dlg,
            parent=self,
            default_filename=default_filename,
        )
        if dlg.exec_() == QDialog.Accepted:
            config = dlg.get_config()
            if config is None:
                return
            self.plot_configs.append(config)
            plot_item_text = f"Plot: {config.get('title') or ', '.join(config.get('groups', []))}"
            self.plots_list.addItem(plot_item_text)
            self._update_plots_empty_state()
            if config.get('create_plot', False):
                self.preview_plot(len(self.plot_configs) - 1)
    
    def edit_plot_config(self, item):
        """Edits the configuration of a selected plot."""
        index = self.plots_list.row(item)
        if index < 0 or index >= len(self.plot_configs):
            return

        config = self.plot_configs[index]
        groups_for_dlg = config.get('groups', [])

        default_filename = None
        if hasattr(self, 'file_path') and self.file_path:
            base_filename = os.path.splitext(os.path.basename(self.file_path))[0]
            default_filename = f"{base_filename}_analyzed"

        samples_for_dlg = {}
        if hasattr(self, 'samples') and self.samples:
            samples_for_dlg = {g: self.samples[g] for g in groups_for_dlg if g in (self.samples or {})}

        dlg = PlotAestheticsDialog(
            groups=groups_for_dlg,
            samples=samples_for_dlg,
            config=config,
            parent=self,
            default_filename=default_filename,
            dependent=config.get('dependent', False),
        )
        if dlg.exec_() == QDialog.Accepted:
            new_config = dlg.get_config()
            self.plot_configs[index] = new_config
            plot_item_text = f"Plot: {new_config.get('title') or ', '.join(new_config.get('groups', []))}"
            item.setText(plot_item_text)
            self.preview_plot(index)
    
    def get_analysis_params(self):
        # Implement this method to return analysis parameters
        return {
            'file_path': self.file_path,
                       'group_col': self.group_col_combo.currentText(),
            # Add more parameters here
        }
    
    def display_results(self, results):
        # Error handling
        if 'error' in results and results['error'] is not None and results['error']:
            QMessageBox.critical(self, "Error", f"Analysis error: {results['error']}")
            return
    
        analysis_log = ""
        
        # Add main results of the ANOVA
        analysis_log += f"Test: {results.get('test', 'Unknown')}\n"
        
        # Safe formatting for p-value
        if 'p_value' in results and results['p_value'] is not None:
            analysis_log += f"p-value: {results['p_value']:.4f}\n"
        else:
            analysis_log += "p-value: Not available\n"
        
        # Safe formatting for test statistic
        if 'statistic' in results and results['statistic'] is not None:
            analysis_log += f"Test statistic: {results['statistic']:.4f}\n"
        else:
            analysis_log += "Test statistic: Not available\n"
        
        # Add status of post-hoc tests
        if results.get('posthoc_status') == 'not_performed':
            analysis_log += f"\nPost-hoc tests: Not performed\n"
            analysis_log += f"Reason: {results.get('posthoc_reason', 'Unknown')}\n"
        
        # Pairwise comparisons, if available
        if 'pairwise_comparisons' in results and results['pairwise_comparisons']:
            analysis_log += "\nPairwise comparisons:\n"
            for comp in results["pairwise_comparisons"]:
                analysis_log += (f"{comp['group1']} vs {comp['group2']}: "
                            f"p = {comp['p_value']:.4g}, "
                            f"significant: {'yes' if comp['significant'] else 'no'}\n")
        else:
            analysis_log += "\nNo pairwise comparisons were performed or calculated.\n"
    
        # Debug output
        print("Analysis results:", analysis_log)
        QMessageBox.information(self, "Analysis Results", analysis_log)
    
    def direct_group_comparison(self):
        """Performs a direct comparison between two selected groups."""
        if not self.available_groups:
            QMessageBox.warning(self, "Error", "No groups available. Please load data first.")
            return
        
        dialog = PairwiseComparisonDialog(self.available_groups, self)
        if dialog.exec_() == QDialog.Accepted:
            comp = dialog.get_comparison()
            if comp['group1'] == comp['group2']:
                QMessageBox.warning(self, "Error", "The two groups must be different!")
                return
            
            self.run_direct_comparison(comp)
    
    def run_direct_comparison(self, comp):
        """Performs a direct statistical comparison between two groups."""
        if not self.samples:
            QMessageBox.warning(self, "Error", "No data loaded.")
            return
        
        try:
            # Extract the two groups
            group1 = comp['group1']
            group2 = comp['group2']
            
            # Check if both groups are present in the data
            if group1 not in self.samples or group2 not in self.samples:
                QMessageBox.warning(self, "Error", f"One or both groups ({group1}, {group2}) not found in the data.")
                return
            
            # Create transformed samples for statistical tests
            groups = [group1, group2]
            transformed_samples, test_recommendation, _ = StatisticalTester.check_normality_and_variance(groups, self.samples)
            
            # Determine test type
            if comp['test_type'] == "t-Test (parametric)":
                test_recommendation = "parametric"
            elif comp['test_type'] == "Mann-Whitney-U (non-parametric)":
                test_recommendation = "non_parametric"
            
            # Perform test and save results
            results = StatisticalTester.perform_statistical_test(groups, transformed_samples, self.samples, 
                                dependent=comp['dependent'], 
                                test_recommendation=test_recommendation)
            
            # Visualization
            colors = DEFAULT_COLORS[:2]  # Use the first two default colors
            hatches = DEFAULT_HATCHES[:2]  # Use the first two default hatches
            
            # Extract pairwise_comparisons from results
            pairwise_comparisons = results.get('pairwise_comparisons', None)
            
            DataVisualizer.plot_bar(groups, self.samples, width=8, height=6, 
                colors=colors, hatches=hatches, 
                compare=[(group1, group2)],
                test_recommendation=test_recommendation,
                pairwise_results=pairwise_comparisons)  # Pass pairwise_comparisons
    
            # Show results
            self.display_results(results)
            
            QMessageBox.information(self, "Success", 
                                f"Direct comparison between {group1} and {group2} was performed and visualized.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error performing the comparison: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def remove_plot(self):
        """Removes the selected plot from the list of configurations."""
        current_row = self.plots_list.currentRow()
        if current_row >= 0:
            self.plots_list.takeItem(current_row)
            self.plot_configs.pop(current_row)
            self._update_plots_empty_state()

            # Clear the preview if no plot is left
            if len(self.plot_configs) == 0:
                if hasattr(self, 'plot_preview_widget') and self.plot_preview_widget:
                    self.plot_preview_widget._show_placeholder()
                elif hasattr(self, 'figure') and hasattr(self, 'canvas'):
                    self.figure.clear()
                    self.canvas.draw()

    def _update_plots_empty_state(self):
        """Shows or hides the plots empty-state label based on list content."""
        has_plots = self.plots_list.count() > 0
        self.plots_list.setVisible(has_plots)
        self.plots_empty_label.setVisible(not has_plots)
    
    def preview_selected_plot(self):
        """Creates a preview of the selected plot."""
        current_row = self.plots_list.currentRow()
        if current_row >= 0:
            # Always show preview, regardless of appearance settings
            self.preview_plot(current_row)
        else:
            QMessageBox.warning(self, "Error", "Please select a plot from the list.")
    
    def preview_plot(self, plot_idx):
        """Creates a preview of a plot based on its configuration."""
        if self.samples is None:
            QMessageBox.warning(self, "Error", "No data loaded.")
            return
            
        if plot_idx < 0 or plot_idx >= len(self.plot_configs):
            return

        plot_config = self.plot_configs[plot_idx]

        try:
            # Check if we have the new PlotPreviewWidget
            if hasattr(self, 'plot_preview_widget') and self.plot_preview_widget:
                # Use the new preview widget
                groups = plot_config.get('groups', [])
                samples = {group: self.samples[group] for group in groups if group in self.samples}
                
                if samples:
                    self.plot_preview_widget.set_data(groups, samples)
                    
                    # Convert appearance settings to new format if available
                    appearance = plot_config.get('appearance_settings', {})
                    if appearance:
                        self.plot_preview_widget.update_plot(appearance)
                    else:
                        # Use default preview
                        self.plot_preview_widget.update_plot({'plot_type': 'Bar'})
                return
            
            # Fallback to old matplotlib canvas (if new widget not available)
            if not hasattr(self, 'figure'):
                return
                
            # Clear the figure
            self.figure.clear()
            ax = self.figure.add_subplot(111)

            # Prepare data - with error checking
            plot_samples = {}
            if 'groups' not in plot_config:
                QMessageBox.warning(self, "Error", "Configuration contains no groups.")
                return

            for group in plot_config.get('groups', []):
                if self.samples and group in self.samples:
                    plot_samples[group] = self.samples[group]

            if not plot_samples:
                QMessageBox.warning(self, "Warning", "No data found for the selected groups.")
                return

            # Prepare data for DataFrame (for possible future use)
            plot_data = []
            for group, values in plot_samples.items():
                for value in values:
                    plot_data.append({'Group': group, 'Value': value})
            df = pd.DataFrame(plot_data)

            # --- APPEARANCE SETTINGS ---
            appearance = plot_config.get('appearance_settings', None)
            use_appearance = plot_config.get('create_plot', True) and appearance is not None

            if use_appearance:
                # Handle both old and new format
                if 'colors' in appearance and isinstance(appearance['colors'], dict):
                    colors = [appearance['colors'].get(group, DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
                            for i, group in enumerate(plot_config['groups'])]
                else:
                    colors = DEFAULT_COLORS[:len(plot_config['groups'])]
                
                if 'hatches' in appearance and isinstance(appearance['hatches'], dict):
                    hatches = [appearance['hatches'].get(group, DEFAULT_HATCHES[i % len(DEFAULT_HATCHES)])
                            for i, group in enumerate(plot_config['groups'])]
                else:
                    hatches = [''] * len(plot_config['groups'])  # No hatches by default
                
                alpha = appearance.get('alpha', 0.8)
                axis_linewidth = appearance.get('axis_linewidth', 0.7)
                bar_linewidth = appearance.get('bar_linewidth', 1.0)
                gridline_width = appearance.get('gridline_width', 0.5)
                grid = appearance.get('grid', False)  # Default should be False
                minor_ticks = appearance.get('minor_ticks', False)
                despine = appearance.get('despine', True)
                fontsize_axis = appearance.get('fontsize_axis', 11)
                fontsize_ticks = appearance.get('fontsize_ticks', 11)
                fontsize_groupnames = appearance.get('fontsize_groupnames', 11)
                fontsize_title = appearance.get('fontsize_title', 11)
                show_title = appearance.get('show_title', True)
                title = plot_config.get('title', 'Preview')
                bar_edge_color = appearance.get('bar_edge_color', 'black')
                plot_type = appearance.get('plot_type', 'Bar')
            else:
                colors = ['#CCCCCC'] * len(plot_config['groups'])
                hatches = [''] * len(plot_config['groups'])
                alpha = 1.0
                axis_linewidth = 1.0
                bar_linewidth = 1.0
                gridline_width = 0.5
                grid = False  # Default to False like in appearance settings
                minor_ticks = False
                despine = True
                fontsize_axis = 11
                fontsize_ticks = 11
                fontsize_groupnames = 11
                fontsize_title = 11
                show_title = True
                title = plot_config.get('title', 'Preview')
                bar_edge_color = 'black'
                plot_type = 'Bar'

            # Error type from configuration
            error_type = plot_config.get('error_type', 'sd')

            groups = plot_config['groups']
            samples = {g: plot_samples[g] for g in groups}
            import numpy as np  # Ensure np is available in this scope
            means = [np.mean(samples[g]) if samples[g] else 0 for g in groups]
            
            # Calculate errors based on error_type
            if error_type == 'se':
                errors = [np.std(samples[g]) / np.sqrt(len(samples[g])) if samples[g] and len(samples[g]) > 0 else 0 for g in groups]
            else:  # 'sd'
                errors = [np.std(samples[g]) if samples[g] else 0 for g in groups]
            
            bars = None

            # --- First ensure that grid is turned off by default ---
            ax.grid(False)
            
            # --- Plot type selection ---
            if plot_type == "Bar":
                bars = ax.bar(
                    groups, means, yerr=errors,
                    color=colors,
                    hatch=hatches,
                    alpha=alpha,
                    linewidth=bar_linewidth,
                    edgecolor=bar_edge_color
                )
                
                # Add individual data points with jitter
                for i, g in enumerate(groups):
                    vals = samples[g]
                    x = np.full(len(vals), i)
                    jitter = np.random.uniform(-0.2, 0.2, size=len(vals))
                    ax.scatter(x + jitter, vals, color='black', alpha=0.6, zorder=3, s=40, edgecolors='white', linewidths=0.5)
                    
            elif plot_type == "Box":
                bp = ax.boxplot(
                    [samples[g] for g in groups],
                    patch_artist=True
                )
                # Apply colors and styles to each box individually
                for i, (patch, color) in enumerate(zip(bp['boxes'], colors)):
                    patch.set_facecolor(color)
                    patch.set_edgecolor(bar_edge_color)
                    patch.set_linewidth(bar_linewidth)
                    
                # Also style the whiskers, caps, medians, etc.
                for whisker in bp['whiskers']:
                    whisker.set_color(bar_edge_color)
                    whisker.set_linewidth(bar_linewidth)
                for cap in bp['caps']:
                    cap.set_color(bar_edge_color)
                    cap.set_linewidth(bar_linewidth)
                for median in bp['medians']:
                    median.set_color(bar_edge_color)
                    median.set_linewidth(bar_linewidth)
                for flier in bp['fliers']:
                    flier.set_markeredgecolor(bar_edge_color)
                    
                # Add individual data points with jitter
                for i, g in enumerate(groups):
                    vals = samples[g]
                    x = np.full(len(vals), i + 1)  # boxplot positions are 1-indexed
                    jitter = np.random.uniform(-0.2, 0.2, size=len(vals))
                    ax.scatter(x + jitter, vals, color='black', alpha=0.6, zorder=3, s=40, edgecolors='white', linewidths=0.5)

            elif plot_type == "Violin":
                vp = ax.violinplot(
                    [samples[g] for g in groups],
                    showmeans=True, showmedians=True
                )
                # Set edge color and face color for violins
                for i, pc in enumerate(vp['bodies']):
                    pc.set_edgecolor(bar_edge_color)
                    pc.set_linewidth(bar_linewidth)
                    pc.set_alpha(alpha)
                    pc.set_facecolor(colors[i % len(colors)])
                    
                # Add individual data points with jitter
                for i, g in enumerate(groups):
                    vals = samples[g]
                    x = np.full(len(vals), i + 1)  # violin positions are 1-indexed
                    jitter = np.random.uniform(-0.15, 0.15, size=len(vals))
                    ax.scatter(x + jitter, vals, color='black', alpha=0.5, zorder=3, s=30, edgecolors='white', linewidths=0.3)
            elif plot_type == "Raincloud":
                # --- Raincloud-Plot mit systematischer Positionierung wie in stats_functions.py ---
                ax.clear()
                import numpy as np
                from scipy import stats
                
                # Daten vorbereiten
                data_x = [np.array(samples[g]) for g in groups]
                n_groups = len(groups)
                
                # Use group_spacing for systematic positioning (consistent with stats_functions.py)
                group_spacing = 0.5  # Default spacing for compact visualization
                positions = [i * group_spacing for i in range(n_groups)]
                
                # Farben wie im Beispiel, aber dynamisch
                boxplots_colors = ["#0f766e", "#1f7a5a", "#b7791f", "#9f3a38", "#1d4ed8", "#7c3aed"]
                violin_colors = ["#7c3aed", "#1d4ed8", "#b7791f", "#9f3a38", "#0f766e", "#1f7a5a"]
                scatter_colors = ["#9f3a38", "#b7791f", "#1d4ed8", "#7c3aed", "#0f766e", "#1f7a5a"]
                
                # Boxplot mit systematischen Positionen
                bp = ax.boxplot(data_x, patch_artist=True, vert=False, positions=positions)
                for patch, color in zip(bp['boxes'], boxplots_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.4)
                    
                # Violinplot mit systematischen Positionen
                vp = ax.violinplot(data_x, points=500, showmeans=False, showextrema=False, showmedians=False, vert=False, positions=positions)
                for idx, b in enumerate(vp['bodies']):
                    pos = positions[idx]
                    # Nur obere Hälfte anzeigen
                    b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1], pos, pos + group_spacing)
                    b.set_color(violin_colors[idx % len(violin_colors)])
                    
                # Scatter mit systematischen Positionen
                for idx, features in enumerate(data_x):
                    pos = positions[idx]
                    y = np.full(len(features), pos - group_spacing * 0.2)  # Offset for scatter
                    idxs = np.arange(len(y))
                    out = y.astype(float)
                    out.flat[idxs] += np.random.uniform(low=-0.05, high=0.05, size=len(idxs))
                    y = out
                    ax.scatter(features, y, s=10, c=scatter_colors[idx % len(scatter_colors)], alpha=0.8)
                    
                # Achsen und Labels mit systematischen Positionen
                ax.set_yticks(positions)
                ax.set_yticklabels(groups, fontsize=fontsize_groupnames)
                ax.set_xlabel("Values", fontsize=fontsize_axis)
                ax.set_ylabel("")
                ax.set_title(title, fontsize=fontsize_title)
                # Layout - make y-axis more compact with systematic positioning
                ax.set_xlim(left=min([min(d) for d in data_x if len(d)>0])-1, right=max([max(d) for d in data_x if len(d)>0])+1)
                y_min = min(positions) - group_spacing * 0.5
                y_max = max(positions) + group_spacing * 0.5
                ax.set_ylim(y_min, y_max)
                ax.grid(False)

            # --- Formatting ---
            if show_title and title:
                ax.set_title(title, fontsize=fontsize_title)
            if plot_config.get('x_label'):
                ax.set_xlabel(plot_config['x_label'], fontsize=fontsize_axis)
            if plot_config.get('y_label'):
                ax.set_ylabel(plot_config['y_label'], fontsize=fontsize_axis)
            plt = get_matplotlib()
            plt.setp(ax.get_xticklabels(), fontsize=fontsize_groupnames)
            plt.setp(ax.get_yticklabels(), fontsize=fontsize_ticks)
            ax.tick_params(axis='x', rotation=45)

            # Grid and minor ticks
            # First make sure all grid is off by default
            ax.grid(False)
            
            # Then conditionally turn on grid only if explicitly requested
            if grid:
                ax.grid(True, axis='y', alpha=0.2, linewidth=gridline_width)
                
            # Enable minor ticks if requested
            if minor_ticks:
                ax.minorticks_on()
                ax.tick_params(which='minor', length=3, color='black', width=0.5)

            # Despine if requested
            if despine:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            for spine in ax.spines.values():
                spine.set_linewidth(axis_linewidth)

            if hasattr(self, 'figure'):
                self.figure.tight_layout()
            if hasattr(self, 'canvas'):
                self.canvas.draw()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error creating preview: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def run_selected_analysis(self):
        """Runs the analysis for the selected plot only."""
        if self.samples is None:
            QMessageBox.warning(self, "Error", "No data loaded.")
            return
            
        current_row = self.plots_list.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "Error", "Please select a plot from the list.")
            return
            
        # Ask for output directory
        output_dir = QFileDialog.getExistingDirectory(self, "Select output directory")
        if not output_dir:
            return
            
        try:
            plot_config = self.plot_configs[current_row]
            self.run_single_analysis(plot_config, output_dir)
            # Success dialog is now handled in run_single_analysis via centralized method
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during analysis: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def run_all_analyses(self):
        """Runs all configured analyses in sequence."""
        if self.samples is None:
            QMessageBox.warning(self, "Error", "No data loaded.")
            return
            
        if not self.plot_configs:
            QMessageBox.warning(self, "Warning", "No plot configurations available.")
            return
        
        # Ask for output directory
        output_dir = QFileDialog.getExistingDirectory(self, "Select output directory")
        if not output_dir:
            return
        
        # Run all analyses and collect files
        success_count = 0
        all_files = []
        original_cwd = os.getcwd()
        
        # Import needed modules
        from stats_functions import AnalysisManager
        import traceback
        
        try:
            os.chdir(output_dir)
            
            for i, plot_config in enumerate(self.plot_configs):
                try:
                    # Manually run analysis logic (like in run_single_analysis but without success dialog)
                    # This avoids recursive calls and dialog conflicts
                    
                    # Check if all groups have data
                    for group in plot_config['groups']:
                        if group not in self.samples or not self.samples[group]:
                            QMessageBox.warning(self, "Warning", f"Group '{group}' has no data or does not exist.")
                            continue

                    # Determine columns to use for analysis
                    value_cols = self.selected_columns if len(self.selected_columns) > 1 else [self.value_cols_combo.currentText()]

                    # Prepare parameters for analyze()
                    kwargs = {
                        'file_path': self.file_path,
                        'group_col': self.group_col_combo.currentText(),
                        'groups': plot_config['groups'],
                        'sheet_name': self.sheet_combo.currentText() if self.sheet_combo.isEnabled() else 0,
                        'width': plot_config['width'],
                        'height': plot_config['height'],
                        'dependent': plot_config['dependent'],
                        'combine_columns': self.combine_columns,
                        'skip_plots': not plot_config.get('create_plot', True),
                        'skip_excel': False,
                        'x_label': plot_config.get('x_label'),
                        'y_label': plot_config.get('y_label'),
                        'title': plot_config.get('title', 'Preview'),
                        'error_type': plot_config.get('error_type', 'sd'),
                        'file_name': plot_config.get('file_name') or "_".join(plot_config['groups']),
                        'show_individual_lines': plot_config.get('show_individual_lines', True),
                        'value_cols': value_cols,
                    }

                    # Merge appearance settings if available
                    if 'appearance_settings' in plot_config:
                        appearance = plot_config['appearance_settings']
                        kwargs.update({
                            'colors': appearance.get('colors', plot_config.get('colors', {})),
                            'hatches': appearance.get('hatches', plot_config.get('hatches', {})),
                            'plot_type': appearance.get('plot_type', 'Bar'),
                            # ... other appearance settings would go here
                        })

                    # Run the analysis via AnalysisManager
                    AnalysisManager.analyze(**kwargs)
                    
                    # Collect files from this analysis
                    base = kwargs['file_name']
                    excel_path = os.path.join(output_dir, f"{base}.xlsx")
                    if os.path.exists(excel_path):
                        all_files.append(excel_path)
                    
                    create_plot = not kwargs.get('skip_plots', True)
                    if create_plot:
                        for ext in ('pdf', 'png'):
                            plot_path = os.path.join(output_dir, f"{base}.{ext}")
                            if os.path.exists(plot_path):
                                all_files.append(plot_path)
                    
                    success_count += 1
                    
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Error in plot {i+1}: {str(e)}")
                    traceback.print_exc()

            # Show single centralized success dialog for all analyses
            if success_count > 0:
                self.show_analysis_success_dialog(f"All plots analysis ({success_count}/{len(self.plot_configs)} successful)", all_files, output_dir)
                
        finally:
            os.chdir(original_cwd)

        # Add this cleanup code
        plt = get_matplotlib()
        plt.close('all')  # Close all matplotlib figures to free memory
    
    def run_single_analysis(self, plot_config, output_dir=None):
        print("DEBUG EXECUTION: run_single_analysis started")
        print(f"DEBUG EXECUTION: plot_config = {plot_config}")
        print(f"DEBUG EXECUTION: output_dir = {output_dir}")

        # Initialize results variable
        results = {}

        if self.samples is None:
            raise ValueError("No data loaded.")

        # Check if all groups have data
        for group in plot_config['groups']:
            if group not in self.samples or not self.samples[group]:
                QMessageBox.warning(self, "Warning", f"Group '{group}' has no data or does not exist.")
                return

        # Determine columns to use for analysis
        value_cols = self.selected_columns if len(self.selected_columns) > 1 else [self.value_cols_combo.currentText()]

        # Prepare parameters for analyze()
        kwargs = {
            'file_path': self.file_path,
            'group_col': self.group_col_combo.currentText(),
            'groups': plot_config['groups'],
            'sheet_name': self.sheet_combo.currentText() if self.sheet_combo.isEnabled() else 0,
            'width': plot_config['width'],
            'height': plot_config['height'],
            'dependent': plot_config['dependent'],
            'combine_columns': self.combine_columns,
            'skip_plots': not plot_config.get('create_plot', True),
            'skip_excel': False,  # always write Excel
            'x_label': plot_config.get('x_label'),
            'y_label': plot_config.get('y_label'),
            'title': plot_config.get('title', 'Preview'),
            'error_type': plot_config.get('error_type', 'sd'),
            'file_name': plot_config.get('file_name') or "_".join(plot_config['groups']),
            'show_individual_lines': plot_config.get('show_individual_lines', True),
            'value_cols': value_cols,
        }

        # Merge appearance settings
        if 'appearance_settings' in plot_config:
            appearance = plot_config['appearance_settings']
            kwargs.update({
                'plot_type': appearance.get('plot_type', 'Bar'),
                'dpi': appearance.get('dpi', 300),
                'aspect': appearance.get('aspect', None),
                
                'font_main': appearance.get('font_main', 'Arial'),
                'font_axis': appearance.get('font_axis', 'Arial'),
                'show_title': appearance.get('show_title', True),
                'fontsize_title': appearance.get('fontsize_title', 11),
                'fontsize_axis': appearance.get('fontsize_axis', 11),
                'fontsize_ticks': appearance.get('fontsize_ticks', 11),
                'fontsize_groupnames': appearance.get('fontsize_groupnames', 11),
                
                'axis_linewidth': appearance.get('axis_linewidth', 0.7),
                'bar_linewidth': appearance.get('bar_linewidth', 1.0),
                'gridline_width': appearance.get('gridline_width', 0.5),
                'grid': appearance.get('grid', False),
                'minor_ticks': appearance.get('minor_ticks', False),
                'logy': appearance.get('logy', False),
                'logx': appearance.get('logx', False),
                'despine': appearance.get('despine', True),
                
                'alpha': appearance.get('alpha', 0.8),
                'bar_edge_color': appearance.get('bar_edge_color', 'black'),
                
                'refline': appearance.get('refline', False),
                'panel_labels': appearance.get('panel_labels', False),
                'value_annotations': appearance.get('value_annotations', False),
                'significance_mode': appearance.get('significance_mode', 'letters'),
                
                # Map to plot_bar function parameter names
                'bar_edge_width': appearance.get('bar_linewidth', 1.0),
                'point_size': 80,  # Larger default point size
                'show_points': True,  # Enable individual points
                'grid_style': 'none' if not appearance.get('grid', False) else 'major',
                'spine_style': 'minimal' if appearance.get('despine', True) else 'default',
                'tick_label_size': appearance.get('fontsize_ticks', 11),
                'x_label_size': appearance.get('fontsize_axis', 11),
                'y_label_size': appearance.get('fontsize_axis', 11),
                'title_size': appearance.get('fontsize_title', 11),
                
                # IMPORTANT: Pass colors from appearance settings
                'colors': appearance.get('colors', plot_config.get('colors', {})),
                'hatches': appearance.get('hatches', plot_config.get('hatches', {})),
            })
        else:
            # If no appearance settings, use colors from plot_config  
            kwargs.update({
                'colors': plot_config.get('colors', {}),
                'hatches': plot_config.get('hatches', {}),
            })

        # Additional factors for advanced ANOVAs
        if plot_config.get('additional_factors'):
            kwargs['additional_factors'] = plot_config['additional_factors']
            
            # Determine which advanced test to use based on configuration
            if plot_config['dependent']:
                # Dependent samples with additional factors
                # This could be mixed ANOVA or repeated measures ANOVA
                # The specific logic to distinguish these needs to be implemented
                # For now, default to mixed ANOVA
                kwargs['test'] = 'mixed_anova'
            else:
                # Independent samples with additional factors = two-way ANOVA
                kwargs['test'] = 'two_way_anova'
        elif plot_config['dependent'] and plot_config.get('needs_subject_selection'):
            # Dependent samples without additional factors but needing subject selection
            # This suggests repeated measures ANOVA with a single within factor
            kwargs['test'] = 'repeated_measures_anova'

        # Pairwise comparisons
        if plot_config.get('comparisons'):
            kwargs['compare'] = [(c['group1'], c['group2']) for c in plot_config['comparisons']]

        original_cwd = os.getcwd()
        try:
            # Change into output directory
            if output_dir:
                print(f"DEBUG: cd {original_cwd} -> {output_dir}")
                os.chdir(output_dir)
                print(f"DEBUG: cwd now {os.getcwd()}")

            # If dependent requires a subject dialog, show it first
            if plot_config['dependent'] and plot_config['needs_subject_selection']:
                dlg = QDialog(self)
                _configure_dialog(dlg, object_name="subjectWithinFactorDialog")
                dlg.setWindowTitle("Select Subject & Within Factor")
                layout = QFormLayout(dlg)
                subject_cb = QComboBox(); subject_cb.addItems(self.df.columns)
                within_cb = QComboBox(); within_cb.addItems(self.df.columns)
                layout.addRow("Subject column:", subject_cb)
                layout.addRow("Within factor:", within_cb)
                btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
                btns.accepted.connect(dlg.accept); btns.rejected.connect(dlg.reject)
                layout.addWidget(btns)
                if dlg.exec_() == QDialog.Accepted:
                    kwargs['subject_column'] = subject_cb.currentText()
                    kwargs['within_column'] = within_cb.currentText()
                else:
                    return

            # Always run the analysis
            print("DEBUG: Calling AnalysisManager.analyze with:")
            for k, v in kwargs.items(): print(f"  {k}: {v}")
            results = AnalysisManager.analyze(**kwargs)
            print("DEBUG: Analysis complete, results keys:", list(results.keys()) if isinstance(results, dict) else type(results))

            # Collect and report output files
            files = []
            base = kwargs['file_name']
            create_plot = kwargs.get('skip_plots', True) == False  # skip_plots=False means create plot
            
            # Always check for Excel file (it should always be created)
            excel_path = os.path.join(os.getcwd(), f"{base}.xlsx")
            if os.path.exists(excel_path):
                files.append(excel_path)
            
            # Only check for plot files if plotting was enabled
            if create_plot:
                for ext in ('pdf', 'png'):
                    plot_path = os.path.join(os.getcwd(), f"{base}.{ext}")
                    if os.path.exists(plot_path):
                        files.append(plot_path)
            
            # Use centralized success dialog
            self.show_analysis_success_dialog("Single plot analysis", files, os.getcwd())

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Analysis error: {e}")
        finally:
            print(f"DEBUG: cd back to {original_cwd}")
            os.chdir(original_cwd)
            plt = get_matplotlib()
            plt.close('all')

    def clear_plot_config_after_analysis(self, analyzed_config):
        """Removes a specific plot configuration after successful analysis"""
        try:
            # Find the config in the list and remove it
            for i, config in enumerate(self.plot_configs):
                if (config.get('groups') == analyzed_config.get('groups') and 
                    config.get('title') == analyzed_config.get('title') and
                    config.get('file_name') == analyzed_config.get('file_name')):
                    
                    # Remove from both list and configs
                    self.plots_list.takeItem(i)
                    self.plot_configs.pop(i)
                    
                    # Clear preview if no plots left
                    if len(self.plot_configs) == 0:
                        if hasattr(self, 'plot_preview_widget') and self.plot_preview_widget:
                            self.plot_preview_widget._show_placeholder()
                        elif hasattr(self, 'figure') and hasattr(self, 'canvas'):
                            self.figure.clear()
                            self.canvas.draw()
                    break
                    
        except Exception as e:
            print(f"Error clearing plot config: {e}")
    
    def show_analysis_success_dialog(self, analysis_type, files, output_dir):
        """Central method for success dialogs after analyses with single clear confirmation"""
        if not files:
            QMessageBox.warning(self, "Warning", 
                f"{analysis_type} completed, but no output files were found in the expected location.\n"
                f"Please check the output directory: {output_dir}")
            return False
        
        # Determine file types
        file_types = []
        if any(f.endswith('.xlsx') for f in files):
            file_types.append("Excel results")
        if any(f.endswith(('.pdf', '.png')) for f in files):
            file_types.append("plots")
        
        # Create success message
        success_msg = f"{analysis_type} completed successfully!\n\n"
        success_msg += f"Created: {', '.join(file_types)}\n"
        success_msg += f"Output directory: {output_dir}\n\n"
        success_msg += "Files:\n" + "\n".join([os.path.basename(f) for f in files])
        success_msg += "\n\nWould you like to clear all plot configurations to start fresh?"
        
        reply = QMessageBox.question(self, "Analysis Complete", success_msg,
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            # Complete reset of the application state
            self.reset_application_state()
            return True
        
        return False
    
    def reset_application_state(self):
        """Complete reset of the application to initial state"""
        try:
            # Clear all plot configurations
            self.plot_configs.clear()
            self.plots_list.clear()
            
            # Clear preview
            if hasattr(self, 'plot_preview_widget') and self.plot_preview_widget:
                self.plot_preview_widget._show_placeholder()
            elif hasattr(self, 'figure') and hasattr(self, 'canvas'):
                self.figure.clear()
                self.canvas.draw()
            
            # Clear temporary appearance settings
            if hasattr(self, 'temp_plot_appearance_settings'):
                self.temp_plot_appearance_settings = None
            
            # Clear group selection
            if hasattr(self, 'groups_list'):
                self.groups_list.clearSelection()
            
            # Close all matplotlib figures
            plt = get_matplotlib()
            plt.close('all')
            
            print("DEBUG: Application state reset to initial state")
            
        except Exception as e:
            print(f"Error resetting application state: {e}")
    
    def auto_generate_preview(self):
        """Automatically creates a preview with all available groups"""
        if not self.samples or not self.available_groups:
            # Clear preview if no data
            if hasattr(self, 'plot_preview_widget') and self.plot_preview_widget:
                self.plot_preview_widget._show_placeholder()
            elif hasattr(self, 'figure') and hasattr(self, 'canvas'):
                self.figure.clear()
                self.canvas.draw()
            return
            
        try:
            # Create a temporary plot config with all available groups
            temp_config = {
                'groups': self.available_groups[:],  # Copy all available groups
                'title': 'Data Preview',
                'x_label': None,
                'y_label': None,
                'error_type': 'sd',
                'create_plot': True
            }
            
            # WICHTIG: Merge mit temporären Plot-Appearance-Einstellungen
            if hasattr(self, 'temp_plot_appearance_settings') and self.temp_plot_appearance_settings:
                temp_config.update(self.temp_plot_appearance_settings)
                print(f"DEBUG: Using temp appearance settings in preview: {self.temp_plot_appearance_settings}")
            else:
                # No user appearance settings, use grayscale for analysis-only preview
                grayscale_colors = {
                    group: ['#2C2C2C', '#4A4A4A', '#686868', '#868686', '#A4A4A4', '#C2C2C2'][i % 6]
                    for i, group in enumerate(self.available_groups)
                }
                temp_config['colors'] = grayscale_colors
                # DEBUG: Removed noisy debug message about grayscale colors
            
            # Use the existing preview_plot logic but with the temp config
            self.preview_auto_plot(temp_config)
            
        except Exception as e:
            print(f"Error in auto_generate_preview: {e}")
            import traceback
            traceback.print_exc()
    
    def preview_auto_plot(self, plot_config):
        """Creates an automatic preview based on a temporary configuration"""
        try:
            # Use the preview widget if available
            if hasattr(self, 'plot_preview_widget') and self.plot_preview_widget:
                # Set data in the preview widget
                if hasattr(self, 'groups') and hasattr(self, 'samples'):
                    self.plot_preview_widget.set_data(self.groups, self.samples)
                    # Update the plot with the configuration
                    self.plot_preview_widget.update_plot(plot_config)
                else:
                    # Show placeholder if no data
                    self.plot_preview_widget._show_placeholder()
            else:
                print("Warning: plot_preview_widget not available")
            
        except Exception as e:
            print(f"Error in preview_auto_plot: {e}")
            import traceback
            traceback.print_exc()
    
    def update_preview_on_selection_change(self):
        """Updates the preview based on the current group selection"""
        if not self.samples:
            return
            
        # Get selected groups from the list
        selected_items = self.groups_list.selectedItems()
        if selected_items:
            # Use only selected groups for preview
            selected_groups = [item.text() for item in selected_items]
        else:
            # If nothing selected, show all groups
            selected_groups = self.available_groups[:]
        
        if selected_groups:
           
            temp_config = {
                'groups': selected_groups,
                'title': f'Preview: {", ".join(selected_groups)}' if len(selected_groups) <= 3 else f'Preview: {len(selected_groups)} groups',
                'x_label': None,
                'y_label': None,
                'error_type': 'sd',
                'create_plot': True
            }
            
            # Add grayscale colors for selection preview (analysis-only context)
            if not hasattr(self, 'temp_plot_appearance_settings') or not self.temp_plot_appearance_settings:
                grayscale_colors = {
                    group: ['#2C2C2C', '#4A4A4A', '#686868', '#868686', '#A4A4A4', '#C2C2C2'][i % 6]
                    for i, group in enumerate(selected_groups)
                }
                temp_config['colors'] = grayscale_colors
            
            self.preview_auto_plot(temp_config)
    
    # Neue Methode für die Anzeige einer Hilfefunktion zu abhängigen Stichproben
    def show_dependent_samples_help(self):
        QMessageBox.information(
            self,
            "Help for Dependent Samples",
            "<h3>When are samples dependent?</h3>"
            "<p>Dependent samples arise when:</p>"
            "<ul>"
            "<li>Measurements are taken on the <b>same subject</b> at different time points</li>"
            "<li>Measurements are naturally paired (e.g. left and right eye)</li>"
            "<li>Experiments are conducted with repeated measurements</li>"
            "</ul>"
            "<h3>Data structure for dependent tests</h3>"
            "<p>For dependent tests, each group must:</p>"
            "<ul>"
            "<li>Contain the <b>same number</b> of measurements</li>"
            "<li>Have measurements in <b>matching order</b></li>"
            "</ul>"
            "<p>Example: Measurement 1 in group A and measurement 1 in group B must be from the same subject</p>"
            "<h3>Available tests</h3>"
            "<ul>"
            "<li><b>Two groups:</b> Paired t-test or Wilcoxon signed-rank test</li>"
            "<li><b>More than two groups:</b> Repeated Measures ANOVA or Friedman test</li>"
            "</ul>"
        )
    
    def show_getting_started_help(self):
        """Shows a comprehensive getting started guide for first-time users."""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextBrowser, QPushButton
        
        dlg = QDialog(self)
        _configure_dialog(dlg, object_name="gettingStartedHelpDialog")
        dlg.setWindowTitle("Getting Started with BioMedStatX")
        dlg.resize(1000, 800)
        layout = QVBoxLayout(dlg)
        
        browser = QTextBrowser()
        browser.setObjectName("helpDialogBrowser")
        browser.setHtml("""
            <h2>Getting Started with BioMedStatX</h2>
            <p><i>A step-by-step guide for first-time users</i></p>
            
            <h3>Step 1: Prepare Your Data</h3>
            <p>BioMedStatX works with <b>Excel files</b> (.xlsx or .xls). Your data should be organized in columns, in a long format:</p>
            <ul>
                <li><b>Group column:</b> Contains group names (e.g., "Control", "Treatment A", "Treatment B")</li>
                <li><b>Value column:</b> Contains the measurements you want to analyze</li>
                <li><b>Subject column (optional):</b> For dependent/paired data - unique identifiers for each subject</li>
            <p>Take a look into the template excel file, if you need an idea of how to structure your data for the different types of analysis</p>
            </ul>

            
            <h3>Step 2: Upload Your Excel File</h3>
            <p>1. Click the <b>"Browse"</b> button in the main window</p>
            <p>2. Select your Excel file from your computer</p>
            <p>3. The file path will appear in the text field</p>
            
            <h3>Step 3: Select Your Worksheet</h3>
            <p>If your Excel file has multiple sheets:</p>
            <ul>
                <li>Use the <b>Sheet dropdown</b> to choose the correct worksheet</li>
                <li>The program will automatically detect available sheets</li>
            </ul>
            
            <h3>Step 4: Configure Your Columns</h3>
            <p>Tell the program which columns contain your data:</p>
            <ul>
                <li><b>Group Column:</b> Select the column with your group names</li>
                <li><b>Value Column:</b> Select the column with your measurements</li>
            </ul>
            
            <h3>Step 5: Choose Your Analysis Type</h3>
            
            <h4>A) Basic Statistical Tests (Automatic Selection)</h4>
            <p>Click <b>"Run Statistical Analysis"</b> for automatic test selection:</p>
            <ul>
                <li><b>2 groups:</b> t-test or Mann-Whitney U test</li>
                <li><b>3+ groups:</b> One-way ANOVA or Kruskal-Wallis test</li>
                <li>The program automatically chooses parametric vs. non-parametric based on your data</li>
            </ul>
            
            <h4>B) Complex ANOVA Designs</h4>
            <p>For repeated and multi-factor designs, map your columns in Smart Mapping and then run <b>Start Auto Analysis</b>:</p>
            <ul>
                <li><b>Repeated Measures ANOVA:</b> Same subjects measured multiple times</li>
                <li><b>Two-Way ANOVA:</b> Two independent factors (e.g., treatment × gender)</li>
                <li><b>Mixed ANOVA:</b> Combination of between- and within-subject factors</li>
            </ul>
            <p>Need a template? Open <b>Help -> Help Hub (Recipes)</b> and copy the long-format example directly into Excel.</p>
            
            <h3>Step 6: Additional Analysis Options</h3>
            
            <h4>Outlier Detection</h4>
            <p>After uploading your data, you can:</p>
            <ul>
                <li>Use <b>Analysis → Detect Outliers</b> to identify unusual data points</li>
                <li>Choose from multiple outlier detection methods</li>
                <li>Decide whether to keep or remove outliers</li>
            </ul>
            
            <h4>Multi-Dataset Analysis</h4>
            <p>To compare multiple related datasets:</p>
            <ul>
                <li>Click <b>Multiple columns...</b> and click <b>Separate analysis per dataset with shared excel file</b> and all the groups you want to analyse
                <li>Click <b>"Multi-Dataset Analysis"</b> in the main window</li>
                <li>Each dataset gets its own analysis and plot</li>
                <li>Results are combined in a single Excel report</li>
            </ul>
            
            <h3>Step 7: Customize Your Results</h3>
            
            <h4>Plot Customization</h4>
            <ul>
                <li>Choose between <b>Bar, Box, Violin, or Strip plots</b></li>
                <li>Customize colors, fonts, and error bars</li>
                <li>Add statistical significance annotations</li>
            </ul>
            
            <p><b>Need more help?</b> Check the other help sections for specific topics!</p>
        """)
        
        layout.addWidget(browser)
        
        btn = QPushButton("OK")
        btn.clicked.connect(dlg.accept)
        layout.addWidget(btn)
        
        dlg.exec_()
    
    def closeEvent(self, event):
        """Cleanup temporäre Daten beim Schließen des Programms"""
        print("DEBUG: Cleaning up temporary plot appearance settings...")
        self.temp_plot_appearance_settings = None
        try:
            if hasattr(self, 'decision_tree_panel') and self.decision_tree_panel is not None:
                self.decision_tree_panel.cleanup()
        except Exception as close_exc:
            print(f"DEBUG: Decision tree cleanup warning during close: {close_exc}")
        super().closeEvent(event)
    
    def run_multi_dataset_analysis(self):
        """Runs separate analyses for multiple datasets, 
        with individual plot configuration and a shared Excel file."""
        print("DEBUG MULTI: ENTERED run_multi_dataset_analysis()")
        print("DEBUG MULTI:   self.multi_dataset_analysis =", getattr(self, "multi_dataset_analysis", None))
        print("DEBUG MULTI:   self.selected_columns =", getattr(self, "selected_columns", None))
        
        if not hasattr(self, 'multi_dataset_analysis') or not self.multi_dataset_analysis:
            QMessageBox.warning(self, "Warning", "Please select multi-dataset analysis in the column selection dialog first.")
            return
            
        if len(self.selected_columns) <= 1:
            QMessageBox.warning(self, "Warning", "Multi-dataset analysis requires multiple selected datasets.")
            return
        print("DEBUG MULTI: Passed all pre-checks.  → proceed with multi-dataset loop")
        print("DEBUG MULTI:   selected_columns =", self.selected_columns)
        print("DEBUG MULTI:   available_groups =", self.available_groups)
        try:
            # Ask for output directory
            output_dir = QFileDialog.getExistingDirectory(
                self, "Select output directory for multi-dataset analysis")
            if not output_dir:
                print("No output directory selected")
                return

            # Select groups for analysis
            print("Opening group dialog...")
            dialog = GroupSelectionDialog(self.available_groups, self)
            if dialog.exec_() != QDialog.Accepted:
                print("Group dialog cancelled")
                return
                
            selected_groups = dialog.get_selected_groups()
            if not selected_groups:
                QMessageBox.warning(self, "Warning", "Please select at least one group.")
                return

            all_results = {}
            plot_configs = {}

            # Remember current working directory
            original_cwd = os.getcwd()
            
            try:
                os.chdir(output_dir)
                print(f"Changed to output directory: {output_dir}")

                # ── PHASE A: Analyze all datasets first (no dialog yet) ─────────────────
                progress = QMessageBox()
                progress.setWindowTitle("Analyzing datasets...")
                progress.setText(f"Pre-analyzing {len(self.selected_columns)} dataset(s). Please wait…")
                progress.setStandardButtons(QMessageBox.NoButton)
                progress.show()
                QApplication.processEvents()

                for i, column in enumerate(self.selected_columns):
                    progress.setText(f"Analyzing dataset {i+1}/{len(self.selected_columns)}: {column}")
                    QApplication.processEvents()
                    print(f"DEBUG MULTI: Pre-analyzing '{column}' ({i+1}/{len(self.selected_columns)})")
                    try:
                        stat_kwargs = {
                            'file_path': self.file_path,
                            'group_col': self.group_col_combo.currentText(),
                            'groups': selected_groups,
                            'sheet_name': self.sheet_combo.currentText() if self.sheet_combo.isEnabled() else 0,
                            'value_cols': [column],
                            'combine_columns': False,
                            'skip_plots': True,   # statistics only — no rendering yet
                            'skip_excel': True,
                            'dataset_name': column,
                        }
                        results = AnalysisManager.analyze(**stat_kwargs)
                        if isinstance(results, dict) and results.get('error'):
                            print(f"WARNING: analysis error for '{column}': {results['error']}")
                            all_results[column] = None
                        else:
                            all_results[column] = results
                            print(f"✓ '{column}' analyzed — pairwise: "
                                  f"{len((results or {}).get('pairwise_comparisons', []))} comparisons")
                    except Exception as e:
                        print(f"ERROR pre-analyzing '{column}': {e}")
                        traceback.print_exc()
                        all_results[column] = None  # isolated failure — others continue

                try:
                    progress.close()
                except Exception:
                    pass

                # ── PHASE B: Open PlotAestheticsDialog per dataset (analysis-first) ───
                base_filename = ""
                if hasattr(self, 'file_path') and self.file_path:
                    base_filename = os.path.splitext(os.path.basename(self.file_path))[0]

                n_valid = sum(1 for r in all_results.values() if r is not None)
                dialog_idx = 0
                for column, result in all_results.items():
                    if result is None:
                        print(f"Skipping dialog for '{column}' (analysis failed)")
                        continue
                    dialog_idx += 1
                    samples_for_dlg = result.get('raw_data') or result.get('samples') or {}
                    default_filename = f"{base_filename}_{column}_analyzed" if base_filename else f"{column}_analyzed"
                    dlg = PlotAestheticsDialog(
                        groups=selected_groups,
                        samples=samples_for_dlg,
                        analysis_result=result,
                        parent=self,
                        default_filename=default_filename,
                        show_export_controls=True,
                    )
                    # Pre-fill sensible file name
                    if hasattr(dlg, 'file_name_edit'):
                        dlg.file_name_edit.setText(f"{column}_analysis")
                    dlg.setWindowTitle(f"Configure plot for '{column}' ({dialog_idx}/{n_valid})")
                    if dlg.exec_() != QDialog.Accepted:
                        print(f"Dialog for '{column}' cancelled — skipping")
                        continue
                    plot_config = dlg.get_config()
                    # Guarantee original group keys are preserved
                    plot_config['groups'] = list(selected_groups)
                    plot_configs[column] = plot_config
                    print(f"Config for '{column}' saved")

                if not plot_configs:
                    QMessageBox.warning(self, "Aborted", "No datasets were configured for export.")
                    return

                # ── PHASE C: Render configured datasets ─────────────────────────────────
                progress = QMessageBox()
                progress.setWindowTitle("Rendering plots…")
                progress.setText("Rendering plots. Please wait…")
                progress.setStandardButtons(QMessageBox.NoButton)
                progress.show()
                QApplication.processEvents()

                for i, (column, plot_config) in enumerate(plot_configs.items()):
                    progress.setText(f"Rendering dataset {i+1}/{len(plot_configs)}: {column}")
                    QApplication.processEvents()
                    print(f"Starting render for '{column}'…")
                    try:
                        kwargs = {
                            'file_path': self.file_path,
                            'group_col': self.group_col_combo.currentText(),
                            'groups': plot_config['groups'],
                            'sheet_name': self.sheet_combo.currentText() if self.sheet_combo.isEnabled() else 0,
                            'value_cols': [column],
                            'combine_columns': False,
                            'width': plot_config.get('width', 12),
                            'height': plot_config.get('height', 10),
                            'dependent': plot_config.get('dependent', False),
                            'skip_plots': not plot_config.get('create_plot', True),
                            'skip_excel': True,
                            'x_label': plot_config.get('x_label'),
                            'y_label': plot_config.get('y_label'),
                            'title': plot_config.get('title', column),
                            'error_type': plot_config.get('error_type', 'sd'),
                            'file_name': plot_config.get('file_name', f"{column}_analysis"),
                            'dataset_name': column,
                            'plot_type': plot_config.get('plot_type', 'Bar'),
                            'dpi': plot_config.get('dpi', 300),
                            'fontsize_title': plot_config.get('fontsize_title', 12),
                            'fontsize_axis': plot_config.get('fontsize_axis', 9),
                            'fontsize_ticks': plot_config.get('fontsize_ticks', 7),
                            'logy': plot_config.get('logy', False),
                            'logx': plot_config.get('logx', False),
                            'despine': plot_config.get('despine', True),
                            'alpha': plot_config.get('alpha', 0.8),
                            'bar_edge_color': plot_config.get('bar_edge_color', 'black'),
                        }
                        # Colors — use original group keys for matching (never display labels)
                        colors_dict = plot_config.get('colors', {})
                        kwargs['colors'] = [
                            colors_dict.get(group, DEFAULT_COLORS[j % len(DEFAULT_COLORS)])
                            for j, group in enumerate(plot_config['groups'])
                        ]
                        # Hatches
                        hatches_dict = plot_config.get('hatches', {})
                        if hatches_dict:
                            kwargs['hatches'] = [hatches_dict.get(group, '') for group in plot_config['groups']]
                        # Pass pairwise results from pre-analysis for significance display
                        pre_result = all_results.get(column)
                        if pre_result:
                            kwargs['pairwise_results'] = pre_result.get('pairwise_comparisons', [])

                        start_time = time.time()
                        results = AnalysisManager.analyze(**kwargs)
                        analysis_time = time.time() - start_time

                        if isinstance(results, dict):
                            if results.get('error'):
                                print(f"WARNING: render error for '{column}': {results['error']}")
                            else:
                                all_results[column] = results  # update with rendered results
                        print(f"✓ Render for '{column}' done in {analysis_time:.2f}s")

                        # --- Export with font embedding / metadata if requested ---
                        if plot_config.get('create_plot', True) and (
                            plot_config.get('embed_fonts', False) or plot_config.get('add_metadata', False)
                        ):
                            filename = plot_config.get('file_name', f"{column}_analysis")
                            filetype = "pdf"
                            out_path = os.path.join(os.getcwd(), f"{filename}.{filetype}")
                            fig = results.get('figure', None) if isinstance(results, dict) else None
                            if fig is None:
                                import matplotlib.pyplot as _plt
                                fig = _plt.gcf()
                            DataVisualizer.export_with_metadata(
                                fig, out_path,
                                metadata={"Title": plot_config.get('title', column), "Description": ""},
                                embed_fonts=plot_config.get('embed_fonts', True),
                                dpi=plot_config.get('dpi', 300),
                                filetype=filetype
                            )
                    except Exception as e:
                        QMessageBox.critical(self, "Error", f"Error rendering '{column}': {e}")
                        traceback.print_exc()

                try:
                    progress.close()
                except Exception:
                    pass

                # Filter all_results to only successfully completed datasets
                all_results = {k: v for k, v in all_results.items() if v is not None}

                if all_results:
                    print("DEBUG MULTI: About to call export_multi_dataset_results()")
                    excel_path = os.path.join(output_dir, "All_Datasets_Analysis.xlsx")
                    print(f"DEBUG MULTI: Excel path will be: {excel_path}")
                    export_result = ExportDispatcher.export_multi_dataset_results(all_results, excel_path)
                    if export_result.get("warning"):
                        print(f"WARNING: {export_result['warning']}")
                    print("DEBUG MULTI: export_multi_dataset_results() completed successfully")

                    # Collect all output files for centralized success dialog
                    files = []
                    excel_path = os.path.join(output_dir, "All_Datasets_Analysis.xlsx")
                    if os.path.exists(excel_path):
                        files.append(excel_path)

                    # Check for plot files if any plots were created
                    any_plots = any(plot_config.get('create_plot', True) for plot_config in plot_configs.values())
                    if any_plots:
                        for column, plot_config in plot_configs.items():
                            if plot_config.get('create_plot', True):
                                file_name = plot_config.get('file_name', f"{column}_analysis")
                                for ext in ('pdf', 'png'):
                                    plot_path = os.path.join(output_dir, f"{file_name}.{ext}")
                                    if os.path.exists(plot_path):
                                        files.append(plot_path)

                    # Use centralized success dialog
                    analysis_type = f"Multi-dataset analysis ({len(all_results)} datasets: {', '.join(all_results.keys())})"
                    self.show_analysis_success_dialog(analysis_type, files, output_dir)

                    # Update cockpit to reflect multi-dataset completion
                    n = len(all_results)
                    names = ", ".join(all_results.keys())
                    multi_summary = {
                        "subtitle": f"Multi-dataset analysis complete — {n} dataset(s) analyzed.",
                        "metric_normality": "Per dataset (see sheets)",
                        "metric_variance": "Per dataset (see sheets)",
                        "inference_main_test": f"{n} dataset(s) analyzed:\n{names}\n\nSee Excel file for full results.",
                        "inference_effect_size": "Per dataset (reported in dedicated result sheets)",
                        "context_design": "Multi-dataset mode — each dependent variable is analyzed with its own inferred model.",
                        "context_sample_overview": f"Datasets analyzed: {n}",
                        "context_analysis_scope": "Results consolidated into one shared Excel workbook.",
                    }
                    self.result_cockpit.set_summary(multi_summary, enable_plot=False, enable_output=True)
                    self.current_output_dir = output_dir
                else:
                    print("DEBUG MULTI: all_results is empty, skipping export")
                    QMessageBox.warning(self, "No Results", "No analysis results were generated.")

            except Exception as e:
                print(f"ERROR in main flow of multi-dataset analysis: {str(e)}")
                import traceback
                traceback.print_exc()
                QMessageBox.critical(self, "Critical error", 
                                f"An unexpected error occurred: {str(e)}")

        except Exception as e:
            print(f"CRITICAL ERROR in run_multi_dataset_analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Critical error", 
                            f"A serious error occurred: {str(e)}")
    
        os.chdir(original_cwd)
            
    def configure_two_way_anova(self):
        """Configure Two-Way ANOVA"""
        # IMPORTANT: Use a local variable for the status
        current_factors = None

        # Create and execute dialog
        dialog = TwoWayAnovaDialog(self.available_groups, self)
        if dialog.exec_() == QDialog.Accepted:
            # Store status only in local variable
            current_factors = dialog.get_factor_data()

            # Use status only for this action, not as an instance variable
            if current_factors:
                # Immediately run analysis with the factors
                self.run_analysis_with_factors(current_factors)

    def run_analysis_with_factors(self, factors):
        """Run analysis with the given factors"""
        try:
            # Prepare parameters for the analysis
            params = self.get_analysis_params()

            # Explicitly add factors for this call
            params['additional_factors'] = factors

            # Run analysis
            results = AnalysisManager.analyze(**params)

            # Show results
            self.display_results(results)

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error during analysis with factors: {str(e)}")

    def run_pairwise_comparison(self):
        """Perform pairwise comparison"""
        dialog = PairwiseComparisonDialog(self.available_groups, self)
        if dialog.exec_() == QDialog.Accepted:
            comparison_data = dialog.get_comparison()
            if comparison_data:
                # Use the existing run_direct_comparison
                self.run_direct_comparison(comparison_data)

    def run_outlier_detection(self):
        """Run outlier detection analysis"""
        # Check if data is loaded
        if self.df is None:
            QMessageBox.warning(self, "Error", "No data loaded. Please load a file first.")
            return
        
        try:
            # Create and show the outlier detection dialog
            dialog = OutlierDetectionDialog(self.df, self)
            if dialog.exec_() == QDialog.Accepted:
                config = dialog.get_config()
                if config is None:
                    return
                
                # Create progress dialog
                progress = QMessageBox()
                progress.setWindowTitle("Outlier Detection")
                progress.setText("Running outlier detection analysis...")
                progress.setStandardButtons(QMessageBox.NoButton)
                progress.show()
                QApplication.processEvents()
                
                try:
                    if config['is_multi_dataset']:
                        # Multi-dataset analysis
                        results = OutlierDetector.run_multi_dataset_outlier_detection(
                            df=self.df,
                            group_col=config['group_column'],
                            dataset_columns=config['dataset_columns'],
                            # alpha parameter removed
                            iterate=config['iterate'],
                            run_grubbs=config['run_grubbs'], 
                            run_modz=config['run_modz'],
                            output_path=config['output_file']
                        )
                        
                        # Show multi-dataset results
                        summary = f"Multi-dataset outlier detection completed!\n\n"
                        summary += f"Analyzed {len(config['dataset_columns'])} datasets:\n"
                        summary += f"{', '.join(config['dataset_columns'])}\n\n"
                        summary += f"Results saved to: {config['output_file']}"
                        
                    else:
                        # Single dataset analysis (existing code)
                        detector = OutlierDetector(
                            df=self.df.copy(),
                            group_col=config['group_column'],
                            value_col=config['value_column']
                        )

                        if config['run_modz']:
                            detector.run_mod_z_score(threshold=3.5, iterate=config['iterate'])
                        
                        if config['run_grubbs']:
                            detector.run_grubbs_test(alpha=0.05, iterate=config['iterate'])
                        
                        detector.save_results(config['output_file'])
                        
                        outlier_count = 0
                        if 'ModZ_Outlier' in detector.df.columns:
                            outlier_count += detector.df['ModZ_Outlier'].sum()
                        if 'Grubbs_Outlier' in detector.df.columns:
                            outlier_count += detector.df['Grubbs_Outlier'].sum()
                        
                        summary = f"Outlier detection completed!\n\n"
                        summary += f"Dataset: {config['value_column']}\n"
                        summary += f"Total outliers found: {outlier_count}\n\n"
                        summary += f"Results saved to: {config['output_file']}"
                    
                    progress.close()
                    
                    # Collect output files for centralized success dialog
                    files = []
                    if os.path.exists(config['output_file']):
                        files.append(config['output_file'])
                    
                    # Use centralized success dialog
                    output_dir = os.path.dirname(config['output_file'])
                    self.show_analysis_success_dialog("Outlier detection", files, output_dir)
                    
                except Exception as e:
                    progress.close()
                    QMessageBox.critical(self, "Error", f"Error during outlier detection: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error opening outlier detection dialog: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def setup_updater(self):
        """Initialize the auto-updater"""
        if UPDATE_AVAILABLE:
            self.updater = AutoUpdater(self)
            # Auto-check for updates 5 seconds after startup
            from PyQt5.QtCore import QTimer
            startup_timer = QTimer()
            startup_timer.singleShot(5000, lambda: self.updater.check_for_updates(silent=True))
        else:
            self.updater = None
    
    def check_for_updates(self):
        """Manual update check triggered from menu"""
        if UPDATE_AVAILABLE and self.updater:
            self.updater.check_for_updates(silent=False)
        else:
            QMessageBox.information(
                self,
                "Updates Not Available",
                "Update functionality is not available in this build.\n\n"
                "Please check the GitHub repository manually for updates:\n"
                "https://github.com/philippkrumm/BioMedStatX---Code"
            )


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


def _load_auto_pilot_stylesheet():
    stylesheet_paths = [
        resource_path("assets/BioMedStatX_2_0.qss"),
        resource_path("assets/StyleSheet.qss"),
    ]
    for path in stylesheet_paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                return handle.read()
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
    browse_button = QPushButton("Load Excel / CSV")
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

    preview_label = QLabel("Table Preview")
    preview_label.setObjectName("sectionLabel")
    left_layout.addWidget(preview_label)
    self.preview_table = QTableWidget(0, 0)
    self.preview_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    self.preview_table.setSelectionMode(QAbstractItemView.NoSelection)
    self.preview_table.setAlternatingRowColors(True)
    left_layout.addWidget(self.preview_table, 1)

    cards_label = QLabel("Excel Headers")
    cards_label.setObjectName("sectionLabel")
    left_layout.addWidget(cards_label)
    cards_scroll = QScrollArea()
    cards_scroll.setWidgetResizable(True)
    cards_scroll.setObjectName("headerCardScroll")
    self.header_cards_widget = QWidget()
    self.header_cards_layout = QVBoxLayout(self.header_cards_widget)
    self.header_cards_layout.setContentsMargins(0, 0, 0, 0)
    self.header_cards_layout.setSpacing(10)
    self.header_cards_layout.addStretch()
    cards_scroll.setWidget(self.header_cards_widget)
    left_layout.addWidget(cards_scroll, 1)

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

    self.start_analysis_button = QPushButton("Start Auto Analysis")
    self.start_analysis_button.setObjectName("primaryButton")
    self.start_analysis_button.clicked.connect(self.determine_and_run_test)
    self.start_analysis_button.setEnabled(False)
    center_layout.addWidget(self.start_analysis_button)

    center_layout.addStretch()

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
    _apply_elevation(self.result_cockpit)
    _apply_elevation(self.decision_tree_panel)

    splitter.addWidget(right_panel)
    splitter.setSizes([450, 430, 520])

    central_scroll.setWidget(central_widget)
    self.setCentralWidget(central_scroll)

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
        return

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

    if self.df is None:
        self.header_cards_layout.addStretch()
        return

    for column_name in self.df.columns:
        series = self.df[column_name]
        column_kind = _infer_column_kind(series)
        preview_values = [str(value) for value in series.dropna().head(3).tolist()]
        preview_text = "Preview: " + (", ".join(preview_values) if preview_values else "No preview values")
        card = DraggableColumnCard(column_name, column_kind, preview_text)
        self.header_cards_layout.addWidget(card)
    self.header_cards_layout.addStretch()


def _ap_apply_mapping_heuristics(self):
    for bucket in (self.dv_bucket, self.factor1_bucket, self.factor2_bucket, self.subject_bucket):
        bucket.clear_assignments()

    if self.df is None:
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

    if subject_column:
        self.subject_bucket.assign_column(subject_column, _infer_column_kind(self.df[subject_column]))


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
        from correlation_models import _is_continuous as _corr_is_continuous
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


def _ap_browse_file(self):
    file_path, _ = QFileDialog.getOpenFileName(
        self, "Open Excel or CSV file", "",
        "Excel files (*.xlsx *.xls);;CSV files (*.csv);;All files (*.*)"
    )
    if file_path:
        self.file_path = file_path
        self.load_file()


def _ap_load_file(self):
    if not self.file_path:
        return
    try:
        if self.file_path.endswith('.csv'):
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
        self._maybe_pivot()
        self.numeric_columns = [column for column in self.df.columns if pd.api.types.is_numeric_dtype(self.df[column])]
        self._refresh_preview_table()
        self._rebuild_column_cards()
        self._apply_mapping_heuristics()
        self._set_workflow_state("map", "Dataset loaded")
        self.result_cockpit.clear()
        self.decision_tree_panel.show_placeholder("Map the columns, then run the auto-pilot analysis.")
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
            from correlation_models import _is_continuous as _corr_is_continuous
            if _corr_is_continuous(analysis_df, factor_columns[0]):
                if covariate_columns:
                    context["inferred_test"] = "linear_regression"
                else:
                    context["inferred_test"] = "correlation"
                context["x_variable"] = factor_columns[0]
        except Exception:
            pass  # correlation_models not available — skip silently

    if context["mode"] == "single" and len(dv_columns) != 1:
        raise ValueError("Single mode requires exactly one measurement column.")
    if context["mode"] == "multi" and len(dv_columns) < 2:
        raise ValueError("Multi mode requires at least two measurement columns (for example two or more genes).")
    if context["mode"] == "multi" and context["inferred_test"] in {"independent_ttest", "paired_ttest", "logistic_regression"}:
        raise ValueError("Multi mode is restricted to ANOVA-capable designs.")

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
        "correlation": "Korrelationsanalyse (Spearman/Pearson)",
        "linear_regression": "Lineare Regression (OLS)",
    }
    return labels.get(context["inferred_test"], context["inferred_test"])


def _ap_execute_single_analysis(self, context, dv_column, output_dir, skip_plots=True, title_suffix=None):
    if not self.file_path:
        raise ValueError("No input file selected.")

    base_name = os.path.splitext(os.path.basename(self.file_path))[0]
    file_base = os.path.join(output_dir, f"{_safe_file_slug(base_name)}_{_safe_file_slug(dv_column)}")
    group_labels = context.get("group_labels", [])
    colors = [DEFAULT_COLORS[index % len(DEFAULT_COLORS)] for index, _ in enumerate(group_labels)]
    hatches = [DEFAULT_HATCHES[index % len(DEFAULT_HATCHES)] for index, _ in enumerate(group_labels)]
    single_context = dict(context)
    single_context["dv_columns"] = [dv_column]
    single_context["current_dv"] = dv_column
    if getattr(self, '_wide_format_info', None) is not None:
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
        skip_excel=False,
        analysis_context=single_context,
        subject_column=context.get("subject_column"),
    )
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
        if equal_variance is True:
            return "Homogeneous"
        if equal_variance is False:
            return "Heterogeneous"
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
        "r": "r (rank correlation)",
        "eta_squared": "Eta-squared",
        "partial_eta_squared": "Partial eta-squared",
        "epsilon_squared": "Epsilon-squared",
        "kendall_w": "Kendall's W",
        "rank_biserial_r": "Rank-biserial r",
        "ICC": "ICC",
        "AUC": "AUC",
    }
    type_label = labels.get(effect_size_type, effect_size_type.replace("_", " ").title() if effect_size_type else "Effect size")
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
    self.result_cockpit.set_summary(summary, enable_plot=True, enable_output=bool(output_dir))
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

    output_dir = QFileDialog.getExistingDirectory(self, "Select output directory for analysis")
    if not output_dir:
        return

    self._set_workflow_state("analyze", "Running analysis", running=True)
    self.mapping_feedback_label.setText("Auto-pilot is analyzing the mapped design.")
    self.decision_tree_panel.show_placeholder("Analyzing data and tracing the statistical decision path...")
    QApplication.processEvents()

    try:
        if context["mode"] == "single":
            result = self._execute_single_analysis(context, context["dv_columns"][0], output_dir, skip_plots=True)
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

            base_name = _safe_file_slug(os.path.splitext(os.path.basename(self.file_path))[0])
            combined_excel = os.path.join(output_dir, f"{base_name}_multi_dataset_results.xlsx")
            export_result = ExportDispatcher.export_multi_dataset_results(all_results, combined_excel)
            if export_result.get("warning"):
                print(f"WARNING: {export_result['warning']}")

            lead_dv = context["dv_columns"][0]
            lead_result = all_results[lead_dv]
            lead_context = dict(context)
            lead_context["dv_columns"] = [lead_dv]
            lead_context["current_dv"] = lead_dv
            self._render_result_summary(
                lead_context,
                lead_result,
                output_dir,
                subtitle=f"Multi-dataset analysis completed for {len(all_results)} dependent variables. Combined Excel: {os.path.basename(combined_excel)}"
            )
            self.current_multi_results = all_results
            self.current_analysis_result["combined_excel"] = combined_excel

    except Exception as exc:
        self._set_workflow_state("map", "Analysis failed")
        self.decision_tree_panel.show_placeholder(f"Analysis failed: {exc}")
        QMessageBox.critical(self, "Analysis Error", str(exc))


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
            skip_excel=False,
            analysis_context=context,
            subject_column=context.get("subject_column"),
            plot_type=appearance.get("plot_type", "Bar"),
            dpi=appearance.get("dpi", 300),
            colors_override=plot_config["colors"],
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


StatisticalAnalyzerApp.init_ui = _ap_init_ui
StatisticalAnalyzerApp.browse_file = _ap_browse_file
StatisticalAnalyzerApp.load_file = _ap_load_file
StatisticalAnalyzerApp.load_sheet = _ap_load_sheet
StatisticalAnalyzerApp._refresh_preview_table = _ap_refresh_preview_table
StatisticalAnalyzerApp._rebuild_column_cards = _ap_rebuild_column_cards
StatisticalAnalyzerApp._apply_mapping_heuristics = _ap_apply_mapping_heuristics
StatisticalAnalyzerApp.update_mode_constraints = _ap_update_mode_constraints
StatisticalAnalyzerApp.on_mapping_changed = _ap_on_mapping_changed
StatisticalAnalyzerApp._set_workflow_state = _ap_set_workflow_state
StatisticalAnalyzerApp._is_binary_outcome_for_help = _ap_is_binary_outcome_for_help
StatisticalAnalyzerApp._is_continuous_factor1_for_help = _ap_is_continuous_factor1_for_help
StatisticalAnalyzerApp._resolve_help_recipe_for_bucket = _ap_resolve_help_recipe_for_bucket
StatisticalAnalyzerApp._ap_get_available_analysis_groups = _ap_get_available_analysis_groups
StatisticalAnalyzerApp._ap_update_analysis_group_selection_ui = _ap_update_analysis_group_selection_ui
StatisticalAnalyzerApp.open_analysis_group_selector = _ap_open_analysis_group_selector
StatisticalAnalyzerApp._build_analysis_context = _ap_build_analysis_context
StatisticalAnalyzerApp._detected_test_label = _ap_detected_test_label
StatisticalAnalyzerApp._execute_single_analysis = _ap_execute_single_analysis
StatisticalAnalyzerApp._format_assumptions = _ap_format_assumptions
StatisticalAnalyzerApp._extract_normality_metric = _ap_extract_normality_metric
StatisticalAnalyzerApp._extract_variance_metric = _ap_extract_variance_metric
StatisticalAnalyzerApp._format_main_test_metric = _ap_format_main_test_metric
StatisticalAnalyzerApp._format_effect_size_metric = _ap_format_effect_size_metric
StatisticalAnalyzerApp._is_ttest_result = _ap_is_ttest_result
StatisticalAnalyzerApp._format_rationale = _ap_format_rationale
StatisticalAnalyzerApp._format_posthoc_status = _ap_format_posthoc_status
StatisticalAnalyzerApp._format_context_design = _ap_format_context_design
StatisticalAnalyzerApp._format_context_sample_overview = _ap_format_context_sample_overview
StatisticalAnalyzerApp._format_context_analysis_scope = _ap_format_context_analysis_scope
StatisticalAnalyzerApp._render_result_summary = _ap_render_result_summary
StatisticalAnalyzerApp.determine_and_run_test = _ap_determine_and_run_test
StatisticalAnalyzerApp.configure_plot_from_result = _ap_configure_plot_from_result
StatisticalAnalyzerApp.open_current_output_folder = _ap_open_current_output_folder
StatisticalAnalyzerApp.reset_application_state = _ap_reset_application_state
StatisticalAnalyzerApp._maybe_pivot = _ap_maybe_pivot


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


StatisticalAnalyzerApp.open_exploratory_matrix_dialog = _ap_open_exploratory_matrix_dialog

if __name__ == "__main__":
    try:
        # Timer-Warnungen unterdrücken
        import os
        os.environ["QT_LOGGING_RULES"] = "qt.core.qobject.timer=false"

        # Enforce high-DPI behavior before QApplication is created.
        if hasattr(Qt, "AA_EnableHighDpiScaling"):
            QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        if hasattr(Qt, "AA_UseHighDpiPixmaps"):
            QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        
        # Apply stylesheet if available
        try:
            stylesheet = _load_auto_pilot_stylesheet()
            print("Stylesheet loaded successfully" if stylesheet else "No stylesheet found")
        except:
            stylesheet = ""
            print("No stylesheet found")
        
        app = QApplication(sys.argv)
        app.setStyleSheet(stylesheet)
        window = StatisticalAnalyzerApp()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Error starting application: {str(e)}")
        traceback.print_exc()
