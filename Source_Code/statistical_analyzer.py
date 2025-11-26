import sys
import time 
import os
from PyQt5.QtWidgets import QDesktopWidget
# Core imports - always needed
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QWidget, QVBoxLayout,
                           QHBoxLayout, QLabel, QComboBox, QPushButton, QListWidget, 
                           QTabWidget, QGroupBox, QCheckBox, QSpinBox, QColorDialog, 
                           QMessageBox, QScrollArea, QListWidgetItem, QDialog, QDialogButtonBox,
                           QGridLayout, QLineEdit, QRadioButton, QAction, QFormLayout, QAbstractItemView, QDoubleSpinBox, QButtonGroup)
from PyQt5.QtGui import QColor, QIcon, QPixmap
from PyQt5.QtCore import Qt

# Initialize lazy loading system
from lazy_imports import preload_critical_modules
preload_critical_modules()

# Heavy modules will be imported lazily when needed
_matplotlib_plt = None
_seaborn = None

def get_matplotlib():
    """Lazy import matplotlib.pyplot"""
    global _matplotlib_plt
    if _matplotlib_plt is None:
        import matplotlib.pyplot as plt
        _matplotlib_plt = plt
    return _matplotlib_plt

def get_seaborn():
    """Lazy import seaborn"""
    global _seaborn
    if _seaborn is None:
        import seaborn as sns
        _seaborn = sns
    return _seaborn

# DISABLED: Nonparametric fallbacks are not yet supported
# from nonparametricanovas import NonParametricFactory, NonParametricRMANOVA
from stats_functions import (
    DataImporter, AnalysisManager, 
    UIDialogManager, OutlierDetector, OUTLIER_IMPORTS_AVAILABLE
)
from resultsexporter import ResultsExporter
from datavisualizer import DataVisualizer
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

DEFAULT_COLORS = ['#FF69B4', '#32CD32', '#FFD700', '#00BFFF', '#DA70D6', '#D8BFD8']  # Pink, Green, Gold, DeepSkyBlue, Orchid, Thistle
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
    """Dialog for selecting groups for a plot"""
    def __init__(self, available_groups, parent=None):
        if not available_groups:
            QMessageBox.critical(parent, "Error", "No groups available! Dialog will not open.")
            raise ValueError("No groups passed to GroupSelectionDialog.")
        super().__init__(parent)
        # Remove the question mark from the window
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setWindowTitle("Select Groups")
        self.resize(300, 400)
        
        layout = QVBoxLayout(self)
        layout.setObjectName("lyoGroupSelection")
        
        # Explanation
        label = QLabel("Select the groups to be displayed in the plot:")
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


class ColumnSelectionDialog(QDialog):
    """Dialog for selecting measurement columns for a dataset"""
    def __init__(self, available_columns, parent=None):
        if not available_columns:
            QMessageBox.critical(parent, "Error", "No measurement columns available! Dialog will not open.")
            raise ValueError("No measurement columns passed to ColumnSelectionDialog.")
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
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
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
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
        hint_label.setStyleSheet("color: #666; font-style: italic;")
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
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
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
        factor_name = self.factor_name.text()
        if not factor_name:
            QMessageBox.warning(self, "Warning", "Please specify a factor name!")
            return None
        
        factor_data = {}
        for group, field in self.factor_values.items():
            value = field.text()
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

class AdvancedTestDialog(QDialog):
    def __init__(self, parent=None, df=None, default_test=None):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.df = df  # Store original DataFrame
        self.default_test = default_test
        self.setWindowTitle("Advanced Statistical Tests")
        self.setupUI()

    def setupUI(self):
        layout = QVBoxLayout()
        
        # 1. Test type selection
        self.testTypeGroup = QGroupBox("Select test")
        testTypeLayout = QVBoxLayout()
        self.mixedAnovaBtn = QRadioButton("Mixed ANOVA (Between + Within)")
        self.rmAnovaBtn = QRadioButton("Repeated Measures ANOVA (Within only)")
        self.twoWayAnovaBtn = QRadioButton("Two-Way ANOVA (Between only)")
        testTypeLayout.addWidget(self.mixedAnovaBtn)
        testTypeLayout.addWidget(self.rmAnovaBtn)
        testTypeLayout.addWidget(self.twoWayAnovaBtn)
        self.testTypeGroup.setLayout(testTypeLayout)
        
        # Preselection based on detected data format
        if self.df is not None:
            columns = list(self.df.columns)
            if 'Subject' in columns or 'subject' in columns:
                self.mixedAnovaBtn.setChecked(True)
            elif any('Factor' in col for col in columns):
                self.twoWayAnovaBtn.setChecked(True)
            else:
                self.mixedAnovaBtn.setChecked(True)  # Default case
            
        # 2. Column selection
        self.columnsGroup = QGroupBox("Assign variables")
        columnsLayout = QFormLayout()
        
        # Dependent variable
        self.dvCombo = QComboBox()
        if self.df is not None:
            self.dvCombo.addItems(self.df.columns)
            # Preselect value column
            value_idx = -1
            for i, col in enumerate(self.df.columns):
                if col.lower() == 'value' or 'wert' in col.lower():
                    value_idx = i
            if value_idx >= 0:
                self.dvCombo.setCurrentIndex(value_idx)
                    
        columnsLayout.addRow("Dependent variable:", self.dvCombo)
        
        # Subject ID
        self.subjectCombo = QComboBox()
        if self.df is not None:
            self.subjectCombo.addItems(self.df.columns)
            # Preselect subject column
            subject_idx = -1
            for i, col in enumerate(self.df.columns):
                if col.lower() in ('subject', 'subjekt', 's01'):
                    subject_idx = i
            if subject_idx >= 0:
                self.subjectCombo.setCurrentIndex(subject_idx)
                
        columnsLayout.addRow("Subject/ID variable:", self.subjectCombo)
        
        # Within factors (multiple possible)
        self.withinList = QListWidget()
        if self.df is not None:
            self.withinList.addItems(self.df.columns)
            # Preselect within factors
            for i, col in enumerate(self.df.columns):
                if col.lower() in ('timepoint', 'time', 'zeit', 'messzeitpunkt'):
                    self.withinList.item(i).setSelected(True)
                    
        self.withinList.setSelectionMode(QAbstractItemView.MultiSelection)
        # Limit height for many columns
        self.withinList.setMaximumHeight(120)
        columnsLayout.addRow("Within factors:", self.withinList)
        
        # Between factors (multiple possible)
        self.betweenList = QListWidget()
        if self.df is not None:
            self.betweenList.addItems(self.df.columns)
            # Preselect between factors
            for i, col in enumerate(self.df.columns):
                if col.lower() in ('betweengrp', 'gruppe', 'group', 'factora', 'factorb'):
                    self.betweenList.item(i).setSelected(True)
                    
        self.betweenList.setSelectionMode(QAbstractItemView.MultiSelection)
        # Limit height for many columns
        self.betweenList.setMaximumHeight(120)
        columnsLayout.addRow("Between factors:", self.betweenList)
        
        self.columnsGroup.setLayout(columnsLayout)
        
        # Connect methods for test type change
        self.mixedAnovaBtn.toggled.connect(self.update_field_visibility)
        self.rmAnovaBtn.toggled.connect(self.update_field_visibility)
        self.twoWayAnovaBtn.toggled.connect(self.update_field_visibility)
        
        # Immediately update fields based on initial test
        self.update_field_visibility()
        
        # 3. Buttons
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        
        # Always set a default button
        self.mixedAnovaBtn.setChecked(True)
        
        # Preselect based on detected format
        if self.default_test == "mixed_anova":
            self.mixedAnovaBtn.setChecked(True)
        elif self.default_test == "two_way_anova":
            self.twoWayAnovaBtn.setChecked(True)
        elif self.default_test == "repeated_measures_anova":
            self.rmAnovaBtn.setChecked(True)
        
        # Assemble layout
        layout.addWidget(self.testTypeGroup)
        layout.addWidget(self.columnsGroup)
        layout.addWidget(buttonBox)
        
        self.setLayout(layout)
        
    def update_field_visibility(self):
        """Updates the visibility of fields depending on the selected test type"""
        is_two_way = self.twoWayAnovaBtn.isChecked()
        is_mixed = self.mixedAnovaBtn.isChecked()
        is_repeated = self.rmAnovaBtn.isChecked()
        
        # For Two-Way ANOVA we don't need Subject
        self.subjectCombo.setEnabled(not is_two_way)
        
        # Within factors only for Mixed ANOVA or Repeated Measures
        self.withinList.setEnabled(is_mixed or is_repeated)
        
        # Between factors only for Mixed ANOVA or Two-Way ANOVA
        self.betweenList.setEnabled(is_mixed or is_two_way)


# --- Unified PlotConfigDialog (new system) ---
class PlotConfigDialog(QDialog):
    def __init__(self, groups, parent=None, default_filename=None):
        if not groups:
            QMessageBox.critical(parent, "Error", "No groups passed for plot configuration!")
            raise ValueError("No groups for PlotConfigDialog.")
        if len(set(groups)) != len(groups):
            QMessageBox.warning(parent, "Warning", "There are duplicate group names!")
            # Optional: raise ValueError("Duplicate groups for PlotConfigDialog.")
        super().__init__(parent)
        screen = QDesktopWidget().screenGeometry()
        width = int(screen.width() * 0.40)
        height = int(screen.height() * 0.50)
        self.resize(width, height)
        self.move(
            (screen.width() - width) // 2,
            (screen.height() - height) // 2
        )
        # Remove the question mark from the window
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setWindowTitle("Configure Plot")
        self.groups = groups
        
        # IMPORTANT: Explicitly initialize all status variables
        self.comparisons = []  # Empty list as initial state
        self.colors = None     # No preselected colors
        self.hatches = None    # No preselected hatches
        
        # Initialize UI elements
        layout = QVBoxLayout(self)
        
        # COMMENTED OUT: Comparisons list and reset button - not needed as post-hoc tests provide all comparisons
        # comp_group = QGroupBox("Pairwise Comparisons")
        # comp_layout = QVBoxLayout()
        # 
        # self.comp_list = QListWidget()
        # comp_layout.addWidget(self.comp_list)
        # 
        # # Reset button for explicit reset
        # reset_button = QPushButton("Reset comparisons")
        # reset_button.clicked.connect(self.reset_comparisons)
        # comp_layout.addWidget(reset_button)
        
        # File name configuration
        file_group = QGroupBox("File Configuration")
        file_group.setObjectName("grpFileConfig")
        file_layout = QGridLayout(file_group)
        file_layout.setObjectName("lyoFileConfig")

        # File name only
        file_label = QLabel("File Name:")
        file_label.setObjectName("lblFileName")
        file_layout.addWidget(file_label, 0, 0)
        self.file_name_edit = QLineEdit("")
        self.file_name_edit.setObjectName("edtFileName")
        self.file_name_edit.setPlaceholderText("Default: automatically generated from group names")
        
        # Set default filename if provided
        if default_filename:
            clean_filename = "".join(c for c in default_filename if c.isalnum() or c in (' ', '-', '_')).strip()
            if clean_filename:
                self.file_name_edit.setText(clean_filename)
                
        file_layout.addWidget(self.file_name_edit, 0, 1)
        
        layout.addWidget(file_group)
        
        # Statistical options
        stats_group = QGroupBox("Statistical Options")
        stats_group.setObjectName("grpStatOptions")
        stats_layout = QVBoxLayout(stats_group)
        stats_layout.setObjectName("lyoStatOptions")
        
        # Improved description for dependent samples
        dependent_layout = QHBoxLayout()
        self.dependent_check = QCheckBox("Dependent samples (paired tests)")
        self.dependent_check.setObjectName("chkDependentSamples")
        dependent_layout.addWidget(self.dependent_check)

        # --- Gruppenreihenfolge (Drag & Drop) ---
        order_group = QGroupBox("Group order (drag to sort)")
        order_layout = QVBoxLayout(order_group)
        self.order_list = QListWidget()
        self.order_list.setDragDropMode(QListWidget.InternalMove)
        # Limit height for many groups
        self.order_list.setMaximumHeight(150)
        for group in groups:
            self.order_list.addItem(str(group))
        order_layout.addWidget(self.order_list)
        layout.addWidget(order_group)
        
        # Add an info button with tooltip
        dependent_info = QPushButton("?")
        dependent_info.setObjectName("btnDependentInfo")
        dependent_info.setMaximumWidth(20)
        dependent_info.setToolTip(
            "Dependent samples are measurements on the same subject at different time points or "
            "under different conditions. Examples: pre/post measurements, repeated measurements.\n\n"
            "Important: For dependent tests, all groups must have the same number of measurements and "
            "the order of the values must match."
        )
        dependent_info.clicked.connect(self.show_dependent_help)
        dependent_layout.addWidget(dependent_info)
        dependent_layout.addStretch()
        
        stats_layout.addLayout(dependent_layout)
        
        # Options for dependent visualization
        self.dependent_visualization_options = QWidget()
        dependent_vis_layout = QVBoxLayout(self.dependent_visualization_options)
        dependent_vis_layout.setObjectName("lyoDependentVisOptions")
        
        self.show_individual_lines = QCheckBox("Show individual connection lines")
        self.show_individual_lines.setObjectName("chkShowIndividualLines")
        self.show_individual_lines.setChecked(True)
        self.show_individual_lines.setToolTip(
            "If enabled, lines are drawn connecting the values of the same subject. "
            "This is useful to visualize individual changes."
        )
        
        dependent_vis_layout.addWidget(self.show_individual_lines)
        # Hide initially
        self.dependent_visualization_options.setVisible(False)
        
        # Insert below the dependent-check
        stats_layout.addWidget(self.dependent_visualization_options)

        # -- Options for dependent tests: select subject ID and within factor
        self.dependent_data_options = QWidget()
        dep_data_layout = QFormLayout(self.dependent_data_options)

        # Placeholder text for missing column data
        self.subject_info_label = QLabel(
            "Note: After closing this dialog, subject ID and within factor will be required. "
            "You can select these later in the advanced test options or in the data import dialog."
        )
        self.subject_info_label.setWordWrap(True)
        self.subject_info_label.setStyleSheet("color: #666; font-style: italic;")
        dep_data_layout.addRow(self.subject_info_label)

        # Hide initially
        self.dependent_data_options.setVisible(False)
        stats_layout.addWidget(self.dependent_data_options)
        
        # right after you’ve created your create_plot_check…
        self.create_plot_check = QCheckBox("Create plot")
        self.create_plot_check.setObjectName("chkCreatePlot")
        self.create_plot_check.setChecked(False)
        stats_layout.addWidget(self.create_plot_check)

        # add your new button, hidden by default:
        self.appearance_btn = QPushButton("Configure appearance…")
        self.appearance_btn.setObjectName("btnConfigureAppearance")
        self.appearance_btn.hide()
        self.appearance_btn.clicked.connect(self.open_appearance_dialog)
        stats_layout.addWidget(self.appearance_btn)

        # wire it up so it only shows when create_plot is checked
        self.create_plot_check.toggled.connect(self.appearance_btn.setVisible)

        # Initialize appearance button text
        self.update_appearance_button_text()
        
        # COMMENTED OUT: Manual comparison selection - not needed as post-hoc tests provide all comparisons
        # # List of comparisons
        # comparisons_label = QLabel("Specific comparisons:")
        # comparisons_label.setObjectName("lblComparisons")
        # stats_layout.addWidget(comparisons_label)
        # 
        # self.comparisons_list = QListWidget()
        # self.comparisons_list.setObjectName("lstComparisons")
        # stats_layout.addWidget(self.comparisons_list)
        # 
        # # Buttons for comparisons
        # compare_buttons = QHBoxLayout()
        # compare_buttons.setObjectName("lyoCompareButtons")
        # 
        # add_compare_button = QPushButton("Add comparison")
        # add_compare_button.setObjectName("btnAddComparison")
        # add_compare_button.clicked.connect(self.add_comparison)
        # 
        # remove_compare_button = QPushButton("Remove comparison")
        # remove_compare_button.setObjectName("btnRemoveComparison")
        # remove_compare_button.clicked.connect(self.remove_comparison)
        # 
        # compare_buttons.addWidget(add_compare_button)
        # compare_buttons.addWidget(remove_compare_button)
        # stats_layout.addLayout(compare_buttons)
        
        layout.addWidget(stats_group)
        
        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.setObjectName("btnDialogButtons")
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        # Save comparisons
        self.comparisons = []
        
    def toggle_dependent_options(self, checked):
        """Show or hide dependent options (visualization + data selection)."""
        self.dependent_visualization_options.setVisible(checked)
        self.dependent_data_options.setVisible(checked)

    def open_appearance_dialog(self):
        print("DEBUG: open_appearance_dialog called!")
        print(f"DEBUG: PLOT_MODULES_AVAILABLE = {PLOT_MODULES_AVAILABLE}")
        print(f"DEBUG: PlotAestheticsDialog = {PlotAestheticsDialog}")
        
        # Get the ordered groups from the list widget instead of using self.groups
        ordered_groups = [self.order_list.item(i).text() for i in range(self.order_list.count())]
        print(f"DEBUG: ordered_groups = {ordered_groups}")
        
        # Only open PlotAestheticsDialog for appearance configuration, not as main dialog
        if PLOT_MODULES_AVAILABLE and PlotAestheticsDialog:
            print("DEBUG: Opening new PlotAestheticsDialog!")
            # Access parent app for temporary settings
            parent_app = self.parent()
            
            # IMPORTANT: Get the current plot config to preserve user's colors
            current_plot_config = self.get_config() or {}
            current_appearance_config = getattr(parent_app, 'temp_plot_appearance_settings', None) or {}
            
            # Merge configs: appearance settings override plot config, but preserve user's colors if no appearance colors
            merged_config = current_plot_config.copy()
            merged_config.update(current_appearance_config)
            
            # Ensure colors are from plot config if not overridden by appearance
            if 'colors' in current_plot_config and ('colors' not in current_appearance_config or not current_appearance_config['colors']):
                merged_config['colors'] = current_plot_config['colors']
            
            print(f"DEBUG: merged_config colors = {merged_config.get('colors', {})}")
            
            # Determine context: this is a user plot (not analysis-only)
            context = "user_plot"
            
            dlg = PlotAestheticsDialog(
                ordered_groups,
                parent_app.samples if hasattr(parent_app, "samples") else {},
                config=merged_config,
                parent=self,
                context=context
            )
            print("DEBUG: PlotAestheticsDialog created successfully")
            if dlg.exec_() == dlg.Accepted:
                print("DEBUG: Dialog accepted, saving settings")
                # Speichere in temporären Einstellungen der parent App
                parent_app.temp_plot_appearance_settings = dlg.get_config()
                self.update_appearance_button_text()
                # WICHTIG: Aktualisiere die Live-Preview im Hauptfenster
                parent_app.auto_generate_preview()
            else:
                print("DEBUG: Dialog cancelled")
        else:
            print("DEBUG: PlotAestheticsDialog not available, showing warning")
            QMessageBox.warning(self, "Feature not available", 
                               "The new plot appearance dialog is not available. "
                               "Please ensure plot_aesthetics_dialog.py is in the same directory.")
    
    def update_appearance_button_text(self):
        """Update the appearance button text to show if settings are configured"""
        # Greife auf die parent App zu, um die temporären Einstellungen zu prüfen
        parent_app = self.parent()
        if hasattr(parent_app, 'temp_plot_appearance_settings') and parent_app.temp_plot_appearance_settings:
            self.appearance_btn.setText("Configure appearance… ✓")
            self.appearance_btn.setToolTip("Appearance settings are configured. Click to modify.")
        else:
            self.appearance_btn.setText("Configure appearance…")
            self.appearance_btn.setToolTip("Configure plot appearance settings.")
            
    def show_dependent_help(self):
        QMessageBox.information(
            self, "Dependent samples",
            "<b>When to choose dependent samples?</b><br><br>"
            "Dependent (or paired) samples are present when you:<br>"
            "• Measure the same subjects/individuals multiple times (e.g. before/after a treatment)<br>"
            "• Perform measurements on naturally paired pairs (e.g. twin studies)<br>"
            "• Take multiple measurements on the same material under different conditions<br><br>"
            "<b>Requirements for dependent tests:</b><br>"
            "• All groups must have the <i>same number</i> of measurements<br>"
            "• The values must be in the <i>same order</i> (i.e. measurement 1 from group A "
            "belongs to the same subject as measurement 1 from group B)<br>"
            "• For more than 2 groups, a special repeated measures ANOVA is performed<br><br>"
            "<b>Visualization:</b><br>"
            "For dependent samples, an additional line plot is created "
            "showing the individual changes."
        )
        
    def set_groups(self, groups): self.groups = groups
    def set_width(self, w): self.width_spin.setValue(w)
    def set_height(self, h): self.height_spin.setValue(h)
    def set_color(self, group, color):
        if group in self.color_buttons:
            self.color_buttons[group].setStyleSheet(f"background-color: {color};")
        else:
            QMessageBox.warning(self, "Warning", f"Group {group} not in color table!")
    def set_hatch(self, group, hatch):
        if group in self.hatch_combos:
            idx = self.hatch_combos[group].findText(hatch)
            if idx >= 0:
                self.hatch_combos[group].setCurrentIndex(idx)
            else:
                QMessageBox.warning(self, "Warning", f"Hatch '{hatch}' not available for group {group}!")
        else:
            QMessageBox.warning(self, "Warning", f"Group {group} not in hatch table!")
    def set_dependent(self, dep): self.dependent_check.setChecked(dep)
    def set_create_plot(self, create): self.create_plot_check.setChecked(create)
    # COMMENTED OUT: Manual comparison methods - not needed as post-hoc tests provide all comparisons
    # def set_comparisons(self, comps):
    #     self.comparisons = comps
    #     self.comparisons_list.clear()
    #     for comp in comps:
    #         self.comparisons_list.addItem(f"{comp['group1']} vs {comp['group2']} ({comp['test_type']})")
    def set_file_name(self, name): self.file_name_edit.setText(name)

    def select_color(self, group):
        """Opens a color dialog for a group"""
        color = QColorDialog.getColor()
        if color.isValid():
            self.color_buttons[group].setStyleSheet(f"background-color: {color.name()};")

    # COMMENTED OUT: Manual comparison methods - not needed as post-hoc tests provide all comparisons
    # def add_comparison(self):
    #     dialog = PairwiseComparisonDialog(self.groups, self)
    #     if dialog.exec_() == QDialog.Accepted:
    #         comp = dialog.get_comparison()
    #         if comp['group1'] == comp['group2']:
    #             QMessageBox.warning(self, "Error", "The two groups must be different.")
    #             return
    #                 # Set a default value for the test type, will be replaced later by post-hoc results
    #         comp['test_type'] = "From post-hoc test"
    # 
    #         self.comparisons.append(comp)
    #         self.comparisons_list.addItem(f"{comp['group1']} vs {comp['group2']} ({comp['test_type']})")
    # 
    # def remove_comparison(self):
    #     current_row = self.comparisons_list.currentRow()
    #     if current_row >= 0:
    #         self.comparisons_list.takeItem(current_row)
    #         self.comparisons.pop(current_row)
              
    def get_config(self):
        if not self.groups:
            QMessageBox.critical(self, "Error", "Plot without groups is not possible!")
            return None
        try:
            # Use Greys palette by default instead of the bright default colors
            try:
                import seaborn as sns
                # Generate Greys palette colors for groups
                greys_colors = sns.color_palette('Greys', n_colors=len(self.groups))
                colors_dict = {}
                for i, g in enumerate(self.groups):
                    # Convert RGB tuple to hex
                    rgb = greys_colors[i]
                    hex_color = "#{:02x}{:02x}{:02x}".format(
                        int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
                    )
                    colors_dict[g] = hex_color
            except ImportError:
                # Fallback to default colors if seaborn is not available
                colors_dict = {}
                for i, g in enumerate(self.groups):
                    colors_dict[g] = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
            hatches_dict = {}
            for g in self.groups:
                hatches_dict[g] = ""  # Default: no hatch
            config = {
                'file_name': self.file_name_edit.text() if self.file_name_edit.text() else None,
                'groups': [self.order_list.item(i).text() for i in range(self.order_list.count())],
                'group_order': [self.order_list.item(i).text() for i in range(self.order_list.count())],
                'width': 12,
                'height': 10,
                'colors': colors_dict,
                'hatches': hatches_dict,
                'create_plot': self.create_plot_check.isChecked(),
                # REMOVED: 'comparisons': self.comparisons.copy() if self.comparisons else [],  # Not needed - post-hoc tests provide comparisons
                'error_type': 'sd',  # Default to standard deviation (handled by PlotAestheticsDialog)
                'dependent': self.dependent_check.isChecked(),
                'show_individual_lines': self.show_individual_lines.isChecked() if self.dependent_check.isChecked() else False,
                'needs_subject_selection': self.dependent_check.isChecked(),
                # Add default Seaborn palette settings
                'seaborn_palette': 'Greys',
                'use_seaborn_styling': True,
                'seaborn_context': 'paper'
            }
            # Always add appearance_settings from PlotAestheticsDialog if present
            parent_app = self.parent()
            if hasattr(parent_app, 'temp_plot_appearance_settings') and parent_app.temp_plot_appearance_settings:
                config['appearance_settings'] = parent_app.temp_plot_appearance_settings
            return config
        except Exception as e:
            print(f"Error in get_config: {str(e)}")
            traceback.print_exc()
            # Return a minimal config to prevent crashes
            try:
                import seaborn as sns
                # Generate Greys palette colors for groups
                greys_colors = sns.color_palette('Greys', n_colors=len(self.groups))
                colors_dict = {}
                for i, g in enumerate(self.groups):
                    # Convert RGB tuple to hex
                    rgb = greys_colors[i]
                    hex_color = "#{:02x}{:02x}{:02x}".format(
                        int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
                    )
                    colors_dict[g] = hex_color
            except ImportError:
                # Fallback to default colors if seaborn is not available
                colors_dict = {g: DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i, g in enumerate(self.groups)}
            
            return {
                'groups': [self.order_list.item(i).text() for i in range(self.order_list.count())],
                'group_order': [self.order_list.item(i).text() for i in range(self.order_list.count())],
                'title': None,
                'x_label': None,
                'y_label': None,
                'file_name': None,
                'width': 12,
                'height': 10,
                'colors': colors_dict,
                'hatches': {g: "" for g in self.groups},
                'dependent': False,
                'create_plot': True,
                # REMOVED: 'comparisons': [],  # Not needed - post-hoc tests provide comparisons
                'error_type': 'sd',
                # Add default Seaborn palette settings
                'seaborn_palette': 'Greys',
                'use_seaborn_styling': True,
                'seaborn_context': 'paper'
            }
    
    # COMMENTED OUT: Manual comparison reset - not needed as post-hoc tests provide all comparisons
    # def reset_comparisons(self):
    #     """Explicit method to reset comparisons"""
    #     self.comparisons = []
    #     self.comparisons_list.clear()
        
    def _convert_old_config_to_new(self, old_config):
        """
        Konvertiert alte Plot-Konfiguration zum neuen Format.
        """
        config = {
            'plot_type': 'Bar',  # Default
            'width': 8,
            'height': 6,
            'dpi': 300,
            'colors': {},
            'hatches': {},
            'alpha': 0.8,
            'theme': 'default',
            'grid': False,
            'despine': True,
            'show_error_bars': True,
            'error_type': 'sd',
            'error_style': 'caps',
            'capsize': 0.05,
            'show_points': True,
            'point_size': 80,
            'jitter_strength': 0.3,
            'show_significance_letters': True,
            'x_label': '',
            'y_label': '',
            'title': old_config.get('title', ''),
            'fontsize_title': 14,
            'fontsize_axis': 12,
            'fontsize_ticks': 10,
            'bar_linewidth': 0.5,
            'bar_edge_color': 'black'
        }
        
        # Konvertiere alte appearance_settings
        if 'appearance_settings' in old_config:
            appearance = old_config['appearance_settings']
            
            # Plot type
            config['plot_type'] = appearance.get('plot_type', 'Bar')
            
            # Farben und Hatches
            if 'colors' in appearance:
                config['colors'] = appearance['colors']
            if 'hatches' in appearance:
                config['hatches'] = appearance['hatches']
                
            # Weitere Einstellungen übernehmen
            config.update({
                'alpha': appearance.get('alpha', 0.8),
                'grid': appearance.get('grid', False),
                'despine': appearance.get('despine', True),
                'fontsize_title': appearance.get('fontsize_title', 14),
                'fontsize_axis': appearance.get('fontsize_axis', 12),
                'fontsize_ticks': appearance.get('fontsize_ticks', 10),
                'bar_linewidth': appearance.get('bar_linewidth', 0.5),
                'bar_edge_color': appearance.get('bar_edge_color', 'black')
            })
        
        # Use global settings if available
        if hasattr(self, 'global_appearance_settings') and self.global_appearance_settings:
            config.update(self.global_appearance_settings)
        
        return config

    def preview_plot(self, plot_idx):
        """Creates a preview of a plot based on its configuration."""
        if self.samples is None:
            QMessageBox.warning(self, "Error", "No data loaded.")
            return
            
        if plot_idx < 0 or plot_idx >= len(self.plot_configs):
            return

        plot_config = self.plot_configs[plot_idx]

        try:
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

            # Use the unified DataVisualizer.plot_from_config for preview
            import matplotlib.pyplot as plt
            from stats_functions import DataVisualizer
            
            # Check if we have the new preview widget
            if hasattr(self, 'plot_preview_widget') and self.plot_preview_widget:
                # Use the new preview widget
                config = self._convert_old_config_to_new(plot_config)
                self.plot_preview_widget.set_data(plot_config['groups'], plot_samples)
                self.plot_preview_widget.update_plot(config)
            elif hasattr(self, 'figure'):
                # Use the old figure/canvas approach
                fig = self.figure
                fig.clear()
                ax = fig.add_subplot(111)
                config = self._convert_old_config_to_new(plot_config)
                try:
                    DataVisualizer.plot_from_config(ax, plot_config['groups'], plot_samples, config)
                    fig.tight_layout()
                    if hasattr(self, 'canvas'):
                        self.canvas.draw()
                except Exception as e:
                    print(f"Error in DataVisualizer.plot_from_config: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    # Fallback to simple preview
                    ax.text(0.5, 0.5, f'Error creating preview:\n{str(e)}', 
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=10, color='red')
                    fig.tight_layout()
                    if hasattr(self, 'canvas'):
                        self.canvas.draw()
            else:
                print("Warning: No preview widget or figure available")
                
        except Exception as e:
            print(f"Error in preview_plot: {str(e)}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Preview Error", f"An error occurred while creating the preview: {str(e)}")

    def update_preview_on_selection_change(self):
        """Updates the preview when group selection changes."""
        if not PLOT_MODULES_AVAILABLE or not self.preview_widget:
            return
            
        selected_items = self.groups_list.selectedItems()
        if selected_items and self.samples:
            selected_groups = [item.text() for item in selected_items]
            
            # Use global settings or default config
            config = getattr(self, 'global_appearance_settings', {})
            if not config:
                config = {
                    'plot_type': 'Bar',
                    'colors': {group: DEFAULT_COLORS[i % len(DEFAULT_COLORS)] 
                              for i, group in enumerate(selected_groups)},
                    'show_points': True,
                    'show_significance_letters': True
                }
            
            self.preview_widget.set_data(selected_groups, self.samples)
            self.preview_widget.update_plot(config)
    
    def integrate_statistical_results(self, plot_config, statistical_results):
        """
        Integriert statistische Ergebnisse (Signifikanzen, Post-hoc Tests) in die Plot-Konfiguration.
        
        Parameters:
        -----------
        plot_config : dict
            Bestehende Plot-Konfiguration
        statistical_results : dict
            Ergebnisse der statistischen Analyse
        
        Returns:
        --------
        dict
            Erweiterte Plot-Konfiguration mit statistischen Informationen
        """
        # Konvertiere zur neuen Konfiguration
        config = self._convert_old_config_to_new(plot_config)
        
        # Füge statistische Informationen hinzu
        if statistical_results:
            # P-value for title or annotation
            if 'p_value' in statistical_results:
                p_val = statistical_results['p_value']
                if isinstance(p_val, (int, float)):
                    if p_val < 0.001:
                        p_text = "p < 0.001"
                    else:
                        p_text = f"p = {p_val:.4f}"
                    
                    # Add P-value to title if desired
                    current_title = config.get('title', '')
                    if current_title:
                        config['title'] = f"{current_title} ({p_text})"
                    else:
                        config['title'] = f"Analysis Results ({p_text})"
            
            # Enable significance letters if post-hoc tests are available
            if 'pairwise_comparisons' in statistical_results and statistical_results['pairwise_comparisons']:
                config['show_significance_letters'] = True
                config['pairwise_results'] = statistical_results['pairwise_comparisons']
            
            # Test recommendation for significance letters
            if 'test_recommendation' in statistical_results:
                config['test_recommendation'] = statistical_results['test_recommendation']
            else:
                config['test_recommendation'] = 'parametric'  # Default
        
        return config

    def create_plot_with_statistics(self, groups, title="", statistical_results=None):
        """
        Erstellt einen Plot mit integrierten statistischen Ergebnissen.
        
        Parameters:
        -----------
        groups : list
            Liste der Gruppennamen
        title : str
            Title for the plot
        statistical_results : dict, optional
            Statistische Analyseergebnisse
        """
        if not self.samples:
            QMessageBox.warning(self, "No data", "No data available for plotting.")
            return
        
        # Basis-Konfiguration
        plot_config = {
            'groups': groups,
            'title': title,
            'create_plot': True
        }
        
        # Integriere statistische Ergebnisse
        config = self.integrate_statistical_results(plot_config, statistical_results)
        
        # Use global appearance settings if available
        if hasattr(self, 'global_appearance_settings') and self.global_appearance_settings:
            config.update(self.global_appearance_settings)
        
        try:
            # Filter samples for selected groups
            plot_samples = {}
            for group in groups:
                if group in self.samples:
                    plot_samples[group] = self.samples[group]
            
            if not plot_samples:
                QMessageBox.warning(self, "No data", "No data found for selected groups.")
                return
            
            # Use DataVisualizer for final plot
            plot_type = config.get('plot_type', 'Bar')
            
            if plot_type == 'Bar':
                fig, ax = DataVisualizer.plot_bar(
                    groups, plot_samples,
                    title=config.get('title', ''),
                    colors=config.get('colors', {}),
                    hatches=config.get('hatches', {}),
                    alpha=config.get('alpha', 0.8),
                    show_points=config.get('show_points', True),
                    show_significance_letters=config.get('show_significance_letters', True),
                    test_recommendation=config.get('test_recommendation', 'parametric'),
                    pairwise_results=config.get('pairwise_results'),
                    **{k: v for k, v in config.items() if k not in ['groups', 'title', 'colors', 'hatches', 'alpha', 'show_points', 'show_significance_letters', 'test_recommendation', 'pairwise_results']}
                )
            elif plot_type == 'Box':
                fig, ax = DataVisualizer.plot_box(
                    groups, plot_samples,
                    title=config.get('title', ''),
                    colors=config.get('colors', {}),
                    show_points=config.get('show_points', True),
                    show_significance_letters=config.get('show_significance_letters', True),
                    test_recommendation=config.get('test_recommendation', 'parametric'),
                    **{k: v for k, v in config.items() if k not in ['groups', 'title', 'colors', 'show_points', 'show_significance_letters', 'test_recommendation']}
                )
            elif plot_type == 'Violin':
                fig, ax = DataVisualizer.plot_violin(
                    groups, plot_samples,
                    title=config.get('title', ''),
                    colors=config.get('colors', {}),
                    show_points=config.get('show_points', True),
                    show_significance_letters=config.get('show_significance_letters', True),
                    test_recommendation=config.get('test_recommendation', 'parametric'),
                    **{k: v for k, v in config.items() if k not in ['groups', 'title', 'colors', 'show_points', 'show_significance_letters', 'test_recommendation']}
                )
            elif plot_type == 'Raincloud':
                fig, ax = DataVisualizer.plot_raincloud(
                    groups, plot_samples,
                    title=config.get('title', ''),
                    colors=config.get('colors', {}),
                    show_points=config.get('show_points', True),
                    show_significance_letters=config.get('show_significance_letters', True),
                    test_recommendation=config.get('test_recommendation', 'parametric'),
                    **{k: v for k, v in config.items() if k not in ['groups', 'title', 'colors', 'show_points', 'show_significance_letters', 'test_recommendation']}
                )
            
            plt = get_matplotlib()
            plt.show()
            
        except Exception as e:
            print(f"Error creating plot: {str(e)}")
            QMessageBox.critical(self, "Plot Error", f"An error occurred while creating the plot: {str(e)}")
    
    def demo_new_plot_system(self):
        """
        Demo-Methode, die das neue Plot-System mit allen Features zeigt.
        """
        if not self.samples or not self.available_groups:
            QMessageBox.warning(self, "No data", "Please load data first.")
            return
        
        # Wähle die ersten 3-4 Gruppen für Demo
        demo_groups = self.available_groups[:min(4, len(self.available_groups))]
        
        # Simuliere statistische Ergebnisse
        import numpy as np
        demo_statistical_results = {
            'test': 'One-Way ANOVA',
            'p_value': 0.023,
            'test_recommendation': 'parametric',
            'pairwise_comparisons': [
                {'group1': demo_groups[0], 'group2': demo_groups[1], 'p_value': 0.045, 'significant': True},
                {'group1': demo_groups[0], 'group2': demo_groups[2], 'p_value': 0.156, 'significant': False}
            ] if len(demo_groups) >= 3 else []
        }
        
        # Erstelle Plot mit allen Features
        self.create_plot_with_statistics(
            groups=demo_groups,
            title="Demo Analysis with Statistics",
            statistical_results=demo_statistical_results
        )
        
        QMessageBox.information(self, "Demo completed", 
                               f"Created demo plot for groups: {', '.join(demo_groups)}\n"
                               f"With simulated statistical results (p = {demo_statistical_results['p_value']})")
        


class TransformationDialog(QDialog):
    def __init__(self, test_info=None, parent=None):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.test_info = test_info
        self.transformation = None
        self.setWindowTitle("Select Data Transformation")
        self.setupUI()
        
    def setupUI(self):
        layout = QVBoxLayout()
        
        # Info text
        info = QLabel("The data do not meet the requirements for parametric tests.\n"
                     "Please select a transformation:")
        layout.addWidget(info)
        
        # Transformation options
        self.log10RB = QRadioButton("Log10 transformation (for positive, right-skewed data)")
        self.boxcoxRB = QRadioButton("Box-Cox transformation (automatic lambda optimization)")
        self.arcsinRB = QRadioButton("Arcsin square root transformation (for percentages/proportions)")
        
        # Default selection
        self.log10RB.setChecked(True)
        
        # Add options to layout
        layout.addWidget(self.log10RB)
        layout.addWidget(self.boxcoxRB)
        layout.addWidget(self.arcsinRB)
        
        # Buttons
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        layout.addWidget(buttonBox)
        
        self.setLayout(layout)
        
    def get_transformation(self):
        if self.log10RB.isChecked():
            return "log10"
        elif self.boxcoxRB.isChecked():
            return "boxcox"
        elif self.arcsinRB.isChecked():
            return "arcsin_sqrt"
        return "log10"  # Default fallback

class OutlierDetectionDialog(QDialog):
    """Dialog for configuring outlier detection analysis"""
    def __init__(self, df, parent=None):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setWindowTitle("Outlier Detection")
        self.resize(600, 500)  # Make it slightly wider
        self.df = df
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the dialog UI"""
        layout = QVBoxLayout(self)
        
        # Check if outlier detection is available
        if not OUTLIER_IMPORTS_AVAILABLE:
            warning_label = QLabel("Warning: Outlier detection is not available.\n"
                                "Please install required packages: outliers, pingouin, openpyxl")
            warning_label.setStyleSheet("color: red; font-weight: bold;")
            layout.addWidget(warning_label)
            
            button_box = QDialogButtonBox(QDialogButtonBox.Ok)
            button_box.accepted.connect(self.reject)
            layout.addWidget(button_box)
            return
        
        # Info section
        info_label = QLabel("Select data columns and parameters for outlier detection:")
        info_label.setStyleSheet("font-weight: bold;")
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
        mode_label.setStyleSheet("font-weight: bold;")
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
        test_label.setStyleSheet("font-weight: bold;")
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

class StatisticalAnalyzerApp(QMainWindow):
    """Main application for statistical analysis of data from Excel/CSV files."""
    
    def __init__(self):
        """Initializes the application with all UI elements."""
        super().__init__()
        screen = QDesktopWidget().screenGeometry()
        width = int(screen.width() * 0.72)
        height = int(screen.height() * 0.72)
        self.resize(width, height)
        self.move(
            (screen.width() - width) // 2,
            (screen.height() - height) // 2
        )
        self.setWindowTitle("BioMedStatX v1.0.1 - Comprehensive Statistical Analysis Tool")
        self.setGeometry(100, 50, 1600, 1300)
        
        # Set window icon
        try:
            icon_path = resource_path("assets/Institutslogo.ico")
            if os.path.exists(icon_path):
                self.setWindowIcon(QIcon(icon_path))
                print(f"SUCCESS: Window icon set from {icon_path}")
            else:
                print(f"WARNING: Icon file not found at {icon_path}")
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

        # Advanced tests in help menu
        advanced_action = QAction('Advanced Tests', self)
        advanced_action.triggered.connect(self.show_advanced_tests_help)
        help_menu.addAction(advanced_action)

        help_menu.addSeparator()
        
        # Check for updates
        update_action = QAction('Check for Updates...', self)  
        update_action.triggered.connect(self.check_for_updates)
        help_menu.addAction(update_action)
        
        # About should be last
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

        # Analysis menu (create new or use existing)
        analysis_menu = menubar.addMenu('&Analysis')

        # Action for advanced tests
        advanced_test_action = QAction('Run Advanced Tests', self)
        advanced_test_action.triggered.connect(self.open_advanced_tests)
        analysis_menu.addAction(advanced_test_action)
        
        # Action for outlier detection
        outlier_action = QAction('Detect Outliers', self)
        outlier_action.triggered.connect(self.run_outlier_detection)
        analysis_menu.addAction(outlier_action)

        
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

    def show_graph_visualization_help(self):
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextBrowser, QPushButton
        dlg = QDialog(self)
        dlg.setWindowTitle("Graph Visualization")
        dlg.resize(800, 600)
        layout = QVBoxLayout(dlg)
        browser = QTextBrowser()
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
        dlg.setWindowTitle("Statistical Tests & Excel Export")
        dlg.resize(900, 600)
        layout = QVBoxLayout(dlg)
        browser = QTextBrowser()
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
                        <li><i>Note: Advanced tests like Mixed ANOVA, Two-Way ANOVA, and Repeated-Measures ANOVA are explained in the separate "Advanced Tests" help section.</i></li>
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
            <p style='color:gray; font-size:90%'>Note: Advanced/complex tests (e.g. Mixed ANOVA, Two-Way ANOVA) are explained in a separate help page.</p>
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
        
        # File selection
        file_section = QGroupBox("Data Source")
        file_section.setObjectName("grpDataSource")
        file_layout = QHBoxLayout(file_section)
        file_layout.setObjectName("lyoFileSection")
        
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
        self.combine_columns_label.setStyleSheet("color: gray; font-style: italic;")
        data_layout.addWidget(self.combine_columns_label)
        
        main_layout.addWidget(data_section)
        
        # Available groups and plot management
        groups_and_plots = QHBoxLayout()
        groups_and_plots.setObjectName("lyoGroupsAndPlots")
        
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
        analyze_button = QPushButton("Start all analyses")
        analyze_button.setObjectName("btnAnalyzeAll")
        analyze_button.clicked.connect(self.run_all_analyses)
        analyze_selected_button = QPushButton("Start selected analysis")
        analyze_selected_button.setObjectName("btnAnalyzeSelected")
        analyze_selected_button.clicked.connect(self.run_selected_analysis)
        multi_analyze_button = QPushButton("Start multi-dataset analysis")
        multi_analyze_button.setObjectName("btnMultiDatasetAnalyze")
        multi_analyze_button.clicked.connect(self.run_multi_dataset_analysis)
        outlier_button = QPushButton("Detect outliers")
        outlier_button.setObjectName("btnDetectOutliers")
        outlier_button.clicked.connect(self.run_outlier_detection)
        actions_layout.addWidget(analyze_button)
        actions_layout.addWidget(analyze_selected_button)
        actions_layout.addWidget(multi_analyze_button)
        actions_layout.addWidget(outlier_button)
        
        main_layout.addLayout(actions_layout)
        
        self.setCentralWidget(central_widget)
    
    def browse_file(self):
        """Opens a dialog to select an Excel or CSV file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Excel file", "", 
            "Excel files (*.xlsx *.xls);;CSV files (*.csv);;All files (*.*)"
        )
        
        if file_path:
            # Check if file exists and has a supported format
            if not os.path.exists(file_path):
                QMessageBox.critical(self, "Error", f"The file {file_path} does not exist.")
                return
                
            if not file_path.endswith(('.xlsx', '.xls', '.csv')):
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
            if self.file_path.endswith('.csv'):
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
        if index < 0 or not self.file_path or not self.file_path.endswith(('.xlsx', '.xls')):
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
    
    def open_advanced_tests(self):
        """Opens the dialog for advanced statistical tests"""
        import traceback  # Add this line
        
        if self.df is None:
            QMessageBox.warning(self, "Error", "No data loaded.")
            return
            
        # Phase 1: Select test and parameters
        data_format = self.detect_data_format(self.df)
        dialog = AdvancedTestDialog(parent=self, df=self.df, default_test=data_format)
        
        if dialog.exec_() == QDialog.Accepted:
            # Read test parameters
            test_type = None
            if dialog.mixedAnovaBtn.isChecked():
                test_type = "mixed_anova"
            elif dialog.rmAnovaBtn.isChecked():
                test_type = "repeated_measures_anova"
            elif dialog.twoWayAnovaBtn.isChecked():
                test_type = "two_way_anova"
                
            # Validate inputs based on test type
            between = [item.text() for item in dialog.betweenList.selectedItems()]
            within = [item.text() for item in dialog.withinList.selectedItems()]
            
            if test_type == "mixed_anova" and (not between or not within):
                QMessageBox.warning(self, "Error", "Mixed ANOVA requires both between and within factors!")
                return
            elif test_type == "repeated_measures_anova" and not within:
                QMessageBox.warning(self, "Error", "Repeated Measures ANOVA requires at least one within factor!")
                return
            elif test_type == "two_way_anova" and len(between) < 2:
                QMessageBox.warning(self, "Error", "Two-Way ANOVA requires at least two between factors!")
                return
    
            # NEW: Dialog to select save path for Excel file
            excel_path, _ = QFileDialog.getSaveFileName(
                self, "Save Excel results file", "",
                "Excel files (*.xlsx);;All files (*.*)"
            )
            if not excel_path:  # User cancelled
                return
                
            # Add .xlsx at the end if not present
            if not excel_path.lower().endswith('.xlsx'):
                excel_path += '.xlsx'
    
            # Phase 2: Check prerequisites and ask for transformation if needed
            df_copy = self.df.copy()
            
            try:
                # Prepare test and check prerequisites
                test_preparation = StatisticalTester.prepare_advanced_test(
                    df=df_copy,
                    test=test_type,
                    dv=dialog.dvCombo.currentText(),
                    subject=dialog.subjectCombo.currentText(),
                    between=between,
                    within=within
                )
                
                # Check if an error occurred
                if "error" in test_preparation:
                    QMessageBox.critical(self, "Error", f"Error preparing the test: {test_preparation['error']}")
                    return
                
                # Use the prepared results directly - no need for additional dialog
                # Phase 3: Perform test with specified Excel path using prepared results
                results = StatisticalTester.perform_advanced_test(
                    df=self.df,
                    test=test_type,
                    dv=dialog.dvCombo.currentText(),
                    subject=dialog.subjectCombo.currentText(),
                    between=between,
                    within=within,
                    alpha=0.05,
                    force_parametric=False,
                    # Pass the prepared results to avoid duplicate dialog
                    transformed_samples=test_preparation["transformed_samples"],
                    recommendation=test_preparation["recommendation"],
                    test_info=test_preparation["test_info"],
                    file_name=excel_path,
                    skip_excel=False
                )

                print("DEBUG: After RM ANOVA execution")
                print("DEBUG: Current results structure:", results.keys() if isinstance(results, dict) else type(results))
                if isinstance(results, dict) and 'error' in results and results['error']:
                    print("DEBUG: Error detected:", results['error'])
                    traceback.print_exc()

                QMessageBox.information(self, "Test completed", 
                    f"The advanced test was successfully performed.\nResults were saved in:\n{excel_path}")
            except Exception as e:
                # Add this except block to handle errors
                QMessageBox.critical(self, "Error", f"Error performing the test: {str(e)}")
                traceback.print_exc()

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
        # Get Excel filename for default filename
        default_filename = None
        if hasattr(self, 'file_path') and self.file_path:
            # Get filename without path and extension, add "_analyzed"
            base_filename = os.path.splitext(os.path.basename(self.file_path))[0]
            default_filename = f"{base_filename}_analyzed"
        
        # Open the main plot configuration dialog first
        dlg = PlotConfigDialog(groups, parent=self, default_filename=default_filename)
        if dlg.exec_() == dlg.Accepted:
            config = dlg.get_config()
            if config is None:
                return
            self.plot_configs.append(config)
            plot_item_text = f"Plot: {config.get('title') or ', '.join(config.get('groups', []))}"
            self.plots_list.addItem(plot_item_text)
            # Only show preview if 'create_plot' is checked
            if config.get('create_plot', False):
                self.preview_plot(len(self.plot_configs) - 1)
    
    def edit_plot_config(self, item):
        """Edits the configuration of a selected plot."""
        index = self.plots_list.row(item)
        
        if index < 0 or index >= len(self.plot_configs):
            return
        
        config = self.plot_configs[index]
        
        # Get Excel filename for default filename
        default_filename = None
        if hasattr(self, 'file_path') and self.file_path:
            # Get filename without path and extension, add "_analyzed"
            base_filename = os.path.splitext(os.path.basename(self.file_path))[0]
            default_filename = f"{base_filename}_analyzed"
        
        # KORREKTUR: Gehe zuerst zu PlotConfigDialog, nicht direkt zu PlotAestheticsDialog
        dlg = PlotConfigDialog(config.get('groups', []), parent=self, default_filename=default_filename)
        
        # Lade die bestehende Konfiguration in den Dialog (nur verfügbare Felder)
        if 'file_name' in config:
            dlg.file_name_edit.setText(config['file_name'])
        if 'dependent' in config:
            dlg.dependent_check.setChecked(config['dependent'])
        if 'create_plot' in config:
            dlg.create_plot_check.setChecked(config['create_plot'])
        # Note: error_type is now handled by PlotAestheticsDialog only
            
        if dlg.exec_() == dlg.Accepted:
            # Hole die neue Konfiguration und merge mit der alten
            new_config = dlg.get_config()
            # Keep appearance_settings if available
            if 'appearance_settings' in config:
                new_config['appearance_settings'] = config['appearance_settings']
            
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
            
            # Clear the preview if no plot is left
            if len(self.plot_configs) == 0:
                if hasattr(self, 'plot_preview_widget') and self.plot_preview_widget:
                    self.plot_preview_widget._show_placeholder()
                elif hasattr(self, 'figure') and hasattr(self, 'canvas'):
                    self.figure.clear()
                    self.canvas.draw()
    
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
                boxplots_colors = ["yellowgreen", "olivedrab", "gold", "deepskyblue", "orchid", "thistle"]
                violin_colors = ["thistle", "orchid", "gold", "deepskyblue", "yellowgreen", "olivedrab"]
                scatter_colors = ["tomato", "darksalmon", "deepskyblue", "orchid", "yellowgreen", "olivedrab"]
                
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
    
    def show_advanced_tests_help(self):
        """Shows detailed information about advanced ANOVA types and their applications."""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextBrowser, QPushButton
        
        dlg = QDialog(self)
        dlg.setWindowTitle("Advanced Statistical Tests - ANOVA Types")
        dlg.resize(900, 700)
        layout = QVBoxLayout(dlg)
        
        browser = QTextBrowser()
        browser.setHtml("""
            <h2>Advanced ANOVA Types</h2>
            
            <h3>1. Repeated Measures ANOVA (Within-Subjects)</h3>
            <p><b>What it does:</b> Analyzes data where the same subjects are measured multiple times under different conditions or at different time points.</p>
            
            <p><b>Data structure required:</b></p>
            <ul>
                <li><b>Subject column:</b> Unique identifier for each participant (e.g., Subject_01, Subject_02...)</li>
                <li><b>Within factor:</b> The repeated condition (e.g., Time: Pre, Post, Follow-up)</li>
                <li><b>Dependent variable:</b> The measured outcome (e.g., blood pressure, test score)</li>
            </ul>
            
            <p><b>Example experiments:</b></p>
            <ul>
                <li><b>Drug efficacy over time:</b> Measure blood pressure in the same patients before treatment, after 1 week, and after 1 month</li>
                <li><b>Learning effects:</b> Test the same students' performance on 3 different learning methods</li>
                <li><b>Cognitive tasks:</b> Measure reaction time under different stimulus conditions in the same participants</li>
                <li><b>Exercise intervention:</b> Measure fitness scores in athletes at baseline, mid-training, and post-training</li>
            </ul>
            
            <p><b>Advantages:</b> Controls for individual differences, requires fewer participants, higher statistical power</p>
            <p><b>Assumptions:</b> Sphericity (equal variances of differences), normality of differences</p>
            
            <hr>
            
            <h3>2. Mixed ANOVA (Between + Within Factors)</h3>
            <p><b>What it does:</b> Combines between-subjects factors (different groups) with within-subjects factors (repeated measures).</p>
            
            <p><b>Data structure required:</b></p>
            <ul>
                <li><b>Subject column:</b> Unique identifier for each participant</li>
                <li><b>Between factor:</b> Groups that subjects belong to (e.g., Treatment: Drug vs Placebo)</li>
                <li><b>Within factor:</b> Repeated conditions (e.g., Time: Pre, Post, Follow-up)</li>
                <li><b>Dependent variable:</b> The measured outcome</li>
            </ul>
            
            <p><b>Example experiments:</b></p>
            <ul>
                <li><b>Clinical trial:</b> Compare drug vs placebo (between) measured at pre/post/follow-up (within)</li>
                <li><b>Gender and learning:</b> Male vs female students (between) tested on 3 different learning methods (within)</li>
                <li><b>Age groups and memory:</b> Young vs elderly participants (between) with memory tests under 3 difficulty levels (within)</li>
                <li><b>Diet intervention:</b> Low-carb vs low-fat diet groups (between) with weight measured monthly over 6 months (within)</li>
            </ul>
            
            <p><b>Analysis results:</b></p>
            <ul>
                <li><b>Main effect of between factor:</b> Do groups differ overall?</li>
                <li><b>Main effect of within factor:</b> Do measurements change over time/conditions?</li>
                <li><b>Interaction effect:</b> Do groups respond differently over time/conditions?</li>
            </ul>
            
            <p><b>Advantages:</b> Most comprehensive design, tests multiple hypotheses simultaneously</p>
            
            <hr>
            
            <h3>3. Two-Way ANOVA (Between-Subjects Only)</h3>
            <p><b>What it does:</b> Analyzes the effects of two independent categorical factors on a continuous outcome, where each subject belongs to only one combination of factor levels.</p>
            
            <p><b>Data structure required:</b></p>
            <ul>
                <li><b>Factor A:</b> First categorical variable (e.g., Treatment: Drug A, Drug B, Placebo)</li>
                <li><b>Factor B:</b> Second categorical variable (e.g., Gender: Male, Female)</li>
                <li><b>Dependent variable:</b> The measured outcome</li>
                <li>Each subject appears in only one cell of the design</li>
            </ul>
            
            <p><b>Example experiments:</b></p>
            <ul>
                <li><b>Drug and gender effects:</b> Test 3 medications (Factor A) in both male and female patients (Factor B) on pain relief</li>
                <li><b>Teaching methods and class size:</b> Compare 2 teaching methods (Factor A) in small vs large classes (Factor B) on test scores</li>
                <li><b>Fertilizer and plant variety:</b> Test 3 fertilizers (Factor A) on 2 plant varieties (Factor B) measuring growth</li>
                <li><b>Exercise and diet:</b> Compare 2 exercise programs (Factor A) with 3 diet types (Factor B) on weight loss</li>
            </ul>
            
            <p><b>Analysis results:</b></p>
            <ul>
                <li><b>Main effect of Factor A:</b> Do levels of Factor A differ when averaging across Factor B?</li>
                <li><b>Main effect of Factor B:</b> Do levels of Factor B differ when averaging across Factor A?</li>
                <li><b>Interaction effect:</b> Does the effect of Factor A depend on the level of Factor B?</li>
            </ul>
            
            <p><b>Design considerations:</b> Requires larger sample sizes, each cell should have equal sample sizes when possible</p>
            
            <hr>
        """)
        
        layout.addWidget(browser)
        
        btn = QPushButton("OK")
        btn.clicked.connect(dlg.accept)
        layout.addWidget(btn)
        
        dlg.exec_()
    
    def show_getting_started_help(self):
        """Shows a comprehensive getting started guide for first-time users."""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextBrowser, QPushButton
        
        dlg = QDialog(self)
        dlg.setWindowTitle("Getting Started with BioMedStatX")
        dlg.resize(1000, 800)
        layout = QVBoxLayout(dlg)
        
        browser = QTextBrowser()
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
            
            <h4>B) Advanced ANOVA Tests</h4>
            <p>For more complex designs, use <b>Analysis → Run Advanced Tests</b>:</p>
            <ul>
                <li><b>Repeated Measures ANOVA:</b> Same subjects measured multiple times</li>
                <li><b>Two-Way ANOVA:</b> Two independent factors (e.g., treatment × gender)</li>
                <li><b>Mixed ANOVA:</b> Combination of between- and within-subject factors</li>
            </ul>
            
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

                # 1. For each value column (e.g. gene): plot configuration & analysis
                for i, column in enumerate(self.selected_columns):
                    print(f"Configuring plot for column {column} ({i+1}/{len(self.selected_columns)})")
                    print(f"DEBUG MULTI:   ► iterating dataset #{i+1}: '{column}'")
                    print(f"DEBUG MULTI:      → plot_configs keys so far: {list(plot_configs.keys())}")
                    
                    # Get Excel filename for default filename
                    default_filename = None
                    if hasattr(self, 'file_path') and self.file_path:
                        # Get filename without path and extension, add column name and "_analyzed"
                        base_filename = os.path.splitext(os.path.basename(self.file_path))[0]
                        default_filename = f"{base_filename}_{column}_analyzed"
                    
                    # --- Use the main PlotConfigDialog for each dataset ---
                    dlg = PlotConfigDialog(selected_groups, parent=self, default_filename=default_filename)
                    dlg.setWindowTitle(f"Configure plot for '{column}' ({i+1}/{len(self.selected_columns)})")
                    # Pre-fill the file name with the column name
                    dlg.file_name_edit.setText(f"{column}_analysis")
                    if dlg.exec_() != dlg.Accepted:
                        print(f"Configuration for {column} cancelled")
                        continue
                    plot_config = dlg.get_config()
                    # Ensure groups are stored
                    plot_config['groups'] = selected_groups.copy() if isinstance(selected_groups, list) else list(selected_groups)
                    plot_configs[column] = plot_config
                    print(f"Configuration for {column} saved")

                if not plot_configs:
                    QMessageBox.warning(self, "Aborted", "No datasets were configured.")
                    return

                # Progress dialog for analysis
                print("Creating progress dialog...")
                progress = QMessageBox()
                progress.setWindowTitle("Analysis running...")
                progress.setText("The multi-dataset analysis is running. Please wait...")
                progress.setStandardButtons(QMessageBox.NoButton)
                progress.show()
                QApplication.processEvents()

                # 2. Analyze all configured datasets in sequence
                for i, (column, plot_config) in enumerate(plot_configs.items()):
                    progress_text = f"({i+1}/{len(plot_configs)})"
                    progress.setText(f"Analyzing dataset {i+1}/{len(plot_configs)}: {column}")
                    QApplication.processEvents()
                    
                    print(f"Starting analysis for {column}...")
                    try:
                        # --- Prepare values for analyze() ---
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
                            'dialog_column': column,
                            'dialog_progress': progress_text
                        }
                        # Apply appearance settings if available
                        if 'appearance_settings' in plot_config:
                            appearance = plot_config['appearance_settings']
                            # Plot type and dimensions
                            kwargs['plot_type'] = appearance.get('plot_type', 'Bar')
                            kwargs['dpi'] = appearance.get('dpi', 300)
                            kwargs['aspect'] = appearance.get('aspect', None)
                            
                            # Typography
                            kwargs['font_main'] = appearance.get('font_main', 'Arial')
                            kwargs['font_axis'] = appearance.get('font_axis', 'Arial')
                            kwargs['show_title'] = appearance.get('show_title', True)
                            kwargs['fontsize_title'] = appearance.get('fontsize_title', 12)
                            kwargs['fontsize_axis'] = appearance.get('fontsize_axis', 9)
                            kwargs['fontsize_ticks'] = appearance.get('fontsize_ticks', 7)
                            kwargs['fontsize_groupnames'] = appearance.get('fontsize_groupnames', 8)
                            
                            # Lines and axes
                            kwargs['axis_linewidth'] = appearance.get('axis_linewidth', 0.7)
                            kwargs['bar_linewidth'] = appearance.get('bar_linewidth', 1.0)
                            kwargs['gridline_width'] = appearance.get('gridline_width', 0.5)
                            kwargs['grid'] = appearance.get('grid', False)
                            kwargs['minor_ticks'] = appearance.get('minor_ticks', False)
                            kwargs['logy'] = appearance.get('logy', False)
                            kwargs['logx'] = appearance.get('logx', False)
                            kwargs['despine'] = appearance.get('despine', True)
                            
                            # Colors and appearance
                            kwargs['alpha'] = appearance.get('alpha', 0.8)
                            kwargs['bar_edge_color'] = appearance.get('bar_edge_color', 'black')
                            
                            # Annotations
                            kwargs['refline'] = appearance.get('refline', False)
                            kwargs['panel_labels'] = appearance.get('panel_labels', False)
                            kwargs['value_annotations'] = appearance.get('value_annotations', False)
                            kwargs['significance_mode'] = appearance.get('significance_mode', 'letters')
                            
                            # Export options
                            kwargs['embed_fonts'] = appearance.get('embed_fonts', False)
                            kwargs['add_metadata'] = appearance.get('add_metadata', False)
                            
                            # Handle hatches consistently with colors
                            if 'hatches' in appearance:
                                kwargs['hatches'] = [appearance['hatches'].get(group, '') for group in plot_config['groups']]

                        # Two-Way ANOVA factors
                        if 'additional_factors' in plot_config and plot_config['additional_factors']:
                            kwargs['additional_factors'] = plot_config['additional_factors']
                            
                        # Specific comparisons
                        if 'comparisons' in plot_config and plot_config['comparisons']:
                            compare_list = [(comp['group1'], comp['group2']) for comp in plot_config['comparisons']]
                            kwargs['compare'] = compare_list
                            
                        # Colors and hatches
                        colors = [plot_config['colors'].get(group, DEFAULT_COLORS[j % len(DEFAULT_COLORS)])
                                for j, group in enumerate(plot_config['groups'])]
                        kwargs['colors'] = colors
                        
                    # Run analysis (without individual Excel export)
                        print(f"Calling AnalysisManager.analyze() for {column} with skip_excel=True...")
                        start_time = time.time()  # Track execution time
                        results = AnalysisManager.analyze(**kwargs)

                        # --- Export with metadata and font embedding if requested ---
                        if plot_config.get('create_plot', True) and (
                            plot_config.get('embed_fonts', False) or plot_config.get('add_metadata', False)
                        ):
                            filename = plot_config.get('file_name', f"{column}_analysis")
                            filetype = "pdf"  # or "svg", depending on your UI
                            out_path = os.path.join(os.getcwd(), f"{filename}.{filetype}")
                            fig = results.get('figure', None)
                            if fig is None:
                                plt = get_matplotlib()
                                fig = plt.gcf()
                            DataVisualizer.export_with_metadata(
                                fig, out_path,
                                metadata={"Title": plot_config.get('title', column), "Description": ""},
                                embed_fonts=plot_config.get('embed_fonts', True),
                                dpi=plot_config.get('dpi', 300) if 'dpi' in plot_config else 300,
                                filetype=filetype
                            )
                        analysis_time = time.time() - start_time
                        
                        # Validate results structure
                        print(f"Analysis for {column} completed in {analysis_time:.2f} seconds")
                        print(f"Result type: {type(results)}")
                        print(f"Keys in results: {list(results.keys()) if isinstance(results, dict) else 'Not a dictionary'}")
                        
                        # Check for critical keys
                        if isinstance(results, dict):
                            if 'error' in results and results['error']:
                                print(f"WARNING: Error found in results: {results['error']}")
                            if 'test' in results:
                                print(f"Test performed: {results['test']}")
                            if 'p_value' in results:
                                print(f"p-value: {results['p_value']}")
                        
                        # Store results
                        all_results[column] = results
                        print(f"✓ Results for '{column}' successfully stored in all_results dictionary")
                        print(f"Analysis for {column} completed")

                    except Exception as e:
                        QMessageBox.critical(self, "Error",
                                        f"Error analyzing {column}: {str(e)}")
                        traceback.print_exc()
                        
                print("DEBUG MULTI: Finished analyzing all columns.")
                print("DEBUG MULTI: all_results keys:", list(all_results.keys()))
                print("DEBUG MULTI: all_results contents:")
                for key, value in all_results.items():
                    print(f"  {key}: {type(value)} with keys {list(value.keys()) if isinstance(value, dict) else 'not a dict'}")

                if all_results:
                    print("DEBUG MULTI: About to call export_multi_dataset_results()")
                    excel_path = os.path.join(output_dir, "All_Datasets_Analysis.xlsx")
                    print(f"DEBUG MULTI: Excel path will be: {excel_path}")
                    ResultsExporter.export_multi_dataset_results(all_results, excel_path)
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
                else:
                    print("DEBUG MULTI: all_results is empty, skipping export")
                    QMessageBox.warning(self, "No Results", "No analysis results were generated.")

                # Ensure the progress dialog is closed
                try:
                    progress.close()
                    progress = None  # Delete reference
                except Exception as e:
                    print(f"Error closing progress dialog: {str(e)}")
                    
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

if __name__ == "__main__":
    try:
        # Timer-Warnungen unterdrücken
        import os
        os.environ["QT_LOGGING_RULES"] = "qt.core.qobject.timer=false"
        
        # Apply stylesheet if available
        try:
            with open(resource_path("assets/StyleSheet.qss"), "r", encoding="utf-8") as f:
                stylesheet = f.read()
                print("Stylesheet loaded successfully")
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