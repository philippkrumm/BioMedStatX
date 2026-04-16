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


from statistical_analyzer_dialogs import (
    ColumnSelectionDialog,
    DebugConsoleWindow,
    ExploratoryMatrixDialog,
    GroupSelectionDialog,
    HelpHubDialog,
    OutlierDetectionDialog,
    PairwiseComparisonDialog,
    TwoWayAnovaDialog,
)

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
        actions_layout.addWidget(analyze_button)
        actions_layout.addWidget(analyze_selected_button)
        
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

        plot_config = self.plot_configs[current_row]
        suggested = plot_config.get('file_name') or "_".join(plot_config['groups'])
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Analysis Results", suggested + ".xlsx",
            "Excel Files (*.xlsx);;All Files (*)"
        )
        if not file_path:
            return
        if not file_path.lower().endswith('.xlsx'):
            file_path += '.xlsx'
        output_dir = os.path.dirname(file_path) or os.getcwd()
        file_name_override = os.path.splitext(os.path.basename(file_path))[0]

        try:
            self.run_single_analysis(plot_config, output_dir, file_name_override=file_name_override)
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
    
    def run_single_analysis(self, plot_config, output_dir=None, file_name_override=None):
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
            'file_name': file_name_override or plot_config.get('file_name') or "_".join(plot_config['groups']),
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



from statistical_analyzer_autopilot_pipeline import (
    _load_auto_pilot_stylesheet,
    attach_autopilot_methods,
)

attach_autopilot_methods(StatisticalAnalyzerApp)

class _CrashSafeApp(QApplication):
    """QApplication subclass that catches exceptions in Qt event handlers."""
    def notify(self, receiver, event):
        try:
            return super().notify(receiver, event)
        except Exception:
            import traceback as _tb
            _tb.print_exc()
            return False


def _install_global_excepthook():
    """Install a global exception hook that logs crashes to a file and shows a dialog."""
    import traceback as _tb
    log_path = os.path.join(os.path.dirname(__file__), "..", "crash_log.txt")

    def _excepthook(exc_type, exc_value, exc_tb):
        msg = "".join(_tb.format_exception(exc_type, exc_value, exc_tb))
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                import datetime
                f.write(f"\n=== {datetime.datetime.now()} ===\n{msg}\n")
        except Exception:
            pass
        print(msg, file=sys.stderr)
        # Show dialog if a QApplication exists
        try:
            if QApplication.instance():
                QMessageBox.critical(
                    None,
                    "Unerwarteter Fehler",
                    f"Ein Fehler ist aufgetreten:\n\n{exc_type.__name__}: {exc_value}\n\n"
                    f"Details wurden in crash_log.txt gespeichert.",
                )
        except Exception:
            pass

    sys.excepthook = _excepthook


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

        _install_global_excepthook()

        # Apply stylesheet if available
        try:
            stylesheet = _load_auto_pilot_stylesheet()
            print("Stylesheet loaded successfully" if stylesheet else "No stylesheet found")
        except:
            stylesheet = ""
            print("No stylesheet found")

        app = _CrashSafeApp(sys.argv)
        app.setStyleSheet(stylesheet)
        window = StatisticalAnalyzerApp()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Error starting application: {str(e)}")
        traceback.print_exc()
