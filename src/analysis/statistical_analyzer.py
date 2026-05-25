import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
if sys.platform == "darwin":
    # Keep Qt environment clean so PyQt uses one runtime only.
    os.environ.pop("DYLD_FRAMEWORK_PATH", None)
    os.environ.pop("DYLD_LIBRARY_PATH", None)
    os.environ.pop("QT_PLUGIN_PATH", None)
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
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

# Initialize central logging before anything else may emit messages.
from core.logger_config import configure_logging
configure_logging()

# Initialize lazy loading system
from core.lazy_imports import preload_critical_modules, get_matplotlib_pyplot as get_matplotlib
preload_critical_modules()

from analysis.stats_functions import (
    OutlierDetector, OUTLIER_IMPORTS_AVAILABLE
)
# Import updater for auto-update functionality
try:
    from core.updater import AutoUpdater
    UPDATE_AVAILABLE = True
except ImportError:
    UPDATE_AVAILABLE = False
    print("Warning: Updater module not available")
try:
    from core.help_content import HELP_RECIPES
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


from ui.dialogs.statistical_analyzer_dialogs import (
    DebugConsoleWindow,
    ExploratoryMatrixDialog,
    HelpHubDialog,
    OutlierDetectionDialog,
)

from autopilot.statistical_analyzer_autopilot_pipeline import (
    AutopilotMixin,
    _load_auto_pilot_stylesheet,
)


class StatisticalAnalyzerApp(AutopilotMixin, QMainWindow):
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
