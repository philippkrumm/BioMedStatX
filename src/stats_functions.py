import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import xlsxwriter
import os
import warnings
import string
from itertools import combinations
from datetime import datetime
from matplotlib.ticker import ScalarFormatter
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QRadioButton, QDialogButtonBox
from PyQt5.QtCore import Qt
from decisiontreevisualizer import DecisionTreeVisualizer
from lazy_imports import (
    get_pingouin, get_scipy_stats, get_seaborn,
    get_statsmodels_multitest, get_pairwise_tukeyhsd, get_scikit_posthocs,
)

# Late import functions to avoid circular imports
def get_results_exporter():
    """Get ResultsExporter class lazily"""
    from resultsexporter import ResultsExporter
    return ResultsExporter

def get_export_dispatcher():
    """Get ExportDispatcher class lazily"""
    from export_dispatcher import ExportDispatcher
    return ExportDispatcher

def get_data_visualizer():
    """Get DataVisualizer class lazily"""
    from datavisualizer import DataVisualizer
    return DataVisualizer

def get_statistical_tester():
    """Get StatisticalTester class lazily"""
    from statisticaltester import StatisticalTester
    return StatisticalTester

def get_stats_module():
    """Get scipy.stats — delegates to canonical lazy_imports loader"""
    return get_scipy_stats()

def get_pingouin_module():
    """Get pingouin — delegates to canonical lazy_imports loader"""
    return get_pingouin()

def get_multicomp_module():
    """Get (pairwise_tukeyhsd, multipletests) via canonical lazy_imports loaders"""
    return get_pairwise_tukeyhsd(), get_statsmodels_multitest()

def get_boxcox_functions():
    """Get scipy boxcox and boxcox_normmax via canonical lazy_imports loader"""
    stats = get_scipy_stats()
    return stats.boxcox, stats.boxcox_normmax

# --------------------------------------------------------------
#  Fallback QApplication to prevent dialogs blocking when script
#  is run purely via CLI
# --------------------------------------------------------------
from PyQt5.QtWidgets import QApplication
import time
from contextlib import contextmanager

@contextmanager
def working_directory(path):
    """Context manager for safely changing directories"""
    previous_dir = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(previous_dir)

print(f"DEBUG: RUNNING FILE VERSION FROM {time.time()} - {os.path.abspath(__file__)}")

warnings.simplefilter(action='ignore', category=FutureWarning)

def safe_format(val, fmt="{:.4f}", none_text="N/A"):
    """
    Format a value safely: if numeric, use the given format;
    if None, return the none_text; otherwise, cast to string.
    """
    if isinstance(val, (float, int)):
        try:
            return fmt.format(val)
        except Exception:
            return str(val)
    elif val is None:
        return none_text
    else:
        return str(val)


class AssumptionVisualizer:
    """
    Creates visual examinations for statistical test assumptions (normality and homoscedasticity).
    """
    
    @staticmethod
    def create_normality_plot(data_dict, title_suffix="", transformation=None, results=None):
        """
        Create Q-Q plot for normality examination using MODEL RESIDUALS (not individual groups).
        
        Parameters:
        -----------
        data_dict : dict
            Dictionary with group names as keys and data arrays as values (for fallback)
        title_suffix : str
            Suffix to add to the plot title (e.g., "Before Transformation")
        transformation : str
            Name of transformation applied (if any)
        results : dict
            Results dictionary containing model residuals if available
            
        Returns:
        --------
        str
            Path to the saved plot file, or None if failed
        """
        try:
            import tempfile
            import time
            stats = get_stats_module()
            
            # Try to get model residuals first (correct approach)
            residuals = None
            if results and 'model_residuals' in results:
                residuals = results['model_residuals']
                data_source = "Model Residuals"
            elif results and 'residuals' in results:
                residuals = results['residuals']
                data_source = "Model Residuals"
            else:
                # Fallback: combine all data from groups (less ideal but better than nothing)
                if not data_dict:
                    return None
                all_values = []
                for values in data_dict.values():
                    clean_vals = [v for v in values if not (pd.isna(v) if pd else np.isnan(v))]
                    all_values.extend(clean_vals)
                residuals = np.array(all_values)
                data_source = "Combined Group Data"
            
            if residuals is None or len(residuals) < 3:
                return None
            
            # Remove NaN values from residuals
            clean_residuals = np.array([r for r in residuals if not (pd.isna(r) if pd else np.isnan(r))])
            
            if len(clean_residuals) < 3:
                return None
            
            # Create single Q-Q plot - use FIXED size for consistency between before/after
            # Fixed size ensures both before and after plots are identical
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))  # Fixed 12x6 for all QQ plots
            
            # Create Q-Q plot of residuals
            stats.probplot(clean_residuals, dist="norm", plot=ax)
            
            # Customize plot
            ax.set_title(f"Normality Check - Q-Q Plot of {data_source}\n(n={len(clean_residuals)} observations)", 
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel("Theoretical Normal Quantiles", fontsize=12)
            ax.set_ylabel("Sample Quantiles", fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Style the reference line
            line = ax.get_lines()[1]  # Second line is the reference line
            line.set_color('red')
            line.set_linewidth(3)
            line.set_alpha(0.8)
            
            # Style the data points
            points = ax.get_lines()[0]
            points.set_markersize(6)
            points.set_alpha(0.7)
            
            # Add transformation info to title if applicable
            transform_text = f" ({transformation})" if transformation and transformation.lower() != "none" else ""
            fig.suptitle(f"Normality Examination{transform_text}{title_suffix}", 
                        fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            
            # Save to temporary file
            fd, temp_path = tempfile.mkstemp(suffix='.png', prefix='normality_plot_')
            os.close(fd)
            
            # Use consistent saving parameters - no tight bbox to ensure consistent sizing
            fig.savefig(temp_path, dpi=300, bbox_inches=None, facecolor='white', pad_inches=0.2)
            plt.close(fig)
            
            print(f"DEBUG: Generated normality plot: {temp_path}")
            return temp_path
            
        except Exception as e:
            print(f"DEBUG: Error creating normality plot: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    @staticmethod
    def create_homoscedasticity_plot(data_dict, title_suffix="", transformation=None):
        """
        Create boxplots for homoscedasticity (variance homogeneity) examination.
        
        Parameters:
        -----------
        data_dict : dict
            Dictionary with group names as keys and data arrays as values
        title_suffix : str
            Suffix to add to the plot title (e.g., "Before Transformation")
        transformation : str
            Name of transformation applied (if any)
            
        Returns:
        --------
        str
            Path to the saved plot file, or None if failed
        """
        try:
            import tempfile
            import time
            
            if not data_dict:
                return None
            
            # Create figure with FIXED size for consistency between before/after
            # Fixed size ensures both before and after plots are identical
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))  # Fixed 12x6 for all boxplots
            
            # Prepare data for boxplot
            group_names = []
            group_data = []
            
            for group_name, values in data_dict.items():
                # Remove NaN values
                clean_values = [v for v in values if not (pd.isna(v) if pd else np.isnan(v))]
                if clean_values:
                    group_names.append(f"{group_name}\n(n={len(clean_values)})")
                    group_data.append(clean_values)
            
            if not group_data:
                ax.text(0.5, 0.5, "No valid data for plot", ha='center', va='center', 
                       transform=ax.transAxes, fontsize=14)
                ax.set_title("Variance Homogeneity Examination", fontsize=16, fontweight='bold')
                return None
            
            # Matplotlib >=3.9 renamed `labels` to `tick_labels`.
            try:
                bp = ax.boxplot(
                    group_data,
                    tick_labels=group_names,
                    patch_artist=True,
                    notch=True,
                    showmeans=True,
                    meanline=True,
                )
            except TypeError:
                bp = ax.boxplot(
                    group_data,
                    labels=group_names,
                    patch_artist=True,
                    notch=True,
                    showmeans=True,
                    meanline=True,
                )
            
            # Color the boxes with distinct colors
            colors = plt.cm.Set2(np.linspace(0, 1, len(group_data)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.8)
                patch.set_linewidth(1.5)
            
            # Style other elements
            for element in ['whiskers', 'fliers', 'medians', 'caps']:
                plt.setp(bp[element], linewidth=1.5)
            
            # Customize plot appearance
            ax.set_ylabel('Values', fontsize=14, fontweight='bold')
            ax.set_xlabel('Groups', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45, labelsize=12)
            ax.tick_params(axis='y', labelsize=12)
            
            # Add main title
            transform_text = f" ({transformation})" if transformation and transformation.lower() != "none" else ""
            ax.set_title(f"Variance Homogeneity Examination - Boxplots{transform_text}{title_suffix}", 
                        fontsize=16, fontweight='bold', pad=20)
            
            plt.tight_layout()
            
            # Save to temporary file
            fd, temp_path = tempfile.mkstemp(suffix='.png', prefix='homoscedasticity_plot_')
            os.close(fd)
            
            # Use consistent saving parameters - no tight bbox to ensure consistent sizing
            fig.savefig(temp_path, dpi=300, bbox_inches=None, facecolor='white', pad_inches=0.2)
            plt.close(fig)
            
            print(f"DEBUG: Generated homoscedasticity plot: {temp_path}")
            return temp_path
            
        except Exception as e:
            print(f"DEBUG: Error creating homoscedasticity plot: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    @staticmethod
    def generate_assumption_plots(results):
        """
        Generate both normality and homoscedasticity plots for the given results.
        Creates before/after transformation plots when applicable.
        
        Parameters:
        -----------
        results : dict
            Results dictionary containing test data and transformation info
            
        Returns:
        --------
        dict
            Dictionary with plot paths: {
                'normality_before': path,
                'normality_after': path,
                'homoscedasticity_before': path,  
                'homoscedasticity_after': path
            }
        """
        plot_paths = {
            'normality_before': None,
            'normality_after': None,
            'homoscedasticity_before': None,
            'homoscedasticity_after': None
        }
        
        try:
            # Get original data
            raw_data = results.get('raw_data', results.get('original_data', {}))
            if not raw_data:
                print("DEBUG: No raw data found for assumption plots")
                return plot_paths
            
            # Get transformation info
            transformation = results.get('transformation', 'None')
            transformed_data = results.get('raw_data_transformed', results.get('transformed_data', {}))
            
            # Filter out non-data keys
            raw_data_filtered = {k: v for k, v in raw_data.items() if str(k).lower() not in ['group', 'sample', '']}
            
            # Generate BEFORE transformation plots
            if raw_data_filtered:
                print(f"DEBUG: Generating BEFORE plots for {len(raw_data_filtered)} groups: {list(raw_data_filtered.keys())}")
                plot_paths['normality_before'] = AssumptionVisualizer.create_normality_plot(
                    raw_data_filtered, " - Before Transformation" if transformation and transformation.lower() != 'none' else "",
                    results=results
                )
                print(f"DEBUG: Q-Q plot BEFORE path: {plot_paths['normality_before']}")
                
                plot_paths['homoscedasticity_before'] = AssumptionVisualizer.create_homoscedasticity_plot(
                    raw_data_filtered, " - Before Transformation" if transformation and transformation.lower() != 'none' else ""
                )
                print(f"DEBUG: Boxplot BEFORE path: {plot_paths['homoscedasticity_before']}")
            else:
                print("DEBUG: No valid raw data found after filtering")
            
            # Generate AFTER transformation plots (if transformation was applied)
            if transformed_data and transformation and transformation.lower() != 'none':
                transformed_filtered = {k: v for k, v in transformed_data.items() if str(k).lower() not in ['group', 'sample', '']}
                if transformed_filtered:
                    print(f"DEBUG: Generating AFTER plots for {len(transformed_filtered)} groups: {list(transformed_filtered.keys())}")
                    plot_paths['normality_after'] = AssumptionVisualizer.create_normality_plot(
                        transformed_filtered, " - After Transformation", transformation, results=results
                    )
                    plot_paths['homoscedasticity_after'] = AssumptionVisualizer.create_homoscedasticity_plot(
                        transformed_filtered, " - After Transformation", transformation
                    )
                    print(f"DEBUG: Q-Q plot AFTER path: {plot_paths['normality_after']}")
                    print(f"DEBUG: Boxplot AFTER path: {plot_paths['homoscedasticity_after']}")
            
            # Track all generated files for cleanup
            ResultsExporter = get_results_exporter()
            for plot_path in plot_paths.values():
                if plot_path:
                    ResultsExporter.track_temp_file(plot_path)
            
            return plot_paths
            
        except Exception as e:
            print(f"DEBUG: Error generating assumption plots: {str(e)}")
            import traceback
            traceback.print_exc()
            return plot_paths
    

# Extracted post-hoc framework (compatibility re-exports)
from posthoc_core import (
    PostHocAnalyzer,
    TwoWayPostHocAnalyzer,
    MixedAnovaPostHocAnalyzer,
    RMAnovaPostHocAnalyzer,
    PostHocStatistics,
    TukeyHSD,
    GamesHowellTest,
    DunnettTest,
    DunnTest,
    DependentPostHoc,
    PostHocFactory,
)

class DataImporter:
    @staticmethod
    def import_data(file_path, sheet_name=0, group_col="Group", value_cols=None, combine_columns=False):
        file_path = os.path.abspath(file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} was not found.")
        if not os.path.isfile(file_path):
            raise ValueError(f"Path does not point to a file: {file_path}")
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        if group_col not in df.columns:
            raise ValueError(f"The group column '{group_col}' was not found. Available columns: {', '.join(df.columns)}")
        if value_cols is None:
            numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != group_col]
            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found that can be used as measurements.")
            value_cols = numeric_cols
        for col in value_cols:
            if col not in df.columns:
                raise ValueError(f"The value column '{col}' was not found. Available columns: {', '.join(df.columns)}")
        groups = sorted(df[group_col].unique())
        samples = {}
        if combine_columns:
            for group in groups:
                combined_values = []
                for col in value_cols:
                    values = df[df[group_col] == group][col].dropna().tolist()
                    combined_values.extend(values)
                samples[group] = combined_values
        else:  # if combine_columns=False
            if len(value_cols) > 1:
                print("Warning: Multiple value columns specified, but combine_columns=False. Only the first column will be used.")
            
            for group in groups:
                values = df[df[group_col] == group][value_cols[0]].dropna().tolist()
                samples[group] = values
        return samples, df

try:
    import scikit_posthocs as sp
    HAS_SCPH = True
except ImportError:
    HAS_SCPH = False

# GLMMTwoWayANOVA, GEERMANOVA, GLMMMixedANOVA, auto_anova_decision removed (dead code).


            
class UIDialogManager:
    @staticmethod
    def _ensure_qt_application():
        """Create QApplication lazily only when a dialog is actually needed."""
        app = QApplication.instance()
        if app is not None:
            return app

        if os.environ.get("PYTEST_CURRENT_TEST"):
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

        return QApplication([])

    @staticmethod
    def _configure_dialog(dialog, object_name=None):
        if object_name:
            dialog.setObjectName(object_name)
        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)

    @staticmethod
    def select_posthoc_test_dialog(parent=None, progress_text=None, column_name=None, default_method=None, equal_variance=None):
        UIDialogManager._ensure_qt_application()
        dialog = QDialog(parent)
        UIDialogManager._configure_dialog(dialog, object_name="posthocSelectionDialog")
        layout = QVBoxLayout(dialog)

        # Set window title
        title = "Select Post-hoc Test"
        if column_name:
            title += f" for '{column_name}'"
        if progress_text:
            title += f" {progress_text}"
        dialog.setWindowTitle(title)

        info_text = "The ANOVA has revealed significant differences. Please select a post-hoc test:"
        if progress_text and ("two_way_anova" in progress_text or "mixed_anova" in progress_text or "repeated_measures_anova" in progress_text):
            info_text = ("The advanced ANOVA has revealed significant differences. For advanced ANOVAs, "
                        "paired t-tests are often preferred to examine specific interaction effects. "
                        "Please select a post-hoc test:")
        elif progress_text and "two_way_anova" in progress_text:
            info_text = ("The Two-Way ANOVA has revealed significant differences. For Two-Way ANOVA, "
                        "paired t-tests are often preferred to examine specific interaction effects. "
                        "Please select a post-hoc test:")
        
        info = QLabel(info_text)
        info.setWordWrap(True)
        layout.addWidget(info)

        # RadioButtons for post-hoc tests - options depend on context
        if progress_text and ("two_way_anova" in progress_text or "mixed_anova" in progress_text or "repeated_measures_anova" in progress_text):
            # For advanced ANOVAs: only offer Tukey and Custom paired t-tests (no Dunnett)
            options = [
                ("Tukey-HSD Test (compares all pairs)", "tukey"),
                ("Custom paired t-tests (you select specific pairs to compare)", "paired_custom"),
            ]
        else:
            # For One-Way ANOVA: offer all options
            options = [
                ("Tukey-HSD Test (compares all pairs, equal variances assumed)", "tukey"),
                ("Dunnett Test (compares all groups against ONE control group)", "dunnett"),
                ("Custom paired t-tests (you select specific pairs to compare)", "paired_custom"),
            ]
            if equal_variance is False:
                options.insert(1, ("Games-Howell Test (compares all pairs, robust to unequal variances)", "games_howell"))
        radio_buttons = []
        for label, value in options:
            rb = QRadioButton(label)
            layout.addWidget(rb)
            radio_buttons.append((rb, value))
        
        # Set default selection based on context
        if default_method is None:
            default_method = "tukey"  # Original default
        
        for i, (rb, value) in enumerate(radio_buttons):
            if value == default_method:
                rb.setChecked(True)
                break
        else:
            # If default_method not found, default to first option
            radio_buttons[0][0].setChecked(True)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        if dialog.exec_() == QDialog.Accepted:
            for rb, value in radio_buttons:
                if rb.isChecked():
                    return value
        return None

    @staticmethod
    def select_nonparametric_posthoc_dialog(parent=None, progress_text=None, column_name=None):
        """
        Dialog for nonparametric post-hoc tests: Dunn or Mann-Whitney U (custom pairs)
        """
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QRadioButton, QDialogButtonBox

        UIDialogManager._ensure_qt_application()
        dialog = QDialog(parent)
        UIDialogManager._configure_dialog(dialog, object_name="nonparamPosthocDialog")
        layout = QVBoxLayout(dialog)

        title = "Nonparametric Post-hoc Test"
        if column_name:
            title += f" for '{column_name}'"
        if progress_text:
            title += f" {progress_text}"
        dialog.setWindowTitle(title)

        info = QLabel("Please select the desired nonparametric post-hoc test:")
        layout.addWidget(info)

        options = [
            ("Dunn Test (all pairs, Holm-Sidak correction)", "dunn"),
            ("Mann-Whitney-U Tests (custom pairs, Sidak correction)", "mw_custom"),
        ]
        radio_buttons = []
        for label, value in options:
            rb = QRadioButton(label)
            layout.addWidget(rb)
            radio_buttons.append((rb, value))
        radio_buttons[0][0].setChecked(True)  # Default: Dunn

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        if dialog.exec_() == QDialog.Accepted:
            for rb, value in radio_buttons:
                if rb.isChecked():
                    return value
        return None
    
    @staticmethod
    def select_custom_pairs_dialog(groups):
        """
        Dialog to select custom group pairs for paired t-tests.
        Returns a list of (group1, group2) tuples.
        """
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QCheckBox, QDialogButtonBox, QWidget, QHBoxLayout, QScrollArea

        UIDialogManager._ensure_qt_application()

        class PairSelectionDialog(QDialog):
            def __init__(self, groups, parent=None):
                super().__init__(parent)
                UIDialogManager._configure_dialog(self, object_name="pairSelectionDialog")
                self.setWindowTitle("Select Group Pairs for Paired t-tests")
                self.selected_pairs = []
                layout = QVBoxLayout(self)
                label = QLabel("Select the group pairs to compare (paired t-test):")
                layout.addWidget(label)
                
                # Create scroll area for many group pairs
                scroll = QScrollArea(self)
                scroll.setWidgetResizable(True)
                scroll_content = QWidget()
                scroll_layout = QVBoxLayout(scroll_content)
                
                self.checkboxes = []
                for i, g1 in enumerate(groups):
                    for g2 in groups[i+1:]:
                        pair_str = f"{g1} vs {g2}"
                        cb = QCheckBox(pair_str)
                        scroll_layout.addWidget(cb)
                        self.checkboxes.append((cb, (g1, g2)))
                
                scroll_content.setLayout(scroll_layout)
                scroll.setWidget(scroll_content)
                
                # Limit maximum height to prevent dialog from becoming too large
                # Calculate dynamic height: max 350px or 50% of screen height, whichever is smaller
                from PyQt5.QtWidgets import QApplication
                if QApplication.instance():
                    screen = QApplication.instance().primaryScreen()
                    if screen:
                        screen_height = screen.geometry().height()
                        max_height = min(350, int(screen_height * 0.5))
                        scroll.setMaximumHeight(max_height)
                    else:
                        scroll.setMaximumHeight(350)  # Fallback
                else:
                    scroll.setMaximumHeight(350)  # Fallback
                    
                layout.addWidget(scroll)
                
                buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
                buttons.accepted.connect(self.accept)
                buttons.rejected.connect(self.reject)
                layout.addWidget(buttons)

            def accept(self):
                self.selected_pairs = [pair for cb, pair in self.checkboxes if cb.isChecked()]
                super().accept()

        dialog = PairSelectionDialog(groups)
        if dialog.exec_() == QDialog.Accepted:
            return dialog.selected_pairs
        return []

    @staticmethod
    def select_control_group_dialog(groups):
        """Opens a dialog window to select the control group"""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QRadioButton, QDialogButtonBox

        UIDialogManager._ensure_qt_application()

        class ControlGroupDialog(QDialog):
            def __init__(self, available_groups, parent=None):
                super().__init__(parent)
                UIDialogManager._configure_dialog(self, object_name="controlGroupDialog")
                self.setWindowTitle("Select Control Group")
                self.selected_group = None
                self.groups = available_groups

                layout = QVBoxLayout(self)
                label = QLabel("Please select the control group for the Dunnett test:")
                layout.addWidget(label)

                self.group_buttons = []
                for group in self.groups:
                    rb = QRadioButton(str(group))
                    self.group_buttons.append(rb)
                    layout.addWidget(rb)

                # Select first group by default
                if self.group_buttons:
                    self.group_buttons[0].setChecked(True)

                button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
                button_box.accepted.connect(self.accept)
                button_box.rejected.connect(self.reject)
                layout.addWidget(button_box)

            def accept(self):
                for i, button in enumerate(self.group_buttons):
                    if button.isChecked():
                        self.selected_group = self.groups[i]
                        break
                super().accept()

        # Always create a new dialog
        dialog = ControlGroupDialog(groups)
        if dialog.exec_() == QDialog.Accepted:
            return dialog.selected_group
        return groups[0]  # Default: first group
    
    @staticmethod
    def select_transformation_dialog(parent=None, progress_text=None, column_name=None, force_show=False):
        # NO CACHING - Each analysis starts fresh and shows the dialog every time
        # This ensures consistent behavior between normal tests and advanced tests
        UIDialogManager._ensure_qt_application()
        
        dialog = QDialog(parent)
        UIDialogManager._configure_dialog(dialog, object_name="transformationSelectionDialog")
        layout = QVBoxLayout(dialog)

        # Set window title
        title = "Select Transformation"
        if column_name:
            title += f" for '{column_name}'"
        if progress_text:
            title += f" {progress_text}"
        dialog.setWindowTitle(title)

        # Info text - CONSISTENT WORDING
        info = QLabel("Please select the desired transformation:")
        layout.addWidget(info)

        # RadioButtons for transformations
        options = [
            ("Log10 transformation (for positive, right-skewed data)", "log10"),
            ("Box-Cox transformation (automatic lambda optimization)", "boxcox"),
            ("Arcsin square root transformation (for percentages/proportions)", "arcsin_sqrt"),
        ]
        radio_buttons = []
        for label, value in options:
            rb = QRadioButton(label)
            layout.addWidget(rb)
            radio_buttons.append((rb, value))
        radio_buttons[0][0].setChecked(True)  # Default: Log10

        # OK/Cancel buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            for rb, value in radio_buttons:
                if rb.isChecked():
                    return value
        
        # If canceled, return None
        return None
       

from analysis_core import DatasetSelector, AnalysisManager, get_output_path
from outlier_core import OUTLIER_IMPORTS_AVAILABLE, OutlierDetector


# Note: Classes are imported lazily to avoid circular imports.
# Use get_data_visualizer(), get_statistical_tester(), get_results_exporter() functions instead.
ResultsExporter = get_results_exporter()
DataVisualizer = get_data_visualizer()
