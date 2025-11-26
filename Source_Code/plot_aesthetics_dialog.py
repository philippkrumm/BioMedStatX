"""
Plot Aesthetics Dialog with tabbed interface and live preview.
Enables comprehensive customization of plot appearance.
"""
from PyQt5.QtWidgets import QDesktopWidget
import sys
import os
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, 
                             QWidget, QLabel, QComboBox, QSpinBox, QDoubleSpinBox, 
                             QCheckBox, QPushButton, QColorDialog, QSlider, 
                             QGroupBox, QGridLayout, QDialogButtonBox, QLineEdit,
                             QApplication, QMainWindow, QSplitter, QFrame)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPalette

# Import des Preview Widgets
try:
    from plot_preview import PlotPreviewWidget
except ImportError:
    print("Warning: Could not import PlotPreviewWidget")
    PlotPreviewWidget = None

class ColorButton(QPushButton):
    """Custom button for color selection"""
    colorChanged = pyqtSignal(str)
    
    def __init__(self, color="#3357FF", parent=None):
        super().__init__(parent)
        self.current_color = color
        self.setFixedSize(40, 25)
        self.update_color()
        self.clicked.connect(self.open_color_dialog)
    
    def update_color(self):
        self.setStyleSheet(f"QPushButton {{ background-color: {self.current_color}; border: 1px solid gray; }}")
    
    def open_color_dialog(self):
        color = QColorDialog.getColor(QColor(self.current_color), self)
        if color.isValid():
            self.current_color = color.name()
            self.update_color()
            self.colorChanged.emit(self.current_color)
    
    def get_color(self):
        return self.current_color
    
    def set_color(self, color):
        self.current_color = color
        self.update_color()


class SizeTab(QWidget):
    """Tab for size settings"""
    settingsChanged = pyqtSignal()
    
    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Figure Size Group
        fig_group = QGroupBox("Figure Size")
        fig_layout = QGridLayout(fig_group)
        
        # Width
        fig_layout.addWidget(QLabel("Width (inches):"), 0, 0)
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(1, 20)
        self.width_spin.setValue(self.config.get('width', 8))
        self.width_spin.setSingleStep(0.5)
        self.width_spin.valueChanged.connect(self.settingsChanged)
        fig_layout.addWidget(self.width_spin, 0, 1)
        
        # Height
        fig_layout.addWidget(QLabel("Height (inches):"), 1, 0)
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(1, 20)
        self.height_spin.setValue(self.config.get('height', 6))
        self.height_spin.setSingleStep(0.5)
        self.height_spin.valueChanged.connect(self.settingsChanged)
        fig_layout.addWidget(self.height_spin, 1, 1)
        
        # DPI
        fig_layout.addWidget(QLabel("DPI:"), 2, 0)
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 600)
        self.dpi_spin.setValue(self.config.get('dpi', 300))
        self.dpi_spin.valueChanged.connect(self.settingsChanged)
        fig_layout.addWidget(self.dpi_spin, 2, 1)
        
        layout.addWidget(fig_group)
        layout.addStretch()
    
    def get_settings(self):
        return {
            'width': self.width_spin.value(),
            'height': self.height_spin.value(),
            'dpi': self.dpi_spin.value()
        }


class TypographyTab(QWidget):
    """Tab for font settings"""
    settingsChanged = pyqtSignal()
    
    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        # Track if user has explicitly changed font sizes
        self.font_sizes_modified = {
            'title': False,
            'axis': False,
            'ticks': False
        }
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Font Family Group
        font_family_group = QGroupBox("Font Family")
        font_family_layout = QGridLayout(font_family_group)
        
        font_family_layout.addWidget(QLabel("Font Family:"), 0, 0)
        self.font_family_combo = QComboBox()
        
        # Use FontManager to get available system fonts
        try:
            # Import the FontManager from datavisualizer
            import sys
            import os
            # Add src directory to path if not already there
            src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
            if src_path not in sys.path:
                sys.path.insert(0, src_path)
            
            from datavisualizer import FontManager
            font_families = FontManager.get_available_fonts()
            print(f"Loaded {len(font_families)} system fonts for UI")
        except Exception as e:
            print(f"Warning: Could not load system fonts: {e}")
            # Fallback to common fonts
            font_families = [
                'Arial', 'Times New Roman', 'Calibri',
                'Segoe UI', 'Georgia', 'Helvetica',
                'Trebuchet MS', 'Impact', 'DejaVu Sans'
            ]
        
        self.font_family_combo.addItems(font_families)
        current_font = self.config.get('font_family', 'Arial')
        
        # Validate and set current font
        try:
            if current_font in font_families:
                self.font_family_combo.setCurrentText(current_font)
            else:
                # Find closest match or use first available
                self.font_family_combo.setCurrentText(font_families[0])
                print(f"Font '{current_font}' not available, using '{font_families[0]}'")
        except:
            self.font_family_combo.setCurrentText(font_families[0])
            
        # Improved signal connection for immediate updates
        self.font_family_combo.currentTextChanged.connect(self.on_font_changed)
        font_family_layout.addWidget(self.font_family_combo, 0, 1)
        
        layout.addWidget(font_family_group)
        
        # Font Sizes Group
        font_group = QGroupBox("Font Sizes")
        font_layout = QGridLayout(font_group)
        
        # Title Font Size
        font_layout.addWidget(QLabel("Title:"), 0, 0)
        self.title_size_spin = QSpinBox()
        self.title_size_spin.setRange(8, 48)
        self.title_size_spin.setValue(self.config.get('fontsize_title', 14))
        self.title_size_spin.valueChanged.connect(self.on_title_size_changed)
        font_layout.addWidget(self.title_size_spin, 0, 1)
        
        # Axis Label Font Size
        font_layout.addWidget(QLabel("Axis Labels:"), 1, 0)
        self.axis_size_spin = QSpinBox()
        self.axis_size_spin.setRange(8, 24)
        self.axis_size_spin.setValue(self.config.get('fontsize_axis', 12))
        self.axis_size_spin.valueChanged.connect(self.on_axis_size_changed)
        font_layout.addWidget(self.axis_size_spin, 1, 1)
        
        # Tick Label Font Size
        font_layout.addWidget(QLabel("Tick Labels:"), 2, 0)
        self.ticks_size_spin = QSpinBox()
        self.ticks_size_spin.setRange(6, 20)
        self.ticks_size_spin.setValue(self.config.get('fontsize_ticks', 10))
        self.ticks_size_spin.valueChanged.connect(self.on_ticks_size_changed)
        font_layout.addWidget(self.ticks_size_spin, 2, 1)
        
        layout.addWidget(font_group)
        
        # Labels Group
        labels_group = QGroupBox("Labels")
        labels_layout = QGridLayout(labels_group)
        
        # Title
        labels_layout.addWidget(QLabel("Title:"), 0, 0)
        self.title_edit = QLineEdit(self.config.get('title', ''))
        self.title_edit.textChanged.connect(self.settingsChanged)
        labels_layout.addWidget(self.title_edit, 0, 1)
        
        # X Label
        labels_layout.addWidget(QLabel("X Label:"), 1, 0)
        self.x_label_edit = QLineEdit(self.config.get('x_label', ''))
        self.x_label_edit.textChanged.connect(self.settingsChanged)
        labels_layout.addWidget(self.x_label_edit, 1, 1)
        
        # Y Label
        labels_layout.addWidget(QLabel("Y Label:"), 2, 0)
        self.y_label_edit = QLineEdit(self.config.get('y_label', ''))
        self.y_label_edit.textChanged.connect(self.settingsChanged)
        labels_layout.addWidget(self.y_label_edit, 2, 1)
        
        layout.addWidget(labels_group)
        layout.addStretch()
    
    def on_font_changed(self):
        """Special handling for font changes with immediate update"""
        # Force immediate update for fonts
        self.settingsChanged.emit()
    
    def on_title_size_changed(self):
        """Handler for title size changes"""
        self.font_sizes_modified['title'] = True
        self.settingsChanged.emit()
    
    def on_axis_size_changed(self):
        """Handler for axis size changes"""
        self.font_sizes_modified['axis'] = True
        self.settingsChanged.emit()
    
    def on_ticks_size_changed(self):
        """Handler for ticks size changes"""
        self.font_sizes_modified['ticks'] = True
        self.settingsChanged.emit()
    
    def get_settings(self):
        settings = {
            'font_family': self.font_family_combo.currentText(),
            'title': self.title_edit.text(),
            'x_label': self.x_label_edit.text(),
            'y_label': self.y_label_edit.text(),
            'show_title': bool(self.title_edit.text().strip())
        }
        
        # Only include font sizes if they were explicitly modified by the user
        if self.font_sizes_modified['title']:
            settings['fontsize_title'] = self.title_size_spin.value()
        if self.font_sizes_modified['axis']:
            settings['fontsize_axis'] = self.axis_size_spin.value()
        if self.font_sizes_modified['ticks']:
            settings['fontsize_ticks'] = self.ticks_size_spin.value()
        
        return settings


class ColorsTab(QWidget):
    """Tab für Farbeinstellungen"""
    settingsChanged = pyqtSignal()
    
    def __init__(self, groups=None, config=None, context="user_plot"):
        super().__init__()
        self.groups = groups or []
        self.config = config or {}
        self.context = context  # "user_plot" or "analysis_only"
        self.color_buttons = {}
        self.hatch_combos = {}
        self.dialog_ref = None  # Reference to main dialog will be set later
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Seaborn Style and Palette Selection
        seaborn_group = QGroupBox("Color Style & Palette")
        seaborn_layout = QGridLayout(seaborn_group)
        
        # Style context selector
        seaborn_layout.addWidget(QLabel("Style Context:"), 0, 0)
        self.style_context_combo = QComboBox()
        self.style_context_combo.addItems(['notebook', 'paper', 'talk', 'poster'])
        self.style_context_combo.setCurrentText(self.config.get('seaborn_context', 'notebook'))
        self.style_context_combo.currentTextChanged.connect(self.on_seaborn_settings_changed)
        seaborn_layout.addWidget(self.style_context_combo, 0, 1)
        
        # Color palette selector
        seaborn_layout.addWidget(QLabel("Color Palette:"), 1, 0)
        self.palette_combo = QComboBox()
        # Professional palettes - excluding rainbow/childish ones
        professional_palettes = [
            'deep', 'muted', 'dark', 'colorblind',
            'viridis', 'plasma', 'inferno', 'magma', 'mako',
            'Greys', 'Paired', 'tab10'
        ]
        self.palette_combo.addItems(professional_palettes)
        self.palette_combo.setCurrentText(self.config.get('seaborn_palette', 'Greys'))
        self.palette_combo.currentTextChanged.connect(self.on_seaborn_settings_changed)
        seaborn_layout.addWidget(self.palette_combo, 1, 1)
        
        # Enable/disable Seaborn styling
        self.use_seaborn_checkbox = QCheckBox("Use Seaborn Styling")
        self.use_seaborn_checkbox.setChecked(self.config.get('use_seaborn_styling', True))
        self.use_seaborn_checkbox.stateChanged.connect(self.on_seaborn_settings_changed)
        seaborn_layout.addWidget(self.use_seaborn_checkbox, 2, 0, 1, 2)
        
        layout.addWidget(seaborn_group)
        
        # Individual Colors Group
        colors_group = QGroupBox("Individual Colors")
        self.colors_layout = QGridLayout(colors_group)
        self.update_color_buttons()
        
        layout.addWidget(colors_group)
        
        # Hatches Group
        hatches_group = QGroupBox("Hatches (Patterns)")
        self.hatches_layout = QGridLayout(hatches_group)
        self.update_hatch_selectors()
        
        layout.addWidget(hatches_group)
        layout.addStretch()
    
    def update_hatch_selectors(self):
        """Update hatch selector dropdowns for each group"""
        # Clear existing selectors
        for i in reversed(range(self.hatches_layout.count())):
            self.hatches_layout.itemAt(i).widget().setParent(None)
        
        # Available hatch patterns
        hatch_patterns = [
            ('None', ''),
            ('Diagonal /', '/'),
            ('Diagonal \\', '\\'),
            ('Vertical |', '|'),
            ('Horizontal -', '-'),
            ('Plus +', '+'),
            ('Cross x', 'x'),
            ('Dots .', '.'),
            ('Circles o', 'o'),
            ('Stars *', '*'),
            ('Dense ///', '///'),
            ('Dense \\\\\\', '\\\\\\'),
            ('Dense |||', '|||'),
            ('Dense ---', '---'),
            ('Dense +++', '+++')
        ]
        
        self.hatch_combos = {}
        
        for i, group in enumerate(self.groups):
            label = QLabel(f"{group}:")
            self.hatches_layout.addWidget(label, i, 0)
            
            hatch_combo = QComboBox()
            hatch_combo.addItems([pattern[0] for pattern in hatch_patterns])
            
            # Set current hatch if available
            current_hatch = self.config.get('hatches', {}).get(group, '')
            for j, (name, pattern) in enumerate(hatch_patterns):
                if pattern == current_hatch:
                    hatch_combo.setCurrentIndex(j)
                    break
            
            hatch_combo.currentTextChanged.connect(self.settingsChanged)
            self.hatch_combos[group] = hatch_combo
            self.hatches_layout.addWidget(hatch_combo, i, 1)
    
    def update_color_buttons(self):
        # Clear existing buttons
        for i in reversed(range(self.colors_layout.count())):
            self.colors_layout.itemAt(i).widget().setParent(None)
        self.color_buttons.clear()
        
        # Add buttons for each group
        for i, group in enumerate(self.groups):
            label = QLabel(f"{group}:")
            self.colors_layout.addWidget(label, i, 0)
            
            # Get color from config or use context-appropriate defaults
            if self.config.get('colors') and group in self.config['colors']:
                # Use existing colors from configuration (user's plot colors)
                color = self.config['colors'][group]
            else:
                # Choose defaults based on context
                if self.context == "analysis_only":
                    # Use grayscale for analysis-only visualization
                    default_colors = [
                        '#2C2C2C', '#4A4A4A', '#686868', '#868686', '#A4A4A4', '#C2C2C2',
                        '#3C3C3C', '#5A5A5A', '#787878', '#969696'
                    ]
                    color = default_colors[i % len(default_colors)]
                else:
                    # Use colorful defaults for user plots (same as DEFAULT_COLORS in statistical_analyzer.py)
                    # Use system default colors for user plots
                    default_colors = ['#FF69B4', '#32CD32', '#FFD700', '#00BFFF', '#DA70D6', '#D8BFD8']  # Pink, Green, Gold, etc.
                    color = default_colors[i % len(default_colors)]
            
            color_btn = ColorButton(color)
            color_btn.colorChanged.connect(self.settingsChanged)
            self.color_buttons[group] = color_btn
            self.colors_layout.addWidget(color_btn, i, 1)
    
    def on_seaborn_settings_changed(self):
        """Handle changes to Seaborn style context or palette"""
        # Reset font size modifications when style context changes
        # This allows seaborn context to take full effect
        if self.dialog_ref and hasattr(self.dialog_ref, 'typography_tab'):
            self.dialog_ref.typography_tab.font_sizes_modified = {
                'title': False,
                'axis': False,
                'ticks': False
            }
        
        if self.use_seaborn_checkbox.isChecked():
            # Apply seaborn palette colors to color buttons
            try:
                import seaborn as sns
                palette_name = self.palette_combo.currentText()
                
                # Get colors from the selected palette
                if palette_name in ['viridis', 'plasma', 'inferno', 'magma', 'mako', 'Greys']:
                    # For continuous palettes, sample discrete colors
                    palette_colors = sns.color_palette(palette_name, n_colors=len(self.groups))
                else:
                    # For discrete palettes
                    palette_colors = sns.color_palette(palette_name, n_colors=len(self.groups))
                
                # Convert to hex colors and update buttons
                for i, group in enumerate(self.groups):
                    if i < len(palette_colors):
                        # Convert RGB tuple to hex
                        rgb = palette_colors[i]
                        hex_color = "#{:02x}{:02x}{:02x}".format(
                            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
                        )
                        self.color_buttons[group].set_color(hex_color)
            except ImportError:
                pass  # Seaborn not available
        
        self.settingsChanged.emit()
    
    def get_settings(self):
        colors = {}
        for group, button in self.color_buttons.items():
            colors[group] = button.get_color()
        
        # Get hatch patterns
        hatches = {}
        hatch_patterns = {
            'None': '',
            'Diagonal /': '/',
            'Diagonal \\': '\\',
            'Vertical |': '|',
            'Horizontal -': '-',
            'Plus +': '+',
            'Cross x': 'x',
            'Dots .': '.',
            'Circles o': 'o',
            'Stars *': '*',
            'Dense ///': '///',
            'Dense \\\\\\': '\\\\\\',
            'Dense |||': '|||',
            'Dense ---': '---',
            'Dense +++': '+++'
        }
        
        for group, combo in getattr(self, 'hatch_combos', {}).items():
            pattern_name = combo.currentText()
            hatches[group] = hatch_patterns.get(pattern_name, '')
        
        return {
            'colors': colors,
            'hatches': hatches,
            'seaborn_context': self.style_context_combo.currentText(),
            'seaborn_palette': self.palette_combo.currentText(),
            'use_seaborn_styling': self.use_seaborn_checkbox.isChecked()
        }
    
    def set_groups(self, groups):
        self.groups = groups
        self.update_color_buttons()
        self.update_hatch_selectors()


class StyleTab(QWidget):
    """Tab für Style-Einstellungen"""
    settingsChanged = pyqtSignal()
    
    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Plot Type Group
        type_group = QGroupBox("Plot Type")
        type_layout = QHBoxLayout(type_group)
        
        type_layout.addWidget(QLabel("Type:"))
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(['Bar', 'Box', 'Violin', 'Raincloud'])
        self.plot_type_combo.setCurrentText(self.config.get('plot_type', 'Bar'))
        self.plot_type_combo.currentTextChanged.connect(self.settingsChanged)
        type_layout.addWidget(self.plot_type_combo)
        type_layout.addStretch()
        
        layout.addWidget(type_group)
        
        # Appearance Group
        appearance_group = QGroupBox("Appearance")
        appearance_layout = QGridLayout(appearance_group)
        
        # Alpha
        appearance_layout.addWidget(QLabel("Transparency:"), 0, 0)
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.1, 1.0)
        self.alpha_spin.setSingleStep(0.1)
        self.alpha_spin.setValue(self.config.get('alpha', 0.8))
        self.alpha_spin.valueChanged.connect(self.settingsChanged)
        appearance_layout.addWidget(self.alpha_spin, 0, 1)
        
        # Edge Width
        appearance_layout.addWidget(QLabel("Edge Width:"), 1, 0)
        self.edge_width_spin = QDoubleSpinBox()
        self.edge_width_spin.setRange(0.0, 5.0)
        self.edge_width_spin.setSingleStep(0.1)
        self.edge_width_spin.setValue(self.config.get('bar_linewidth', 0.5))
        self.edge_width_spin.valueChanged.connect(self.settingsChanged)
        appearance_layout.addWidget(self.edge_width_spin, 1, 1)
        
        # Grid
        self.grid_check = QCheckBox("Show Grid")
        self.grid_check.setChecked(self.config.get('grid', False))
        self.grid_check.toggled.connect(self.settingsChanged)
        appearance_layout.addWidget(self.grid_check, 2, 0)
        
        # Minor Ticks
        self.minor_ticks_check = QCheckBox("Show Minor Ticks")
        self.minor_ticks_check.setChecked(self.config.get('minor_ticks', False))
        self.minor_ticks_check.toggled.connect(self.settingsChanged)
        appearance_layout.addWidget(self.minor_ticks_check, 2, 1)
        
        # Despine
        self.despine_check = QCheckBox("Remove Spines")
        self.despine_check.setChecked(self.config.get('despine', True))
        self.despine_check.toggled.connect(self.settingsChanged)
        appearance_layout.addWidget(self.despine_check, 3, 0)
        
        # Axis Thickness
        appearance_layout.addWidget(QLabel("Axis Thickness:"), 4, 0)
        self.axis_thickness_spin = QDoubleSpinBox()
        self.axis_thickness_spin.setRange(0.1, 3.0)
        self.axis_thickness_spin.setSingleStep(0.1)
        self.axis_thickness_spin.setValue(self.config.get('axis_thickness', 0.7))
        self.axis_thickness_spin.valueChanged.connect(self.settingsChanged)
        appearance_layout.addWidget(self.axis_thickness_spin, 4, 1)
        
        layout.addWidget(appearance_group)
        
        # Data Points Group
        points_group = QGroupBox("Data Points")
        points_layout = QGridLayout(points_group)
        
        # Show Points
        self.points_check = QCheckBox("Show Individual Points")
        self.points_check.setChecked(self.config.get('show_points', True))
        self.points_check.toggled.connect(self.settingsChanged)
        points_layout.addWidget(self.points_check, 0, 0, 1, 2)
        
        # Point Size
        points_layout.addWidget(QLabel("Point Size:"), 1, 0)
        self.point_size_spin = QSpinBox()
        self.point_size_spin.setRange(10, 200)
        self.point_size_spin.setValue(self.config.get('point_size', 80))
        self.point_size_spin.valueChanged.connect(self.settingsChanged)
        points_layout.addWidget(self.point_size_spin, 1, 1)
        
        # Jitter Strength
        points_layout.addWidget(QLabel("Jitter Strength:"), 2, 0)
        self.jitter_spin = QDoubleSpinBox()
        self.jitter_spin.setRange(0.0, 1.0)
        self.jitter_spin.setSingleStep(0.1)
        self.jitter_spin.setValue(self.config.get('jitter_strength', 0.3))
        self.jitter_spin.valueChanged.connect(self.settingsChanged)
        points_layout.addWidget(self.jitter_spin, 2, 1)
        
        layout.addWidget(points_group)
        
        # Legend Group
        legend_group = QGroupBox("Legend")
        legend_layout = QGridLayout(legend_group)
        
        # Show Legend
        self.legend_check = QCheckBox("Show Legend")
        self.legend_check.setChecked(self.config.get('show_legend', True))
        self.legend_check.toggled.connect(self.settingsChanged)
        legend_layout.addWidget(self.legend_check, 0, 0, 1, 2)
        
        layout.addWidget(legend_group)
        layout.addStretch()
    
    def get_settings(self):
        return {
            'plot_type': self.plot_type_combo.currentText(),
            'alpha': self.alpha_spin.value(),
            'bar_linewidth': self.edge_width_spin.value(),
            'grid': self.grid_check.isChecked(),
            'grid_style': 'major' if self.grid_check.isChecked() else 'none',  # Für DataVisualizer
            'minor_ticks': self.minor_ticks_check.isChecked(),
            'despine': self.despine_check.isChecked(),
            'axis_thickness': self.axis_thickness_spin.value(),
            'show_points': self.points_check.isChecked(),
            'point_size': self.point_size_spin.value(),
            'jitter_strength': self.jitter_spin.value(),
            'show_legend': self.legend_check.isChecked()
        }


class RaincloudTab(QWidget):
    """Tab für erweiterte Raincloud-Einstellungen"""
    settingsChanged = pyqtSignal()
    
    def __init__(self, groups=None, config=None):
        super().__init__()
        self.groups = groups or []
        self.config = config or {}
        self.violin_color_buttons = {}
        self.box_color_buttons = {}
        self.point_color_buttons = {}
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Violin Colors Group
        violin_group = QGroupBox("Violin Colors")
        self.violin_layout = QGridLayout(violin_group)
        
        # Box Colors Group
        box_group = QGroupBox("Box Colors")
        self.box_layout = QGridLayout(box_group)
        
        # Point Colors Group
        point_group = QGroupBox("Point Colors")
        self.point_layout = QGridLayout(point_group)
        
        # Update color buttons
        self.update_color_buttons()
        
        layout.addWidget(violin_group)
        layout.addWidget(box_group)
        layout.addWidget(point_group)
        
        # Spacing and Layout Group
        spacing_group = QGroupBox("Spacing and Layout")
        spacing_layout = QGridLayout(spacing_group)
        
        # Group Spacing
        spacing_layout.addWidget(QLabel("Group Spacing:"), 0, 0)
        self.group_spacing_spin = QDoubleSpinBox()
        self.group_spacing_spin.setRange(0.3, 2.0)
        self.group_spacing_spin.setSingleStep(0.1)
        self.group_spacing_spin.setValue(self.config.get('group_spacing', 0.90))
        self.group_spacing_spin.valueChanged.connect(self.settingsChanged)
        spacing_layout.addWidget(self.group_spacing_spin, 0, 1)
        
        # Point Vertical Offset
        spacing_layout.addWidget(QLabel("Point Vertical Offset:"), 1, 0)
        self.point_offset_spin = QDoubleSpinBox()
        self.point_offset_spin.setRange(0.1, 0.5)
        self.point_offset_spin.setSingleStep(0.05)
        self.point_offset_spin.setValue(self.config.get('point_offset', 0.2))
        self.point_offset_spin.valueChanged.connect(self.settingsChanged)
        spacing_layout.addWidget(self.point_offset_spin, 1, 1)
        
        # Point Horizontal Jitter
        spacing_layout.addWidget(QLabel("Point Horizontal Jitter:"), 2, 0)
        self.point_jitter_spin = QDoubleSpinBox()
        self.point_jitter_spin.setRange(0.01, 0.1)
        self.point_jitter_spin.setSingleStep(0.01)
        self.point_jitter_spin.setValue(self.config.get('point_jitter', 0.05))
        self.point_jitter_spin.valueChanged.connect(self.settingsChanged)
        spacing_layout.addWidget(self.point_jitter_spin, 2, 1)
        
        # Violin Width
        spacing_layout.addWidget(QLabel("Violin Width:"), 3, 0)
        self.violin_width_spin = QDoubleSpinBox()
        self.violin_width_spin.setRange(0.3, 1.5)
        self.violin_width_spin.setSingleStep(0.1)
        self.violin_width_spin.setValue(self.config.get('violin_width', 0.8))
        self.violin_width_spin.valueChanged.connect(self.settingsChanged)
        spacing_layout.addWidget(self.violin_width_spin, 3, 1)
        
        # Box Width
        spacing_layout.addWidget(QLabel("Box Width:"), 4, 0)
        self.box_width_spin = QDoubleSpinBox()
        self.box_width_spin.setRange(0.1, 0.8)
        self.box_width_spin.setSingleStep(0.1)
        self.box_width_spin.setValue(self.config.get('box_width', 0.2))
        self.box_width_spin.valueChanged.connect(self.settingsChanged)
        spacing_layout.addWidget(self.box_width_spin, 4, 1)
        
        layout.addWidget(spacing_group)
        layout.addStretch()
    
    def update_color_buttons(self):
        """Update color buttons for all groups"""
        # Clear existing buttons
        for layout in [self.violin_layout, self.box_layout, self.point_layout]:
            for i in reversed(range(layout.count())):
                layout.itemAt(i).widget().setParent(None)
        
        self.violin_color_buttons.clear()
        self.box_color_buttons.clear()
        self.point_color_buttons.clear()
        
        # Default colors for different components - erweitert für bis zu 10 Gruppen in Grautönen
        default_violin_colors = [
            "lightgray", "silver", "darkgray", "gray", "dimgray", "gainsboro",
            "whitesmoke", "lightslategray", "slategray", "darkslategray"
        ]
        default_box_colors = [
            "dimgray", "gainsboro", "darkgray", "gray", "silver", "lightgray",
            "darkslategray", "slategray", "lightslategray", "whitesmoke"
        ]
        default_point_colors = [
            "black", "dimgray", "gray", "darkgray", "silver", "lightgray",
            "slategray", "lightslategray", "gainsboro", "darkslategray"
        ]
        
        for i, group in enumerate(self.groups):
            # Violin colors
            violin_label = QLabel(f"{group}:")
            self.violin_layout.addWidget(violin_label, i, 0)
            
            violin_color = self.config.get('violin_colors', {}).get(group, default_violin_colors[i % len(default_violin_colors)])
            violin_btn = ColorButton(violin_color)
            violin_btn.colorChanged.connect(self.settingsChanged)
            self.violin_color_buttons[group] = violin_btn
            self.violin_layout.addWidget(violin_btn, i, 1)
            
            # Box colors
            box_label = QLabel(f"{group}:")
            self.box_layout.addWidget(box_label, i, 0)
            
            box_color = self.config.get('box_colors', {}).get(group, default_box_colors[i % len(default_box_colors)])
            box_btn = ColorButton(box_color)
            box_btn.colorChanged.connect(self.settingsChanged)
            self.box_color_buttons[group] = box_btn
            self.box_layout.addWidget(box_btn, i, 1)
            
            # Point colors
            point_label = QLabel(f"{group}:")
            self.point_layout.addWidget(point_label, i, 0)
            
            point_color = self.config.get('point_colors', {}).get(group, default_point_colors[i % len(default_point_colors)])
            point_btn = ColorButton(point_color)
            point_btn.colorChanged.connect(self.settingsChanged)
            self.point_color_buttons[group] = point_btn
            self.point_layout.addWidget(point_btn, i, 1)
    
    def get_settings(self):
        """Get all raincloud-specific settings"""
        violin_colors = {}
        box_colors = {}
        point_colors = {}
        
        for group in self.groups:
            if group in self.violin_color_buttons:
                violin_colors[group] = self.violin_color_buttons[group].get_color()
            if group in self.box_color_buttons:
                box_colors[group] = self.box_color_buttons[group].get_color()
            if group in self.point_color_buttons:
                point_colors[group] = self.point_color_buttons[group].get_color()
        
        return {
            'violin_colors': violin_colors,
            'box_colors': box_colors,
            'point_colors': point_colors,
            'group_spacing': self.group_spacing_spin.value(),
            'point_offset': self.point_offset_spin.value(),
            'point_jitter': self.point_jitter_spin.value(),
            'violin_width': self.violin_width_spin.value(),
            'box_width': self.box_width_spin.value()
        }
    
    def set_groups(self, groups):
        """Update groups and rebuild color buttons"""
        self.groups = groups
        self.update_color_buttons()


class ErrorBarsTab(QWidget):
    """Tab für Error Bar Einstellungen"""
    settingsChanged = pyqtSignal()
    
    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Error Bars Group
        error_group = QGroupBox("Error Bars")
        error_layout = QGridLayout(error_group)
        
        # Show Error Bars
        self.show_error_check = QCheckBox("Show Error Bars")
        self.show_error_check.setChecked(self.config.get('show_error_bars', True))
        self.show_error_check.toggled.connect(self.settingsChanged)
        error_layout.addWidget(self.show_error_check, 0, 0, 1, 2)
        
        # Error Type
        error_layout.addWidget(QLabel("Error Type:"), 1, 0)
        self.error_type_combo = QComboBox()
        self.error_type_combo.addItems(['sd', 'se', 'ci'])
        self.error_type_combo.setCurrentText(self.config.get('error_type', 'sd'))
        self.error_type_combo.currentTextChanged.connect(self.settingsChanged)
        error_layout.addWidget(self.error_type_combo, 1, 1)
        
        # Error Style
        error_layout.addWidget(QLabel("Error Style:"), 2, 0)
        self.error_style_combo = QComboBox()
        self.error_style_combo.addItems(['caps', 'line'])
        self.error_style_combo.setCurrentText(self.config.get('error_style', 'caps'))
        self.error_style_combo.currentTextChanged.connect(self.settingsChanged)
        error_layout.addWidget(self.error_style_combo, 2, 1)
        
        # Cap Size
        error_layout.addWidget(QLabel("Cap Size:"), 3, 0)
        self.capsize_spin = QDoubleSpinBox()
        self.capsize_spin.setRange(0.0, 1.0)
        self.capsize_spin.setSingleStep(0.01)
        self.capsize_spin.setValue(self.config.get('capsize', 0.05))
        self.capsize_spin.valueChanged.connect(self.settingsChanged)
        error_layout.addWidget(self.capsize_spin, 3, 1)
        
        layout.addWidget(error_group)
        layout.addStretch()
    
    def get_settings(self):
        return {
            'show_error_bars': self.show_error_check.isChecked(),
            'error_type': self.error_type_combo.currentText(),
            'error_style': self.error_style_combo.currentText(),
            'capsize': self.capsize_spin.value()
        }


class SignificanceTab(QWidget):
    """Tab für Signifikanz-Einstellungen (Buchstaben und Balken)"""
    settingsChanged = pyqtSignal()
    
    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Significance Letters Group
        letters_group = QGroupBox("Significance Letters")
        letters_layout = QGridLayout(letters_group)
        
        # Show Significance Letters
        self.show_letters_check = QCheckBox("Show Significance Letters")
        self.show_letters_check.setChecked(self.config.get('show_significance_letters', True))
        self.show_letters_check.toggled.connect(self.settingsChanged)
        letters_layout.addWidget(self.show_letters_check, 0, 0, 1, 2)
        
        # Letters Font Size
        letters_layout.addWidget(QLabel("Font Size:"), 1, 0)
        self.letters_fontsize_spin = QSpinBox()
        self.letters_fontsize_spin.setRange(6, 24)
        self.letters_fontsize_spin.setValue(self.config.get('significance_font_size', 12))
        self.letters_fontsize_spin.valueChanged.connect(self.settingsChanged)
        letters_layout.addWidget(self.letters_fontsize_spin, 1, 1)
        
        # Letters Height Offset
        letters_layout.addWidget(QLabel("Height Offset:"), 2, 0)
        self.letters_offset_spin = QDoubleSpinBox()
        self.letters_offset_spin.setRange(0.0, 0.5)
        self.letters_offset_spin.setSingleStep(0.01)
        self.letters_offset_spin.setValue(self.config.get('significance_height_offset', 0.05))
        self.letters_offset_spin.valueChanged.connect(self.settingsChanged)
        letters_layout.addWidget(self.letters_offset_spin, 2, 1)
        
        layout.addWidget(letters_group)
        
        # Significance Brackets Group
        brackets_group = QGroupBox("Significance Brackets")
        brackets_layout = QGridLayout(brackets_group)
        
        # Bracket Line Width
        brackets_layout.addWidget(QLabel("Line Width:"), 0, 0)
        self.bracket_linewidth_spin = QDoubleSpinBox()
        self.bracket_linewidth_spin.setRange(0.5, 5.0)
        self.bracket_linewidth_spin.setSingleStep(0.1)
        self.bracket_linewidth_spin.setValue(self.config.get('bracket_line_width', 2.0))
        self.bracket_linewidth_spin.valueChanged.connect(self.settingsChanged)
        brackets_layout.addWidget(self.bracket_linewidth_spin, 0, 1)
        
        # Bracket Font Size
        brackets_layout.addWidget(QLabel("Font Size:"), 1, 0)
        self.bracket_fontsize_spin = QSpinBox()
        self.bracket_fontsize_spin.setRange(8, 30)
        self.bracket_fontsize_spin.setValue(self.config.get('bracket_font_size', 16))
        self.bracket_fontsize_spin.valueChanged.connect(self.settingsChanged)
        brackets_layout.addWidget(self.bracket_fontsize_spin, 1, 1)
        
        # Bracket Vertical Length
        brackets_layout.addWidget(QLabel("Vertical Length:"), 2, 0)
        self.bracket_vertical_spin = QDoubleSpinBox()
        self.bracket_vertical_spin.setRange(0.1, 1.0)
        self.bracket_vertical_spin.setSingleStep(0.05)
        self.bracket_vertical_spin.setValue(self.config.get('bracket_vertical_fraction', 0.25))
        self.bracket_vertical_spin.valueChanged.connect(self.settingsChanged)
        brackets_layout.addWidget(self.bracket_vertical_spin, 2, 1)
        
        # Bracket Spacing
        brackets_layout.addWidget(QLabel("Spacing:"), 3, 0)
        self.bracket_spacing_spin = QDoubleSpinBox()
        self.bracket_spacing_spin.setRange(0.05, 0.5)
        self.bracket_spacing_spin.setSingleStep(0.01)
        self.bracket_spacing_spin.setValue(self.config.get('bracket_spacing', 0.1))
        self.bracket_spacing_spin.valueChanged.connect(self.settingsChanged)
        brackets_layout.addWidget(self.bracket_spacing_spin, 3, 1)
        
        layout.addWidget(brackets_group)
        layout.addStretch()
    
    def get_settings(self):
        return {
            'show_significance_letters': self.show_letters_check.isChecked(),
            'significance_font_size': self.letters_fontsize_spin.value(),
            'significance_height_offset': self.letters_offset_spin.value(),
            # Bracket settings
            'bracket_line_width': self.bracket_linewidth_spin.value(),
            'bracket_font_size': self.bracket_fontsize_spin.value(),
            'bracket_vertical_fraction': self.bracket_vertical_spin.value(),
            'bracket_spacing': self.bracket_spacing_spin.value(),
            'bracket_color': '#000000'  # Always black
        }


class PlotAestheticsDialog(QDialog):
    """
    Hauptdialog für Plot-Einstellungen mit Tab-Interface und Live-Preview
    """
    
    def __init__(self, groups=None, samples=None, config=None, parent=None, context="user_plot"):
        super().__init__(parent)
        self.groups = groups or []
        self.samples = samples or {}
        self.config = config or {}
        self.context = context  # "user_plot" or "analysis_only"
        
        self.setWindowTitle("Plot Appearance Settings")
        self.setModal(True)
        screen = QDesktopWidget().screenGeometry()
        
        # Adaptive sizing for different display types and resolutions
        screen_width = screen.width()
        screen_height = screen.height()
        
        # High-resolution displays (Retina, 4K, etc.) - like MacBook Air 2880x1864
        if screen_width >= 2560:  # High-res displays
            width = min(1400, int(screen_width * 0.50))   # 50% of screen, max 1400px
            height = min(550, int(screen_height * 0.30))  # 30% of screen, max 550px (much shorter!)
        # Medium resolution displays
        elif screen_width >= 1920:  # Full HD and similar
            width = min(1200, int(screen_width * 0.60))   # 60% of screen, max 1200px
            height = min(500, int(screen_height * 0.45))  # 45% of screen, max 500px
        # Standard/smaller displays
        else:  # < 1920px width
            width = min(1000, int(screen_width * 0.65))   # 65% of screen, max 1000px
            height = min(450, int(screen_height * 0.50))  # 50% of screen, max 450px
            
        self.resize(width, height)
        self.move(
            (screen.width() - width) // 2,
            (screen.height() - height) // 2
        )
        
        self.init_ui()
        self.connect_signals()
        
        # Initial update für Raincloud Tab Sichtbarkeit
        self.update_raincloud_tab_visibility()
        
        # Initiale Preview
        if self.groups and self.samples:
            self.update_preview()
    
    def init_ui(self):
        main_layout = QHBoxLayout(self)
        
        # Splitter für Tabs und Preview
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Linke Seite: Tab Widget für Einstellungen
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Tab Widget
        self.tab_widget = QTabWidget()
        
        # Tabs erstellen
        self.size_tab = SizeTab(self.config)
        self.typography_tab = TypographyTab(self.config)
        self.colors_tab = ColorsTab(self.groups, self.config, self.context)
        self.style_tab = StyleTab(self.config)
        self.raincloud_tab = RaincloudTab(self.groups, self.config)
        self.error_tab = ErrorBarsTab(self.config)
        self.significance_tab = SignificanceTab(self.config)
        
        # Set dialog reference for cross-tab communication
        self.colors_tab.dialog_ref = self
        
        # Tabs hinzufügen
        self.tab_widget.addTab(self.size_tab, "Size")
        self.tab_widget.addTab(self.typography_tab, "Typography")
        self.tab_widget.addTab(self.colors_tab, "Colors")
        self.tab_widget.addTab(self.style_tab, "Style")
        # Raincloud Tab wird nur hinzugefügt wenn Plot Type = Raincloud
        self.tab_widget.addTab(self.error_tab, "Error Bars")
        self.tab_widget.addTab(self.significance_tab, "Significance")
        
        left_layout.addWidget(self.tab_widget)
        
        # Dialog Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        left_layout.addWidget(button_box)
        
        splitter.addWidget(left_widget)
        
        # Rechte Seite: Preview
        if PlotPreviewWidget:
            preview_frame = QFrame()
            preview_frame.setFrameStyle(QFrame.StyledPanel)
            preview_layout = QVBoxLayout(preview_frame)
            
            preview_label = QLabel("Live Preview")
            preview_label.setAlignment(Qt.AlignCenter)
            preview_label.setStyleSheet("font-weight: bold; padding: 2px; font-size: 10px;")
            preview_label.setMaximumHeight(20)  # Begrenze die Höhe des Labels
            preview_layout.addWidget(preview_label)
            
            self.preview = PlotPreviewWidget()
            if self.groups and self.samples:
                self.preview.set_data(self.groups, self.samples)
            preview_layout.addWidget(self.preview)
            
            splitter.addWidget(preview_frame)
            splitter.setSizes([500, 800])  # Weniger Platz für Tabs, mehr für Preview
        else:
            # Fallback ohne Preview
            no_preview_label = QLabel("Preview not available")
            no_preview_label.setAlignment(Qt.AlignCenter)
            splitter.addWidget(no_preview_label)
            splitter.setSizes([500, 300])
    
    def connect_signals(self):
        """Verbinde alle Signals für Live-Update"""
        self.size_tab.settingsChanged.connect(self.update_preview)
        self.typography_tab.settingsChanged.connect(self.update_preview_immediately)
        self.colors_tab.settingsChanged.connect(self.update_preview)
        self.style_tab.settingsChanged.connect(self.update_preview)
        self.style_tab.settingsChanged.connect(self.update_raincloud_tab_visibility)
        self.raincloud_tab.settingsChanged.connect(self.update_preview)
        # FIXED: Add missing signal connections for complete preview updates
        self.error_tab.settingsChanged.connect(self.update_preview)
        self.significance_tab.settingsChanged.connect(self.update_preview)
        self.error_tab.settingsChanged.connect(self.update_preview)
        self.significance_tab.settingsChanged.connect(self.update_preview)
    
    def update_preview_immediately(self):
        """Sofortige Preview-Aktualisierung für Schriftarten-Änderungen"""
        if hasattr(self, 'preview') and self.preview:
            config = self.get_config()
            
            # Font-Management ist jetzt im StylingManager integriert
            # Einfach das normale Update verwenden - der neue Manager handhabt Fonts optimal
            self.preview.update_plot(config)
            
            # Force immediate redraw for font changes
            try:
                if hasattr(self.preview, 'draw'):
                    self.preview.draw()
                if hasattr(self.preview, 'flush_events'):
                    self.preview.flush_events()
            except Exception as e:
                print(f"Warning: Could not force redraw: {e}")
    
    def update_raincloud_tab_visibility(self):
        """Zeigt/versteckt den Raincloud Tab basierend auf dem Plot Type"""
        plot_type = self.style_tab.plot_type_combo.currentText()
        raincloud_tab_index = self.tab_widget.indexOf(self.raincloud_tab)
        
        if plot_type == 'Raincloud':
            # Tab anzeigen wenn er versteckt ist
            if raincloud_tab_index == -1:
                # Tab ist versteckt, wieder hinzufügen an Position 4 (vor Error Bars)
                self.tab_widget.insertTab(4, self.raincloud_tab, "Raincloud")
        else:
            # Tab verstecken wenn er sichtbar ist
            if raincloud_tab_index != -1:
                self.tab_widget.removeTab(raincloud_tab_index)
    
    def update_preview(self):
        """Aktualisiert die Live-Preview"""
        if hasattr(self, 'preview') and self.preview:
            config = self.get_config()
            self.preview.update_plot(config)
    
    def get_config(self):
        """Sammelt alle Einstellungen aus den Tabs"""
        config = {}
        
        # Sammle Einstellungen von allen Tabs
        config.update(self.size_tab.get_settings())
        config.update(self.typography_tab.get_settings())
        config.update(self.colors_tab.get_settings())
        config.update(self.style_tab.get_settings())
        config.update(self.raincloud_tab.get_settings())
        config.update(self.error_tab.get_settings())
        config.update(self.significance_tab.get_settings())
        
        # AUTOMATISCHE GRÖßENANPASSUNG BASIEREND AUF GRUPPENANZAHL
        num_groups = len(self.groups)
        plot_type = config.get('plot_type', 'Bar')
        
        # Basis-Größen aus Size Tab
        base_width = config.get('width', 8)
        base_height = config.get('height', 6)
        
        if num_groups > 0:
            if plot_type == 'Raincloud':
                # Raincloud ist horizontal: Höhe muss mit Gruppen skalieren
                config['width'] = max(base_width, 8 + num_groups * 0.5)  # Mindestens 8, dann +0.5 pro Gruppe
                config['height'] = max(base_height, 4 + num_groups * 1.2)  # Deutlich mehr Höhe pro Gruppe
            else:
                # Bar, Box, Violin sind vertikal: Breite muss mit Gruppen skalieren
                config['width'] = max(base_width, 6 + num_groups * 1.0)  # Mindestens 6, dann +1.0 pro Gruppe
                config['height'] = max(base_height, 6)  # Mindesthöhe beibehalten
                
            # Zusätzliche Skalierung für sehr viele Gruppen
            if num_groups > 6:
                if plot_type == 'Raincloud':
                    config['height'] += (num_groups - 6) * 0.8  # Extra Höhe ab 6 Gruppen
                else:
                    config['width'] += (num_groups - 6) * 0.5   # Extra Breite ab 6 Gruppen
        
        # Sicherstellen, dass Farben immer gesetzt sind
        # Only set default colors if no colors are configured at all
        if 'colors' not in config or not config['colors']:
            # Use context to determine appropriate default colors
            if self.context == "analysis_only":
                # Use grayscale for analysis-only visualization
                default_colors = [
                    '#2C2C2C', '#4A4A4A', '#686868', '#868686', '#A4A4A4', '#C2C2C2',
                    '#3C3C3C', '#5A5A5A', '#787878', '#969696'
                ]
            else:
                # Use colorful defaults for user plots
                default_colors = ['#FF69B4', '#32CD32', '#FFD700', '#00BFFF', '#DA70D6', '#D8BFD8']
                
            config['colors'] = {}
            for i, group in enumerate(self.groups):
                config['colors'][group] = default_colors[i % len(default_colors)]
        
        return config
    
    def set_groups(self, groups, samples):
        """Aktualisiert Gruppen und Samples"""
        self.groups = groups
        self.samples = samples
        self.colors_tab.set_groups(groups)
        self.raincloud_tab.set_groups(groups)
        if hasattr(self, 'preview'):
            self.preview.set_data(groups, samples)
        self.update_preview()


# Test-Anwendung
if __name__ == "__main__":
    import numpy as np
    
    app = QApplication(sys.argv)
    
    # Test-Daten
    test_groups = ['Control', 'Treatment A', 'Treatment B']
    test_samples = {
        'Control': np.random.normal(10, 2, 50),
        'Treatment A': np.random.normal(12, 3, 45),
        'Treatment B': np.random.normal(8, 1.5, 55)
    }
    
    # Dialog testen
    dialog = PlotAestheticsDialog(test_groups, test_samples)
    if dialog.exec_() == QDialog.Accepted:
        config = dialog.get_config()
        print("User configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    sys.exit(app.exec_())
