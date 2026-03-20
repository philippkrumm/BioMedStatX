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
                             QApplication, QMainWindow, QSplitter, QFrame,
                             QListWidget, QScrollArea)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
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
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        scroll.setWidget(content_widget)
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll)

        # Figure Size Group
        fig_group = QGroupBox("Figure Size")
        fig_layout = QGridLayout(fig_group)
        
        # Width
        fig_layout.addWidget(QLabel("Width (inches):"), 0, 0)
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setMinimumHeight(25)
        self.width_spin.setRange(1, 20)
        self.width_spin.setValue(self.config.get('width', 8))
        self.width_spin.setSingleStep(0.5)
        self.width_spin.valueChanged.connect(self.settingsChanged)
        fig_layout.addWidget(self.width_spin, 0, 1)

        # Height
        fig_layout.addWidget(QLabel("Height (inches):"), 1, 0)
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setMinimumHeight(25)
        self.height_spin.setRange(1, 20)
        self.height_spin.setValue(self.config.get('height', 6))
        self.height_spin.setSingleStep(0.5)
        self.height_spin.valueChanged.connect(self.settingsChanged)
        fig_layout.addWidget(self.height_spin, 1, 1)

        # DPI
        fig_layout.addWidget(QLabel("DPI:"), 2, 0)
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setMinimumHeight(25)
        self.dpi_spin.setRange(72, 600)
        self.dpi_spin.setValue(self.config.get('dpi', 300))
        self.dpi_spin.valueChanged.connect(self.settingsChanged)
        fig_layout.addWidget(self.dpi_spin, 2, 1)

        # Aspect Ratio Presets
        aspect_group = QGroupBox("Aspect Ratio Presets")
        aspect_layout = QHBoxLayout(aspect_group)

        self.square_btn = QPushButton("Square")
        self.square_btn.setMinimumHeight(25)
        self.square_btn.clicked.connect(lambda: self.apply_aspect_preset('square'))
        aspect_layout.addWidget(self.square_btn)

        self.landscape_btn = QPushButton("Landscape (4:3)")
        self.landscape_btn.setMinimumHeight(25)
        self.landscape_btn.clicked.connect(lambda: self.apply_aspect_preset('landscape_4_3'))
        aspect_layout.addWidget(self.landscape_btn)

        self.portrait_btn = QPushButton("Portrait (3:4)")
        self.portrait_btn.setMinimumHeight(25)
        self.portrait_btn.clicked.connect(lambda: self.apply_aspect_preset('portrait_3_4'))
        aspect_layout.addWidget(self.portrait_btn)

        content_layout.addWidget(fig_group)
        content_layout.addWidget(aspect_group)
        content_layout.addStretch()

    def apply_aspect_preset(self, preset):
        current_width = self.width_spin.value()
        current_height = self.height_spin.value()

        if preset == 'square':
            target = max(current_width, current_height)
            self.width_spin.setValue(target)
            self.height_spin.setValue(target)
        elif preset == 'landscape_4_3':
            base = max(current_width, current_height, 4.0)
            self.width_spin.setValue(base)
            self.height_spin.setValue(round(base * 0.75, 2))
        elif preset == 'portrait_3_4':
            base = max(current_width, current_height, 4.0)
            self.height_spin.setValue(base)
            self.width_spin.setValue(round(base * 0.75, 2))

        self.settingsChanged.emit()
    
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
        self.base_font_sizes = {
            'title': self.config.get('fontsize_title', 14),
            'axis': self.config.get('fontsize_axis', 12),
            'ticks': self.config.get('fontsize_ticks', 10)
        }
        # Track if user has explicitly changed font sizes
        self.font_sizes_modified = {
            'title': False,
            'axis': False,
            'ticks': False
        }
        self.init_ui()
        
    def init_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        scroll.setWidget(content_widget)
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll)

        # Font Family Group
        font_family_group = QGroupBox("Font Family")
        font_family_layout = QGridLayout(font_family_group)

        font_family_layout.addWidget(QLabel("Font Family:"), 0, 0)
        self.font_family_combo = QComboBox()
        self.font_family_combo.setMinimumHeight(25)

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

        content_layout.addWidget(font_family_group)

        # Font Sizes Group
        font_group = QGroupBox("Font Sizes")
        font_layout = QGridLayout(font_group)

        # Title Font Size
        font_layout.addWidget(QLabel("Title:"), 0, 0)
        self.title_size_spin = QSpinBox()
        self.title_size_spin.setMinimumHeight(25)
        self.title_size_spin.setRange(8, 48)
        self.title_size_spin.setValue(self.config.get('fontsize_title', 14))
        self.title_size_spin.valueChanged.connect(self.on_title_size_changed)
        font_layout.addWidget(self.title_size_spin, 0, 1)

        # Axis Label Font Size
        font_layout.addWidget(QLabel("Axis Labels:"), 1, 0)
        self.axis_size_spin = QSpinBox()
        self.axis_size_spin.setMinimumHeight(25)
        self.axis_size_spin.setRange(8, 24)
        self.axis_size_spin.setValue(self.config.get('fontsize_axis', 12))
        self.axis_size_spin.valueChanged.connect(self.on_axis_size_changed)
        font_layout.addWidget(self.axis_size_spin, 1, 1)

        # Tick Label Font Size
        font_layout.addWidget(QLabel("Tick Labels:"), 2, 0)
        self.ticks_size_spin = QSpinBox()
        self.ticks_size_spin.setMinimumHeight(25)
        self.ticks_size_spin.setRange(6, 20)
        self.ticks_size_spin.setValue(self.config.get('fontsize_ticks', 10))
        self.ticks_size_spin.valueChanged.connect(self.on_ticks_size_changed)
        font_layout.addWidget(self.ticks_size_spin, 2, 1)

        content_layout.addWidget(font_group)

        # Global typography scaling to preserve manual relative proportions
        scale_group = QGroupBox("Scale All Fonts")
        scale_layout = QGridLayout(scale_group)

        scale_layout.addWidget(QLabel("Scale:"), 0, 0)
        self.font_scale_slider = QSlider(Qt.Horizontal)
        self.font_scale_slider.setMinimumHeight(25)
        self.font_scale_slider.setRange(50, 200)
        self.font_scale_slider.setValue(int(self.config.get('font_scale_percent', 100)))
        self.font_scale_slider.valueChanged.connect(self.on_font_scale_changed)
        scale_layout.addWidget(self.font_scale_slider, 0, 1)

        self.font_scale_label = QLabel(f"{self.font_scale_slider.value()}%")
        scale_layout.addWidget(self.font_scale_label, 0, 2)

        content_layout.addWidget(scale_group)

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

        # Automatic scientific unit formatting with MathText
        self.auto_format_units_check = QCheckBox("Auto-format Units (MathText)")
        self.auto_format_units_check.setChecked(self.config.get('auto_format_units', False))
        self.auto_format_units_check.toggled.connect(self.settingsChanged)
        labels_layout.addWidget(self.auto_format_units_check, 3, 0, 1, 2)

        content_layout.addWidget(labels_group)
        content_layout.addStretch()
    
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

    def on_font_scale_changed(self):
        """Scale all typography values proportionally."""
        scale_factor = self.font_scale_slider.value() / 100.0
        self.font_scale_label.setText(f"{self.font_scale_slider.value()}%")

        self.title_size_spin.blockSignals(True)
        self.axis_size_spin.blockSignals(True)
        self.ticks_size_spin.blockSignals(True)

        self.title_size_spin.setValue(max(self.title_size_spin.minimum(), min(self.title_size_spin.maximum(), int(round(self.base_font_sizes['title'] * scale_factor)))))
        self.axis_size_spin.setValue(max(self.axis_size_spin.minimum(), min(self.axis_size_spin.maximum(), int(round(self.base_font_sizes['axis'] * scale_factor)))))
        self.ticks_size_spin.setValue(max(self.ticks_size_spin.minimum(), min(self.ticks_size_spin.maximum(), int(round(self.base_font_sizes['ticks'] * scale_factor)))))

        self.title_size_spin.blockSignals(False)
        self.axis_size_spin.blockSignals(False)
        self.ticks_size_spin.blockSignals(False)

        self.font_sizes_modified = {'title': True, 'axis': True, 'ticks': True}
        self.settingsChanged.emit()
    
    def get_settings(self):
        settings = {
            'font_family': self.font_family_combo.currentText(),
            'title': self.title_edit.text(),
            'x_label': self.x_label_edit.text(),
            'y_label': self.y_label_edit.text(),
            'show_title': bool(self.title_edit.text().strip()),
            'auto_format_units': self.auto_format_units_check.isChecked(),
            'fontsize_title': self.title_size_spin.value(),
            'fontsize_axis': self.axis_size_spin.value(),
            'fontsize_ticks': self.ticks_size_spin.value(),
            'font_scale_percent': self.font_scale_slider.value()
        }
        
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
        self.journal_palettes = {
            'Nature': ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7'],
            'Science': ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#56B4E9', '#E69F00', '#999999'],
            'NEJM': ['#BC3C29', '#0072B5', '#E18727', '#20854E', '#7876B1', '#6F99AD', '#FFDC91'],
            'Lancet': ['#00468B', '#ED0000', '#42B540', '#0099B4', '#925E9F', '#FDAF91', '#AD002A']
        }
        self.init_ui()
        
    def init_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        scroll.setWidget(content_widget)
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll)

        # Seaborn Style and Palette Selection
        seaborn_group = QGroupBox("Color Style & Palette")
        seaborn_layout = QGridLayout(seaborn_group)

        # Style context selector
        seaborn_layout.addWidget(QLabel("Style Context:"), 0, 0)
        self.style_context_combo = QComboBox()
        self.style_context_combo.setMinimumHeight(25)
        self.style_context_combo.addItems(['notebook', 'paper', 'talk', 'poster'])
        self.style_context_combo.setCurrentText(self.config.get('seaborn_context', 'notebook'))
        self.style_context_combo.currentTextChanged.connect(self.on_seaborn_settings_changed)
        seaborn_layout.addWidget(self.style_context_combo, 0, 1)

        # Color palette selector
        seaborn_layout.addWidget(QLabel("Color Palette:"), 1, 0)
        self.palette_combo = QComboBox()
        self.palette_combo.setMinimumHeight(25)
        # Professional palettes - excluding rainbow/childish ones
        professional_palettes = [
            'Nature', 'Science', 'NEJM', 'Lancet',
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

        content_layout.addWidget(seaborn_group)

        # Individual Colors Group
        colors_group = QGroupBox("Individual Colors")
        self.colors_layout = QGridLayout(colors_group)
        self.update_color_buttons()

        content_layout.addWidget(colors_group)

        # Hatches Group
        hatches_group = QGroupBox("Hatches (Patterns)")
        self.hatches_layout = QGridLayout(hatches_group)
        self.update_hatch_selectors()

        content_layout.addWidget(hatches_group)
        content_layout.addStretch()
    
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
        palette_name = self.palette_combo.currentText()

        if palette_name in self.journal_palettes:
            palette_colors = self.journal_palettes[palette_name]
            for i, group in enumerate(self.groups):
                self.color_buttons[group].set_color(palette_colors[i % len(palette_colors)])
            self.settingsChanged.emit()
            return
        
        if self.use_seaborn_checkbox.isChecked():
            # Apply seaborn palette colors to color buttons
            try:
                import seaborn as sns
                
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
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        scroll.setWidget(content_widget)
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll)

        # Plot Type Group
        type_group = QGroupBox("Plot Type")
        type_layout = QHBoxLayout(type_group)

        type_layout.addWidget(QLabel("Type:"))
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.setMinimumHeight(25)
        self.plot_type_combo.addItems(['Bar', 'Box', 'Violin', 'Raincloud'])
        self.plot_type_combo.setCurrentText(self.config.get('plot_type', 'Bar'))
        self.plot_type_combo.currentTextChanged.connect(self.settingsChanged)
        type_layout.addWidget(self.plot_type_combo)
        type_layout.addStretch()

        content_layout.addWidget(type_group)
        
        # Appearance Group
        appearance_group = QGroupBox("Appearance")
        appearance_layout = QGridLayout(appearance_group)
        
        # Alpha
        appearance_layout.addWidget(QLabel("Transparency:"), 0, 0)
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setMinimumHeight(25)
        self.alpha_spin.setRange(0.1, 1.0)
        self.alpha_spin.setSingleStep(0.1)
        self.alpha_spin.setValue(self.config.get('alpha', 0.8))
        self.alpha_spin.valueChanged.connect(self.settingsChanged)
        appearance_layout.addWidget(self.alpha_spin, 0, 1)

        # Edge Width
        appearance_layout.addWidget(QLabel("Edge Width:"), 1, 0)
        self.edge_width_spin = QDoubleSpinBox()
        self.edge_width_spin.setMinimumHeight(25)
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
        self.axis_thickness_spin.setMinimumHeight(25)
        self.axis_thickness_spin.setRange(0.1, 3.0)
        self.axis_thickness_spin.setSingleStep(0.1)
        self.axis_thickness_spin.setValue(self.config.get('axis_thickness', 0.7))
        self.axis_thickness_spin.valueChanged.connect(self.settingsChanged)
        appearance_layout.addWidget(self.axis_thickness_spin, 4, 1)

        content_layout.addWidget(appearance_group)

        # Publication Standards Group
        publication_group = QGroupBox("Publication Standards")
        publication_layout = QGridLayout(publication_group)

        self.prism_look_check = QCheckBox("Prism Look")
        self.prism_look_check.setChecked(self.config.get('prism_look', False))
        self.prism_look_check.toggled.connect(self.apply_prism_look)
        publication_layout.addWidget(self.prism_look_check, 0, 0, 1, 2)

        self.offset_axes_check = QCheckBox("Offset Axes")
        self.offset_axes_check.setChecked(self.config.get('offset_axes', False))
        self.offset_axes_check.toggled.connect(self.settingsChanged)
        publication_layout.addWidget(self.offset_axes_check, 1, 0, 1, 2)

        publication_layout.addWidget(QLabel("Tick Direction:"), 2, 0)
        self.tick_direction_combo = QComboBox()
        self.tick_direction_combo.setMinimumHeight(25)
        self.tick_direction_combo.addItems(['In', 'Out', 'Both'])
        self.tick_direction_combo.setCurrentText(self.config.get('tick_direction', 'Out').title())
        self.tick_direction_combo.currentTextChanged.connect(self.settingsChanged)
        publication_layout.addWidget(self.tick_direction_combo, 2, 1)

        self.grayscale_preview_check = QCheckBox("Grayscale Preview")
        self.grayscale_preview_check.setChecked(self.config.get('grayscale_preview', False))
        self.grayscale_preview_check.toggled.connect(self.settingsChanged)
        publication_layout.addWidget(self.grayscale_preview_check, 3, 0, 1, 2)

        content_layout.addWidget(publication_group)

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
        self.point_size_spin.setMinimumHeight(25)
        self.point_size_spin.setRange(10, 200)
        self.point_size_spin.setValue(self.config.get('point_size', 80))
        self.point_size_spin.valueChanged.connect(self.settingsChanged)
        points_layout.addWidget(self.point_size_spin, 1, 1)

        # Jitter Strength
        points_layout.addWidget(QLabel("Jitter Strength:"), 2, 0)
        self.jitter_spin = QDoubleSpinBox()
        self.jitter_spin.setMinimumHeight(25)
        self.jitter_spin.setRange(0.0, 1.0)
        self.jitter_spin.setSingleStep(0.1)
        self.jitter_spin.setValue(self.config.get('jitter_strength', 0.3))
        self.jitter_spin.valueChanged.connect(self.settingsChanged)
        points_layout.addWidget(self.jitter_spin, 2, 1)

        # Point style / layout algorithm
        points_layout.addWidget(QLabel("Point Layout:"), 3, 0)
        self.point_style_combo = QComboBox()
        self.point_style_combo.setMinimumHeight(25)
        self.point_style_combo.addItems(['Jitter', 'Beeswarm', 'Strip'])
        point_style_map = {
            'jitter': 'Jitter',
            'swarm': 'Beeswarm',
            'strip': 'Strip'
        }
        self.point_style_combo.setCurrentText(point_style_map.get(self.config.get('point_style', 'jitter'), 'Jitter'))
        self.point_style_combo.currentTextChanged.connect(self.settingsChanged)
        points_layout.addWidget(self.point_style_combo, 3, 1)

        # Superimposed summary+points look
        self.superimposed_mode_check = QCheckBox("Superimposed Look (summary low-alpha, points high-contrast)")
        self.superimposed_mode_check.setChecked(self.config.get('superimposed_mode', False))
        self.superimposed_mode_check.toggled.connect(self.settingsChanged)
        points_layout.addWidget(self.superimposed_mode_check, 4, 0, 1, 2)

        points_layout.addWidget(QLabel("Summary Alpha:"), 5, 0)
        self.summary_alpha_spin = QDoubleSpinBox()
        self.summary_alpha_spin.setMinimumHeight(25)
        self.summary_alpha_spin.setRange(0.1, 1.0)
        self.summary_alpha_spin.setSingleStep(0.05)
        self.summary_alpha_spin.setValue(self.config.get('summary_alpha', self.config.get('alpha', 0.8)))
        self.summary_alpha_spin.valueChanged.connect(self.settingsChanged)
        points_layout.addWidget(self.summary_alpha_spin, 5, 1)

        points_layout.addWidget(QLabel("Point Alpha:"), 6, 0)
        self.point_alpha_spin = QDoubleSpinBox()
        self.point_alpha_spin.setMinimumHeight(25)
        self.point_alpha_spin.setRange(0.1, 1.0)
        self.point_alpha_spin.setSingleStep(0.05)
        self.point_alpha_spin.setValue(self.config.get('point_alpha', 0.8))
        self.point_alpha_spin.valueChanged.connect(self.settingsChanged)
        points_layout.addWidget(self.point_alpha_spin, 6, 1)

        # Violin KDE bandwidth
        points_layout.addWidget(QLabel("Violin Bandwidth:"), 7, 0)
        self.violin_bandwidth_spin = QDoubleSpinBox()
        self.violin_bandwidth_spin.setMinimumHeight(25)
        self.violin_bandwidth_spin.setRange(0.1, 3.0)
        self.violin_bandwidth_spin.setSingleStep(0.1)
        self.violin_bandwidth_spin.setValue(self.config.get('violin_bandwidth', 1.0))
        self.violin_bandwidth_spin.valueChanged.connect(self.settingsChanged)
        points_layout.addWidget(self.violin_bandwidth_spin, 7, 1)

        content_layout.addWidget(points_group)

        # Axis Dynamics Group
        dynamics_group = QGroupBox("Axis Dynamics")
        dynamics_layout = QGridLayout(dynamics_group)

        self.logx_check = QCheckBox("Log X (base 10)")
        self.logx_check.setChecked(self.config.get('logx', False))
        self.logx_check.toggled.connect(self.settingsChanged)
        dynamics_layout.addWidget(self.logx_check, 0, 0)

        self.logy_check = QCheckBox("Log Y (base 10)")
        self.logy_check.setChecked(self.config.get('logy', False))
        self.logy_check.toggled.connect(self.settingsChanged)
        dynamics_layout.addWidget(self.logy_check, 0, 1)

        self.axis_break_check = QCheckBox("Y-Axis Break (Gap)")
        self.axis_break_check.setChecked(self.config.get('axis_break_enabled', False))
        self.axis_break_check.toggled.connect(self.settingsChanged)
        dynamics_layout.addWidget(self.axis_break_check, 1, 0, 1, 2)

        dynamics_layout.addWidget(QLabel("Break Start:"), 2, 0)
        self.axis_break_start_spin = QDoubleSpinBox()
        self.axis_break_start_spin.setMinimumHeight(25)
        self.axis_break_start_spin.setRange(-1e6, 1e6)
        self.axis_break_start_spin.setDecimals(4)
        self.axis_break_start_spin.setValue(self.config.get('axis_break_start', 20.0))
        self.axis_break_start_spin.valueChanged.connect(self.settingsChanged)
        dynamics_layout.addWidget(self.axis_break_start_spin, 2, 1)

        dynamics_layout.addWidget(QLabel("Break End:"), 3, 0)
        self.axis_break_end_spin = QDoubleSpinBox()
        self.axis_break_end_spin.setMinimumHeight(25)
        self.axis_break_end_spin.setRange(-1e6, 1e6)
        self.axis_break_end_spin.setDecimals(4)
        self.axis_break_end_spin.setValue(self.config.get('axis_break_end', 80.0))
        self.axis_break_end_spin.valueChanged.connect(self.settingsChanged)
        dynamics_layout.addWidget(self.axis_break_end_spin, 3, 1)

        content_layout.addWidget(dynamics_group)

        # Legend Group
        legend_group = QGroupBox("Legend")
        legend_layout = QGridLayout(legend_group)

        # Show Legend
        self.legend_check = QCheckBox("Show Legend")
        self.legend_check.setChecked(self.config.get('show_legend', True))
        self.legend_check.toggled.connect(self.settingsChanged)
        legend_layout.addWidget(self.legend_check, 0, 0, 1, 2)

        # Stable legend anchoring coordinates
        legend_layout.addWidget(QLabel("Legend Anchor X:"), 1, 0)
        self.legend_anchor_x_spin = QDoubleSpinBox()
        self.legend_anchor_x_spin.setMinimumHeight(25)
        self.legend_anchor_x_spin.setRange(-5.0, 5.0)
        self.legend_anchor_x_spin.setSingleStep(0.05)
        self.legend_anchor_x_spin.setValue(self.config.get('legend_anchor_x', 1.15))
        self.legend_anchor_x_spin.valueChanged.connect(self.settingsChanged)
        legend_layout.addWidget(self.legend_anchor_x_spin, 1, 1)

        legend_layout.addWidget(QLabel("Legend Anchor Y:"), 2, 0)
        self.legend_anchor_y_spin = QDoubleSpinBox()
        self.legend_anchor_y_spin.setMinimumHeight(25)
        self.legend_anchor_y_spin.setRange(-5.0, 5.0)
        self.legend_anchor_y_spin.setSingleStep(0.05)
        self.legend_anchor_y_spin.setValue(self.config.get('legend_anchor_y', 1.0))
        self.legend_anchor_y_spin.valueChanged.connect(self.settingsChanged)
        legend_layout.addWidget(self.legend_anchor_y_spin, 2, 1)

        # Export padding in millimeters
        legend_layout.addWidget(QLabel("Padding Left (mm):"), 3, 0)
        self.padding_left_mm_spin = QDoubleSpinBox()
        self.padding_left_mm_spin.setMinimumHeight(25)
        self.padding_left_mm_spin.setRange(0.0, 100.0)
        self.padding_left_mm_spin.setSingleStep(0.5)
        self.padding_left_mm_spin.setValue(self.config.get('padding_left_mm', 8.0))
        self.padding_left_mm_spin.valueChanged.connect(self.settingsChanged)
        legend_layout.addWidget(self.padding_left_mm_spin, 3, 1)

        legend_layout.addWidget(QLabel("Padding Right (mm):"), 4, 0)
        self.padding_right_mm_spin = QDoubleSpinBox()
        self.padding_right_mm_spin.setMinimumHeight(25)
        self.padding_right_mm_spin.setRange(0.0, 100.0)
        self.padding_right_mm_spin.setSingleStep(0.5)
        self.padding_right_mm_spin.setValue(self.config.get('padding_right_mm', 6.0))
        self.padding_right_mm_spin.valueChanged.connect(self.settingsChanged)
        legend_layout.addWidget(self.padding_right_mm_spin, 4, 1)

        legend_layout.addWidget(QLabel("Padding Top (mm):"), 5, 0)
        self.padding_top_mm_spin = QDoubleSpinBox()
        self.padding_top_mm_spin.setMinimumHeight(25)
        self.padding_top_mm_spin.setRange(0.0, 100.0)
        self.padding_top_mm_spin.setSingleStep(0.5)
        self.padding_top_mm_spin.setValue(self.config.get('padding_top_mm', 6.0))
        self.padding_top_mm_spin.valueChanged.connect(self.settingsChanged)
        legend_layout.addWidget(self.padding_top_mm_spin, 5, 1)

        legend_layout.addWidget(QLabel("Padding Bottom (mm):"), 6, 0)
        self.padding_bottom_mm_spin = QDoubleSpinBox()
        self.padding_bottom_mm_spin.setMinimumHeight(25)
        self.padding_bottom_mm_spin.setRange(0.0, 100.0)
        self.padding_bottom_mm_spin.setSingleStep(0.5)
        self.padding_bottom_mm_spin.setValue(self.config.get('padding_bottom_mm', 6.0))
        self.padding_bottom_mm_spin.valueChanged.connect(self.settingsChanged)
        legend_layout.addWidget(self.padding_bottom_mm_spin, 6, 1)

        content_layout.addWidget(legend_group)
        content_layout.addStretch()

    def apply_prism_look(self, checked):
        """Apply Prism-like defaults for publication-focused appearance."""
        if checked:
            self.despine_check.setChecked(True)
            self.offset_axes_check.setChecked(True)
            self.tick_direction_combo.setCurrentText('Out')
            self.axis_thickness_spin.setValue(1.0)
        self.settingsChanged.emit()
    
    def get_settings(self):
        point_style_map = {
            'Jitter': 'jitter',
            'Beeswarm': 'swarm',
            'Strip': 'strip'
        }

        return {
            'plot_type': self.plot_type_combo.currentText(),
            'alpha': self.summary_alpha_spin.value(),
            'summary_alpha': self.summary_alpha_spin.value(),
            'bar_linewidth': self.edge_width_spin.value(),
            'grid': self.grid_check.isChecked(),
            'grid_style': 'major' if self.grid_check.isChecked() else 'none',  # Für DataVisualizer
            'minor_ticks': self.minor_ticks_check.isChecked(),
            'logx': self.logx_check.isChecked(),
            'logy': self.logy_check.isChecked(),
            'axis_break_enabled': self.axis_break_check.isChecked(),
            'axis_break_start': self.axis_break_start_spin.value(),
            'axis_break_end': self.axis_break_end_spin.value(),
            'despine': self.despine_check.isChecked(),
            'axis_thickness': self.axis_thickness_spin.value(),
            'prism_look': self.prism_look_check.isChecked(),
            'offset_axes': self.offset_axes_check.isChecked(),
            'axis_offset_points': self.config.get('axis_offset_points', 10),
            'tick_direction': self.tick_direction_combo.currentText().lower(),
            'grayscale_preview': self.grayscale_preview_check.isChecked(),
            'show_points': self.points_check.isChecked(),
            'point_style': point_style_map.get(self.point_style_combo.currentText(), 'jitter'),
            'point_size': self.point_size_spin.value(),
            'point_alpha': self.point_alpha_spin.value(),
            'jitter_strength': self.jitter_spin.value(),
            'superimposed_mode': self.superimposed_mode_check.isChecked(),
            'violin_bandwidth': self.violin_bandwidth_spin.value(),
            'show_legend': self.legend_check.isChecked(),
            'legend_anchor_x': self.legend_anchor_x_spin.value(),
            'legend_anchor_y': self.legend_anchor_y_spin.value(),
            'padding_left_mm': self.padding_left_mm_spin.value(),
            'padding_right_mm': self.padding_right_mm_spin.value(),
            'padding_top_mm': self.padding_top_mm_spin.value(),
            'padding_bottom_mm': self.padding_bottom_mm_spin.value()
        }


class SymbolsTab(QWidget):
    """Tab for marker shape settings per group"""
    settingsChanged = pyqtSignal()

    def __init__(self, groups=None, config=None):
        super().__init__()
        self.groups = groups or []
        self.config = config or {}
        self.symbol_combos = {}
        self.symbol_map = {
            'Circle': 'o',
            'Square': 's',
            'Triangle': '^',
            'Diamond': 'D'
        }
        self.init_ui()

    def init_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        scroll.setWidget(content_widget)
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll)

        symbols_group = QGroupBox("Marker Shapes")
        self.symbols_layout = QGridLayout(symbols_group)
        self.update_symbol_selectors()

        content_layout.addWidget(symbols_group)
        content_layout.addStretch()

    def update_symbol_selectors(self):
        for i in reversed(range(self.symbols_layout.count())):
            self.symbols_layout.itemAt(i).widget().setParent(None)

        self.symbol_combos = {}
        current_shapes = self.config.get('marker_shapes', {})

        for i, group in enumerate(self.groups):
            self.symbols_layout.addWidget(QLabel(f"{group}:"), i, 0)
            combo = QComboBox()
            combo.addItems(list(self.symbol_map.keys()))

            current_marker = current_shapes.get(group, 'o')
            for name, marker in self.symbol_map.items():
                if marker == current_marker:
                    combo.setCurrentText(name)
                    break

            combo.currentTextChanged.connect(self.settingsChanged)
            self.symbol_combos[group] = combo
            self.symbols_layout.addWidget(combo, i, 1)

    def get_settings(self):
        marker_shapes = {}
        for group, combo in self.symbol_combos.items():
            marker_shapes[group] = self.symbol_map.get(combo.currentText(), 'o')
        return {'marker_shapes': marker_shapes}

    def set_groups(self, groups):
        self.groups = groups
        self.update_symbol_selectors()


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
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        scroll.setWidget(content_widget)
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll)

        self.global_group_color_check = QCheckBox("Global Group Color Mode (auto shades)")
        self.global_group_color_check.setChecked(self.config.get('global_group_color_mode', True))
        self.global_group_color_check.toggled.connect(self.settingsChanged)
        content_layout.addWidget(self.global_group_color_check)

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

        content_layout.addWidget(violin_group)
        content_layout.addWidget(box_group)
        content_layout.addWidget(point_group)

        # Spacing and Layout Group
        spacing_group = QGroupBox("Spacing and Layout")
        spacing_layout = QGridLayout(spacing_group)

        # Group Spacing
        spacing_layout.addWidget(QLabel("Group Spacing:"), 0, 0)
        self.group_spacing_spin = QDoubleSpinBox()
        self.group_spacing_spin.setMinimumHeight(25)
        self.group_spacing_spin.setRange(0.3, 2.0)
        self.group_spacing_spin.setSingleStep(0.1)
        self.group_spacing_spin.setValue(self.config.get('group_spacing', 0.90))
        self.group_spacing_spin.valueChanged.connect(self.settingsChanged)
        spacing_layout.addWidget(self.group_spacing_spin, 0, 1)

        # Point Vertical Offset
        spacing_layout.addWidget(QLabel("Point Vertical Offset:"), 1, 0)
        self.point_offset_spin = QDoubleSpinBox()
        self.point_offset_spin.setMinimumHeight(25)
        self.point_offset_spin.setRange(0.1, 0.5)
        self.point_offset_spin.setSingleStep(0.05)
        self.point_offset_spin.setValue(self.config.get('point_offset', 0.2))
        self.point_offset_spin.valueChanged.connect(self.settingsChanged)
        spacing_layout.addWidget(self.point_offset_spin, 1, 1)

        # Point Horizontal Jitter
        spacing_layout.addWidget(QLabel("Point Horizontal Jitter:"), 2, 0)
        self.point_jitter_spin = QDoubleSpinBox()
        self.point_jitter_spin.setMinimumHeight(25)
        self.point_jitter_spin.setRange(0.01, 0.1)
        self.point_jitter_spin.setSingleStep(0.01)
        self.point_jitter_spin.setValue(self.config.get('point_jitter', 0.05))
        self.point_jitter_spin.valueChanged.connect(self.settingsChanged)
        spacing_layout.addWidget(self.point_jitter_spin, 2, 1)

        # Violin Width
        spacing_layout.addWidget(QLabel("Violin Width:"), 3, 0)
        self.violin_width_spin = QDoubleSpinBox()
        self.violin_width_spin.setMinimumHeight(25)
        self.violin_width_spin.setRange(0.3, 1.5)
        self.violin_width_spin.setSingleStep(0.1)
        self.violin_width_spin.setValue(self.config.get('violin_width', 0.8))
        self.violin_width_spin.valueChanged.connect(self.settingsChanged)
        spacing_layout.addWidget(self.violin_width_spin, 3, 1)

        # Box Width
        spacing_layout.addWidget(QLabel("Box Width:"), 4, 0)
        self.box_width_spin = QDoubleSpinBox()
        self.box_width_spin.setMinimumHeight(25)
        self.box_width_spin.setRange(0.1, 0.8)
        self.box_width_spin.setSingleStep(0.1)
        self.box_width_spin.setValue(self.config.get('box_width', 0.2))
        self.box_width_spin.valueChanged.connect(self.settingsChanged)
        spacing_layout.addWidget(self.box_width_spin, 4, 1)

        content_layout.addWidget(spacing_group)
        content_layout.addStretch()
    
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
            'global_group_color_mode': self.global_group_color_check.isChecked(),
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
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        scroll.setWidget(content_widget)
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll)

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
        self.error_type_combo.setMinimumHeight(25)
        self.error_type_combo.addItems(['sd', 'se', 'ci'])
        self.error_type_combo.setCurrentText(self.config.get('error_type', 'sd'))
        self.error_type_combo.currentTextChanged.connect(self.settingsChanged)
        error_layout.addWidget(self.error_type_combo, 1, 1)

        # Error Style
        error_layout.addWidget(QLabel("Error Style:"), 2, 0)
        self.error_style_combo = QComboBox()
        self.error_style_combo.setMinimumHeight(25)
        self.error_style_combo.addItems(['caps', 'line'])
        self.error_style_combo.setCurrentText(self.config.get('error_style', 'caps'))
        self.error_style_combo.currentTextChanged.connect(self.settingsChanged)
        error_layout.addWidget(self.error_style_combo, 2, 1)

        # Cap Size
        error_layout.addWidget(QLabel("Cap Size:"), 3, 0)
        self.capsize_spin = QDoubleSpinBox()
        self.capsize_spin.setMinimumHeight(25)
        self.capsize_spin.setRange(0.0, 1.0)
        self.capsize_spin.setSingleStep(0.01)
        self.capsize_spin.setValue(self.config.get('capsize', 0.05))
        self.capsize_spin.valueChanged.connect(self.settingsChanged)
        error_layout.addWidget(self.capsize_spin, 3, 1)

        content_layout.addWidget(error_group)
        content_layout.addStretch()
    
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

    def __init__(self, config=None, analysis_result=None):
        super().__init__()
        self.config = config or {}
        self.analysis_result = analysis_result
        self.pair_checkboxes = {}  # (group1, group2) -> QCheckBox
        self.init_ui()
        
    def init_ui(self):
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        scroll_area.setWidget(content_widget)
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll_area)

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
        self.letters_fontsize_spin.setMinimumHeight(25)
        self.letters_fontsize_spin.setRange(6, 24)
        self.letters_fontsize_spin.setValue(self.config.get('significance_font_size', 12))
        self.letters_fontsize_spin.valueChanged.connect(self.settingsChanged)
        letters_layout.addWidget(self.letters_fontsize_spin, 1, 1)

        # Letters Height Offset
        letters_layout.addWidget(QLabel("Height Offset:"), 2, 0)
        self.letters_offset_spin = QDoubleSpinBox()
        self.letters_offset_spin.setMinimumHeight(25)
        self.letters_offset_spin.setRange(0.0, 0.5)
        self.letters_offset_spin.setSingleStep(0.01)
        self.letters_offset_spin.setValue(self.config.get('significance_height_offset', 0.05))
        self.letters_offset_spin.valueChanged.connect(self.settingsChanged)
        letters_layout.addWidget(self.letters_offset_spin, 2, 1)

        content_layout.addWidget(letters_group)

        # Significance Brackets Group
        brackets_group = QGroupBox("Significance Brackets")
        brackets_layout = QGridLayout(brackets_group)

        # Bracket Line Width
        brackets_layout.addWidget(QLabel("Line Width:"), 0, 0)
        self.bracket_linewidth_spin = QDoubleSpinBox()
        self.bracket_linewidth_spin.setMinimumHeight(25)
        self.bracket_linewidth_spin.setRange(0.5, 5.0)
        self.bracket_linewidth_spin.setSingleStep(0.1)
        self.bracket_linewidth_spin.setValue(self.config.get('bracket_line_width', 2.0))
        self.bracket_linewidth_spin.valueChanged.connect(self.settingsChanged)
        brackets_layout.addWidget(self.bracket_linewidth_spin, 0, 1)

        # Bracket Font Size
        brackets_layout.addWidget(QLabel("Font Size:"), 1, 0)
        self.bracket_fontsize_spin = QSpinBox()
        self.bracket_fontsize_spin.setMinimumHeight(25)
        self.bracket_fontsize_spin.setRange(8, 30)
        self.bracket_fontsize_spin.setValue(self.config.get('bracket_font_size', 16))
        self.bracket_fontsize_spin.valueChanged.connect(self.settingsChanged)
        brackets_layout.addWidget(self.bracket_fontsize_spin, 1, 1)

        # Bracket Vertical Length
        brackets_layout.addWidget(QLabel("Vertical Length:"), 2, 0)
        self.bracket_vertical_spin = QDoubleSpinBox()
        self.bracket_vertical_spin.setMinimumHeight(25)
        self.bracket_vertical_spin.setRange(0.1, 1.0)
        self.bracket_vertical_spin.setSingleStep(0.05)
        self.bracket_vertical_spin.setValue(self.config.get('bracket_vertical_fraction', 0.25))
        self.bracket_vertical_spin.valueChanged.connect(self.settingsChanged)
        brackets_layout.addWidget(self.bracket_vertical_spin, 2, 1)

        # Bracket Spacing
        brackets_layout.addWidget(QLabel("Spacing:"), 3, 0)
        self.bracket_spacing_spin = QDoubleSpinBox()
        self.bracket_spacing_spin.setMinimumHeight(25)
        self.bracket_spacing_spin.setRange(0.05, 0.5)
        self.bracket_spacing_spin.setSingleStep(0.01)
        self.bracket_spacing_spin.setValue(self.config.get('bracket_spacing', 0.1))
        self.bracket_spacing_spin.valueChanged.connect(self.settingsChanged)
        brackets_layout.addWidget(self.bracket_spacing_spin, 3, 1)

        # P-value style
        brackets_layout.addWidget(QLabel("P-value Style:"), 4, 0)
        self.pvalue_style_combo = QComboBox()
        self.pvalue_style_combo.setMinimumHeight(25)
        self.pvalue_style_combo.addItems(['GP: 0.0332 (*)', 'Exact p-value', 'Fixed stars'])
        self.pvalue_style_combo.setCurrentText(self.config.get('p_value_style', 'Fixed stars'))
        self.pvalue_style_combo.currentTextChanged.connect(self.settingsChanged)
        brackets_layout.addWidget(self.pvalue_style_combo, 4, 1)

        content_layout.addWidget(brackets_group)

        # Pairwise comparisons section — populated from real analysis_result
        pairs_group = QGroupBox("Bracket Comparisons (from analysis)")
        pairs_layout = QVBoxLayout(pairs_group)

        pairwise = (self.analysis_result or {}).get('pairwise_comparisons', [])
        if pairwise:
            info_label = QLabel("Select comparisons to display as brackets:")
            info_label.setWordWrap(True)
            pairs_layout.addWidget(info_label)
            inner_scroll = QScrollArea()
            inner_scroll.setWidgetResizable(True)
            inner_scroll.setMaximumHeight(160)
            inner = QWidget()
            inner_layout = QVBoxLayout(inner)
            inner_layout.setContentsMargins(4, 4, 4, 4)
            inner_layout.setSpacing(2)
            prechecked = set()
            if isinstance(self.config.get('selected_pairs'), list):
                for p in self.config['selected_pairs']:
                    prechecked.add(tuple(p))
            for comp in pairwise:
                g1 = comp.get('group1', '')
                g2 = comp.get('group2', '')
                if not g1 or not g2:
                    continue
                p_val = comp.get('p_value')
                sig = comp.get('significant', False)
                p_str = f"p={p_val:.4f}" if p_val is not None else "p=N/A"
                sig_str = " *" if sig else ""
                label = f"{g1} vs {g2}  ({p_str}{sig_str})"
                cb = QCheckBox(label)
                cb.setChecked(len(prechecked) == 0 or (g1, g2) in prechecked)
                cb.stateChanged.connect(self.settingsChanged)
                self.pair_checkboxes[(g1, g2)] = cb
                inner_layout.addWidget(cb)
            inner_layout.addStretch()
            inner.setLayout(inner_layout)
            inner_scroll.setWidget(inner)
            pairs_layout.addWidget(inner_scroll)
            # Select all / none buttons
            btn_row = QHBoxLayout()
            btn_all = QPushButton("All")
            btn_none = QPushButton("None")
            btn_all.setMaximumWidth(60)
            btn_none.setMaximumWidth(60)
            btn_all.clicked.connect(lambda: self._set_all_pairs(True))
            btn_none.clicked.connect(lambda: self._set_all_pairs(False))
            btn_row.addWidget(btn_all)
            btn_row.addWidget(btn_none)
            btn_row.addStretch()
            pairs_layout.addLayout(btn_row)
        else:
            no_data_label = QLabel("No post-hoc data available.\nRun analysis with a post-hoc test to enable bracket selection.")
            no_data_label.setWordWrap(True)
            no_data_label.setStyleSheet("color: gray; font-style: italic;")
            pairs_layout.addWidget(no_data_label)

        content_layout.addWidget(pairs_group)
        content_layout.addStretch()

    def _set_all_pairs(self, checked: bool):
        for cb in self.pair_checkboxes.values():
            cb.setChecked(checked)

    def get_settings(self):
        selected_pairs = [list(key) for key, cb in self.pair_checkboxes.items() if cb.isChecked()]
        return {
            'show_significance_letters': self.show_letters_check.isChecked(),
            'significance_font_size': self.letters_fontsize_spin.value(),
            'significance_height_offset': self.letters_offset_spin.value(),
            # Bracket settings
            'bracket_line_width': self.bracket_linewidth_spin.value(),
            'bracket_font_size': self.bracket_fontsize_spin.value(),
            'bracket_vertical_fraction': self.bracket_vertical_spin.value(),
            'bracket_spacing': self.bracket_spacing_spin.value(),
            'p_value_style': self.pvalue_style_combo.currentText(),
            'bracket_color': '#000000',  # Always black
            # Selected pairs (original group keys, never display labels)
            'selected_pairs': selected_pairs,
        }


class PlotAestheticsDialog(QDialog):
    """
    Hauptdialog für Plot-Einstellungen mit Tab-Interface und Live-Preview
    """
    
    def __init__(self, groups=None, samples=None, config=None, parent=None, context="user_plot",
                 default_filename=None, show_export_controls=True, analysis_result=None, dependent=False):
        super().__init__(parent)
        self.setObjectName("plotAestheticsDialog")
        self.groups = groups or []
        self.samples = samples or {}
        self.config = config or {}
        self.context = context  # "user_plot" or "analysis_only"
        self.default_filename = default_filename
        self.show_export_controls = show_export_controls
        self.analysis_result = analysis_result
        self.dependent = dependent
        
        self.setWindowTitle("Plot Appearance Settings")
        self.setModal(True)
        screen = QDesktopWidget().screenGeometry()
        
        # Adaptive sizing for different display types and resolutions
        screen_width = screen.width()
        screen_height = screen.height()
        
        # High-resolution displays (Retina, 4K, etc.) - like MacBook Air 2880x1864
        if screen_width >= 2560:  # High-res displays
            width = min(1500, int(screen_width * 0.72))
            height = min(980, int(screen_height * 0.82))
        # Medium resolution displays
        elif screen_width >= 1920:  # Full HD and similar
            width = min(1320, int(screen_width * 0.78))
            height = min(900, int(screen_height * 0.84))
        # Standard/smaller displays
        else:  # < 1920px width
            width = min(1180, int(screen_width * 0.88))
            height = min(860, int(screen_height * 0.88))
            
        self.setMinimumSize(900, 600)
        self.resize(width, height)
        self.move(
            (screen.width() - width) // 2,
            (screen.height() - height) // 2
        )
        
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(250)
        self._preview_timer.timeout.connect(self._do_update_preview)

        self.init_ui()
        self.connect_signals()

        # Initial update für Raincloud Tab Sichtbarkeit
        self.update_raincloud_tab_visibility()
        
        # Initiale Preview
        if self.groups and self.samples:
            self.update_preview()
    
    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)
        
        # Splitter für Tabs und Preview
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        self.splitter = splitter  # store for resizeEvent
        main_layout.addWidget(splitter)
        
        # Linke Seite: Tab Widget für Einstellungen
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        if self.show_export_controls:
            export_group = QGroupBox("Export")
            export_layout = QGridLayout(export_group)
            export_layout.setContentsMargins(8, 8, 8, 8)
            export_layout.setHorizontalSpacing(8)
            export_layout.setVerticalSpacing(6)

            export_layout.addWidget(QLabel("Output file name:"), 0, 0)
            self.file_name_edit = QLineEdit()
            configured_name = self.config.get('file_name') or self.default_filename or ""
            self.file_name_edit.setText(configured_name)
            self.file_name_edit.setPlaceholderText("Default: dataset-based filename")
            export_layout.addWidget(self.file_name_edit, 0, 1)

            export_layout.addWidget(QLabel("Group order (drag to sort):"), 1, 0, Qt.AlignTop)
            self.order_list = QListWidget()
            self.order_list.setDragDropMode(QListWidget.InternalMove)
            self.order_list.setMaximumHeight(130)
            for group in self.groups:
                self.order_list.addItem(str(group))
            export_layout.addWidget(self.order_list, 1, 1)

            left_layout.addWidget(export_group)

            # Keep preview and all tabs in sync with current group ordering.
            if self.order_list.model() is not None:
                self.order_list.model().rowsMoved.connect(self._on_group_order_changed)
        
        # Tab Widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setUsesScrollButtons(True)
        self.tab_widget.setElideMode(Qt.ElideRight)
        
        # Tabs erstellen
        self.size_tab = SizeTab(self.config)
        self.typography_tab = TypographyTab(self.config)
        self.colors_tab = ColorsTab(self.groups, self.config, self.context)
        self.style_tab = StyleTab(self.config)
        self.symbols_tab = SymbolsTab(self.groups, self.config)
        self.raincloud_tab = RaincloudTab(self.groups, self.config)
        self.error_tab = ErrorBarsTab(self.config)
        self.significance_tab = SignificanceTab(self.config, analysis_result=self.analysis_result)
        
        # Set dialog reference for cross-tab communication
        self.colors_tab.dialog_ref = self
        
        # Tabs hinzufügen
        self.tab_widget.addTab(self.size_tab, "Size")
        self.tab_widget.addTab(self.typography_tab, "Typography")
        self.tab_widget.addTab(self.colors_tab, "Colors")
        self.tab_widget.addTab(self.symbols_tab, "Symbols")
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
            preview_frame.setFrameStyle(QFrame.Sunken)
            preview_layout = QVBoxLayout(preview_frame)
            preview_layout.setContentsMargins(2, 2, 2, 2)

            self.preview = PlotPreviewWidget()
            if self.groups and self.samples:
                self.preview.set_data(self.groups, self.samples)
            preview_layout.addWidget(self.preview)
            
            splitter.addWidget(preview_frame)
            splitter.setStretchFactor(0, 2)
            splitter.setStretchFactor(1, 3)
            splitter.setSizes([560, 840])
        else:
            # Fallback ohne Preview
            no_preview_label = QLabel("Preview not available")
            no_preview_label.setAlignment(Qt.AlignCenter)
            splitter.addWidget(no_preview_label)
            splitter.setStretchFactor(0, 2)
            splitter.setStretchFactor(1, 3)
            splitter.setSizes([560, 840])

    def _get_ordered_groups(self):
        if hasattr(self, 'order_list') and self.order_list is not None:
            return [self.order_list.item(i).text() for i in range(self.order_list.count())]
        return list(self.groups)

    def _on_group_order_changed(self, *args):
        ordered_groups = self._get_ordered_groups()
        self.groups = ordered_groups
        if hasattr(self, 'preview') and self.preview:
            ordered_samples = {g: self.samples[g] for g in ordered_groups if g in self.samples}
            self.preview.set_data(ordered_groups, ordered_samples)
        self.update_preview()
    
    def connect_signals(self):
        """Verbinde alle Signals für Live-Update"""
        # Block signals on all tabs during wiring to prevent spurious preview calls
        tabs = [self.size_tab, self.typography_tab, self.colors_tab,
                self.symbols_tab, self.style_tab, self.raincloud_tab,
                self.error_tab, self.significance_tab]
        for tab in tabs:
            tab.blockSignals(True)

        self.size_tab.settingsChanged.connect(self.update_preview)
        self.typography_tab.settingsChanged.connect(self.update_preview_immediately)
        self.colors_tab.settingsChanged.connect(self.update_preview)
        self.symbols_tab.settingsChanged.connect(self.update_preview)
        self.style_tab.settingsChanged.connect(self.update_preview)
        self.style_tab.settingsChanged.connect(self.update_raincloud_tab_visibility)
        self.raincloud_tab.settingsChanged.connect(self.update_preview)
        self.error_tab.settingsChanged.connect(self.update_preview)
        self.significance_tab.settingsChanged.connect(self.update_preview)

        for tab in tabs:
            tab.blockSignals(False)
    
    def update_preview_immediately(self):
        """Sofortige Preview-Aktualisierung für Schriftarten-Änderungen"""
        if hasattr(self, 'preview') and self.preview:
            config = self.get_config()
            config['pairwise_results'] = (self.analysis_result or {}).get('pairwise_comparisons', [])
            self._apply_preview_aspect(config)
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
                # Tab ist versteckt, wieder hinzufügen vor Error Bars
                self.tab_widget.insertTab(5, self.raincloud_tab, "Raincloud")
        else:
            # Tab verstecken wenn er sichtbar ist
            if raincloud_tab_index != -1:
                self.tab_widget.removeTab(raincloud_tab_index)
    
    def _apply_preview_aspect(self, config):
        """Preview always fills the available canvas — aspect ratio only matters for export."""
        pass

    def update_preview(self):
        """Schedules a debounced preview update (250 ms quiet period)."""
        self._preview_timer.start()

    def _do_update_preview(self):
        """Aktualisiert die Live-Preview"""
        if hasattr(self, 'preview') and self.preview:
            ordered_groups = self._get_ordered_groups()
            if ordered_groups != self.groups:
                self.groups = ordered_groups
            if self.samples:
                ordered_samples = {g: self.samples[g] for g in self.groups if g in self.samples}
                self.preview.set_data(self.groups, ordered_samples)
            config = self.get_config()
            # Inject real pairwise results for significance rendering
            config['pairwise_results'] = (self.analysis_result or {}).get('pairwise_comparisons', [])
            self._apply_preview_aspect(config)
            self.preview.update_plot(config)
    
    def get_config(self):
        """Sammelt alle Einstellungen aus den Tabs"""
        config = {}
        
        # Sammle Einstellungen von allen Tabs
        config.update(self.size_tab.get_settings())
        config.update(self.typography_tab.get_settings())
        config.update(self.colors_tab.get_settings())
        config.update(self.symbols_tab.get_settings())
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

        config['groups'] = self._get_ordered_groups()
        config['group_order'] = config['groups'][:]
        if hasattr(self, 'file_name_edit') and self.file_name_edit is not None:
            config['file_name'] = self.file_name_edit.text().strip() or None
        config['create_plot'] = True
        config['dependent'] = self.dependent

        return config
    
    def set_dependent(self, val, show_lines=True):
        """Setzt den Paired/Dependent-Status."""
        self.dependent = bool(val)


    def set_groups(self, groups, samples):
        """Aktualisiert Gruppen und Samples"""
        self.groups = groups
        self.samples = samples
        self.colors_tab.set_groups(groups)
        self.symbols_tab.set_groups(groups)
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
