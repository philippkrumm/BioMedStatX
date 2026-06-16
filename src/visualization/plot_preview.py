"""
Preview widget for plot display
Shows live preview of plot settings
"""

import matplotlib
matplotlib.use('Qt5Agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import logging
logger = logging.getLogger(__name__)

# Import the DataVisualizer class
try:
    from visualization.datavisualizer import DataVisualizer
except ImportError:
    try:
        # Fallback: try to get it from analysis.stats_functions
        from analysis.stats_functions import get_data_visualizer
        DataVisualizer = get_data_visualizer()
    except ImportError:
        # Final fallback if import does not work
        logger.info("Warning: Could not import DataVisualizer")
        DataVisualizer = None

class PlotPreviewWidget(FigureCanvasQTAgg):
    """
    Widget for live preview of plots based on configuration.
    Uses the central plot_from_config method for consistent rendering.
    """
    
    def __init__(self, parent=None, figsize=(5, 4), dpi=100):
        """
        Initializes the preview widget with live preview.
        """
        self.fig = Figure(figsize=figsize, dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)

        # Initial subplot
        self.ax = self.fig.add_subplot(111)

        # Data container
        self.groups = []
        self.samples = {}

        # Default configuration for all appearance options
        self.default_config = {
            'plot_type': 'Bar',
            'colors': {},
            'hatches': {},
            'alpha': 0.8,
            'error_type': 'sd',
            'error_style': 'caps',
            'bar_edge_color': 'black',
            'bar_linewidth': 0.5,
            'grid': False,
            'grid_style': 'none',
            'minor_ticks': False,
            'logx': False,
            'logy': False,
            'axis_break_enabled': False,
            'axis_break_start': 20.0,
            'axis_break_end': 80.0,
            'despine': True,
            'prism_look': False,
            'offset_axes': False,
            'axis_offset_points': 10,
            'tick_direction': 'out',
            'axis_thickness': 0.7,
            'grayscale_preview': False,
            'font_family': 'Arial',
            'show_error_bars': True,
            'show_points': True,
            'point_style': 'jitter',
            'point_size': 80,
            'point_alpha': 0.8,
            'summary_alpha': 0.8,
            'marker_shapes': {},
            'jitter_strength': 0.3,
            'point_style': 'jitter',
            'superimposed_mode': False,
            'violin_bandwidth': 1.0,
            'show_significance_letters': True,
            'show_legend': True,
            'legend_anchor_x': 1.15,
            'legend_anchor_y': 1.0,
            'padding_left_mm': 8.0,
            'padding_right_mm': 6.0,
            'padding_top_mm': 6.0,
            'padding_bottom_mm': 6.0,
            'capsize': 0.05,
            'x_label': '',
            'y_label': '',
            'title': '',
            'fontsize_axis': 12,
            'fontsize_title': 14,
            'fontsize_ticks': 10,
            'grid_alpha': 0.3,
            'show_title': True,
            'auto_format_units': False,
            'width': 8,
            'height': 6,
            'dpi': 300,
            'theme': 'default',
            # Raincloud specific defaults
            'group_spacing': 0.90,
            'point_offset': 0.2,
            'point_jitter': 0.05,
            'violin_width': 0.8,
            'box_width': 0.2,
            'global_group_color_mode': True,
            'p_value_style': 'Fixed stars',
            'frame_thickness': 0.7
        }

        # Initial empty display
        self._show_placeholder()
    
    def _show_placeholder(self):
        """Shows placeholder text when no data is available"""
        self.ax.clear()
        self.ax.text(0.5, 0.5, 'No data available\nfor preview', 
                     ha='center', va='center', transform=self.ax.transAxes,
                     fontsize=12, color='gray')
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis('off')
        self.draw()
    
    def set_data(self, groups, samples):
        """
        Sets the current data for the preview.
        
        Parameters:
        -----------
        groups : list
            List of group names
        samples : dict
            Dictionary with group names as keys and data values as values
        """
        self.groups = groups if groups else []
        self.samples = samples if samples else {}
        
        # Validate data
        if not self.groups or not self.samples:
            self._show_placeholder()
            return
        
        # Check if all groups have data
        valid_groups = []
        valid_samples = {}
        
        for group in self.groups:
            if group in self.samples and len(self.samples[group]) > 0:
                valid_groups.append(group)
                valid_samples[group] = self.samples[group]
        
        self.groups = valid_groups
        self.samples = valid_samples
        
        if not self.groups:
            self._show_placeholder()
    
    def update_plot(self, config=None):
        """
        Updates the plot display based on the configuration.
        
        Parameters:
        -----------
        config : dict, optional
            Configuration dictionary with plot parameters.
            If None, the default configuration is used.
        """
        if not self.groups or not self.samples:
            self._show_placeholder()
            return
        
        if DataVisualizer is None:
            self.ax.clear()
            self.ax.text(0.5, 0.5, 'DataVisualizer not available', 
                         ha='center', va='center', transform=self.ax.transAxes,
                         fontsize=12, color='red')
            self.ax.axis('off')
            self.draw()
            return
        
        # Use default config if none provided
        if config is None:
            config = self.default_config.copy()
        else:
            # Merge with default config for missing keys
            merged_config = self.default_config.copy()
            merged_config.update(config)
            config = merged_config
            
        # Ensure colors are set if not provided
        if not config.get('colors') and self.groups:
            # Only set default colors if absolutely no colors are provided
            # This should rarely happen since configs should always include colors
            DEFAULT_COLORS = ['#0f766e', '#1f7a5a', '#b7791f', '#9f3a38', '#1d4ed8', '#7c3aed']
            colors_dict = {}
            for i, group in enumerate(self.groups):
                colors_dict[group] = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
            config['colors'] = colors_dict
            logger.debug(f"DEBUG: PlotPreviewWidget set fallback colors: {colors_dict}")
        else:
            logger.debug(f"DEBUG: PlotPreviewWidget using provided colors: {config.get('colors', {})}")
        
        try:
            # Clear and redraw
            self.ax.clear()

            # Markiere als Preview für optimiertes Styling
            config['_is_preview'] = True

            # Extract pairwise results before passing config to avoid key leakage
            pairwise_results = config.pop('pairwise_results', []) or []

            # Use the central dispatcher method (Font-Management ist jetzt integriert)
            DataVisualizer.plot_from_config(self.ax, self.groups, self.samples, config,
                                            pairwise_results=pairwise_results)
            
            # Force immediate redraw
            self.draw_idle()
            self.flush_events()
            
        except Exception as e:
            logger.error(f"Error in plot update: {str(e)}")
            self.ax.clear()
            self.ax.text(0.5, 0.5, f'Error while drawing:\n{str(e)}', 
                         ha='center', va='center', transform=self.ax.transAxes,
                         fontsize=10, color='red')
            self.ax.axis('off')
            self.draw()
    
    def get_current_config(self):
        """
        Returns the current configuration.
        
        Returns:
        --------
        dict
            Current configuration
        """
        return self.default_config.copy()
    
    def set_default_config(self, config):
        """
        Sets a new default configuration.
        
        Parameters:
        -----------
        config : dict
            New default configuration
        """
        if config:
            self.default_config.update(config)
