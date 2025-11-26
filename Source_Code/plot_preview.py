"""
Preview widget for plot display
Shows live preview of plot settings
"""

import sys
import os
import matplotlib
matplotlib.use('Qt5Agg')

from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

# Import the DataVisualizer class
try:
    from datavisualizer import DataVisualizer
except ImportError:
    try:
        # Fallback: try to get it from stats_functions
        from stats_functions import get_data_visualizer
        DataVisualizer = get_data_visualizer()
    except ImportError:
        # Final fallback if import does not work
        print("Warning: Could not import DataVisualizer")
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
            'despine': True,
            'axis_thickness': 0.7,
            'font_family': 'Arial',
            'show_error_bars': True,
            'show_points': True,
            'point_style': 'jitter',
            'point_size': 80,
            'jitter_strength': 0.3,
            'show_significance_letters': True,
            'show_legend': True,
            'capsize': 0.05,
            'x_label': '',
            'y_label': '',
            'title': '',
            'fontsize_axis': 12,
            'fontsize_title': 14,
            'fontsize_ticks': 10,
            'grid_alpha': 0.3,
            'show_title': True,
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
            DEFAULT_COLORS = ['#FF69B4', '#32CD32', '#FFD700', '#00BFFF', '#DA70D6', '#D8BFD8']
            colors_dict = {}
            for i, group in enumerate(self.groups):
                colors_dict[group] = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
            config['colors'] = colors_dict
            print(f"DEBUG: PlotPreviewWidget set fallback colors: {colors_dict}")
        else:
            print(f"DEBUG: PlotPreviewWidget using provided colors: {config.get('colors', {})}")
        
        try:
            # Clear and redraw
            self.ax.clear()
            
            # Markiere als Preview für optimiertes Styling
            config['_is_preview'] = True
            
            # Use the central dispatcher method (Font-Management ist jetzt integriert)
            DataVisualizer.plot_from_config(self.ax, self.groups, self.samples, config)
            
            # Force immediate redraw
            self.draw_idle()
            self.flush_events()
            
        except Exception as e:
            print(f"Error in plot update: {str(e)}")
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
