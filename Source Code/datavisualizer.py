import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import scipy.stats as stats
import string
import os
from scipy.stats import ttest_ind, mannwhitneyu
from matplotlib.ticker import ScalarFormatter, FuncFormatter
from resultsexporter import ResultsExporter
from stats_functions import get_output_path

class DataVisualizer:
    @staticmethod
    def _add_pairwise_comparisons(ax, groups, compare, pairwise_results, df=None, line_height=0.1, font_size=14):
        """
    Draws significance brackets for pairwise comparisons.
    ax: matplotlib axes
    groups: List of group names (in plot order)
    compare: Order of the groups (usually same as groups)
    pairwise_results: List of dicts with 'group1', 'group2', 'p_value', optionally 'significant'
    df: DataFrame with plot data (optional)
    line_height: Spacing between brackets
    font_size: Font size for p-values
        """
        import matplotlib.pyplot as plt
        import numpy as np
        # Settings for publication-ready bars
        LINEWIDTH = 2
        FONT_SIZE = 16
        SHORT_VERT = 0.25  # Vertical ends are now significantly shorter (as fraction of step)

        group_pos = {g: i for i, g in enumerate(compare)}
        # Find the current maximum on the y-axis
        if df is not None and 'Value' in df:
            y_max = df['Value'].max()
        else:
            y_max = None
            try:
                y_max = max([p.get_height() for p in getattr(ax, 'patches', []) if hasattr(p, 'get_height')])
            except Exception:
                y_max = 1.0
        if y_max is None or y_max == 0:
            y_max = 1.0
        # Start-Offset für die Balken
        base_height = y_max * 1.05
        step = y_max * line_height
        current_height = base_height
        for comp in pairwise_results:
            g1 = comp.get('group1')
            g2 = comp.get('group2')
            p = comp.get('p_value')
            if g1 not in group_pos or g2 not in group_pos:
                continue
            x1 = group_pos[g1]
            x2 = group_pos[g2]
            # Vertical ends are now significantly shorter
            vert = step * SHORT_VERT
            # Left vertical line
            ax.plot([x1, x1], [current_height, current_height+vert], color='#222', linewidth=LINEWIDTH)
            # Right vertical line
            ax.plot([x2, x2], [current_height, current_height+vert], color='#222', linewidth=LINEWIDTH)
            # Horizontal line at top
            ax.plot([x1, x2], [current_height+vert, current_height+vert], color='#222', linewidth=LINEWIDTH)
            # p-value or asterisks
            if p is not None:
                if p < 0.001:
                    text = '***'
                elif p < 0.01:
                    text = '**'
                elif p < 0.05:
                    text = '*'
                else:
                    text = 'n.s.'
                # Place asterisks directly above the horizontal line
                ax.text((x1+x2)/2, current_height+vert, text, ha='center', va='bottom', fontsize=FONT_SIZE, color='#222', fontweight='bold', fontname='Arial')
            current_height += step * 1.2
    """Advanced data visualization class with extensive customization options"""
    
    # Default colors for plots
    DEFAULT_COLORS = ['#3357FF', '#FF5733', '#33FF57', '#F033FF', '#FF3366', '#33FFEC']
    
    @staticmethod
    def _auto_adjust_figure_size(width, height, groups, plot_type='Bar'):
        """Automatic size adjustment based on number of groups"""
        num_groups = len(groups)
        
        if num_groups <= 2:
            return width, height
            
        if plot_type == 'Raincloud':
            # Raincloud is horizontal: height must scale with number of groups
            adjusted_width = max(width, 8 + num_groups * 0.5)
            adjusted_height = max(height, 4 + num_groups * 1.2)
            # Extra scaling for many groups
            if num_groups > 6:
                adjusted_height += (num_groups - 6) * 0.8
        else:
            # Bar, Box, Violin are vertical: width must scale with number of groups
            adjusted_width = max(width, 6 + num_groups * 1.0)
            adjusted_height = max(height, 6)
            # Extra scaling for many groups
            if num_groups > 6:
                adjusted_width += (num_groups - 6) * 0.5
                
        return adjusted_width, adjusted_height

    @staticmethod
    def _apply_seaborn_settings(seaborn_context=None, seaborn_palette=None, use_seaborn_styling=True):
        """Apply Seaborn style context and palette settings"""
        if not use_seaborn_styling:
            return
            
        try:
            import seaborn as sns
            
            # Apply style context if specified
            if seaborn_context:
                sns.set_context(seaborn_context)
            
            # Set default style for better aesthetics
            sns.set_style("whitegrid")
            
            # Apply palette if specified
            if seaborn_palette:
                sns.set_palette(seaborn_palette)
                
        except ImportError:
            # Seaborn not available, skip styling
            pass

    @staticmethod
    def plot_bar(groups, samples, 
                 # Basic plot settings
                 width=8, height=6, dpi=300,
                 # Styling and theme
                 theme='default', colors=None, hatches=None, 
                 color_palette='Greys', alpha=0.8,
                 # Bar customization
                 bar_width=0.8, bar_edge_color='black', bar_edge_width=0.5,
                 capsize=0.05, error_type="sd", show_error_bars=True,
                 # Data points
                 show_points=True, point_style='jitter', max_points_per_group=None,
                 point_size=80, point_alpha=0.8, point_edge_width=0.5,
                 jitter_strength=0.3, strip_dodge=False,
                 # Statistical annotations
                 show_significance_letters=True,
                 significance_height_offset=0.05, comparison_line_height=0.1,
                 significance_font_size=12, comparison_font_size=14,
                 # Axes and labels
                 x_label=None, y_label=None, title=None,
                 x_label_size=12, y_label_size=12, title_size=14,
                 tick_label_size=10, rotate_x_labels=0,
                 # Axis formatting
                 y_axis_format='auto', y_limits=None, x_limits=None,
                 grid_style='none', grid_alpha=0.3,
                 # Legend
                 show_legend=True, legend_position='upper right',
                 legend_bbox=(1.15, 1), legend_fontsize=9,
                 legend_title="Samples", legend_title_size=12,
                 # Advanced styling
                 spine_style='minimal', background_color='white',
                 figure_face_color='white',
                 # Seaborn styling
                 seaborn_context=None, seaborn_palette=None, use_seaborn_styling=True,
                 # Output options
                 save_plot=True, file_formats=['png', 'svg'], 
                 file_name=None, group_order=None,
                 # Statistical data
                 compare=None, test_recommendation="parametric",
                 pairwise_results=None,
                 posthoc_method=None,
                 error_style="caps",     # "caps" (Whisker-Caps), "line" (nur Strich)
                 # Advanced customization
                 custom_annotations=None, watermark=None,
                 subplot_margins=None, tight_layout=True,
                 # Legend colors
                 legend_colors=None,
                 # Optional ax parameter for direct plotting
                 ax=None):
      
        # Apply theme (simplified - use default colors if none provided)
        if colors is None:
            colors = DataVisualizer.DEFAULT_COLORS
        
        # Apply Seaborn styling
        DataVisualizer._apply_seaborn_settings(seaborn_context, seaborn_palette, use_seaborn_styling)
        
        # Prepare data and groups
        if group_order is not None:
            groups = [g for g in group_order if g in samples and len(samples[g]) > 0]
        else:
            groups = [g for g in groups if len(samples.get(g, [])) > 0]
        
        # AUTOMATISCHE GRÖßENANPASSUNG
        width, height = DataVisualizer._auto_adjust_figure_size(width, height, groups, 'Bar')
        if group_order is not None:
            groups = [g for g in group_order if g in samples and len(samples[g]) > 0]
        else:
            groups = [g for g in groups if len(samples.get(g, [])) > 0]
        
        # Color and hatch preparation
        if colors is None:
            if color_palette:
                colors = sns.color_palette(color_palette, len(groups))
            else:
                colors = DataVisualizer.DEFAULT_COLORS
        
        colors = DataVisualizer._extend_list(colors, len(groups))
        hatches = DataVisualizer._extend_list(hatches or [''] * len(groups), len(groups))
        
        # Create figure nur wenn kein ax übergeben wurde
        if ax is None:
            fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
            fig.patch.set_facecolor(figure_face_color)
            ax.set_facecolor(background_color)
            created_fig = True
        else:
            fig = ax.figure
            created_fig = False
        
        # Prepare data for plotting
        plot_data = DataVisualizer._prepare_plot_data(groups, samples, colors)
        df = pd.DataFrame(plot_data)
        
        # Fehler-Balken-Style vorbereiten
        if error_style == "caps":
            capsize_val = capsize   # wie bisher, zeigt whisker-caps
        elif error_style == "line":
            capsize_val = 0         # keine Caps, nur durchgezogene Linie
        else:
            raise ValueError(f"Unbekannter error_style: {error_style}")

        # Create bar plot
        if show_error_bars:
            bars = sns.barplot(
                x='Group', y='Value', data=df, ax=ax,
                errorbar=error_type, palette=colors, 
                capsize=capsize_val, alpha=alpha,
                order=groups, width=bar_width,
                edgecolor=bar_edge_color, linewidth=bar_edge_width
            )
        else:
            bars = sns.barplot(
                x='Group', y='Value', data=df, ax=ax,
                errorbar=None, palette=colors,
                order=groups, width=bar_width,
                edgecolor=bar_edge_color, linewidth=bar_edge_width,
                alpha=alpha
            )
        
        # Apply hatches
        if any(h for h in hatches):
            for i, patch in enumerate(bars.patches):
                if i < len(hatches) and hatches[i]:
                    patch.set_hatch(hatches[i])
        
        # Add data points
        if show_points:
            DataVisualizer._add_data_points(
                ax, groups, samples, point_style, point_size, 
                point_alpha, jitter_strength, max_points_per_group,
                point_edge_width, strip_dodge
            )
        
        # Format axes
        DataVisualizer._format_axes(
            ax, y_axis_format, y_limits, x_limits, 
            grid_style, grid_alpha, spine_style
        )
        
        # Add labels and title
        DataVisualizer._add_labels(
            ax, x_label, y_label, title,
            x_label_size, y_label_size, title_size,
            tick_label_size, rotate_x_labels
        )
        
        # Entscheide, ob Buchstaben oder Bars angezeigt werden sollen basierend auf Post-hoc Test Typ
        show_letters = True  # Standard: Buchstaben zeigen
        show_bars = False
        
        # Prüfe zuerst, ob wir den Post-hoc Test Typ aus pairwise_results ermitteln können
        if pairwise_results is not None and len(pairwise_results) > 0:
            # Schaue auf den ersten Vergleich, um den Test-Typ zu bestimmen
            first_comparison = pairwise_results[0]
            test_name = first_comparison.get('test', '').lower()
            
            # Nur bei Pairwise T-Tests oder Mann-Whitney Tests Bars zeigen
            if any(keyword in test_name for keyword in [
                'pairwise t-test', 'pairwise t test', 'paired t-test', 'independent t-test',
                'pairwise mann-whitney', 'pairwise mann whitney', 'mann-whitney',
                'wilcoxon', 'pairwise wilcoxon'
            ]):
                show_letters = False
                show_bars = True
                print(f"DEBUG: Using bars for test: {test_name}")
            else:
                show_letters = True
                show_bars = False
                print(f"DEBUG: Using letters for test: {test_name}")
        
        # Fallback auf posthoc_method falls pairwise_results test nicht erkannt wurde
        elif posthoc_method is not None and isinstance(posthoc_method, str):
            method_lower = posthoc_method.lower()
            if any(keyword in method_lower for keyword in [
                "pairwise t-test", "pairwise t test", "pairwise mann-whitney", 
                "pairwise mann whitney", "pairwise_mannwhitney", "pairwise_ttest"
            ]):
                show_letters = False
                show_bars = True
                print(f"DEBUG: Using bars for posthoc_method: {posthoc_method}")
            else:
                show_letters = True
                show_bars = False
                print(f"DEBUG: Using letters for posthoc_method: {posthoc_method}")
        else:
            # Kein Post-hoc Test: Standard Letters
            show_letters = True
            show_bars = False
            print("DEBUG: No post-hoc method detected, using letters")
        if show_letters and show_significance_letters:
            DataVisualizer._add_significance_letters(
                ax, df, groups, samples, test_recommendation,
                significance_height_offset, significance_font_size,
                error_type, pairwise_results=pairwise_results
            )
        if show_bars and pairwise_results:
            DataVisualizer._add_pairwise_comparisons(
                ax, groups, compare if compare else groups, pairwise_results, df,
                comparison_line_height, comparison_font_size
            )
        
        # Add legend
        if show_legend:
            if legend_colors is not None:
                # Use the provided legend colors
                colors_to_use = legend_colors
            else:
                # Use the colors from the plot
                colors_to_use = colors
                
            # Create patches for the legend with the correct colors
            legend_patches = []
            for i, group in enumerate(groups):
                color = colors_to_use.get(group, colors[i % len(colors)]) if isinstance(colors_to_use, dict) else colors[i % len(colors)]
                patch = mpatches.Patch(color=color, label=str(group), alpha=alpha)
                legend_patches.append(patch)
                
            ax.legend(
                handles=legend_patches,
                loc=legend_position,
                bbox_to_anchor=legend_bbox,
                fontsize=legend_fontsize,
                title=legend_title,
                title_fontsize=legend_title_size,
                frameon=False
            )
        
        # Apply final formatting including tick control
        DataVisualizer._apply_final_formatting(
            ax, groups, 'Bar', tight_layout, created_fig, 
            subplot_margins, custom_annotations, watermark
        )
        
        # Save plot nur wenn neue Figure erstellt wurde
        if save_plot and created_fig:
            DataVisualizer._save_plot(fig, file_name, groups, file_formats, dpi)
        
        return fig, ax if created_fig else ax
    
    @staticmethod
    def plot_violin(
        groups, samples,
        width=8, height=6, dpi=300,
        theme='default', colors=None, hatches=None, color_palette='Greys', alpha=0.8,
        violin_width=0.8, edge_color='black', edge_width=0.5,
        show_points=True, point_style='jitter', max_points_per_group=None,
        point_size=80, point_alpha=0.8, point_edge_width=0.5,
        jitter_strength=0.3, strip_dodge=False,
        show_significance_letters=True, significance_height_offset=0.05, significance_font_size=12,
        x_label=None, y_label=None, title=None,
        x_label_size=12, y_label_size=12, title_size=14,
        tick_label_size=10, rotate_x_labels=0,
        y_axis_format='auto', y_limits=None, x_limits=None,
        grid_style='none', grid_alpha=0.3,
        show_legend=True, legend_position='upper right',
        legend_bbox=(1.15, 1), legend_fontsize=9,
        legend_title="Samples", legend_title_size=12,
        spine_style='minimal', background_color='white',
        figure_face_color='white',
        # Seaborn styling
        seaborn_context=None, seaborn_palette=None, use_seaborn_styling=True,
        save_plot=True, file_formats=['png', 'svg'],
        file_name=None, group_order=None,
        test_recommendation="parametric",
        pairwise_results=None,
        posthoc_method=None,
        custom_annotations=None, watermark=None,
        subplot_margins=None, tight_layout=True,
        legend_colors=None,
        # Optional ax parameter for direct plotting
        ax=None
    ):

        # Apply theme (simplified - use default colors if none provided)
        if colors is None:
            colors = DataVisualizer.DEFAULT_COLORS
        
        # Apply Seaborn styling
        DataVisualizer._apply_seaborn_settings(seaborn_context, seaborn_palette, use_seaborn_styling)
        
        if colors is None:
            colors = sns.color_palette(color_palette, len(groups))
        colors = DataVisualizer._extend_list(colors, len(groups))

        if group_order is not None:
            groups = [g for g in group_order if g in samples and len(samples[g]) > 0]
        else:
            groups = [g for g in groups if len(samples.get(g, [])) > 0]

        # AUTOMATISCHE GRÖßENANPASSUNG
        width, height = DataVisualizer._auto_adjust_figure_size(width, height, groups, 'Violin')

        plot_data = DataVisualizer._prepare_plot_data(groups, samples, colors)
        df = pd.DataFrame(plot_data)

        # Create figure nur wenn kein ax übergeben wurde
        if ax is None:
            fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
            fig.patch.set_facecolor(figure_face_color)
            ax.set_facecolor(background_color)
            created_fig = True
        else:
            fig = ax.figure
            created_fig = False

        sns.violinplot(
            x='Group', y='Value', data=df, ax=ax,
            palette=colors, order=groups, width=violin_width,
            linewidth=edge_width, edgecolor=edge_color, alpha=alpha
        )
        
        # Apply hatches to violins
        if any(h for h in hatches):
            # For violin plots, we need to access the collections (violin bodies)
            for i, collection in enumerate(ax.collections):
                if i < len(hatches) and hatches[i] and hasattr(collection, 'set_hatch'):
                    collection.set_hatch(hatches[i])

        if show_points:
            DataVisualizer._add_data_points(
                ax, groups, samples, point_style, point_size,
                point_alpha, jitter_strength, max_points_per_group,
                point_edge_width, strip_dodge
            )

        DataVisualizer._format_axes(
            ax, y_axis_format, y_limits, x_limits,
            grid_style, grid_alpha, spine_style
        )
        DataVisualizer._add_labels(
            ax, x_label, y_label, title,
            x_label_size, y_label_size, title_size,
            tick_label_size, rotate_x_labels
        )
        # Entscheide, ob Buchstaben oder Bars angezeigt werden sollen basierend auf Post-hoc Test Typ (VIOLIN)
        show_letters = True  # Standard: Buchstaben zeigen
        show_bars = False
        
        # Prüfe zuerst, ob wir den Post-hoc Test Typ aus pairwise_results ermitteln können
        if pairwise_results is not None and len(pairwise_results) > 0:
            # Schaue auf den ersten Vergleich, um den Test-Typ zu bestimmen
            first_comparison = pairwise_results[0]
            test_name = first_comparison.get('test', '').lower()
            
            # Nur bei Pairwise T-Tests oder Mann-Whitney Tests Bars zeigen
            if any(keyword in test_name for keyword in [
                'pairwise t-test', 'pairwise t test', 'paired t-test', 'independent t-test',
                'pairwise mann-whitney', 'pairwise mann whitney', 'mann-whitney',
                'wilcoxon', 'pairwise wilcoxon'
            ]):
                show_letters = False
                show_bars = True
                print(f"DEBUG: Using bars for test: {test_name}")
            else:
                show_letters = True
                show_bars = False
                print(f"DEBUG: Using letters for test: {test_name}")
        
        # Fallback auf posthoc_method falls pairwise_results test nicht erkannt wurde
        elif posthoc_method is not None and isinstance(posthoc_method, str):
            method_lower = posthoc_method.lower()
            if any(keyword in method_lower for keyword in [
                "pairwise t-test", "pairwise t test", "pairwise mann-whitney", 
                "pairwise mann whitney", "pairwise_mannwhitney", "pairwise_ttest"
            ]):
                show_letters = False
                show_bars = True
                print(f"DEBUG: Using bars for posthoc_method: {posthoc_method}")
            else:
                show_letters = True
                show_bars = False
                print(f"DEBUG: Using letters for posthoc_method: {posthoc_method}")
        else:
            # Kein Post-hoc Test: Standard Letters
            show_letters = True
            show_bars = False
            print("DEBUG: No post-hoc method detected, using letters")
        if show_letters and show_significance_letters:
            DataVisualizer._add_significance_letters(
                ax, df, groups, samples, test_recommendation,
                significance_height_offset, significance_font_size,
                "sd", pairwise_results=pairwise_results
            )
        if show_bars and pairwise_results:
            DataVisualizer._add_pairwise_comparisons(
                ax, groups, groups, pairwise_results, df,
                significance_height_offset, significance_font_size
            )
        if show_legend and show_points:
            if legend_colors is not None:
                # Use the provided legend colors
                colors_to_use = legend_colors
            else:
                # Use the colors from the plot
                colors_to_use = colors
                
            # Create patches for the legend with the correct colors
            legend_patches = []
            for i, group in enumerate(groups):
                color = colors_to_use.get(group, colors[i % len(colors)]) if isinstance(colors_to_use, dict) else colors[i % len(colors)]
                patch = mpatches.Patch(color=color, label=str(group), alpha=alpha)
                legend_patches.append(patch)
                
            ax.legend(
                handles=legend_patches,
                loc=legend_position,
                bbox_to_anchor=legend_bbox,
                fontsize=legend_fontsize,
                title=legend_title,
                title_fontsize=legend_title_size,
                frameon=False
            )
        if custom_annotations:
            DataVisualizer._add_custom_annotations(ax, custom_annotations)
        if watermark:
            DataVisualizer._add_watermark(fig, watermark)
        # Adjust layout nur wenn neue Figure erstellt wurde
        if tight_layout and created_fig:
            if subplot_margins:
                plt.subplots_adjust(**subplot_margins)
            else:
                fig.tight_layout()
        
        # Save plot nur wenn neue Figure erstellt wurde
        if save_plot and created_fig:
            DataVisualizer._save_plot(fig, file_name, groups, file_formats, dpi)
        
        return fig, ax if created_fig else ax

    @staticmethod
    def plot_box(
        groups, samples,
        width=8, height=6, dpi=300,
        theme='default', colors=None, hatches=None, color_palette='Greys', alpha=0.8,
        box_width=0.8, edge_color='black', edge_width=0.5,
        show_points=True, point_style='jitter', max_points_per_group=None,
        point_size=80, point_alpha=0.8, point_edge_width=0.5,
        jitter_strength=0.3, strip_dodge=False,
        show_error_bars=True, error_type="sd", capsize=0.05,
        show_significance_letters=True, significance_height_offset=0.05, significance_font_size=12,
        x_label=None, y_label=None, title=None,
        x_label_size=12, y_label_size=12, title_size=14,
        tick_label_size=10, rotate_x_labels=0,
        y_axis_format='auto', y_limits=None, x_limits=None,
        grid_style='none', grid_alpha=0.3,
        show_legend=True, legend_position='upper right',
        legend_bbox=(1.15, 1), legend_fontsize=9,
        legend_title="Samples", legend_title_size=12,
        spine_style='minimal', background_color='white',
        figure_face_color='white', error_style="caps", 
        # Seaborn styling
        seaborn_context=None, seaborn_palette=None, use_seaborn_styling=True,
        save_plot=True, file_formats=['png', 'svg'],
        file_name=None, group_order=None,
        test_recommendation="parametric",
        pairwise_results=None,
        posthoc_method=None,
        custom_annotations=None, watermark=None,
        subplot_margins=None, tight_layout=True,
        legend_colors=None,
        # Optional ax parameter for direct plotting
        ax=None
    ):

        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np

        # Apply theme (simplified - use default colors if none provided)
        if colors is None:
            colors = DataVisualizer.DEFAULT_COLORS
        
        # Apply Seaborn styling
        DataVisualizer._apply_seaborn_settings(seaborn_context, seaborn_palette, use_seaborn_styling)
        
        if colors is None:
            colors = sns.color_palette(color_palette, len(groups))
        colors = DataVisualizer._extend_list(colors, len(groups))

        if group_order is not None:
            groups = [g for g in group_order if g in samples and len(samples[g]) > 0]
        else:
            groups = [g for g in groups if len(samples.get(g, [])) > 0]

        # AUTOMATISCHE GRÖßENANPASSUNG
        width, height = DataVisualizer._auto_adjust_figure_size(width, height, groups, 'Box')

        plot_data = DataVisualizer._prepare_plot_data(groups, samples, colors)
        df = pd.DataFrame(plot_data)

        # Create figure nur wenn kein ax übergeben wurde
        if ax is None:
            fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
            fig.patch.set_facecolor(figure_face_color)
            ax.set_facecolor(background_color)
            created_fig = True
        else:
            fig = ax.figure
            created_fig = False

        sns.boxplot(
            x='Group', y='Value', data=df, ax=ax,
            palette=colors, order=groups, width=box_width,
            linewidth=edge_width, fliersize=0, boxprops=dict(alpha=alpha)
        )
        
        # Apply hatches to boxes
        if any(h for h in hatches):
            for i, patch in enumerate(ax.patches):
                if i < len(hatches) and hatches[i]:
                    patch.set_hatch(hatches[i])
        
        if show_error_bars:
            import numpy as np
            means = [np.mean(samples[g]) for g in groups]
            if error_type == "sd":
                errs = [np.std(samples[g], ddof=1) for g in groups]
            else:  # "se"
                errs = [np.std(samples[g], ddof=1)/np.sqrt(len(samples[g])) for g in groups]
            xs = range(len(groups))
            eb_caps = capsize if error_style == "caps" else 0
            ax.errorbar(xs, means, yerr=errs,
                        fmt='none',
                        color='black',
                        elinewidth=edge_width,
                        capsize=eb_caps)

        if show_points:
            DataVisualizer._add_data_points(
                ax, groups, samples, point_style, point_size,
                point_alpha, jitter_strength, max_points_per_group,
                point_edge_width, strip_dodge
            )

        DataVisualizer._format_axes(
            ax, y_axis_format, y_limits, x_limits,
            grid_style, grid_alpha, spine_style
        )
        DataVisualizer._add_labels(
            ax, x_label, y_label, title,
            x_label_size, y_label_size, title_size,
            tick_label_size, rotate_x_labels
        )
        # Entscheide, ob Buchstaben oder Bars angezeigt werden sollen basierend auf Post-hoc Test Typ (BOX)
        show_letters = True  # Standard: Buchstaben zeigen
        show_bars = False
        
        # Prüfe zuerst, ob wir den Post-hoc Test Typ aus pairwise_results ermitteln können
        if pairwise_results is not None and len(pairwise_results) > 0:
            # Schaue auf den ersten Vergleich, um den Test-Typ zu bestimmen
            first_comparison = pairwise_results[0]
            test_name = first_comparison.get('test', '').lower()
            
            # Nur bei Pairwise T-Tests oder Mann-Whitney Tests Bars zeigen
            if any(keyword in test_name for keyword in [
                'pairwise t-test', 'pairwise t test', 'paired t-test', 'independent t-test',
                'pairwise mann-whitney', 'pairwise mann whitney', 'mann-whitney',
                'wilcoxon', 'pairwise wilcoxon'
            ]):
                show_letters = False
                show_bars = True
                print(f"DEBUG: Using bars for test: {test_name}")
            else:
                show_letters = True
                show_bars = False
                print(f"DEBUG: Using letters for test: {test_name}")
        
        # Fallback auf posthoc_method falls pairwise_results test nicht erkannt wurde
        elif posthoc_method is not None and isinstance(posthoc_method, str):
            method_lower = posthoc_method.lower()
            if any(keyword in method_lower for keyword in [
                "pairwise t-test", "pairwise t test", "pairwise mann-whitney", 
                "pairwise mann whitney", "pairwise_mannwhitney", "pairwise_ttest"
            ]):
                show_letters = False
                show_bars = True
                print(f"DEBUG: Using bars for posthoc_method: {posthoc_method}")
            else:
                show_letters = True
                show_bars = False
                print(f"DEBUG: Using letters for posthoc_method: {posthoc_method}")
        else:
            # Kein Post-hoc Test: Standard Letters
            show_letters = True
            show_bars = False
            print("DEBUG: No post-hoc method detected, using letters")
        if show_letters and show_significance_letters:
            DataVisualizer._add_significance_letters(
                ax, df, groups, samples, test_recommendation,
                significance_height_offset, significance_font_size,
                "sd", pairwise_results=pairwise_results
            )
        if show_bars and pairwise_results:
            DataVisualizer._add_pairwise_comparisons(
                ax, groups, groups, pairwise_results, df,
                significance_height_offset, significance_font_size
            )
        if show_legend and show_points:
            if legend_colors is not None:
                # Use the provided legend colors
                colors_to_use = legend_colors
            else:
                # Use the colors from the plot
                colors_to_use = colors
                
            # Create patches for the legend with the correct colors
            legend_patches = []
            for i, group in enumerate(groups):
                color = colors_to_use.get(group, colors[i % len(colors)]) if isinstance(colors_to_use, dict) else colors[i % len(colors)]
                patch = mpatches.Patch(color=color, label=str(group), alpha=alpha)
                legend_patches.append(patch)
                
            ax.legend(
                handles=legend_patches,
                loc=legend_position,
                bbox_to_anchor=legend_bbox,
                fontsize=legend_fontsize,
                title=legend_title,
                title_fontsize=legend_title_size,
                frameon=False
            )
        if custom_annotations:
            DataVisualizer._add_custom_annotations(ax, custom_annotations)
        if watermark:
            DataVisualizer._add_watermark(fig, watermark)
        
        # Adjust layout nur wenn neue Figure erstellt wurde
        if tight_layout and created_fig:
            if subplot_margins:
                plt.subplots_adjust(**subplot_margins)
            else:
                fig.tight_layout()
        
        # Save plot nur wenn neue Figure erstellt wurde
        if save_plot and created_fig:
            DataVisualizer._save_plot(fig, file_name, groups, file_formats, dpi)
        
        return fig, ax if created_fig else ax

    @staticmethod
    def plot_raincloud(
        groups, samples,
        width=8, height=6, dpi=300,
        theme='default', colors=None, hatches=None, color_palette='Greys', alpha=0.8,
        violin_width=0.8, box_width=0.2, edge_color='black', edge_width=0.5,
        show_points=True, point_style='jitter', max_points_per_group=None,
        point_size=80, point_alpha=0.8, point_edge_width=0.5,
        jitter_strength=0.3, strip_dodge=False,
        show_significance_letters=True, significance_height_offset=0.05, significance_font_size=12,
        x_label=None, y_label=None, title=None,
        x_label_size=12, y_label_size=12, title_size=14,
        tick_label_size=10, rotate_x_labels=0,
        y_axis_format='auto', y_limits=None, x_limits=None,
        grid_style='none', grid_alpha=0.3,
        show_legend=True, legend_position='center right',
        legend_bbox=(1.08, 0.7), legend_fontsize=9,
        legend_title="Samples", legend_title_size=12,
        spine_style='minimal', background_color='white',
        figure_face_color='white',
        save_plot=True, file_formats=['png', 'svg'],
        file_name=None, group_order=None,
        test_recommendation="parametric",
        pairwise_results=None,
        posthoc_method=None,
        custom_annotations=None, watermark=None,
        subplot_margins=None, tight_layout=True,
        # Appearance options
        font_main=None, font_axis=None, fontsize_title=None, fontsize_axis=None,
        fontsize_ticks=None, fontsize_groupnames=None, legend_colors=None, group_spacing=0.5,
        # Raincloud-specific color options
        violin_colors=None, box_colors=None, point_colors=None,
        # Spacing options
        point_offset=0.2, point_jitter=0.05,
        # Line thickness options
        frame_thickness=0.7, axis_thickness=0.5,
        # Seaborn styling
        seaborn_context=None, seaborn_palette=None, use_seaborn_styling=True,
        # Optional ax parameter for direct plotting
        ax=None
    ):
        """Creates a raincloud plot (violin + box + points) with half violins above boxplots and data points below."""
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        import matplotlib.patches as mpatches
        import scipy.stats as stats

        # Apply font settings if provided
        if fontsize_title is not None:
            title_size = fontsize_title
        if fontsize_axis is not None:
            x_label_size = fontsize_axis
            y_label_size = fontsize_axis
        if fontsize_ticks is not None:
            tick_label_size = fontsize_ticks
        if fontsize_groupnames is not None:
            rotate_x_labels = 0  # Override rotation if group names font size is specified

        # Apply theme settings (simplified - use default colors if none provided)
        if colors is None:
            colors = DataVisualizer.DEFAULT_COLORS
        
        # Apply Seaborn styling
        DataVisualizer._apply_seaborn_settings(seaborn_context, seaborn_palette, use_seaborn_styling)
        
        if colors is None:
            colors = sns.color_palette(color_palette, len(groups))
        colors = DataVisualizer._extend_list(colors, len(groups))

        # Filter groups
        if group_order is not None:
            groups = [g for g in group_order if g in samples and len(samples[g]) > 0]
        else:
            groups = [g for g in groups if len(samples.get(g, [])) > 0]

        # AUTOMATISCHE GRÖßENANPASSUNG
        width, height = DataVisualizer._auto_adjust_figure_size(width, height, groups, 'Raincloud')

        # Create figure nur wenn kein ax übergeben wurde
        if ax is None:
            fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
            fig.patch.set_facecolor(figure_face_color)
            ax.set_facecolor(background_color)
            created_fig = True
        else:
            fig = ax.figure
            created_fig = False
        
        # Use the CORRECT Raincloud implementation from Live Preview
        ax.clear()
        
        # Prepare data like in Live Preview
        data_x = [np.array(data) for data in [samples[group] for group in groups]]
        n_groups = len(groups)
        
        # Einheitliche Farblogik: Wenn keine spezifischen Farben für Violin/Box/Points übergeben wurden,
        # werden die allgemeinen colors verwendet (wie in Bar/Box/Violin)
        if violin_colors is None:
            violin_colors = colors
        if box_colors is None:
            box_colors = colors
        if point_colors is None:
            point_colors = colors

        # Falls dict, auf Gruppen abbilden
        if isinstance(violin_colors, dict):
            violin_colors = [violin_colors.get(group, colors[i % len(colors)]) for i, group in enumerate(groups)]
        if isinstance(box_colors, dict):
            box_colors = [box_colors.get(group, colors[i % len(colors)]) for i, group in enumerate(groups)]
        if isinstance(point_colors, dict):
            point_colors = [point_colors.get(group, colors[i % len(colors)]) for i, group in enumerate(groups)]

        # Use these colors for the legend as well
        actual_legend_colors = {groups[i]: box_colors[i % len(box_colors)] for i in range(len(groups))}
        
        # Calculate explicit positions based on group_spacing
        positions = np.arange(1, n_groups+1) * group_spacing
        
        # Create horizontal raincloud plot with explicit positions
        bp = ax.boxplot(data_x, patch_artist=True, vert=False, positions=positions)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(alpha)
            
        # Violinplot with correct orientation and explicit positions
        vp = ax.violinplot(data_x, points=500, showmeans=False, showextrema=False, showmedians=False, vert=False, positions=positions)
        for idx, b in enumerate(vp['bodies']):
            m = np.mean(b.get_paths()[0].vertices[:, 0])
            # Only show upper half (half-violin) - adjust clipping to new positions
            pos = positions[idx]
            b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1], pos, pos+1)
            b.set_color(violin_colors[idx % len(violin_colors)])
            
        # Scatter points with correct positioning based on positions - improved spacing
        for idx, features in enumerate(data_x):
            y = np.full(len(features), positions[idx] - point_offset)  # Use configurable offset
            idxs = np.arange(len(y))
            out = y.astype(float)
            # Use configurable jitter strength
            out.flat[idxs] += np.random.uniform(low=-point_jitter, high=point_jitter, size=len(idxs))
            y = out
            ax.scatter(features, y, s=point_size/8, c=point_colors[idx % len(point_colors)], alpha=point_alpha)
            
        # Set up axes and labels with explicit positions
        ax.set_yticks(positions)
        ax.set_yticklabels([str(g) for g in groups], fontsize=fontsize_groupnames or tick_label_size)
        ax.set_xlabel(y_label or "Values", fontsize=y_label_size)
        ax.set_ylabel("")
        
        if title:
            ax.set_title(title, fontsize=title_size, pad=20)
            
        # Set tick parameters
        ax.tick_params(axis='x', labelsize=tick_label_size)
        ax.tick_params(axis='y', labelsize=fontsize_groupnames or tick_label_size)
        
        # Set proper limits like in Live Preview
        if data_x and any(len(d) > 0 for d in data_x):
            ax.set_xlim(left=min([min(d) for d in data_x if len(d)>0])-1, right=max([max(d) for d in data_x if len(d)>0])+1)
        # Set y-limits based on calculated positions and group_spacing
        ax.set_ylim(positions[0] - group_spacing*0.4, positions[-1] + group_spacing*0.4)
        ax.grid(False)
        
        # Make sure spines are consistent with configurable thickness
        for spine in ax.spines.values():
            spine.set_linewidth(frame_thickness)
        
        # Set axis line thickness
        ax.tick_params(axis='both', which='major', width=axis_thickness)
        ax.tick_params(axis='both', which='minor', width=axis_thickness * 0.7)
        
        # Remove y-ticks if too many groups
        if len(groups) > 10:
            ax.set_yticks([])
        
        # Grid with configurable line width
        if grid_style and grid_style != 'none':
            grid_linewidth = axis_thickness * 0.7  # Grid lines slightly thinner than axis
            if grid_style == 'major':
                ax.grid(True, axis='x', which='major', alpha=grid_alpha, linestyle='-', linewidth=grid_linewidth)
            elif grid_style == 'minor':
                ax.grid(True, axis='x', which='minor', alpha=grid_alpha, linestyle='-', linewidth=grid_linewidth)
            elif grid_style == 'both':
                ax.grid(True, axis='x', which='both', alpha=grid_alpha, linestyle='-', linewidth=grid_linewidth)
        else:
            ax.grid(False)
            
        # Spine styling
        if spine_style == 'minimal':
            sns.despine(ax=ax, top=True, right=True)
        elif spine_style == 'none':
            sns.despine(ax=ax, top=True, right=True, left=True, bottom=True)
        
        # Set axis limits - for horizontal plot, x_limits controls value range
        if x_limits:
            ax.set_xlim(x_limits)
        if y_limits:  # y_limits would control group range (rarely used)
            ax.set_ylim(y_limits)
            
        # Entscheide, ob Buchstaben oder Bars angezeigt werden sollen
        show_letters = True
        show_bars = False
        if posthoc_method is not None and isinstance(posthoc_method, str):
            method_lower = posthoc_method.lower()
            if method_lower in ["pairwise t-test", "pairwise t test", "pairwise mann-whitney", "pairwise mann whitney", "pairwise_mannwhitney", "pairwise_ttest"]:
                show_letters = False
                show_bars = True
        elif pairwise_results is not None and len(pairwise_results) > 0:
            show_letters = False
            show_bars = True
        if show_letters and show_significance_letters:
            DataVisualizer._add_significance_letters_raincloud(
                ax, groups, samples, test_recommendation,
                significance_height_offset, significance_font_size, positions, pairwise_results=pairwise_results
            )
        if show_bars and pairwise_results:
            DataVisualizer._add_pairwise_comparisons(
                ax, groups, groups, pairwise_results, None,
                significance_height_offset, significance_font_size
            )
            
        # Add legend with correct colors
        if show_legend:
            if legend_colors is not None:
                # Use the provided legend colors
                colors_to_use = legend_colors
            else:
                # Use the actual colors from the raincloud plot
                colors_to_use = actual_legend_colors
                
            # Create patches for the legend with the correct colors
            legend_patches = []
            for i, group in enumerate(groups):
                color = colors_to_use.get(group, actual_legend_colors[group]) if isinstance(colors_to_use, dict) else actual_legend_colors[group]
                patch = mpatches.Patch(color=color, label=str(group), alpha=alpha)
                legend_patches.append(patch)
                
            ax.legend(
                handles=legend_patches,
                loc=legend_position,
                bbox_to_anchor=legend_bbox,
                fontsize=legend_fontsize,
                title=legend_title,
                title_fontsize=legend_title_size,
                frameon=False
            )
        
        # Add any custom annotations
        if custom_annotations:
            DataVisualizer._add_custom_annotations(ax, custom_annotations)
            
        # Add watermark if provided
        if watermark:
            DataVisualizer._add_watermark(fig, watermark)
            
        # Layout adjustments - optimize for raincloud plots
        if tight_layout and created_fig:
            if subplot_margins:
                plt.subplots_adjust(**subplot_margins)
            else:
                # Better tight_layout for raincloud plots - less aggressive compression
                fig.tight_layout(rect=[0, 0.05, 0.85, 0.93])
                
        # Save plot if requested
        if save_plot and created_fig:
            DataVisualizer._save_plot(fig, file_name, groups, file_formats, dpi)
            
        return fig, ax if created_fig else ax
    
    # Helper methods
    @staticmethod
    def _extend_list(lst, target_length):
        """Extend a list to target length by repeating elements"""
        if not lst:
            return [''] * target_length
        
        # Ensure lst is a list or tuple - convert if it's something else
        if isinstance(lst, dict):
            # If it's a dictionary, assume we want the values
            lst = list(lst.values())
        elif not isinstance(lst, (list, tuple)):
            # If it's some other iterable, convert to list
            try:
                lst = list(lst)
            except TypeError:
                # If we can't convert it, return empty strings
                return [''] * target_length
        
        return (lst * (target_length // len(lst) + 1))[:target_length]
    
    @staticmethod
    def _prepare_plot_data(groups, samples, colors):
        """Prepare data for plotting"""
        data = []
        for i, group in enumerate(groups):
            values = samples.get(group, [])
            for val in values:
                data.append({
                    'Group': group, 
                    'Value': val, 
                    'Color': colors[i % len(colors)]
                })
        return data
    
    @staticmethod
    def _add_data_points(ax, groups, samples, style, size, alpha, 
                        jitter_strength, max_points, edge_width, dodge):
        """Add individual data points to the plot"""
        markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', '+', 'x']
        
        for i, group in enumerate(groups):
            values = samples.get(group, [])
            if max_points and len(values) > max_points:
                values = np.random.choice(values, size=max_points, replace=False)
            
            x_pos = i
            
            if style == 'jitter':
                # Custom jitter implementation
                if len(values) > 1:
                    jitter = np.random.uniform(-jitter_strength/2, jitter_strength/2, len(values))
                else:
                    jitter = [0]
                
                for j, (val, jit) in enumerate(zip(values, jitter)):
                    marker_idx = j % len(markers)
                    marker = markers[marker_idx]
                    
                    # Nur für filled markers edgecolors setzen
                    scatter_kwargs = {
                        'color': 'black',
                        'marker': marker,
                        's': size,
                        'zorder': 3,
                        'alpha': alpha
                    }
                    
                    # Nur für filled markers (nicht +, x) edgecolors setzen
                    if marker not in ['+', 'x']:
                        scatter_kwargs['edgecolors'] = 'white'
                        scatter_kwargs['linewidth'] = edge_width
                    
                    ax.scatter(x_pos + jit, val, **scatter_kwargs)
            
            elif style == 'strip':
                # Use seaborn stripplot
                df_group = pd.DataFrame({'x': [i] * len(values), 'y': values})
                sns.stripplot(data=df_group, x='x', y='y', ax=ax, 
                            size=size/10, alpha=alpha, color='black',
                            jitter=jitter_strength, dodge=dodge)
            
            elif style == 'swarm':
                # Use seaborn swarmplot
                df_group = pd.DataFrame({'x': [group] * len(values), 'y': values})
                sns.swarmplot(data=df_group, x='x', y='y', ax=ax,
                            size=size/10, alpha=alpha, color='black')
    
    @staticmethod
    def _format_axes(ax, y_format, y_limits, x_limits, grid_style, grid_alpha, spine_style):
        """Format axes according to specifications"""
        # Y-axis formatting
        if y_format == 'scientific':
            formatter = ScalarFormatter(useOffset=False, useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((0, 0))
            ax.yaxis.set_major_formatter(formatter)
        elif y_format == 'percentage':
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.1%}'))
        elif y_format == 'decimal':
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.2f}'))

        # Set limits
        if y_limits:
            ax.set_ylim(y_limits)
        if x_limits:
            ax.set_xlim(x_limits)

        # --- TICK CONTROL: Ensure ticks are always visible unless explicitly removed ---
        # Always show major ticks on both axes
        ax.xaxis.set_visible(True)
        ax.yaxis.set_visible(True)
        # Ensure tick labels are visible
        for label in ax.get_xticklabels():
            label.set_visible(True)
        for label in ax.get_yticklabels():
            label.set_visible(True)
        # Ensure ticks are drawn (for matplotlib >=3.1)
        ax.tick_params(axis='both', which='both', length=4, width=0.7, direction='out', bottom=True, top=False, left=True, right=False)

        # Make sure grid is off by default, only turn on if explicitly requested
        ax.grid(False)  # First turn off all grid lines

        # Then conditionally turn on grid if explicitly requested
        if grid_style and grid_style != 'none':
            if grid_style == 'major':
                ax.grid(True, which='major', alpha=grid_alpha, linestyle='-', linewidth=0.5)
            elif grid_style == 'minor':
                ax.grid(True, which='minor', alpha=grid_alpha, linestyle='-', linewidth=0.5)
            elif grid_style == 'both':
                ax.grid(True, which='both', alpha=grid_alpha, linestyle='-', linewidth=0.5)

        # Spine styling
        if spine_style == 'minimal':
            sns.despine(ax=ax, top=True, right=True)
        elif spine_style == 'none':
            sns.despine(ax=ax, top=True, right=True, left=True, bottom=True)
        elif spine_style == 'box':
            # Keep all spines
            pass
    
    @staticmethod
    def _add_labels(ax, x_label, y_label, title, x_size, y_size, title_size, tick_size, rotation):
        """Add and format labels"""
        if x_label:
            ax.set_xlabel(x_label, fontsize=x_size, fontweight='bold')
        if y_label:
            ax.set_ylabel(y_label, fontsize=y_size, fontweight='bold')
        if title:
            ax.set_title(title, fontsize=title_size, fontweight='bold', pad=20)
        
        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        
        if rotation != 0:
            plt.setp(ax.get_xticklabels(), rotation=rotation, ha='right')
    
    @staticmethod
    def _add_custom_annotations(ax, annotations):
        """Add custom text annotations"""
        for annotation in annotations:
            ax.annotate(
                annotation.get('text', ''),
                xy=annotation.get('xy', (0, 0)),
                xytext=annotation.get('xytext', (0, 0)),
                fontsize=annotation.get('fontsize', 10),
                color=annotation.get('color', 'black'),
                ha=annotation.get('ha', 'center'),
                va=annotation.get('va', 'center'),
                arrowprops=annotation.get('arrowprops', None)
            )

    @staticmethod
    def _add_legend(ax, samples, groups, legend_position, legend_bbox,
                    legend_fontsize, legend_title, legend_title_size):
        """
        Adds a legend to the plot.
        """
        handles, labels = ax.get_legend_handles_labels()
        if not handles:
            # Create dummy handles if none exist
            import matplotlib.patches as mpatches
            handles = [mpatches.Patch(label=str(g)) for g in groups]
            labels = [str(g) for g in groups]
        legend = ax.legend(
            handles, labels,
            loc=legend_position,
            bbox_to_anchor=legend_bbox,
            fontsize=legend_fontsize,
            title=legend_title,
            title_fontsize=legend_title_size,
            frameon=False
        )
        return legend
    
    @staticmethod
    def _add_watermark(fig, watermark_text):
        """Add watermark to the figure"""
        fig.text(0.95, 0.05, watermark_text, 
                fontsize=8, color='gray', alpha=0.5,
                ha='right', va='bottom', rotation=0)

    @staticmethod
    def _apply_final_formatting(ax, groups, plot_type, tight_layout, created_fig, 
                               subplot_margins, custom_annotations, watermark):
        """Apply final formatting including custom annotations, watermarks, and layout adjustments"""
        # Get the figure from the axes
        fig = ax.figure if hasattr(ax, 'figure') else plt.gcf()
        
        # Add custom annotations if provided
        if custom_annotations:
            DataVisualizer._add_custom_annotations(ax, custom_annotations)
        
        # Add watermark if provided
        if watermark:
            DataVisualizer._add_watermark(fig, watermark)
        
        # Adjust layout only if a new figure was created
        if tight_layout and created_fig:
            if subplot_margins:
                plt.subplots_adjust(**subplot_margins)
            else:
                fig.tight_layout()
    

    @staticmethod
    def _control_ticks_for_many_groups(ax, groups, plot_type):
        """
        Placeholder for tick control logic when there are many groups.
        Currently does nothing, but can be extended to handle tick label density, rotation, etc.
        """
        pass

    @staticmethod
    def _save_plot(fig, file_name, groups, formats, dpi):
        """Save plot in multiple formats with absolute paths"""
        print(f"DEBUG PLOT: _save_plot called")
        print(f"DEBUG PLOT: Current working directory: {os.getcwd()}")
        print(f"DEBUG PLOT: file_name = {file_name}")
        print(f"DEBUG PLOT: formats = {formats}")
        
        if file_name is None:
            file_name = "_".join(map(str, groups))
            print(f"DEBUG PLOT: Generated file_name = {file_name}")
        
        # Store original directory
        original_dir = os.getcwd()
        
        for fmt in formats:
            if fmt == 'pdf':
                pdf_path = get_output_path(file_name, "pdf")
                print(f"DEBUG: Attempting to save PDF to: {pdf_path}")
                fig.savefig(pdf_path, dpi=dpi, bbox_inches='tight', format='pdf')
                print(f"DEBUG: PDF file exists after save: {os.path.exists(pdf_path)}")
            elif fmt == 'svg':
                svg_path = get_output_path(file_name, "svg")
                print(f"DEBUG: Attempting to save SVG to: {svg_path}")
                fig.savefig(svg_path, bbox_inches='tight', format='svg')
                print(f"DEBUG: SVG file exists after save: {os.path.exists(svg_path)}")
            elif fmt == 'png':
                png_path = get_output_path(file_name, "png")
                print(f"DEBUG: Attempting to save PNG to: {png_path}")
                fig.savefig(png_path, dpi=dpi, bbox_inches='tight', format='png')
                print(f"DEBUG: PNG file exists after save: {os.path.exists(png_path)}")
        
        # Restore original directory if it changed
        if os.getcwd() != original_dir:
            os.chdir(original_dir)
            print(f"DEBUG PLOT: Restored original directory: {original_dir}")

    @staticmethod
    def _control_ticks_for_many_groups(ax, groups, plot_type):
        """
        Placeholder for tick control logic when there are many groups.
        Currently does nothing, but can be extended to handle tick label density, rotation, etc.
        """
        pass
    
    @staticmethod
    def get_significance_letters_from_posthoc(groups, pairwise_results, alpha=0.05, sweep=True, sort_by=None):
        """
        Generate compact letter display (CLD) using Piepho's algorithm.
        Groups sharing a letter are not significantly different.

        Parameters:
        -----------
        groups : list[str]
            List of group names
        pairwise_results : list[dict]
            List of dictionaries containing post-hoc comparison results
            Each dict should have: 'group1', 'group2', 'p_value'
        alpha : float
            Significance level (default: 0.05)
        sweep : bool
            If True, perform sweeping to remove redundant letter columns
        sort_by : dict
            Dictionary mapping group to value for ordering (e.g., means)

        Returns:
        --------
        dict[str, str]
            Dictionary with groups as keys and significance letters as values
        """
        import string
        import numpy as np
        
        # Handle edge cases
        if len(groups) <= 1:
            return {groups[0]: 'a'} if groups else {}
        
        # Check if this is a Dunnett test (all comparisons are vs control)
        is_dunnett = False
        control_group = None
        if pairwise_results:
            # Check if all comparisons share one common group (control)
            all_groups_in_comparisons = set()
            for comp in pairwise_results:
                all_groups_in_comparisons.add(comp.get('group1'))
                all_groups_in_comparisons.add(comp.get('group2'))
            
            # Find potential control group (appears in all comparisons)
            for group in groups:
                if group in all_groups_in_comparisons:
                    appears_in_all = True
                    for comp in pairwise_results:
                        if group not in [comp.get('group1'), comp.get('group2')]:
                            appears_in_all = False
                            break
                    if appears_in_all:
                        control_group = group
                        is_dunnett = True
                        break
        
        print(f"DEBUG: is_dunnett = {is_dunnett}, control_group = {control_group}")
        
        # Special handling for Dunnett test
        if is_dunnett and control_group:
            letters = {control_group: 'a'}
            treatment_groups = [g for g in groups if g != control_group]

            # Für jede Behandlungsgruppe einmal p-Wert gegen Kontrolle holen
            for grp in treatment_groups:
                # suchen, ob es einen entsprechenden Vergleichseintrag gibt
                comp = next(
                    (c for c in pairwise_results
                    if {c['group1'], c['group2']} == {grp, control_group}),
                    None
                )
                p_val = comp.get('p_value', 1.0) if comp else 1.0

                # signifikant? b, sonst a
                letters[grp] = 'b' if p_val < alpha else 'a'

            return letters
        
        # Original algorithm for full pairwise comparisons
        # Prepare index mapping and matrix of non-significance
        n = len(groups)
        idx = {g: i for i, g in enumerate(groups)}
        not_diff = np.eye(n, dtype=bool)
        
        # Process pairwise results
        for comp in pairwise_results:
            g1, g2, p = comp.get('group1'), comp.get('group2'), comp.get('p_value')
            if p is not None and g1 in idx and g2 in idx and p >= alpha:
                i, j = idx[g1], idx[g2]
                not_diff[i, j] = not_diff[j, i] = True

        # Optional: sort groups (as in multcompView) to prioritize letter assignment
        order = list(range(n))
        if sort_by:
            order = sorted(order, key=lambda i: sort_by.get(groups[i], 0), reverse=True)
        
        # Build initial columns per Piepho (insert-and-absorb)
        cols = []
        for i in order:
            col = np.zeros(n, dtype=bool)
            col[i] = True
            for j in order:
                if not_diff[i, j]:
                    col[j] = True
            # Absorption: merge if superset
            merged = False
            for existing in cols:
                if np.all(existing <= col):
                    existing[:] = col
                    merged = True
                    break
            if not merged:
                cols.append(col)

        # Optional sweeping: remove redundant columns
        if sweep:
            cols = [col for col in cols if not any(not np.array_equal(col, other) and np.all(col <= other) for other in cols)]

        # Assign letters
        letters = {g: '' for g in groups}
        alphabet = string.ascii_lowercase
        for idx_col, col in enumerate(cols):
            letter = alphabet[idx_col] if idx_col < len(alphabet) else (
                alphabet[idx_col//26 - 1] + alphabet[idx_col % 26])
            for i, included in enumerate(col):
                if included:
                    letters[groups[i]] += letter

        return letters

    @staticmethod
    def get_significance_letters(samples, groups, test_recommendation="parametric", alpha=0.05):
        """
        Calculate significance letters for groups based on statistical comparisons.
        Groups that share a letter are not significantly different.

        Parameters:
        -----------
        samples : dict
            Dictionary with group names as keys and measurement values as lists
        groups : list
            List of groups to compare
        test_recommendation : str
            Type of test to perform ("parametric" or "non_parametric")
        alpha : float
            Significance level

        Returns:
        --------
        dict
            Dictionary with groups as keys and significance letters as values
        """
        import string
        from scipy.stats import ttest_ind, mannwhitneyu

        # Initialize all groups with 'a'
        letters = {group: 'a' for group in groups}

        # If we have only one group, return immediately
        if len(groups) <= 1:
            return letters

        # Create matrix of p-values
        p_values = {}
        for i, group1 in enumerate(groups):
            for j, group2 in enumerate(groups):
                if i < j:  # Only compute once per pair
                    if test_recommendation == "parametric":
                        _, p_val = ttest_ind(samples[group1], samples[group2])
                    else:
                        _, p_val = mannwhitneyu(samples[group1], samples[group2], alternative='two-sided')
                    p_values[(group1, group2)] = p_val

        # Assign letters based on significant differences
        available_letters = list(string.ascii_lowercase)
        current_letter_idx = 0

        # Start assigning from the first group
        for i, group1 in enumerate(groups):
            # If this is the first group or it's significantly different from all previous,
            # we might need a new letter
            needs_new_letter = True

            # Check all previous groups
            for j, group2 in enumerate(groups[:i]):
                # If current group is NOT significantly different from a previous group
                pair = (group2, group1) if (group2, group1) in p_values else (group1, group2)
                if p_values[pair] >= alpha:  # Not significant
                    # Use the same letter as that group
                    if letters[group2] not in letters[group1]:
                        letters[group1] += letters[group2]
                    needs_new_letter = False

            # If we need a new letter and haven't exhausted the alphabet
            if needs_new_letter and current_letter_idx < len(available_letters):
                current_letter_idx += 1
                if current_letter_idx < len(available_letters):
                    letters[group1] = available_letters[current_letter_idx]

        return letters
    
    @staticmethod
    def _add_significance_letters(ax, df, groups, samples, test_recommendation, 
                                height_offset, font_size, error_type, pairwise_results=None):
        """Add significance letters with enhanced formatting"""
        try:
            # Debug output
            print(f"DEBUG: _add_significance_letters called with {len(groups)} groups")
            print(f"DEBUG: pairwise_results = {pairwise_results}")

            # Always use post-hoc results if available and non-empty
            if pairwise_results is not None:
                if len(pairwise_results) > 0:
                    means_dict = {g: np.mean(samples[g]) for g in groups}
                    n = len(groups)
                    is_full_pairwise = len(pairwise_results) == n*(n-1)//2
                    is_dunnett_t3 = any(
                        str(c.get('test','')).lower().startswith("dunnett's t3")
                        for c in pairwise_results
                    )
                    sweep_flag = not (is_full_pairwise or is_dunnett_t3)
                    letters = DataVisualizer.get_significance_letters_from_posthoc(
                        groups,
                        pairwise_results,
                        alpha=0.05,
                        sweep=sweep_flag,
                        sort_by=means_dict
                    )
                else:
                    print("WARNING: pairwise_results provided but empty; falling back to simple method.")
                    letters = DataVisualizer.get_significance_letters(
                        samples, groups, test_recommendation=test_recommendation
                    )
            else:
                print("WARNING: pairwise_results is None; falling back to simple method. If post-hoc results are expected, check upstream logic.")
                letters = DataVisualizer.get_significance_letters(
                    samples, groups, test_recommendation=test_recommendation
                )

            print(f"DEBUG: Generated letters = {letters}")
            
            y_max = df['Value'].max()
            y_offset = height_offset * y_max
            
            # Calculate bar heights with error bars
            bar_heights = []
            for group in groups:
                values = samples[group]
                mean_val = np.mean(values)
                if error_type == 'sd':
                    error = np.std(values, ddof=1)
                elif error_type == 'se':
                    error = np.std(values, ddof=1) / np.sqrt(len(values))
                else:  # ci
                    error = 1.96 * np.std(values, ddof=1) / np.sqrt(len(values))
                bar_heights.append(mean_val + error)
            
            # Place letters with enhanced styling
            for i, group in enumerate(groups):
                letter = letters[group]
                print(f"DEBUG: Adding letter '{letter}' to group '{group}' at position {i}")
                ax.text(i, bar_heights[i] + y_offset, letter,
                       horizontalalignment='center', 
                       verticalalignment='bottom',
                       color='black', fontweight='bold',
                       fontsize=font_size,
                       bbox=dict(boxstyle="round,pad=0.3", 
                                facecolor="white", 
                                edgecolor="gray",
                                alpha=0.8))
        except Exception as e:
            print(f"Error adding significance letters: {str(e)}")
            import traceback
            traceback.print_exc()

    @staticmethod
    def _add_significance_letters_raincloud(ax, groups, samples, test_recommendation, 
                                          height_offset, font_size, positions=None, pairwise_results=None):
        """Add significance letters for horizontal raincloud plots"""
        try:
            import string
            import numpy as np
            
            # Debug output
            print(f"DEBUG: _add_significance_letters_raincloud called with {len(groups)} groups")
            print(f"DEBUG: pairwise_results = {pairwise_results}")
            
            # Always use post-hoc results if available and non-empty
            if pairwise_results is not None:
                if len(pairwise_results) > 0:
                    print("DEBUG: Using post-hoc results for raincloud significance letters")
                    means_dict = {group: np.mean(samples[group]) for group in groups}
                    letters = DataVisualizer.get_significance_letters_from_posthoc(
                        groups, pairwise_results, alpha=0.05, sweep=True, sort_by=means_dict
                    )
                else:
                    print("WARNING: pairwise_results provided but empty; falling back to simple method.")
                    letters = DataVisualizer.get_significance_letters(
                        samples, groups, test_recommendation=test_recommendation
                    )
            else:
                print("WARNING: pairwise_results is None; falling back to simple method. If post-hoc results are expected, check upstream logic.")
                letters = DataVisualizer.get_significance_letters(
                    samples, groups, test_recommendation=test_recommendation
                )
            
            print(f"DEBUG: Generated raincloud letters = {letters}")
            
            # For horizontal raincloud, find the rightmost x position
            x_positions = []
            for group in groups:
                values = samples[group]
                if hasattr(values, '__len__') and len(values) > 0:
                    x_positions.append(max(values))
            
            if x_positions:
                x_max = max(x_positions)
                # Reduce the offset to move letters closer to the plots (about 1.5-2 cm)
                x_range = max(x_positions) - min([min(samples[group]) for group in groups if hasattr(samples[group], '__len__') and len(samples[group]) > 0])
                x_offset = x_max + (x_range * 0.12)  # Much closer to the plots
                
                # Use provided positions or fall back to default spacing
                if positions is None:
                    positions = np.arange(1, len(groups)+1)
                
                # Place letters to the right of the plot elements using actual positions
                for i, group in enumerate(groups):
                    letter = letters[group]
                    y_pos = positions[i]  # Use calculated positions instead of i+1
                    ax.text(x_offset, y_pos, letter,
                           horizontalalignment='left', 
                           verticalalignment='center',
                           color='black', fontweight='bold',
                           fontsize=font_size,
                           bbox=dict(boxstyle="round,pad=0.3", 
                                    facecolor="white", 
                                    edgecolor="gray",
                                    alpha=0.8))
        except Exception as e:
            print(f"Error adding raincloud significance letters: {str(e)}")

    @staticmethod
    def set_global_font(family="Arial", main_text_family="Times New Roman", use_latex=False):
        """
        Set global font for all plots. Optionally enable LaTeX rendering.
        """
        import matplotlib
        import matplotlib.pyplot as plt
        if use_latex:
            plt.rcParams['text.usetex'] = True
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.serif'] = [main_text_family]
        else:
            plt.rcParams['font.family'] = family
            plt.rcParams['axes.labelweight'] = 'bold'
            plt.rcParams['axes.titlesize'] = 12
            plt.rcParams['axes.labelsize'] = 9
            plt.rcParams['xtick.labelsize'] = 7
            plt.rcParams['ytick.labelsize'] = 7

    @staticmethod
    def apply_custom_colormap(ax, groups, colormap="gray", accent="#0072B2", accent_idx=0, greyscale=True, user_colors=None):
        """
        Apply a two-tone or accent color scheme to bars/boxes/violins.
        """
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        n = len(groups)
        if user_colors:
            colors = user_colors
        elif greyscale:
            colors = [str(0.2 + 0.6*i/(n-1)) for i in range(n)]
            if accent:
                colors[accent_idx % n] = accent
        else:
            cmap = cm.get_cmap(colormap, n)
            colors = [mcolors.to_hex(cmap(i)) for i in range(n)]
        for i, patch in enumerate(ax.patches):
            patch.set_facecolor(colors[i % n])
        return colors

    @staticmethod
    def add_panel_labels(fig, axes, labels=None, fontdict=None, x=0.01, y=0.98):
        """
        Add bold panel labels (A, B, C, ...) to each subplot.
        """
        import string
        if not isinstance(axes, (list, tuple, np.ndarray)):
            axes = [axes]
        if labels is None:
            labels = list(string.ascii_uppercase)
        for i, ax in enumerate(axes):
            ax.text(x, y, labels[i], transform=ax.transAxes,
                    fontsize=10, fontweight='bold', va='top', ha='left',
                    fontdict=fontdict if fontdict else None)

    @staticmethod
    def annotate_bar_values(ax, bars, values, errors=None, fmt="{:.2f}", font_size=6, y_offset=0.01, error_fmt="±{:.2f}"):
        """
        Annotate mean±SEM or other values above bars.
        """
        for bar, val, err in zip(bars, values, errors if errors is not None else [None]*len(values)):
            height = bar.get_height()
            label = fmt.format(val)
            if err is not None:
                label += error_fmt.format(err)
            ax.text(bar.get_x() + bar.get_width()/2, height + y_offset*height, label,
                    ha='center', va='bottom', fontsize=font_size, color='black')

    @staticmethod
    def annotate_box_medians(ax, boxplot, medians, fmt="{:.2f}", font_size=6, y_offset=0.01):
        """
        Annotate median values above boxplots.
        """
        for median_line, val in zip(boxplot['medians'], medians):
            x = median_line.get_xdata().mean()
            y = median_line.get_ydata().mean()
            ax.text(x, y + y_offset*abs(y), fmt.format(val),
                    ha='center', va='bottom', fontsize=font_size, color='black')

    @staticmethod
    def add_reference_line(ax, y=0, style='--', color='grey', linewidth=0.7, alpha=0.5, label=None):
        """
        Add a horizontal reference line (e.g., baseline).
        """
        ax.axhline(y, linestyle=style, color=color, linewidth=linewidth, alpha=alpha, label=label)

    @staticmethod
    def add_inset_axes(ax, bounds=[0.6, 0.6, 0.35, 0.35], zoom_xlim=None, zoom_ylim=None, plot_func=None, **plot_kwargs):
        """
        Add an inset axes for zoomed-in region or sub-distribution.
        """
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        inset_ax = inset_axes(ax, width="35%", height="35%", loc='upper right', bbox_to_anchor=bounds, bbox_transform=ax.transAxes)
        if plot_func:
            plot_func(inset_ax, **plot_kwargs)
        if zoom_xlim:
            inset_ax.set_xlim(*zoom_xlim)
        if zoom_ylim:
            inset_ax.set_ylim(*zoom_ylim)
        return inset_ax

    @staticmethod
    def set_axis_linewidths(ax, width=0.7):
        """
        Set axes and box/violin/bar outlines to a specific line width.
        """
        for spine in ax.spines.values():
            spine.set_linewidth(width)
        for line in ax.get_lines():
            line.set_linewidth(width)
        for patch in getattr(ax, 'patches', []):
            patch.set_linewidth(width)

    @staticmethod
    def set_grid_style(ax, which='major', axis='y', color='#cccccc', alpha=0.15, linewidth=0.5):
        """
        Set grid lines to light grey or only horizontal minor grid lines.
        """
        ax.grid(True, which=which, axis=axis, color=color, alpha=alpha, linewidth=linewidth)

    @staticmethod
    def set_minor_ticks(ax, x_minor=True, y_minor=True, x_locator=None, y_locator=None, inward=False):
        """
        Enable minor ticks and custom locators.
        """
        import matplotlib.ticker as mticker
        if x_minor:
            ax.xaxis.set_minor_locator(x_locator or mticker.AutoMinorLocator())
        if y_minor:
            ax.yaxis.set_minor_locator(y_locator or mticker.AutoMinorLocator())
        if inward:
            ax.tick_params(axis='both', which='both', direction='in', length=3)

    @staticmethod
    def set_tick_format(ax, axis='y', style='si'):
        """
        Custom tick formatting: SI prefixes, percentages, currency, etc.
        """
        import matplotlib.ticker as mticker
        if style == 'si':
            ax.yaxis.set_major_formatter(mticker.EngFormatter())
        elif style == 'percent':
            ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        elif style == 'currency':
            ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))
        # Add more as needed

    @staticmethod
    def add_stat_bracket(ax, x1, x2, y, text, height=0.02, linewidth=0.7, fontsize=8, color='black'):
        """
        Draw a single-line significance bracket with centered text (e.g., *, **, a, b).
        """
        ax.plot([x1, x1, x2, x2], [y, y+height, y+height, y], lw=linewidth, c=color, clip_on=False)
        ax.text((x1+x2)/2, y+height, text, ha='center', va='bottom', color=color, fontsize=fontsize, fontweight='bold')

    @staticmethod
    def add_letter_grouping(ax, groups, letters, y_positions, font_size=8, box_color="#eeeeee", text_color="grey", pad=0.01):
        """
        Add small grey boxes with letter groupings above bars/boxes.
        """
        for i, group in enumerate(groups):
            ax.text(i, y_positions[i]+pad, letters[group],
                    ha='center', va='bottom', fontsize=font_size,
                    color=text_color, bbox=dict(boxstyle="round,pad=0.2", fc=box_color, ec='none', alpha=0.8))

    @staticmethod
    def set_opacity_gradient(ax, base_alpha=0.8, fade_to=0.3):
        """
        Fade earlier bars/violins/boxes to show density/focus.
        """
        patches = getattr(ax, 'patches', [])
        n = len(patches)
        for i, patch in enumerate(patches):
            alpha = base_alpha - (base_alpha-fade_to)*i/(n-1) if n > 1 else base_alpha
            patch.set_alpha(alpha)

    @staticmethod
    def set_panel_background(ax, motif="striped", color1="#ffffff", color2="#f7f7f7"):
        """
        Add alternating-row background shading.
        """
        ylim = ax.get_ylim()
        for i in range(int(ylim[0]), int(ylim[1]), 2):
            ax.axhspan(i, i+1, facecolor=color2, alpha=0.3, zorder=0)

    @staticmethod
    def set_aspect_ratio(ax, ratio=1.0):
        """
        Set custom aspect ratio (height/width).
        """
        ax.set_aspect(ratio)

    @staticmethod
    def set_log_axis(ax, axis='y', base=10, symlog=False, linthresh=1):
        """
        Set log or symlog scaling on an axis.
        """
        if symlog:
            if axis == 'y':
                ax.set_yscale('symlog', linthresh=linthresh)
            else:
                ax.set_xscale('symlog', linthresh=linthresh)
        else:
            if axis == 'y':
                ax.set_yscale('log', base=base)
            else:
                ax.set_xscale('log', base=base)

    @staticmethod
    def add_secondary_axis(ax, which='y', label=None, func=None, inv_func=None):
        """
        Add a secondary y-axis (or x-axis) for e.g. percentages.
        """
        if which == 'y':
            secax = ax.secondary_yaxis('right', functions=(func, inv_func) if func and inv_func else None)
            if label:
                secax.set_ylabel(label)
            return secax
        else:
            secax = ax.secondary_xaxis('top', functions=(func, inv_func) if func and inv_func else None)
            if label:
                secax.set_xlabel(label)
            return secax

    @staticmethod
    def add_broken_axis(ax, break_y, gap=0.02):
        """
        Simple broken y-axis: add a zigzag or break mark at break_y.
        """
        ylim = ax.get_ylim()
        ax.plot([-.1, .1], [break_y-gap, break_y+gap], color='k', lw=1, transform=ax.get_yaxis_transform(), clip_on=False)

    @staticmethod
    def export_with_metadata(fig, filename, metadata=None, embed_fonts=True, dpi=300, filetype="pdf"):
        """
        Save figure with embedded fonts and metadata (PDF/SVG).
        """
        from matplotlib import font_manager
        import matplotlib.pyplot as plt
        if embed_fonts and filetype in ["pdf", "svg"]:
            plt.rcParams['pdf.fonttype'] = 42
            plt.rcParams['svg.fonttype'] = 'none'
        if metadata is None:
            metadata = {}
        metadata.setdefault("Creator", "DataVisualizer")
        metadata.setdefault("Title", filename)
        metadata.setdefault("Description", "Generated by DataVisualizer")
        fig.savefig(filename, dpi=dpi, bbox_inches='tight', metadata=metadata)

    @staticmethod
    def add_metadata_block(fig, params, version="1.0", date=None):
        """
        Add a tiny text block with version, date, and parameters to the PDF/SVG.
        """
        import datetime
        if date is None:
            date = datetime.datetime.now().strftime("%Y-%m-%d")
        meta_text = f"DataVisualizer v{version} | {date}\nParams: {params}"
        fig.text(0.01, 0.01, meta_text, fontsize=4, color='gray', ha='left', va='bottom', alpha=0.7)

    @staticmethod
    def plot_from_config(ax, groups, samples, config):
        """
        Zentrale Dispatcher-Methode für alle Plot-Typen.
        Zeichnet auf dem übergebenen ax-Objekt genau den Plot,
        den config spezifiziert (Typ, Farben, Hatches, Linien, Grid, etc.).
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            Die Axes auf die gezeichnet werden soll
        groups : list
            Liste der Gruppennamen
        samples : dict
            Dictionary mit Gruppennamen als Keys und Messwerten als Listen
        config : dict
            Konfiguration mit allen Plot-Parametern
        """
        # 1. Alle nötigen Parameter aus config auslesen
        plot_type = config.get('plot_type', 'Bar')
        
        # Farben für Gruppen extrahieren
        colors_dict = config.get('colors', {})
        colors = [colors_dict.get(g, '#3357FF') for g in groups] if isinstance(colors_dict, dict) else colors_dict
        
        # Hatches für Gruppen extrahieren
        hatches_dict = config.get('hatches', {})
        hatches = [hatches_dict.get(g, '') for g in groups] if isinstance(hatches_dict, dict) else hatches_dict
        
        # Standard-Parameter
        alpha = config.get('alpha', 0.8)
        error_type = config.get('error_type', 'sd')
        error_style = config.get('error_style', 'caps')
        bar_edge_color = config.get('bar_edge_color', 'black')
        bar_linewidth = config.get('bar_linewidth', 0.5)
        grid_style = config.get('grid_style', 'none')
        despine = config.get('despine', True)
        
        # 2. Basis-kwargs für alle Plot-Typen (nur gemeinsame Parameter)
        base_kwargs = {
            'width': config.get('width', 8),
            'height': config.get('height', 6),
            'dpi': config.get('dpi', 300),
            'theme': config.get('theme', 'default'),
            'colors': colors,
            'hatches': hatches,
            'alpha': alpha,
            # Seaborn styling parameters
            'seaborn_context': config.get('seaborn_context', 'notebook'),
            'seaborn_palette': config.get('seaborn_palette', 'Greys'),
            'use_seaborn_styling': config.get('use_seaborn_styling', True),
            'save_plot': False,  # NIE speichern in der Preview!
            'file_formats': [],  # NIE speichern in der Preview!
            'file_name': None,
            'ax': ax,  # Wichtig: auf übergebene ax zeichnen
            'show_points': config.get('show_points', True),
            'point_style': config.get('point_style', 'jitter'),
            'point_size': config.get('point_size', 80),
            'jitter_strength': config.get('jitter_strength', 0.3),  # Jitter Parameter hinzugefügt
            'show_significance_letters': config.get('show_significance_letters', True),
            'significance_height_offset': config.get('significance_height_offset', 0.05),
            'significance_font_size': config.get('significance_font_size', 12),
            'x_label': config.get('x_label', ''),
            'y_label': config.get('y_label', ''),
            'title': config.get('title', ''),
            'x_label_size': config.get('fontsize_axis', 12),
            'y_label_size': config.get('fontsize_axis', 12),
            'title_size': config.get('fontsize_title', 14),
            'tick_label_size': config.get('fontsize_ticks', 10),
            'grid_style': grid_style,
            'spine_style': 'minimal' if despine else 'box',
            'show_legend': config.get('show_legend', True),
            'legend_position': config.get('legend_position', 'upper right')
        }
        
        # 3. Plot-Typ spezifische Parameter und Dispatch
        if plot_type == "Bar":
            bar_kwargs = base_kwargs.copy()
            bar_kwargs.update({
                'show_error_bars': config.get('show_error_bars', True),
                'error_type': error_type,
                'capsize': 0 if error_style == 'line' else config.get('capsize', 0.05),
                'bar_edge_color': bar_edge_color,
                'bar_edge_width': bar_linewidth,
                'error_style': error_style
            })
            DataVisualizer.plot_bar(groups, samples, **bar_kwargs)
            
        elif plot_type == "Box":
            box_kwargs = base_kwargs.copy()
            box_kwargs.update({
                'show_error_bars': config.get('show_error_bars', True),
                'error_type': error_type,
                'capsize': 0 if error_style == 'line' else config.get('capsize', 0.05),
                'edge_color': bar_edge_color,
                'edge_width': bar_linewidth,
                'error_style': error_style,
                'box_width': config.get('box_width', 0.8)
            })
            DataVisualizer.plot_box(groups, samples, **box_kwargs)
            
        elif plot_type == "Violin":
            # Violin hat keine show_error_bars oder error_type Parameter
            violin_kwargs = base_kwargs.copy()
            violin_kwargs.update({
                'edge_color': bar_edge_color,
                'edge_width': bar_linewidth,
                'violin_width': config.get('violin_width', 0.8)
            })
            DataVisualizer.plot_violin(groups, samples, **violin_kwargs)
            
        elif plot_type == "Raincloud":
            # Raincloud hat spezifische Parameter
            raincloud_kwargs = base_kwargs.copy()
            raincloud_kwargs.update({
                'edge_color': bar_edge_color,
                'edge_width': bar_linewidth,
                'violin_width': config.get('violin_width', 0.8),
                'box_width': config.get('box_width', 0.2),
                'group_spacing': config.get('group_spacing', 0.5),
                'fontsize_groupnames': config.get('fontsize_groupnames', None),
                # Raincloud-specific color settings
                'violin_colors': config.get('violin_colors', None),
                'box_colors': config.get('box_colors', None),
                'point_colors': config.get('point_colors', None),
                # Spacing settings
                'point_offset': config.get('point_offset', 0.2),
                'point_jitter': config.get('point_jitter', 0.05),
                # Line thickness settings
                'frame_thickness': config.get('frame_thickness', 0.7),
                'axis_thickness': config.get('axis_thickness', 0.5)
            })
            DataVisualizer.plot_raincloud(groups, samples, **raincloud_kwargs)
        else:
            raise ValueError(f"Unbekannter plot_type: {plot_type}")
        
        # 4. Zusätzliche Achsen-Formatierungen
        
        # Font Family setzen (matplotlib rcParams)
        font_family = config.get('font_family', 'Arial')
        import matplotlib.pyplot as plt
        plt.rcParams['font.family'] = font_family
        
        # Title - only set fontsize if explicitly provided in config
        if config.get('show_title', True) and config.get('title'):
            title_kwargs = {}
            if 'fontsize_title' in config:
                title_kwargs['fontsize'] = config['fontsize_title']
            ax.set_title(config.get('title', ''), **title_kwargs)
        
        # Axis labels - only set fontsize if explicitly provided in config
        if config.get('x_label'):
            xlabel_kwargs = {}
            if 'fontsize_axis' in config:
                xlabel_kwargs['fontsize'] = config['fontsize_axis']
            ax.set_xlabel(config.get('x_label', ''), **xlabel_kwargs)
        if config.get('y_label'):
            ylabel_kwargs = {}
            if 'fontsize_axis' in config:
                ylabel_kwargs['fontsize'] = config['fontsize_axis']
            ax.set_ylabel(config.get('y_label', ''), **ylabel_kwargs)
        
        # Tick label font sizes - only set if explicitly provided in config
        if 'fontsize_ticks' in config:
            ax.tick_params(axis='both', which='major', labelsize=config['fontsize_ticks'])
        
        # Axis thickness
        axis_thickness = config.get('axis_thickness', 0.7)
        for spine in ax.spines.values():
            spine.set_linewidth(axis_thickness)
        ax.tick_params(axis='both', which='major', width=axis_thickness)
        ax.tick_params(axis='both', which='minor', width=axis_thickness * 0.7)
        
        # Minor Ticks - only on numerical axes
        if config.get('minor_ticks', False):
            plot_type = config.get('plot_type', 'Bar')
            # For Bar, Box, Violin: y is numeric, x is categorical
            # For Raincloud: x is numeric, y is categorical
            if plot_type == 'Raincloud':
                DataVisualizer.set_minor_ticks(ax, x_minor=True, y_minor=False)
            else:
                DataVisualizer.set_minor_ticks(ax, x_minor=False, y_minor=True)
        
        # Control ticks for many groups to prevent matplotlib errors
        DataVisualizer._control_ticks_for_many_groups(ax, groups, config.get('plot_type', 'Bar'))
        
        # Grid-Einstellungen
        if grid_style and grid_style != 'none':
            grid_alpha = config.get('grid_alpha', 0.3)
            if config.get('minor_ticks', False):
                ax.grid(True, which='both', alpha=grid_alpha, linestyle='-', linewidth=axis_thickness * 0.5)
            else:
                ax.grid(True, which='major', alpha=grid_alpha, linestyle='-', linewidth=axis_thickness * 0.5)
        else:
            ax.grid(False)
            
        # Spine-Einstellungen (Remove Spines)
        if despine:
            import seaborn as sns
            sns.despine(ax=ax)   