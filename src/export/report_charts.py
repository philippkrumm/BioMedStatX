"""Plotly chart / figure builders for the HTML report.

Extracted from ``html_exporter.py`` (Phase 5 of the god-file split): the per-
design plot builders (LMM, trajectory, interaction, profile, ROC, regression,
correlation, ANCOVA, group comparison), the chart bundle assembler, figure->
HTML serialization, significance brackets and plot-data prep. Stateless
``@staticmethod`` helpers mixed into ``HTMLExporter``; call sites unchanged.
"""

import math

import numpy as np
from scipy import stats

from export.report_association import _AssociationMixin
from export.report_formatting import _FormattingMixin

try:
    from core.logger_config import get_logger
except ImportError:  # pragma: no cover — fallback when logger_config missing
    import logging as _logging

    def get_logger(name):
        return _logging.getLogger(name)


logger = get_logger(__name__)


class _ChartsMixin:
    """Charts helpers mixed into ``HTMLExporter``."""

    @staticmethod
    def _build_lmm_chart(results: dict) -> dict | None:
        if results.get("model_type") != "LMM":
            return None
        fe_table = results.get("fixed_effects_table") or []
        if not fe_table:
            return None
        try:
            import plotly.graph_objects as go

            valid = [
                fe for fe in fe_table
                if fe.get("coefficient") is not None
                and fe.get("ci_lower") is not None
                and fe.get("ci_upper") is not None
            ]
            if not valid:
                return None

            # Reverse so intercept ends up at bottom of the y-axis
            valid = list(reversed(valid))
            params = [str(fe.get("parameter", "")) for fe in valid]
            coefs = [float(fe["coefficient"]) for fe in valid]
            lowers = [float(fe["ci_lower"]) for fe in valid]
            uppers = [float(fe["ci_upper"]) for fe in valid]
            pvals = [fe.get("p_value") for fe in valid]

            colors = [
                "#0f766e" if (pv is not None and float(pv) < 0.05) else "#94a3b8"
                for pv in pvals
            ]
            error_minus = [c - l for c, l in zip(coefs, lowers)]
            error_plus = [u - c for c, u in zip(coefs, uppers)]

            figure = go.Figure()
            figure.add_vline(x=0, line=dict(color="#64748b", width=1, dash="dot"))
            figure.add_trace(go.Scatter(
                x=coefs,
                y=params,
                mode="markers",
                marker=dict(
                    size=10,
                    color=colors,
                    line=dict(width=1.2, color="#16313a"),
                ),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=error_plus,
                    arrayminus=error_minus,
                    color="#64748b",
                    thickness=1.8,
                    width=6,
                ),
                hovertemplate="<b>%{y}</b><br>β = %{x:.4f}<extra></extra>",
                name="Fixed effect",
            ))
            figure.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#fffdf8",
                margin=dict(l=180, r=30, t=24, b=48),
                font=dict(family="Segoe UI, Helvetica Neue, sans-serif", color="#16313a"),
                xaxis_title="Coefficient (β)",
                yaxis=dict(title="", automargin=True),
                showlegend=False,
                height=max(260, len(params) * 44 + 80),
            )
            html = _ChartsMixin._figure_to_html(figure, div_id="biomedstatx-lmm-chart")
            if not html:
                return None
            return {
                "title": "Fixed Effects (Coefficient Plot)",
                "subtitle": (
                    "Dots = coefficient estimate (β), lines = 95% CI. "
                    "Teal = significant (p < 0.05), grey = not significant. "
                    "Vertical dashed line at β = 0."
                ),
                "html": html,
                "div_id": "biomedstatx-lmm-chart",
            }
        except Exception as exc:
            logger.warning("LMM chart generation failed: %s", exc, exc_info=True)
            return None

    @staticmethod
    def _build_trajectory_chart(results: dict) -> dict | None:
        trajectories = results.get("plot_subject_trajectories")
        if not isinstance(trajectories, list) or len(trajectories) < 1:
            return None
        try:
            import plotly.graph_objects as go

            figure = go.Figure()
            palette = ["#0f766e", "#1f7a5a", "#b7791f", "#9f3a38", "#1d4ed8", "#7c3aed"]

            # Collect all timepoints/conditions in order
            all_groups: list[str] = []
            seen: set[str] = set()
            for traj in trajectories:
                for pt in (traj.get("points") or []):
                    g = str(pt.get("group", ""))
                    if g and g not in seen:
                        all_groups.append(g)
                        seen.add(g)

            if not all_groups:
                return None

            group_idx = {g: i for i, g in enumerate(all_groups)}

            # Subject lines (thin grey)
            for traj in trajectories:
                pts = traj.get("points") or []
                if len(pts) < 2:
                    continue
                x_pts = [group_idx[pt["group"]] for pt in pts if pt.get("group") in group_idx]
                y_pts = [pt["value"] for pt in pts if pt.get("group") in group_idx]
                figure.add_trace(go.Scatter(
                    x=x_pts, y=y_pts,
                    mode="lines+markers",
                    line=dict(color="rgba(100,120,130,0.28)", width=1),
                    marker=dict(size=4, color="rgba(100,120,130,0.4)"),
                    showlegend=False,
                    hoverinfo="skip",
                ))

            # Group mean line (bold, colored)
            from collections import defaultdict
            group_values: dict[str, list[float]] = defaultdict(list)
            for traj in trajectories:
                for pt in (traj.get("points") or []):
                    g = str(pt.get("group", ""))
                    if g in group_idx:
                        group_values[g].append(float(pt["value"]))

            mean_x = sorted(group_values.keys(), key=lambda g: group_idx[g])
            mean_y = [float(np.mean(group_values[g])) for g in mean_x]
            mean_x_idx = [group_idx[g] for g in mean_x]

            figure.add_trace(go.Scatter(
                x=mean_x_idx, y=mean_y,
                mode="lines+markers",
                line=dict(color=palette[0], width=2.5),
                marker=dict(size=8, color=palette[0]),
                name="Group mean",
                showlegend=True,
            ))

            figure.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#fffdf8",
                margin=dict(l=48, r=20, t=24, b=56),
                font=dict(family="Segoe UI, Helvetica Neue, sans-serif", color="#16313a"),
                xaxis=dict(
                    tickmode="array",
                    tickvals=list(range(len(all_groups))),
                    ticktext=all_groups,
                    title="Condition / Timepoint",
                ),
                yaxis_title="Observed values",
                legend=dict(orientation="h", x=0.01, y=1.08),
            )

            html = _ChartsMixin._figure_to_html(figure, div_id="biomedstatx-trajectory-chart")
            if not html:
                return None
            n_subjects = len(trajectories)
            return {
                "title": "Subject Trajectories",
                "subtitle": f"Individual profiles (n={n_subjects}) with group mean overlay.",
                "html": html,
                "div_id": "biomedstatx-trajectory-chart",
            }
        except Exception as exc:
            logger.warning("trajectory chart generation failed: %s", exc, exc_info=True)
            return None

    @staticmethod
    def _build_interaction_plot(results: dict) -> dict | None:
        """Interaction plot for Two-Way ANOVA and Mixed ANOVA — cell means ± SE."""
        model_type = results.get("model_type", "")
        if model_type not in ("TwoWayANOVA", "MixedANOVA"):
            return None
        descriptive = results.get("descriptive") or {}
        if not descriptive:
            return None
        factors = results.get("factors") or []
        if model_type == "MixedANOVA":
            between_list = [f["factor"] for f in factors if f.get("type") == "between"]
            within_list = [f["factor"] for f in factors if f.get("type") == "within"]
            if not between_list or not within_list:
                return None
            # x-axis = between-factor levels, one line per within-factor level
            factor_x = between_list[0]
            factor_line = within_list[0]
        else:
            between_list = [f["factor"] for f in factors if f.get("type") == "between"]
            if len(between_list) >= 2:
                factor_x, factor_line = between_list[0], between_list[1]
            elif len(factors) >= 2:
                factor_x = factors[0]["factor"]
                factor_line = factors[1]["factor"]
            else:
                return None
        # Build cell grid from descriptive
        cell_data: dict = {}
        x_levels_order: list = []
        line_levels_order: list = []
        seen_x: set = set()
        seen_line: set = set()
        for key, stats in descriptive.items():
            parts_dict: dict = {}
            for part in key.split(", "):
                if "=" in part:
                    fname, fval = part.split("=", 1)
                    parts_dict[fname.strip()] = fval.strip()
            x_val = parts_dict.get(factor_x)
            line_val = parts_dict.get(factor_line)
            if x_val is None or line_val is None:
                continue
            if x_val not in seen_x:
                x_levels_order.append(x_val)
                seen_x.add(x_val)
            if line_val not in seen_line:
                line_levels_order.append(line_val)
                seen_line.add(line_val)
            cell_data.setdefault(x_val, {})[line_val] = {
                "mean": float(stats["mean"]) if stats.get("mean") is not None else None,
                "se": float(stats.get("stderr") or stats.get("se") or 0),
                "n": stats.get("n", 0),
            }
        if not cell_data:
            return None
        try:
            import plotly.graph_objects as go
            palette = ["#0f766e", "#b7791f", "#1d4ed8", "#9f3a38", "#7c3aed", "#1f7a5a"]
            interactions = results.get("interactions") or []
            interaction_sig = any(
                isinstance(inter.get("p_value"), (int, float)) and inter["p_value"] < 0.05
                for inter in interactions
            )
            fig = go.Figure()
            for idx, line_level in enumerate(line_levels_order):
                color = palette[idx % len(palette)]
                y_vals = []
                y_err = []
                hover_texts = []
                for x_val in x_levels_order:
                    cell = cell_data.get(x_val, {}).get(line_level)
                    if cell and cell["mean"] is not None:
                        y_vals.append(cell["mean"])
                        y_err.append(cell["se"])
                        hover_texts.append(
                            f"{factor_x}={x_val}, {factor_line}={line_level}<br>"
                            f"Mean: {cell['mean']:.3f} ± {cell['se']:.3f} SE<br>n={cell['n']}"
                        )
                    else:
                        y_vals.append(None)
                        y_err.append(0)
                        hover_texts.append("")
                fig.add_trace(go.Scatter(
                    x=x_levels_order,
                    y=y_vals,
                    mode="lines+markers",
                    name=f"{factor_line}={line_level}",
                    line=dict(color=color, width=2),
                    marker=dict(size=8, color=color),
                    error_y=dict(type="data", array=y_err, visible=True,
                                 color=color, thickness=1.5, width=4),
                    hovertext=hover_texts,
                    hoverinfo="text",
                ))
            interaction_note = ""
            if interactions:
                inter = interactions[0]
                ip = inter.get("p_value")
                if isinstance(ip, (int, float)):
                    interaction_note = (
                        f"Interaction p = {_FormattingMixin._format_p_value(ip)}"
                        + (" — significant" if ip < 0.05 else "")
                    )
            fig.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#fffdf8",
                margin=dict(l=56, r=20, t=36, b=64),
                font=dict(family="Segoe UI, Helvetica Neue, sans-serif", color="#16313a"),
                xaxis=dict(title=factor_x, automargin=True),
                yaxis=dict(title="Cell Mean"),
                legend=dict(title=dict(text=factor_line), orientation="h", x=0.01, y=1.1),
                annotations=[dict(
                    x=0.5, y=-0.2, xref="paper", yref="paper",
                    text=interaction_note, showarrow=False,
                    font=dict(size=11, color="#b7791f" if interaction_sig else "#555"),
                )] if interaction_note else [],
            )
            html = _ChartsMixin._figure_to_html(fig, div_id="biomedstatx-interaction-plot")
            if not html:
                return None
            return {
                "title": "Interaction Plot — Cell Means ± SE",
                "subtitle": f"{factor_x} × {factor_line}",
                "html": html,
                "div_id": "biomedstatx-interaction-plot",
                "info": (
                    "Each line represents one level of the line-factor, plotted across levels of the x-axis factor.\n"
                    "Crossing or diverging lines indicate an interaction: the effect of one factor depends on the level of the other.\n"
                    "Parallel lines indicate no interaction — each factor acts independently.\n"
                    "Error bars show the standard error of the mean (SE) for each cell.\n\n"
                    "Hover over any point to see exact cell mean, SE, and n."
                ),
            }
        except Exception as exc:
            logger.warning("interaction plot failed: %s", exc, exc_info=True)
            return None

    @staticmethod
    def _build_profile_plot(results: dict) -> dict | None:
        """Profile plot for RM-ANOVA — group mean ± SE with individual trajectories."""
        if results.get("model_type") != "RepeatedMeasuresANOVA":
            return None
        descriptive = results.get("descriptive") or {}
        if not descriptive:
            return None
        # Determine within factor name from factors list or infer from descriptive keys
        factors = results.get("factors") or []
        within_list = [f["factor"] for f in factors if f.get("type") == "within"]
        if within_list:
            within_factor = within_list[0]
        else:
            within_factor = None
            for key in descriptive:
                if "=" in key and ", " not in key:
                    within_factor = key.split("=")[0].strip()
                    break
        if not within_factor:
            return None
        # Parse timepoint → stats
        level_stats: dict = {}
        for key, stats in descriptive.items():
            if "=" in key:
                fname, fval = key.split("=", 1)
                if fname.strip() == within_factor:
                    level_stats[fval.strip()] = stats
        if not level_stats:
            return None
        levels = list(level_stats.keys())
        try:
            levels_sorted = sorted(levels, key=lambda x: float(x))
        except (ValueError, TypeError):
            levels_sorted = sorted(levels)
        means = [level_stats[lv].get("mean") for lv in levels_sorted]
        ses = [float(level_stats[lv].get("stderr") or level_stats[lv].get("se") or 0) for lv in levels_sorted]
        try:
            import plotly.graph_objects as go
            palette = ["#0f766e", "#1f7a5a", "#b7791f", "#9f3a38", "#1d4ed8", "#7c3aed"]
            fig = go.Figure()
            level_idx = {lv: i for i, lv in enumerate(levels_sorted)}
            # Individual subject trajectories (grey thin lines)
            trajectories = results.get("plot_subject_trajectories") or []
            for traj in trajectories:
                pts = traj.get("points") or []
                if len(pts) < 2:
                    continue
                x_pts = []
                y_pts = []
                for pt in pts:
                    group_name = str(pt.get("group", ""))
                    if "=" in group_name:
                        _, lv = group_name.split("=", 1)
                        lv = lv.strip()
                    else:
                        lv = group_name
                    if lv in level_idx:
                        x_pts.append(level_idx[lv])
                        y_pts.append(float(pt["value"]))
                if len(x_pts) < 2:
                    continue
                fig.add_trace(go.Scatter(
                    x=x_pts, y=y_pts,
                    mode="lines+markers",
                    line=dict(color="rgba(100,120,130,0.22)", width=1),
                    marker=dict(size=3, color="rgba(100,120,130,0.35)"),
                    showlegend=False,
                    hoverinfo="skip",
                ))
            # Group mean ± SE
            fig.add_trace(go.Scatter(
                x=list(range(len(levels_sorted))),
                y=means,
                mode="lines+markers",
                name="Group mean ± SE",
                line=dict(color=palette[0], width=2.5),
                marker=dict(size=9, color=palette[0]),
                error_y=dict(type="data", array=ses, visible=True,
                             color=palette[0], thickness=1.5, width=5),
            ))
            fig.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#fffdf8",
                margin=dict(l=56, r=20, t=36, b=60),
                font=dict(family="Segoe UI, Helvetica Neue, sans-serif", color="#16313a"),
                xaxis=dict(
                    tickmode="array",
                    tickvals=list(range(len(levels_sorted))),
                    ticktext=levels_sorted,
                    title=within_factor,
                    automargin=True,
                ),
                yaxis=dict(title="Observed values"),
                legend=dict(orientation="h", x=0.01, y=1.1),
            )
            n_subjects = len(trajectories)
            subtitle = f"Mean ± SE across {within_factor} levels"
            if n_subjects:
                subtitle += f" | n={n_subjects} subjects"
            html = _ChartsMixin._figure_to_html(fig, div_id="biomedstatx-profile-plot")
            if not html:
                return None
            return {
                "title": "Profile Plot — Means ± SE across Timepoints",
                "subtitle": subtitle,
                "html": html,
                "div_id": "biomedstatx-profile-plot",
                "info": (
                    "Shows the group mean (±SE) at each level of the within-subject factor.\n"
                    "Grey lines trace individual subject trajectories — they reveal whether each participant follows the overall group trend.\n"
                    "Stable, parallel individual trajectories support the assumption of a consistent within-subject effect.\n"
                    "Error bars show the standard error of the mean (SE).\n\n"
                    "Hover over any point to see exact mean, SE, and n."
                ),
            }
        except Exception as exc:
            logger.warning("profile plot failed: %s", exc, exc_info=True)
            return None

    @staticmethod
    def _build_mixed_profile_plot(results: dict) -> dict | None:
        """Profile plot for Mixed ANOVA — one line per between-group over within-factor levels."""
        if results.get("model_type") != "MixedANOVA":
            return None
        descriptive = results.get("descriptive") or {}
        if not descriptive:
            return None
        factors = results.get("factors") or []
        between_list = [f["factor"] for f in factors if f.get("type") == "between"]
        within_list = [f["factor"] for f in factors if f.get("type") == "within"]
        if not between_list or not within_list:
            return None
        factor_between = between_list[0]
        factor_within = within_list[0]
        # Collect cell data: {between_level: {within_level: {mean, se}}}
        profile_data: dict = {}
        between_order: list = []
        within_order: list = []
        seen_b: set = set()
        seen_w: set = set()
        for key, stats in descriptive.items():
            parts_dict: dict = {}
            for part in key.split(", "):
                if "=" in part:
                    fname, fval = part.split("=", 1)
                    parts_dict[fname.strip()] = fval.strip()
            b_val = parts_dict.get(factor_between)
            w_val = parts_dict.get(factor_within)
            if b_val is None or w_val is None:
                continue
            if b_val not in seen_b:
                between_order.append(b_val)
                seen_b.add(b_val)
            if w_val not in seen_w:
                within_order.append(w_val)
                seen_w.add(w_val)
            profile_data.setdefault(b_val, {})[w_val] = {
                "mean": float(stats["mean"]) if stats.get("mean") is not None else None,
                "se": float(stats.get("stderr") or stats.get("se") or 0),
                "n": stats.get("n", 0),
            }
        if not profile_data:
            return None
        try:
            within_sorted = sorted(within_order, key=lambda x: float(x))
        except (ValueError, TypeError):
            within_sorted = sorted(within_order)
        try:
            import plotly.graph_objects as go
            palette = ["#0f766e", "#b7791f", "#1d4ed8", "#9f3a38", "#7c3aed", "#1f7a5a"]
            fig = go.Figure()
            for idx, b_level in enumerate(between_order):
                color = palette[idx % len(palette)]
                y_vals = []
                y_err = []
                hover_texts = []
                for w_level in within_sorted:
                    cell = profile_data.get(b_level, {}).get(w_level)
                    if cell and cell["mean"] is not None:
                        y_vals.append(cell["mean"])
                        y_err.append(cell["se"])
                        hover_texts.append(
                            f"{factor_between}={b_level}, {factor_within}={w_level}<br>"
                            f"Mean: {cell['mean']:.3f} ± {cell['se']:.3f} SE<br>n={cell['n']}"
                        )
                    else:
                        y_vals.append(None)
                        y_err.append(0)
                        hover_texts.append("")
                fig.add_trace(go.Scatter(
                    x=within_sorted,
                    y=y_vals,
                    mode="lines+markers",
                    name=f"{factor_between}={b_level}",
                    line=dict(color=color, width=2.5),
                    marker=dict(size=9, color=color),
                    error_y=dict(type="data", array=y_err, visible=True,
                                 color=color, thickness=1.5, width=5),
                    hovertext=hover_texts,
                    hoverinfo="text",
                ))
            fig.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#fffdf8",
                margin=dict(l=56, r=20, t=36, b=60),
                font=dict(family="Segoe UI, Helvetica Neue, sans-serif", color="#16313a"),
                xaxis=dict(title=factor_within, automargin=True),
                yaxis=dict(title="Group mean"),
                legend=dict(title=dict(text=factor_between), orientation="h", x=0.01, y=1.1),
            )
            html = _ChartsMixin._figure_to_html(fig, div_id="biomedstatx-mixed-profile-plot")
            if not html:
                return None
            return {
                "title": "Profile Plot — Group Means ± SE",
                "subtitle": f"{factor_between} groups over {factor_within} levels",
                "html": html,
                "div_id": "biomedstatx-mixed-profile-plot",
                "info": (
                    "Each line represents one between-subject group, plotted across levels of the within-subject factor.\n"
                    "Parallel lines indicate that the within-factor effect is consistent across groups (no interaction).\n"
                    "Crossing or diverging lines indicate a Group × Time interaction — the effect of the within-factor differs by group.\n"
                    "Error bars show the standard error of the mean (SE) for each cell.\n\n"
                    "Hover over any point to see exact cell mean, SE, and n."
                ),
            }
        except Exception as exc:
            logger.warning("mixed profile plot failed: %s", exc, exc_info=True)
            return None

    @staticmethod
    def _build_single_chart_bundle(results: dict) -> list[dict]:
        charts = []
        lmm_chart = _ChartsMixin._build_lmm_chart(results)
        if lmm_chart:
            charts.append({
                "title": lmm_chart["title"],
                "subtitle": lmm_chart["subtitle"],
                "html": lmm_chart["html"],
                "div_id": lmm_chart["div_id"],
            })
        ancova_chart = _ChartsMixin._build_ancova_chart(results)
        if ancova_chart:
            charts.append({
                "title": ancova_chart["title"],
                "subtitle": ancova_chart["subtitle"],
                "html": ancova_chart["html"],
                "div_id": ancova_chart["div_id"],
            })
        model_type = results.get("model_type", "")
        if model_type == "LogisticRegression":
            # OR table as inline HTML block (6-column, not 2-col stats table)
            or_block = _AssociationMixin._build_or_table_html(results)
            if or_block:
                charts.append(or_block)
            # ROC curve replaces meaningless boxplot of binary outcome
            roc_block = _ChartsMixin._build_roc_chart(results)
            if roc_block:
                charts.append(roc_block)
        elif model_type == "BetaRegression":
            # Coefficient table as inline HTML block
            beta_coef_block = _AssociationMixin._build_beta_coefficient_table_html(results)
            if beta_coef_block:
                charts.append(beta_coef_block)
            # Scatter + fitted curve replaces meaningless boxplot for proportion outcome
            beta_chart = _ChartsMixin._build_beta_regression_chart(results)
            if beta_chart:
                charts.append(beta_chart)
        elif model_type == "CorrelationMatrix":
            # Heatmaps replace meaningless boxplot — no group data, matrix data only
            charts.extend(_ChartsMixin._build_correlation_matrix_charts(results))
        elif model_type in ("TwoWayANOVA", "MixedANOVA"):
            interactions = results.get("interactions") or []
            interaction_sig = any(
                isinstance(inter.get("p_value"), (int, float)) and inter["p_value"] < 0.05
                for inter in interactions
            )
            interaction_plot = _ChartsMixin._build_interaction_plot(results)
            if model_type == "MixedANOVA":
                mixed_profile = _ChartsMixin._build_mixed_profile_plot(results)
                if interaction_sig:
                    # Significant interaction → interaction plot is primary
                    if interaction_plot:
                        charts.append(interaction_plot)
                    if mixed_profile:
                        charts.append(mixed_profile)
                else:
                    # Not significant → profile plot is primary
                    if mixed_profile:
                        charts.append(mixed_profile)
                    if interaction_plot:
                        charts.append(interaction_plot)
            else:
                # TwoWayANOVA: interaction plot first when significant, boxplot always shown
                if interaction_sig and interaction_plot:
                    charts.append(interaction_plot)
                group_chart = _ChartsMixin._build_group_comparison_chart(results)
                if group_chart:
                    charts.append({
                        "title": "Group Comparison",
                        "subtitle": "Distribution overview with boxplots and individual observations.",
                        "html": group_chart["html"],
                        "group_order": group_chart["group_order"],
                        "div_id": "biomedstatx-group-chart",
                    })
                if not interaction_sig and interaction_plot:
                    charts.append(interaction_plot)
        elif model_type == "RepeatedMeasuresANOVA":
            # Profile plot with subject trajectories is primary
            profile_plot = _ChartsMixin._build_profile_plot(results)
            if profile_plot:
                charts.append(profile_plot)
            # Boxplot as secondary context
            group_chart = _ChartsMixin._build_group_comparison_chart(results)
            if group_chart:
                charts.append({
                    "title": "Group Comparison",
                    "subtitle": "Distribution overview with boxplots and individual observations.",
                    "html": group_chart["html"],
                    "group_order": group_chart["group_order"],
                    "div_id": "biomedstatx-group-chart",
                })
        else:
            group_chart = _ChartsMixin._build_group_comparison_chart(results)
            if group_chart:
                charts.append({
                    "title": "Group Comparison",
                    "subtitle": "Distribution overview with boxplots and individual observations.",
                    "html": group_chart["html"],
                    "group_order": group_chart["group_order"],
                    "div_id": "biomedstatx-group-chart",
                })
        # Trajectory chart for repeated/paired designs
        # Skip for RM-ANOVA and Mixed ANOVA — profile plots already incorporate trajectories
        if model_type not in ("RepeatedMeasuresANOVA", "MixedANOVA"):
            trajectory_chart = _ChartsMixin._build_trajectory_chart(results)
            if trajectory_chart:
                charts.append({
                    "title": trajectory_chart["title"],
                    "subtitle": trajectory_chart["subtitle"],
                    "html": trajectory_chart["html"],
                    "div_id": trajectory_chart["div_id"],
                })
        correlation_chart = _ChartsMixin._build_correlation_chart(results)
        if correlation_chart:
            charts.append({
                "title": str(correlation_chart.get("title") or "Association Overview"),
                "subtitle": str(correlation_chart.get("subtitle") or "Scatter-based visualization of paired variables."),
                "html": correlation_chart.get("html"),
                "div_id": correlation_chart.get("div_id"),
            })
        return charts

    @staticmethod
    def _build_group_comparison_chart(results: dict) -> str | None:
        raw_data = results.get("raw_data") or results.get("samples") or {}
        if not isinstance(raw_data, dict) or len(raw_data) == 0:
            return None
        try:
            import plotly.graph_objects as go

            def _hex_to_rgba(hex_color: str, alpha: float) -> str:
                color = str(hex_color or "").strip().lstrip("#")
                if len(color) != 6:
                    return f"rgba(15,118,110,{alpha})"
                try:
                    r = int(color[0:2], 16)
                    g = int(color[2:4], 16)
                    b = int(color[4:6], 16)
                except Exception:
                    return f"rgba(15,118,110,{alpha})"
                return f"rgba({r},{g},{b},{alpha})"

            figure = go.Figure()
            palette = ["#0f766e", "#1f7a5a", "#b7791f", "#9f3a38", "#1d4ed8", "#7c3aed"]
            group_order = []
            for idx, (group_name, values) in enumerate(raw_data.items()):
                numeric = _FormattingMixin._coerce_numeric_sequence(values)
                if not numeric:
                    continue
                group_order.append(str(group_name))
                label = f"{group_name} (n={len(numeric)})"
                color = palette[idx % len(palette)]
                figure.add_trace(
                    go.Box(
                        y=numeric,
                        name=label,
                        boxpoints="all",
                        jitter=0.45,
                        pointpos=0,
                        fillcolor=_hex_to_rgba(color, 0.18),
                        line=dict(color=color),
                        marker=dict(size=7, color=color, opacity=0.78),
                    )
                )
            if not figure.data:
                return None
            figure.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#fffdf8",
                margin=dict(l=40, r=20, t=24, b=56),
                font=dict(family="Segoe UI, Helvetica Neue, sans-serif", color="#16313a"),
                xaxis=dict(automargin=True),
                yaxis_title="Observed values",
                showlegend=False,
            )
            html = _ChartsMixin._figure_to_html(figure, div_id="biomedstatx-group-chart")
            if not html:
                return None
            return {"html": html, "group_order": group_order}
        except Exception as exc:
            logger.warning("group chart generation failed: %s", exc, exc_info=True)
            return None

    @staticmethod
    def _build_roc_chart(results: dict) -> dict | None:
        """Builds a Plotly ROC curve with diagonal reference line and AUC annotation."""
        if results.get("model_type") != "LogisticRegression":
            return None
        roc = results.get("roc_data") or {}
        fpr = roc.get("fpr") or []
        tpr = roc.get("tpr") or []
        auc = roc.get("auc")
        if len(fpr) < 2 or len(tpr) < 2:
            return None
        try:
            import plotly.graph_objects as go

            figure = go.Figure()
            # Diagonal reference line (random classifier)
            figure.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(color="#9f3a38", width=1, dash="dash"),
                name="Random classifier",
                hoverinfo="skip",
            ))
            # ROC curve
            auc_label = f"AUC = {auc:.3f}" if auc is not None else "ROC Curve"
            figure.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                line=dict(color="#0f766e", width=2.5),
                name=auc_label,
                hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
            ))
            # AUC annotation in lower-right area
            if auc is not None:
                figure.add_annotation(
                    x=0.65,
                    y=0.12,
                    text=f"<b>AUC = {auc:.3f}</b>",
                    showarrow=False,
                    font=dict(size=14, color="#0f766e"),
                    bgcolor="rgba(255,253,248,0.85)",
                    bordercolor="#0f766e",
                    borderwidth=1,
                    borderpad=6,
                )
            figure.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#fffdf8",
                margin=dict(l=50, r=20, t=24, b=56),
                font=dict(family="Segoe UI, Helvetica Neue, sans-serif", color="#16313a"),
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1]),
                showlegend=True,
                legend=dict(x=0.55, y=0.06),
            )
            html = _ChartsMixin._figure_to_html(figure, div_id="biomedstatx-roc-chart")
            if not html:
                return None
            subtitle = f"Receiver Operating Characteristic — {auc_label}" if auc is not None else "Receiver Operating Characteristic"
            return {
                "title": "ROC Curve",
                "subtitle": subtitle,
                "html": html,
                "div_id": "biomedstatx-roc-chart",
            }
        except Exception as exc:
            logger.warning("ROC chart generation failed: %s", exc, exc_info=True)
            return None

    @staticmethod
    def _build_beta_regression_chart(results: dict) -> dict | None:
        """Scatter plot of observed proportions vs primary predictor with fitted curve overlay."""
        if results.get("model_type") != "BetaRegression":
            return None
        fitted = results.get("fitted_values") or []
        xy_data = results.get("xy_data") or {}
        x_values = _FormattingMixin._coerce_numeric_sequence(xy_data.get("x"))
        y_values = _FormattingMixin._coerce_numeric_sequence(xy_data.get("y"))
        if not x_values or not y_values or len(x_values) != len(fitted):
            return None
        try:
            import plotly.graph_objects as go

            x_arr = np.array(x_values, dtype=float)
            y_arr = np.array(y_values, dtype=float)
            fitted_arr = np.array(fitted, dtype=float)
            sort_idx = np.argsort(x_arr)
            x_label = _FormattingMixin._prettify_label(xy_data.get("x_label") or "Predictor")

            figure = go.Figure()
            figure.add_trace(go.Scatter(
                x=x_arr,
                y=y_arr,
                mode="markers",
                marker=dict(size=7, color="#0f766e", opacity=0.72),
                name="Observed",
                hovertemplate=f"{x_label}: %{{x:.3f}}<br>Observed: %{{y:.3f}}<extra></extra>",
            ))
            figure.add_trace(go.Scatter(
                x=x_arr[sort_idx],
                y=fitted_arr[sort_idx],
                mode="lines",
                line=dict(color="#b7791f", width=2.5),
                name="Fitted",
                hovertemplate="Fitted: %{y:.3f}<extra></extra>",
            ))
            figure.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#fffdf8",
                margin=dict(l=50, r=20, t=24, b=56),
                font=dict(family="Segoe UI, Helvetica Neue, sans-serif", color="#16313a"),
                xaxis_title=x_label,
                yaxis_title="Proportion (outcome)",
                yaxis=dict(range=[0, 1]),
                showlegend=True,
                legend=dict(x=0.75, y=0.06),
            )
            html = _ChartsMixin._figure_to_html(figure, div_id="biomedstatx-beta-chart")
            if not html:
                return None
            return {
                "title": "Beta Regression: Observed vs. Fitted",
                "subtitle": "Proportion outcome (y-axis fixed [0, 1]). Orange line = model-fitted values.",
                "html": html,
                "div_id": "biomedstatx-beta-chart",
            }
        except Exception as exc:
            logger.warning("Beta regression chart failed: %s", exc, exc_info=True)
            return None

    @staticmethod
    def _build_correlation_matrix_charts(results: dict) -> list[dict]:
        """Builds heatmap chart(s) for CorrelationMatrix model type.

        Returns a list of chart dicts — one r-heatmap + one p-heatmap for the
        overall matrix, repeated per stratum when stratified.
        """
        if results.get("model_type") != "CorrelationMatrix":
            return []

        variables = results.get("variables") or []
        if len(variables) < 2:
            return []

        try:
            import plotly.graph_objects as go
            import math as _math
        except Exception:
            return []

        correction_map = {"fdr_bh": "FDR-corrected", "bonferroni": "Bonferroni-corrected"}
        correction = results.get("correction")
        p_label = correction_map.get(str(correction or "").lower(), "corrected") if correction else "uncorrected"

        def _make_heatmap_pair(r_mat_d: dict, p_corr_d: dict, title_prefix: str) -> list[dict]:
            """Build r-heatmap and p-heatmap for one matrix."""
            k = len(variables)
            # Build 2-D lists row=y(variable_i), col=x(variable_j)
            z_r, z_p, text_r, text_p = [], [], [], []
            for vi in variables:
                row_r, row_p, tr, tp = [], [], [], []
                for vj in variables:
                    r_val = (r_mat_d.get(vi) or {}).get(vj)
                    p_val = (p_corr_d.get(vi) or {}).get(vj)
                    rv = r_val if (r_val is not None and not _math.isnan(r_val)) else float("nan")
                    pv = p_val if (p_val is not None and not _math.isnan(p_val)) else float("nan")
                    row_r.append(rv)
                    row_p.append(pv)
                    tr.append(f"{rv:.2f}" if not _math.isnan(rv) else "")
                    tp.append(f"{pv:.3f}" if not _math.isnan(pv) else "")
                z_r.append(row_r)
                z_p.append(row_p)
                text_r.append(tr)
                text_p.append(tp)

            var_labels = [_FormattingMixin._prettify_label(v) for v in variables]
            cell_px = max(55, min(90, 700 // k))
            fig_h = max(350, k * cell_px + 120)

            # --- Chart 1: r-value heatmap ---
            # Build per-cell annotations coloured by significance
            annots_r = []
            for i, vi in enumerate(variables):
                for j, vj in enumerate(variables):
                    r_val = (r_mat_d.get(vi) or {}).get(vj)
                    p_val = (p_corr_d.get(vi) or {}).get(vj)
                    is_diag = vi == vj
                    is_sig = (
                        not is_diag
                        and isinstance(p_val, (int, float))
                        and not _math.isnan(p_val)
                        and p_val < 0.05
                    )
                    if r_val is not None and not _math.isnan(r_val):
                        text = f"<b>{r_val:.2f}</b>" if is_sig else f"{r_val:.2f}"
                    else:
                        text = ""
                    font_color = "#111111" if is_sig else "#aaaaaa"
                    annots_r.append(dict(
                        x=var_labels[j],
                        y=var_labels[i],
                        text=text,
                        font=dict(size=11, color=font_color),
                        showarrow=False,
                    ))

            fig_r = go.Figure(go.Heatmap(
                z=z_r,
                x=var_labels,
                y=var_labels,
                zmin=-1, zmax=1,
                colorscale=[
                    [0.0, "#2166ac"],
                    [0.25, "#92c5de"],
                    [0.5, "#f7f7f7"],
                    [0.75, "#f4a582"],
                    [1.0, "#d6604d"],
                ],
                showscale=True,
                colorbar=dict(title="r", thickness=14, len=0.8),
                hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>r = %{z:.3f}<extra></extra>",
            ))
            fig_r.update_layout(
                annotations=annots_r,
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#fffdf8",
                height=fig_h,
                margin=dict(l=20, r=20, t=36, b=20),
                font=dict(family="Segoe UI, Helvetica Neue, sans-serif", size=12, color="#16313a"),
                xaxis=dict(side="bottom", tickangle=-35),
                yaxis=dict(autorange="reversed"),
            )
            div_r = f"biomedstatx-corrmat-r-{title_prefix.replace(' ', '-').lower()}"
            html_r = _ChartsMixin._figure_to_html(fig_r, div_id=div_r)

            # --- Chart 2: p-value heatmap ---
            annots_p = []
            for i, vi in enumerate(variables):
                for j, vj in enumerate(variables):
                    p_val = (p_corr_d.get(vi) or {}).get(vj)
                    is_diag = vi == vj
                    if is_diag:
                        text = "—"
                        font_color = "#cccccc"
                    elif p_val is not None and not _math.isnan(p_val):
                        text = f"{p_val:.3f}"
                        font_color = "#111111" if p_val < 0.05 else "#888888"
                    else:
                        text = ""
                        font_color = "#aaaaaa"
                    annots_p.append(dict(
                        x=var_labels[j],
                        y=var_labels[i],
                        text=text,
                        font=dict(size=10, color=font_color),
                        showarrow=False,
                    ))

            # Clamp diagonal NaN to 1.0 for display (diagonal has no p-value)
            z_p_display = []
            for i, vi in enumerate(variables):
                row = []
                for j, vj in enumerate(variables):
                    p_val = z_p[i][j]
                    if vi == vj:
                        row.append(1.0)
                    elif not _math.isnan(p_val):
                        row.append(p_val)
                    else:
                        row.append(float("nan"))
                z_p_display.append(row)

            fig_p = go.Figure(go.Heatmap(
                z=z_p_display,
                x=var_labels,
                y=var_labels,
                zmin=0, zmax=1,
                colorscale=[
                    [0.0, "#1a7340"],
                    [0.05, "#52b788"],
                    [0.1, "#b7e4c7"],
                    [0.5, "#f0f0f0"],
                    [1.0, "#ffffff"],
                ],
                showscale=True,
                colorbar=dict(title="p", thickness=14, len=0.8),
                hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>p = %{z:.4f}<extra></extra>",
            ))
            fig_p.update_layout(
                annotations=annots_p,
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#fffdf8",
                height=fig_h,
                margin=dict(l=20, r=20, t=36, b=20),
                font=dict(family="Segoe UI, Helvetica Neue, sans-serif", size=12, color="#16313a"),
                xaxis=dict(side="bottom", tickangle=-35),
                yaxis=dict(autorange="reversed"),
            )
            div_p = f"biomedstatx-corrmat-p-{title_prefix.replace(' ', '-').lower()}"
            html_p = _ChartsMixin._figure_to_html(fig_p, div_id=div_p)

            out = []
            if html_r:
                out.append({
                    "title": f"Correlation Matrix (r values){' — ' + title_prefix if title_prefix else ''}",
                    "subtitle": f"Significance based on {p_label} p-values. Bold black = significant (p < 0.05). Grey = non-significant.",
                    "html": html_r,
                    "div_id": div_r,
                })
            if html_p:
                out.append({
                    "title": f"FDR-corrected p-values{' — ' + title_prefix if title_prefix else ''}",
                    "subtitle": f"Darker green = smaller p-value. Dark cells (p < 0.05) indicate significant correlations.",
                    "html": html_p,
                    "div_id": div_p,
                })
            return out

        charts = []
        try:
            # Overall matrix
            r_mat = results.get("r_matrix") or {}
            pc_mat = results.get("p_corrected_matrix") or {}
            charts.extend(_make_heatmap_pair(r_mat, pc_mat, ""))
        except Exception as exc:
            logger.warning("CorrelationMatrix overall heatmap failed: %s", exc, exc_info=True)

        # Stratified matrices
        strata = results.get("strata") or {}
        for stratum_name, stratum_data in strata.items():
            try:
                r_s = stratum_data.get("r_matrix") or {}
                pc_s = stratum_data.get("p_corrected_matrix") or {}
                charts.extend(_make_heatmap_pair(r_s, pc_s, str(stratum_name)))
            except Exception as exc:
                logger.warning("CorrelationMatrix stratum %r heatmap failed: %s", stratum_name, exc, exc_info=True)

        return charts

    @staticmethod
    def _build_correlation_chart(results: dict, div_id: str = "biomedstatx-assoc-chart") -> dict | None:
        model_type = str(results.get("model_type") or "")
        if model_type not in {"Correlation", "LinearRegression"}:
            return None
        try:
            import plotly.graph_objects as go

            payload = _AssociationMixin._extract_association_payload(results)
            if payload is None:
                return None

            x_values = payload.get("x_values") or []
            y_values = payload.get("y_values") or []
            if len(x_values) < 2 or len(y_values) < 2:
                return None

            figure = go.Figure()

            fit_x = payload.get("fit_x") or []
            fit_y = payload.get("fit_y") or []
            fit_ci_lower = payload.get("fit_ci_lower") or []
            fit_ci_upper = payload.get("fit_ci_upper") or []

            if fit_x and fit_y and len(fit_x) == len(fit_y):
                has_band = (
                    fit_ci_lower
                    and fit_ci_upper
                    and len(fit_ci_lower) == len(fit_x)
                    and len(fit_ci_upper) == len(fit_x)
                )
                if has_band:
                    figure.add_trace(
                        go.Scatter(
                            x=fit_x,
                            y=fit_ci_lower,
                            mode="lines",
                            line=dict(width=0),
                            hoverinfo="skip",
                            showlegend=False,
                            name="95% CI lower",
                        )
                    )
                    figure.add_trace(
                        go.Scatter(
                            x=fit_x,
                            y=fit_ci_upper,
                            mode="lines",
                            line=dict(width=0),
                            fill="tonexty",
                            fillcolor="rgba(15,118,110,0.14)",
                            hoverinfo="skip",
                            showlegend=True,
                            name="95% CI",
                        )
                    )

                figure.add_trace(
                    go.Scatter(
                        x=fit_x,
                        y=fit_y,
                        mode="lines",
                        line=dict(width=2.2, color="#0f766e"),
                        name="Trend",
                        showlegend=True,
                    )
                )

            figure.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="markers",
                    name="Observed",
                    marker=dict(size=8, color="#0f766e", opacity=0.82, line=dict(width=1, color="#16313a")),
                )
            )

            figure.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#fffdf8",
                margin=dict(l=40, r=20, t=24, b=40),
                font=dict(family="Segoe UI, Helvetica Neue, sans-serif", color="#16313a"),
                xaxis_title=str(payload.get("x_label") or "X"),
                yaxis_title=str(payload.get("y_label") or "Y"),
                showlegend=True,
                legend=dict(orientation="h", x=0.01, y=1.08),
            )
            html = _ChartsMixin._figure_to_html(figure, div_id=div_id)
            if not html:
                return None
            corr_method = str(results.get("method") or "").lower()
            if model_type == "LinearRegression":
                chart_title = "Regression Overview"
                chart_subtitle = "Observed values with OLS trend estimate and 95% confidence band."
            elif corr_method == "spearman":
                chart_title = "Association Overview"
                chart_subtitle = "Spearman correlation — no parametric trend line shown."
            else:
                chart_title = "Association Overview"
                chart_subtitle = "Observed values with trend estimate and 95% confidence band."
            return {
                "title": chart_title,
                "subtitle": chart_subtitle,
                "html": html,
                "div_id": "biomedstatx-assoc-chart",
            }
        except Exception as exc:
            logger.warning("association chart generation failed: %s", exc, exc_info=True)
            return None

    @staticmethod
    def _build_ancova_chart(results: dict) -> dict | None:
        if results.get("model_type") != "ANCOVA":
            return None
        adjusted_means = results.get("adjusted_means") or {}
        covariates_used = results.get("covariates_used") or []
        if not adjusted_means:
            return None
        try:
            import plotly.graph_objects as go

            figure = go.Figure()
            palette = ["#0f766e", "#1f7a5a", "#b7791f", "#9f3a38", "#1d4ed8", "#7c3aed"]

            for _factor, levels in adjusted_means.items():
                if not isinstance(levels, dict):
                    continue
                group_labels, adj_means_vals, raw_sds, ns = [], [], [], []
                for level, stats in levels.items():
                    if not isinstance(stats, dict):
                        continue
                    adj_mean = stats.get("adjusted_mean")
                    if adj_mean is None:
                        continue
                    group_labels.append(str(level))
                    adj_means_vals.append(float(adj_mean))
                    raw_sds.append(float(stats.get("raw_sd") or 0))
                    ns.append(int(stats.get("n") or 0))

                for i, (label, mean, sd, n) in enumerate(zip(group_labels, adj_means_vals, raw_sds, ns)):
                    color = palette[i % len(palette)]
                    figure.add_trace(go.Bar(
                        x=[label],
                        y=[mean],
                        name=f"{label} (n={n})",
                        error_y=dict(type="data", array=[sd], visible=True, color=color),
                        marker_color=color,
                        marker_opacity=0.82,
                        width=0.45,
                    ))

            if not figure.data:
                return None

            cov_str = ", ".join(covariates_used) if covariates_used else "none"
            figure.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#fffdf8",
                margin=dict(l=48, r=20, t=24, b=56),
                font=dict(family="Segoe UI, Helvetica Neue, sans-serif", color="#16313a"),
                yaxis_title="Outcome",
                xaxis_title="Group",
                showlegend=True,
                legend=dict(orientation="h", x=0.01, y=1.08),
            )
            html = _ChartsMixin._figure_to_html(figure, div_id="biomedstatx-ancova-chart")
            if not html:
                return None
            return {
                "title": "Adjusted Group Means",
                "subtitle": f"Estimated marginal means ± SD. Adjusted for: {cov_str}.",
                "html": html,
                "div_id": "biomedstatx-ancova-chart",
            }
        except Exception as exc:
            logger.warning("ANCOVA chart generation failed: %s", exc, exc_info=True)
            return None

    @staticmethod
    def _figure_to_html(figure, div_id: str | None = None) -> str | None:
        try:
            import plotly.io as pio

            kwargs = {"div_id": div_id} if div_id else {}
            return pio.to_html(
                figure,
                full_html=False,
                include_plotlyjs=False,
                config={"responsive": True, "displayModeBar": False},
                default_width="100%",
                **kwargs,
            )
        except Exception as exc:
            logger.warning("figure HTML generation failed: %s", exc, exc_info=True)
            return None

    @staticmethod
    def _build_significance_brackets(figure, results: dict, group_order: list) -> None:
        """Add significance bracket annotations (*, **, ***) to a Plotly group comparison figure."""
        try:
            pairwise = results.get("pairwise_comparisons") or []
            sig_pairs = [p for p in pairwise if p.get("significant")]
            if not sig_pairs:
                return
            group_to_idx = {name: i for i, name in enumerate(group_order)}
            y_vals = []
            for trace in figure.data:
                if hasattr(trace, "y") and trace.y is not None:
                    y_vals.extend(v for v in trace.y if v is not None)
            if not y_vals:
                return
            y_min, y_max = min(y_vals), max(y_vals)
            y_range = max(abs(y_max - y_min), 1e-9)
            step = y_range * 0.13
            tick = step * 0.28
            brackets = []
            for pair in sig_pairs:
                g1 = pair.get("group1") or pair.get("comparison", "").split(" vs ")[0].strip()
                g2 = pair.get("group2") or pair.get("comparison", "").split(" vs ")[-1].strip()
                i1, i2 = group_to_idx.get(str(g1)), group_to_idx.get(str(g2))
                if i1 is None or i2 is None:
                    continue
                if i1 > i2:
                    i1, i2 = i2, i1
                p_val = pair.get("p_value")
                if not isinstance(p_val, (int, float)):
                    continue
                stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*"
                brackets.append((i1, i2, stars, i2 - i1))
            brackets.sort(key=lambda b: (b[3], b[0]))
            line_style = dict(color="rgba(22,49,58,0.65)", width=1.5)
            for level, (i1, i2, stars, _) in enumerate(brackets):
                y = y_max + step * (level + 1)
                figure.add_shape(type="line", x0=i1, x1=i2, y0=y, y1=y, xref="x", yref="y", line=line_style)
                figure.add_shape(type="line", x0=i1, x1=i1, y0=y - tick, y1=y, xref="x", yref="y", line=line_style)
                figure.add_shape(type="line", x0=i2, x1=i2, y0=y - tick, y1=y, xref="x", yref="y", line=line_style)
                figure.add_annotation(
                    x=(i1 + i2) / 2, y=y, text=f"<b>{stars}</b>", showarrow=False,
                    xref="x", yref="y", yshift=8,
                    font=dict(size=13, color="#16313a"),
                )
            figure.update_yaxes(range=[y_min - y_range * 0.05, y_max + step * (len(brackets) + 1.8)])
        except Exception as exc:
            logger.warning("significance brackets failed: %s", exc, exc_info=True)

    @staticmethod
    def _build_plot_subject_trajectories(results: dict, group_order: list[str], plot_data: dict) -> list[dict]:
        raw_trajectories = results.get("plot_subject_trajectories") or []
        if not isinstance(raw_trajectories, list):
            return []

        allowed_groups = set(group_order or list((plot_data or {}).keys()))
        group_rank = {group: idx for idx, group in enumerate(group_order or [])}

        normalized = []
        for idx, trajectory in enumerate(raw_trajectories):
            if not isinstance(trajectory, dict):
                continue
            subject_id = str(trajectory.get("subject_id") or trajectory.get("subject") or f"S{idx + 1}")
            points_raw = trajectory.get("points") or []
            if not isinstance(points_raw, list):
                continue

            points = []
            for point in points_raw:
                if not isinstance(point, dict):
                    continue
                group_name = str(point.get("group") or point.get("condition") or "")
                if not group_name:
                    continue
                if allowed_groups and group_name not in allowed_groups:
                    continue
                try:
                    numeric_value = float(point.get("value"))
                    if math.isnan(numeric_value) or math.isinf(numeric_value):
                        continue
                except Exception:
                    continue
                points.append({"group": group_name, "value": numeric_value})

            if len(points) < 2:
                continue

            points.sort(key=lambda item: (group_rank.get(item["group"], 10_000), item["group"]))
            normalized.append({"subject_id": subject_id, "points": points})

        if len(normalized) > 2000:
            normalized = normalized[:2000]
        return normalized

    @staticmethod
    def _build_plot_reference_lines(results: dict) -> list[dict]:
        raw_lines = []
        for key in ("thresholds", "plot_thresholds", "reference_lines"):
            candidate = results.get(key)
            if isinstance(candidate, list):
                raw_lines.extend(candidate)

        normalized = []
        for index, line in enumerate(raw_lines):
            value = None
            label = None
            dash = "dash"
            color = "rgba(159,58,56,0.82)"
            width = 1.5

            if isinstance(line, (int, float)) and np.isfinite(float(line)):
                value = float(line)
                label = f"Threshold {index + 1}"
            elif isinstance(line, dict):
                for key in ("value", "y", "threshold"):
                    candidate = line.get(key)
                    try:
                        numeric = float(candidate)
                    except Exception:
                        continue
                    if np.isfinite(numeric):
                        value = numeric
                        break

                if value is None:
                    continue

                raw_label = line.get("label") or line.get("name")
                label = str(raw_label).strip() if raw_label is not None else ""
                if not label:
                    label = f"Threshold {index + 1}"

                raw_dash = str(line.get("dash") or "dash").strip().lower()
                if raw_dash in {"solid", "dash", "dot", "dashdot"}:
                    dash = raw_dash

                raw_color = line.get("color")
                if isinstance(raw_color, str) and raw_color.strip():
                    color = raw_color.strip()

                raw_width = line.get("width")
                try:
                    width_candidate = float(raw_width)
                    if np.isfinite(width_candidate):
                        width = max(0.6, min(4.0, width_candidate))
                except (TypeError, ValueError):
                    logger.debug("line-overlay width=%r not numeric; keeping default", raw_width)
            else:
                continue

            normalized.append({
                "value": float(value),
                "label": label,
                "dash": dash,
                "color": color,
                "width": width,
            })

        if len(normalized) > 30:
            normalized = normalized[:30]
        return normalized
