import base64
import copy
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from jinja2 import Environment
from scipy import stats


class _ResultsEncoder(json.JSONEncoder):
    def default(self, obj):
        return HTMLExporter._normalize_for_json(obj)


class HTMLExporter:
    @staticmethod
    def export_results_to_html(results: dict, output_file: str, analysis_log=None) -> str | None:
        try:
            output_path = Path(output_file).resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            context = HTMLExporter._prepare_single_report_context(results, analysis_log=analysis_log)
            html = HTMLExporter._render_template(context, mode="single")
            with open(output_path, "w", encoding="utf-8") as handle:
                handle.write(html)
            return str(output_path)
        except Exception as exc:
            print(f"WARNING HTML EXPORT: Failed to write single report to '{output_file}': {exc}")
            return None

    @staticmethod
    def export_multi_dataset_results_to_html(all_results: dict, output_file: str) -> str | None:
        try:
            output_path = Path(output_file).resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            context = HTMLExporter._prepare_multi_report_context(all_results)
            html = HTMLExporter._render_template(context, mode="multi")
            with open(output_path, "w", encoding="utf-8") as handle:
                handle.write(html)
            return str(output_path)
        except Exception as exc:
            print(f"WARNING HTML EXPORT: Failed to write multi report to '{output_file}': {exc}")
            return None

    @staticmethod
    def _normalize_for_json(value: Any):
        if isinstance(value, pd.DataFrame):
            normalized = value.replace({np.nan: None})
            return {
                "columns": [str(col) for col in normalized.columns],
                "data": [
                    [HTMLExporter._normalize_for_json(cell) for cell in row]
                    for row in normalized.itertuples(index=False, name=None)
                ],
            }
        if isinstance(value, pd.Series):
            return [HTMLExporter._normalize_for_json(item) for item in value.tolist()]
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return HTMLExporter._normalize_for_json(value.item())
        if isinstance(value, np.ndarray):
            return [HTMLExporter._normalize_for_json(item) for item in value.tolist()]
        if isinstance(value, dict):
            return {
                str(key): HTMLExporter._normalize_for_json(val)
                for key, val in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [HTMLExporter._normalize_for_json(item) for item in value]
        if isinstance(value, float):
            if math.isnan(value):
                return None
            if math.isinf(value):
                return "Infinity" if value > 0 else "-Infinity"
            return value
        if isinstance(value, (str, bool, int)) or value is None:
            return value
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        return str(value)

    @staticmethod
    def _prepare_single_report_context(results: dict, analysis_log=None) -> dict:
        results_copy = copy.deepcopy(results or {})
        normalized = HTMLExporter._normalize_for_json(results_copy)
        analysis_log_text = analysis_log if analysis_log is not None else results_copy.get("analysis_log", "")
        hero = HTMLExporter._build_hero_context(results_copy)
        metrics = HTMLExporter._build_statistical_rows(results_copy)
        assumptions = HTMLExporter._build_assumption_summary(results_copy)
        descriptive = HTMLExporter._build_descriptive_summary(results_copy)
        pairwise = HTMLExporter._build_pairwise_rows(results_copy)
        raw_table = HTMLExporter._build_raw_data_table(results_copy)
        charts = HTMLExporter._build_single_chart_bundle(results_copy)
        decision_tree_image = HTMLExporter._embed_decision_tree(results_copy)
        decision_path = HTMLExporter._build_decision_path_model(results_copy)
        methods_text = HTMLExporter._build_methods_text(results_copy, analysis_log_text)
        return {
            "mode": "single",
            "report_title": hero["title"],
            "subtitle": hero["subtitle"],
            "hero": hero,
            "decision_path": decision_path,
            "decision_tree_image": decision_tree_image,
            "statistical_rows": metrics,
            "assumptions": assumptions,
            "descriptive": descriptive,
            "pairwise_rows": pairwise,
            "raw_data_table": raw_table,
            "chart_blocks": charts,
            "methods_text": methods_text,
            "generated_warning": results_copy.get("error"),
            "normalized_results_json": json.dumps(normalized, cls=_ResultsEncoder, ensure_ascii=False),
        }

    @staticmethod
    def _prepare_multi_report_context(all_results: dict) -> dict:
        cards = []
        significant_count = 0
        for dataset_name, results in (all_results or {}).items():
            hero = HTMLExporter._build_hero_context(results or {}, dataset_name=str(dataset_name))
            assumptions = HTMLExporter._build_assumption_summary(results or {})
            cards.append({
                "dataset_name": str(dataset_name),
                "test_name": hero["test_name"],
                "p_value_display": hero["p_value_display"],
                "significance_label": hero["significance_label"],
                "significance_class": hero["significance_class"],
                "transformation": str((results or {}).get("transformation") or "None"),
                "pairwise_count": len((results or {}).get("pairwise_comparisons") or []),
                "summary_note": hero["summary_note"],
                "assumptions": assumptions,
            })
            if hero["is_significant"]:
                significant_count += 1
        return {
            "mode": "multi",
            "report_title": "BioMedStatX Multi-Dataset Scientific Report",
            "subtitle": f"{len(cards)} datasets summarized, {significant_count} significant main results.",
            "dataset_cards": cards,
            "generated_warning": None,
        }

    @staticmethod
    def _build_hero_context(results: dict, dataset_name: str | None = None) -> dict:
        test_name = str(
            results.get("final_test_label")
            or results.get("tested_against")
            or results.get("test")
            or results.get("model_type")
            or "Statistical analysis"
        )
        p_value = results.get("p_value")
        is_significant = isinstance(p_value, (int, float)) and p_value < 0.05
        effect_size = results.get("effect_size")
        effect_label = str(results.get("effect_size_type") or "Effect size")
        title = f"Scientific Report: {dataset_name}" if dataset_name else "BioMedStatX Scientific Report"
        subtitle = dataset_name if dataset_name else str(results.get("dataset_name") or test_name)
        return {
            "title": title,
            "subtitle": subtitle,
            "test_name": test_name,
            "p_value_display": HTMLExporter._format_p_value(p_value),
            "is_significant": is_significant,
            "significance_label": "Significant" if is_significant else "Not significant",
            "significance_class": "is-significant" if is_significant else "is-neutral",
            "effect_size_display": HTMLExporter._format_metric(effect_size),
            "effect_label": effect_label,
            "summary_note": HTMLExporter._build_summary_note(results, test_name, p_value),
            "alpha_display": HTMLExporter._format_metric(results.get("alpha"), digits=3),
        }

    @staticmethod
    def _build_summary_note(results: dict, test_name: str, p_value: Any) -> str:
        if results.get("error"):
            return str(results["error"])
        if isinstance(p_value, (int, float)):
            if p_value < 0.05:
                return f"{test_name} detected evidence against the null hypothesis."
            return f"{test_name} did not show evidence against the null hypothesis."
        return f"{test_name} completed without a numeric p-value."

    @staticmethod
    def _build_statistical_rows(results: dict) -> list[dict]:
        rows = []
        for label, key in [
            ("Test", "test"),
            ("Model type", "model_type"),
            ("Statistic", "statistic"),
            ("p-value", "p_value"),
            ("Adjusted p-value", "p_value_fdr"),
            ("Effect size", "effect_size"),
            ("Effect size type", "effect_size_type"),
            ("Confidence interval", "confidence_interval"),
            ("Degrees of freedom 1", "df1"),
            ("Degrees of freedom 2", "df2"),
            ("Transformation", "transformation"),
            ("Post-hoc test", "posthoc_test"),
        ]:
            value = results.get(key)
            if key in results and HTMLExporter._has_display_value(value):
                if key.startswith("p_value"):
                    display = HTMLExporter._format_p_value(value)
                elif key == "confidence_interval":
                    display = HTMLExporter._format_confidence_interval(value)
                else:
                    display = HTMLExporter._format_metric(value)
                rows.append({"label": label, "value": display})
        for factor in results.get("factors", []) or []:
            factor_name = factor.get("factor", "Factor")
            factor_value = (
                f"F({HTMLExporter._format_metric(factor.get('df1'))}, "
                f"{HTMLExporter._format_metric(factor.get('df2'))}) = {HTMLExporter._format_metric(factor.get('F'))}, "
                f"{HTMLExporter._format_p_value(factor.get('p_value'))}"
            )
            rows.append({"label": f"Main effect: {factor_name}", "value": factor_value})
        for interaction in results.get("interactions", []) or []:
            factors = interaction.get("factors") or ["Interaction"]
            label = " x ".join(map(str, factors))
            value = (
                f"F({HTMLExporter._format_metric(interaction.get('df1'))}, "
                f"{HTMLExporter._format_metric(interaction.get('df2'))}) = {HTMLExporter._format_metric(interaction.get('F'))}, "
                f"{HTMLExporter._format_p_value(interaction.get('p_value'))}"
            )
            rows.append({"label": f"Interaction: {label}", "value": value})
        if not rows:
            rows.append({"label": "Status", "value": "No structured statistical summary available."})
        return rows

    @staticmethod
    def _build_assumption_summary(results: dict) -> dict:
        rows = []
        normality_tests = results.get("normality_tests", {}) or {}
        for label, payload in normality_tests.items():
            if not isinstance(payload, dict):
                continue
            rows.append({
                "name": f"Normality: {HTMLExporter._prettify_label(label)}",
                "statistic": HTMLExporter._format_metric(payload.get("statistic")),
                "p_value": HTMLExporter._format_p_value(payload.get("p_value")),
                "status_label": HTMLExporter._bool_label(payload.get("is_normal")),
                "status_class": HTMLExporter._bool_class(payload.get("is_normal")),
            })
        variance_test = results.get("variance_test", {}) or {}
        if isinstance(variance_test, dict) and variance_test:
            rows.append({
                "name": "Variance homogeneity",
                "statistic": HTMLExporter._format_metric(variance_test.get("statistic")),
                "p_value": HTMLExporter._format_p_value(variance_test.get("p_value")),
                "status_label": HTMLExporter._bool_label(variance_test.get("equal_variance")),
                "status_class": HTMLExporter._bool_class(variance_test.get("equal_variance")),
            })
            transformed = variance_test.get("transformed")
            if isinstance(transformed, dict):
                rows.append({
                    "name": "Variance homogeneity (transformed)",
                    "statistic": HTMLExporter._format_metric(transformed.get("statistic")),
                    "p_value": HTMLExporter._format_p_value(transformed.get("p_value")),
                    "status_label": HTMLExporter._bool_label(transformed.get("equal_variance")),
                    "status_class": HTMLExporter._bool_class(transformed.get("equal_variance")),
                })
        sphericity = results.get("sphericity_test", {}) or {}
        if isinstance(sphericity, dict) and sphericity:
            status_value = sphericity.get("sphericity_met")
            if status_value is None and sphericity.get("p_value") is not None:
                status_value = sphericity.get("p_value") >= 0.05
            rows.append({
                "name": "Sphericity",
                "statistic": HTMLExporter._format_metric(sphericity.get("W") or sphericity.get("statistic")),
                "p_value": HTMLExporter._format_p_value(sphericity.get("p_value")),
                "status_label": HTMLExporter._bool_label(status_value),
                "status_class": HTMLExporter._bool_class(status_value),
            })
        return {
            "rows": rows,
            "transformation": str(results.get("transformation") or "None"),
            "interpretation": HTMLExporter._build_assumption_interpretation(results, rows),
            "qq_plot_html": HTMLExporter._build_assumption_visuals(results),
            "distribution_plot_html": HTMLExporter._build_distribution_dashboard_chart(results),
            "residual_plot_html": HTMLExporter._build_residuals_vs_fitted_chart(results),
        }

    @staticmethod
    def _build_descriptive_summary(results: dict) -> dict:
        raw_data = results.get("raw_data") or results.get("samples") or {}
        transformed = results.get("raw_data_transformed") or results.get("transformed_data") or {}
        rows = []
        if raw_data:
            for group_name, values in raw_data.items():
                numeric = HTMLExporter._coerce_numeric_sequence(values)
                if not numeric:
                    continue
                rows.append({
                    "group": str(group_name),
                    "n": len(numeric),
                    "mean": HTMLExporter._format_metric(np.mean(numeric)),
                    "median": HTMLExporter._format_metric(np.median(numeric)),
                    "sd": HTMLExporter._format_metric(np.std(numeric, ddof=1) if len(numeric) > 1 else None),
                    "sem": HTMLExporter._format_metric(stats.sem(numeric) if len(numeric) > 1 else None),
                    "min": HTMLExporter._format_metric(np.min(numeric)),
                    "max": HTMLExporter._format_metric(np.max(numeric)),
                })
        if not rows and results.get("descriptive"):
            for group_name, payload in (results.get("descriptive") or {}).items():
                if not isinstance(payload, dict):
                    continue
                rows.append({
                    "group": str(group_name),
                    "n": HTMLExporter._format_metric(payload.get("n")),
                    "mean": HTMLExporter._format_metric(payload.get("mean")),
                    "median": HTMLExporter._format_metric(payload.get("median")),
                    "sd": HTMLExporter._format_metric(payload.get("sd") or payload.get("std")),
                    "sem": HTMLExporter._format_metric(payload.get("sem")),
                    "min": HTMLExporter._format_metric(payload.get("min")),
                    "max": HTMLExporter._format_metric(payload.get("max")),
                })
        return {
            "rows": rows,
            "has_transformed": bool(transformed and transformed != raw_data),
        }

    @staticmethod
    def _build_pairwise_rows(results: dict) -> list[dict]:
        rows = []
        for comp in results.get("pairwise_comparisons", []) or []:
            rows.append({
                "comparison": f"{comp.get('group1', 'Group 1')} vs {comp.get('group2', 'Group 2')}",
                "test": str(comp.get("test") or results.get("posthoc_test") or "Pairwise comparison"),
                "statistic": HTMLExporter._format_metric(comp.get("statistic")),
                "p_value": HTMLExporter._format_p_value(comp.get("p_value")),
                "effect_size": HTMLExporter._format_metric(comp.get("effect_size")),
                "significant": bool(comp.get("significant")),
                "row_class": "is-significant" if comp.get("significant") else "is-neutral",
            })
        return rows

    @staticmethod
    def _build_raw_data_table(results: dict) -> dict:
        raw_data = results.get("raw_data") or results.get("samples") or {}
        transformed = results.get("raw_data_transformed") or results.get("transformed_data") or {}
        rows = []
        if isinstance(raw_data, dict):
            for group_name, values in raw_data.items():
                raw_values = list(values) if values is not None else []
                transformed_source = transformed.get(group_name, []) if isinstance(transformed, dict) else []
                transformed_values = list(transformed_source) if transformed_source is not None else []
                max_len = max(len(raw_values), len(transformed_values), 1)
                for index in range(max_len):
                    raw_value = raw_values[index] if index < len(raw_values) else None
                    transformed_value = transformed_values[index] if index < len(transformed_values) else None
                    rows.append({
                        "group": str(group_name),
                        "index": index + 1,
                        "raw_value": HTMLExporter._format_metric(raw_value, digits=6),
                        "transformed_value": HTMLExporter._format_metric(transformed_value, digits=6),
                    })
        return {
            "rows": rows,
            "has_transformed": any(row["transformed_value"] != "N/A" for row in rows),
        }

    @staticmethod
    def _build_single_chart_bundle(results: dict) -> list[dict]:
        charts = []
        group_chart = HTMLExporter._build_group_comparison_chart(results)
        if group_chart:
            charts.append({
                "title": "Group Comparison",
                "subtitle": "Distribution overview with boxplots and individual observations.",
                "html": group_chart,
            })
        correlation_chart = HTMLExporter._build_correlation_chart(results)
        if correlation_chart:
            charts.append({
                "title": "Association Overview",
                "subtitle": "Scatter-based visualization of paired variables.",
                "html": correlation_chart,
            })
        return charts

    @staticmethod
    def _build_group_comparison_chart(results: dict) -> str | None:
        raw_data = results.get("raw_data") or results.get("samples") or {}
        if not isinstance(raw_data, dict) or len(raw_data) == 0:
            return None
        try:
            import plotly.graph_objects as go
            import plotly.io as pio

            figure = go.Figure()
            for group_name, values in raw_data.items():
                numeric = HTMLExporter._coerce_numeric_sequence(values)
                if not numeric:
                    continue
                figure.add_trace(
                    go.Box(
                        y=numeric,
                        name=str(group_name),
                        boxpoints="all",
                        jitter=0.45,
                        pointpos=0,
                        fillcolor="rgba(15,118,110,0.18)",
                        line=dict(color="#16313a"),
                        marker=dict(size=7, color="#0f766e", opacity=0.78),
                    )
                )
            if not figure.data:
                return None
            figure.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#fffdf8",
                margin=dict(l=40, r=20, t=24, b=40),
                font=dict(family="Segoe UI, Helvetica Neue, sans-serif", color="#16313a"),
                yaxis_title="Observed values",
                xaxis_title="Groups",
                showlegend=False,
            )
            return HTMLExporter._figure_to_html(figure)
        except Exception as exc:
            print(f"WARNING HTML EXPORT: group chart generation failed: {exc}")
            return None

    @staticmethod
    def _build_correlation_chart(results: dict) -> str | None:
        if str(results.get("model_type")) != "Correlation":
            return None
        raw_data = results.get("raw_data") or {}
        if not isinstance(raw_data, dict) or len(raw_data) < 2:
            return None
        try:
            import plotly.graph_objects as go
            import plotly.io as pio

            names = list(raw_data.keys())[:2]
            x_values = HTMLExporter._coerce_numeric_sequence(raw_data.get(names[0], []))
            y_values = HTMLExporter._coerce_numeric_sequence(raw_data.get(names[1], []))
            if not x_values or not y_values:
                return None
            paired_length = min(len(x_values), len(y_values))
            figure = go.Figure(
                data=[
                    go.Scatter(
                        x=x_values[:paired_length],
                        y=y_values[:paired_length],
                        mode="markers",
                        marker=dict(size=9, color="#0f766e", opacity=0.8, line=dict(width=1, color="#16313a")),
                    )
                ]
            )
            figure.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#fffdf8",
                margin=dict(l=40, r=20, t=24, b=40),
                font=dict(family="Segoe UI, Helvetica Neue, sans-serif", color="#16313a"),
                xaxis_title=str(names[0]),
                yaxis_title=str(names[1]),
                showlegend=False,
            )
            return HTMLExporter._figure_to_html(figure)
        except Exception as exc:
            print(f"WARNING HTML EXPORT: correlation chart generation failed: {exc}")
            return None

    @staticmethod
    def _build_assumption_visuals(results: dict) -> str | None:
        values = None
        if "model_residuals" in results:
            values = HTMLExporter._coerce_numeric_sequence(results.get("model_residuals"))
        elif "residuals" in results:
            values = HTMLExporter._coerce_numeric_sequence(results.get("residuals"))
        else:
            raw_data = results.get("raw_data") or results.get("samples") or {}
            if isinstance(raw_data, dict):
                combined = []
                for group_values in raw_data.values():
                    combined.extend(HTMLExporter._coerce_numeric_sequence(group_values))
                values = combined
        if not values or len(values) < 3:
            return None
        try:
            import plotly.graph_objects as go
            import plotly.io as pio

            osm, osr = stats.probplot(values, dist="norm", fit=False)
            slope, intercept, _, _, _ = stats.linregress(osm, osr)
            line_x = np.linspace(min(osm), max(osm), 50)
            line_y = slope * line_x + intercept
            figure = go.Figure()
            figure.add_trace(
                go.Scatter(
                    x=osm,
                    y=osr,
                    mode="markers",
                    marker=dict(size=7, color="#0f766e", opacity=0.82),
                    name="Observed",
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=line_x,
                    y=line_y,
                    mode="lines",
                    line=dict(color="#9f3a38", width=2),
                    name="Reference",
                )
            )
            figure.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#fffdf8",
                margin=dict(l=40, r=20, t=24, b=40),
                font=dict(family="Segoe UI, Helvetica Neue, sans-serif", color="#16313a"),
                xaxis_title="Theoretical quantiles",
                yaxis_title="Observed quantiles",
                legend=dict(orientation="h", y=1.08, x=0),
            )
            return HTMLExporter._figure_to_html(figure)
        except Exception as exc:
            print(f"WARNING HTML EXPORT: QQ plot generation failed: {exc}")
            return None

    @staticmethod
    def _build_distribution_dashboard_chart(results: dict) -> str | None:
        raw_data = results.get("raw_data") or results.get("samples") or {}
        if not isinstance(raw_data, dict) or not raw_data:
            return None
        try:
            import plotly.graph_objects as go

            figure = go.Figure()
            palette = ["#0f766e", "#1f7a5a", "#b7791f", "#9f3a38", "#1d4ed8", "#7c3aed"]
            added = 0
            for idx, (group_name, values) in enumerate(raw_data.items()):
                numeric = HTMLExporter._coerce_numeric_sequence(values)
                if len(numeric) < 2:
                    continue
                color = palette[idx % len(palette)]
                figure.add_trace(
                    go.Box(
                        y=numeric,
                        name=str(group_name),
                        marker=dict(color=color, size=7, opacity=0.7),
                        line=dict(color=color),
                        fillcolor="rgba(15,118,110,0.12)" if idx == 0 else None,
                        boxpoints="all",
                        jitter=0.32,
                        pointpos=0,
                        whiskerwidth=0.7,
                        showlegend=False,
                    )
                )
                added += 1
            if added == 0:
                return None
            figure.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#fffdf8",
                margin=dict(l=48, r=20, t=24, b=48),
                font=dict(family="Segoe UI, Helvetica Neue, sans-serif", color="#16313a"),
                height=500,
            )
            figure.update_xaxes(title_text="Groups")
            figure.update_yaxes(title_text="Observed values", zeroline=False)
            return HTMLExporter._figure_to_html(figure)
        except Exception as exc:
            print(f"WARNING HTML EXPORT: group distribution chart generation failed: {exc}")
            return None

    @staticmethod
    def _build_residuals_vs_fitted_chart(results: dict) -> str | None:
        residuals = HTMLExporter._coerce_numeric_sequence(results.get("model_residuals") or results.get("residuals"))
        fitted = HTMLExporter._coerce_numeric_sequence(results.get("fitted_values") or results.get("fitted"))
        if len(residuals) < 3 or len(fitted) < 3:
            return None
        paired_length = min(len(residuals), len(fitted))
        try:
            import plotly.graph_objects as go
            import plotly.io as pio

            figure = go.Figure(
                data=[
                    go.Scatter(
                        x=fitted[:paired_length],
                        y=residuals[:paired_length],
                        mode="markers",
                        marker=dict(size=8, color="#0f766e", opacity=0.8, line=dict(color="#16313a", width=1)),
                    )
                ]
            )
            figure.add_hline(y=0, line=dict(color="#9f3a38", dash="dash"))
            figure.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#fffdf8",
                margin=dict(l=48, r=20, t=24, b=42),
                font=dict(family="Segoe UI, Helvetica Neue, sans-serif", color="#16313a"),
                xaxis_title="Fitted values",
                yaxis_title="Residuals",
            )
            return HTMLExporter._figure_to_html(figure)
        except Exception as exc:
            print(f"WARNING HTML EXPORT: residual-vs-fitted chart generation failed: {exc}")
            return None

    @staticmethod
    def _build_assumption_interpretation(results: dict, rows: list[dict]) -> str:
        flagged = [row["name"] for row in rows if row.get("status_class") == "is-danger"]
        if flagged:
            top = ", ".join(flagged[:2])
            if len(flagged) > 2:
                top += ", and additional flagged diagnostics"
            return f"Diagnostic review flagged {top}. The selected procedure should be interpreted in light of these assumption results."
        transformation = str(results.get("transformation") or "None")
        if transformation and transformation.lower() != "none":
            return f"All available diagnostics are acceptable after applying the reported {transformation} transformation."
        return "Available diagnostics do not indicate a major assumption problem for the selected procedure."

    @staticmethod
    def _estimate_kde(values: list[float]):
        if len(values) < 2:
            return None, None
        try:
            kde = stats.gaussian_kde(values)
            x_grid = np.linspace(min(values), max(values), 160)
            return x_grid, kde(x_grid)
        except Exception:
            return None, None

    @staticmethod
    def _build_decision_path_model(results: dict) -> list[dict]:
        test_info = results.get("test_info", {}) or {}
        steps = [{
            "title": "Data screening",
            "detail": "BioMedStatX evaluated assumptions and available structure before selecting the analysis path.",
            "active": True,
        }]
        pre = test_info.get("pre_transformation", {}) if isinstance(test_info, dict) else {}
        if pre:
            detail_parts = []
            residuals = pre.get("residuals_normality", {})
            variance = pre.get("variance", {})
            if residuals.get("p_value") is not None:
                detail_parts.append(f"Residual normality {HTMLExporter._format_p_value(residuals.get('p_value'))}")
            if variance.get("p_value") is not None:
                detail_parts.append(f"Variance homogeneity {HTMLExporter._format_p_value(variance.get('p_value'))}")
            steps.append({
                "title": "Assumption checks",
                "detail": ", ".join(detail_parts) if detail_parts else "Assumption checks executed.",
                "active": True,
            })
        transformation = results.get("transformation") or test_info.get("transformation")
        steps.append({
            "title": "Transformation",
            "detail": f"Applied transformation: {transformation or 'None'}",
            "active": bool(transformation and str(transformation).lower() != "none"),
        })
        steps.append({
            "title": "Test selection",
            "detail": str(results.get("final_test_label") or results.get("test") or "Selected statistical model"),
            "active": True,
        })
        if results.get("posthoc_test") or results.get("pairwise_comparisons"):
            steps.append({
                "title": "Post-hoc layer",
                "detail": str(results.get("posthoc_test") or f"{len(results.get('pairwise_comparisons') or [])} pairwise comparisons"),
                "active": True,
            })
        steps.append({
            "title": "Inference",
            "detail": HTMLExporter._build_summary_note(
                results,
                str(results.get("final_test_label") or results.get("test") or "Analysis"),
                results.get("p_value"),
            ),
            "active": True,
        })
        return steps

    @staticmethod
    def _embed_decision_tree(results: dict) -> str | None:
        temp_path = None
        try:
            from decisiontreevisualizer import DecisionTreeVisualizer

            temp_path = DecisionTreeVisualizer.generate_and_save_for_excel(results)
            if not temp_path or not os.path.exists(temp_path):
                return None
            with open(temp_path, "rb") as handle:
                encoded = base64.b64encode(handle.read()).decode("ascii")
            return f"data:image/png;base64,{encoded}"
        except Exception as exc:
            print(f"WARNING HTML EXPORT: decision tree embedding failed: {exc}")
            return None
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    @staticmethod
    def _build_methods_text(results: dict, analysis_log: Any) -> str:
        if isinstance(analysis_log, list):
            return "\n".join(str(line) for line in analysis_log)
        if isinstance(analysis_log, str) and analysis_log.strip():
            return analysis_log
        lines = [
            f"Test performed: {results.get('final_test_label') or results.get('test') or 'Not specified'}",
            f"Alpha level: {HTMLExporter._format_metric(results.get('alpha'), digits=3)}",
            f"Transformation: {results.get('transformation') or 'None'}",
        ]
        if results.get("posthoc_test"):
            lines.append(f"Post-hoc procedure: {results.get('posthoc_test')}")
        if results.get("effect_size_type"):
            lines.append(f"Effect size metric: {results.get('effect_size_type')}")
        return "\n".join(lines)

    @staticmethod
    def _render_template(context: dict, mode: str) -> str:
        assumptions = context.get("assumptions", {}) or {}
        plotly_enabled = any([
            bool(context.get("chart_blocks")),
            bool(assumptions.get("qq_plot_html")),
            bool(assumptions.get("distribution_plot_html")),
            bool(assumptions.get("residual_plot_html")),
        ])
        plotly_bundle = HTMLExporter._plotly_bundle() if plotly_enabled else ""
        env = Environment(autoescape=True, trim_blocks=True, lstrip_blocks=True)
        template = env.from_string(HTMLExporter._template())
        return template.render(context=context, mode=mode, plotly_bundle=plotly_bundle)

    @staticmethod
    def _plotly_bundle() -> str:
        try:
            from plotly.offline.offline import get_plotlyjs

            return f"<script>{get_plotlyjs()}</script>"
        except Exception as exc:
            print(f"WARNING HTML EXPORT: plotly bundle unavailable: {exc}")
            return ""

    @staticmethod
    def _figure_to_html(figure) -> str | None:
        try:
            import plotly.io as pio

            return pio.to_html(
                figure,
                full_html=False,
                include_plotlyjs=False,
                config={"responsive": True, "displayModeBar": False},
                default_width="100%",
            )
        except Exception as exc:
            print(f"WARNING HTML EXPORT: figure HTML generation failed: {exc}")
            return None

    @staticmethod
    def _template() -> str:
        return r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{{ context.report_title }}</title>
<style>
:root{--surface:#fffdf8;--surface-2:#fff;--ink:#16313a;--muted:#6b7c84;--line:rgba(22,49,58,.12);--accent:#0f766e;--success:#1f7a5a;--warning:#b7791f;--danger:#9f3a38;--shadow:0 18px 40px rgba(22,49,58,.08)}*{box-sizing:border-box}html{font-size:16px}
body{margin:0;font-family:"Segoe UI","Helvetica Neue",Arial,sans-serif;line-height:1.58;color:var(--ink);background:radial-gradient(circle at top left,rgba(15,118,110,.08),transparent 28%),linear-gradient(180deg,#f9f7f2 0,#f3efe8 100%)}h1,h2,h3{font-family:Georgia,"Times New Roman",serif;letter-spacing:-.02em;margin:0 0 .4rem}p,td,th{overflow-wrap:break-word;word-break:break-word;hyphens:auto;margin:0 0 1rem}
.page{max-width:1240px;margin:0 auto;padding:1.75rem 1.25rem 3.5rem}.hero,.section,.dataset-card,.metric-card{background:var(--surface);border:1px solid var(--line);border-radius:24px;box-shadow:var(--shadow)}.hero{padding:2.125rem;background:linear-gradient(135deg,rgba(22,49,58,.98),rgba(15,118,110,.88));color:#f9fafb}.eyebrow,.section-kicker{font-size:.8rem;text-transform:uppercase;letter-spacing:.12em;color:var(--muted)}.hero .eyebrow{color:rgba(249,250,251,.75)}
.hero-grid,.metrics-grid,.decision-layout,.dataset-grid,.assumption-visual-grid{display:grid;gap:18px;min-width:0}.hero-grid>* ,.metrics-grid>* ,.decision-layout>* ,.dataset-grid>* ,.assumption-visual-grid>*{min-width:0}.hero-grid{grid-template-columns:1.6fr 1fr;align-items:end}.metrics-grid{grid-template-columns:repeat(auto-fit,minmax(250px,1fr));margin-top:20px}.hero-title{font-size:clamp(2rem,4vw,3.35rem)}.hero-subtitle{color:rgba(249,250,251,.84)}.metric-card{padding:1rem 1.1rem;color:var(--ink);min-height:7rem}.metric-label{font-size:.78rem;text-transform:uppercase;letter-spacing:.12em;color:var(--muted)}.metric-value{font-size:clamp(1.2rem,2vw,2rem);font-weight:700;margin-top:.5rem;font-variant-numeric:tabular-nums}
.section{margin-top:22px;padding:1.5rem;opacity:0;transform:translateY(18px);transition:opacity .45s ease,transform .45s ease}.section.is-visible{opacity:1;transform:none}.section-head{display:flex;justify-content:space-between;gap:20px;align-items:end;margin-bottom:16px;flex-wrap:wrap}.badge{display:inline-flex;align-items:center;padding:.4rem .85rem;border-radius:999px;font-size:.84rem;font-weight:700;max-width:100%}.is-significant{background:rgba(31,122,90,.14);color:var(--success)}.is-danger{background:rgba(159,58,56,.14);color:var(--danger)}.is-neutral{background:rgba(22,49,58,.08);color:var(--ink)}
.table-shell{overflow-x:auto;border:1px solid var(--line);border-radius:18px;background:var(--surface-2)}table{width:100%;border-collapse:collapse;font-size:.96rem}th,td{padding:12px 14px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{background:rgba(22,49,58,.04);font-size:.82rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}tr:hover td{background:rgba(15,118,110,.05)}
.decision-layout{grid-template-columns:minmax(0,.84fr) minmax(0,1.16fr)}.decision-track{position:relative;padding-left:24px}.decision-track:before{content:"";position:absolute;left:7px;top:8px;bottom:8px;width:2px;background:linear-gradient(180deg,rgba(15,118,110,.15),rgba(15,118,110,.55))}.decision-step{position:relative;padding:0 0 18px 18px;opacity:.55;transform:translateX(-6px);transition:opacity .3s ease,transform .3s ease}.decision-step.is-active{opacity:1;transform:none}.decision-step:before{content:"";position:absolute;left:-24px;top:7px;width:14px;height:14px;border-radius:50%;background:var(--surface);border:3px solid rgba(15,118,110,.35)}.decision-step.is-active:before{border-color:var(--accent);background:var(--accent)}
.decision-tree-frame,.chart-card,.methods,.empty-state,.modal-panel{border:1px solid var(--line);border-radius:18px;background:var(--surface-2)}.decision-tree-frame{padding:12px;min-height:480px;display:flex;align-items:center;justify-content:center;overflow:hidden;cursor:zoom-in}.decision-tree-frame img{width:100%;height:auto;max-height:780px;object-fit:contain}.chart-card{padding:16px;margin-bottom:16px;min-width:0;overflow:hidden}.chart-card .js-plotly-plot,.chart-card .plotly-graph-div,.chart-card .plot-container,.chart-card .svg-container{width:100%!important;max-width:100%!important}.chart-card .main-svg{max-width:100%!important}.assumption-visual-grid{grid-template-columns:repeat(auto-fit,minmax(320px,1fr));margin-top:1rem;align-items:start}.toolbar{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:12px}.toolbar input,.toolbar button,.tree-button,.modal-close{font:inherit;border-radius:12px;border:1px solid var(--line);background:#fff;color:var(--ink);padding:.75rem .95rem}.toolbar button,.tree-button,.modal-close{cursor:pointer}.methods{white-space:pre-wrap;font-family:"Cascadia Mono",Consolas,monospace;padding:16px;background:rgba(22,49,58,.03)}.dataset-grid{grid-template-columns:repeat(auto-fit,minmax(260px,1fr))}.dataset-card{padding:20px;opacity:0;transform:translateY(16px);animation:riseIn .45s ease forwards}.muted{color:var(--muted)}.small{font-size:.92rem}.empty-state{padding:18px;color:var(--muted)}.footer-note{margin-top:26px;color:var(--muted);font-size:.92rem;text-align:center}.modal-backdrop{position:fixed;inset:0;background:rgba(6,10,12,.75);display:none;align-items:center;justify-content:center;padding:1.5rem;z-index:1000}.modal-backdrop.is-open{display:flex}.modal-panel{max-width:min(94vw,96rem);max-height:92vh;padding:1rem}.modal-panel img{display:block;max-width:100%;max-height:80vh;object-fit:contain}.modal-toolbar{display:flex;justify-content:space-between;align-items:center;gap:1rem;margin-bottom:.75rem}
@keyframes riseIn{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:none}}@media (max-width:900px){.hero-grid,.decision-layout{grid-template-columns:1fr}}@media (prefers-reduced-motion:reduce){*,*:before,*:after{animation:none!important;transition:none!important}.section,.dataset-card{opacity:1;transform:none}}
</style>{{ plotly_bundle | safe }}</head><body><div class="page">
<header class="hero"><div class="eyebrow">BioMedStatX Offline Scientific Report</div><div class="hero-grid"><div><h1 class="hero-title">{{ context.report_title }}</h1><p class="hero-subtitle">{{ context.subtitle }}</p>{% if mode == "single" %}<div class="metrics-grid"><article class="metric-card"><div class="metric-label">Selected Test</div><div class="metric-value">{{ context.hero.test_name }}</div></article><article class="metric-card"><div class="metric-label">p-value</div><div class="metric-value">{{ context.hero.p_value_display }}</div></article><article class="metric-card"><div class="metric-label">{{ context.hero.effect_label }}</div><div class="metric-value">{{ context.hero.effect_size_display }}</div></article></div>{% endif %}</div><div>{% if mode == "single" %}<div class="badge {{ context.hero.significance_class }}">{{ context.hero.significance_label }}</div><p style="margin-top:14px;">{{ context.hero.summary_note }}</p>{% else %}<div class="badge is-neutral">Overview Report</div><p style="margin-top:14px;">This companion report summarizes the main outcome of each dataset while preserving fully offline viewing.</p>{% endif %}</div></div></header>
{% if mode == "single" %}
<section class="section" data-reveal><div class="section-head"><div><div class="section-kicker">Decision Path</div><h2>How BioMedStatX reached this decision</h2><p>The decision timeline provides a readable path through the selected analysis route.</p></div>{% if context.decision_tree_image %}<button type="button" class="tree-button" id="open-tree-modal">Open enlarged decision tree</button>{% endif %}</div><div class="decision-layout"><div class="decision-track" id="decision-path">{% for step in context.decision_path %}<div class="decision-step{% if step.active %} is-active{% endif %}" data-step="{{ loop.index0 }}"><h3>{{ step.title }}</h3><p class="muted">{{ step.detail }}</p></div>{% endfor %}</div><div class="decision-tree-frame" {% if context.decision_tree_image %}id="tree-preview-trigger" aria-label="Open enlarged decision tree" role="button" tabindex="0"{% endif %}>{% if context.decision_tree_image %}<img src="{{ context.decision_tree_image }}" alt="Decision tree visualization">{% else %}<div class="empty-state">Decision tree preview was not available for this analysis.</div>{% endif %}</div></div></section>
<section class="section" data-reveal><div class="section-head"><div><div class="section-kicker">Statistical Engine</div><h2>Main results</h2></div></div><div class="table-shell"><table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>{% for row in context.statistical_rows %}<tr><td>{{ row.label }}</td><td>{{ row.value }}</td></tr>{% endfor %}</tbody></table></div></section>
<section class="section" data-reveal><div class="section-head"><div><div class="section-kicker">Assumptions</div><h2>Model validity checks</h2><p class="muted">{{ context.assumptions.interpretation }}</p></div><div class="badge is-neutral">Transformation: {{ context.assumptions.transformation }}</div></div>{% if context.assumptions.rows %}<div class="table-shell"><table><thead><tr><th>Check</th><th>Statistic</th><th>p-value</th><th>Status</th></tr></thead><tbody>{% for row in context.assumptions.rows %}<tr class="{{ row.status_class }}"><td>{{ row.name }}</td><td>{{ row.statistic }}</td><td>{{ row.p_value }}</td><td>{{ row.status_label }}</td></tr>{% endfor %}</tbody></table></div>{% else %}<div class="empty-state">No structured assumption summary was available for this result.</div>{% endif %}<div class="assumption-visual-grid">{% if context.assumptions.qq_plot_html %}<div class="chart-card"><div class="section-kicker">Q-Q Diagnostic</div><h3>Observed quantiles against the normal reference line</h3>{{ context.assumptions.qq_plot_html | safe }}</div>{% endif %}{% if context.assumptions.distribution_plot_html %}<div class="chart-card"><div class="section-kicker">Group Distribution View</div><h3>Boxplots with jittered observations</h3>{{ context.assumptions.distribution_plot_html | safe }}</div>{% endif %}{% if context.assumptions.residual_plot_html %}<div class="chart-card"><div class="section-kicker">Residual Structure</div><h3>Residuals versus fitted values</h3>{{ context.assumptions.residual_plot_html | safe }}</div>{% endif %}{% if not context.assumptions.qq_plot_html and not context.assumptions.distribution_plot_html and not context.assumptions.residual_plot_html %}<div class="empty-state">Visual assumption diagnostics were not available for this result structure.</div>{% endif %}</div></section>
<section class="section" data-reveal><div class="section-head"><div><div class="section-kicker">Descriptive Statistics</div><h2>Group-level summary</h2></div></div>{% if context.descriptive.rows %}<div class="table-shell"><table><thead><tr><th>Group</th><th>n</th><th>Mean</th><th>Median</th><th>SD</th><th>SEM</th><th>Min</th><th>Max</th></tr></thead><tbody>{% for row in context.descriptive.rows %}<tr><td>{{ row.group }}</td><td>{{ row.n }}</td><td>{{ row.mean }}</td><td>{{ row.median }}</td><td>{{ row.sd }}</td><td>{{ row.sem }}</td><td>{{ row.min }}</td><td>{{ row.max }}</td></tr>{% endfor %}</tbody></table></div>{% else %}<div class="empty-state">No descriptive summary could be derived from the available result payload.</div>{% endif %}</section>
{% if context.pairwise_rows %}<section class="section" data-reveal><div class="section-head"><div><div class="section-kicker">Pairwise Comparisons</div><h2>Post-hoc findings</h2></div></div><div class="table-shell"><table><thead><tr><th>Comparison</th><th>Procedure</th><th>Statistic</th><th>p-value</th><th>Effect size</th><th>Interpretation</th></tr></thead><tbody>{% for row in context.pairwise_rows %}<tr class="{{ row.row_class }}"><td>{{ row.comparison }}</td><td>{{ row.test }}</td><td>{{ row.statistic }}</td><td>{{ row.p_value }}</td><td>{{ row.effect_size }}</td><td>{{ "Significant" if row.significant else "Not significant" }}</td></tr>{% endfor %}</tbody></table></div></section>{% endif %}
<section class="section" data-reveal><div class="section-head"><div><div class="section-kicker">Interactive Charts</div><h2>Visual evidence</h2></div></div>{% if context.chart_blocks %}{% for chart in context.chart_blocks %}<div class="chart-card"><div class="section-kicker">{{ chart.title }}</div><h3>{{ chart.subtitle }}</h3>{{ chart.html | safe }}</div>{% endfor %}{% else %}<div class="empty-state">No interactive chart could be created from the current result structure.</div>{% endif %}</section>
<section class="section" data-reveal><div class="section-head"><div><div class="section-kicker">Raw Data Vault</div><h2>Searchable raw values</h2></div></div><div class="toolbar"><input id="raw-search" type="search" placeholder="Filter raw data"><button type="button" onclick="copyTable('raw-data-table')">Copy table</button><button type="button" onclick="downloadTableCSV('raw-data-table','biomedstatx_raw_data.csv')">Download CSV</button></div>{% if context.raw_data_table.rows %}<div class="table-shell"><table id="raw-data-table"><thead><tr><th>Group</th><th>Index</th><th>Raw value</th>{% if context.raw_data_table.has_transformed %}<th>Transformed value</th>{% endif %}</tr></thead><tbody>{% for row in context.raw_data_table.rows %}<tr><td>{{ row.group }}</td><td>{{ row.index }}</td><td>{{ row.raw_value }}</td>{% if context.raw_data_table.has_transformed %}<td>{{ row.transformed_value }}</td>{% endif %}</tr>{% endfor %}</tbody></table></div>{% else %}<div class="empty-state">No raw data were embedded in this result structure.</div>{% endif %}</section>
<section class="section" data-reveal><div class="section-head"><div><div class="section-kicker">Methods Snippet</div><h2>Reusable narrative text</h2></div></div><div class="toolbar"><button type="button" onclick="copyText('methods-text')">Copy methods text</button></div><div id="methods-text" class="methods">{{ context.methods_text }}</div></section>
{% else %}
<section class="section is-visible"><div class="section-head"><div><div class="section-kicker">Dataset Overview</div><h2>Summary across exported analyses</h2></div></div><div class="dataset-grid">{% for card in context.dataset_cards %}<article class="dataset-card"><div class="section-kicker">{{ card.dataset_name }}</div><h3>{{ card.test_name }}</h3><p class="small muted">{{ card.summary_note }}</p><div style="display:flex;gap:10px;flex-wrap:wrap;margin:12px 0 14px;"><span class="badge {{ card.significance_class }}">{{ card.significance_label }}</span><span class="badge is-neutral">{{ card.p_value_display }}</span></div><p class="small">Transformation: {{ card.transformation }}</p><p class="small">Pairwise comparisons: {{ card.pairwise_count }}</p>{% if card.assumptions.rows %}<div class="table-shell" style="margin-top:14px;"><table><thead><tr><th>Check</th><th>p-value</th><th>Status</th></tr></thead><tbody>{% for row in card.assumptions.rows[:4] %}<tr class="{{ row.status_class }}"><td>{{ row.name }}</td><td>{{ row.p_value }}</td><td>{{ row.status_label }}</td></tr>{% endfor %}</tbody></table></div>{% endif %}</article>{% endfor %}</div></section>
{% endif %}
<div class="footer-note">{% if context.generated_warning %}{{ context.generated_warning }}{% else %}Generated by BioMedStatX as a fully offline HTML scientific report.{% endif %}</div></div>{% if mode == "single" and context.decision_tree_image %}<div class="modal-backdrop" id="tree-modal" aria-hidden="true"><div class="modal-panel" role="dialog" aria-modal="true" aria-labelledby="tree-modal-title"><div class="modal-toolbar"><div><div class="section-kicker">Decision Tree</div><h3 id="tree-modal-title">Expanded analysis routing view</h3></div><button type="button" class="modal-close" id="close-tree-modal">Close</button></div><img src="{{ context.decision_tree_image }}" alt="Expanded decision tree visualization"></div></div>{% endif %}
<script>
const reduceMotion=window.matchMedia&&window.matchMedia('(prefers-reduced-motion: reduce)').matches;if(!reduceMotion&&'IntersectionObserver'in window){const observer=new IntersectionObserver((entries)=>{entries.forEach((entry)=>{if(entry.isIntersecting){entry.target.classList.add('is-visible');observer.unobserve(entry.target);}})},{threshold:.12});document.querySelectorAll('[data-reveal]').forEach((node)=>observer.observe(node));}else{document.querySelectorAll('[data-reveal]').forEach((node)=>node.classList.add('is-visible'));}
const decisionSteps=Array.from(document.querySelectorAll('#decision-path .decision-step'));if(decisionSteps.length&&!reduceMotion){decisionSteps.forEach((step)=>step.classList.remove('is-active'));decisionSteps.forEach((step,index)=>setTimeout(()=>step.classList.add('is-active'),220*index));}
function copyText(elementId){const node=document.getElementById(elementId);if(!node)return;const text=node.innerText||node.textContent||'';navigator.clipboard.writeText(text).catch(()=>{});}
function tableToTSV(tableId){const table=document.getElementById(tableId);if(!table)return'';return Array.from(table.querySelectorAll('tr')).map((row)=>Array.from(row.querySelectorAll('th,td')).map((cell)=>(cell.innerText||'').replace(/\n/g,' ').trim()).join('\t')).join('\n');}
function copyTable(tableId){const text=tableToTSV(tableId);if(text){navigator.clipboard.writeText(text).catch(()=>{});}}
function downloadTableCSV(tableId,fileName){const table=document.getElementById(tableId);if(!table)return;const rows=Array.from(table.querySelectorAll('tr')).map((row)=>Array.from(row.querySelectorAll('th,td')).map((cell)=>{const value=(cell.innerText||'').replace(/\n/g,' ').trim().replace(/"/g,'""');return `"${value}"`;}).join(',')).join('\n');const blob=new Blob([rows],{type:'text/csv;charset=utf-8;'});const url=URL.createObjectURL(blob);const link=document.createElement('a');link.href=url;link.download=fileName;document.body.appendChild(link);link.click();document.body.removeChild(link);URL.revokeObjectURL(url);}
const rawSearch=document.getElementById('raw-search');if(rawSearch){rawSearch.addEventListener('input',(event)=>{const needle=String(event.target.value||'').toLowerCase();document.querySelectorAll('#raw-data-table tbody tr').forEach((row)=>{row.style.display=row.innerText.toLowerCase().includes(needle)?'':'none';});});}
const treeModal=document.getElementById('tree-modal');const openTreeButton=document.getElementById('open-tree-modal');const treePreviewTrigger=document.getElementById('tree-preview-trigger');const closeTreeButton=document.getElementById('close-tree-modal');function openTreeModal(){if(!treeModal)return;treeModal.classList.add('is-open');treeModal.setAttribute('aria-hidden','false');}function closeTreeModal(){if(!treeModal)return;treeModal.classList.remove('is-open');treeModal.setAttribute('aria-hidden','true');}if(openTreeButton){openTreeButton.addEventListener('click',openTreeModal);}if(treePreviewTrigger){treePreviewTrigger.addEventListener('click',openTreeModal);treePreviewTrigger.addEventListener('keydown',(event)=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();openTreeModal();}});}if(closeTreeButton){closeTreeButton.addEventListener('click',closeTreeModal);}if(treeModal){treeModal.addEventListener('click',(event)=>{if(event.target===treeModal){closeTreeModal();}});document.addEventListener('keydown',(event)=>{if(event.key==='Escape'){closeTreeModal();}});}
</script></body></html>"""

    @staticmethod
    def _coerce_numeric_sequence(values: Any) -> list[float]:
        sequence = []
        if values is None:
            return sequence
        for item in list(values):
            try:
                numeric = float(item)
                if math.isnan(numeric) or math.isinf(numeric):
                    continue
                sequence.append(numeric)
            except Exception:
                continue
        return sequence

    @staticmethod
    def _format_metric(value: Any, digits: int = 4) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            return ", ".join(HTMLExporter._format_metric(item, digits=digits) for item in value)
        if isinstance(value, (int, float, np.generic)):
            numeric = float(value)
            if math.isnan(numeric):
                return "N/A"
            if math.isinf(numeric):
                return "Infinity" if numeric > 0 else "-Infinity"
            if abs(numeric) >= 1000 or (abs(numeric) > 0 and abs(numeric) < 0.001):
                return f"{numeric:.3e}"
            return f"{numeric:.{digits}f}"
        return str(value)

    @staticmethod
    def _format_p_value(value: Any) -> str:
        if not isinstance(value, (int, float, np.generic)):
            return "N/A" if value in (None, "", "N/A") else str(value)
        numeric = float(value)
        if math.isnan(numeric):
            return "N/A"
        if numeric < 0.001:
            return "p < 0.001"
        return f"p = {numeric:.3f}"

    @staticmethod
    def _format_confidence_interval(value: Any) -> str:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return f"[{HTMLExporter._format_metric(value[0])}, {HTMLExporter._format_metric(value[1])}]"
        return HTMLExporter._format_metric(value)

    @staticmethod
    def _prettify_label(value: str) -> str:
        return str(value).replace("_", " ").replace("-", " ").title()

    @staticmethod
    def _bool_label(value: Any) -> str:
        if value is True:
            return "Passed"
        if value is False:
            return "Flagged"
        return "Not available"

    @staticmethod
    def _bool_class(value: Any) -> str:
        if value is True:
            return "is-significant"
        if value is False:
            return "is-danger"
        return "is-neutral"

    @staticmethod
    def _has_display_value(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return value != ""
        if isinstance(value, dict):
            return bool(value)
        if isinstance(value, (list, tuple, set)):
            return len(value) > 0
        if isinstance(value, np.ndarray):
            return value.size > 0
        return True
