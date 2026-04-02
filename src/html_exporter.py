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
        group_chart_block = next((c for c in charts if c.get("div_id") == "biomedstatx-group-chart"), None)
        bracket_data = [
            {"pair_id": r["pair_id"], "group1": r["group1"], "group2": r["group2"],
             "stars": r["stars"], "significant": r["significant"]}
            for r in pairwise
        ]
        group_order = group_chart_block["group_order"] if group_chart_block else []
        decision_tree_image = HTMLExporter._embed_decision_tree(results_copy)
        decision_tree_json = HTMLExporter._build_decision_tree_json(results_copy)
        decision_path = HTMLExporter._build_decision_path_model(results_copy)
        methods_text = HTMLExporter._build_methods_text(results_copy, analysis_log_text)
        return {
            "mode": "single",
            "report_title": hero["title"],
            "subtitle": hero["subtitle"],
            "hero": hero,
            "decision_path": decision_path,
            "decision_tree_image": decision_tree_image,
            "decision_tree_json": json.dumps(decision_tree_json, ensure_ascii=False) if decision_tree_json else "null",
            "decision_path_json": json.dumps(decision_path, ensure_ascii=False),
            "statistical_rows": metrics,
            "assumptions": assumptions,
            "descriptive": descriptive,
            "pairwise_rows": pairwise,
            "bracket_data_json": json.dumps(bracket_data, ensure_ascii=False),
            "group_order_json": json.dumps(group_order, ensure_ascii=False),
            "group_chart_div_id": "biomedstatx-group-chart" if group_chart_block else "",
            "raw_data_table": raw_table,
            "chart_blocks": charts,
            "methods_text": methods_text,
            "info_texts": HTMLExporter._info_texts(),
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
            "effect_magnitude": HTMLExporter._effect_size_magnitude(effect_size, effect_label),
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
            if not isinstance(factor, dict):
                continue
            factor_name = factor.get("factor", "Factor")
            factor_value = (
                f"F({HTMLExporter._format_metric(factor.get('df1'))}, "
                f"{HTMLExporter._format_metric(factor.get('df2'))}) = {HTMLExporter._format_metric(factor.get('F'))}, "
                f"{HTMLExporter._format_p_value(factor.get('p_value'))}"
            )
            rows.append({"label": f"Main effect: {factor_name}", "value": factor_value})
        for interaction in results.get("interactions", []) or []:
            if not isinstance(interaction, dict):
                continue
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
        # Fallback: extract from nested test_info structure used by one-way ANOVA path
        test_info_raw = results.get("test_info", {}) or {}
        if not normality_tests and test_info_raw:
            _has_tr = test_info_raw.get("transformation") not in (None, "None", "No further")
            _phase = "post_transformation" if _has_tr else "pre_transformation"
            _norm = test_info_raw.get(_phase, {}).get("residuals_normality", {})
            if _norm:
                normality_tests = {"Model residuals": _norm}
        for label, payload in normality_tests.items():
            if not isinstance(payload, dict):
                continue
            rows.append({
                "name": f"Normality: {HTMLExporter._prettify_label(label)}",
                "statistic": HTMLExporter._format_metric(payload.get("statistic")),
                "p_value": HTMLExporter._format_p_value(payload.get("p_value")),
                "p_value_style": HTMLExporter._p_heat_style(payload.get("p_value")),
                "status_label": HTMLExporter._bool_label(payload.get("is_normal")),
                "status_class": HTMLExporter._bool_class(payload.get("is_normal")),
            })
        variance_test = results.get("variance_test", {}) or {}
        # Fallback: extract from nested test_info structure
        if not variance_test and test_info_raw:
            _has_tr = test_info_raw.get("transformation") not in (None, "None", "No further")
            _phase = "post_transformation" if _has_tr else "pre_transformation"
            variance_test = test_info_raw.get(_phase, {}).get("variance", {}) or {}
        if isinstance(variance_test, dict) and variance_test:
            _var_name = variance_test.get("test_name", "Levene")
            rows.append({
                "name": f"Variance homogeneity ({_var_name})",
                "statistic": HTMLExporter._format_metric(variance_test.get("statistic")),
                "p_value": HTMLExporter._format_p_value(variance_test.get("p_value")),
                "p_value_style": HTMLExporter._p_heat_style(variance_test.get("p_value")),
                "status_label": HTMLExporter._bool_label(variance_test.get("equal_variance")),
                "status_class": HTMLExporter._bool_class(variance_test.get("equal_variance")),
            })
            transformed = variance_test.get("transformed")
            if isinstance(transformed, dict):
                _var_name_tr = transformed.get("test_name", _var_name)
                rows.append({
                    "name": f"Variance homogeneity ({_var_name_tr}, transformed)",
                    "statistic": HTMLExporter._format_metric(transformed.get("statistic")),
                    "p_value": HTMLExporter._format_p_value(transformed.get("p_value")),
                    "p_value_style": HTMLExporter._p_heat_style(transformed.get("p_value")),
                    "status_label": HTMLExporter._bool_label(transformed.get("equal_variance")),
                    "status_class": HTMLExporter._bool_class(transformed.get("equal_variance")),
                })
        sphericity = results.get("sphericity_test", {}) or {}
        sphericity_correction_note = None
        if isinstance(sphericity, dict) and sphericity:
            status_value = sphericity.get("sphericity_met")
            if status_value is None and sphericity.get("p_value") is not None:
                status_value = sphericity.get("p_value") >= 0.05
            rows.append({
                "name": "Sphericity (Mauchly's W)",
                "statistic": HTMLExporter._format_metric(sphericity.get("W") or sphericity.get("statistic")),
                "p_value": HTMLExporter._format_p_value(sphericity.get("p_value")),
                "p_value_style": HTMLExporter._p_heat_style(sphericity.get("p_value")),
                "status_label": HTMLExporter._bool_label(status_value),
                "status_class": HTMLExporter._bool_class(status_value),
            })
            if status_value is False:
                corr = (sphericity.get("correction") or sphericity.get("correction_applied") or "").lower()
                gg_eps = sphericity.get("greenhouse_geisser") or sphericity.get("gg_epsilon") or sphericity.get("epsilon_gg")
                hf_eps = sphericity.get("huynh_feldt") or sphericity.get("hf_epsilon") or sphericity.get("epsilon_hf")
                if "huynh" in corr or "hf" in corr:
                    label = "Huynh-Feldt"
                    eps = hf_eps or gg_eps
                elif gg_eps or "greenhouse" in corr or "gg" in corr:
                    label = "Greenhouse-Geisser"
                    eps = gg_eps
                else:
                    label, eps = "Greenhouse-Geisser", gg_eps
                if label:
                    eps_str = f" (ε = {HTMLExporter._format_metric(eps)})" if eps else ""
                    sphericity_correction_note = f"Sphericity violated → {label} correction applied{eps_str}"
        _icons = {"is-significant": "✓ ", "is-danger": "✗ ", "is-neutral": "~ "}
        for row in rows:
            row["status_label"] = _icons.get(row["status_class"], "") + row["status_label"]
        return {
            "rows": rows,
            "transformation": str(results.get("transformation") or "None"),
            "interpretation": HTMLExporter._build_assumption_interpretation(results, rows),
            "sphericity_correction_note": sphericity_correction_note,
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
        for i, comp in enumerate(results.get("pairwise_comparisons", []) or []):
            p_val = comp.get("p_value")
            is_sig = bool(comp.get("significant"))
            p_numeric = float(p_val) if isinstance(p_val, (int, float)) else None
            stars = (
                "***" if p_numeric is not None and p_numeric < 0.001 else
                "**" if p_numeric is not None and p_numeric < 0.01 else
                "*" if p_numeric is not None and p_numeric < 0.05 else ""
            )
            rows.append({
                "pair_id": i,
                "group1": str(comp.get("group1", "")),
                "group2": str(comp.get("group2", "")),
                "comparison": f"{comp.get('group1', 'Group 1')} vs {comp.get('group2', 'Group 2')}",
                "test": str(comp.get("test") or results.get("posthoc_test") or "Pairwise comparison"),
                "statistic": HTMLExporter._format_metric(comp.get("statistic")),
                "p_value": HTMLExporter._format_p_value(p_val),
                "p_value_style": HTMLExporter._p_heat_style(p_val),
                "effect_size": HTMLExporter._format_metric(comp.get("effect_size")),
                "significant": is_sig,
                "stars": stars,
                "p_value_raw": p_numeric,
                "row_class": "is-significant" if is_sig else "is-neutral",
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
                "html": group_chart["html"],
                "group_order": group_chart["group_order"],
                "div_id": "biomedstatx-group-chart",
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

            figure = go.Figure()
            group_order = []
            for group_name, values in raw_data.items():
                numeric = HTMLExporter._coerce_numeric_sequence(values)
                if not numeric:
                    continue
                group_order.append(str(group_name))
                label = f"{group_name} (n={len(numeric)})"
                figure.add_trace(
                    go.Box(
                        y=numeric,
                        name=label,
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
                margin=dict(l=40, r=20, t=24, b=56),
                font=dict(family="Segoe UI, Helvetica Neue, sans-serif", color="#16313a"),
                yaxis_title="Observed values",
                showlegend=False,
            )
            html = HTMLExporter._figure_to_html(figure, div_id="biomedstatx-group-chart")
            if not html:
                return None
            return {"html": html, "group_order": group_order}
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
                margin=dict(l=56, r=16, t=30, b=60),
                font=dict(family="Segoe UI, Helvetica Neue, sans-serif", size=11, color="#16313a"),
                xaxis=dict(title=dict(text="Theoretical quantiles", font=dict(size=11), standoff=8)),
                yaxis=dict(title=dict(text="Observed quantiles", font=dict(size=11), standoff=8)),
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
    def _build_decision_tree_json(results: dict) -> dict | None:
        try:
            from decisiontreevisualizer import DecisionTreeVisualizer
            return DecisionTreeVisualizer.get_tree_json(results)
        except Exception as exc:
            print(f"WARNING HTML EXPORT: decision tree JSON failed: {exc}")
            return None

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
            print(f"WARNING HTML EXPORT: figure HTML generation failed: {exc}")
            return None

    @staticmethod
    def _info_texts() -> dict:
        return {
            "decision": (
                "This section traces the automated hypothesis-testing workflow. "
                "Each node represents a decision point (e.g. normality check, group count, "
                "study design) that led BioMedStatX to select the reported statistical procedure. "
                "The highlighted path is the route taken for this specific dataset."
            ),
            "results": (
                "This section contains the main results of the statistical analysis: "
                "test statistic, p-value, effect size, confidence interval, and statistical power.\n"
                "For permutation-based nonparametric ANOVA, p-values are computed using "
                "the Freedman\u2013Lane scheme."
            ),
            "assumptions": (
                "Normality is assessed with Shapiro\u2013Wilk (W): p\u202f<\u202f0.05 indicates "
                "departure from normality.\n"
                "Variance homogeneity is assessed with Levene\u2019s test (Brown-Forsythe variant, center\u202f=\u202fmedian): p\u202f<\u202f0.05 "
                "indicates unequal variances across groups.\n"
                "Sphericity (Mauchly\u2019s W) is only relevant for repeated-measures designs. "
                "A violation triggers an adjustment of degrees of freedom: "
                "Greenhouse\u2013Geisser (\u03b5\u202f<\u202f0.75) or Huynh\u2013Feldt "
                "(\u03b5\u202f\u2265\u202f0.75) is selected automatically."
            ),
            "descriptive": (
                "Summary statistics for each group:\n"
                "  \u2022 n \u2014 sample size\n"
                "  \u2022 Mean with 95\u202f% confidence interval\n"
                "  \u2022 Median, SD (standard deviation), SEM (standard error of the mean)\n"
                "  \u2022 Min / Max\n"
                "If a data transformation was applied, transformed values are shown as well."
            ),
            "pairwise": (
                "Parametric post-hoc tests (normality met):\n"
                "  \u2022 Tukey HSD \u2014 all-pair comparisons, controls family-wise error rate\n"
                "  \u2022 Dunnett \u2014 all groups vs. a single control group\n"
                "  \u2022 Custom paired t-tests with Holm\u2013Sid\u00e1k correction\n\n"
                "Non-parametric post-hoc tests (normality violated):\n"
                "  \u2022 Dunn test \u2014 rank-based all-pair comparisons with Holm\u2013Sid\u00e1k\n"
                "  \u2022 Custom Mann\u2013Whitney U with Sid\u00e1k correction (assumes independence)\n"
                "  \u2022 Dependent post-hoc (paired t-tests or Wilcoxon) for repeated measures\n\n"
                "Effect size interpretation (Cohen\u2019s d / Hedges\u2019 g, Cohen 1988; "
                "Hedges\u2019 g preferred for n\u202f<\u202f20): "
                "small\u202f\u2264\u202f0.2, medium\u202f\u2264\u202f0.5, large\u202f\u2264\u202f0.8.\n"
                "Significance: *\u202fp\u202f<\u202f0.05 \u2002 **\u202fp\u202f<\u202f0.01 \u2002 ***\u202fp\u202f<\u202f0.001"
            ),
            "charts": (
                "Interactive Plotly charts rendered fully offline inside this file.\n"
                "Boxplots show the median (central line), interquartile range (box), "
                "1.5\u00d7IQR whiskers, and individual observations as jittered points.\n"
                "Hover over any element to see exact values."
            ),
            "raw": (
                "The original data values exactly as submitted.\n"
                "Use the search box to filter rows. Download as CSV for use in other tools."
            ),
            "methods": (
                "A ready-to-paste methods paragraph for a scientific manuscript. "
                "It describes the statistical procedure, transformation applied (if any), "
                "post-hoc correction, effect size metric, and alpha threshold. "
                "The BioMedStatX version is included for reproducibility."
            ),
        }

    @staticmethod
    def _template() -> str:
        return r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{{ context.report_title }}</title>
<style>
:root{--surface:#fffdf8;--surface-2:#fff;--ink:#16313a;--muted:#6b7c84;--line:rgba(22,49,58,.12);--accent:#0f766e;--info:#0369a1;--success:#1f7a5a;--warning:#b7791f;--danger:#9f3a38;--shadow:0 18px 40px rgba(22,49,58,.08)}*{box-sizing:border-box}html{font-size:16px}
body{margin:0;font-family:"Segoe UI","Helvetica Neue",Arial,sans-serif;line-height:1.58;color:var(--ink);background:radial-gradient(circle at top left,rgba(15,118,110,.08),transparent 28%),linear-gradient(180deg,#f9f7f2 0,#f3efe8 100%)}h1,h2,h3{font-family:Georgia,"Times New Roman",serif;letter-spacing:-.02em;margin:0 0 .4rem}p,td,th{overflow-wrap:break-word;word-break:break-word;hyphens:auto;margin:0 0 1rem}
.page{max-width:1240px;margin:0 auto;padding:1.75rem 1.25rem 3.5rem}.hero,.section,.dataset-card,.metric-card{background:var(--surface);border:1px solid var(--line);border-radius:24px;box-shadow:var(--shadow)}.hero{padding:2.125rem;background:linear-gradient(135deg,rgba(22,49,58,.98),rgba(15,118,110,.88));color:#f9fafb}.eyebrow,.section-kicker{font-size:.8rem;text-transform:uppercase;letter-spacing:.12em;color:var(--info)}.hero .eyebrow{color:rgba(249,250,251,.75)}
.hero-grid,.metrics-grid,.decision-layout,.dataset-grid,.assumption-visual-grid{display:grid;gap:18px;min-width:0}.hero-grid>* ,.metrics-grid>* ,.decision-layout>* ,.dataset-grid>* ,.assumption-visual-grid>*{min-width:0}.hero-grid{grid-template-columns:1.6fr 1fr;align-items:end}.metrics-grid{grid-template-columns:repeat(auto-fit,minmax(250px,1fr));margin-top:20px}.hero-title{font-size:clamp(2rem,4vw,3.35rem)}.hero-subtitle{color:rgba(249,250,251,.84)}.metric-card{padding:1rem 1.1rem;color:var(--ink);min-height:7rem}.metric-label{font-size:.78rem;text-transform:uppercase;letter-spacing:.12em;color:var(--muted)}.metric-value{font-size:clamp(1.2rem,2vw,2rem);font-weight:700;margin-top:.5rem;font-variant-numeric:tabular-nums}
.section{margin-top:22px;padding:1.5rem;opacity:0;transform:translateY(18px);transition:opacity .45s ease,transform .45s ease}.section.is-visible{opacity:1;transform:none}.section-head{display:flex;justify-content:space-between;gap:20px;align-items:end;margin-bottom:16px;flex-wrap:wrap}.badge{display:inline-flex;align-items:center;padding:.4rem .85rem;border-radius:999px;font-size:.84rem;font-weight:700;max-width:100%}.is-significant{background:rgba(31,122,90,.14);color:var(--success)}.is-danger{background:rgba(159,58,56,.14);color:var(--danger)}.is-neutral{background:rgba(22,49,58,.08);color:var(--ink)}.is-info{background:rgba(3,105,161,.12);color:var(--info)}.hero .badge.is-significant{background:rgba(52,211,153,.22);color:#bbf7d0}.hero .badge.is-danger{background:rgba(252,165,165,.22);color:#fecaca}.hero .badge.is-neutral{background:rgba(249,250,251,.15);color:#f9fafb}
.table-shell{overflow-x:auto;border:1px solid var(--line);border-radius:18px;background:var(--surface-2)}table{width:100%;border-collapse:collapse;font-size:.96rem}th,td{padding:12px 14px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{background:rgba(22,49,58,.04);font-size:.82rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}tr:hover td{background:rgba(15,118,110,.05)}
.decision-layout{grid-template-columns:minmax(0,.84fr) minmax(0,1.16fr)}.decision-track{position:relative;padding-left:24px}.decision-track:before{content:"";position:absolute;left:7px;top:8px;bottom:8px;width:2px;background:linear-gradient(180deg,rgba(15,118,110,.15),rgba(15,118,110,.55))}.decision-step{position:relative;padding:0 0 18px 18px;opacity:.55;transform:translateX(-6px);transition:opacity .3s ease,transform .3s ease}.decision-step.is-active{opacity:1;transform:none}.decision-step:before{content:"";position:absolute;left:-24px;top:7px;width:14px;height:14px;border-radius:50%;background:var(--surface);border:3px solid rgba(15,118,110,.35)}.decision-step.is-active:before{border-color:var(--accent);background:var(--accent)}
.decision-tree-frame,.chart-card,.methods,.empty-state,.modal-panel{border:1px solid var(--line);border-radius:18px;background:var(--surface-2)}.decision-tree-frame{padding:0;height:560px;display:flex;flex-direction:column;overflow:hidden;position:relative}.decision-tree-frame.is-empty{height:auto;min-height:0}.decision-tree-frame img{width:100%;height:auto;max-height:780px;object-fit:contain}#tree-viewport{flex:1;overflow:hidden;cursor:grab;user-select:none;position:relative}#tree-viewport.is-dragging{cursor:grabbing}#tree-canvas{transform-origin:0 0;will-change:transform}#tree-toolbar{display:flex;align-items:center;gap:8px;padding:10px 14px;border-bottom:1px solid var(--line);background:var(--surface-2);border-radius:18px 18px 0 0}.tree-ctrl{font:inherit;font-size:.8rem;border-radius:8px;border:1px solid var(--line);background:var(--surface);color:var(--ink);padding:.3rem .7rem;cursor:pointer;line-height:1}.tree-ctrl:hover{background:var(--surface-2)}.tree-zoom-label{font-size:.8rem;color:var(--muted);min-width:38px;text-align:center}#tree-tooltip{position:absolute;pointer-events:none;background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:.45rem .7rem;font-size:.82rem;color:var(--ink);box-shadow:var(--shadow);white-space:nowrap;opacity:0;transition:opacity .15s;z-index:10}.chart-card{padding:16px;margin-bottom:16px;min-width:0;overflow:hidden}.chart-card .js-plotly-plot,.chart-card .plotly-graph-div,.chart-card .plot-container,.chart-card .svg-container{width:100%!important;max-width:100%!important}.chart-card .main-svg{max-width:100%!important}.assumption-visual-grid{grid-template-columns:repeat(auto-fit,minmax(320px,1fr));margin-top:1rem;align-items:start}.toolbar{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:12px}.toolbar input,.toolbar button,.tree-button,.modal-close{font:inherit;border-radius:12px;border:1px solid var(--line);background:#fff;color:var(--ink);padding:.75rem .95rem}.toolbar button,.tree-button,.modal-close{cursor:pointer}.methods{white-space:pre-wrap;font-family:"Cascadia Mono",Consolas,monospace;padding:16px;background:rgba(22,49,58,.03)}.dataset-grid{grid-template-columns:repeat(auto-fit,minmax(260px,1fr))}.dataset-card{padding:20px;opacity:0;transform:translateY(16px);animation:riseIn .45s ease forwards}.muted{color:var(--muted)}.small{font-size:.92rem}.empty-state{padding:18px;color:var(--muted)}.footer-note{margin-top:26px;color:var(--muted);font-size:.92rem;text-align:center}.modal-backdrop{position:fixed;inset:0;background:rgba(6,10,12,.75);display:none;align-items:center;justify-content:center;padding:1.5rem;z-index:1000}.modal-backdrop.is-open{display:flex}.modal-panel{max-width:min(94vw,96rem);max-height:92vh;padding:1rem}.modal-panel img{display:block;max-width:100%;max-height:80vh;object-fit:contain}.modal-toolbar{display:flex;justify-content:space-between;align-items:center;gap:1rem;margin-bottom:.75rem}
@keyframes riseIn{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:none}}@keyframes treeIn{from{opacity:0}to{opacity:1}}#dyn-tree-host{width:100%;display:block}.num-cell{font-family:"Cascadia Mono",Consolas,"Courier New",monospace;font-variant-numeric:tabular-nums}.magnitude-large{background:rgba(31,122,90,.14);color:var(--success)}.magnitude-medium{background:rgba(183,121,31,.14);color:var(--warning)}.magnitude-small,.magnitude-negligible{background:rgba(22,49,58,.1);color:var(--muted)}#decision-path .decision-step::before{display:none}#decision-path .decision-step{padding:0;transform:none;opacity:1}
@media (prefers-color-scheme:dark){:root{--surface:#0f1a1c;--surface-2:#162428;--ink:#e8f0f2;--muted:#8ba4ac;--line:rgba(232,240,242,.1);--accent:#2dd4bf;--info:#38bdf8;--success:#34d399;--warning:#fbbf24;--danger:#f87171;--shadow:0 18px 40px rgba(0,0,0,.35)}body{background:#0a1214}.hero{background:linear-gradient(135deg,rgba(10,18,20,.98),rgba(15,118,110,.6))}th{background:rgba(232,240,242,.06)}.toolbar input,.toolbar button,.tree-button,.modal-close{background:#1a2e33;color:#e8f0f2}}
@media print{ #toc,.toolbar button,.tree-button,.modal-backdrop{display:none!important}.section{opacity:1!important;transform:none!important;page-break-inside:avoid}.hero{-webkit-print-color-adjust:exact;print-color-adjust:exact}.page{max-width:100%;padding:0}.decision-layout,.hero-grid{grid-template-columns:1fr}body{font-size:11pt;background:white;color:black}}
#toc{position:fixed;right:1rem;top:50%;transform:translateY(-50%);display:flex;flex-direction:column;gap:10px;z-index:100}#toc a{width:10px;height:10px;border-radius:50%;background:var(--line);text-decoration:none;transition:background .2s,transform .2s;position:relative}#toc a.is-active{background:var(--accent);transform:scale(1.4)}#toc a[data-label]:hover::after{content:attr(data-label);position:absolute;right:1.6rem;top:50%;transform:translateY(-50%);background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:.3rem .65rem;font-size:.78rem;white-space:nowrap;color:var(--ink);pointer-events:none;box-shadow:var(--shadow)}@media (max-width:1100px){ #toc{display:none}}@media (max-width:900px){.hero-grid,.decision-layout{grid-template-columns:1fr}}@media (prefers-reduced-motion:reduce){*,*:before,*:after{animation:none!important;transition:none!important}.section,.dataset-card{opacity:1;transform:none}}
.info-btn{display:inline-flex;align-items:center;justify-content:center;width:20px;height:20px;margin-left:8px;border-radius:50%;border:1.5px solid var(--muted);background:transparent;color:var(--muted);font-size:.92rem;font-family:"Segoe UI Symbol","Arial Unicode MS","Noto Sans Symbols","Segoe UI",sans-serif;font-style:normal;font-weight:700;line-height:1;cursor:pointer;vertical-align:middle;flex-shrink:0;transition:border-color .18s,color .18s,background .18s}
.info-btn:hover,.info-btn[aria-expanded="true"]{border-color:var(--accent);color:var(--accent);background:rgba(15,118,110,.09)}
.info-btn:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.info-panel{overflow:hidden;max-height:0;opacity:0;margin-top:0;transition:max-height .35s cubic-bezier(.4,0,.2,1),opacity .25s ease,margin-top .25s ease}
.info-panel.is-open{opacity:1;margin-top:8px}
.info-panel-inner{padding:.7rem .95rem;border-left:3px solid var(--accent);border-radius:0 10px 10px 0;background:rgba(15,118,110,.05);font-size:.875rem;line-height:1.65;color:var(--ink);white-space:pre-wrap;overflow-wrap:break-word;word-break:break-word}
@media(prefers-color-scheme:dark){.info-panel-inner{background:rgba(45,212,191,.06)}.is-info{background:rgba(56,189,248,.1)}}
@media print{.info-btn{display:none!important}.info-panel.is-open{max-height:none!important}}
</style>{{ plotly_bundle | safe }}</head><body><div class="page">
<header class="hero"><div class="eyebrow">BioMedStatX Offline Scientific Report</div><div class="hero-grid"><div><h1 class="hero-title">{{ context.report_title }}</h1><p class="hero-subtitle">{{ context.subtitle }}</p>{% if mode == "single" %}<div class="metrics-grid"><article class="metric-card"><div class="metric-label">Selected Test</div><div class="metric-value">{{ context.hero.test_name }}</div></article><article class="metric-card"><div class="metric-label">p-value</div><div class="metric-value">{{ context.hero.p_value_display }}</div></article><article class="metric-card"><div class="metric-label">{{ context.hero.effect_label }}</div><div class="metric-value">{{ context.hero.effect_size_display }}</div>{% if context.hero.effect_magnitude %}<div style="margin-top:.5rem"><span class="badge magnitude-{{ context.hero.effect_magnitude }}" style="font-size:.75rem;padding:.2rem .6rem;text-transform:capitalize">{{ context.hero.effect_magnitude }}</span></div>{% endif %}</article></div>{% endif %}</div><div>{% if mode == "single" %}<div class="badge {{ context.hero.significance_class }}">{{ context.hero.significance_label }}</div><p style="margin-top:14px;">{{ context.hero.summary_note }}</p>{% else %}<div class="badge is-neutral">Overview Report</div><p style="margin-top:14px;">This companion report summarizes the main outcome of each dataset while preserving fully offline viewing.</p>{% endif %}</div></div></header>
{% if mode == "single" %}<nav id="toc" aria-label="Sections"><a href="#sec-decision" data-label="Decision Path"></a><a href="#sec-results" data-label="Main Results"></a><a href="#sec-assumptions" data-label="Assumptions"></a><a href="#sec-descriptive" data-label="Descriptives"></a><a href="#sec-pairwise" data-label="Pairwise"></a><a href="#sec-charts" data-label="Charts"></a><a href="#sec-raw" data-label="Raw Data"></a><a href="#sec-methods" data-label="Methods"></a></nav>
<section id="sec-decision" class="section" data-reveal><div class="section-head"><div><div class="section-kicker">Decision Path</div><h2 id="hd-decision">How BioMedStatX reached this decision<button type="button" class="info-btn" aria-label="About this section" aria-expanded="false" aria-controls="info-decision">&#9432;</button></h2><div class="info-panel" id="info-decision" role="region" aria-labelledby="hd-decision"><div class="info-panel-inner">{{ context.info_texts.decision }}</div></div></div></div><div id="decision-path" style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px">{% for step in context.decision_path %}<div class="decision-step{% if step.active %} is-active{% endif %}" data-step="{{ loop.index0 }}" style="display:flex;align-items:center;gap:6px;padding:.35rem .8rem;border-radius:999px;border:1px solid var(--line);background:var(--surface-2);font-size:.82rem"><span style="width:8px;height:8px;border-radius:50%;flex-shrink:0;background:{% if step.active %}var(--accent){% else %}var(--line){% endif %}"></span><span>{{ step.title }}</span></div>{% endfor %}</div><div class="decision-tree-frame{% if context.decision_tree_json == 'null' %} is-empty{% endif %}">{% if context.decision_tree_json != "null" %}<div id="tree-toolbar"><button class="tree-ctrl" id="tree-zoom-in">＋</button><button class="tree-ctrl" id="tree-zoom-out">－</button><span class="tree-zoom-label" id="tree-zoom-pct">100%</span><button class="tree-ctrl" id="tree-reset">Reset</button><span style="flex:1"></span><button class="tree-ctrl" id="tree-replay" title="Replay path animation">&#9654; Replay</button><span class="small muted" style="margin-left:8px">Scroll to zoom · Drag to pan</span></div><div id="tree-viewport"><div id="tree-canvas"><div id="dyn-tree-host"></div></div><div id="tree-tooltip"></div></div>{% else %}<div class="empty-state">Decision tree was not available for this analysis.</div>{% endif %}</div></section>
<section id="sec-results" class="section" data-reveal><div class="section-head"><div><div class="section-kicker">Statistical Engine</div><h2 id="hd-results">Main results<button type="button" class="info-btn" aria-label="About this section" aria-expanded="false" aria-controls="info-results">&#9432;</button></h2><div class="info-panel" id="info-results" role="region" aria-labelledby="hd-results"><div class="info-panel-inner">{{ context.info_texts.results }}</div></div></div></div><div class="table-shell"><table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>{% for row in context.statistical_rows %}<tr><td>{{ row.label }}</td><td class="num-cell">{{ row.value }}</td></tr>{% endfor %}</tbody></table></div></section>
<section id="sec-assumptions" class="section" data-reveal><div class="section-head"><div><div class="section-kicker">Assumptions</div><h2 id="hd-assumptions">Model validity checks<button type="button" class="info-btn" aria-label="About this section" aria-expanded="false" aria-controls="info-assumptions">&#9432;</button></h2><div class="info-panel" id="info-assumptions" role="region" aria-labelledby="hd-assumptions"><div class="info-panel-inner">{{ context.info_texts.assumptions }}</div></div><p class="muted">{{ context.assumptions.interpretation }}</p>{% if context.assumptions.sphericity_correction_note %}<p class="muted" style="margin-top:.35rem;font-size:.88rem;color:var(--warning)">&#9888; {{ context.assumptions.sphericity_correction_note }}</p>{% endif %}</div><div class="badge is-info">Transformation: {{ context.assumptions.transformation }}</div></div>{% if context.assumptions.rows %}<div class="table-shell"><table><thead><tr><th>Check</th><th>Statistic</th><th>p-value</th><th>Status</th></tr></thead><tbody>{% for row in context.assumptions.rows %}<tr class="{{ row.status_class }}"><td>{{ row.name }}</td><td class="num-cell">{{ row.statistic }}</td><td class="num-cell" style="{{ row.p_value_style }}">{{ row.p_value }}</td><td>{{ row.status_label }}</td></tr>{% endfor %}</tbody></table></div>{% else %}<div class="empty-state">No structured assumption summary was available for this result.</div>{% endif %}<div class="assumption-visual-grid">{% if context.assumptions.qq_plot_html %}<div class="chart-card"><div class="section-kicker">Q-Q Diagnostic</div><h3>Observed quantiles against the normal reference line</h3>{{ context.assumptions.qq_plot_html | safe }}</div>{% endif %}{% if context.assumptions.distribution_plot_html %}<div class="chart-card"><div class="section-kicker">Group Distribution View</div><h3>Boxplots with jittered observations</h3>{{ context.assumptions.distribution_plot_html | safe }}</div>{% endif %}{% if context.assumptions.residual_plot_html %}<div class="chart-card"><div class="section-kicker">Residual Structure</div><h3>Residuals versus fitted values</h3>{{ context.assumptions.residual_plot_html | safe }}</div>{% endif %}{% if not context.assumptions.qq_plot_html and not context.assumptions.distribution_plot_html and not context.assumptions.residual_plot_html %}<div class="empty-state">Visual assumption diagnostics were not available for this result structure.</div>{% endif %}</div></section>
<section id="sec-descriptive" class="section" data-reveal><div class="section-head"><div><div class="section-kicker">Descriptive Statistics</div><h2 id="hd-descriptive">Group-level summary<button type="button" class="info-btn" aria-label="About this section" aria-expanded="false" aria-controls="info-descriptive">&#9432;</button></h2><div class="info-panel" id="info-descriptive" role="region" aria-labelledby="hd-descriptive"><div class="info-panel-inner">{{ context.info_texts.descriptive }}</div></div></div></div>{% if context.descriptive.rows %}<div class="table-shell"><table><thead><tr><th>Group</th><th>n</th><th>Mean</th><th>Median</th><th>SD</th><th>SEM</th><th>Min</th><th>Max</th></tr></thead><tbody>{% for row in context.descriptive.rows %}<tr><td>{{ row.group }}</td><td class="num-cell">{{ row.n }}</td><td class="num-cell">{{ row.mean }}</td><td class="num-cell">{{ row.median }}</td><td class="num-cell">{{ row.sd }}</td><td class="num-cell">{{ row.sem }}</td><td class="num-cell">{{ row.min }}</td><td class="num-cell">{{ row.max }}</td></tr>{% endfor %}</tbody></table></div>{% else %}<div class="empty-state">No descriptive summary could be derived from the available result payload.</div>{% endif %}</section>
{% if context.pairwise_rows %}<section id="sec-pairwise" class="section" data-reveal><div class="section-head"><div><div class="section-kicker">Pairwise Comparisons</div><h2 id="hd-pairwise">Post-hoc findings<button type="button" class="info-btn" aria-label="About this section" aria-expanded="false" aria-controls="info-pairwise">&#9432;</button></h2><div class="info-panel" id="info-pairwise" role="region" aria-labelledby="hd-pairwise"><div class="info-panel-inner">{{ context.info_texts.pairwise }}</div></div></div></div><div class="table-shell"><table><thead><tr>{% if context.group_chart_div_id %}<th title="Show bracket in chart">Chart</th>{% endif %}<th>Comparison</th><th>Procedure</th><th>Statistic</th><th>p-value</th><th>Effect size</th><th>Interpretation</th></tr></thead><tbody>{% for row in context.pairwise_rows %}<tr class="{{ row.row_class }}">{% if context.group_chart_div_id %}<td style="text-align:center"><input type="checkbox" class="bracket-toggle" data-pair-id="{{ row.pair_id }}" {% if row.significant and row.stars %}checked{% endif %} aria-label="Show bracket for {{ row.comparison }}"></td>{% endif %}<td>{{ row.comparison }}</td><td>{{ row.test }}</td><td class="num-cell">{{ row.statistic }}</td><td class="num-cell" style="{{ row.p_value_style }}">{{ row.p_value }}</td><td class="num-cell">{{ row.effect_size }}</td><td>{{ "Significant" if row.significant else "Not significant" }}</td></tr>{% endfor %}</tbody></table></div></section>{% endif %}
<section id="sec-charts" class="section" data-reveal><div class="section-head"><div><div class="section-kicker">Interactive Charts</div><h2 id="hd-charts">Visual evidence<button type="button" class="info-btn" aria-label="About this section" aria-expanded="false" aria-controls="info-charts">&#9432;</button></h2><div class="info-panel" id="info-charts" role="region" aria-labelledby="hd-charts"><div class="info-panel-inner">{{ context.info_texts.charts }}</div></div></div></div>{% if context.chart_blocks %}{% for chart in context.chart_blocks %}<div class="chart-card"><div class="section-kicker">{{ chart.title }}</div><h3>{{ chart.subtitle }}</h3>{{ chart.html | safe }}</div>{% endfor %}{% else %}<div class="empty-state">No interactive chart could be created from the current result structure.</div>{% endif %}</section>
<section id="sec-raw" class="section" data-reveal><div class="section-head"><div><div class="section-kicker">Raw Data Vault</div><h2 id="hd-raw">Searchable raw values<button type="button" class="info-btn" aria-label="About this section" aria-expanded="false" aria-controls="info-raw">&#9432;</button></h2><div class="info-panel" id="info-raw" role="region" aria-labelledby="hd-raw"><div class="info-panel-inner">{{ context.info_texts.raw }}</div></div></div></div><div class="toolbar"><input id="raw-search" type="search" placeholder="Filter raw data"><button type="button" onclick="copyTable('raw-data-table')">Copy table</button><button type="button" onclick="downloadTableCSV('raw-data-table','biomedstatx_raw_data.csv')">Download CSV</button></div>{% if context.raw_data_table.rows %}<div class="table-shell"><table id="raw-data-table"><thead><tr><th>Group</th><th>Index</th><th>Raw value</th>{% if context.raw_data_table.has_transformed %}<th>Transformed value</th>{% endif %}</tr></thead><tbody>{% for row in context.raw_data_table.rows %}<tr><td>{{ row.group }}</td><td>{{ row.index }}</td><td>{{ row.raw_value }}</td>{% if context.raw_data_table.has_transformed %}<td>{{ row.transformed_value }}</td>{% endif %}</tr>{% endfor %}</tbody></table></div>{% else %}<div class="empty-state">No raw data were embedded in this result structure.</div>{% endif %}</section>
<section id="sec-methods" class="section" data-reveal><div class="section-head"><div><div class="section-kicker">Methods Snippet</div><h2 id="hd-methods">Reusable narrative text<button type="button" class="info-btn" aria-label="About this section" aria-expanded="false" aria-controls="info-methods">&#9432;</button></h2><div class="info-panel" id="info-methods" role="region" aria-labelledby="hd-methods"><div class="info-panel-inner">{{ context.info_texts.methods }}</div></div></div></div><div class="toolbar"><button type="button" onclick="copyText('methods-text')">Copy methods text</button></div><div id="methods-text" class="methods">{{ context.methods_text }}</div></section>
{% else %}
<section class="section is-visible"><div class="section-head"><div><div class="section-kicker">Dataset Overview</div><h2>Summary across exported analyses</h2></div></div><div class="dataset-grid">{% for card in context.dataset_cards %}<article class="dataset-card"><div class="section-kicker">{{ card.dataset_name }}</div><h3>{{ card.test_name }}</h3><p class="small muted">{{ card.summary_note }}</p><div style="display:flex;gap:10px;flex-wrap:wrap;margin:12px 0 14px;"><span class="badge {{ card.significance_class }}">{{ card.significance_label }}</span><span class="badge is-neutral">{{ card.p_value_display }}</span></div><p class="small">Transformation: {{ card.transformation }}</p><p class="small">Pairwise comparisons: {{ card.pairwise_count }}</p>{% if card.assumptions.rows %}<div class="table-shell" style="margin-top:14px;"><table><thead><tr><th>Check</th><th>p-value</th><th>Status</th></tr></thead><tbody>{% for row in card.assumptions.rows[:4] %}<tr class="{{ row.status_class }}"><td>{{ row.name }}</td><td>{{ row.p_value }}</td><td>{{ row.status_label }}</td></tr>{% endfor %}</tbody></table></div>{% endif %}</article>{% endfor %}</div></section>
{% endif %}
<div class="footer-note">{% if context.generated_warning %}{{ context.generated_warning }}{% else %}Generated by BioMedStatX as a fully offline HTML scientific report.{% endif %}</div></div>{% if mode == "single" and context.decision_tree_image %}<div class="modal-backdrop" id="tree-modal" aria-hidden="true"><div class="modal-panel" role="dialog" aria-modal="true" aria-labelledby="tree-modal-title"><div class="modal-toolbar"><div><div class="section-kicker">Decision Tree</div><h3 id="tree-modal-title">Expanded analysis routing view</h3></div><button type="button" class="modal-close" id="close-tree-modal">Close</button></div><img src="{{ context.decision_tree_image }}" alt="Expanded decision tree visualization"></div></div>{% endif %}
<script>
const reduceMotion=window.matchMedia&&window.matchMedia('(prefers-reduced-motion: reduce)').matches;if(!reduceMotion&&'IntersectionObserver'in window){const observer=new IntersectionObserver((entries)=>{entries.forEach((entry)=>{if(entry.isIntersecting){entry.target.classList.add('is-visible');observer.unobserve(entry.target);}})},{threshold:.12});document.querySelectorAll('[data-reveal]').forEach((node)=>observer.observe(node));}else{document.querySelectorAll('[data-reveal]').forEach((node)=>node.classList.add('is-visible'));}
const decisionSteps=Array.from(document.querySelectorAll('#decision-path .decision-step'));if(decisionSteps.length&&!reduceMotion){decisionSteps.forEach((step)=>step.classList.remove('is-active'));decisionSteps.forEach((step,index)=>setTimeout(()=>step.classList.add('is-active'),220*index));}
{% if mode == "single" %}(function(){
const TREE_DATA={{ context.decision_tree_json | safe }};
const host=document.getElementById('dyn-tree-host');
const viewport=document.getElementById('tree-viewport');
const canvas=document.getElementById('tree-canvas');
const tooltip=document.getElementById('tree-tooltip');
if(!TREE_DATA||!host||!viewport)return;
const ns='http://www.w3.org/2000/svg';
function svgEl(tag,attrs){const e=document.createElementNS(ns,tag);Object.entries(attrs||{}).forEach(([k,v])=>e.setAttribute(k,String(v)));return e;}
const nodes=TREE_DATA.nodes,edges=TREE_DATA.edges;
const xs=nodes.map(n=>n.x),ys=nodes.map(n=>n.y);
const minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);
const NW=110,NH=44,PAD=80,SCALE=42;
const toSvg=(x,y)=>[(x-minX)*SCALE+PAD,(maxY-y)*SCALE+PAD];
const svgW=(maxX-minX)*SCALE+PAD*2,svgH=(maxY-minY)*SCALE+PAD*2;
const svg=svgEl('svg',{viewBox:'0 0 '+svgW+' '+svgH,width:svgW,height:svgH});
const defs=svgEl('defs');
['act','dim'].forEach(function(k){
  const mk=svgEl('marker',{id:'mk-'+k,markerWidth:9,markerHeight:7,refX:8,refY:3.5,orient:'auto'});
  const p=svgEl('polygon',{points:'0 0,9 3.5,0 7',fill:k==='act'?'#0f766e':'rgba(22,49,58,0.18)'});
  mk.appendChild(p);defs.appendChild(mk);
});
svg.appendChild(defs);
const nodeMap={};nodes.forEach(function(n){nodeMap[n.id]=n;});
const activeEdges=edges.filter(function(e){return e.isActive;});
function edgeKey(e){return String(e.source)+'->'+String(e.target);}
function nodeSvgY(nodeId){
    const n=nodeMap[nodeId];
    if(!n)return Number.POSITIVE_INFINITY;
    return toSvg(n.x,n.y)[1];
}
const activeBySource={};
const activeInDegree={};
activeEdges.forEach(function(e){
    if(!activeBySource[e.source])activeBySource[e.source]=[];
    activeBySource[e.source].push(e);
    activeInDegree[e.target]=(activeInDegree[e.target]||0)+1;
    if(activeInDegree[e.source]===undefined)activeInDegree[e.source]=0;
});
const startNodes=Object.keys(activeBySource)
    .filter(function(nodeId){return !activeInDegree[nodeId];})
    .sort(function(a,b){return nodeSvgY(a)-nodeSvgY(b);});
const orderedActiveEdges=[];
const seenActive={};
function appendOutgoing(nodeId){
    const outgoing=(activeBySource[nodeId]||[]).slice().sort(function(a,b){
        const dy=nodeSvgY(a.target)-nodeSvgY(b.target);
        if(dy!==0)return dy;
        return String(a.target).localeCompare(String(b.target));
    });
    outgoing.forEach(function(e){
        const key=edgeKey(e);
        if(seenActive[key])return;
        seenActive[key]=true;
        orderedActiveEdges.push(e);
        appendOutgoing(e.target);
    });
}
startNodes.forEach(appendOutgoing);
activeEdges.forEach(function(e){
    const key=edgeKey(e);
    if(!seenActive[key]){
        seenActive[key]=true;
        orderedActiveEdges.push(e);
    }
});
const activeOrderIndex={};
orderedActiveEdges.forEach(function(e,index){activeOrderIndex[edgeKey(e)]=index;});
const animLines=[];
[false,true].forEach(function(isAct){
  edges.forEach(function(e){
    if(e.isActive!==isAct)return;
    const s=nodeMap[e.source],t=nodeMap[e.target];if(!s||!t)return;
    const sp=toSvg(s.x,s.y),tp=toSvg(t.x,t.y);
    const x1=sp[0],y1=sp[1],x2=tp[0],y2=tp[1];
    const dx=x2-x1,dy=y2-y1,len=Math.sqrt(dx*dx+dy*dy)||1;
    const nx=dx/len,ny=dy/len;
    const absnx=Math.abs(nx),absny=Math.abs(ny);
    const ex1=x1+nx*Math.sqrt(Math.pow(NW/2*absnx+4,2)+Math.pow(NH/2*absny+3,2));
    const ey1=y1+ny*Math.sqrt(Math.pow(NW/2*absnx+4,2)+Math.pow(NH/2*absny+3,2));
    const ex2=x2-nx*Math.sqrt(Math.pow(NW/2*absnx+4,2)+Math.pow(NH/2*absny+3,2));
    const ey2=y2-ny*Math.sqrt(Math.pow(NW/2*absnx+4,2)+Math.pow(NH/2*absny+3,2));
    const line=svgEl('line',{x1:ex1,y1:ey1,x2:ex2,y2:ey2,
      stroke:isAct?'#0f766e':'rgba(22,49,58,0.13)',
      'stroke-width':isAct?2.5:1,
      'marker-end':isAct?'url(#mk-act)':'url(#mk-dim)'});
    if(isAct&&!reduceMotion){
      const el=Math.sqrt(Math.pow(ex2-ex1,2)+Math.pow(ey2-ey1,2));
      line.setAttribute('stroke-dasharray',el);
      line.setAttribute('stroke-dashoffset',el);
            const ord=(activeOrderIndex[edgeKey(e)]!==undefined)?activeOrderIndex[edgeKey(e)]:activeEdges.indexOf(e);
            animLines.push({line:line,len:el,ord:ord});
    }
    svg.appendChild(line);
  });
});
nodes.forEach(function(n){
  const p=toSvg(n.x,n.y),cx=p[0],cy=p[1];
  const g=svgEl('g',{});
  const fill=n.isActive?'rgba(15,118,110,0.12)':'rgba(22,49,58,0.03)';
  const stroke=n.isActive?'#0f766e':'rgba(22,49,58,0.20)';
  const rect=svgEl('rect',{x:cx-NW/2,y:cy-NH/2,width:NW,height:NH,rx:n.isSquare?5:16,fill:fill,stroke:stroke,'stroke-width':n.isActive?2:1});
  g.appendChild(rect);
  const lines=n.label.split('\n');
  const fs=n.isActive?11:9.5,lh=fs*1.3,totalH=lines.length*lh;
  lines.forEach(function(ln,li){
    const t=svgEl('text',{x:cx,y:cy-totalH/2+li*lh+lh*0.75,'text-anchor':'middle',
      'font-family':'"Segoe UI","Helvetica Neue",Arial,sans-serif',
      'font-size':fs,'font-weight':n.isActive?700:400,
      fill:n.isActive?'#0f766e':'rgba(22,49,58,0.68)'});
    t.textContent=ln;g.appendChild(t);
  });
  if(tooltip){
    const lbl=n.label.replace(/\n/g,' · ');
    g.addEventListener('mouseenter',function(){tooltip.textContent=lbl;tooltip.style.opacity='1';});
    g.addEventListener('mousemove',function(ev){const vr=viewport.getBoundingClientRect();tooltip.style.left=(ev.clientX-vr.left+14)+'px';tooltip.style.top=(ev.clientY-vr.top-10)+'px';});
    g.addEventListener('mouseleave',function(){tooltip.style.opacity='0';});
  }
  svg.appendChild(g);
});
host.appendChild(svg);
function playAnimation(){
  animLines.forEach(function(a){a.line.style.transition='none';a.line.setAttribute('stroke-dashoffset',a.len);});
  setTimeout(function(){animLines.forEach(function(a){setTimeout(function(){a.line.style.transition='stroke-dashoffset .5s ease';a.line.setAttribute('stroke-dashoffset',0);},a.ord*180);});},80);
}
if(!reduceMotion&&animLines.length)setTimeout(playAnimation,500);
const activeNodes=nodes.filter(function(n){return n.isActive;});
function buildFocusRect(){
    if(!activeNodes.length)return null;
    let minFx=Number.POSITIVE_INFINITY,maxFx=Number.NEGATIVE_INFINITY;
    let minFy=Number.POSITIVE_INFINITY,maxFy=Number.NEGATIVE_INFINITY;
    activeNodes.forEach(function(n){
        const p=toSvg(n.x,n.y),cx=p[0],cy=p[1];
        minFx=Math.min(minFx,cx);maxFx=Math.max(maxFx,cx);
        minFy=Math.min(minFy,cy);maxFy=Math.max(maxFy,cy);
    });
    const focusPad=26;
    minFx=minFx-NW/2-focusPad;maxFx=maxFx+NW/2+focusPad;
    minFy=minFy-NH/2-focusPad;maxFy=maxFy+NH/2+focusPad;
    minFx=Math.max(0,minFx);minFy=Math.max(0,minFy);
    maxFx=Math.min(svgW,maxFx);maxFy=Math.min(svgH,maxFy);
    return {x:minFx,y:minFy,w:Math.max(64,maxFx-minFx),h:Math.max(64,maxFy-minFy)};
}
const focusRect=buildFocusRect();
function getDefaultView(vw,vh){
    const margin=40;
    if(focusRect){
        const fit=Math.min((vw-margin)/focusRect.w,(vh-margin)/focusRect.h);
        const s=Math.min(Math.max(fit,0.15),4);
        return {sc:s,tx:(vw-focusRect.w*s)/2-focusRect.x*s,ty:(vh-focusRect.h*s)/2-focusRect.y*s};
    }
    const s=Math.min((vw-margin)/svgW,(vh-margin)/svgH,1);
    return {sc:s,tx:(vw-svgW*s)/2,ty:(vh-svgH*s)/2};
}
const vw=viewport.clientWidth||800,vh=viewport.clientHeight||500;
const initialView=getDefaultView(vw,vh);
let sc=initialView.sc,tx=initialView.tx,ty=initialView.ty;
function applyT(){canvas.style.transform='translate('+tx+'px,'+ty+'px) scale('+sc+')';const p=document.getElementById('tree-zoom-pct');if(p)p.textContent=Math.round(sc*100)+'%';}
function clampPan(){var mg=80,cw=viewport.clientWidth||800,ch=viewport.clientHeight||500;tx=Math.max(mg-svgW*sc,Math.min(cw-mg,tx));ty=Math.max(mg-svgH*sc,Math.min(ch-mg,ty));}
applyT();
let drag=false,dragX=0,dragY=0,stx=0,sty=0;
viewport.addEventListener('mousedown',function(e){drag=true;dragX=e.clientX;dragY=e.clientY;stx=tx;sty=ty;viewport.classList.add('is-dragging');});
window.addEventListener('mousemove',function(e){if(!drag)return;tx=stx+(e.clientX-dragX);ty=sty+(e.clientY-dragY);clampPan();applyT();});
window.addEventListener('mouseup',function(){drag=false;viewport.classList.remove('is-dragging');});
viewport.addEventListener('wheel',function(e){e.preventDefault();var raw=e.deltaY*0.0012;var zoom=Math.exp(-Math.max(-0.12,Math.min(0.12,raw)));var nsc=Math.min(Math.max(sc*zoom,0.12),4);var r=viewport.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;var act=nsc/sc;tx=mx-(mx-tx)*act;ty=my-(my-ty)*act;sc=nsc;clampPan();applyT();},{passive:false});
let t0=null;
viewport.addEventListener('touchstart',function(e){if(e.touches.length===1)t0={x:e.touches[0].clientX,y:e.touches[0].clientY,tx:tx,ty:ty};},{passive:true});
viewport.addEventListener('touchmove',function(e){if(e.touches.length===1&&t0){tx=t0.tx+(e.touches[0].clientX-t0.x);ty=t0.ty+(e.touches[0].clientY-t0.y);clampPan();applyT();}},{passive:true});
function zoomC(f){var cw=viewport.clientWidth/2,ch=viewport.clientHeight/2;tx=cw-(cw-tx)*f;ty=ch-(ch-ty)*f;sc=Math.min(Math.max(sc*f,0.15),4);clampPan();applyT();}
const zi=document.getElementById('tree-zoom-in'),zo=document.getElementById('tree-zoom-out'),rst=document.getElementById('tree-reset'),rpl=document.getElementById('tree-replay');
if(zi)zi.addEventListener('click',function(){zoomC(1.25);});
if(zo)zo.addEventListener('click',function(){zoomC(0.8);});
if(rst)rst.addEventListener('click',function(){var vw2=viewport.clientWidth||800,vh2=viewport.clientHeight||500;var d=getDefaultView(vw2,vh2);sc=d.sc;tx=d.tx;ty=d.ty;applyT();});
if(rpl)rpl.addEventListener('click',function(){playAnimation();});
})();{% endif %}
function copyText(elementId){const node=document.getElementById(elementId);if(!node)return;const text=node.innerText||node.textContent||'';navigator.clipboard.writeText(text).catch(()=>{});}
function tableToTSV(tableId){const table=document.getElementById(tableId);if(!table)return'';return Array.from(table.querySelectorAll('tr')).map((row)=>Array.from(row.querySelectorAll('th,td')).map((cell)=>(cell.innerText||'').replace(/\n/g,' ').trim()).join('\t')).join('\n');}
function copyTable(tableId){const text=tableToTSV(tableId);if(text){navigator.clipboard.writeText(text).catch(()=>{});}}
function downloadTableCSV(tableId,fileName){const table=document.getElementById(tableId);if(!table)return;const rows=Array.from(table.querySelectorAll('tr')).map((row)=>Array.from(row.querySelectorAll('th,td')).map((cell)=>{const value=(cell.innerText||'').replace(/\n/g,' ').trim().replace(/"/g,'""');return `"${value}"`;}).join(',')).join('\n');const blob=new Blob([rows],{type:'text/csv;charset=utf-8;'});const url=URL.createObjectURL(blob);const link=document.createElement('a');link.href=url;link.download=fileName;document.body.appendChild(link);link.click();document.body.removeChild(link);URL.revokeObjectURL(url);}
const rawSearch=document.getElementById('raw-search');if(rawSearch){rawSearch.addEventListener('input',(event)=>{const needle=String(event.target.value||'').toLowerCase();document.querySelectorAll('#raw-data-table tbody tr').forEach((row)=>{row.style.display=row.innerText.toLowerCase().includes(needle)?'':'none';});});}
(function(){function toggleInfo(btn){var p=document.getElementById(btn.getAttribute('aria-controls'));if(!p)return;var open=p.classList.toggle('is-open');btn.setAttribute('aria-expanded',open?'true':'false');if(open){p.style.maxHeight=p.scrollHeight+'px';}else{p.style.maxHeight='0';}}document.querySelectorAll('.info-btn').forEach(function(b){b.addEventListener('click',function(){toggleInfo(b);});b.addEventListener('keydown',function(e){if(e.key==='Enter'||e.key===' '){e.preventDefault();toggleInfo(b);}});});}());
const tocLinks=document.querySelectorAll('#toc a');if(tocLinks.length){const tocObserver=new IntersectionObserver((entries)=>{entries.forEach((entry)=>{const link=document.querySelector(`#toc a[href="#${entry.target.id}"]`);if(link)link.classList.toggle('is-active',entry.isIntersecting);});},{threshold:.25,rootMargin:'-10% 0px -65% 0px'});document.querySelectorAll('section[id]').forEach((sec)=>tocObserver.observe(sec));}
const treeModal=document.getElementById('tree-modal');const openTreeButton=document.getElementById('open-tree-modal');const treePreviewTrigger=document.getElementById('tree-preview-trigger');const closeTreeButton=document.getElementById('close-tree-modal');function openTreeModal(){if(!treeModal)return;treeModal.classList.add('is-open');treeModal.setAttribute('aria-hidden','false');}function closeTreeModal(){if(!treeModal)return;treeModal.classList.remove('is-open');treeModal.setAttribute('aria-hidden','true');}if(openTreeButton){openTreeButton.addEventListener('click',openTreeModal);}if(treePreviewTrigger){treePreviewTrigger.addEventListener('click',openTreeModal);treePreviewTrigger.addEventListener('keydown',(event)=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();openTreeModal();}});}if(closeTreeButton){closeTreeButton.addEventListener('click',closeTreeModal);}if(treeModal){treeModal.addEventListener('click',(event)=>{if(event.target===treeModal){closeTreeModal();}});document.addEventListener('keydown',(event)=>{if(event.key==='Escape'){closeTreeModal();}});}
{% if context.group_chart_div_id %}(function(){const BRACKET_DATA={{ context.bracket_data_json | safe }};const GROUP_ORDER={{ context.group_order_json | safe }};const CHART_DIV_ID="{{ context.group_chart_div_id }}";if(!BRACKET_DATA||!GROUP_ORDER||!CHART_DIV_ID)return;function getCheckedIds(){return new Set(Array.from(document.querySelectorAll('.bracket-toggle:checked')).map(cb=>parseInt(cb.dataset.pairId)));}function drawBrackets(){const chartDiv=document.getElementById(CHART_DIV_ID);const traces=chartDiv&&(chartDiv.data||chartDiv._fullData);if(!traces||!traces.length)return;const checkedIds=getCheckedIds();const groupToIdx={};GROUP_ORDER.forEach((name,i)=>{groupToIdx[name]=i;});const yVals=[];traces.forEach(trace=>{if(trace.y)trace.y.forEach(v=>{if(v!=null)yVals.push(v);});});if(!yVals.length)return;const yMin=Math.min(...yVals),yMax=Math.max(...yVals);const yRange=Math.max(Math.abs(yMax-yMin),1e-9);const step=yRange*0.13,tick=step*0.28;const active=BRACKET_DATA.filter(b=>checkedIds.has(b.pair_id)&&b.stars).map(b=>{let i1=groupToIdx[b.group1],i2=groupToIdx[b.group2];if(i1===undefined||i2===undefined)return null;if(i1>i2){[i1,i2]=[i2,i1];}return{...b,i1,i2};}).filter(Boolean);active.sort((a,b)=>(a.i2-a.i1)-(b.i2-b.i1)||a.i1-b.i1);const lineStyle={color:'rgba(22,49,58,0.65)',width:1.5};const shapes=[],annotations=[];active.forEach(({i1,i2,stars},level)=>{const y=yMax+step*(level+1);shapes.push({type:'line',x0:i1,x1:i2,y0:y,y1:y,xref:'x',yref:'y',line:lineStyle},{type:'line',x0:i1,x1:i1,y0:y-tick,y1:y,xref:'x',yref:'y',line:lineStyle},{type:'line',x0:i2,x1:i2,y0:y-tick,y1:y,xref:'x',yref:'y',line:lineStyle});annotations.push({x:(i1+i2)/2,y:y,text:'<b>'+stars+'</b>',showarrow:false,xref:'x',yref:'y',yshift:8,font:{size:13,color:'#16313a'}});});const newYMax=active.length?yMax+step*(active.length+1.8):yMax+yRange*0.05;Plotly.relayout(CHART_DIV_ID,{shapes:shapes,annotations:annotations,'yaxis.range':[yMin-yRange*0.05,newYMax]});}document.addEventListener('change',e=>{if(e.target.classList.contains('bracket-toggle'))drawBrackets();});function tryDraw(n){const d=document.getElementById(CHART_DIV_ID);const t=d&&(d.data||d._fullData);if(t&&t.length){drawBrackets();}else if(n>0){setTimeout(()=>tryDraw(n-1),250);}}tryDraw(30);})();{% endif %}
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
        stars = " ***" if numeric < 0.001 else " **" if numeric < 0.01 else " *" if numeric < 0.05 else " ns"
        p_str = "p < 0.001" if numeric < 0.001 else f"p = {numeric:.3f}"
        return f"{p_str}{stars}"

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
        if value is None:
            return "Not available"
        try:
            return "Passed" if bool(value) else "Flagged"
        except Exception:
            return "Not available"

    @staticmethod
    def _bool_class(value: Any) -> str:
        if value is None:
            return "is-neutral"
        try:
            return "is-significant" if bool(value) else "is-danger"
        except Exception:
            return "is-neutral"

    @staticmethod
    def _p_heat_style(p_val: Any) -> str:
        if not isinstance(p_val, (int, float)) or math.isnan(p_val):
            return ""
        if p_val < 0.001:
            return "background:rgba(31,122,90,.22)"
        if p_val < 0.01:
            return "background:rgba(31,122,90,.13)"
        if p_val < 0.05:
            return "background:rgba(183,121,31,.13)"
        if p_val < 0.1:
            return "background:rgba(159,58,56,.08)"
        return ""

    @staticmethod
    def _effect_size_magnitude(effect_size: Any, effect_type: str) -> str | None:
        """Cohen (1988) magnitude labels for common effect size metrics."""
        if not isinstance(effect_size, (int, float)) or math.isnan(float(effect_size)):
            return None
        es = abs(float(effect_size))
        et = str(effect_type or "").lower()
        if any(k in et for k in ["cohen", "hedge", " d", "'d"]):
            if es >= 0.8: return "large"
            if es >= 0.5: return "medium"
            if es >= 0.2: return "small"
            return "negligible"
        if any(k in et for k in ["eta", "omega", "epsilon", "η", "ω"]):
            if es >= 0.14: return "large"
            if es >= 0.06: return "medium"
            if es >= 0.01: return "small"
            return "negligible"
        if any(k in et for k in ["rho", "pearson", "spearman", "correlation"]) or et.strip() in ("r", "ρ"):
            if es >= 0.5: return "large"
            if es >= 0.3: return "medium"
            if es >= 0.1: return "small"
            return "negligible"
        if "cramer" in et or et.strip() == "v":
            if es >= 0.5: return "large"
            if es >= 0.3: return "medium"
            if es >= 0.1: return "small"
            return "negligible"
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
            print(f"WARNING HTML EXPORT: significance brackets failed: {exc}")

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
