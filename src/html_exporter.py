import base64
import copy
import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, TemplateNotFound
from scipy import stats


class _ResultsEncoder(json.JSONEncoder):
    def default(self, obj):
        return HTMLExporter._normalize_for_json(obj)


class HTMLExporter:
    _INLINE_LATEX_RE = re.compile(r"(?<!\\)\$(.+?)(?<!\\)\$")
    _BEGIN_ENV_RE = re.compile(r"\\\\begin\{[^}]+\}")
    _CSS_URL_RE = re.compile(r"url\((?P<quote>['\"]?)(?P<path>[^\)\"']+)(?P=quote)\)", re.IGNORECASE)

    @staticmethod
    def export_results_to_html(results: dict, output_file: str, analysis_log=None, pre_generated_tree=None) -> str | None:
        try:
            output_path = Path(output_file).resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            context = HTMLExporter._prepare_single_report_context(
                results, analysis_log=analysis_log, pre_generated_tree=pre_generated_tree
            )
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
    def _prepare_single_report_context(results: dict, analysis_log=None, pre_generated_tree=None) -> dict:
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

        raw_data = results_copy.get("raw_data") or results_copy.get("samples") or {}
        plot_data = {}
        stats_summary = {}
        max_points = 5000
        if isinstance(raw_data, dict):
            for group_name, values in raw_data.items():
                cleaned = HTMLExporter._coerce_numeric_sequence(values)
                if not cleaned:
                    continue
                group_key = str(group_name)
                plot_data[group_key] = HTMLExporter._downsample_for_display(cleaned, max_points=max_points)
                stats_summary[group_key] = HTMLExporter._summarize_numeric_group(cleaned)

        pairwise_payload = [
            {
                "pair_id": row.get("pair_id"),
                "group1": row.get("group1"),
                "group2": row.get("group2"),
                "comparison": row.get("comparison"),
                "p_value": row.get("p_value_raw"),
                "stars": row.get("stars", ""),
                "significant": bool(row.get("significant")),
            }
            for row in pairwise
        ]

        group_order = group_chart_block["group_order"] if group_chart_block else []
        if not group_order:
            group_order = list(plot_data.keys())

        plot_subject_trajectories = HTMLExporter._build_plot_subject_trajectories(
            results_copy,
            group_order=group_order,
            plot_data=plot_data,
        )
        plot_reference_lines = HTMLExporter._build_plot_reference_lines(results_copy)

        plot_designer_enabled = bool(plot_data)
        decision_tree_image = HTMLExporter._embed_decision_tree(results_copy, pre_generated_path=pre_generated_tree)
        decision_tree_json = HTMLExporter._build_decision_tree_json(results_copy)
        decision_path = HTMLExporter._build_decision_path_model(results_copy)
        methods_text = HTMLExporter._build_methods_text(results_copy, analysis_log_text)
        math_render_enabled = HTMLExporter._requires_math_rendering(results_copy, hero)
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
            "pairwise_data_json": json.dumps(pairwise_payload, cls=_ResultsEncoder, ensure_ascii=False),
            "plot_data_json": json.dumps(plot_data, cls=_ResultsEncoder, ensure_ascii=False),
            "plot_subject_trajectories_json": json.dumps(plot_subject_trajectories, cls=_ResultsEncoder, ensure_ascii=False),
            "plot_reference_lines_json": json.dumps(plot_reference_lines, cls=_ResultsEncoder, ensure_ascii=False),
            "stats_summary_json": json.dumps(stats_summary, cls=_ResultsEncoder, ensure_ascii=False),
            "plot_stats_json": json.dumps(stats_summary, cls=_ResultsEncoder, ensure_ascii=False),
            "plot_designer_enabled": plot_designer_enabled,
            "group_order_json": json.dumps(group_order, ensure_ascii=False),
            "group_chart_div_id": "biomedstatx-group-chart" if group_chart_block else "",
            "raw_data_table": raw_table,
            "chart_blocks": charts,
            "methods_text": methods_text,
            "info_texts": HTMLExporter._info_texts(),
            "generated_warning": results_copy.get("error"),
            "normalized_results_json": json.dumps(normalized, cls=_ResultsEncoder, ensure_ascii=False),
            "math_render_enabled": math_render_enabled,
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

        math_render_enabled = any(
            HTMLExporter._requires_math_rendering((results or {}), HTMLExporter._build_hero_context(results or {}, dataset_name=str(dataset_name)))
            for dataset_name, results in (all_results or {}).items()
        )

        return {
            "mode": "multi",
            "report_title": "BioMedStatX Multi-Dataset Scientific Report",
            "subtitle": f"{len(cards)} datasets summarized, {significant_count} significant main results.",
            "dataset_cards": cards,
            "generated_warning": None,
            "math_render_enabled": math_render_enabled,
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
    def _build_ancova_statistical_rows(results: dict) -> list[dict]:
        rows = []
        anova_table = results.get("anova_table") or []
        covariate_effects = results.get("covariate_effects") or []
        covariates_used = results.get("covariates_used") or []

        # --- Table 1: ANOVA Table (Type II SS) ---
        rows.append({"label": "── ANOVA Table (Type II SS) ──", "value": ""})
        rows.append({"label": "Source", "value": "Sum of Sq. | df | F | p-value | η²"})
        total_ss = sum(float(r.get("sum_sq") or 0) for r in anova_table)
        for entry in anova_table:
            source = str(entry.get("source", ""))
            ss = entry.get("sum_sq")
            df = entry.get("df")
            f_val = entry.get("F")
            p_val = entry.get("p_value")
            eta_sq = float(ss / total_ss) if (ss is not None and total_ss > 0) else None
            is_cov = any(cov in source for cov in covariates_used) and "C(" not in source
            label = f"[Cov] {source}" if is_cov else source
            value = (
                f"{HTMLExporter._format_metric(ss)} | "
                f"df={HTMLExporter._format_metric(df)} | "
                f"F={HTMLExporter._format_metric(f_val)} | "
                f"{HTMLExporter._format_p_value(p_val)} | "
                f"η²={HTMLExporter._format_metric(eta_sq)}"
            )
            rows.append({"label": label, "value": value})

        # --- Table 2: Covariate Effects ---
        if covariate_effects:
            rows.append({"label": "── Covariate Effects ──", "value": ""})
            rows.append({"label": "Covariate", "value": "Coefficient | SE | t | p-value | 95% CI"})
            for eff in covariate_effects:
                cov = str(eff.get("covariate", ""))
                beta = eff.get("coefficient")
                se = eff.get("std_err")
                t_val = eff.get("t_value")
                p_val = eff.get("p_value")
                ci_l = eff.get("ci_lower")
                ci_u = eff.get("ci_upper")
                value = (
                    f"β={HTMLExporter._format_metric(beta)} | "
                    f"SE={HTMLExporter._format_metric(se)} | "
                    f"t={HTMLExporter._format_metric(t_val)} | "
                    f"{HTMLExporter._format_p_value(p_val)} | "
                    f"[{HTMLExporter._format_metric(ci_l)}, {HTMLExporter._format_metric(ci_u)}]"
                )
                rows.append({"label": cov, "value": value})

        # --- Table 3: Model Fit ---
        rows.append({"label": "── Model Fit ──", "value": ""})
        rows.append({"label": "R²", "value": HTMLExporter._format_metric(results.get("r_squared"))})
        rows.append({"label": "Adjusted R²", "value": HTMLExporter._format_metric(results.get("r_squared_adj"))})
        rows.append({"label": "AIC", "value": HTMLExporter._format_metric(results.get("aic"))})
        rows.append({"label": "N observations", "value": HTMLExporter._format_metric(results.get("n_observations"))})

        return rows

    @staticmethod
    def _lmm_icc_interpretation(icc_val) -> str:
        if icc_val is None:
            return "N/A"
        try:
            v = float(icc_val)
        except (TypeError, ValueError):
            return "N/A"
        if v < 0.1:
            return "Negligible clustering"
        if v < 0.3:
            return "Small clustering"
        if v < 0.6:
            return "Moderate clustering"
        return "Strong clustering — LMM justified"

    @staticmethod
    def _build_lmm_statistical_rows(results: dict) -> list[dict]:
        rows = []

        # Table 1: Fixed Effects
        fe_table = results.get("fixed_effects_table") or []
        rows.append({"label": "── Fixed Effects ──", "value": ""})
        rows.append({"label": "Parameter", "value": "Coefficient | SE | z | p-value | 95% CI"})
        for fe in fe_table:
            param = str(fe.get("parameter", ""))
            coef = fe.get("coefficient")
            se = fe.get("std_err")
            z = fe.get("z_value")
            p_val = fe.get("p_value")
            ci_l = fe.get("ci_lower")
            ci_u = fe.get("ci_upper")
            value = (
                f"β={HTMLExporter._format_metric(coef)} | "
                f"SE={HTMLExporter._format_metric(se)} | "
                f"z={HTMLExporter._format_metric(z)} | "
                f"{HTMLExporter._format_p_value(p_val)} | "
                f"[{HTMLExporter._format_metric(ci_l)}, {HTMLExporter._format_metric(ci_u)}]"
            )
            rows.append({"label": param, "value": value})

        # Table 2: Random Effects & Model Fit
        rows.append({"label": "── Random Effects & Model Fit ──", "value": ""})
        rows.append({
            "label": "Random intercept variance",
            "value": HTMLExporter._format_metric(results.get("random_effects_variance")),
        })
        rows.append({
            "label": "Residual variance",
            "value": HTMLExporter._format_metric(results.get("residual_variance")),
        })
        icc_val = results.get("icc")
        icc_interp = HTMLExporter._lmm_icc_interpretation(icc_val)
        rows.append({
            "label": "Intraclass Correlation (ICC)",
            "value": f"{HTMLExporter._format_metric(icc_val)} — {icc_interp}",
        })
        rows.append({"label": "AIC", "value": HTMLExporter._format_metric(results.get("aic"))})
        rows.append({"label": "BIC", "value": HTMLExporter._format_metric(results.get("bic"))})
        rows.append({"label": "Log-likelihood", "value": HTMLExporter._format_metric(results.get("log_likelihood"))})
        rows.append({"label": "N subjects", "value": HTMLExporter._format_metric(results.get("n_subjects"))})
        rows.append({"label": "N observations", "value": HTMLExporter._format_metric(results.get("n_observations"))})

        converged = results.get("converged")
        if converged is None:
            conv_str = "Not available"
        elif converged:
            conv_str = "Yes"
        else:
            conv_str = "No — results may be unreliable"
        rows.append({"label": "Converged", "value": conv_str})

        return rows

    @staticmethod
    def _build_corr_matrix_statistical_rows(results: dict) -> list[dict]:
        """Dedicated statistical rows for CorrelationMatrix."""
        rows = [{"label": "── Correlation Matrix ──", "value": ""}]
        method_map = {
            "pearson": "Pearson",
            "spearman": "Spearman",
            "auto": "Auto (Pearson or Spearman per pair based on normality)",
        }
        method = str(results.get("method") or "").lower()
        rows.append({"label": "Method", "value": method_map.get(method, method or "—")})

        correction_map = {
            "fdr_bh": "Benjamini-Hochberg FDR",
            "bonferroni": "Bonferroni",
        }
        correction = results.get("correction")
        rows.append({
            "label": "Correction method",
            "value": correction_map.get(str(correction or "").lower(), str(correction) if correction else "None"),
        })

        variables = results.get("variables") or []
        n_vars = len(variables)
        rows.append({"label": "N variables", "value": str(n_vars)})

        n_tests = n_vars * (n_vars - 1) // 2
        rows.append({"label": "N pairwise tests", "value": str(n_tests)})

        # Count significant pairs (upper triangle only)
        p_matrix = results.get("p_matrix") or {}
        p_corr_matrix = results.get("p_corrected_matrix") or {}
        n_sig_raw = 0
        n_sig_corr = 0
        for idx_i, var_i in enumerate(variables):
            for idx_j, var_j in enumerate(variables):
                if idx_j <= idx_i:
                    continue
                p_raw = (p_matrix.get(var_i) or {}).get(var_j)
                p_corr = (p_corr_matrix.get(var_i) or {}).get(var_j)
                if isinstance(p_raw, (int, float)) and p_raw < 0.05:
                    n_sig_raw += 1
                if isinstance(p_corr, (int, float)) and p_corr < 0.05:
                    n_sig_corr += 1
        rows.append({"label": "N significant (uncorrected)", "value": str(n_sig_raw)})
        rows.append({"label": "N significant (FDR-corrected)", "value": str(n_sig_corr)})

        if results.get("pairwise_deletion"):
            rows.append({"label": "Missing data handling", "value": "Pairwise deletion — n varies per pair"})

        strata = results.get("strata")
        if strata:
            rows.append({"label": "N strata", "value": str(len(strata))})

        if not rows[1:]:
            rows.append({"label": "Status", "value": "No structured statistical summary available."})
        return rows

    @staticmethod
    def _build_beta_statistical_rows(results: dict) -> list[dict]:
        """Dedicated statistical rows for Beta Regression.
        The coefficient table is rendered separately as an HTML block via chart_blocks."""
        rows = [{"label": "── Model Fit ──", "value": ""}]
        for label, key in [
            ("Test", "test"),
            ("Model type", "model_type"),
            ("p-value (primary predictor)", "p_value"),
        ]:
            value = results.get(key)
            if key in results and HTMLExporter._has_display_value(value):
                display = HTMLExporter._format_p_value(value) if key.startswith("p_value") else HTMLExporter._format_metric(value)
                rows.append({"label": label, "value": display})

        pseudo_r2 = results.get("pseudo_r_squared")
        if pseudo_r2 is not None:
            rows.append({"label": "Pseudo-R² (McFadden)", "value": HTMLExporter._format_metric(pseudo_r2)})

        phi = results.get("phi")
        if phi is not None:
            phi_f = float(phi)
            if phi_f < 1:
                phi_interp = "High variance relative to mean"
            elif phi_f <= 5:
                phi_interp = "Moderate dispersion"
            else:
                phi_interp = "Low dispersion — precise estimates"
            rows.append({"label": "Dispersion parameter (φ)", "value": f"{HTMLExporter._format_metric(phi)} — {phi_interp}"})

        for label, key in [
            ("AIC", "aic"),
            ("BIC", "bic"),
            ("N observations", "n_observations"),
        ]:
            value = results.get(key)
            if value is not None:
                rows.append({"label": label, "value": HTMLExporter._format_metric(value)})

        bc = results.get("bias_corrected")
        if bc is not None:
            rows.append({"label": "Bias corrected", "value": "Yes" if bc else "No"})
            if bc:
                bc_method = results.get("bias_correction_method")
                if bc_method:
                    rows.append({"label": "Bias correction method", "value": str(bc_method)})

        if len(rows) <= 1:
            rows.append({"label": "Status", "value": "No structured statistical summary available."})
        return rows

    @staticmethod
    def _build_logistic_statistical_rows(results: dict) -> list[dict]:
        """Dedicated statistical rows for Logistic Regression: Model Fit only.
        The OR table is rendered separately as an HTML block via chart_blocks."""
        rows = [
            {"label": "── Model Fit ──", "value": ""},
        ]
        for label, key in [
            ("Test", "test"),
            ("Model type", "model_type"),
            ("p-value (primary predictor)", "p_value"),
            ("Adjusted p-value", "p_value_fdr"),
        ]:
            value = results.get(key)
            if key in results and HTMLExporter._has_display_value(value):
                display = HTMLExporter._format_p_value(value) if key.startswith("p_value") else HTMLExporter._format_metric(value)
                rows.append({"label": label, "value": display})

        auc = results.get("effect_size")
        if auc is not None:
            rows.append({"label": "AUC (ROC)", "value": HTMLExporter._format_metric(auc)})

        pseudo_r2 = results.get("pseudo_r_squared")
        if pseudo_r2 is not None:
            rows.append({"label": "McFadden pseudo-R²", "value": HTMLExporter._format_metric(pseudo_r2)})

        aic = results.get("aic")
        if aic is not None:
            rows.append({"label": "AIC", "value": HTMLExporter._format_metric(aic)})

        bic = results.get("bic")
        if bic is not None:
            rows.append({"label": "BIC", "value": HTMLExporter._format_metric(bic)})

        n = results.get("n_observations")
        if n is not None:
            rows.append({"label": "N observations", "value": HTMLExporter._format_metric(n)})

        if len(rows) <= 1:
            rows.append({"label": "Status", "value": "No structured statistical summary available."})
        return rows

    @staticmethod
    def _build_statistical_rows(results: dict) -> list[dict]:
        rows = []
        model_type = results.get("model_type", "")
        statistic_type = results.get("statistic_type", "")

        if model_type == "ANCOVA":
            return HTMLExporter._build_ancova_statistical_rows(results)

        if model_type == "LMM":
            return HTMLExporter._build_lmm_statistical_rows(results)

        if model_type == "LogisticRegression":
            return HTMLExporter._build_logistic_statistical_rows(results)

        if model_type == "BetaRegression":
            return HTMLExporter._build_beta_statistical_rows(results)

        if model_type == "CorrelationMatrix":
            return HTMLExporter._build_corr_matrix_statistical_rows(results)

        # Determine the label for the "statistic" row based on model type
        if statistic_type == "odds_ratio":
            stat_label = "Odds Ratio (first predictor)"
        elif statistic_type == "coefficient":
            stat_label = "Coefficient (first predictor)"
        else:
            stat_label = "Statistic"

        for label, key in [
            ("Test", "test"),
            ("Model type", "model_type"),
            (stat_label, "statistic"),
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

        # For Logistic Regression: add Wald z-statistic from odds_ratios table
        if model_type == "LogisticRegression":
            or_table = results.get("odds_ratios") or []
            if or_table and isinstance(or_table[0], dict):
                z_val = or_table[0].get("z_value")
                if z_val is not None:
                    rows.append({"label": "Wald z-statistic (first predictor)", "value": HTMLExporter._format_metric(z_val)})

        # For Correlation: add sample size, interpretation, and statistic type
        if model_type == "Correlation":
            n_val = results.get("n")
            if n_val is not None:
                rows.append({"label": "Sample size (n)", "value": HTMLExporter._format_metric(n_val)})
            method = str(results.get("method") or "").capitalize()
            if method:
                rows.append({"label": "Correlation method", "value": method})
            interp = results.get("interpretation")
            if interp:
                rows.append({"label": "Interpretation", "value": str(interp)})

        # For Brunner-Langer ATS: add statistic type label and RTE table rows
        if model_type == "BrunnerLangerATS":
            rows.append({"label": "Statistic type", "value": "ANOVA-Type Statistic (ATS) — not a standard F-value"})
            rte = results.get("RTE")
            if rte is not None:
                rte_rows = []
                try:
                    if isinstance(rte, pd.DataFrame):
                        rte_rows = rte.to_dict(orient="records")
                except Exception:
                    pass
                if not rte_rows and isinstance(rte, dict) and "data" in rte:
                    cols = rte.get("columns", [])
                    rte_rows = [dict(zip(cols, row)) for row in (rte.get("data") or [])]
                if rte_rows:
                    rows.append({"label": "Relative Treatment Effects (RTE)", "value": "RTE near 0.5 = no effect"})
                    for rte_row in rte_rows:
                        between = rte_row.get("between_group", "")
                        within = rte_row.get("within_level", "")
                        rte_val = rte_row.get("RTE")
                        n_cell = rte_row.get("n")
                        group_label = f"RTE: {between} / {within}"
                        rte_display = HTMLExporter._format_metric(rte_val)
                        if n_cell is not None:
                            rte_display += f"  (n={int(n_cell)})"
                        rows.append({"label": group_label, "value": rte_display})

        # For Freedman-Lane Permutation: add n_permutations
        if model_type == "FreedmanLanePermutation":
            n_perm = results.get("n_permutations")
            if n_perm is not None:
                rows.append({"label": "Permutations (n)", "value": HTMLExporter._format_metric(n_perm)})
            stat_type = results.get("StatisticType")
            if stat_type:
                rows.append({"label": "Statistic type", "value": str(stat_type)})

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
        model_type = results.get("model_type", "")

        # --- Beta Regression: residual normality, S-V transformation, EPV ---
        if model_type == "BetaRegression":
            residuals = HTMLExporter._coerce_numeric_sequence(results.get("residuals"))
            if residuals and len(residuals) >= 3:
                try:
                    from scipy import stats as _stats
                    sw_stat, sw_p = _stats.shapiro(residuals)
                    sw_normal = sw_p >= 0.05
                    rows.append({
                        "name": "Residual normality (Shapiro-Wilk)",
                        "statistic": HTMLExporter._format_metric(sw_stat),
                        "p_value": HTMLExporter._format_p_value(sw_p),
                        "p_value_style": HTMLExporter._p_heat_style(sw_p),
                        "status_label": HTMLExporter._bool_label(sw_normal),
                        "status_class": HTMLExporter._bool_class(sw_normal),
                    })
                except Exception:
                    rows.append({
                        "name": "Residual normality (Shapiro-Wilk)",
                        "statistic": "—",
                        "p_value": "—",
                        "p_value_style": "",
                        "status_label": "Assessed visually via Q-Q plot",
                        "status_class": "is-neutral",
                    })
            else:
                rows.append({
                    "name": "Residual normality (Shapiro-Wilk)",
                    "statistic": "—",
                    "p_value": "—",
                    "p_value_style": "",
                    "status_label": "Assessed visually via Q-Q plot",
                    "status_class": "is-neutral",
                })

            if results.get("sv_transformed"):
                rows.append({
                    "name": "Smithson-Verkuilen transformation",
                    "statistic": "Applied",
                    "p_value": "—",
                    "p_value_style": "",
                    "status_label": "Boundary values present — squeezed from [0,1] to strictly (0,1)",
                    "status_class": "is-neutral",
                })

            epv = results.get("epv")
            if epv is not None:
                epv_f = float(epv)
                if epv_f < 10:
                    epv_label = f"EPV = {HTMLExporter._format_metric(epv)} — Small sample relative to predictors — bias-corrected estimation applied"
                    epv_class = "is-danger"
                else:
                    epv_label = f"EPV = {HTMLExporter._format_metric(epv)} — Adequate sample size"
                    epv_class = "is-significant"
                rows.append({
                    "name": "Events per variable (EPV)",
                    "statistic": HTMLExporter._format_metric(epv),
                    "p_value": "—",
                    "p_value_style": "",
                    "status_label": epv_label,
                    "status_class": epv_class,
                })

        # --- CorrelationMatrix: method justification ---
        if model_type == "CorrelationMatrix":
            method = str(results.get("method") or "").lower()
            method_map = {
                "pearson": "Pearson",
                "spearman": "Spearman",
                "auto": "Auto (Pearson or Spearman selected per pair based on Shapiro-Wilk normality test)",
            }
            method_label = method_map.get(method, method or "—")
            if method == "pearson":
                status_label = "Pearson requires bivariate normality — verify via Q-Q plots"
                status_class = "is-neutral"
            elif method == "spearman":
                status_label = "Spearman is distribution-free — no normality required"
                status_class = "is-significant"
            else:
                status_label = "Method auto-selected per pair — verify individual pair choices"
                status_class = "is-neutral"
            rows.append({
                "name": f"Correlation method: {method_label}",
                "statistic": "—",
                "p_value": "—",
                "p_value_style": "",
                "status_label": status_label,
                "status_class": status_class,
            })
            correction = results.get("correction")
            correction_map = {"fdr_bh": "Benjamini-Hochberg FDR", "bonferroni": "Bonferroni"}
            corr_label = correction_map.get(str(correction or "").lower(), str(correction) if correction else "None")
            rows.append({
                "name": f"Multiple testing correction: {corr_label}",
                "statistic": "—",
                "p_value": "—",
                "p_value_style": "",
                "status_label": "Controls false discovery rate across all pairwise tests" if correction else "No correction applied",
                "status_class": "is-significant" if correction else "is-neutral",
            })
            if results.get("pairwise_deletion"):
                rows.append({
                    "name": "Missing data handling",
                    "statistic": "—",
                    "p_value": "—",
                    "p_value_style": "",
                    "status_label": "Pairwise deletion — each pair uses all available complete cases",
                    "status_class": "is-neutral",
                })

        # --- Correlation: normality_check (Shapiro-Wilk per variable for method selection) ---
        normality_check = results.get("normality_check") or {}
        if model_type == "Correlation" and normality_check:
            x_var = results.get("x_variable", "")
            y_var = results.get("y_variable", "")
            for var_key, var_label in [(x_var, results.get("x_variable_display", x_var)),
                                       (y_var, results.get("y_variable_display", y_var))]:
                payload = normality_check.get(var_key, {})
                if isinstance(payload, dict) and "p_value" in payload:
                    rows.append({
                        "name": f"Normality: {HTMLExporter._prettify_label(var_label)} (Shapiro-Wilk)",
                        "statistic": HTMLExporter._format_metric(payload.get("statistic")),
                        "p_value": HTMLExporter._format_p_value(payload.get("p_value")),
                        "p_value_style": HTMLExporter._p_heat_style(payload.get("p_value")),
                        "status_label": HTMLExporter._bool_label(payload.get("normal")),
                        "status_class": HTMLExporter._bool_class(payload.get("normal")),
                    })

        # --- Correlation: document pre-transformation shift (reproducibility requirement) ---
        if model_type == "Correlation":
            x_shift = results.get("x_transform_shift") or 0.0
            y_shift = results.get("y_transform_shift") or 0.0
            x_tr = results.get("x_transform") or "none"
            y_tr = results.get("y_transform") or "none"
            x_lam = results.get("x_boxcox_lambda")
            y_lam = results.get("y_boxcox_lambda")
            for axis, tr_name, shift, lam, disp in [
                ("X", x_tr, x_shift, x_lam, results.get("x_variable_display") or results.get("x_variable") or "X"),
                ("Y", y_tr, y_shift, y_lam, results.get("y_variable_display") or results.get("y_variable") or "Y"),
            ]:
                if tr_name == "none":
                    continue
                min_raw = (1.0 - shift) if tr_name in ('log10', 'boxcox') else (-shift)
                if shift != 0.0:
                    detail = (
                        f"c={shift:.4f} automatically added (min raw = {min_raw:.4f}); "
                        f"determined from data, not a researcher choice"
                    )
                    if lam is not None:
                        detail += f"; λ={lam:.4f}"
                else:
                    detail = "No shift required (all values > 0)"
                    if lam is not None:
                        detail += f"; λ={lam:.4f}"
                rows.append({
                    "name": f"Shift before {tr_name}: {HTMLExporter._prettify_label(disp)} ({axis}-axis)",
                    "statistic": f"c={shift:.4f}",
                    "p_value": "—",
                    "p_value_style": "",
                    "status_label": detail,
                    "status_class": "is-neutral",
                })

        # --- Linear Regression: diagnostics (Shapiro-Wilk residuals, Breusch-Pagan, Ramsey RESET) ---
        elif model_type == "LinearRegression":
            diag = results.get("diagnostics") or {}
            norm_d = diag.get("normality") or {}
            if norm_d and "p_value" in norm_d:
                rows.append({
                    "name": "Normality of residuals (Shapiro-Wilk)",
                    "statistic": HTMLExporter._format_metric(norm_d.get("statistic")),
                    "p_value": HTMLExporter._format_p_value(norm_d.get("p_value")),
                    "p_value_style": HTMLExporter._p_heat_style(norm_d.get("p_value")),
                    "status_label": HTMLExporter._bool_label(norm_d.get("assumption_holds")),
                    "status_class": HTMLExporter._bool_class(norm_d.get("assumption_holds")),
                })
            homo_d = diag.get("homoscedasticity") or {}
            if homo_d and "p_value" in homo_d:
                rows.append({
                    "name": "Homoscedasticity (Breusch-Pagan)",
                    "statistic": HTMLExporter._format_metric(homo_d.get("statistic")),
                    "p_value": HTMLExporter._format_p_value(homo_d.get("p_value")),
                    "p_value_style": HTMLExporter._p_heat_style(homo_d.get("p_value")),
                    "status_label": HTMLExporter._bool_label(homo_d.get("assumption_holds")),
                    "status_class": HTMLExporter._bool_class(homo_d.get("assumption_holds")),
                })
            lin_d = diag.get("linearity") or {}
            if lin_d and "p_value" in lin_d:
                rows.append({
                    "name": "Linearity (Ramsey RESET)",
                    "statistic": HTMLExporter._format_metric(lin_d.get("statistic")),
                    "p_value": HTMLExporter._format_p_value(lin_d.get("p_value")),
                    "p_value_style": HTMLExporter._p_heat_style(lin_d.get("p_value")),
                    "status_label": HTMLExporter._bool_label(lin_d.get("assumption_holds")),
                    "status_class": HTMLExporter._bool_class(lin_d.get("assumption_holds")),
                })

        # --- Logistic Regression: Hosmer-Lemeshow goodness-of-fit + AUC interpretation ---
        elif model_type == "LogisticRegression":
            hl = results.get("hosmer_lemeshow") or {}
            if hl and "p_value" in hl:
                hl_p = hl.get("p_value")
                # HL passes (good fit) when p > 0.05 (no significant deviation from expected)
                goodness_of_fit = isinstance(hl_p, (int, float)) and hl_p > 0.05
                rows.append({
                    "name": "Goodness-of-fit (Hosmer-Lemeshow)",
                    "statistic": HTMLExporter._format_metric(hl.get("chi2")),
                    "p_value": HTMLExporter._format_p_value(hl_p),
                    "p_value_style": HTMLExporter._p_heat_style(hl_p),
                    "status_label": HTMLExporter._bool_label(goodness_of_fit),
                    "status_class": HTMLExporter._bool_class(goodness_of_fit),
                })
            auc = results.get("effect_size")
            if auc is not None and isinstance(auc, (int, float)):
                if auc < 0.6:
                    auc_interp, auc_ok = "Poor discrimination", False
                elif auc < 0.7:
                    auc_interp, auc_ok = "Acceptable", True
                elif auc < 0.8:
                    auc_interp, auc_ok = "Good", True
                elif auc < 0.9:
                    auc_interp, auc_ok = "Excellent", True
                else:
                    auc_interp, auc_ok = "Outstanding", True
                rows.append({
                    "name": "Discrimination (AUC interpretation)",
                    "statistic": HTMLExporter._format_metric(auc),
                    "p_value": "—",
                    "p_value_style": "",
                    "status_label": auc_interp,
                    "status_class": "is-significant" if auc_ok else "is-danger",
                })

        # --- ANCOVA: residual normality + slope homogeneity ---
        elif model_type == "ANCOVA":
            normality_tests = results.get("normality_tests") or {}
            if normality_tests:
                for label, payload in normality_tests.items():
                    if not isinstance(payload, dict):
                        continue
                    rows.append({
                        "name": f"Normality: {HTMLExporter._prettify_label(label)} (Shapiro-Wilk)",
                        "statistic": HTMLExporter._format_metric(payload.get("statistic")),
                        "p_value": HTMLExporter._format_p_value(payload.get("p_value")),
                        "p_value_style": HTMLExporter._p_heat_style(payload.get("p_value")),
                        "status_label": HTMLExporter._bool_label(payload.get("is_normal")),
                        "status_class": HTMLExporter._bool_class(payload.get("is_normal")),
                    })
            else:
                rows.append({
                    "name": "Residual Normality (Shapiro-Wilk)",
                    "statistic": "N/A",
                    "p_value": "—",
                    "p_value_style": "",
                    "status_label": "Assessed visually via Q-Q plot",
                    "status_class": "is-neutral",
                })
            slope_hom = results.get("slope_homogeneity") or {}
            for interaction_key, payload in slope_hom.items():
                if not isinstance(payload, dict):
                    continue
                f_val = payload.get("F")
                p_val = payload.get("p_value")
                holds = payload.get("assumption_holds")
                rows.append({
                    "name": f"Slope Homogeneity: {interaction_key}",
                    "statistic": HTMLExporter._format_metric(f_val),
                    "p_value": HTMLExporter._format_p_value(p_val),
                    "p_value_style": HTMLExporter._p_heat_style(p_val),
                    "status_label": HTMLExporter._bool_label(holds),
                    "status_class": HTMLExporter._bool_class(holds),
                })

        # --- LMM: convergence + ICC + residual normality ---
        elif model_type == "LMM":
            rows.append({
                "name": "Residual Normality",
                "statistic": "—",
                "p_value": "—",
                "p_value_style": "",
                "status_label": "Assessed visually via Q-Q plot",
                "status_class": "is-neutral",
            })
            converged = results.get("converged")
            conv_holds = bool(converged) if converged is not None else None
            rows.append({
                "name": "Model Convergence",
                "statistic": "—",
                "p_value": "—",
                "p_value_style": "",
                "status_label": "Yes" if converged is True else ("No" if converged is False else "Not available"),
                "status_class": HTMLExporter._bool_class(conv_holds),
            })
            icc_val = results.get("icc")
            icc_interp = HTMLExporter._lmm_icc_interpretation(icc_val)
            icc_justified = (float(icc_val) >= 0.1) if isinstance(icc_val, (int, float)) else None
            rows.append({
                "name": "Intraclass Correlation (ICC) — ICC > 0.1 justifies LMM",
                "statistic": HTMLExporter._format_metric(icc_val),
                "p_value": icc_interp,
                "p_value_style": "",
                "status_label": "LMM justified" if icc_justified else ("ICC < 0.1 — LMM may be unnecessary" if icc_justified is False else "N/A"),
                "status_class": "is-significant" if icc_justified else ("is-neutral" if icc_justified is None else "is-danger"),
            })

        # --- Standard tests: normality_tests + fallback from test_info ---
        elif model_type == "BetaRegression":
            pass  # handled above in dedicated Beta Regression block
        elif model_type == "CorrelationMatrix":
            pass  # handled above in dedicated CorrelationMatrix block
        else:
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
        # For ANCOVA/LMM: inject model-specific note shown by the template
        model_type_check = results.get("model_type", "")
        if model_type_check == "ANCOVA":
            slope_hom = results.get("slope_homogeneity") or {}
            any_violated = any(
                isinstance(v, dict) and v.get("assumption_holds") is False
                for v in slope_hom.values()
            )
            if any_violated:
                sphericity_correction_note = (
                    "Homogeneity of regression slopes violated — the covariate effect differs between groups. "
                    "ANCOVA results should be interpreted with caution; consider interaction models."
                )
            else:
                sphericity_correction_note = (
                    "Homogeneity of regression slopes: key ANCOVA assumption. "
                    "Parallel regression slopes support valid covariate adjustment."
                )
        elif model_type_check == "LMM":
            converged_check = results.get("converged")
            if converged_check is False:
                sphericity_correction_note = (
                    "Model did not converge — results may be unreliable. "
                    "Consider simplifying the random effects structure."
                )
            else:
                sphericity_correction_note = None
        else:
            sphericity_correction_note = None

        sphericity = results.get("sphericity_test", {}) or {}
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
        model_type = results.get("model_type", "")

        # --- ANCOVA: adjusted group means ---
        if model_type == "ANCOVA":
            adjusted_means = results.get("adjusted_means") or {}
            covariates_used = results.get("covariates_used") or []
            rows = []
            for factor, levels in adjusted_means.items():
                if not isinstance(levels, dict):
                    continue
                for level, level_data in levels.items():
                    if not isinstance(level_data, dict):
                        continue
                    rows.append({
                        "group": str(level),
                        "n": HTMLExporter._format_metric(level_data.get("n")),
                        "raw_mean": HTMLExporter._format_metric(level_data.get("raw_mean")),
                        "adj_mean": HTMLExporter._format_metric(level_data.get("adjusted_mean")),
                        "raw_sd": HTMLExporter._format_metric(level_data.get("raw_sd")),
                    })
            cov_str = ", ".join(covariates_used) if covariates_used else "—"
            return {
                "rows": rows,
                "has_transformed": False,
                "title": "Adjusted Group Means",
                "group_col_label": "Group",
                "columns": ["Group", "N", "Raw Mean", "Adj. Mean", "Raw SD"],
                "column_keys": ["group", "n", "raw_mean", "adj_mean", "raw_sd"],
                "note": f"Means adjusted for covariates: {cov_str}",
            }

        # --- Correlation / Linear Regression: bivariate variable summary from association_points ---
        if model_type in ("Correlation", "LinearRegression"):
            points = results.get("association_points") or []
            x_label = results.get("x_variable_display") or results.get("x_variable") or "X"
            y_label = results.get("y_variable_display") or results.get("y_variable") or "Y"
            rows = []
            if points:
                x_vals = [p["x"] for p in points if "x" in p]
                y_vals = [p["y"] for p in points if "y" in p]
                for label, vals in [(x_label, x_vals), (y_label, y_vals)]:
                    if not vals:
                        continue
                    arr = np.array(vals, dtype=float)
                    rows.append({
                        "group": str(label),
                        "n": len(arr),
                        "mean": HTMLExporter._format_metric(float(np.mean(arr))),
                        "median": HTMLExporter._format_metric(float(np.median(arr))),
                        "sd": HTMLExporter._format_metric(float(np.std(arr, ddof=1)) if len(arr) > 1 else None),
                        "sem": HTMLExporter._format_metric(float(stats.sem(arr)) if len(arr) > 1 else None),
                        "min": HTMLExporter._format_metric(float(np.min(arr))),
                        "max": HTMLExporter._format_metric(float(np.max(arr))),
                    })
            return {
                "rows": rows,
                "has_transformed": False,
                "title": "Variable summary",
                "group_col_label": "Variable",
            }

        # --- CorrelationMatrix: variable N summary from n_matrix diagonal ---
        if model_type == "CorrelationMatrix":
            variables = results.get("variables") or []
            n_matrix = results.get("n_matrix") or {}
            rows = []
            for var in variables:
                n_val = (n_matrix.get(var) or {}).get(var)
                rows.append({
                    "group": HTMLExporter._prettify_label(var),
                    "n": HTMLExporter._format_metric(int(n_val) if n_val is not None else None),
                    "mean": "—",
                    "median": "—",
                    "sd": "—",
                    "min": "—",
                    "max": "—",
                })
            return {
                "rows": rows,
                "has_transformed": False,
                "title": "Variable overview",
                "group_col_label": "Variable",
                "note": "N = non-missing observations per variable. Descriptive statistics not stored — run per-variable analysis for full summary.",
            }

        # --- Logistic Regression: no meaningful group-level summary ---
        if model_type == "LogisticRegression":
            return {
                "rows": [],
                "has_transformed": False,
                "title": "Descriptive Statistics",
                "group_col_label": "Group",
            }

        # --- LMM: model structure summary ---
        if model_type == "LMM":
            fe_used = results.get("fixed_effects_used") or []
            covariates = results.get("covariates_used") or []
            desc_rows = [
                {
                    "label": "N subjects (random intercept grouping)",
                    "value": HTMLExporter._format_metric(results.get("n_subjects")),
                },
                {
                    "label": "N total observations",
                    "value": HTMLExporter._format_metric(results.get("n_observations")),
                },
                {
                    "label": "Fixed effects",
                    "value": ", ".join(str(f) for f in fe_used) if fe_used else "—",
                },
                {
                    "label": "Random intercept variable",
                    "value": str(results.get("random_intercept") or "—"),
                },
                {
                    "label": "Covariates",
                    "value": ", ".join(str(c) for c in covariates) if covariates else "None",
                },
            ]
            return {
                "rows": desc_rows,
                "has_transformed": False,
                "title": "Model Structure Summary",
                "group_col_label": "Parameter",
                "columns": ["Parameter", "Value"],
                "column_keys": ["label", "value"],
                "note": "Random intercept model — observations nested within subjects.",
            }

        # --- Standard group comparison tests ---
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
            "title": "Group-level summary",
            "group_col_label": "Group",
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
            html = HTMLExporter._figure_to_html(figure, div_id="biomedstatx-lmm-chart")
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
            print(f"WARNING HTML EXPORT: LMM chart generation failed: {exc}")
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

            html = HTMLExporter._figure_to_html(figure, div_id="biomedstatx-trajectory-chart")
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
            print(f"WARNING HTML EXPORT: trajectory chart generation failed: {exc}")
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
                        f"Interaction p = {HTMLExporter._format_p_value(ip)}"
                        + (" — significant" if ip < 0.05 else "")
                    )
            fig.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#fffdf8",
                margin=dict(l=56, r=20, t=36, b=64),
                font=dict(family="Segoe UI, Helvetica Neue, sans-serif", color="#16313a"),
                xaxis=dict(title=factor_x),
                yaxis=dict(title="Cell Mean"),
                legend=dict(title=dict(text=factor_line), orientation="h", x=0.01, y=1.1),
                annotations=[dict(
                    x=0.5, y=-0.2, xref="paper", yref="paper",
                    text=interaction_note, showarrow=False,
                    font=dict(size=11, color="#b7791f" if interaction_sig else "#555"),
                )] if interaction_note else [],
            )
            html = HTMLExporter._figure_to_html(fig, div_id="biomedstatx-interaction-plot")
            if not html:
                return None
            return {
                "title": "Interaction Plot — Cell Means ± SE",
                "subtitle": f"{factor_x} × {factor_line}",
                "html": html,
                "div_id": "biomedstatx-interaction-plot",
            }
        except Exception as exc:
            print(f"WARNING HTML EXPORT: interaction plot failed: {exc}")
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
                ),
                yaxis=dict(title="Observed values"),
                legend=dict(orientation="h", x=0.01, y=1.1),
            )
            n_subjects = len(trajectories)
            subtitle = f"Mean ± SE across {within_factor} levels"
            if n_subjects:
                subtitle += f" | n={n_subjects} subjects"
            html = HTMLExporter._figure_to_html(fig, div_id="biomedstatx-profile-plot")
            if not html:
                return None
            return {
                "title": "Profile Plot — Means ± SE across Timepoints",
                "subtitle": subtitle,
                "html": html,
                "div_id": "biomedstatx-profile-plot",
            }
        except Exception as exc:
            print(f"WARNING HTML EXPORT: profile plot failed: {exc}")
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
                xaxis=dict(title=factor_within),
                yaxis=dict(title="Group mean"),
                legend=dict(title=dict(text=factor_between), orientation="h", x=0.01, y=1.1),
            )
            html = HTMLExporter._figure_to_html(fig, div_id="biomedstatx-mixed-profile-plot")
            if not html:
                return None
            return {
                "title": "Profile Plot — Group Means ± SE",
                "subtitle": f"{factor_between} groups over {factor_within} levels",
                "html": html,
                "div_id": "biomedstatx-mixed-profile-plot",
            }
        except Exception as exc:
            print(f"WARNING HTML EXPORT: mixed profile plot failed: {exc}")
            return None

    @staticmethod
    def _build_single_chart_bundle(results: dict) -> list[dict]:
        charts = []
        lmm_chart = HTMLExporter._build_lmm_chart(results)
        if lmm_chart:
            charts.append({
                "title": lmm_chart["title"],
                "subtitle": lmm_chart["subtitle"],
                "html": lmm_chart["html"],
                "div_id": lmm_chart["div_id"],
            })
        ancova_chart = HTMLExporter._build_ancova_chart(results)
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
            or_block = HTMLExporter._build_or_table_html(results)
            if or_block:
                charts.append(or_block)
            # ROC curve replaces meaningless boxplot of binary outcome
            roc_block = HTMLExporter._build_roc_chart(results)
            if roc_block:
                charts.append(roc_block)
        elif model_type == "BetaRegression":
            # Coefficient table as inline HTML block
            beta_coef_block = HTMLExporter._build_beta_coefficient_table_html(results)
            if beta_coef_block:
                charts.append(beta_coef_block)
            # Scatter + fitted curve replaces meaningless boxplot for proportion outcome
            beta_chart = HTMLExporter._build_beta_regression_chart(results)
            if beta_chart:
                charts.append(beta_chart)
        elif model_type == "CorrelationMatrix":
            # Heatmaps replace meaningless boxplot — no group data, matrix data only
            charts.extend(HTMLExporter._build_correlation_matrix_charts(results))
        elif model_type in ("TwoWayANOVA", "MixedANOVA"):
            interactions = results.get("interactions") or []
            interaction_sig = any(
                isinstance(inter.get("p_value"), (int, float)) and inter["p_value"] < 0.05
                for inter in interactions
            )
            interaction_plot = HTMLExporter._build_interaction_plot(results)
            if model_type == "MixedANOVA":
                mixed_profile = HTMLExporter._build_mixed_profile_plot(results)
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
                group_chart = HTMLExporter._build_group_comparison_chart(results)
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
            profile_plot = HTMLExporter._build_profile_plot(results)
            if profile_plot:
                charts.append(profile_plot)
            # Boxplot as secondary context
            group_chart = HTMLExporter._build_group_comparison_chart(results)
            if group_chart:
                charts.append({
                    "title": "Group Comparison",
                    "subtitle": "Distribution overview with boxplots and individual observations.",
                    "html": group_chart["html"],
                    "group_order": group_chart["group_order"],
                    "div_id": "biomedstatx-group-chart",
                })
        else:
            group_chart = HTMLExporter._build_group_comparison_chart(results)
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
            trajectory_chart = HTMLExporter._build_trajectory_chart(results)
            if trajectory_chart:
                charts.append({
                    "title": trajectory_chart["title"],
                    "subtitle": trajectory_chart["subtitle"],
                    "html": trajectory_chart["html"],
                    "div_id": trajectory_chart["div_id"],
                })
        correlation_chart = HTMLExporter._build_correlation_chart(results)
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
                numeric = HTMLExporter._coerce_numeric_sequence(values)
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
    def _build_or_table_html(results: dict) -> dict | None:
        """Renders the Odds Ratios table as an inline HTML block for chart_blocks injection."""
        or_table = results.get("odds_ratios") or []
        if not or_table:
            return None
        rows_html = ""
        for row in or_table:
            p_val = row.get("p_value")
            is_sig = isinstance(p_val, (int, float)) and p_val < 0.05
            or_display = HTMLExporter._format_metric(row.get("odds_ratio"))
            if is_sig:
                or_display = f"<strong>{or_display}</strong>"
            p_style = "color:var(--success)" if is_sig else "color:var(--muted)"
            rows_html += (
                f"<tr>"
                f"<td>{row.get('parameter', '')}</td>"
                f"<td class='num-cell'>{or_display}</td>"
                f"<td class='num-cell'>{HTMLExporter._format_metric(row.get('ci_lower'))}</td>"
                f"<td class='num-cell'>{HTMLExporter._format_metric(row.get('ci_upper'))}</td>"
                f"<td class='num-cell'>{HTMLExporter._format_metric(row.get('z_value'))}</td>"
                f"<td class='num-cell' style='{p_style}'>{HTMLExporter._format_p_value(p_val)}</td>"
                f"</tr>"
            )
        html = (
            "<div class='table-shell'>"
            "<table>"
            "<thead><tr>"
            "<th>Parameter</th><th>OR</th><th>95% CI Lower</th><th>95% CI Upper</th><th>z</th><th>p-value</th>"
            "</tr></thead>"
            f"<tbody>{rows_html}</tbody>"
            "</table></div>"
        )
        return {
            "title": "Odds Ratios",
            "subtitle": "Exponentiated coefficients with 95% confidence intervals. Bold OR = p &lt; 0.05.",
            "html": html,
            "div_id": "biomedstatx-or-table",
        }

    @staticmethod
    def _build_beta_coefficient_table_html(results: dict) -> dict | None:
        """Renders the Beta Regression coefficient table as an inline HTML block."""
        coef_table = results.get("coefficients") or []
        if not coef_table:
            return None
        rows_html = ""
        for row in coef_table:
            p_val = row.get("p_value")
            is_sig = isinstance(p_val, (int, float)) and p_val < 0.05
            coef_display = HTMLExporter._format_metric(row.get("coefficient"))
            if is_sig:
                coef_display = f"<strong>{coef_display}</strong>"
            p_style = "color:var(--success)" if is_sig else "color:var(--muted)"
            rows_html += (
                f"<tr>"
                f"<td>{row.get('parameter', '')}</td>"
                f"<td class='num-cell'>{coef_display}</td>"
                f"<td class='num-cell'>{HTMLExporter._format_metric(row.get('std_err'))}</td>"
                f"<td class='num-cell'>{HTMLExporter._format_metric(row.get('z_value'))}</td>"
                f"<td class='num-cell' style='{p_style}'>{HTMLExporter._format_p_value(p_val)}</td>"
                f"<td class='num-cell'>{HTMLExporter._format_metric(row.get('ci_lower'))}</td>"
                f"<td class='num-cell'>{HTMLExporter._format_metric(row.get('ci_upper'))}</td>"
                f"</tr>"
            )
        html = (
            "<div class='table-shell'>"
            "<p style='font-size:0.78rem;color:var(--muted);margin:0 0 6px 0'>"
            "Coefficients on the logit scale. Bold = p &lt; 0.05.</p>"
            "<table>"
            "<thead><tr>"
            "<th>Parameter</th><th>Coefficient</th><th>SE</th><th>z</th>"
            "<th>p-value</th><th>95% CI Lower</th><th>95% CI Upper</th>"
            "</tr></thead>"
            f"<tbody>{rows_html}</tbody>"
            "</table></div>"
        )
        return {
            "title": "Coefficients (logit scale)",
            "subtitle": "Log-odds scale coefficients with standard errors and 95% confidence intervals.",
            "html": html,
            "div_id": "biomedstatx-beta-coef-table",
        }

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
            html = HTMLExporter._figure_to_html(figure, div_id="biomedstatx-roc-chart")
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
            print(f"WARNING HTML EXPORT: ROC chart generation failed: {exc}")
            return None

    @staticmethod
    def _build_beta_regression_chart(results: dict) -> dict | None:
        """Scatter plot of observed proportions vs primary predictor with fitted curve overlay."""
        if results.get("model_type") != "BetaRegression":
            return None
        fitted = results.get("fitted_values") or []
        xy_data = results.get("xy_data") or {}
        x_values = HTMLExporter._coerce_numeric_sequence(xy_data.get("x"))
        y_values = HTMLExporter._coerce_numeric_sequence(xy_data.get("y"))
        if not x_values or not y_values or len(x_values) != len(fitted):
            return None
        try:
            import plotly.graph_objects as go

            x_arr = np.array(x_values, dtype=float)
            y_arr = np.array(y_values, dtype=float)
            fitted_arr = np.array(fitted, dtype=float)
            sort_idx = np.argsort(x_arr)
            x_label = HTMLExporter._prettify_label(xy_data.get("x_label") or "Predictor")

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
            html = HTMLExporter._figure_to_html(figure, div_id="biomedstatx-beta-chart")
            if not html:
                return None
            return {
                "title": "Beta Regression: Observed vs. Fitted",
                "subtitle": "Proportion outcome (y-axis fixed [0, 1]). Orange line = model-fitted values.",
                "html": html,
                "div_id": "biomedstatx-beta-chart",
            }
        except Exception as exc:
            print(f"WARNING HTML EXPORT: Beta regression chart failed: {exc}")
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

            var_labels = [HTMLExporter._prettify_label(v) for v in variables]
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
            html_r = HTMLExporter._figure_to_html(fig_r, div_id=div_r)

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
            html_p = HTMLExporter._figure_to_html(fig_p, div_id=div_p)

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
            print(f"WARNING HTML EXPORT: CorrelationMatrix overall heatmap failed: {exc}")

        # Stratified matrices
        strata = results.get("strata") or {}
        for stratum_name, stratum_data in strata.items():
            try:
                r_s = stratum_data.get("r_matrix") or {}
                pc_s = stratum_data.get("p_corrected_matrix") or {}
                charts.extend(_make_heatmap_pair(r_s, pc_s, str(stratum_name)))
            except Exception as exc:
                print(f"WARNING HTML EXPORT: CorrelationMatrix stratum '{stratum_name}' heatmap failed: {exc}")

        return charts

    @staticmethod
    def _build_correlation_chart(results: dict) -> dict | None:
        model_type = str(results.get("model_type") or "")
        if model_type not in {"Correlation", "LinearRegression"}:
            return None
        try:
            import plotly.graph_objects as go

            payload = HTMLExporter._extract_association_payload(results)
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
            html = HTMLExporter._figure_to_html(figure, div_id="biomedstatx-assoc-chart")
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
            print(f"WARNING HTML EXPORT: association chart generation failed: {exc}")
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
            html = HTMLExporter._figure_to_html(figure, div_id="biomedstatx-ancova-chart")
            if not html:
                return None
            return {
                "title": "Adjusted Group Means",
                "subtitle": f"Estimated marginal means ± SD. Adjusted for: {cov_str}.",
                "html": html,
                "div_id": "biomedstatx-ancova-chart",
            }
        except Exception as exc:
            print(f"WARNING HTML EXPORT: ANCOVA chart generation failed: {exc}")
            return None

    @staticmethod
    def _extract_association_payload(results: dict) -> dict | None:
        def _pair_points(points):
            x_out = []
            y_out = []
            if not isinstance(points, list):
                return x_out, y_out
            for point in points:
                if not isinstance(point, dict):
                    continue
                try:
                    x_val = float(point.get("x"))
                    y_val = float(point.get("y"))
                    if not (np.isfinite(x_val) and np.isfinite(y_val)):
                        continue
                except Exception:
                    continue
                x_out.append(x_val)
                y_out.append(y_val)
            return x_out, y_out

        x_label = str(results.get("x_variable_display") or results.get("x_variable") or "X")
        y_label = str(results.get("y_variable_display") or results.get("y_variable") or "Y")

        x_values = []
        y_values = []
        fit_x = []
        fit_y = []
        fit_ci_lower = []
        fit_ci_upper = []

        regression_payload = results.get("plot_regression")
        if isinstance(regression_payload, dict):
            x_values, y_values = _pair_points(regression_payload.get("points"))
            x_label = str(regression_payload.get("x_label") or x_label)
            y_label = str(regression_payload.get("y_label") or y_label)
            fit = regression_payload.get("fit") if isinstance(regression_payload.get("fit"), dict) else {}
            fit_x = HTMLExporter._coerce_numeric_sequence(fit.get("x") or [])
            fit_y = HTMLExporter._coerce_numeric_sequence(fit.get("y") or [])
            fit_ci_lower = HTMLExporter._coerce_numeric_sequence(fit.get("ci_lower") or [])
            fit_ci_upper = HTMLExporter._coerce_numeric_sequence(fit.get("ci_upper") or [])

        if not x_values or not y_values:
            association_points = results.get("association_points")
            x_values, y_values = _pair_points(association_points)

        if not x_values or not y_values:
            raw_data = results.get("raw_data") or {}
            if isinstance(raw_data, dict) and len(raw_data) >= 2:
                names = list(raw_data.keys())[:2]
                x_candidate = HTMLExporter._coerce_numeric_sequence(raw_data.get(names[0], []))
                y_candidate = HTMLExporter._coerce_numeric_sequence(raw_data.get(names[1], []))
                paired_length = min(len(x_candidate), len(y_candidate))
                x_values = x_candidate[:paired_length]
                y_values = y_candidate[:paired_length]
                x_label = str(names[0])
                y_label = str(names[1])

        if not x_values or not y_values:
            return None

        method = str(results.get("method") or "").lower()
        if not (fit_x and fit_y and len(fit_x) == len(fit_y)) and method != "spearman":
            fit_data = HTMLExporter._simple_linear_fit_with_ci(x_values, y_values, alpha=float(results.get("alpha", 0.05)))
            if fit_data is not None:
                fit_x = fit_data["x"]
                fit_y = fit_data["y"]
                fit_ci_lower = fit_data["ci_lower"]
                fit_ci_upper = fit_data["ci_upper"]

        return {
            "x_label": x_label,
            "y_label": y_label,
            "x_values": x_values,
            "y_values": y_values,
            "fit_x": fit_x,
            "fit_y": fit_y,
            "fit_ci_lower": fit_ci_lower,
            "fit_ci_upper": fit_ci_upper,
        }

    @staticmethod
    def _simple_linear_fit_with_ci(x_values: list[float], y_values: list[float], alpha: float = 0.05) -> dict | None:
        try:
            x = np.asarray(x_values, dtype=float)
            y = np.asarray(y_values, dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            x = x[valid]
            y = y[valid]
            if x.size < 3 or y.size < 3:
                return None

            x_min = float(np.min(x))
            x_max = float(np.max(x))
            if np.isclose(x_min, x_max):
                return None

            slope, intercept, _, _, _ = stats.linregress(x, y)
            x_grid = np.linspace(x_min, x_max, 180)
            y_fit = intercept + slope * x_grid

            n = x.size
            dof = n - 2
            if dof <= 0:
                return {
                    "x": [float(v) for v in x_grid.tolist()],
                    "y": [float(v) for v in y_fit.tolist()],
                    "ci_lower": [],
                    "ci_upper": [],
                }

            residuals = y - (intercept + slope * x)
            s_err = np.sqrt(np.sum(residuals ** 2) / dof)
            x_mean = float(np.mean(x))
            ss_x = float(np.sum((x - x_mean) ** 2))
            if ss_x <= 0:
                return {
                    "x": [float(v) for v in x_grid.tolist()],
                    "y": [float(v) for v in y_fit.tolist()],
                    "ci_lower": [],
                    "ci_upper": [],
                }

            t_critical = stats.t.ppf(1 - alpha / 2, dof)
            se_fit = s_err * np.sqrt((1 / n) + ((x_grid - x_mean) ** 2 / ss_x))
            ci_delta = t_critical * se_fit
            ci_lower = y_fit - ci_delta
            ci_upper = y_fit + ci_delta

            return {
                "x": [float(v) for v in x_grid.tolist()],
                "y": [float(v) for v in y_fit.tolist()],
                "ci_lower": [float(v) for v in ci_lower.tolist()],
                "ci_upper": [float(v) for v in ci_upper.tolist()],
            }
        except Exception:
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
            return HTMLExporter._figure_to_html(figure, div_id="biomedstatx-qq-chart")
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
    def _embed_decision_tree(results: dict, pre_generated_path: str | None = None) -> str | None:
        # If caller already generated the tree, encode it directly without re-generating or deleting.
        if pre_generated_path and os.path.exists(pre_generated_path):
            try:
                with open(pre_generated_path, "rb") as handle:
                    encoded = base64.b64encode(handle.read()).decode("ascii")
                return f"data:image/png;base64,{encoded}"
            except Exception as exc:
                print(f"WARNING HTML EXPORT: decision tree embedding (pre-generated) failed: {exc}")
                return None

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

        # For Correlation: document any automatic shift applied before transformation.
        # Reviewers need c to reproduce the transformed values (y = f(x + c)).
        if results.get("model_type") == "Correlation":
            x_shift = results.get("x_transform_shift") or 0.0
            y_shift = results.get("y_transform_shift") or 0.0
            x_tr = results.get("x_transform") or "none"
            y_tr = results.get("y_transform") or "none"
            for axis, tr_name, shift in [("X", x_tr, x_shift), ("Y", y_tr, y_shift)]:
                if tr_name != "none" and shift != 0.0:
                    # Reconstruct the minimum raw value that triggered the shift.
                    # log10/boxcox: shift = -min_val + 1.0  → min_val = 1.0 - shift
                    # sqrt:         shift = -min_val         → min_val = -shift
                    min_raw = (1.0 - shift) if tr_name in ('log10', 'boxcox') else (-shift)
                    lines.append(
                        f"Note ({axis}-axis): a constant c={shift:.4f} was automatically added "
                        f"to all {axis.lower()}-values prior to {tr_name} transformation to satisfy "
                        f"the positivity requirement (minimum raw value was {min_raw:.4f}). "
                        f"This constant was determined from the data, not set by the researcher."
                    )

        return "\n".join(lines)

    @staticmethod
    def _render_template(context: dict, mode: str) -> str:
        assumptions = context.get("assumptions", {}) or {}
        plotly_enabled = any([
            bool(context.get("chart_blocks")),
            bool(context.get("plot_designer_enabled")),
            bool(assumptions.get("qq_plot_html")),
            bool(assumptions.get("distribution_plot_html")),
            bool(assumptions.get("residual_plot_html")),
        ])
        plotly_bundle = HTMLExporter._plotly_bundle() if plotly_enabled else ""
        math_render_enabled = bool(context.get("math_render_enabled"))
        if math_render_enabled:
            math_bundle, math_status = HTMLExporter._math_bundle(preferred="katex")
        else:
            math_bundle, math_status = "", "disabled-no-latex"
        env = Environment(
            loader=FileSystemLoader(str(HTMLExporter._templates_dir())),
            autoescape=True,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        template_name = HTMLExporter._template_name(mode)
        try:
            template = env.get_template(template_name)
            return template.render(
                context=context,
                mode=mode,
                plotly_bundle=plotly_bundle,
                mathjax_bundle=math_bundle,
                mathjax_status=math_status,
            )
        except TemplateNotFound:
            template = env.from_string(HTMLExporter._template())
            return template.render(
                context=context,
                mode=mode,
                plotly_bundle=plotly_bundle,
                mathjax_bundle=math_bundle,
                mathjax_status=math_status,
            )

    @staticmethod
    def _has_latex_syntax(value: str) -> bool:
        if not isinstance(value, str):
            return False
        if HTMLExporter._BEGIN_ENV_RE.search(value):
            return True
        return bool(HTMLExporter._INLINE_LATEX_RE.search(value))

    @staticmethod
    def _requires_math_rendering(results: dict, hero: dict | None = None) -> bool:
        strings_to_scan = []

        if isinstance(results, dict):
            for key in [
                "title", "subtitle", "dataset_name", "column_name", "dependent_variable",
                "unit", "units", "x_label", "xlabel", "y_label", "ylabel",
            ]:
                value = results.get(key)
                if isinstance(value, str):
                    strings_to_scan.append(value)

        if isinstance(hero, dict):
            for key in ["title", "subtitle", "test_name"]:
                value = hero.get(key)
                if isinstance(value, str):
                    strings_to_scan.append(value)

        return any(HTMLExporter._has_latex_syntax(text) for text in strings_to_scan)

    @staticmethod
    def _math_bundle(preferred: str = "katex") -> tuple[str, str]:
        order = [preferred, "mathjax"] if preferred != "mathjax" else ["mathjax"]
        for engine in order:
            if engine == "katex":
                bundle, status = HTMLExporter._katex_bundle()
            else:
                bundle, status = HTMLExporter._mathjax_bundle()
            if status.startswith("loaded"):
                return bundle, status

        missing_bundle = (
            "<script>"
            "window.BioMedStatXMath={enabled:false,status:'missing-local-runtime'};"
            "window.BioMedStatXTypesetMath=function(){return Promise.resolve();};"
            "</script>"
        )
        return missing_bundle, "missing-local-runtime"

    @staticmethod
    def _katex_runtime_candidates() -> list[Path]:
        templates_dir = HTMLExporter._templates_dir()
        source_root = Path(__file__).resolve().parent
        candidates = [
            templates_dir / "vendor" / "katex",
            source_root / "templates" / "vendor" / "katex",
        ]
        frozen_root = getattr(sys, "_MEIPASS", None)
        if frozen_root:
            candidates.append(Path(frozen_root) / "templates" / "vendor" / "katex")
        unique_candidates = []
        seen = set()
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            unique_candidates.append(candidate)
        return unique_candidates

    @staticmethod
    def _katex_bundle() -> tuple[str, str]:
        for candidate_root in HTMLExporter._katex_runtime_candidates():
            try:
                css_path = candidate_root / "katex.min.css"
                js_path = candidate_root / "katex.min.js"
                autorender_path = candidate_root / "auto-render.min.js"
                if not (css_path.exists() and js_path.exists() and autorender_path.exists()):
                    continue

                css_text = css_path.read_text(encoding="utf-8")
                css_text = HTMLExporter._inline_local_css_assets(css_text, candidate_root)
                js_text = js_path.read_text(encoding="utf-8")
                autorender_text = autorender_path.read_text(encoding="utf-8")

                bootstrap = (
                    "<style>" + css_text + "</style>"
                    "<script>"
                    "window.BioMedStatXMath={enabled:false,status:'loaded',engine:'katex'};"
                    "window.BioMedStatXTypesetMath=function(root){"
                    "if(typeof renderMathInElement==='function'){"
                    "renderMathInElement(root||document.body,{"
                    "delimiters:["
                    "{left:'$$',right:'$$',display:true},"
                    "{left:'$',right:'$',display:false},"
                    "{left:'\\\\(',right:'\\\\)',display:false},"
                    "{left:'\\\\[',right:'\\\\]',display:true}"
                    "]"
                    "});"
                    "}"
                    "return Promise.resolve();"
                    "};"
                    "</script>"
                )
                runtime = f"<script>{js_text}</script><script>{autorender_text}</script>"
                finalize = (
                    "<script>"
                    "window.BioMedStatXMath={enabled:true,status:'loaded',engine:'katex'};"
                    "document.addEventListener('DOMContentLoaded',function(){"
                    "window.BioMedStatXTypesetMath(document.body);"
                    "});"
                    "</script>"
                )
                return bootstrap + runtime + finalize, "loaded-katex"
            except Exception as exc:
                print(f"WARNING HTML EXPORT: failed to load KaTeX runtime '{candidate_root}': {exc}")
        return "", "missing-katex-runtime"

    @staticmethod
    def _inline_local_css_assets(css_text: str, assets_root: Path) -> str:
        if not isinstance(css_text, str) or not isinstance(assets_root, Path):
            return css_text

        def replace_url(match: re.Match) -> str:
            raw_path = (match.group("path") or "").strip()
            if not raw_path:
                return match.group(0)
            if raw_path.startswith(("data:", "http:", "https:", "//", "#")):
                return match.group(0)

            cleaned_path = raw_path.split("?", 1)[0].split("#", 1)[0]
            candidate = (assets_root / cleaned_path).resolve()
            if not candidate.exists() or not candidate.is_file():
                return match.group(0)

            mime_type = HTMLExporter._guess_embedded_asset_mime(candidate.suffix.lower())
            if mime_type is None:
                return match.group(0)

            try:
                encoded = base64.b64encode(candidate.read_bytes()).decode("ascii")
            except Exception:
                return match.group(0)

            return f"url('data:{mime_type};base64,{encoded}')"

        return HTMLExporter._CSS_URL_RE.sub(replace_url, css_text)

    @staticmethod
    def _guess_embedded_asset_mime(extension: str) -> str | None:
        mapping = {
            ".woff2": "font/woff2",
            ".woff": "font/woff",
            ".ttf": "font/ttf",
            ".otf": "font/otf",
            ".eot": "application/vnd.ms-fontobject",
            ".svg": "image/svg+xml",
        }
        return mapping.get(extension)

    @staticmethod
    def _templates_dir() -> Path:
        module_templates = Path(__file__).resolve().parent / "templates"
        if module_templates.exists():
            return module_templates
        frozen_root = getattr(sys, "_MEIPASS", None)
        if frozen_root:
            frozen_templates = Path(frozen_root) / "templates"
            if frozen_templates.exists():
                return frozen_templates
        return module_templates

    @staticmethod
    def _template_name(mode: str) -> str:
        return "report_multi.html.j2" if mode == "multi" else "report_single.html.j2"

    @staticmethod
    def _read_template(template_name: str) -> str:
        return (HTMLExporter._templates_dir() / template_name).read_text(encoding="utf-8")

    @staticmethod
    def _plotly_bundle() -> str:
        try:
            from plotly.offline.offline import get_plotlyjs

            return f"<script>{get_plotlyjs()}</script>"
        except Exception as exc:
            print(f"WARNING HTML EXPORT: plotly bundle unavailable: {exc}")
            return ""

    @staticmethod
    def _mathjax_runtime_candidates() -> list[Path]:
        templates_dir = HTMLExporter._templates_dir()
        source_root = Path(__file__).resolve().parent
        candidates = [
            templates_dir / "vendor" / "mathjax" / "tex-svg.js",
            templates_dir / "vendor" / "mathjax" / "tex-mml-chtml.js",
            source_root / "templates" / "vendor" / "mathjax" / "tex-svg.js",
            source_root / "templates" / "vendor" / "mathjax" / "tex-mml-chtml.js",
        ]
        frozen_root = getattr(sys, "_MEIPASS", None)
        if frozen_root:
            frozen_path = Path(frozen_root)
            candidates.extend([
                frozen_path / "templates" / "vendor" / "mathjax" / "tex-svg.js",
                frozen_path / "templates" / "vendor" / "mathjax" / "tex-mml-chtml.js",
            ])
        unique_candidates = []
        seen = set()
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            unique_candidates.append(candidate)
        return unique_candidates

    @staticmethod
    def _mathjax_bundle() -> tuple[str, str]:
        for candidate in HTMLExporter._mathjax_runtime_candidates():
            try:
                if not candidate.exists():
                    continue
                runtime = candidate.read_text(encoding="utf-8")
                bootstrap = (
                    "<script>"
                    "window.MathJax={"
                    "tex:{inlineMath:[['$','$'],['\\\\(','\\\\)']],displayMath:[['$$','$$'],['\\\\[','\\\\]']]},"
                    "svg:{fontCache:'none'},"
                    "options:{skipHtmlTags:['script','noscript','style','textarea','pre','code']}"
                    "};"
                    "window.BioMedStatXMath={enabled:false,status:'loaded'};"
                    "window.BioMedStatXTypesetMath=function(root){"
                    "if(window.MathJax&&window.MathJax.typesetPromise){"
                    "return window.MathJax.typesetPromise(root?[root]:undefined);"
                    "}"
                    "return Promise.resolve();"
                    "};"
                    "</script>"
                )
                runtime_script = f"<script>{runtime}</script>"
                finalize = (
                    "<script>"
                    "window.BioMedStatXMath={enabled:true,status:'loaded',engine:'mathjax'};"
                    "document.addEventListener('DOMContentLoaded',function(){"
                    "window.BioMedStatXTypesetMath(document.body);"
                    "});"
                    "</script>"
                )
                return bootstrap + runtime_script + finalize, "loaded-mathjax"
            except Exception as exc:
                print(f"WARNING HTML EXPORT: failed to load MathJax runtime '{candidate}': {exc}")
        return "", "missing-mathjax-runtime"

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
        try:
            return HTMLExporter._read_template("report_single.html.j2")
        except Exception:
            pass
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
.info-btn{display:inline-flex;align-items:center;justify-content:center;width:18px;height:18px;margin-left:8px;border-radius:50%;border:1.5px solid var(--muted);background:transparent;color:var(--muted);font-size:.75rem;font-family:"Segoe UI","Helvetica Neue",Arial,sans-serif;font-style:italic;font-weight:600;line-height:1;cursor:pointer;vertical-align:middle;flex-shrink:0;transition:border-color .18s,color .18s,background .18s}
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
<section id="sec-decision" class="section" data-reveal><div class="section-head"><div><div class="section-kicker">Decision Path</div><h2 id="hd-decision">How BioMedStatX reached this decision<button type="button" class="info-btn" aria-label="About this section" aria-expanded="false" aria-controls="info-decision">i</button></h2><div class="info-panel" id="info-decision" role="region" aria-labelledby="hd-decision"><div class="info-panel-inner">{{ context.info_texts.decision }}</div></div></div></div><div id="decision-path" style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px">{% for step in context.decision_path %}<div class="decision-step{% if step.active %} is-active{% endif %}" data-step="{{ loop.index0 }}" style="display:flex;align-items:center;gap:6px;padding:.35rem .8rem;border-radius:999px;border:1px solid var(--line);background:var(--surface-2);font-size:.82rem"><span style="width:8px;height:8px;border-radius:50%;flex-shrink:0;background:{% if step.active %}var(--accent){% else %}var(--line){% endif %}"></span><span>{{ step.title }}</span></div>{% endfor %}</div><div class="decision-tree-frame{% if context.decision_tree_json == 'null' %} is-empty{% endif %}">{% if context.decision_tree_json != "null" %}<div id="tree-toolbar"><button class="tree-ctrl" id="tree-zoom-in">＋</button><button class="tree-ctrl" id="tree-zoom-out">－</button><span class="tree-zoom-label" id="tree-zoom-pct">100%</span><button class="tree-ctrl" id="tree-reset">Reset</button><span style="flex:1"></span><button class="tree-ctrl" id="tree-replay" title="Replay path animation">&#9654; Replay</button><span class="small muted" style="margin-left:8px">Scroll to zoom · Drag to pan</span></div><div id="tree-viewport"><div id="tree-canvas"><div id="dyn-tree-host"></div></div><div id="tree-tooltip"></div></div>{% else %}<div class="empty-state">Decision tree was not available for this analysis.</div>{% endif %}</div></section>
<section id="sec-results" class="section" data-reveal><div class="section-head"><div><div class="section-kicker">Statistical Engine</div><h2 id="hd-results">Main results<button type="button" class="info-btn" aria-label="About this section" aria-expanded="false" aria-controls="info-results">i</button></h2><div class="info-panel" id="info-results" role="region" aria-labelledby="hd-results"><div class="info-panel-inner">{{ context.info_texts.results }}</div></div></div></div><div class="table-shell"><table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>{% for row in context.statistical_rows %}<tr><td>{{ row.label }}</td><td class="num-cell">{{ row.value }}</td></tr>{% endfor %}</tbody></table></div></section>
<section id="sec-assumptions" class="section" data-reveal><div class="section-head"><div><div class="section-kicker">Assumptions</div><h2 id="hd-assumptions">Model validity checks<button type="button" class="info-btn" aria-label="About this section" aria-expanded="false" aria-controls="info-assumptions">i</button></h2><div class="info-panel" id="info-assumptions" role="region" aria-labelledby="hd-assumptions"><div class="info-panel-inner">{{ context.info_texts.assumptions }}</div></div><p class="muted">{{ context.assumptions.interpretation }}</p>{% if context.assumptions.sphericity_correction_note %}<p class="muted" style="margin-top:.35rem;font-size:.88rem;color:var(--warning)">&#9888; {{ context.assumptions.sphericity_correction_note }}</p>{% endif %}</div><div class="badge is-info">Transformation: {{ context.assumptions.transformation }}</div></div>{% if context.assumptions.rows %}<div class="table-shell"><table><thead><tr><th>Check</th><th>Statistic</th><th>p-value</th><th>Status</th></tr></thead><tbody>{% for row in context.assumptions.rows %}<tr class="{{ row.status_class }}"><td>{{ row.name }}</td><td class="num-cell">{{ row.statistic }}</td><td class="num-cell" style="{{ row.p_value_style }}">{{ row.p_value }}</td><td>{{ row.status_label }}</td></tr>{% endfor %}</tbody></table></div>{% else %}<div class="empty-state">No structured assumption summary was available for this result.</div>{% endif %}<div class="assumption-visual-grid">{% if context.assumptions.qq_plot_html %}<div class="chart-card"><div class="section-kicker">Q-Q Diagnostic</div><h3>Observed quantiles against the normal reference line</h3>{{ context.assumptions.qq_plot_html | safe }}<div class="toolbar" style="margin-top:12px"><button type="button" onclick="downloadPlotlyChart('biomedstatx-qq-chart','svg','biomedstatx_qq_plot')">Download SVG</button><button type="button" onclick="downloadPlotlyChart('biomedstatx-qq-chart','png','biomedstatx_qq_plot')">Download PNG</button></div></div>{% endif %}{% if context.assumptions.distribution_plot_html %}<div class="chart-card"><div class="section-kicker">Group Distribution View</div><h3>Boxplots with jittered observations</h3>{{ context.assumptions.distribution_plot_html | safe }}</div>{% endif %}{% if context.assumptions.residual_plot_html %}<div class="chart-card"><div class="section-kicker">Residual Structure</div><h3>Residuals versus fitted values</h3>{{ context.assumptions.residual_plot_html | safe }}</div>{% endif %}{% if not context.assumptions.qq_plot_html and not context.assumptions.distribution_plot_html and not context.assumptions.residual_plot_html %}<div class="empty-state">Visual assumption diagnostics were not available for this result structure.</div>{% endif %}</div></section>
<section id="sec-descriptive" class="section" data-reveal><div class="section-head"><div><div class="section-kicker">Descriptive Statistics</div><h2 id="hd-descriptive">{{ context.descriptive.title or "Group-level summary" }}<button type="button" class="info-btn" aria-label="About this section" aria-expanded="false" aria-controls="info-descriptive">i</button></h2><div class="info-panel" id="info-descriptive" role="region" aria-labelledby="hd-descriptive"><div class="info-panel-inner">{{ context.info_texts.descriptive }}</div></div></div></div>{% if context.descriptive.rows %}<div class="table-shell"><table><thead><tr><th>{{ context.descriptive.group_col_label or "Group" }}</th><th>n</th><th>Mean</th><th>Median</th><th>SD</th><th>SEM</th><th>Min</th><th>Max</th></tr></thead><tbody>{% for row in context.descriptive.rows %}<tr><td>{{ row.group }}</td><td class="num-cell">{{ row.n }}</td><td class="num-cell">{{ row.mean }}</td><td class="num-cell">{{ row.median }}</td><td class="num-cell">{{ row.sd }}</td><td class="num-cell">{{ row.sem }}</td><td class="num-cell">{{ row.min }}</td><td class="num-cell">{{ row.max }}</td></tr>{% endfor %}</tbody></table></div>{% else %}<div class="empty-state">No descriptive summary could be derived from the available result payload.</div>{% endif %}</section>
{% if context.pairwise_rows %}<section id="sec-pairwise" class="section" data-reveal><div class="section-head"><div><div class="section-kicker">Pairwise Comparisons</div><h2 id="hd-pairwise">Post-hoc findings<button type="button" class="info-btn" aria-label="About this section" aria-expanded="false" aria-controls="info-pairwise">i</button></h2><div class="info-panel" id="info-pairwise" role="region" aria-labelledby="hd-pairwise"><div class="info-panel-inner">{{ context.info_texts.pairwise }}</div></div></div></div><div class="table-shell"><table><thead><tr>{% if context.group_chart_div_id %}<th title="Show bracket in chart">Chart</th>{% endif %}<th>Comparison</th><th>Procedure</th><th>Statistic</th><th>p-value</th><th>Effect size</th><th>Interpretation</th></tr></thead><tbody>{% for row in context.pairwise_rows %}<tr class="{{ row.row_class }}">{% if context.group_chart_div_id %}<td style="text-align:center"><input type="checkbox" class="bracket-toggle" data-pair-id="{{ row.pair_id }}" {% if row.significant and row.stars %}checked{% endif %} aria-label="Show bracket for {{ row.comparison }}"></td>{% endif %}<td>{{ row.comparison }}</td><td>{{ row.test }}</td><td class="num-cell">{{ row.statistic }}</td><td class="num-cell" style="{{ row.p_value_style }}">{{ row.p_value }}</td><td class="num-cell">{{ row.effect_size }}</td><td>{{ "Significant" if row.significant else "Not significant" }}</td></tr>{% endfor %}</tbody></table></div></section>{% endif %}
<section id="sec-charts" class="section" data-reveal><div class="section-head"><div><div class="section-kicker">Interactive Charts</div><h2 id="hd-charts">Visual evidence<button type="button" class="info-btn" aria-label="About this section" aria-expanded="false" aria-controls="info-charts">i</button></h2><div class="info-panel" id="info-charts" role="region" aria-labelledby="hd-charts"><div class="info-panel-inner">{{ context.info_texts.charts }}</div></div></div></div>{% if context.chart_blocks %}{% for chart in context.chart_blocks %}<div class="chart-card"><div class="section-kicker">{{ chart.title }}</div><h3>{{ chart.subtitle }}</h3>{{ chart.html | safe }}{% if chart.div_id %}<div class="toolbar" style="margin-top:12px"><button type="button" onclick="downloadPlotlyChart('{{ chart.div_id }}','svg','biomedstatx_{{ chart.title | lower | replace(' ','_') }}')">Download SVG</button><button type="button" onclick="downloadPlotlyChart('{{ chart.div_id }}','png','biomedstatx_{{ chart.title | lower | replace(' ','_') }}')">Download PNG</button></div>{% endif %}</div>{% endfor %}{% else %}<div class="empty-state">No interactive chart could be created from the current result structure.</div>{% endif %}</section>
<section id="sec-raw" class="section" data-reveal><div class="section-head"><div><div class="section-kicker">Raw Data Vault</div><h2 id="hd-raw">Searchable raw values<button type="button" class="info-btn" aria-label="About this section" aria-expanded="false" aria-controls="info-raw">i</button></h2><div class="info-panel" id="info-raw" role="region" aria-labelledby="hd-raw"><div class="info-panel-inner">{{ context.info_texts.raw }}</div></div></div></div><div class="toolbar"><input id="raw-search" type="search" placeholder="Filter raw data"><button type="button" onclick="copyTable('raw-data-table')">Copy table</button><button type="button" onclick="downloadTableCSV('raw-data-table','biomedstatx_raw_data.csv')">Download CSV</button></div>{% if context.raw_data_table.rows %}<div class="table-shell"><table id="raw-data-table"><thead><tr><th>Group</th><th>Index</th><th>Raw value</th>{% if context.raw_data_table.has_transformed %}<th>Transformed value</th>{% endif %}</tr></thead><tbody>{% for row in context.raw_data_table.rows %}<tr><td>{{ row.group }}</td><td>{{ row.index }}</td><td>{{ row.raw_value }}</td>{% if context.raw_data_table.has_transformed %}<td>{{ row.transformed_value }}</td>{% endif %}</tr>{% endfor %}</tbody></table></div>{% else %}<div class="empty-state">No raw data were embedded in this result structure.</div>{% endif %}</section>
<section id="sec-methods" class="section" data-reveal><div class="section-head"><div><div class="section-kicker">Methods Snippet</div><h2 id="hd-methods">Reusable narrative text<button type="button" class="info-btn" aria-label="About this section" aria-expanded="false" aria-controls="info-methods">i</button></h2><div class="info-panel" id="info-methods" role="region" aria-labelledby="hd-methods"><div class="info-panel-inner">{{ context.info_texts.methods }}</div></div></div></div><div class="toolbar"><button type="button" onclick="copyText('methods-text')">Copy methods text</button></div><div id="methods-text" class="methods">{{ context.methods_text }}</div></section>
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
const NW=110,NH=44,PAD=80,SCALE=42,Y_SCALE=62;
const toSvg=(x,y)=>[(x-minX)*SCALE+PAD,(maxY-y)*Y_SCALE+PAD];
const svgW=(maxX-minX)*SCALE+PAD*2,svgH=(maxY-minY)*Y_SCALE+PAD*2;
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
function downloadPlotlyChart(divId,fmt,fileName){const el=document.getElementById(divId);if(!el||!window.Plotly){alert('Chart not available for download.');return;}Plotly.downloadImage(el,{format:fmt,filename:fileName||'biomedstatx_chart',width:1200,height:900,scale:3});}
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
    def _downsample_for_display(values: list[float], max_points: int = 5000) -> list[float]:
        if len(values) <= max_points:
            return values
        rng = random.Random(42)
        selected_indices = sorted(rng.sample(range(len(values)), max_points))
        return [values[index] for index in selected_indices]

    @staticmethod
    def _summarize_numeric_group(values: list[float]) -> dict:
        n = len(values)
        if n == 0:
            return {}
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1)) if n > 1 else 0.0
        sem = float(sd / math.sqrt(n)) if n > 0 else 0.0
        ci_half_width = float(stats.t.ppf(0.975, n - 1) * sem) if n > 1 else 0.0
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        q1 = float(q1)
        median = float(median)
        q3 = float(q3)
        iqr = float(q3 - q1)

        min_value = float(min(values))
        max_value = float(max(values))
        if iqr > 0:
            lower_limit = q1 - 1.5 * iqr
            upper_limit = q3 + 1.5 * iqr
            sorted_values = sorted(values)
            lower_candidates = [v for v in sorted_values if v >= lower_limit]
            upper_candidates = [v for v in sorted_values if v <= upper_limit]
            lower_fence = float(lower_candidates[0]) if lower_candidates else min_value
            upper_fence = float(upper_candidates[-1]) if upper_candidates else max_value
        else:
            lower_fence = min_value
            upper_fence = max_value

        return {
            "n": n,
            "mean": mean,
            "sd": sd,
            "sem": sem,
            "ci95_lower": float(mean - ci_half_width),
            "ci95_upper": float(mean + ci_half_width),
            "min": min_value,
            "max": max_value,
            "q1": q1,
            "median": median,
            "q3": q3,
            "iqr": iqr,
            "lower_fence": lower_fence,
            "upper_fence": upper_fence,
        }

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
                except Exception:
                    pass
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
