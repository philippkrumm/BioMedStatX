import copy
import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from export.report_formatting import _FormattingMixin
from export.report_assets import _AssetsMixin
from export.report_stat_rows import _StatRowsMixin
from export.report_association import _AssociationMixin
from export.report_charts import _ChartsMixin
from export.report_summaries import _SummariesMixin

try:
    from core.logger_config import get_logger
except ImportError:  # pragma: no cover — fallback when logger_config missing
    import logging as _logging
    def get_logger(name):
        return _logging.getLogger(name)

logger = get_logger(__name__)


def _effect_is_ratio(effect_type) -> bool:
    """True for ratio-scale effect measures (centered at 1, log-friendly):
    odds/risk/hazard/rate ratio, linear fold change. False for differences —
    including any *log* scale (log2 fold change, log-odds), which is centered
    at 0 and may be negative, so it must not get a reference line at 1 or a
    log axis.
    """
    t = str(effect_type or "").lower()
    if "log" in t:
        return False
    return "ratio" in t or "odds" in t or "fold" in t


class _ResultsEncoder(json.JSONEncoder):
    def default(self, obj):
        return HTMLExporter._normalize_for_json(obj)


class HTMLExporter(_FormattingMixin, _AssetsMixin, _StatRowsMixin, _AssociationMixin, _ChartsMixin, _SummariesMixin):

    @staticmethod
    def export_results_to_html(results: dict, output_file: str, analysis_log=None) -> str | None:
        try:
            output_path = Path(output_file).resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            context = HTMLExporter._prepare_single_report_context(
                results, analysis_log=analysis_log
            )
            html = HTMLExporter._render_template(context, mode="single")
            with open(output_path, "w", encoding="utf-8") as handle:
                handle.write(html)
            return str(output_path)
        except Exception as exc:
            logger.error("failed to write single report to %r: %s", output_file, exc, exc_info=True)
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
            logger.error("failed to write multi report to %r: %s", output_file, exc, exc_info=True)
            return None

    @staticmethod
    def _prepare_single_report_context(results: dict, analysis_log=None) -> dict:
        results_copy = copy.deepcopy(results or {})
        normalized = HTMLExporter._normalize_for_json(results_copy)
        analysis_log_text = analysis_log if analysis_log is not None else results_copy.get("analysis_log", "")
        hero = HTMLExporter._build_hero_context(results_copy)
        metrics = HTMLExporter._build_statistical_rows(results_copy)
        _es_type_top = str(results_copy.get("effect_size_type") or "")
        _es_value_top = results_copy.get("effect_size")
        _es_magnitude_top = HTMLExporter._effect_size_magnitude(_es_value_top, _es_type_top)
        for _r in metrics:
            if "info" not in _r:
                _r["info"] = HTMLExporter._stat_row_info(_r.get("label", ""))
            if _r.get("label") == "Effect size" and _es_magnitude_top:
                _r["magnitude"] = _es_magnitude_top
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
                "effect_size": row.get("effect_size_value"),
                "effect_size_type": row.get("effect_size_type") or row.get("effect_type") or "",
                "ci_lower": row.get("ci_lower"),
                "ci_upper": row.get("ci_upper"),
                "is_ratio": _effect_is_ratio(row.get("effect_size_type") or row.get("effect_type")),
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
            "decision_tree_json": json.dumps(decision_tree_json, ensure_ascii=False) if decision_tree_json else "null",
            "decision_path_json": json.dumps(decision_path, ensure_ascii=False),
            "statistical_rows": metrics,
            "assumptions": assumptions,
            "sphericity_correction_note": assumptions.get("sphericity_correction_note"),
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
            "group_factor_map_json": json.dumps(results_copy.get("group_factor_map", {}), ensure_ascii=False),
            "info_texts": HTMLExporter._info_texts(),
            "generated_warning": results_copy.get("error"),
            "normalized_results_json": json.dumps(normalized, cls=_ResultsEncoder, ensure_ascii=False),
            "math_render_enabled": math_render_enabled,
        }

    @staticmethod
    def _prepare_multi_report_context(all_results: dict) -> dict:
        cards = []
        significant_count = 0
        for idx, (dataset_name, results) in enumerate((all_results or {}).items()):
            r = results or {}
            hero = HTMLExporter._build_hero_context(r, dataset_name=str(dataset_name))
            assumptions = HTMLExporter._build_assumption_summary(r)

            # --- detail context (accordion panel) ---
            stat_rows = HTMLExporter._build_statistical_rows(r)
            _es_type_top = str(r.get("effect_size_type") or "")
            _es_value_top = r.get("effect_size")
            _es_magnitude_top = HTMLExporter._effect_size_magnitude(_es_value_top, _es_type_top)
            for _r in stat_rows:
                if "info" not in _r:
                    _r["info"] = HTMLExporter._stat_row_info(_r.get("label", ""))
                if _r.get("label") == "Effect size" and _es_magnitude_top:
                    _r["magnitude"] = _es_magnitude_top
            decision_path = HTMLExporter._build_decision_path_model(r)
            methods_text = HTMLExporter._build_methods_text(r, r.get("analysis_log", ""))

            # Scatter / correlation chart as inline Plotly HTML
            # Each card gets a unique div_id so Plotly can render all of them
            scatter_html = ""
            try:
                chart = HTMLExporter._build_correlation_chart(r, div_id=f"biomedstatx-scatter-{idx}")
                if chart and chart.get("html"):
                    scatter_html = chart["html"]
            except Exception:
                logger.error("correlation chart build failed for card %s", idx, exc_info=True)

            # FDR-adjusted p if available
            p_fdr = r.get("p_value_fdr")
            p_fdr_display = HTMLExporter._format_p_value(p_fdr) if p_fdr is not None else None

            card_id = f"card-{idx}"
            cards.append({
                "card_id": card_id,
                "dataset_name": str(dataset_name),
                "test_name": hero["test_name"],
                "p_value_display": hero["p_value_display"],
                "p_fdr_display": p_fdr_display,
                "significance_label": hero["significance_label"],
                "significance_class": hero["significance_class"],
                "effect_size_display": hero["effect_size_display"],
                "effect_label": hero["effect_label"],
                "effect_magnitude": hero["effect_magnitude"],
                "transformation": str(r.get("transformation") or "None"),
                "pairwise_count": len(r.get("pairwise_comparisons") or []),
                "summary_note": hero["summary_note"],
                "assumptions": assumptions,
                "sphericity_correction_note": assumptions.get("sphericity_correction_note"),
                # detail fields
                "stat_rows": stat_rows,
                "decision_path": decision_path,
                "methods_text": methods_text,
                "scatter_html": scatter_html,
            })
            if hero["is_significant"]:
                significant_count += 1

        math_render_enabled = any(
            HTMLExporter._requires_math_rendering((results or {}), HTMLExporter._build_hero_context(results or {}, dataset_name=str(dataset_name)))
            for dataset_name, results in (all_results or {}).items()
        )

        n_valid_for_fdr = sum(
            1 for res in (all_results or {}).values()
            if not (res or {}).get("error") and (res or {}).get("p_value") is not None
        )
        fdr_note = None
        if any((res or {}).get("p_value_fdr") is not None for res in (all_results or {}).values()):
            fdr_note = f"FDR correction (Benjamini-Hochberg) applied to m = {n_valid_for_fdr} tests."

        return {
            "mode": "multi",
            "report_title": "BioMedStatX Multi-Dataset Scientific Report",
            "subtitle": f"{len(cards)} datasets summarized, {significant_count} significant main results.",
            "fdr_note": fdr_note,
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
    def _build_decision_path_model(results: dict) -> list[dict]:
        """Decision-path breadcrumb. See :mod:`report_methods`."""
        from export.report_methods import build_decision_path_model
        return build_decision_path_model(
            results,
            format_p_value=HTMLExporter._format_p_value,
            build_summary_note=HTMLExporter._build_summary_note,
        )

    @staticmethod
    def _build_decision_tree_json(results: dict) -> dict | None:
        try:
            from visualization.decisiontreevisualizer import DecisionTreeVisualizer
            return DecisionTreeVisualizer.get_tree_json(results)
        except Exception as exc:
            logger.warning("decision tree JSON failed: %s", exc, exc_info=True)
            return None

    @staticmethod
    def _build_methods_text(results: dict, analysis_log: Any) -> str:
        """Methods-section paragraph. See :mod:`report_methods`."""
        from export.report_methods import build_methods_text
        return build_methods_text(
            results,
            analysis_log,
            format_metric=HTMLExporter._format_metric,
        )

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
        # Multi-mode: scatter charts live inside dataset_cards, not chart_blocks
        if mode == "multi" and not plotly_enabled:
            plotly_enabled = any(
                bool(card.get("scatter_html"))
                for card in (context.get("dataset_cards") or [])
            )
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
        # Templates are bundled as package data (see BioMedStatX.spec), so a
        # missing template is a packaging bug: let it raise loudly rather than
        # silently falling back to a hand-maintained inline copy that can drift.
        template = env.get_template(template_name)
        return template.render(
            context=context,
            mode=mode,
            plotly_bundle=plotly_bundle,
            mathjax_bundle=math_bundle,
            mathjax_status=math_status,
        )

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
                "Interaction plots and profile plots show cell means \u00b1 SE across factor levels.\n"
                "Click the \u24d8 button on each chart for a description of what it shows.\n"
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
