"""Assumption / descriptive / raw-data summaries for the HTML report.

Extracted from ``html_exporter.py`` (Phase 6 of the god-file split): the
assumption-check summary and its visuals (distribution dashboard, residuals-
vs-fitted, interpretation, KDE), the descriptive-statistics summary, and the
raw-data table. Stateless ``@staticmethod`` helpers mixed into ``HTMLExporter``;
call sites unchanged via the MRO.
"""

import numpy as np
from core.level_order import natural_order, order_is_defined
from scipy import stats

from export.report_charts import _ChartsMixin
from export.report_formatting import _FormattingMixin
from export.report_stat_rows import _StatRowsMixin

try:
    from core.logger_config import get_logger
except ImportError:  # pragma: no cover — fallback when logger_config missing
    import logging as _logging

    def get_logger(name):
        return _logging.getLogger(name)


logger = get_logger(__name__)


class _SummariesMixin:
    """Summaries helpers mixed into ``HTMLExporter``."""

    @staticmethod
    def _build_assumption_summary(results: dict) -> dict:
        rows = []
        model_type = results.get("model_type", "")

        # C6-1: Model-agnostic converged row
        converged = results.get("converged")
        if converged is not None:
            rows.append({
                "name": "Model Convergence",
                "statistic": "—",
                "p_value": "—",
                "p_value_style": "",
                "status_label": "Model converged successfully" if converged else "Did NOT converge — results may be unreliable",
                "status_class": "is-significant" if converged else "is-error",
            })


        # --- CorrelationMatrix: method justification ---
        if model_type == "CorrelationMatrix":
            method = str(results.get("method") or "").lower()
            method_map = {
                "pearson": "Pearson",
                "spearman": "Spearman",
                "auto": "Auto (Pearson or Spearman selected per pair by skewness/kurtosis tiers)",
            }
            method_label = method_map.get(method, method or "—")
            if method == "pearson":
                status_label = "Pearson requires bivariate normality — verify via Q-Q plots"
                status_class = "is-neutral"
            elif method == "spearman":
                status_label = "Spearman is distribution-free — no normality required"
                status_class = "is-significant"
            else:
                status_label = ("Method chosen per pair by skewness/kurtosis; see the "
                                "per-pair method matrix (Shapiro-Wilk is informational, not decisive)")
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

        # --- Correlation: method-selection basis (honest about what decides) ---
        if model_type == "Correlation":
            corr_method = str(results.get("method") or "").lower()
            if corr_method in ("pearson", "spearman"):
                rows.append({
                    "name": f"Correlation method: {corr_method.capitalize()}",
                    "statistic": "—",
                    "p_value": "—",
                    "p_value_style": "",
                    "status_label": ("Selected by skewness/kurtosis tiers "
                                     "(Shapiro-Wilk below is informational, not decisive)"),
                    "status_class": "is-neutral",
                })

        # --- Correlation: normality_check (Shapiro-Wilk per variable, informational) ---
        # The method is chosen by skewness/kurtosis tiers (see
        # _select_correlation_method); the per-variable Shapiro-Wilk shown here
        # documents distributional shape but does not drive Pearson vs Spearman.
        normality_check = results.get("normality_check") or {}
        if model_type == "Correlation" and normality_check:
            x_var = results.get("x_variable", "")
            y_var = results.get("y_variable", "")
            for var_key, var_label in [(x_var, results.get("x_variable_display", x_var)),
                                       (y_var, results.get("y_variable_display", y_var))]:
                payload = normality_check.get(var_key, {})
                if isinstance(payload, dict) and "p_value" in payload:
                    rows.append({
                        "name": f"Normality: {_FormattingMixin._prettify_label(var_label)} (Shapiro-Wilk)",
                        "statistic": _FormattingMixin._format_metric(payload.get("statistic")),
                        "p_value": _FormattingMixin._format_p_value(payload.get("p_value")),
                        "p_value_style": _FormattingMixin._p_heat_style(payload.get("p_value")),
                        "status_label": _FormattingMixin._bool_label(payload.get("normal")),
                        "status_class": _FormattingMixin._bool_class(payload.get("normal")),
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
                    "name": f"Shift before {tr_name}: {_FormattingMixin._prettify_label(disp)} ({axis}-axis)",
                    "statistic": f"c={shift:.4f}",
                    "p_value": "—",
                    "p_value_style": "",
                    "status_label": detail,
                    "status_class": "is-neutral",
                })

        # --- Linear Regression: diagnostics (Shapiro-Wilk residuals, Breusch-Pagan, Ramsey RESET) ---
        if model_type == "LinearRegression":
            diag = results.get("diagnostics") or {}
            
            norm_d = diag.get("normality") or {}
            if norm_d:
                if "error" in norm_d:
                    rows.append({
                        "name": "Normality of residuals (Shapiro-Wilk)",
                        "statistic": "N/A",
                        "p_value": "—",
                        "p_value_style": "",
                        "status_label": f"Test failed: {norm_d.get('error')}",
                        "status_class": "is-neutral",
                    })
                elif "p_value" in norm_d:
                    rows.append({
                        "name": "Normality of residuals (Shapiro-Wilk)",
                        "statistic": _FormattingMixin._format_metric(norm_d.get("statistic")),
                        "p_value": _FormattingMixin._format_p_value(norm_d.get("p_value")),
                        "p_value_style": _FormattingMixin._p_heat_style(norm_d.get("p_value")),
                        "status_label": _FormattingMixin._bool_label(norm_d.get("assumption_holds")),
                        "status_class": _FormattingMixin._bool_class(norm_d.get("assumption_holds")),
                    })
                    
            homo_d = diag.get("homoscedasticity") or {}
            if homo_d:
                if "error" in homo_d:
                    rows.append({
                        "name": "Homoscedasticity (Breusch-Pagan)",
                        "statistic": "N/A",
                        "p_value": "—",
                        "p_value_style": "",
                        "status_label": f"Test failed: {homo_d.get('error')}",
                        "status_class": "is-neutral",
                    })
                elif "p_value" in homo_d:
                    rows.append({
                        "name": "Homoscedasticity (Breusch-Pagan)",
                        "statistic": _FormattingMixin._format_metric(homo_d.get("statistic")),
                        "p_value": _FormattingMixin._format_p_value(homo_d.get("p_value")),
                        "p_value_style": _FormattingMixin._p_heat_style(homo_d.get("p_value")),
                        "status_label": _FormattingMixin._bool_label(homo_d.get("assumption_holds")),
                        "status_class": _FormattingMixin._bool_class(homo_d.get("assumption_holds")),
                    })
                    
            lin_d = diag.get("linearity") or {}
            if lin_d:
                if "error" in lin_d:
                    rows.append({
                        "name": "Linearity (Ramsey RESET)",
                        "statistic": "N/A",
                        "p_value": "—",
                        "p_value_style": "",
                        "status_label": f"Test failed: {lin_d.get('error')}",
                        "status_class": "is-neutral",
                    })
                elif "p_value" in lin_d:
                    rows.append({
                        "name": "Linearity (Ramsey RESET)",
                        "statistic": _FormattingMixin._format_metric(lin_d.get("statistic")),
                        "p_value": _FormattingMixin._format_p_value(lin_d.get("p_value")),
                        "p_value_style": _FormattingMixin._p_heat_style(lin_d.get("p_value")),
                        "status_label": _FormattingMixin._bool_label(lin_d.get("assumption_holds")),
                        "status_class": _FormattingMixin._bool_class(lin_d.get("assumption_holds")),
                    })

        # --- Logistic Regression: Hosmer-Lemeshow goodness-of-fit + AUC interpretation ---
        if model_type == "LogisticRegression":
            hl = results.get("hosmer_lemeshow") or {}
            if hl and "p_value" in hl:
                hl_p = hl.get("p_value")
                # HL passes (good fit) when p > 0.05 (no significant deviation from expected)
                goodness_of_fit = isinstance(hl_p, (int, float)) and hl_p > 0.05
                rows.append({
                    "name": "Goodness-of-fit (Hosmer-Lemeshow)",
                    "statistic": _FormattingMixin._format_metric(hl.get("chi2")),
                    "p_value": _FormattingMixin._format_p_value(hl_p),
                    "p_value_style": _FormattingMixin._p_heat_style(hl_p),
                    "status_label": _FormattingMixin._bool_label(goodness_of_fit),
                    "status_class": _FormattingMixin._bool_class(goodness_of_fit),
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
                    "statistic": _FormattingMixin._format_metric(auc),
                    "p_value": "—",
                    "p_value_style": "",
                    "status_label": auc_interp,
                    "status_class": "is-significant" if auc_ok else "is-danger",
                })

        # --- ANCOVA: residual normality + slope homogeneity ---
        if model_type == "ANCOVA":
            normality_tests = results.get("normality_tests") or {}
            if normality_tests:
                for label, payload in normality_tests.items():
                    if not isinstance(payload, dict):
                        continue
                    rows.append({
                        "name": f"Normality: {_FormattingMixin._prettify_label(label)} (Shapiro-Wilk)",
                        "statistic": _FormattingMixin._format_metric(payload.get("statistic")),
                        "p_value": _FormattingMixin._format_p_value(payload.get("p_value")),
                        "p_value_style": _FormattingMixin._p_heat_style(payload.get("p_value")),
                        "status_label": _FormattingMixin._bool_label(payload.get("is_normal")),
                        "status_class": _FormattingMixin._bool_class(payload.get("is_normal")),
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
                    "statistic": _FormattingMixin._format_metric(f_val),
                    "p_value": _FormattingMixin._format_p_value(p_val),
                    "p_value_style": _FormattingMixin._p_heat_style(p_val),
                    "status_label": _FormattingMixin._bool_label(holds),
                    "status_class": _FormattingMixin._bool_class(holds),
                })

        # --- LMM: convergence + ICC + residual normality ---
        if model_type == "LMM":
            normality_tests = results.get("normality_tests") or {}
            if normality_tests:
                for label, payload in normality_tests.items():
                    if not isinstance(payload, dict):
                        continue
                    rows.append({
                        "name": f"Normality: {_FormattingMixin._prettify_label(label)} (Shapiro-Wilk)",
                        "statistic": _FormattingMixin._format_metric(payload.get("statistic")),
                        "p_value": _FormattingMixin._format_p_value(payload.get("p_value")),
                        "p_value_style": _FormattingMixin._p_heat_style(payload.get("p_value")),
                        "status_label": _FormattingMixin._bool_label(payload.get("is_normal")),
                        "status_class": _FormattingMixin._bool_class(payload.get("is_normal")),
                    })
            else:
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
                "status_class": _FormattingMixin._bool_class(conv_holds),
            })
            icc_val = results.get("icc")
            icc_interp = _StatRowsMixin._lmm_icc_interpretation(icc_val)
            icc_justified = (float(icc_val) >= 0.1) if isinstance(icc_val, (int, float)) else None
            rows.append({
                "name": "Intraclass Correlation (ICC) — ICC > 0.1 justifies LMM",
                "statistic": _FormattingMixin._format_metric(icc_val),
                "p_value": icc_interp,
                "p_value_style": "",
                "status_label": "LMM justified" if icc_justified else ("ICC < 0.1 — LMM may be unnecessary" if icc_justified is False else "N/A"),
                "status_class": "is-significant" if icc_justified else ("is-neutral" if icc_justified is None else "is-danger"),
            })

        # --- Standard tests: normality_tests + fallback from test_info ---
        if model_type not in ("CorrelationMatrix", "Correlation", "LinearRegression", "LogisticRegression", "ANCOVA", "LMM"):
            normality_tests = results.get("normality_tests", {}) or {}
            # Fallback: extract from nested test_info structure used by one-way ANOVA path
            test_info_raw = results.get("test_info", {}) or {}
            if not normality_tests and test_info_raw:
                # Fill-gate, not name-gate (presence-vs-value audit 2026-08): read post
                # only when the post block is present -- today absent only on the
                # already-transformed "No further" early-return, and robust if a future
                # engine skips the post recompute when no transform was needed.
                _phase = "post_transformation" if test_info_raw.get("post_transformation") else "pre_transformation"
                _norm = test_info_raw.get(_phase, {}).get("residuals_normality", {})
                if _norm:
                    normality_tests = {"Model residuals": _norm}
            for label, payload in normality_tests.items():
                if not isinstance(payload, dict):
                    continue
                rows.append({
                    "name": f"Normality: {_FormattingMixin._prettify_label(label)}",
                    "statistic": _FormattingMixin._format_metric(payload.get("statistic")),
                    "p_value": _FormattingMixin._format_p_value(payload.get("p_value")),
                    "p_value_style": _FormattingMixin._p_heat_style(payload.get("p_value")),
                    "status_label": _FormattingMixin._bool_label(payload.get("is_normal")),
                    "status_class": _FormattingMixin._bool_class(payload.get("is_normal")),
                })
            variance_test = results.get("variance_test", {}) or {}
            # Fallback: extract from nested test_info structure
            if not variance_test and test_info_raw:
                # Fill-gate, not name-gate (presence-vs-value audit 2026-08): read post
                # only when the post block is present -- today absent only on the
                # already-transformed "No further" early-return, and robust if a future
                # engine skips the post recompute when no transform was needed.
                _phase = "post_transformation" if test_info_raw.get("post_transformation") else "pre_transformation"
                variance_test = test_info_raw.get(_phase, {}).get("variance", {}) or {}
            if isinstance(variance_test, dict) and variance_test:
                _var_name = variance_test.get("test_name", "Levene")
                rows.append({
                    "name": f"Variance homogeneity ({_var_name})",
                    "statistic": _FormattingMixin._format_metric(variance_test.get("statistic")),
                    "p_value": _FormattingMixin._format_p_value(variance_test.get("p_value")),
                    "p_value_style": _FormattingMixin._p_heat_style(variance_test.get("p_value")),
                    "status_label": _FormattingMixin._bool_label(variance_test.get("equal_variance")),
                    "status_class": _FormattingMixin._bool_class(variance_test.get("equal_variance")),
                })
                transformed = variance_test.get("transformed")
                if isinstance(transformed, dict):
                    _var_name_tr = transformed.get("test_name", _var_name)
                    rows.append({
                        "name": f"Variance homogeneity ({_var_name_tr}, transformed)",
                        "statistic": _FormattingMixin._format_metric(transformed.get("statistic")),
                        "p_value": _FormattingMixin._format_p_value(transformed.get("p_value")),
                        "p_value_style": _FormattingMixin._p_heat_style(transformed.get("p_value")),
                        "status_label": _FormattingMixin._bool_label(transformed.get("equal_variance")),
                        "status_class": _FormattingMixin._bool_class(transformed.get("equal_variance")),
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

        # RM writes "sphericity_test"; Mixed writes the same information under a
        # "within_" prefix, because a mixed design has two effect families and the
        # within verdict has to be qualified. Reading only the RM key rendered an
        # EMPTY assumption summary for Mixed even at Mauchly p = 1.6e-22.
        #
        # Keyed on PRESENCE, not truthiness: an RM payload whose sphericity_test is
        # an empty-but-present dict must stay on the RM branch instead of silently
        # borrowing the mixed field. `x or y` would fall through in that case.
        if "sphericity_test" in results:
            sphericity = results.get("sphericity_test") or {}
            _corr_key, _corr_block_key = "correction_used", "sphericity_corrections"
        else:
            sphericity = results.get("within_sphericity_test") or {}
            _corr_key, _corr_block_key = "within_correction_used", "within_sphericity_corrections"
        if isinstance(sphericity, dict) and sphericity:
            status_value = sphericity.get("sphericity_met")
            if status_value is None and sphericity.get("p_value") is not None:
                status_value = sphericity.get("p_value") >= 0.05
            rows.append({
                "name": "Sphericity (Mauchly's W)",
                "statistic": _FormattingMixin._format_metric(sphericity.get("W") or sphericity.get("statistic")),
                "p_value": _FormattingMixin._format_p_value(sphericity.get("p_value")),
                "p_value_style": _FormattingMixin._p_heat_style(sphericity.get("p_value")),
                "status_label": _FormattingMixin._bool_label(status_value),
                "status_class": _FormattingMixin._bool_class(status_value),
            })
            if status_value is False:
                # Primary source: statisticaltester.py writes the correction label
                # to the top-level "correction_used" key and the epsilon values
                # nested under "sphericity_corrections", NOT into the
                # "sphericity_test" sub-dict this function reads for W/p_value.
                # _corr_key/_corr_block_key point at the RM or the Mixed spelling,
                # chosen above by the same presence check.
                top_correction = str(results.get(_corr_key) or "")
                sph_corrections = results.get(_corr_block_key) or {}
                gg_block = sph_corrections.get("greenhouse_geisser") or {}
                hf_block = sph_corrections.get("huynh_feldt") or {}
                if not (gg_block or hf_block):
                    # Mixed nests its epsilon one level deeper, under the effect it
                    # belongs to.
                    _main = sph_corrections.get("main_effect") or {}
                    if isinstance(_main, dict):
                        gg_block = _main.get("greenhouse_geisser") or {}
                        hf_block = _main.get("huynh_feldt") or {}
                gg_eps = gg_block.get("epsilon") if isinstance(gg_block, dict) else None
                hf_eps = hf_block.get("epsilon") if isinstance(hf_block, dict) else None
                corr = top_correction.lower()
                if not corr:
                    # Fallback for older/serialized payloads that only populated
                    # the sphericity_test sub-dict directly.
                    corr = (sphericity.get("correction") or sphericity.get("correction_applied") or "").lower()
                    if gg_eps is None:
                        gg_eps = sphericity.get("greenhouse_geisser") or sphericity.get("gg_epsilon") or sphericity.get("epsilon_gg")
                    if hf_eps is None:
                        hf_eps = sphericity.get("huynh_feldt") or sphericity.get("hf_epsilon") or sphericity.get("epsilon_hf")
                if "huynh" in corr or "hf" in corr:
                    label = "Huynh-Feldt"
                    eps = hf_eps if hf_eps is not None else gg_eps
                elif gg_eps is not None or "greenhouse" in corr or "gg" in corr:
                    label = "Greenhouse-Geisser"
                    eps = gg_eps
                else:
                    label, eps = "Greenhouse-Geisser", gg_eps
                if label:
                    eps_str = f" (ε = {_FormattingMixin._format_metric(eps)})" if eps is not None else ""
                    sphericity_correction_note = f"Sphericity violated → {label} correction applied{eps_str}"
        _icons = {"is-significant": "✓ ", "is-danger": "✗ ", "is-neutral": "~ "}
        for row in rows:
            row["status_label"] = _icons.get(row["status_class"], "") + row["status_label"]
        _trafo_label = str(results.get("transformation") or "").strip()
        # Gate the transformed-data plots on an ACTUAL value change, not the label
        # (presence-vs-value audit 2026-08): mirror the value-comparison already used
        # for the transformed COLUMN below, so a named-but-inert transform does not
        # spawn a transformed Q-Q / distribution plot identical to the raw one.
        from statistical_testing.validators import grouped_samples_changed
        _has_transform = grouped_samples_changed(
            results.get("raw_data", {}) or {},
            results.get("raw_data_transformed") or results.get("transformed_data") or {},
        )
        _test_info = results.get("test_info") if isinstance(results.get("test_info"), dict) else {}
        transform_warning = results.get("transform_warning") or _test_info.get("transform_warning")
        # Level-ordering transparency. This note used to fire on every composite
        # interaction-cell label, because the old test asked whether a label was
        # *entirely* numeric rather than whether a number had decided its
        # position; it was muted rather than corrected. With order_is_defined()
        # it speaks only when the order really is alphabetical guesswork, which
        # is worth saying: it is also the reason a plot will not connect
        # individual subjects across those levels.
        order_defined, order_note = order_is_defined(results.get("groups") or [])
        return {
            "rows": rows,
            # Kept out of data_health_warnings on purpose: that block is the
            # red "pre-analysis data quality" table, and a level order being
            # alphabetical is neither a data defect nor a danger. It is a
            # statement about how the axis was arranged, so it renders as a
            # plain note.
            "level_order_note": "" if order_defined else order_note,
            "data_health_warnings": _SummariesMixin._build_data_health_warnings(results),
            "transformation": _trafo_label or "None",
            "interpretation": _SummariesMixin._build_assumption_interpretation(results, rows),
            "sphericity_correction_note": sphericity_correction_note,
            "transform_warning": transform_warning,
            "qq_plot_html": _SummariesMixin._build_assumption_visuals(results),
            "qq_plot_transformed_html": _SummariesMixin._build_assumption_visuals(results, source="transformed") if _has_transform else None,
            "distribution_plot_html": _SummariesMixin._build_distribution_dashboard_chart(results),
            "distribution_plot_transformed_html": _SummariesMixin._build_distribution_dashboard_chart(results, source="transformed") if _has_transform else None,
            "residual_plot_html": _SummariesMixin._build_residuals_vs_fitted_chart(results),
            "transformation_label": _trafo_label,
        }

    @staticmethod
    def _build_data_health_warnings(results: dict) -> list[str]:
        """Pre-analysis data-quality findings from DataHealthScanner.

        The scanner runs on every clinical model (covariate outliers, Little's
        MCAR, VIF, quasi-separation, group sizes) and writes into
        ``results["data_health"]``. Returns an empty list when the data is
        clean, so the template renders nothing rather than an empty block.
        """
        health = results.get("data_health") or {}
        if not isinstance(health, dict):
            return []
        warnings_list = health.get("warnings") or []
        return [str(w) for w in warnings_list if str(w).strip()]

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
                        "n": _FormattingMixin._format_metric(level_data.get("n")),
                        "raw_mean": _FormattingMixin._format_metric(level_data.get("raw_mean")),
                        "adj_mean": _FormattingMixin._format_metric(level_data.get("adjusted_mean")),
                        "raw_sd": _FormattingMixin._format_metric(level_data.get("raw_sd")),
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
                    stats_dict = {
                        "mean": float(np.mean(arr)),
                        "median": float(np.median(arr)),
                        "sd": float(np.std(arr, ddof=1)) if len(arr) > 1 else None,
                        "sem": float(stats.sem(arr)) if len(arr) > 1 else None,
                        "min": float(np.min(arr)),
                        "max": float(np.max(arr)),
                    }
                    formatted = _FormattingMixin._format_metric_row(stats_dict)
                    rows.append({
                        "group": str(label),
                        "n": len(arr),
                        **formatted
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
                    "group": _FormattingMixin._prettify_label(var),
                    "n": _FormattingMixin._format_metric(int(n_val) if n_val is not None else None),
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
                    "value": _FormattingMixin._format_metric(results.get("n_subjects")),
                },
                {
                    "label": "N total observations",
                    "value": _FormattingMixin._format_metric(results.get("n_observations")),
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
                numeric = _FormattingMixin._coerce_numeric_sequence(values)
                if not numeric:
                    continue
                stats_dict = {
                    "mean": np.mean(numeric),
                    "median": np.median(numeric),
                    "sd": np.std(numeric, ddof=1) if len(numeric) > 1 else None,
                    "sem": stats.sem(numeric) if len(numeric) > 1 else None,
                    "min": np.min(numeric),
                    "max": np.max(numeric),
                }
                formatted = _FormattingMixin._format_metric_row(stats_dict)
                rows.append({
                    "group": str(group_name),
                    "n": len(numeric),
                    **formatted
                })
        if not rows and results.get("descriptive"):
            for group_name, payload in (results.get("descriptive") or {}).items():
                if not isinstance(payload, dict):
                    continue
                stats_dict = {
                    "mean": payload.get("mean"),
                    "median": payload.get("median"),
                    "sd": payload.get("sd") or payload.get("std"),
                    "sem": payload.get("sem"),
                    "min": payload.get("min"),
                    "max": payload.get("max"),
                }
                formatted = _FormattingMixin._format_metric_row(stats_dict)
                rows.append({
                    "group": str(group_name),
                    "n": _FormattingMixin._format_metric(payload.get("n")),
                    **formatted
                })
        has_tr = bool(transformed and transformed != raw_data)
        note_str = None
        if has_tr:
            tr_notes = []
            for group_name, values in transformed.items():
                numeric = _FormattingMixin._coerce_numeric_sequence(values)
                if not numeric:
                    continue
                mean = np.mean(numeric)
                sd = np.std(numeric, ddof=1) if len(numeric) > 1 else 0.0
                mean_str = _FormattingMixin._format_metric(float(mean))
                sd_str = _FormattingMixin._format_metric(float(sd))
                tr_notes.append(f"{group_name}: {mean_str} ± {sd_str}")
            if tr_notes:
                note_str = "Transformed-scale means (Mean ± SD): " + "; ".join(tr_notes)

        return {
            "rows": rows,
            "has_transformed": has_tr,
            "title": "Group-level summary",
            "group_col_label": "Group",
            "note": note_str,
        }

    @staticmethod
    def _build_raw_data_table(results: dict) -> dict:
        # Column-mode: correlation/regression embeds raw data as named columns
        raw_data_columns = results.get("raw_data_columns")
        if raw_data_columns and isinstance(raw_data_columns, dict):
            col_names = list(raw_data_columns.keys())
            col_values = [raw_data_columns[c] for c in col_names]
            n_rows = max((len(v) for v in col_values), default=0)
            rows = []
            for i in range(n_rows):
                row = {"index": i + 1}
                for col, vals in zip(col_names, col_values):
                    row[col] = _FormattingMixin._format_metric(vals[i] if i < len(vals) else None, digits=6)
                rows.append(row)
            return {
                "rows": rows,
                "has_transformed": False,
                "column_mode": True,
                "columns": col_names,
            }

        # Group-mode: group-based analyses embed raw data keyed by group name
        raw_data = results.get("raw_data") or results.get("samples") or {}
        transformed = results.get("raw_data_transformed") or results.get("transformed_data") or {}
        subjects = results.get("raw_data_subjects") or {}
        if not isinstance(subjects, dict):
            subjects = {}
        rows = []
        if isinstance(raw_data, dict):
            for group_name, values in raw_data.items():
                raw_values = list(values) if values is not None else []
                transformed_source = transformed.get(group_name, []) if isinstance(transformed, dict) else []
                transformed_values = list(transformed_source) if transformed_source is not None else []
                max_len = max(len(raw_values), len(transformed_values), 1)
                # A per-group row number is deliberately absent. In an
                # independent design the order inside a group is import order
                # and carries nothing; in a repeated-measures design a shared
                # row number reads across levels as "the same subject" and is
                # not, because the values are filtered per level in whatever
                # order the frame holds. Where the design actually has subjects
                # the extractor hands them over and they are printed instead --
                # a real identity rather than a position that resembles one.
                subject_ids = subjects.get(group_name) or []
                for index in range(max_len):
                    raw_value = raw_values[index] if index < len(raw_values) else None
                    transformed_value = transformed_values[index] if index < len(transformed_values) else None
                    row = {
                        "group": str(group_name),
                        "raw_value": _FormattingMixin._format_metric(raw_value, digits=6),
                        "transformed_value": _FormattingMixin._format_metric(transformed_value, digits=6),
                    }
                    if index < len(subject_ids):
                        row["subject"] = str(subject_ids[index])
                    rows.append(row)
        return {
            "rows": rows,
            # Show the transformed column only when a transformation actually
            # changed the values. Some engines populate raw_data_transformed
            # with a no-op copy of the raw data (transformation selected but
            # identity, e.g. Box-Cox with lambda 1); a transformed column that
            # merely mirrors the raw column is meaningless and confusing.
            "has_transformed": any(
                row["transformed_value"] != "N/A"
                and row["transformed_value"] != row["raw_value"]
                for row in rows
            ),
            "column_mode": False,
            "columns": [],
            "has_subjects": any("subject" in row for row in rows),
        }

    @staticmethod
    def _build_assumption_visuals(results: dict, *, source: str = "raw") -> str | None:
        """Build a Q-Q plot.

        source='raw'         → residuals/raw data on the original scale
        source='transformed' → values from results['raw_data_transformed']
                               (group-combined). Returns None if no transform.
        """
        values = None
        if source == "transformed":
            transformed = results.get("raw_data_transformed") or results.get("transformed_data") or {}
            if not isinstance(transformed, dict) or not transformed:
                return None
            combined = []
            for group_values in transformed.values():
                combined.extend(_FormattingMixin._coerce_numeric_sequence(group_values))
            values = combined
        elif "model_residuals" in results:
            values = _FormattingMixin._coerce_numeric_sequence(results.get("model_residuals"))
        elif "residuals" in results:
            values = _FormattingMixin._coerce_numeric_sequence(results.get("residuals"))
        else:
            raw_data = results.get("raw_data") or results.get("samples") or {}
            if isinstance(raw_data, dict):
                combined = []
                for group_values in raw_data.values():
                    combined.extend(_FormattingMixin._coerce_numeric_sequence(group_values))
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
            figure.update_layout(**_ChartsMixin._base_layout(
                margin=dict(l=56, r=16, t=30, b=60),
                xaxis=dict(title=dict(text="Theoretical quantiles", font=dict(size=11), standoff=8)),
                yaxis=dict(title=dict(text="Observed quantiles", font=dict(size=11), standoff=8)),
                legend=dict(orientation="h", y=1.08, x=0)
            ))
            div_id = "biomedstatx-qq-chart" if source == "raw" else "biomedstatx-qq-chart-transformed"
            return _ChartsMixin._figure_to_html(figure, div_id=div_id)
        except Exception as exc:
            logger.warning("QQ plot generation failed: %s", exc, exc_info=True)
            return None

    @staticmethod
    def _build_distribution_dashboard_chart(results: dict, *, source: str = "raw") -> str | None:
        if source == "transformed":
            raw_data = results.get("raw_data_transformed") or results.get("transformed_data") or {}
        else:
            raw_data = results.get("raw_data") or results.get("samples") or {}
        if not isinstance(raw_data, dict) or not raw_data:
            return None
        try:
            import plotly.graph_objects as go

            figure = go.Figure()
            palette = ["#0f766e", "#1f7a5a", "#b7791f", "#9f3a38", "#1d4ed8", "#7c3aed"]
            added = 0
            for idx, (group_name, values) in enumerate(raw_data.items()):
                numeric = _FormattingMixin._coerce_numeric_sequence(values)
                if len(numeric) < 2:
                    continue
                color = palette[idx % len(palette)]
                figure.add_trace(
                    go.Box(
                        y=numeric,
                        name=_FormattingMixin._esc(group_name),
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
            figure.update_layout(**_ChartsMixin._base_layout(
                margin=dict(l=48, r=20, t=24, b=48),
                height=500
            ))
            figure.update_xaxes(title_text="Groups")
            figure.update_yaxes(title_text="Observed values", zeroline=False)
            return _ChartsMixin._figure_to_html(figure)
        except Exception as exc:
            logger.warning("group distribution chart generation failed: %s", exc, exc_info=True)
            return None

    @staticmethod
    def _build_residuals_vs_fitted_chart(results: dict) -> str | None:
        residuals = _FormattingMixin._coerce_numeric_sequence(results.get("model_residuals") or results.get("residuals"))
        fitted = _FormattingMixin._coerce_numeric_sequence(results.get("fitted_values") or results.get("fitted"))
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
            figure.update_layout(**_ChartsMixin._base_layout(
                margin=dict(l=48, r=20, t=24, b=42),
                xaxis_title="Fitted values",
                yaxis_title="Residuals"
            ))
            return _ChartsMixin._figure_to_html(figure)
        except Exception as exc:
            logger.warning("residual-vs-fitted chart generation failed: %s", exc, exc_info=True)
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
