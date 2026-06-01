"""Per-test statistical results table rows for the HTML report.

Extracted from ``html_exporter.py`` (Phase 3 of the god-file split). Each
method maps an analysis ``results`` dict to the list of display rows for one
test family (ANCOVA, LMM, correlation matrix, beta/logistic regression,
factorial ANOVA, generic, pairwise). Stateless ``@staticmethod`` helpers;
``HTMLExporter`` mixes them in so existing call sites keep working via MRO.

Formatting helpers are referenced on ``_FormattingMixin`` directly to avoid
importing ``html_exporter`` back (no circular import).
"""

import pandas as pd

from export.report_formatting import _FormattingMixin

try:
    from core.logger_config import get_logger
except ImportError:  # pragma: no cover — fallback when logger_config missing
    import logging as _logging

    def get_logger(name):
        return _logging.getLogger(name)


logger = get_logger(__name__)


class _StatRowsMixin:
    """Statistical-rows builders mixed into ``HTMLExporter``."""

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
                f"{_FormattingMixin._format_metric(ss)} | "
                f"df={_FormattingMixin._format_metric(df)} | "
                f"F={_FormattingMixin._format_metric(f_val)} | "
                f"{_FormattingMixin._format_p_value(p_val)} | "
                f"η²={_FormattingMixin._format_metric(eta_sq)}"
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
                    f"β={_FormattingMixin._format_metric(beta)} | "
                    f"SE={_FormattingMixin._format_metric(se)} | "
                    f"t={_FormattingMixin._format_metric(t_val)} | "
                    f"{_FormattingMixin._format_p_value(p_val)} | "
                    f"[{_FormattingMixin._format_metric(ci_l)}, {_FormattingMixin._format_metric(ci_u)}]"
                )
                rows.append({"label": cov, "value": value})

        # --- Table 3: Simple Slopes & Johnson-Neyman ---
        ssa = results.get("simple_slopes_analysis")
        if ssa:
            rows.append({"label": "── Simple Slopes & Johnson-Neyman ──", "value": ""})
            rows.append({"label": "Moderator/Covariate", "value": str(ssa.get("covariate_name", ""))})
            rows.append({"label": "Primary Factor", "value": str(ssa.get("factor_name", ""))})
            for slope in ssa.get("simple_slopes", []):
                lvl = slope.get("covariate_label", "")
                val = slope.get("covariate_value")
                beta = slope.get("coefficient")
                se = slope.get("std_err")
                t_val = slope.get("t_value")
                p_val = slope.get("p_value")
                ci_l = slope.get("ci_lower")
                ci_u = slope.get("ci_upper")
                lbl = f"Simple Slope at {lvl} ({_FormattingMixin._format_metric(val)})"
                value = (
                    f"β={_FormattingMixin._format_metric(beta)} | "
                    f"SE={_FormattingMixin._format_metric(se)} | "
                    f"t={_FormattingMixin._format_metric(t_val)} | "
                    f"{_FormattingMixin._format_p_value(p_val)} | "
                    f"[{_FormattingMixin._format_metric(ci_l)}, {_FormattingMixin._format_metric(ci_u)}]"
                )
                rows.append({"label": lbl, "value": value})
            
            jn = ssa.get("johnson_neyman")
            if jn:
                roots = jn.get("roots", [])
                if len(roots) >= 2:
                    rows.append({"label": "J-N Critical Interval", "value": f"[{roots[0]:.4f}, {roots[1]:.4f}]"})
                sig_regs = jn.get("significant_regions", [])
                rows.append({"label": "J-N Significant Regions", "value": ", ".join(sig_regs) if sig_regs else "None in range"})

        # --- Table 4: Model Fit ---
        rows.append({"label": "── Model Fit ──", "value": ""})
        rows.append({"label": "R²", "value": _FormattingMixin._format_metric(results.get("r_squared"))})
        rows.append({"label": "Adjusted R²", "value": _FormattingMixin._format_metric(results.get("r_squared_adj"))})
        _xt_fit = results.get("x_transform", "none") or "none"
        _yt_fit = results.get("y_transform", "none") or "none"
        if _xt_fit != "none" or _yt_fit != "none":
            rows.append({
                "label": "R² Note",
                "value": (
                    "R² is on the transformed scale (X: {}, Y: {}). "
                    "It is not directly comparable to untransformed models and "
                    "does not reflect variance explained in the original units."
                ).format(_xt_fit, _yt_fit),
            })
        rows.append({"label": "AIC", "value": _FormattingMixin._format_metric(results.get("aic"))})
        rows.append({"label": "N observations", "value": _FormattingMixin._format_metric(results.get("n_observations"))})

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
        rows.append({"label": "Parameter", "value": "Coefficient | SE | df | t/z | p-value | 95% CI"})
        for fe in fe_table:
            param = str(fe.get("parameter", ""))
            coef = fe.get("coefficient")
            se = fe.get("std_err")
            df_val = fe.get("df")
            z = fe.get("z_value")
            p_val = fe.get("p_value")
            ci_l = fe.get("ci_lower")
            ci_u = fe.get("ci_upper")
            df_str = str(df_val) if df_val is not None else "N/A"
            value = (
                f"β={_FormattingMixin._format_metric(coef)} | "
                f"SE={_FormattingMixin._format_metric(se)} | "
                f"df={df_str} | "
                f"t/z={_FormattingMixin._format_metric(z)} | "
                f"{_FormattingMixin._format_p_value(p_val)} | "
                f"[{_FormattingMixin._format_metric(ci_l)}, {_FormattingMixin._format_metric(ci_u)}]"
            )
            rows.append({"label": param, "value": value})

        # Table 2: Random Effects & Model Fit
        rows.append({"label": "── Random Effects & Model Fit ──", "value": ""})
        rows.append({
            "label": "Random intercept variance",
            "value": _FormattingMixin._format_metric(results.get("random_effects_variance")),
        })
        rows.append({
            "label": "Residual variance",
            "value": _FormattingMixin._format_metric(results.get("residual_variance")),
        })
        icc_val = results.get("icc")
        icc_interp = _StatRowsMixin._lmm_icc_interpretation(icc_val)
        rows.append({
            "label": "Intraclass Correlation (ICC)",
            "value": f"{_FormattingMixin._format_metric(icc_val)} — {icc_interp}",
        })
        rows.append({"label": "AIC", "value": _FormattingMixin._format_metric(results.get("aic"))})
        rows.append({"label": "BIC", "value": _FormattingMixin._format_metric(results.get("bic"))})
        rows.append({"label": "Log-likelihood", "value": _FormattingMixin._format_metric(results.get("log_likelihood"))})
        rows.append({"label": "N subjects", "value": _FormattingMixin._format_metric(results.get("n_subjects"))})
        rows.append({"label": "N observations", "value": _FormattingMixin._format_metric(results.get("n_observations"))})
        rows.append({"label": "Degrees of freedom method", "value": str(results.get("df_method") or "N/A")})
        rows.append({"label": "Random structure chosen", "value": str(results.get("random_structure_chosen") or "N/A")})
        
        lrt_perf = results.get("lrt_performed")
        if lrt_perf:
            lrt_stat = results.get("lrt_statistic")
            lrt_p = results.get("lrt_p_value")
            rows.append({
                "label": "Random slope LRT",
                "value": f"χ²(2) = {_FormattingMixin._format_metric(lrt_stat)}, {_FormattingMixin._format_p_value(lrt_p)}"
            })
        else:
            rows.append({"label": "Random slope LRT", "value": "Not performed"})

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
            if key in results and _FormattingMixin._has_display_value(value):
                display = _FormattingMixin._format_p_value(value) if key.startswith("p_value") else _FormattingMixin._format_metric(value)
                rows.append({"label": label, "value": display})

        pseudo_r2 = results.get("pseudo_r_squared")
        if pseudo_r2 is not None:
            rows.append({"label": "Pseudo-R² (McFadden)", "value": _FormattingMixin._format_metric(pseudo_r2)})

        phi = results.get("phi")
        if phi is not None:
            phi_f = float(phi)
            if phi_f < 1:
                phi_interp = "High variance relative to mean"
            elif phi_f <= 5:
                phi_interp = "Moderate dispersion"
            else:
                phi_interp = "Low dispersion — precise estimates"
            rows.append({"label": "Dispersion parameter (φ)", "value": f"{_FormattingMixin._format_metric(phi)} — {phi_interp}"})

        for label, key in [
            ("AIC", "aic"),
            ("BIC", "bic"),
            ("N observations", "n_observations"),
        ]:
            value = results.get(key)
            if value is not None:
                rows.append({"label": label, "value": _FormattingMixin._format_metric(value)})

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
            ("Model variant", "model_variant"),
            ("p-value (primary predictor)", "p_value"),
            ("Adjusted p-value", "p_value_fdr"),
        ]:
            value = results.get(key)
            if key in results and _FormattingMixin._has_display_value(value):
                display = _FormattingMixin._format_p_value(value) if key.startswith("p_value") else _FormattingMixin._format_metric(value)
                rows.append({"label": label, "value": display})

        auc = results.get("effect_size")
        if auc is not None:
            rows.append({"label": "AUC (ROC)", "value": _FormattingMixin._format_metric(auc)})

        brier = results.get("brier_score")
        if brier is not None:
            rows.append({"label": "Brier score", "value": _FormattingMixin._format_metric(brier)})

        cal_slope = results.get("calibration_slope")
        if cal_slope is not None:
            rows.append({"label": "Calibration slope", "value": _FormattingMixin._format_metric(cal_slope)})

        cal_intercept = results.get("calibration_intercept")
        if cal_intercept is not None:
            rows.append({"label": "Calibration intercept", "value": _FormattingMixin._format_metric(cal_intercept)})

        pseudo_r2 = results.get("pseudo_r_squared")
        if pseudo_r2 is not None:
            rows.append({"label": "McFadden pseudo-R²", "value": _FormattingMixin._format_metric(pseudo_r2)})

        aic = results.get("aic")
        if aic is not None:
            rows.append({"label": "AIC", "value": _FormattingMixin._format_metric(aic)})

        bic = results.get("bic")
        if bic is not None:
            rows.append({"label": "BIC", "value": _FormattingMixin._format_metric(bic)})

        n = results.get("n_observations")
        if n is not None:
            rows.append({"label": "N observations", "value": _FormattingMixin._format_metric(n)})

        if len(rows) <= 1:
            rows.append({"label": "Status", "value": "No structured statistical summary available."})
        return rows

    @staticmethod
    def _build_factorial_anova_statistical_rows(results: dict) -> list[dict]:
        """ANOVA effects table for TwoWayANOVA, MixedANOVA, and RepeatedMeasuresANOVA.
        Shows all main effects and interactions with F, df, p-value, and η²p."""
        rows = []
        model_type = results.get("model_type", "")

        # Basic identity rows
        for label, key in [
            ("Test", "test"),
            ("Model type", "model_type"),
            ("Transformation", "transformation"),
            ("Post-hoc test", "posthoc_test"),
        ]:
            value = results.get(key)
            if key in results and _FormattingMixin._has_display_value(value):
                rows.append({"label": label, "value": _FormattingMixin._format_metric(value)})

        factors = results.get("factors") or []
        interactions = results.get("interactions") or []

        if factors or interactions:
            rows.append({"label": "── ANOVA Effects Table ──", "value": ""})
            rows.append({"label": "Source", "value": "F (df₁, df₂) | p-value | η²p"})

            for factor in factors:
                name = str(factor.get("factor", ""))
                ftype = factor.get("type", "")
                F = factor.get("F")
                df1 = factor.get("df1")
                df2 = factor.get("df2")
                p = factor.get("p_value")
                eta = factor.get("effect_size")

                label = name
                if ftype == "between":
                    label += " (between-subject)"
                elif ftype == "within":
                    label += " (within-subject)"

                parts = []
                if F is not None:
                    if df1 is not None and df2 is not None:
                        parts.append(f"F({df1:.0f}, {df2:.0f}) = {F:.4f}")
                    else:
                        parts.append(f"F = {F:.4f}")
                if p is not None:
                    parts.append(_FormattingMixin._format_p_value(p))
                if eta is not None:
                    try:
                        parts.append(f"η²p = {float(eta):.4f}")
                    except (TypeError, ValueError):
                        pass

                rows.append({"label": label, "value": " | ".join(parts)})

            for inter in interactions:
                inter_factors = inter.get("factors") or []
                name = " × ".join(str(f) for f in inter_factors) if inter_factors else "Interaction"
                F = inter.get("F")
                df1 = inter.get("df1")
                df2 = inter.get("df2")
                p = inter.get("p_value")
                eta = inter.get("effect_size")

                parts = []
                if F is not None:
                    if df1 is not None and df2 is not None:
                        parts.append(f"F({df1:.0f}, {df2:.0f}) = {F:.4f}")
                    else:
                        parts.append(f"F = {F:.4f}")
                if p is not None:
                    parts.append(_FormattingMixin._format_p_value(p))
                if eta is not None:
                    try:
                        parts.append(f"η²p = {float(eta):.4f}")
                    except (TypeError, ValueError):
                        pass

                rows.append({"label": name + " (interaction)", "value": " | ".join(parts)})
        else:
            # Fallback: single primary effect summary (RM-ANOVA single factor)
            for label, key in [
                ("Statistic", "statistic"),
                ("p-value", "p_value"),
                ("Effect size", "effect_size"),
                ("Effect size type", "effect_size_type"),
                ("Degrees of freedom 1", "df1"),
                ("Degrees of freedom 2", "df2"),
            ]:
                value = results.get(key)
                if key in results and _FormattingMixin._has_display_value(value):
                    if key == "p_value":
                        rows.append({"label": label, "value": _FormattingMixin._format_p_value(value)})
                    else:
                        rows.append({"label": label, "value": _FormattingMixin._format_metric(value)})

        # Primary effect summary line (e.g. "Main effect: Timepoint")
        primary_effect = results.get("primary_effect") or {}
        if isinstance(primary_effect, dict):
            primary_factor = primary_effect.get("factor") or primary_effect.get("source")
            if not primary_factor and factors:
                # fall back: find factor with smallest p-value or the primary one
                primary_factor = factors[0].get("factor")
            p_primary = primary_effect.get("p_value")
            F_primary = primary_effect.get("F")
            df1_p = primary_effect.get("df1")
            df2_p = primary_effect.get("df2")
            if primary_factor and F_primary is not None and p_primary is not None and df1_p is not None and df2_p is not None:
                summary = (
                    f"F({df1_p:.4f}, {df2_p:.4f}) = {F_primary:.4f}, "
                    f"{_FormattingMixin._format_p_value(p_primary)}"
                )
                rows.append({"label": f"Main effect: {primary_factor}", "value": summary})
        else:
            # Try the legacy "main_effect_*" style key
            for key, value in results.items():
                if key.startswith("main_effect:") or key.startswith("main_effect_"):
                    rows.append({"label": key.replace("_", " ").replace(":", ":"), "value": str(value)})

        return rows

    @staticmethod
    def _stat_row_info(label: str) -> str:
        """Tooltip copy for a stat-row label. See :mod:`report_tooltips`."""
        from export.report_tooltips import stat_row_info
        return stat_row_info(label)

    @staticmethod
    def _build_statistical_rows(results: dict) -> list[dict]:
        rows = []
        model_type = results.get("model_type", "")
        statistic_type = results.get("statistic_type", "")

        if model_type == "ANCOVA":
            return _StatRowsMixin._build_ancova_statistical_rows(results)

        if model_type in ("TwoWayANOVA", "MixedANOVA", "RepeatedMeasuresANOVA"):
            return _StatRowsMixin._build_factorial_anova_statistical_rows(results)

        if model_type == "LMM":
            return _StatRowsMixin._build_lmm_statistical_rows(results)

        if model_type == "LogisticRegression":
            return _StatRowsMixin._build_logistic_statistical_rows(results)

        if model_type == "BetaRegression":
            return _StatRowsMixin._build_beta_statistical_rows(results)

        if model_type == "CorrelationMatrix":
            return _StatRowsMixin._build_corr_matrix_statistical_rows(results)

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
            if key in results and _FormattingMixin._has_display_value(value):
                if key.startswith("p_value"):
                    display = _FormattingMixin._format_p_value(value)
                elif key == "confidence_interval":
                    display = _FormattingMixin._format_confidence_interval(value)
                else:
                    display = _FormattingMixin._format_metric(value)
                rows.append({"label": label, "value": display})

        # For Logistic Regression: add Wald z-statistic from odds_ratios table
        if model_type == "LogisticRegression":
            or_table = results.get("odds_ratios") or []
            if or_table and isinstance(or_table[0], dict):
                z_val = or_table[0].get("z_value")
                if z_val is not None:
                    rows.append({"label": "Wald z-statistic (first predictor)", "value": _FormattingMixin._format_metric(z_val)})

        # For Correlation: add sample size, interpretation, and statistic type
        if model_type == "Correlation":
            n_val = results.get("n")
            if n_val is not None:
                rows.append({"label": "Sample size (n)", "value": _FormattingMixin._format_metric(n_val)})
            method = str(results.get("method") or "").capitalize()
            if method:
                rows.append({"label": "Correlation method", "value": method})
            interp = results.get("interpretation")
            if interp:
                rows.append({"label": "Interpretation", "value": str(interp)})

        # For Linear Regression: add transformation labels and coefficient interpretation disclaimer
        if model_type == "LinearRegression":
            xt = results.get("x_transform", "none") or "none"
            yt = results.get("y_transform", "none") or "none"
            if xt != "none" or yt != "none":
                rows.append({
                    "label": "Variable transformations",
                    "value": f"X: {xt} | Y: {yt}",
                })
            coef_interp = results.get("coef_interpretation")
            if coef_interp:
                rows.append({
                    "label": "\u03b2 Interpretation",
                    "value": str(coef_interp),
                })

        # For Brunner-Langer ATS: add statistic type label and RTE table rows
        if model_type == "BrunnerLangerATS":
            rows.append({"label": "Statistic type", "value": "ANOVA-Type Statistic (ATS) — not a standard F-value"})
            rte = results.get("RTE")
            if rte is not None:
                rte_rows = []
                try:
                    if isinstance(rte, pd.DataFrame):
                        rte_rows = rte.to_dict(orient="records")
                except (ValueError, TypeError):
                    logger.error("RTE DataFrame → dict conversion failed", exc_info=True)
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
                        rte_display = _FormattingMixin._format_metric(rte_val)
                        if n_cell is not None:
                            rte_display += f"  (n={int(n_cell)})"
                        rows.append({"label": group_label, "value": rte_display})

        # For Freedman-Lane Permutation: add n_permutations
        if model_type == "FreedmanLanePermutation":
            n_perm = results.get("n_permutations")
            if n_perm is not None:
                rows.append({"label": "Permutations (n)", "value": _FormattingMixin._format_metric(n_perm)})
            stat_type = results.get("StatisticType")
            if stat_type:
                rows.append({"label": "Statistic type", "value": str(stat_type)})

        for factor in results.get("factors", []) or []:
            if not isinstance(factor, dict):
                continue
            factor_name = factor.get("factor", "Factor")
            factor_value = (
                f"F({_FormattingMixin._format_metric(factor.get('df1'))}, "
                f"{_FormattingMixin._format_metric(factor.get('df2'))}) = {_FormattingMixin._format_metric(factor.get('F'))}, "
                f"{_FormattingMixin._format_p_value(factor.get('p_value'))}"
            )
            rows.append({"label": f"Main effect: {factor_name}", "value": factor_value})
        for interaction in results.get("interactions", []) or []:
            if not isinstance(interaction, dict):
                continue
            factors = interaction.get("factors") or ["Interaction"]
            label = " x ".join(map(str, factors))
            value = (
                f"F({_FormattingMixin._format_metric(interaction.get('df1'))}, "
                f"{_FormattingMixin._format_metric(interaction.get('df2'))}) = {_FormattingMixin._format_metric(interaction.get('F'))}, "
                f"{_FormattingMixin._format_p_value(interaction.get('p_value'))}"
            )
            rows.append({"label": f"Interaction: {label}", "value": value})
        if not rows:
            rows.append({"label": "Status", "value": "No structured statistical summary available."})
        return rows

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
            es_type = comp.get("effect_size_type") or results.get("effect_size_type") or ""
            magnitude = _FormattingMixin._effect_size_magnitude(
                comp.get("effect_size"), str(es_type)
            )
            rows.append({
                "pair_id": i,
                "group1": str(comp.get("group1", "")),
                "group2": str(comp.get("group2", "")),
                "comparison": f"{comp.get('group1', 'Group 1')} vs {comp.get('group2', 'Group 2')}",
                "test": str(comp.get("test") or results.get("posthoc_test") or "Pairwise comparison"),
                "statistic": _FormattingMixin._format_metric(comp.get("statistic")),
                "p_value": _FormattingMixin._format_p_value(p_val),
                "p_value_style": _FormattingMixin._p_heat_style(p_val),
                "effect_size": _FormattingMixin._format_metric(comp.get("effect_size")),
                "effect_size_type": str(es_type) if es_type else "",
                "effect_magnitude": magnitude or "",
                "significant": is_sig,
                "stars": stars,
                "p_value_raw": p_numeric,
                "row_class": "is-significant" if is_sig else "is-neutral",
            })
        return rows
