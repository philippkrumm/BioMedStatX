"""Association / regression tables and payload helpers for the HTML report.

Extracted from ``html_exporter.py`` (Phase 4 of the god-file split): odds-ratio
and beta-coefficient tables, the association data payload, and a simple OLS fit
with confidence band. Stateless ``@staticmethod`` helpers mixed into
``HTMLExporter``; call sites unchanged via the MRO.
"""

import numpy as np
from scipy import stats

from export.report_formatting import _FormattingMixin


class _AssociationMixin:
    """Association / regression helpers mixed into ``HTMLExporter``."""

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
            or_display = _FormattingMixin._format_metric(row.get("odds_ratio"))
            if is_sig:
                or_display = f"<strong>{or_display}</strong>"
            p_style = "color:var(--success)" if is_sig else "color:var(--muted)"
            rows_html += (
                f"<tr>"
                f"<td>{row.get('parameter', '')}</td>"
                f"<td class='num-cell'>{or_display}</td>"
                f"<td class='num-cell'>{_FormattingMixin._format_metric(row.get('ci_lower'))}</td>"
                f"<td class='num-cell'>{_FormattingMixin._format_metric(row.get('ci_upper'))}</td>"
                f"<td class='num-cell'>{_FormattingMixin._format_metric(row.get('z_value'))}</td>"
                f"<td class='num-cell' style='{p_style}'>{_FormattingMixin._format_p_value(p_val)}</td>"
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
        subtitle = "Exponentiated coefficients with 95% confidence intervals. Bold OR = p &lt; 0.05."
        # Surface the inference method when it deviates from the standard Wald
        # logit (e.g. Firth penalized likelihood) so the reader knows how the
        # CIs and p-values were derived.
        ci_method = next((r.get("ci_method") for r in or_table if r.get("ci_method")), None)
        p_method = next((r.get("p_value_method") for r in or_table if r.get("p_value_method")), None)
        if ci_method or p_method:
            parts = []
            if ci_method:
                parts.append(f"CI: {ci_method}")
            if p_method:
                parts.append(f"p-value: {p_method}")
            subtitle += " · " + "; ".join(parts) + "."
        return {
            "title": "Odds Ratios",
            "subtitle": subtitle,
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
            coef_display = _FormattingMixin._format_metric(row.get("coefficient"))
            if is_sig:
                coef_display = f"<strong>{coef_display}</strong>"
            p_style = "color:var(--success)" if is_sig else "color:var(--muted)"
            rows_html += (
                f"<tr>"
                f"<td>{row.get('parameter', '')}</td>"
                f"<td class='num-cell'>{coef_display}</td>"
                f"<td class='num-cell'>{_FormattingMixin._format_metric(row.get('std_err'))}</td>"
                f"<td class='num-cell'>{_FormattingMixin._format_metric(row.get('z_value'))}</td>"
                f"<td class='num-cell' style='{p_style}'>{_FormattingMixin._format_p_value(p_val)}</td>"
                f"<td class='num-cell'>{_FormattingMixin._format_metric(row.get('ci_lower'))}</td>"
                f"<td class='num-cell'>{_FormattingMixin._format_metric(row.get('ci_upper'))}</td>"
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
            fit_x = _FormattingMixin._coerce_numeric_sequence(fit.get("x") or [])
            fit_y = _FormattingMixin._coerce_numeric_sequence(fit.get("y") or [])
            fit_ci_lower = _FormattingMixin._coerce_numeric_sequence(fit.get("ci_lower") or [])
            fit_ci_upper = _FormattingMixin._coerce_numeric_sequence(fit.get("ci_upper") or [])

        if not x_values or not y_values:
            association_points = results.get("association_points")
            x_values, y_values = _pair_points(association_points)

        if not x_values or not y_values:
            raw_data = results.get("raw_data") or {}
            if isinstance(raw_data, dict) and len(raw_data) >= 2:
                names = list(raw_data.keys())[:2]
                x_candidate = _FormattingMixin._coerce_numeric_sequence(raw_data.get(names[0], []))
                y_candidate = _FormattingMixin._coerce_numeric_sequence(raw_data.get(names[1], []))
                paired_length = min(len(x_candidate), len(y_candidate))
                x_values = x_candidate[:paired_length]
                y_values = y_candidate[:paired_length]
                x_label = str(names[0])
                y_label = str(names[1])

        if not x_values or not y_values:
            return None

        method = str(results.get("method") or "").lower()
        if not (fit_x and fit_y and len(fit_x) == len(fit_y)) and method != "spearman":
            fit_data = _AssociationMixin._simple_linear_fit_with_ci(x_values, y_values, alpha=float(results.get("alpha", 0.05)))
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
