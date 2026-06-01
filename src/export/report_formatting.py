"""Pure, stateless formatting and numeric helpers for the HTML report.

Extracted from ``html_exporter.py`` (Phase 1 of the god-file split). Every
method here is a deterministic ``@staticmethod`` with no report state — they map
values to display strings or summarize numeric sequences. ``HTMLExporter`` mixes
this in, so existing ``HTMLExporter._format_metric(...)`` call sites keep working
unchanged via the MRO.

Internal cross-calls reference ``_FormattingMixin`` directly (not
``HTMLExporter``) to keep this module free of any import back into
``html_exporter`` (no circular import).
"""
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

try:
    from core.logger_config import get_logger
except ImportError:  # pragma: no cover — fallback when logger_config missing
    import logging as _logging

    def get_logger(name):
        return _logging.getLogger(name)


logger = get_logger(__name__)


class _FormattingMixin:
    """Stateless formatting / numeric helpers mixed into ``HTMLExporter``."""

    @staticmethod
    def _normalize_for_json(value: Any):
        if isinstance(value, pd.DataFrame):
            normalized = value.replace({np.nan: None})
            return {
                "columns": [str(col) for col in normalized.columns],
                "data": [
                    [_FormattingMixin._normalize_for_json(cell) for cell in row]
                    for row in normalized.itertuples(index=False, name=None)
                ],
            }
        if isinstance(value, pd.Series):
            return [_FormattingMixin._normalize_for_json(item) for item in value.tolist()]
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return _FormattingMixin._normalize_for_json(value.item())
        if isinstance(value, np.ndarray):
            return [_FormattingMixin._normalize_for_json(item) for item in value.tolist()]
        if isinstance(value, dict):
            return {
                str(key): _FormattingMixin._normalize_for_json(val)
                for key, val in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [_FormattingMixin._normalize_for_json(item) for item in value]
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
        except (TypeError, ValueError) as exc:
            logger.debug("_normalize_for_json: pd.isna rejected %r (%s)", type(value).__name__, exc)
        return str(value)

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
    def _format_metric(value: Any, digits: int = 4) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            return ", ".join(_FormattingMixin._format_metric(item, digits=digits) for item in value)
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
    def _sci_notation(value: float) -> str:
        """Format as e.g. '1.89 × 10⁻⁶' using Unicode superscripts."""
        s = f"{value:.2e}"
        mantissa, exp = s.split("e")
        exp_int = int(exp)
        digits = str(abs(exp_int)).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
        sign = "⁻" if exp_int < 0 else ""
        return f"{mantissa} × 10{sign}{digits}"

    @staticmethod
    def _format_p_value(value: Any) -> str:
        if not isinstance(value, (int, float, np.generic)):
            return "N/A" if value in (None, "", "N/A") else str(value)
        numeric = float(value)
        if math.isnan(numeric):
            return "N/A"
        stars = " ***" if numeric < 0.001 else " **" if numeric < 0.01 else " *" if numeric < 0.05 else " ns"
        if numeric < 0.001:
            if numeric > 0:
                p_str = f"p < 0.001 (p = {_FormattingMixin._sci_notation(numeric)})"
            else:
                p_str = "p < 0.001"
        else:
            p_str = f"p = {numeric:.3f}"
        return f"{p_str}{stars}"

    @staticmethod
    def _format_confidence_interval(value: Any) -> str:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return f"[{_FormattingMixin._format_metric(value[0])}, {_FormattingMixin._format_metric(value[1])}]"
        return _FormattingMixin._format_metric(value)

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
    def _effect_size_magnitude(effect_size: Any, effect_type: str,
                               *, df_star: int | None = None) -> str | None:
        """Cohen-style magnitude label for an effect size.

        Thin delegate to :func:`effect_sizes.classify` — see that module for
        threshold sources (Cohen 1988, Koo & Li 2016, Hosmer-Lemeshow,
        McFadden 1979) and the canonicalization rules.
        """
        from analysis.effect_sizes import classify
        return classify(effect_size, effect_type, df_star=df_star)

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
