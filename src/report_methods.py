"""Narrative "methods" content for the report — extracted from html_exporter.

This module answers the *what was done* question: it builds the decision-path
breadcrumbs and the methods-section paragraph that explains the analysis to a
reader. It is intentionally free of Jinja/HTML/template concerns — the
return values are plain Python (list[dict] / str) and the caller is
responsible for rendering them.

Formatting helpers (p-value, metric, summary note) are injected by the caller
so this module does not depend on ``HTMLExporter``. That keeps the
dependency arrow one-way: html_exporter → report_methods, never back.
"""
from __future__ import annotations

from typing import Any, Callable, Optional


# Type aliases for the injected formatters — keep signatures small so callers
# can pass HTMLExporter._format_p_value etc. directly.
PValueFormatter = Callable[[Any], str]
MetricFormatter = Callable[..., str]
SummaryNoteBuilder = Callable[[dict, str, Any], str]


_ATS_EFFECT_NOTE = (
    "\n\nEffect-size note: ATS (ANOVA-Type Statistic) evaluates effects via "
    "Relative Treatment Effects (RTE). RTE values range from 0 to 1, where "
    "0.5 indicates the global null effect. No standardized Cohen-style "
    "magnitude thresholds apply to rank-based longitudinal designs; the "
    "RTE table is the appropriate effect metric."
)

# Narrow no-break spaces ( ) tighten the gaps around < and ≥ so the
# inequalities render as one visual unit in the HTML report; en-spaces ( )
# separate the bands. Preserved verbatim from the original copy.
_CORR_R_INTERPRETATION = (
    "Effect size interpretation (Cohen 1988 conventions for r): "
    "negligible < 0.1, small ≥ 0.1,"
    " medium ≥ 0.3, large ≥ 0.5 (|ρ| or |r|)."
)


def build_decision_path_model(
    results: dict,
    *,
    format_p_value: PValueFormatter,
    build_summary_note: SummaryNoteBuilder,
) -> list[dict]:
    """Return ordered breadcrumb steps describing the analysis decision path.

    Each step is a dict with keys: ``title``, ``detail``, ``active``.
    ``active=False`` indicates the step was skipped (e.g. no transformation
    applied) but is kept in the trail for transparency.
    """
    test_info = results.get("test_info", {}) or {}
    steps: list[dict] = [{
        "title": "Data screening",
        "detail": "BioMedStatX evaluated assumptions and available structure before selecting the analysis path.",
        "active": True,
    }]

    pre = test_info.get("pre_transformation", {}) if isinstance(test_info, dict) else {}
    if pre:
        detail_parts: list[str] = []
        residuals = pre.get("residuals_normality", {})
        variance = pre.get("variance", {})
        if residuals.get("p_value") is not None:
            detail_parts.append(f"Residual normality {format_p_value(residuals.get('p_value'))}")
        if variance.get("p_value") is not None:
            detail_parts.append(f"Variance homogeneity {format_p_value(variance.get('p_value'))}")
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
            "detail": str(
                results.get("posthoc_test")
                or f"{len(results.get('pairwise_comparisons') or [])} pairwise comparisons"
            ),
            "active": True,
        })

    steps.append({
        "title": "Inference",
        "detail": build_summary_note(
            results,
            str(results.get("final_test_label") or results.get("test") or "Analysis"),
            results.get("p_value"),
        ),
        "active": True,
    })
    return steps


def build_methods_text(
    results: dict,
    analysis_log: Any,
    *,
    format_metric: MetricFormatter,
) -> str:
    """Compose the Methods-section paragraph.

    Precedence:
        1. ``results["methodology_trace"].to_methods_paragraph()`` — structured trace.
        2. ``analysis_log`` (list joined w/ newlines, or a non-empty string).
        3. Fallback summary built from key result fields.

    The ATS effect-size note is appended whenever ``model_type ==
    "BrunnerLangerATS"`` to satisfy peer-review integrity (no Cohen-style
    heuristic for rank-based longitudinal designs).
    """
    ats_note = _ATS_EFFECT_NOTE if results.get("model_type") == "BrunnerLangerATS" else ""

    trace = results.get("methodology_trace")
    if trace is not None and hasattr(trace, "to_methods_paragraph"):
        return trace.to_methods_paragraph() + ats_note
    if isinstance(analysis_log, list):
        return "\n".join(str(line) for line in analysis_log) + ats_note
    if isinstance(analysis_log, str) and analysis_log.strip():
        return analysis_log + ats_note

    lines = [
        f"Test performed: {results.get('final_test_label') or results.get('test') or 'Not specified'}",
        f"Alpha level: {format_metric(results.get('alpha'), digits=3)}",
        f"Transformation: {results.get('transformation') or 'None'}",
    ]
    if results.get("posthoc_test"):
        lines.append(f"Post-hoc procedure: {results.get('posthoc_test')}")
    if results.get("effect_size_type"):
        lines.append(f"Effect size metric: {results.get('effect_size_type')}")

    # Correlation: append Cohen-r conventions + transformation shift note.
    if results.get("model_type") == "Correlation":
        es_type = str(results.get("effect_size_type") or "").lower().strip()
        if (
            any(k in es_type for k in ("rho", "pearson", "spearman", "correlation"))
            or es_type in ("r", "ρ")
        ):
            lines.append(_CORR_R_INTERPRETATION)

        x_shift = results.get("x_transform_shift") or 0.0
        y_shift = results.get("y_transform_shift") or 0.0
        x_tr = results.get("x_transform") or "none"
        y_tr = results.get("y_transform") or "none"
        for axis, tr_name, shift in (("X", x_tr, x_shift), ("Y", y_tr, y_shift)):
            if tr_name != "none" and shift != 0.0:
                # Reconstruct the raw minimum that triggered the shift, so a
                # peer reviewer can reproduce y = f(x + c) exactly:
                #   log10 / boxcox: shift = -min + 1.0 → min = 1.0 - shift
                #   sqrt:           shift = -min       → min = -shift
                min_raw = (1.0 - shift) if tr_name in ("log10", "boxcox") else (-shift)
                lines.append(
                    f"Note ({axis}-axis): a constant c={shift:.4f} was automatically added "
                    f"to all {axis.lower()}-values prior to {tr_name} transformation to satisfy "
                    f"the positivity requirement (minimum raw value was {min_raw:.4f}). "
                    f"This constant was determined from the data, not set by the researcher."
                )

    return "\n".join(lines) + ats_note
