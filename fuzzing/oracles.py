"""Statistical fuzzing oracles.

A try/except only catches hard crashes. These oracles catch *silent* statistical
failures — a result that looks fine structurally but is mathematically impossible
or internally inconsistent. A violation means the pipeline produced a wrong-but-
unflagged answer, which is worse than a crash.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _check_df_positive(result: dict, violations: List[str]) -> None:
    """Degrees of freedom must be positive when present."""
    for key in ("df", "df1", "df2", "df_between", "df_within", "df_residual"):
        val = result.get(key)
        if _is_number(val) and math.isfinite(val) and val <= 0:
            violations.append(f"df '{key}' = {val} is not positive")

    for i, factor in enumerate(result.get("factors") or []):
        if not isinstance(factor, dict):
            continue
        for key in ("df", "df1", "df2"):
            val = factor.get(key)
            if _is_number(val) and math.isfinite(val) and val <= 0:
                violations.append(f"factors[{i}].{key} = {val} is not positive")


def _check_f_p_consistency(result: dict, violations: List[str]) -> None:
    """If F >> 1 (e.g. F > 100), p must be very small (< 0.001); if F < 1, p should be > alpha.

    This is a soft heuristic — only flag extreme inconsistencies to catch sign
    errors or wrong distribution lookups.
    """
    blocked = result.get("blocked") is True
    if blocked:
        return

    stat = result.get("statistic")
    p = result.get("p_value")
    if not (_is_number(stat) and _is_number(p)):
        return
    if not (math.isfinite(stat) and math.isfinite(p)):
        return

    # Large F / chi-square → p must be tiny
    if stat > 500 and p > 0.05:
        violations.append(
            f"F/chi2={stat:.1f} is huge but p_value={p:.4f} is not small — likely wrong tail"
        )
    # F < 0 is impossible (sum-of-squares ratio)
    if stat < 0 and result.get("test") and any(
        kw in str(result.get("test", "")).lower()
        for kw in ("anova", "f-test", "welch")
    ):
        violations.append(f"F-statistic={stat} is negative — impossible for ANOVA")


def _check_effect_size_bounds(result: dict, violations: List[str]) -> None:
    """η², partial η², Cohen's d, r, R² must be in physically valid ranges."""
    es = result.get("effect_size")
    es_type = str(result.get("effect_size_type", "")).lower()
    if not _is_number(es) or not math.isfinite(es):
        return

    if "eta" in es_type or "r_squared" in es_type or "omega" in es_type:
        if not (0.0 <= es <= 1.0):
            violations.append(f"effect_size ({es_type}) = {es:.4f} outside [0, 1]")
    elif "cohen" in es_type or es_type == "d":
        if abs(es) > 50:
            violations.append(f"Cohen's d = {es:.2f} is implausibly large (|d| > 50)")
    elif "r" == es_type or "pearson" in es_type or "spearman" in es_type:
        if not (-1.0 <= es <= 1.0):
            violations.append(f"correlation effect_size = {es:.4f} outside [-1, 1]")


def check_result(result: Any) -> List[str]:
    """Return a list of contract violations (empty == clean)."""
    violations: List[str] = []

    if not isinstance(result, dict):
        return [f"result is not a dict: {type(result).__name__}"]

    blocked = result.get("blocked") is True
    has_error = bool(result.get("error"))

    # A graceful result must self-identify: either a test label, a block, or an error.
    if result.get("test") is None and not blocked and not has_error:
        violations.append("no 'test' label and not marked blocked/error")

    p = result.get("p_value")
    if _is_number(p):
        if not math.isfinite(p):
            if not blocked:
                violations.append(f"p_value is non-finite ({p}) but result not blocked")
        elif not (0.0 <= p <= 1.0):
            violations.append(f"p_value {p} outside [0, 1]")
    elif p is not None:
        violations.append(f"p_value is neither number nor None: {p!r}")

    stat = result.get("statistic")
    if _is_number(stat) and not math.isfinite(stat) and not blocked:
        violations.append(f"statistic is non-finite ({stat}) but result not blocked")

    # Pairwise comparison p-values must also be physical.
    for i, comp in enumerate(result.get("pairwise_comparisons") or []):
        if not isinstance(comp, dict):
            violations.append(f"pairwise_comparisons[{i}] is not a dict")
            continue
        cp = comp.get("p_value")
        if _is_number(cp) and math.isfinite(cp) and not (0.0 <= cp <= 1.0):
            violations.append(f"pairwise_comparisons[{i}] p_value {cp} outside [0, 1]")

    # Multi-factor results: check each factor
    for i, factor in enumerate(result.get("factors") or []):
        if not isinstance(factor, dict):
            violations.append(f"factors[{i}] is not a dict")
            continue
        fp = factor.get("p_value")
        if _is_number(fp) and math.isfinite(fp) and not (0.0 <= fp <= 1.0):
            violations.append(f"factors[{i}] p_value {fp} outside [0, 1]")

    # New structural + plausibility checks
    _check_df_positive(result, violations)
    _check_f_p_consistency(result, violations)
    _check_effect_size_bounds(result, violations)

    return violations
