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

    return violations
