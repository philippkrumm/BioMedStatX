"""Data-driven catalog of pathological datasets for the robustness suites.

Each ``EdgeCase`` carries a group->values mapping plus the expected outcome
(``"ok"`` or ``"blocked"``). Values may include NaN, +/-Inf, huge magnitudes, or
non-numeric junk (text, whitespace) on purpose — the pipeline must degrade
gracefully (clean block or valid result), never crash or return a silently-wrong
number.

Adding a new edge case = one entry here; both the core-level and analyze()-level
suites pick it up automatically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd

NaN = float("nan")
Inf = float("inf")


@dataclass(frozen=True)
class EdgeCase:
    name: str
    samples: Dict[str, List[Any]]
    expectation: str  # "ok" | "blocked"
    dependent: bool = False
    # Optional: the block code we expect (only asserted when set and blocked).
    expected_code: str | None = None


CATALOG: List[EdgeCase] = [
    # --- benign / should run -------------------------------------------------
    EdgeCase("nan_scattered", {"a": [1, 2, NaN, 4, 5], "b": [2, 3, 4, NaN, 6, 7]}, "ok"),
    EdgeCase("text_in_numeric", {"a": [1.2, 3.4, "foo", 5.6, 7.8], "b": [2, 3, 4, 5, 6]}, "ok"),
    EdgeCase("whitespace_only_strings", {"a": [1.2, 3.4, "   ", 5.6, 7.0], "b": [1, 2, 3, 4, 5]}, "ok"),
    EdgeCase("two_identical_groups", {"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5]}, "ok"),
    EdgeCase("unbalanced_extreme", {"a": [1, 2, 3], "b": list(range(1, 41))}, "ok"),
    EdgeCase("huge_outlier", {"a": [1, 2, 3, 4, 1000], "b": [2, 3, 4, 5, 6]}, "ok"),

    # --- must block ----------------------------------------------------------
    EdgeCase("nan_whole_column", {"a": [NaN, NaN, NaN, NaN, NaN], "b": [1, 2, 3, 4, 5]}, "blocked", expected_code="EMPTY_GROUP"),
    EdgeCase("nan_entire_group", {"a": [1, 2, 3, 4, 5], "b": [NaN, NaN, NaN, NaN]}, "blocked", expected_code="EMPTY_GROUP"),
    EdgeCase("zero_variance_one_group", {"a": [5.0] * 6, "b": [1, 2, 3, 4, 5, 6]}, "blocked", expected_code="VAR_ZERO"),
    EdgeCase("zero_variance_all", {"a": [5.0] * 5, "b": [7.0] * 5}, "blocked", expected_code="VAR_ZERO"),
    EdgeCase("near_zero_variance", {"a": [100.0, 100.0 + 1e-14, 100.0 - 1e-14], "b": [1, 2, 3]}, "blocked", expected_code="VAR_ZERO"),
    EdgeCase("n1_single_value", {"a": [5], "b": [1, 2, 3, 4]}, "blocked", expected_code="N_BELOW_MIN"),
    EdgeCase("n2_below_block", {"a": [5, 6], "b": [1, 2, 3, 4]}, "blocked", expected_code="N_BELOW_MIN"),
    EdgeCase("inf_values", {"a": [1, 2, Inf, 4], "b": [1, 2, 3, 4]}, "blocked", expected_code="INF_VALUES"),
    EdgeCase("computational_overflow_risk", {"a": [1e160, 1.5e160, 2e160], "b": [1, 2, 3]}, "blocked", expected_code="NUM_OVERFLOW"),
    EdgeCase("single_group", {"a": [1, 2, 3, 4, 5]}, "blocked", expected_code="TOO_FEW_GROUPS"),
    EdgeCase("constant_all_groups", {"a": [5.0] * 5, "b": [8.0] * 5, "c": [2.0] * 5}, "blocked", expected_code="VAR_ZERO"),

    # --- dependent / paired --------------------------------------------------
    EdgeCase("paired_identical_groups", {"a": [1, 5, 20, 8, 3], "b": [2, 6, 21, 9, 4]}, "blocked", dependent=True, expected_code="VAR_DIFF_ZERO"),
    EdgeCase("paired_nonconsecutive_constant_diff", {"a": [1, 5, 20], "b": [3, 10, 2], "c": [5, 9, 24]}, "blocked", dependent=True, expected_code="VAR_DIFF_ZERO"),
]


def coerce_samples(samples: Dict[str, List[Any]]) -> Dict[str, list]:
    """Mimic the upstream numeric coercion analysis_core applies at sample-build:
    text / whitespace -> NaN -> dropped; +/-Inf preserved so the Inf guard sees
    it. Used by the core-level suite, which receives already-extracted samples."""
    out: Dict[str, list] = {}
    for group, values in samples.items():
        series = pd.to_numeric(pd.Series(list(values), dtype="object"), errors="coerce")
        arr = series.to_numpy(dtype=float)
        # Keep Inf (guard must see it), drop NaN (missing/junk).
        out[group] = [float(v) for v in arr if not np.isnan(v)]
    return out


def to_long_df(samples: Dict[str, List[Any]], group_col: str = "Grp", value_col: str = "Val") -> pd.DataFrame:
    """Long-format Group/Value frame for the analyze() suite. Raw values are kept
    (strings/Inf included) so the pipeline's own coercion is exercised."""
    rows = []
    for group, values in samples.items():
        for v in values:
            rows.append({group_col: group, value_col: v})
    return pd.DataFrame(rows)
