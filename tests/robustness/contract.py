"""Robustness contract: what every result dict must satisfy regardless of input.

A graceful outcome is either a valid result or a clean labeled block — never a
crash and never a NaN/Inf masquerading as a real p-value.
"""
from __future__ import annotations

import math
from typing import Any, Dict


def assert_graceful(result: Dict[str, Any], expectation: str) -> None:
    assert isinstance(result, dict), f"result is not a dict: {type(result)!r}"
    assert result.get("test") is not None, "result has no 'test' label"

    p = result.get("p_value")
    assert p is None or (isinstance(p, (int, float)) and math.isfinite(p)), (
        f"p_value must be None or finite, got {p!r}"
    )

    stat = result.get("statistic")
    assert stat is None or (isinstance(stat, (int, float)) and math.isfinite(stat)), (
        f"statistic must be None or finite, got {stat!r}"
    )

    if expectation == "blocked":
        assert result.get("blocked") is True or result.get("error"), (
            f"expected a block/error, got test={result.get('test')!r} p={p!r}"
        )
    elif expectation == "ok":
        assert not result.get("blocked"), (
            f"expected a runnable result, but it was blocked: {result.get('block_reason')!r}"
        )
    else:  # pragma: no cover - guards typos in the catalog
        raise ValueError(f"unknown expectation {expectation!r}")
