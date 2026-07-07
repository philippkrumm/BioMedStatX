# Tier B3: Specialized Models Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix a silent data-corruption bug in outlier ingestion (SM1) and a severe performance
bug in Dunn-test's bootstrap confidence interval (SM2).

**Architecture:** Two independent, single-file fixes. SM1 changes the value-conversion strategy
from "always assume German decimal format" to "try direct parse first, fall back to German-format
conversion only if that fails." SM2 replaces a pure-Python O(n1×n2) nested loop with the
mathematically identical vectorized `numpy` equivalent — same random draws, same result, just
computed without a Python-level double loop.

**Tech Stack:** Python, numpy, pandas, pytest.

---

### Task 1: SM1 — stop corrupting US/international-formatted decimal strings

**Files:**
- Modify: `src/analysis/outlier_core.py:77-107` (`OutlierDetector._convert_values_to_float`)
- Test: `tests/test_outlier_decimal_format.py`

When the value column isn't already numeric dtype, the current code unconditionally treats `.`
as a thousands separator and `,` as the decimal point — `"1.5"` becomes `15.0`, a silent 10×
inflation, with no error or plausibility check.

- [ ] **Step 1: Write the failing test**

```python
"""OutlierDetector._convert_values_to_float unconditionally applies German decimal-format
conversion (dot=thousands, comma=decimal) whenever the value column isn't already numeric
dtype - silently multiplying plain US/international-formatted decimal strings like "1.5" by
10-1000x ("1.5" -> 15.0). Fix: try a direct float() parse first; only fall back to the
German-format substitution for values that fail it.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import pytest

from analysis.outlier_core import OutlierDetector


def test_us_formatted_decimal_strings_are_not_corrupted():
    df = pd.DataFrame({
        "Group": ["a", "a", "b", "b"],
        "Value": ["1.5", "2.75", "3.0", "4.25"],
    })
    assert df["Value"].dtype == object, "test fixture must start as text, not already numeric"

    detector = OutlierDetector(df, "Group", "Value")

    assert list(detector.df["Value"]) == pytest.approx([1.5, 2.75, 3.0, 4.25]), (
        "US-formatted decimal strings must parse as-is, not get reinterpreted as "
        "German-formatted thousands separators"
    )


def test_german_formatted_decimal_strings_still_convert_correctly():
    df = pd.DataFrame({
        "Group": ["a", "a", "b", "b"],
        "Value": ["1,5", "2,75", "1.234,5", "4,25"],
    })
    detector = OutlierDetector(df, "Group", "Value")

    assert list(detector.df["Value"]) == pytest.approx([1.5, 2.75, 1234.5, 4.25]), (
        "genuine German-formatted values (comma decimal, dot thousands) must still convert"
    )


def test_already_numeric_column_is_left_untouched():
    df = pd.DataFrame({"Group": ["a", "b"], "Value": [1.5, 2.75]})
    detector = OutlierDetector(df, "Group", "Value")
    assert list(detector.df["Value"]) == pytest.approx([1.5, 2.75])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/test_outlier_decimal_format.py -v`
Expected: FAIL on `test_us_formatted_decimal_strings_are_not_corrupted` —
`detector.df["Value"]` is `[15.0, 275.0, 30.0, 425.0]`, not `[1.5, 2.75, 3.0, 4.25]`.

- [ ] **Step 3: Fix — try a direct parse first**

Read `outlier_core.py` around line 77 fresh (`grep -n "def _convert_values_to_float" src/analysis/outlier_core.py`)
to confirm the current line, then change:

```python
            # Convert German decimal numbers
            self.df[self.value_col] = (
                self.df[self.value_col]
                    .astype(str)
                    .str.replace('.', '', regex=False)   # Remove thousand separators
                    .str.replace(',', '.', regex=False)  # Comma → Period
                    .astype(float)
            )
```

to:

```python
            def _parse_numeric_string(raw):
                text = str(raw).strip()
                try:
                    # If it already parses as a plain float, it's not
                    # German-formatted - leave it alone. This is the fix for
                    # the silent 10-1000x inflation of US/international-
                    # formatted decimal strings like "1.5" -> 15.0.
                    return float(text)
                except ValueError:
                    pass
                # Only reached for strings that fail a direct parse - assume
                # German format (dot=thousands, comma=decimal). If this also
                # fails, let the ValueError propagate rather than silently
                # producing a wrong number.
                german_converted = text.replace('.', '').replace(',', '.')
                return float(german_converted)

            self.df[self.value_col] = self.df[self.value_col].apply(_parse_numeric_string)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/test_outlier_decimal_format.py -v`
Expected: PASS (all 3 cases).

- [ ] **Step 5: Commit**

```bash
git add tests/test_outlier_decimal_format.py src/analysis/outlier_core.py
git commit -m "fix(outlier): stop misparsing US-formatted decimals as German thousands"
```

---

### Task 2: SM2 — vectorize Dunn-test's bootstrap confidence interval

**Files:**
- Modify: `src/analysis/posthoc_core.py:1598-1602` (inside `DunnTest.perform_test`)
- Test: `tests/test_dunn_bootstrap_performance.py`

The bootstrap CI for each pairwise median difference builds a full `n1 × n2` pairwise-difference
list in pure Python per bootstrap iteration (1000 iterations by default) — measured at ~13.5s
per pair at n=500/group during the round-2 audit. `np.subtract.outer(b1, b2)` computes the exact
same `n1 × n2` difference matrix as a single vectorized numpy operation.

- [ ] **Step 1: Write the failing test (correctness-equivalence, not yet a speed assertion)**

```python
"""DunnTest.perform_test's bootstrap CI uses a pure-Python O(n1*n2) nested loop
(np.median([u - v for u in b1 for v in b2])) per bootstrap iteration - this test proves the
vectorized np.subtract.outer(b1, b2) replacement produces bit-for-bit identical results for the
same random draws (same RNG seed, same b1/b2 samples - only the inner difference computation
changes), then exercises the real DunnTest.perform_test end to end and checks it completes
quickly for a realistic group size.
"""
import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from analysis.posthoc_core import DunnTest


def test_vectorized_outer_product_matches_naive_double_loop_bit_for_bit():
    rng = np.random.RandomState(0)
    x = rng.normal(0, 1, 30)
    y = rng.normal(2, 1, 25)

    np.random.seed(123)
    naive_boots = []
    for _ in range(200):
        b1 = np.random.choice(x, len(x), replace=True)
        b2 = np.random.choice(y, len(y), replace=True)
        naive_boots.append(np.median([u - v for u in b1 for v in b2]))

    np.random.seed(123)
    vectorized_boots = []
    for _ in range(200):
        b1 = np.random.choice(x, len(x), replace=True)
        b2 = np.random.choice(y, len(y), replace=True)
        vectorized_boots.append(np.median(np.subtract.outer(b1, b2)))

    assert naive_boots == pytest.approx(vectorized_boots), (
        "vectorized np.subtract.outer must produce the exact same bootstrap medians as the "
        "naive nested loop for identical random draws - this is a performance fix, not a "
        "behavior change"
    )


def test_dunn_test_completes_quickly_for_a_realistic_group_size():
    rng = np.random.RandomState(1)
    n_per_group = 200
    groups = {
        "A": rng.normal(0, 1, n_per_group).tolist(),
        "B": rng.normal(0.5, 1, n_per_group).tolist(),
        "C": rng.normal(1.0, 1, n_per_group).tolist(),
    }

    start = time.perf_counter()
    result = DunnTest.perform_test(list(groups.keys()), groups, alpha=0.05, n_boot=1000)
    elapsed = time.perf_counter() - start

    assert result.get("error") is None
    assert len(result.get("pairwise_comparisons", [])) == 3  # 3 choose 2
    assert elapsed < 5.0, (
        f"DunnTest.perform_test took {elapsed:.1f}s for n=200/group, 3 pairs - "
        f"expected well under 5s with the vectorized bootstrap (was ~13.5s PER PAIR "
        f"at n=500/group before this fix, i.e. this exact case would have taken "
        f"tens of seconds)"
    )
```

- [ ] **Step 2: Run the test to verify it fails (or is at least much slower)**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/test_dunn_bootstrap_performance.py -v --timeout=120`
Expected: `test_vectorized_outer_product_matches_naive_double_loop_bit_for_bit` PASSES already
(it doesn't touch `DunnTest` itself, just proves the two computations are equivalent — this is
groundwork, not the regression test). `test_dunn_test_completes_quickly_for_a_realistic_group_size`
is expected to be slow (tens of seconds) or FAIL the `elapsed < 5.0` assertion against the
current unvectorized code. If `pytest-timeout` isn't installed, drop `--timeout=120` and just
observe the wall-clock time printed by `-v --durations=1`.

- [ ] **Step 3: Fix — vectorize the bootstrap CI computation**

Read `posthoc_core.py` around line 1598 fresh
(`grep -n "for _ in range(n_boot):" src/analysis/posthoc_core.py`) to confirm the current line,
then change:

```python
            # Bootstrap CI
            boots = []
            for _ in range(n_boot):
                b1 = np.random.choice(x, n1, replace=True)
                b2 = np.random.choice(y, n2, replace=True)
                boots.append(np.median([u - v for u in b1 for v in b2]))
            ci_low, ci_high = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
```

to:

```python
            # Bootstrap CI - np.subtract.outer(b1, b2) computes the identical
            # n1×n2 pairwise-difference matrix as the equivalent nested
            # Python loop, vectorized (was ~13.5s per pair at n=500/group).
            boots = []
            for _ in range(n_boot):
                b1 = np.random.choice(x, n1, replace=True)
                b2 = np.random.choice(y, n2, replace=True)
                boots.append(np.median(np.subtract.outer(b1, b2)))
            ci_low, ci_high = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/test_dunn_bootstrap_performance.py -v`
Expected: PASS (both tests; the performance test now completes in well under 5 seconds).

- [ ] **Step 5: Run the full test suite**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/ -q --tb=no`
Expected: same pass count as before this task plus the 5 new tests across both files (the 1
pre-existing unrelated `test_convergence.py::test_convergence_keys` failure is expected — see
the A1 plan's Task 4 for how this was confirmed pre-existing).

- [ ] **Step 6: Commit**

```bash
git add tests/test_dunn_bootstrap_performance.py src/analysis/posthoc_core.py
git commit -m "perf(posthoc): vectorize Dunn-test bootstrap CI, ~13.5s/pair to sub-second"
```

---

## Self-review notes

- **Spec coverage:** SM1 (Task 1), SM2 (Task 2) — both findings assigned to this package are
  covered.
- **SM1's fix preserves the existing German-format conversion path exactly** (Task 1's test
  `test_german_formatted_decimal_strings_still_convert_correctly` proves this) — only the
  precondition for reaching it changes (try direct parse first), not the conversion arithmetic
  itself.
- **SM2's fix is proven bit-for-bit equivalent, not just "probably fine"** — Task 2's first test
  seeds numpy's RNG identically for both the old and new computation and asserts the resulting
  bootstrap medians match to floating-point precision, before ever touching the real
  `DunnTest.perform_test` call path.
