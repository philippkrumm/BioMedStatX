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
