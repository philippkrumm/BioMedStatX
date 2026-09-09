"""Wave-3 repair: two SHOULD-FIX findings in the nonparametric paths.

#1  Kruskal-Wallis effect size is computed as (H-k+1)/(n-k) = eta-squared[H]
    (rstatix::kruskal_effsize convention) but was labeled "epsilon_squared".
    True epsilon-squared is H/(n-1). The number is a correct eta-squared[H];
    only the metric NAME was wrong, and it is rendered in the report. Fix:
    relabel to "eta_squared" (the number is left unchanged).

#2  DunnTest bootstrap CI used the unseeded global numpy RNG, so the reported
    median-difference confidence interval drifted run-to-run. Fix: seed a local
    Generator so the CI is reproducible.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from scipy import stats as sp

from statistical_testing.engines.comparison import ComparisonEngine
from analysis.posthoc_core import DunnTest


def _kw_samples():
    rng = np.random.default_rng(1)
    return {
        "A": list(rng.normal(10, 2, 9)),
        "B": list(rng.normal(13, 2, 10)),
        "C": list(rng.normal(16, 2, 11)),
    }


def test_kruskal_wallis_effect_size_named_for_what_it_computes():
    samples = _kw_samples()
    groups = ["A", "B", "C"]
    kw = ComparisonEngine()._run_kruskal_wallis(
        groups=groups, samples=samples, alpha=0.05, results={}
    )
    H = float(sp.kruskal(*[samples[g] for g in groups]).statistic)
    n = sum(len(samples[g]) for g in groups)
    k = len(groups)
    eta2_H = (H - k + 1) / (n - k)   # what the code actually computes
    eps2 = H / (n - 1)               # true epsilon-squared

    # the number is (and stays) eta-squared[H], not epsilon-squared
    assert kw["effect_size"] == pytest.approx(eta2_H, rel=1e-9)
    assert not np.isclose(kw["effect_size"], eps2), "value must remain eta^2[H], not become eps^2"

    # the label must match the computation
    assert kw["effect_size_type"] == "eta_squared", (
        f"KW effect size named {kw['effect_size_type']!r} but computes eta^2[H]"
    )


def test_dunn_bootstrap_ci_is_reproducible_run_to_run():
    samples = {
        "A": list(np.random.default_rng(1).normal(10, 2, 12)),
        "B": list(np.random.default_rng(2).normal(13, 2, 12)),
        "C": list(np.random.default_rng(3).normal(16, 2, 12)),
    }
    groups = ["A", "B", "C"]
    d1 = DunnTest.perform_test(groups, samples, alpha=0.05, n_boot=400)
    d2 = DunnTest.perform_test(groups, samples, alpha=0.05, n_boot=400)

    ci1 = [tuple(c["confidence_interval"]) for c in d1["pairwise_comparisons"]]
    ci2 = [tuple(c["confidence_interval"]) for c in d2["pairwise_comparisons"]]
    assert ci1 == ci2, f"Dunn bootstrap CI not reproducible:\n  run1={ci1}\n  run2={ci2}"
