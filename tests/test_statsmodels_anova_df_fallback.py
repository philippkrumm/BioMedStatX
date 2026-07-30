"""Regression guard for the statsmodels-ANOVA degrees-of-freedom key.

The statsmodels fallback paths of RM / Mixed / Two-Way ANOVA read the df
column out of ``sm.stats.anova_lm()`` output. That column is named ``"df"``,
but eight call sites in ``statisticaltester.py`` were mis-typed as ``"d"``,
raising ``KeyError: 'd'`` the moment the fallback was actually reached. It was
never caught because pingouin is the primary engine and normally handles these
designs — the buggy branch only runs when pingouin is unavailable, which no
existing test forced. (See commit fbdc675 + stat_fix.patch: the same typo was
patched twice before, one site at a time.)

Each test here FORCES the fallback by making ``get_pingouin_module`` raise
``ImportError``, then asserts the ``[statsmodels]`` marker so it cannot pass
silently via the pingouin path — the exact gap that hid the bug.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

import analysis.statisticaltester as st_mod
from analysis.statisticaltester import StatisticalTester


@pytest.fixture
def force_statsmodels_fallback(monkeypatch):
    """Make the pingouin lookup raise ImportError so every ANOVA drops into
    its statsmodels fallback branch (the one that carried the df-key bug)."""
    def _no_pingouin():
        raise ImportError("pingouin forced unavailable for fallback regression test")
    monkeypatch.setattr(st_mod, "get_pingouin_module", _no_pingouin)


def _assert_took_statsmodels_fallback(results):
    # If the monkeypatch failed and pingouin ran, this marker is absent — the
    # test would then be vacuous, so this assertion is load-bearing.
    assert "[statsmodels]" in (results.get("test") or ""), (
        f"fallback not exercised; test={results.get('test')!r}"
    )


def _assert_df_sane(results):
    assert results.get("error") is None, f"unexpected error: {results.get('error')!r}"
    for factor in results.get("factors", []):
        assert isinstance(factor.get("df1"), int) and factor["df1"] >= 1, factor
        assert isinstance(factor.get("df2"), int) and factor["df2"] >= 1, factor
    for inter in results.get("interactions", []):
        assert isinstance(inter.get("df1"), int) and inter["df1"] >= 1, inter
        assert isinstance(inter.get("df2"), int) and inter["df2"] >= 1, inter


def test_mixed_anova_statsmodels_fallback_df(force_statsmodels_fallback):
    rng = np.random.RandomState(7)
    n_subj = 8
    rows = []
    for s in range(n_subj):
        grp = "A" if s < n_subj // 2 else "B"
        base = rng.randn()
        for t in ("T1", "T2", "T3"):
            rows.append({"subject": f"S{s}", "group": grp, "time": t,
                         "dv": base + rng.randn()})
    df = pd.DataFrame(rows)

    results = StatisticalTester._run_mixed_anova(
        df=df, dv="dv", subject="subject", between=["group"], within=["time"], alpha=0.05
    )

    _assert_took_statsmodels_fallback(results)
    _assert_df_sane(results)
    assert isinstance(results.get("df1"), int) and results["df1"] >= 1


def test_repeated_measures_anova_statsmodels_fallback_df(force_statsmodels_fallback):
    rng = np.random.RandomState(11)
    rows = []
    for s in range(6):
        base = rng.randn()
        for t in ("T1", "T2", "T3"):
            rows.append({"subject": f"S{s}", "time": t, "dv": base + rng.randn()})
    df = pd.DataFrame(rows)

    results = StatisticalTester._run_repeated_measures_anova(
        df=df, dv="dv", subject="subject", within=["time"], alpha=0.05
    )

    _assert_took_statsmodels_fallback(results)
    _assert_df_sane(results)
    assert isinstance(results.get("df1"), int) and results["df1"] >= 1


def test_two_way_anova_statsmodels_fallback_df(force_statsmodels_fallback):
    rng = np.random.RandomState(13)
    rows = []
    for fa in ("X", "Y"):
        for fb in ("P", "Q"):
            for _ in range(4):
                rows.append({"fa": fa, "fb": fb, "dv": rng.randn()})
    df = pd.DataFrame(rows)

    results = StatisticalTester._run_two_way_anova(
        df=df, dv="dv", between=["fa", "fb"], alpha=0.05
    )

    _assert_took_statsmodels_fallback(results)
    _assert_df_sane(results)
    assert isinstance(results.get("df1"), int) and results["df1"] >= 1
