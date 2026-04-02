"""
conftest.py — Shared fixtures for BioMedStatX decision-tree test suite.

Provides:
  - make_excel_fixture: generates synthetic Excel files for each design
  - patch_dialogs: session-scoped autouse fixture that silences all Qt dialogs
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

# NumPy 2 removed np.unicode_; older scientific deps (xarray/pingouin stacks)
# may still import it. Keep tests stable by restoring a compatible alias.
if not hasattr(np, "unicode_"):
    np.unicode_ = str

# Ensure PyQt uses a headless backend during automated test runs.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
for _p in [str(ROOT), str(SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Session-level: silence all Qt dialogs so tests run headless
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True, scope="session")
def patch_dialogs():
    """
    Replace every UIDialogManager dialog with a headless default:
      - posthoc (parametric)  → "tukey"
      - posthoc (nonparam)    → "dunn"
      - control group         → first group
      - custom pairs          → empty list (skip)
    Also suppresses QApplication creation attempts.
    """
    mock_manager = MagicMock()
    mock_manager.select_posthoc_test_dialog.return_value = "tukey"
    mock_manager.select_nonparametric_posthoc_dialog.return_value = "dunn"
    mock_manager.select_control_group_dialog.return_value = None
    mock_manager.select_custom_pairs_dialog.return_value = []
    mock_manager.select_transformation_dialog.return_value = None  # no transformation

    # Patch before any stats_functions import loads UIDialogManager
    with patch("stats_functions.UIDialogManager", mock_manager), \
         patch("statisticaltester.UIDialogManager", mock_manager):
        yield mock_manager


# ---------------------------------------------------------------------------
# Per-test: Excel fixture generator
# ---------------------------------------------------------------------------

def _bimodal(rng, n, loc1=1.0, loc2=50.0, scale=0.3):
    """Bimodal distribution — resistant to log/sqrt/boxcox normalization."""
    half = n // 2
    return np.concatenate([
        rng.normal(loc1, scale, half),
        rng.normal(loc2, scale, n - half),
    ])


def _make_two_group_long(dist, n_per_group, add_nan=False, seed=42):
    """Long-format 2-group DataFrame: columns [Group, Value]."""
    rng = np.random.default_rng(seed)
    if dist == "normal":
        g1 = rng.normal(loc=5.0, scale=1.0, size=n_per_group)
        g2 = rng.normal(loc=7.0, scale=1.0, size=n_per_group)   # clear effect
    else:
        # Bimodal: guaranteed non-normal even after log/sqrt/boxcox transformation
        g1 = _bimodal(rng, n_per_group, loc1=1.0, loc2=50.0)
        g2 = _bimodal(rng, n_per_group, loc1=2.0, loc2=55.0)
    if add_nan:
        g1[0] = np.nan
        g2[1] = np.nan
    groups = (["Control"] * n_per_group) + (["Treatment"] * n_per_group)
    values = np.concatenate([g1, g2])
    return pd.DataFrame({"Group": groups, "Value": values})


def _make_two_group_paired(dist, n_subjects, seed=42):
    """Long-format paired DataFrame: columns [Subject, Group, Value]."""
    rng = np.random.default_rng(seed)
    subjects = [f"S{i+1}" for i in range(n_subjects)]
    if dist == "normal":
        before = rng.normal(loc=5.0, scale=1.0, size=n_subjects)
        after  = before + rng.normal(loc=2.0, scale=0.5, size=n_subjects)
    else:
        # Bimodal: resistant to log/sqrt/boxcox normalization
        before = _bimodal(rng, n_subjects, loc1=1.0, loc2=50.0)
        after  = _bimodal(rng, n_subjects, loc1=2.0, loc2=55.0)
    rows = []
    for i, s in enumerate(subjects):
        rows.append({"Subject": s, "Group": "Before", "Value": before[i]})
        rows.append({"Subject": s, "Group": "After",  "Value": after[i]})
    return pd.DataFrame(rows)


def _make_three_group_long(dist, n_per_group, seed=42):
    """Long-format 3-group DataFrame: columns [Group, Value]."""
    rng = np.random.default_rng(seed)
    if dist == "normal":
        g1 = rng.normal(loc=5.0, scale=1.0, size=n_per_group)
        g2 = rng.normal(loc=7.0, scale=1.0, size=n_per_group)
        g3 = rng.normal(loc=9.0, scale=1.0, size=n_per_group)
    else:
        # Bimodal: resistant to log/sqrt/boxcox normalization
        g1 = _bimodal(rng, n_per_group, loc1=1.0, loc2=50.0)
        g2 = _bimodal(rng, n_per_group, loc1=2.0, loc2=55.0)
        g3 = _bimodal(rng, n_per_group, loc1=3.0, loc2=60.0)
    groups = (["G1"] * n_per_group) + (["G2"] * n_per_group) + (["G3"] * n_per_group)
    values = np.concatenate([g1, g2, g3])
    return pd.DataFrame({"Group": groups, "Value": values})


def _make_three_group_repeated(dist, n_subjects, seed=42):
    """Long-format RM DataFrame: columns [Subject, Group, Value]."""
    rng = np.random.default_rng(seed)
    subjects = [f"S{i+1}" for i in range(n_subjects)]
    rows = []
    for i, s in enumerate(subjects):
        if dist == "normal":
            base = rng.normal(5.0, 1.0)
            vals = [base, base + rng.normal(2.0, 0.5), base + rng.normal(4.0, 0.5)]
        else:
            # Bimodal: half subjects cluster near 1.0, half near 50.0 — resists all transformations
            if i < n_subjects // 2:
                base = rng.normal(1.0, 0.3)
                delta = rng.normal(0.2, 0.1)
            else:
                base = rng.normal(50.0, 0.3)
                delta = rng.normal(0.2, 0.1)
            vals = [base, base + delta, base + 2 * delta]
        for t, v in zip(["T1", "T2", "T3"], vals):
            rows.append({"Subject": s, "Group": t, "Value": v})
    return pd.DataFrame(rows)


def _make_two_way(dist, n_per_cell, seed=42):
    """Long-format 2-way independent DataFrame: columns [FactorA, FactorB, Value]."""
    rng = np.random.default_rng(seed)
    rows = []
    means = {"A1_B1": 5.0, "A1_B2": 7.0, "A2_B1": 6.0, "A2_B2": 10.0}
    for fa in ["A1", "A2"]:
        for fb in ["B1", "B2"]:
            m = means[f"{fa}_{fb}"]
            if dist == "normal":
                vals = rng.normal(m, 1.0, size=n_per_cell)
            else:
                vals = rng.lognormal(np.log(max(m, 0.1)), 1.2, size=n_per_cell)
            for v in vals:
                rows.append({"FactorA": fa, "FactorB": fb, "Value": v})
    return pd.DataFrame(rows)


def _make_mixed_anova(dist, n_subjects_per_group, seed=42):
    """Long-format mixed ANOVA DataFrame: columns [Subject, Group, Time, Value]."""
    rng = np.random.default_rng(seed)
    rows = []
    for gi, group in enumerate(["Control", "Treatment"]):
        for si in range(n_subjects_per_group):
            subj = f"{group[0]}{si+1}"
            base_effect = gi * 2.0  # between effect
            if dist == "normal":
                base = rng.normal(5.0 + base_effect, 1.0)
                for ti, time in enumerate(["T1", "T2", "T3"]):
                    val = base + ti * 1.5 + rng.normal(0, 0.5)
                    rows.append({"Subject": subj, "Group": group, "Time": time, "Value": val})
            else:
                base = rng.lognormal(np.log(5.0 + base_effect), 0.8)
                for ti, time in enumerate(["T1", "T2", "T3"]):
                    val = base * ((ti + 1) * 0.5) * rng.lognormal(0, 0.4)
                    rows.append({"Subject": subj, "Group": group, "Time": time, "Value": val})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Design definitions — used by test_all_paths.py
# ---------------------------------------------------------------------------

DESIGNS = [
    {
        "name": "indep_ttest_parametric",
        "df_factory": lambda: _make_two_group_long("normal", 15),
        "factor_columns": ["Group"],
        "dv_columns": ["Value"],
        "group_labels": ["Control", "Treatment"],
        "subject_column": None,
        "dependent": False,
        "inferred_test": "independent_ttest",
        "expected_test_keywords": ["t-test", "t_test", "independent"],
        "r_test": "indep_ttest",
        "levels": 2,
        "factors": 1,
    },
    {
        "name": "indep_ttest_nonparametric",
        "df_factory": lambda: _make_two_group_long("skewed", 15),
        "factor_columns": ["Group"],
        "dv_columns": ["Value"],
        "group_labels": ["Control", "Treatment"],
        "subject_column": None,
        "dependent": False,
        "inferred_test": "independent_ttest",
        "expected_test_keywords": ["mann", "whitney", "wilcoxon", "u-test"],
        "r_test": "mann_whitney",
        "levels": 2,
        "factors": 1,
    },
    {
        "name": "paired_ttest_parametric",
        "df_factory": lambda: _make_two_group_paired("normal", 12),
        "factor_columns": ["Group"],
        "dv_columns": ["Value"],
        "group_labels": ["Before", "After"],
        "subject_column": "Subject",
        "dependent": True,
        "inferred_test": "paired_ttest",
        "expected_test_keywords": ["paired", "t-test", "t_test"],
        "r_test": "paired_ttest",
        "levels": 2,
        "factors": 1,
    },
    {
        "name": "paired_ttest_nonparametric",
        "df_factory": lambda: _make_two_group_paired("skewed", 12),
        "factor_columns": ["Group"],
        "dv_columns": ["Value"],
        "group_labels": ["Before", "After"],
        "subject_column": "Subject",
        "dependent": True,
        "inferred_test": "paired_ttest",
        "expected_test_keywords": ["wilcoxon", "signed", "paired"],
        "r_test": "wilcoxon",
        "levels": 2,
        "factors": 1,
    },
    {
        "name": "one_way_anova_parametric",
        "df_factory": lambda: _make_three_group_long("normal", 12),
        "factor_columns": ["Group"],
        "dv_columns": ["Value"],
        "group_labels": ["G1", "G2", "G3"],
        "subject_column": None,
        "dependent": False,
        "inferred_test": "one_way_anova",
        "expected_test_keywords": ["anova", "one-way", "one_way", "f-test"],
        "r_test": "one_way_anova",
        "levels": 3,
        "factors": 1,
    },
    {
        "name": "one_way_anova_nonparametric",
        "df_factory": lambda: _make_three_group_long("skewed", 12),
        "factor_columns": ["Group"],
        "dv_columns": ["Value"],
        "group_labels": ["G1", "G2", "G3"],
        "subject_column": None,
        "dependent": False,
        "inferred_test": "one_way_anova",
        "expected_test_keywords": ["kruskal", "wallis"],
        "r_test": "kruskal_wallis",
        "levels": 3,
        "factors": 1,
    },
    {
        "name": "repeated_anova_parametric",
        "df_factory": lambda: _make_three_group_repeated("normal", 10),
        "factor_columns": ["Group"],
        "dv_columns": ["Value"],
        "group_labels": ["T1", "T2", "T3"],
        "subject_column": "Subject",
        "dependent": True,
        "inferred_test": "repeated_measures_anova",
        "expected_test_keywords": ["repeated", "rm", "anova"],
        "r_test": "repeated_anova",
        "levels": 3,
        "factors": 1,
    },
    {
        "name": "repeated_anova_nonparametric",
        "df_factory": lambda: _make_three_group_repeated("skewed", 10),
        "factor_columns": ["Group"],
        "dv_columns": ["Value"],
        "group_labels": ["T1", "T2", "T3"],
        "subject_column": "Subject",
        "dependent": True,
        "inferred_test": "repeated_measures_anova",
        "expected_test_keywords": ["friedman"],
        "r_test": "friedman",
        "levels": 3,
        "factors": 1,
    },
    {
        "name": "two_way_anova_parametric",
        "df_factory": lambda: _make_two_way("normal", 10),
        "factor_columns": ["FactorA", "FactorB"],
        "dv_columns": ["Value"],
        "group_labels": [],   # auto-determined from factors
        "subject_column": None,
        "dependent": False,
        "inferred_test": "two_way_anova",
        "expected_test_keywords": ["two-way", "two_way", "anova", "factorial"],
        "r_test": "two_way_anova",
        "levels": 4,
        "factors": 2,
    },
    {
        "name": "two_way_anova_nonparametric",
        "df_factory": lambda: _make_two_way("skewed", 10),
        "factor_columns": ["FactorA", "FactorB"],
        "dv_columns": ["Value"],
        "group_labels": [],
        "subject_column": None,
        "dependent": False,
        "inferred_test": "two_way_anova",
        "expected_test_keywords": ["two-way", "two_way", "anova", "non-parametric", "nonparam"],
        "r_test": None,   # No valid R equivalent: Python uses rank/permutation, R uses parametric aov()
        "levels": 4,
        "factors": 2,
    },
    {
        "name": "mixed_anova_parametric",
        "df_factory": lambda: _make_mixed_anova("normal", 8),
        "factor_columns": ["Group", "Time"],
        "dv_columns": ["Value"],
        "group_labels": [],
        "subject_column": "Subject",
        "dependent": True,
        "inferred_test": "mixed_anova",
        "expected_test_keywords": ["mixed", "anova"],
        "r_test": "mixed_anova",
        "levels": 6,
        "factors": 2,
        "between_factors": ["Group"],
        "within_factors": ["Time"],
    },
    {
        "name": "mixed_anova_nonparametric",
        "df_factory": lambda: _make_mixed_anova("skewed", 8),
        "factor_columns": ["Group", "Time"],
        "dv_columns": ["Value"],
        "group_labels": [],
        "subject_column": "Subject",
        "dependent": True,
        "inferred_test": "mixed_anova",
        "expected_test_keywords": ["mixed", "anova", "non-parametric", "brunner", "glmm"],
        "r_test": "mixed_anova",
        "levels": 6,
        "factors": 2,
        "between_factors": ["Group"],
        "within_factors": ["Time"],
    },
    {
        "name": "nan_robustness",
        "df_factory": lambda: _make_two_group_long("normal", 15, add_nan=True),
        "factor_columns": ["Group"],
        "dv_columns": ["Value"],
        "group_labels": ["Control", "Treatment"],
        "subject_column": None,
        "dependent": False,
        "inferred_test": "independent_ttest",
        "expected_test_keywords": ["t-test", "t_test", "mann", "whitney"],
        "r_test": None,   # no R validation for robustness test (NaN handling differs)
        "levels": 2,
        "factors": 1,
    },
]


@pytest.fixture
def make_excel_fixture(tmp_path):
    """
    Returns a factory function.
    Usage: excel_path = make_excel_fixture(design)
    """
    def _generate(design: dict) -> str:
        df = design["df_factory"]()
        out = tmp_path / f"{design['name']}.xlsx"
        df.to_excel(str(out), index=False)
        return str(out)
    return _generate
