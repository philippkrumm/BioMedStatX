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

    # The parametric post-hoc dialog now also fires for the Welch one-way path
    # (Games-Howell default). Mirror each context's real default so headless
    # runs keep their expected post-hoc: Games-Howell for the Welch one-way
    # (signalled by default_method="games_howell"), Tukey for advanced ANOVAs.
    def _posthoc_choice(*args, **kwargs):
        return "games_howell" if kwargs.get("default_method") == "games_howell" else "tukey"

    mock_manager.select_posthoc_test_dialog.side_effect = _posthoc_choice
    mock_manager.select_nonparametric_posthoc_dialog.return_value = "dunn"
    mock_manager.select_control_group_dialog.return_value = None
    mock_manager.select_custom_pairs_dialog.return_value = []
    mock_manager.select_transformation_dialog.return_value = None  # no transformation

    # Patch before any stats_functions import loads UIDialogManager
    with patch("analysis.stats_functions.UIDialogManager", mock_manager), \
         patch("analysis.statisticaltester.UIDialogManager", mock_manager):
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
            # Massive outlier to destroy normality of residuals and force Friedman test
            base = rng.normal(10.0, 2.0)
            vals = [base, base + rng.normal(2.0, 0.5), base + rng.normal(4.0, 0.5)]
            if i == 0:
                vals[0] += 1000.0  # Extreme outlier
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
            if dist == "normal":
                # Tight per-subject base (scale=0.5) + small time noise (scale=0.3) keeps
                # per-cell residuals clearly Gaussian — reliably passes Shapiro-Wilk at n=15.
                base = rng.normal(5.0 + gi * 3.0, 0.5)
                for ti, time in enumerate(["T1", "T2", "T3"]):
                    val = base + ti * 2.0 + rng.normal(0, 0.3)
                    rows.append({"Subject": subj, "Group": group, "Time": time, "Value": val})
            else:
                base = rng.lognormal(np.log(5.0 + gi * 2.0), 0.8)
                for ti, time in enumerate(["T1", "T2", "T3"]):
                    val = base * ((ti + 1) * 0.5) * rng.lognormal(0, 0.4)
                    rows.append({"Subject": subj, "Group": group, "Time": time, "Value": val})
    return pd.DataFrame(rows)


def _make_ancova(n_per_group=12, seed=42):
    """Long-format ANCOVA: columns [Group, Covariate, Value].
    Two groups with a continuous covariate that explains ~50% variance.
    """
    rng = np.random.default_rng(seed)
    groups, covs, vals = [], [], []
    for gi, grp in enumerate(["Control", "Treatment"]):
        cov = rng.normal(5.0, 1.0, size=n_per_group)
        val = (gi * 3.0) + 0.8 * cov + rng.normal(0, 0.5, size=n_per_group)
        groups.extend([grp] * n_per_group)
        covs.extend(cov.tolist())
        vals.extend(val.tolist())
    return pd.DataFrame({"Group": groups, "Covariate": covs, "Value": vals})


def _make_correlation(n=30, seed=42):
    """Bivariate normal data: columns [Group, X, Y] with ~0.7 Pearson correlation.
    Normal marginals ensure Python auto-selects Pearson.
    Group="Sample" (constant) avoids float-precision issues in group dispatch;
    correlation routing uses x_variable / y from analysis_context.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(10.0, 2.0, size=n)
    y = 0.7 * x + rng.normal(0, 1.5, size=n)
    return pd.DataFrame({"Group": ["Sample"] * n, "X": x, "Y": y})


def _make_correlation_spearman(n=30, seed=42):
    """Bimodal X with monotone Y: columns [Group, X, Y].
    Non-normal marginals ensure Python auto-selects Spearman.
    """
    rng = np.random.default_rng(seed)
    x = _bimodal(rng, n, loc1=1.0, loc2=20.0, scale=0.5)
    y = 0.8 * x + rng.normal(0, 0.3, size=n)
    return pd.DataFrame({"Group": ["Sample"] * n, "X": x, "Y": y})


def _make_regression(n=30, seed=99):
    """Simple OLS: columns [Group, X, Y] with strong linear relationship.
    Uses different seed from correlation fixtures to avoid accidental overlap.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, size=n)
    y = 2.5 * x + 5.0 + rng.normal(0, 1.5, size=n)
    return pd.DataFrame({"Group": ["Sample"] * n, "X": x, "Y": y})


def _make_welch_anova_heteroscedastic(n_per_group=12, seed=7):
    """3 normally distributed groups with unequal variances (σ ratio 1:2:4).

    n_per_group kept small so Shapiro-Wilk on pooled residuals has low power
    (passes), while Brown-Forsythe detects the 1:4 variance ratio.
    → Routing: normal + unequal var → Welch ANOVA → Games-Howell.
    If normality fails anyway (scale-mixture heavy tails), Kruskal-Wallis is
    the correct fallback — expected_test_keywords accepts both.
    """
    rng = np.random.default_rng(seed)
    sds = [1.0, 2.0, 4.0]
    rows = []
    for grp, sd in zip(["G1", "G2", "G3"], sds):
        vals = rng.normal(10.0, sd, size=n_per_group)
        for v in vals:
            rows.append({"Group": grp, "Value": v})
    return pd.DataFrame(rows)


def _make_sphericity_violation_rm(n_subjects=16, seed=13):
    """RM design where Mauchly's sphericity test will fail.
    T1-T2: tight correlation (r≈0.95), T2-T3: independent with 5× variance.
    Var(T1-T2) << Var(T2-T3) → compound symmetry broken → GG/HF correction.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subjects):
        base = rng.normal(10.0, 1.0)
        t1 = base + rng.normal(0, 0.15)
        t2 = base + rng.normal(0, 0.15)   # tight with T1
        t3 = rng.normal(12.0, 5.0)        # independent, large variance
        for lvl, val in zip(["T1", "T2", "T3"], [t1, t2, t3]):
            rows.append({"Subject": f"S{s+1}", "Group": lvl, "Value": val})
    return pd.DataFrame(rows)


def _make_lmm_hierarchical(n_subjects=20, seed=17):
    """Hierarchical design: subjects measured at 3 time points with between-subject grouping.
    Routes to LMM when inferred_test='lmm'.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subjects):
        grp = "Trt" if s < n_subjects // 2 else "Ctrl"
        subj_re = rng.normal(0, 1.5)
        for ti, time in enumerate(["T1", "T2", "T3"]):
            val = (5.0 if grp == "Trt" else 0.0) + ti * 1.5 + subj_re + rng.normal(0, 0.8)
            rows.append({"Subject": f"S{s+1}", "Group": grp, "Time": time, "Value": val})
    return pd.DataFrame(rows)


def _make_logistic_regression(n=60, seed=23):
    """Binary outcome with group predictor + continuous covariate.
    Routes to logistic_regression when inferred_test='logistic_regression'.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for gi, grp in enumerate(["Ctrl", "Trt"]):
        n_g = n // 2
        cov = rng.normal(50.0, 10.0, size=n_g)
        logit = -2.0 + gi * 1.5 + 0.04 * cov
        prob = 1.0 / (1.0 + np.exp(-logit))
        outcome = (rng.uniform(size=n_g) < prob).astype(int)
        for c, o in zip(cov, outcome):
            rows.append({"Group": grp, "Covariate": float(c), "Outcome": int(o)})
    return pd.DataFrame(rows)


def _make_skewed_lognormal(n_per_group=20, seed=31):
    """Strongly right-skewed log-normal data (3 groups).
    Shapiro-Wilk fails → transformation dialog is triggered.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for gi, grp in enumerate(["G1", "G2", "G3"]):
        vals = rng.lognormal(mean=gi * 0.5, sigma=1.2, size=n_per_group)
        for v in vals:
            rows.append({"Group": grp, "Value": v})
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
        "r_output_format": ["p_value", "statistic", "effect_size"],  # Cohen's d
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
        "r_output_format": ["p_value", "statistic"],
        # Mann-Whitney: both sides use normal approximation → tighter tolerance
        "r_tolerance": 1e-3,
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
        "r_output_format": ["p_value", "statistic", "effect_size"],  # Cohen's d paired
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
        "r_output_format": ["p_value", "statistic"],
        # Wilcoxon: Python uses normal approx, R may use exact distribution → loose tolerance
        "r_tolerance": 0.05,
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
        "r_output_format": [
            "p_value", "statistic", "eta_squared", "cohens_f",
            "p_tukey_1", "p_tukey_2", "p_tukey_3",
        ],
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
        "r_output_format": ["p_value", "statistic"],
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
        "r_output_format": ["p_value", "statistic"],
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
        "r_output_format": ["p_value", "statistic"],
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
        "r_output_format": [
            "p_FactorA", "p_FactorB", "p_Interaction",
            "F_FactorA",  "F_FactorB",  "F_Interaction",
            "peta_FactorA", "peta_FactorB", "peta_Interaction",
        ],
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
        "r_output_format": ["p_value", "statistic"],
        "levels": 4,
        "factors": 2,
    },
    {
        "name": "mixed_anova_parametric",
        "df_factory": lambda: _make_mixed_anova("normal", 15, seed=2),
        "factor_columns": ["Group", "Time"],
        "dv_columns": ["Value"],
        "group_labels": [],
        "subject_column": "Subject",
        "dependent": True,
        "inferred_test": "mixed_anova",
        "expected_test_keywords": ["mixed", "anova"],
        "r_test": "mixed_anova",
        "r_output_format": [
            "p_between", "p_within", "p_interaction",
            "F_between",  "F_within",  "F_interaction",
            "peta_between", "peta_within", "peta_interaction",
        ],
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
        "r_output_format": [
            "p_between", "p_within", "p_interaction",
            "F_between",  "F_within",  "F_interaction",
            "peta_between", "peta_within", "peta_interaction",
        ],
        # Nonparametric mixed ANOVA uses different test (Brunner-Langer) — large tolerance
        "r_tolerance": 0.5,
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
        "r_output_format": ["p_value", "statistic"],
        "levels": 2,
        "factors": 1,
    },
    # ── New designs: ANCOVA, Correlation (Pearson + Spearman), OLS Regression ──
    {
        "name": "ancova_parametric",
        "df_factory": lambda: _make_ancova(12),
        "factor_columns": ["Group"],
        "dv_columns": ["Value"],
        "group_labels": ["Control", "Treatment"],
        "subject_column": None,
        "dependent": False,
        "inferred_test": "ancova",
        "expected_test_keywords": ["ancova", "covariate"],
        "r_test": "ancova",
        "r_output_format": ["p_value", "statistic", "eta_squared", "p_covariate"],
        "levels": 2,
        "factors": 1,
        "covariate_columns": ["Covariate"],
    },
    {
        "name": "correlation_pearson",
        "df_factory": lambda: _make_correlation(30),
        # Group="Sample" (constant) avoids float-precision issues; x_column is passed
        # via analysis_context["x_variable"] which analysis_core.py reads at line 589
        "factor_columns": ["Group"],
        "dv_columns": ["Y"],
        "group_labels": ["Sample"],
        "subject_column": None,
        "dependent": False,
        "inferred_test": "correlation",
        "expected_test_keywords": ["correlation", "pearson"],
        "r_test": "correlation",
        "r_output_format": ["p_value", "statistic"],
        "r_extra_args": ["pearson"],   # passed as 2nd CLI arg to correlation.R
        "levels": 1,
        "factors": 1,
        "x_column": "X",
        "y_column": "Y",
    },
    {
        "name": "correlation_spearman",
        "df_factory": lambda: _make_correlation_spearman(30),
        "factor_columns": ["Group"],
        "dv_columns": ["Y"],
        "group_labels": ["Sample"],
        "subject_column": None,
        "dependent": False,
        "inferred_test": "correlation",
        "expected_test_keywords": ["correlation", "spearman"],
        "r_test": "correlation",
        "r_output_format": ["p_value", "statistic"],
        "r_extra_args": ["spearman"],  # passed as 2nd CLI arg to correlation.R
        "levels": 1,
        "factors": 1,
        "x_column": "X",
        "y_column": "Y",
    },
    {
        "name": "regression_ols",
        "df_factory": lambda: _make_regression(30),
        "factor_columns": ["Group"],
        "dv_columns": ["Y"],
        "group_labels": ["Sample"],
        "subject_column": None,
        "dependent": False,
        "inferred_test": "linear_regression",
        "expected_test_keywords": ["linear", "regression", "ols"],
        "r_test": "regression",
        "r_output_format": ["p_value", "statistic", "r_squared"],
        "levels": 1,
        "factors": 1,
        "x_column": "X",
        "y_column": "Y",
    },
    # ── New designs: Welch, Sphericity, LMM, Logistic ──────────────────────
    {
        "name": "welch_anova_heteroscedastic",
        "df_factory": lambda: _make_welch_anova_heteroscedastic(20),
        "factor_columns": ["Group"],
        "dv_columns": ["Value"],
        "group_labels": ["G1", "G2", "G3"],
        "subject_column": None,
        "dependent": False,
        "inferred_test": "one_way_anova",
        # If normality passes: Welch ANOVA + Games-Howell. If pooled residuals
        # fail Shapiro-Wilk (scale-mixture tails), Kruskal-Wallis is the correct
        # non-parametric fallback — both routes are accepted.
        "expected_test_keywords": ["welch", "games", "howell", "anova", "kruskal", "wallis"],
        "r_test": None,  # Welch ANOVA p-value differs from standard F; skip R cross-check
        "r_output_format": ["p_value", "statistic"],
        "levels": 3,
        "factors": 1,
    },
    {
        "name": "sphericity_violation_rm",
        "df_factory": lambda: _make_sphericity_violation_rm(16),
        "factor_columns": ["Group"],
        "dv_columns": ["Value"],
        "group_labels": ["T1", "T2", "T3"],
        "subject_column": "Subject",
        "dependent": True,
        "inferred_test": "repeated_measures_anova",
        "expected_test_keywords": ["repeated", "rm", "anova"],
        # Mauchly will fail; test checks for sphericity_correction key in result
        "requires_sphericity_correction": True,
        "r_test": None,  # GG-corrected RM ANOVA differs from R's standard aov(); skip
        "r_output_format": ["p_value", "statistic"],
        "levels": 3,
        "factors": 1,
    },
    {
        "name": "lmm_hierarchical",
        "df_factory": lambda: _make_lmm_hierarchical(20),
        "factor_columns": ["Group"],
        "dv_columns": ["Value"],
        "group_labels": ["Trt", "Ctrl"],
        "subject_column": "Subject",
        "dependent": True,
        "inferred_test": "lmm",
        "expected_test_keywords": ["lmm", "mixed", "linear", "model"],
        "r_test": None,
        "r_output_format": ["p_value", "statistic"],
        "levels": 2,
        "factors": 1,
        "between_factors": ["Group"],
        "within_factors": ["Time"],
    },
    {
        "name": "logistic_regression",
        "df_factory": lambda: _make_logistic_regression(60),
        "factor_columns": ["Group"],
        "dv_columns": ["Outcome"],
        "group_labels": ["Ctrl", "Trt"],
        "subject_column": None,
        "dependent": False,
        "inferred_test": "logistic_regression",
        "expected_test_keywords": ["logistic", "regression"],
        "r_test": None,
        "r_output_format": ["p_value", "statistic"],
        "levels": 2,
        "factors": 1,
        "covariate_columns": ["Covariate"],
    },
    {
        "name": "skewed_lognormal_boxcox",
        # Log-normal data; test_all_paths.py overrides transform mock → "box_cox" for this design
        "df_factory": lambda: _make_skewed_lognormal(20),
        "factor_columns": ["Group"],
        "dv_columns": ["Value"],
        "group_labels": ["G1", "G2", "G3"],
        "subject_column": None,
        "dependent": False,
        "inferred_test": "one_way_anova",
        "expected_test_keywords": ["anova", "kruskal", "wallis", "mann", "whitney"],
        "requires_transform_mock": "box_cox",
        "r_test": None,
        "r_output_format": ["p_value", "statistic"],
        "levels": 3,
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
