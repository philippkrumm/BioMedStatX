"""F1/B5/S10: unify the two auto correlation engines on skew/kurtosis tiers.

Before this change the two engines in correlation_models.py disagreed on
identical data:

  * CorrelationModel (single pair) chose Pearson vs Spearman by sample-size
    tier and shape (|skew|, |excess kurtosis|).
  * ExploratoryCorrelationMatrix chose per pair by a Shapiro-Wilk pre-test.

On the disagreement fixture below (Seed 7, n=25: lognormal X with skew ~1.03,
normal Y) Shapiro passed on both variables (px=0.078, py=0.861) so the matrix
picked Pearson (r=-0.110223), while the single-pair engine picked Spearman on
the skew (r=-0.100000). Same module, same data, two answers.

Both engines now route through _select_correlation_method, so the matrix now
also returns Spearman here -- the delivered number actually changes
(-0.110223 -> -0.100000), which is the discriminance proof that the switch is
real and not just a relabel. B5 adds a per-pair method_matrix so a mixed 'auto'
matrix records which method actually ran for each cell.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from analysis.correlation_models import (
    CorrelationModel,
    ExploratoryCorrelationMatrix,
    _select_correlation_method,
)


def _disagreement_pair(seed=7, n=25):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"X": rng.lognormal(0, 0.55, n), "Y": rng.normal(0, 1, n)})


def test_shared_helper_is_the_single_source_of_truth():
    """Both engines must call the same selector -- verify it exists and tiers."""
    # n < 20 -> always spearman regardless of perfect shape
    assert _select_correlation_method(10, 0.0, 0.0, 0.0, 0.0) == "spearman"
    # mid tier, clean shape -> pearson; skew>1 -> spearman
    assert _select_correlation_method(50, 0.5, 0.5, 1.0, 1.0) == "pearson"
    assert _select_correlation_method(50, 1.5, 0.0, 0.0, 0.0) == "spearman"
    # large n tolerates moderate skew, rejects extreme
    assert _select_correlation_method(150, 1.5, 1.5, 3.0, 3.0) == "pearson"
    assert _select_correlation_method(150, 2.5, 0.0, 0.0, 0.0) == "spearman"


def test_premise_shapiro_and_shape_disagree_on_the_fixture():
    """Guard the fixture: Shapiro would pick Pearson, shape picks Spearman."""
    df = _disagreement_pair()
    x, y = df["X"].values, df["Y"].values
    _, px = scipy_stats.shapiro(x)
    _, py = scipy_stats.shapiro(y)
    old_shapiro_choice = "pearson" if (px > 0.05 and py > 0.05) else "spearman"
    sx, sy = scipy_stats.skew(x), scipy_stats.skew(y)
    kx, ky = scipy_stats.kurtosis(x), scipy_stats.kurtosis(y)
    new_choice = _select_correlation_method(len(x), sx, sy, kx, ky)
    assert old_shapiro_choice == "pearson", "fixture no longer trips the old Shapiro rule"
    assert new_choice == "spearman", "fixture no longer trips the skew tier"


def test_both_engines_agree_on_the_disagreement_fixture():
    df = _disagreement_pair()

    single = CorrelationModel()
    single.fit(df, x_col="X", y_col="Y", method="auto")
    single_res = single.as_results_dict()

    matrix = ExploratoryCorrelationMatrix()
    matrix.fit(df, ["X", "Y"], method="auto", correction=None)
    mat_res = matrix.as_results_dict()

    # Same method from both engines now.
    assert single_res["method"] == "spearman"
    assert mat_res["method_matrix"]["X"]["Y"] == "spearman"

    # And the same coefficient (the matrix number moved off the Pearson value).
    assert mat_res["r_matrix"]["X"]["Y"] == single_res["r"]


def test_matrix_result_actually_changed_off_the_pearson_value():
    """Discriminance: the delivered r is the Spearman value, not the old Pearson one."""
    df = _disagreement_pair()
    x, y = df["X"].values, df["Y"].values
    pearson_r = float(scipy_stats.pearsonr(x, y)[0])
    spearman_r = float(scipy_stats.spearmanr(x, y)[0])
    assert abs(pearson_r - spearman_r) > 0.005, "fixture too weak to tell the methods apart"

    matrix = ExploratoryCorrelationMatrix()
    matrix.fit(df, ["X", "Y"], method="auto", correction=None)
    r_delivered = matrix.as_results_dict()["r_matrix"]["X"]["Y"]
    assert abs(r_delivered - spearman_r) < 1e-9, "matrix did not switch to Spearman"
    assert abs(r_delivered - pearson_r) > 0.005, "matrix still delivers the old Pearson value"


def test_positive_control_genuine_normal_pair_stays_pearson():
    """A clean bivariate-normal pair must keep Pearson in both engines."""
    rng = np.random.default_rng(11)
    n = 60
    x = rng.normal(50, 10, n)
    df = pd.DataFrame({"X": x, "Y": 2 * x + rng.normal(0, 8, n)})

    single = CorrelationModel()
    single.fit(df, x_col="X", y_col="Y", method="auto")
    assert single.as_results_dict()["method"] == "pearson"

    matrix = ExploratoryCorrelationMatrix()
    matrix.fit(df, ["X", "Y"], method="auto", correction=None)
    assert matrix.as_results_dict()["method_matrix"]["X"]["Y"] == "pearson"


def test_mixed_matrix_records_method_per_pair():
    """B5: a mixed 'auto' matrix must record which method ran for each cell."""
    rng = np.random.default_rng(3)
    n = 60
    a = rng.normal(0, 1, n)
    b = 0.7 * a + rng.normal(0, 1, n)          # A-B: both ~normal -> pearson
    c = rng.lognormal(0, 0.9, n)               # any pair with C: skewed -> spearman
    df = pd.DataFrame({"A": a, "B": b, "C": c})

    matrix = ExploratoryCorrelationMatrix()
    matrix.fit(df, ["A", "B", "C"], method="auto", correction=None)
    mm = matrix.as_results_dict()["method_matrix"]

    assert mm["A"]["B"] == "pearson"
    assert mm["A"]["C"] == "spearman"
    assert mm["B"]["C"] == "spearman"
    # Symmetric and diagonal is None (never a "method" for a self-pair).
    assert mm["B"]["A"] == mm["A"]["B"]
    assert mm["A"]["A"] is None


def test_fixed_method_is_still_recorded_in_the_matrix():
    """A non-auto matrix still logs the method it was told to use, per pair."""
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"A": rng.normal(0, 1, 40), "B": rng.normal(0, 1, 40)})
    matrix = ExploratoryCorrelationMatrix()
    matrix.fit(df, ["A", "B"], method="spearman", correction=None)
    assert matrix.as_results_dict()["method_matrix"]["A"]["B"] == "spearman"


def test_report_rows_surface_the_per_pair_method():
    """B5/S10: the matrix report must show which method ran per pair, and must
    not claim Shapiro-Wilk decides."""
    from export.report_stat_rows import _StatRowsMixin

    rng = np.random.default_rng(3)
    n = 60
    a = rng.normal(0, 1, n)
    b = 0.7 * a + rng.normal(0, 1, n)
    c = rng.lognormal(0, 0.9, n)
    df = pd.DataFrame({"A": a, "B": b, "C": c})
    matrix = ExploratoryCorrelationMatrix()
    matrix.fit(df, ["A", "B", "C"], method="auto", correction="fdr_bh")
    rows = _StatRowsMixin._build_corr_matrix_statistical_rows(matrix.as_results_dict())

    flat = {r["label"].strip(): r["value"] for r in rows}
    # Honest method description, no Shapiro claim.
    assert "skewness/kurtosis" in flat["Method"]
    assert "normality" not in flat["Method"].lower()
    # Per-pair breakdown present.
    assert flat.get("A × B") == "Pearson"
    assert flat.get("A × C") == "Spearman"
    assert flat.get("B × C") == "Spearman"
    assert flat["Pairs run as Pearson"] == "1"
    assert flat["Pairs run as Spearman"] == "2"


def test_strata_carry_the_method_matrix():
    """Per-stratum matrices must also expose their per-pair method record."""
    rng = np.random.default_rng(9)
    n = 40
    grp = np.array(["G1"] * n + ["G2"] * n)
    x = np.concatenate([rng.normal(0, 1, n), rng.normal(0, 1, n)])
    y = np.concatenate([2 * x[:n] + rng.normal(0, 1, n), rng.lognormal(0, 0.9, n)])
    df = pd.DataFrame({"X": x, "Y": y, "Grp": grp})

    matrix = ExploratoryCorrelationMatrix()
    matrix.fit(df, ["X", "Y"], method="auto", correction=None, stratify_by="Grp")
    res = matrix.as_results_dict()
    assert "strata" in res
    for grp_key, mats in res["strata"].items():
        assert "method_matrix" in mats
        assert mats["method_matrix"]["X"]["Y"] in ("pearson", "spearman")
