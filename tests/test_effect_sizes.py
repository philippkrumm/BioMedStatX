"""Unit tests for effect_sizes module — pure-domain, no PyQt / Jinja needed."""
import math
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analysis.effect_sizes import (  # noqa: E402
    EffectSizeKind,
    canonicalize,
    classify,
    magnitude,
)


# ---------------------------------------------------------------------------
# canonicalize — substring rules must not bleed between kinds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,expected", [
    ("Cohen's f", EffectSizeKind.COHEN_F),
    ("cohens_f", EffectSizeKind.COHEN_F),
    ("f", EffectSizeKind.COHEN_F),
    ("f (2, 18)", EffectSizeKind.COHEN_F),
    ("Cohen's d", EffectSizeKind.COHEN_D),
    ("cohen_d", EffectSizeKind.COHEN_D),
    ("Hedges' g", EffectSizeKind.COHEN_D),
    ("hedges_g", EffectSizeKind.COHEN_D),
    ("d", EffectSizeKind.COHEN_D),
    ("g", EffectSizeKind.COHEN_D),
    ("McFadden pseudo R²", EffectSizeKind.MCFADDEN_R2),
    ("pseudo_r2", EffectSizeKind.MCFADDEN_R2),
    ("ICC", EffectSizeKind.ICC),
    ("ICC3k", EffectSizeKind.ICC),
    ("AUC", EffectSizeKind.AUC),
    ("ROC AUC", EffectSizeKind.AUC),
    ("eta_squared", EffectSizeKind.ETA_SQUARED),
    ("partial_eta_squared", EffectSizeKind.ETA_SQUARED),
    ("omega_squared", EffectSizeKind.ETA_SQUARED),
    ("epsilon_squared", EffectSizeKind.ETA_SQUARED),
    ("η²", EffectSizeKind.ETA_SQUARED),
    ("r_squared", EffectSizeKind.R_SQUARED),
    ("R²", EffectSizeKind.R_SQUARED),
    ("rank_biserial", EffectSizeKind.RANK_BISERIAL),
    ("Kendall's W", EffectSizeKind.RANK_BISERIAL),
    ("w", EffectSizeKind.RANK_BISERIAL),
    ("pearson r", EffectSizeKind.CORRELATION_R),
    ("Spearman rho", EffectSizeKind.CORRELATION_R),
    ("correlation", EffectSizeKind.CORRELATION_R),
    ("r", EffectSizeKind.CORRELATION_R),
    ("ρ", EffectSizeKind.CORRELATION_R),
    ("Cramer's V", EffectSizeKind.CRAMER_V),
    ("cramér_v", EffectSizeKind.CRAMER_V),
    ("v", EffectSizeKind.CRAMER_V),
])
def test_canonicalize_known_labels(label, expected):
    assert canonicalize(label) is expected


@pytest.mark.parametrize("label", ["", None, "  ", "magic_effect", "xyz"])
def test_canonicalize_unknown_or_empty(label):
    assert canonicalize(label) is None


def test_canonicalize_case_insensitive_and_trimmed():
    assert canonicalize("  COHEN'S D  ") is EffectSizeKind.COHEN_D


def test_cohen_f_does_not_bleed_into_cohen_d():
    """Regression: substring "cohen" in "Cohen's f" must NOT route to COHEN_D."""
    assert canonicalize("Cohen's f") is EffectSizeKind.COHEN_F


def test_partial_eta_squared_does_not_bleed_into_r_squared():
    """Regression: "partial_eta_squared" contains "squared" but must be ETA family."""
    assert canonicalize("partial_eta_squared") is EffectSizeKind.ETA_SQUARED


# ---------------------------------------------------------------------------
# magnitude — threshold bands
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("es,expected", [
    (0.05, "negligible"),
    (0.10, "small"),
    (0.24, "small"),
    (0.25, "medium"),
    (0.39, "medium"),
    (0.40, "large"),
    (1.5, "large"),
])
def test_magnitude_cohen_f_thresholds(es, expected):
    assert magnitude(es, EffectSizeKind.COHEN_F) == expected


@pytest.mark.parametrize("es,expected", [
    (0.10, "negligible"),
    (0.20, "small"),
    (0.50, "medium"),
    (0.79, "medium"),
    (0.80, "large"),
])
def test_magnitude_cohen_d_thresholds(es, expected):
    assert magnitude(es, EffectSizeKind.COHEN_D) == expected


def test_magnitude_cohen_d_uses_absolute_value():
    assert magnitude(-1.2, EffectSizeKind.COHEN_D) == "large"


@pytest.mark.parametrize("es,expected", [
    (0.0, "negligible"),
    (0.49, "negligible"),
    (0.50, "small"),
    (0.74, "small"),
    (0.75, "medium"),
    (0.89, "medium"),
    (0.90, "large"),
    (0.99, "large"),
])
def test_magnitude_icc_koo_li(es, expected):
    assert magnitude(es, EffectSizeKind.ICC) == expected


@pytest.mark.parametrize("es,expected", [
    (0.65, "negligible"),
    (0.70, "small"),
    (0.79, "small"),
    (0.80, "medium"),
    (0.90, "large"),
])
def test_magnitude_auc_hosmer_lemeshow(es, expected):
    assert magnitude(es, EffectSizeKind.AUC) == expected


@pytest.mark.parametrize("es,expected", [
    (0.005, "negligible"),
    (0.01, "small"),
    (0.06, "medium"),
    (0.14, "large"),
])
def test_magnitude_eta_squared(es, expected):
    assert magnitude(es, EffectSizeKind.ETA_SQUARED) == expected


@pytest.mark.parametrize("es,expected", [
    (0.01, "negligible"),
    (0.02, "small"),
    (0.13, "medium"),
    (0.26, "large"),
])
def test_magnitude_r_squared(es, expected):
    assert magnitude(es, EffectSizeKind.R_SQUARED) == expected


@pytest.mark.parametrize("es,expected", [
    (0.05, "negligible"),
    (0.10, "small"),
    (0.20, "medium"),
    (0.40, "large"),
])
def test_magnitude_mcfadden(es, expected):
    assert magnitude(es, EffectSizeKind.MCFADDEN_R2) == expected


# ---------------------------------------------------------------------------
# Cramer's V — df*-aware scaling
# ---------------------------------------------------------------------------

def test_cramer_v_df_star_1_uses_base_thresholds():
    """df*=1 (2x2 table) → 0.10 / 0.30 / 0.50."""
    assert magnitude(0.30, EffectSizeKind.CRAMER_V, df_star=1) == "medium"
    assert magnitude(0.10, EffectSizeKind.CRAMER_V, df_star=1) == "small"
    assert magnitude(0.50, EffectSizeKind.CRAMER_V, df_star=1) == "large"


def test_cramer_v_df_star_4_scales_thresholds_down():
    """df*=4 → divide by sqrt(4) = 2. Small=0.05, medium=0.15, large=0.25."""
    assert magnitude(0.05, EffectSizeKind.CRAMER_V, df_star=4) == "small"
    assert magnitude(0.15, EffectSizeKind.CRAMER_V, df_star=4) == "medium"
    assert magnitude(0.25, EffectSizeKind.CRAMER_V, df_star=4) == "large"


def test_cramer_v_default_df_star_acts_as_1():
    assert magnitude(0.30, EffectSizeKind.CRAMER_V) == "medium"


def test_cramer_v_negative_df_star_defensively_falls_back():
    """Invalid df* (0, negative) should fall back to df*=1 — not raise."""
    assert magnitude(0.30, EffectSizeKind.CRAMER_V, df_star=0) == "medium"
    assert magnitude(0.30, EffectSizeKind.CRAMER_V, df_star=-2) == "medium"


# ---------------------------------------------------------------------------
# magnitude — boundary / invalid inputs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [None, "string", float("nan"), [], {}])
def test_magnitude_invalid_inputs_return_none(bad):
    assert magnitude(bad, EffectSizeKind.COHEN_D) is None


def test_magnitude_unknown_kind_returns_none():
    assert magnitude(0.5, None) is None


# ---------------------------------------------------------------------------
# classify — composition smoke tests + regression of the Cohen-f bug
# ---------------------------------------------------------------------------

def test_classify_cohen_f_uses_f_thresholds_not_d_thresholds():
    """Regression: 0.30 is "medium" for f (>=0.25) but "small" for d (<0.50).
    The old substring-matching bug routed Cohen's f → d thresholds → "small"."""
    assert classify(0.30, "Cohen's f") == "medium"
    assert classify(0.30, "Cohen's d") == "small"


def test_classify_unknown_type_returns_none():
    assert classify(0.5, "made_up_effect") is None


def test_classify_round_trips_via_canonicalize_and_magnitude():
    for label in ("Cohen's d", "ICC", "AUC", "eta_squared", "Cramer's V"):
        kind = canonicalize(label)
        assert kind is not None
        assert classify(0.5, label) == magnitude(0.5, kind)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
