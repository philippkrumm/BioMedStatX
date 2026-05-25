"""Effect-size magnitude classification — pure domain module.

Extracted from ``html_exporter.py`` to:
    * Remove a statistical-domain function from a reporting/templating module.
    * Replace ambiguous substring matching ("cohen" → d *or* f?) with an
      enum-based dispatch that fails closed on unknown types.
    * Make threshold conventions unit-testable without importing Jinja / Qt.

Threshold sources (all peer-reviewed conventions):
    * Cohen, J. (1988). *Statistical Power Analysis for the Behavioral
      Sciences* (2nd ed.). — d, f, η², r, V.
    * Koo, T. K., & Li, M. Y. (2016). Selecting Appropriate Statistical
      Methods for ICC. *J Chiropractic Med*, 15(2), 155–163. — ICC.
    * Hosmer, D. W., & Lemeshow, S. (2000). *Applied Logistic Regression*. — AUC.
    * McFadden, D. (1979). Quantitative methods for analyzing travel
      behaviour: an overview. — pseudo-R².

Cramer's V scales with df* = min(rows-1, cols-1) per Cohen (1988); callers
must pass ``df_star`` when computing V on tables larger than 2x2. df_star=1
(default) reproduces the unscaled 0.10/0.30/0.50 thresholds.
"""
from __future__ import annotations

import math
from enum import Enum
from typing import Literal, Optional, Union


Magnitude = Literal["large", "medium", "small", "negligible"]


class EffectSizeKind(Enum):
    """Canonical effect-size families. Maps many spelling variants → one kind."""
    COHEN_F = "cohen_f"
    COHEN_D = "cohen_d"          # also Hedges' g (same thresholds)
    MCFADDEN_R2 = "mcfadden_r2"
    ICC = "icc"
    AUC = "auc"
    ETA_SQUARED = "eta_squared"  # η², partial η², ω², ε² — variance-explained family
    R_SQUARED = "r_squared"      # OLS R²
    RANK_BISERIAL = "rank_biserial"  # also Kendall's W
    CORRELATION_R = "correlation_r"  # Pearson r, Spearman ρ
    CRAMER_V = "cramer_v"


# Lookup is order-sensitive: more specific patterns first. ``cohen's f`` must
# match COHEN_F before any generic "cohen" substring routes to COHEN_D.
_PATTERN_RULES: tuple[tuple[tuple[str, ...], EffectSizeKind], ...] = (
    # Cohen's f — specific exact / prefix patterns. NOT a generic substring,
    # otherwise "cohen_d" would erroneously match (contains "cohen" + "_" + …).
    (("cohen's f", "cohen f", "cohens_f", "cohen_f"), EffectSizeKind.COHEN_F),
    # Cohen's d / Hedges' g — substrings safe because COHEN_F handled above.
    (("cohen_d", "cohen's d", "cohen d", "hedge", "hedges_g"), EffectSizeKind.COHEN_D),
    # McFadden pseudo-R² — distinct scale from OLS R², must precede R² match.
    (("mcfadden", "pseudo_r", "pseudo r", "pseudo-r"), EffectSizeKind.MCFADDEN_R2),
    # ICC — exact + prefix (icc2, icc3k, …).
    (("icc",), EffectSizeKind.ICC),
    # AUC / ROC.
    (("auc", "roc"), EffectSizeKind.AUC),
    # eta family — η², partial η², ω², ε². Must precede generic R² to avoid
    # "partial_eta_squared" matching r-squared via the "squared" suffix.
    (("eta", "omega", "epsilon", "η", "ω"), EffectSizeKind.ETA_SQUARED),
    # OLS R².
    (("r_squared", "r-squared", "r²", "r2"), EffectSizeKind.R_SQUARED),
    # Rank-biserial, Kendall's W.
    (("rank_biserial", "rank biserial", "kendall"), EffectSizeKind.RANK_BISERIAL),
    # Correlation r / ρ.
    (("rho", "pearson", "spearman", "correlation"), EffectSizeKind.CORRELATION_R),
    # Cramer's V.
    (("cramer", "cramér"), EffectSizeKind.CRAMER_V),
)

# Single-character exact matches — separate table because substring fallback
# would over-match (e.g. "r" inside "correlation"). Order matters here too:
# "f" before "g" before "d" so an ambiguous "f" lands on Cohen f not future
# additions.
_EXACT_RULES: dict[str, EffectSizeKind] = {
    "f": EffectSizeKind.COHEN_F,
    "d": EffectSizeKind.COHEN_D,
    "g": EffectSizeKind.COHEN_D,
    "w": EffectSizeKind.RANK_BISERIAL,
    "r": EffectSizeKind.CORRELATION_R,
    "ρ": EffectSizeKind.CORRELATION_R,
    "v": EffectSizeKind.CRAMER_V,
}


def canonicalize(effect_type: Optional[str]) -> Optional[EffectSizeKind]:
    """Map a free-form effect-size type label to its canonical kind.

    Returns ``None`` when the label is empty or unrecognised. Matching is
    case-insensitive and whitespace-trimmed.
    """
    if not effect_type:
        return None
    et = str(effect_type).lower().strip()
    if not et:
        return None
    if et in _EXACT_RULES:
        return _EXACT_RULES[et]
    for patterns, kind in _PATTERN_RULES:
        for pat in patterns:
            if pat in et:
                return kind
    # Special handling: "f (" → Cohen f formatted as "f (1, 23)" or similar.
    if et.startswith("f ("):
        return EffectSizeKind.COHEN_F
    return None


# Thresholds per kind. Tuple = (small, medium, large) — exclusive lower bounds
# for each band; values below ``small`` are negligible.
_THRESHOLDS: dict[EffectSizeKind, tuple[float, float, float]] = {
    EffectSizeKind.COHEN_F:        (0.10, 0.25, 0.40),  # Cohen 1988
    EffectSizeKind.COHEN_D:        (0.20, 0.50, 0.80),  # Cohen 1988
    EffectSizeKind.MCFADDEN_R2:    (0.10, 0.20, 0.40),  # McFadden 1979
    EffectSizeKind.ICC:            (0.50, 0.75, 0.90),  # Koo & Li 2016
    EffectSizeKind.AUC:            (0.70, 0.80, 0.90),  # Hosmer-Lemeshow 2000
    EffectSizeKind.ETA_SQUARED:    (0.01, 0.06, 0.14),  # Cohen 1988
    EffectSizeKind.R_SQUARED:      (0.02, 0.13, 0.26),  # Cohen f²-derived
    EffectSizeKind.RANK_BISERIAL:  (0.10, 0.30, 0.50),  # Cohen r conventions
    EffectSizeKind.CORRELATION_R:  (0.10, 0.30, 0.50),  # Cohen 1988
    EffectSizeKind.CRAMER_V:       (0.10, 0.30, 0.50),  # scaled by df* — see magnitude()
}


def magnitude(
    effect_size: Union[int, float, None],
    kind: Optional[EffectSizeKind],
    *,
    df_star: Optional[int] = None,
) -> Optional[Magnitude]:
    """Return Cohen-style magnitude label for ``effect_size`` interpreted as ``kind``.

    Args:
        effect_size: Numeric effect size. ``None`` / NaN / non-numeric → ``None``.
        kind: Canonical kind. ``None`` → ``None`` (caller can't classify).
        df_star: For ``CRAMER_V`` on >2x2 tables, ``df* = min(rows-1, cols-1)``.
                 Cohen (1988) scales thresholds by ``1 / sqrt(df*)``. Defaults to
                 1 (unscaled 2x2 case).

    Returns:
        ``"large" | "medium" | "small" | "negligible" | None``.
    """
    if kind is None:
        return None
    if not isinstance(effect_size, (int, float)):
        return None
    try:
        if math.isnan(float(effect_size)):
            return None
    except (TypeError, ValueError):
        return None
    es = abs(float(effect_size))
    small, medium, large = _THRESHOLDS[kind]

    if kind is EffectSizeKind.CRAMER_V:
        d = df_star if isinstance(df_star, int) and df_star >= 1 else 1
        scale = 1.0 / math.sqrt(d)
        small *= scale
        medium *= scale
        large *= scale

    if es >= large:
        return "large"
    if es >= medium:
        return "medium"
    if es >= small:
        return "small"
    return "negligible"


def classify(
    effect_size: Union[int, float, None],
    effect_type: Optional[str],
    *,
    df_star: Optional[int] = None,
) -> Optional[Magnitude]:
    """Convenience: ``canonicalize`` + ``magnitude`` in one call.

    Returns ``None`` when the type is unrecognised — callers should treat that
    as "no badge", not "negligible".
    """
    return magnitude(effect_size, canonicalize(effect_type), df_star=df_star)
