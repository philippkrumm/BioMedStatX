"""Layperson-readable tooltips for statistical output rows.

Extracted from ``html_exporter.py`` so the explanatory copy can be edited,
reviewed, and translated without touching report rendering code. Wording aims
to balance technical correctness with accessibility for non-statisticians.
"""
from __future__ import annotations

from typing import Mapping


STAT_ROW_DESCRIPTIONS: Mapping[str, str] = {
    "Test": "Name of the statistical procedure used to compute the p-value.",
    "Model type": "Underlying statistical model family (e.g. ANOVA, regression, non-parametric).",
    "Statistic": (
        "Number the test computes from your data to decide significance. "
        "For F, t, χ² and z, larger magnitude = stronger signal against the "
        "null hypothesis. For rank-based statistics (U, W), what matters is "
        "how far the value sits from its null-expected midpoint, not its raw size."
    ),
    "Odds Ratio (first predictor)": (
        "Multiplicative change in the odds of the outcome per unit increase of "
        "the predictor. OR=1 means no effect; >1 increases the odds, <1 decreases them."
    ),
    "Coefficient (first predictor)": (
        "Regression coefficient for the first predictor on the model's link scale."
    ),
    "p-value": (
        "How surprising your data are if there is really no effect. Formally: "
        "the probability of obtaining a test statistic at least as extreme as the "
        "observed one when the null hypothesis (no difference / no effect) is true. "
        "Small p = data unlikely under H₀. It is NOT the probability that H₀ is true. "
        "Conventional thresholds: * p<0.05, ** p<0.01, *** p<0.001."
    ),
    "Adjusted p-value": (
        "p-value after correction for multiple testing (e.g. Benjamini–Hochberg FDR). "
        "Use this when several hypotheses are tested simultaneously. "
        "Note: BH assumes independent or positively dependent tests. For arbitrary dependence, "
        "Benjamini-Yekutieli is technically cleaner, but BH remains a valid choice for exploratory screening."
    ),
    "Effect size": (
        "Standardised magnitude of the effect, independent of sample size. "
        "Tells you whether a significant result is also practically meaningful."
    ),
    "Effect size type": (
        "Which effect-size metric is reported (e.g. Cohen's d, η², partial η², r). "
        "Each has its own scale and benchmarks."
    ),
    "Confidence interval": (
        "Plausible range for the true value, given your data. Formally (95%): "
        "if you repeated the experiment many times, 95% of the intervals computed "
        "this way would contain the true parameter. Narrower = more precise estimate. "
        "Wide intervals signal limited sample size or high variability."
    ),
    "Degrees of freedom 1": (
        "Numerator (between-groups) df. For ANOVA: number of groups − 1. "
        "Reflects how many independent contrasts the model fits."
    ),
    "Degrees of freedom 2": (
        "Denominator (residual/error) df. NOT the number of groups. For ANOVA: "
        "N − k (total observations minus groups). Roughly: how much information "
        "remains after fitting. Larger df₂ → more precise estimate."
    ),
    "Transformation": (
        "Mathematical transformation applied to the response before testing "
        "(e.g. log10, sqrt) to better satisfy normality / variance assumptions."
    ),
    "Post-hoc test": (
        "Follow-up pairwise comparison procedure used after an overall significant "
        "omnibus result."
    ),
    "Wald z-statistic (first predictor)": (
        "z-statistic for the first predictor in a logistic / GLM model. "
        "|z| > 1.96 corresponds to p < 0.05."
    ),
    "Sample size (n)": "Number of observations entering the analysis.",
    "Correlation method": (
        "Which correlation coefficient was computed (Pearson = linear, "
        "Spearman/Kendall = rank-based)."
    ),
    "Interpretation": "Human-readable summary of the result's direction and magnitude.",
    "Variable transformations": (
        "Transformations applied to predictor (X) and response (Y) before fitting the model."
    ),
    "β Interpretation": (
        "Plain-language interpretation of the regression coefficient on the chosen scale."
    ),
}


def stat_row_info(label: str) -> str:
    """Return tooltip text for ``label``, or empty string when no copy exists."""
    return STAT_ROW_DESCRIPTIONS.get(label, "")
