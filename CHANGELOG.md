# Changelog

All notable changes to this project will be documented in this file.

## [2.0.0] - 2026-06-29

Welcome to BioMedStatX 2.0.0! We've done a massive under-the-hood statistical audit to ensure our results are more robust, conservative, and statistically sound than ever before. Here is what's new and what has changed.

### 🛑 Breaking Changes (Behavioral Updates)
- **Smarter Beta Regression P-Values**: When running Beta regressions, the main p-value you see is now a true Omnibus Likelihood-Ratio (LR) test. This gives you a much better overall picture of the model's significance compared to just looking at the first predictor.
- **Conservative Sphericity Defaults**: Safety first! If the data doesn't allow us to formally test for sphericity (e.g., due to incomplete tables), we now conservatively apply the Greenhouse-Geisser correction by default. Previously, we optimistically assumed sphericity was met, which could inflate Type-I error rates.
- **True Hedges' g**: We now strictly apply the $J$-correction factor to all effect sizes labeled as Hedges' g. This ensures small sample biases are properly penalized. (In earlier versions, some Welch's test branches accidentally reported uncorrected Cohen's d under the Hedges label).
- **Strict Dunnett-RM**: Dunnett's test is specifically designed for comparing treatments against a single control group. We've tightened our Repeated Measures implementation to strictly perform control-only comparisons, preventing accidental (and statistically flawed) all-pairwise comparisons.

### 🐛 Bug Fixes & Stability
- **Rich Analysis Logs Are Back**: Fixed a glitch where the beautifully detailed analysis logs were accidentally discarded during standard exports. You'll now see the full story in your HTML reports again.
- **Working Directory Safety**: We've wrapped our standard export paths in robust error handlers. If an export fails, it will no longer contaminate the working directory for subsequent datasets.
- **Convergence Transparency**: Logistic and Beta models now strictly monitor and report their convergence status. If a model fails to converge (or if our Firth penalized fallback fails), you will be clearly warned rather than presented with misleading outputs.
- **Invalid p-Value Guards**: We've added strict boundaries. The system will now gracefully intercept mathematically impossible p-values (like negative numbers or NaNs) and flag them as `invalid`, rather than blindly formatting them as `< 0.001`.

### 🔬 Statistical Corrections
- **Degrees of Freedom Consistency (`ddof=1`)**: We've standardized standard deviation computations across the board (including Cohen's d for Repeated Measures and our bootstrapping methods) to correctly use sample standard deviations (`ddof=1`) instead of population estimators (`ddof=0`).
- **Flexible Confidence Intervals**: Confidence intervals for bootstraps and effect sizes now strictly respect your dynamically chosen `alpha` level, rather than hardcoding a 95% (1.96) cutoff.
