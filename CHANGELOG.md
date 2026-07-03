# Changelog

All notable changes to this project will be documented in this file.

## [2.0.0] - 2026-06-29

This release is the result of a statistical audit. The changes below make the default behavior more conservative and correct some effect-size and p-value computations. Several of these are behavioral changes that can alter reported results, so read the breaking-changes section before upgrading.

### Breaking changes (behavioral)

- Beta regression now reports an omnibus likelihood-ratio (LR) test as the main p-value instead of the p-value of the first predictor. The LR test reflects overall model significance.
- When sphericity cannot be formally tested (for example, with incomplete tables), the Greenhouse-Geisser correction is now applied by default. Earlier versions assumed sphericity was met, which could inflate the Type-I error rate.
- The J-correction factor is now applied to every effect size labeled Hedges' g. Some Welch's test branches previously reported uncorrected Cohen's d under the Hedges label.
- Repeated-measures Dunnett now performs control-only comparisons. Earlier versions could fall through to all-pairwise comparisons, which Dunnett's test is not designed for.

### Bug fixes and stability

- Detailed analysis logs are no longer discarded during standard exports. They appear again in the HTML reports.
- Standard export paths are wrapped in error handlers. A failed export no longer leaves the working directory in a state that affects later datasets.
- Logistic and Beta models now check and report convergence status. If a model fails to converge, or if the Firth penalized fallback fails, the output says so instead of presenting a misleading result.
- Invalid p-values (negative numbers, NaN) are now flagged as `invalid` rather than formatted as `< 0.001`.

### Statistical corrections

- Standard deviation computations now use the sample estimator (`ddof=1`) consistently, including Cohen's d for repeated measures and the bootstrap methods. Earlier code used the population estimator (`ddof=0`) in some places.
- Confidence intervals for bootstraps and effect sizes now use the chosen `alpha` level instead of a hardcoded 95% (1.96) cutoff.
