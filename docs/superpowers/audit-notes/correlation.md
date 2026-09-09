# Audit note: `correlation` recipe

Recipe location: `src/core/help_content.py:388` (`"id": "correlation"`).

**Outcome: no recipe change.** Every factual claim in the recipe is accurate
against the code. This recipe is the spec's named benchmark for recipe economy
(design doc line 66: "match the economy of the existing `correlation` recipe's
one-sentence 'the app picks Pearson or Spearman automatically' treatment"), and
the audit confirms it earns that role: the buckets, data layout, and the
one-sentence auto-selection description all match the code. The exact
N-tier / skewness-kurtosis selection rule is recorded below for the human
record and deliberately kept out of the shipped text per the recipe-economy
rule.

The citation anchor (symbol name or quoted string) is authoritative; the line
number is only a navigation hint.

## Ground-truth dispatch path (traced, not guessed)

Bucket-to-test routing happens in `_ap_build_analysis_context`
(`src/autopilot/statistical_analyzer_autopilot_pipeline.py`). Step 4 of the
upgrade chain (`:1248`-`1267`) fires only when `len(factor_columns) == 1 and not
subject_column` (`:1251`) and the single factor is continuous
(`_is_continuous`, `correlation_models.py:46`, called via
`_corr_is_continuous` at pipeline `:1254`). With no covariate and the
regression toggle off it sets `context["inferred_test"] = "correlation"`
(`:1264`); a covariate present forces `"linear_regression"` (`:1256`); the
toggle on forces `"linear_regression"` (`:1264`).

The dispatch then reaches the clinical block in
`AnalysisManager.analyze` (`src/analysis/analysis_core.py`), which for
`clinical_test == 'correlation'` instantiates `model = CorrelationModel()`
(`analysis_core.py:707`) and calls `model.fit(analysis_df, x_col=..., y_col=...,
method='auto', x_transform=..., y_transform=...)` (`:708`-`715`). No `alpha`
is passed, so `CorrelationModel.fit`'s default `alpha=0.05`
(`correlation_models.py:225`) is used. The statistics live in
`CorrelationModel` (`correlation_models.py:198`).

## What the model actually does (verified)

- **Coefficient selection (`method='auto'`)** — `CorrelationModel.fit`
  (`correlation_models.py:270`-`297`). The decision is **N-tier gated on
  skewness / excess kurtosis**, not on the Shapiro-Wilk p-value:
  - `n < MIN_N_SMALL` (MIN_N_SMALL = 20, `validators.py:10`) → always
    **Spearman** (`correlation_models.py:283`-`284`), regardless of shape.
  - `20 <= n < 100` → **Pearson** iff `|skew_x|,|skew_y| <= 1.0` and
    `|kurt_x|,|kurt_y| <= 2.0` (Fisher excess kurtosis), else Spearman
    (`:285`-`291`).
  - `n >= 100` → **Pearson** unless extreme asymmetry (`|skew| > 2.0` or
    `|excess kurtosis| > 4.0` for either variable) → Spearman (`:292`-`297`).
  - Shapiro-Wilk **is** computed on each variable (`scipy_stats.shapiro`,
    `:277`-`278`) capped at 5000 values (`x_vals[:5000]`, `y_vals[:5000]`) and
    stored as `both_normal_sw` in the `normality_check` dict (`:279`,
    `:333`), but the **method choice does not use `px`/`py`** — only the
    skew/kurtosis thresholds and the N tier. So the recipe's plain-language
    "based on whether the data is normally distributed" describes a distribution-
    shape check (skewness/kurtosis are normality diagnostics); it is a fair lay
    summary, not a claim that Shapiro-Wilk's p-value drives the switch.
- **Explicit `method='pearson'`/`'spearman'`** bypasses auto and uses the named
  method (`:339`-`340`). The autopilot always passes `method='auto'`
  (`analysis_core.py:712`), so the live path is always auto-selected.
- **Statistics computed:** Pearson via `scipy_stats.pearsonr(x_vals, y_vals)`
  on transformed values (`:344`); Spearman via
  `scipy_stats.spearmanr(x_raw, y_raw)` on **raw** values, discarding any
  transform bookkeeping (`:347`-`353`). For `20 <= n < 100` Spearman, the
  p-value is replaced by a t-approximation
  `t = r*sqrt((n-2)/(1-r^2))`, two-sided `t.sf` (`:356`-`361`).
- **Confidence interval:** 95% (from `alpha=0.05`) via Fisher z-transform,
  `_fisher_z_ci` (`correlation_models.py:59`-`80`). Pearson SE = `1/sqrt(n-3)`;
  Spearman uses the wider Bonett-Wright SE = `sqrt((1 + r^2/2)/(n-3))`
  (`:71`-`74`). Requires `n >= 4` else `(None, None)` (`:66`-`67`).
- **Minimum n:** `fit` raises `ValueError` for `n < 4` after pairwise deletion
  ("Correlation requires at least 4 complete value pairs", `:246`-`250`).
- **Missing data:** pairwise deletion — only rows with no NaN in `x_col`/`y_col`
  are kept (`df.dropna(subset=[x_col, y_col])`, `:243`).
- **Reported statistics** (`as_results_dict`, `:414`-`449`): test label
  `"Correlation (Pearson|Spearman)"`, `r`, `p_value`, `ci_lower`/`ci_upper`
  (95% CI), `n`, `alpha`, a strength/direction `interpretation`
  (`_interpret`, `:369`-`383`), and the scatter `association_points`.

## Claim table

| # | Claim (title/summary/html) | Verdict | Citation |
|---|-----------------------------|---------|----------|
| 1 | Title "Do two measurements go up and down together? (Correlation)" | correct | test label `f"Correlation ({self._method_used.capitalize()})"` (`correlation_models.py:419`, `CorrelationModel.as_results_dict`) |
| 2 | Summary "Measure the relationship between two continuous variables. No groups." | correct | routing requires a single continuous factor and no grouping/subject column: `len(factor_columns) == 1 and not subject_column` + `_is_continuous` (pipeline `:1251`-`1254`; `_is_continuous`, `correlation_models.py:46`) |
| 3 | "two number columns, one measurement per subject in each"; both columns are numbers | correct | correlation is only reached when the factor is numeric with >10 unique values (`_is_continuous`, `correlation_models.py:52`-`56`); values cast to float in `fit` (`:258`-`259`) |
| 4 | "when one goes up, does the other tend to go up (or down) too?" — strength and direction | correct | `_interpret` returns direction (positive/negative from sign of r) and strength band (`correlation_models.py:369`-`383`) |
| 5 | "does not give a prediction formula and does not prove causation; for a slope or to control other variables use Regression" | correct | CorrelationModel reports only `r`/`p`/CI, no slope (`as_results_dict`, `:414`-`449`); a slope requires `SimpleLinearRegressionModel` (`correlation_models.py:456`); a covariate or the regression toggle re-routes to `linear_regression` (pipeline `:1256`, `:1264`). Causation is a domain caution, not code-checkable; no code contradicts it. |
| 6 | Data layout: "One row per subject. Two numeric columns, one per variable." | correct | pairwise deletion over the two named columns, one value pair per row (`df.dropna(subset=[x_col, y_col])`, `:243`; scatter points built per row, `:262`) |
| 7 | Common mistake: averaged groups instead of individual values loses power / changes apparent strength | correct (domain reasoning; code operates on individual rows) | `fit` correlates individual rows after pairwise deletion (`:243`, `:344`/`:347`); the 3-row group-mean example has n=3 and would use the Spearman small-n branch. Aggregating rows into group means is a standard power/ecological-fallacy caution; no code contradicts it. |
| 8 | Bucket mapping — DV = outcome; Factor 1 = other numeric variable; Covariates empty (else Regression); Subject ID empty (else Mixed Model); Factor 2/Filter not needed | correct | one factor + no subject + no covariate + toggle off → `"correlation"` (pipeline `:1251`-`1264`); a covariate → `"linear_regression"` (`:1256`); a subject column fails the `not subject_column` guard (`:1251`) so correlation is never inferred — with a within-factor it routes toward LMM/mixed handling (`:1216`-`1235`). x/y bound from `x_variable`/dv columns at the clinical call (`analysis_core.py:708`-`715`). |
| 9 | "The app picks Pearson or Spearman automatically based on whether the data is normally distributed. You do not need to choose." | correct (fair lay summary; see nuance) | `method='auto'` selection via N-tier + skewness/kurtosis thresholds (`correlation_models.py:281`-`297`); autopilot always passes `method='auto'` (`analysis_core.py:712`). Skewness/kurtosis (and the Shapiro-Wilk also computed at `:277`-`278`) are normality diagnostics, so "normally distributed" is an accurate plain-language description. Kept to one sentence per the recipe-economy rule; the exact N<20-forces-Spearman rule is recorded above, not in the recipe. |
| 10 | Checklist: both columns numbers, no text/labels; one row per subject, no repeats; raw values not averages/bins; Covariates and Subject ID empty | correct | numeric requirement (claim 3); one-row-per-subject pairwise design (claim 6); raw individual values (claim 7); empty Covariates/Subject ID keep the routing on `correlation` (claim 8) |

## Alpha / CI-level control check

Default `alpha` is 0.05. The clinical dispatch calls `model.fit(...)` without an
`alpha` argument (`analysis_core.py:708`-`715`), so `CorrelationModel.fit`'s
default `alpha=0.05` (`correlation_models.py:225`) is used, stored as
`self._alpha` (`:239`), and threaded into both the auto-selection normality
thresholds (`px > alpha`, `:279`) and the Fisher-z CI half-width
(`z_crit = norm.ppf(1 - alpha/2)`, `_fisher_z_ci`, `:75`), giving a 95% CI. The
recipe prints no numeric alpha or CI level, so there is nothing to contradict.

## Data-structure control check

The recipe's layout (two numeric columns, one value pair per subject row)
matches `CorrelationModel.fit`, which reads exactly `x_col`/`y_col`, applies
pairwise deletion (`:243`), and requires `n >= 4` (`:246`-`250`). No auto-pivot
applies: correlation is a single-continuous-factor, no-subject signature
(pipeline `:1251`), which `_ap_maybe_pivot` (wide paired/repeated detection)
does not touch. The "averaged groups" example is correctly flagged as the wrong
shape (claim 7).

## Unclear / possible code bug

None affecting the recipe. Two internal observations, neither contradicting the
recipe text (which names no specific test), recorded for completeness:

1. **`method='auto'` never uses the Shapiro-Wilk p-value it computes.**
   `fit` runs `scipy_stats.shapiro` on both variables (`correlation_models.py:277`-`278`)
   and stores `both_normal_sw` in `normality_check` (`:279`, `:333`), but the
   Pearson-vs-Spearman decision is driven only by the N tier and
   skewness/excess-kurtosis thresholds (`:281`-`297`); `px`/`py` feed only the
   reported diagnostic dict, not the branch. This is an internal design choice
   (skew/kurtosis gating is deliberate — Shapiro over-rejects at large n), not a
   bug, and the recipe's "normally distributed" wording is consistent with it.
   No recipe or code change.

2. **Spearman p-value is recomputed with a t-approximation only for
   `20 <= n < 100`** (`correlation_models.py:356`-`361`); for `n < 20` and
   `n >= 100` the `scipy_stats.spearmanr` p-value is kept as-is. This is an
   internal accuracy refinement, invisible to the recipe. Recorded, no change.

3. **The `CorrelationModel` class docstring is stale and describes the wrong
   mechanism** (`correlation_models.py:199`-`203`): it says `method='auto'`
   "applies Shapiro-Wilk to both variables and uses Pearson when both are
   normally distributed (p > alpha)," but the actual `fit` logic is the
   skew/excess-kurtosis N-tier gating described in observation 1 above, which
   does not read the Shapiro p-value at all. This is a genuine code
   documentation bug (not a recipe issue — the recipe never named a specific
   mechanism), found during spec review of this task. Worth fixing at the code
   level: update the docstring to match the real `fit` logic. No recipe or
   code change made here, per the audit's scope.
