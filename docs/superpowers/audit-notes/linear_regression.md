# Audit note: `linear_regression` recipe

Recipe location: `src/core/help_content.py:444` (`"id": "linear_regression"`).

**Outcome: recipe changed.** Three edits: (1) corrected the checkbox label the
recipe quotes for triggering simple regression (it was in English; the live
widget label is German), (2) completed the transform dropdown list (recipe named
only log10 and sqrt; the widget also offers `log10(x+1)` and `boxcox`), and
(3) added that the report also shows R² alongside the primary predictor's
coefficient and p-value.

**Correction (post spec-review):** an earlier version of edit (3) went further
and claimed the report shows per-coefficient 95% confidence intervals and both
R² and adjusted R². That overstated what reaches the user. `coefficient_table`
(built in `as_results_dict`, `correlation_models.py:780`-`790`, with CIs from
`conf_int()`) is written into the result dict but never read by any exporter;
grep of `report_stat_rows.py`, `report_association.py`, `report_methods.py`,
and `report_charts.py` finds no consumer of `coefficient_table` for
`LinearRegression`. The R² / adjusted R² rows at `report_stat_rows.py:117`-`118`
live inside `_build_ancova_statistical_rows`, gated on `model_type == "ANCOVA"`
(`report_stat_rows.py:559`-`560`) and unreachable for `model_type ==
"LinearRegression"`. What actually reaches the user for linear regression is
the primary predictor's raw `p_value` and a single plain R² shown as the report
hero's "Effect size" row (`effect_label = results.get("effect_size_type") or
"Effect size"`, `html_exporter.py:284`, with `effect_size_type = "R_squared"`
set at `correlation_models.py:835`). The recipe wording below now reflects only
that: the coefficient's p-value and a plain R², no coefficient CI, no adjusted
R² claim.

The plan's original ground-truth guess ("some regression path in
`src/analysis/`") was wrong. The real model is
`SimpleLinearRegressionModel` in `src/analysis/correlation_models.py:456`, the
sibling class to `CorrelationModel` in the same file.

The citation anchor (symbol name or quoted string) is authoritative; the line
number is only a navigation hint.

## Ground-truth dispatch path (traced, not guessed)

Bucket-to-test routing happens in `_ap_build_analysis_context`
(`src/autopilot/statistical_analyzer_autopilot_pipeline.py`). Step 4 of the
upgrade chain (`:1248`-`1267`) fires only when `len(factor_columns) == 1 and not
subject_column` (`:1251`) and the single factor is continuous
(`_is_continuous`, `correlation_models.py:46`, called via
`_corr_is_continuous` at pipeline `:1254`). Two ways to reach
`linear_regression`:

- **Covariate present** → forced to `"linear_regression"` (pipeline `:1255`-`1256`),
  regardless of the toggle. This is the "multiple regression" trigger.
- **No covariate, regression toggle checked** → `"linear_regression"`; unchecked
  → `"correlation"` (pipeline `:1259`-`1264`, gated on
  `self.corr_regression_toggle.isChecked()`). This is the "simple regression"
  trigger.

`context["x_variable"] = factor_columns[0]` is set here (`:1265`). Transforms are
read from the two combo boxes into the context at pipeline `:1276`-`1282`
(`context['x_transform']`, `context['y_transform']`), defaulting to `'none'`.

The dispatch then reaches the clinical block in `AnalysisManager.analyze`
(`src/analysis/analysis_core.py`). For `clinical_test == 'linear_regression'` it
binds `x_col = analysis_context.get('x_variable') or factor_columns[0]`
(`:693`), `y_col = value_cols[0]` (`:694`), instantiates
`model = SimpleLinearRegressionModel()` (`:718`) and calls
`model.fit(analysis_df, x_col=x_col, y_col=y_col, covariates=covariates or None,
x_transform=..., y_transform=...)` (`:719`-`722`). No `alpha` is passed, so
`SimpleLinearRegressionModel.fit`'s default `alpha=0.05`
(`correlation_models.py:487`) is used. `covariates` come from the Covariates
bucket (`analysis_core.py:547`, threaded from `analysis_context`).

## What the model actually does (verified)

- **Estimator** — ordinary least squares via `statsmodels.formula.api.ols`
  imported inside `fit` (`import statsmodels.formula.api as smf`,
  `correlation_models.py:499`) and fitted at `:575`
  (`self.result = smf.ols(formula, data=self._df).fit()`). The formula is
  `Y ~ X + cov1 + cov2 + ...` built from the sanitized column names
  (`:572`-`573`). Covariates are added as additional continuous predictor terms
  (`:520`, `:544`, `:572`).
- **Statistics computed / reported** (`as_results_dict`,
  `correlation_models.py:772`-`870`):
  - Primary predictor: `beta` (`self.result.params[self._x]`, `:795`),
    `p_value` (`self.result.pvalues[self._x]`, `:796`), `statistic` = t-value
    with `statistic_type="t"` (`:797`, `:832`-`833`).
  - Full `coefficient_table` (`:780`-`790`): for every parameter (intercept, X,
    each covariate) it emits `coefficient`, `std_err` (`self.result.bse`),
    `t_value`, `p_value`, and `ci_lower`/`ci_upper` from
    `self.result.conf_int()` (`:779`, `:788`-`789`). `conf_int()` uses the
    default 95% level (alpha 0.05).
  - Model fit: `r_squared` (`self.result.rsquared`, `:837`), `r_squared_adj`
    (`:838`), overall F-test `f_statistic` / `f_p_value` (`:799`-`800`,
    `:839`-`840`), `aic`/`bic` (`:841`-`842`), `n_observations` (`:843`).
  - Effect size: `effect_size = float(self.result.rsquared)` with
    `effect_size_type = "R_squared"` (`:834`-`835`). So the reported effect size
    is **R²**. `effect_sizes.py` maps `"r_squared"` →
    `EffectSizeKind.R_SQUARED` (`effect_sizes.py:40`, `:64`) with Cohen
    f²-derived magnitude bands `(0.02, 0.13, 0.26)` (`effect_sizes.py:120`).
  - Test label: `"Linear Regression (OLS)"`, `model_type = "LinearRegression"`
    (`:828`-`829`).
- **Heteroscedasticity → HC3 robust standard errors (verified).**
  After the OLS fit, `fit` runs `self.diagnostics()` and reads the Breusch-Pagan
  p-value (`correlation_models.py:577`-`580`, comment
  "Run diagnostics on OLS to check homoscedasticity for HC3 switch"). If
  `n >= 20 and bp_p is not None and bp_p < alpha` it replaces the result with
  `self.result.get_robustcov_results(cov_type='HC3')` and sets
  `self._cov_type = "HC3"` (`:582`-`588`); otherwise `cov_type` stays
  `"nonrobust"`. So the standard errors (and therefore the coefficient p-values
  and CIs, where those are used) **do** adapt to detected heteroscedasticity,
  but only at n ≥ 20 and only when Breusch-Pagan is significant at the alpha
  level. `cov_type` is emitted in the result dict (`:869`) and **does surface
  to the user**, though not as a table row or an explicit "HC3" label: it drives
  a branch in the interactive decision tree/flowchart.
  `model_type == "LinearRegression"` is one of the types
  `DecisionTreeVisualizer.get_tree_json` (`decisiontreevisualizer.py:164`-`167`)
  delegates to `FlowchartVisualizer.get_tree_json`
  (`flowchartvisualizer.py:428`-`443`), which reads
  `cov_type = str(results.get("cov_type", "") or "").lower()` and
  `is_hc3 = "hc3" in cov_type or cov_type == "robust"` (`:431`-`432`) to choose
  between two qualitative nodes: "Uneven spread detected -> adjusted
  estimation" (`COV_HC3`, `:442`) versus "Even spread confirmed -> standard
  estimation" (`COV_NONROBUST`, `:443`). This tree is wired into both the live
  UI panel (`DecisionTreeVisualizer.get_tree_json` called at
  `statistical_analyzer_autopilot_ui.py:998`) and the exported HTML report
  (`HTMLExporter._build_decision_tree_json` at `html_exporter.py:150`, calling
  `DecisionTreeVisualizer.get_tree_json` at `html_exporter.py:326`). So the HC3
  switch is not a silent internal-only step: the user sees it as a plain-language
  branch label, just not as a "cov_type: HC3" field anywhere. Per the
  recipe-economy rule the recipe body still does not name HC3 or the decision
  tree explicitly (the tree is a separate UI surface from the Help Hub recipe,
  and the recipe never claims a specific SE method), but this is recorded here
  as a correction to the original (incorrect) "never surfaced" claim.
- **Box-Cox on Y optimizes over OLS residuals (verified).** For
  `y_transform == 'boxcox'` (`correlation_models.py:531`-`563`) the model does
  **not** use the marginal Box-Cox of the CorrelationModel-style
  `_apply_transform`. Instead it builds the design matrix (intercept + X +
  covariates) via `patsy.dmatrix` (`:543`-`546`) and calls
  `_optimize_boxcox_for_regression(y_clean, x_matrix)`
  (`:554`; helper at `correlation_models.py:143`), which maximizes the profile
  log-likelihood of the **OLS residuals**, then applies
  `scipy.special.boxcox` with that lambda (`:557`-`560`). Falls back to the
  marginal transform when `len(temp_df) < 5` (`:538`-`540`). The chosen lambda is
  stored as `y_boxcox_lambda` and returned in the result dict (`:562`, `:859`).
  X-transforms (and the non-Box-Cox Y-transforms) go through the shared
  `_apply_transform` (`correlation_models.py:86`-`141`), which supports exactly
  `none`, `log10`, `log10(x+1)`, `sqrt`, `boxcox` (`_VALID_TRANSFORMS`, checked
  at `:105`). Non-positive inputs are set to NaN and dropped, no auto-shift
  (`:110`-`141`; docstring `:88`-`91`).
- **Transform reporting to the user.** When a transform is active the report
  adds a "Variable transformations" row (`report_stat_rows.py:638`-`645`), a
  "β Interpretation" row from `coef_interpretation`
  (`correlation_models.py:592`-`649`, `report_stat_rows.py:646`-`651`), and an
  "R² Note" warning that R² is on the transformed scale
  (`report_stat_rows.py:119`-`129`). Box-Cox is offered in the UI dropdown
  (`_TRANSFORMS`, pipeline `:370`) and the Box-Cox interpretation falls through
  to the generic "requires back-transformation" branch
  (`correlation_models.py:643`-`648`).
- **Minimum observations / missing data:** listwise deletion over
  `[x_col, y_col] + covariates` (`df.dropna(subset=all_cols)`,
  `correlation_models.py:504`-`505`); raises `ValueError` when
  `n < max(5, len(all_cols) + 2)` (`:508`-`513`).
- **Assumption diagnostics (non-blocking).** `diagnostics()`
  (`correlation_models.py:722`-`770`) runs Shapiro-Wilk on residuals,
  Breusch-Pagan for homoscedasticity, and Ramsey RESET (power=2) for linearity.
  All three are reported as diagnostics but none block the analysis; the
  Breusch-Pagan p-value additionally drives the HC3 switch above.

## Claim table

| # | Claim (title/summary/html) | Verdict | Citation |
|---|-----------------------------|---------|----------|
| 1 | Title "Predicting a measurement from other measurements (Linear Regression)" / summary "How much does the outcome change per unit of the predictor? Add control variables." | correct | test label `"Linear Regression (OLS)"` (`correlation_models.py:828`, `SimpleLinearRegressionModel.as_results_dict`); slope β reported (`:795`, `:836`); covariates enter as extra predictors (`:572`) |
| 2 | Unlike correlation, regression gives a specific slope number and lets you control for additional variables | correct | β = `self.result.params[self._x]` (`:795`, `:836`); covariates added to the OLS formula (`:520`, `:572`-`573`); CorrelationModel reports only r/p/CI, no slope (`correlation_models.py:414`-`449`) |
| 3 | Simple regression: drag only Factor 1, then tick the checkbox to run regression; without the tick the app runs Correlation | correct (label corrected) | no covariate + toggle checked → `"linear_regression"`, unchecked → `"correlation"` (pipeline `:1259`-`1264`); the widget label is the German string, not the English one the recipe originally quoted (`corr_regression_toggle` = `QCheckBox("Als Lineare Regression analysieren (Y = a + bX)")`, pipeline `:353`). Note: this German string in an otherwise all-English application is itself worth fixing at the code level (`statistical_analyzer_autopilot_pipeline.py:353`) since it is user-facing outside of Help and the codebase has no i18n system to explain it; out of scope for this audit, flagged for the human. |
| 4 | Multiple regression: drag Factor 1 plus one or more Covariates; regression mode activates automatically when Covariates is populated | correct | `if covariate_columns: context["inferred_test"] = "linear_regression"` (pipeline `:1255`-`1256`), fires regardless of the toggle |
| 5 | Transform dropdowns for X or Y appear in regression mode; use for skew/heteroscedasticity | correct (options completed) | transform combos populated from `_TRANSFORMS = ["none", "log10", "log10(x+1)", "sqrt", "boxcox"]` (pipeline `:370`, `:374`-`383`); container shown only in regression mode (`_update_transform_warning`, pipeline `:401`-`408`); recipe previously named only log10/sqrt, now lists all four |
| 6 | Transforming changes interpretation; log-Y is multiplicative; Log-Log β is an elasticity (1% X → β% Y) | correct | `_build_coef_interpretation` log-log branch returns the elasticity sentence (`correlation_models.py:615`-`619`); log10-Y branch returns the multiplicative `10^β` sentence (`:602`-`609`) |
| 7 | Data layout: one row per subject; one numeric outcome column, one or more numeric predictor columns | correct | `x_col`/`y_col` + covariate columns read per row; listwise deletion over all of them (`correlation_models.py:504`-`505`); continuous-factor requirement (`_is_continuous`, pipeline `:1254`) |
| 8 | Bucket mapping: DV = outcome; Factor 1 = main numeric predictor; Covariates = additional numeric controls; Factor 2 / Subject ID empty | correct | `y_col = value_cols[0]` (`analysis_core.py:694`); `x_col = x_variable / factor_columns[0]` (`:693`); covariates from the Covariates bucket (`:547`, `:719`-`720`); routing requires `not subject_column` (pipeline `:1251`), and a within-factor / subject column diverts to LMM/mixed before this step (pipeline `:1216`-`1235`) |
| 9 | "The main output is the coefficient (β)…" with the holding-constant interpretation | correct | β and its holding-others-constant meaning follow directly from the multi-predictor OLS coefficient (`correlation_models.py:795`, `:836`; `_build_coef_interpretation` "no-transform" branch `:597`-`601`) |
| 10 | (added) the report also shows R² alongside the coefficient and p-value | correct (missing feature added; wording corrected post-review) | `effect_size = float(self.result.rsquared)` with `effect_size_type = "R_squared"` (`correlation_models.py:834`-`835`) surfaces via the report hero's "Effect size" row (`effect_label = results.get("effect_size_type") or "Effect size"`, `html_exporter.py:284`). An earlier draft of this claim additionally said the report shows per-coefficient 95% CIs and adjusted R²; that was wrong and has been removed. `coefficient_table` with `ci_lower`/`ci_upper` (`correlation_models.py:780`-`790`) is written to the result dict but has no exporter consumer (checked `report_stat_rows.py`, `report_association.py`, `report_methods.py`, `report_charts.py`). `r_squared_adj` (`:838`) is likewise emitted but the only reader of `report_stat_rows.py:117`-`118` ("R²"/"Adjusted R²" rows) is `_build_ancova_statistical_rows`, gated on `model_type == "ANCOVA"` (`report_stat_rows.py:559`-`560`), never reached for `model_type == "LinearRegression"` (that model type's only extra rows are transform labels and `coef_interpretation`, `report_stat_rows.py:638`-`651`). |
| 11 | Checklist: predictors/covariates numeric; outcome numeric; one row per subject; no column in both Factor 1 and Covariates | correct | numeric continuous predictor (`_is_continuous`, pipeline `:1254`); one row per subject (listwise deletion, `correlation_models.py:504`); a column cannot be both the primary factor and a covariate because `x_col` and `covariates` are distinct bucket slots (`analysis_core.py:693`-`694`, `:719`-`720`) |

## Alpha / CI-level control check

Default `alpha` is 0.05. The clinical dispatch calls `model.fit(...)` without an
`alpha` argument (`analysis_core.py:719`-`722`), so
`SimpleLinearRegressionModel.fit`'s default `alpha=0.05`
(`correlation_models.py:487`) is used and stored as `self._alpha` (`:501`). It
threads into: the assumption-diagnostic thresholds
(`sw_p > self._alpha` etc., `:738`, `:752`, `:765`), the HC3 switch condition
(`bp_p < alpha`, `:583`), and the regression-plot prediction band
(`summary_frame(alpha=self._alpha)`, `:695`). The coefficient CIs use
`self.result.conf_int()` with the statsmodels default 95% level (`:779`). The
recipe now states a 95% CI and does not print a numeric alpha, so there is
nothing to contradict.

## Data-structure control check

The recipe's layout (one outcome column + one or more predictor/covariate
columns, one value row per subject) matches `SimpleLinearRegressionModel.fit`,
which reads exactly `x_col` / `y_col` / `covariates`, applies listwise deletion
over all of them (`correlation_models.py:504`-`505`), and requires
`n >= max(5, k+2)` (`:508`-`513`). No auto-pivot applies: linear regression is a
single-continuous-factor, no-subject signature (pipeline `:1251`), which
`_ap_maybe_pivot` (wide paired/repeated detection) does not touch. A subject
column would fail the `not subject_column` guard and divert the design to
LMM/mixed handling before this branch (pipeline `:1216`-`1235`), consistent with
the recipe's "Subject ID: leave empty" instruction.

## Unclear / possible code bug

None affecting the recipe. Internal observations recorded for the human:

1. **HC3 robust standard errors surface as a decision-tree branch, not a
   labeled field (corrected finding).** When `n >= 20` and Breusch-Pagan is
   significant (`bp_p < alpha`), `fit` swaps in HC3 covariance
   (`correlation_models.py:582`-`588`) and records `cov_type = "HC3"` in the
   result dict (`:869`). This is read by
   `FlowchartVisualizer.get_tree_json` (`flowchartvisualizer.py:431`-`432`,
   `:442`-`443`) and shown to the user as one of two qualitative nodes ("Uneven
   spread detected -> adjusted estimation" vs "Even spread confirmed ->
   standard estimation") in the decision-tree panel
   (`statistical_analyzer_autopilot_ui.py:998`) and the exported HTML report
   (`html_exporter.py:150`, `:326`). An earlier version of this note claimed
   `cov_type` was "not surfaced in any report row" at all; that was wrong,
   found and corrected during spec review. It is still true that no report row
   prints the literal text "HC3" or a numeric comparison of robust vs.
   nonrobust standard errors, so a user who does not open the decision-tree
   view will not learn the SE method switched. Not a correctness bug. Kept out
   of the recipe body per the recipe-economy rule (the decision tree is a
   separate UI surface from the Help Hub recipe).

2. **`coefficient_table` (with per-coefficient SE, t, p, and 95% CI) is
   computed but never rendered.** `as_results_dict` builds it in full
   (`correlation_models.py:780`-`790`) but no exporter reads the
   `coefficient_table` key for `LinearRegression`; only the primary predictor's
   `p_value` (top-level) and the "Effect size" R² row reach the user. This
   looks like dead/unused output, or an intended-but-unfinished report feature
   (a full coefficient table would be useful for multi-predictor models with
   covariates, where the user currently sees only the primary predictor's
   stats). Worth a code-level decision: either wire it into the report or
   remove it. No code change made here, per the audit's scope.

3. **German checkbox label in an all-English application.** The simple
   regression trigger is `QCheckBox("Als Lineare Regression analysieren (Y = a
   + bX)")` (`statistical_analyzer_autopilot_pipeline.py:353`). No i18n system
   exists anywhere in this codebase, so this reads as an untranslated string
   left over from development rather than a deliberate localization choice.
   User-facing outside of Help, so this is worth fixing at the code level (swap
   in the English equivalent, e.g. "Analyze as Linear Regression (Y = a +
   bX)"). Out of scope for this audit; the recipe itself now quotes the actual
   (German) label so the two stay consistent until the code is fixed.

4. **Box-Cox on Y uses a regression-aware lambda; X Box-Cox does not.** The Y
   Box-Cox path optimizes lambda over OLS residuals
   (`_optimize_boxcox_for_regression`, `correlation_models.py:554`), while an X
   Box-Cox goes through the marginal `_apply_transform` /
   `bounded_boxcox_lambda` path (`:529`, `:139`). This asymmetry is intentional
   (residual-based Box-Cox is the standard for the response), invisible to the
   recipe, and recorded only for completeness. No change.

5. **The recipe intentionally omits the assumption diagnostics.** Shapiro-Wilk
   on residuals, Breusch-Pagan, and Ramsey RESET all run non-blocking
   (`diagnostics`, `correlation_models.py:722`-`770`). Naming them in the recipe
   body would violate the recipe-economy rule (design doc lines 58-68), so the
   recipe says only "the report shows R²…" and the named tests live here. No
   recipe change on this point.
