# Audit note: `logistic_regression` recipe

Recipe location: `src/core/help_content.py:502` (`"id": "logistic_regression"`).

**Outcome: recipe changed.** Three substantive corrections plus one added
feature sentence:

1. **AUC interpretation bands were wrong.** The recipe said "0.5 = chance;
   0.70 to 0.80 = acceptable; above 0.80 = good." The code's actual bands
   (which the user literally sees in a report row) are: below 0.6 poor,
   0.6-0.7 acceptable, 0.7-0.8 good, 0.8-0.9 excellent, 0.9+ outstanding
   (`report_summaries.py:263`-`272`). The recipe now states the code's bands.
2. **The OR "per one-unit increase" interpretation was wrong for Factor 1.**
   Factor 1 predictors are wrapped in `C()` in the model formula
   (`clinical_models.py:1080`), so they are dummy-coded: each OR compares one
   level against the reference level, not a per-unit change. The per-unit
   interpretation the recipe gave is only correct for **covariates**, which
   enter as continuous terms (`clinical_models.py:1081`-`1082`). The recipe now
   distinguishes the two cases and the "what to drag where" example uses a
   categorical Factor 1 (`Treatment`) with continuous covariates (`Age`,
   `BP_baseline`).
3. **The example put a continuous predictor (`Dosage_mg`) in Factor 1**, which
   the code would dummy-code into one OR per dose level. Reworked to a
   genuinely categorical Factor 1 so the example matches what the model does.
4. **(added feature)** one sentence that the report lists every OR with its
   95% CI and shows the ROC curve. Verified all the way to the renderers per
   the Task 8 lesson (see below).

**The plan's original ground-truth guess was corrected in the task prompt** and
independently re-verified here: the real model is `LogisticRegressionModel`
(`src/analysis/clinical_models.py:1041`), which fits via
`smf.glm(formula, data=..., family=sm.families.Binomial())`
(`clinical_models.py:1085`), **not** `sm.Logit` directly. On convergence
failure or detected separation it falls back to Firth Penalized Likelihood via
Newton-Raphson (`_fit_firth_logistic`, `clinical_models.py:1145`; profile CI at
`_firth_profile_ci`, `:1251`).

The citation anchor (symbol name or quoted string) is authoritative; the line
number is only a navigation hint.

## Ground-truth dispatch path (traced, not guessed)

Binary-DV detection and routing happen in `_ap_build_analysis_context`
(`src/autopilot/statistical_analyzer_autopilot_pipeline.py`). A binary outcome
is detected at `:1119`-`1124` (2 unique values that are 0/1-or-two-strings and
whose column name is not a grouping hint), setting
`context["outcome_type"] = "binary"` (`:1126`). The clinical-upgrade block then
forces `context["inferred_test"] = "logistic_regression"` and this **overrides
everything** (`:1204`-`1205`, comment "Binary DV -> Logistic Regression
(overrides everything)"). Live bucket-hint routing to the recipe uses
`_ap_is_binary_outcome_for_help` (`:717`-`736`) and returns
`"logistic_regression"` from both the Factor 1 and Covariates buckets when the
DV is binary (`:772`-`773`, `:777`-`778`).

The dispatch reaches the clinical block in `AnalysisManager.analyze`
(`src/analysis/analysis_core.py`). The block is gated at `:540`
(`if kwargs.get('test') in (..., 'logistic_regression', ...)`), and for
`clinical_test == 'logistic_regression'` it instantiates
`model = LogisticRegressionModel()` (`:626`), reads
`predictors = analysis_context.get('factor_columns', [])` (`:627`), and calls
`model.fit(df, dv=value_cols[0], predictors=predictors, covariates=covariates
or None)` (`:628`), then `test_results = model.as_results_dict()` (`:629`).
`fit` takes no `alpha` argument, so there is no configurable alpha; the CIs are
fixed at 95% (see the alpha/CI control check below).

## What the model actually does (verified)

- **Estimator.** `smf.glm(formula, data=self._df, family=sm.families.Binomial())`
  fitted with `.fit()` (`clinical_models.py:1085`-`1086`). The formula is
  `DV ~ C(pred1) + C(pred2) + ... + cov1 + cov2 + ...`
  (`:1080`-`1083`): **Factor 1 predictors are categorical (`C()`)**, covariates
  are continuous. Empirically confirmed: a numeric `Dosage_mg` placed in
  Factor 1 produces params `C(Dosage_mg)[T.20]`, `[T.30]`, ... (one OR per level
  vs the reference), while an `Age` covariate produces a single per-unit OR.
- **Outcome encoding.** Requires exactly 2 unique DV values or raises
  `ValueError("Logistic regression requires exactly 2 outcome levels, ...")`
  (`clinical_models.py:1074`-`1076`). If the two values are not already `{0, 1}`
  it maps `unique_vals[1] -> 1` via `(series == unique_vals[1]).astype(int)`
  (`:1077`-`1078`). So it accepts **any two distinct values** (text or numeric);
  0/1 is the clearest but not required.
- **Convergence check + Firth fallback.** After the GLM fit it flags separation
  when any coefficient SE exceeds `SEPARATION_BSE_THRESHOLD = 5.0`
  (`clinical_models.py:1088`-`1097`). If `not self.result.converged or
  has_large_se` (`:1106`), it fits Firth Penalized Likelihood
  (`_fit_firth_logistic`, `:1145`) and sets
  `self._model_variant = "Firth Penalized Likelihood"` (`:1110`). If the Firth
  solver itself fails, it keeps the standard logit result and sets
  `_firth_failed = True` (`:1127`), which forces `converged = False` in the
  result dict (`:1528`-`1529`).
- **Odds ratios + CI.** `odds_ratios()` (`clinical_models.py:1302`-`1367`)
  returns OR = `exp(coef)` per non-intercept parameter. Standard path: 95% Wald
  CI from `self.result.conf_int()` (`:1353`, `:1360`-`1361`). Firth path: 95%
  penalized **profile-likelihood** CI (`_firth_profile_ci(..., alpha=0.05)`,
  `:1327`-`1330`), falling back to Wald only if the root search fails
  (`:1334`-`1337`); p-values prefer the penalized LR test over Wald
  (`_firth_plr_pvals`, `:1116`-`1125`, `:1321`, `:1347`-`1350`). The chosen
  `ci_method`/`p_value_method` are carried per row and surfaced in the report
  subtitle (`report_association.py:52`-`63`).
- **AUC / ROC.** `roc_data()` (`clinical_models.py:1466`-`1500`) builds FPR/TPR
  over thresholds and integrates via `np.trapezoid`/`np.trapz` to get AUC. In
  the result dict `effect_size = roc["auc"]`, `effect_size_type = "AUC"`
  (`:1544`-`1545`).
- **Other reported fields** (`as_results_dict`, `:1502`-`1561`): `model_variant`
  (`:1540`), `p_value` = primary predictor's p (`:1541`), `statistic` = primary
  OR with `statistic_type = "odds_ratio"` (`:1542`-`1543`), the full
  `odds_ratios` table (`:1546`), Hosmer-Lemeshow (`:1547`), `roc_data` (`:1548`),
  Brier score / calibration slope / intercept (`:1549`-`1551`), McFadden
  `pseudo_r_squared` (`:1553`, standard-ML only), AIC/BIC/log-likelihood
  (standard-ML only, `:1554`-`1556`), `n_observations` (`:1557`),
  `predictors_used`/`covariates_used` (`:1558`-`1559`). `model_type =
  "LogisticRegression"`, `test = "Logistic Regression"` (`:1536`-`1537`).
- **Missing data.** Listwise deletion over `[dv] + predictors + covariates`
  (`df.dropna(subset=all_cols)`, `clinical_models.py:1064`-`1065`).

## What actually reaches the user (traced to the renderers, per Task 8)

- **OR table with 95% CI** — YES. `_build_or_table_html`
  (`report_association.py:19`-`69`) renders a 6-column table (Parameter, OR,
  95% CI Lower, 95% CI Upper, z, p-value). It is called for
  `model_type == "LogisticRegression"` from `_build_single_chart_bundle`
  (`report_charts.py:604`, `:606`-`608`), whose `charts` list becomes
  `"chart_blocks"` in the single-report context (`html_exporter.py:99`, `:178`).
- **ROC curve + AUC annotation** — YES. `_build_roc_chart`
  (`report_charts.py:772`-`842`, gated on `model_type == "LogisticRegression"`
  at `:774`) is appended in the same bundle (`:610`-`612`). AUC also appears as
  the report hero's "Effect size" row via
  `effect_label = str(results.get("effect_size_type") or "Effect size")`
  (`html_exporter.py:284`, with `effect_size_type = "AUC"`), the same mechanism
  the linear_regression audit verified for R².
- **AUC interpretation bands** — YES, and this is what makes the old recipe's
  bands a factual contradiction. `_build_assumption_summary`
  (`report_summaries.py:33`, called at `html_exporter.py:95`, into the
  `assumptions` context key at `:163`) has a `LogisticRegression` branch
  (`:247`) that emits a "Discrimination (AUC interpretation)" row with labels
  Poor (<0.6) / Acceptable (0.6-0.7) / Good (0.7-0.8) / Excellent (0.8-0.9) /
  Outstanding (>=0.9) (`:263`-`280`).
- **Model variant (Standard ML vs Firth)** — YES. The dedicated
  `_build_logistic_statistical_rows` (`report_stat_rows.py:326`-`374`), reached
  via `_build_statistical_rows`'s `model_type == "LogisticRegression"` early
  return (`:568`-`569`), renders a "Model Fit" section including a "Model
  variant" row (`:335`), plus AUC (ROC), Brier score, calibration
  slope/intercept, McFadden pseudo-R², AIC, BIC, N observations. So the Firth
  fallback **is** surfaced to the user, though the recipe body does not name it
  (see recipe-economy note).
- **Convergence / "results may be unreliable"** — YES, via the assumptions
  summary (`report_summaries.py:37`-`46`, `:426`-`429`) and the primary-predictor
  p-value in the Model Fit rows. (Note: the model-agnostic converged row and the
  second `model_type == "LogisticRegression"` branch at
  `report_stat_rows.py:610`-`623` are **unreachable for logistic** because
  `_build_statistical_rows` returns early at `:569`; convergence still surfaces
  through the assumptions summary.)

## Claim table

| # | Claim (title/summary/html) | Verdict | Citation |
|---|-----------------------------|---------|----------|
| 1 | Title "Predicting a yes/no outcome (Logistic Regression)" / summary "outcome is binary: 0/1, yes/no, event/no-event" | correct | `test = "Logistic Regression"`, `model_type = "LogisticRegression"` (`clinical_models.py:1536`-`1537`); binary-only DV (`:1074`-`1076`) |
| 2 | Use when the outcome can only be one of two values (yes/no, 0/1, survived/died) | correct | exactly-2-levels guard (`clinical_models.py:1074`-`1076`); binary-DV detection (`pipeline:1119`-`1126`) |
| 3 | "The app detects this automatically: when the DV has exactly two distinct values, logistic runs without manual selection" | correct | binary DV overrides all other test inference (`pipeline:1204`-`1205`); bucket hints return `"logistic_regression"` (`:772`-`773`, `:777`-`778`) |
| 4 | Data: one row per subject; outcome exactly two values; 0/1 clearest; two text labels work; app picks a reference level | correct (reworded) | listwise deletion per row (`clinical_models.py:1064`-`1065`); any-two-values encoding, `unique_vals[1] -> 1` (`:1077`-`1078`); `C()` uses a reference level for a categorical predictor, and the DV's non-reference level is modeled (`:1078`, `:1080`) |
| 5 | Common mistake: 3+ outcome levels fail; collapse to two | correct | `ValueError` when `len(unique_vals) != 2` (`clinical_models.py:1075`-`1076`) |
| 6 | Bucket mapping: DV binary; Factor 1 = grouping predictor compared level by level; Covariates = numeric per-unit predictors; Factor 2 / Subject empty | correct (was wrong: Factor 1 previously described as "the main numeric predictor" with a per-unit reading) | Factor 1 -> `C(pred)` categorical (`clinical_models.py:1080`); covariates continuous (`:1081`-`1082`); `predictors = factor_columns`, `dv = value_cols[0]`, covariates threaded (`analysis_core.py:627`-`628`); binary DV routes here regardless of Subject/Factor 2, and a subject+within design diverts earlier (`pipeline:1204` overrides after the `:1217` LMM branch is only reached in the `elif`) |
| 7 | OR interpretation: Factor 1 group OR = group vs reference; covariate OR = per one-unit increase; OR<1 lower odds; OR=1 no effect; report lists every OR with 95% CI | correct (was wrong: old text applied per-unit reading to all predictors) | categorical vs continuous formula terms (`clinical_models.py:1080`-`1082`, empirically confirmed); OR = `exp(coef)` (`:1340`, `:1359`); 95% CI (`:1353`, `:1327`-`1330`); OR table rendered with 95% CI columns (`report_association.py:44`-`46`, called from `report_charts.py:606`) |
| 8 | AUC bands: below 0.6 poor, 0.6-0.7 acceptable, 0.7-0.8 good, 0.8-0.9 excellent, 0.9+ outstanding; 0.5 = chance; report shows ROC curve | correct (was wrong: old text said 0.70-0.80 acceptable, >0.80 good) | exact bands in `report_summaries.py:263`-`272`; ROC chart rendered (`report_charts.py:772`-`842`); AUC = `roc["auc"]`, `effect_size_type = "AUC"` (`clinical_models.py:1544`-`1545`) |
| 9 | Checklist: DV exactly two values; covariates numeric (Factor 1 may be text/numeric); one row per subject; >=10 events per predictor rule of thumb | correct (reworded) | 2-level DV (`clinical_models.py:1075`); Factor 1 dummy-coded so text is fine (`:1080`), covariates must be numeric continuous terms (`:1081`-`1082`); one row per subject (listwise deletion, `:1064`); the 10-EPV figure is standard advice, not enforced in code for logistic (no EPV gate in the logistic path; EPV is only computed/shown for Beta Regression, `report_summaries.py:98`-`104`) |
| 10 | (added) "The report lists every OR with its 95% confidence interval" and "the report shows the ROC curve" | correct (missing feature added) | OR table (`report_association.py:19`-`69` via `report_charts.py:606`-`608`); ROC chart (`report_charts.py:610`-`612`, `:772`-`842`); both flow into `chart_blocks` (`html_exporter.py:99`, `:178`) |

## Alpha / CI-level control check

There is **no configurable alpha** for logistic regression. `fit` is called
without an `alpha` argument (`analysis_core.py:628`) and `fit` has no `alpha`
parameter (`clinical_models.py:1060`). All intervals are fixed at 95%: the
standard path uses `self.result.conf_int()` at the statsmodels default 95%
(`clinical_models.py:1353`); the Firth path hardcodes `z_crit =
norm.ppf(0.975)` (`:1310`) and `_firth_profile_ci(..., alpha=0.05)` (`:1329`);
the AUC-interpretation and Hosmer-Lemeshow "good fit" cutoffs use a fixed 0.05
(`report_summaries.py:252`, `report_stat_rows.py`). The recipe prints no numeric
alpha and states "95% confidence interval", so there is nothing to contradict.

## Data-structure control check

The recipe's layout (one binary outcome column, one grouping predictor in
Factor 1, numeric covariates, one row per subject) matches
`LogisticRegressionModel.fit`, which reads exactly `dv` / `predictors` /
`covariates`, dummy-codes predictors and treats covariates as continuous
(`clinical_models.py:1080`-`1082`), and applies listwise deletion over all of
them (`:1064`-`1065`). No auto-pivot applies: `_ap_maybe_pivot` handles wide
paired/repeated layouts, which the binary-DV signature does not trigger. The
"Subject ID: leave empty" instruction is consistent with the binary-DV override
firing before any repeated-measures routing takes effect (`pipeline:1204`).

## Unclear / possible code bug

None that falsify the (now-corrected) recipe. Internal observations for the
human record:

1. **Binary-outcome detection has an operator-precedence quirk.** In
   `_ap_build_analysis_context` (`pipeline:1119`-`1124`) the condition is
   `len(_unique) == 2 and pd.api.types.is_numeric_dtype(...) or _is_str and
   (_is_01 or _is_str) and not _name_is_grouping`. Because `and` binds tighter
   than `or`, this parses as `(len==2 and is_numeric) OR (_is_str and (...) and
   not _name_is_grouping)`. Consequences: (a) a **numeric** 2-value column is
   treated as binary even when its values are not 0/1 (e.g. `{2, 5}`) and even
   if its name is a grouping hint (the `not _name_is_grouping` guard does not
   apply to the numeric branch); (b) a **string** 2-value column also needs
   `len == 2`, but the first operand's `len == 2` does not gate the string
   branch, so in principle a >2-value all-string column could slip through the
   `is_binary` flag before the model's own 2-level guard rejects it. The model's
   `ValueError` at `clinical_models.py:1075`-`1076` is the real backstop, so no
   user reaches a wrong result, but the flag logic reads as an unintended
   precedence bug. The live Help-hint detector `_ap_is_binary_outcome_for_help`
   (`:730`-`736`) is stricter and correct (`is_01 or is_str`, both requiring
   `len == 2`). Recorded for a human decision; no code change made.

2. **Factor 1 predictors are always dummy-coded (`C()`), including continuous
   ones.** `clinical_models.py:1080` wraps every Factor 1 predictor in `C()`
   unconditionally. A continuous predictor (e.g. a dose in mg) placed in Factor 1
   therefore yields one OR per distinct value against a reference, not a per-unit
   slope. This is a deliberate bucket-semantics choice (Factor 1 = grouping
   factor, Covariates = continuous controls, matching the ANOVA convention), but
   it is a trap for a user who expects a continuous predictor in the "main
   predictor" slot to give a per-unit OR. The recipe now steers continuous
   predictors to Covariates and explains the difference. Whether the model should
   auto-detect a continuous Factor 1 and skip `C()` is a design question left for
   the human; no code change made.

3. **Firth fallback and convergence are surfaced, but only in report rows the
   recipe does not name.** The "Model variant" row
   (`report_stat_rows.py:335`) and the convergence status
   (`report_summaries.py:37`-`46`) do reach the user. Per the recipe-economy
   rule the recipe body does not name Firth Penalized Likelihood, the separation
   threshold (SE > 5.0, `clinical_models.py:1088`), or the profile-likelihood CI;
   those live here. No recipe change on this point.

4. **The recipe intentionally omits the extra fit statistics.** Brier score,
   calibration slope/intercept, McFadden pseudo-R², AIC/BIC, and
   Hosmer-Lemeshow all reach the report
   (`report_stat_rows.py:344`-`374`, `report_summaries.py:248`-`260`). Naming
   them in a triage recipe would violate the recipe-economy rule, so the recipe
   mentions only the OR table, the 95% CI, and the ROC/AUC. No recipe change.
