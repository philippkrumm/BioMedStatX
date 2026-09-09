# BioMedStatX — Global Data Schema (Audit Reference v1)

Derived from code (2026-06-27). This is the **absolute reference** for all chunks.
Confidence tags: [OBS] = directly observed in code; [INF] = inferred from call flow.
If a fix changes a signature, this doc must be re-issued before the next chunk.

---

## Boundary 0 — File → Preprocessing
`pd.read_csv` / `pd.read_excel(sheet_name)` → `self.df` (autopilot injects it as
`analysis_context["injected_df"]`; `AnalysisManager` does NOT re-read from disk).
`_ap_maybe_pivot()` may melt wide→long before this.

## Boundary 1 — Preprocessing → Engine (TWO stages)

### 1a. Caller → `AnalysisManager.analyze(...)`  [OBS analysis_core.py:143]
Positional/contract args:
`file_path, group_col, groups, sheet_name=0, value_cols=None,
selected_datasets=None, combine_columns=False, dependent=False, compare=None,
error_type="sd", dataset_name=None, additional_factors=None, **kwargs`
`kwargs["analysis_context"]` carries `injected_df`, design flags, transform opts.

### 1b. `_analyze_single_dataset` → prepared-data bundle  [OBS analysis_core.py:285]
```python
{
  "d": working_df,            # pd.DataFrame (cleaned, long-form)
  "samples": {group: seq},    # dict[str, list|ndarray] ← engine input
                              #   autopilot path -> python list (to_numeric.dropna().tolist())
                              #   legacy DataImporter path -> ndarray. Type is NOT uniform.
  "filtered_samples": {...},  # currently == samples
  "groups": [str, ...],
  "group_col": str,
  "value_cols": [str, ...],
  "dependent": bool,
  "additional_factors": [...] | None,
  "kwargs": {... , "analysis_context": {...}},
}
```
Multi-dataset path returns `{dataset_name: result_dict}` (+ `failed_datasets`).

---

## [INF RESOLVED — Chunk 1] `samples` consume points / engine orchestration
- `samples` built in `_prepare_contextual_inputs` (analysis_core.py:257 autopilot / :188 legacy).
- **Simple-test family consumes `samples`:**
  `StatisticalTester.check_normality_and_variance(groups, filtered_samples, …)` [analysis_core.py:833]
  → `perform_statistical_test(groups, transformed_samples, filtered_samples, …)` [:1015]
  → `perform_refactored_posthoc_testing(valid_groups, transformed_samples, …)` [:1083/:1158].
- **Advanced designs (mixed/two-way/RM) DO NOT use `samples`** — they consume the
  DataFrame `df` via `prepare_advanced_test(df, …)` [:897/:935/:978] +
  `perform_advanced_test(df=df, …)` [:904/:942/:985].
- **Clinical models** consume `df` via `model.fit(df, …)` [:592/:615/:623/:663/:703/:714].
- Convergence funnel: all paths land in `results`, pass `nonfinite_block(results)`
  [:1274] then `ExportDispatcher.export_analysis_results(results, …)` [:1416/:778].

### New result keys observed in Chunk 1 (extend Family-A/B):
`p_value_fdr` (multi-dataset, :367), `_file_paths` (:1526), `data_health` (:721),
`group_factor_map` (:532), `methodology_trace` (:373/:666), `raw_data_columns`
(:750, clinical no-group), `selected_groups`, `group_column`, `dependent_variable`,
`filter_applied`, `transformed_data`, `analysis_log`.

## Boundary 2 — Engine result (THE result dict) — dual representation

### 2a. Canonical DTO  [OBS statistical_testing/models.py:5]
`StatisticalResult(frozen)`:
| field | type | notes |
|---|---|---|
| test_name | str | required |
| statistic_value | Optional[float] | |
| p_value | Optional[float] | |
| degrees_of_freedom_1 / _2 | Optional[float] | df may be non-int (Welch) |
| effect_size | Optional[float] | |
| effect_size_type | Optional[str] | |
| metadata | Dict[str,Any] | catch-all |

Bridges: `from_legacy_dict()` / `to_legacy_dict()`. **The legacy dict is the real
wire format** — DTO is only used inside strategy engines.

### 2b. Legacy result dict — the actual contract  [OBS]
Core keys (mapped by the bridge):
`test`, `final_test_label`, `tested_against`, `statistic`, `p_value`,
`df1`, `df2`, `effect_size`, `effect_size_type`.

Two result FAMILIES share this envelope and add their own keys:

**Family A — group-comparison** (t/ANOVA/nonparam/RM/Mixed):
`comparisons` (post-hoc list, see 2c), `descriptive`, `samples`, `groups`,
`normality_check`, `normality_tests`, `variance_test`, `transformation`,
`power`, `warnings`, `recommendation` / `test_recommendation`, `test_type`,
`sphericity_test`, `sphericity_corrections`, `within_sphericity_corrections`.

**Family B — model-based** (ANCOVA/LMM/logistic/correlation), built by
`as_results_dict()` [OBS correlation_models.py, clinical_models.py]:
`model_type`, `method`, `coefficient`, `odds_ratios`, `confidence_interval`,
`ci_lower`, `ci_upper`, `fixed_effects_table`, `anova_table`, `r_squared`,
`r_squared_adj`, `pseudo_r_squared`, `aic`, `bic`, `log_likelihood`,
`converged`, `n` / `n_observations` / `n_subjects`, `icc`,
`random_effects_variance`, `residual_variance`, `diagnostics`,
`hosmer_lemeshow`, `brier_score`, `calibration_slope/intercept`,
`r_matrix` / `p_matrix` / `p_corrected_matrix` / `n_matrix` (correlation),
`roc_data`, `lrt_*`, `x_/y_transform*`, `x_/y_boxcox_lambda`.

### Cohen's d ddof convention — REFERENCE (for C2-2 fix)  [OBS posthoc_core.py:1302]
Canonical = `PostHocStatistics.calculate_cohens_d` (:1302-1311):
- paired:      `mean(diff) / np.std(diff, ddof=1)`,  guard `>0 else 0`
- independent: pooled SD with `np.var(..., ddof=1)`,  guard `s_pooled>0 else 0`
RM internal (:1103-1106) matches: `np.std(differences, ddof=1)`, `else 0`.
`calculate_ci_mean_diff` (:1314) is alpha-aware: `t.ppf(1-alpha/2, df)` (≠ statisticaltester's
hardcoded 0.95 — see C2-8). **CONVENTION = ddof=1; zero-SD → 0.**
INCONSISTENCY across codebase: zero-SD sentinel is `0` (posthoc_core) vs `None`
(statisticaltester `_paired_ttest` :467). C2-2 fix (statisticaltester:1815) → adopt
ddof=1 + zero-guard; pick ONE sentinel (recommend the local RM convention: ddof=1, →0,
type "cohen_d_rm").

### 2c. Post-hoc comparison sub-dict — shapes  [OBS posthoc_core.py]
✅ RESOLVED (Chunk 3): Shape-2 (`level1/level2/p_val/mean_dif/se_dif`, :1114) is an
INTERNAL scratch list only; every live analyzer converts it to canonical **Shape-1**
via `PostHocAnalyzer.add_comparison(...)` before returning in `pairwise_comparisons`
(RM :1217; Mixed :752/:950; TwoWay :176/:236/:291; Tukey/GH/Dunnett/Dunn/Dependent).
Engine forwards only `pairwise_comparisons` (Shape-1). The hybrid Shape-2 emitter
`_perform_test_legacy` (:476) is DEAD. ⇒ **Shape-2 never reaches a consumer; the wire
format is Shape-1.** analysis_core :1364 hard-indexing is SAFE in practice.
Residual key duality kept only for historical reference below:
| concept | Shape-1 keys | Shape-2 keys |
|---|---|---|
| groups | `group1`, `group2` | `level1`, `level2` |
| p-value | `p_value` | `p_val` |
| statistic | (t_stat / statistic) | `t_stat` |
| effect size | `effect_size` | `d` |
| mean diff | — | `mean_dif`, `se_dif` |
| CI | `ci_lower`, `ci_upper` | `ci_lower`, `ci_upper` |
| n | `n_pairs` | `n_pairs` |
| type | `test_type`, `comparison_type` | `test_type`, `comparison_type` |
| raw data | `data1`, `data2`, `differences` | `data1`, `data2`, `differences` |
**HAZARD:** export `_build_pairwise_rows` must absorb both; a path producing the
wrong key-set renders blank cells silently. Flag per chunk.

### 2d. Assumptions block  [OBS]
`normality_check` / `normality_tests`, `variance_test` (+ `levene`),
`sphericity_test`, `slope_homogeneity`. Sub-values carry Shapiro W + p, etc.

---

## Boundary 3 — Engine → Export

`ExportDispatcher.export_analysis_results(results, output_file, analysis_log)`
[OBS export_dispatcher.py:9] → `HTMLExporter.export_results_to_html(results, ...)`.
Exporter builds a **context dict** consumed by Jinja templates
(`report_single.html.j2`, `report_multi.html.j2`) via `template.render(context=...)`.

`_build_*` consumers all take `results: dict` and branch on
`model_type` / `test_type`:
- `_build_statistical_rows`, `_build_pairwise_rows`, `_build_ancova/_lmm/_logistic/
  _factorial_anova/_corr_matrix/_beta_statistical_rows` [OBS report_stat_rows.py]
- `_build_*_chart` family [OBS report_charts.py]
- `_build_or_table_html`, `_build_beta_coefficient_table_html` [report_association.py]
- `_build_hero_context`, `_build_decision_path_model`, `_build_decision_tree_json`,
  `_build_methods_text` [html_exporter.py]

Context top-level keys feeding templates (subset): `assumptions`, `chart_blocks`,
`dataset_cards` (multi), `plot_designer_enabled`, `math_render_enabled`,
`descriptive`, `samples`, `raw_data` / `raw_data_transformed`, `residuals`,
`transformation`, `diagnostics`, `adjusted_means`.

---

## Output-band invariants the audit will enforce (Dimension 3)
- `p_value`, every `p_val`/`p_adjusted`/matrix p ∈ [0,1] or None — never a str.
- `df1/df2` > 0; integer except Welch/Satterthwaite (float allowed).
- `effect_size` finite; |Cohen d| flagged if > 5 (bug if > 50).
- CI: `ci_lower ≤ point ≤ ci_upper`; OR/HR CIs asymmetric & strictly > 0.
- All numeric fields are float/None at the ENGINE→EXPORT boundary (results dict) —
  verified Chunks 1-5. CORRECTION (Chunk 6): formatting happens in the **export
  Python layer** (`report_formatting._format_metric/_format_p_value`), NOT in Jinja.
  Templates are dumb string-renderers (no `|round`/`{:.Nf}` filters) → they render
  pre-formatted strings. So `_format_*` is the SINGLE point for NaN/band/format logic.
  NaN→"N/A", inf→"Infinity" (explicit, good). NO [0,1] band guard → out-of-band
  finite p rendered verbatim; NEGATIVE p → falsely "p < 0.001 ***" (C6 finding).
- `within_pairwise_comparisons` is MERGED into `pairwise_comparisons` by the engine
  (statisticaltester:1958-1959) before export → exporter reads only pairwise_comparisons.

## Post-hoc shape tracking ledger (active)
| Module | Emits | Consumes | How |
|---|---|---|---|
| analysis_core.py | **Shape-1** (via `PostHocAnalyzer.add_comparison(group1=,group2=,p_value=,statistic=,effect_size=,…)` :1132) | **Shape-1, HARD indexing** `comp['group1']`/`comp['p_value']`/`comp['significant']` (:1364-1372, :1668-1675) | Shape-2 (level1/p_val) input → **KeyError**, log build crashes into outer except |

## Known structural hazards (carry into every chunk)
1. Post-hoc dict key duality (2c) — silent blank-cell risk. analysis_core consumes
   Shape-1 with non-defensive indexing → Shape-2 crashes (not blank).
2. Legacy-dict ↔ DTO round-trip: `metadata` catch-all can swallow a renamed key
   without error (Dimension 6 blind spot).
3. pandas-3.0 default str dtype on `groups`/factor cols → NA/group semantics.
4. Cohen's-d ddof inconsistency (statisticaltester.py:1815 ddof=0 vs ddof=1).
