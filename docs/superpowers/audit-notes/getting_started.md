# Audit note: `getting_started` recipe

Recipe location: `src/core/help_content.py:14` (`"id": "getting_started"`).
Ground-truth modules: `src/autopilot/statistical_analyzer_autopilot_pipeline.py`
(`_ap_load_file`, `_ap_load_sheet`, `_ap_maybe_pivot`, `_ap_build_analysis_context`,
`_ap_export_example_template`), plus the wide-format helpers in
`src/autopilot/statistical_analyzer_autopilot_ui.py` and the menu wiring in
`src/analysis/statistical_analyzer.py`.

The citation anchor (symbol name or quoted string) is authoritative; the line number is
only a navigation hint.

## Claim table

| # | Claim (from title/html) | Verdict | Citation |
|---|-------------------------|---------|----------|
| 1 | Title "Getting started: read this first" | correct | `src/core/help_content.py:16` (`"title"`) — matches the recipe title; heading is descriptive, no code dependency |
| 2 | The app has six drag-and-drop buckets in the center | correct | `statistical_analyzer_autopilot_pipeline.py:324` (`_ap_init_ui`, loop over `dv_bucket, factor1_bucket, factor2_bucket, subject_bucket, covariates_bucket, filter_bucket`) — six widgets added to `mapping_panel` |
| 3 | You drag column names into buckets to assign each column a role | correct | `statistical_analyzer_autopilot_pipeline.py:250` (`_ap_init_ui`, `MappingBucketWidget(...)`); roles read back in `_ap_build_analysis_context:1055` via `get_assigned_columns()` |
| 4 | The app selects the test automatically; you never pick one manually | correct | `statistical_analyzer_autopilot_pipeline.py:1166` (`_ap_build_analysis_context`, `context["inferred_test"] = ...`) — the test is derived from mapping, not chosen by the user |
| 5 | Bucket "Dependent Variable" = the measured number | correct | `statistical_analyzer_autopilot_pipeline.py:250` (`_ap_init_ui`, `MappingBucketWidget("Dependent Variable", ..., accepted_kinds={"numeric"})`) |
| 6 | Bucket "Factor 1" = main predictor; group column for group comparison, predictor for a numeric relationship | correct | `statistical_analyzer_autopilot_pipeline.py:263` (`_ap_init_ui`, `"Factor 1"`, info_text "Categorical -> t-Test or ANOVA. Continuous -> Correlation or Regression"); routing at `:1168` (categorical) and `:1254` (`_corr_is_continuous` -> correlation/regression) |
| 7 | Bucket "Factor 2" = a second grouping column, only for two-way splits | correct | `statistical_analyzer_autopilot_pipeline.py:275` (`_ap_init_ui`, `"Factor 2"`); two-factor branch at `_ap_build_analysis_context:1169` -> `two_way_anova`/`mixed_anova` |
| 8 | Bucket "Subject ID" = who the measurement belongs to, only when a subject appears more than once | correct | `statistical_analyzer_autopilot_pipeline.py:288` (`_ap_init_ui`, `"Subject ID"`); repeated-measures gate at `_ap_build_analysis_context:1162` (`subject_span ... nunique`) raises if the subject does not repeat across the factor |
| 9 | Bucket "Covariates" = a background variable to correct for | correct | `statistical_analyzer_autopilot_pipeline.py:303` (`_ap_init_ui`, `"Covariates (optional)"`, `accepted_kinds={"numeric"}`, `allow_multiple=True`); ANCOVA upgrade at `_ap_build_analysis_context:1240` |
| 10 | Bucket "Filter" = restricts the whole analysis to one subgroup | correct | `statistical_analyzer_autopilot_pipeline.py:317` (`_ap_init_ui`, `FilterBucketWidget`); applied at `_ap_build_analysis_context:1091` (`analysis_df = analysis_df[analysis_df[filter_col] == filter_val]`) — single column-equals-value subset |
| 11 | The app expects long format: one row is one measurement | correct | `_ap_build_analysis_context:1157` (`levels = _sorted_unique(analysis_df[factor].tolist())`) — the factor column is read as row-wise group labels, i.e. long format |
| 12 | In long format a "Group" column maps to Factor 1; in wide format group names are hidden in headers and cannot be extracted | correct | `_ap_build_analysis_context:1096` (`analysis_df[factor_columns[0]]`) reads groups from a column's values, not from headers |
| 13 | "Wide format does not work here" (blanket statement) | wrong / missing-relevant-feature | `_ap_maybe_pivot:1961` calls `_detect_wide_format`; `statistical_analyzer_autopilot_ui.py:128` (`_detect_wide_format`) auto-detects a specific wide-format paired/repeated signature and `:176` (`_pivot_wide_to_long`) melts it to long automatically. So wide-format group-comparison data still fails, but wide-format paired/repeated data is auto-converted. The blanket "does not work" was corrected to describe the one wide case that is handled. |
| 14 | Factor 1 = group labels -> t-Test or ANOVA | correct | `_ap_build_analysis_context:1168` (`"independent_ttest" if len(levels) == 2 else "one_way_anova"`) |
| 15 | Factor 1 = numbers -> Correlation or Regression | correct | `_ap_build_analysis_context:1254` (`_corr_is_continuous(...)` -> `:1264` `"correlation"`/`"linear_regression"`) |
| 16 | Factor 1 and Factor 2 both filled -> Two-Way ANOVA or Mixed ANOVA | correct | `_ap_build_analysis_context:1198` (`"two_way_anova"`) and `:1195` (`"mixed_anova"` when Subject ID is also present) |
| 17 | Subject ID filled -> paired or repeated-measures design | correct | `_ap_build_analysis_context:1166` (`"paired_ttest" if len(levels) == 2 else "repeated_measures_anova"`) |
| 18 | Covariates filled -> ANCOVA or Multiple Regression | correct | `_ap_build_analysis_context:1240` (`"ancova"`) and `:1256` (`"linear_regression"` when Factor 1 is continuous and covariates are present) |
| 19 | Outcome has exactly two values (0/1 or Yes/No) -> Logistic Regression | correct | `_ap_build_analysis_context:1119` (`is_binary = len(_unique) == 2 ...`, `_is_01` or two strings) -> `:1205` (`"logistic_regression"`) |
| 20 | The status line "always shows which test would run right now, before you click Start" | wrong | The status line is `mapping_feedback_label`, set in `_ap_on_mapping_changed`. Its text is validation/mapping guidance, not the inferred test name: `statistical_analyzer_autopilot_pipeline.py:896` (`"Mapping looks valid. Start the analysis when you are ready."`), `:854` (wide-format pivot notice), `:892` (covariate warnings). The decision path with the actual test only renders after the run: `statistical_analyzer_autopilot_ui.py:960` (`DecisionTreePanel`, `"Decision path renders after analysis."`). Corrected to say the line confirms the mapping is valid and enables Start. |
| 21 | "Save example template" reference (task control check) | correct once added | Real menu action is "Save Example Template..." under Help: `src/analysis/statistical_analyzer.py:187` (`QAction('Save Example Template...')` -> `:188` `export_example_template`), which copies `assets/BioMedStatX_Excel_Template.xlsx`: `_ap_export_example_template:2253`. The original recipe did not mention it; added a short pointer with the exact menu label. |

## Data-structure control check

`_detect_wide_format` (`statistical_analyzer_autopilot_ui.py:128`) auto-pivots a
DataFrame to long only when all of the following hold: exactly one subject-like column
(`_looks_like_subject`, `:89`), 2 to 8 numeric value columns (`:158`), no categorical
column with exactly 2 unique values (`:163`, the wide-vs-long group discriminator), and
subject-column uniqueness ratio at least 0.8 (`:169`). The melt produces columns
`[subject_col, "Condition", "Value"]` (`_pivot_wide_to_long:181`). This matches the
recipe's corrected wording: wide group-comparison data (group names as column headers)
is still rejected, but wide paired/repeated data is converted automatically.

## Alpha / adjustment control check

Not applicable to this recipe. `getting_started` makes no significance-level, p-value
adjustment, sided-ness, or post-hoc claim; those live in the test-specific recipes.

## Unclear / possible code bug

None found for this recipe. The binary-detection boolean in
`_ap_build_analysis_context:1119` (`is_binary = len(_unique) == 2 and
pd.api.types.is_numeric_dtype(self.df[dv_col]) or _is_str and (_is_01 or _is_str) and
not _name_is_grouping`) mixes `and`/`or` without parentheses, which is fragile, but it
does not contradict the recipe claim (claim 19 only states the two-value outcome case)
and is out of scope for a content audit. Flagging it here for the human as a readability
concern, not a recipe correction.
