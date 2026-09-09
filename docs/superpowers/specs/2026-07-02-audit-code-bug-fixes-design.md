# Audit-flagged code bug fixes design

Date: 2026-07-02
Status: approved

## Problem

The Help Hub content audit (`docs/superpowers/audit-notes/SUMMARY.md`) found 10
code-level issues while verifying recipe text against behavior, deliberately left
unfixed because the audit's scope was recipe text only. Six of them have a single
correct fix with no product decision required. The other four (sphericity
outer-exception fallback, logistic regression's Factor 1 dummy-coding trap, the
unrendered `coefficient_table` in linear regression, and ANCOVA's dead vs-control
post-hoc path) require a product decision before any code is written and are out of
scope for this pass.

## Scope

In scope, one fix each:

1. Post-hoc method label desyncs from the actually-applied correction in Two-Way
   ANOVA, Mixed ANOVA, and Repeated Measures ANOVA (one shared root cause).
2. Dead `"Strip"` branch and silent wrong-plot fallback in the plot export dispatch.
3. Untranslated German checkbox label for simple linear regression.
4. Operator-precedence bug in the binary-outcome detector.
5. Missing subject-column check for `mixed_anova` in `validate_test_design`.
6. Stale `CorrelationModel` class docstring.

Out of scope: the four product-decision items listed above (remain in SUMMARY.md
for a later design pass); any other code the audit did not flag; any change to
`src/core/help_content.py` recipe text beyond the one places explicitly noted below.

## Fix 1: Post-hoc label desync (shared root cause)

**File:** `src/statistical_testing/advanced_pipeline.py:256-266`

**Current code:**
```python
current_posthoc = res.get("posthoc_test", "")
new_posthoc = advanced_posthoc_updates.get("posthoc_test") or advanced_posthoc_result.test_name
should_override = (
    not current_posthoc
    or current_posthoc == "Two-Way ANOVA Post-hoc Tests"
    or "parametric paired t-tests" in current_posthoc.lower()
    or "pairwise paired t-tests" in current_posthoc.lower()
    or ("Pingouin" in str(current_posthoc) and new_posthoc and "Tukey" in str(new_posthoc))
)
if should_override:
    res["posthoc_test"] = new_posthoc
```

This guard only overrides `posthoc_test` when `current_posthoc` matches one of five
specific string patterns. But `res["pairwise_comparisons"]` (three lines above, line
255) is unconditionally replaced with the advanced engine's output whenever
`advanced_posthoc_updates.get("pairwise_comparisons")` is truthy — the block this
guard lives inside only runs after that replacement already happened. There is no
scenario where the pairwise comparisons come from the advanced engine but the label
should still describe some other, stale method. The guard is solving a problem that
doesn't exist and, by pattern-matching specific inline labels, breaks silently
whenever a new or reworded inline label doesn't happen to match one of the five
patterns — which is exactly what happened for Two-Way ANOVA's `"Tukey HSD Test
(Pingouin)"`, Mixed ANOVA's `"Pairwise t-tests for interaction (Holm-Bonferroni)"`,
and Repeated Measures ANOVA's equivalent inline label (all three go through this
same code path).

**Fix:** remove the `should_override` computation and the conditional; always set
`res["posthoc_test"] = new_posthoc` in the same place `pairwise_comparisons` is
replaced (inside the `if advanced_posthoc_updates.get("pairwise_comparisons"):`
block, which already gates this whole section).

## Fix 2: Dead Strip branch and silent plot-type fallback

**File:** `src/analysis/analysis_core.py:1469-1535` (inside `AnalysisManager.analyze`)

Two problems in the same dispatch:
- Lines 1500-1512: an `elif plot_type == "Strip":` branch with the comment "Strip
  plot doesn't exist, fall back to box plot with points." Unreachable — `"Strip"` was
  removed from `plot_type_combo.addItems([...])` in
  `src/ui/dialogs/plot_aesthetics_dialog.py:646` (now `['Bar', 'Box', 'Violin',
  'Raincloud']`), but this branch was never deleted.
- Lines 1522-1535: the catch-all `else` silently renders a Bar plot and only logs
  `logger.warning(f"WARNING: Unknown plot type '{plot_type}', falling back to Bar
  plot")` — no exception. The sibling *preview* dispatch
  (`src/visualization/datavisualizer.py:plot_from_config`, ~line 2905) raises
  `ValueError(f"Unbekannter plot_type: {plot_type}")` for the identical situation.

**Fix:** delete the dead `elif plot_type == "Strip":` branch entirely. Replace the
catch-all `else`'s silent-Bar-plus-warning body with
`raise ValueError(f"Unknown plot type: {plot_type!r}")`, matching the preview
dispatch's behavior. This is safe today because the dropdown can only ever supply
`Bar`/`Box`/`Violin`/`Raincloud`; the fix only changes behavior for a value that
cannot currently occur, so it is a pure latent-bug fix, not a behavior change for any
real user path.

## Fix 3: German checkbox label

**File:** `src/autopilot/statistical_analyzer_autopilot_pipeline.py:353`

**Current:** `QCheckBox("Als Lineare Regression analysieren (Y = a + bX)")`
**Fix:** `QCheckBox("Analyze as Linear Regression (Y = a + bX)")`

**Also update:** the already-audited `linear_regression` recipe in
`src/core/help_content.py` currently quotes the German label verbatim (added
deliberately during the content audit, commit `578b8e1`, specifically to match what
the live UI showed at the time). Once the UI label changes to English, update that
quote in the recipe to match, so the recipe continues to describe the live UI
accurately. This is the one recipe-text touch in this plan; it is a direct
consequence of fixing the code, not a new content audit.

## Fix 4: Binary-outcome detector operator precedence

**File:** `src/autopilot/statistical_analyzer_autopilot_pipeline.py:1119-1124`

**Current code:**
```python
is_binary = (
    len(_unique) == 2
    and pd.api.types.is_numeric_dtype(self.df[dv_col]) or _is_str
    and (_is_01 or _is_str)
    and not _name_is_grouping
)
```

Because `and` binds tighter than `or`, this parses as
`(len==2 and is_numeric) or (_is_str and (_is_01 or _is_str) and not _name_is_grouping)`.
Since `_is_str` implies `(_is_01 or _is_str)` is trivially true, this reduces to
`(len==2 and is_numeric) or (_is_str and not _name_is_grouping)` — meaning: (a) any
numeric 2-value column is flagged binary regardless of its actual values (not just
0/1) and regardless of a grouping-hint name; (b) any all-string column is flagged
binary regardless of how many unique values it has, as long as the name isn't a
grouping hint.

The comment immediately above the block (lines 1112-1113) states the intended
semantics: "exactly 2 values that are 0/1 (or two strings), AND column name does not
hint at a grouping variable." Cross-checked against the ground-truth reference
implementation `_ap_is_binary_outcome_for_help`
(`statistical_analyzer_autopilot_pipeline.py:717-736`), which independently confirms:
`len(unique_values) != 2: return False` gates first, unconditionally, then
`is_01 or is_str` decides. That function has no dtype check and no grouping-name
check (it's a different, narrower helper used elsewhere for a Help-hint), but its
`len==2` gate applying unconditionally to both branches confirms the intended shape.

**Fix:**
```python
is_binary = (
    len(_unique) == 2
    and (_is_01 or _is_str)
    and not _name_is_grouping
)
```
This drops the `pd.api.types.is_numeric_dtype(self.df[dv_col])` check entirely: it is
redundant once `_is_01`/`_is_str` are computed from the actual unique values present
(a set of `{0, 1}`-like values is inherently numeric; an all-string unique set is
inherently non-numeric), and keeping it was part of what caused the precedence bug in
the first place. The `len(_unique) == 2` gate now unconditionally applies to both the
numeric and string cases, and `not _name_is_grouping` applies to both, matching the
comment's stated intent and the ground-truth helper's shape.

## Fix 5: Missing subject check for mixed_anova

**File:** `src/statistical_testing/validators.py:254-260` (function `validate_test_design`)

**Current code (exact, confirmed):**
```python
    if test_name == "mixed_anova":
        if not between or not within:
            raise ModelDesignError("Mixed ANOVA requires between and within factor.")
    elif test_name == "repeated_measures_anova":
        if not within:
            raise ModelDesignError("RM-ANOVA requires within factor.")
        if subject is None:
            raise ModelDesignError("RM-ANOVA requires subject column.")
```

The `mixed_anova` branch checks for between + within factors but not for `subject`,
unlike the sibling `repeated_measures_anova` branch directly below it. Harmless today
because the autopilot only ever routes to `mixed_anova` with a Subject ID present, but
a non-autopilot caller (or a future code path) could reach `pg.mixed_anova(subject=...)`
with no subject and get a raw pingouin exception instead of the app's own clean
`ModelDesignError`.

**Fix:**
```python
    if test_name == "mixed_anova":
        if not between or not within:
            raise ModelDesignError("Mixed ANOVA requires between and within factor.")
        if subject is None:
            raise ModelDesignError("Mixed ANOVA requires subject column.")
```

## Fix 6: Stale CorrelationModel docstring

**File:** `src/analysis/correlation_models.py:199-203`

**Current:**
```python
class CorrelationModel:
    """Pearson or Spearman correlation with 95 % CI (Fisher z-transform).

    method='auto' applies Shapiro-Wilk to both variables and uses Pearson when
    both are normally distributed (p > alpha), otherwise Spearman.
    Pairwise deletion: only rows without NaN in x_col or y_col are used.
    """
```

This describes a Shapiro-Wilk-p-value-driven selection. The actual `fit` logic
(`correlation_models.py:281-297`, confirmed during the content audit) is N-tier gated
on skewness/excess-kurtosis, not on the Shapiro p-value: `n<20` always Spearman;
`20<=n<100` Pearson iff `|skew|<=1.0` and `|excess kurtosis|<=2.0` for both variables;
`n>=100` Pearson unless extreme asymmetry (`|skew|>2.0` or `|excess kurtosis|>4.0`).
Shapiro-Wilk is computed and stored in a diagnostics dict but never read by the
branch.

**Fix:** rewrite the docstring's second paragraph to describe the real skew/kurtosis
gating and its thresholds, and note that Shapiro-Wilk is computed for diagnostic
reporting only, not used for the method decision.

## Testing

TDD per fix, targeted unit tests that reproduce each bug before fixing it:

- **Fix 1:** a test that drives the advanced post-hoc path for a design whose inline
  label doesn't match any of the five old patterns (e.g. the Mixed ANOVA inline label
  `"Pairwise t-tests for interaction (Holm-Bonferroni)"`), asserts `posthoc_test` in
  the result now equals the advanced engine's actual method name, not the stale
  inline label. Should also cover Two-Way and RM inline labels as parametrized cases
  of the same test, since it's one shared code path.
- **Fix 2:** a test that calls the export dispatch with an unrecognized `plot_type`
  and asserts it raises `ValueError` (not a silent Bar-plot render). A second test
  confirms `plot_type == "Strip"` is no longer a special-cased branch (also raises,
  same as any other unrecognized value).
- **Fix 3:** no test needed (pure string change); confirmed by updating any test that
  currently asserts on the German string, if one exists (grep for it first).
- **Fix 4:** parametrized test covering: numeric 2-value column with non-0/1 values
  and a grouping-hint name (should NOT be binary after the fix, WAS incorrectly
  binary before); all-string column with >2 unique values (should NOT be binary
  after the fix, WAS incorrectly binary before via the string branch's missing len
  check); the normal 0/1 and Yes/No cases (should remain binary, regression check).
- **Fix 5:** a test that calls `validate_test_design` for `mixed_anova` with no
  subject column and asserts it raises `ModelDesignError` (not a downstream pingouin
  exception).
- **Fix 6:** no test needed (docstring only).

Plus a full `pytest tests/` run after all fixes, and `ruff check` on every touched
file.

## Risks

- Fix 1 changes a code path exercised by three different ANOVA designs; the
  parametrized test must cover all three inline-label shapes to avoid silently
  breaking one while fixing another.
- Fix 4's parenthesization changes real routing behavior (a column that was
  incorrectly treated as a binary/logistic-regression outcome before will no longer
  be) — this is the intended correction, but the test must include a case that was
  previously (incorrectly) treated as binary to prove the fix actually changes
  behavior, not just that it doesn't crash.
- Fix 2's `raise` on an unknown `plot_type` is a stricter contract than before; if any
  code path can currently reach this dispatch with a transient/legacy plot_type value
  the audit didn't find, this fix would surface it as a hard error instead of a silent
  Bar-plot fallback. Mitigated by the audit's confirmation that the dropdown is the
  sole source of `plot_type` values reaching this dispatch and only supplies four
  known-good values.
