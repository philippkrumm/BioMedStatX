# Help Hub Content Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring every Help Hub recipe's content into agreement with the current code, fixing wrong statements and adding relevant missing features and statistical specifics, each correction backed by a verifiable code citation.

**Architecture:** One task per recipe (12 recipes in `src/core/help_content.py`). Each task audits one recipe against authoritative code modules, rewrites its `html` (and `title` if wrong), records an audit note with anchor-based citations, and commits independently so progress is durable and resumable. A final task does whole-diff verification and collates flagged items.

**Tech Stack:** Python 3.12, PyQt5, pytest 7.4 (headless Qt via root `conftest.py`, `QT_QPA_PLATFORM=offscreen`, `src/` on `sys.path`). Spec: `docs/superpowers/specs/2026-06-30-help-content-audit-design.md`.

---

## Shared audit procedure (every recipe task follows this)

For the recipe assigned to the task:

1. Read the recipe dict in `src/core/help_content.py` (find it by `"id": "<recipe_id>"`). List every factual claim in its `html` and `title`.
2. For each claim, open the authoritative module(s) listed in the task and find the supporting code. Record a citation in this format (anchor authoritative, line is a hint):
   `path/to/file.py:NNN (ClassName.method)` or `path/to/file.py:NNN ("string literal")`.
   For a statistical claim, open the actual function and confirm the EXACT behavior: p-value adjustment method (Tukey HSD / Bonferroni / Holm / none), one- vs two-sided, default `alpha`, and the key call parameters. Citing "a test of this family exists" is not enough.
3. Run the three mandatory control checks where applicable:
   - alpha / adjustment match (recipe's stated significance handling == code default + actual correction);
   - data-structure invariance (wide/long instruction matches the parser; for paired/RM/mixed check `advanced_pipeline.py` and `_ap_maybe_pivot`);
   - assumption/correction claims (e.g. sphericity → Greenhouse-Geisser, normality → non-parametric fallback) match code.
4. Classify each claim: correct / wrong / missing-relevant-feature / unclear.
5. Rewrite the recipe `html` to fix wrong claims and add missing relevant features. Constraints:
   - Do NOT change the recipe `id` or `category`.
   - Apply the `anthropic-skills:humanizer` skill to every NEW or CHANGED sentence
     (not just the unchanged parts): no emoji, no typographic dashes
     (— – ― ‒ ‐ ‑), no rule-of-three/copula-avoidance/inflated-significance
     patterns, sentence-case headings, exactly one opening `<h2>` then `<h3>`
     sections. Text you did not touch does not need re-humanizing, but anything you
     write or rewrite must pass the same bar as the Task 4 humanizer pass.
   - Preserve existing `<table>` structure and any `.badge` spans; only change prose.
   - If code contradicts the recipe and the CODE looks wrong (not the recipe), do NOT rewrite to match a bug — flag it.
6. Write an audit note file at `docs/superpowers/audit-notes/<recipe_id>.md` containing: the claim list with verdict + citation for each, and an "Unclear / possible code bug" section (may be empty).
7. Run `pytest tests/test_help_hub.py -q` then `pytest tests/ -q`. Both must stay green (the invariant tests enforce no-emoji/no-dash/one-h2/id-stability).
8. Commit: stage `src/core/help_content.py` and `docs/superpowers/audit-notes/<recipe_id>.md`. Subject `docs(help): audit <recipe_id> recipe against code`. Trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

There are no new unit tests per recipe; the existing `tests/test_help_hub.py` invariants plus the citation-verified audit note are the gate. The spec-compliance reviewer for each task re-checks the citations by locating each anchor in code.

---

### Task 1: Audit `getting_started`

**Files:** Modify `src/core/help_content.py` (recipe `getting_started`); Create `docs/superpowers/audit-notes/getting_started.md`.

- [ ] **Step 1:** Follow the shared audit procedure for `getting_started`.
- [ ] **Step 2:** Ground-truth modules: `src/autopilot/statistical_analyzer_autopilot_pipeline.py` (`_ap_load_file`, `_ap_load_sheet`, `_ap_maybe_pivot`, `_ap_build_analysis_context`, `export_example_template` / `_ap_export_example_template`). Verify: the described load → column-mapping → run workflow matches the actual UI flow; the "save example template" reference matches the real menu action; any stated data format matches `_ap_maybe_pivot` behavior.
- [ ] **Step 3:** Rewrite html per findings; write audit note.
- [ ] **Step 4:** `pytest tests/test_help_hub.py -q` and `pytest tests/ -q` → green.
- [ ] **Step 5:** Commit (subject `docs(help): audit getting_started recipe against code`).

### Task 2: Audit `one_way_anova`

**Files:** Modify `src/core/help_content.py` (recipe `one_way_anova`); Create `docs/superpowers/audit-notes/one_way_anova.md`.

- [ ] **Step 1:** Follow the shared audit procedure for `one_way_anova`.
- [ ] **Step 2:** Ground-truth: `src/analysis/statisticaltester.py` (t-test vs one-way ANOVA selection, Shapiro-Wilk normality, Levene/Brown-Forsythe variance, Welch correction, Kruskal-Wallis fallback), `src/analysis/analysis_core.py`, `src/statistical_testing/validators.py`. Verify the post-hoc method and its exact p-adjustment (e.g. Tukey HSD vs Holm) and default `alpha`. Confirm the data layout the recipe shows matches the long-format the analyzer expects.
- [ ] **Step 3:** Rewrite html; write audit note.
- [ ] **Step 4:** Tests green.
- [ ] **Step 5:** Commit (`docs(help): audit one_way_anova recipe against code`).

### Task 3: Audit `two_way_anova`

**Files:** Modify `src/core/help_content.py` (recipe `two_way_anova`); Create `docs/superpowers/audit-notes/two_way_anova.md`.

- [ ] **Step 1:** Follow the shared audit procedure for `two_way_anova`.
- [ ] **Step 2:** Ground-truth: `src/statistical_testing/advanced_pipeline.py`, `src/statistical_testing/engines/advanced_posthoc.py`, `src/statistical_testing/validators.py`, `src/analysis/analysis_core.py`. Verify: main-effects + interaction handling, the exact post-hoc method/adjustment, required two-factor long-format data structure vs the parser, and assumption checks.
- [ ] **Step 3:** Rewrite html; write audit note.
- [ ] **Step 4:** Tests green.
- [ ] **Step 5:** Commit (`docs(help): audit two_way_anova recipe against code`).

### Task 4: Audit `repeated_measures_anova`

**Files:** Modify `src/core/help_content.py` (recipe `repeated_measures_anova`); Create `docs/superpowers/audit-notes/repeated_measures_anova.md`.

- [ ] **Step 1:** Follow the shared audit procedure for `repeated_measures_anova`.
- [ ] **Step 2:** Ground-truth: `src/statistical_testing/advanced_pipeline.py` dispatches (`test == "repeated_measures_anova"` at `advanced_pipeline.py:166`) into `StatisticalTester._run_repeated_measures_anova_logged` in `src/analysis/statisticaltester.py`, where the actual sphericity/Greenhouse-Geisser logic lives (`_perform_comprehensive_sphericity_test` at `statisticaltester.py:1967`, correction application around `statisticaltester.py:1936`, `sphericity_assumed`/`mauchly_p`/`epsilon` fields at `:1978-1980`); `validators.py`. Verify the sphericity → Greenhouse-Geisser claim against the actual `_perform_comprehensive_sphericity_test` behavior (control check 3), the LMM-fallback note, the post-hoc method, and the wide-vs-long data expectation against `_ap_maybe_pivot` and the RM parser.
- [ ] **Step 3:** Rewrite html; write audit note.
- [ ] **Step 4:** Tests green.
- [ ] **Step 5:** Commit (`docs(help): audit repeated_measures_anova recipe against code`).

### Task 5: Audit `mixed_anova`

**Files:** Modify `src/core/help_content.py` (recipe `mixed_anova`); Create `docs/superpowers/audit-notes/mixed_anova.md`.

- [ ] **Step 1:** Follow the shared audit procedure for `mixed_anova`.
- [ ] **Step 2:** Ground-truth: `src/statistical_testing/advanced_pipeline.py` dispatches (`test == "mixed_anova"` at `advanced_pipeline.py:164`) into `StatisticalTester._run_mixed_anova_logged` in `src/analysis/statisticaltester.py` (mixed-design sphericity via `_test_mixed_anova_within_sphericity:1703`); post-hoc via `src/analysis/emm_posthoc.py` (`rm_dunnett_emm_mvt:163`, `mixed_dunnett_emm_mvt:200` — EMM + multivariate-t Dunnett, method name `emm_mvt`); `validators.py`. Verify the between/within factor explanation, the post-hoc method and its adjustment, and the data structure expectation.
- [ ] **Step 3:** Rewrite html; write audit note.
- [ ] **Step 4:** Tests green.
- [ ] **Step 5:** Commit (`docs(help): audit mixed_anova recipe against code`).

### Task 6: Audit `ancova`

**Files:** Modify `src/core/help_content.py` (recipe `ancova`); Create `docs/superpowers/audit-notes/ancova.md`.

- [ ] **Step 1:** Follow the shared audit procedure for `ancova`.
- [ ] **Step 2:** Ground-truth: the actual model is `src/analysis/clinical_models.py:90 (ANCOVAModel)` — "ANCOVA via statsmodels OLS with Type III SS (Sum contrasts)" per its docstring, instantiated from `src/analysis/analysis_core.py:595 (model = ANCOVAModel())`, imported at `analysis_core.py:542`; `src/statistical_testing/advanced_pipeline.py` only dispatches into the `ancova`/`two_way_ancova` branch, it does not itself implement ANCOVA. Homogeneity-of-regression-slopes is a real method: `check_regression_slope_homogeneity` at `clinical_models.py:141`. `validators.py`. Verify the covariate handling description, the slope-homogeneity assumption claim against `check_regression_slope_homogeneity`, post-hoc method/adjustment on adjusted means (`adjusted_means` at `clinical_models.py:181`), and the data structure (group + covariate columns).
- [ ] **Step 3:** Rewrite html; write audit note.
- [ ] **Step 4:** Tests green.
- [ ] **Step 5:** Commit (`docs(help): audit ancova recipe against code`).

### Task 7: Audit `correlation`

**Files:** Modify `src/core/help_content.py` (recipe `correlation`); Create `docs/superpowers/audit-notes/correlation.md`.

- [ ] **Step 1:** Follow the shared audit procedure for `correlation`.
- [ ] **Step 2:** Ground-truth: `src/analysis/correlation_models.py` (Pearson vs Spearman selection, Shapiro-Wilk on inputs at `correlation_models.py` ~277-302, the 5000-sample cap, CI handling). Verify which coefficient is used when, the normality-driven choice, and any reported statistics (r, rho, p, CI).
- [ ] **Step 3:** Rewrite html; write audit note.
- [ ] **Step 4:** Tests green.
- [ ] **Step 5:** Commit (`docs(help): audit correlation recipe against code`).

### Task 8: Audit `linear_regression`

**Files:** Modify `src/core/help_content.py` (recipe `linear_regression`); Create `docs/superpowers/audit-notes/linear_regression.md`.

- [ ] **Step 1:** Follow the shared audit procedure for `linear_regression`.
- [ ] **Step 2:** Ground-truth: `src/analysis/correlation_models.py:456 (SimpleLinearRegressionModel)` — OLS via `statsmodels.formula.api` (verified: `correlation_models.py:499` imports `smf`, `fit()` at `:487`), continuous predictor + optional covariates, Box-Cox handling on Y via OLS residuals; `src/analysis/effect_sizes.py`. Verify reported coefficients/CI/alpha, R^2 / effect-size reporting, and the predictor/outcome data structure.
- [ ] **Step 3:** Rewrite html; write audit note.
- [ ] **Step 4:** Tests green.
- [ ] **Step 5:** Commit (`docs(help): audit linear_regression recipe against code`).

### Task 9: Audit `logistic_regression`

**Files:** Modify `src/core/help_content.py` (recipe `logistic_regression`); Create `docs/superpowers/audit-notes/logistic_regression.md`.

- [ ] **Step 1:** Follow the shared audit procedure for `logistic_regression`.
- [ ] **Step 2:** Ground-truth: `src/analysis/clinical_models.py:1041 (LogisticRegressionModel)` — CORRECTION to the spec's assumption: this uses `statsmodels` GLM with a Binomial family (`smf.glm(formula, ..., family=sm.families.Binomial())` at `clinical_models.py:~1145`), NOT `sm.Logit` directly. On convergence failure it falls back to Firth Penalized Likelihood via Newton-Raphson (`_fit_firth_logistic` at `clinical_models.py:1145`, profile CI at `_firth_profile_ci:1251`). `effect_sizes.py`. Verify the recipe's convergence/Firth-fallback description matches this exact mechanism (do not describe it as plain "statsmodels Logit"), OR/AUC reporting and ranges, default `alpha`, and the yes/no outcome data structure.
- [ ] **Step 3:** Rewrite html; write audit note.
- [ ] **Step 4:** Tests green.
- [ ] **Step 5:** Commit (`docs(help): audit logistic_regression recipe against code`).

### Task 10: Audit `dependent_samples`

**Files:** Modify `src/core/help_content.py` (recipe `dependent_samples`); Create `docs/superpowers/audit-notes/dependent_samples.md`.

- [ ] **Step 1:** Follow the shared audit procedure for `dependent_samples`.
- [ ] **Step 2:** Ground-truth: `src/analysis/statisticaltester.py` (`_wilcoxon_test:509-531`, `_mannwhitney_test:633` for contrast) for the two-group case; Friedman is NOT in statisticaltester.py — it lives in `src/analysis/nonparametricanovas.py:189 (perform_friedman_test)`, using `scipy.stats.friedmanchisquare`, with a documented warning for n<3 subjects/cells and a note suggesting Wilcoxon over Friedman for exactly 2 time points (`nonparametricanovas.py:215`). Verify the "two groups → paired t / Wilcoxon; more than two → RM ANOVA / Friedman" claim against these two modules, the matched-order / equal-n data requirement against the actual paired-data handling, and any normality-driven choice.
- [ ] **Step 3:** Rewrite html; write audit note.
- [ ] **Step 4:** Tests green.
- [ ] **Step 5:** Commit (`docs(help): audit dependent_samples recipe against code`).

### Task 11: Audit `graph_visualization`

**Files:** Modify `src/core/help_content.py` (recipe `graph_visualization`); Create `docs/superpowers/audit-notes/graph_visualization.md`.

- [ ] **Step 1:** Follow the shared audit procedure for `graph_visualization`.
- [ ] **Step 2:** Ground-truth: `src/ui/dialogs/plot_aesthetics_dialog.py` (`plot_aesthetics_dialog.py:646 (PlotAestheticsDialog ... addItems(['Bar','Box','Violin','Raincloud']))`), `src/autopilot/statistical_analyzer_autopilot_pipeline.py` (`_ap_configure_plot_from_result` ~1833/1850 — the live entry point), `src/visualization/datavisualizer.py` (point styles incl. `strip`/`swarm` ~1858-1868, error bars, significance brackets/letters), `src/templates/plot_designer.html` (Forest/Estimation, Jitter/Beeswarm).
  Apply seed findings from the spec: the live dialog offers Bar/Box/Violin/Raincloud; "strip" is a point style not a plot type; SD/SEM error bars are real; the "caps or line only" claim is unverified. BEFORE adding Forest/Estimation, prove `plot_designer.html` is reachable by citing its instantiation/entry path; if there is none, declare it dead code in the audit note and do not add those types.
- [ ] **Step 3:** Rewrite html to list the actually-available plot types and overlay/error-bar/annotation options, with citations; write audit note.
- [ ] **Step 4:** Tests green.
- [ ] **Step 5:** Commit (`docs(help): audit graph_visualization recipe against code`).

### Task 12: Audit `statistical_tests_html`

**Files:** Modify `src/core/help_content.py` (recipe `statistical_tests_html`); Create `docs/superpowers/audit-notes/statistical_tests_html.md`.

- [ ] **Step 1:** Follow the shared audit procedure for `statistical_tests_html`.
- [ ] **Step 2:** Ground-truth: test-decision logic in `src/analysis/statisticaltester.py` and `src/analysis/analysis_core.py`, `src/export/report_methods.py` (HTML report sections/content). Spec seed: the test-selection claims spot-checked accurate (t/Mann-Whitney, paired-t/Wilcoxon, ANOVA/Kruskal, Shapiro, Levene, auto post-hoc, letters/brackets) — re-verify with citations and confirm the described HTML report sections match what `report_methods.py` actually emits.
- [ ] **Step 3:** Rewrite html (likely light changes); write audit note.
- [ ] **Step 4:** Tests green.
- [ ] **Step 5:** Commit (`docs(help): audit statistical_tests_html recipe against code`).

### Task 13: Whole-diff verification and collation

**Files:** Create `docs/superpowers/audit-notes/SUMMARY.md`.

- [ ] **Step 1:** Confirm no recipe `id` or `category` changed across the whole audit:
  `git -C . diff <first-audit-parent>..HEAD -- src/core/help_content.py | grep -E '^\+\s*"(id|category)"'` → expect no output (no added id/category lines that differ).
- [ ] **Step 2:** Run the full suite: `pytest tests/ -q` → green.
- [ ] **Step 3:** Run ruff on the touched file: `python -m ruff check src/core/help_content.py` → no new errors.
- [ ] **Step 4:** Collate every audit note's "Unclear / possible code bug" section into `docs/superpowers/audit-notes/SUMMARY.md` for human review, with the recipe id and the cited code location for each item.
- [ ] **Step 5:** Commit (`docs(help): collate audit summary and possible-code-bug flags`).

---

## Self-review

- **Spec coverage:** every recipe in the spec's source map has a task (Tasks 1-12); the citation format, correction policy, invariants, and three control checks are embedded in the shared procedure; whole-diff id/category check + green suite + flagged-item collation are Task 13. Covered.
- **Resumability:** one recipe per task per commit, matching the spec's session-limit risk mitigation.
- **No new false tests:** the plan adds no per-recipe unit tests (which would risk asserting wrong "expected" content); it relies on the existing invariant suite plus citation-verified audit notes reviewed against code.
- **Placeholder scan:** module paths are concrete; where a path is a "path in src/analysis", the auditor must locate the exact module (linear/logistic regression live across a few files) — this is investigation, not a placeholder, and the citation requirement forces the exact anchor.
