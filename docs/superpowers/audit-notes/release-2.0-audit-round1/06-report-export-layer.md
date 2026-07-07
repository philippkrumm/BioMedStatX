# AUDIT: BioMedStatX — Report / Export Layer @ b16cf24

**Scope:** `src/export/report_charts.py`, `report_summaries.py`, `report_stat_rows.py`,
`html_exporter.py`, `report_association.py`, `report_assets.py`, `report_formatting.py`,
`outlier_html_exporter.py`, `report_methods.py`, `report_tooltips.py`, `export_dispatcher.py`.

**Verdict.** No live emergency. The writer/reader key-contract problem this batch was
tasked to hunt (report reads a key the analysis layer never populates, or populates under
a different name/shape) is now mostly clean after this session's earlier fixes — I found
one more live instance (sphericity status masked by a wrong key, silently papered over by
a fallback) and two harmless dead-code remnants, not a new epidemic. The more serious find
is a **real, reproducible HTML/script-injection path**: three coefficient-table builders in
`report_association.py` build raw HTML via f-strings with an unescaped `parameter` cell,
and that HTML reaches the page through Jinja's `{{ chart.html | safe }}` — bypassing the
template's own autoescaping. I reproduced the payload end-to-end with a real statsmodels
fit: a categorical group named `Zebra<script>alert(1)</script>` survives verbatim into the
rendered report as a live `<script>` tag (HIGH).

## What I mechanically verified (not eyeballed)

| Check | Method | Result |
|---|---|---|
| Every `.get("key")` in the 4 target builder files has a real writer in `src/analysis/` or `src/statistical_testing/` | `git grep -c "\"key\""` per key, ~140 distinct keys | All have a writer except 2 report-internal false positives (payload built earlier in the same file, not read from `results`) and 2 confirmed-dead fallback keys (below) |
| Nesting/shape match for every non-trivial nested structure read (`sphericity_test`, `descriptive` two-way/mixed key format, `anova_table`, `fixed_effects_table`, `odds_ratios`, `coefficients`/`coefficient_table`, `RTE`, `simple_slopes_analysis`/`johnson_neyman`, `roc_data`, `xy_data`, `association_points`) | Read the writer function body, compared field-by-field against the reader | All match **except** `sphericity_test["sphericity_assumed"]` vs reader's `sphericity.get("sphericity_met")` (EX1) |
| Every `_build_*` chart/table builder in `report_charts.py`/`report_association.py` is reachable from `_build_single_chart_bundle`'s dispatch | `grep` each `def _build_*` name, counted call sites elsewhere in `src/export/` | All 17 are wired in; 0 orphaned builders (the linear-regression-coefficient-table bug from earlier this session is fixed and now correctly dispatched at `report_charts.py:622-627`) |
| Every `_build_*` in `report_summaries.py`/`report_stat_rows.py` is reachable | Same method | All 15 wired in |
| HTML-escaping discipline across the mixin package | Read `report_formatting.py` in full; grepped for `html.escape`/`|safe`/raw f-string `<td>`/`<tr>` construction | `_FormattingMixin` has **no escaping helper at all**; `outlier_html_exporter.py` has its own correct `_esc()` used everywhere; `report_association.py`'s 3 coefficient-table builders build raw HTML with **zero** escaping of the `parameter` field |
| Confirmed the injection is reachable with real user data, not just in theory | Ran `statsmodels.formula.api.ols("y ~ C(group)")` locally with a group value `Zebra<script>alert(1)</script>` | Parameter name comes out as `C(group)[T.Zebra<script>alert(1)</script>]` verbatim — patsy/statsmodels does not sanitize category labels |
| Jinja autoescape configuration | Read `html_exporter.py:_render_template` | `Environment(..., autoescape=True)` — correct default; the 3 injection points bypass it deliberately via `| safe` on `chart.html` |

## Findings — severity ranked

### HIGH

**EX1 — Unescaped `parameter` cell in 3 injected HTML tables → stored/reflected HTML injection in the exported report.**
`src/export/report_association.py:34` (`_build_or_table_html`), `:87` (`_build_beta_coefficient_table_html`),
`:135` (`_build_linear_regression_coefficient_table_html`) each do:
```python
rows_html += f"<tr><td>{row.get('parameter', '')}</td>..."
```
`parameter` is `str(param)` from a statsmodels `.params.index` entry
(`src/analysis/clinical_models.py:1354,1373`, `src/analysis/correlation_models.py:787`) — for a
categorical predictor wrapped `C(col)` (`clinical_models.py:1095,1625`), patsy encodes the raw
category *value* into the parameter name (`C(group)[T.<value>]`), and that value is whatever the
user typed into their group/category column. All three tables are wired into `chart_blocks`
(`report_charts.py:606,615,625`) and rendered via `{{ chart.html | safe }}`
(`src/templates/report_single.html.j2:32`), which explicitly **opts out** of Jinja's
`autoescape=True`. A group literally named `Zebra<script>alert(1)</script>` (verified locally with
a live `statsmodels.formula.api.ols` fit) reaches the exported `.html` file as a live `<script>`
tag. **Impact:** the app is a single-user local desktop tool (per `my-environment.md`), so this is
not a multi-tenant/server compromise — but the generated `.html` file is explicitly designed to be
portable/shared (methods paragraph, "ready-to-paste" language, offline bundling of Plotly/KaTeX).
If a report is emailed to a collaborator, uploaded to a shared drive, or opened by a second person,
attacker-controlled content in a CSV/Excel group column becomes script execution in whoever's
browser opens the file — a realistic path for a biomedical/clinical tool that ingests
externally-sourced spreadsheets. **Fix:** escape `row.get('parameter', '')` (and every other
free-text cell value) with `html.escape()` before interpolation in all three functions — the
`<strong>`/style wrapping around numeric `coef_display`/`p_style` is fine to keep since those are
internally-formatted numbers, not user strings. Cheapest fix: add an `_esc()` static helper to
`_FormattingMixin` (mirroring `outlier_html_exporter.py`'s existing, correct `_esc()`) and apply it
at the `parameter` interpolation site in all three builders.

### MEDIUM

**EX2 — Inconsistent escaping of group/factor-level labels in Plotly trace `name=`.**
`src/export/report_charts.py:740-743` (`_build_group_comparison_chart`) explicitly HTML-escapes
the group name (`html.escape(str(group_name))`) before using it in a Plotly trace `name=`. The
near-duplicate `src/export/report_summaries.py:816` (`_build_distribution_dashboard_chart`) does
**not** escape (`name=str(group_name)`), and neither does `report_charts.py:299`
(`f"{factor_line}={line_level}"`, interaction plot), `:546` (`f"{factor_between}={b_level}"`, mixed
profile plot), or `:1271` (`f"{label} (n={n})"`, ANCOVA adjusted-means bars) — all four take
factor-level strings straight from the `descriptive`/`adjusted_means` dict keys, which originate
from the user's raw category values. Plotly.js legend `name` typically renders as SVG `<text>`
(escaped by the browser's own SVG text node handling, not `innerHTML`), so this is **lower
confidence than EX1** — I did not reproduce script execution through this path, only confirmed the
unescaped interpolation exists and is inconsistent with the one function that *does* guard it.
**Fix:** either verify Plotly's `name=`/`hovertext=` rendering is provably safe for all supported
Plotly versions and document why, or apply the same `html.escape()` used at
`report_charts.py:741` uniformly at all four sites — cheap, and removes the inconsistency
regardless of the exploitability question.

### LOW

**EX3 — Sphericity status row reads a key (`sphericity_met`) no writer ever sets; masked by a fallback that only partially covers the gap.**
`src/export/report_summaries.py:439`: `status_value = sphericity.get("sphericity_met")`. Every
writer of the `sphericity_test` dict (`src/analysis/statisticaltester.py:2624-2788`,
`src/statistical_testing/mixed_assumptions.py` — 14 write sites total) uses the key
`sphericity_assumed`, never `sphericity_met`. Line 440-441 has a fallback
(`if status_value is None and sphericity.get("p_value") is not None: status_value = p_value >= 0.05`)
that recomputes the right answer in the common case, which is why this wasn't caught as a total
outage. But the fallback itself is incomplete: for a 2-level within-subject factor, the writer sets
`p_value: None` **and** `sphericity_assumed: True` (sphericity is trivially satisfied with only 2
levels — `statisticaltester.py:2624-2632`) — in that case both the wrong-key lookup and the
`p_value` fallback come up empty, so `status_value` stays `None` and the report renders "Not
available" / neutral styling for a condition that is actually definitionally `True`.
**Impact:** narrow — only the 2-level RM-ANOVA path, and only cosmetic (a "not available" badge
instead of "passed"), not a wrong p-value or corrected-test omission. **Fix:** read
`sphericity.get("sphericity_assumed")` instead of `sphericity.get("sphericity_met")` at line 439;
keep the `p_value` fallback for defense-in-depth on any writer that's missed in the future.

**EX4 — Two dead reader-side fallback keys with no writer anywhere (inert, not exploitable).**
1. `src/export/report_charts.py:1400`: `point.get("group") or point.get("condition")` — every
   trajectory-point writer (`src/analysis/statisticaltester.py:434-441`) only ever sets `"group"`;
   `"condition"` has zero writers. Harmless because `"group"` is always present.
2. `src/export/report_stat_rows.py:541-543`: the `else` branch matching keys prefixed
   `main_effect:`/`main_effect_` on the top-level `results` dict — `git grep` across
   `src/analysis/` and `src/statistical_testing/` found **no** writer that ever sets such a
   top-level key (the current writer path always populates the structured `primary_effect` dict,
   which the `if isinstance(primary_effect, dict)` branch immediately above already handles).
   This is legacy/dead code from an older results shape. **Fix (optional, cosmetic):** delete both
   dead branches, or leave them — they cannot fire incorrectly, so this is a cleanliness note, not
   a correctness bug.

**EX5 — `_FormattingMixin` has no HTML-escaping helper, unlike its sibling `outlier_html_exporter.py`.**
`src/export/report_formatting.py`'s `_FormattingMixin` provides `_format_metric`, `_format_p_value`,
`_prettify_label`, etc., but no `_esc`/`html.escape` wrapper — every other field that reaches the
main Jinja-templated tables (`row.label`, `row.value`, `assumptions.rows[].name`, etc.) is protected
only by Jinja's `autoescape=True`, which is correct and sufficient *as long as no caller opts out
with `| safe`*. The three EX1 sites are exactly that opt-out, and there is no shared, reusable
escaping helper in this file to reach for when a raw-HTML block **is** legitimately needed (chart
tables, KaTeX/MathJax bootstrapping). Structural root cause behind EX1/EX2.
**Fix:** add a static `_esc(value) -> str` (thin `html.escape(str(value))` wrapper) to
`_FormattingMixin`, matching `outlier_html_exporter.py`'s pattern, and use it at every raw-HTML
f-string site in `report_association.py` and the two unescaped `name=` sites in EX2.

## Strengths (verified)

- **The writer/reader contract is now overwhelmingly clean.** Of ~140 distinct `.get("key")` calls
  cross-referenced against every write site in `src/analysis/` and `src/statistical_testing/`, only
  one live shape mismatch remains (EX3) and two are inert dead branches (EX4) — down from the 3
  same-class bugs already fixed this session (sphericity correction/epsilon, RTE
  `between_group`/`within_level`, the orphaned linear-regression coefficient table). The ANCOVA
  (`anova_table`, `covariate_effects`, `simple_slopes_analysis`/`johnson_neyman`), LMM
  (`fixed_effects_table`), Beta/Logistic/Linear regression coefficient tables, ROC data
  (`roc_data.fpr/tpr/auc`), and TwoWay/Mixed-ANOVA `descriptive` dict key format
  (`f"{factor_a}={a_val}, {factor_b}={b_val}"`) were all checked field-by-field against their
  writers and match exactly.
- **Dispatch completeness is now total.** Every `_build_*` chart, table, and stat-row function in
  `report_charts.py`, `report_association.py`, `report_summaries.py`, and `report_stat_rows.py` is
  reachable from `_build_single_chart_bundle` or `_build_statistical_rows` — zero orphaned builders
  found, confirming the earlier fix for the linear-regression coefficient table closed that class
  of bug for this release.
- **`outlier_html_exporter.py` gets escaping right.** Its `_esc()` helper (`html.escape`) is applied
  consistently to every user-controlled string (dataset name, group value, error message) before
  interpolation into `<td>`/`<h2>`/`<img alt>` — a clean template for how `report_association.py`
  should be fixed.
- **Jinja autoescaping is correctly configured** (`autoescape=True` in `html_exporter.py:365`), and
  the vast majority of report fields (`statistical_rows`, `assumptions.rows`, `descriptive.rows`,
  `pairwise_rows`) go through it un-bypassed — the `| safe` opt-outs are narrowly scoped to
  Plotly/KaTeX/MathJax bundles and the handful of chart-table blocks, not a blanket bypass.
- **`_safe_json_dumps`** (`html_exporter.py:46-48`) neutralizes `</script>` sequences
  (`s.replace("</", "<\\/")`) before embedding JSON payloads in inline `<script>` blocks — correct
  defense against breaking out of a script context via a crafted string inside `normalized_results_json`,
  `pairwise_data_json`, etc.
- **The already-fixed sphericity-correction-label logic in `report_summaries.py:451-480`** correctly
  prefers the top-level `correction_used` string and the nested `sphericity_corrections.{greenhouse_geisser,huynh_feldt}.epsilon`
  fields (verified against 5 real write sites in `statisticaltester.py`/`mixed_assumptions.py`), with a defensive
  fallback to older/serialized shapes — this is the right pattern and should be the template for EX3's fix.

## Recommended remediation order

1. **EX1 (HIGH)** — add `html.escape()` around the `parameter` cell in all three
   `report_association.py` builders. Smallest possible diff, closes the only reproducible
   injection path found in this batch.
2. **EX5** — add a shared `_esc()` helper to `_FormattingMixin` as part of the same change (EX1
   depends on it existing somewhere reusable; do it once, not three times inline).
3. **EX2 (MEDIUM)** — apply the same escaping at the four unescaped Plotly `name=` sites for
   consistency, even if the exploitability is unconfirmed — cheap and removes a latent risk.
4. **EX3 (LOW)** — one-line key fix (`sphericity_met` → `sphericity_assumed`) in
   `report_summaries.py:439`.
5. **EX4 (LOW, optional)** — delete the two dead fallback branches (`condition`, `main_effect:`/`main_effect_`)
   during a future cleanup pass; no urgency, they cannot misfire.
