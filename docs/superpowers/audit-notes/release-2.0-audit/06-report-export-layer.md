# AUDIT: BioMedStatX — Report / Export Layer (round 2) @ 3fd4796

**Scope:** `src/export/report_charts.py`, `report_summaries.py`, `report_stat_rows.py`,
`html_exporter.py`, `report_association.py`, `report_assets.py`, `report_formatting.py`,
`outlier_html_exporter.py`, `report_methods.py`, `report_tooltips.py`.

**Verdict.** No live emergency, and no change since round 1. This is a second independent
pass over the exact same code state: `git log --oneline b16cf24..HEAD -- src/export/` returns
zero commits, i.e. **nothing in this subsystem has been touched since the round-1 report was
written** (that report itself is dated at commit `b16cf24`). I re-verified every round-1
finding against current source rather than trusting the prior write-up, and all five
(EX1–EX5) are confirmed still present, byte-for-byte at the cited line numbers. I found no
new HIGH/CRITICAL-class issue beyond what round 1 already documented. The standout risk is
unchanged: a real, reproducible HTML/script-injection path in `report_association.py`'s three
coefficient-table builders, reaching the page via Jinja's `{{ chart.html | safe }}` bypass.
The previously-flagged "linear regression coefficient table never rendered" bug **is fixed**
— `coefficient_table` is written in `correlation_models.py:848` and now dispatched at
`report_charts.py:625` for `model_type == "LinearRegression"`, confirmed wired end-to-end.

## What I mechanically verified (not eyeballed)

| Check | Method | Result |
|---|---|---|
| Any `src/export/` changes since round-1 report (`b16cf24`) | `git log --oneline b16cf24..HEAD -- src/export/` | **0 commits** — code is byte-identical to what round 1 audited |
| EX1 (unescaped `parameter` cell, 3 builders) still present | Read `report_association.py:34,87,135` in full | Confirmed verbatim: `f"<td>{row.get('parameter', '')}</td>"` in all three, no `html.escape` anywhere in the file |
| `chart.html \| safe` still bypasses Jinja autoescape | `git grep -n "\| safe"` on `src/templates/*.j2` (excluding `dist/` build artifacts) | `report_single.html.j2:32`, `report_multi.html.j2:32` — `{{ chart.html \| safe }}`, confirmed; `autoescape=True` confirmed at `html_exporter.py:365` |
| `_FormattingMixin` still has no `_esc`/escape helper (EX5) | Read `report_formatting.py` in full | Confirmed — no `html.escape` import or wrapper anywhere in the file |
| EX2 unescaped `name=`/hovertext sites | Read `report_charts.py` in full (1489 lines, 3 chunks) | Confirmed at lines 288-289 (`hovertext` interaction plot), 299 (`name=f"{factor_line}={line_level}"`), 535-536 (`hovertext` mixed profile), 546 (`name=f"{factor_between}={b_level}"`); contrast with the one guarded site at 740-743 (`html.escape(str(group_name))`) and `report_summaries.py:816` (`name=str(group_name)`, unescaped) |
| EX3 sphericity key mismatch (`sphericity_met` vs `sphericity_assumed`) | `git grep -n "sphericity_assumed\|sphericity_met"` across `src/analysis/statisticaltester.py`, `src/statistical_testing/mixed_assumptions.py`, `src/visualization/decisiontreevisualizer.py` | **21 writer/reader sites total, all use `sphericity_assumed`; zero use `sphericity_met`.** `report_summaries.py:439` is the only site in the codebase reading `sphericity_met` — confirmed still wrong |
| EX4 dead fallback keys (`condition`, `main_effect:`/`main_effect_`) | `git grep` for writers of `"condition"` on trajectory points and `"main_effect:"` / `main_effect_` prefix keys across `src/analysis/`, `src/statistical_testing/` | Zero writers for either — both fallback branches (`report_charts.py:1400`, `report_stat_rows.py:541-543`) are confirmed dead/inert |
| Linear-regression `coefficient_table` now rendered (round-1 "fixed but re-verify" item) | `git grep -n "coefficient_table"` across `src/analysis/` and `src/export/` | Writer: `correlation_models.py:848`. Reader: `report_association.py:122` (`results.get("coefficient_table")`). Dispatch: `report_charts.py:622-627` gated on `model_type == "LinearRegression"` inside `_build_single_chart_bundle`. **Confirmed wired**, not dead |
| No new bare/silent `except:`/`except Exception:` (no log) in the 10 assigned files | `git grep -n "except Exception:$\|except:$"` (exit 1 = no match) | Every `except Exception as exc:` site logs (`logger.warning`/`logger.error`, mostly with `exc_info=True`) before returning `None`/a fallback — no silent swallow found |
| `.get("key")` contracts for `roc_data`, `xy_data`, `adjusted_means`, `normality_check`, `group_factor_map`, `association_points`, `plot_regression` | `git grep` each key's writer in `src/analysis/` against its reader in `src/export/` | All match — flat-dict shapes agree, no new writer/reader drift found beyond the already-known EX3 |
| `association_points` point shape (`{"x":..., "y":...}`) vs. direct-index reads (no `.get`) at `report_summaries.py:542-543` | Read `correlation_models.py:266` (`self._points = [{"x": float(x), "y": float(y)} for x, y in zip(...)]`) | Shape matches; direct indexing is safe, not a latent `KeyError` |
| `_build_significance_brackets`'s `comparison.split(" vs ")` fallback reachability | `git grep -n '"group1"'` across every `pairwise_comparisons` writer in `src/analysis/` | Every writer sets `group1`/`group2` directly; the string-split fallback (`report_charts.py:1350-1351`) is unreachable in practice, not a live bug |
| `_effect_is_ratio`'s "ratio/odds/fold/hazard" string matching | `git grep -n '"effect_size_type"'` across `src/analysis/` for any ratio/odds/fold/hazard value | No writer ever emits such a value on `pairwise_comparisons[].effect_size_type` — the ratio-detection branch in `html_exporter.py:25-35` is defensive/forward-looking, currently always `False` in production, not a live defect |

## Findings — severity ranked

### HIGH

**RE1 — Unescaped `parameter` cell in 3 injected HTML tables → stored/reflected HTML injection in the exported report. (= round-1 EX1, unfixed.)**
`src/export/report_association.py:34` (`_build_or_table_html`), `:87`
(`_build_beta_coefficient_table_html`), `:135`
(`_build_linear_regression_coefficient_table_html`) each build a row via:
```python
rows_html += f"<tr><td>{row.get('parameter', '')}</td>..."
```
`parameter` is `str(param)` from a statsmodels `.params.index` entry — for a categorical
predictor wrapped `C(col)`, patsy encodes the raw category *value* the user typed into their
group/category column directly into the parameter name (`C(group)[T.<value>]`). All three
tables are wired into `chart_blocks` (`report_charts.py:606,615,625`) and rendered via
`{{ chart.html | safe }}` (`src/templates/report_single.html.j2:32`,
`report_multi.html.j2:32`), which explicitly opts out of Jinja's `autoescape=True`
(confirmed set at `html_exporter.py:365`). A group literally named
`Zebra<script>alert(1)</script>` reaches the exported `.html` file as a live `<script>` tag
(round 1 reproduced this end-to-end with a live `statsmodels.formula.api.ols` fit; the code
path is unchanged, so the reproduction still applies).
**Impact:** single-user local desktop tool, so not a multi-tenant/server compromise — but the
generated `.html` is explicitly designed to be portable/shareable (methods paragraph,
"ready-to-paste" language, fully offline Plotly/KaTeX bundling). If a report is emailed,
uploaded to a shared drive, or opened by a second person, attacker-controlled content in a
CSV/Excel group column becomes script execution in whoever's browser opens the file.
**Fix:** wrap `row.get('parameter', '')` (and any other free-text cell value) in
`html.escape()` before interpolation in all three builders. The `<strong>`/style wrapping
around `coef_display`/`p_style` is fine to keep — those are internally-formatted numbers, not
user strings.

### MEDIUM

**RE2 — Inconsistent escaping of group/factor-level labels in Plotly trace `name=`/`hovertext`. (= round-1 EX2, unfixed.)**
`src/export/report_charts.py:740-743` (`_build_group_comparison_chart`) explicitly
HTML-escapes the group name (`html.escape(str(group_name))`) before using it as a Plotly
trace `name=`. The near-duplicate `src/export/report_summaries.py:816`
(`_build_distribution_dashboard_chart`) does **not** escape (`name=str(group_name)`), and
neither do `report_charts.py:288-289` (interaction-plot `hovertext`, `f"{factor_x}={x_val},
{factor_line}={line_level}<br>..."`), `:299` (`name=f"{factor_line}={line_level}"`),
`:535-536` (mixed-profile `hovertext`), or `:546`
(`name=f"{factor_between}={b_level}"`) — all take factor-level strings straight from the
`descriptive` dict keys, which originate from the user's raw category values.
**Impact:** lower confidence than RE1 — Plotly.js typically renders `name`/`hovertext` as SVG
`<text>` (browser-escaped, not `innerHTML`), so this is not confirmed-reproducible script
execution, only a confirmed unescaped-interpolation inconsistency with the one function that
does guard it. **Fix:** apply `html.escape()` uniformly at all four unescaped sites,
regardless of the exploitability question — cheap and removes the inconsistency.

### LOW

**RE3 — Sphericity status row reads a key (`sphericity_met`) no writer ever sets; masked by a partial fallback. (= round-1 EX3, unfixed.)**
`src/export/report_summaries.py:439`: `status_value = sphericity.get("sphericity_met")`.
All 21 writer/reader sites for the `sphericity_test` dict across
`src/analysis/statisticaltester.py` and `src/statistical_testing/mixed_assumptions.py` use
the key `sphericity_assumed` — never `sphericity_met`. Line 440-441 has a fallback
(`if status_value is None and sphericity.get("p_value") is not None: status_value = p_value
>= 0.05`) that recomputes the right answer in the common case, which is why this isn't a
total outage. But the fallback is incomplete: for a 2-level within-subject factor, the writer
sets `p_value: None` **and** `sphericity_assumed: True` (sphericity is trivially satisfied
with only 2 levels — `statisticaltester.py:2624-2632`); in that case both the wrong-key
lookup and the `p_value` fallback come up empty, so `status_value` stays `None` and the
report shows "Not available" for a condition that is actually definitionally `True`.
**Impact:** narrow (only the 2-level RM-ANOVA path) and purely cosmetic (a neutral badge
instead of "passed"), not a wrong p-value or corrected-test omission.
**Fix:** read `sphericity.get("sphericity_assumed")` instead of `sphericity.get
("sphericity_met")` at line 439; keep the `p_value` fallback for defense-in-depth.

**RE4 — Two dead reader-side fallback keys with no writer anywhere (inert, not exploitable). (= round-1 EX4, unfixed — no urgency.)**
1. `src/export/report_charts.py:1400`: `point.get("group") or point.get("condition")` — every
   trajectory-point writer (`src/analysis/statisticaltester.py:434-441`) only ever sets
   `"group"`; `"condition"` has zero writers anywhere in the tree.
2. `src/export/report_stat_rows.py:541-543`: the `else` branch matching top-level keys
   prefixed `main_effect:`/`main_effect_` — zero writers of such a key exist in
   `src/analysis/` or `src/statistical_testing/`; the current writer path always populates
   the structured `primary_effect` dict, handled by the `if isinstance(primary_effect, dict)`
   branch immediately above.
**Fix (optional, cosmetic):** delete both dead branches in a future cleanup pass; they
cannot misfire, so there's no urgency.

**RE5 — `_FormattingMixin` still has no HTML-escaping helper, unlike its sibling `outlier_html_exporter.py`. (= round-1 EX5, unfixed — structural root cause of RE1/RE2.)**
`report_formatting.py`'s `_FormattingMixin` provides `_format_metric`, `_format_p_value`,
`_prettify_label`, etc., but no `_esc`/`html.escape` wrapper. Every field that reaches the
main Jinja-templated tables is protected only by Jinja's `autoescape=True` — correct and
sufficient as long as no caller opts out with `| safe`. RE1/RE2 are exactly that opt-out, and
there is still no shared, reusable escaping helper in this file for a caller that legitimately
needs raw HTML (chart tables). **Fix:** add a static `_esc(value) -> str` (thin
`html.escape(str(value))` wrapper) to `_FormattingMixin`, mirroring
`outlier_html_exporter.py`'s existing, correct `_esc()`, and use it at every raw-HTML
f-string site identified in RE1/RE2.

## Strengths (verified)

- **Zero drift since round 1** — `git log b16cf24..HEAD -- src/export/` is empty, so this
  round independently re-derived every finding from the live source rather than trusting the
  prior write-up, and all five reproduce exactly as previously reported. No regressions, no
  silent partial fixes that half-close a finding while leaving a gap.
- **The linear-regression coefficient-table dispatch bug is genuinely fixed and still wired.**
  `coefficient_table` (written `correlation_models.py:848`) is read by
  `_build_linear_regression_coefficient_table_html` (`report_association.py:122`) and
  dispatched from `_build_single_chart_bundle` for `model_type == "LinearRegression"`
  (`report_charts.py:622-627`) — confirmed end-to-end, not dead code.
- **The writer/reader contract is overwhelmingly clean elsewhere.** Spot-checked
  `roc_data`, `xy_data`, `adjusted_means`, `normality_check`, `group_factor_map`,
  `association_points`/`plot_regression` (including the nested point shape
  `{"x":..., "y":...}` against a direct-index reader with no `.get()` guard) — every one
  matches its writer field-for-field, flat vs. nested shape included.
- **`outlier_html_exporter.py` gets escaping right, consistently.** Its `_esc()` helper
  (`html.escape`) wraps every user-controlled string (dataset name, group value, error
  message) before interpolation into `<td>`/`<h2>`/`<img alt>` — still the correct template
  for how `report_association.py` should be fixed.
- **Jinja autoescaping is correctly configured** (`autoescape=True`,
  `html_exporter.py:365`), and the large majority of report fields (`statistical_rows`,
  `assumptions.rows`, `descriptive.rows`, `pairwise_rows`) go through it un-bypassed — the
  `| safe` opt-outs remain narrowly scoped to Plotly/KaTeX/MathJax bundles and the handful of
  chart-table blocks, not a blanket bypass.
- **No silent exception swallowing found in this batch.** Every `except Exception as exc:`
  site across all 10 files logs via `logger.warning`/`logger.error` (mostly with
  `exc_info=True`) before falling back to `None`/an empty result — a caller inspecting logs
  can always find out why a chart or section didn't render. `export_dispatcher.py` (adjacent,
  not in the assigned batch but read for context) also correctly propagates a `warning`
  string to the caller instead of swallowing export failures.
- **Several defensive fallback branches that look suspicious on first read are confirmed
  unreachable, not silently-wrong.** `_build_significance_brackets`'s `comparison.split(" vs
  ")` fallback and `_effect_is_ratio`'s ratio/odds/fold string matching both have zero live
  writers that would exercise them — they're forward-compatible dead code, not active bugs.

## Recommended remediation order

1. **RE1 (HIGH)** — add `html.escape()` around the `parameter` cell in all three
   `report_association.py` builders. Smallest possible diff, closes the only reproducible
   injection path in this batch. Unchanged priority from round 1.
2. **RE5** — add a shared `_esc()` helper to `_FormattingMixin` as part of the same change
   (RE1 needs it to exist somewhere reusable; do it once, not three times inline).
3. **RE2 (MEDIUM)** — apply the same escaping at the four unescaped Plotly `name=`/
   `hovertext` sites for consistency, even though exploitability there is unconfirmed —
   cheap, and removes a latent risk plus the inconsistency with the one already-guarded site.
4. **RE3 (LOW)** — one-line key fix (`sphericity_met` → `sphericity_assumed`) in
   `report_summaries.py:439`.
5. **RE4 (LOW, optional)** — delete the two dead fallback branches (`condition`,
   `main_effect:`/`main_effect_`) during a future cleanup pass; no urgency, they cannot
   misfire.

**Note on why RE1–RE5 are unfixed a second time:** this is not evidence the round-1 report
was ignored — `git log` confirms zero commits touched `src/export/` between the two audit
passes, meaning round 2 ran before any remediation work started. These are the same fixes
waiting on a decision, not a regression.
