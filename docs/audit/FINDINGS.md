# BioMedStatX Audit — Master Findings (prioritized)

Severity: 🔴 high · 🟡 med · ⚪ low. Status: OPEN unless marked ✅.
Audit scope: Phase A (deps) + Chunks 1-6 (analysis_core, statisticaltester, posthoc_core,
nonparametricanovas, clinical_models, export) + Phase C (tests).
~40 substantive findings. Line numbers as of audit date; re-verify before editing.

Verdicts: A ✗ · C1 ✗ · C2 ✗ · C3 ✓* · C4 ✓* · C5 ✗ · C6 ✗ · Tests ✓**

---

## ✅ DONE (Phase A — already fixed + verified)
- **A-1** wilcoxon `exact=` dead crutch → `method='exact'/'approx'` — posthoc_core.py:1640, nonparametricanovas.py:124. FIXED+runtime-verified.
- **A-2** Dependency pinning: requirements.txt core pinned + requirements.lock written; jinja2 3.1.4→3.1.6 (CVE-2024-56326/CVE-2025-27516), plotly 5.24.1→6.6.0. FIXED.

---

## 🔴 HIGH (6) — fix first

### C1-1 — analysis_core.py:1679 (+ :1616/:1618/:1622/:1624/:1626; :788 vs :1680) — Dim 7+6
Accumulated rich `analysis_log` is discarded; `build_analysis_log` rebuilds from keys that
are never set (`import_status`/`group_sizes`/`test_recommendation`/`normality_p`/`levene_p`)
→ degraded log in `results["analysis_log"]`. HTML export uses the rich log (:1416), the
returned dict the poor one. Clinical path stores the rich log (:788) → Clinical/Standard divergence.
**Fix:** delete the `build_analysis_log` rebuild; set `results["analysis_log"]` = the
accumulated `analysis_log` (the one already passed to export at :1416). Unifies both paths.

### C1-2 — analysis_core.py:1408-1424 — Dim 5
Standard-path export has no try/finally; an export exception jumps to the outer `except`
(:1683) and `os.chdir(original_dir)` is never restored → wrong cwd contaminates the next
dataset in `_analyze_multiple_datasets`.
**Fix:** wrap export in `try/finally: os.chdir(original_dir)` (mirror the clinical path :771-786).

### C2-1 — statisticaltester.py:589-596 — Dim 3+6
Welch (`equal_var=False`) branch labels `effect_size_type="hedges_g"` but computes plain
pooled Cohen's d — identical formula to the equal-var branch, no Hedges J correction.
**Fix:** apply `J = 1 - 3/(4*(n1+n2)-9)` (`g = d*J`) and keep label "hedges_g", OR relabel
"cohen_d". Decide pooled vs unpooled SD for Welch while there.

### C2-2 — statisticaltester.py:1815 — Dim 2+3
RM within-pairwise Cohen's d = `mean(diff)/np.std(diff)` → ddof=0 (population), inconsistent
with ddof=1 everywhere else; also no zero-SD guard → inf/nan d on constant differences.
**Fix:** `PostHocStatistics.calculate_cohens_d(data1, data2, paired=True)` (ddof=1, zero→0),
type "cohen_d_rm". Reference convention: posthoc_core.py:1302.

### C5-1 — clinical_models.py:1104 / :1125 / :1524 — Dim 2+4
Logistic `converged` used internally to trigger Firth but absent from `as_results_dict`.
Worst case (standard non-converged + Firth raises) → "Keeping standard logit results" (:1125)
unflagged → unreliable/inf ORs presented as "Standard Maximum Likelihood".
**Fix:** add `"converged"` to the dict; on the Firth-fail branch set converged=False +
append a result warning. (Pairs with C6-1.)

### C5-2 — clinical_models.py:1602 / :1713 — Dim 2+4
Beta main fit never inspects convergence; no `converged` in dict. Non-converged beta
(φ→∞, quasi-separation in proportions) → garbage coef/p/phi emitted unflagged; `phi` extract
(:1688) unguarded vs inf.
**Fix:** read `self.result.mle_retvals.get('converged')` (or `.converged`) post-fit; add to
dict; guard phi finite; warn on non-convergence.

---

## 🟡 MED (19)

| ID | Module:line | Dim | Problem | Fix-vector |
|---|---|---|---|---|
| C1-3 | analysis_core.py:~1274 | 4/3 | No p∈[0,1] band check at export funnel; nonfinite_block catches only NaN/Inf | Extend the funnel guard to block/flag p∉[0,1] (also TV-2, C6-2) |
| C1-4 | analysis_core.py:1683-1689 | 4 | Outer except collapses all errors to {"error":str(e)}; `traceback.print_exc()` to dead stdout | `logger.exception(...)`; include exc type in the error dict |
| C1-5 | analysis_core.py:783-789 | 4 | Clinical export failure swallowed → returns "success" w/o report | Set `results["export_warning"]` / surface to caller |
| C1-6 | analysis_core.py:361-367 | 1 | Multi-dataset FDR filter `isinstance(p,(float,int))` keeps NaN + bool → NaN into multipletests | Filter `isinstance(p,float) and not bool and math.isfinite(p)` |
| C1-7 | analysis_core.py:1672 | 6 | `f"{p_val:.4f}"` unguarded vs safe_format at :1369 | Use safe_format (moot if C1-1 removes the rebuild) |
| C1-8 | analysis_core.py:243 | 5 | Mutates shared injected `analysis_context["group_factor_map"]` | Return via prepared dict / write to a local copy |
| C2-3 | statisticaltester.py:537-541 | 3 | Wilcoxon "power" = rank-biserial r×0.955 into TTestPower = meaningless | `power=None` (mirror MWU :649) |
| C2-4 | statisticaltester.py:917 | 3 | Welch ANOVA Cohen's f = sqrt(F·df1/N), non-standard, mislabeled | `η²=F·df1/(F·df1+df2)`, `f=sqrt(η²/(1−η²))` |
| C2-5 | statisticaltester.py:2899/:2831/:2909 | 3 | Sphericity "conservative" default = sphericity_assumed=True on undetermined → no correction → inflated Type-I | Default to applying GG (truly conservative) or flag "indeterminate" hard |
| C2-6 | statisticaltester.py:902/:1817 | 6 | Container-key proliferation effects/factors/interactions/within_pairwise | RESOLVED at export (within merged :1958-1959) — monitor; optionally unify names |
| C2-7 | statisticaltester.py:262/:267/:278/:304/:1088 | 1/2 | `not values` / `orig!=trans` raise "ambiguous truth value" on ndarray samples (legacy path) | Normalize samples to list at ingestion OR use len()/np.array_equal |
| C2-8 | statisticaltester.py:476/:606/:1098 | 3/6 | CI hardcoded 0.95, ignores `alpha` | Use `1-alpha` confidence (also C5-5) |
| C3-1 | posthoc_core.py:1122 | 6 | Internal key `"d"` holds degrees-of-freedom, not Cohen's d (footgun) | Rename to `"df"` |
| C3-2 | posthoc_core.py:1185+:1217 | 3/6 | Dunnett-RM emits non-control pairs at forced p=1.0 as if tested | Emit only control comparisons, or flag non-control "not in family" |
| C4-1 | nonparametricanovas.py:465-466 | 3/4 | Freedman-Lane `df1<1 → df1=1` masks rank-deficient design | Detect df1<1 → flag/block degenerate design |
| C4-2 | nonparametricanovas.py:756/:778/:793 | 3/4 | ATS empty-cell RTE=nan → ATS=nan; `tr_TV<=0` guard misses nan; per-effect nan p slips | nan-guard p_hat/tr_TV → block/warn |
| C5-3 | clinical_models.py:365 | 3 | Simple-slopes cov_sd ddof=0 (pick-a-point Mean±1SD) | ddof=1 (Aiken-West) |
| C5-4 | clinical_models.py:1627 | 3 | Bootstrap SE `boot_arr.std()` ddof=0 → underestimate (~5% at B=10) | ddof=1 |
| C5-5 | clinical_models.py:1643-1645 | 6 | Beta bootstrap CI hardcoded 1.96 + normal z, ignores alpha | t-crit with alpha (or document normal-approx) |
| C6-1 | report_stat_rows.py:214 / report_summaries.py:311 | 4 | Converged display LMM-only → logistic/beta convergence end-to-end blind | Model-agnostic converged row + logistic/beta builder rows (pairs w/ C5-1/C5-2) |
| C6-2 | report_formatting.py:186-190 | 3 | Out-of-band finite p verbatim; negative p → "p<0.001 ***" + green heat | Clamp/guard `0<=p<=1` in `_format_p_value` → "invalid (p=X)" |
| C6-3 | export_dispatcher.py:28-29 | 4 | Nested `except Exception: pass` swallows export-handling failure | Remove bare except/pass; surface the error |
| TV-1 | tests/test_rm_anova_lmm_redirect.py:47 | 7 | Strawman `len(pairwise)>0` — value not asserted → C2-2 invisible | Assert cohen_d_rm value/sign + add golden |
| TV-2 | tests/robustness/contract.py:17 | 3 | `assert_graceful` checks p finite, not [0,1]; band only in out-of-CI fuzzer | Add `0<=p<=1` assertion to the contract |
| TV-3 | tests/* (none) | 6 | No test asserts a `converged` key → C5-1/C5-2/C6-1 invisible | Add converged-key tests w/ non-converged fixtures (logistic/beta) |
| TV-4 | pyproject testpaths + fuzzing/run_fuzzer.py | 7 | Strongest oracle (p∈[0,1], df>0, F/p-tail) is out-of-CI | Add a bounded seeded fuzz campaign to CI |

(Count note: C2-6 downgraded to monitor; TV-2/C1-3/C6-2 are the same band-gap at three layers.)

---

## ⚪ LOW (13)

| ID | Module:line | Dim | Problem | Fix-vector |
|---|---|---|---|---|
| C1-9 | analysis_core.py:825-830/:1558/:424 | 7 | Identical elif/else; stale "line 5647" comment; dup comment; unused `requested_transform` | Dedup/cleanup |
| C1-10 | analysis_core.py:257 vs :198 | 6 | Samples value type non-uniform (list autopilot / ndarray legacy) | Normalize at prepare boundary |
| C2-9 | statisticaltester.py:954-1084 | 7 | `_perform_dunnett_t3_posthoc` dead (0 callers), own unguarded se=0/pooled-SD=0 | Delete |
| C2-10 | statisticaltester.py:1239-1242 | 7 | Redundant double `except` (ValidationError + Exception identical) | Single `except Exception` |
| C2-11 | statisticaltester.py:259+:277 | 7 | Descriptive computed twice | Remove the duplicate |
| C2-12 | statisticaltester.py:480 | 6 | paired-t power via `abs(None)`→except vs explicit guard at :609 | `if cohen_d is not None` |
| C2-13 | statisticaltester.py:1183 | 2 | Formula sanitization spaces-only; other Patsy-special chars unhandled | Robust sanitize (non-alnum→_) + consistent col rename |
| C3-3 | posthoc_core.py:1156/:1270 | 4/7 | Bare `except:`; `traceback.print_exc()` to dead stdout | `except Exception` + `logger.exception` |
| C3-4 | posthoc_core.py:332-630 | 7 | `_perform_test_legacy` dead (~300 LOC), emits hybrid Shape-2 | Delete |
| C3-5 | posthoc_core.py:1093 | 1 | RM pairing silently `continue`s pairs with unequal len / <3 | Collect+surface skipped pairs in warnings |
| C4-3 | nonparametricanovas.py:480 | 5 | Freedman-Lane `df.copy()` in perm loop → ~15k copies, ~30k OLS fits | Hoist copy / mutate preallocated column |
| C4-4 | nonparametricanovas.py:134 vs statisticaltester.py:532 | 6 | Rank-biserial normalizer n(incl zeros) vs n_eff(excl) | Unify on n_eff |
| C4-5 | nonparametricanovas.py:775 | 1 | `RTE_mat … .values[0]` IndexError on empty between-level | Guard / default nan |
| C4-6 | nonparametricanovas.py:201 | 1 | Friedman/ATS silent complete-case dropna | Surface dropped-subject count |
| C5-6 | clinical_models.py:1092 | 2 | Logistic separation heuristic `bse>5.0` magic threshold | Named const + comment / add condition-number check |
| C5-7 | clinical_models.py:1671 | 6 | Beta `main_p` = first coefficient only (arbitrary multi-predictor) | Document or use omnibus |
| C6-4 | templates (CSV export) | 6 | CSV/stat-table download exports display strings, not raw numbers | data-* raw-value attributes for CSV |
| TV-5 | fuzzing/generators.py | 1 | No partial-ties mutation (only extreme via zero-var/constant) | Add "rank_ties" mutation (bin to few distinct values) |
| TV-6 | tests/robustness/edge_cases.py | 1 | CATALOG simple-path only; no RM/mixed/logistic/beta degenerate cases | Extend CATALOG with advanced/clinical degenerate designs |

---

## Cross-cutting themes (fix once, resolves several)
1. **p∈[0,1] band invariant missing at 3 layers**: runtime funnel (C1-3), export formatter (C6-2),
   CI test contract (TV-2). Out-of-CI fuzzer oracle is the only check. → add one shared band guard + assertion.
2. **Convergence detectability**: model omits flag (C5-1/C5-2) → export can't show it (C6-1) → tests don't assert it (TV-3). LMM is the good template; replicate.
3. **ddof=0 strays**: C2-2 (RM d), C5-3 (cov SD), C5-4 (bootstrap SE). Reference = posthoc_core.py:1302 (ddof=1, zero→0).
4. **traceback.print_exc to dead stdout**: C1-4, C3-3 (+ pattern in clinical export). → logger.exception.
5. **Dead code**: C2-9 (Dunnett T3 ~130 LOC), C3-4 (legacy mixed posthoc ~300 LOC).

---
---
# APPENDIX — per-chunk notes & ledgers (working detail)

## Shape ledger (CLOSED)
- Shape-2 (`level1/p_val/mean_dif/se_dif`) is INTERNAL scratch only (RMAnovaPostHocAnalyzer :1114),
  converted to canonical Shape-1 via add_comparison (:1217) before output. Hybrid emitter
  `_perform_test_legacy` (:476) is DEAD. Engine forwards only `pairwise_comparisons` (Shape-1).
  ⇒ analysis_core :1364 hard-indexing SAFE. `within_pairwise_comparisons` merged into
  `pairwise_comparisons` by engine (statisticaltester:1958-1959) → export reads one container.

## ddof Cohen's d reference (C2-2/C5-3/C5-4 fix target)
Canonical = PostHocStatistics.calculate_cohens_d (posthoc_core.py:1302): paired
`mean(diff)/np.std(diff,ddof=1)` guard `>0 else 0`; independent pooled `np.var(...,ddof=1)`
guard `s_pooled>0 else 0`. CI helper (:1314) is alpha-aware. Zero-SD sentinel: 0 (posthoc_core)
vs None (statisticaltester _paired_ttest:467) — pick one; RM context → 0, type "cohen_d_rm".

## Convergence-flag propagation (Q-Chunk5)
LMM=PASSED (clinical_models.py:868). Logistic=DROPPED. Beta=DROPPED+unchecked. Export converged
display (report_stat_rows:214, report_summaries:311) is LMM-only.

## Phase C — what CI actually guarantees
Golden-vs-R (car/afex/emmeans/lme4/glm/logistf/nparLD @ tol≈1e-4) = strong value-correctness on
happy paths. Robustness CATALOG (18 cases, edge_cases.py) = simple-path data-quality blocks
(EMPTY_GROUP/VAR_ZERO/N_BELOW_MIN/INF_VALUES/NUM_OVERFLOW/TOO_FEW_GROUPS/VAR_DIFF_ZERO).
Fuzzer (fuzzing/) = exploratory, NOT in CI; findings distilled into CATALOG + regression tests
(test_nonfinite_block). Gaps = TV-1..6, aligned with the audit's HIGH findings.

## Coverage caveats (not fully line-audited → optional sub-chunks)
- 2b: statisticaltester _run_mixed/_run_rm/_run_two_way bodies (1602-2713)
- 5b: ANCOVA fit/adjusted_means/emm (90-600), Firth inner loop (1142-1205), HL/calibration/ROC (1366-1499)
- 6b: html_exporter context assembly (273-447), report_charts plot builders, report_association
- Not audited as chunks: advanced_pipeline.py, validators.py/robustness layer, visualization/

---

## ✅ V21-1 — legacy mixed-interaction post-hoc is dead code (⚠️ pingouin-version-coupled)

**Location:** `statisticaltester.py:1587-1592`, inside `_run_mixed_anova` (def :1470).
Live call chain: `_run_mixed_anova` ← `_run_mixed_anova_logged` (:1294) ← `advanced_pipeline.py:196`.
**Status: RESOLVED** in `1d72e98` — the whole inline block (154 lines) was removed and the
mixed post-hoc now always comes from `AdvancedPostHocEngine`. The recommendation below
("remove, do not repair") was followed.

> ### ⚠️ Correction to this finding's own verdict
> The line "No user-visible defect remains in v2.0" was **wrong**, and the way it was wrong is
> the lesson worth keeping. This analysis examined the `if interaction significant:` branch,
> proved it could not complete on pingouin ≥ 0.6, and concluded the *block* was dead. It never
> looked at the `else:` branch of the same `try`.
>
> That else branch was live, and harmful. It ran whenever the interaction was **not**
> significant — the ordinary case when a real main effect exists — and paired observations by
> dataframe position rather than by subject:
>
> ```python
> data1 = df[df[rm_factor] == group1][dv].values
> data2 = df[df[rm_factor] == group2][dv].values
> t_stat, p_val = stats.ttest_rel(data1, data2)
> ```
>
> Reordering the rows of the same file moved a contrast from p = 0.012 to p = 0.298 and
> flipped comparison directions. The results were merged into `pairwise_comparisons`
> (`:1908-1912`) and labelled "Paired t-tests (Holm-Bonferroni)", so the report looked normal.
>
> It was also *doubly* masked: `advanced_pipeline` gates the post-hoc on `res["p_value"]`,
> which for a mixed design is the **interaction's** p — so in exactly the case where the
> broken branch ran, the checked Feature-B path was switched off and could not supersede it.
>
> **Takeaway:** "this branch is dead" is a statement about one branch, not about the block
> that contains it. Read the whole `try`.

**Fix 1 / Fix 2 below are now moot** — both mitigated symptoms of a block that no longer
exists. The pingouin-version coupling described below no longer applies to this code path;
it is kept as the reasoning record.

### What is wrong
The inline interaction post-hoc reads four keys off `pg.pairwise_tests(...)` that do not
match the library's output:

| access | line | exists on pingouin 0.6.1? |
|---|---|---|
| `r['Time']` | 1587-1588 | only if the within factor is *literally* named `Time` |
| `r['Type']` | 1589 | ❌ no |
| `r["p-corr"]` | 1591 | ❌ no (it is `p_corr`) |
| `r["significant"]` | 1592 | ❌ no |

Actual columns (verified, pingouin 0.6.1, between+within):
`['Contrast','<within>','A','B','Paired','Parametric','T','dof','alternative','p_unc','p_corr','p_adjust','BF10','hedges']`

`r['Time']` is a hardcoded column name in a tool whose factors are user-named — structurally
fragile independent of the version issue. The app's own Excel template uses `Timepoint`
(`assets/BioMedStatX_Excel_Template.xlsx`, sheets RM_1F / Mixed ANOVA / RM_EMM_1F).

### Why it is NOT a v2.0 blocker (proven, not assumed)
The path was driven end-to-end with the within column renamed literally to `Time`: it gets
past `r['Time']` and then dies on `'Type'`. So the branch **cannot complete for any column
name** on pingouin ≥0.6 — it is dead code, not merely hard to reach. It therefore never emits
numbers, and when the omnibus is significant `AdvancedPostHocEngine` (Feature B) replaces
`pairwise_comparisons` wholesale anyway (`advanced_pipeline.py:285-288`). No wrong values can
reach a user.

### ⚠️ This verdict is coupled to pingouin 0.6.1
On **every** pingouin upgrade, re-verify which of these holds:
- **(a)** branch still dead → keep as is;
- **(b)** branch runs further but breaks on a *new* column schema → re-assess;
- **(c)** branch runs and works → the v2.0 mitigations below become wrong and must be revisited.

In cases (b)/(c) both v2.0 fixes must be re-checked, not assumed:
- **Fix 1** (`statisticaltester.py:~1698`): the raw exception no longer goes into
  `results["warnings"]`, only to `logger.warning`. Justified *only* because the branch is
  known-dead and superseded. If it can run again, suppressing its error hides a real failure
  from the user — it lands in a log nobody reads.
- **Fix 2** (right after): top-level `posthoc_test` is consolidated from
  `within_posthoc_test` / `between_posthoc_test` when the main-effects branch produced
  contrasts.

### Recommendation (prioritized): **remove, do not repair**
Feature B (`analysis/mixed_simple_effects.py` + `MixedAnovaPostHocAnalyzer`) has already
superseded this branch functionally and is R-verified (golden: `tests/golden/references_feature_b*.json`),
Levene-gated for the pooled/isolated error term, and covered by routing + gating tests
(`tests/test_feature_b_golden.py`). Repairing the legacy branch means maintaining **two
parallel mixed-interaction post-hoc implementations**, one of them obsolete, untested and
built on hardcoded column names. Deleting it is less v2.1 work than fixing a path that will
likely be dropped anyway.

**If "repair" is chosen instead**, the precedent to copy already exists in the same file:
`statisticaltester.py:2391` does `p_corr → p_unc → p-unc` as a fallback chain. Apply the same
for the p-column and replace `r['Time']` with the actual `rm_factor` variable.

### Fixed in v2.0 (user-visible symptoms only)
- spurious `"Post-hoc failed: 'Time'"` in `results["warnings"]` next to a correct Feature-B
  result → now `logger.warning` only;
- `posthoc_test: None` (contrasts shown with no method name in the report) whenever the
  omnibus was n.s. and the advanced engine did not run → now consolidated.

Related live bug found and fixed during the same investigation: `posthoc_core.py` read
pingouin's uncorrected-p column as `"p-unc"` while ≥0.6 emits `p_unc`; the `KeyError` was
swallowed, so the Feature-B effect gate silently degraded to simple main effects for *every*
design. Now reads both spellings, logs at WARNING, and exposes `gating_applied` /
`gating_fallback_reason` in the result. Regression guard:
`tests/test_feature_b_golden.py::test_gating_through_the_real_analyzer_entry_point`.

---

## 🔁 Recurring pattern — layer boundaries narrow the dict they forward

**Not a bug. A shape to expect when adding a field.**

Six times during the pre-2.0 audit, information that one layer computed correctly was lost
because the next layer rebuilt the dict from a hand-written list of keys instead of forwarding
it. Each instance was found only by reading the *consuming* function end to end; none was
visible from the producing side, because the producer was right.

Confirmed instances:

| Producer writes | Consumer reads / forwards | Effect |
|---|---|---|
| `MixedAnovaPostHocAnalyzer`: `posthoc_mode`, `gating_applied`, `gating_fallback_reason` | `engines/advanced_posthoc.py` returned a 2-key dict | diagnostics dropped |
| same | `advanced_pipeline.py` copied only `posthoc_test` + `pairwise_comparisons` | dropped a second time |
| Mixed: `within_sphericity_test`, `within_correction_used` | `report_summaries._build_assumption_summary` read only the RM spellings | Mixed reports showed **no** sphericity row at Mauchly p = 1.6e-22 |
| `pg.mixed_anova` / `rm_anova` (0.6 column names) | `mixed_assumptions` read the pre-rename hyphenated names | branch never taken |
| `pg.pairwise_tests` (0.6 columns) | `statisticaltester` read `Type` / `p-corr` / `significant` | branch raised, error swallowed |
| `pg.sphericity` namedtuple `(spher, W, chi2, dof, pval)` | unpacked as `W, pval, spher` | violated sphericity reported as **met** |

Two properties make this class hard to see:

1. **The producer is correct**, so reviewing it finds nothing. Only the consumer is wrong.
2. **The failure is silence, not an exception** — a missing row, an absent key, a `None`. Tests
   that assert on what *is* present pass; nothing asserts on what should have been there.

### When adding a field to a result dict
Trace it to its consumer and confirm it arrives. `grep` for the key name across `src/` — if the
only hits are the write sites, the field has no reader. Prefer forwarding the dict over
re-listing keys; where a narrow dict is deliberate, say so in a comment naming what is dropped.

### When a result differs between designs
RM writes `sphericity_test` / `correction_used`; Mixed writes the same information under a
`within_` prefix, because a mixed design has two effect families. Consumers must handle both.
Key on **presence** (`if "sphericity_test" in results`), not truthiness — `a or b` silently
falls through when `a` is an empty-but-present dict and borrows the other design's value.
