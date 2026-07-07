# Mixed ANOVA Sphericity Correction Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Mixed ANOVA's Greenhouse-Geisser sphericity correction actually gate the
significance verdict and post-hoc dispatch, instead of being computed and silently discarded.

**Architecture:** Replace the existing, broken sphericity-lookup chain in
`src/statistical_testing/mixed_assumptions.py` (which calls a redundant, always-failing
`pg.sphericity()` and then checks column names — `'GG-eps'`/`'p-GG'` — that never existed in any
pingouin version) with a single function that reads the real `eps`/`p_GG_corr`/`sphericity`/
`W_spher`/`p_spher` columns pingouin's `mixed_anova()` already computes for the within-factor row
(confirmed against pingouin 0.6.1's own source and empirically). Apply that same epsilon to the
Interaction row too (it shares the identical error term/denominator df, confirmed from pingouin's
source — this is the standard SPSS/afex/JASP convention, not a novel computation). Wire both
corrected p-values into the canonical fields in `src/analysis/statisticaltester.py::_run_mixed_anova`,
mirroring RM-ANOVA's already-correct `# E1` pattern one function away.

**Tech Stack:** Python, pingouin 0.6.1, scipy.stats, pytest.

**Spec:** `docs/superpowers/specs/2026-07-07-mixed-anova-sphericity-fix-design.md`

---

## Reference data for the new test (derived during planning, do not regenerate)

Synthetic dataset (seed=65, 24 subjects × 3 within-levels × between-group) produces a genuine
significance-boundary crossing after correction:

| | F | df1 | df2 | uncorrected p (`p_unc`) | GG-corrected p |
|---|---|---|---|---|---|
| within factor "time" | 0.294514 | 2 | 44 | 0.746350 | 0.650295 (pingouin's own `p_GG_corr`) |
| Interaction | 3.379686 | 2 | 44 | **0.043112** (significant) | **0.071045** (computed: `scipy.stats.f.sf(3.379686, 2*eps, 44*eps)`) |

`eps = 0.594881` (pingouin's own value for the within-factor row). The interaction crosses from
significant to non-significant once corrected — this is the exact bug this fix closes.

---

### Task 1: Write the failing test proving the bug (RED)

**Files:**
- Create: `tests/test_mixed_anova_sphericity_correction.py`

- [ ] **Step 1: Write the failing test**

```python
"""Mixed ANOVA's Greenhouse-Geisser sphericity correction must gate the
canonical p_value (and therefore the significance verdict and post-hoc
dispatch), not just sit in a side-channel dict nobody reads. RM-ANOVA's
sibling path already does this (statisticaltester.py:1973-1974, tagged
"E1"); this test targets the Mixed-ANOVA path, which does not.

Seed=65 synthetic dataset (heteroscedastic within-factor variances, a small
interaction signal) was chosen during planning specifically because it
produces a genuine significance-boundary crossing: the Interaction term's
uncorrected p (0.043) is significant at alpha=0.05, but the Greenhouse-Geisser
corrected p (0.071, using the within-factor's own eps=0.595 - pingouin never
computes a separate epsilon for the interaction term, but it shares the same
error term/denominator df as the within-factor, per pingouin's own
mixed_anova() source, so applying the same epsilon is the standard
SPSS/afex/JASP convention) is not.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from analysis.statisticaltester import StatisticalTester


def _seed65_mixed_design_df():
    rng = np.random.RandomState(65)
    n_subj = 24
    subjects = np.repeat(np.arange(n_subj), 3)
    time = np.tile(["T1", "T2", "T3"], n_subj)
    group = np.repeat(rng.choice(["A", "B"], n_subj), 3)
    base = rng.randn(n_subj)
    t1 = base + rng.randn(n_subj) * 1
    t2 = base + rng.randn(n_subj) * 4
    t3 = base + rng.randn(n_subj) * 10
    interaction_bump = (group[0::3] == "A") * rng.normal(0, 0.6, n_subj)
    t3 = t3 + interaction_bump
    dv = np.empty(n_subj * 3)
    dv[0::3] = t1
    dv[1::3] = t2
    dv[2::3] = t3
    return pd.DataFrame({"subject": subjects, "time": time, "group": group, "dv": dv})


def test_mixed_anova_sphericity_correction_flips_the_verdict():
    df = _seed65_mixed_design_df()

    results = StatisticalTester._run_mixed_anova(
        df=df, dv="dv", subject="subject", between=["group"], within=["time"], alpha=0.05
    )

    assert results.get("error") is None

    within_entry = next(f for f in results["factors"] if f["factor"] == "time")
    interaction_entry = results["interactions"][0]

    # Raw, uncorrected values must still be inspectable under "p_unc".
    assert within_entry["p_unc"] == pytest.approx(0.746350, abs=1e-4)
    assert interaction_entry["p_unc"] == pytest.approx(0.043112, abs=1e-4)

    # "p_value" must now be the Greenhouse-Geisser corrected value, not p_unc.
    assert within_entry["p_value"] == pytest.approx(0.650295, abs=1e-4)
    assert interaction_entry["p_value"] == pytest.approx(0.071045, abs=1e-3)

    # The actual bug: the interaction is "significant" uncorrected but NOT
    # significant once correctly adjusted for sphericity violation.
    assert interaction_entry["p_unc"] < 0.05
    assert interaction_entry["p_value"] > 0.05

    # The top-level canonical field (what analysis_core.py:1087 gates the
    # verdict and post-hoc dispatch on) must reflect the corrected value.
    assert results["p_value"] == pytest.approx(interaction_entry["p_value"], abs=1e-9)
    assert results["p_value"] > 0.05

    # F itself is unchanged by a sphericity correction - only df/p move.
    assert results["statistic"] == pytest.approx(3.379686, abs=1e-4)

    assert "Greenhouse-Geisser" in results["within_correction_used"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/test_mixed_anova_sphericity_correction.py -v`

Expected: FAIL — `within_entry["p_unc"]`/`interaction_entry["p_unc"]` raise `KeyError` (neither
key exists yet on current code), or `assert within_entry["p_value"] == pytest.approx(0.650295...)`
fails because `p_value` is still the uncorrected `0.746350`/`0.043112`.

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test_mixed_anova_sphericity_correction.py
git commit -m "test(mixed-anova): add failing test for sphericity-correction verdict bug"
```

---

### Task 2: Replace the broken sphericity-lookup chain in mixed_assumptions.py

**Files:**
- Modify: `src/statistical_testing/mixed_assumptions.py:386-733` (four functions to delete, one to
  write)

The four functions being deleted (`_test_mixed_anova_within_sphericity`,
`_extract_mixed_sphericity_from_anova_table`, `_apply_mixed_anova_sphericity_corrections`,
`_apply_corrections_to_effect_row`) and `_generate_within_factor_recommendations` have **no
external callers** other than the staticmethod bindings in `statisticaltester.py` (verified via
`git grep` during planning — every hit outside `mixed_assumptions.py` was one of those binding
lines, removed in Task 3). Safe to delete the function bodies wholesale.

- [ ] **Step 1: Read the current line range to delete**

Run: `sed -n '386,733p' src/statistical_testing/mixed_assumptions.py | tail -5`
Expected: last line printed is the closing of `_generate_within_factor_recommendations` (a
`return recommendations` followed by blank line before the next method). Confirm the exact end
line number with `grep -n "_test_mixed_anova_between_assumptions" src/statistical_testing/mixed_assumptions.py`
— the next method after the block being replaced.

- [ ] **Step 2: Delete lines 386-733 (all five old functions) and replace with the new one**

Use the Edit tool with this exact replacement (old text starts at the
`@staticmethod` immediately before `def _test_mixed_anova_within_sphericity` and ends at the
blank line immediately before the next method in the file — read the file first to get the exact
`old_string` boundaries, since line numbers may have drifted slightly from this plan). The
replacement adds one small helper (`_interpret_sphericity_test`, needed because
`mixed_assumptions.py` cannot import `StatisticalTester` — that would create a circular import,
since `statisticaltester.py` imports `MixedAnovaAssumptionEngine` from this file — so this class
gets its own copy rather than reusing `StatisticalTester`'s RM-ANOVA one) followed by the new
`_test_mixed_anova_within_sphericity`:

```python
    @staticmethod
    def _interpret_sphericity_test(p_value, sphericity_met):
        if p_value is None:
            return "Sphericity not tested"
        if sphericity_met:
            return f"Sphericity assumption met (Mauchly's W, p = {p_value:.3f} >= 0.05)"
        return f"Sphericity assumption violated (Mauchly's W, p = {p_value:.3f} < 0.05) - Greenhouse-Geisser correction applied"

    @staticmethod
    def _test_mixed_anova_within_sphericity(df, dv, subject, within_factor, aov, alpha=0.05):
        """
        Reads the within-factor's Greenhouse-Geisser sphericity correction
        directly from the aov table pingouin's mixed_anova() already computed
        (mirrors StatisticalTester._apply_sphericity_corrections for RM-ANOVA,
        which reads the same columns from pg.rm_anova(..., correction=True)).

        No separate pg.sphericity() call: mixed_anova() calls rm_anova()
        internally for the within-factor row, so eps/p_GG_corr/sphericity/
        W_spher/p_spher are already present on that row whenever within has
        >=3 levels. pingouin never computes a separate epsilon for the
        Interaction row (confirmed from pingouin 0.6.1's own source) - the
        caller (StatisticalTester._run_mixed_anova) applies this same within
        row's epsilon to the Interaction row too, since both share the same
        error term/denominator df.

        Parameters
        ----------
        df : DataFrame
            Data containing mixed design variables (kept for the k<=2 level
            count; not used for a second sphericity computation).
        dv : str
            Dependent variable column name (unused directly; kept for
            signature parity with the previous implementation's callers).
        subject : str
            Subject identifier column name (unused directly; see dv).
        within_factor : str
            Within-subjects factor column name.
        aov : DataFrame
            ANOVA results table from pingouin's mixed_anova() call.
        alpha : float
            Significance level (unused here; kept for signature parity).

        Returns
        -------
        dict
            within_sphericity_test, within_corrected_p_value,
            within_correction_used, and (only when sphericity is violated)
            within_sphericity_corrections["main_effect"] with the corrected
            df1/df2/p_value the caller needs to also correct the Interaction
            row.
        """
        sphericity_results = {}

        within_levels = df[within_factor].unique()
        k = len(within_levels)

        if k <= 2:
            sphericity_results["within_sphericity_test"] = {
                "test_name": "Mauchly's Test for Sphericity (Within-Factor)",
                "factor": within_factor,
                "W": None,
                "chi_square": None,
                "df": None,
                "p_value": None,
                "sphericity_assumed": True,
                "note": "Sphericity assumption is always met with 2 levels",
                "interpretation": "No correction needed - only 2 conditions compared"
            }
            sphericity_results["within_corrected_p_value"] = None
            sphericity_results["within_correction_used"] = "None (sphericity assumption met)"
            sphericity_results["within_sphericity_corrections"] = {
                "needed": False,
                "reason": "Sphericity assumption is always met with 2 levels"
            }
            return sphericity_results

        within_mask = aov["Source"] == within_factor
        if not within_mask.any():
            sphericity_results["within_sphericity_test"] = {
                "test_name": "Mauchly's Test for Sphericity (Within-Factor)",
                "factor": within_factor,
                "W": None,
                "p_value": None,
                "sphericity_assumed": None,
                "note": f"Within-factor '{within_factor}' not found in ANOVA table",
                "interpretation": "Indeterminate - within-factor row missing"
            }
            sphericity_results["within_corrected_p_value"] = None
            sphericity_results["within_correction_used"] = "None (within-factor not found in ANOVA table)"
            sphericity_results["within_sphericity_corrections"] = {
                "needed": False,
                "reason": f"Within-factor '{within_factor}' not found in ANOVA table"
            }
            return sphericity_results

        within_row = aov.loc[within_mask].iloc[0]
        has_sphericity_data = "sphericity" in aov.columns and pd.notna(within_row.get("sphericity"))

        if not has_sphericity_data:
            sphericity_results["within_sphericity_test"] = {
                "test_name": "Mauchly's Test for Sphericity (Within-Factor)",
                "factor": within_factor,
                "W": None,
                "p_value": None,
                "sphericity_assumed": None,
                "note": "pingouin did not compute a sphericity test for this design",
                "interpretation": "Sphericity not tested"
            }
            sphericity_results["within_corrected_p_value"] = None
            sphericity_results["within_correction_used"] = "None (sphericity not tested)"
            sphericity_results["within_sphericity_corrections"] = {
                "needed": False,
                "reason": "pingouin did not compute a sphericity test for this design"
            }
            return sphericity_results

        spher = bool(within_row["sphericity"])
        W = float(within_row["W_spher"]) if pd.notna(within_row.get("W_spher")) else None
        p_spher = float(within_row["p_spher"]) if pd.notna(within_row.get("p_spher")) else None

        sphericity_results["within_sphericity_test"] = {
            "test_name": "Mauchly's Test for Sphericity (Within-Factor)",
            "factor": within_factor,
            "W": W,
            "p_value": p_spher,
            "sphericity_assumed": spher,
            "df": int((k * (k - 1)) / 2 - 1),
            "interpretation": MixedAnovaAssumptionEngine._interpret_sphericity_test(p_spher, spher),
            "levels_tested": k,
            "comparisons": int(k * (k - 1) / 2)
        }

        if spher:
            sphericity_results["within_corrected_p_value"] = None
            sphericity_results["within_correction_used"] = "None (sphericity assumption met)"
            sphericity_results["within_sphericity_corrections"] = {
                "needed": False,
                "reason": "Sphericity assumption is met for within-factor"
            }
            return sphericity_results

        # Sphericity violated: pingouin always provides eps/p_GG_corr
        # together with sphericity=False on this row.
        epsilon = float(within_row["eps"])
        gg_p_value = float(within_row["p_GG_corr"])
        corrected_df1 = float(within_row["DF1"]) * epsilon
        corrected_df2 = float(within_row["DF2"]) * epsilon

        main_effect = {
            "effect": f"within-factor ({within_factor})",
            "greenhouse_geisser": {
                "epsilon": epsilon,
                "corrected_df1": corrected_df1,
                "corrected_df2": corrected_df2,
                "p_value": gg_p_value,
                "conservative": True,
                "description": f"Greenhouse-Geisser correction for within-factor ({within_factor})"
            },
            "recommended_correction": "greenhouse_geisser",
            "final_p_value": gg_p_value,
            "correction_used": f"Greenhouse-Geisser (ε = {epsilon:.3f})"
        }

        sphericity_results["within_sphericity_corrections"] = {
            "needed": True,
            "main_effect": main_effect
        }
        sphericity_results["within_corrected_p_value"] = gg_p_value
        sphericity_results["within_correction_used"] = f"Greenhouse-Geisser (ε = {epsilon:.3f})"

        return sphericity_results
```

- [ ] **Step 3: Run a quick syntax/import check**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -c "from statistical_testing.mixed_assumptions import MixedAnovaAssumptionEngine; print(MixedAnovaAssumptionEngine._test_mixed_anova_within_sphericity)"`
Expected: prints the function object with no `ImportError`/`SyntaxError`.

- [ ] **Step 4: Commit**

```bash
git add src/statistical_testing/mixed_assumptions.py
git commit -m "fix(mixed-anova): read GG sphericity correction from real pingouin columns"
```

---

### Task 3: Remove the now-dead staticmethod bindings in statisticaltester.py

**Files:**
- Modify: `src/analysis/statisticaltester.py:2905-2908`

- [ ] **Step 1: Delete the three dead bindings**

Find this block (currently at lines 2905-2908, confirm exact location with
`grep -n "_extract_mixed_sphericity_from_anova_table\|_apply_mixed_anova_sphericity_corrections\|_apply_corrections_to_effect_row\|_generate_within_factor_recommendations" src/analysis/statisticaltester.py`):

```python
    _extract_mixed_sphericity_from_anova_table = staticmethod(MixedAnovaAssumptionEngine._extract_mixed_sphericity_from_anova_table)
    _apply_mixed_anova_sphericity_corrections = staticmethod(MixedAnovaAssumptionEngine._apply_mixed_anova_sphericity_corrections)
    _apply_corrections_to_effect_row = staticmethod(MixedAnovaAssumptionEngine._apply_corrections_to_effect_row)
    _generate_within_factor_recommendations = staticmethod(MixedAnovaAssumptionEngine._generate_within_factor_recommendations)
```

Delete these 4 lines entirely. Leave the line above them
(`_test_mixed_anova_within_sphericity = staticmethod(MixedAnovaAssumptionEngine._test_mixed_anova_within_sphericity)`)
untouched — it now binds the new, simplified function from Task 2.

- [ ] **Step 2: Run a quick import check**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -c "from analysis.statisticaltester import StatisticalTester; print(StatisticalTester._test_mixed_anova_within_sphericity)"`
Expected: prints the function object, no `AttributeError`.

- [ ] **Step 3: Commit**

```bash
git add src/analysis/statisticaltester.py
git commit -m "chore(mixed-anova): remove staticmethod bindings for deleted dead functions"
```

---

### Task 4: Wire the corrected p-values into the canonical fields (the actual bug fix)

**Files:**
- Modify: `src/analysis/statisticaltester.py` (inside `_run_mixed_anova`, right after the existing
  sphericity call — confirm exact line with
  `grep -n "within_sphericity_results = StatisticalTester._test_mixed_anova_within_sphericity" src/analysis/statisticaltester.py`,
  currently around line 1701-1706)

- [ ] **Step 1: Add the write-back immediately after `results.update(within_sphericity_results)`**

Find:

```python
                # Enhanced Within-Factor Sphericity Testing for Mixed ANOVA
                rm_factor = within[0]
                within_sphericity_results = StatisticalTester._test_mixed_anova_within_sphericity(
                    df, dv, subject, rm_factor, aov, alpha
                )
                results.update(within_sphericity_results)
```

Replace with:

```python
                # Enhanced Within-Factor Sphericity Testing for Mixed ANOVA
                rm_factor = within[0]
                within_sphericity_results = StatisticalTester._test_mixed_anova_within_sphericity(
                    df, dv, subject, rm_factor, aov, alpha
                )
                results.update(within_sphericity_results)

                # E1 (Mixed ANOVA): if sphericity was violated, wire the
                # Greenhouse-Geisser corrected p-value into the canonical
                # fields the verdict/post-hoc dispatch actually reads -
                # mirrors RM-ANOVA's existing fix one function away
                # (statisticaltester.py, tagged "E1"). The Interaction row
                # gets the SAME epsilon as the within-factor row: pingouin
                # never computes a separate one for it, but both terms share
                # the same error term/denominator df (confirmed from
                # pingouin's mixed_anova() source), so this is the standard
                # SPSS/afex/JASP convention, not an approximation.
                main_effect_corr = within_sphericity_results.get(
                    "within_sphericity_corrections", {}
                ).get("main_effect")
                if main_effect_corr is not None:
                    epsilon = main_effect_corr["greenhouse_geisser"]["epsilon"]

                    for f in results["factors"]:
                        if f["factor"] == rm_factor:
                            f["p_unc"] = f["p_value"]
                            f["p_value"] = main_effect_corr["final_p_value"]
                            f["df1"] = main_effect_corr["greenhouse_geisser"]["corrected_df1"]
                            f["df2"] = main_effect_corr["greenhouse_geisser"]["corrected_df2"]

                    if results["interactions"]:
                        inter = results["interactions"][0]
                        inter_df1_corr = inter["df1"] * epsilon
                        inter_df2_corr = inter["df2"] * epsilon
                        inter_p_corr = float(stats.f.sf(inter["F"], inter_df1_corr, inter_df2_corr))

                        inter["p_unc"] = inter["p_value"]
                        inter["p_value"] = inter_p_corr
                        inter["df1"] = inter_df1_corr
                        inter["df2"] = inter_df2_corr

                        interaction_key = f"{rm_factor} * {between_factor}"
                        results["within_sphericity_corrections"]["interactions"] = {
                            interaction_key: {
                                "effect": f"interaction ({interaction_key})",
                                "greenhouse_geisser": {
                                    "epsilon": epsilon,
                                    "corrected_df1": inter_df1_corr,
                                    "corrected_df2": inter_df2_corr,
                                    "p_value": inter_p_corr,
                                    "conservative": True,
                                    "description": f"Greenhouse-Geisser correction for interaction ({interaction_key})"
                                },
                                "recommended_correction": "greenhouse_geisser",
                                "final_p_value": inter_p_corr,
                                "correction_used": f"Greenhouse-Geisser (ε = {epsilon:.3f})"
                            }
                        }

                        # Top-level canonical fields: the Interaction row
                        # currently always drives these (see the "Set
                        # top-level fields" block earlier in this function).
                        # F itself does not change under a sphericity
                        # correction - only df/p do.
                        results["p_value"] = inter_p_corr
                        results["df1"] = inter_df1_corr
                        results["df2"] = inter_df2_corr
```

- [ ] **Step 2: Run the Task 1 test to verify it now passes**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/test_mixed_anova_sphericity_correction.py -v`
Expected: PASS.

- [ ] **Step 3: Run the existing golden test to confirm no regression**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/test_golden_r_advanced.py::test_golden_afex_mixed_anova -v`
Expected: PASS — this test already checks `if "p_unc" in app_eff: p_val = app_eff["p_unc"]` before
comparing to R's uncorrected afex p-value, so adding the `p_unc` key makes it pick up the raw
value automatically, with no test-code change.

- [ ] **Step 4: Run the full test suite**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/ -x -q`
Expected: all tests pass (same pass count as before this change, plus the one new test from
Task 1). If anything besides `test_mixed_anova_sphericity_correction.py` newly fails, stop and
investigate before proceeding — do not paper over an unexpected failure.

- [ ] **Step 5: Commit**

```bash
git add src/analysis/statisticaltester.py
git commit -m "fix(mixed-anova): wire GG-corrected p-value into the canonical verdict field"
```

---

### Task 5: Negative control — prove the test actually catches the bug

**Files:** none modified permanently; this task temporarily reverts Task 4's change to confirm
the new test fails with the *predicted* symptom, then restores the fix.

- [ ] **Step 1: Temporarily revert the Task 4 write-back**

```bash
git revert --no-commit HEAD~1
```

(This reverts the "wire GG-corrected p-value" commit specifically — confirm with
`git log --oneline -5` that `HEAD~1` is that commit before running; adjust the ref if other
commits landed in between.)

- [ ] **Step 2: Run the new test and confirm it fails with the exact predicted symptom**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/test_mixed_anova_sphericity_correction.py -v`
Expected: FAIL on `assert within_entry["p_value"] == pytest.approx(0.650295, abs=1e-4)` (or the
`KeyError` on `"p_unc"`, depending on how much of Task 2/3 is also reverted) — confirms the test
is actually exercising the bug, not vacuously passing regardless of the fix.

- [ ] **Step 3: Restore the fix**

```bash
git revert --abort
```

(Discards the temporary revert; the working tree returns to the fixed state committed in Task 4.)
Run `git status` to confirm the working tree is clean and matches the last commit before
proceeding.

---

### Task 6: Correct the Help Hub doc claim (GD1)

**Files:**
- Modify: `src/core/help_content.py:282`

- [ ] **Step 1: Read the current text**

Run: `sed -n '275,290p' src/core/help_content.py`

- [ ] **Step 2: Update the sentence**

Find the sentence claiming the app "applies a conservative correction rather than assuming
[sphericity] holds" for Mixed ANOVA. Replace it with wording that matches what's now actually
implemented — Greenhouse-Geisser only (pingouin's `mixed_anova()` never returns Huynh-Feldt
columns, confirmed during planning), applied to both the within-factor effect and the
interaction (since this fix now does both). Use the Edit tool with the exact surrounding text
read in Step 1 as `old_string` — do not guess at the exact current wording from this plan, read
it fresh, since `help_content.py` may have been touched by an unrelated Tier B doc fix (GD9) by
the time this task runs.

- [ ] **Step 3: Commit**

```bash
git add src/core/help_content.py
git commit -m "docs(help): correct mixed_anova recipe's sphericity-correction claim"
```

---

## Self-review notes (from writing this plan)

- **Spec coverage:** all 5 numbered items in the spec's "Fix" section are covered — simplification
  (Task 2), interaction-row correction (Task 4), canonical-field write-back (Task 4), doc
  correction (Task 6), golden-test compatibility + new unit test (Tasks 1, 4 Step 3).
- **Consumer compatibility check done during planning, not assumed:** `src/export/report_stat_rows.py:428,473`
  reads `results["within_sphericity_corrections"]["main_effect"]`/`["interactions"]` to render a
  "(GG)" suffix in the HTML report — the new function and the Task 4 write-back preserve this
  exact nested shape (including the `interaction_key` substring-matching convention
  `report_stat_rows.py` expects: `f"{rm_factor} * {between_factor}"`), so that display path keeps
  working and now agrees with the corrected canonical `p_value` instead of independently
  re-deriving it. `src/visualization/decisiontreevisualizer.py:183,283` reads
  `within_sphericity_test`/`within_correction_used` — both keys preserved with the same meaning.
- **Also discovered, explicitly out of scope:** the *old* code's primary sphericity path called
  `MixedAnovaAssumptionEngine._interpret_sphericity_test(...)`, a method that was never defined on
  that class (only `StatisticalTester` had one, for RM-ANOVA) — this would have raised
  `AttributeError` every time `pg.sphericity()` itself succeeded, silently caught by the
  surrounding broad `except Exception`. This plan's new function avoids the issue entirely (adds
  its own `_interpret_sphericity_test` to `MixedAnovaAssumptionEngine` in Task 2) rather than
  fixing the old dead path, since the old path is deleted wholesale.
