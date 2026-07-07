# Design: Mixed ANOVA sphericity correction fix (A1)

Date: 2026-07-07. Branch: `feature/advanced-stats-automation`. Item A1 from
`docs/superpowers/specs/2026-07-07-audit-fix-clustering-design.md`. Root findings: SC2
(`docs/superpowers/audit-notes/release-2.0-audit/02-statistical-core-dispatch.md`), AT1/AT2/AT3
(`04-advanced-testing-engines.md`), GD1 (`07-gui-docs-parity.md`).

## Problem

Mixed ANOVA's within-factor sphericity correction is computed but never gates the significance
verdict or post-hoc dispatch. `src/analysis/statisticaltester.py:1548-1549` sets the canonical
`results["p_value"]` (and every `results["factors"]`/`results["interactions"]` entry) from
pingouin's raw, uncorrected interaction p-value. `_test_mixed_anova_within_sphericity`
(`src/statistical_testing/mixed_assumptions.py:414-499`) computes a Greenhouse-Geisser-corrected
p-value but it only reaches a nested nowhere-read key
(`within_sphericity_corrections.main_effect.final_p_value`). RM-ANOVA's sibling path
(`statisticaltester.py:1973-1974`, tagged `# E1`) has the correct write-back; it was never
propagated to Mixed ANOVA.

## What the audit got right, and what this design corrects

The round-2 audit (AT2) additionally claimed pingouin 0.6.1's `mixed_anova()` never returns the
`GG-eps`/`p-GG`/`HF-eps`/`p-HF` columns the correction code looks for, concluding the correction
would need to be computed from scratch. **Verified directly against pingouin 0.6.1's own
source** (`site-packages/pingouin/parametric.py::mixed_anova`), this is half right:

- `mixed_anova()` internally calls `rm_anova(..., correction=correction, detailed=True)` for the
  within-factor row, so that row **does** get real `eps`/`p_GG_corr`/`sphericity`/`W_spher`/
  `p_spher` values whenever `within` has ≥3 levels and pingouin's own Mauchly's test runs — the
  code's lookup just uses the wrong column names (`'GG-eps'`/`'p-GG'` instead of the real
  `'eps'`/`'p_GG_corr'`), so it always falls through to "corrections not available" even though
  the data is right there. No new computation needed for the within-factor row — just fix the
  column names.
- The "Interaction" row **never** gets its own correction columns from pingouin — this part of
  AT2/AT3 is confirmed: `mixed_anova()` builds the interaction row as a fresh dict with only
  `Source, SS, DF1, MS, F, p_unc, <effsize>`, then explicitly sets `eps = NaN` for it. AT3's fix
  suggestion (match the literal `"Interaction"` Source label instead of a `"*"` filter) would
  make the lookup *run*, but it would still find nothing, because pingouin genuinely has nothing
  there.
- Critically, pingouin's own arithmetic shows the within-effect and the interaction **share the
  same denominator** (`ms_reswith`/`df_reswith` — both `f_with` and `f_inter` divide by it). This
  is exactly the classical justification (Maxwell & Delaney 2004; the same convention SPSS/afex/
  JASP use) for applying the **same** Greenhouse-Geisser epsilon to correct both the within main
  effect and the interaction, not just the within main effect alone.

## Fix

**1. Simplify Mixed ANOVA's sphericity plumbing to mirror RM-ANOVA's already-correct pattern.**
Replace `_test_mixed_anova_within_sphericity` / `_apply_corrections_to_effect_row` /
`_extract_mixed_sphericity_from_anova_table` (`mixed_assumptions.py`) with a single function that
reads `eps`, `p_GG_corr`, `sphericity`, `W_spher`, `p_spher` directly off the within-factor's row
in the `aov` table already computed by `_run_mixed_anova` — no second, redundant `pg.sphericity()`
call, no guessing at column names. This mirrors `_apply_sphericity_corrections`
(`statisticaltester.py:2794`), which already does exactly this for RM-ANOVA and is proven correct.
The `k <= 2` short-circuit ("sphericity always met") is unaffected — still returns early before
any of this runs.

**2. Apply the same epsilon to the interaction row.** Once the within-row's `eps` is known: for
the interaction row, compute `df1_corr = df1 * eps`, `df2_corr = df2 * eps` (`df2` is the shared
`df_reswith`), and `p_corr = scipy.stats.f.sf(F, df1_corr, df2_corr)`. This does not require a new
statistical method — it's the identical GG-scaling formula pingouin already applies to the
within-row, applied to the interaction row's own F/df1/df2 with the within-row's epsilon.

**3. Wire both corrected values into the canonical fields, mirroring RM-ANOVA's `E1` pattern.**
In `_run_mixed_anova` (`statisticaltester.py`), after building `results["factors"]` and
`results["interactions"]`:
- Add a `"p_unc"` key (the raw, uncorrected p) alongside the existing `"p_value"` key on both the
  within-factor entry in `results["factors"]` and the entry in `results["interactions"]`.
- Overwrite `"p_value"` on both entries with the corrected value when sphericity was violated
  (leave as the uncorrected value, unchanged, when sphericity holds — including the `k<=2` case).
- Overwrite the top-level `results["p_value"]`/`results["statistic"]`/`results["df1"]`/
  `results["df2"]` (currently always copied from the interaction row) with the interaction's
  corrected values, since this is the field `analysis_core.py:1087` gates the significance verdict
  and post-hoc dispatch on.

**4. Fix the doc claim.** `src/core/help_content.py:282` (GD1) currently states the app "applies a
conservative correction rather than assuming [sphericity] holds" for Mixed ANOVA. Once the code
fix lands this becomes true; update the wording only if it still overstates what's implemented
(e.g. don't claim Huynh-Feldt is available for Mixed ANOVA — pingouin's `mixed_anova()` never
returns HF columns at all, confirmed from the same source read, unlike RM-ANOVA).

## Why this doesn't touch `_apply_sphericity_corrections`'s own dead Huynh-Feldt branch

While reading `_apply_sphericity_corrections` (the RM-ANOVA function this design mirrors), its
Huynh-Feldt branch (`if 'p_HF_corr' in row and 'eps_HF' in row`) is *also* dead — confirmed
`pg.rm_anova(..., correction=True)` on pingouin 0.6.1 never returns `eps_HF`/`p_HF_corr` either.
This isn't a live bug: the function always prefers Greenhouse-Geisser when both are present
anyway ("Use Greenhouse-Geisser unconditionally," per its own comment), and GG's columns *do*
exist, so RM-ANOVA's actual output is correct today. Out of scope for this fix — flagging so it
isn't mistaken for something this design was supposed to have caught and missed. Worth a follow-up
cleanup ticket, not a correctness fix.

## Testing

- **Existing golden test unaffected.** `tests/test_golden_r_advanced.py::test_golden_afex_mixed_anova`
  already checks `if "p_unc" in app_eff: p_val = app_eff["p_unc"]` before falling back to
  `"p_value"` — this fix adds exactly that key, so the existing test keeps comparing against R
  afex's uncorrected p-value with no test-code changes needed.
- **New unit test proving the fix changes the verdict.** The frozen golden dataset's `eps` for
  `time` is `0.9546` (near 1 — sphericity barely violated), too weak to prove the correction
  actually flips a decision. Add a new test (style: `tests/test_sphericity_outer_exception.py`)
  using a synthetic dataset with a strongly heteroscedastic within-factor covariance structure
  (e.g. `Var(T1)=1, Var(T2)=9, Var(T3)=64` on a shared per-subject baseline — reproduced and
  confirmed to give `eps≈0.67` during this brainstorm) where the corrected p-value crosses
  `alpha=0.05` while the uncorrected one doesn't (or vice versa), asserting `results["p_value"]`
  reflects the corrected value and differs from `results["p_unc"]` by more than a trivial amount.
- **Negative control**, per this project's established practice: revert the write-back step,
  confirm the new test fails with the exact "uncorrected p used" symptom, restore the fix.

## Out of scope

- RM-ANOVA's dead Huynh-Feldt branch (see above) — separate, non-blocking cleanup.
- Changing which effect (interaction vs. main effect) drives the top-level `results["p_value"]`
  when the interaction isn't the significant one — pre-existing design choice, unrelated to
  sphericity, not part of this fix.
- Any UI/export-layer change beyond what's needed to display the now-correct `p_value` — the
  existing render paths already read `p_value` from these dicts; SC3/RE1/etc. from the audit are
  separate Tier B work packages.
