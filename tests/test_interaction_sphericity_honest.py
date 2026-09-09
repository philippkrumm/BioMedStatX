"""`_test_interaction_sphericity` could not do what its name promised, and said
so with a message that described a different problem than the real one.

Six defects sat in one ~80-line function, all name/shape mismatches against
pingouin 0.6.1:

  1. the row was looked up as "<within> * <between>"; pingouin labels it
     "Interaction", so the function returned
     "error: Interaction effect not found in ANOVA table" -- a false diagnosis;
  2. sphericity columns were read as 'W-spher' / 'p-spher' (pre-rename,
     hyphenated); 0.6.1 emits 'W_spher' / 'p_spher';
  3. corrections were gated on 'GG-eps' / 'HF-eps'; 0.6.1 emits a single 'eps';
  4. `W, pval, spher = pg.sphericity(...)[:3]` mis-ordered the namedtuple, whose
     fields are (spher, W, chi2, dof, pval). `sphericity_assumed` became
     bool(chi2) -- i.e. essentially always True. Measured on a set with
     Mauchly p = 1.6e-14 it reported sphericity as MET;
  5. that fallback built an interaction_factor "<time>_<group>" and ran Mauchly
     on it. In a mixed design every subject sits in exactly one between-group,
     so those cells are structurally empty and the covariance matrix is
     singular: LinAlgError, by construction rather than by data;
  6. `_apply_corrections_to_effect_row` is called but defined nowhere --
     AttributeError, currently shielded by defect 3.

Defects 4-6 were only unreachable because defect 1 returned early. Fixing the
lookup alone would have armed them, which is why the whole function was reduced
to what is actually computable rather than patched name by name.

pingouin computes no separate epsilon for the Interaction row (stated in this
module's own docstring and confirmed against 0.6.1). The verdict therefore stays
conservative -- sphericity_assumed=False -- exactly as before; the real
correction is applied in StatisticalTester._run_mixed_anova using the
within-factor epsilon. This path is reporting only: its sole consumer is
_generate_interaction_recommendations, which produces display strings.
"""
import numpy as np
import pandas as pd
import pytest
import pingouin as pg

from statistical_testing.mixed_assumptions import MixedAnovaAssumptionEngine as Engine


def _mixed_frame():
    """Violated within-sphericity: the T3/T4 difference variances dwarf T1/T2."""
    rng = np.random.default_rng(7)
    rows = []
    for g in ("ctrl", "trt"):
        for s in range(15):
            b = rng.normal(0, 1)
            off = 0.5 if g == "trt" else 0.0
            vals = {"T1": b + off + rng.normal(0, 0.3),
                    "T2": b + off + 0.6 + rng.normal(0, 0.3),
                    "T3": b + off + 1.0 + rng.normal(0, 3.5),
                    "T4": b + off + 1.4 + rng.normal(0, 3.5)}
            for t, y in vals.items():
                rows.append({"subj": f"{g}{s}", "grp": g, "time": t, "y": y})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def result():
    df = _mixed_frame()
    aov = pg.mixed_anova(data=df, dv="y", within="time", subject="subj",
                         between="grp", correction=True)
    return Engine._test_interaction_sphericity(df, "y", "grp", "time", "subj", aov, 0.05)


def test_no_false_not_found_diagnosis(result):
    """The row IS in the table -- under the name pingouin gives it."""
    err = str(result.get("error") or "")
    assert "not found" not in err.lower(), result
    assert result.get("interaction") == "Interaction", result


def test_verdict_stays_conservative(result):
    assert result["sphericity_assumed"] is False, result


def test_the_note_points_at_where_the_correction_actually_happens(result):
    note = " ".join(str(v) for v in result.values())
    assert "_run_mixed_anova" in note, result
    assert "within" in note.lower(), result


def test_never_claims_sphericity_met_without_a_real_p_value(result):
    """Regression guard for the mis-ordered namedtuple: `sphericity_assumed`
    must never be True unless an actual Mauchly p-value backs it."""
    if result.get("sphericity_assumed") is True:
        assert isinstance(result.get("p_value"), float), result


def test_pingouin_still_gives_the_interaction_row_no_sphericity(result):
    """Version-drift guard. If a future pingouin starts filling W_spher on the
    Interaction row, this fails and the branch reading it should be revisited."""
    df = _mixed_frame()
    aov = pg.mixed_anova(data=df, dv="y", within="time", subject="subj",
                         between="grp", correction=True)
    inter = aov[aov["Source"] == "Interaction"].iloc[0]
    assert pd.isnull(inter["W_spher"]), (
        "pingouin now reports interaction sphericity -- revisit "
        "_test_interaction_sphericity, which currently assumes it never does")


def test_sphericity_namedtuple_order_is_what_we_assume():
    """Pins the field order the removed fallback got wrong."""
    rng = np.random.default_rng(7)
    rows = []
    for s in range(20):
        b = rng.normal(0, 1)
        vals = {"T1": b + rng.normal(0, 0.3), "T2": b + 0.6 + rng.normal(0, 0.3),
                "T3": b + 1.0 + rng.normal(0, 3.5), "T4": b + 1.4 + rng.normal(0, 3.5)}
        for t, y in vals.items():
            rows.append({"subj": f"s{s}", "time": t, "y": y})
    res = pg.sphericity(pd.DataFrame(rows), dv="y", subject="subj", within="time")
    assert res._fields == ("spher", "W", "chi2", "dof", "pval"), res._fields
    assert isinstance(res.spher, (bool, np.bool_))
