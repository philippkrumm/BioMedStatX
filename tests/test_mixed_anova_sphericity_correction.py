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
