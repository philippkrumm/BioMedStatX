"""ANCOVAModel.emm_contrasts() writes "t"/"se" while LinearMixedModel.emm_contrasts() writes
"statistic"/"std_err" for the exact same kind of comparison dict - the HTML export layer
(report_stat_rows.py) reads "statistic", so ANCOVA's t-column silently renders blank. Fix:
ANCOVA's dict shape must match LMM's.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from analysis.clinical_models import ANCOVAModel


def test_ancova_emm_contrasts_use_the_same_keys_as_lmm():
    rng = np.random.RandomState(0)
    n = 60
    df = pd.DataFrame({
        "Group": np.repeat(["ctrl", "a", "b"], n // 3),
        "Cov": rng.randn(n),
    })
    df["Value"] = (
        df["Cov"] * 1.5
        + df["Group"].map({"ctrl": 0.0, "a": 2.0, "b": 4.0})
        + rng.randn(n) * 0.5
    )

    model = ANCOVAModel()
    model.fit(df, dv="Value", between_factors=["Group"], covariates=["Cov"])
    contrasts = model.emm_contrasts(method="pairwise")

    assert contrasts, "expected at least one pairwise contrast"
    for c in contrasts:
        assert "statistic" in c, f"missing 'statistic' key (found: {sorted(c.keys())})"
        assert "std_err" in c, f"missing 'std_err' key (found: {sorted(c.keys())})"
        assert "t" not in c, "stale 't' key should have been renamed to 'statistic'"
        assert "se" not in c, "stale 'se' key should have been renamed to 'std_err'"
        assert c.get("test") == "ANCOVA EMM Contrast"
        assert c.get("corrected") is True
        assert c.get("correction") == "Holm-Bonferroni"


def test_ancova_emm_contrasts_vs_control_correction_label():
    rng = np.random.RandomState(1)
    n = 60
    df = pd.DataFrame({
        "Group": np.repeat(["ctrl", "a", "b"], n // 3),
        "Cov": rng.randn(n),
    })
    df["Value"] = (
        df["Cov"] * 1.5
        + df["Group"].map({"ctrl": 0.0, "a": 2.0, "b": 4.0})
        + rng.randn(n) * 0.5
    )

    model = ANCOVAModel()
    model.fit(df, dv="Value", between_factors=["Group"], covariates=["Cov"])
    contrasts = model.emm_contrasts(method="vs_control", control_group="ctrl")

    assert contrasts
    assert all(c.get("correction") == "multivariate-t" for c in contrasts)
