"""The two nonparametric post-hoc descriptions used to name each other's
correction: "dunn" advertised Holm-Bonferroni while DunnTest applies statsmodels'
'holm-sidak', and "mw_custom" advertised Sidak while the Mann-Whitney branch
applies 'holm' (Holm-Bonferroni).

These pin each label against the correction the code actually runs, so the two
cannot drift apart again.
"""
import numpy as np
import pytest

from analysis.stats_functions import NONPARAMETRIC_POSTHOC_OPTIONS

LABELS = dict((value, label) for label, value in NONPARAMETRIC_POSTHOC_OPTIONS)


def _samples():
    rng = np.random.default_rng(3)
    return {"A": list(rng.normal(10, 2, 10)),
            "B": list(rng.normal(13, 2, 10)),
            "C": list(rng.normal(16, 2, 10))}


def test_both_options_are_offered():
    assert set(LABELS) == {"dunn", "mw_custom"}


def test_dunn_label_names_the_correction_dunn_actually_applies():
    from analysis.posthoc_core import DunnTest

    samples = _samples()
    res = DunnTest.perform_test(list(samples), samples, alpha=0.05)
    applied = res["pairwise_comparisons"][0]["correction"]

    assert applied == "Holm-Šidák", applied
    assert "Holm-Šidák" in LABELS["dunn"], LABELS["dunn"]
    assert "Bonferroni" not in LABELS["dunn"], LABELS["dunn"]


def test_mann_whitney_label_names_the_correction_it_actually_applies(monkeypatch):
    import statistical_testing.posthoc_fallback as pf

    class _UI:
        @staticmethod
        def select_nonparametric_posthoc_dialog(**kwargs):
            return "mw_custom"

        @staticmethod
        def select_custom_pairs_dialog(groups):
            return [("A", "B")]

        @staticmethod
        def select_control_group_dialog(groups):
            return None

    monkeypatch.setattr(pf, "_get_ui_dialog_manager", lambda: _UI)
    samples = _samples()
    res = pf.PosthocFallbackEngine.perform_refactored_posthoc_testing(
        list(samples), samples, "non_parametric", alpha=0.05,
        posthoc_choice=None, control_group=None, is_dependent=False,
    )
    applied = res["pairwise_comparisons"][0]["correction"]

    assert applied == "Holm-Bonferroni", applied
    assert "Holm-Bonferroni" in LABELS["mw_custom"], LABELS["mw_custom"]
    assert "Šidák" not in LABELS["mw_custom"], LABELS["mw_custom"]


def test_the_two_labels_do_not_name_the_same_correction():
    """The original defect was symmetric -- each named the other's method."""
    assert LABELS["dunn"] != LABELS["mw_custom"]
    assert ("Šidák" in LABELS["dunn"]) != ("Šidák" in LABELS["mw_custom"])
