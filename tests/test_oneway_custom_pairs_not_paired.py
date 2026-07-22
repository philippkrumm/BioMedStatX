"""BLOCKER (pre-2.0 audit): the one-way "custom pairs" post-hoc ran a PAIRED
t-test on INDEPENDENT groups.

`perform_refactored_posthoc_testing`'s `paired_custom` / `paired_fdr` branch
called `scipy.stats.ttest_rel`, which pairs observation *i* of one group with
observation *i* of the other. For a one-way design with independent groups no
such pairing exists, so:

  * the reported t / df / p were those of a paired test on unpaired data;
  * the result depended on the ROW ORDER of the input file -- reordering rows
    within a group silently changed the p-value;
  * unequal group sizes raised inside scipy and degraded to an error string.

The branch is only ever reachable with independent data: the sole caller that
passes ``is_dependent=True`` (``perform_dependent_posthoc_tests``) forces
``posthoc_choice="dependent"``, which routes to ``DependentPostHoc`` instead.
The dialog nevertheless offers this option for one-way ANOVA
(``stats_functions.select_posthoc_test_dialog``), so a normal user reaches it.
"""
import numpy as np
import pytest
from scipy import stats

import statistical_testing.posthoc_fallback as pf


def _stub_dialogs(monkeypatch, pairs, choice="paired_custom"):
    class _UI:
        @staticmethod
        def select_posthoc_test_dialog(**kwargs):
            return choice

        @staticmethod
        def select_custom_pairs_dialog(groups):
            return pairs

        @staticmethod
        def select_control_group_dialog(groups):
            return None

    monkeypatch.setattr(pf, "_get_ui_dialog_manager", lambda: _UI)


def _run(samples, groups, recommendation="parametric"):
    return pf.PosthocFallbackEngine.perform_refactored_posthoc_testing(
        groups, samples, recommendation, alpha=0.05,
        posthoc_choice=None, control_group=None, is_dependent=False,
    )


def _independent_samples():
    rng = np.random.default_rng(0)
    return {"A": list(rng.normal(10, 2, 8)), "B": list(rng.normal(13, 2, 8))}


def test_custom_pairs_on_independent_groups_is_row_order_invariant(monkeypatch):
    """The core defect: an independent comparison must not care which row of A
    happens to sit next to which row of B."""
    _stub_dialogs(monkeypatch, [("A", "B")])
    samples = _independent_samples()

    res_a = _run(samples, ["A", "B"])
    p_a = res_a["pairwise_comparisons"][0]["p_value"]
    t_a = res_a["pairwise_comparisons"][0]["statistic"]

    # Same data, same groups -- only the order of the values inside group B.
    shuffled = {"A": list(samples["A"]), "B": list(reversed(samples["B"]))}
    res_b = _run(shuffled, ["A", "B"])
    p_b = res_b["pairwise_comparisons"][0]["p_value"]
    t_b = res_b["pairwise_comparisons"][0]["statistic"]

    assert p_a == pytest.approx(p_b, abs=1e-12), (
        f"p-value changed when rows were reordered: {p_a} -> {p_b}. "
        "The comparison is being computed as if the groups were paired."
    )
    assert t_a == pytest.approx(t_b, abs=1e-12)


def test_custom_pairs_on_independent_groups_matches_ttest_ind(monkeypatch):
    """Anchor the statistic to scipy's independent t-test."""
    _stub_dialogs(monkeypatch, [("A", "B")])
    samples = _independent_samples()
    res = _run(samples, ["A", "B"])
    comp = res["pairwise_comparisons"][0]

    ref = stats.ttest_ind(samples["A"], samples["B"], equal_var=True)
    assert comp["statistic"] == pytest.approx(ref.statistic, abs=1e-12)
    # single pair -> the step-down correction leaves the raw p unchanged
    assert comp["p_value"] == pytest.approx(ref.pvalue, abs=1e-12)

    wrong = stats.ttest_rel(samples["A"], samples["B"])
    assert comp["statistic"] != pytest.approx(wrong.statistic, abs=1e-9)


def test_custom_pairs_handles_unequal_group_sizes(monkeypatch):
    """ttest_rel raised on unequal n; an independent comparison must not."""
    _stub_dialogs(monkeypatch, [("A", "B")])
    rng = np.random.default_rng(1)
    samples = {"A": list(rng.normal(10, 2, 8)), "B": list(rng.normal(13, 2, 11))}

    res = _run(samples, ["A", "B"])
    assert not res.get("error"), res.get("error")
    assert res["pairwise_comparisons"], res
    ref = stats.ttest_ind(samples["A"], samples["B"], equal_var=True)
    assert res["pairwise_comparisons"][0]["statistic"] == pytest.approx(
        ref.statistic, abs=1e-12)


def test_custom_pairs_labels_do_not_claim_pairing_for_independent_groups(monkeypatch):
    _stub_dialogs(monkeypatch, [("A", "B")])
    res = _run(_independent_samples(), ["A", "B"])
    comp = res["pairwise_comparisons"][0]

    assert "Paired" not in comp["test"], comp["test"]
    assert "paired" not in res["posthoc_test"].lower(), res["posthoc_test"]
    assert "RM" not in (comp.get("effect_size_type") or ""), comp.get("effect_size_type")


def test_custom_pairs_uses_welch_when_the_omnibus_was_welch(monkeypatch):
    """The app routes Student vs Welch on the variance assumption everywhere
    else; the custom-pairs branch must follow the same recommendation."""
    _stub_dialogs(monkeypatch, [("A", "B")])
    rng = np.random.default_rng(2)
    samples = {"A": list(rng.normal(10, 1, 10)), "B": list(rng.normal(13, 6, 10))}

    res = _run(samples, ["A", "B"], recommendation="welch")
    comp = res["pairwise_comparisons"][0]
    ref = stats.ttest_ind(samples["A"], samples["B"], equal_var=False)
    assert comp["statistic"] == pytest.approx(ref.statistic, abs=1e-12)
    assert comp["p_value"] == pytest.approx(ref.pvalue, abs=1e-12)
    assert "Welch" in comp["test"], comp["test"]


def test_the_dialog_text_names_the_test_the_branch_actually_runs(monkeypatch):
    """The option label is what the user reads BEFORE choosing. It outlived the
    ttest_rel -> ttest_ind fix and still promised a paired test."""
    from analysis.stats_functions import ONEWAY_POSTHOC_OPTIONS

    labels = dict((value, label) for label, value in ONEWAY_POSTHOC_OPTIONS)
    assert set(labels) == {"games_howell", "dunnett", "paired_custom"}

    _stub_dialogs(monkeypatch, [("A", "B")])
    comp = _run(_independent_samples(), ["A", "B"])["pairwise_comparisons"][0]

    assert "Paired" not in comp["test"], comp["test"]
    assert "paired" not in labels["paired_custom"].lower(), labels["paired_custom"]
    assert "independent" in labels["paired_custom"].lower(), labels["paired_custom"]


def test_dependent_designs_still_get_a_paired_test(monkeypatch):
    """Guard against over-correcting: when the caller really does pass
    dependent data with an explicit paired_custom choice, keep ttest_rel."""
    _stub_dialogs(monkeypatch, [("A", "B")])
    rng = np.random.default_rng(3)
    base = rng.normal(0, 1, 9)
    samples = {"A": list(base + rng.normal(0, 0.3, 9)),
               "B": list(base + 1.2 + rng.normal(0, 0.3, 9))}

    res = pf.PosthocFallbackEngine.perform_refactored_posthoc_testing(
        ["A", "B"], samples, "parametric", alpha=0.05,
        posthoc_choice="paired_custom", control_group=None, is_dependent=True,
    )
    comp = res["pairwise_comparisons"][0]
    ref = stats.ttest_rel(samples["A"], samples["B"])
    assert comp["statistic"] == pytest.approx(ref.statistic, abs=1e-12)
    assert "Paired" in comp["test"], comp["test"]
