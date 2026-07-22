"""BLOCKER (pre-2.0 audit): the Conover-Iman post-hoc after Friedman applied no
multiplicity correction at all.

`perform_friedman_test` called ``sp.posthoc_conover_friedman(wide[level_cols])``
without ``p_adjust``. scikit-posthocs defaults that parameter to ``None``, so the
raw pairwise p-values were emitted, marked ``"corrected": False`` -- and the
``significant`` flag, which is what the HTML report renders, was computed on the
uncorrected values. This is the DEFAULT path: it runs automatically after every
significant Friedman omnibus, with no dialog and no way to decline it.

Every other post-hoc family in the app corrects. The correction used here is
Holm-Bonferroni, following the app's own C3b precedent
(``analysis_core.py`` and ``posthoc_fallback.py``): Holm-Bonferroni controls the
FWER under arbitrary dependence, while Sidak-based step-down assumes an
independence that pairwise comparisons sharing groups do not have. Conover-Friedman
contrasts are maximally dependent -- shared ranks, shared blocks, shared error term.

The dataset below is the one the audit measured the defect on: 14 subjects,
5 timepoints, a 0.30/step drift. Two of its ten comparisons flip verdict at
alpha=0.05 between raw and Holm-adjusted p-values.
"""
import numpy as np
import pandas as pd
import pytest

from analysis.nonparametricanovas import perform_friedman_test

# Comparisons whose significance verdict flips between raw and Holm-adjusted
# p-values on this dataset (raw -> holm):
#   T1-T4 : 0.027168 -> 0.217346
#   T3-T5 : 0.037104 -> 0.259728
FLIPPERS = {("T1", "T4"), ("T3", "T5")}
ALPHA = 0.05


def _audit_frame():
    rng = np.random.default_rng(9)
    rows = []
    for s in range(14):
        base = rng.normal(0, 1)
        for i, t in enumerate(["T1", "T2", "T3", "T4", "T5"]):
            rows.append({"subj": f"s{s}", "time": t, "y": base + 0.30 * i + rng.normal(0, 1.0)})
    return pd.DataFrame(rows)


def _levels(comp):
    """Comparison labels are prefixed with the factor name ('time=T1')."""
    return tuple(sorted(str(comp[k]).split("=")[-1] for k in ("group1", "group2")))


@pytest.fixture(scope="module")
def result():
    res = perform_friedman_test(_audit_frame(), dv="y", within_factor="time",
                                subject_col="subj", alpha=ALPHA)
    assert res.get("p_value") is not None and res["p_value"] < ALPHA, (
        "omnibus must be significant, otherwise the post-hoc never runs")
    return res


def test_conover_posthoc_is_actually_produced(result):
    comps = result.get("pairwise_comparisons") or []
    assert len(comps) == 10, comps
    assert any("Conover" in str(c.get("test", "")) for c in comps), comps


def test_conover_pvalues_are_multiplicity_corrected(result):
    """The core defect: raw p-values were reported and flagged significant."""
    for comp in result["pairwise_comparisons"]:
        assert comp.get("corrected") is True, (
            f"{_levels(comp)} still reported as uncorrected: {comp}")


def test_the_two_measured_flippers_are_no_longer_significant(result):
    """T1-T4 (raw p=0.027) and T3-T5 (raw p=0.037) are significant only without
    a correction. After Holm they must not be flagged."""
    by_pair = {_levels(c): c for c in result["pairwise_comparisons"]}
    for pair in FLIPPERS:
        comp = by_pair[pair]
        assert comp["p_value"] > ALPHA, (
            f"{pair}: p={comp['p_value']} still <= alpha -- correction not applied")
        assert not comp["significant"], f"{pair} still flagged significant: {comp}"


def test_the_genuinely_significant_pairs_survive_the_correction(result):
    """Guard against over-correcting: T1-T5 and T2-T5 stay significant."""
    by_pair = {_levels(c): c for c in result["pairwise_comparisons"]}
    for pair in [("T1", "T5"), ("T2", "T5")]:
        comp = by_pair[pair]
        assert comp["p_value"] < ALPHA, f"{pair} lost significance: {comp}"
        assert comp["significant"], comp


def test_reported_pvalues_equal_holm_over_the_unique_pairs(result):
    """Anchor to statsmodels, independent of what scikit-posthocs does
    internally."""
    import scikit_posthocs as sp
    from statsmodels.stats.multitest import multipletests

    wide = _audit_frame().pivot(index="subj", columns="time", values="y")
    raw = sp.posthoc_conover_friedman(wide)
    cols = list(wide.columns)
    pairs = [(cols[i], cols[j]) for i in range(len(cols)) for j in range(i + 1, len(cols))]
    raw_ps = [float(raw.loc[a, b]) for a, b in pairs]
    expected = dict(zip([tuple(sorted(p)) for p in pairs],
                        multipletests(raw_ps, method="holm")[1]))

    # `_apply_holm` rounds to 6 decimals (pre-existing convention shared with the
    # Wilcoxon fallback path), so compare at that resolution.
    for comp in result["pairwise_comparisons"]:
        assert comp["p_value"] == pytest.approx(expected[_levels(comp)], abs=1e-6), _levels(comp)


def test_scikit_posthocs_padjust_still_means_what_we_assume():
    """Version-drift guard, in the spirit of the pingouin p_unc/p-unc bug: the
    library's own `p_adjust='holm'` must keep adjusting over the k(k-1)/2 unique
    pairs -- not the full symmetric matrix or the diagonal."""
    import scikit_posthocs as sp
    from statsmodels.stats.multitest import multipletests

    wide = _audit_frame().pivot(index="subj", columns="time", values="y")
    raw = sp.posthoc_conover_friedman(wide)
    lib = sp.posthoc_conover_friedman(wide, p_adjust="holm")
    cols = list(wide.columns)
    idx = [(i, j) for i in range(len(cols)) for j in range(i + 1, len(cols))]

    assert list(raw.index) == cols and list(raw.columns) == cols
    assert np.allclose(raw.values, raw.values.T), "expected a symmetric matrix"

    raw_ps = [float(raw.iloc[i, j]) for i, j in idx]
    lib_ps = [float(lib.iloc[i, j]) for i, j in idx]
    own = multipletests(raw_ps, method="holm")[1]
    np.testing.assert_allclose(lib_ps, own, atol=1e-12)
