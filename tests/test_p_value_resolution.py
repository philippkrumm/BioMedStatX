"""Reported p-values must not claim precision the method cannot deliver.

Two engines estimate their p-values rather than deriving them. The
multivariate-t CDF behind the EMM/Dunnett post-hoc is integrated by Monte
Carlo, and the Freedman-Lane test permutes. Both produce numbers that look
exact and are not: the mvt figure loses its leading digit deep in the tail, and
a permutation p can never fall below 1/(n_perm+1), so a value sitting exactly
there means "nothing beat the observed statistic", not a measured magnitude.

Each such engine attaches ``p_value_resolution`` and the report shows a bound
below it. Analytic p-values carry no resolution and must be untouched.
"""

import pandas as pd
import pytest

from analysis.emm_posthoc import MVT_P_RESOLUTION, rm_dunnett_emm_mvt
from analysis.nonparametricanovas import perform_freedman_lane_test
from analysis.posthoc_core import PostHocAnalyzer
from export.report_formatting import _FormattingMixin
from export.report_stat_rows import _StatRowsMixin

fmt = _FormattingMixin._format_p_value


def test_analytic_p_values_are_unchanged():
    """No resolution passed means the old behaviour, exactly."""
    assert fmt(0.5) == "p = 0.500 ns"
    assert fmt(0.03) == "p = 0.030 *"
    assert "3.40" in fmt(3.4e-9)


def test_a_p_below_the_resolution_is_shown_as_a_bound():
    shown = fmt(7.96e-7, MVT_P_RESOLUTION)
    assert shown.startswith("p < ")
    assert "7.96" not in shown, "the unresolvable figure must not be printed"
    assert shown.endswith("***"), "the verdict against alpha is unaffected"


def test_a_p_exactly_at_the_resolution_is_a_bound_too():
    """The permutation floor is reached exactly, not approached.

    The add-one estimator lands on 1/(n_perm+1) whenever nothing beat the
    observed statistic, so an exclusive comparison would let precisely the case
    this exists for slip through and print the floor as a measurement. Asserting
    "starts with p <" is not enough -- the fallback "p < 0.001" also does.
    """
    floor = 1.0 / 5001
    shown = fmt(floor, floor)
    assert shown == "p < " + _FormattingMixin._sci_notation(floor) + " ***"
    assert "0.001" not in shown


def test_a_p_above_the_resolution_keeps_its_figure():
    assert fmt(2.12e-4, MVT_P_RESOLUTION) == fmt(2.12e-4)
    assert fmt(0.005, MVT_P_RESOLUTION) == "p = 0.005 **"


@pytest.mark.parametrize("bad", [None, 0.0, 1.0, -1e-6, float("nan"), "1e-6"])
def test_an_unusable_resolution_is_ignored_rather_than_trusted(bad):
    assert fmt(3.4e-9, bad) == fmt(3.4e-9)


def test_mvt_contrasts_declare_their_resolution():
    rows = []
    for i in range(10):
        base = 10 + 0.3 * i
        rows += [{"Subject": f"S{i}", "Time": "T0", "Y": base},
                 {"Subject": f"S{i}", "Time": "T1", "Y": base + 12.0}]
    contrasts = rm_dunnett_emm_mvt(pd.DataFrame(rows), dv="Y", subject="Subject",
                                   within="Time", control_level="T0")
    assert contrasts
    assert all(c["p_value_resolution"] == MVT_P_RESOLUTION for c in contrasts)


def test_the_resolution_survives_into_the_report_row():
    """The pairwise table is where a reader actually sees the number."""
    result = PostHocAnalyzer.create_result_template("EMM")
    PostHocAnalyzer.add_comparison(
        result, group1="Control", group2="Dose", test="EMM + multivariate-t",
        p_value=1e-9, statistic=12.0, significant=True,
        p_value_resolution=MVT_P_RESOLUTION,
    )
    PostHocAnalyzer.add_comparison(
        result, group1="Control", group2="Vehicle", test="Tukey HSD",
        p_value=1e-9, statistic=11.0, significant=True,
    )
    bounded, analytic = _StatRowsMixin._build_pairwise_rows(result)
    assert bounded["p_value"].startswith("p < ") and "1.00" in bounded["p_value"]
    # The analytic neighbour in the same table keeps its figure.
    assert "1.00 × 10⁻⁹" in analytic["p_value"]


def test_permutation_resolution_is_the_grid_not_a_constant():
    """1/(n_perm+1), so it tracks whatever permutation count was used."""
    rows = []
    for a in ("A1", "A2"):
        for b in ("B1", "B2"):
            shift = 6.0 if a == "A2" else 0.0
            for k in range(8):
                rows.append({"FA": a, "FB": b, "Y": float(shift + k * 0.5)})
    result = perform_freedman_lane_test(pd.DataFrame(rows), dv="Y",
                                        factor_a="FA", factor_b="FB",
                                        n_permutations=99)
    assert result["p_value_resolution"] == pytest.approx(1.0 / 100)
    assert result["p_value"] >= result["p_value_resolution"], (
        "the add-one estimator cannot go below its own grid"
    )
