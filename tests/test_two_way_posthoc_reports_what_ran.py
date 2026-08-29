"""The two-way post-hoc block must report the procedure it actually ran.

Found by the fuzzer's post-hoc-name oracle on seed 50278, which read the
finished page and the result and found them disagreeing: the heading said
"Tukey HSD Test (Pingouin)" while every comparison recorded "Pairwise t-test".
Digging into it turned up a second, heavier defect underneath.

``pg.pairwise_tests(padjust='holm')`` returns ``p_corr`` as a COLUMN, so it is
present on every row -- and pingouin leaves it NaN for a family that has only
one comparison, where there is nothing to correct. The code asked
``'p_corr' in ph_row``, which is always True, and took the NaN. On a 2x2 design
that is BOTH main effects: a main effect at p = 0.000175 was reported as
``p = NaN`` and "not significant", indistinguishable from a genuine null.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from analysis.statisticaltester import StatisticalTester


def _two_by_two(seed=7):
    """A 2x2 layout with a large FacA effect and a real interaction.

    Both are needed: the post-hoc block is only entered when the interaction is
    significant, and the FacA main effect is the row whose p-value was lost.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for a in ("A0", "A1"):
        for b in ("B0", "B1"):
            mean = (0.0 if a == "A0" else 4.0) + (1.5 if (a == "A1" and b == "B1") else 0.0)
            for _ in range(8):
                rows.append({"FacA": a, "FacB": b, "Val": float(rng.normal(mean, 1.0))})
    return pd.DataFrame(rows)


def _comparisons(results):
    return results.get("pairwise_comparisons") or []


def test_a_family_with_nothing_to_correct_keeps_its_uncorrected_p():
    """The row that was reported as NaN, and so as not significant."""
    results = StatisticalTester._run_two_way_anova(_two_by_two(), "Val", ["FacA", "FacB"])

    comparisons = _comparisons(results)
    assert comparisons, "no post-hoc ran, so this test proves nothing"

    main_effect = [c for c in comparisons
                   if {c["group1"], c["group2"]} == {"A0", "A1"}]
    assert len(main_effect) == 1
    row = main_effect[0]
    assert row["p_value"] == row["p_value"], "p is NaN -- the corrected column was empty"
    assert row["p_value"] < 0.01
    assert row["significant"] is True or row["significant"] == True  # noqa: E712


def test_every_reported_p_value_is_a_number():
    results = StatisticalTester._run_two_way_anova(_two_by_two(), "Val", ["FacA", "FacB"])
    for row in _comparisons(results):
        assert row["p_value"] == row["p_value"], f"NaN p for {row['group1']} vs {row['group2']}"


def test_a_row_says_whether_it_was_corrected():
    """Both answers occur in one table, so neither may be assumed.

    The interaction family has more than one comparison and is Holm-adjusted;
    a two-level main effect has one comparison and nothing to adjust. Claiming
    a correction on the second would be as false as omitting it on the first.
    """
    results = StatisticalTester._run_two_way_anova(_two_by_two(), "Val", ["FacA", "FacB"])
    corrected = {row.get("corrected") for row in _comparisons(results)}
    assert "Holm-Bonferroni" in corrected
    assert None in corrected


def test_the_heading_names_the_procedure_that_ran():
    """Tukey uses the studentized range and would give different p-values.

    The comparisons have always recorded "Pairwise t-test"; only the heading
    claimed otherwise, and a reader quoting the method from it -- which is what
    a methods section is written from -- would have misreported it.
    """
    results = StatisticalTester._run_two_way_anova(_two_by_two(), "Val", ["FacA", "FacB"])
    name = results.get("posthoc_test") or ""
    assert _comparisons(results), "no post-hoc ran, so the name proves nothing"
    assert "Tukey" not in name
    assert "t-test" in name.lower() and "holm" in name.lower()
