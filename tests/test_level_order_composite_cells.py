"""A two-factor cell carries one level per factor, and each is ranked on its own.

``natural_order`` strips the ``factor=`` prefixes and used to rejoin what was
left into a single string before looking it up. A cell is not a level, so the
lookup matched nothing: ``"M, WT"`` is in no reference-term table, and the pair
fell back to the alphabet. ``Sex=F, Geno=KO`` therefore sorted ahead of
``Sex=M, Geno=WT``, while the same two levels on their own put WT first.

Numbers hid it. ``_natural_key`` finds digits anywhere in the string, so a cell
whose within levels were T0/T1 came out in the right order and looked like proof
the composite case worked -- it was the numbers doing it, not the ranking. Only
word levels (WT/KO, Pre/Post, Control/Treated) exposed the difference, and only
where the alphabet disagreed with the meaning.
"""
from __future__ import annotations

import random

import pytest

# Root conftest.py puts src/ on sys.path.
from core.level_order import natural_order, order_is_defined


@pytest.mark.parametrize("given,expected", [
    # The case that was wrong: KO before WT in every cell.
    (["Sex=M, Geno=WT", "Sex=M, Geno=KO", "Sex=F, Geno=WT", "Sex=F, Geno=KO"],
     ["Sex=F, Geno=WT", "Sex=F, Geno=KO", "Sex=M, Geno=WT", "Sex=M, Geno=KO"]),
    # Pre/Post, where the alphabet and the meaning also disagree.
    (["Arm=B, Time=Post", "Arm=A, Time=Pre", "Arm=A, Time=Post", "Arm=B, Time=Pre"],
     ["Arm=A, Time=Pre", "Arm=A, Time=Post", "Arm=B, Time=Pre", "Arm=B, Time=Post"]),
    # A reference term on the FIRST factor puts its whole block first.
    (["Geno=KO, Time=6h", "Geno=WT, Time=24h", "Geno=WT, Time=6h", "Geno=KO, Time=24h"],
     ["Geno=WT, Time=6h", "Geno=WT, Time=24h", "Geno=KO, Time=6h", "Geno=KO, Time=24h"]),
    # Numeric within levels: right before the fix, and still right.
    (["Arm=B, Time=T2", "Arm=B, Time=T0", "Arm=A, Time=T2", "Arm=A, Time=T0"],
     ["Arm=A, Time=T0", "Arm=A, Time=T2", "Arm=B, Time=T0", "Arm=B, Time=T2"]),
    # Control/Treated: alphabetical luck agreed with the meaning here, so this
    # one has to keep coming out the same.
    (["Dose=High, Arm=Treated", "Dose=High, Arm=Control",
      "Dose=Low, Arm=Treated", "Dose=Low, Arm=Control"],
     ["Dose=High, Arm=Control", "Dose=High, Arm=Treated",
      "Dose=Low, Arm=Control", "Dose=Low, Arm=Treated"]),
])
def test_each_factors_levels_are_ranked_on_their_own(given, expected):
    assert natural_order(given) == expected


def test_the_primary_factor_still_decides_the_grouping():
    """Ranking per factor must not gather every control cell to the front.

    The chart is read as "all of A's bars, then all of B's". Sorting by the rank
    of both factors before either name would put ``A=x, B=Control`` next to
    ``A=y, B=Control`` and split each block in two.
    """
    cells = ["Site=Bonn, Arm=Treated", "Site=Aachen, Arm=Control",
             "Site=Bonn, Arm=Control", "Site=Aachen, Arm=Treated"]
    ordered = natural_order(cells)
    assert ordered == ["Site=Aachen, Arm=Control", "Site=Aachen, Arm=Treated",
                       "Site=Bonn, Arm=Control", "Site=Bonn, Arm=Treated"]
    first_factor = [label.split(",")[0] for label in ordered]
    assert first_factor == sorted(first_factor), "the first factor's blocks were split"


def test_single_factor_labels_are_unchanged():
    """The whole existing vocabulary still behaves the way it did."""
    assert natural_order(["KO", "WT"]) == ["WT", "KO"]
    assert natural_order(["Timepoint=Post", "Timepoint=Pre"]) == [
        "Timepoint=Pre", "Timepoint=Post"]
    assert natural_order(["24h", "48h", "6h"]) == ["6h", "24h", "48h"]
    assert natural_order(["48h", "Baseline", "6h"]) == ["Baseline", "6h", "48h"]


def test_order_is_still_independent_of_input_order():
    cells = ["Sex=M, Geno=WT", "Sex=M, Geno=KO", "Sex=F, Geno=WT", "Sex=F, Geno=KO"]
    canonical = natural_order(cells)
    for seed in range(6):
        shuffled = cells[:]
        random.Random(seed).shuffle(shuffled)
        assert natural_order(shuffled) == canonical


def test_a_cell_ordered_by_the_alphabet_is_declared_as_such():
    """Recognising the levels is what makes the order defensible, not the join."""
    defined, _ = order_is_defined(["Sex=M, Geno=WT", "Sex=F, Geno=WT"])
    assert not defined, "M before F is a guess and has to be declared"

    defined, reason = order_is_defined(["Arm=A, Time=Pre", "Arm=A, Time=Post"])
    assert defined, reason


def test_a_guess_in_one_factor_is_declared_even_when_the_other_is_ranked():
    """The regression that ranking per factor introduced, and its guard.

    The ambiguity test used to group labels into runs of equal rank and compare
    only inside a run. One rank per label made that work; one rank per FACTOR
    fragmented the runs -- four cells over two factors can land in four runs of
    one -- and a run of one is never compared with anything. So the alphabetical
    Aachen/Bonn decision below stopped being declared the moment the ranking
    improved. The question is now asked of each adjacent pair, at the factor
    that separates them.
    """
    cells = ["Site=Aachen, Time=T0", "Site=Aachen, Time=T1",
             "Site=Bonn, Time=T0", "Site=Bonn, Time=T1"]
    defined, reason = order_is_defined(cells)
    assert not defined, "Aachen before Bonn is a guess and has to be declared"
    assert "Site=Aachen, Time=T1" in reason and "Site=Bonn, Time=T0" in reason, reason

    # And the same shape with a ranked first factor stays silent.
    ranked = ["Geno=WT, Time=T0", "Geno=WT, Time=T1",
              "Geno=KO, Time=T0", "Geno=KO, Time=T1"]
    defined, reason = order_is_defined(ranked)
    assert defined, reason
