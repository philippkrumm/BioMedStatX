"""The run-level calibration check: a rate, and only where a rate is meaningful.

The generator draws the effect each design is built with and, for the whole life
of this fuzzer, threw it away -- so the one dimension it could have graded itself
against was varied and never looked at. The check holds each built term against
the p-value for that same term and reports two rates over a run: how often a null
term is called significant (should be about alpha) and how often a real one is
found (power).

Two mistakes are pinned here because both were made while building it. Holding a
built MAIN effect against the headline p-value measures the interaction for a
mixed design -- which those data are built without -- and reported 7% "power"
that was really the interaction's type-I error. And a rate from a handful of
terms is not a rate; below a floor the honest answer is that the run is too small
to say.
"""
from __future__ import annotations

import pytest

from fuzzing.run_fuzzer import _calibration, _calibration_verdict


def _record(test, truth, term_p_values, *, mutations=("none",), category="OK",
            datasets=1, blocked=False):
    return {"category": category, "test": test, "mutations": list(mutations),
            "datasets": datasets, "blocked": blocked,
            "truth": truth, "term_p_values": term_p_values}


def test_a_term_is_judged_against_its_own_p_value():
    """The built effect and the reported p must name the same term."""
    records = [
        _record("two_way_anova", {"FacA": 0.0, "FacB": 2.0, "FacA:FacB": 0.0},
                {"FacA": 0.9, "FacB": 0.001, "FacA:FacB": 0.4}),
    ]
    result = _calibration(records)
    assert result["null_terms"] == 2 and result["null_rejected"] == 0
    assert result["effect_terms"] == 1 and result["effect_found"] == 1
    assert result["power"] == 1.0


def test_a_headline_cannot_stand_in_for_a_term():
    """The mistake that produced a false 7% power reading.

    A mixed design is built with main effects and no interaction. If the check
    reads the headline -- which for that design IS the interaction -- it reports
    a null term as a missed effect.
    """
    records = [_record("mixed_anova", {"Time": 3.0, "Between": 3.0},
                       {"Time": 0.0001, "Between": 0.0002, "Time:Between": 0.8})]
    result = _calibration(records)
    assert result["effect_terms"] == 2 and result["effect_found"] == 2
    # The interaction was reported but never built, so it is nobody's evidence.
    assert result["null_terms"] == 0


def test_a_built_term_that_was_never_reported_is_surfaced():
    """Dropping it silently is how a calibration measures three terms quietly."""
    records = [_record("two_way_anova", {"FacA": 0.0, "FacB": 0.0}, {"FacA": 0.7})]
    result = _calibration(records)
    assert result["null_terms"] == 1
    assert result["unmatched_terms"] == ["two_way_anova:FacB"]


@pytest.mark.parametrize("kwargs", [
    {"mutations": ("nan_scatter",)},   # an assumption deliberately broken
    {"datasets": 3},                   # the headline is per dataset, not per run
    {"blocked": True},                 # no p-value to judge
    {"category": "ORACLE_VIOLATION"},  # the run already has a finding
])
def test_seeds_that_cannot_speak_to_calibration_are_excluded(kwargs):
    records = [_record("two_way_anova", {"FacA": 0.0}, {"FacA": 0.001}, **kwargs)]
    result = _calibration(records)
    assert result["null_terms"] == 0 and result["effect_terms"] == 0


def test_designs_without_a_drawn_effect_contribute_nothing():
    """LMM is always built without one and correlation always with one.

    Counting either would measure the generator rather than the app.
    """
    records = [_record("lmm", {}, {"Between": 0.4})]
    assert _calibration(records)["null_terms"] == 0


def test_the_verdict_refuses_to_read_a_rate_from_a_handful():
    records = [_record("two_way_anova", {"FacA": 0.0}, {"FacA": 0.9})] * 5
    verdict = _calibration_verdict(_calibration(records))
    assert "too few" in verdict


def test_a_calibrated_run_is_reported_as_such():
    """One in twenty null terms rejected is what alpha promises."""
    records = ([_record("two_way_anova", {"FacA": 0.0}, {"FacA": 0.9})] * 95
               + [_record("two_way_anova", {"FacA": 0.0}, {"FacA": 0.01})] * 5)
    result = _calibration(records)
    assert result["null_rate"] == pytest.approx(0.05)
    assert "consistent with alpha" in _calibration_verdict(result)


def test_a_test_that_rejects_the_null_far_too_often_is_named():
    records = ([_record("two_way_anova", {"FacA": 0.0}, {"FacA": 0.01})] * 40
               + [_record("two_way_anova", {"FacA": 0.0}, {"FacA": 0.9})] * 60)
    verdict = _calibration_verdict(_calibration(records))
    assert "REJECTS THE NULL TOO OFTEN" in verdict


def test_a_test_that_never_rejects_anything_is_named_too():
    """A p-value that is never small is as broken as one that always is."""
    records = [_record("two_way_anova", {"FacA": 0.0}, {"FacA": 0.9})] * 300
    verdict = _calibration_verdict(_calibration(records))
    assert "implausibly rarely" in verdict


def test_the_breakdown_keeps_the_designs_apart():
    records = [
        _record("two_way_anova", {"FacA": 0.0}, {"FacA": 0.01}),
        _record("mixed_anova", {"Time": 2.0}, {"Time": 0.01}),
    ]
    per_design = _calibration(records)["per_design"]
    assert per_design["two_way_anova"]["null"] == [1, 1]
    assert per_design["mixed_anova"]["effect"] == [1, 1]
