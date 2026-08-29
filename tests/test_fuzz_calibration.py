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


def test_the_design_filter_reads_the_same_draw_the_generator_makes():
    """The filter skips seeds before spawning them, so it must not guess.

    A second copy of the selection rule would keep filtering on a rule the
    generator had stopped following, and the run would quietly measure a
    different sample than the one it names. Both go through the same draw.
    """
    from fuzzing.generators import build_case, design_for_seed
    for seed in range(25):
        assert build_case(seed).test_label == design_for_seed(seed)


def test_count_still_means_seeds_run_when_a_filter_is_on():
    """Otherwise every denominator in the summary silently means something else."""
    from fuzzing.generators import design_for_seed
    from fuzzing.run_fuzzer import _seeds_to_run

    wanted = {"rm_anova", "mixed_anova", "two_way_anova"}
    seeds = list(_seeds_to_run(0, 12, wanted))
    assert len(seeds) == 12
    assert all(design_for_seed(s) in wanted for s in seeds)
    # It walked past the seeds it skipped rather than renumbering them.
    assert seeds == sorted(seeds) and seeds[-1] > 12


def test_no_filter_leaves_the_seed_range_exactly_as_it_was():
    from fuzzing.run_fuzzer import _seeds_to_run
    assert list(_seeds_to_run(7, 4, set())) == [7, 8, 9, 10]


def _summary(**over):
    base = {"start": 0, "count": 100, "last_seed": 99, "designs_filter": [],
            "elapsed_sec": 12.0, "findings": [], "categories": {"OK": 100},
            "coverage": {"oracles_fired": {"a": 3, "b": 0}},
            "calibration": _calibration([])}
    base.update(over)
    return base


def test_a_run_is_recorded_so_the_rate_can_be_read_across_runs(tmp_path):
    """The only question the fuzzer is finally judged by is a trend."""
    from fuzzing.run_fuzzer import _record_run

    path = tmp_path / "history.jsonl"
    _record_run(_summary(findings=[{"seed": 1}, {"seed": 2}]), ["a", "b"], path=str(path))
    _record_run(_summary(findings=[]), ["a", "b"], path=str(path))

    import json
    lines = [json.loads(l) for l in path.read_text().splitlines()]
    assert [e["findings"] for e in lines] == [2, 0]
    assert lines[0]["count"] == 100


def test_the_record_keeps_what_makes_a_falling_rate_readable(tmp_path):
    """Fewer findings from fewer questions is not progress.

    The oracle count and the design filter are kept beside the rate precisely so
    a run that asked less cannot be read as a product that improved.
    """
    from fuzzing.run_fuzzer import _record_run

    path = tmp_path / "history.jsonl"
    entry = _record_run(_summary(designs_filter=["rm_anova"]), ["a", "b", "c"],
                        path=str(path))
    assert entry["oracles"] == 3
    assert entry["oracles_that_fired"] == 1
    assert entry["designs"] == ["rm_anova"]


def test_a_partial_report_says_it_is_partial(tmp_path):
    """A long run must not be all-or-nothing.

    A 2000-seed run killed near the end left nothing behind, having spent forty
    minutes. The report is now written as the run goes, and a reader can tell a
    partial one from a finished one by the flag rather than by guessing from the
    seed count.
    """
    import json
    from fuzzing.run_fuzzer import _write_report

    path = tmp_path / "r.json"
    _write_report({"complete": False, "seeds_run": 100}, str(path))
    assert json.loads(path.read_text())["complete"] is False
    _write_report({"complete": True, "seeds_run": 250}, str(path))
    written = json.loads(path.read_text())
    assert written["complete"] is True and written["seeds_run"] == 250
    # Nothing half-written is left where a reader would look.
    assert not (tmp_path / "r.json.partial").exists()

def test_a_repeated_measures_design_claims_only_the_term_it_is_analysed_on():
    """The between column is built for it and never analysed.

    It enters as one constant offset per subject and is absorbed into the
    subject effect, so claiming it as a built term would have the run looking
    for a p-value nothing reports.
    """
    from fuzzing.generators import build_case, design_for_seed

    seen = 0
    for seed in range(400):
        if design_for_seed(seed) != "rm_anova":
            continue
        seen += 1
        assert set(build_case(seed).truth) == {"Time"}
        if seen == 3:
            break
    assert seen == 3


def test_a_main_effect_is_not_judged_when_the_design_carries_an_interaction():
    """The truth is a COEFFICIENT; the ANOVA tests MARGINAL means.

    An interaction moves the marginal means even where the main coefficient is
    zero, so such a term is not a null term and rejecting it is not an error.
    Measured over 2500 seeds before this rule existed: two-way main effects were
    called significant 37% of the time where an interaction was present against
    6.8% on a purely null design, and the 8.5% that looked like an engine
    problem was this. The engine was right.
    """
    records = [_record("two_way_anova",
                       {"FacA": 0.0, "FacB": 0.0, "FacA:FacB": 2.0},
                       {"FacA": 0.001, "FacB": 0.002, "FacA:FacB": 0.0001})]
    result = _calibration(records)

    assert result["null_terms"] == 0, "a main effect under an interaction is not null"
    assert result["not_marginal_terms"] == 2
    # The interaction itself is still judged, and it was built with an effect.
    assert result["effect_terms"] == 1 and result["effect_found"] == 1


def test_the_same_main_effects_are_judged_without_an_interaction():
    """The rule must not quietly excuse every main effect."""
    records = [_record("two_way_anova",
                       {"FacA": 0.0, "FacB": 0.0, "FacA:FacB": 0.0},
                       {"FacA": 0.001, "FacB": 0.9, "FacA:FacB": 0.7})]
    result = _calibration(records)

    assert result["null_terms"] == 3 and result["null_rejected"] == 1
    assert result["not_marginal_terms"] == 0


def test_a_zero_interaction_does_not_excuse_the_main_effects():
    """An interaction TERM is present in every two-way truth; only a non-zero
    one changes what a main effect means."""
    records = [_record("two_way_anova",
                       {"FacA": 0.0, "FacA:FacB": 0.0},
                       {"FacA": 0.02, "FacA:FacB": 0.5})]
    result = _calibration(records)
    assert result["null_terms"] == 2 and result["null_rejected"] == 1
