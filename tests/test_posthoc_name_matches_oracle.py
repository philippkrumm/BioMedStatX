"""The procedure the page names must be the one the comparisons came from.

Two facts already lived in every result and were never compared: the headline
``posthoc_test``, printed in the main results table, and the ``test`` each
comparison carries, written by whichever engine produced it. A fallback that
forgets to rename the headline leaves the report naming a method that did not
run -- which is exactly what the mixed EMM/multivariate-t branch did when it
refused its control group and reverted to isolated t-tests with only a log line
to say so.

The headline is read off the page and the comparisons out of the result, so the
check crosses the seam. It must NOT be read from the rendered pairwise table:
that column renders ``comp.get("test") or posthoc_test``, so a comparison with
no procedure of its own inherits the headline and agrees by construction.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from export.report_selfcheck import (  # noqa: E402
    _oracle_posthoc_name_matches_what_ran, load_report)


def _page(tmp_path, headline, name="report.html"):
    body = (
        "<html><body><table><tbody>"
        "<tr><td>Test</td><td class='num-cell'>One-Way ANOVA</td></tr>"
        f"<tr><td>Post-hoc test <button class='info-btn'>i</button></td>"
        f"<td class='num-cell'>{headline}</td></tr>"
        "</tbody></table></body></html>"
    )
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return load_report(str(path))


def _judge(report, result):
    violations = []
    fired = _oracle_posthoc_name_matches_what_ran(report, result, violations)
    return fired, violations


def _pairs(names, test):
    return [{"group1": a, "group2": b, "test": test} for a, b in names]


ALL_PAIRS = [("G1", "G2"), ("G1", "G3"), ("G2", "G3")]
VS_CONTROL = [("G1", "G2"), ("G1", "G3")]
THREE_GROUPS = {"raw_data": {"G1": [1.0], "G2": [2.0], "G3": [3.0]}}


def test_a_headline_naming_another_family_is_caught(tmp_path):
    result = dict(THREE_GROUPS,
                  pairwise_comparisons=_pairs(ALL_PAIRS, "Paired t-test"))
    fired, violations = _judge(
        _page(tmp_path, "Dunnett-type (EMM + multivariate-t, Mixed)"), result)
    assert fired
    assert any("which the name does not mention" in v for v in violations), violations


def test_a_matching_family_is_quiet(tmp_path):
    result = dict(THREE_GROUPS,
                  pairwise_comparisons=_pairs(ALL_PAIRS, "Tukey HSD"))
    fired, violations = _judge(_page(tmp_path, "Tukey HSD Test (Pingouin)"), result)
    assert fired and not violations, violations


def test_a_control_family_over_all_pairs_is_caught(tmp_path):
    """Naming Dunnett while running every pair is a different correction."""
    result = dict(THREE_GROUPS,
                  pairwise_comparisons=_pairs(ALL_PAIRS, "Dunnett"))
    fired, violations = _judge(_page(tmp_path, "Dunnett Test"), result)
    assert fired
    assert any("multiplicity correction" in v for v in violations), violations


def test_a_control_family_over_its_own_contrasts_is_quiet(tmp_path):
    result = dict(THREE_GROUPS,
                  pairwise_comparisons=_pairs(VS_CONTROL, "Dunnett"))
    fired, violations = _judge(_page(tmp_path, "Dunnett Test"), result)
    assert fired and not violations, violations


def test_comparisons_without_their_own_procedure_are_not_evidence(tmp_path):
    """They inherit the headline when rendered, so they can never disagree."""
    result = dict(THREE_GROUPS, pairwise_comparisons=[
        {"group1": "G1", "group2": "G2"}, {"group1": "G1", "group2": "G3"}])
    fired, violations = _judge(_page(tmp_path, "Tukey HSD Test"), result)
    assert fired and not violations, violations


def test_a_row_without_a_procedure_cannot_vouch_for_the_headline(tmp_path):
    """The dangerous shape: some rows name another family, others name nothing.

    Letting the silent rows fall back to the headline puts the headline's own
    family into the evidence, and the check then finds itself agreeing with
    itself -- the disagreement the other rows carry is suppressed. That is the
    circular reading this oracle was written to avoid, and it hides rather than
    invents, so only a mixed result can expose it.
    """
    result = dict(THREE_GROUPS, pairwise_comparisons=[
        {"group1": "G1", "group2": "G2", "test": "Paired t-test"},
        {"group1": "G1", "group2": "G3"},
    ])
    fired, violations = _judge(
        _page(tmp_path, "Dunnett-type (EMM + multivariate-t, Mixed)"), result)
    assert fired
    assert any("which the name does not mention" in v for v in violations), violations


def test_a_label_that_names_two_procedures_covers_both(tmp_path):
    """A mixed design compares paired within-levels and independent between-groups.

    Its post-hoc says so -- "Pairwise Wilcoxon / Mann-Whitney U (within / between
    simple effects, Holm-corrected)" -- and a run that happened to make only
    within-comparisons uses one of the two. Reading one family per label reported
    that as a mismatch on the check's first real run.
    """
    headline = ("Pairwise Wilcoxon / Mann-Whitney U (within / between simple "
                "effects, Holm-corrected)")
    result = dict(THREE_GROUPS,
                  pairwise_comparisons=_pairs(ALL_PAIRS, "Wilcoxon signed-rank"))
    fired, violations = _judge(_page(tmp_path, headline), result)
    assert fired and not violations, violations

    # But a family the label does not mention is still a finding.
    result = dict(THREE_GROUPS, pairwise_comparisons=_pairs(ALL_PAIRS, "Tukey HSD"))
    _fired, violations = _judge(_page(tmp_path, headline), result)
    assert any("tukey" in v for v in violations), violations


def test_dunn_and_dunnett_are_not_the_same_test(tmp_path):
    """One name is a substring of the other; they are different procedures."""
    result = dict(THREE_GROUPS, pairwise_comparisons=_pairs(VS_CONTROL, "Dunn"))
    fired, violations = _judge(_page(tmp_path, "Dunnett Test"), result)
    assert fired
    assert any("dunn" in v for v in violations), violations


@pytest.mark.parametrize("headline", ["Some Bespoke Procedure", "", "   "])
def test_an_unrecognised_headline_stays_silent(tmp_path, headline):
    """An oracle that fires on labels it does not know is noise, and noise gets muted."""
    result = dict(THREE_GROUPS, pairwise_comparisons=_pairs(ALL_PAIRS, "Tukey HSD"))
    _fired, violations = _judge(_page(tmp_path, headline), result)
    assert not violations, violations


def test_no_comparisons_means_the_check_does_not_apply(tmp_path):
    fired, violations = _judge(_page(tmp_path, "Tukey HSD Test"),
                               dict(THREE_GROUPS, pairwise_comparisons=[]))
    assert not fired and not violations


def test_a_page_without_the_row_does_not_apply(tmp_path):
    path = tmp_path / "bare.html"
    path.write_text("<html><body><p>nothing here</p></body></html>", encoding="utf-8")
    fired, violations = _judge(load_report(str(path)),
                               dict(THREE_GROUPS,
                                    pairwise_comparisons=_pairs(ALL_PAIRS, "Tukey")))
    assert not fired and not violations
