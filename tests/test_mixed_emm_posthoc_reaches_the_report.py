"""The mixed EMM/mvt post-hoc has to survive the seam it is called across.

The engine in ``analysis.emm_posthoc`` matches its R reference and is covered by
its own tests, which call it with the arguments it documents. The pipeline calls
it through ``MixedAnovaPostHocAnalyzer.perform_test``, and there the three things
that cross the seam were each wrong in a different way:

* ``between`` / ``within`` arrive as **lists** from the advanced pipeline. They
  were passed straight into ``df[[subject, between, within, dv]]``, which raises
  ``TypeError: unhashable type: 'list'``.
* ``control_group`` arrives as a **cell label** ("Between=B0, Time=T0"), because
  that is what the control-group dialog offers. The engine wants a between-factor
  level and rejected it as "not present", so the run degraded to isolated
  t-tests without anyone seeing why.
* The comparisons came back labelled ``"B0:T0"``, a vocabulary nothing else in
  the report uses, so the chart could not match them to any group and drew no
  brackets.

Each of these hides behind the next, so all three are asserted separately.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Root conftest.py puts src/ on sys.path and forces headless Qt.
from analysis.posthoc_core import MixedAnovaPostHocAnalyzer

BETWEEN, WITHIN, DV, SUBJECT = "Between", "Time", "Val", "Subject"
CELL_CONTROL = "Between=B0, Time=T0"


def _split_plot(n_subjects: int = 16) -> pd.DataFrame:
    """A balanced, complete split-plot -- the design the closed form covers."""
    rng = np.random.default_rng(9)
    offsets = rng.normal(0, 0.6, size=n_subjects)
    return pd.DataFrame([
        {SUBJECT: f"S{s}", WITHIN: level, BETWEEN: f"B{s % 2}",
         DV: 10.0 + 2.5 * index + 3.0 * (s % 2) + offsets[s] + rng.normal(0, 0.5)}
        for s in range(n_subjects) for index, level in enumerate(("T0", "T1"))
    ])


def _run(**overrides):
    kwargs = dict(
        df=_split_plot(), between=[BETWEEN], within=[WITHIN], dv=DV, subject=SUBJECT,
        alpha=0.05, selected_comparisons=None, method="emm_mvt",
        control_group=CELL_CONTROL,
    )
    kwargs.update(overrides)
    return MixedAnovaPostHocAnalyzer.perform_test(**kwargs)


def test_the_pipelines_argument_shape_reaches_the_emm_engine():
    """Lists for between/within, exactly as advanced_pipeline passes them."""
    result = _run()
    assert result.get("error") is None, result.get("error")
    assert result["posthoc_test"] == "Dunnett-type (EMM + multivariate-t, Mixed)", (
        "the EMM/mvt branch was not reached; the run fell back to something else"
    )
    assert len(result["pairwise_comparisons"]) == 2, result["pairwise_comparisons"]


def test_bare_column_names_still_work():
    """Direct callers pass strings; both shapes have to land in the same place."""
    result = _run(between=BETWEEN, within=WITHIN)
    assert result["posthoc_test"] == "Dunnett-type (EMM + multivariate-t, Mixed)"


@pytest.mark.parametrize("control", [CELL_CONTROL, "B0"])
def test_a_cell_label_control_resolves_to_its_between_level(control):
    """The dialog offers cells; the contrast family is defined on between levels."""
    result = _run(control_group=control)
    assert result["posthoc_test"] == "Dunnett-type (EMM + multivariate-t, Mixed)"
    controls = {row["group1"] for row in result["pairwise_comparisons"]}
    assert controls == {f"{BETWEEN}=B0, {WITHIN}=T0", f"{BETWEEN}=B0, {WITHIN}=T1"}, controls


def test_a_control_that_names_nothing_is_not_quietly_replaced():
    """Unresolvable stays unresolvable, so the engine's own refusal still fires."""
    assert MixedAnovaPostHocAnalyzer._between_level_of(
        "Between=B9, Time=T0", _split_plot(), BETWEEN) == "Between=B9, Time=T0"
    result = _run(control_group="Between=B9, Time=T0")
    # Falls back to the effect-driven post-hoc rather than contrasting against
    # an arbitrary group nobody chose.
    assert result["posthoc_test"] != "Dunnett-type (EMM + multivariate-t, Mixed)"


def test_the_comparison_labels_are_the_vocabulary_the_chart_uses():
    """``b=level, w=level`` -- the same spelling advanced_posthoc builds groups with."""
    result = _run()
    assert result["posthoc_test"] == "Dunnett-type (EMM + multivariate-t, Mixed)", (
        "these are the fallback's labels, not the EMM branch's"
    )
    group_names = []
    for b_val in ("B0", "B1"):
        for w_val in ("T0", "T1"):
            group_names.append(f"{BETWEEN}={b_val}, {WITHIN}={w_val}")
    for row in result["pairwise_comparisons"]:
        assert row["group1"] in group_names, row["group1"]
        assert row["group2"] in group_names, row["group2"]


def test_a_failure_reports_its_own_cause(monkeypatch):
    """The handler used to raise UnboundLocalError and lose the real exception.

    ``result`` was created below the EMM branch but written to by the ``except``
    at the bottom, so anything raised in that branch replaced the diagnosis with
    "cannot access local variable 'result'" -- which is what the pipeline then
    recorded as the post-hoc's error.
    """
    import analysis.emm_posthoc as emm

    def _boom(*args, **kwargs):
        raise ValueError("the real cause")

    monkeypatch.setattr(emm, "mixed_dunnett_emm_mvt", _boom)
    result = _run()
    assert "the real cause" in str(result.get("error")), result.get("error")
    assert "local variable" not in str(result.get("error"))
