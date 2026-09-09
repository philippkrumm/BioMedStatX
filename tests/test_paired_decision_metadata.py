"""The paired assumption metadata claimed a Welch t-test on a paired design.

check_normality_and_variance() has the correct ``is_paired = model_type ==
"paired"`` in scope, but the decision-strategy call hardcoded is_paired=False:

    decision_strategy = select_comparison_test(
        is_normal=post_norm, is_homoscedastic=post_var,
        is_paired=False,               # <- ignores the design
        group_count=len(valid_groups))

So a paired, normal design produced decision.strategy="welch_ttest" and a note
reading "Welch's t-test will be used", which the report embeds in the decision
tree next to the actual "Paired t-test" leaf and an "Assumptions met" node --
three statements, two of them wrong. The executed test was always correct
(statisticaltester recomputes the strategy with is_paired=dependent); only the
logged rationale was broken.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from analysis.statisticaltester import StatisticalTester


@pytest.fixture(scope="module", autouse=True)
def _qt_and_dialogs():
    """check_normality_and_variance opens a transformation dialog when a group
    is non-normal. Neutralise every modal, same as the golden-core suite."""
    try:
        from PyQt5.QtWidgets import QApplication, QDialog
    except Exception:
        yield
        return
    app = QApplication.instance() or QApplication([])
    mp = pytest.MonkeyPatch()
    mp.setattr(QDialog, "exec_", lambda self, *a, **k: 0, raising=False)
    mp.setattr(QDialog, "exec", lambda self, *a, **k: 0, raising=False)
    try:
        from analysis.statisticaltester import UIDialogManager
        # "skip" = continue without transformation (non-parametric); None now
        # means the user cancelled the dialog, which aborts the whole analysis.
        mp.setattr(UIDialogManager, "select_transformation_dialog", staticmethod(lambda *a, **k: "skip"), raising=False)
        for name in ("select_posthoc_test_dialog", "select_nonparametric_posthoc_dialog",
                     "select_control_group_dialog", "select_custom_pairs_dialog"):
            mp.setattr(UIDialogManager, name, staticmethod(lambda *a, **k: None), raising=False)
    except Exception:
        pass
    yield app
    mp.undo()


def _paired_samples(seed=21, n=20):
    """Audit fixture: seed 21, N=20, a normal paired shift A -> B."""
    rng = np.random.default_rng(seed)
    a = rng.normal(10, 2, n)
    b = rng.normal(12, 2, n)
    return {"A": list(a), "B": list(b)}


def _nonnormal_paired_samples(seed=202, n=18):
    rng = np.random.default_rng(seed)
    base = rng.lognormal(1.0, 1.1, n)
    a = list(base * rng.lognormal(0, 0.25, n))
    b = list(base * 2.2 * rng.lognormal(0, 0.25, n))
    a[0], b[1] = 900.0, -700.0
    return {"A": a, "B": b}


def _decide(samples, model_type):
    _t, _rec, info = StatisticalTester.check_normality_and_variance(
        ["A", "B"], samples, model_type=model_type, formula="Value ~ C(Group)",
    )
    return info.get("decision", {}), info.get("note")


def test_paired_normal_reports_paired_strategy():
    decision, note = _decide(_paired_samples(), model_type="paired")
    assert decision.get("model_type") == "paired"
    assert decision.get("strategy") == "paired_ttest", (
        f"paired design logged strategy {decision.get('strategy')!r}"
    )
    if note:
        assert "Welch" not in note, f"paired run still claims Welch: {note!r}"


def test_paired_nonnormal_reports_wilcoxon_strategy():
    decision, _ = _decide(_nonnormal_paired_samples(), model_type="paired")
    assert decision.get("strategy") == "wilcoxon", (
        f"non-normal paired design logged {decision.get('strategy')!r}"
    )


def test_unpaired_normal_still_reports_welch():
    """Positive control: the independent path must be unchanged."""
    decision, note = _decide(_paired_samples(), model_type="ttest")
    assert decision.get("strategy") == "welch_ttest"
    assert decision.get("model_type") == "ttest"


def test_logged_strategy_matches_the_executed_test():
    """End-to-end guard: the metadata must agree with what actually ran."""
    samples = _paired_samples()
    decision, _ = _decide(samples, model_type="paired")

    results = {"test": "", "descriptive": {}, "raw_data": {}}
    executed = StatisticalTester._paired_ttest(
        results, "A", "B", np.array(samples["A"]), np.array(samples["B"]), 0.05,
    )
    assert "Paired" in executed["test"]
    # the logged strategy now names the same family the engine actually ran
    assert decision.get("strategy") == "paired_ttest"
