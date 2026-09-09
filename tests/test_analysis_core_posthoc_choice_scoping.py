"""analysis_core.py's post-hoc block only assigns posthoc_choice inside the parametric
dialog's else-branch (~line 1116), but reads it afterwards (~line 1196) regardless of which
branch ran. When the significant result comes from a non-parametric test (Kruskal-Wallis /
Friedman) AND the earlier, separate post-hoc dispatch inside statisticaltester.py's own
perform_statistical_test/_stat_test_multi_groups didn't already populate
test_results['pairwise_comparisons'], analysis_core.py's own re-entry branch runs
perform_refactored_posthoc_testing a second time without ever assigning posthoc_choice - so
the read at line 1196 raises UnboundLocalError, caught by the outer except-Exception and
surfaced as a confusing UNHANDLED_EXCEPTION block instead of the intended silent no-op
(Dunnett's control-group key never applies to the non-parametric branch in the first place).

Reproducing this precisely requires two things confirmed empirically during TDD for this fix:
1. StatisticalTester.check_normality_and_variance mocked to return "non_parametric" (the real
   Shapiro-Wilk/Levene detection is data-dependent and unreliable to target directly).
2. perform_refactored_posthoc_testing mocked STATEFULLY - empty pairwise_comparisons on the
   first call (simulating the earlier statisticaltester.py-internal dispatch not populating it,
   e.g. because its own dialog step was skipped) and a populated result on the second call
   (analysis_core.py's own re-entry, the vulnerable one). A single non-stateful mock that always
   returns a populated result satisfies BOTH call sites' checks and never reaches the buggy
   branch at all - confirmed via sys.settrace during initial TDD that this silently no-ops
   instead of reproducing the bug.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import pytest

from analysis.analysis_core import AnalysisManager
from analysis.statisticaltester import StatisticalTester


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch):
    from PyQt5.QtWidgets import QDialog
    monkeypatch.setattr(QDialog, "exec_", lambda self, *a, **k: 0, raising=False)
    monkeypatch.setattr(QDialog, "exec", lambda self, *a, **k: 0, raising=False)


@pytest.fixture
def dummy_file(tmp_path):
    path = tmp_path / "dummy.xlsx"
    pd.DataFrame({"a": [1]}).to_excel(path, index=False)
    return str(path)


def test_kruskal_significant_result_does_not_crash_with_nameerror(dummy_file, tmp_path, monkeypatch):
    call_count = {"n": 0}

    def _fake_perform_refactored_posthoc_testing(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Simulate the earlier, separate post-hoc dispatch inside
            # statisticaltester.py not finding/returning comparisons yet.
            return {"posthoc_test": "No post-hoc tests performed", "pairwise_comparisons": [], "error": None}
        # analysis_core.py's own re-entry branch (the vulnerable one).
        return {"posthoc_test": "Dunn (Holm-Sidak)", "pairwise_comparisons": [
            {"group1": "a", "group2": "b", "p_value": 0.01, "significant": True}
        ], "error": None}

    def _fake_check_normality_and_variance(groups, samples, **kwargs):
        return dict(samples), "non_parametric", {
            "pre_transformation": {}, "post_transformation": {},
            "transformation": None, "validation_notes": [],
        }

    monkeypatch.setattr(
        StatisticalTester, "perform_refactored_posthoc_testing",
        staticmethod(_fake_perform_refactored_posthoc_testing), raising=False
    )
    monkeypatch.setattr(
        StatisticalTester, "check_normality_and_variance",
        staticmethod(_fake_check_normality_and_variance), raising=False
    )

    df = pd.DataFrame({
        "Group": ["a"] * 5 + ["b"] * 5 + ["c"] * 5,
        "Value": [1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24],
    })
    ctx = {
        "injected_df": df,
        "factor_columns": ["Group"],
        "between_factors": ["Group"],
        "dv_columns": ["Value"],
        "group_labels": ["a", "b", "c"],
        "mode": "single",
    }

    result = AnalysisManager.analyze(
        file_path=dummy_file,
        group_col="Group",
        groups=["a", "b", "c"],
        value_cols=["Value"],
        save_plot=False,
        skip_plots=True,
        file_name=str(tmp_path / "out"),
        analysis_context=ctx,
    )

    assert call_count["n"] >= 2, (
        "test setup didn't reach analysis_core.py's own post-hoc re-entry branch "
        "(perform_refactored_posthoc_testing was called fewer than 2 times) - the bug "
        "this test targets never got exercised"
    )
    assert result.get("block_code") != "UNHANDLED_EXCEPTION", (
        f"got block_reason={result.get('block_reason')!r} - the posthoc_choice "
        f"UnboundLocalError fired"
    )
    assert "posthoc_choice" not in str(result.get("block_reason", ""))
    assert result.get("pairwise_comparisons") == [
        {"group1": "a", "group2": "b", "p_value": 0.01, "significant": True}
    ]
