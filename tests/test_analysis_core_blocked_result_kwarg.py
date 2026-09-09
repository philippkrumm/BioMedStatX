"""make_blocked_result() has signature (reason, *, code, details=None, warnings=None) - no
test_name parameter. analysis_core.py's Mixed ANOVA invalid-design check passes test_name=
anyway, so the TypeError this raises gets caught by the outer except-Exception in
AnalysisManager._analyze_single_dataset and replaces the intended "Mixed ANOVA requires two
factors" message with a confusing Python signature error instead.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import pytest

from analysis.analysis_core import AnalysisManager


@pytest.fixture
def dummy_file(tmp_path):
    path = tmp_path / "dummy.xlsx"
    pd.DataFrame({"a": [1]}).to_excel(path, index=False)
    return str(path)


def test_mixed_anova_invalid_design_reports_the_real_message_not_a_typeerror(dummy_file, tmp_path):
    df = pd.DataFrame({
        "Group": ["ctrl", "ctrl", "a", "a"],
        "Time": ["T1", "T2", "T1", "T2"],
        "Value": [1.0, 2.0, 3.0, 4.0],
    })
    ctx = {
        "injected_df": df,
        "factor_columns": ["Group"],
        "between_factors": ["Group"],
        "dv_columns": ["Value"],
        "group_labels": ["ctrl", "a"],
        "mode": "single",
    }

    result = AnalysisManager.analyze(
        file_path=dummy_file,
        group_col="Group",
        groups=["ctrl", "a"],
        value_cols=["Value"],
        save_plot=False,
        skip_plots=True,
        file_name=str(tmp_path / "out"),
        analysis_context=ctx,
        test="mixed_anova",
        additional_factors=["Time"],  # only 1 factor: triggers the "requires two factors" block
    )

    assert result.get("blocked") is True
    assert result.get("block_code") == "INVALID_DESIGN", (
        f"expected INVALID_DESIGN, got {result.get('block_code')!r} "
        f"(reason={result.get('block_reason')!r}) - the TypeError from the bad "
        f"test_name= kwarg is likely being caught and relabeled UNHANDLED_EXCEPTION"
    )
    assert result.get("block_reason") == "Mixed ANOVA requires two factors (between and within)"
