"""Guards the fix for the post-hoc-label desync bug found in the Help Hub content
audit: `posthoc_test` must always be resynced to the actually-applied method once
`pairwise_comparisons` is replaced by the advanced post-hoc engine, regardless of
what the inline (pre-advanced-engine) label happened to say.
"""
import pandas as pd
import pytest

from statistical_testing.advanced_pipeline import perform_advanced_test_pipeline
from statistical_testing.engines.advanced_posthoc import AdvancedPostHocEngine
from statistical_testing.models import StatisticalResult


def _canned_posthoc_result(new_label):
    return StatisticalResult(
        test_name=new_label,
        statistic_value=None,
        p_value=None,
        metadata={
            "pairwise_comparisons": [{"group1": "A", "group2": "B", "p_value": 0.01}],
            "posthoc_test": new_label,
        },
    )


@pytest.mark.parametrize(
    "test_name,stale_inline_label,real_method_name",
    [
        # real_method_name is what the (mocked) engine reports back; the values
        # are the labels the current post-hoc layer actually produces, so this
        # parametrisation cannot advertise methods the app no longer has.
        ("two_way_anova", "Tukey HSD Test (Pingouin)", "Pairwise t-tests (independent, Holm-Sidak)"),
        ("mixed_anova", "Pairwise t-tests for interaction (Holm-Bonferroni)",
         "Simple main effects (Holm-Sidak per family)"),
        ("repeated_measures_anova", "Paired t-tests (Holm-Bonferroni)", "RM Post-hoc (paired_custom)"),
    ],
)
def test_posthoc_label_synced_to_real_method(
    monkeypatch, test_name, stale_inline_label, real_method_name
):
    from analysis.statisticaltester import StatisticalTester

    def _canned_initial_result(*args, **kwargs):
        return {
            "p_value": 0.001,
            "posthoc_test": stale_inline_label,
            "test_info": None,
        }

    monkeypatch.setattr(
        StatisticalTester, "_run_two_way_anova_logged",
        staticmethod(_canned_initial_result), raising=False,
    )
    monkeypatch.setattr(
        StatisticalTester, "_run_mixed_anova_logged",
        staticmethod(_canned_initial_result), raising=False,
    )
    monkeypatch.setattr(
        StatisticalTester, "_run_repeated_measures_anova_logged",
        staticmethod(_canned_initial_result), raising=False,
    )
    monkeypatch.setattr(
        AdvancedPostHocEngine, "execute",
        lambda self, data: _canned_posthoc_result(real_method_name),
    )

    # perform_advanced_test_pipeline runs validate_test_design() and the
    # ExtractionEngine before the (mocked) test-runner is ever called, so the
    # df/kwargs must satisfy each test's real design requirements even though
    # the statistical computation itself is short-circuited by the monkeypatch.
    df = pd.DataFrame(
        {
            "dv": [1, 2, 3, 4, 5, 6, 7, 8],
            "factorA": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "factorB": ["X", "X", "Y", "Y", "X", "X", "Y", "Y"],
            "subject": [1, 2, 1, 2, 1, 2, 1, 2],
        }
    )
    pipeline_kwargs = {
        "two_way_anova": dict(subject=None, between=["factorA", "factorB"], within=None),
        "mixed_anova": dict(subject="subject", between=["factorA"], within=["factorB"]),
        "repeated_measures_anova": dict(subject="subject", between=None, within=["factorB"]),
    }[test_name]

    result = perform_advanced_test_pipeline(
        df=df,
        test=test_name,
        dv="dv",
        force_parametric=True,
        alpha=0.05,
        **pipeline_kwargs,
    )

    assert result["posthoc_test"] == real_method_name
    assert result["posthoc_test"] != stale_inline_label
