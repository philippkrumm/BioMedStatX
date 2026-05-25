import sys
from pathlib import Path
from importlib import import_module

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_core = import_module("statistical_testing.core")
_decision_logic = import_module("statistical_testing.decision_logic")
_comparison_engine = import_module("statistical_testing.engines.comparison")
_advanced_posthoc_engine = import_module("statistical_testing.engines.advanced_posthoc")
_posthoc_engine = import_module("statistical_testing.engines.posthoc")
_reporting_engine = import_module("statistical_testing.engines.reporting")
_finalization_engine = import_module("statistical_testing.engines.finalization")
_assumption_bridge_engine = import_module("statistical_testing.engines.assumption_bridge")
_transformation_engine = import_module("statistical_testing.engines.transformation")
_recommendation_engine = import_module("statistical_testing.engines.recommendation")
_extraction_engine = import_module("statistical_testing.engines.extraction")
_models = import_module("statistical_testing.models")
_statisticaltester = import_module("analysis.statisticaltester")
_validators = import_module("statistical_testing.validators")

StatisticalTesterCore = _core.StatisticalTesterCore
AssumptionState = _decision_logic.AssumptionState
DecisionInput = _decision_logic.DecisionInput
choose_comparison_strategy = _decision_logic.choose_comparison_strategy
extract_assumption_state = _decision_logic.extract_assumption_state
select_comparison_test = _decision_logic.select_comparison_test
strategy_to_recommendation = _decision_logic.strategy_to_recommendation
ComparisonEngine = _comparison_engine.ComparisonEngine
AdvancedPostHocEngine = _advanced_posthoc_engine.AdvancedPostHocEngine
PostHocEngine = _posthoc_engine.PostHocEngine
ReportingEngine = _reporting_engine.ReportingEngine
FinalizationEngine = _finalization_engine.FinalizationEngine
AssumptionBridgeEngine = _assumption_bridge_engine.AssumptionBridgeEngine
TransformationEngine = _transformation_engine.TransformationEngine
RecommendationEngine = _recommendation_engine.RecommendationEngine
ExtractionEngine = _extraction_engine.ExtractionEngine
StatisticalResult = _models.StatisticalResult
StatisticalTester = _statisticaltester.StatisticalTester
ValidationError = _validators.ValidationError


class _StubEngine:
    def __init__(self, name: str):
        self.name = name

    def execute(self, data):
        return StatisticalResult(test_name=self.name, statistic_value=1.0, p_value=0.05, metadata={"data": data})


def test_select_comparison_test_two_group_mapping():
    assert select_comparison_test(is_normal=True, is_homoscedastic=True, is_paired=False, group_count=2) == "welch_ttest"
    assert select_comparison_test(is_normal=True, is_homoscedastic=False, is_paired=False, group_count=2) == "welch_ttest"
    assert select_comparison_test(is_normal=False, is_homoscedastic=False, is_paired=False, group_count=2) == "mann_whitney_u"


def test_select_comparison_test_paired_mapping():
    assert select_comparison_test(is_normal=True, is_homoscedastic=True, is_paired=True, group_count=2) == "paired_ttest"
    assert select_comparison_test(is_normal=False, is_homoscedastic=True, is_paired=True, group_count=2) == "wilcoxon"


def test_select_comparison_test_multi_group_mapping():
    # normal + homoscedastic -> one_way_anova
    assert (
        select_comparison_test(
            is_normal=True, is_homoscedastic=True, is_paired=False, group_count=3
        )
        == "one_way_anova"
    )

    # normal + heteroscedastic -> welch_anova
    assert (
        select_comparison_test(
            is_normal=True, is_homoscedastic=False, is_paired=False, group_count=3
        )
        == "welch_anova"
    )
    assert select_comparison_test(is_normal=False, is_homoscedastic=False, is_paired=False, group_count=4) == "kruskal_wallis"


def test_extract_assumption_state_new_structure_prefers_post_transformation():
    state = extract_assumption_state(
        {
            "transformation": "log10",
            "post_transformation": {
                "residuals_normality": {"is_normal": True},
                "variance": {"equal_variance": False},
            },
            "pre_transformation": {
                "residuals_normality": {"is_normal": False},
                "variance": {"equal_variance": True},
            },
        }
    )
    assert state == AssumptionState(residuals_normal=True, equal_variance=False)


def test_strategy_to_recommendation_mapping():
    assert strategy_to_recommendation("welch_ttest") == "welch"
    assert strategy_to_recommendation("kruskal_wallis") == "non_parametric"
    assert strategy_to_recommendation("student_ttest") == "parametric"


def test_choose_comparison_strategy_compatibility_wrapper():
    decision = DecisionInput(group_count=2, dependent=False, residuals_normal=True, equal_variance=False)
    assert choose_comparison_strategy(decision) == "welch_ttest"


def test_core_uses_decision_strategy_and_engine_mapping():
    core = StatisticalTesterCore(
        engines={
            "welch_ttest": _StubEngine("welch_ttest"),
        }
    )

    result = core.run(
        {
            "samples": {"A": [1.0, 2.0], "B": [2.0, 3.0]},
            "group_count": 2,
            "dependent": False,
            "residuals_normal": True,
            "equal_variance": True,
        }
    )

    assert isinstance(result, StatisticalResult)
    assert result.test_name == "welch_ttest"


def test_core_raises_on_validation_errors_before_selection():
    core = StatisticalTesterCore(engines={"student_ttest": _StubEngine("student_ttest")})

    with pytest.raises(ValidationError):
        core.run(
            {
                "samples": {"A": [1.0], "B": [2.0]},
                "group_count": 2,
                "dependent": False,
                "residuals_normal": True,
                "equal_variance": True,
            }
        )


def test_comparison_engine_executes_welch_ttest_strategy():
    engine = ComparisonEngine()
    result = engine.execute(
        {
            "strategy": "welch_ttest",
            "groups": ["A", "B"],
            "samples": {
                "A": [1.0, 2.0, 3.0, 4.0, 5.0],
                "B": [1.2, 1.4, 1.6, 1.8, 2.0],
            },
            "alpha": 0.05,
            "results": {"pairwise_comparisons": []},
        }
    )

    assert isinstance(result, StatisticalResult)
    legacy = StatisticalTester.from_statistical_result(result)
    assert legacy["test"] == "Welch's t-test (unequal variances)"
    assert isinstance(legacy.get("p_value"), float)


def test_statisticaltester_two_group_uses_comparison_engine(monkeypatch):
    called = {}

    def _fake_execute(self, data):
        called["strategy"] = data.get("strategy")
        return StatisticalResult(
            test_name="engine_stub_test",
            statistic_value=1.0,
            p_value=0.5,
            metadata={"pairwise_comparisons": [], "descriptive": {}, "descriptive_transformed": {}},
        )

    monkeypatch.setattr(_statisticaltester.ComparisonEngine, "execute", _fake_execute)

    result = StatisticalTester._stat_test_two_groups(
        results={
            "pairwise_comparisons": [],
            "descriptive": {},
            "descriptive_transformed": {},
            "alpha": 0.05,
        },
        valid_groups=["A", "B"],
        samples_to_use={"A": [1.0, 2.0, 3.0, 4.0, 5.0], "B": [1.5, 2.5, 3.5, 4.5, 5.5]},
        original_samples={"A": [1.0, 2.0, 3.0, 4.0, 5.0], "B": [1.5, 2.5, 3.5, 4.5, 5.5]},
        dependent=False,
        test_recommendation="parametric",
        alpha=0.05,
        test_info=None,
    )

    assert called.get("strategy") == "welch_ttest"
    assert result["test"] == "engine_stub_test"


def test_comparison_engine_executes_kruskal_strategy():
    engine = ComparisonEngine()
    result = engine.execute(
        {
            "strategy": "kruskal_wallis",
            "groups": ["A", "B", "C"],
            "samples": {
                "A": [1.0, 2.0, 3.0, 4.0, 5.0],
                "B": [1.2, 1.4, 1.6, 1.8, 2.0],
                "C": [2.1, 2.5, 2.7, 2.9, 3.1],
            },
            "alpha": 0.05,
            "results": {"pairwise_comparisons": []},
        }
    )

    assert isinstance(result, StatisticalResult)
    legacy = StatisticalTester.from_statistical_result(result)
    assert legacy["test"] == "Kruskal-Wallis test"
    assert isinstance(legacy.get("p_value"), float)


def test_statisticaltester_multigroup_uses_comparison_engine(monkeypatch):
    called = {}

    def _fake_execute(self, data):
        called["strategy"] = data.get("strategy")
        return StatisticalResult(
            test_name="engine_multi_stub",
            statistic_value=2.0,
            p_value=0.6,
            metadata={
                "pairwise_comparisons": [],
                "descriptive": {},
                "descriptive_transformed": {},
                "alpha": 0.05,
            },
        )

    monkeypatch.setattr(_statisticaltester.ComparisonEngine, "execute", _fake_execute)

    result = StatisticalTester._stat_test_multi_groups(
        results={
            "pairwise_comparisons": [],
            "descriptive": {},
            "descriptive_transformed": {},
            "alpha": 0.05,
        },
        valid_groups=["A", "B", "C"],
        samples_to_use={
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
            "B": [1.5, 2.5, 3.5, 4.5, 5.5],
            "C": [2.0, 2.5, 3.0, 3.5, 4.0],
        },
        dependent=False,
        test_recommendation="parametric",
        alpha=0.05,
        test_info=None,
        trace=None,
    )

    assert called.get("strategy") == "one_way_anova"
    assert result["test"] == "engine_multi_stub"


def test_posthoc_engine_delegates_to_refactored_posthoc(monkeypatch):
    def _fake_posthoc(*args, **kwargs):
        return {
            "posthoc_test": "Holm-Sidak",
            "pairwise_comparisons": [{"group1": "A", "group2": "B", "p_value": 0.04}],
        }

    monkeypatch.setattr(_statisticaltester.StatisticalTester, "perform_refactored_posthoc_testing", staticmethod(_fake_posthoc))

    result = PostHocEngine().execute(
        {
            "groups": ["A", "B", "C"],
            "samples": {
                "A": [1.0, 2.0, 3.0, 4.0, 5.0],
                "B": [1.5, 2.5, 3.5, 4.5, 5.5],
                "C": [2.0, 2.2, 2.4, 2.6, 2.8],
            },
            "test_recommendation": "parametric",
            "alpha": 0.05,
            "test_info": None,
        }
    )

    assert isinstance(result, StatisticalResult)
    assert result.test_name == "Holm-Sidak"
    assert result.metadata.get("pairwise_comparisons")


def test_statisticaltester_multigroup_uses_posthoc_engine(monkeypatch):
    calls = {}

    def _fake_primary_execute(self, data):
        return StatisticalResult(
            test_name="engine_multi_primary",
            statistic_value=3.1,
            p_value=0.01,
            metadata={
                "pairwise_comparisons": [],
                "descriptive": {},
                "descriptive_transformed": {},
                "alpha": 0.05,
            },
        )

    def _fake_posthoc_execute(self, data):
        calls["test_recommendation"] = data.get("test_recommendation")
        return StatisticalResult(
            test_name="Holm-Sidak",
            statistic_value=None,
            p_value=None,
            metadata={
                "posthoc_test": "Holm-Sidak",
                "pairwise_comparisons": [{"group1": "A", "group2": "B", "p_value": 0.04}],
            },
        )

    monkeypatch.setattr(_statisticaltester.ComparisonEngine, "execute", _fake_primary_execute)
    monkeypatch.setattr(_statisticaltester.PostHocEngine, "execute", _fake_posthoc_execute)

    result = StatisticalTester._stat_test_multi_groups(
        results={
            "pairwise_comparisons": [],
            "descriptive": {},
            "descriptive_transformed": {},
            "alpha": 0.05,
        },
        valid_groups=["A", "B", "C"],
        samples_to_use={
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
            "B": [1.5, 2.5, 3.5, 4.5, 5.5],
            "C": [2.0, 2.5, 3.0, 3.5, 4.0],
        },
        dependent=False,
        test_recommendation="parametric",
        alpha=0.05,
        test_info=None,
        trace=None,
    )

    assert calls.get("test_recommendation") == "parametric"
    assert result["posthoc_test"] == "Holm-Sidak"
    assert len(result.get("pairwise_comparisons", [])) == 1


def test_advanced_posthoc_engine_nonparam_skip_when_not_significant():
    engine = AdvancedPostHocEngine()
    result = engine.execute(
        {
            "mode": "nonparametric_fallback",
            "res": {"p_value": 0.3, "error": None, "pairwise_comparisons": []},
            "test": "mixed_anova",
            "alpha": 0.05,
        }
    )

    assert isinstance(result, StatisticalResult)
    assert result.metadata.get("posthoc_skipped") is True
    assert "not significant" in str(result.metadata.get("posthoc_skip_reason", "")).lower()


def test_advanced_posthoc_engine_rejects_unknown_mode():
    engine = AdvancedPostHocEngine()
    result = engine.execute({"mode": "unknown"})
    assert result.test_name == "advanced_posthoc_failed"
    assert "Unsupported" in str(result.metadata.get("error", ""))


def test_reporting_engine_builds_modern_fallback_analysis_log():
    engine = ReportingEngine()
    result = engine.execute(
        {
            "mode": "modern_fallback_analysis_log",
            "test": "mixed_anova",
            "dv": "score",
            "transformation_type": "log10",
            "test_info": {
                "pre_transformation": {"residuals_normality": {"p_value": 0.01, "is_normal": False}},
                "post_transformation": {"residuals_normality": {"p_value": 0.20, "is_normal": True}},
            },
            "res": {
                "analysis_note": "Fallback used due to model incompatibility.",
                "model_class": "MixedLMResults",
                "model_family": "gaussian",
                "p_value": 0.04,
                "posthoc_test": "Pairwise t-tests",
                "pairwise_comparisons": [{"group1": "A", "group2": "B", "p_value": 0.03}],
            },
        }
    )

    assert isinstance(result, StatisticalResult)
    log = str(result.metadata.get("analysis_log", ""))
    assert "Fallback path: statsmodels modern model" in log
    assert "Original data normality: p = 0.0100" in log
    assert "After transformation normality: p = 0.2000" in log
    assert "Applied transformation before fallback: log10" in log
    assert "Fallback result p-value: 0.0400" in log
    assert "Pairwise comparisons generated: 1" in log


def test_reporting_engine_appends_error_to_analysis_note_when_present():
    engine = ReportingEngine()
    result = engine.execute(
        {
            "mode": "modern_fallback_analysis_log",
            "test": "mixed_anova",
            "dv": "score",
            "res": {
                "analysis_note": "Fallback used due to incompatibility.",
                "error": "Model failed to converge",
            },
        }
    )

    assert isinstance(result, StatisticalResult)
    assert "Error: Model failed to converge" in str(result.metadata.get("analysis_log", ""))
    assert result.metadata.get("analysis_note") == (
        "Fallback used due to incompatibility. Error: Model failed to converge"
    )


def test_finalization_engine_sets_legacy_label_fields_without_export():
    result = FinalizationEngine().execute(
        {
            "mode": "advanced_result",
            "res": {"test": "Mixed ANOVA"},
        }
    )

    assert isinstance(result, StatisticalResult)
    assert result.metadata.get("final_test_label") == "Mixed ANOVA"
    assert result.metadata.get("tested_against") == "Mixed ANOVA"
    assert "excel_file" not in result.metadata




def test_assumption_bridge_engine_projects_pre_post_to_legacy_fields():
    bridge_result = AssumptionBridgeEngine().execute(
        {
            "mode": "advanced_assumption_projection",
            "res": {},
            "test_info": {
                "transformation": "log10",
                "boxcox_lambda": 0.3,
                "pre_transformation": {
                    "residuals_normality": {"statistic": 0.91, "p_value": 0.02, "is_normal": False},
                    "variance": {"statistic": 2.1, "p_value": 0.04, "equal_variance": False},
                },
                "post_transformation": {
                    "residuals_normality": {"statistic": 0.97, "p_value": 0.22, "is_normal": True},
                    "variance": {"statistic": 1.2, "p_value": 0.31, "equal_variance": True},
                },
            },
        }
    )

    assert isinstance(bridge_result, StatisticalResult)
    updates = dict(bridge_result.metadata or {})
    assert "test_info" in updates
    assert updates.get("transformation") == "log10"
    assert updates.get("boxcox_lambda") == 0.3
    assert updates.get("normality_tests", {}).get("all_data", {}).get("p_value") == 0.02
    assert updates.get("normality_tests", {}).get("transformed_data", {}).get("p_value") == 0.22
    assert updates.get("variance_test", {}).get("equal_variance") is False
    assert updates.get("variance_test", {}).get("transformed", {}).get("equal_variance") is True


def test_assumption_bridge_engine_preserves_existing_legacy_fields():
    bridge_result = AssumptionBridgeEngine().execute(
        {
            "mode": "advanced_assumption_projection",
            "res": {
                "normality_tests": {"all_data": {"p_value": 0.9}},
                "variance_test": {"equal_variance": True},
                "transformation": "none",
            },
            "test_info": {
                "transformation": "log10",
                "normality_tests": {"all_data": {"p_value": 0.2}},
                "variance_test": {"equal_variance": False},
            },
        }
    )

    assert isinstance(bridge_result, StatisticalResult)
    updates = dict(bridge_result.metadata or {})
    assert "normality_tests" not in updates
    assert "variance_test" not in updates
    assert "transformation" not in updates


def test_transformation_engine_applies_log10_and_rebuilds_samples():
    import pandas as pd

    df = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "score": [1.0, 10.0, 100.0, 1000.0],
        }
    )

    result = TransformationEngine().execute(
        {
            "mode": "advanced_transformation",
            "df": df,
            "dv": "score",
            "test": "repeated_measures_anova",
            "within": ["group"],
            "test_info": {"transformation": "log10"},
            "transformed_samples": {"A": [1.0, 10.0], "B": [100.0, 1000.0]},
        }
    )

    assert isinstance(result, StatisticalResult)
    updates = dict(result.metadata or {})
    assert updates.get("transformation_type") == "log10"
    assert updates.get("error") is None
    transformed_samples = updates.get("transformed_samples", {})
    assert transformed_samples.get("A") == [0.0, 1.0]
    assert transformed_samples.get("B") == [2.0, 3.0]


def test_transformation_engine_keeps_existing_samples_when_no_transform():
    import pandas as pd

    df = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "score": [1.0, 2.0, 3.0, 4.0],
        }
    )
    existing = {"A": [1.0, 2.0], "B": [3.0, 4.0]}

    result = TransformationEngine().execute(
        {
            "mode": "advanced_transformation",
            "df": df,
            "dv": "score",
            "test": "repeated_measures_anova",
            "within": ["group"],
            "test_info": {"transformation": "none"},
            "transformed_samples": existing,
        }
    )

    assert isinstance(result, StatisticalResult)
    updates = dict(result.metadata or {})
    assert updates.get("transformation_type") == "none"
    assert updates.get("transformed_samples") == existing
    assert updates.get("error") is None


def test_recommendation_engine_force_parametric_override():
    result = RecommendationEngine().execute(
        {
            "mode": "advanced_recommendation",
            "recommendation": "non_parametric",
            "force_parametric": True,
            "test_info": {
                "post_transformation": {
                    "residuals_normality": {"is_normal": False}
                }
            },
        }
    )

    assert isinstance(result, StatisticalResult)
    updates = dict(result.metadata or {})
    assert updates.get("effective_recommendation") == "parametric"
    assert updates.get("forced") is True


def test_recommendation_engine_uses_post_transformation_normality_guard():
    result = RecommendationEngine().execute(
        {
            "mode": "advanced_recommendation",
            "recommendation": "parametric",
            "force_parametric": False,
            "test_info": {
                "post_transformation": {
                    "residuals_normality": {"is_normal": False}
                },
                "pre_transformation": {
                    "residuals_normality": {"is_normal": True}
                },
            },
        }
    )

    assert isinstance(result, StatisticalResult)
    updates = dict(result.metadata or {})
    assert updates.get("effective_recommendation") == "non_parametric"


def test_recommendation_engine_preserves_recommendation_when_normal():
    result = RecommendationEngine().execute(
        {
            "mode": "advanced_recommendation",
            "recommendation": "parametric",
            "force_parametric": False,
            "test_info": {
                "post_transformation": {
                    "residuals_normality": {"is_normal": True}
                }
            },
        }
    )

    assert isinstance(result, StatisticalResult)
    updates = dict(result.metadata or {})
    assert updates.get("effective_recommendation") == "parametric"


def test_extraction_engine_builds_rm_groups_and_original_samples():
    import pandas as pd

    df = pd.DataFrame(
        {
            "time": ["t1", "t1", "t1", "t1", "t1", "t2", "t2", "t2", "t2", "t2"],
            "score": [1.0, 2.0, 1.5, 2.5, 3.0, 1.5, 2.5, 2.0, 3.0, 3.5],
        }
    )

    result = ExtractionEngine().execute(
        {
            "mode": "advanced_group_extraction",
            "df": df,
            "test": "repeated_measures_anova",
            "dv": "score",
            "within": ["time"],
        }
    )

    assert isinstance(result, StatisticalResult)
    updates = dict(result.metadata or {})
    assert updates.get("error") is None
    assert updates.get("groups") == ["t1", "t2"]
    samples = updates.get("samples", {})
    originals = updates.get("original_samples", {})
    assert samples.get("t1") == [1.0, 2.0, 1.5, 2.5, 3.0]
    assert samples.get("t2") == [1.5, 2.5, 2.0, 3.0, 3.5]
    assert originals == samples
    assert originals.get("t1") is not samples.get("t1")


def test_extraction_engine_returns_rm_balance_error_for_unbalanced_data():
    import pandas as pd

    df = pd.DataFrame(
        {
            "time": ["t1", "t1", "t2"],
            "score": [1.0, 2.0, 1.5],
        }
    )

    result = ExtractionEngine().execute(
        {
            "mode": "advanced_group_extraction",
            "df": df,
            "test": "repeated_measures_anova",
            "dv": "score",
            "within": ["time"],
        }
    )

    assert isinstance(result, StatisticalResult)
    updates = dict(result.metadata or {})
    assert "error" in updates and updates.get("error")
    assert updates.get("test") == "Repeated Measures ANOVA (failed)"
