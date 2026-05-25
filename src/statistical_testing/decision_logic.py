from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class DecisionInput:
    group_count: int
    dependent: bool
    residuals_normal: bool
    equal_variance: bool


@dataclass(frozen=True)
class AssumptionState:
    residuals_normal: bool
    equal_variance: bool


def extract_assumption_state(test_info: Mapping[str, Any] | None) -> AssumptionState:
    """Extract normalized assumption flags from legacy/new test_info structures."""
    test_info = dict(test_info or {})

    use_post = bool(test_info.get("transformation"))
    if use_post:
        normality = test_info.get("post_transformation", {}).get("residuals_normality", {})
        variance = test_info.get("post_transformation", {}).get("variance", {})
    else:
        normality = test_info.get("pre_transformation", {}).get("residuals_normality", {})
        variance = test_info.get("pre_transformation", {}).get("variance", {})

    # Backward compatibility for older structures still referenced by legacy code.
    if not normality:
        normality = test_info.get("normality_tests", {}).get("model_residuals", {})
    if not variance:
        variance = test_info.get("variance_test", {})

    residuals_normal = bool(normality.get("is_normal", False))
    equal_variance = bool(variance.get("equal_variance", False))
    return AssumptionState(residuals_normal=residuals_normal, equal_variance=equal_variance)


def select_comparison_test(
    *,
    is_normal: bool,
    is_homoscedastic: bool,
    is_paired: bool,
    group_count: int,
) -> str:
    """Map assumption metadata to a concrete strategy key."""
    if group_count <= 1:
        return "descriptive_only"

    if group_count == 2:
        if is_paired:
            return "paired_ttest" if is_normal else "wilcoxon"
        if is_normal:
            return "welch_ttest"  # Unconditional default (A1 Fix)
        return "mann_whitney_u"

    if is_paired:
        return "repeated_measures_required"

    if is_normal:
        return "welch_anova"  # Unconditional default (A1 Fix)
    return "kruskal_wallis"


def choose_comparison_strategy(decision: DecisionInput) -> str:
    """Backwards-compatible entry point for strategy selection."""
    return select_comparison_test(
        is_normal=decision.residuals_normal,
        is_homoscedastic=decision.equal_variance,
        is_paired=decision.dependent,
        group_count=decision.group_count,
    )


def strategy_to_recommendation(strategy: str) -> str:
    """Map a concrete strategy key to the legacy recommendation family."""
    if strategy in {"mann_whitney_u", "wilcoxon", "kruskal_wallis"}:
        return "non_parametric"
    if strategy in {"welch_ttest", "welch_anova"}:
        return "welch"
    return "parametric"
