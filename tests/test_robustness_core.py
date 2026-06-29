"""Core-level robustness suite.

Replays the edge-case catalog through the production guard composition
(validate_samples_for_test -> block, else perform_statistical_test) — the exact
path analysis_core uses, minus file IO. Every case must degrade gracefully: no
raised exception, contract satisfied.
"""
import pytest

from analysis.statisticaltester import StatisticalTester
from statistical_testing.validators import validate_samples_for_test

from tests.robustness.contract import assert_graceful
from tests.robustness.edge_cases import CATALOG, coerce_samples


def _run_guarded(samples, dependent):
    """Mirror analysis_core: gate first, then the test. Numeric coercion is an
    upstream responsibility, so it is applied here as the pipeline does."""
    coerced = coerce_samples(samples)
    groups = list(samples.keys())
    report = validate_samples_for_test(coerced, groups, dependent=dependent)
    if report.blocking_issue is not None:
        return StatisticalTester.make_blocked_result(
            report.blocking_issue.message,
            code=report.blocking_issue.code,
            details={"groups": groups},
            warnings=report.warnings,
        )
    return StatisticalTester.perform_statistical_test(
        groups, coerced, coerced,
        dependent=dependent, test_recommendation="parametric", test_info=None,
    )


@pytest.mark.parametrize("ec", CATALOG, ids=lambda e: e.name)
def test_core_edge_case_is_graceful(ec):
    if ec.test_type != "simple":
        pytest.skip(f"Core suite only supports simple tests, got {ec.test_type}")
    try:
        result = _run_guarded(ec.samples, ec.dependent)
    except Exception as exc:  # pragma: no cover - failure path
        pytest.fail(f"{ec.name}: pipeline raised instead of degrading: {exc!r}")

    assert_graceful(result, ec.expectation)

    if ec.expectation == "blocked" and ec.expected_code:
        assert result.get("block_code") == ec.expected_code, (
            f"{ec.name}: expected block_code {ec.expected_code}, got {result.get('block_code')}"
        )
