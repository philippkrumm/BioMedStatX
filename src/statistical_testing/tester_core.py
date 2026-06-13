from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from .decision_logic import DecisionInput, choose_comparison_strategy
from .models import StatisticalResult
from .validators import ValidationError, validate_samples


class TestEngine(Protocol):
    def execute(self, data: Mapping[str, Any]) -> StatisticalResult:
        ...


@dataclass
class StatisticalTesterCore:
    """Orchestrator that validates inputs, selects strategy, and delegates execution."""

    engines: Mapping[str, TestEngine]

    def run(self, data: Mapping[str, Any]) -> StatisticalResult:
        samples = data.get("samples") if isinstance(data, Mapping) else None
        if isinstance(samples, dict):
            issues = validate_samples(samples, min_group_size=2)
            if issues:
                raise ValidationError("; ".join(issue.message for issue in issues))

        decision = DecisionInput(
            group_count=int(data.get("group_count", 0)),
            dependent=bool(data.get("dependent", False)),
            residuals_normal=bool(data.get("residuals_normal", False)),
            equal_variance=bool(data.get("equal_variance", False)),
        )
        strategy = choose_comparison_strategy(decision)
        engine = self.engines.get(strategy)
        if engine is None:
            raise ValidationError(f"No engine configured for strategy '{strategy}'.")
        return engine.execute(data)
