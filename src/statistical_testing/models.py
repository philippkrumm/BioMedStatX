from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional


@dataclass(frozen=True)
class StatisticalResult:
    """Canonical test result DTO for strategy-based statistical engines."""

    test_name: str
    statistic_value: Optional[float]
    p_value: Optional[float]
    degrees_of_freedom_1: Optional[float] = None
    degrees_of_freedom_2: Optional[float] = None
    effect_size: Optional[float] = None
    effect_size_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def from_legacy_dict(cls, payload: Mapping[str, Any]) -> "StatisticalResult":
        payload = dict(payload or {})
        return cls(
            test_name=str(payload.get("final_test_label") or payload.get("test") or "Unknown test"),
            statistic_value=cls._coerce_float(payload.get("statistic")),
            p_value=cls._coerce_float(payload.get("p_value")),
            degrees_of_freedom_1=cls._coerce_float(payload.get("df1")),
            degrees_of_freedom_2=cls._coerce_float(payload.get("df2")),
            effect_size=cls._coerce_float(payload.get("effect_size")),
            effect_size_type=(str(payload.get("effect_size_type")) if payload.get("effect_size_type") is not None else None),
            metadata={
                key: value
                for key, value in payload.items()
                if key not in {
                    "test",
                    "final_test_label",
                    "tested_against",
                    "statistic",
                    "p_value",
                    "df1",
                    "df2",
                    "effect_size",
                    "effect_size_type",
                }
            },
        )

    def to_legacy_dict(self) -> Dict[str, Any]:
        legacy = dict(self.metadata)
        legacy["test"] = self.test_name
        legacy["final_test_label"] = self.test_name
        legacy.setdefault("tested_against", self.test_name)
        legacy["statistic"] = self.statistic_value
        legacy["p_value"] = self.p_value
        legacy["df1"] = self.degrees_of_freedom_1
        legacy["df2"] = self.degrees_of_freedom_2
        legacy["effect_size"] = self.effect_size
        legacy["effect_size_type"] = self.effect_size_type
        return legacy
