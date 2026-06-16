from typing import Any, Mapping

from ..models import StatisticalResult
from ..validators import ValidationError, ensure_equal_group_sizes, MIN_N_BLOCK


class ExtractionEngine:
    """Extracts grouped samples for advanced tests and validates RM balance."""

    def execute(self, data: Mapping[str, Any]) -> StatisticalResult:
        payload = dict(data or {})
        mode = str(payload.get("mode") or "")

        if mode != "advanced_group_extraction":
            return StatisticalResult(
                test_name="extraction_failed",
                statistic_value=None,
                p_value=None,
                metadata={"error": f"Unsupported extraction mode '{mode}'."},
            )

        df = payload.get("df")
        test = payload.get("test")
        dv = payload.get("dv")
        between = payload.get("between")
        within = payload.get("within")

        if df is None or dv is None or test is None:
            return StatisticalResult(
                test_name="extraction_failed",
                statistic_value=None,
                p_value=None,
                metadata={"error": "df, test and dv are required."},
            )

        samples: dict[str, list[float]] = {}
        groups: list[str] = []
        df_original = df.copy()

        try:
            if test == "mixed_anova":
                b_factor, w_factor = between[0], within[0]
                for b_val in df[b_factor].unique():
                    for w_val in df[w_factor].unique():
                        group_label = f"{b_factor}={b_val}, {w_factor}={w_val}"
                        subset = df[(df[b_factor] == b_val) & (df[w_factor] == w_val)]
                        samples[group_label] = subset[dv].tolist()
                groups = list(samples.keys())

            elif test == "repeated_measures_anova":
                w_factor = within[0]
                for lvl in df[w_factor].unique():
                    samples[lvl] = df[df[w_factor] == lvl][dv].tolist()
                groups = list(samples.keys())
                ensure_equal_group_sizes(samples, groups, min_n=MIN_N_BLOCK)

            elif test == "two_way_anova":
                fA, fB = between
                for a_val in df[fA].unique():
                    for b_val in df[fB].unique():
                        group_label = f"{fA}={a_val}, {fB}={b_val}"
                        subset = df[(df[fA] == a_val) & (df[fB] == b_val)]
                        samples[group_label] = subset[dv].tolist()
                groups = list(samples.keys())

            else:
                return StatisticalResult(
                    test_name="extraction_failed",
                    statistic_value=None,
                    p_value=None,
                    metadata={
                        "error": f"Invalid test type: {test}",
                        "test": f"{test} (failed)",
                    },
                )
        except ValidationError as exc:
            return StatisticalResult(
                test_name="extraction_failed",
                statistic_value=None,
                p_value=None,
                metadata={"error": str(exc), "test": "Repeated Measures ANOVA (failed)"},
            )

        original_samples = {k: v.copy() for k, v in samples.items()}

        return StatisticalResult(
            test_name="extraction_completed",
            statistic_value=None,
            p_value=None,
            metadata={
                "error": None,
                "samples": samples,
                "groups": groups,
                "df_original": df_original,
                "original_samples": original_samples,
            },
        )