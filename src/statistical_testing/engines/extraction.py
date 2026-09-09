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
        subject = payload.get("subject")

        if df is None or dv is None or test is None:
            return StatisticalResult(
                test_name="extraction_failed",
                statistic_value=None,
                p_value=None,
                metadata={"error": "df, test and dv are required."},
            )

        samples: dict[str, list[float]] = {}
        # Who each extracted value belongs to, in the same order, filled in the
        # same loop. Kept here rather than derived later because a second
        # extraction is a second chance to disagree: the raw table prints value
        # and subject side by side, and the two used to come from different
        # frames -- one of which had been replicate-averaged -- so every printed
        # row named the wrong subject. Only within-subject designs have an
        # answer; the rest leave this empty and the table shows no Subject
        # column, which is correct for them.
        subjects: dict[str, list[str]] = {}
        groups: list[str] = []
        df_original = df.copy()

        def _labels(subset):
            if not subject or subject not in subset.columns:
                return None
            return [str(value) for value in subset[subject].tolist()]

        try:
            if test == "mixed_anova":
                b_factor, w_factor = between[0], within[0]
                for b_val in df[b_factor].unique():
                    for w_val in df[w_factor].unique():
                        group_label = f"{b_factor}={b_val}, {w_factor}={w_val}"
                        subset = df[(df[b_factor] == b_val) & (df[w_factor] == w_val)]
                        samples[group_label] = subset[dv].tolist()
                        labels = _labels(subset)
                        if labels is not None:
                            subjects[group_label] = labels
                groups = list(samples.keys())

            elif test == "repeated_measures_anova":
                w_factor = within[0]
                for lvl in df[w_factor].unique():
                    subset = df[df[w_factor] == lvl]
                    samples[lvl] = subset[dv].tolist()
                    labels = _labels(subset)
                    if labels is not None:
                        subjects[lvl] = labels
                groups = list(samples.keys())
                ensure_equal_group_sizes(samples, groups, min_n=MIN_N_BLOCK)

            elif test == "two_way_anova" or test == "two_way_ancova":
                fA, fB = between
                for a_val in df[fA].unique():
                    for b_val in df[fB].unique():
                        group_label = f"{fA}={a_val}, {fB}={b_val}"
                        subset = df[(df[fA] == a_val) & (df[fB] == b_val)]
                        samples[group_label] = subset[dv].tolist()
                groups = list(samples.keys())

            elif test == "ancova" or test == "logistic_regression":
                if between:
                    fA = between[0]
                    for a_val in df[fA].unique():
                        group_label = f"{fA}={a_val}"
                        subset = df[df[fA] == a_val]
                        samples[group_label] = subset[dv].tolist()
                groups = list(samples.keys())

            elif test == "lmm":
                # For LMM, the groups for visualization/assumptions are typically the combinations of all fixed categorical effects.
                fixed = (between or []) + (within or [])
                if not fixed:
                    samples["all"] = df[dv].tolist()
                elif len(fixed) == 1:
                    fA = fixed[0]
                    for a_val in df[fA].unique():
                        group_label = f"{fA}={a_val}"
                        subset = df[df[fA] == a_val]
                        samples[group_label] = subset[dv].tolist()
                else:
                    # Just use the first two for extraction grouping
                    fA, fB = fixed[0], fixed[1]
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
                "subjects": subjects or None,
            },
        )