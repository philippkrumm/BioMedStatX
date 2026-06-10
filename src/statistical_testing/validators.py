from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

MIN_N_HARD = 5    # Absolute minimum before blocking or critical warnings
MIN_N_SMALL = 20  # Threshold for forcing robust defaults (exact methods, non-parametric)

class ValidationError(Exception):
    """Base class for statistical input validation failures."""


class SampleSizeError(ValidationError):
    """Raised when sample size is below method requirements."""


class MissingValuesError(ValidationError):
    """Raised when NaN values violate a method's input contract."""


class InfiniteValuesError(ValidationError):
    """Raised when Inf/-Inf values are detected in numeric inputs."""


class PairedDataError(ValidationError):
    """Raised when paired data assumptions are violated."""


class GroupValidationError(ValidationError):
    """Raised when group-level structural constraints are violated."""


class ModelDesignError(ValidationError):
    """Raised when model design parameters are invalid for a chosen test."""


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str


def _coerce_numeric_array(values: Sequence[float], *, label: str) -> np.ndarray:
    if values is None:
        raise ValidationError(f"{label}: no values provided.")
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{label}: values are not numeric.") from exc
    if array.ndim == 0:
        array = array.reshape(1)
    return array.ravel()


def validate_finite_values(
    data: Sequence[float],
    *,
    label: str = "data",
    allow_missing: bool = False,
) -> np.ndarray:
    array = _coerce_numeric_array(data, label=label)
    if np.isinf(array).any():
        raise InfiniteValuesError(f"{label}: contains +/-Inf values.")
    if not allow_missing and np.isnan(array).any():
        raise MissingValuesError(f"{label}: contains NaN values.")
    return array


def validate_minimum_n(
    data: Sequence[float],
    *,
    min_n: int = MIN_N_HARD,
    label: str = "data",
    allow_missing: bool = False,
) -> np.ndarray:
    import logging
    logger = logging.getLogger(__name__)

    array = validate_finite_values(data, label=label, allow_missing=allow_missing)
    valid_n = int(np.count_nonzero(~np.isnan(array)))
    
    if valid_n < min_n:
        raise SampleSizeError(
            f"{label}: sample size n={valid_n} is below absolute minimum n={min_n}."
        )
        
    return array


def validate_paired_data(
    group_a: Sequence[float],
    group_b: Sequence[float],
    *,
    group_a_label: str = "group_a",
    group_b_label: str = "group_b",
    min_n: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    arr_a = validate_minimum_n(group_a, min_n=min_n, label=group_a_label, allow_missing=False)
    arr_b = validate_minimum_n(group_b, min_n=min_n, label=group_b_label, allow_missing=False)
    if arr_a.size != arr_b.size:
        raise PairedDataError(
            "Paired tests require equal sample sizes: "
            f"n({group_a_label})={arr_a.size}, n({group_b_label})={arr_b.size}."
        )
    return arr_a, arr_b


def ensure_equal_group_sizes(
    samples: Mapping[str, Sequence[float]],
    groups: Iterable[str],
    *,
    min_n: int = 2,
) -> int:
    normalized_groups = [str(group) for group in groups]
    if not normalized_groups:
        raise GroupValidationError("No groups provided.")

    sizes: Dict[str, int] = {}
    for group in normalized_groups:
        if group not in samples:
            raise GroupValidationError(f"Group '{group}' not found in samples.")
        arr = validate_minimum_n(samples[group], min_n=min_n, label=group, allow_missing=False)
        sizes[group] = int(arr.size)

    unique_sizes = set(sizes.values())
    if len(unique_sizes) != 1:
        detail = ", ".join(f"{group}: {size}" for group, size in sizes.items())
        raise PairedDataError(
            "Dependent tests require equal sample sizes across groups. "
            f"Observed sizes: {detail}."
        )

    return next(iter(unique_sizes))


def validate_transformed_values(values: Sequence[float], *, transformation_name: str) -> np.ndarray:
    label = f"transformation '{transformation_name}'"
    return validate_finite_values(values, label=label, allow_missing=True)


def validate_group_count(groups: Iterable[str], *, min_groups: int = 2, label: str = "groups") -> List[str]:
    normalized_groups = [str(group) for group in groups]
    if len(normalized_groups) < min_groups:
        raise GroupValidationError(
            f"{label}: requires at least {min_groups} group(s), got {len(normalized_groups)}."
        )
    return normalized_groups


def validate_residuals_for_shapiro(residuals: Sequence[float] | Any, *, label: str = "residuals") -> np.ndarray:
    array = validate_finite_values(residuals, label=label, allow_missing=False)
    if array.size < 3:
        raise SampleSizeError(f"{label}: Shapiro-Wilk requires n>=3 residuals.")
    if np.unique(array).size <= 1:
        raise GroupValidationError(f"{label}: zero variance residuals; Shapiro-Wilk not applicable.")
    return array


def validate_levene_inputs(
    group_data: Sequence[Sequence[float]],
    *,
    min_groups: int = 2,
    min_n_per_group: int = 3,
    label: str = "levene",
) -> List[np.ndarray]:
    if len(group_data) < min_groups:
        raise GroupValidationError(
            f"{label}: requires at least {min_groups} groups, got {len(group_data)}."
        )
    arrays: List[np.ndarray] = []
    for index, values in enumerate(group_data):
        arr = validate_minimum_n(
            values,
            min_n=min_n_per_group,
            label=f"{label}_group_{index + 1}",
            allow_missing=False,
        )
        arrays.append(arr)
    return arrays


def validate_test_design(
    *,
    test_name: str,
    between: Sequence[str] | None = None,
    within: Sequence[str] | None = None,
    subject: str | None = None,
) -> None:
    between = list(between or [])
    within = list(within or [])

    if test_name == "mixed_anova":
        if not between or not within:
            raise ModelDesignError("Mixed ANOVA requires between and within factor.")
    elif test_name == "repeated_measures_anova":
        if not within:
            raise ModelDesignError("RM-ANOVA requires within factor.")
        if subject is None:
            raise ModelDesignError("RM-ANOVA requires subject column.")
    elif test_name == "two_way_anova":
        if len(between) != 2:
            raise ModelDesignError("Two-Way ANOVA requires two between factors.")
    else:
        raise ModelDesignError(f"Unknown/invalid test type: {test_name}")


def validate_balanced_design(samples: Mapping[str, Sequence[float]], groups: Iterable[str]) -> None:
    normalized_groups = validate_group_count(groups, min_groups=2, label="balanced_design_groups")
    sizes: List[int] = []
    for group in normalized_groups:
        if group not in samples:
            raise GroupValidationError(f"Group '{group}' not found in samples.")
        arr = validate_minimum_n(samples[group], min_n=1, label=group, allow_missing=False)
        sizes.append(int(arr.size))
    if len(set(sizes)) > 1:
        detail = {group: size for group, size in zip(normalized_groups, sizes)}
        raise GroupValidationError(
            f"Unbalanced design detected: {detail}. Effect-size estimates may be biased."
        )


def validate_samples(samples: Dict[str, Sequence[float]], min_group_size: int = 2) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    if not samples:
        return [ValidationIssue(code="empty_input", message="No groups provided.")]

    for group, values in samples.items():
        try:
            validate_minimum_n(values, min_n=min_group_size, label=str(group), allow_missing=False)
        except SampleSizeError as exc:
            issues.append(ValidationIssue(code="small_group", message=str(exc)))
        except MissingValuesError as exc:
            issues.append(ValidationIssue(code="missing_values", message=str(exc)))
        except InfiniteValuesError as exc:
            issues.append(ValidationIssue(code="infinite_values", message=str(exc)))
        except ValidationError as exc:
            issues.append(ValidationIssue(code="invalid_group", message=str(exc)))

    return issues
