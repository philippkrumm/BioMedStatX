import itertools
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

MIN_N_BLOCK = 3   # Absolute minimum to run a test at all (variance/df need n>=3)
MIN_N_HARD = 5    # Small-sample warning threshold (n<5 => low power, interpret with caution)
MIN_N_SMALL = 20  # Threshold for forcing robust defaults (exact methods, non-parametric)

# Relative+absolute tolerance for constancy (zero-variance) detection. A static
# global epsilon is unusable across arbitrary biomed scales (proportions 0-1,
# fluorescence ~1e6, cell counts), so np.allclose-style rtol/atol is used.
VAR_RTOL = 1e-05
VAR_ATOL = 1e-08
# Flag a design as severely unbalanced when the largest group is this many times
# the smallest (warning only, not blocking).
IMBALANCE_RATIO = 10.0


def _max_safe_abs(n: int) -> float:
    """Safe per-element magnitude bound for a variance / sum-of-squares over n
    points. Variance sums n squared terms, so the safe bound is
    sqrt(float64_max / n), NOT sqrt(float64_max) — otherwise adding n near-limit
    squares overflows the global float64 max (~1.79e308)."""
    return float(np.sqrt(np.finfo(np.float64).max / max(int(n), 1)))

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


class DataQualityError(ValidationError):
    """Raised when a sample fails a pre-flight data-quality check (zero variance,
    overflow risk, constant paired differences, etc.)."""


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


def bounded_boxcox_lambda(data, bounds: tuple = (-3.0, 3.0)) -> tuple:
    """Maximum-likelihood Box-Cox lambda with a divergence guard.

    On extremely right-skewed assay data (e.g. luciferase signal-to-background),
    ``scipy.stats.boxcox_normmax`` can diverge to |lambda| >> 1. Forcing such a
    lambda potentiates the variance instead of stabilizing it, destroying the
    homoscedasticity assumption of downstream parametric tests and producing
    transformed values on the order of 1e16.

    If the ML estimate falls outside ``bounds`` (default the strict interval
    [-3, 3]) — or cannot be computed — the estimate is REJECTED and we
    hard-fall-back to lambda = 0 (natural log), the established standard for
    such data. Clamping to the boundary is methodologically invalid and is
    never done.

    Args:
        data: 1-D array of strictly positive values (non-positive/NaN dropped).
        bounds: (low, high) validity interval for the ML lambda.

    Returns:
        (lambda, reverted_to_log): ``reverted_to_log`` is True when the ML
        estimate was rejected and lambda was forced to 0.0.
    """
    from scipy.stats import boxcox_normmax

    arr = np.asarray(data, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size < 3:
        return 0.0, True
    try:
        lam = float(boxcox_normmax(arr))
    except Exception:
        return 0.0, True
    if not np.isfinite(lam) or lam < bounds[0] or lam > bounds[1]:
        return 0.0, True
    return lam, False


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
    elif test_name in ("ancova", "two_way_ancova", "lmm", "logistic_regression"):
        pass # The specialized model classes will perform their own specific validation
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


# ---------------------------------------------------------------------------
# Pre-flight sample quality gate (central chokepoint)
# ---------------------------------------------------------------------------

# Block code -> human-readable message template. Wording lives here so the UI
# and the HTML report stay consistent with a single source of truth.
BLOCK_MESSAGES: Dict[str, str] = {
    "INF_VALUES": "Group '{group}' contains infinite (+/-Inf) values — cannot run a test.",
    "EMPTY_GROUP": "Group '{group}' has no usable numeric values after removing missing data.",
    "N_BELOW_MIN": "Group '{group}' has only n={n} usable value(s); a test needs at least n={min_n}.",
    "NUM_OVERFLOW": "Group '{group}' has values too large for a numerically stable variance / sum-of-squares calculation.",
    "VAR_ZERO": "Group '{group}' has (near) zero variance — all values are effectively identical, so a test is undefined.",
    "TOO_FEW_GROUPS": "At least 2 groups with usable data are required, found {n_groups}.",
    "PAIRED_SIZE_MISMATCH": "Paired/repeated-measures analysis requires equal group sizes; observed: {detail}.",
    "VAR_DIFF_ZERO": "Paired differences between '{a}' and '{b}' are (near) constant — the test statistic is undefined (singular covariance).",
}


@dataclass(frozen=True)
class SampleQualityReport:
    """Result of the pre-flight gate. ``blocking_issue`` is the first hard-stop
    reason (or None when the data may proceed). ``warnings`` are non-blocking
    notes (small samples, severe imbalance)."""
    blocking_issue: "ValidationIssue | None"
    warnings: List[str]

    @property
    def ok(self) -> bool:
        return self.blocking_issue is None


def _coerce_quality_array(values: Sequence[Any]) -> np.ndarray:
    """Coerce arbitrary cell values to a float array. Whitespace-only strings and
    non-numeric text become NaN (never raise); +/-Inf is preserved so the Inf
    guard can see it before NaN-dropping."""
    series = pd.to_numeric(pd.Series(list(values), dtype="object"), errors="coerce")
    return series.to_numpy(dtype=float)


def validate_samples_for_test(
    samples: Mapping[str, Sequence[Any]],
    groups: Iterable[str],
    *,
    dependent: bool = False,
    min_n_block: int = MIN_N_BLOCK,
) -> SampleQualityReport:
    """Central pre-flight gate. Returns the first blocking issue (if any) plus
    soft warnings, WITHOUT raising. Run before any statistical test so that
    pathological inputs become a clean labeled block instead of a crash or a
    silently-wrong result (e.g. zero-variance Welch -> p=1.0)."""
    normalized = [str(group) for group in groups]
    warnings: List[str] = []
    valid_by_group: Dict[str, np.ndarray] = {}

    def issue(code: str, **fmt) -> SampleQualityReport:
        return SampleQualityReport(
            blocking_issue=ValidationIssue(code=code, message=BLOCK_MESSAGES[code].format(**fmt)),
            warnings=warnings,
        )

    for group in normalized:
        raw = samples.get(group, [])
        arr = _coerce_quality_array(raw)

        if np.isinf(arr).any():
            return issue("INF_VALUES", group=group)

        valid = arr[~np.isnan(arr)]
        if valid.size == 0:
            return issue("EMPTY_GROUP", group=group)
        if valid.size < min_n_block:
            return issue("N_BELOW_MIN", group=group, n=int(valid.size), min_n=min_n_block)
        if float(np.max(np.abs(valid))) >= _max_safe_abs(valid.size):
            return issue("NUM_OVERFLOW", group=group)
        if np.allclose(valid, valid[0], rtol=VAR_RTOL, atol=VAR_ATOL):
            return issue("VAR_ZERO", group=group)

        valid_by_group[group] = valid
        if valid.size < MIN_N_HARD:
            warnings.append(
                f"Group '{group}' is a small sample (n={int(valid.size)} < {MIN_N_HARD}); "
                "statistical power is low — interpret with caution."
            )

    usable = list(valid_by_group)
    if len(usable) < 2:
        return issue("TOO_FEW_GROUPS", n_groups=len(usable))

    sizes = {group: int(valid_by_group[group].size) for group in usable}
    if max(sizes.values()) >= IMBALANCE_RATIO * min(sizes.values()):
        warnings.append(
            "Severely unbalanced group sizes "
            f"({', '.join(f'{g}: {n}' for g, n in sizes.items())}); "
            "power and assumption checks may be affected."
        )

    if dependent:
        if len(set(sizes.values())) != 1:
            detail = ", ".join(f"{g}: {n}" for g, n in sizes.items())
            return issue("PAIRED_SIZE_MISMATCH", detail=detail)
        # RM-ANOVA (k>=3) needs an invertible covariance matrix: a constant
        # difference between ANY pair makes it singular, so check all pairs.
        for a, b in itertools.combinations(usable, 2):
            diff = valid_by_group[a] - valid_by_group[b]
            if np.allclose(diff, diff[0], rtol=VAR_RTOL, atol=VAR_ATOL):
                return issue("VAR_DIFF_ZERO", a=a, b=b)

    return SampleQualityReport(blocking_issue=None, warnings=warnings)


def validate_outcome(values, *, label="outcome", min_n_block=MIN_N_BLOCK):
    """Single-vector degeneracy gate for regression-style models (ANCOVA, LMM,
    logistic/linear/beta regression, correlation) whose data shape doesn't fit
    the group-based gate. Returns the first blocking ValidationIssue or None.
    Catches a constant / empty / too-small / Inf / overflow outcome or predictor
    that would make the fit meaningless or singular."""
    arr = _coerce_quality_array(values)
    if np.isinf(arr).any():
        return ValidationIssue(code="INF_VALUES", message=BLOCK_MESSAGES["INF_VALUES"].format(group=label))
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return ValidationIssue(code="EMPTY_GROUP", message=BLOCK_MESSAGES["EMPTY_GROUP"].format(group=label))
    if valid.size < min_n_block:
        return ValidationIssue(code="N_BELOW_MIN",
                               message=BLOCK_MESSAGES["N_BELOW_MIN"].format(group=label, n=int(valid.size), min_n=min_n_block))
    if float(np.max(np.abs(valid))) >= _max_safe_abs(valid.size):
        return ValidationIssue(code="NUM_OVERFLOW", message=BLOCK_MESSAGES["NUM_OVERFLOW"].format(group=label))
    if np.allclose(valid, valid[0], rtol=VAR_RTOL, atol=VAR_ATOL):
        return ValidationIssue(code="VAR_ZERO", message=BLOCK_MESSAGES["VAR_ZERO"].format(group=label))
    return None
