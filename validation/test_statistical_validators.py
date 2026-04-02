import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from statistical_testing.validators import (
    GroupValidationError,
    InfiniteValuesError,
    MissingValuesError,
    ModelDesignError,
    PairedDataError,
    SampleSizeError,
    ensure_equal_group_sizes,
    validate_group_count,
    validate_levene_inputs,
    validate_minimum_n,
    validate_paired_data,
    validate_residuals_for_shapiro,
    validate_test_design,
    validate_transformed_values,
)


def test_validate_paired_data_requires_equal_lengths():
    with pytest.raises(PairedDataError):
        validate_paired_data([1.0, 2.0, 3.0], [1.0, 2.0], group_a_label="A", group_b_label="B")


def test_validate_paired_data_rejects_missing_values():
    with pytest.raises(MissingValuesError):
        validate_paired_data([1.0, float("nan")], [1.2, 1.3], group_a_label="A", group_b_label="B")


def test_validate_minimum_n_enforces_threshold():
    with pytest.raises(SampleSizeError):
        validate_minimum_n([1.0, 2.0], min_n=3, label="group_A")


def test_ensure_equal_group_sizes_returns_common_n():
    n = ensure_equal_group_sizes({"A": [1.0, 2.0], "B": [2.0, 3.0]}, ["A", "B"], min_n=2)
    assert n == 2


def test_validate_transformed_values_rejects_inf():
    with pytest.raises(InfiniteValuesError) as exc_info:
        validate_transformed_values([1.0, float("inf")], transformation_name="boxcox")
    assert "Inf" in str(exc_info.value)


def test_validate_group_count_requires_minimum():
    with pytest.raises(GroupValidationError):
        validate_group_count(["A"], min_groups=2)


def test_validate_residuals_for_shapiro_requires_variation_and_n():
    with pytest.raises(SampleSizeError):
        validate_residuals_for_shapiro([1.0, 2.0], label="res")

    with pytest.raises(GroupValidationError):
        validate_residuals_for_shapiro([1.0, 1.0, 1.0], label="res")


def test_validate_levene_inputs_enforces_group_requirements():
    with pytest.raises(SampleSizeError):
        validate_levene_inputs([[1.0, 2.0], [3.0, 4.0]], min_n_per_group=3)


def test_validate_test_design_rejects_invalid_model_spec():
    with pytest.raises(ModelDesignError):
        validate_test_design(test_name="mixed_anova", between=["A"], within=None, subject="id")
