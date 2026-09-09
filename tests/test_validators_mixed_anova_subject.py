"""Guards the fix for the missing subject-column check on the mixed_anova branch
of validate_test_design, found in the Help Hub content audit. Without a subject,
pg.mixed_anova(subject=...) fails with a raw pingouin exception instead of the
app's own clean ModelDesignError.
"""
import pytest

from statistical_testing.validators import ModelDesignError, validate_test_design


def test_mixed_anova_without_subject_raises_model_design_error():
    with pytest.raises(ModelDesignError, match="subject"):
        validate_test_design(
            test_name="mixed_anova",
            between=["Group"],
            within=["Time"],
            subject=None,
        )


def test_mixed_anova_with_subject_does_not_raise():
    validate_test_design(
        test_name="mixed_anova",
        between=["Group"],
        within=["Time"],
        subject="SubjectID",
    )


def test_repeated_measures_anova_subject_check_unchanged():
    # Regression guard: this task must not touch the sibling RM branch.
    with pytest.raises(ModelDesignError, match="subject"):
        validate_test_design(
            test_name="repeated_measures_anova",
            within=["Time"],
            subject=None,
        )
