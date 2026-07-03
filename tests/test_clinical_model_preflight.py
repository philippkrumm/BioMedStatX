"""Guards the pre-flight ModelDesignError checks added to ANCOVAModel,
LinearMixedModel, and LogisticRegressionModel.fit(). Before this fix, a
missing structural field (empty between_factors/covariates/fixed_effects/
predictors, or a missing subject column) either crashed later inside
as_results_dict() with an incidental IndexError/KeyError, or (for LMM fixed
effects) silently degraded to a meaningless intercept-only model instead of
being rejected outright. See docs/superpowers/specs/2026-07-03-clinical-model-preflight-validation-design.md.
"""
import pandas as pd
import pytest

from analysis.clinical_models import ANCOVAModel, LinearMixedModel, LogisticRegressionModel
from statistical_testing.validators import ModelDesignError


def _ancova_df():
    return pd.DataFrame({
        "Y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "Group": ["A", "A", "A", "B", "B", "B"],
        "Cov": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    })


def test_ancova_without_between_factors_raises_model_design_error():
    with pytest.raises(ModelDesignError, match="between-subjects factor"):
        ANCOVAModel().fit(_ancova_df(), dv="Y", between_factors=[], covariates=["Cov"])


def test_ancova_without_covariates_raises_model_design_error():
    with pytest.raises(ModelDesignError, match="at least one covariate"):
        ANCOVAModel().fit(_ancova_df(), dv="Y", between_factors=["Group"], covariates=[])


def _lmm_df():
    return pd.DataFrame({
        "Y": [1.0, 2.0, 1.5, 2.5, 1.2, 2.2, 1.8, 2.8],
        "Time": ["T1", "T2", "T1", "T2", "T1", "T2", "T1", "T2"],
        "Subject": ["S1", "S1", "S2", "S2", "S3", "S3", "S4", "S4"],
    })


def test_lmm_without_fixed_effects_raises_model_design_error():
    with pytest.raises(ModelDesignError, match="fixed effect"):
        LinearMixedModel().fit(_lmm_df(), dv="Y", fixed_effects=[], random_intercept="Subject")


def test_lmm_without_random_intercept_raises_model_design_error():
    with pytest.raises(ModelDesignError, match="subject/ID column"):
        LinearMixedModel().fit(_lmm_df(), dv="Y", fixed_effects=["Time"], random_intercept=None)
