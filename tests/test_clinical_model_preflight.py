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
