from typing import Any, Mapping

import numpy as np
from scipy import stats

from ..models import StatisticalResult
from ..validators import ValidationError, validate_transformed_values, bounded_boxcox_lambda


class TransformationEngine:
    """Applies advanced-test transformations and rebuilds grouped samples."""

    def execute(self, data: Mapping[str, Any]) -> StatisticalResult:
        payload = dict(data or {})
        mode = str(payload.get("mode") or "")

        if mode != "advanced_transformation":
            return StatisticalResult(
                test_name="transformation_failed",
                statistic_value=None,
                p_value=None,
                metadata={"error": f"Unsupported transformation mode '{mode}'."},
            )

        df = payload.get("df")
        dv = payload.get("dv")
        test = payload.get("test")
        between = payload.get("between")
        within = payload.get("within")
        transformed_samples = payload.get("transformed_samples")
        test_info = payload.get("test_info")

        if df is None or dv is None:
            return StatisticalResult(
                test_name="transformation_failed",
                statistic_value=None,
                p_value=None,
                metadata={"error": "df and dv are required for transformation."},
            )

        df_transformed = df.copy()
        transformation_type = None
        if isinstance(test_info, Mapping) and test_info.get("transformation") is not None:
            transformation_type = test_info.get("transformation")

        updates: dict[str, Any] = {
            "df_transformed": df_transformed,
            "transformed_samples": transformed_samples,
            "transformation_type": transformation_type,
            "error": None,
        }

        if transformation_type and transformation_type not in ["none", "None", "Keine"]:
            try:
                if transformation_type == "log10":
                    min_val = df[dv].min()
                    shift = -min_val + 1 if min_val <= 0 else 0
                    df_transformed[dv] = np.log10(df[dv] + shift)
                elif transformation_type == "boxcox":
                    min_val = df[dv].dropna().min()
                    shift = -min_val + 1 if min_val <= 0 else 0
                    shifted = df[dv] + shift
                    lambda_val = test_info.get("boxcox_lambda") if isinstance(test_info, Mapping) else None
                    if lambda_val is None:
                        valid_bc = shifted.dropna()
                        valid_bc = valid_bc[valid_bc > 0].values
                        # Guard against optimizer divergence: reject out-of-bounds
                        # lambda and fall back to log (lambda=0) rather than
                        # potentiating the variance. See bounded_boxcox_lambda.
                        lambda_val, reverted = bounded_boxcox_lambda(valid_bc)
                        if reverted:
                            updates["transform_warning"] = (
                                "Maximum-likelihood estimation of the Box-Cox parameter "
                                "lambda diverged (out of bounds). Fell back to a log "
                                "transformation (lambda = 0)."
                            )
                        if isinstance(test_info, Mapping):
                            test_info["boxcox_lambda"] = lambda_val
                        updates["boxcox_lambda"] = lambda_val
                    # Apply BoxCox element-wise, preserving NaN
                    import pandas as _pd
                    mask = shifted.notna() & (shifted > 0)
                    safe = shifted.where(mask, 1.0).values
                    bc_vals = stats.boxcox(safe, lambda_val)
                    df_transformed[dv] = _pd.Series(
                        np.where(mask.values, bc_vals, np.nan), index=df.index
                    )
                elif transformation_type == "arcsin_sqrt":
                    min_val = df[dv].min()
                    max_val = df[dv].max()
                    if min_val < 0 or max_val > 1:
                        df_transformed[dv] = (df[dv] - min_val) / (max_val - min_val)
                    df_transformed[dv] = np.arcsin(np.sqrt(df_transformed[dv]))

                transformed_values = validate_transformed_values(
                    df_transformed[dv].values,
                    transformation_name=str(transformation_type),
                )
                # Keep the coercion side-effect parity from validator output.
                df_transformed[dv] = transformed_values

                updates["transformed_samples"] = self._extract_transformed_samples(
                    df_transformed=df_transformed,
                    dv=str(dv),
                    test=str(test),
                    between=between,
                    within=within,
                )
            except ValidationError as exc:
                updates["error"] = f"ERROR: {exc}. Analysis aborted."
                updates["test"] = transformation_type
            except Exception as exc:
                updates["error"] = f"ERROR: {exc}. Analysis aborted."
                updates["test"] = transformation_type

        return StatisticalResult(
            test_name="transformation_completed",
            statistic_value=None,
            p_value=None,
            metadata=updates,
        )

    @staticmethod
    def _extract_transformed_samples(
        *,
        df_transformed,
        dv: str,
        test: str,
        between,
        within,
    ) -> dict[str, list[float]]:
        samples_for_transform: dict[str, list[float]] = {}

        if test == "mixed_anova":
            b_factor, w_factor = between[0], within[0]
            for b_val in df_transformed[b_factor].unique():
                for w_val in df_transformed[w_factor].unique():
                    group_label = f"{b_factor}={b_val}, {w_factor}={w_val}"
                    subset = df_transformed[(df_transformed[b_factor] == b_val) & (df_transformed[w_factor] == w_val)]
                    samples_for_transform[group_label] = subset[dv].tolist()
        elif test == "repeated_measures_anova":
            w_factor = within[0]
            for lvl in df_transformed[w_factor].unique():
                samples_for_transform[lvl] = df_transformed[df_transformed[w_factor] == lvl][dv].tolist()
        elif test == "two_way_anova":
            fA, fB = between
            for a_val in df_transformed[fA].unique():
                for b_val in df_transformed[fB].unique():
                    group_label = f"{fA}={a_val}, {fB}={b_val}"
                    subset = df_transformed[(df_transformed[fA] == a_val) & (df_transformed[fB] == b_val)]
                    samples_for_transform[group_label] = subset[dv].tolist()

        return samples_for_transform