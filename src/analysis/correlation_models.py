"""
Correlation and Regression Models for BioMedStatX.

Provides Pearson/Spearman correlation, OLS linear regression with diagnostics,
and an exploratory correlation matrix with multiple-testing correction.

All model classes follow the pattern established in clinical_models.py:
    model = SomeModel()
    model.fit(df, ...)
    results = model.as_results_dict()
"""

import re
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _sanitize_columns(df, columns):
    """Rename columns with special characters so patsy / statsmodels can parse them.

    Returns a dict mapping original name -> sanitized name.
    The DataFrame is renamed in-place (same pattern as clinical_models.py).
    """
    mapping = {}
    for col in columns:
        safe = re.sub(r'[^A-Za-z0-9_]', '_', str(col))
        if safe != col:
            base = safe
            i = 2
            while safe in df.columns and safe != col:
                safe = f"{base}_{i}"
                i += 1
            df.rename(columns={col: safe}, inplace=True)
        mapping[col] = safe
    return mapping


def _is_continuous(df, col, threshold=10):
    """Return True if col is numeric with more than `threshold` unique values.

    Used by the auto-pilot to distinguish continuous predictors from
    categorical group columns.
    """
    if col not in df.columns:
        return False
    if not pd.api.types.is_numeric_dtype(df[col]):
        return False
    return df[col].nunique() > threshold


def _fisher_z_ci(r, n, alpha=0.05):
    """95 % CI for a Pearson or Spearman r via Fisher z-transform."""
    if n < 4:
        return (None, None)
    try:
        r_clamped = float(np.clip(r, -0.9999, 0.9999))
        z = np.arctanh(r_clamped)
        se = 1.0 / np.sqrt(n - 3)
        z_crit = scipy_stats.norm.ppf(1 - alpha / 2)
        lo = float(np.tanh(z - z_crit * se))
        hi = float(np.tanh(z + z_crit * se))
        return (lo, hi)
    except Exception:
        return (None, None)


_VALID_TRANSFORMS = ('none', 'log10', 'log10(x+1)', 'sqrt', 'boxcox')


def _apply_transform(vals: np.ndarray, name: str):
    """Apply a named transformation to a 1-D float array.

    Explicitly handles non-positive values by setting them to np.nan, 
    ensuring they are dropped via listwise deletion later in the pipeline.
    There is no automatic data shifting (c=0.0 always).

    Args:
        vals: 1-D numpy array of finite floats.
        name: one of 'none', 'log10', 'log10(x+1)', 'sqrt', 'boxcox'.

    Returns:
        Tuple (transformed_array, boxcox_lambda, shift).
        ``boxcox_lambda`` is a float when name == 'boxcox'; None otherwise.
        ``shift`` is always 0.0 (kept for API compatibility).

    Raises:
        ValueError: if name is not a recognised transformation.
    """
    if name not in _VALID_TRANSFORMS:
        raise ValueError(
            f"Unknown transformation '{name}'. "
            f"Choose from: {', '.join(_VALID_TRANSFORMS)}."
        )
    if name == 'none':
        return vals.copy(), None, 0.0

    v = vals.astype(float).copy()

    if name == 'log10':
        v[v <= 0] = np.nan
        return np.log10(v), None, 0.0

    if name == 'log10(x+1)':
        v[v <= -1] = np.nan
        # Use log1p for numerical stability as requested
        return np.log1p(v) / np.log(10), None, 0.0

    if name == 'sqrt':
        v[v < 0] = np.nan
        return np.sqrt(v), None, 0.0

    if name == 'boxcox':
        from scipy.stats import boxcox as _scipy_boxcox
        from statistical_testing.validators import bounded_boxcox_lambda
        v[v <= 0] = np.nan
        # Only optimize over valid (positive) values
        valid_mask = ~np.isnan(v)
        if not np.any(valid_mask):
            return v, None, 0.0

        # Reject a divergent (out-of-bounds) ML lambda and fall back to log
        # (lambda=0) — boxcox(x, 0) == ln(x). Never clamp to a boundary.
        lam, _reverted = bounded_boxcox_lambda(v[valid_mask])
        v[valid_mask] = _scipy_boxcox(v[valid_mask], lam)
        return v, float(lam), 0.0

def _optimize_boxcox_for_regression(y: np.ndarray, x_matrix: np.ndarray):
    """Optimize Box-Cox lambda by maximizing the profile log-likelihood of OLS residuals.
    
    Uses the geometric mean scaling approach:
    y_dot = geometric_mean(y)
    y_scaled_lambda = ( (y / y_dot)^lambda - 1 ) / lambda
    The optimal lambda minimizes the Residual Sum of Squares (RSS) of OLS(y_scaled_lambda ~ x_matrix).
    
    Args:
        y: 1-D array of outcome values (must be > 0)
        x_matrix: 2-D exogenous design matrix (including intercept)
        
    Returns:
        optimal_lambda (float)
    """
    from scipy.optimize import minimize_scalar
    from scipy.stats import gmean
    from scipy.special import boxcox as scipy_boxcox
    
    valid_mask = (y > 0) & ~np.isnan(y)
    if not np.any(valid_mask):
        return 1.0 # Fallback if no valid data
        
    y_valid = y[valid_mask]
    x_valid = x_matrix[valid_mask]
    
    y_dot = gmean(y_valid)
    
    def rss_for_lambda(lam):
        # Scale Y and apply Box-Cox
        y_scaled = y_valid / y_dot
        y_trans = scipy_boxcox(y_scaled, lam)
        
        # Fit OLS (np.linalg.lstsq is fastest)
        try:
            beta, residuals, rank, s = np.linalg.lstsq(x_valid, y_trans, rcond=None)
            if len(residuals) > 0:
                return residuals[0]
            else:
                # If exact fit (residuals sum to 0), compute manually
                y_hat = x_valid @ beta
                return np.sum((y_trans - y_hat)**2)
        except np.linalg.LinAlgError:
            return np.inf
            
    res = minimize_scalar(rss_for_lambda, bounds=(-2.0, 2.0), method='bounded')
    if res.success:
        return res.x
    return 1.0 # Fallback to no transformation (lambda=1)


# ---------------------------------------------------------------------------
# 1. CorrelationModel
# ---------------------------------------------------------------------------

class CorrelationModel:
    """Pearson or Spearman correlation with 95 % CI (Fisher z-transform).

    method='auto' applies Shapiro-Wilk to both variables and uses Pearson when
    both are normally distributed (p > alpha), otherwise Spearman.
    Pairwise deletion: only rows without NaN in x_col or y_col are used.
    """

    def __init__(self):
        self.r = None
        self.p = None
        self.ci = (None, None)
        self.n = None
        self._method_used = None
        self._x = None
        self._y = None
        self._x_label = None
        self._y_label = None
        self._points = []
        self._alpha = 0.05
        self._x_transform = 'none'
        self._y_transform = 'none'
        self._x_boxcox_lambda = None
        self._y_boxcox_lambda = None
        self._x_transform_shift = 0.0
        self._y_transform_shift = 0.0

    def fit(self, df, x_col, y_col, method='auto', alpha=0.05,
            x_transform='none', y_transform='none'):
        """Fit the correlation model.

        Args:
            df:           DataFrame
            x_col:        first variable (predictor)
            y_col:        second variable (outcome)
            method:       'auto', 'pearson', or 'spearman'
            alpha:        significance level for auto-detection and CI width
            x_transform:  transformation applied to x before testing
                          ('none', 'log10', 'sqrt', 'boxcox')
            y_transform:  transformation applied to y before testing
        """
        self._alpha = alpha
        self._x_transform = x_transform or 'none'
        self._y_transform = y_transform or 'none'

        work = df.dropna(subset=[x_col, y_col]).copy()
        self.n = len(work)

        if self.n < 4:
            raise ValueError(
                f"Correlation requires at least 4 complete value pairs "
                f"(after pairwise deletion: n={self.n})."
            )

        col_map = _sanitize_columns(work, [x_col, y_col])
        self._x = col_map[x_col]
        self._y = col_map[y_col]
        self._x_label = str(x_col)
        self._y_label = str(y_col)

        x_raw = work[self._x].values.astype(float)
        y_raw = work[self._y].values.astype(float)

        # Store scatter points from raw (untransformed) values for plots
        self._points = [{"x": float(x), "y": float(y)} for x, y in zip(x_raw, y_raw)]

        # --- Apply transformations ---
        x_vals, self._x_boxcox_lambda, self._x_transform_shift = _apply_transform(x_raw, self._x_transform)
        y_vals, self._y_boxcox_lambda, self._y_transform_shift = _apply_transform(y_raw, self._y_transform)

        # --- Normality check (auto mode only) ---
        self._normality_check = None
        if method == 'auto':
            # Calculate N, skewness, excess kurtosis for the values being tested (x_vals, y_vals)
            skew_x = float(scipy_stats.skew(x_vals))
            skew_y = float(scipy_stats.skew(y_vals))
            kurt_x = float(scipy_stats.kurtosis(x_vals))  # Excess kurtosis (Fisher: normal = 0)
            kurt_y = float(scipy_stats.kurtosis(y_vals))
            
            sw_stat_x, px = scipy_stats.shapiro(x_vals[:5000])
            sw_stat_y, py = scipy_stats.shapiro(y_vals[:5000])
            both_normal_sw = bool(px > alpha and py > alpha)
            
            # Determine method based on N-tier:
            from statistical_testing.validators import MIN_N_SMALL
            if self.n < MIN_N_SMALL:
                self._method_used = 'spearman'
            elif MIN_N_SMALL <= self.n < 100:
                # Pearson if |skewness| <= 1.0 and |excess kurtosis| <= 2.0 for both
                if (abs(skew_x) <= 1.0 and abs(skew_y) <= 1.0 and 
                        abs(kurt_x) <= 2.0 and abs(kurt_y) <= 2.0):
                    self._method_used = 'pearson'
                else:
                    self._method_used = 'spearman'
            else: # N >= 100
                # Pearson unless extreme asymmetry
                if abs(skew_x) > 2.0 or abs(skew_y) > 2.0 or abs(kurt_x) > 4.0 or abs(kurt_y) > 4.0:
                    self._method_used = 'spearman'
                else:
                    self._method_used = 'pearson'
                    
            has_transform = (self._x_transform != 'none' or self._y_transform != 'none')
            if has_transform:
                sw_stat_x_raw, px_raw = scipy_stats.shapiro(x_raw[:5000])
                sw_stat_y_raw, py_raw = scipy_stats.shapiro(y_raw[:5000])
                skew_x_raw = float(scipy_stats.skew(x_raw))
                skew_y_raw = float(scipy_stats.skew(y_raw))
                kurt_x_raw = float(scipy_stats.kurtosis(x_raw))
                kurt_y_raw = float(scipy_stats.kurtosis(y_raw))

                self._normality_check = {
                    "test": "Skewness/Kurtosis & Shapiro-Wilk check",
                    "transform_attempted": self._x_transform,
                    "transform_reverted": self._method_used == 'spearman',
                    "pre_transform": {
                        x_col: {"statistic": float(sw_stat_x_raw), "p_value": float(px_raw), "normal": bool(px_raw > alpha), "skewness": skew_x_raw, "kurtosis": kurt_x_raw},
                        y_col: {"statistic": float(sw_stat_y_raw), "p_value": float(py_raw), "normal": bool(py_raw > alpha), "skewness": skew_y_raw, "kurtosis": kurt_y_raw},
                        "both_normal": bool(px_raw > alpha and py_raw > alpha),
                    },
                    "post_transform": {
                        x_col: {"statistic": float(sw_stat_x), "p_value": float(px), "normal": bool(px > alpha), "skewness": skew_x, "kurtosis": kurt_x},
                        y_col: {"statistic": float(sw_stat_y), "p_value": float(py), "normal": bool(py > alpha), "skewness": skew_y, "kurtosis": kurt_y},
                        "both_normal": both_normal_sw,
                    },
                    "both_normal": both_normal_sw,
                }
            else:
                self._normality_check = {
                    "test": "Skewness/Kurtosis & Shapiro-Wilk check",
                    "skew_x": skew_x,
                    "skew_y": skew_y,
                    "kurtosis_x": kurt_x,
                    "kurtosis_y": kurt_y,
                    "shapiro_x_p": float(px),
                    "shapiro_y_p": float(py),
                    "shapiro_both_normal": both_normal_sw,
                    "both_normal": self._method_used == 'pearson',
                    x_col: {"statistic": float(sw_stat_x), "p_value": float(px), "normal": bool(px > alpha)},
                    y_col: {"statistic": float(sw_stat_y), "p_value": float(py), "normal": bool(py > alpha)},
                    "both_normal": both_normal_sw,
                }
        else:
            self._method_used = method

        # --- Compute r and p ---
        if self._method_used == 'pearson':
            self.r, self.p = scipy_stats.pearsonr(x_vals, y_vals)
        else:
            # Spearman on raw data; discard transform bookkeeping
            self.r, self.p = scipy_stats.spearmanr(x_raw, y_raw)
            self._x_transform = 'none'
            self._y_transform = 'none'
            self._x_boxcox_lambda = None
            self._y_boxcox_lambda = None
            self._x_transform_shift = 0.0
            self._y_transform_shift = 0.0
            
            # For 20 <= N < 100, calculate t-approximation for p-value
            if 20 <= self.n < 100:
                if np.isclose(abs(self.r), 1.0):
                    self.p = 0.0
                else:
                    t_stat = self.r * np.sqrt((self.n - 2) / (1.0 - self.r**2))
                    self.p = float(scipy_stats.t.sf(abs(t_stat), df=self.n - 2) * 2)

        self.r = float(self.r)
        self.p = float(self.p)
        self.ci = _fisher_z_ci(self.r, self.n, alpha)

        return self

    @staticmethod
    def _interpret(r):
        abs_r = abs(r)
        direction = "positive" if r >= 0 else "negative"
        if abs_r < 0.2:
            strength = "negligible"
        elif abs_r < 0.4:
            strength = "weak"
        elif abs_r < 0.6:
            strength = "moderate"
        elif abs_r < 0.8:
            strength = "strong"
        else:
            strength = "very strong"
        return f"{direction.capitalize()}, {strength} (|r| = {abs_r:.3f})"

    def _build_transformation_label(self) -> str:
        """Return a human-readable transformation string for the results export.

        Embeds all parameters required for full mathematical reproducibility:
        - Box-Cox:  boxcox(λ=0.3141, c=1.00)  — both λ and shift c are needed
                    to reconstruct y = ((x+c)^λ - 1) / λ
        - log10:    log10(c=4.50)              — only shown when shift != 0
        - sqrt:     sqrt(c=0.50)               — only shown when shift != 0
        - no shift: boxcox(λ=0.3141)           — c omitted when 0.0
        Falls back to plain name when lambda optimisation failed.
        Returns 'none' when no transformation was applied to either variable.
        """
        def _fmt(name, lam, shift):
            shift_part = f", c={shift:.4f}" if shift != 0.0 else ""
            if name == 'boxcox':
                if lam is not None:
                    return f"boxcox(λ={lam:.4f}{shift_part})"
                return f"boxcox→log{shift_part}" if shift != 0.0 else "boxcox→log"
            if name in ('log10', 'sqrt', 'log10(x+1)') and shift != 0.0:
                return f"{name}(c={shift:.4f})"
            return name

        x_label = _fmt(self._x_transform, self._x_boxcox_lambda, self._x_transform_shift)
        y_label = _fmt(self._y_transform, self._y_boxcox_lambda, self._y_transform_shift)

        if self._x_transform == 'none' and self._y_transform == 'none':
            return 'none'
        return f"{x_label}/{y_label}"

    def as_results_dict(self):
        if self.r is None:
            return {"error": "Model not fitted"}

        return {
            "test": f"Correlation ({self._method_used.capitalize()})",
            "model_type": "Correlation",
            "p_value": self.p,
            "statistic": self.r,
            "statistic_type": "r",
            "effect_size": abs(self.r),
            "effect_size_type": "r",
            "r": self.r,
            "method": self._method_used,
            "ci_lower": self.ci[0],
            "ci_upper": self.ci[1],
            "confidence_interval": [self.ci[0], self.ci[1]],
            "n": self.n,
            "alpha": self._alpha,
            "interpretation": self._interpret(self.r),
            "x_variable": self._x,
            "y_variable": self._y,
            "x_variable_display": self._x_label,
            "y_variable_display": self._y_label,
            "association_points": self._points,
            "normality_check": self._normality_check,
            "transformation": self._build_transformation_label(),
            "x_transform": getattr(self, '_x_transform', 'none'),
            "y_transform": getattr(self, '_y_transform', 'none'),
            "x_boxcox_lambda": getattr(self, '_x_boxcox_lambda', None),
            "y_boxcox_lambda": getattr(self, '_y_boxcox_lambda', None),
            # Shift c applied before each transformation (0.0 = no shift needed).
            # Required alongside λ for full reproducibility: y = ((x+c)^λ−1)/λ
            "x_transform_shift": getattr(self, '_x_transform_shift', 0.0),
            "y_transform_shift": getattr(self, '_y_transform_shift', 0.0),
        }


# ---------------------------------------------------------------------------
# 2. SimpleLinearRegressionModel
# ---------------------------------------------------------------------------

class SimpleLinearRegressionModel:
    """OLS linear regression with a continuous primary predictor + optional covariates.

    Regression diagnostics included (all non-blocking):
      - Shapiro-Wilk on residuals     (normality assumption)
      - Breusch-Pagan test            (homoscedasticity)
      - Ramsey RESET test             (linearity / functional form)

    For categorical group predictors use ANCOVAModel (clinical_models.py).
    Covariates are added as additional continuous predictors to the OLS formula.
    """

    def __init__(self):
        self.result = None
        self._df = None
        self._x = None
        self._y = None
        self._x_label = None
        self._y_label = None
        self._covariates = None
        self._alpha = 0.05
        self._x_transform = 'none'
        self._y_transform = 'none'
        self._x_boxcox_lambda = None
        self._y_boxcox_lambda = None
        self._x_transform_shift = 0.0
        self._y_transform_shift = 0.0
        self._x_raw_vals = None
        self._y_raw_vals = None
        self._cov_type = "nonrobust"

    def fit(self, df, x_col, y_col, covariates=None, alpha=0.05,
            x_transform='none', y_transform='none'):
        """Fit OLS model.

        Args:
            x_col:        continuous primary predictor column name
            y_col:        outcome column name
            covariates:   list of additional continuous predictor column names
            alpha:        significance level for diagnostics
            x_transform:  pre-fit transformation for X ('none', 'log10', 'sqrt', 'boxcox')
            y_transform:  pre-fit transformation for Y ('none', 'log10', 'sqrt', 'boxcox')
        """
        import statsmodels.formula.api as smf

        self._alpha = alpha
        self._x_transform = x_transform or 'none'
        self._y_transform = y_transform or 'none'
        all_cols = [x_col, y_col] + (covariates or [])
        work = df.dropna(subset=all_cols).copy()
        n = len(work)

        min_obs = max(5, len(all_cols) + 2)
        if n < min_obs:
            raise ValueError(
                f"Too few observations after listwise deletion (n={n}, "
                f"required: ≥{min_obs} for {len(all_cols)} variables)."
            )

        col_map = _sanitize_columns(work, all_cols)
        self._x = col_map[x_col]
        self._y = col_map[y_col]
        self._x_label = str(x_col)
        self._y_label = str(y_col)
        self._covariates = [col_map[c] for c in (covariates or [])]
        self._df = work

        # Apply pre-fit transformations (reuses the same helper as CorrelationModel).
        x_raw = pd.to_numeric(self._df[self._x], errors='coerce').to_numpy(dtype=float)
        y_raw = pd.to_numeric(self._df[self._y], errors='coerce').to_numpy(dtype=float)
        self._x_raw_vals = x_raw.copy()
        self._y_raw_vals = y_raw.copy()
        
        x_vals, self._x_boxcox_lambda, self._x_transform_shift = _apply_transform(x_raw, self._x_transform)
        
        if self._y_transform == 'boxcox':
            # Box-Cox on Y must optimize over the OLS residuals, not the marginal distribution.
            # 1. Temporarily drop NaNs from X (and covariates) and Y.
            temp_df = self._df.copy()
            temp_df[self._x] = x_vals
            temp_df = temp_df.dropna(subset=[self._x, self._y] + self._covariates)
            
            if len(temp_df) < 5:
                # Not enough data, fallback
                y_vals, self._y_boxcox_lambda, self._y_transform_shift = _apply_transform(y_raw, self._y_transform)
            else:
                # Construct design matrix (intercept + X + covariates)
                import patsy
                terms = [self._x] + self._covariates
                formula = f"~ {' + '.join(terms)}"
                x_matrix = patsy.dmatrix(formula, temp_df, return_type='dataframe').to_numpy()
                y_clean = pd.to_numeric(temp_df[self._y], errors='coerce').to_numpy(dtype=float)
                
                # Check for negative values
                y_raw_copy = y_raw.copy()
                y_raw_copy[y_raw_copy <= 0] = np.nan
                
                # Optimize lambda
                lam = _optimize_boxcox_for_regression(y_clean, x_matrix)
                
                # Apply transformation with the optimized lambda
                from scipy.special import boxcox as scipy_boxcox
                y_vals = y_raw_copy
                valid_mask = ~np.isnan(y_vals)
                y_vals[valid_mask] = scipy_boxcox(y_vals[valid_mask], lam)
                
                self._y_boxcox_lambda = float(lam)
                self._y_transform_shift = 0.0
        else:
            y_vals, self._y_boxcox_lambda, self._y_transform_shift = _apply_transform(y_raw, self._y_transform)

        # Write transformed values back into the working DataFrame so statsmodels sees them.
        self._df = self._df.copy()
        self._df[self._x] = x_vals
        self._df[self._y] = y_vals

        terms = [self._x] + self._covariates
        formula = f"{self._y} ~ {' + '.join(terms)}"

        self.result = smf.ols(formula, data=self._df).fit()

        # Run diagnostics on OLS to check homoscedasticity for HC3 switch
        diag = self.diagnostics()
        bp = diag.get("homoscedasticity", {})
        bp_p = bp.get("p_value", None)

        self._cov_type = "nonrobust"
        if n >= 20 and bp_p is not None and bp_p < alpha:
            try:
                self.result = self.result.get_robustcov_results(cov_type='HC3')
                self._cov_type = "HC3"
            except Exception as exc:
                print(f"WARNING: Failed to apply HC3 covariance: {exc}")

        return self

    def _build_coef_interpretation(self, beta):
        """Return a human-readable sentence describing β given the active transforms."""
        xt = self._x_transform
        yt = self._y_transform

        if xt == 'none' and yt == 'none':
            return (
                f"Absolute change: a 1-unit increase in X is associated with a "
                f"β = {beta:.4f} unit change in Y."
            )
        if xt == 'none' and yt == 'log10':
            factor = 10 ** beta
            direction = "increase" if beta >= 0 else "decrease"
            return (
                f"Log-linear model (Y log10-transformed): a 1-unit increase in X "
                f"multiplies Y by 10^β ≈ {factor:.4f} "
                f"(i.e., a {direction} by factor {factor:.4f})."
            )
        if xt == 'log10' and yt == 'none':
            return (
                f"Lin-log model (X log10-transformed): a 10-fold increase in X "
                f"is associated with a β = {beta:.4f} unit change in Y."
            )
        if xt == 'log10' and yt == 'log10':
            return (
                f"Log-log model — elasticity: a 1% increase in X is associated "
                f"with approximately a β = {beta:.4f}% change in Y."
            )
        if xt == 'log10(x+1)' and yt == 'none':
            return (
                f"Lin-log1p model (X log10(x+1)-transformed): a 10-fold increase in X "
                f"is associated with a β = {beta:.4f} unit change in Y."
            )
        if xt == 'none' and yt == 'log10(x+1)':
            factor = 10 ** beta
            direction = "increase" if beta >= 0 else "decrease"
            return (
                f"Log1p-linear model (Y log10(x+1)-transformed): a 1-unit increase in X "
                f"multiplies (Y+1) by 10^β ≈ {factor:.4f} "
                f"(i.e., a {direction} by factor {factor:.4f})."
            )
        if xt == 'none' and yt == 'sqrt':
            return (
                f"Sqrt(Y)-linear model: a 1-unit increase in X changes √Y by β = {beta:.4f}; "
                f"back-transformation required for units of Y."
            )
        if xt == 'sqrt' and yt == 'none':
            return (
                f"Lin-√X model: a 1-unit increase in √X is associated with a "
                f"β = {beta:.4f} unit change in Y."
            )
        # Generic fallback for boxcox or other combinations
        xt_label = xt if xt != 'none' else 'untransformed'
        yt_label = yt if yt != 'none' else 'untransformed'
        return (
            f"Transformed model (X: {xt_label}, Y: {yt_label}): β = {beta:.4f} "
            f"on the transformed scale. Direct interpretation requires back-transformation."
        )

    @staticmethod
    def _transform_axis_label(label: str, transform: str) -> str:
        """Prefix an axis label with the active transform so the plot scale is clear."""
        if not transform or transform == 'none':
            return label
        _pretty = {
            'log10': 'log\u2081\u2080',
            'log10(x+1)': 'log\u2081\u2080(x+1)',
            'log':   'ln',
            'sqrt':  '\u221a',
            'boxcox': 'BoxCox',
        }
        prefix = _pretty.get(transform, transform)
        if transform == 'sqrt':
            return f"{prefix}({label})"
        return f"{prefix}({label})"

    def _build_regression_plot_payload(self, n_points=180):
        if self.result is None or self._df is None:
            return None
        try:
            # self._df holds *transformed* values — use those for scatter + fit line
            # so that points and line are on the same (transformed) scale.
            x_obs = pd.to_numeric(self._df[self._x], errors="coerce").to_numpy(dtype=float)
            y_obs = pd.to_numeric(self._df[self._y], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(x_obs) & np.isfinite(y_obs)
            x_obs = x_obs[valid]
            y_obs = y_obs[valid]
            if x_obs.size < 2 or y_obs.size < 2:
                return None

            x_min = float(np.min(x_obs))
            x_max = float(np.max(x_obs))
            if np.isclose(x_min, x_max):
                x_fit = np.array([x_min], dtype=float)
            else:
                x_fit = np.linspace(x_min, x_max, int(max(25, n_points)))

            prediction_frame = pd.DataFrame({self._x: x_fit})
            for covariate in self._covariates:
                cov_values = pd.to_numeric(self._df[covariate], errors="coerce").to_numpy(dtype=float)
                cov_mean = np.nanmean(cov_values)
                prediction_frame[covariate] = float(cov_mean) if np.isfinite(cov_mean) else 0.0

            prediction = self.result.get_prediction(prediction_frame).summary_frame(alpha=self._alpha)
            y_fit = prediction["mean"].to_numpy(dtype=float)
            ci_lower = prediction["mean_ci_lower"].to_numpy(dtype=float)
            ci_upper = prediction["mean_ci_upper"].to_numpy(dtype=float)

            # Axis labels reflect the active transformation so the reader knows the scale.
            x_label = self._transform_axis_label(self._x_label, self._x_transform)
            y_label = self._transform_axis_label(self._y_label, self._y_transform)

            return {
                "x_label": x_label,
                "y_label": y_label,
                "x_transform": self._x_transform,
                "y_transform": self._y_transform,
                "points": [{"x": float(x), "y": float(y)} for x, y in zip(x_obs, y_obs)],
                "fit": {
                    "x": [float(value) for value in x_fit.tolist()],
                    "y": [float(value) for value in y_fit.tolist()],
                    "ci_lower": [float(value) for value in ci_lower.tolist()],
                    "ci_upper": [float(value) for value in ci_upper.tolist()],
                    "alpha": float(self._alpha),
                    "confidence_level": float(1.0 - self._alpha),
                },
            }
        except Exception:
            return None

    def diagnostics(self):
        """Run regression assumption checks. Returns dict (all non-blocking)."""
        if self.result is None:
            return {}

        diag = {}
        residuals = self.result.resid.values
        n = len(residuals)

        # 1. Normality of residuals (Shapiro-Wilk)
        try:
            sw_stat, sw_p = scipy_stats.shapiro(residuals[:5000])
            diag["normality"] = {
                "test": "Shapiro-Wilk (Residuals)",
                "statistic": float(sw_stat),
                "p_value": float(sw_p),
                "assumption_holds": bool(sw_p > self._alpha),
            }
        except Exception as exc:
            diag["normality"] = {"test": "Shapiro-Wilk", "error": str(exc)}

        # 2. Homoscedasticity (Breusch-Pagan)
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan
            X_exog = self.result.model.exog
            bp_lm, bp_p, _, _ = het_breuschpagan(residuals, X_exog)
            diag["homoscedasticity"] = {
                "test": "Breusch-Pagan",
                "statistic": float(bp_lm),
                "p_value": float(bp_p),
                "assumption_holds": bool(bp_p > self._alpha),
            }
        except Exception as exc:
            diag["homoscedasticity"] = {"test": "Breusch-Pagan", "error": str(exc)}

        # 3. Linearity (Ramsey RESET, power=2)
        try:
            from statsmodels.stats.diagnostic import linear_reset
            reset = linear_reset(self.result, power=2, use_f=True)
            diag["linearity"] = {
                "test": "Ramsey RESET",
                "statistic": float(reset.statistic),
                "p_value": float(reset.pvalue),
                "assumption_holds": bool(reset.pvalue > self._alpha),
            }
        except Exception as exc:
            diag["linearity"] = {"test": "Ramsey RESET", "error": str(exc)}

        return diag

    def as_results_dict(self):
        if self.result is None:
            return {"error": "Model not fitted"}

        diag = self.diagnostics()

        # Coefficient table
        conf = self.result.conf_int()
        coef_table = []
        for param in self.result.params.index:
            coef_table.append({
                "parameter": str(param),
                "coefficient": float(self.result.params[param]),
                "std_err": float(self.result.bse[param]),
                "t_value": float(self.result.tvalues[param]),
                "p_value": float(self.result.pvalues[param]),
                "ci_lower": float(conf.loc[param, 0]),
                "ci_upper": float(conf.loc[param, 1]),
            })

        # Primary predictor stats
        main_beta = main_p = main_t = None
        if self._x in self.result.params.index:
            main_beta = float(self.result.params[self._x])
            main_p = float(self.result.pvalues[self._x])
            main_t = float(self.result.tvalues[self._x])

        f_stat = float(self.result.fvalue) if hasattr(self.result, 'fvalue') and self.result.fvalue is not None else None
        f_p = float(self.result.f_pvalue) if hasattr(self.result, 'f_pvalue') and self.result.f_pvalue is not None else None

        # Raw scatter points for descriptive summary and charts (original scale, pre-transform)
        association_points = []
        if self._x_raw_vals is not None and self._y_raw_vals is not None:
            try:
                x_obs = self._x_raw_vals
                y_obs = self._y_raw_vals
                valid = np.isfinite(x_obs) & np.isfinite(y_obs)
                association_points = [
                    {"x": float(xv), "y": float(yv)}
                    for xv, yv in zip(x_obs[valid], y_obs[valid])
                ]
            except Exception:
                pass

        # Residuals and fitted values at top level for QQ plot and residuals-vs-fitted chart
        residuals_list = None
        fitted_list = None
        try:
            residuals_list = [float(v) for v in self.result.resid.values]
            fitted_list = [float(v) for v in self.result.fittedvalues.values]
        except Exception:
            pass

        coef_interp = self._build_coef_interpretation(main_beta) if main_beta is not None else None

        return {
            "test": "Linear Regression (OLS)",
            "model_type": "LinearRegression",
            "alpha": self._alpha,
            "p_value": main_p,
            "statistic": main_t,
            "statistic_type": "t",
            "effect_size": float(self.result.rsquared),
            "effect_size_type": "R_squared",
            "beta": main_beta,
            "r_squared": float(self.result.rsquared),
            "r_squared_adj": float(self.result.rsquared_adj),
            "f_statistic": f_stat,
            "f_p_value": f_p,
            "aic": float(self.result.aic),
            "bic": float(self.result.bic),
            "n_observations": int(self.result.nobs),
            "coefficient_table": coef_table,
            "diagnostics": diag,
            "residuals": residuals_list,
            "model_residuals": residuals_list,
            "fitted_values": fitted_list,
            "x_variable": self._x,
            "y_variable": self._y,
            "x_variable_display": self._x_label,
            "y_variable_display": self._y_label,
            "covariates_used": self._covariates,
            "association_points": association_points,
            "plot_regression": self._build_regression_plot_payload(),
            # Transform metadata (mirrors CorrelationModel fields for export parity)
            "x_transform": self._x_transform,
            "y_transform": self._y_transform,
            "x_boxcox_lambda": self._x_boxcox_lambda,
            "y_boxcox_lambda": self._y_boxcox_lambda,
            "x_transform_shift": self._x_transform_shift,
            "y_transform_shift": self._y_transform_shift,
            "transformation": (
                f"{self._x_transform}/{self._y_transform}"
                if (self._x_transform != 'none' or self._y_transform != 'none')
                else "none"
            ),
            "coef_interpretation": coef_interp,
            "cov_type": self._cov_type,
        }


# ---------------------------------------------------------------------------
# 3. ExploratoryCorrelationMatrix
# ---------------------------------------------------------------------------

class ExploratoryCorrelationMatrix:
    """Pairwise correlation matrix with multiple-testing correction.

    Pairwise deletion (default): each variable pair uses all available rows
    independently — n per pair is reported in the n-matrix.
    Listwise deletion: only complete cases across ALL selected variables.

    Supports optional stratification: computes a separate matrix per level of
    a categorical column (e.g. OP-Group).
    """

    def __init__(self):
        self._columns = None
        self._method = None
        self._correction = None
        self._pairwise = True
        self._stratify_by = None
        self._alpha = 0.05
        self.r_matrix = None
        self.p_matrix = None
        self.p_corrected_matrix = None
        self.n_matrix = None
        self.strata_results = None

    def fit(self, df, columns, method='spearman', correction='fdr_bh',
            pairwise=True, stratify_by=None, alpha=0.05):
        """Compute the correlation matrix.

        Args:
            df:           DataFrame
            columns:      list of numeric column names to include
            method:       'spearman', 'pearson', or 'auto'
            correction:   'fdr_bh' (Benjamini-Hochberg), 'bonferroni', or None
            pairwise:     True = pairwise deletion (recommended, preserves n)
                          False = listwise deletion (only complete cases)
            stratify_by:  column name for group-stratified matrices (optional)
            alpha:        significance level for auto method-selection
        """
        self._columns = list(columns)
        self._method = method
        self._correction = correction
        self._pairwise = pairwise
        self._stratify_by = stratify_by
        self._alpha = alpha

        # Prepare base DataFrame
        extra = [stratify_by] if (stratify_by and stratify_by in df.columns) else []
        base = df[self._columns + extra].copy()

        if not pairwise:
            base = base.dropna(subset=self._columns)

        # Unstratified matrix (always computed)
        r_m, p_m, pc_m, n_m = self._compute_matrix(base[self._columns])
        self.r_matrix = r_m
        self.p_matrix = p_m
        self.p_corrected_matrix = pc_m
        self.n_matrix = n_m

        # Stratified matrices (optional)
        if stratify_by and stratify_by in base.columns:
            self.strata_results = {}
            for grp_val, grp_df in base.groupby(stratify_by):
                r_s, p_s, pc_s, n_s = self._compute_matrix(grp_df[self._columns])
                self.strata_results[str(grp_val)] = {
                    "r_matrix": r_s,
                    "p_matrix": p_s,
                    "p_corrected_matrix": pc_s,
                    "n_matrix": n_s,
                }

        return self

    def _compute_matrix(self, data):
        """Core computation: returns (r_mat, p_mat, p_corrected_mat, n_mat) as ndarrays."""
        cols = self._columns
        k = len(cols)
        r_mat = np.full((k, k), np.nan)
        p_mat = np.full((k, k), np.nan)
        n_mat = np.zeros((k, k))
        np.fill_diagonal(r_mat, 1.0)
        np.fill_diagonal(n_mat, data[cols].notna().sum().values.astype(float))

        all_p = []
        ij_indices = []

        for i in range(k):
            for j in range(i + 1, k):
                pair = data[[cols[i], cols[j]]].dropna()
                n = len(pair)
                n_mat[i, j] = n_mat[j, i] = float(n)
                if n < 4:
                    continue

                x = pair.iloc[:, 0].values.astype(float)
                y = pair.iloc[:, 1].values.astype(float)

                m = self._method
                if m == 'auto':
                    _, px = scipy_stats.shapiro(x[:5000])
                    _, py = scipy_stats.shapiro(y[:5000])
                    m = 'pearson' if (px > self._alpha and py > self._alpha) else 'spearman'

                try:
                    if m == 'pearson':
                        r, p = scipy_stats.pearsonr(x, y)
                    else:
                        r, p = scipy_stats.spearmanr(x, y)
                    r_mat[i, j] = r_mat[j, i] = float(r)
                    p_mat[i, j] = p_mat[j, i] = float(p)
                    all_p.append(float(p))
                    ij_indices.append((i, j))
                except Exception:
                    pass

        # Multiple testing correction
        pc_mat = p_mat.copy()
        if self._correction and all_p:
            try:
                from statsmodels.stats.multitest import multipletests
                _, p_adj, _, _ = multipletests(all_p, method=self._correction)
                for idx, (i, j) in enumerate(ij_indices):
                    pc_mat[i, j] = pc_mat[j, i] = float(p_adj[idx])
            except Exception:
                pass  # Fall back to uncorrected p-values

        return r_mat, p_mat, pc_mat, n_mat

    @staticmethod
    def _ndarray_to_nested_dict(mat, cols):
        """Convert k×k ndarray to {col_i: {col_j: value}} dict, NaN → None."""
        out = {}
        for i, ci in enumerate(cols):
            out[ci] = {}
            for j, cj in enumerate(cols):
                val = mat[i, j]
                out[ci][cj] = None if np.isnan(val) else float(val)
        return out

    @staticmethod
    def _n_mat_to_dict(mat, cols):
        """Same as above but casts to int (or None)."""
        out = {}
        for i, ci in enumerate(cols):
            out[ci] = {}
            for j, cj in enumerate(cols):
                val = mat[i, j]
                out[ci][cj] = None if np.isnan(val) else int(val)
        return out

    def as_results_dict(self):
        if self.r_matrix is None:
            return {"error": "Matrix not computed"}

        cols = self._columns
        _d = self._ndarray_to_nested_dict
        _n = self._n_mat_to_dict

        result = {
            "test": "Explorative Korrelationsmatrix",
            "model_type": "CorrelationMatrix",
            "method": self._method,
            "correction": self._correction,
            "pairwise_deletion": self._pairwise,
            "variables": cols,
            "r_matrix": _d(self.r_matrix, cols),
            "p_matrix": _d(self.p_matrix, cols),
            "p_corrected_matrix": _d(self.p_corrected_matrix, cols),
            "n_matrix": _n(self.n_matrix, cols),
        }

        if self.strata_results:
            strat_out = {}
            for grp, mats in self.strata_results.items():
                strat_out[grp] = {
                    "r_matrix": _d(mats["r_matrix"], cols),
                    "p_corrected_matrix": _d(mats["p_corrected_matrix"], cols),
                    "n_matrix": _n(mats["n_matrix"], cols),
                }
            result["strata"] = strat_out

        return result


# ---------------------------------------------------------------------------
# 4. RegressionHealthScanner
# ---------------------------------------------------------------------------

class RegressionHealthScanner:
    """Pre-analysis data quality checks specific to linear regression.

    Checks:
      1. Outliers in predictor and covariates (MAD-based Modified Z-Score)
      2. Multicollinearity among predictors (VIF, when ≥2 continuous predictors)
      3. Minimum sample size (n ≥ 10 per predictor recommended)
      4. Missing data summary (n complete vs. dropped)

    Returns the same interface as DataHealthScanner.run():
        {"warnings": [...], "checks": {...}}
    """

    def __init__(self, df, x_col, y_col, covariates=None):
        self._df = df.copy()
        self._x = x_col
        self._y = y_col
        self._covariates = covariates or []
        self.warnings = []
        self.checks = {}

    def run(self):
        all_pred = [self._x] + self._covariates
        all_cols = [self._y] + all_pred

        # Missing data summary
        n_total = len(self._df)
        n_complete = self._df.dropna(subset=all_cols).shape[0]
        n_dropped = n_total - n_complete
        self.checks["missing_data"] = {
            "n_total": n_total,
            "n_complete": n_complete,
            "n_dropped": n_dropped,
        }
        if n_dropped > 0:
            self.warnings.append(
                f"Missing values: {n_dropped} of {n_total} rows excluded "
                f"(listwise deletion)."
            )

        # Minimum sample size
        n_preds = len(all_pred)
        if n_complete < 10 * n_preds:
            self.warnings.append(
                f"Sample size: n={n_complete} for {n_preds} predictors "
                f"— Rule of thumb: ≥10 observations per predictor recommended."
            )
        self.checks["sample_size"] = {"n_complete": n_complete, "n_predictors": n_preds}

        # Outliers in predictors (MAD-based, same as DataHealthScanner)
        outlier_info = {}
        for col in all_pred:
            if col not in self._df.columns:
                continue
            vals = self._df[col].dropna().values
            if len(vals) < 4:
                continue
            median = np.median(vals)
            mad = np.median(np.abs(vals - median))
            if mad == 0:
                continue
            mod_z = 0.6745 * (vals - median) / mad
            n_extreme = int(np.sum(np.abs(mod_z) > 3.5))
            if n_extreme > 0:
                outlier_info[col] = n_extreme
                self.warnings.append(
                    f"Outliers in '{col}': {n_extreme} value(s) with |mod. Z-score| > 3.5."
                )
        self.checks["predictor_outliers"] = outlier_info

        # VIF (multicollinearity) when ≥2 continuous predictors
        if len(all_pred) >= 2:
            try:
                import statsmodels.api as sm
                from statsmodels.stats.outliers_influence import variance_inflation_factor
                cov_data = self._df[all_pred].dropna()
                if len(cov_data) >= len(all_pred) + 2:
                    X = sm.add_constant(cov_data.values, has_constant='add')
                    vif_vals = {}
                    high_vif = []
                    for i, col in enumerate(all_pred):
                        vif = float(variance_inflation_factor(X, i + 1))
                        vif_vals[col] = round(vif, 2)
                        if vif > 10:
                            high_vif.append(f"{col} (VIF={vif:.1f})")
                    self.checks["vif"] = vif_vals
                    if high_vif:
                        self.warnings.append(
                            f"Multicollinearity: {', '.join(high_vif)} — "
                            "VIF > 10, coefficient interpretation is limited."
                        )
            except Exception as exc:
                self.checks["vif"] = {"error": str(exc)}

        return {"warnings": self.warnings, "checks": self.checks}
