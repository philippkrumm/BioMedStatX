"""
Clinical-Grade Statistical Models for BioMedStatX.

This module provides ANCOVA, Linear Mixed Models (LMM), and Logistic Regression
for clinical study data with unbalanced designs, missing data, and confounders.

All model classes follow the pattern established in nonparametricanovas.py:
    model = SomeModel()
    model.fit(df, ...)
    results = model.as_results_dict()
"""

import re
import numpy as np
import pandas as pd

import logging
logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod
from enum import Enum

class DesignType(str, Enum):
    INDEPENDENT = "INDEPENDENT"
    REPEATED = "REPEATED"
    MIXED = "MIXED"

class BaseStatisticalModel(ABC):
    
    @property
    @abstractmethod
    def design_type(self) -> DesignType:
        """Jeder erbende Test MUSS dieses Flag definieren."""
        pass


def _sanitize_columns(df, columns):
    """Rename columns with special characters so patsy can parse them.

    Returns a dict mapping original name -> sanitized name.
    The DataFrame is renamed in-place.
    """
    mapping = {}
    for col in columns:
        safe = re.sub(r'[^A-Za-z0-9_]', '_', str(col))
        
        # Block patsy keywords
        if safe in ('C', 'I', 'Q'):
            safe = f"{safe}_safe"
            
        if safe != col:
            # Avoid collisions
            base = safe
            i = 2
            while safe in df.columns and safe != col:
                safe = f"{base}_{i}"
                i += 1
            df.rename(columns={col: safe}, inplace=True)
        mapping[col] = safe
    return mapping

def _restore_names_in_dict(d, rev_map):
    """Recursively reverse the sanitized names back to original names in dictionaries and lists."""
    if not rev_map:
        return d
        
    if isinstance(d, dict):
        new_d = {}
        for k, v in d.items():
            new_k = k
            if isinstance(k, str):
                # Replace longer safe names first to prevent partial replacements
                for safe, orig in sorted(rev_map.items(), key=lambda x: len(x[0]), reverse=True):
                    if safe != orig:
                        new_k = new_k.replace(safe, str(orig))
            new_d[new_k] = _restore_names_in_dict(v, rev_map)
        return new_d
    elif isinstance(d, list):
        return [_restore_names_in_dict(item, rev_map) for item in d]
    elif isinstance(d, str):
        new_str = d
        for safe, orig in sorted(rev_map.items(), key=lambda x: len(x[0]), reverse=True):
            if safe != orig:
                new_str = new_str.replace(safe, str(orig))
        return new_str
    else:
        return d


class ANCOVAModel(BaseStatisticalModel):
    """ANCOVA via statsmodels OLS with Type III SS (Sum contrasts).

    Type III SS with Sum-to-zero contrasts gives interpretable main effects in
    unbalanced designs even when the factor interaction is present ("C1 fix").
    The regression slope homogeneity check validates the ANCOVA assumption
    that factor and covariate do not interact.
    """

    def __init__(self):
        self._rev_map = {}
        self.result = None
        self.anova_table = None
        self._df = None
        self._dv = None
        self._between_factors = None
        self._covariates = None
        self._alpha = 0.05
        self._control_group = None

    @property
    def design_type(self) -> DesignType:
        return DesignType.INDEPENDENT

    def fit(self, df, dv, between_factors, covariates, alpha=0.05, control_group=None):
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm

        self._df = df.dropna(subset=[dv] + between_factors + covariates).copy()
        self._alpha = alpha
        # Control level for vs-control EMM contrasts (original, un-sanitized
        # label — only column *names* get sanitized, not the level values).
        self._control_group = control_group

        # Sanitize column names for patsy formulas (replace spaces/special chars)
        col_map = _sanitize_columns(self._df, [dv] + between_factors + covariates)
        self._rev_map = {v: k for k, v in col_map.items()}
        self._dv = col_map[dv]
        self._between_factors = [col_map[f] for f in between_factors]
        self._covariates = [col_map[c] for c in covariates]

        # C1: Use Sum contrasts to ensure correct Type III SS in unbalanced designs
        factor_terms = " * ".join([f"C({f}, Sum)" for f in self._between_factors])
        cov_terms = " + ".join(self._covariates)
        formula = f"{self._dv} ~ {factor_terms} + {cov_terms}"

        model = smf.ols(formula, data=self._df).fit()
        self.result = model
        self.anova_table = anova_lm(model, typ=3)
        return self

    def check_regression_slope_homogeneity(self):
        """Test Factor x Covariate interaction (ANCOVA assumption).

        Returns a dict per covariate with F, p, and whether the assumption holds.
        """
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm

        results = {}
        # Use sanitized column names (already renamed in-place during fit)
        col_map = {c: c for c in [self._dv] + self._between_factors + self._covariates}
        # Columns were already sanitized in fit(), just use current df column names
        for cov in self._covariates:
            for factor in self._between_factors:
                # C1: Use Sum contrasts for Type III SS
                factor_term = f"C({factor}, Sum)"
                formula = f"{self._dv} ~ {factor_term} * {cov}"
                
                try:
                    model_interaction = smf.ols(formula, data=self._df).fit()
                    table = anova_lm(model_interaction, typ=3)
                    
                    interaction_term = f"{factor_term}:{cov}"
                    key = f"{factor}:{cov}"
                    if interaction_term in table.index:
                        row = table.loc[interaction_term]
                        p_val = float(row["PR(>F)"])
                        results[key] = {
                            "F": float(row["F"]),
                            "p_value": p_val,
                            "df": row["df"],
                            "assumption_holds": p_val > self._alpha,
                        }
                except Exception:
                    results[f"{factor}:{cov}"] = {
                        "F": None, "p_value": None, "df": None,
                        "assumption_holds": None, "error": "Could not fit interaction model"
                    }
        return results

    def adjusted_means(self):
        """Compute estimated marginal means (adjusted for covariates).

        True EMMs (R emmeans / SPSS EMMEANS): predictions are averaged over a
        balanced reference grid — every level of the OTHER between-factors
        weighted equally, covariates fixed at their grand mean. Averaging over
        the empirical rows instead would weight the other factors by their
        observed (possibly unbalanced) distribution and bias the contrasts.
        """
        from itertools import product as _product

        if self.result is None:
            return {}

        means = {}
        cov_means = {c: self._df[c].mean() for c in self._covariates}

        for factor in self._between_factors:
            levels = self._df[factor].unique()
            other_factors = [f for f in self._between_factors if f != factor]
            other_levels = [self._df[f].unique() for f in other_factors]
            adjusted = {}

            for level in levels:
                subset = self._df[self._df[factor] == level]
                # Balanced reference grid: {level} x all combinations of the
                # other factors' levels, covariates at grand mean.
                grid_rows = []
                for combo in _product(*other_levels) if other_factors else [()]:
                    row = {factor: level}
                    row.update(dict(zip(other_factors, combo)))
                    row.update(cov_means)
                    grid_rows.append(row)
                grid = pd.DataFrame(grid_rows)
                try:
                    predicted = self.result.predict(grid)
                    adjusted[str(level)] = {
                        "adjusted_mean": float(np.mean(predicted)),
                        "n": len(subset),
                        "raw_mean": float(subset[self._dv].mean()),
                        "raw_sd": float(subset[self._dv].std()),
                    }
                except Exception:
                    adjusted[str(level)] = {
                        "adjusted_mean": None, "n": len(subset),
                        "raw_mean": float(subset[self._dv].mean()),
                        "raw_sd": float(subset[self._dv].std()),
                    }
            means[factor] = adjusted
        return means

    def _emm_functional(self, factor, level):
        """Linear functional L (length = n model params) whose dot product with
        the OLS coefficients yields the EMM of ``level`` for ``factor``.

        Built from the *same* balanced reference grid as ``adjusted_means``
        (other between-factors balanced, covariates at their grand mean), so the
        contrast L_A - L_B is exactly the difference of adjusted means.
        """
        from itertools import product as _product
        from patsy import dmatrix

        cov_means = {c: self._df[c].mean() for c in self._covariates}
        other_factors = [f for f in self._between_factors if f != factor]
        other_levels = [self._df[f].unique() for f in other_factors]

        grid_rows = []
        for combo in (_product(*other_levels) if other_factors else [()]):
            row = {factor: level}
            row.update(dict(zip(other_factors, combo)))
            row.update(cov_means)
            grid_rows.append(row)
        grid = pd.DataFrame(grid_rows)

        design_info = self.result.model.data.design_info
        X = np.asarray(dmatrix(design_info, grid, return_type="matrix"))
        return X.mean(axis=0)

    @staticmethod
    def _mvt_pvalues(t_values, R, df):
        """Single-step two-sided multivariate-t adjusted p-values for a family
        of contrasts with correlation matrix ``R`` and ``df`` residual df. This
        is the emmeans ``adjust="mvt"`` rule: it accounts for the correlation
        between the contrasts instead of the conservative Bonferroni/Holm bound.
        """
        from scipy.stats import multivariate_t, t as student_t

        k = len(t_values)
        if k == 1:
            return [float(2.0 * student_t.sf(abs(t_values[0]), df))]
        rv = multivariate_t(loc=np.zeros(k), shape=R, df=df, allow_singular=True)
        out = []
        for tv in t_values:
            c = abs(float(tv))
            p_all = float(rv.cdf(np.full(k, c), lower_limit=np.full(k, -c)))
            out.append(float(min(1.0, max(0.0, 1.0 - p_all))))
        return out

    def emm_contrasts(self, method="vs_control", control_group=None, factor=None):
        """EMM (covariate-adjusted) post-hoc contrasts on the primary between
        factor, computed via ``result.t_test`` on the fitted OLS model.

        method="vs_control": treatment-vs-control family, multivariate-t adjusted
        (optimal for targeted designs with a defined reference, e.g. empty-vector
        controls in Dual-Luciferase assays). Requires ``control_group``.

        method="pairwise": all C(G,2) contrasts, Holm-Bonferroni adjusted.

        Returns a list of comparison dicts (group1, group2, estimate, se, t, df,
        p_value, significant) or ``None`` if the model is not fitted / <2 levels.
        Each estimate has sign ``mean(group1) - mean(group2)``.
        """
        if self.result is None:
            return None
        factor = factor or self._between_factors[0]
        levels = list(pd.unique(self._df[factor]))
        if len(levels) < 2:
            return None

        functionals = {lvl: self._emm_functional(factor, lvl) for lvl in levels}
        beta = np.asarray(self.result.params.values, dtype=float)
        cov_params = np.asarray(self.result.cov_params().values, dtype=float)
        ddf = float(self.result.df_resid)

        if method == "vs_control":
            if control_group is None or control_group not in levels:
                raise ValueError("vs_control requires a control_group present in the data")
            treatments = [lvl for lvl in levels if lvl != control_group]
            pairs = [(trt, control_group) for trt in treatments]
        elif method == "pairwise":
            from itertools import combinations as _combinations
            pairs = list(_combinations(levels, 2))
        else:
            raise ValueError(f"unknown contrast method {method!r}")

        Lmat = np.vstack([functionals[a] - functionals[b] for a, b in pairs])
        est = Lmat @ beta
        cov_c = Lmat @ cov_params @ Lmat.T
        se = np.sqrt(np.clip(np.diag(cov_c), 0.0, None))
        with np.errstate(divide="ignore", invalid="ignore"):
            t_values = np.where(se > 0, est / se, 0.0)

        if method == "vs_control":
            d = np.sqrt(np.clip(np.diag(cov_c), 1e-300, None))
            R = cov_c / np.outer(d, d)
            np.fill_diagonal(R, 1.0)
            p_adj = self._mvt_pvalues(list(t_values), R, ddf)
        else:
            from scipy.stats import t as student_t
            from statsmodels.stats.multitest import multipletests
            p_raw = [float(2.0 * student_t.sf(abs(tv), ddf)) for tv in t_values]
            _, p_adj, _, _ = multipletests(p_raw, alpha=self._alpha, method="holm")
            p_adj = list(p_adj)

        contrasts = []
        for (a, b), e, s, tv, p in zip(pairs, est, se, t_values, p_adj):
            contrasts.append({
                "group1": str(a),
                "group2": str(b),
                "estimate": float(e),
                "se": float(s),
                "t": float(tv),
                "df": ddf,
                "p_value": float(p),
                "significant": bool(p < self._alpha),
            })
        return contrasts

    def run_simple_slopes_and_jn(self):
        """Perform Simple Slopes (Pick-a-Point) and Johnson-Neyman analysis when slopes are heterogeneous."""
        if self.result is None or not self._covariates or not self._between_factors:
            return None

        import statsmodels.formula.api as smf
        from scipy.stats import t as scipy_t

        factor = self._between_factors[0]
        cov = self._covariates[0]

        df_clean = self._df.copy()
        
        # 1. Simple Slopes (Pick-a-Point Approach)
        cov_vals = df_clean[cov].dropna().values
        cov_mean = float(np.mean(cov_vals))
        cov_sd = float(np.std(cov_vals, ddof=1)) if len(cov_vals) > 1 else 0.0
        
        points = {
            "Mean - 1 SD": cov_mean - cov_sd,
            "Mean": cov_mean,
            "Mean + 1 SD": cov_mean + cov_sd
        }
        
        simple_slopes = []
        for label, val in points.items():
            df_centered = df_clean.copy()
            df_centered[f"{cov}_centered"] = df_centered[cov] - val
            
            factor_terms = " * ".join([f"C({f})" for f in self._between_factors])
            other_covs = [c for c in self._covariates if c != cov]
            terms = [f"{factor_terms}"]
            terms.append(f"C({factor}):{cov}_centered")
            terms.append(f"{cov}_centered")
            if other_covs:
                terms.extend(other_covs)
            
            formula = f"{self._dv} ~ {' + '.join(terms)}"
            try:
                fit_centered = smf.ols(formula, data=df_centered).fit()
                for param in fit_centered.params.index:
                    if f"C({factor})" in param and ":" not in param and param != "Intercept":
                        simple_slopes.append({
                            "covariate_value": val,
                            "covariate_label": label,
                            "parameter": param,
                            "coefficient": float(fit_centered.params[param]),
                            "std_err": float(fit_centered.bse[param]),
                            "t_value": float(fit_centered.tvalues[param]),
                            "p_value": float(fit_centered.pvalues[param]),
                            "ci_lower": float(fit_centered.conf_int().loc[param, 0]),
                            "ci_upper": float(fit_centered.conf_int().loc[param, 1]),
                        })
            except Exception as e:
                logger.error(f"Error in simple slopes at {label}: {e}")

        # 2. Johnson-Neyman Interval
        jn_result = None
        levels = df_clean[factor].unique()
        if len(levels) == 2:
            try:
                factor_terms = f"C({factor})"
                other_between = [f for f in self._between_factors if f != factor]
                other_covs = [c for c in self._covariates if c != cov]
                
                terms = [f"{factor_terms} * {cov}"]
                if other_between:
                    terms.append(" * ".join([f"C({f})" for f in other_between]))
                if other_covs:
                    terms.extend(other_covs)
                    
                formula = f"{self._dv} ~ {' + '.join(terms)}"
                fit_int = smf.ols(formula, data=df_clean).fit()
                
                dummy_term = None
                interaction_term = None
                for param in fit_int.params.index:
                    if f"C({factor})" in param and ":" not in param:
                        dummy_term = param
                    elif f"C({factor})" in param and cov in param:
                        interaction_term = param
                        
                if dummy_term and interaction_term:
                    beta1 = fit_int.params[dummy_term]
                    beta3 = fit_int.params[interaction_term]
                    
                    var_beta1 = fit_int.cov_params().loc[dummy_term, dummy_term]
                    var_beta3 = fit_int.cov_params().loc[interaction_term, interaction_term]
                    cov_beta1_beta3 = fit_int.cov_params().loc[dummy_term, interaction_term]
                    
                    df_resid = fit_int.df_resid
                    t_crit = scipy_t.ppf(1 - self._alpha / 2, df_resid)
                    t_crit_sq = t_crit ** 2
                    
                    a = (beta3 ** 2) - (t_crit_sq * var_beta3)
                    b = 2 * beta1 * beta3 - 2 * t_crit_sq * cov_beta1_beta3
                    c = (beta1 ** 2) - (t_crit_sq * var_beta1)
                    
                    discriminant = b ** 2 - 4 * a * c
                    if discriminant >= 0:
                        root1 = (-b - np.sqrt(discriminant)) / (2 * a)
                        root2 = (-b + np.sqrt(discriminant)) / (2 * a)
                        roots = sorted([float(root1), float(root2)])
                        
                        cov_min = float(np.min(cov_vals))
                        cov_max = float(np.max(cov_vals))
                        
                        test_vals = [roots[0] - 1.0, (roots[0] + roots[1]) / 2.0, roots[1] + 1.0]
                        sig_regions = []
                        for val in test_vals:
                            se_val = np.sqrt(var_beta1 + (val ** 2) * var_beta3 + 2 * val * cov_beta1_beta3)
                            t_val = (beta1 + beta3 * val) / se_val
                            p_val = 2 * (1 - scipy_t.cdf(abs(t_val), df_resid))
                            if p_val < self._alpha:
                                if val < roots[0]:
                                    sig_regions.append(f"{cov} < {roots[0]:.4f}")
                                elif val > roots[1]:
                                    sig_regions.append(f"{cov} > {roots[1]:.4f}")
                                else:
                                    sig_regions.append(f"{roots[0]:.4f} < {cov} < {roots[1]:.4f}")
                                    
                        jn_result = {
                            "roots": roots,
                            "significant_regions": sig_regions,
                            "covariate_min": cov_min,
                            "covariate_max": cov_max,
                        }
            except Exception as e:
                logger.error(f"Error calculating Johnson-Neyman: {e}")
                
        return {
            "simple_slopes": simple_slopes,
            "johnson_neyman": jn_result,
            "covariate_name": cov,
            "factor_name": factor
        }

    def as_results_dict(self):
        if self.result is None:
            return {"error": "Model not fitted"}

        slope_homogeneity = self.check_regression_slope_homogeneity()
        adj_means = self.adjusted_means()

        # EMM post-hoc contrasts on the primary between factor. Default to the
        # vs-control family (EMM + multivariate-t) when a control group is
        # identifiable; otherwise fall back to all-pairwise (EMM + Holm) so the
        # pipeline never crashes on missing control metadata.
        emm_comparisons = []
        posthoc_label = None
        try:
            primary = self._between_factors[0]
            primary_levels = list(pd.unique(self._df[primary]))
            # Resolve the control label dtype-safely: the UI hands back a string,
            # but the actual level values may be numeric.
            ctrl = None
            if self._control_group is not None:
                ctrl = next(
                    (lvl for lvl in primary_levels
                     if str(lvl) == str(self._control_group)),
                    None,
                )
            if ctrl is not None:
                emm_comparisons = self.emm_contrasts(method="vs_control", control_group=ctrl)
                posthoc_label = f"EMM contrasts vs control '{ctrl}' (multivariate-t)"
            elif len(primary_levels) >= 2:
                emm_comparisons = self.emm_contrasts(method="pairwise")
                posthoc_label = "EMM pairwise contrasts (Holm-Bonferroni)"
        except Exception as exc:
            logger.error(f"Error computing ANCOVA EMM contrasts: {exc}")
            emm_comparisons = []

        # Determine if slopes are heterogeneous (any interaction p < alpha)
        slopes_heterogeneous = any(v.get("p_value") is not None and v.get("p_value") < self._alpha for v in slope_homogeneity.values())
        
        simple_slopes_analysis = None
        if slopes_heterogeneous:
            simple_slopes_analysis = self.run_simple_slopes_and_jn()

        anova_rows = []
        if self.anova_table is not None:
            for idx, row in self.anova_table.iterrows():
                anova_rows.append({
                    "source": str(idx),
                    "sum_sq": float(row.get("sum_sq", 0)),
                    "df": float(row.get("df", 0)),
                    "F": float(row.get("F", 0)) if pd.notna(row.get("F")) else None,
                    "p_value": float(row.get("PR(>F)", 1)) if pd.notna(row.get("PR(>F)")) else None,
                })

        covariate_effects = []
        for cov in self._covariates:
            if cov in self.result.params.index:
                covariate_effects.append({
                    "covariate": cov,
                    "coefficient": float(self.result.params[cov]),
                    "std_err": float(self.result.bse[cov]),
                    "t_value": float(self.result.tvalues[cov]),
                    "p_value": float(self.result.pvalues[cov]),
                    "ci_lower": float(self.result.conf_int().loc[cov, 0]),
                    "ci_upper": float(self.result.conf_int().loc[cov, 1]),
                })

        # Extract main effect p-value for the primary between factor
        main_p = None
        main_f = None
        primary_factor = self._between_factors[0]
        # C1: match the factor key from the Sum contrast formula
        factor_key = f"C({primary_factor}, Sum)"
        if self.anova_table is not None and factor_key in self.anova_table.index:
            main_p = float(self.anova_table.loc[factor_key, "PR(>F)"])
            main_f = float(self.anova_table.loc[factor_key, "F"])

        # Compute partial eta-squared for main factor (ss_factor / (ss_factor + ss_residual))
        eta_sq = None
        if self.anova_table is not None and factor_key in self.anova_table.index:
            ss_factor = self.anova_table.loc[factor_key, "sum_sq"]
            residual_key = next((k for k in self.anova_table.index if "residual" in k.lower()), None)
            if residual_key is not None:
                ss_residual = self.anova_table.loc[residual_key, "sum_sq"]
                denom = ss_factor + ss_residual
                if denom > 0:
                    eta_sq = float(ss_factor / denom)

        res = {
            "design_type": self.design_type.value,
            "test": "ANCOVA" if len(self._between_factors) == 1 else "Two-Way ANCOVA",
            "model_type": "ANCOVA",
            "p_value": main_p,
            "statistic": main_f,
            "effect_size": eta_sq,
            "effect_size_type": "partial_eta_squared",
            "anova_table": anova_rows,
            "covariate_effects": covariate_effects,
            "adjusted_means": adj_means,
            "pairwise_comparisons": emm_comparisons,
            "posthoc_test": posthoc_label,
            "slope_homogeneity": slope_homogeneity,
            "slopes_heterogeneous": slopes_heterogeneous,
            "simple_slopes_analysis": simple_slopes_analysis,
            "r_squared": float(self.result.rsquared),
            "r_squared_adj": float(self.result.rsquared_adj),
            "aic": float(self.result.aic),
            "bic": float(self.result.bic),
            "n_observations": int(self.result.nobs),
            "covariates_used": self._covariates,
            "between_factors": self._between_factors,
        }
        return _restore_names_in_dict(res, self._rev_map)


class LinearMixedModel(BaseStatisticalModel):
    """LMM for longitudinal clinical data via statsmodels MixedLM.

    Fits Random Intercept and compares with Random Intercept + Random Slope via LRT.
    Applies Between-Within degrees of freedom correction for fixed effects if N < 100.
    """

    def __init__(self):
        self._rev_map = {}
        self.result = None
        self._df = None
        self._dv = None
        self._fixed_effects = None
        self._random_intercept = None
        self._random_slope = None
        self._covariates = None
        self._lrt_performed = False
        self._lrt_stat = None
        self._lrt_p = None
        self._random_structure_chosen = "Random Intercept Only"
        self._alpha = 0.05
        self._control_group = None
        self._groups_vals = None

    @property
    def design_type(self) -> DesignType:
        if not hasattr(self, '_df') or self._df is None:
            return DesignType.REPEATED
        
        between_cols = set()
        for col in self._fixed_effects + (self._covariates or []):
            vals = self._df[col].values
            is_constant = True
            for g_id in self._df[self._random_intercept].unique():
                mask = (self._df[self._random_intercept].values == g_id)
                g_vals = vals[mask]
                if len(g_vals) > 1 and not np.all(g_vals == g_vals[0]):
                    is_constant = False
                    break
            if is_constant:
                between_cols.add(col)
        return DesignType.MIXED if between_cols else DesignType.REPEATED

    def fit(self, df, dv, fixed_effects, random_intercept, covariates=None, random_slope=None, alpha=0.05, control_group=None):
        import statsmodels.formula.api as smf
        from scipy import stats as scipy_stats

        self._alpha = alpha
        self._control_group = control_group

        all_cols = [dv, random_intercept] + fixed_effects + (covariates or [])
        if random_slope and random_slope not in all_cols:
            all_cols.append(random_slope)
        self._df = df.dropna(subset=all_cols).copy()

        col_map = _sanitize_columns(self._df, all_cols)
        self._rev_map = {v: k for k, v in col_map.items()}
        self._dv = col_map[dv]
        self._fixed_effects = [col_map[f] for f in fixed_effects]
        self._random_intercept = col_map[random_intercept]
        self._random_slope = col_map[random_slope] if random_slope else None
        self._covariates = [col_map[c] for c in (covariates or [])]
        self._groups_vals = self._df[self._random_intercept].values

        terms = []
        if self._fixed_effects:
            terms.append(" * ".join([f"C({f})" for f in self._fixed_effects]))
        if self._covariates:
            terms.extend(self._covariates)
        fixed_terms = " + ".join(terms) if terms else "1"
        formula = f"{self._dv} ~ {fixed_terms}"

        # 1. Fit Random Intercept model (baseline, REML)
        model_ri = smf.mixedlm(formula, data=self._df, groups=self._df[self._random_intercept])
        fit_ri = model_ri.fit(reml=True)

        self._lrt_performed = False
        self._lrt_stat = None
        self._lrt_p = None
        self._random_structure_chosen = "Random Intercept Only"
        self.result = fit_ri

        # 2. If random slope candidate is present, fit RI + RS and compare
        if self._random_slope:
            try:
                re_formula = f"~{self._random_slope}"
                model_ri_rs = smf.mixedlm(formula, data=self._df,
                                          groups=self._df[self._random_intercept],
                                          re_formula=re_formula)
                fit_ri_rs = model_ri_rs.fit(reml=True)

                if fit_ri_rs.converged:
                    ll_ri = fit_ri.llf
                    ll_ri_rs = fit_ri_rs.llf
                    D = 2 * (ll_ri_rs - ll_ri)
                    # Diagonal RE structure adds 1 parameter (variance of slope) — df=1
                    df_diff = 1
                    p_val = float(scipy_stats.chi2.sf(D, df_diff))
                    
                    self._lrt_performed = True
                    self._lrt_stat = float(D)
                    self._lrt_p = p_val
                    
                    if p_val < 0.05 and D > 0:
                        self.result = fit_ri_rs
                        self._random_structure_chosen = "Random Intercept + Random Slope"
                    else:
                        self.result = fit_ri
                        self._random_structure_chosen = "Random Intercept Only"
            except Exception as e:
                logger.info(f"LMM fit with random slope failed/did not converge: {e}. Falling back to Random Intercept only.")
                self.result = fit_ri
                self._random_structure_chosen = "Random Intercept Only (Fallback)"

        return self

    def icc(self):
        """Compute Intraclass Correlation Coefficient."""
        if self.result is None:
            return None
        try:
            re_var = float(self.result.cov_re.iloc[0, 0])
            resid_var = float(self.result.scale)
            total_var = re_var + resid_var
            if total_var > 0:
                return re_var / total_var
        except Exception:
            pass
        return None

    def as_results_dict(self):
        if self.result is None:
            return {"error": "Model not fitted"}

        n_groups = self._df[self._random_intercept].nunique()
        n_obs = len(self._df)

        # Classify predictor columns as Between-Subject or Within-Subject
        between_cols = set()
        within_cols = set()
        for col in self._fixed_effects + (self._covariates or []):
            vals = self._df[col].values
            is_constant = True
            for g_id in self._df[self._random_intercept].unique():
                mask = (self._df[self._random_intercept].values == g_id)
                g_vals = vals[mask]
                if len(g_vals) > 1 and not np.all(g_vals == g_vals[0]):
                    is_constant = False
                    break
            if is_constant:
                between_cols.add(col)
            else:
                within_cols.add(col)
                
        n_between_predictors = len(between_cols)
        n_within_predictors = len(within_cols)
        
        apply_correction = (n_groups < 100)
        df_method = "Between-Within (Kenward-Roger / Satterthwaite approximation)" if apply_correction else "Asymptotic (z-test)"

        # Fixed effects table with BW degrees of freedom correction
        fixed_effects_table = []
        for param_name in self.result.fe_params.index:
            coef = float(self.result.fe_params[param_name])
            se = float(self.result.bse_fe[param_name])
            z_val = float(self.result.tvalues[param_name])
            
            if apply_correction:
                if param_name == "Intercept":
                    df_param = n_groups - 1
                else:
                    is_within = False
                    for col in within_cols:
                        if col in param_name:
                            is_within = True
                            break
                    if is_within:
                        df_param = n_obs - n_groups - n_within_predictors
                    else:
                        df_param = n_groups - 1 - n_between_predictors
                
                df_param = max(1, df_param)
                from scipy.stats import t as scipy_t
                p_val = float(2 * (1 - scipy_t.cdf(abs(z_val), df_param)))
                t_crit = scipy_t.ppf(0.975, df_param)
                ci_lower = coef - t_crit * se
                ci_upper = coef + t_crit * se
            else:
                df_param = None
                p_val = float(self.result.pvalues[param_name])
                ci_lower = float(self.result.conf_int().loc[param_name, 0])
                ci_upper = float(self.result.conf_int().loc[param_name, 1])

            fixed_effects_table.append({
                "parameter": str(param_name),
                "coefficient": coef,
                "std_err": se,
                "z_value": z_val,
                "df": df_param,
                "p_value": p_val,
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
            })

        # Random effects variance
        re_var = None
        resid_var = None
        try:
            re_var = float(self.result.cov_re.iloc[0, 0])
            resid_var = float(self.result.scale)
        except Exception:
            pass

        # Extract p-values for main fixed effects
        main_p = None
        main_z = None
        for fe in self._fixed_effects:
            for entry in fixed_effects_table:
                param_name = entry["parameter"]
                if f"C({fe})" in param_name and ":" not in param_name and param_name != "Intercept":
                    main_p = entry["p_value"]
                    main_z = entry["z_value"]
                    break
            if main_p is not None:
                break

        icc_val = self.icc()

        # EMM post-hoc contrasts on the primary fixed effect factor.
        emm_comparisons = []
        posthoc_label = None
        try:
            if self._fixed_effects:
                primary = self._fixed_effects[0]
                primary_levels = list(pd.unique(self._df[primary]))
                ctrl = None
                if self._control_group is not None:
                    ctrl = next(
                        (lvl for lvl in primary_levels
                         if str(lvl) == str(self._control_group)),
                        None,
                    )
                if ctrl is not None:
                    emm_comparisons = self.emm_contrasts(method="vs_control", control_group=ctrl)
                    posthoc_label = f"EMM contrasts vs control '{ctrl}' (multivariate-t)"
                elif len(primary_levels) >= 2:
                    emm_comparisons = self.emm_contrasts(method="pairwise")
                    posthoc_label = "EMM pairwise contrasts (Holm-Bonferroni)"
        except Exception as exc:
            logger.error(f"Error computing LMM EMM contrasts: {exc}")
            emm_comparisons = []

        res = {
            "design_type": self.design_type.value,
            "test": "Linear Mixed Model",
            "model_type": "LMM",
            "p_value": main_p,
            "statistic": main_z,
            "statistic_type": "z",
            "effect_size": icc_val,
            "effect_size_type": "ICC",
            "fixed_effects_table": fixed_effects_table,
            "random_effects_variance": re_var,
            "residual_variance": resid_var,
            "icc": icc_val,
            "aic": float(self.result.aic) if hasattr(self.result, 'aic') else None,
            "bic": float(self.result.bic) if hasattr(self.result, 'bic') else None,
            "log_likelihood": float(self.result.llf) if hasattr(self.result, 'llf') else None,
            "converged": getattr(self.result, 'converged', None),
            "n_subjects": n_groups,
            "n_observations": n_obs,
            "fixed_effects_used": self._fixed_effects,
            "between_effects": list(between_cols),
            "within_effects": list(within_cols),
            "random_intercept": self._random_intercept,
            "covariates_used": self._covariates,
            "lrt_performed": self._lrt_performed,
            "lrt_statistic": self._lrt_stat,
            "lrt_p_value": self._lrt_p,
            "random_structure_chosen": self._random_structure_chosen,
            "df_method": df_method,
            "pairwise_comparisons": emm_comparisons,
            "posthoc_test": posthoc_label,
        }
        return _restore_names_in_dict(res, self._rev_map)

    def emm_contrasts(self, method="vs_control", control_group=None, factor=None):
        """EMM post-hoc contrasts on the fixed effects factors, computed via
        manually evaluated contrasts on the fitted MixedLM model.
        """
        if self.result is None:
            return None
        factor = factor or self._fixed_effects[0]
        levels = list(pd.unique(self._df[factor]))
        if len(levels) < 2:
            return None

        # 1. Classify factors as Between-Subject or Within-Subject for degrees of freedom
        between_cols = set()
        within_cols = set()
        for col in self._fixed_effects + (self._covariates or []):
            vals = self._df[col].values
            is_constant = True
            for g_id in np.unique(self._groups_vals):
                mask = (self._groups_vals == g_id)
                g_vals = vals[mask]
                if len(g_vals) > 1 and not np.all(g_vals == g_vals[0]):
                    is_constant = False
                    break
            if is_constant:
                between_cols.add(col)
            else:
                within_cols.add(col)

        # 2. Determine k_between and k_within from Fixed-Effects design matrix X
        X = self.result.model.exog
        param_names = self.result.fe_params.index.tolist()
        group_ids = self._groups_vals
        
        k_between = 0
        k_within = 0
        for i, col_name in enumerate(param_names):
            if col_name == "Intercept":
                continue
            col_vals = X[:, i]
            is_constant_within_clusters = True
            for g_id in np.unique(group_ids):
                g_vals = col_vals[group_ids == g_id]
                if len(g_vals) > 1 and not np.all(g_vals == g_vals[0]):
                    is_constant_within_clusters = False
                    break
            if is_constant_within_clusters:
                k_between += 1
            else:
                k_within += 1

        n_groups = len(np.unique(self._groups_vals))
        n_obs = len(self._df)

        if factor in within_cols:
            df_bw = n_obs - n_groups - k_within
        else:
            df_bw = n_groups - 1 - k_between
        df_bw = max(1, df_bw)

        # 3. Build EMM functionals for each level
        functionals = {lvl: self._emm_functional(factor, lvl) for lvl in levels}
        
        # 4. Get Fixed Effects coefficients and Fixed Effects covariance submatrix
        beta = np.asarray(self.result.fe_params.values, dtype=float)
        cov_params = np.asarray(self.result.cov_params().iloc[:self.result.k_fe, :self.result.k_fe].values, dtype=float)

        if method == "vs_control":
            if control_group is None or control_group not in levels:
                raise ValueError("vs_control requires a control_group present in the data")
            treatments = [lvl for lvl in levels if lvl != control_group]
            pairs = [(trt, control_group) for trt in treatments]
        elif method == "pairwise":
            from itertools import combinations as _combinations
            pairs = list(_combinations(levels, 2))
        else:
            raise ValueError(f"unknown contrast method {method!r}")

        Lmat = np.vstack([functionals[a] - functionals[b] for a, b in pairs])
        est = Lmat @ beta
        cov_c = Lmat @ cov_params @ Lmat.T
        se = np.sqrt(np.clip(np.diag(cov_c), 0.0, None))
        with np.errstate(divide="ignore", invalid="ignore"):
            t_values = np.where(se > 0, est / se, 0.0)

        if method == "vs_control":
            d = np.sqrt(np.clip(np.diag(cov_c), 1e-300, None))
            R = cov_c / np.outer(d, d)
            np.fill_diagonal(R, 1.0)
            p_adj = self._mvt_pvalues(list(t_values), R, df_bw)
        else:
            from scipy.stats import t as student_t
            from statsmodels.stats.multitest import multipletests
            p_raw = [float(2.0 * student_t.sf(abs(tv), df_bw)) for tv in t_values]
            _, p_adj, _, _ = multipletests(p_raw, alpha=self._alpha, method="holm")
            p_adj = list(p_adj)

        contrasts = []
        for (a, b), e, s, tv, p in zip(pairs, est, se, t_values, p_adj):
            contrasts.append({
                "group1": str(a),
                "group2": str(b),
                "estimate": float(e),
                "std_err": float(s),
                "statistic": float(tv),
                "p_value": float(p),
                "significant": bool(p < self._alpha),
                "test": "LMM EMM Contrast",
                "df": float(df_bw),
                "corrected": True,
                "correction": "multivariate-t" if method == "vs_control" else "Holm-Bonferroni"
            })
        return contrasts

    def _emm_functional(self, factor, level):
        """Linear functional L (length = n model params) whose dot product with
        the Fixed Effects coefficients yields the EMM of ``level`` for ``factor``.
        """
        from itertools import product as _product
        from patsy import dmatrix

        cov_means = {c: self._df[c].mean() for c in self._covariates}
        other_factors = [f for f in self._fixed_effects if f != factor]
        other_levels = [self._df[f].unique() for f in other_factors]

        grid_rows = []
        for combo in (_product(*other_levels) if other_factors else [()]):
            row = {factor: level}
            row.update(dict(zip(other_factors, combo)))
            row.update(cov_means)
            grid_rows.append(row)
        grid = pd.DataFrame(grid_rows)

        design_info = self.result.model.data.design_info
        X = np.asarray(dmatrix(design_info, grid, return_type="matrix"))
        return X.mean(axis=0)

    @staticmethod
    def _mvt_pvalues(t_values, R, df):
        """Single-step two-sided multivariate-t adjusted p-values for a family
        of contrasts with correlation matrix ``R`` and ``df`` residual df.
        """
        from scipy.stats import multivariate_t, t as student_t

        k = len(t_values)
        if k == 1:
            return [float(2.0 * student_t.sf(abs(t_values[0]), df))]
        rv = multivariate_t(loc=np.zeros(k), shape=R, df=df, allow_singular=True)
        out = []
        for tv in t_values:
            c = abs(float(tv))
            p_all = float(rv.cdf(np.full(k, c), lower_limit=np.full(k, -c)))
            out.append(float(min(1.0, max(0.0, 1.0 - p_all))))
        return out


class LogisticRegressionModel(BaseStatisticalModel):
    """Logistic regression for binary outcomes via statsmodels GLM(Binomial) with Firth fallback."""

    def __init__(self):
        self._rev_map = {}
        self.result = None
        self._df = None
        self._dv = None
        self._predictors = None
        self._covariates = None
        self._model_variant = "Standard Maximum Likelihood"
        self._firth_coefs = None
        self._firth_bse = None
        self._firth_cov = None

    @property
    def design_type(self) -> DesignType:
        return DesignType.INDEPENDENT

    def fit(self, df, dv, predictors, covariates=None):
        import statsmodels.formula.api as smf
        import statsmodels.api as sm

        all_cols = [dv] + predictors + (covariates or [])
        self._df = df.dropna(subset=all_cols).copy()

        col_map = _sanitize_columns(self._df, all_cols)
        self._rev_map = {v: k for k, v in col_map.items()}
        self._dv = col_map[dv]
        self._predictors = [col_map[p] for p in predictors]
        self._covariates = [col_map[c] for c in (covariates or [])]

        # Encode DV as 0/1 if needed
        unique_vals = sorted(self._df[self._dv].unique())
        if len(unique_vals) != 2:
            raise ValueError(f"Logistic regression requires exactly 2 outcome levels, found {len(unique_vals)}")
        if set(unique_vals) != {0, 1}:
            self._df[self._dv] = (self._df[self._dv] == unique_vals[1]).astype(int)

        terms = [f"C({p})" for p in self._predictors]
        if self._covariates:
            terms.extend(self._covariates)
        formula = f"{self._dv} ~ {' + '.join(terms)}"

        model = smf.glm(formula, data=self._df, family=sm.families.Binomial())
        self.result = model.fit()

        SEPARATION_BSE_THRESHOLD = 5.0
        # Check standard errors of coefficients to detect separation
        has_large_se = False
        try:
            for param in self.result.params.index:
                if self.result.bse[param] > SEPARATION_BSE_THRESHOLD:
                    has_large_se = True
                    break
        except Exception:
            pass

        self._model_variant = "Standard Maximum Likelihood"
        self._firth_coefs = None
        self._firth_bse = None
        self._firth_cov = None
        self._firth_plr_pvals = {}
        self._firth_failed = False

        if not self.result.converged or has_large_se:
            # Fallback to Firth Penalized Likelihood
            try:
                coefs, bse, cov = self._fit_firth_logistic(model.exog, model.endog)
                self._model_variant = "Firth Penalized Likelihood"
                self._firth_coefs = coefs
                self._firth_bse = bse
                self._firth_cov = cov
                # Penalized likelihood-ratio p-values (logistf default inference;
                # Wald is unreliable in the separation settings Firth targets).
                self._firth_plr_pvals = {}
                for idx, param in enumerate(self.result.params.index):
                    if param == "Intercept":
                        continue
                    try:
                        self._firth_plr_pvals[str(param)] = self._firth_plr_pvalue(
                            model.exog, model.endog, coefs, idx
                        )
                    except Exception:
                        pass  # odds_ratios falls back to Wald for this parameter
            except Exception as e:
                self._firth_failed = True
                logger.warning(f"WARNING: Firth solver failed/did not converge: {e}. Keeping standard logit results.")

        return self

    @staticmethod
    def _penalized_loglik(X, y, beta):
        """Firth-penalized log-likelihood: ll + 0.5*log|X'WX| (Jeffreys prior)."""
        pi = 1.0 / (1.0 + np.exp(-np.dot(X, beta)))
        pi = np.clip(pi, 1e-12, 1.0 - 1e-12)
        ll = float(np.sum(y * np.log(pi) + (1.0 - y) * np.log(1.0 - pi)))
        w = pi * (1.0 - pi)
        xtwx = np.dot(X.T, w[:, None] * X)
        sign, logdet = np.linalg.slogdet(xtwx)
        if sign <= 0:
            return -np.inf
        return ll + 0.5 * float(logdet)

    def _fit_firth_logistic(self, X, y, max_iter=100, tol=1e-6, fixed_zero=None,
                            fixed_values=None):
        """Fit Firth Penalized Likelihood Logistic Regression using Newton-Raphson.

        fixed_zero: optional set of column indices constrained to beta_j = 0
        (used by the penalized likelihood-ratio test).
        fixed_values: optional dict {col_index: value} constraining beta_j to an
        arbitrary value (used by the profile-likelihood CI). The Jeffreys penalty
        is always computed from the FULL design matrix, as in R's logistf.
        """
        n, p = X.shape
        beta = np.zeros(p)
        fixed = set(fixed_zero or ())
        if fixed_values:
            for j, val in fixed_values.items():
                beta[j] = val
                fixed.add(j)
        free = np.array([j for j in range(p) if j not in fixed], dtype=int)

        converged = False
        for iteration in range(max_iter):
            pi = 1.0 / (1.0 + np.exp(-np.dot(X, beta)))
            pi = np.clip(pi, 1e-12, 1.0 - 1e-12)

            w = pi * (1.0 - pi)
            xtwx = np.dot(X.T, w[:, None] * X)
            try:
                xtwx_inv = np.linalg.inv(xtwx)
            except np.linalg.LinAlgError:
                xtwx_inv = np.linalg.pinv(xtwx)

            # Hat matrix diagonal (full model — the penalty term needs it)
            h = np.einsum('ij,jk,ik->i', X, xtwx_inv, X) * w

            # Firth-modified score
            score = np.dot(X.T, (y - pi + h * (0.5 - pi)))

            if np.all(np.abs(score[free]) < tol):
                converged = True
                break

            # Newton step restricted to free parameters
            if len(free) == p:
                step_free = np.dot(xtwx_inv, score)
            else:
                sub = xtwx[np.ix_(free, free)]
                try:
                    step_free = np.linalg.solve(sub, score[free])
                except np.linalg.LinAlgError:
                    step_free = np.dot(np.linalg.pinv(sub), score[free])

            step = np.zeros(p)
            if len(free) == p:
                step = step_free
            else:
                step[free] = step_free

            # Step-halving line search on the penalized log-likelihood
            pll_old = self._penalized_loglik(X, y, beta)
            step_len = 1.0
            for _ in range(10):
                beta_new = beta + step_len * step
                if self._penalized_loglik(X, y, beta_new) >= pll_old:
                    break
                step_len *= 0.5
            beta = beta + step_len * step

            # Secondary convergence criterion: parameter change negligible
            if np.max(np.abs(step_len * step)) < tol:
                converged = True
                break

        if not converged:
            # Accept if the score is small on a practical scale (quasi-separation
            # flattens the likelihood; logistf behaves the same way).
            if np.all(np.abs(score[free]) < 1e-3):
                converged = True
        if not converged:
            raise RuntimeError("Firth solver did not converge within max iterations.")

        # Recompute standard errors
        pi = 1.0 / (1.0 + np.exp(-np.dot(X, beta)))
        pi = np.clip(pi, 1e-12, 1.0 - 1e-12)
        w = pi * (1.0 - pi)
        xtwx = np.dot(X.T, w[:, None] * X)
        try:
            cov = np.linalg.inv(xtwx)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(xtwx)

        bse = np.sqrt(np.diag(cov))
        return beta, bse, cov

    def _firth_plr_pvalue(self, X, y, beta_full, j):
        """Penalized likelihood-ratio test p-value for H0: beta_j = 0.

        PLR = 2*[pl(beta_hat) - max_{beta_j=0} pl(beta)] ~ chi2(1),
        the default inference in R's logistf (Heinze & Schemper 2002).
        """
        from scipy.stats import chi2
        pll_full = self._penalized_loglik(X, y, beta_full)
        beta_constrained, _, _ = self._fit_firth_logistic(X, y, fixed_zero={j})
        pll_constrained = self._penalized_loglik(X, y, beta_constrained)
        stat = max(2.0 * (pll_full - pll_constrained), 0.0)
        return float(chi2.sf(stat, 1))

    def _firth_profile_ci(self, X, y, beta_full, j, alpha=0.05):
        """Penalized profile-likelihood CI for beta_j (logistf default).

        The (1-alpha) interval is the set of values c with
        2*[pl(beta_hat) - max_{beta_j=c} pl(beta)] <= chi2(1, 1-alpha).
        Each bound is the c where that deviance gap equals the critical value;
        found by expanding outward from beta_hat_j then bracketing with brentq.
        More reliable than the Wald interval under (quasi-)separation.
        """
        from scipy.stats import chi2
        from scipy.optimize import brentq

        pll_hat = self._penalized_loglik(X, y, beta_full)
        crit = float(chi2.ppf(1 - alpha, 1))

        def deviance_gap(value):
            beta_c, _, _ = self._fit_firth_logistic(X, y, fixed_values={j: value})
            pll_c = self._penalized_loglik(X, y, beta_c)
            return 2.0 * (pll_hat - pll_c) - crit

        point = float(beta_full[j])
        se = None
        if self._firth_bse is not None:
            se_j = self._firth_bse[j]
            if np.isfinite(se_j) and se_j > 0:
                se = float(se_j)
        step = max(se or 1.0, 0.5)

        def find_bound(direction):
            c_in = point
            c_out = point + direction * step
            for _ in range(60):
                if deviance_gap(c_out) > 0:
                    return float(brentq(deviance_gap, c_in, c_out, xtol=1e-6))
                c_in, c_out = c_out, c_out + direction * step
            return float("nan")

        return find_bound(-1), find_bound(+1)

    def predict(self):
        """Get predicted probabilities from the model (handles Firth adjustment)."""
        if self.result is None:
            return None
        if self._model_variant == "Firth Penalized Likelihood" and self._firth_coefs is not None:
            exog = self.result.model.exog
            logits = np.dot(exog, self._firth_coefs)
            p = 1.0 / (1.0 + np.exp(-logits))
            return np.clip(p, 1e-15, 1.0 - 1e-15)
        else:
            return self.result.predict()

    def odds_ratios(self):
        """Compute odds ratios with 95% CI."""
        if self.result is None:
            return []

        rows = []
        if self._model_variant == "Firth Penalized Likelihood" and self._firth_coefs is not None:
            from scipy.stats import norm
            z_crit = float(norm.ppf(0.975))
            plr_pvals = getattr(self, "_firth_plr_pvals", {}) or {}
            for idx, param in enumerate(self.result.params.index):
                if param == "Intercept":
                    continue
                coef = self._firth_coefs[idx]
                se = self._firth_bse[idx]
                z_val = coef / se if se > 0 else 0.0
                p_wald = 2 * (1 - norm.cdf(abs(z_val)))
                # Prefer the penalized LR p-value (logistf default); Wald only
                # as fallback when the constrained refit failed.
                p_plr = plr_pvals.get(str(param))
                # Penalized profile-likelihood CI (logistf default); Wald only
                # as fallback when the profiling root-search fails. Wald is
                # unreliable in the separation settings Firth targets.
                ci_method = "Profile likelihood"
                try:
                    lo, hi = self._firth_profile_ci(
                        self.result.model.exog, self.result.model.endog,
                        self._firth_coefs, idx, alpha=0.05,
                    )
                    if not (np.isfinite(lo) and np.isfinite(hi)):
                        raise ValueError("profile CI did not bracket")
                    ci_lower, ci_upper = lo, hi
                except Exception:
                    ci_lower = coef - z_crit * se
                    ci_upper = coef + z_crit * se
                    ci_method = "Wald"
                rows.append({
                    "parameter": str(param),
                    "odds_ratio": float(np.exp(coef)),
                    "ci_lower": float(np.exp(ci_lower)),
                    "ci_upper": float(np.exp(ci_upper)),
                    "ci_method": ci_method,
                    "coefficient": float(coef),
                    "std_err": float(se),
                    "z_value": float(z_val),
                    "p_value": float(p_plr) if p_plr is not None else float(p_wald),
                    "p_value_method": (
                        "PLR (penalized likelihood ratio)" if p_plr is not None else "Wald"
                    ),
                })
        else:
            conf = self.result.conf_int()
            for param in self.result.params.index:
                if param == "Intercept":
                    continue
                rows.append({
                    "parameter": str(param),
                    "odds_ratio": float(np.exp(self.result.params[param])),
                    "ci_lower": float(np.exp(conf.loc[param, 0])),
                    "ci_upper": float(np.exp(conf.loc[param, 1])),
                    "coefficient": float(self.result.params[param]),
                    "std_err": float(self.result.bse[param]),
                    "z_value": float(self.result.tvalues[param]),
                    "p_value": float(self.result.pvalues[param]),
                })
        return rows

    def hosmer_lemeshow(self, n_groups=10):
        """Hosmer-Lemeshow goodness-of-fit test (deprecated but kept for backward compatibility)."""
        from scipy import stats as scipy_stats
        if self.result is None:
            return {"chi2": None, "df": None, "p_value": None}
        try:
            predicted = self.predict()
            observed = self._df[self._dv].values
            order = np.argsort(predicted)
            predicted_sorted = predicted[order]
            observed_sorted = observed[order]
            groups = np.array_split(np.arange(len(predicted_sorted)), n_groups)
            chi2 = 0.0
            for group_indices in groups:
                if len(group_indices) == 0:
                    continue
                obs = observed_sorted[group_indices]
                pred = predicted_sorted[group_indices]
                n_g = len(group_indices)
                obs_events = obs.sum()
                exp_events = pred.sum()
                obs_non = n_g - obs_events
                exp_non = n_g - exp_events
                if exp_events > 0:
                    chi2 += (obs_events - exp_events) ** 2 / exp_events
                if exp_non > 0:
                    chi2 += (obs_non - exp_non) ** 2 / exp_non
            df = n_groups - 2
            p_value = float(scipy_stats.chi2.sf(chi2, df))
            return {"chi2": float(chi2), "df": df, "p_value": p_value}
        except Exception:
            return {"chi2": None, "df": None, "p_value": None}

    def calibration_analysis(self):
        """Compute Brier score, calibration slope, and calibration intercept in log-odds space."""
        if self.result is None:
            return {
                "brier_score": None,
                "calibration_slope": None,
                "calibration_intercept": None,
                "calibration_curve": {"predicted": [], "observed": []}
            }

        try:
            predicted = self.predict()
            observed = self._df[self._dv].values.astype(float)

            # 1. Brier score
            brier = float(np.mean((predicted - observed) ** 2))

            # 2. Calibration slope and intercept
            p_hat = np.clip(predicted, 1e-15, 1.0 - 1e-15)
            logit_p = np.log(p_hat / (1.0 - p_hat))

            import statsmodels.api as sm
            X_cal = sm.add_constant(logit_p)
            try:
                cal_model = sm.GLM(observed, X_cal, family=sm.families.Binomial()).fit()
                cal_intercept = float(cal_model.params[0])
                cal_slope = float(cal_model.params[1])
            except Exception:
                try:
                    cal_model = sm.OLS(observed, X_cal).fit()
                    cal_intercept = float(cal_model.params[0])
                    cal_slope = float(cal_model.params[1])
                except Exception:
                    cal_intercept = 0.0
                    cal_slope = 1.0

            # 3. LOESS calibration curve
            from statsmodels.nonparametric.smoothers_lowess import lowess
            sort_idx = np.argsort(predicted)
            pred_sorted = predicted[sort_idx]
            obs_sorted = observed[sort_idx]

            lowess_fit = lowess(obs_sorted, pred_sorted, frac=0.6, it=3, return_sorted=False)
            
            calibration_curve = {
                "predicted": [float(p) for p in pred_sorted],
                "observed_smoothed": [float(o) for o in lowess_fit],
            }

            return {
                "brier_score": brier,
                "calibration_slope": cal_slope,
                "calibration_intercept": cal_intercept,
                "calibration_curve": calibration_curve
            }
        except Exception as e:
            logger.error(f"Error in calibration_analysis: {e}")
            return {
                "brier_score": None,
                "calibration_slope": None,
                "calibration_intercept": None,
                "calibration_curve": {"predicted": [], "observed": []}
            }

    def roc_data(self):
        """Compute ROC curve data (FPR, TPR, thresholds) and AUC."""
        if self.result is None:
            return {"fpr": [], "tpr": [], "auc": None}

        predicted = self.predict()
        observed = self._df[self._dv].values

        thresholds = np.sort(np.unique(predicted))[::-1]
        thresholds = np.concatenate([[thresholds[0] + 0.01], thresholds, [0.0]])

        fpr_list = []
        tpr_list = []
        total_pos = observed.sum()
        total_neg = len(observed) - total_pos

        if total_pos == 0 or total_neg == 0:
            return {"fpr": [0, 1], "tpr": [0, 1], "auc": 0.5, "thresholds": [1, 0]}

        for threshold in thresholds:
            pred_pos = predicted >= threshold
            tp = (pred_pos & (observed == 1)).sum()
            fp = (pred_pos & (observed == 0)).sum()
            tpr_list.append(tp / total_pos)
            fpr_list.append(fp / total_neg)

        trapz_fn = getattr(np, 'trapezoid', None) or getattr(np, 'trapz', None)
        auc = float(np.abs(trapz_fn(tpr_list, fpr_list)))

        return {
            "fpr": [float(x) for x in fpr_list],
            "tpr": [float(x) for x in tpr_list],
            "auc": auc,
            "thresholds": [float(x) for x in thresholds],
        }

    def as_results_dict(self):
        if self.result is None:
            return {"error": "Model not fitted"}

        or_table = self.odds_ratios()
        hl = self.hosmer_lemeshow()
        roc = self.roc_data()
        cal = self.calibration_analysis()

        main_p = None
        main_or = None
        if or_table:
            main_p = or_table[0]["p_value"]
            main_or = or_table[0]["odds_ratio"]

        pseudo_r2 = None
        if self._model_variant != "Firth Penalized Likelihood":
            try:
                ll_model = self.result.llf
                ll_null = self.result.llnull if hasattr(self.result, 'llnull') else None
                if ll_null is not None and ll_null != 0:
                    pseudo_r2 = float(1 - ll_model / ll_null)
            except Exception:
                pass

        converged = getattr(self.result, "converged", True) if self._model_variant != "Firth Penalized Likelihood" else True
        if getattr(self, "_firth_failed", False):
            converged = False
        warnings_list = []
        if not converged:
            warnings_list.append("Logistic regression did not converge. Results may be unreliable.")

        res = {
            "design_type": self.design_type.value,
            "test": "Logistic Regression",
            "model_type": "LogisticRegression",
            "converged": converged,
            "warnings": warnings_list,
            "model_variant": self._model_variant,
            "p_value": main_p,
            "statistic": main_or,
            "statistic_type": "odds_ratio",
            "effect_size": roc["auc"],
            "effect_size_type": "AUC",
            "odds_ratios": or_table,
            "hosmer_lemeshow": hl,
            "roc_data": roc,
            "brier_score": cal["brier_score"],
            "calibration_slope": cal["calibration_slope"],
            "calibration_intercept": cal["calibration_intercept"],
            "calibration_curve": cal["calibration_curve"],
            "pseudo_r_squared": pseudo_r2,
            "aic": float(self.result.aic) if (hasattr(self.result, 'aic') and self._model_variant != "Firth Penalized Likelihood") else None,
            "bic": float(self.result.bic_llf) if (hasattr(self.result, 'bic_llf') and self._model_variant != "Firth Penalized Likelihood") else None,
            "log_likelihood": float(self.result.llf) if (hasattr(self.result, 'llf') and self._model_variant != "Firth Penalized Likelihood") else None,
            "n_observations": int(self.result.nobs),
            "predictors_used": self._predictors,
            "covariates_used": self._covariates,
        }
        return _restore_names_in_dict(res, self._rev_map)


class BetaRegressionModel(BaseStatisticalModel):
    """Beta regression for proportion outcomes strictly in (0, 1).

    Uses statsmodels BetaModel (othermod.betareg) with a logit link.
    Reports coefficients, 95% CI, pseudo-R², dispersion parameter (phi),
    and fitted vs. residual data for diagnostics. The main p-value (main_p) 
    is computed via an Omnibus Likelihood-Ratio test.
    """

    def __init__(self):
        self._rev_map = {}
        self.result = None
        self._df = None
        self._dv = None
        self._predictors = None
        self._covariates = None
        self._bias_corrected = False
        self._boot_se = None  # populated when bias_corrected=True via bootstrap

    @property
    def design_type(self) -> DesignType:
        return DesignType.INDEPENDENT

    def fit(self, df, dv, predictors, covariates=None, bias_corrected=False, alpha=0.05):
        from statsmodels.othermod.betareg import BetaModel

        self._bias_corrected = bias_corrected
        self._alpha = alpha
        all_cols = [dv] + predictors + (covariates or [])
        self._df = df.dropna(subset=all_cols).copy()

        col_map = _sanitize_columns(self._df, all_cols)
        self._rev_map = {v: k for k, v in col_map.items()}
        self._dv = col_map[dv]
        self._predictors = [col_map[p] for p in predictors]
        self._covariates = [col_map[c] for c in (covariates or [])]

        # Validate strictly (0, 1) — S-V transformation must have been applied upstream
        series = self._df[self._dv]
        if series.min() <= 0.0 or series.max() >= 1.0:
            raise ValueError(
                "Beta regression requires outcome values strictly between 0 and 1 (exclusive). "
                "Values at exactly 0 or 1 are present — apply a boundary transformation first "
                "(e.g. y_adj = (y * (n-1) + 0.5) / n)."
            )

        terms = [f"C({p})" for p in self._predictors]
        if self._covariates:
            terms.extend(self._covariates)
        formula = f"{self._dv} ~ {' + '.join(terms)}"

        model = BetaModel.from_formula(formula, data=self._df)
        self.result = model.fit(disp=False)

        if bias_corrected:
            self._boot_se = self._bootstrap_se(formula, n_boot=1000)

        return self

    def _bootstrap_se(self, formula, n_boot=1000):
        """Bootstrapped standard errors as bias correction for small samples."""
        from statsmodels.othermod.betareg import BetaModel
        boot_params = []
        rng = np.random.default_rng(42)
        n = len(self._df)
        for _ in range(n_boot):
            try:
                idx = rng.integers(0, n, size=n)
                boot_df = self._df.iloc[idx].reset_index(drop=True)
                m = BetaModel.from_formula(formula, data=boot_df)
                r = m.fit(disp=False, maxiter=100)
                boot_params.append(r.params.values)
            except Exception:
                continue
        if len(boot_params) < 10:
            return None
        boot_arr = np.array(boot_params)
        return dict(zip(self.result.params.index, boot_arr.std(axis=0, ddof=1)))

    def coefficients(self):
        if self.result is None:
            return []
        alpha = getattr(self, "_alpha", 0.05)
        conf = self.result.conf_int(alpha=alpha)
        rows = []
        for param in self.result.params.index:
            if param in ("Intercept", "phi"):
                continue
            # Use bootstrapped SE if available, fall back to model SE
            if self._boot_se and param in self._boot_se:
                se = self._boot_se[param]
                coef = float(self.result.params[param])
                z = coef / se if se > 0 else float("nan")
                from scipy.stats import norm
                z_crit = norm.ppf(1 - alpha / 2)
                p = float(2 * (1 - norm.cdf(abs(z))))
                ci_lower = coef - z_crit * se
                ci_upper = coef + z_crit * se
                se_source = "bootstrap"
            else:
                se = float(self.result.bse[param])
                z = float(self.result.tvalues[param])
                p = float(self.result.pvalues[param])
                ci_lower = float(conf.loc[param, 0])
                ci_upper = float(conf.loc[param, 1])
                se_source = "model"
            rows.append({
                "parameter": str(param),
                "coefficient": float(self.result.params[param]),
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "std_err": se,
                "std_err_source": se_source,
                "z_value": z,
                "p_value": p,
            })
        return rows

    def as_results_dict(self):
        if self.result is None:
            return {"error": "Model not fitted"}

        coef_table = self.coefficients()
        
        main_p = None
        main_coef = None
        p_val_note = None
        if hasattr(self.result, "llr_pvalue") and not np.isnan(self.result.llr_pvalue):
            main_p = float(self.result.llr_pvalue)
            main_coef = float(self.result.llr) if hasattr(self.result, "llr") else None
            p_val_note = "Omnibus (Likelihood-Ratio)"
            stat_type = "Likelihood Ratio (Chi-Square)"
        elif len(coef_table) > 1:
            # Drop intercept if present
            non_int = [c for c in coef_table if c["parameter"] != "Intercept"]
            if non_int:
                main_p = non_int[0]["p_value"]
                main_coef = non_int[0]["coefficient"]
            else:
                main_p = coef_table[0]["p_value"]
                main_coef = coef_table[0]["coefficient"]
            p_val_note = "p-value: first predictor only"
            stat_type = "coefficient"
        elif coef_table:
            main_p = coef_table[0]["p_value"]
            main_coef = coef_table[0]["coefficient"]
            p_val_note = "p-value: first predictor only"
            stat_type = "coefficient"

        # Pseudo-R² (McFadden)
        pseudo_r2 = None
        try:
            ll_model = self.result.llf
            ll_null = self.result.llnull if hasattr(self.result, "llnull") else None
            if ll_null is not None and ll_null != 0:
                pseudo_r2 = float(1 - ll_model / ll_null)
        except Exception:
            pass

        # Dispersion parameter phi (precision)
        phi = None
        try:
            if "phi" in self.result.params.index:
                phi = float(self.result.params["phi"])
        except Exception:
            pass

        # Fitted values and residuals for diagnostics
        fitted = residuals = None
        try:
            fitted = [float(v) for v in self.result.predict()]
            residuals = [float(v) for v in (self._df[self._dv].values - self.result.predict())]
        except Exception:
            pass

        # Raw x/y data for chart: primary predictor vs observed outcome
        xy_data = {}
        try:
            if self._predictors:
                pred_col = self._predictors[0]
                xy_data = {
                    "x": [float(v) for v in self._df[pred_col].values],
                    "y": [float(v) for v in self._df[self._dv].values],
                    "x_label": pred_col,
                }
        except Exception:
            pass

        converged = getattr(self.result, "mle_retvals", {}).get("converged", True) if hasattr(self.result, "mle_retvals") else True
        warnings_list = []
        if not converged:
            warnings_list.append("Beta regression did not converge. Results may be unreliable.")

        res = {
            "design_type": self.design_type.value,
            "test": "Beta Regression",
            "model_type": "BetaRegression",
            "converged": converged,
            "warnings": warnings_list,
            "p_value_note": p_val_note,
            "p_value": main_p,
            "statistic": main_coef,
            "statistic_type": stat_type if 'stat_type' in locals() else "coefficient",
            "effect_size": pseudo_r2,
            "effect_size_type": "pseudo_R2",
            "coefficients": coef_table,
            "phi": phi,
            "fitted_values": fitted,
            "residuals": residuals,
            "xy_data": xy_data,
            "pseudo_r_squared": pseudo_r2,
            "aic": float(self.result.aic) if hasattr(self.result, "aic") else None,
            "bic": float(self.result.bic) if hasattr(self.result, "bic") else None,
            "log_likelihood": float(self.result.llf) if hasattr(self.result, "llf") else None,
            "n_observations": int(self.result.nobs),
            "predictors_used": self._predictors,
            "covariates_used": self._covariates,
            "detection_note": "Outcome detected as proportion (all values strictly in (0,1))",
            "bias_corrected": self._bias_corrected,
            "bias_correction_method": "bootstrapped SE (n_boot=1000)" if self._bias_corrected and self._boot_se else (
                "requested but bootstrap produced insufficient samples" if self._bias_corrected else None
            ),
        }
        return _restore_names_in_dict(res, self._rev_map)


class DataHealthScanner:
    """Pre-analysis data quality checks for clinical datasets.

    Runs up to 5 checks depending on model type:
      1. Covariate outliers      — MAD-based (Modified Z-Score, Iglewicz & Hoaglin)
      2. Missing data mechanism  — Little's MCAR test (only when missings exist)
      3. Multicollinearity       — VIF per covariate (ANCOVA / LMM, ≥2 covariates)
      4. Quasi-perfect separation — crosstab / correlation check (Logistic Regression)
      5. Minimum group size      — events per group (Logistic Regression)

    Usage:
        scanner = DataHealthScanner(df, model_type='ANCOVA', dv='Score',
                                    covariates=['Age', 'BMI'], factors=['Group'])
        report = scanner.run()
        # report = {"warnings": [...], "checks": {...}}
    """

    def __init__(self, df, model_type, dv, covariates=None, factors=None, subject_col=None):
        self._df = df.copy()
        self._model_type = model_type          # 'ANCOVA', 'LMM', 'LogisticRegression'
        self._dv = dv
        self._covariates = covariates or []
        self._factors = factors or []
        self._subject_col = subject_col
        self.warnings = []
        self.checks = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self):
        """Execute all applicable checks and return the health report."""
        if self._covariates:
            self._check_covariate_outliers()
            # Missing-data mechanism only makes sense when there are actual gaps
            cov_cols = [c for c in self._covariates if c in self._df.columns]
            if cov_cols and self._df[cov_cols].isnull().any().any():
                self._check_mcar(cov_cols)
            if len(self._covariates) >= 2:
                self._check_vif()

        if self._model_type == 'LogisticRegression':
            self._check_separation()
            self._check_group_sizes()

        return {"warnings": self.warnings, "checks": self.checks}

    # ------------------------------------------------------------------
    # Check 1 — Covariate outliers (MAD-based Modified Z-Score)
    # ------------------------------------------------------------------

    def _check_covariate_outliers(self):
        outlier_info = {}
        for col in self._covariates:
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
                    f"Outliers in '{col}': {n_extreme} value(s) with |mod. Z-score| > 3.5 "
                    "(MAD-based, no normality requirement)."
                )
        self.checks["covariate_outliers"] = outlier_info

    # ------------------------------------------------------------------
    # Check 2 — Little's MCAR test
    # ------------------------------------------------------------------

    def _check_mcar(self, cov_cols):
        data = self._df[cov_cols].copy()
        if data.dropna().shape[0] < max(5, len(cov_cols) + 1):
            self.checks["mcar"] = {"note": "Too few complete cases for Little's test."}
            return
        try:
            result = self._littles_mcar_test(data, cov_cols)
            self.checks["mcar"] = result
            if result["p_value"] < 0.05:
                self.warnings.append(
                    f"Little's MCAR Test: p={result['p_value']:.3f} — Missing data pattern is "
                    "not random (MAR/MNAR mechanism). LMM results may be biased."
                )
            else:
                result["interpretation"] = "MCAR not rejected — random data loss is plausible."
        except Exception as exc:
            self.checks["mcar"] = {"error": str(exc)}

    def _littles_mcar_test(self, data, columns):
        """Little (1988) MCAR test via chi-squared statistic."""
        from scipy import stats as scipy_stats

        d = len(columns)
        grand_means = data.mean()   # pandas ignores NaN
        grand_cov = data.cov()      # pairwise complete obs

        # Group rows by which columns are missing
        pattern_series = data.isnull().apply(lambda r: tuple(r), axis=1)

        chi2_stat = 0.0
        df_parts = 0

        for pat, group in data.groupby(pattern_series):
            obs_cols = [columns[i] for i, is_na in enumerate(pat) if not is_na]
            if not obs_cols:
                continue
            n_k = len(group)
            mu_k = group[obs_cols].mean().values
            mu_hat = grand_means[obs_cols].values
            diff = mu_k - mu_hat
            sigma_k = grand_cov.loc[obs_cols, obs_cols].values
            try:
                sigma_inv = np.linalg.inv(sigma_k)
                chi2_stat += float(n_k * (diff @ sigma_inv @ diff))
                df_parts += len(obs_cols)
            except np.linalg.LinAlgError:
                continue  # Singular submatrix — skip this pattern

        df_val = max(df_parts - d, 1)
        p_val = float(scipy_stats.chi2.sf(chi2_stat, df=df_val))
        return {"chi2": round(chi2_stat, 4), "df": df_val, "p_value": round(p_val, 4)}

    # ------------------------------------------------------------------
    # Check 3 — VIF (Multicollinearity among covariates)
    # ------------------------------------------------------------------

    def _check_vif(self):
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            import statsmodels.api as sm

            cov_cols = [c for c in self._covariates if c in self._df.columns]
            cov_data = self._df[cov_cols].dropna()
            if len(cov_data) < len(cov_cols) + 2:
                self.checks["vif"] = {"note": "Too few observations for VIF calculation."}
                return

            X = sm.add_constant(cov_data.values, has_constant='add')
            vif_vals = {}
            high_vif = []
            for i, col in enumerate(cov_cols):
                vif = float(variance_inflation_factor(X, i + 1))  # +1 skips constant column
                vif_vals[col] = round(vif, 2)
                if vif > 10:
                    high_vif.append(f"{col} (VIF={vif:.1f})")

            self.checks["vif"] = vif_vals
            if high_vif:
                self.warnings.append(
                    f"Multicollinearity: {', '.join(high_vif)} — covariates provide "
                    "redundant information (VIF > 10). Coefficient interpretation is limited."
                )
        except Exception as exc:
            self.checks["vif"] = {"error": str(exc)}

    # ------------------------------------------------------------------
    # Check 4 — Quasi-perfect separation (Logistic Regression)
    # ------------------------------------------------------------------

    def _check_separation(self):
        separation_issues = []
        try:
            y = self._df[self._dv].dropna()
            if len(y.unique()) != 2:
                self.checks["separation"] = []
                return

            for col in self._factors + self._covariates:
                if col not in self._df.columns:
                    continue
                col_data = self._df[[col, self._dv]].dropna()
                if col_data.empty:
                    continue

                if self._df[col].dtype == object or self._df[col].nunique() <= 10:
                    # Categorical: any cell = 0 in crosstab means complete separation
                    ct = pd.crosstab(col_data[col], col_data[self._dv])
                    if (ct == 0).any().any():
                        separation_issues.append(col)
                else:
                    # Continuous: near-perfect rank correlation as proxy
                    corr = abs(col_data[col].corr(col_data[self._dv]))
                    if corr > 0.95:
                        separation_issues.append(col)

            self.checks["separation"] = separation_issues
            if separation_issues:
                self.warnings.append(
                    f"Quasi-perfect separation in: {', '.join(separation_issues)}. "
                    "Odds ratios may be extremely large/unstable. "
                    "Consider Firth regression as an alternative."
                )
        except Exception as exc:
            self.checks["separation"] = {"error": str(exc)}

    # ------------------------------------------------------------------
    # Check 5 — Minimum group size (Logistic Regression stability)
    # ------------------------------------------------------------------

    def _check_group_sizes(self):
        if not self._factors:
            self.checks["group_sizes"] = {}
            return
        try:
            factor = self._factors[0]
            if factor not in self._df.columns:
                self.checks["group_sizes"] = {}
                return

            counts = (
                self._df.groupby(factor)[self._dv]
                .value_counts()
                .unstack(fill_value=0)
            )
            small_groups = []
            for grp in counts.index:
                min_count = int(counts.loc[grp].min())
                if min_count < 10:
                    small_groups.append(f"{grp} (min. n={min_count})")

            self.checks["group_sizes"] = counts.to_dict()
            if small_groups:
                self.warnings.append(
                    f"Kleine Gruppenbesetzung: {', '.join(small_groups)}. "
                    "Logistische Regression instabil bei n < 10 pro Outcome-Kategorie."
                )
        except Exception as exc:
            self.checks["group_sizes"] = {"error": str(exc)}
