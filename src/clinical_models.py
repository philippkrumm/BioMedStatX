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


def _sanitize_columns(df, columns):
    """Rename columns with special characters so patsy can parse them.

    Returns a dict mapping original name -> sanitized name.
    The DataFrame is renamed in-place.
    """
    mapping = {}
    for col in columns:
        safe = re.sub(r'[^A-Za-z0-9_]', '_', str(col))
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


class ANCOVAModel:
    """ANCOVA via statsmodels OLS with Type II SS.

    Type II SS is used because it provides higher power than Type III when
    no significant Factor x Covariate interactions exist (typical in clinical data).
    The regression slope homogeneity check validates this assumption.
    """

    def __init__(self):
        self.result = None
        self.anova_table = None
        self._df = None
        self._dv = None
        self._between_factors = None
        self._covariates = None
        self._alpha = 0.05

    def fit(self, df, dv, between_factors, covariates, alpha=0.05):
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm

        self._df = df.dropna(subset=[dv] + between_factors + covariates).copy()
        self._alpha = alpha

        # Sanitize column names for patsy formulas (replace spaces/special chars)
        col_map = _sanitize_columns(self._df, [dv] + between_factors + covariates)
        self._dv = col_map[dv]
        self._between_factors = [col_map[f] for f in between_factors]
        self._covariates = [col_map[c] for c in covariates]

        factor_terms = " * ".join([f"C({f})" for f in self._between_factors])
        cov_terms = " + ".join(self._covariates)
        formula = f"{self._dv} ~ {factor_terms} + {cov_terms}"

        model = smf.ols(formula, data=self._df).fit()
        self.result = model
        self.anova_table = anova_lm(model, typ=2)
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
                interaction_term = f"C({factor}):{cov}"
                factor_terms = " * ".join([f"C({f})" for f in self._between_factors])
                cov_terms = " + ".join(self._covariates)
                formula_with_interaction = f"{self._dv} ~ {factor_terms} + {cov_terms} + {interaction_term}"

                try:
                    model_interaction = smf.ols(formula_with_interaction, data=self._df).fit()
                    table = anova_lm(model_interaction, typ=2)
                    key = f"{factor}:{cov}"
                    if key in table.index:
                        row = table.loc[key]
                        p_val = row["PR(>F)"]
                        results[key] = {
                            "F": row["F"],
                            "p_value": p_val,
                            "df": row["df"],
                            "assumption_holds": p_val > self._alpha,
                        }
                    elif interaction_term in table.index:
                        row = table.loc[interaction_term]
                        p_val = row["PR(>F)"]
                        results[key] = {
                            "F": row["F"],
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
        """Compute estimated marginal means (adjusted for covariates)."""
        if self.result is None:
            return {}

        means = {}
        for factor in self._between_factors:
            levels = self._df[factor].unique()
            adjusted = {}
            cov_means = {c: self._df[c].mean() for c in self._covariates}

            for level in levels:
                subset = self._df[self._df[factor] == level]
                prediction_data = subset.copy()
                for c, m in cov_means.items():
                    prediction_data[c] = m
                try:
                    predicted = self.result.predict(prediction_data)
                    adjusted[str(level)] = {
                        "adjusted_mean": float(predicted.mean()),
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

    def as_results_dict(self):
        if self.result is None:
            return {"error": "Model not fitted"}

        slope_homogeneity = self.check_regression_slope_homogeneity()
        adj_means = self.adjusted_means()

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
        factor_key = f"C({primary_factor})"
        if self.anova_table is not None and factor_key in self.anova_table.index:
            main_p = float(self.anova_table.loc[factor_key, "PR(>F)"])
            main_f = float(self.anova_table.loc[factor_key, "F"])

        # Compute eta-squared for main factor
        eta_sq = None
        if self.anova_table is not None and factor_key in self.anova_table.index:
            ss_factor = self.anova_table.loc[factor_key, "sum_sq"]
            ss_total = self.anova_table["sum_sq"].sum()
            if ss_total > 0:
                eta_sq = float(ss_factor / ss_total)

        return {
            "test": "ANCOVA" if len(self._between_factors) == 1 else "Two-Way ANCOVA",
            "model_type": "ANCOVA",
            "p_value": main_p,
            "statistic": main_f,
            "effect_size": eta_sq,
            "effect_size_type": "eta_squared",
            "anova_table": anova_rows,
            "covariate_effects": covariate_effects,
            "adjusted_means": adj_means,
            "slope_homogeneity": slope_homogeneity,
            "r_squared": float(self.result.rsquared),
            "r_squared_adj": float(self.result.rsquared_adj),
            "aic": float(self.result.aic),
            "bic": float(self.result.bic),
            "n_observations": int(self.result.nobs),
            "covariates_used": self._covariates,
            "between_factors": self._between_factors,
        }


class LinearMixedModel:
    """LMM for longitudinal clinical data via statsmodels MixedLM.

    Default: Random intercept per patient (the clinical standard).
    Uses REML estimation and handles unbalanced/missing data natively.
    """

    def __init__(self):
        self.result = None
        self._df = None
        self._dv = None
        self._fixed_effects = None
        self._random_intercept = None
        self._random_slope = None
        self._covariates = None

    def fit(self, df, dv, fixed_effects, random_intercept, covariates=None, random_slope=None):
        """Fit the Linear Mixed Model.

        Args:
            random_slope: Optional column name for a random slope effect.
                          When provided, fits a correlated random intercept +
                          random slope model (equivalent to lme4's (1 + x | Subject)).
                          Default None = random intercept only.
        """
        import statsmodels.formula.api as smf

        all_cols = [dv, random_intercept] + fixed_effects + (covariates or [])
        if random_slope and random_slope not in all_cols:
            all_cols.append(random_slope)
        self._df = df.dropna(subset=all_cols).copy()

        col_map = _sanitize_columns(self._df, all_cols)
        self._dv = col_map[dv]
        self._fixed_effects = [col_map[f] for f in fixed_effects]
        self._random_intercept = col_map[random_intercept]
        self._random_slope = col_map[random_slope] if random_slope else None
        self._covariates = [col_map[c] for c in (covariates or [])]

        # Build fixed-effects part: categorical factors wrapped in C(), continuous covariates added as-is
        terms = []
        if self._fixed_effects:
            terms.append(" * ".join([f"C({f})" for f in self._fixed_effects]))
        if self._covariates:
            terms.extend(self._covariates)
        fixed_terms = " + ".join(terms) if terms else "1"
        formula = f"{self._dv} ~ {fixed_terms}"

        # re_formula="~x" adds a correlated random slope for x, matching
        # lme4's (1 + x | Subject) syntax. Without it, only random intercept.
        re_formula = f"~{self._random_slope}" if self._random_slope else None
        model = smf.mixedlm(formula, data=self._df,
                            groups=self._df[self._random_intercept],
                            re_formula=re_formula)
        self.result = model.fit(reml=True)
        return self

    def icc(self):
        """Compute Intraclass Correlation Coefficient.

        ICC = var(random intercept) / (var(random intercept) + var(residual))
        """
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

        # Fixed effects table
        fixed_effects_table = []
        for param_name in self.result.fe_params.index:
            fixed_effects_table.append({
                "parameter": str(param_name),
                "coefficient": float(self.result.fe_params[param_name]),
                "std_err": float(self.result.bse_fe[param_name]),
                "z_value": float(self.result.tvalues[param_name]),
                "p_value": float(self.result.pvalues[param_name]),
                "ci_lower": float(self.result.conf_int().loc[param_name, 0]),
                "ci_upper": float(self.result.conf_int().loc[param_name, 1]),
            })

        # Random effects variance
        re_var = None
        resid_var = None
        try:
            re_var = float(self.result.cov_re.iloc[0, 0])
            resid_var = float(self.result.scale)
        except Exception:
            pass

        # Extract p-values for main fixed effects (factors, not interactions/intercept)
        main_p = None
        main_z = None
        for fe in self._fixed_effects:
            for param_name in self.result.fe_params.index:
                if f"C({fe})" in param_name and ":" not in param_name and param_name != "Intercept":
                    main_p = float(self.result.pvalues[param_name])
                    main_z = float(self.result.tvalues[param_name])
                    break
            if main_p is not None:
                break

        icc_val = self.icc()

        # Count subjects and observations
        n_subjects = self._df[self._random_intercept].nunique()
        n_observations = len(self._df)

        return {
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
            "n_subjects": n_subjects,
            "n_observations": n_observations,
            "fixed_effects_used": self._fixed_effects,
            "random_intercept": self._random_intercept,
            "covariates_used": self._covariates,
        }


class LogisticRegressionModel:
    """Logistic regression for binary outcomes via statsmodels GLM(Binomial).

    Provides odds ratios with 95% CI, Hosmer-Lemeshow goodness-of-fit test,
    pseudo-R², and ROC/AUC data.
    """

    def __init__(self):
        self.result = None
        self._df = None
        self._dv = None
        self._predictors = None
        self._covariates = None

    def fit(self, df, dv, predictors, covariates=None):
        import statsmodels.formula.api as smf
        import statsmodels.api as sm

        all_cols = [dv] + predictors + (covariates or [])
        self._df = df.dropna(subset=all_cols).copy()

        col_map = _sanitize_columns(self._df, all_cols)
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
        return self

    def odds_ratios(self):
        """Compute odds ratios with 95% CI."""
        if self.result is None:
            return []

        conf = self.result.conf_int()
        rows = []
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
        """Hosmer-Lemeshow goodness-of-fit test."""
        from scipy import stats as scipy_stats

        if self.result is None:
            return {"chi2": None, "df": None, "p_value": None}

        try:
            predicted = self.result.predict()
            observed = self._df[self._dv].values

            # Sort by predicted probability and divide into groups
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

    def roc_data(self):
        """Compute ROC curve data (FPR, TPR, thresholds) and AUC."""
        if self.result is None:
            return {"fpr": [], "tpr": [], "auc": None}

        predicted = self.result.predict()
        observed = self._df[self._dv].values

        # Manual ROC computation (no sklearn dependency)
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

        # AUC via trapezoidal rule
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

        # Primary predictor odds ratio and p-value
        main_p = None
        main_or = None
        if or_table:
            main_p = or_table[0]["p_value"]
            main_or = or_table[0]["odds_ratio"]

        # Pseudo R-squared (McFadden)
        pseudo_r2 = None
        try:
            ll_model = self.result.llf
            ll_null = self.result.llnull if hasattr(self.result, 'llnull') else None
            if ll_null is not None and ll_null != 0:
                pseudo_r2 = float(1 - ll_model / ll_null)
        except Exception:
            pass

        return {
            "test": "Logistic Regression",
            "model_type": "LogisticRegression",
            "p_value": main_p,
            "statistic": main_or,
            "statistic_type": "odds_ratio",
            "effect_size": roc["auc"],
            "effect_size_type": "AUC",
            "odds_ratios": or_table,
            "hosmer_lemeshow": hl,
            "roc_data": roc,
            "pseudo_r_squared": pseudo_r2,
            "aic": float(self.result.aic) if hasattr(self.result, 'aic') else None,
            "bic": float(self.result.bic_llf) if hasattr(self.result, 'bic_llf') else None,
            "log_likelihood": float(self.result.llf) if hasattr(self.result, 'llf') else None,
            "n_observations": int(self.result.nobs),
            "predictors_used": self._predictors,
            "covariates_used": self._covariates,
        }


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
                    f"Ausreißer in '{col}': {n_extreme} Wert(e) mit |mod. Z-Score| > 3.5 "
                    "(MAD-basiert, kein Normalverteilungs-Zwang)."
                )
        self.checks["covariate_outliers"] = outlier_info

    # ------------------------------------------------------------------
    # Check 2 — Little's MCAR test
    # ------------------------------------------------------------------

    def _check_mcar(self, cov_cols):
        data = self._df[cov_cols].copy()
        if data.dropna().shape[0] < max(5, len(cov_cols) + 1):
            self.checks["mcar"] = {"note": "Zu wenige vollständige Fälle für Little's Test."}
            return
        try:
            result = self._littles_mcar_test(data, cov_cols)
            self.checks["mcar"] = result
            if result["p_value"] < 0.05:
                self.warnings.append(
                    f"Little's MCAR Test: p={result['p_value']:.3f} — Datenlücken sind "
                    "nicht zufällig (MAR/MNAR-Mechanismus). LMM-Ergebnisse könnten verzerrt sein."
                )
            else:
                result["interpretation"] = "MCAR nicht verworfen — zufälliger Datenverlust plausibel."
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
                self.checks["vif"] = {"note": "Zu wenige Beobachtungen für VIF-Berechnung."}
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
                    f"Multikollinearität: {', '.join(high_vif)} — Kovariaten liefern "
                    "redundante Information (VIF > 10). Koeffizienten-Interpretation eingeschränkt."
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
                    f"Quasi-perfekte Separation in: {', '.join(separation_issues)}. "
                    "Odds Ratios könnten extrem groß/instabil sein. "
                    "Firth-Regression als Alternative erwägen."
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
