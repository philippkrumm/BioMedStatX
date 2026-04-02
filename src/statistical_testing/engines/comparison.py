import logging
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy import stats

from ..models import StatisticalResult
from ..validators import (
    GroupValidationError,
    ValidationError,
    validate_balanced_design,
    validate_group_count,
    validate_minimum_n,
)


logger = logging.getLogger(__name__)


class ComparisonEngine:
    """Executes group-comparison strategies via legacy-tested statistical routines."""

    def execute(self, data: Mapping[str, Any]) -> StatisticalResult:
        payload = dict(data or {})
        strategy = str(payload.get("strategy") or "")
        groups = list(payload.get("groups") or [])
        samples = dict(payload.get("samples") or {})
        alpha = float(payload.get("alpha", 0.05))
        results = dict(payload.get("results") or {})

        try:
            # Local import avoids module-cycle side effects at import time.
            from statisticaltester import StatisticalTester

            two_group_strategies = {
                "paired_ttest",
                "wilcoxon",
                "student_ttest",
                "welch_ttest",
                "mann_whitney_u",
            }
            multi_group_strategies = {
                "one_way_anova",
                "welch_anova",
                "kruskal_wallis",
            }

            if strategy in two_group_strategies:
                return self._execute_two_group(
                    StatisticalTester=StatisticalTester,
                    strategy=strategy,
                    groups=groups,
                    samples=samples,
                    alpha=alpha,
                    results=results,
                )

            if strategy in multi_group_strategies:
                return self._execute_multi_group(
                    StatisticalTester=StatisticalTester,
                    strategy=strategy,
                    groups=groups,
                    samples=samples,
                    alpha=alpha,
                    results=results,
                )

            return self._failed(f"Unsupported comparison strategy '{strategy}'.", strategy=strategy)

        except Exception as exc:
            return self._failed(str(exc), strategy=strategy)

    def _execute_two_group(self, *, StatisticalTester, strategy: str, groups: list[str], samples: dict[str, Any], alpha: float, results: dict[str, Any]) -> StatisticalResult:
        if len(groups) != 2:
            return self._failed(f"ComparisonEngine expects exactly 2 groups, got {len(groups)}.", strategy=strategy)

        g1, g2 = groups
        data1 = samples.get(g1)
        data2 = samples.get(g2)
        if data1 is None or data2 is None:
            return self._failed("ComparisonEngine requires samples for both groups.", strategy=strategy)

        if strategy == "paired_ttest":
            legacy = StatisticalTester._paired_ttest(results, g1, g2, data1, data2, alpha)
        elif strategy == "wilcoxon":
            legacy = StatisticalTester._wilcoxon_test(results, g1, g2, data1, data2, alpha)
        elif strategy == "student_ttest":
            legacy = StatisticalTester._independent_ttest(results, g1, g2, data1, data2, alpha, equal_var=True)
        elif strategy == "welch_ttest":
            legacy = StatisticalTester._independent_ttest(results, g1, g2, data1, data2, alpha, equal_var=False)
        elif strategy == "mann_whitney_u":
            legacy = StatisticalTester._mannwhitney_test(results, g1, g2, data1, data2, alpha)
        else:
            return self._failed(f"Unsupported two-group strategy '{strategy}'.", strategy=strategy)

        return StatisticalTester.to_statistical_result(legacy)

    def _execute_multi_group(self, *, StatisticalTester, strategy: str, groups: list[str], samples: dict[str, Any], alpha: float, results: dict[str, Any]) -> StatisticalResult:
        try:
            validate_group_count(groups, min_groups=3, label="comparison_engine_multi_groups")
            for group in groups:
                validate_minimum_n(samples.get(group, []), min_n=2, label=str(group), allow_missing=False)
        except ValidationError as exc:
            return self._failed(str(exc), strategy=strategy)

        if strategy == "welch_anova":
            legacy = StatisticalTester._welch_anova_test(results, groups, samples, alpha)
            return StatisticalTester.to_statistical_result(legacy)

        if strategy == "one_way_anova":
            legacy = self._run_one_way_anova(StatisticalTester=StatisticalTester, groups=groups, samples=samples, alpha=alpha, results=results)
            return StatisticalTester.to_statistical_result(legacy)

        if strategy == "kruskal_wallis":
            legacy = self._run_kruskal_wallis(groups=groups, samples=samples, alpha=alpha, results=results)
            return StatisticalTester.to_statistical_result(legacy)

        return self._failed(f"Unsupported multi-group strategy '{strategy}'.", strategy=strategy)

    def _run_one_way_anova(self, *, StatisticalTester, groups: list[str], samples: dict[str, Any], alpha: float, results: dict[str, Any]) -> dict[str, Any]:
        try:
            from stats_functions import get_pingouin_module

            data_for_anova = []
            for i, group in enumerate(groups):
                for value in samples[group]:
                    data_for_anova.append({"dependent_var": float(value), "group_var": i})

            df_pg = pd.DataFrame(data_for_anova)
            df_pg["dependent_var"] = df_pg["dependent_var"].astype(float)
            df_pg["group_var"] = df_pg["group_var"].astype("category")

            pg = get_pingouin_module()
            aov = pg.anova(data=df_pg, dv="dependent_var", between="group_var", detailed=True)
            results["anova_table"] = aov.copy()

            if len(aov) < 1:
                raise ValueError("Pingouin ANOVA returned empty table")

            row_between = aov.iloc[0]
            if len(aov) > 1:
                row_residual = aov.iloc[1]
                df2 = row_residual["DF"]
                results["df2"] = int(df2) if pd.notnull(df2) else None
            else:
                total_observations = sum(len(samples[g]) for g in groups)
                results["df2"] = total_observations - len(groups)

            results["test"] = "One-way ANOVA (Pingouin)"
            results["statistic"] = float(row_between["F"])
            p_col = StatisticalTester._pingouin_p_column(row_between.index)
            if p_col is None:
                raise ValueError("Pingouin ANOVA table missing p-value column")
            results["p_value"] = float(row_between[p_col])
            np2_col = "np2" if "np2" in row_between.index else "n2"
            results["effect_size"] = float(row_between[np2_col]) if np2_col in row_between.index else None
            results["effect_size_type"] = "partial_eta_squared"
            results["confidence_interval"] = (None, None)
            results["power"] = None
            return results

        except Exception as exc:
            logger.debug("Pingouin ANOVA failed in ComparisonEngine: %s", exc)
            teststat, pval = stats.f_oneway(*[samples[g] for g in groups])
            results["test"] = "One-way ANOVA (SciPy)"
            results["p_value"] = float(pval)
            results["statistic"] = float(teststat)

            all_data = np.concatenate([samples[g] for g in groups])
            grand_mean = np.mean(all_data)
            ss_between = sum(len(samples[g]) * (np.mean(samples[g]) - grand_mean) ** 2 for g in groups)
            ss_total = sum((x - grand_mean) ** 2 for x in all_data)
            eta_sq = ss_between / ss_total if ss_total > 0 else None

            try:
                validate_balanced_design(samples, groups)
            except GroupValidationError as design_warning:
                results["design_note"] = str(design_warning)
                logger.warning(str(design_warning))

            results["effect_size"] = eta_sq
            results["effect_size_type"] = "eta_squared"
            results["anova_table"] = None
            results["confidence_interval"] = (None, None)

            try:
                from statsmodels.stats.power import FTestAnovaPower

                k = len(groups)
                n = sum(len(samples[g]) for g in groups)
                f2 = eta_sq / (1 - eta_sq) if (eta_sq is not None and eta_sq < 1) else 0
                power_analysis = FTestAnovaPower()
                results["power"] = float(power_analysis.power(effect_size=f2, k_groups=k, nobs=n, alpha=alpha))
            except Exception:
                results["power"] = None

            return results

    def _run_kruskal_wallis(self, *, groups: list[str], samples: dict[str, Any], alpha: float, results: dict[str, Any]) -> dict[str, Any]:
        teststat, pval = stats.kruskal(*[samples[g] for g in groups])
        results["test"] = "Kruskal-Wallis test"
        results["p_value"] = float(pval)
        results["statistic"] = float(teststat)

        n = sum(len(samples[g]) for g in groups)
        h = float(teststat)
        k = len(groups)
        epsilon_sq = (h - k + 1) / (n - k) if n > k else None
        if epsilon_sq is not None:
            epsilon_sq = max(0.0, min(1.0, float(epsilon_sq)))

        results["effect_size"] = epsilon_sq
        results["effect_size_type"] = "epsilon_squared"
        results["anova_table"] = None
        results["confidence_interval"] = (None, None)

        try:
            from statsmodels.stats.power import FTestAnovaPower

            f2_approx = (epsilon_sq / (1 - epsilon_sq)) * 0.955 if (epsilon_sq is not None and epsilon_sq < 1) else 0
            power_analysis = FTestAnovaPower()
            results["power"] = float(power_analysis.power(effect_size=f2_approx, k_groups=k, nobs=n, alpha=alpha))
        except Exception:
            results["power"] = None

        return results

    @staticmethod
    def _failed(message: str, *, strategy: str | None = None) -> StatisticalResult:
        metadata = {"error": message}
        if strategy is not None:
            metadata["strategy"] = strategy
        return StatisticalResult(
            test_name="comparison_engine_failed",
            statistic_value=None,
            p_value=None,
            metadata=metadata,
        )
