import logging
from itertools import combinations
from typing import Any, Mapping

from ..models import StatisticalResult


logger = logging.getLogger(__name__)


class AdvancedPostHocEngine:
    """Handles advanced-model post-hoc orchestration and fallback paths."""

    def execute(self, data: Mapping[str, Any]) -> StatisticalResult:
        payload = dict(data or {})
        mode = str(payload.get("mode") or "")

        if mode == "advanced_parametric":
            updates = self._run_advanced_parametric_posthoc(payload)
            return StatisticalResult(
                test_name=str(updates.get("posthoc_test") or "advanced_posthoc"),
                statistic_value=None,
                p_value=None,
                metadata=updates,
            )

        if mode == "nonparametric_fallback":
            updates = self._run_nonparametric_fallback_posthoc(payload)
            return StatisticalResult(
                test_name=str(updates.get("posthoc_test") or "nonparametric_fallback_posthoc"),
                statistic_value=None,
                p_value=None,
                metadata=updates,
            )

        return StatisticalResult(
            test_name="advanced_posthoc_failed",
            statistic_value=None,
            p_value=None,
            metadata={"error": f"Unsupported advanced post-hoc mode '{mode}'."},
        )

    def _run_advanced_parametric_posthoc(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        test = payload.get("test")
        df_transformed = payload.get("df_transformed")
        dv = payload.get("dv")
        subject = payload.get("subject")
        between = payload.get("between")
        within = payload.get("within")
        alpha = float(payload.get("alpha", 0.05))

        if df_transformed is None:
            return {"error": "df_transformed is required for advanced parametric post-hoc."}

        try:
            from stats_functions import UIDialogManager, PostHocFactory

            if test == "two_way_anova":
                group_names = []
                factors = between
                for factor_a_val in sorted(df_transformed[factors[0]].unique()):
                    for factor_b_val in sorted(df_transformed[factors[1]].unique()):
                        group_names.append(f"{factors[0]}={factor_a_val}, {factors[1]}={factor_b_val}")
            elif test == "mixed_anova":
                group_names = []
                b_factor, w_factor = between[0], within[0]
                for b_val in sorted(df_transformed[b_factor].unique()):
                    for w_val in sorted(df_transformed[w_factor].unique()):
                        group_names.append(f"{b_factor}={b_val}, {w_factor}={w_val}")
            elif test == "repeated_measures_anova":
                w_factor = within[0]
                group_names = list(df_transformed[w_factor].unique())
            else:
                group_names = []

            all_comparisons = list(combinations(group_names, 2))

            posthoc_method = "paired_custom"
            control_group = None
            try:
                default_method = "paired_custom" if test == "two_way_anova" else "tukey"
                posthoc_method = UIDialogManager.select_posthoc_test_dialog(
                    parent=None, progress_text=f"({test})", column_name=dv, default_method=default_method
                )
                if posthoc_method is None:
                    posthoc_method = "paired_custom"
                if posthoc_method == "dunnett":
                    control_group = UIDialogManager.select_control_group_dialog(parent=None, groups=group_names)
            except Exception as exc:
                logger.warning("Could not show post-hoc method dialog: %s", exc)
                posthoc_method = "paired_custom"

            if posthoc_method == "dunnett" and control_group:
                selected_comparisons = [(control_group, group) for group in group_names if group != control_group]
            elif posthoc_method == "tukey":
                selected_comparisons = all_comparisons
            elif posthoc_method == "paired_custom":
                try:
                    from comparison_selection_dialog import ComparisonSelectionDialog
                    import sys
                    from PyQt5.QtWidgets import QApplication

                    app = QApplication.instance()
                    if app is None:
                        app = QApplication(sys.argv)
                    dialog = ComparisonSelectionDialog(all_comparisons, checked_by_default=False)
                    if dialog.exec_() == dialog.Accepted:
                        selected_comparisons = dialog.get_selected_comparisons()
                    else:
                        selected_comparisons = []
                except Exception as exc:
                    logger.warning("Could not show comparison selection dialog: %s", exc)
                    selected_comparisons = all_comparisons
            else:
                selected_comparisons = all_comparisons

            def normalize_pair(pair):
                return tuple(sorted([entry.strip() for entry in pair]))

            normalized_selected_comparisons = set(normalize_pair(pair) for pair in selected_comparisons)
            posthoc_kwargs = {
                "selected_comparisons": normalized_selected_comparisons,
                "method": posthoc_method,
            }
            if control_group:
                posthoc_kwargs["control_group"] = control_group

            if test == "two_way_anova":
                posthoc = PostHocFactory.perform_posthoc_for_anova(
                    "two_way", df=df_transformed, dv=dv, between=between, alpha=alpha, **posthoc_kwargs
                )
            elif test == "mixed_anova":
                posthoc = PostHocFactory.perform_posthoc_for_anova(
                    "mixed", df=df_transformed, dv=dv, subject=subject, between=between, within=within, alpha=alpha, **posthoc_kwargs
                )
            elif test == "repeated_measures_anova":
                posthoc = PostHocFactory.perform_posthoc_for_anova(
                    "rm", df=df_transformed, dv=dv, subject=subject, within=within, alpha=alpha, **posthoc_kwargs
                )
            else:
                posthoc = None

            if posthoc and "pairwise_comparisons" in posthoc:
                return {
                    "posthoc_test": posthoc.get("posthoc_test"),
                    "pairwise_comparisons": posthoc.get("pairwise_comparisons", []),
                }

            return {
                "posthoc_test": "No post-hoc tests performed",
                "pairwise_comparisons": [],
            }
        except Exception as exc:
            return {"error": str(exc), "posthoc_test": "No post-hoc tests performed", "pairwise_comparisons": []}

    def _run_nonparametric_fallback_posthoc(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        res = dict(payload.get("res") or {})
        test = payload.get("test")
        df_original = payload.get("df_original")
        dv = payload.get("dv")
        subject = payload.get("subject")
        between = payload.get("between")
        within = payload.get("within")
        alpha = float(payload.get("alpha", 0.05))

        if res.get("error") is not None:
            return {}

        p_value = res.get("p_value")
        if not isinstance(p_value, (float, int)):
            return {}

        if p_value >= alpha:
            return {
                "posthoc_skipped": True,
                "posthoc_skip_reason": f"Post-hoc not performed: main test was not significant (p\u202f=\u202f{p_value:.4f})",
                "pairwise_comparisons": res.get("pairwise_comparisons", []),
            }

        try:
            from statisticaltester import StatisticalTester

            fallback_posthoc = None
            marginaleffects_error = None
            if test == "repeated_measures_anova" and within:
                fallback_posthoc = StatisticalTester._run_rm_marginaleffects_posthoc(res, within[0], alpha=alpha)
            elif test == "mixed_anova" and between and within:
                fallback_posthoc = StatisticalTester._run_mixed_marginaleffects_posthoc(res, between, within, alpha=alpha)

            updates: dict[str, Any] = {}
            warnings_list = list(res.get("warnings", []))

            if fallback_posthoc and fallback_posthoc.get("error"):
                marginaleffects_error = fallback_posthoc["error"]
                if marginaleffects_error not in warnings_list:
                    warnings_list.append(marginaleffects_error)

            _nonparam_classes = {"Friedman", "Freedman-Lane Permutation", "Brunner-Langer ATS"}
            if res.get("model_class") not in _nonparam_classes and (
                not fallback_posthoc or (not fallback_posthoc.get("pairwise_comparisons") and fallback_posthoc.get("error"))
            ):
                fallback_posthoc = StatisticalTester._run_modern_fallback_posthoc(
                    df_original.copy(),
                    test,
                    dv,
                    subject=subject,
                    between=between,
                    within=within,
                    alpha=alpha,
                )
                if marginaleffects_error:
                    fallback_note = (
                        " Post-hoc comparisons used a robust non-parametric fallback "
                        "because the marginaleffects step failed. See warnings for details."
                    )
                    analysis_note = str(res.get("analysis_note", ""))
                    if fallback_note.strip() not in analysis_note:
                        updates["analysis_note"] = f"{analysis_note}{fallback_note}".strip()

            if fallback_posthoc and fallback_posthoc.get("pairwise_comparisons"):
                updates["pairwise_comparisons"] = fallback_posthoc.get("pairwise_comparisons", [])
                updates["posthoc_test"] = fallback_posthoc.get("posthoc_test")
            if fallback_posthoc and fallback_posthoc.get("error") and fallback_posthoc["error"] not in warnings_list:
                warnings_list.append(fallback_posthoc["error"])

            if warnings_list:
                updates["warnings"] = warnings_list

            return updates
        except Exception as exc:
            return {"error": str(exc)}
