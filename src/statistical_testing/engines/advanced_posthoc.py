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
            from analysis.stats_functions import UIDialogManager, PostHocFactory

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
                    from ui.dialogs.comparison_selection_dialog import ComparisonSelectionDialog
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
            from analysis.statisticaltester import StatisticalTester

            fallback_posthoc = None
            marginaleffects_error = None
            if test == "repeated_measures_anova" and within:
                fallback_posthoc = StatisticalTester._run_rm_marginaleffects_posthoc(res, within[0], alpha=alpha)
            elif test == "mixed_anova" and between and within:
                fallback_posthoc = StatisticalTester._run_mixed_marginaleffects_posthoc(res, between, within, alpha=alpha)
            elif (
                test == "two_way_anova" and between and len(between) == 2
                and df_original is not None
                and res.get("model_class") == "Freedman-Lane Permutation"
            ):
                fallback_posthoc = self._freedman_lane_dialog_posthoc(res, df_original, dv, between, alpha)

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

    @staticmethod
    def _freedman_lane_candidate_specs(res, df, between, alpha):
        """Build significance-gated candidate comparisons for the Freedman-Lane
        post-hoc, mirroring the additive-model omnibus structure.

        Marginal level pairs are offered for each significant main effect
        (pooled over the other factor); cell pairs are offered only when the
        interaction is significant. Returns a list of
        ``(label1, label2, kind, payload)`` specs.
        """
        from itertools import combinations as _comb

        factor_a, factor_b = between[0], between[1]
        a_levels = sorted(df[factor_a].dropna().unique())
        b_levels = sorted(df[factor_b].dropna().unique())

        sig_a = sig_b = sig_ab = False
        for f in res.get("factors", []):
            if f.get("p_value") is None:
                continue
            if f.get("factor") == factor_a:
                sig_a = f["p_value"] < alpha
            elif f.get("factor") == factor_b:
                sig_b = f["p_value"] < alpha
        for it in res.get("interactions", []):
            if it.get("p_value") is not None and it["p_value"] < alpha:
                sig_ab = True

        specs = []
        if sig_a and len(a_levels) >= 2:
            for v1, v2 in _comb(a_levels, 2):
                specs.append((f"{factor_a}={v1}", f"{factor_a}={v2}", "marginal_a", (v1, v2)))
        if sig_b and len(b_levels) >= 2:
            for v1, v2 in _comb(b_levels, 2):
                specs.append((f"{factor_b}={v1}", f"{factor_b}={v2}", "marginal_b", (v1, v2)))
        if sig_ab:
            cells = [(av, bv) for av in a_levels for bv in b_levels]
            for (a1, b1), (a2, b2) in _comb(cells, 2):
                specs.append((
                    f"{factor_a}={a1}, {factor_b}={b1}",
                    f"{factor_a}={a2}, {factor_b}={b2}",
                    "cell", ((a1, b1), (a2, b2)),
                ))
        return specs

    @staticmethod
    def _select_comparisons_dialog(all_pairs):
        """Show the ComparisonSelectionDialog and return the chosen pairs.

        Headless / cancelled / error -> return ``all_pairs`` unchanged, preserving
        the non-interactive all-candidates behaviour.
        """
        try:
            import sys
            from PyQt5.QtWidgets import QApplication
            from ui.dialogs.comparison_selection_dialog import ComparisonSelectionDialog

            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)
            dialog = ComparisonSelectionDialog(all_pairs, checked_by_default=True)
            if dialog.exec_() == dialog.Accepted:
                chosen = dialog.get_selected_comparisons()
                return chosen if chosen else all_pairs
            return all_pairs
        except Exception as exc:
            logger.warning("Could not show Freedman-Lane comparison dialog: %s", exc)
            return all_pairs

    @staticmethod
    def _freedman_lane_compute(specs, selected_pairs, df, dv, between, alpha):
        """Run pairwise Mann-Whitney U + Holm on the selected Freedman-Lane specs.

        Marginal pairs pool the dependent variable over the other factor; cell
        pairs use the (A, B) subgroup. Reuses the omnibus MWU / Holm helpers so
        the statistics match the all-pairs path exactly.
        """
        from analysis.nonparametricanovas import _mwu_posthoc_comp, _apply_holm

        factor_a, factor_b = between[0], between[1]
        selected_set = set(selected_pairs)

        raw = []
        for label1, label2, kind, payload in specs:
            if (label1, label2) not in selected_set:
                continue
            if kind == "marginal_a":
                v1, v2 = payload
                arr1 = df[df[factor_a] == v1][dv].values
                arr2 = df[df[factor_a] == v2][dv].values
            elif kind == "marginal_b":
                v1, v2 = payload
                arr1 = df[df[factor_b] == v1][dv].values
                arr2 = df[df[factor_b] == v2][dv].values
            else:  # cell
                (a1, b1), (a2, b2) = payload
                arr1 = df[(df[factor_a] == a1) & (df[factor_b] == b1)][dv].values
                arr2 = df[(df[factor_a] == a2) & (df[factor_b] == b2)][dv].values
            comp = _mwu_posthoc_comp(arr1, arr2, label1, label2, alpha)
            if comp is not None:
                raw.append(comp)

        if not raw:
            return {"pairwise_comparisons": [], "posthoc_test": "No comparisons selected"}

        comps = _apply_holm(raw, alpha)
        return {
            "pairwise_comparisons": comps,
            "posthoc_test": "Pairwise Mann-Whitney U (marginal / cell simple effects, Holm-corrected)",
        }

    def _freedman_lane_dialog_posthoc(self, res, df_original, dv, between, alpha):
        """Dialog-driven pairwise MWU post-hoc for the Freedman-Lane fallback.

        Builds the significance-gated candidate set, lets the user pick a subset
        (all candidates when headless/cancelled), then computes MWU + Holm.
        """
        specs = self._freedman_lane_candidate_specs(res, df_original, between, alpha)
        if not specs:
            return {"pairwise_comparisons": [], "posthoc_test": "No applicable post-hoc comparisons"}
        all_pairs = [(s[0], s[1]) for s in specs]
        selected_pairs = self._select_comparisons_dialog(all_pairs)
        return self._freedman_lane_compute(specs, selected_pairs, df_original, dv, between, alpha)
