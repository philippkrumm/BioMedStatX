import logging
logger = logging.getLogger(__name__)


class DecisionTreeVisualizer:
    """
    Creates visual decision trees for statistical test workflows with the actual path highlighted.
    Uses networkx and matplotlib to generate a directed graph showing the decision-making process.
    """

    WIDE_LAYOUT_TOP_X_SCALE = 1.85
    WIDE_LAYOUT_BOTTOM_X_SCALE = 3.35
    WIDE_LAYOUT_TOP_Y_SCALE = 1.6
    WIDE_LAYOUT_BOTTOM_Y_SCALE = 2.2

    @staticmethod
    def _apply_wide_canvas_layout(nodes_info):
        """Stretch the layout while preserving the geometric decision-tree pattern."""
        f_y = nodes_info['F']["pos"][1]
        stretched = {}

        for node_id, info in nodes_info.items():
            x_pos, y_pos = info["pos"]
            if y_pos < f_y:
                x_scale = DecisionTreeVisualizer.WIDE_LAYOUT_BOTTOM_X_SCALE
                y_scale = DecisionTreeVisualizer.WIDE_LAYOUT_BOTTOM_Y_SCALE
            else:
                x_scale = DecisionTreeVisualizer.WIDE_LAYOUT_TOP_X_SCALE
                y_scale = DecisionTreeVisualizer.WIDE_LAYOUT_TOP_Y_SCALE

            stretched_y = f_y + ((y_pos - f_y) * y_scale)
            stretched[node_id] = {"label": info["label"], "pos": (x_pos * x_scale, stretched_y)}

        return stretched


    @staticmethod
    def format_dynamic_test_label(node_id, base_label, results):
        """
        Dynamically format a test node label with its statistic, df, p-value, and effect size.
        """
        if not results or not isinstance(results, dict):
            return base_label

        stat_val = results.get("statistic", results.get("stat", None))
        p_val = results.get("p_value", results.get("p-val", results.get("p", None)))
        df = results.get("d", None)
        df1 = results.get("df1", results.get("df_num", None))
        df2 = results.get("df2", results.get("df_den", None))
        eff_val = results.get("effect_size", results.get("ef", None))
        eff_name = results.get("effect_size_type", "")

        df_str = ""
        if df1 is not None and df2 is not None:
            try:
                df_str = f"({int(df1)}, {int(df2)})"
            except (ValueError, TypeError):
                df_str = f"({df1}, {df2})"
        elif df is not None:
            try:
                df_str = f"({int(df)})"
            except (ValueError, TypeError):
                df_str = f"({df})"

        p_str = ""
        if isinstance(p_val, str):
            p_str = f"p {p_val}" if ("<" in p_val or ">" in p_val or "=" in p_val) else f"p = {p_val}"
        elif isinstance(p_val, (int, float)):
            p_str = "p < 0.0001" if p_val < 0.0001 else f"p = {p_val:.4f}"
        
        stat_str = ""
        if stat_val is not None:
            try:
                stat_str = f"{stat_val:.3f}" if isinstance(stat_val, (int, float)) else f"{stat_val}"
            except (ValueError, TypeError):
                stat_str = f"{stat_val}"

        eff_str = ""
        if eff_val is not None:
            symbol = "d"
            if eff_name:
                eff_name_lower = eff_name.lower()
                if "eta" in eff_name_lower:
                    symbol = "η²_p"
                elif "epsilon" in eff_name_lower:
                    symbol = "ε²"
                elif "kendall" in eff_name_lower or "w" in eff_name_lower:
                    symbol = "W"
            else:
                if node_id in ['IND_ONE_WAY', 'IND_TWO_WAY', 'RM_ANOVA_STANDARD', 'RM_ANOVA_CORRECTED', 'MIXED_ANOVA_STANDARD', 'MIXED_ANOVA_CORRECTED', 'WELCH_ANOVA']:
                    symbol = "η²_p"
                elif node_id == 'K2_M_IND':
                    symbol = "ε²"
                elif node_id == 'NP_RM_ROBUST':
                    symbol = "W"
                elif node_id in ['K1_2_IND', 'K1_2_DEP', 'WELCH_T_TEST']:
                    symbol = "d"

            try:
                eff_str = f"{symbol} = {eff_val:.3f}" if isinstance(eff_val, (int, float)) else f"{symbol} = {eff_val}"
            except (ValueError, TypeError):
                eff_str = f"{symbol} = {eff_val}"

        stat_symbol = ""
        if node_id in ['K1_2_IND', 'K1_2_DEP', 'WELCH_T_TEST']:
            stat_symbol = "t"
        elif node_id in ['IND_ONE_WAY', 'IND_TWO_WAY', 'RM_ANOVA_STANDARD', 'RM_ANOVA_CORRECTED', 'MIXED_ANOVA_STANDARD', 'MIXED_ANOVA_CORRECTED', 'WELCH_ANOVA']:
            stat_symbol = "F"
        elif node_id == 'K2_2_IND':
            stat_symbol = "U"
        elif node_id == 'K2_2_DEP':
            stat_symbol = "W"
        elif node_id == 'K2_M_IND':
            stat_symbol = "H"
        elif node_id == 'NP_RM_ROBUST':
            stat_symbol = "Q"
        elif node_id == 'NP_MIXED_ROBUST':
            stat_symbol = "ATS"

        if stat_symbol and stat_str and p_str:
            line2 = f"{stat_symbol}{df_str} = {stat_str} | {p_str}"
            if eff_str:
                return f"{base_label}\n{line2}\n{eff_str}"
            else:
                return f"{base_label}\n{line2}"
        
        return base_label

    def create_association_tree(self, results, output_path=None):
        """Generates a flowchart for sequential/pipeline workflows.

        Delegates to FlowchartVisualizer for Correlation, LinearRegression,
        LogisticRegression, ANCOVA, LMM, and CorrelationMatrix model types.
        Returns the saved file path (str) on success, or None on failure.
        """
        from visualization.flowchartvisualizer import FlowchartVisualizer
        return FlowchartVisualizer.visualize(results, output_path)

    @staticmethod
    def _mixed_posthoc_node(posthoc_test: str) -> str:
        """Map a Mixed-ANOVA post-hoc test name to its decision-tree node.

        Treatment-vs-control / EMM contrasts are between-group comparisons, so
        they belong on the between-groups branch, not the within-subject one
        (which would mislead the user about what was actually compared).
        """
        ph = (posthoc_test or "").lower()
        if "tukey" in ph:
            return "MIXED_TUKEY"
        if any(kw in ph for kw in ("between", "dunnett", "emm", "multivariate")):
            return "MIXED_BETWEEN"
        return "MIXED_WITHIN"

    @staticmethod
    def get_tree_json(results: dict) -> dict | None:
        """
        Returns the decision tree topology as a JSON-serializable dict.
        Nodes and edges carry an isActive flag marking the taken path.
        """
        try:
            _model_type = results.get("model_type", "")
            if _model_type in ["Correlation", "LinearRegression", "LogisticRegression",
                                "ANCOVA", "LMM", "CorrelationMatrix"]:
                from visualization.flowchartvisualizer import FlowchartVisualizer
                return FlowchartVisualizer.get_tree_json(results)

            test_name = results.get("test_name", results.get("test", "")) or ""
            test_type = results.get("test_recommendation", results.get("test_type", ""))
            transformation = results.get("transformation", "None")
            p_value = results.get("p_value", None)

            test_info = results.get("test_info", {})
            normality_tests = test_info.get("normality_tests", results.get("normality_tests", {}))
            variance_test = test_info.get("variance_test", results.get("variance_test", {}))
            # Fallback: read from nested pre_transformation / post_transformation structure
            if not normality_tests and test_info:
                _has_tr = test_info.get("transformation") not in (None, "None", "No further")
                _phase = "post_transformation" if _has_tr else "pre_transformation"
                normality_tests = {"all_data": test_info.get(_phase, {}).get("residuals_normality", {})}
                variance_test = test_info.get(_phase, {}).get("variance", {})
            sphericity_test = results.get("sphericity_test", results.get("within_sphericity_test", {}))
            posthoc_test = str(results.get("posthoc_test") or "")

            # dependence_type
            dependence_type = "independent"
            dependent_param = results.get("dependent", None)
            dependent_samples_param = results.get("dependent_samples", None)
            if isinstance(dependent_param, bool):
                dependence_type = "dependent" if dependent_param else "independent"
            elif isinstance(dependent_samples_param, bool):
                dependence_type = "dependent" if dependent_samples_param else "independent"
            elif str(dependent_param or "").lower() in ("true", "1"):
                dependence_type = "dependent"
            elif str(dependent_samples_param or "").lower() in ("true", "1"):
                dependence_type = "dependent"
            elif any(kw in test_name.lower() for kw in ("repeated", "rm ", "within", "mixed", "paired")):
                dependence_type = "dependent"

            # normality / variance
            if isinstance(normality_tests, dict):
                group_results = [
                    v.get("is_normal", v.get("p_value", None) is not None and v.get("p_value", 0) > 0.05)
                    for k, v in normality_tests.items() if k not in ("all_data", "transformed_data")
                ]
                is_normal = all(group_results) if group_results else (
                    normality_tests.get("transformed_data", {}).get("is_normal", False)
                    if "transformed_data" in normality_tests
                    else normality_tests.get("all_data", {}).get("is_normal", False)
                )
            else:
                is_normal = False

            has_equal_variance = (
                variance_test.get("transformed", {}).get("equal_variance", False)
                if "transformed" in variance_test
                else variance_test.get("equal_variance", False)
            )
            
            if not isinstance(sphericity_test, dict):
                sphericity_test = {}
            has_sphericity = sphericity_test.get("sphericity_assumed", None)
            was_transformed = transformation != "None"

            # Resolve retroactive pre-transformation contradiction:
            pre_trans = test_info.get("pre_transformation", {}) if isinstance(test_info, dict) else {}
            pre_is_normal = None
            pre_has_equal_variance = None
            if pre_trans:
                pre_norm = pre_trans.get("residuals_normality", {})
                if isinstance(pre_norm, dict):
                    pre_is_normal = pre_norm.get("is_normal", None)
                pre_var = pre_trans.get("variance", {})
                if isinstance(pre_var, dict):
                    pre_has_equal_variance = pre_var.get("equal_variance", None)

            # Fallbacks:
            if pre_is_normal is None:
                pre_is_normal = is_normal if not was_transformed else False
            if pre_has_equal_variance is None:
                pre_has_equal_variance = has_equal_variance if not was_transformed else True

            auto_switched = (
                "Switching to nonparametric" in str(results.get("analysis_log", "")) or
                test_name.lower().startswith("nonparametric_")
            )

            # n_groups
            n_groups = 0
            for src in ("groups", ):
                if results.get(src):
                    n_groups = len(results[src]); break
            if not n_groups:
                for src in [
                    lambda: results.get("descriptive_stats", {}).get("groups"),
                    lambda: list(results.get("raw_data", {}).keys()) if results.get("raw_data") else None,
                    lambda: list(results.get("descriptive", {}).keys()) if results.get("descriptive") else None,
                    lambda: list((results.get("descriptive_stats", {}).get("means") or {}).keys()),
                ]:
                    v = src()
                    if v:
                        n_groups = len(v); break
            if n_groups == 0:
                n_groups = 2

            n_within_levels = results.get("n_within_levels", None)
            actual_test_type = test_type or results.get("recommendation", "")

            if n_within_levels == 2:
                k1_m_sph_label = "Sphericity N/A\n(< 3 Levels)"
            elif has_sphericity is True:
                k1_m_sph_label = "Sphericity Met\n(Mauchly's p > 0.05)"
            elif has_sphericity is False:
                k1_m_sph_label = "Sphericity Violated\n(Correction Applied)"
            else:
                k1_m_sph_label = "Sphericity Check\n(Mauchly's Test)"
            correction_used = str(results.get("correction_used", "None"))
            within_correction = str(results.get("within_correction_used", "None"))

            strategy = results.get("strategy") or results.get("test_info", {}).get("decision", {}).get("strategy")
            welch_t_condition = (strategy == "welch_ttest")
            welch_anova_condition = (strategy == "welch_anova")

            # labels
            if welch_t_condition or welch_anova_condition:
                f_label = "Normal data, unequal variances\n-> Welch correction"
            elif actual_test_type.lower() == "parametric":
                f_label = "Assumptions met\n-> parametric test"
            elif actual_test_type.lower() in ("non_parametric", "non-parametric"):
                f_label = "Assumptions violated\n-> non-parametric test"
            else:
                f_label = "Test Recommendation"

            # nodes
            nodes_info = {
                'A':  {"label": "Start", "pos": (0, 14)},
                'B':  {"label": f"Are the data normally distributed and variances equal?\nShapiro-Wilk: {pre_is_normal}  |  Brown-Forsythe: {pre_has_equal_variance}", "pos": (0, 12.5)},
                'C':  {"label": f"Assumptions {'met' if pre_is_normal and pre_has_equal_variance else 'violated'}", "pos": (0, 11)},
                'D1': {"label": "Data is ready\n(no transformation needed)", "pos": (-2, 9.5)},
                'D2': {"label": f"Transform the data\n({transformation})", "pos": (2, 9.5)},
                'E':  {"label": f"Check again after transformation\nShapiro-Wilk: {is_normal}  |  Brown-Forsythe: {has_equal_variance}" if was_transformed else "Check again after transformation", "pos": (2, 8)},
                'F':  {"label": f_label, "pos": (0, 6.5)},

                'G1': {"label": "Standard parametric test", "pos": (-10, 5)},
                'H1': {"label": "How many groups?", "pos": (-10, 4)},
                'I1_2': {"label": "2 groups", "pos": (-13, 3)},
                'I1_M': {"label": "3 or more groups", "pos": (-3, 3)},
                'J1_INDEP': {"label": "Independent\n(different subjects)", "pos": (-14, 2)},
                'J1_DEP':   {"label": "Dependent\n(same subjects)", "pos": (-12, 2)},
                'K1_2_IND': {"label": "Welch's t-test", "pos": (-14, 1)},
                'K1_2_DEP': {"label": "Paired t-test", "pos": (-12, 1)},
                'INDEPENDENT_GROUPS': {"label": "Different subjects\nin each group", "pos": (-8, 2)},
                'REPEATED_MEASURES':  {"label": "Same subjects\nmeasured multiple times", "pos": (-2, 2)},
                'MIXED_DESIGN':        {"label": "Mixed design\n(between + within factors)", "pos": (4, 2)},
                'IND_ONE_WAY':   {"label": "Welch's ANOVA", "pos": (-9, 1)},
                'IND_TWO_WAY':   {"label": "Two-way ANOVA", "pos": (-7, 1)},
                'IND_POSTHOC':   {"label": "Which specific groups differ?", "pos": (-8, 0)},
                'IND_TUKEY':     {"label": "Tukey HSD /\nGames-Howell", "pos": (-9.5, -1)},
                'IND_DUNNETT':   {"label": "Dunnett Test", "pos": (-8, -1)},
                'IND_HOLM_SIDAK':{"label": "Pairwise t-tests\n(Holm-Šidák)", "pos": (-6.5, -1)},
                'RM_MAUCHLY':            {"label": k1_m_sph_label, "pos": (-2, 1)},
                'RM_SPHERICITY_OK':      {"label": "Even correlation\n-> no correction needed", "pos": (-3.5, 0)},
                'RM_SPHERICITY_VIOLATED':{"label": "Uneven correlation\n-> correction needed", "pos": (-0.5, 0)},
                'RM_CHOOSE_CORRECTION':  {"label": "Select sphericity\ncorrection", "pos": (-0.5, -1)},
                'RM_GG_CORRECTION':      {"label": "Greenhouse-Geisser\n(conservative)", "pos": (-1.5, -2)},
                'RM_HF_CORRECTION':      {"label": "Huynh-Feldt\n(less conservative)", "pos": (0.5, -2)},
                'RM_ANOVA_STANDARD':     {"label": "RM ANOVA", "pos": (-3.5, -1)},
                'RM_ANOVA_CORRECTED':    {"label": "RM ANOVA\n(Corrected)", "pos": (-0.5, -3)},
                'RM_POSTHOC':            {"label": "Which time points differ?", "pos": (-2, -4)},
                'RM_TUKEY':              {"label": "Tukey HSD\n(RM)", "pos": (-3.5, -5)},
                'RM_PAIRED_TESTS':       {"label": "Pairwise Paired t-tests\n(Holm-Šidák)", "pos": (-0.5, -5)},
                'MIXED_MAUCHLY':             {"label": k1_m_sph_label, "pos": (4, 1)},
                'MIXED_SPHERICITY_OK':       {"label": "Even correlation\n-> no correction needed", "pos": (2.5, 0)},
                'MIXED_SPHERICITY_VIOLATED': {"label": "Uneven correlation\n-> correction needed", "pos": (5.5, 0)},
                'MIXED_CHOOSE_CORRECTION':   {"label": "Select sphericity\ncorrection", "pos": (5.5, -1)},
                'MIXED_GG_CORRECTION':       {"label": "Greenhouse-Geisser\n(conservative)", "pos": (4.5, -2)},
                'MIXED_HF_CORRECTION':       {"label": "Huynh-Feldt\n(less conservative)", "pos": (6.5, -2)},
                'MIXED_ANOVA_STANDARD':      {"label": "Mixed ANOVA", "pos": (2.5, -1)},
                'MIXED_ANOVA_CORRECTED':     {"label": "Mixed ANOVA\n(Within Corrected)", "pos": (5.5, -3)},
                'MIXED_POSTHOC':             {"label": "Which groups / time points differ?", "pos": (4, -4)},
                'MIXED_TUKEY':   {"label": "Mixed Tukey\n(Between/Within)", "pos": (2, -5)},
                'MIXED_BETWEEN': {"label": "Between groups\n(different subjects)", "pos": (4, -5)},
                'MIXED_WITHIN':  {"label": "Within group\n(same subjects over time)", "pos": (6, -5)},
                'G2': {"label": "Non-parametric test\n(rank-based)", "pos": (10, 5)},
                'H2': {"label": "How many groups?", "pos": (10, 4)},
                'I2_2': {"label": "2 groups", "pos": (8, 3)},
                'I2_M': {"label": "3 or more groups", "pos": (14, 3)},
                'J2_INDEP': {"label": "Independent\n(different subjects)", "pos": (7, 2)},
                'J2_DEP':   {"label": "Dependent\n(same subjects)", "pos": (9, 2)},
                'K2_2_IND': {"label": "Mann-Whitney U", "pos": (7, 1)},
                'K2_2_DEP': {"label": "Wilcoxon\nSigned-Rank", "pos": (9, 1)},
                'NP_INDEPENDENT_GROUPS': {"label": "Different subjects\nin each group", "pos": (12.5, 2)},
                'NP_REPEATED_MEASURES':  {"label": "Same subjects\nmeasured multiple times", "pos": (16, 2)},
                'NP_MIXED_DESIGN':       {"label": "Mixed design\n(between + within)", "pos": (20, 2)},
                'K2_M_IND':           {"label": "Kruskal-Wallis", "pos": (11.5, 1)},
                'NP_POSTHOC':         {"label": "Which groups differ?", "pos": (11.5, 0)},
                'NP_DUNN':            {"label": "Dunn Test", "pos": (10.5, -1)},
                'NP_MANN_WHITNEY':    {"label": "Pairwise\nMann-Whitney U", "pos": (12.5, -1)},
                'NP_TWO_WAY_ROBUST':  {"label": "Freedman-Lane\nPermutation", "pos": (13.5, 1)},
                'NP_TWO_WAY_POSTHOC': {"label": "Which groups / conditions differ?", "pos": (13.5, 0)},
                'NP_TWO_WAY_PAIRWISE':{"label": "Marginal Effects\nPairwise", "pos": (13.5, -1)},
                'NP_RM_ROBUST':   {"label": "Friedman Test", "pos": (16, 1)},
                'NP_RM_POSTHOC':  {"label": "Which time points differ?", "pos": (16, 0)},
                'NP_RM_PAIRWISE': {"label": "RM Pairwise\nComparisons", "pos": (16, -1)},
                'NP_MIXED_ROBUST':   {"label": "Brunner-Langer\nATS", "pos": (20, 1)},
                'NP_MIXED_POSTHOC':  {"label": "Which groups / time points differ?", "pos": (20, 0)},
                'NP_MIXED_BETWEEN':  {"label": "Between groups\n(different subjects)", "pos": (19, -1)},
                'NP_MIXED_WITHIN':   {"label": "Within group\n(same subjects over time)", "pos": (21, -1)},
            }
            nodes_info = DecisionTreeVisualizer._apply_wide_canvas_layout(nodes_info)

            edges = {
                ('A','B'),('B','C'),('C','D1'),('C','D2'),('D2','E'),('E','F'),('D1','F'),

                ('F','G1'),('F','G2'),
                ('G1','H1'),('H1','I1_2'),('H1','I1_M'),
                ('I1_2','J1_INDEP'),('I1_2','J1_DEP'),('J1_INDEP','K1_2_IND'),('J1_DEP','K1_2_DEP'),
                ('I1_M','INDEPENDENT_GROUPS'),('I1_M','REPEATED_MEASURES'),('I1_M','MIXED_DESIGN'),
                ('INDEPENDENT_GROUPS','IND_ONE_WAY'),('INDEPENDENT_GROUPS','IND_TWO_WAY'),
                ('IND_ONE_WAY','IND_POSTHOC'),('IND_TWO_WAY','IND_POSTHOC'),
                ('IND_POSTHOC','IND_TUKEY'),('IND_POSTHOC','IND_DUNNETT'),('IND_POSTHOC','IND_HOLM_SIDAK'),
                ('REPEATED_MEASURES','RM_MAUCHLY'),
                ('RM_MAUCHLY','RM_SPHERICITY_OK'),('RM_MAUCHLY','RM_SPHERICITY_VIOLATED'),
                ('RM_SPHERICITY_OK','RM_ANOVA_STANDARD'),
                ('RM_SPHERICITY_VIOLATED','RM_CHOOSE_CORRECTION'),
                ('RM_CHOOSE_CORRECTION','RM_GG_CORRECTION'),('RM_CHOOSE_CORRECTION','RM_HF_CORRECTION'),
                ('RM_GG_CORRECTION','RM_ANOVA_CORRECTED'),('RM_HF_CORRECTION','RM_ANOVA_CORRECTED'),
                ('RM_ANOVA_STANDARD','RM_POSTHOC'),('RM_ANOVA_CORRECTED','RM_POSTHOC'),
                ('RM_POSTHOC','RM_TUKEY'),('RM_POSTHOC','RM_PAIRED_TESTS'),
                ('MIXED_DESIGN','MIXED_MAUCHLY'),
                ('MIXED_MAUCHLY','MIXED_SPHERICITY_OK'),('MIXED_MAUCHLY','MIXED_SPHERICITY_VIOLATED'),
                ('MIXED_SPHERICITY_OK','MIXED_ANOVA_STANDARD'),
                ('MIXED_SPHERICITY_VIOLATED','MIXED_CHOOSE_CORRECTION'),
                ('MIXED_CHOOSE_CORRECTION','MIXED_GG_CORRECTION'),('MIXED_CHOOSE_CORRECTION','MIXED_HF_CORRECTION'),
                ('MIXED_GG_CORRECTION','MIXED_ANOVA_CORRECTED'),('MIXED_HF_CORRECTION','MIXED_ANOVA_CORRECTED'),
                ('MIXED_ANOVA_STANDARD','MIXED_POSTHOC'),('MIXED_ANOVA_CORRECTED','MIXED_POSTHOC'),
                ('MIXED_POSTHOC','MIXED_TUKEY'),('MIXED_POSTHOC','MIXED_BETWEEN'),('MIXED_POSTHOC','MIXED_WITHIN'),
                ('G2','H2'),('H2','I2_2'),('H2','I2_M'),
                ('I2_2','J2_INDEP'),('I2_2','J2_DEP'),('J2_INDEP','K2_2_IND'),('J2_DEP','K2_2_DEP'),
                ('I2_M','NP_INDEPENDENT_GROUPS'),('I2_M','NP_REPEATED_MEASURES'),('I2_M','NP_MIXED_DESIGN'),
                ('NP_INDEPENDENT_GROUPS','K2_M_IND'),('K2_M_IND','NP_POSTHOC'),
                ('NP_POSTHOC','NP_DUNN'),('NP_POSTHOC','NP_MANN_WHITNEY'),
                ('NP_INDEPENDENT_GROUPS','NP_TWO_WAY_ROBUST'),
                ('NP_TWO_WAY_ROBUST','NP_TWO_WAY_POSTHOC'),('NP_TWO_WAY_POSTHOC','NP_TWO_WAY_PAIRWISE'),
                ('NP_REPEATED_MEASURES','NP_RM_ROBUST'),('NP_RM_ROBUST','NP_RM_POSTHOC'),
                ('NP_RM_POSTHOC','NP_RM_PAIRWISE'),
                ('NP_MIXED_DESIGN','NP_MIXED_ROBUST'),('NP_MIXED_ROBUST','NP_MIXED_POSTHOC'),
                ('NP_MIXED_POSTHOC','NP_MIXED_BETWEEN'),('NP_MIXED_POSTHOC','NP_MIXED_WITHIN'),
            }

            # highlighted path — same logic as visualize()
            highlighted = set()
            highlighted.add(('A', 'B'))
            highlighted.add(('B', 'C'))
            if transformation and transformation != "None":
                highlighted.update([('C','D2'),('D2','E'),('E','F')])
            else:
                highlighted.update([('C','D1'),('D1','F')])

            recommendation_text = str(results.get("recommendation", "")).lower()
            model_class_text    = str(results.get("model_class", "")).lower()
            analysis_log_text   = str(results.get("analysis_log", "")).lower()
            test_name_text      = test_name.lower()

            is_modern_or_robust_fallback = any(kw in t for kw in ("fallback","modern model","robust") for t in (recommendation_text, analysis_log_text))
            is_nonparam_two_way_advanced = (("two-way" in test_name_text or "two way" in test_name_text) and (is_modern_or_robust_fallback or "freedman" in model_class_text or "permutation" in model_class_text))
            is_nonparam_rm_advanced      = (("repeated" in test_name_text or "rm anova" in test_name_text) and (is_modern_or_robust_fallback or "friedman" in model_class_text))
            is_nonparam_mixed_advanced   = ("mixed" in test_name_text and (is_modern_or_robust_fallback or "brunner" in model_class_text or "ats" in model_class_text))
            is_nonparametric_test = (
                actual_test_type.lower() in ("non-parametric","non_parametric") or
                test_name.lower().startswith(("non-parametric","nonparametric")) or
                "rank + permutation" in test_name.lower() or auto_switched or
                results.get("recommendation") == "non_parametric" or
                results.get("parametric_assumptions_violated", False)
            )

            alpha = results.get("alpha", 0.05)

            # Ensure welch bypass condition is disabled so we flow into standard tree branch
            welch_t_condition = False
            welch_anova_condition = False

            if is_nonparametric_test:
                if not auto_switched:
                    highlighted.add(('F','G2'))
                highlighted.add(('G2','H2'))
                if n_groups == 2:
                    highlighted.add(('H2','I2_2'))
                    if dependence_type == "dependent" or "paired" in test_name_text or "wilcoxon" in test_name_text:
                        highlighted.update([('I2_2','J2_DEP'),('J2_DEP','K2_2_DEP')])
                    else:
                        highlighted.update([('I2_2','J2_INDEP'),('J2_INDEP','K2_2_IND')])
                else:
                    highlighted.add(('H2','I2_M'))
                    if is_nonparam_two_way_advanced:
                        highlighted.update([('I2_M','NP_INDEPENDENT_GROUPS'),('NP_INDEPENDENT_GROUPS','NP_TWO_WAY_ROBUST')])
                        if p_value is not None and p_value < alpha:
                            highlighted.update([('NP_TWO_WAY_ROBUST','NP_TWO_WAY_POSTHOC'),('NP_TWO_WAY_POSTHOC','NP_TWO_WAY_PAIRWISE')])
                    elif is_nonparam_rm_advanced:
                        highlighted.update([('I2_M','NP_REPEATED_MEASURES'),('NP_REPEATED_MEASURES','NP_RM_ROBUST')])
                        if p_value is not None and p_value < alpha:
                            highlighted.update([('NP_RM_ROBUST','NP_RM_POSTHOC'),('NP_RM_POSTHOC','NP_RM_PAIRWISE')])
                    elif is_nonparam_mixed_advanced:
                        highlighted.update([('I2_M','NP_MIXED_DESIGN'),('NP_MIXED_DESIGN','NP_MIXED_ROBUST')])
                        if p_value is not None and p_value < alpha:
                            highlighted.add(('NP_MIXED_ROBUST','NP_MIXED_POSTHOC'))
                            ph = posthoc_test.lower()
                            if "between" in ph:
                                highlighted.add(('NP_MIXED_POSTHOC','NP_MIXED_BETWEEN'))
                            elif "within" in ph:
                                highlighted.add(('NP_MIXED_POSTHOC','NP_MIXED_WITHIN'))
                            else:
                                highlighted.update([('NP_MIXED_POSTHOC','NP_MIXED_BETWEEN'),('NP_MIXED_POSTHOC','NP_MIXED_WITHIN')])
                    else:
                        highlighted.update([('I2_M','NP_INDEPENDENT_GROUPS'),('NP_INDEPENDENT_GROUPS','K2_M_IND')])
                        if p_value is not None and p_value < alpha:
                            highlighted.add(('K2_M_IND','NP_POSTHOC'))
                            ph = posthoc_test.lower()
                            highlighted.add(('NP_POSTHOC','NP_DUNN') if "dunn" in ph else ('NP_POSTHOC','NP_MANN_WHITNEY'))

            elif (actual_test_type.lower() == "parametric"
                  or "welch" in test_name.lower()
                  or "t-test" in test_name.lower()
                  or ("anova" in test_name.lower() and "non" not in test_name.lower())):
                if not auto_switched:
                    highlighted.add(('F','G1'))
                highlighted.add(('G1','H1'))
                if n_groups == 2:
                    highlighted.add(('H1','I1_2'))
                    if dependence_type == "dependent":
                        highlighted.update([('I1_2','J1_DEP'),('J1_DEP','K1_2_DEP')])
                    else:
                        highlighted.update([('I1_2','J1_INDEP'),('J1_INDEP','K1_2_IND')])
                else:
                    highlighted.add(('H1','I1_M'))
                    if "repeated" in test_name_text or ("rm" in test_name_text and "anova" in test_name_text):
                        highlighted.update([('I1_M','REPEATED_MEASURES'),('REPEATED_MEASURES','RM_MAUCHLY')])
                        if any(kw in correction_used.lower() for kw in ("greenhouse","gg")) or "greenhouse" in within_correction.lower():
                            highlighted.update([('RM_MAUCHLY','RM_SPHERICITY_VIOLATED'),('RM_SPHERICITY_VIOLATED','RM_CHOOSE_CORRECTION'),('RM_CHOOSE_CORRECTION','RM_GG_CORRECTION'),('RM_GG_CORRECTION','RM_ANOVA_CORRECTED')])
                        elif any(kw in correction_used.lower() for kw in ("huynh","h")) or "huynh" in within_correction.lower():
                            highlighted.update([('RM_MAUCHLY','RM_SPHERICITY_VIOLATED'),('RM_SPHERICITY_VIOLATED','RM_CHOOSE_CORRECTION'),('RM_CHOOSE_CORRECTION','RM_HF_CORRECTION'),('RM_HF_CORRECTION','RM_ANOVA_CORRECTED')])
                        else:
                            highlighted.update([('RM_MAUCHLY','RM_SPHERICITY_OK'),('RM_SPHERICITY_OK','RM_ANOVA_STANDARD')])
                        if p_value is not None and p_value < alpha:
                            if any(kw in correction_used.lower() for kw in ("greenhouse","huynh")) or any(kw in within_correction.lower() for kw in ("greenhouse","huynh")):
                                highlighted.add(('RM_ANOVA_CORRECTED','RM_POSTHOC'))
                            else:
                                highlighted.add(('RM_ANOVA_STANDARD','RM_POSTHOC'))
                            highlighted.add(('RM_POSTHOC','RM_TUKEY') if "tukey" in posthoc_test.lower() else ('RM_POSTHOC','RM_PAIRED_TESTS'))
                    elif "mixed" in test_name_text:
                        highlighted.update([('I1_M','MIXED_DESIGN'),('MIXED_DESIGN','MIXED_MAUCHLY')])
                        if any(kw in within_correction.lower() for kw in ("greenhouse","gg")) or "greenhouse" in correction_used.lower():
                            highlighted.update([('MIXED_MAUCHLY','MIXED_SPHERICITY_VIOLATED'),('MIXED_SPHERICITY_VIOLATED','MIXED_CHOOSE_CORRECTION'),('MIXED_CHOOSE_CORRECTION','MIXED_GG_CORRECTION'),('MIXED_GG_CORRECTION','MIXED_ANOVA_CORRECTED')])
                        elif any(kw in within_correction.lower() for kw in ("huynh","h")) or "huynh" in correction_used.lower():
                            highlighted.update([('MIXED_MAUCHLY','MIXED_SPHERICITY_VIOLATED'),('MIXED_SPHERICITY_VIOLATED','MIXED_CHOOSE_CORRECTION'),('MIXED_CHOOSE_CORRECTION','MIXED_HF_CORRECTION'),('MIXED_HF_CORRECTION','MIXED_ANOVA_CORRECTED')])
                        else:
                            highlighted.update([('MIXED_MAUCHLY','MIXED_SPHERICITY_OK'),('MIXED_SPHERICITY_OK','MIXED_ANOVA_STANDARD')])
                        if p_value is not None and p_value < alpha:
                            if any(kw in within_correction.lower() for kw in ("greenhouse","huynh")):
                                highlighted.add(('MIXED_ANOVA_CORRECTED','MIXED_POSTHOC'))
                            else:
                                highlighted.add(('MIXED_ANOVA_STANDARD','MIXED_POSTHOC'))
                            highlighted.add((
                                'MIXED_POSTHOC',
                                DecisionTreeVisualizer._mixed_posthoc_node(posthoc_test),
                            ))
                    elif "two-way" in test_name_text or "two way" in test_name_text:
                        highlighted.update([('I1_M','INDEPENDENT_GROUPS'),('INDEPENDENT_GROUPS','IND_TWO_WAY')])
                        if p_value is not None and p_value < alpha:
                            highlighted.add(('IND_TWO_WAY','IND_POSTHOC'))
                            ph = posthoc_test.lower()
                            if "tukey" in ph or "games" in ph or "howell" in ph: highlighted.add(('IND_POSTHOC','IND_TUKEY'))
                            elif "dunnett" in ph: highlighted.add(('IND_POSTHOC','IND_DUNNETT'))
                            else: highlighted.add(('IND_POSTHOC','IND_HOLM_SIDAK'))
                    else:
                        highlighted.update([('I1_M','INDEPENDENT_GROUPS'),('INDEPENDENT_GROUPS','IND_ONE_WAY')])
                        if p_value is not None and p_value < alpha:
                            highlighted.add(('IND_ONE_WAY','IND_POSTHOC'))
                            ph = posthoc_test.lower()
                            if "tukey" in ph or "games" in ph or "howell" in ph: highlighted.add(('IND_POSTHOC','IND_TUKEY'))
                            elif "dunnett" in ph: highlighted.add(('IND_POSTHOC','IND_DUNNETT'))
                            else: highlighted.add(('IND_POSTHOC','IND_HOLM_SIDAK'))

            # active node set
            active_nodes = set()
            for u, v in highlighted:
                active_nodes.add(u)
                active_nodes.add(v)

            _square_keywords = {
                "Start","Check Assumptions","Assumptions","Parametric Test","Non-parametric Test",
                "Group Structure","Two Groups","Multiple Groups","Independent Samples","Dependent Samples",
                "Repeated Measures","Mixed Design","Post-hoc Tests","Independent Groups",
                "Mauchly's Test","Sphericity","Choose Correction","RM Post-hoc","Mixed Post-hoc",
            }
            def _is_square(node_id):
                lbl = nodes_info[node_id]["label"].replace('\n',' ')
                return any(kw in lbl for kw in _square_keywords)

            test_nodes = {
                'WELCH_T_TEST', 'WELCH_ANOVA', 'K1_2_IND', 'K1_2_DEP', 
                'IND_ONE_WAY', 'IND_TWO_WAY', 'RM_ANOVA_STANDARD', 'RM_ANOVA_CORRECTED', 
                'MIXED_ANOVA_STANDARD', 'MIXED_ANOVA_CORRECTED', 'K2_2_IND', 'K2_2_DEP', 
                'K2_M_IND', 'NP_RM_ROBUST', 'NP_MIXED_ROBUST'
            }
            node_list = []
            for nid, info in nodes_info.items():
                lbl = info["label"]
                if nid in active_nodes and nid in test_nodes:
                    lbl = DecisionTreeVisualizer.format_dynamic_test_label(nid, lbl, results)
                node_list.append({
                    "id": nid,
                    "x": float(info["pos"][0]),
                    "y": float(info["pos"][1]),
                    "label": lbl,
                    "isActive": nid in active_nodes,
                    "isSquare": _is_square(nid),
                })
            edge_list = [
                {"source": u, "target": v, "isActive": (u, v) in highlighted}
                for u, v in edges
            ]
            return {"nodes": node_list, "edges": edge_list}

        except Exception as exc:
            logger.warning(f"WARNING DecisionTreeVisualizer.get_tree_json: {exc}")
            return None

