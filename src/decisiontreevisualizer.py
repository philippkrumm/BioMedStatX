import tempfile
import os


class DecisionTreeVisualizer:
    """
    Creates visual decision trees for statistical test workflows with the actual path highlighted.
    Uses networkx and matplotlib to generate a directed graph showing the decision-making process.
    """

    WIDE_LAYOUT_TOP_X_SCALE = 1.35
    WIDE_LAYOUT_BOTTOM_X_SCALE = 2.25
    WIDE_LAYOUT_TOP_Y_SCALE = 1.15
    WIDE_LAYOUT_BOTTOM_Y_SCALE = 1.55

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
    def _calculate_figure_size(pos):
        """Derive a figure size from node extents so wide layouts are not compressed."""
        if not pos:
            return 24.0, 16.0

        x_values = [xy[0] for xy in pos.values()]
        y_values = [xy[1] for xy in pos.values()]
        x_span = max(x_values) - min(x_values)
        y_span = max(y_values) - min(y_values)

        width = max(24.0, min(42.0, (x_span * 0.58) + 4.0))
        height = max(16.0, min(32.0, (y_span * 1.05) + 5.0))
        return width, height

    @staticmethod
    def format_dynamic_test_label(node_id, base_label, results):
        """
        Dynamically format a test node label with its statistic, df, p-value, and effect size.
        """
        if not results or not isinstance(results, dict):
            return base_label

        stat_val = results.get("statistic", results.get("stat", None))
        p_val = results.get("p_value", results.get("p-val", results.get("p", None)))
        df = results.get("df", None)
        df1 = results.get("df1", results.get("df_num", None))
        df2 = results.get("df2", results.get("df_den", None))
        eff_val = results.get("effect_size", results.get("eff", None))
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

    @staticmethod
    def visualize(results, output_path=None):
        """
        Generate a decision tree visualization based on the provided test results.

        Parameters:
        -----------
        results : dict
            Results dictionary containing test information
        output_path : str, optional
            Path to save the visualization

        Returns:
        --------
        str
            Path to the saved visualization file
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            # Apply a clean style for better aesthetics
            plt.style.use('seaborn-v0_8-whitegrid')

            # Extract key information from results
            test_name = results.get("test_name", results.get("test", ""))
            test_type = results.get("test_recommendation", results.get("test_type", ""))
            transformation = results.get("transformation", "None")
            p_value = results.get("p_value", None)

            # Route to specialized visualizers for association/regression tests
            _model_type = results.get("model_type", "")
            if _model_type in ["Correlation", "LinearRegression", "LogisticRegression", "ANCOVA", "CorrelationMatrix"]:
                return DecisionTreeVisualizer._visualize_association_test(results, _model_type, output_path)

            # Get test_info for more detailed analysis
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
            posthoc_test = results.get("posthoc_test", None)

            # BETTER DETECTION OF TEST TYPE FOR T-TESTS
            # This needs to be at the beginning of the visualize method
            dependence_type = "independent"  # Default
            
            # Check for explicit indicators in the parameters
            dependent_param = results.get("dependent", None)
            dependent_samples_param = results.get("dependent_samples", None)
            print(f"DEBUG TREE: dependent_param={dependent_param}, dependent_samples_param={dependent_samples_param}")
            
            if isinstance(dependent_param, bool):
                dependence_type = "dependent" if dependent_param else "independent"
            elif isinstance(dependent_samples_param, bool):
                dependence_type = "dependent" if dependent_samples_param else "independent"
            elif dependent_param is not None:
                # Fallback: try to interpret string or int
                if str(dependent_param).lower() in ("true", "1"):
                    dependence_type = "dependent"
                else:
                    dependence_type = "independent"
            elif dependent_samples_param is not None:
                # Fallback: try to interpret string or int
                if str(dependent_samples_param).lower() in ("true", "1"):
                    dependence_type = "dependent"
                else:
                    dependence_type = "independent"
            elif "repeated" in test_name.lower() or "rm" in test_name.lower() or "within" in test_name.lower():
                # RM ANOVA and Mixed ANOVA are always dependent samples
                dependence_type = "dependent"
            elif "mixed" in test_name.lower():
                # Mixed ANOVA has both dependent and independent factors, but treat as dependent
                dependence_type = "dependent"
            elif "t-test" in test_name.lower() or "t test" in test_name.lower():
                # Check for "paired" or only match "dependent" when surrounded by spaces/parentheses
                if "paired" in test_name.lower() or " dependent" in test_name.lower() or "(dependent)" in test_name.lower():
                    dependence_type = "dependent"
                elif "independent" in test_name.lower():
                    dependence_type = "independent"
                
            print(f"DEBUG TREE: Final dependence_type='{dependence_type}' from test_name='{test_name}' and params")

            # Logic to determine test path - check transformed data first if available
            if isinstance(normality_tests, dict):
                group_norm_results = [
                    v.get("is_normal", v.get("p_value", None) is not None and v.get("p_value", 0) > 0.05)
                    for k, v in normality_tests.items()
                    if k not in ("all_data", "transformed_data")
                ]
                if group_norm_results:
                    is_normal = all(group_norm_results)
                else:
                    # fallback to summary keys
                    if "transformed_data" in normality_tests:
                        is_normal = normality_tests.get("transformed_data", {}).get("is_normal", False)
                    else:
                        is_normal = normality_tests.get("all_data", {}).get("is_normal", False)
            else:
                is_normal = False

            has_equal_variance = False
            if "transformed" in variance_test:
                has_equal_variance = variance_test.get("transformed", {}).get("equal_variance", False)
            else:
                has_equal_variance = variance_test.get("equal_variance", False)

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
            
            # Define auto_switched flag here
            auto_switched = False
            # Check for auto-switch in analysis log
            if results.get("analysis_log", ""):
                if "Switching to nonparametric" in results.get("analysis_log", ""):
                    auto_switched = True
            # Check test name for indicators
            if test_name.lower().startswith("nonparametric_"):
                auto_switched = True

            # Determine number of groups - check multiple locations WITH EXACT DATA
            n_groups = 0
            groups_found = []
            
            # Priority 1: Direct groups in results
            if "groups" in results and results["groups"]:
                groups_found = results["groups"]
                n_groups = len(groups_found)
                print(f"DEBUG TREE: Found groups from results['groups']: {groups_found}")
            
            # Priority 2: Groups from descriptive stats
            elif "descriptive_stats" in results and "groups" in results["descriptive_stats"]:
                groups_found = results["descriptive_stats"]["groups"]
                n_groups = len(groups_found)
                print(f"DEBUG TREE: Found groups from descriptive_stats: {groups_found}")
            
            # Priority 3: Groups from raw_data keys
            elif "raw_data" in results and results["raw_data"]:
                groups_found = list(results["raw_data"].keys())
                n_groups = len(groups_found)
                print(f"DEBUG TREE: Found groups from raw_data keys: {groups_found}")
            
            # Priority 4: Groups from descriptive dict keys
            elif "descriptive" in results and results["descriptive"]:
                groups_found = list(results["descriptive"].keys())
                n_groups = len(groups_found)
                print(f"DEBUG TREE: Found groups from descriptive keys: {groups_found}")
            
            # Priority 5: Extract from any means/stats data
            elif "descriptive_stats" in results and "means" in results["descriptive_stats"]:
                groups_found = list(results["descriptive_stats"]["means"].keys()) if isinstance(results["descriptive_stats"]["means"], dict) else []
                n_groups = len(groups_found)
                print(f"DEBUG TREE: Found groups from means: {groups_found}")
            
            # NO FALLBACK ASSUMPTIONS - If we can't find groups, something is wrong
            else:
                print(f"DEBUG TREE: WARNING - No groups found in results structure!")
                print(f"DEBUG TREE: Available keys in results: {list(results.keys())}")
                if "descriptive_stats" in results:
                    print(f"DEBUG TREE: Available keys in descriptive_stats: {list(results['descriptive_stats'].keys())}")
                n_groups = 0  # This will force an error rather than wrong assumptions

            print(f"DEBUG TREE: EXACT n_groups={n_groups} from ACTUAL data: {groups_found}")
            
            # Validation: Never assume, always use actual data
            if n_groups == 0:
                print(f"DEBUG TREE: ERROR - Could not determine actual number of groups from data!")
                n_groups = 2  # Minimal fallback only to prevent crashes

            n_within_levels = results.get("n_within_levels", None)

            # Create graph
            G = nx.DiGraph()

            # Decide label for test recommendation node and Welch conditions
            welch_t_condition = (
                # Direct detection from test name
                ("welch" in test_name.lower() and "t-test" in test_name.lower()) or
                ("welch" in test_name.lower() and n_groups == 2) or
                # Condition-based detection for 2 groups
                (is_normal and not has_equal_variance and n_groups == 2 and 
                 ("welch" in test_name.lower() or "independent" in test_name.lower()))
            )
            
            welch_anova_condition = (
                # Direct detection from test name
                ("welch" in test_name.lower() and "anova" in test_name.lower()) or
                ("welch" in test_name.lower() and n_groups > 2) or
                # Condition-based detection for multiple groups
                (is_normal and not has_equal_variance and n_groups > 2 and
                 ("welch" in test_name.lower() or "anova" in test_name.lower()))
            )
            
            actual_test_type = test_type or results.get("recommendation", "")
            print(f"DEBUG TREE: Building recommendation label from: test_type='{test_type}', recommendation='{results.get('recommendation', '')}', actual='{actual_test_type}'")
            print(f"DEBUG TREE: Welch conditions - t-test: {welch_t_condition}, ANOVA: {welch_anova_condition}")

            if welch_t_condition or welch_anova_condition:
                test_recommendation_label = "Test Recommendation:\nNormal distributed\nbut unequal variances"
            elif actual_test_type.lower() == "parametric":
                test_recommendation_label = "Test Recommendation:\nParametric Test"
            elif actual_test_type.lower() == "non_parametric" or actual_test_type.lower() == "non-parametric":
                test_recommendation_label = "Test Recommendation:\nNon-parametric Test"
            else:
                test_recommendation_label = "Test Recommendation"
                print(f"DEBUG TREE: Warning - using default label, actual_test_type was '{actual_test_type}'")

            # Update sphericity label logic
            if n_within_levels == 2:
                k1_m_sph_label = "Sphericity N/A\n(< 3 Levels)"
            elif has_sphericity is True:
                k1_m_sph_label = "Sphericity Met\n(Mauchly's p > 0.05)"
            elif has_sphericity is False:
                k1_m_sph_label = "Sphericity Violated\n(Correction Applied)"
            else:
                k1_m_sph_label = "Sphericity Check\n(Mauchly's Test)"

            # Get sphericity correction info from results
            sphericity_correction = "None"
            if "sphericity_test" in results:
                sph_test = results["sphericity_test"]
                if "correction_used" in results:
                    sphericity_correction = results["correction_used"]
                elif sph_test.get("sphericity_assumed", True) is False:
                    sphericity_correction = "Correction needed"

            # Get detailed correction information
            correction_used = results.get("correction_used", "None")
            within_correction = results.get("within_correction_used", "None")
            
            # Detect specific correction types
            is_greenhouse_geisser = ("greenhouse" in str(correction_used).lower() or 
                                   "gg" in str(correction_used).lower())
            is_huynh_feldt = ("huynh" in str(correction_used).lower() or 
                             "hf" in str(correction_used).lower())

            # Define nodes with positions and labels - LOGICAL GROUPING WITH PROPER SPACING
            nodes_info = {
                # Common path
                'A': {"label": "Start", "pos": (0, 14)},
                'B': {"label": f"Check Assumptions\nShapiro-Wilk: {pre_is_normal}\nBrown-Forsythe: {pre_has_equal_variance}", "pos": (0, 12.5)},
                'C': {"label": f"Assumptions{': ' + ('Met' if pre_is_normal and pre_has_equal_variance else 'Not Met')}", "pos": (0, 11)},

                # Transformation branch point
                'D1': {"label": f"No Transformation\nNeeded", "pos": (-2, 9.5)},
                'D2': {"label": f"Apply Transformation\n{transformation}", "pos": (2, 9.5)},
                'E': {"label": f"Re-check Assumptions\nShapiro-Wilk: {is_normal}\nBrown-Forsythe: {has_equal_variance}" if was_transformed else "Re-check Assumptions", "pos": (2, 8)},

                # Test recommendation
                'F': {"label": f"{test_recommendation_label}", "pos": (0, 6.5)},

                # Welch tests - direct branches from test recommendation
                'WELCH_T_TEST': {"label": "Welch's t-test\n(2 groups)", "pos": (-1.5, 4.5)},
                'WELCH_ANOVA': {"label": "Welch-ANOVA\n(>2 groups)", "pos": (1.5, 4.5)},
                'WELCH_DUNNETT_T3': {"label": "Dunnett T3\nPost-hoc", "pos": (1.5, 3)},

                # Parametric branch
                'G1': {"label": "Parametric Test", "pos": (-10, 5)},
                'H1': {"label": "Group Structure", "pos": (-10, 4)},
                'I1_2': {"label": "Two Groups", "pos": (-13, 3)},         # MOVED CLOSER TO CENTER
                'I1_M': {"label": "Multiple Groups", "pos": (-3, 3)},      # MOVED CLOSER TO CENTER

                # Parametric - Two Groups (MOVED CLOSER TO TWO GROUPS)
                'J1_INDEP': {"label": "Independent\nSamples", "pos": (-14, 2)},     # CLOSER
                'J1_DEP': {"label": "Dependent\nSamples", "pos": (-12, 2)},         # CLOSER
                'K1_2_IND': {"label": "Independent t-test", "pos": (-14, 1)},
                'K1_2_DEP': {"label": "Paired t-test", "pos": (-12, 1)},

                # THREE ANOVA DESIGNS - MOVED INDEPENDENT GROUPS FURTHER LEFT
                'INDEPENDENT_GROUPS': {"label": "Independent\nGroups", "pos": (-8, 2)},     # MOVED FURTHER LEFT
                'REPEATED_MEASURES': {"label": "Repeated\nMeasures", "pos": (-2, 2)},       # SAME 
                'MIXED_DESIGN': {"label": "Mixed\nDesign", "pos": (4, 2)},                  # MOVED CLOSER

                # INDEPENDENT GROUPS PATH - MOVED FURTHER LEFT AND MORE SPACING
                'IND_ONE_WAY': {"label": "One-way ANOVA", "pos": (-9, 1)},                  # MOVED LEFT
                'IND_TWO_WAY': {"label": "Two-way ANOVA", "pos": (-7, 1)},                  # MOVED LEFT
                'IND_POSTHOC': {"label": "Independent\nPost-hoc Tests", "pos": (-8, 0)},    # MOVED LEFT
                'IND_TUKEY': {"label": "Tukey HSD", "pos": (-9.5, -1)},                     # MORE SPACING: 1.5 apart
                'IND_DUNNETT': {"label": "Dunnett Test", "pos": (-8, -1)},                  # MORE SPACING: 1.5 apart
                'IND_HOLM_SIDAK': {"label": "Pairwise t-tests\n(Holm-Bonferroni)", "pos": (-6.5, -1)}, # MORE SPACING: 1.5 apart

                # REPEATED MEASURES PATH - SAME INTERNAL SPACING
                'RM_MAUCHLY': {"label": k1_m_sph_label, "pos": (-2, 1)},
                'RM_SPHERICITY_OK': {"label": "Sphericity\nAssumption Met", "pos": (-3.5, 0)},    # KEEP CLOSE
                'RM_SPHERICITY_VIOLATED': {"label": "Sphericity\nViolated", "pos": (-0.5, 0)},   # KEEP CLOSE
                'RM_CHOOSE_CORRECTION': {"label": "Choose\nCorrection", "pos": (-0.5, -1)},
                'RM_GG_CORRECTION': {"label": "Greenhouse-Geisser\nCorrection", "pos": (-1.5, -2)},  # KEEP CLOSE
                'RM_HF_CORRECTION': {"label": "Huynh-Feldt\nCorrection", "pos": (0.5, -2)},         # KEEP CLOSE
                'RM_ANOVA_STANDARD': {"label": "RM ANOVA", "pos": (-3.5, -1)},
                'RM_ANOVA_CORRECTED': {"label": "RM ANOVA\n(Corrected)", "pos": (-0.5, -3)},
                'RM_POSTHOC': {"label": "RM Post-hoc Tests", "pos": (-2, -4)},
                'RM_TUKEY': {"label": "Tukey HSD\n(RM)", "pos": (-3.5, -5)},
                'RM_PAIRED_TESTS': {"label": "Pairwise Paired t-tests\n(Holm-Bonferroni)", "pos": (-0.5, -5)},

                # MIXED DESIGN PATH - MOVED LEFT TO BE CLOSER, SAME INTERNAL SPACING
                'MIXED_MAUCHLY': {"label": k1_m_sph_label, "pos": (4, 1)},
                'MIXED_SPHERICITY_OK': {"label": "Sphericity\nAssumption Met", "pos": (2.5, 0)},     # KEEP CLOSE
                'MIXED_SPHERICITY_VIOLATED': {"label": "Sphericity\nViolated", "pos": (5.5, 0)},    # KEEP CLOSE
                'MIXED_CHOOSE_CORRECTION': {"label": "Choose\nCorrection", "pos": (5.5, -1)},
                'MIXED_GG_CORRECTION': {"label": "Greenhouse-Geisser\nCorrection", "pos": (4.5, -2)}, # KEEP CLOSE
                'MIXED_HF_CORRECTION': {"label": "Huynh-Feldt\nCorrection", "pos": (6.5, -2)},      # KEEP CLOSE
                'MIXED_ANOVA_STANDARD': {"label": "Mixed ANOVA", "pos": (2.5, -1)},
                'MIXED_ANOVA_CORRECTED': {"label": "Mixed ANOVA\n(Within Corrected)", "pos": (5.5, -3)},
                'MIXED_POSTHOC': {"label": "Mixed Post-hoc Tests", "pos": (4, -4)},
                'MIXED_TUKEY': {"label": "Mixed Tukey\n(Between/Within)", "pos": (2, -5)},      # MORE SPACING: 1.5 apart
                'MIXED_BETWEEN': {"label": "Between-Subjects\nComparisons", "pos": (4, -5)},       # MORE SPACING: 1.5 apart  
                'MIXED_WITHIN': {"label": "Within-Subjects\nComparisons", "pos": (6, -5)},       # MORE SPACING: 1.5 apart

                # Non-parametric branch - MOVED CLOSER TO PARAMETRIC
                'G2': {"label": "Non-parametric Test", "pos": (10, 5)},                      # MOVED CLOSER
                'H2': {"label": "Group Structure", "pos": (10, 4)},
                'I2_2': {"label": "Two Groups", "pos": (8, 3)},                             # MOVED CLOSER
                'I2_M': {"label": "Multiple Groups", "pos": (14, 3)},

                # Non-parametric - Two groups (MOVED CLOSER TO TWO GROUPS)
                'J2_INDEP': {"label": "Independent\nSamples", "pos": (7, 2)},               # MOVED CLOSER
                'J2_DEP': {"label": "Dependent\nSamples", "pos": (9, 2)},                   # MOVED CLOSER
                'K2_2_IND': {"label": "Mann-Whitney U", "pos": (7, 1)},
                'K2_2_DEP': {"label": "Wilcoxon\nSigned-Rank", "pos": (9, 1)},

                # Non-parametric - Multiple groups (advanced layout aligned to parametric pattern)
                'NP_INDEPENDENT_GROUPS': {"label": "Independent\nGroups", "pos": (12.5, 2)},
                'NP_REPEATED_MEASURES': {"label": "Repeated\nMeasures", "pos": (16, 2)},
                'NP_MIXED_DESIGN': {"label": "Mixed\nDesign", "pos": (20, 2)},

                # Non-parametric independent path (classic + robust two-way)
                'K2_M_IND': {"label": "Kruskal-Wallis", "pos": (11.5, 1)},
                'NP_POSTHOC': {"label": "Non-parametric\nPost-hoc Tests", "pos": (11.5, 0)},
                'NP_DUNN': {"label": "Dunn Test", "pos": (10.5, -1)},
                'NP_MANN_WHITNEY': {"label": "Pairwise\nMann-Whitney U", "pos": (12.5, -1)},
                'NP_TWO_WAY_ROBUST': {"label": "Freedman-Lane\nPermutation", "pos": (13.5, 1)},
                'NP_TWO_WAY_POSTHOC': {"label": "Two-way\nPost-hoc", "pos": (13.5, 0)},
                'NP_TWO_WAY_PAIRWISE': {"label": "Marginal Effects\nPairwise", "pos": (13.5, -1)},

                # Non-parametric repeated-measures path
                'NP_RM_ROBUST': {"label": "Friedman Test", "pos": (16, 1)},
                'NP_RM_POSTHOC': {"label": "Friedman\nPost-hoc", "pos": (16, 0)},
                'NP_RM_PAIRWISE': {"label": "RM Pairwise\nComparisons", "pos": (16, -1)},

                # Non-parametric mixed path
                'NP_MIXED_ROBUST': {"label": "Brunner-Langer\nATS", "pos": (20, 1)},
                'NP_MIXED_POSTHOC': {"label": "Mixed Robust\nPost-hoc", "pos": (20, 0)},
                'NP_MIXED_BETWEEN': {"label": "Between-Subjects\nComparisons", "pos": (19, -1)},
                'NP_MIXED_WITHIN': {"label": "Within-Subjects\nComparisons", "pos": (21, -1)},
            }

            # Apply wide-canvas spacing to preserve structure while avoiding label collisions.
            nodes_info = DecisionTreeVisualizer._apply_wide_canvas_layout(nodes_info)

            for node_id, info in nodes_info.items():
                G.add_node(node_id, label=info["label"], pos=info["pos"])

            # Create position dictionary from nodes_info
            pos = {node_id: info["pos"] for node_id, info in nodes_info.items()}

            # Define edges with NEW LOGICAL STRUCTURE - NO CROSS-CONNECTIONS
            edges = {
                # Common path
                ('A', 'B'),
                ('B', 'C'),

                # Transformation decision branch
                ('C', 'D1'),  # No transformation needed
                ('C', 'D2'),  # Apply transformation
                ('D2', 'E'),  # Re-check after transformation
                ('E', 'F'),   # Go to test recommendation after re-check
                ('D1', 'F'),  # Skip re-check if no transformation

                # Welch tests - direct from test recommendation
                ('F', 'WELCH_T_TEST'),      # Two groups with unequal variances
                ('F', 'WELCH_ANOVA'),       # Multiple groups with unequal variances
                ('WELCH_ANOVA', 'WELCH_DUNNETT_T3'),

                # Test type decision
                ('F', 'G1'),  # Parametric
                ('F', 'G2'),  # Non-parametric

                # Parametric branch - Group structure
                ('G1', 'H1'),
                ('H1', 'I1_2'),  # Two groups
                ('H1', 'I1_M'),  # Multiple groups

                # Parametric - Two Groups
                ('I1_2', 'J1_INDEP'),
                ('I1_2', 'J1_DEP'),
                ('J1_INDEP', 'K1_2_IND'),
                ('J1_DEP', 'K1_2_DEP'),

                # Multiple groups - THREE SEPARATE PATHS
                ('I1_M', 'INDEPENDENT_GROUPS'),  # No sphericity
                ('I1_M', 'REPEATED_MEASURES'),   # Sphericity check needed
                ('I1_M', 'MIXED_DESIGN'),        # Sphericity check needed

                # INDEPENDENT GROUPS PATH (NO SPHERICITY)
                ('INDEPENDENT_GROUPS', 'IND_ONE_WAY'),
                ('INDEPENDENT_GROUPS', 'IND_TWO_WAY'),
                ('IND_ONE_WAY', 'IND_POSTHOC'),
                ('IND_TWO_WAY', 'IND_POSTHOC'),
                ('IND_POSTHOC', 'IND_TUKEY'),
                ('IND_POSTHOC', 'IND_DUNNETT'),
                ('IND_POSTHOC', 'IND_HOLM_SIDAK'),

                # REPEATED MEASURES SPHERICITY PATH
                ('REPEATED_MEASURES', 'RM_MAUCHLY'),
                ('RM_MAUCHLY', 'RM_SPHERICITY_OK'),
                ('RM_MAUCHLY', 'RM_SPHERICITY_VIOLATED'),
                ('RM_SPHERICITY_OK', 'RM_ANOVA_STANDARD'),
                ('RM_SPHERICITY_VIOLATED', 'RM_CHOOSE_CORRECTION'),
                ('RM_CHOOSE_CORRECTION', 'RM_GG_CORRECTION'),
                ('RM_CHOOSE_CORRECTION', 'RM_HF_CORRECTION'),
                ('RM_GG_CORRECTION', 'RM_ANOVA_CORRECTED'),
                ('RM_HF_CORRECTION', 'RM_ANOVA_CORRECTED'),
                ('RM_ANOVA_STANDARD', 'RM_POSTHOC'),
                ('RM_ANOVA_CORRECTED', 'RM_POSTHOC'),
                ('RM_POSTHOC', 'RM_TUKEY'),
                ('RM_POSTHOC', 'RM_PAIRED_TESTS'),

                # MIXED DESIGN SPHERICITY PATH
                ('MIXED_DESIGN', 'MIXED_MAUCHLY'),
                ('MIXED_MAUCHLY', 'MIXED_SPHERICITY_OK'),
                ('MIXED_MAUCHLY', 'MIXED_SPHERICITY_VIOLATED'),
                ('MIXED_SPHERICITY_OK', 'MIXED_ANOVA_STANDARD'),
                ('MIXED_SPHERICITY_VIOLATED', 'MIXED_CHOOSE_CORRECTION'),
                ('MIXED_CHOOSE_CORRECTION', 'MIXED_GG_CORRECTION'),
                ('MIXED_CHOOSE_CORRECTION', 'MIXED_HF_CORRECTION'),
                ('MIXED_GG_CORRECTION', 'MIXED_ANOVA_CORRECTED'),
                ('MIXED_HF_CORRECTION', 'MIXED_ANOVA_CORRECTED'),
                ('MIXED_ANOVA_STANDARD', 'MIXED_POSTHOC'),
                ('MIXED_ANOVA_CORRECTED', 'MIXED_POSTHOC'),
                ('MIXED_POSTHOC', 'MIXED_TUKEY'),
                ('MIXED_POSTHOC', 'MIXED_BETWEEN'),
                ('MIXED_POSTHOC', 'MIXED_WITHIN'),

                # Non-parametric branch - Group structure
                ('G2', 'H2'),
                ('H2', 'I2_2'),  # Two groups
                ('H2', 'I2_M'),  # Multiple groups

                # Non-parametric - Two groups
                ('I2_2', 'J2_INDEP'),
                ('I2_2', 'J2_DEP'),
                ('J2_INDEP', 'K2_2_IND'),  # Mann-Whitney U
                ('J2_DEP', 'K2_2_DEP'),    # Wilcoxon

                # Non-parametric - Multiple groups (advanced cluster, geometry aligned to parametric branch)
                ('I2_M', 'NP_INDEPENDENT_GROUPS'),
                ('I2_M', 'NP_REPEATED_MEASURES'),
                ('I2_M', 'NP_MIXED_DESIGN'),

                # Independent branch: classic Kruskal + robust two-way fallback
                ('NP_INDEPENDENT_GROUPS', 'K2_M_IND'),
                ('K2_M_IND', 'NP_POSTHOC'),
                ('NP_POSTHOC', 'NP_DUNN'),
                ('NP_POSTHOC', 'NP_MANN_WHITNEY'),
                ('NP_INDEPENDENT_GROUPS', 'NP_TWO_WAY_ROBUST'),
                ('NP_TWO_WAY_ROBUST', 'NP_TWO_WAY_POSTHOC'),
                ('NP_TWO_WAY_POSTHOC', 'NP_TWO_WAY_PAIRWISE'),

                # Repeated-measures robust branch
                ('NP_REPEATED_MEASURES', 'NP_RM_ROBUST'),
                ('NP_RM_ROBUST', 'NP_RM_POSTHOC'),
                ('NP_RM_POSTHOC', 'NP_RM_PAIRWISE'),

                # Mixed robust branch
                ('NP_MIXED_DESIGN', 'NP_MIXED_ROBUST'),
                ('NP_MIXED_ROBUST', 'NP_MIXED_POSTHOC'),
                ('NP_MIXED_POSTHOC', 'NP_MIXED_BETWEEN'),
                ('NP_MIXED_POSTHOC', 'NP_MIXED_WITHIN'),
            }

            # Add edges to graph
            for start, end in edges:
                G.add_edge(start, end)

            # Determine highlighted path based on actual test performed
            highlighted = set()

            # Common path handling transformation loop
            highlighted.add(('A', 'B'))
            highlighted.add(('B', 'C'))

            # Transformation branch
            if transformation and transformation != "None":
                highlighted.add(('C', 'D2'))
                highlighted.add(('D2', 'E'))
                highlighted.add(('E', 'F'))
            else:
                highlighted.add(('C', 'D1'))
                highlighted.add(('D1', 'F'))

            # Debug information
            print(f"DEBUG TREE: test_type='{test_type}', test_name='{test_name}', n_groups={n_groups}")
            print(f"DEBUG TREE: is_normal={is_normal}, has_equal_variance={has_equal_variance}")
            print(f"DEBUG TREE: welch_t_condition={welch_t_condition}, welch_anova_condition={welch_anova_condition}")
            
            # Fix: Check both test_type and recommendation from results
            actual_test_type = test_type or results.get("recommendation", "")
            print(f"DEBUG TREE: actual_test_type='{actual_test_type}'")
            
            # Better test type branching logic
            print(f"DEBUG TREE: Determining test path...")
            print(f"DEBUG TREE: test_type='{test_type}', actual_test_type='{actual_test_type}'")
            print(f"DEBUG TREE: test_name='{test_name}'")
            print(f"DEBUG TREE: auto_switched={auto_switched}")

            recommendation_text = str(results.get("recommendation", "")).lower()
            model_class_text = str(results.get("model_class", "")).lower()
            model_type_text = str(results.get("model_type", "")).lower()
            analysis_log_text = str(results.get("analysis_log", "")).lower()
            test_name_text = str(test_name).lower()

            is_modern_or_robust_fallback = (
                "fallback" in recommendation_text or
                "modern model" in analysis_log_text or
                "robust" in recommendation_text or
                "[gee fallback]" in test_name_text or
                "[glm fallback]" in test_name_text
            )

            is_nonparam_two_way_advanced = (
                ("two-way" in test_name_text or "two way" in test_name_text) and
                (
                    is_modern_or_robust_fallback or
                    "freedman" in model_class_text or
                    "permutation" in model_class_text
                )
            )

            is_nonparam_rm_advanced = (
                ("repeated" in test_name_text or "rm anova" in test_name_text) and
                (
                    is_modern_or_robust_fallback or
                    "friedman" in model_class_text
                )
            )

            is_nonparam_mixed_advanced = (
                "mixed" in test_name_text and
                (
                    is_modern_or_robust_fallback or
                    "brunner" in model_class_text or
                    "ats" in model_class_text
                )
            )
            
            # Check if this is a non-parametric test
            is_nonparametric_test = (
                actual_test_type.lower() in ["non-parametric", "non_parametric"] or
                test_name.lower().startswith("non-parametric") or
                test_name.lower().startswith("nonparametric") or
                "rank + permutation" in test_name.lower() or
                auto_switched or
                results.get("recommendation") == "non_parametric" or  # NEW: Check recommendation
                results.get("parametric_assumptions_violated", False)  # NEW: Check if assumptions failed
            )
            
            print(f"DEBUG TREE: is_nonparametric_test={is_nonparametric_test}")
            
            # Welch test path (both t-test and ANOVA)
            if (welch_t_condition or welch_anova_condition) and not auto_switched and not is_nonparametric_test:
                
                if welch_t_condition:
                    # Welch's t-test path - direct from test recommendation
                    highlighted.add(('F', 'WELCH_T_TEST'))
                    print(f"DEBUG TREE: Highlighting Welch's t-test path")
                elif welch_anova_condition:
                    # Welch ANOVA path - direct from test recommendation
                    highlighted.add(('F', 'WELCH_ANOVA'))
                    # Only highlight post-hoc if significant
                    alpha = results.get("alpha", 0.05)
                    if p_value is not None and p_value < alpha:
                        if posthoc_test and "dunnett" in posthoc_test.lower() and "t3" in posthoc_test.lower():
                            highlighted.add(('WELCH_ANOVA', 'WELCH_DUNNETT_T3'))
                    print(f"DEBUG TREE: Highlighting Welch ANOVA path")
                    
            elif is_nonparametric_test:
                print(f"DEBUG TREE: Taking non-parametric path")
                if not auto_switched:
                    highlighted.add(('F', 'G2'))  # Non-parametric path
                highlighted.add(('G2', 'H2'))

                # Group structure for non-parametric
                if n_groups == 2:
                    highlighted.add(('H2', 'I2_2'))
                    if dependence_type == "dependent" or "paired" in test_name.lower() or "wilcoxon" in test_name.lower():
                        highlighted.add(('I2_2', 'J2_DEP'))
                        highlighted.add(('J2_DEP', 'K2_2_DEP'))  # Wilcoxon
                    else:
                        highlighted.add(('I2_2', 'J2_INDEP'))
                        highlighted.add(('J2_INDEP', 'K2_2_IND'))  # Mann-Whitney U
                else:
                    highlighted.add(('H2', 'I2_M'))

                    alpha = results.get("alpha", 0.05)

                    if is_nonparam_two_way_advanced:
                        highlighted.add(('I2_M', 'NP_INDEPENDENT_GROUPS'))
                        highlighted.add(('NP_INDEPENDENT_GROUPS', 'NP_TWO_WAY_ROBUST'))
                        if p_value is not None and p_value < alpha:
                            highlighted.add(('NP_TWO_WAY_ROBUST', 'NP_TWO_WAY_POSTHOC'))
                            highlighted.add(('NP_TWO_WAY_POSTHOC', 'NP_TWO_WAY_PAIRWISE'))

                    elif is_nonparam_rm_advanced:
                        highlighted.add(('I2_M', 'NP_REPEATED_MEASURES'))
                        highlighted.add(('NP_REPEATED_MEASURES', 'NP_RM_ROBUST'))
                        if p_value is not None and p_value < alpha:
                            highlighted.add(('NP_RM_ROBUST', 'NP_RM_POSTHOC'))
                            highlighted.add(('NP_RM_POSTHOC', 'NP_RM_PAIRWISE'))

                    elif is_nonparam_mixed_advanced:
                        highlighted.add(('I2_M', 'NP_MIXED_DESIGN'))
                        highlighted.add(('NP_MIXED_DESIGN', 'NP_MIXED_ROBUST'))
                        if p_value is not None and p_value < alpha:
                            highlighted.add(('NP_MIXED_ROBUST', 'NP_MIXED_POSTHOC'))
                            posthoc_text = str(results.get("posthoc_test", "")).lower()
                            if "between" in posthoc_text:
                                highlighted.add(('NP_MIXED_POSTHOC', 'NP_MIXED_BETWEEN'))
                            elif "within" in posthoc_text:
                                highlighted.add(('NP_MIXED_POSTHOC', 'NP_MIXED_WITHIN'))
                            else:
                                highlighted.add(('NP_MIXED_POSTHOC', 'NP_MIXED_BETWEEN'))
                                highlighted.add(('NP_MIXED_POSTHOC', 'NP_MIXED_WITHIN'))

                    else:
                        # Classic non-parametric multiple-groups path (Kruskal-Wallis)
                        highlighted.add(('I2_M', 'NP_INDEPENDENT_GROUPS'))
                        highlighted.add(('NP_INDEPENDENT_GROUPS', 'K2_M_IND'))

                        if p_value is not None and p_value < alpha:
                            highlighted.add(('K2_M_IND', 'NP_POSTHOC'))

                            posthoc_text = str(results.get("posthoc_test", "")).lower()
                            if "dunn" in posthoc_text:
                                highlighted.add(('NP_POSTHOC', 'NP_DUNN'))
                            else:
                                highlighted.add(('NP_POSTHOC', 'NP_MANN_WHITNEY'))
                        
            elif (actual_test_type.lower() == "parametric" or 
                  (test_name.lower().find("anova") != -1 and test_name.lower().find("non") == -1)) and \
                 not welch_t_condition and not welch_anova_condition:
                print(f"DEBUG TREE: Taking parametric path")
                if not auto_switched:
                    highlighted.add(('F', 'G1'))  # Parametric path
                highlighted.add(('G1', 'H1'))

                # Group structure
                if n_groups == 2:
                    highlighted.add(('H1', 'I1_2'))
                    if dependence_type == "dependent":
                        highlighted.add(('I1_2', 'J1_DEP'))
                        highlighted.add(('J1_DEP', 'K1_2_DEP'))
                    else:
                        highlighted.add(('I1_2', 'J1_INDEP'))
                        highlighted.add(('J1_INDEP', 'K1_2_IND'))
                        
                    # IMPORTANT: For 2-group tests, NEVER highlight post-hoc paths
                    # because post-hoc tests are only needed for 3+ groups
                    print(f"DEBUG TREE: 2-group test detected, skipping ALL post-hoc path highlighting")
                else:
                    highlighted.add(('H1', 'I1_M'))
                    
                    # Better logic for advanced ANOVA detection with explicit prioritization
                    if "repeated" in test_name.lower() or ("rm" in test_name.lower() and "anova" in test_name.lower()):
                        # Repeated Measures ANOVA path
                        print(f"DEBUG TREE: Detected RM ANOVA: {test_name}")
                        highlighted.add(('I1_M', 'REPEATED_MEASURES'))
                        highlighted.add(('REPEATED_MEASURES', 'RM_MAUCHLY'))
                        
                        # Check if sphericity correction was applied
                        sphericity_correction = results.get("correction_used", "None")
                        within_correction = results.get("within_correction_used", "None")
                        
                        if ("greenhouse" in str(sphericity_correction).lower() or 
                            "gg" in str(sphericity_correction).lower() or
                            "greenhouse" in str(within_correction).lower()):
                            # Greenhouse-Geisser correction path
                            highlighted.add(('RM_MAUCHLY', 'RM_SPHERICITY_VIOLATED'))
                            highlighted.add(('RM_SPHERICITY_VIOLATED', 'RM_CHOOSE_CORRECTION'))
                            highlighted.add(('RM_CHOOSE_CORRECTION', 'RM_GG_CORRECTION'))
                            highlighted.add(('RM_GG_CORRECTION', 'RM_ANOVA_CORRECTED'))
                        elif ("huynh" in str(sphericity_correction).lower() or 
                              "hf" in str(sphericity_correction).lower() or
                              "huynh" in str(within_correction).lower()):
                            # Huynh-Feldt correction path
                            highlighted.add(('RM_MAUCHLY', 'RM_SPHERICITY_VIOLATED'))
                            highlighted.add(('RM_SPHERICITY_VIOLATED', 'RM_CHOOSE_CORRECTION'))
                            highlighted.add(('RM_CHOOSE_CORRECTION', 'RM_HF_CORRECTION'))
                            highlighted.add(('RM_HF_CORRECTION', 'RM_ANOVA_CORRECTED'))
                        else:
                            # No correction needed or sphericity met
                            highlighted.add(('RM_MAUCHLY', 'RM_SPHERICITY_OK'))
                            highlighted.add(('RM_SPHERICITY_OK', 'RM_ANOVA_STANDARD'))
                        
                        # Only add post-hoc if ANOVA is significant:
                        alpha = results.get("alpha", 0.05)
                        if p_value is not None and p_value < alpha:
                            if ("greenhouse" in str(sphericity_correction).lower() or 
                                "huynh" in str(sphericity_correction).lower() or
                                "greenhouse" in str(within_correction).lower() or
                                "huynh" in str(within_correction).lower()):
                                highlighted.add(('RM_ANOVA_CORRECTED', 'RM_POSTHOC'))
                            else:
                                highlighted.add(('RM_ANOVA_STANDARD', 'RM_POSTHOC'))
                            
                            # Determine specific RM post-hoc test
                            posthoc_test = results.get("posthoc_test", "")
                            if "tukey" in posthoc_test.lower():
                                highlighted.add(('RM_POSTHOC', 'RM_TUKEY'))
                            else:
                                highlighted.add(('RM_POSTHOC', 'RM_PAIRED_TESTS'))
                            
                    elif "mixed" in test_name.lower():
                        # Mixed ANOVA path
                        print(f"DEBUG TREE: Detected Mixed ANOVA: {test_name}")
                        highlighted.add(('I1_M', 'MIXED_DESIGN'))
                        highlighted.add(('MIXED_DESIGN', 'MIXED_MAUCHLY'))
                        
                        # Check if within-factor sphericity correction was applied
                        within_correction = results.get("within_correction_used", "None")
                        sphericity_correction = results.get("correction_used", "None")
                        
                        if ("greenhouse" in str(within_correction).lower() or 
                            "gg" in str(within_correction).lower() or
                            "greenhouse" in str(sphericity_correction).lower()):
                            # Greenhouse-Geisser correction for within-factor
                            highlighted.add(('MIXED_MAUCHLY', 'MIXED_SPHERICITY_VIOLATED'))
                            highlighted.add(('MIXED_SPHERICITY_VIOLATED', 'MIXED_CHOOSE_CORRECTION'))
                            highlighted.add(('MIXED_CHOOSE_CORRECTION', 'MIXED_GG_CORRECTION'))
                            highlighted.add(('MIXED_GG_CORRECTION', 'MIXED_ANOVA_CORRECTED'))
                        elif ("huynh" in str(within_correction).lower() or 
                              "hf" in str(within_correction).lower() or
                              "huynh" in str(sphericity_correction).lower()):
                            # Huynh-Feldt correction for within-factor
                            highlighted.add(('MIXED_MAUCHLY', 'MIXED_SPHERICITY_VIOLATED'))
                            highlighted.add(('MIXED_SPHERICITY_VIOLATED', 'MIXED_CHOOSE_CORRECTION'))
                            highlighted.add(('MIXED_CHOOSE_CORRECTION', 'MIXED_HF_CORRECTION'))
                            highlighted.add(('MIXED_HF_CORRECTION', 'MIXED_ANOVA_CORRECTED'))
                        else:
                            # No within-factor correction needed
                            highlighted.add(('MIXED_MAUCHLY', 'MIXED_SPHERICITY_OK'))
                            highlighted.add(('MIXED_SPHERICITY_OK', 'MIXED_ANOVA_STANDARD'))
                        
                        # Only add post-hoc if Mixed ANOVA is significant:
                        alpha = results.get("alpha", 0.05)
                        if p_value is not None and p_value < alpha:
                            if ("greenhouse" in str(within_correction).lower() or 
                                "huynh" in str(within_correction).lower()):
                                highlighted.add(('MIXED_ANOVA_CORRECTED', 'MIXED_POSTHOC'))
                            else:
                                highlighted.add(('MIXED_ANOVA_STANDARD', 'MIXED_POSTHOC'))
                                
                            # Determine specific Mixed post-hoc test
                            posthoc_test = results.get("posthoc_test") or ""
                            if "tukey" in posthoc_test.lower():
                                highlighted.add(('MIXED_POSTHOC', 'MIXED_TUKEY'))
                            elif "between" in posthoc_test.lower():
                                highlighted.add(('MIXED_POSTHOC', 'MIXED_BETWEEN'))
                            else:
                                highlighted.add(('MIXED_POSTHOC', 'MIXED_WITHIN'))
                            
                    elif "two-way" in test_name.lower() or "two way" in test_name.lower():
                        # Two-Way ANOVA path (explicit detection) - Independent Groups
                        print(f"DEBUG TREE: Detected Two-Way ANOVA: {test_name}")
                        highlighted.add(('I1_M', 'INDEPENDENT_GROUPS'))
                        highlighted.add(('INDEPENDENT_GROUPS', 'IND_TWO_WAY'))
                        alpha = results.get("alpha", 0.05)
                        if p_value is not None and p_value < alpha:
                            highlighted.add(('IND_TWO_WAY', 'IND_POSTHOC'))
                            # Determine specific post-hoc test
                            posthoc_test = results.get("posthoc_test", "")
                            if "tukey" in posthoc_test.lower():
                                highlighted.add(('IND_POSTHOC', 'IND_TUKEY'))
                            elif "dunnett" in posthoc_test.lower():
                                highlighted.add(('IND_POSTHOC', 'IND_DUNNETT'))
                            else:
                                highlighted.add(('IND_POSTHOC', 'IND_HOLM_SIDAK'))
                            
                    else:
                        # One-Way ANOVA path (default for unspecified multiple group parametric tests)
                        print(f"DEBUG TREE: Detected One-Way ANOVA (default): {test_name}")
                        highlighted.add(('I1_M', 'INDEPENDENT_GROUPS'))
                        highlighted.add(('INDEPENDENT_GROUPS', 'IND_ONE_WAY'))
                        alpha = results.get("alpha", 0.05)
                        if p_value is not None and p_value < alpha:
                            highlighted.add(('IND_ONE_WAY', 'IND_POSTHOC'))
                            # Determine specific post-hoc test
                            posthoc_test = results.get("posthoc_test", "")
                            if "tukey" in posthoc_test.lower():
                                highlighted.add(('IND_POSTHOC', 'IND_TUKEY'))
                            elif "dunnett" in posthoc_test.lower():
                                highlighted.add(('IND_POSTHOC', 'IND_DUNNETT'))
                            else:
                                highlighted.add(('IND_POSTHOC', 'IND_HOLM_SIDAK'))
            # Generate edge lists for drawing
            highlighted_edges = [(u, v) for u, v in G.edges() if (u, v) in highlighted]
            regular_edges = [(u, v) for u, v in G.edges() if (u, v) not in highlighted]
            transformation_edges = [e for e in highlighted_edges if e[0] == 'D2' and e[1] == 'E']
            check_edges = [e for e in G.edges() if e[0] == 'B' or e[1] == 'B']

            # Add debug visualization info
            print(f"DEBUG VISUALIZATION: Test name: {test_name}")
            print(f"DEBUG VISUALIZATION: Path type: {'parametric' if actual_test_type.lower() == 'parametric' else 'non-parametric'}")
            print(f"DEBUG VISUALIZATION: Groups: {n_groups}")
            print(f"DEBUG VISUALIZATION: Dependence: {dependence_type}")
            print(f"DEBUG VISUALIZATION: Number of highlighted edges: {len(highlighted)}")

            # Draw the graph
            pos = nx.get_node_attributes(G, 'pos')
            fig_width, fig_height = DecisionTreeVisualizer._calculate_figure_size(pos)
            plt.figure(figsize=(fig_width, fig_height))
            node_labels = nx.get_node_attributes(G, 'label')

            # Highlighted nodes for color/highlighting
            highlighted_nodes = set()
            for u, v in highlighted:
                highlighted_nodes.add(u)
                highlighted_nodes.add(v)

            # Apply dynamic test label formatting to active leaf nodes
            test_nodes = {
                'WELCH_T_TEST', 'WELCH_ANOVA', 'K1_2_IND', 'K1_2_DEP', 
                'IND_ONE_WAY', 'IND_TWO_WAY', 'RM_ANOVA_STANDARD', 'RM_ANOVA_CORRECTED', 
                'MIXED_ANOVA_STANDARD', 'MIXED_ANOVA_CORRECTED', 'K2_2_IND', 'K2_2_DEP', 
                'K2_M_IND', 'NP_RM_ROBUST', 'NP_MIXED_ROBUST'
            }
            for node_id in highlighted_nodes:
                if node_id in test_nodes and node_id in node_labels:
                    node_labels[node_id] = DecisionTreeVisualizer.format_dynamic_test_label(
                        node_id, node_labels[node_id], results
                    )

            # --- Shape logic for nodes ---
            always_square_labels = {
                "Start", "Check Assumptions", "Assumptions: Met", "Assumptions: Not Met",
                "Parametric Test", "Non-parametric Test", "Group Structure",
                "Two Groups", "Multiple Groups", "Independent Samples",
                "Dependent Samples", "Sphericity Check", "Repeated Measures",
                "Mixed Design", "Post-hoc Tests", "Independent Groups",
                "Mauchly's Test", "Sphericity", "Choose Correction",  # NEW
                "RM Post-hoc Tests", "Mixed Post-hoc Tests"  # NEW
            }

            def is_always_square(node_id):
                # Check if node exists in nodes_info first
                if node_id not in nodes_info:
                    return False
                label = nodes_info[node_id]["label"].replace('\n', ' ').replace(':', ': ').replace('  ', ' ')
                for square_label in always_square_labels:
                    if square_label in label:
                        return True
                return False

            # Draw nodes
            square_nodes = [n for n in G.nodes() if is_always_square(n)]
            round_nodes = [n for n in G.nodes() if n not in square_nodes]

            # Draw all nodes: highlighted ones in color, others in white
            nx.draw_networkx_nodes(G, pos, nodelist=[n for n in square_nodes if n in highlighted_nodes], node_size=3000,
                                    node_color='#ffcccc', edgecolors='black', linewidths=1.5, node_shape='s')
            nx.draw_networkx_nodes(G, pos, nodelist=[n for n in square_nodes if n not in highlighted_nodes], node_size=3000,
                                    node_color='white', edgecolors='black', linewidths=1.5, node_shape='s')
            nx.draw_networkx_nodes(G, pos, nodelist=[n for n in round_nodes if n in highlighted_nodes], node_size=3000,
                                    node_color='#ffcccc', edgecolors='black', linewidths=1.5, node_shape='o')
            nx.draw_networkx_nodes(G, pos, nodelist=[n for n in round_nodes if n not in highlighted_nodes], node_size=3000,
                                    node_color='white', edgecolors='black', linewidths=1.5, node_shape='o')

            # Draw all edges with different styles
            nx.draw_networkx_edges(G, pos, edgelist=highlighted_edges, width=4, edge_color='red')
            nx.draw_networkx_edges(G, pos, edgelist=transformation_edges, width=4, edge_color='blue', style='dashed')
            nx.draw_networkx_edges(G, pos, edgelist=check_edges, width=2, edge_color='gray', style='dotted')
            nx.draw_networkx_edges(G, pos, edgelist=regular_edges, width=1, edge_color='black', style='solid')

            # Draw node labels with background boxes
            nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12.5,
                    font_family='sans-serif', font_weight='bold',
                    bbox=dict(boxstyle='round,pad=0.28', facecolor='white',
                                alpha=0.7, edgecolor='lightgray'))

            # Add title using figure-level title
            fig = plt.gcf()
            fig.suptitle(f"Statistical Decision Path: {test_name}", fontsize=16, y=0.98)

            # Create a proper legend
            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch

            legend_elements = [
                Line2D([0], [0], color='red', lw=4, label='Taken path'),
                Line2D([0], [0], color='blue', lw=4, linestyle='dashed', label='Transformation loop'),
                Line2D([0], [0], color='gray', lw=2, linestyle='dotted', label='Assumption checks'),
                Patch(facecolor='#ffcccc', edgecolor='black', label='Steps performed'),
                Patch(facecolor='white', edgecolor='black', label='Alternative steps'),
                Line2D([0], [0], marker='s', color='none', markerfacecolor='#ffcccc', 
                    markeredgecolor='black', markersize=15, label='Decision nodes'),
                Line2D([0], [0], marker='o', color='none', markerfacecolor='#ffcccc',
                    markeredgecolor='black', markersize=15, label='Statistical tests'),
            ]

            # Create legend with larger font size
            plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.99),
                    fontsize=11, frameon=True, facecolor='white', edgecolor='black',
                    framealpha=0.9, shadow=True)

            # Remove axis
            plt.axis('off')
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            
            # Set figure size early and maintain it
            fig = plt.gcf()
            fig.set_size_inches(fig_width, fig_height, forward=True)
            
            print(f"DEBUG: About to save figure to {'temp file' if not output_path else output_path}")
            print(f"DEBUG: Using matplotlib backend: {plt.get_backend()}")
            print(f"DEBUG: Graph has {len(G.nodes())} nodes and {len(G.edges())} edges")
            print(f"DEBUG: Figure size: {plt.gcf().get_size_inches()}")
            print(f"DEBUG: Number of highlighted edges: {len(highlighted_edges)}")

            # Save the image if path provided
            if output_path:
                output_file = f"{output_path}.png"
                plt.savefig(output_file, format="png", dpi=200, transparent=False, 
                        facecolor='white', bbox_inches='tight')
                plt.close('all')
                
                print(f"DEBUG: Image saved as {output_file}")
                return output_file
            else:
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    plt.savefig(tmp.name, format="png", dpi=200, transparent=False,
                            facecolor='white', bbox_inches='tight')
                    path = tmp.name
                
                if os.path.exists(path) and os.path.getsize(path) > 1000:
                    plt.close('all')
                print(f"DEBUG: Image saved to temp file {path}")
                return path

        except Exception as e:
            print(f"Error generating decision tree with NetworkX: {e}")
            import traceback
            traceback.print_exc()
            return None

    @staticmethod
    def generate_and_save_for_excel(results):
        """
        Generates a decision tree visualization and saves it as a temporary PNG file
        for embedding in Excel.
        
        Parameters:
        -----------
        results : dict
            Results dictionary containing test information
            
        Returns:
        --------
        str
            Path to the saved PNG file, or None if generation failed
        """
        try:
            import os
            import time
            import tempfile
            
            # Use system temp directory instead of Documents folder
            fd, temp_path = tempfile.mkstemp(suffix='.png', prefix='decision_tree_')
            os.close(fd)

            print(f"DEBUG: Generating decision tree visualization to: {temp_path}")

            # Generate visualization with the PNG path (remove extension for base path)
            output_path = DecisionTreeVisualizer.visualize(results, output_path=temp_path.replace(".png", ""))
            print(f"DEBUG: Decision tree visualization returned path: {output_path}")
            
            # Double check file exists
            if output_path and os.path.exists(output_path):
                print(f"DEBUG: Decision tree file verified at: {output_path}")
                return output_path
            else:
                print(f"DEBUG: ERROR - Decision tree image not found at expected path: {temp_path}")
                return None
                
        except Exception as e:
            print(f"DEBUG: Exception in generate_and_save_for_excel: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    @staticmethod
    def get_tree_json(results: dict) -> dict | None:
        """
        Returns the decision tree topology as a JSON-serializable dict.
        Nodes and edges carry an isActive flag marking the taken path.
        """
        try:
            _model_type = results.get("model_type", "")
            if _model_type in ["Correlation", "LinearRegression", "LogisticRegression", "ANCOVA", "LMM", "CorrelationMatrix"]:
                return DecisionTreeVisualizer._get_association_tree_json(results, _model_type)

            test_name = results.get("test_name", results.get("test", ""))
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

            welch_t_condition = (
                ("welch" in test_name.lower() and "t-test" in test_name.lower()) or
                ("welch" in test_name.lower() and n_groups == 2) or
                (is_normal and not has_equal_variance and n_groups == 2 and
                 ("welch" in test_name.lower() or "independent" in test_name.lower()))
            )
            welch_anova_condition = (
                ("welch" in test_name.lower() and "anova" in test_name.lower()) or
                ("welch" in test_name.lower() and n_groups > 2) or
                (is_normal and not has_equal_variance and n_groups > 2 and
                 ("welch" in test_name.lower() or "anova" in test_name.lower()))
            )

            # labels
            if welch_t_condition or welch_anova_condition:
                f_label = "Test Recommendation:\nNormal distributed\nbut unequal variances"
            elif actual_test_type.lower() == "parametric":
                f_label = "Test Recommendation:\nParametric Test"
            elif actual_test_type.lower() in ("non_parametric", "non-parametric"):
                f_label = "Test Recommendation:\nNon-parametric Test"
            else:
                f_label = "Test Recommendation"

            # nodes
            nodes_info = {
                'A':  {"label": "Start", "pos": (0, 14)},
                'B':  {"label": f"Check Assumptions\nShapiro-Wilk: {pre_is_normal}\nBrown-Forsythe: {pre_has_equal_variance}", "pos": (0, 12.5)},
                'C':  {"label": f"Assumptions: {'Met' if pre_is_normal and pre_has_equal_variance else 'Not Met'}", "pos": (0, 11)},
                'D1': {"label": "No Transformation\nNeeded", "pos": (-2, 9.5)},
                'D2': {"label": f"Apply Transformation\n{transformation}", "pos": (2, 9.5)},
                'E':  {"label": f"Re-check Assumptions\nShapiro-Wilk: {is_normal}\nBrown-Forsythe: {has_equal_variance}" if was_transformed else "Re-check Assumptions", "pos": (2, 8)},
                'F':  {"label": f_label, "pos": (0, 6.5)},
                'WELCH_T_TEST':   {"label": "Welch's t-test\n(2 groups)", "pos": (-1.5, 4.5)},
                'WELCH_ANOVA':    {"label": "Welch-ANOVA\n(>2 groups)", "pos": (1.5, 4.5)},
                'WELCH_DUNNETT_T3': {"label": "Dunnett T3\nPost-hoc", "pos": (1.5, 3)},
                'G1': {"label": "Parametric Test", "pos": (-10, 5)},
                'H1': {"label": "Group Structure", "pos": (-10, 4)},
                'I1_2': {"label": "Two Groups", "pos": (-13, 3)},
                'I1_M': {"label": "Multiple Groups", "pos": (-3, 3)},
                'J1_INDEP': {"label": "Independent\nSamples", "pos": (-14, 2)},
                'J1_DEP':   {"label": "Dependent\nSamples", "pos": (-12, 2)},
                'K1_2_IND': {"label": "Independent t-test", "pos": (-14, 1)},
                'K1_2_DEP': {"label": "Paired t-test", "pos": (-12, 1)},
                'INDEPENDENT_GROUPS': {"label": "Independent\nGroups", "pos": (-8, 2)},
                'REPEATED_MEASURES':  {"label": "Repeated\nMeasures", "pos": (-2, 2)},
                'MIXED_DESIGN':        {"label": "Mixed\nDesign", "pos": (4, 2)},
                'IND_ONE_WAY':   {"label": "One-way ANOVA", "pos": (-9, 1)},
                'IND_TWO_WAY':   {"label": "Two-way ANOVA", "pos": (-7, 1)},
                'IND_POSTHOC':   {"label": "Independent\nPost-hoc Tests", "pos": (-8, 0)},
                'IND_TUKEY':     {"label": "Tukey HSD", "pos": (-9.5, -1)},
                'IND_DUNNETT':   {"label": "Dunnett Test", "pos": (-8, -1)},
                'IND_HOLM_SIDAK':{"label": "Pairwise t-tests\n(Holm-Bonferroni)", "pos": (-6.5, -1)},
                'RM_MAUCHLY':            {"label": k1_m_sph_label, "pos": (-2, 1)},
                'RM_SPHERICITY_OK':      {"label": "Sphericity\nAssumption Met", "pos": (-3.5, 0)},
                'RM_SPHERICITY_VIOLATED':{"label": "Sphericity\nViolated", "pos": (-0.5, 0)},
                'RM_CHOOSE_CORRECTION':  {"label": "Choose\nCorrection", "pos": (-0.5, -1)},
                'RM_GG_CORRECTION':      {"label": "Greenhouse-Geisser\nCorrection", "pos": (-1.5, -2)},
                'RM_HF_CORRECTION':      {"label": "Huynh-Feldt\nCorrection", "pos": (0.5, -2)},
                'RM_ANOVA_STANDARD':     {"label": "RM ANOVA", "pos": (-3.5, -1)},
                'RM_ANOVA_CORRECTED':    {"label": "RM ANOVA\n(Corrected)", "pos": (-0.5, -3)},
                'RM_POSTHOC':            {"label": "RM Post-hoc Tests", "pos": (-2, -4)},
                'RM_TUKEY':              {"label": "Tukey HSD\n(RM)", "pos": (-3.5, -5)},
                'RM_PAIRED_TESTS':       {"label": "Pairwise Paired t-tests\n(Holm-Bonferroni)", "pos": (-0.5, -5)},
                'MIXED_MAUCHLY':             {"label": k1_m_sph_label, "pos": (4, 1)},
                'MIXED_SPHERICITY_OK':       {"label": "Sphericity\nAssumption Met", "pos": (2.5, 0)},
                'MIXED_SPHERICITY_VIOLATED': {"label": "Sphericity\nViolated", "pos": (5.5, 0)},
                'MIXED_CHOOSE_CORRECTION':   {"label": "Choose\nCorrection", "pos": (5.5, -1)},
                'MIXED_GG_CORRECTION':       {"label": "Greenhouse-Geisser\nCorrection", "pos": (4.5, -2)},
                'MIXED_HF_CORRECTION':       {"label": "Huynh-Feldt\nCorrection", "pos": (6.5, -2)},
                'MIXED_ANOVA_STANDARD':      {"label": "Mixed ANOVA", "pos": (2.5, -1)},
                'MIXED_ANOVA_CORRECTED':     {"label": "Mixed ANOVA\n(Within Corrected)", "pos": (5.5, -3)},
                'MIXED_POSTHOC':             {"label": "Mixed Post-hoc Tests", "pos": (4, -4)},
                'MIXED_TUKEY':   {"label": "Mixed Tukey\n(Between/Within)", "pos": (2, -5)},
                'MIXED_BETWEEN': {"label": "Between-Subjects\nComparisons", "pos": (4, -5)},
                'MIXED_WITHIN':  {"label": "Within-Subjects\nComparisons", "pos": (6, -5)},
                'G2': {"label": "Non-parametric Test", "pos": (10, 5)},
                'H2': {"label": "Group Structure", "pos": (10, 4)},
                'I2_2': {"label": "Two Groups", "pos": (8, 3)},
                'I2_M': {"label": "Multiple Groups", "pos": (14, 3)},
                'J2_INDEP': {"label": "Independent\nSamples", "pos": (7, 2)},
                'J2_DEP':   {"label": "Dependent\nSamples", "pos": (9, 2)},
                'K2_2_IND': {"label": "Mann-Whitney U", "pos": (7, 1)},
                'K2_2_DEP': {"label": "Wilcoxon\nSigned-Rank", "pos": (9, 1)},
                'NP_INDEPENDENT_GROUPS': {"label": "Independent\nGroups", "pos": (12.5, 2)},
                'NP_REPEATED_MEASURES':  {"label": "Repeated\nMeasures", "pos": (16, 2)},
                'NP_MIXED_DESIGN':       {"label": "Mixed\nDesign", "pos": (20, 2)},
                'K2_M_IND':           {"label": "Kruskal-Wallis", "pos": (11.5, 1)},
                'NP_POSTHOC':         {"label": "Non-parametric\nPost-hoc Tests", "pos": (11.5, 0)},
                'NP_DUNN':            {"label": "Dunn Test", "pos": (10.5, -1)},
                'NP_MANN_WHITNEY':    {"label": "Pairwise\nMann-Whitney U", "pos": (12.5, -1)},
                'NP_TWO_WAY_ROBUST':  {"label": "Freedman-Lane\nPermutation", "pos": (13.5, 1)},
                'NP_TWO_WAY_POSTHOC': {"label": "Two-way\nPost-hoc", "pos": (13.5, 0)},
                'NP_TWO_WAY_PAIRWISE':{"label": "Marginal Effects\nPairwise", "pos": (13.5, -1)},
                'NP_RM_ROBUST':   {"label": "Friedman Test", "pos": (16, 1)},
                'NP_RM_POSTHOC':  {"label": "Friedman\nPost-hoc", "pos": (16, 0)},
                'NP_RM_PAIRWISE': {"label": "RM Pairwise\nComparisons", "pos": (16, -1)},
                'NP_MIXED_ROBUST':   {"label": "Brunner-Langer\nATS", "pos": (20, 1)},
                'NP_MIXED_POSTHOC':  {"label": "Mixed Robust\nPost-hoc", "pos": (20, 0)},
                'NP_MIXED_BETWEEN':  {"label": "Between-Subjects\nComparisons", "pos": (19, -1)},
                'NP_MIXED_WITHIN':   {"label": "Within-Subjects\nComparisons", "pos": (21, -1)},
            }
            nodes_info = DecisionTreeVisualizer._apply_wide_canvas_layout(nodes_info)

            edges = {
                ('A','B'),('B','C'),('C','D1'),('C','D2'),('D2','E'),('E','F'),('D1','F'),
                ('F','WELCH_T_TEST'),('F','WELCH_ANOVA'),('WELCH_ANOVA','WELCH_DUNNETT_T3'),
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

            if (welch_t_condition or welch_anova_condition) and not auto_switched and not is_nonparametric_test:
                if welch_t_condition:
                    highlighted.add(('F','WELCH_T_TEST'))
                else:
                    highlighted.add(('F','WELCH_ANOVA'))
                    if p_value is not None and p_value < alpha:
                        if "dunnett" in posthoc_test.lower() and "t3" in posthoc_test.lower():
                            highlighted.add(('WELCH_ANOVA','WELCH_DUNNETT_T3'))

            elif is_nonparametric_test:
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

            elif (actual_test_type.lower() == "parametric" or ("anova" in test_name.lower() and "non" not in test_name.lower())) and not welch_t_condition and not welch_anova_condition:
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
                        elif any(kw in correction_used.lower() for kw in ("huynh","hf")) or "huynh" in within_correction.lower():
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
                        elif any(kw in within_correction.lower() for kw in ("huynh","hf")) or "huynh" in correction_used.lower():
                            highlighted.update([('MIXED_MAUCHLY','MIXED_SPHERICITY_VIOLATED'),('MIXED_SPHERICITY_VIOLATED','MIXED_CHOOSE_CORRECTION'),('MIXED_CHOOSE_CORRECTION','MIXED_HF_CORRECTION'),('MIXED_HF_CORRECTION','MIXED_ANOVA_CORRECTED')])
                        else:
                            highlighted.update([('MIXED_MAUCHLY','MIXED_SPHERICITY_OK'),('MIXED_SPHERICITY_OK','MIXED_ANOVA_STANDARD')])
                        if p_value is not None and p_value < alpha:
                            if any(kw in within_correction.lower() for kw in ("greenhouse","huynh")):
                                highlighted.add(('MIXED_ANOVA_CORRECTED','MIXED_POSTHOC'))
                            else:
                                highlighted.add(('MIXED_ANOVA_STANDARD','MIXED_POSTHOC'))
                            ph = posthoc_test.lower()
                            if "tukey" in ph:
                                highlighted.add(('MIXED_POSTHOC','MIXED_TUKEY'))
                            elif "between" in ph:
                                highlighted.add(('MIXED_POSTHOC','MIXED_BETWEEN'))
                            else:
                                highlighted.add(('MIXED_POSTHOC','MIXED_WITHIN'))
                    elif "two-way" in test_name_text or "two way" in test_name_text:
                        highlighted.update([('I1_M','INDEPENDENT_GROUPS'),('INDEPENDENT_GROUPS','IND_TWO_WAY')])
                        if p_value is not None and p_value < alpha:
                            highlighted.add(('IND_TWO_WAY','IND_POSTHOC'))
                            ph = posthoc_test.lower()
                            if "tukey" in ph: highlighted.add(('IND_POSTHOC','IND_TUKEY'))
                            elif "dunnett" in ph: highlighted.add(('IND_POSTHOC','IND_DUNNETT'))
                            else: highlighted.add(('IND_POSTHOC','IND_HOLM_SIDAK'))
                    else:
                        highlighted.update([('I1_M','INDEPENDENT_GROUPS'),('INDEPENDENT_GROUPS','IND_ONE_WAY')])
                        if p_value is not None and p_value < alpha:
                            highlighted.add(('IND_ONE_WAY','IND_POSTHOC'))
                            ph = posthoc_test.lower()
                            if "tukey" in ph: highlighted.add(('IND_POSTHOC','IND_TUKEY'))
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
            print(f"WARNING DecisionTreeVisualizer.get_tree_json: {exc}")
            return None

    @staticmethod
    def _build_tree_topology(results: dict, model_type: str, alpha: float = 0.05):
        p_value = results.get("p_value", None)
        method = results.get("method", "")
        transformation = results.get("transformation", "none")

        # Extract assumption check results
        _nc = results.get("normality_check")
        normality_check = _nc if isinstance(_nc, dict) else {}
        diagnostics = results.get("diagnostics") or {}
        slope_homogeneity = results.get("slope_homogeneity") or {}

        # Correlation: normality of both variables
        _nc_keys = list(normality_check.keys()) if normality_check else []
        _nc_v0 = normality_check.get(_nc_keys[0]) if len(_nc_keys) >= 1 else None
        _nc_v1 = normality_check.get(_nc_keys[1]) if len(_nc_keys) >= 2 else None
        norm_x_ok = _nc_v0.get("normal", None) if isinstance(_nc_v0, dict) else None
        norm_y_ok = _nc_v1.get("normal", None) if isinstance(_nc_v1, dict) else None
        both_normal = normality_check.get("both_normal", None)

        # Linear Regression diagnostics
        norm_resid_ok = diagnostics.get("normality", {}).get("assumption_holds", None)
        homosced_ok = diagnostics.get("homoscedasticity", {}).get("assumption_holds", None)
        linearity_ok = diagnostics.get("linearity", {}).get("assumption_holds", None)

        # ANCOVA: slope homogeneity
        slopes_ok = all(v.get("assumption_holds", True) for v in slope_homogeneity.values()) if slope_homogeneity else None
        if results.get("simple_slopes_analysis") is not None or "simple_slopes" in results:
            slopes_ok = False

        # Logistic Regression: Hosmer-Lemeshow
        hl = results.get("hosmer_lemeshow") or {}
        hl_p = hl.get("p_value", None)
        hl_ok = (hl_p is not None and hl_p > 0.05) if hl_p is not None else None

        sig = (p_value is not None and p_value < alpha)

        def _yn(val):
            if val is True:  return "Yes"
            if val is False: return "No"
            return "n/a"

        nodes_info = {}
        edges = set()
        highlighted = set()
        
        n_samples = results.get("n", 100)
        if model_type == "Correlation":
            n_samples = results.get("n", 50)
            
        from statistical_testing.validators import MIN_N_SMALL
        if n_samples < MIN_N_SMALL:
            calculated_tier = f"Micro-Sample Mode (N < {MIN_N_SMALL})"
        elif n_samples < 100:
            calculated_tier = "Clinical Mode (20 <= N < 100)"
        else:
            calculated_tier = "Asymptotic Mode (N >= 100)"
            
        if "calculated_tier" in results:
            calculated_tier = results["calculated_tier"]
        elif "tier" in results:
            calculated_tier = results["tier"]
            
        tree_meta = {
            "calculated_tier": calculated_tier,
            "total_samples": int(n_samples)
        }

        if model_type == "Correlation":
            used_pearson = method.lower() == "pearson" or (both_normal is True)
            r_val    = results.get("r", None)
            r_label  = f"r = {r_val:.3f}" if r_val is not None else ""
            p_label  = f"p = {p_value:.4f}" if p_value is not None else ""
            sig_label = "Significant" if sig else "Not Significant"

            nodes_info = {
                'START':           {"label": "Start\nCorrelation Analysis",                                "pos": (0, 10), "isSquare": True},
                'TIER_MICRO':      {"label": f"Micro-Sample Logic\n(N < {MIN_N_SMALL})",                               "pos": (-4, 8), "isSquare": True},
                'TIER_CLINICAL':   {"label": "Clinical Adaptive Logic\n(20 <= N < 100)",                    "pos": (0, 8), "isSquare": True},
                'TIER_ASYMPTOTIC': {"label": "Asymptotic Logic\n(N >= 100)",                                "pos": (4, 8), "isSquare": True},
                'SKEW_KURT_CHECK': {"label": "Skewness/Kurtosis Check\n(Pearson assumptions)",              "pos": (2, 6), "isSquare": True},
                'PEARSON':         {"label": "Pearson Correlation\n(parametric)",                          "pos": (-2, 4), "isSquare": False},
                'SPEARMAN':        {"label": "Spearman Correlation\n(non-parametric)",                     "pos": (2, 4), "isSquare": False},
                'RESULT':          {"label": f"Result\n{r_label}  {p_label}\n{sig_label}",                "pos": (0, 2), "isSquare": False},
                'CI':              {"label": "95% CI\n(Fisher z-transform)",                               "pos": (-2, 0.5), "isSquare": False},
                'EFFECT':          {"label": "Effect Size  |r|\n(small\u2265.1 med\u2265.3 large\u2265.5)", "pos": (2, 0.5), "isSquare": False},
            }
            edges = {
                ('START',      'TIER_MICRO'),
                ('START',      'TIER_CLINICAL'),
                ('START',      'TIER_ASYMPTOTIC'),
                ('TIER_MICRO', 'SPEARMAN'),
                ('TIER_CLINICAL', 'SKEW_KURT_CHECK'),
                ('TIER_ASYMPTOTIC', 'SKEW_KURT_CHECK'),
                ('SKEW_KURT_CHECK', 'PEARSON'),
                ('SKEW_KURT_CHECK', 'SPEARMAN'),
                ('PEARSON',    'RESULT'),
                ('SPEARMAN',   'RESULT'),
                ('RESULT',     'CI'),
                ('RESULT',     'EFFECT'),
            }
            
            highlighted = {('START', 'TIER_MICRO') if n_samples < MIN_N_SMALL else (('START', 'TIER_CLINICAL') if n_samples < 100 else ('START', 'TIER_ASYMPTOTIC'))}
            
            if n_samples < MIN_N_SMALL:
                highlighted.add(('TIER_MICRO', 'SPEARMAN'))
            else:
                tier_node = 'TIER_CLINICAL' if n_samples < 100 else 'TIER_ASYMPTOTIC'
                highlighted.add((tier_node, 'SKEW_KURT_CHECK'))
                if used_pearson:
                    highlighted.add(('SKEW_KURT_CHECK', 'PEARSON'))
                else:
                    highlighted.add(('SKEW_KURT_CHECK', 'SPEARMAN'))
                    
            if used_pearson:
                highlighted.add(('PEARSON', 'RESULT'))
            else:
                highlighted.add(('SPEARMAN', 'RESULT'))
                
            highlighted.add(('RESULT', 'CI'))
            highlighted.add(('RESULT', 'EFFECT'))

        elif model_type == "LinearRegression":
            r2 = results.get("r_squared", None)
            r2_label = f"R² = {r2:.3f}" if r2 is not None else ""
            f_p = results.get("f_p_value", None)
            f_p_label = f"F p = {f_p:.4f}" if f_p is not None else ""
            cov_type = results.get("cov_type", "nonrobust")
            
            nodes_info = {
                'START':            {"label": "Start\nLinear Regression",                                  "pos": (0, 12), "isSquare": True},
                'OLS_FIT':          {"label": "Ordinary Least Squares (OLS)\nParameter Estimation",        "pos": (0, 10.5), "isSquare": True},
                'DIAGNOSTICS':      {"label": "Assumption Testing\n(Breusch-Pagan, Shapiro-Wilk)",         "pos": (0, 9), "isSquare": True},
                'ROBUST_BRANCH':    {"label": "Robust Covariance Switch\n(HC3 if BP p < 0.05)",            "pos": (0, 7.5), "isSquare": True},
                'COV_NONROBUST':    {"label": "Standard OLS Covariance\n(Homoscedasticity holds)",          "pos": (-3, 6), "isSquare": False},
                'COV_HC3':          {"label": "HC3 Robust Covariance\n(Heteroscedasticity detected)",     "pos": (3, 6), "isSquare": False},
                'COEFFICIENTS':     {"label": "Coefficient Estimates\n(β, SE, t, p, 95% CI)",              "pos": (0, 4), "isSquare": False},
                'MODEL_FIT':        {"label": f"Model Fit\n{r2_label}  {f_p_label}",                       "pos": (-2.5, 2.5), "isSquare": False},
                'EFFECT':           {"label": "Effect Size (R²)\n(small\u2265.01 med\u2265.09 large\u2265.25)", "pos": (2.5, 2.5), "isSquare": False},
            }
            edges = {
                ('START',         'OLS_FIT'),
                ('OLS_FIT',       'DIAGNOSTICS'),
                ('DIAGNOSTICS',   'ROBUST_BRANCH'),
                ('ROBUST_BRANCH', 'COV_NONROBUST'),
                ('ROBUST_BRANCH', 'COV_HC3'),
                ('COV_NONROBUST', 'COEFFICIENTS'),
                ('COV_HC3',       'COEFFICIENTS'),
                ('COEFFICIENTS',  'MODEL_FIT'),
                ('COEFFICIENTS',  'EFFECT'),
            }
            highlighted = {
                ('START', 'OLS_FIT'),
                ('OLS_FIT', 'DIAGNOSTICS'),
                ('DIAGNOSTICS', 'ROBUST_BRANCH'),
            }
            if cov_type == "HC3":
                highlighted.add(('ROBUST_BRANCH', 'COV_HC3'))
                highlighted.add(('COV_HC3', 'COEFFICIENTS'))
            else:
                highlighted.add(('ROBUST_BRANCH', 'COV_NONROBUST'))
                highlighted.add(('COV_NONROBUST', 'COEFFICIENTS'))
                
            highlighted.add(('COEFFICIENTS', 'MODEL_FIT'))
            highlighted.add(('COEFFICIENTS', 'EFFECT'))

        elif model_type == "LogisticRegression":
            auc = results.get("effect_size", results.get("roc_data", {}).get("auc", None))
            auc_label = f"AUC = {auc:.3f}" if auc is not None else ""
            pseudo_r2 = results.get("pseudo_r_squared", None)
            pr2_label = f"McFadden R² = {pseudo_r2:.3f}" if pseudo_r2 is not None else ""
            p_label = f"p = {p_value:.4f}" if p_value is not None else ""
            
            is_firth = results.get("model_variant") == "Firth Penalized Likelihood"

            nodes_info = {
                'START':                {"label": "Start\nLogistic Regression",                                "pos": (0, 12), "isSquare": True},
                'SEPARATION_CHECK':     {"label": "Check Separation / Convergence\n(SE > 5.0 check)",          "pos": (0, 10.5), "isSquare": True},
                'STANDARD_LOGIT':       {"label": "Standard Logit (MLE)\nEstimation",                          "pos": (-3, 9), "isSquare": False},
                'FIRTH_REGRESSION':     {"label": "Firth Penalized Likelihood\n(Separation/Non-convergence)",  "pos": (3, 9), "isSquare": False},
                'CALIBRATION_ANALYSIS': {"label": "Calibration & Discrimination\n(Brier, AUC, Lowess)",        "pos": (0, 7), "isSquare": True},
                'SIG_YES':              {"label": "Significant Predictors\n(Interpret Odds Ratios)",           "pos": (-3, 5), "isSquare": False},
                'SIG_NO':               {"label": "No Significant Predictors",                                 "pos": (3, 5), "isSquare": False},
                'OR_TABLE':             {"label": "Odds Ratio Table\n(OR, 95% CI, p)",                         "pos": (-4, 3), "isSquare": False},
                'ROC_AUC':              {"label": f"Discrimination (ROC AUC)\n{auc_label}",                    "pos": (0, 3), "isSquare": False},
                'AIC_BIC':              {"label": "Model Comparison\nAIC / BIC",                               "pos": (4, 3), "isSquare": False},
            }
            edges = {
                ('START',                'SEPARATION_CHECK'),
                ('SEPARATION_CHECK',     'STANDARD_LOGIT'),
                ('SEPARATION_CHECK',     'FIRTH_REGRESSION'),
                ('STANDARD_LOGIT',       'CALIBRATION_ANALYSIS'),
                ('FIRTH_REGRESSION',     'CALIBRATION_ANALYSIS'),
                ('CALIBRATION_ANALYSIS', 'SIG_YES'),
                ('CALIBRATION_ANALYSIS', 'SIG_NO'),
                ('SIG_YES',              'OR_TABLE'),
                ('SIG_YES',              'ROC_AUC'),
                ('SIG_YES',              'AIC_BIC'),
                ('SIG_NO',               'ROC_AUC'),
                ('SIG_NO',               'AIC_BIC'),
            }
            
            highlighted = {
                ('START', 'SEPARATION_CHECK'),
            }
            if is_firth:
                highlighted.add(('SEPARATION_CHECK', 'FIRTH_REGRESSION'))
                highlighted.add(('FIRTH_REGRESSION', 'CALIBRATION_ANALYSIS'))
            else:
                highlighted.add(('SEPARATION_CHECK', 'STANDARD_LOGIT'))
                highlighted.add(('STANDARD_LOGIT', 'CALIBRATION_ANALYSIS'))
                
            if sig:
                highlighted.add(('CALIBRATION_ANALYSIS', 'SIG_YES'))
                highlighted.add(('SIG_YES', 'OR_TABLE'))
                highlighted.add(('SIG_YES', 'ROC_AUC'))
                highlighted.add(('SIG_YES', 'AIC_BIC'))
            else:
                highlighted.add(('CALIBRATION_ANALYSIS', 'SIG_NO'))
                highlighted.add(('SIG_NO', 'ROC_AUC'))
                highlighted.add(('SIG_NO', 'AIC_BIC'))

        elif model_type == "ANCOVA":
            eta2 = results.get("effect_size", None)
            eta2_label = f"η² = {eta2:.3f}" if eta2 is not None else ""
            p_label = f"p = {p_value:.4f}" if p_value is not None else ""
            
            nodes_info = {
                'START':           {"label": "Start\nANCOVA Analysis",                                     "pos": (0, 12), "isSquare": True},
                'COVAR':           {"label": "Define Covariate(s)\n& Factor(s)",                           "pos": (0, 10.5), "isSquare": True},
                'SLOPE_HOM':       {"label": f"Test Homogeneity of Slopes\nBoth: {_yn(slopes_ok)}",        "pos": (0, 9), "isSquare": True},
                'SLOPE_OK':        {"label": "Slopes Homogeneous\n(ANCOVA assumption met)",                "pos": (-3, 7.5), "isSquare": False},
                'SLOPE_FAIL':      {"label": "Slopes Heterogeneous\n(Interaction p < 0.05)",                "pos": (3, 7.5), "isSquare": False},
                'SIMPLE_SLOPES_JN':{"label": "Simple Slopes & Johnson-Neyman\n(Pick-a-Point & regions)",   "pos": (3, 6), "isSquare": True},
                'FIT':             {"label": f"Model Fit & Effects Table\n{p_label}  {eta2_label}",        "pos": (0, 4.5), "isSquare": True},
                'SIG_YES':         {"label": "Significant Main Effects\n(Adjusted Means differ)",          "pos": (-2.5, 3), "isSquare": False},
                'SIG_NO':          {"label": "No Significant Main Effects",                                "pos": (2.5, 3), "isSquare": False},
                'POSTHOC':         {"label": "Post-Hoc Pairwise Comparisons\n(Bonferroni/Tukey adjusted)", "pos": (-3, 1.5), "isSquare": False},
                'EFFECT':          {"label": "Effect Size (Partial η²)\n(small\u2265.01 med\u2265.06 large\u2265.14)", "pos": (3, 1.5), "isSquare": False},
            }
            edges = {
                ('START',           'COVAR'),
                ('COVAR',           'SLOPE_HOM'),
                ('SLOPE_HOM',       'SLOPE_OK'),
                ('SLOPE_HOM',       'SLOPE_FAIL'),
                ('SLOPE_OK',        'FIT'),
                ('SLOPE_FAIL',      'SIMPLE_SLOPES_JN'),
                ('SIMPLE_SLOPES_JN','FIT'),
                ('FIT',             'SIG_YES'),
                ('FIT',             'SIG_NO'),
                ('SIG_YES',         'POSTHOC'),
                ('SIG_YES',         'EFFECT'),
                ('SIG_NO',          'EFFECT'),
            }
            highlighted = {
                ('START', 'COVAR'),
                ('COVAR', 'SLOPE_HOM'),
            }
            if slopes_ok is False:
                highlighted.add(('SLOPE_HOM', 'SLOPE_FAIL'))
                highlighted.add(('SLOPE_FAIL', 'SIMPLE_SLOPES_JN'))
                highlighted.add(('SIMPLE_SLOPES_JN', 'FIT'))
            else:
                highlighted.add(('SLOPE_HOM', 'SLOPE_OK'))
                highlighted.add(('SLOPE_OK', 'FIT'))
                
            if sig:
                highlighted.add(('FIT', 'SIG_YES'))
                highlighted.add(('SIG_YES', 'POSTHOC'))
                highlighted.add(('SIG_YES', 'EFFECT'))
            else:
                highlighted.add(('FIT', 'SIG_NO'))
                highlighted.add(('SIG_NO', 'EFFECT'))

        elif model_type == "LMM":
            icc_val = results.get("icc", results.get("effect_size", None))
            icc_label = f"ICC = {icc_val:.3f}" if icc_val is not None else ""
            p_label = f"p = {p_value:.4f}" if p_value is not None else ""
            df_method = results.get("df_method", "Asymptotic")
            is_bw = (df_method == "Between-Within" or n_samples < 100)

            nodes_info = {
                'START':                {"label": "Start\nLinear Mixed Model",                                 "pos": (0, 12), "isSquare": True},
                'RANDOM_STRUCTURE_LRT': {"label": "Random Structure LRT Check\n(Random Intercept vs. Intercept + Slope)", "pos": (0, 10.5), "isSquare": True},
                'FINAL_FIT':            {"label": "Final Fit Estimation\n(REML optimization)",                  "pos": (0, 9), "isSquare": True},
                'BW_DF_CORRECTION':     {"label": "Between-Within DF Correction\n(Applied because N < 100)",  "pos": (-3, 7.5), "isSquare": True},
                'ASYMP_DF':             {"label": "Asymptotic Wald Inference\n(No correction, N >= 100)",      "pos": (3, 7.5), "isSquare": True},
                'FIXED_EFFECTS_TEST':   {"label": f"Fixed Effects Significance\n{p_label}",                   "pos": (0, 6), "isSquare": True},
                'SIG_YES':              {"label": "Significant Fixed Effects",                                 "pos": (-2.5, 4.5), "isSquare": False},
                'SIG_NO':               {"label": "No Significant Fixed Effects",                             "pos": (2.5, 4.5), "isSquare": False},
                'FIXEF_TABLE':          {"label": "Fixed Effects Coefficients\n(β, SE, t/z, p, 95% CI)",      "pos": (-3, 3), "isSquare": False},
                'ICC_VAL':              {"label": f"Intraclass Correlation (ICC)\n{icc_label}",                "pos": (0, 3), "isSquare": False},
                'MODEL_FIT':            {"label": "Model Information Criteria\n(AIC/BIC comparison)",          "pos": (3, 3), "isSquare": False},
            }
            edges = {
                ('START',                'RANDOM_STRUCTURE_LRT'),
                ('RANDOM_STRUCTURE_LRT', 'FINAL_FIT'),
                ('FINAL_FIT',            'BW_DF_CORRECTION'),
                ('FINAL_FIT',            'ASYMP_DF'),
                ('BW_DF_CORRECTION',     'FIXED_EFFECTS_TEST'),
                ('ASYMP_DF',             'FIXED_EFFECTS_TEST'),
                ('FIXED_EFFECTS_TEST',   'SIG_YES'),
                ('FIXED_EFFECTS_TEST',   'SIG_NO'),
                ('SIG_YES',              'FIXEF_TABLE'),
                ('SIG_YES',              'ICC_VAL'),
                ('SIG_NO',               'ICC_VAL'),
                ('ICC_VAL',              'MODEL_FIT'),
                ('SIG_NO',               'MODEL_FIT'),
            }
            highlighted = {
                ('START', 'RANDOM_STRUCTURE_LRT'),
                ('RANDOM_STRUCTURE_LRT', 'FINAL_FIT'),
            }
            if is_bw:
                highlighted.add(('FINAL_FIT', 'BW_DF_CORRECTION'))
                highlighted.add(('BW_DF_CORRECTION', 'FIXED_EFFECTS_TEST'))
            else:
                highlighted.add(('FINAL_FIT', 'ASYMP_DF'))
                highlighted.add(('ASYMP_DF', 'FIXED_EFFECTS_TEST'))
                
            if sig:
                highlighted.add(('FIXED_EFFECTS_TEST', 'SIG_YES'))
                highlighted.add(('SIG_YES', 'FIXEF_TABLE'))
                highlighted.add(('SIG_YES', 'ICC_VAL'))
            else:
                highlighted.add(('FIXED_EFFECTS_TEST', 'SIG_NO'))
                highlighted.add(('SIG_NO', 'ICC_VAL'))
                highlighted.add(('SIG_NO', 'MODEL_FIT'))
            highlighted.add(('ICC_VAL', 'MODEL_FIT'))

        else: # CorrelationMatrix
            method_label = results.get("method", "auto").capitalize()
            correction_label = results.get("correction", "None") or "None"
            variables = results.get("variables", [])
            n_vars = len(variables)

            nodes_info = {
                'START':    {"label": f"Start\nExploratory Correlation Matrix\n({n_vars} variables)", "pos": (0, 8), "isSquare": True},
                'METHOD':   {"label": f"Method: {method_label}\n(auto = Shapiro-Wilk per pair)", "pos": (0, 6.5), "isSquare": True},
                'CORRECT':  {"label": f"Multiple Testing Correction\n{correction_label}", "pos": (0, 5), "isSquare": True},
                'MATRIX':   {"label": "Correlation Matrix\n(r / ρ per pair)", "pos": (-2.5, 3.5), "isSquare": False},
                'SIG_MAP':  {"label": "Significance Map\n(corrected p-values)", "pos": (2.5, 3.5), "isSquare": False},
                'INTERPRET':{"label": "Interpret significant pairs\n(effect size, direction)", "pos": (0, 2), "isSquare": False},
            }
            edges = {
                ('START', 'METHOD'),
                ('METHOD', 'CORRECT'),
                ('CORRECT', 'MATRIX'),
                ('CORRECT', 'SIG_MAP'),
                ('MATRIX', 'INTERPRET'),
                ('SIG_MAP', 'INTERPRET'),
            }
            highlighted = {
                ('START', 'METHOD'), ('METHOD', 'CORRECT'),
                ('CORRECT', 'MATRIX'), ('CORRECT', 'SIG_MAP'),
                ('MATRIX', 'INTERPRET'), ('SIG_MAP', 'INTERPRET'),
            }

        return nodes_info, edges, highlighted, tree_meta

    @staticmethod
    def _get_association_tree_json(results: dict, model_type: str) -> dict | None:
        try:
            alpha = results.get("alpha", 0.05)
            nodes_info, edges, highlighted, tree_meta = DecisionTreeVisualizer._build_tree_topology(results, model_type, alpha)

            active_nodes = set()
            for u, v in highlighted:
                active_nodes.add(u)
                active_nodes.add(v)

            node_list = []
            for nid, info in nodes_info.items():
                node_list.append({
                    "id": nid,
                    "x": float(info["pos"][0]),
                    "y": float(info["pos"][1]),
                    "label": info["label"],
                    "isActive": nid in active_nodes,
                    "active": nid in active_nodes,
                    "isSquare": info["isSquare"],
                })

            edge_list = [
                {
                    "source": u,
                    "from": u,
                    "target": v,
                    "to": v,
                    "isActive": (u, v) in highlighted,
                    "active": (u, v) in highlighted,
                }
                for u, v in edges
            ]
            return {"tree_meta": tree_meta, "nodes": node_list, "edges": edge_list}

        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"WARNING DecisionTreeVisualizer._get_association_tree_json: {exc}")
            return None

    def create_association_tree(self, results, output_path=None):
        """
        Public instance-method wrapper around _visualize_association_test.
        Returns the saved file path (str) on success, or None on failure.
        """
        model_type = results.get("model_type", "Correlation")
        return DecisionTreeVisualizer._visualize_association_test(results, model_type, output_path)

    @staticmethod
    def _visualize_association_test(results, model_type, output_path=None):
        """
        Generates a decision tree for association/regression tests:
        Correlation (Pearson/Spearman), Linear Regression, Logistic Regression, ANCOVA.
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            import tempfile
            plt.style.use('seaborn-v0_8-whitegrid')

            test_name = results.get("test", results.get("test_name", model_type))
            alpha = results.get("alpha", 0.05)

            nodes_info, edges, highlighted, tree_meta = DecisionTreeVisualizer._build_tree_topology(results, model_type, alpha)

            G = nx.DiGraph()
            for node_id, info in nodes_info.items():
                G.add_node(node_id, label=info["label"], pos=info["pos"])
            for edge in edges:
                G.add_edge(*edge)

            pos_dict = {nid: info["pos"] for nid, info in nodes_info.items()}

            highlighted_edges = [(u, v) for u, v in G.edges() if (u, v) in highlighted]
            regular_edges     = [(u, v) for u, v in G.edges() if (u, v) not in highlighted]

            highlighted_nodes = set()
            for u, v in highlighted:
                highlighted_nodes.add(u)
                highlighted_nodes.add(v)

            # ── Figure size ───────────────────────────────────────────────────
            x_vals = [xy[0] for xy in pos_dict.values()]
            y_vals = [xy[1] for xy in pos_dict.values()]
            x_span = max(x_vals) - min(x_vals) if x_vals else 10
            y_span = max(y_vals) - min(y_vals) if y_vals else 10
            fig_width  = max(14.0, min(28.0, x_span * 1.6 + 6.0))
            fig_height = max(10.0, min(22.0, y_span * 0.9 + 4.0))

            plt.figure(figsize=(fig_width, fig_height))

            node_labels = nx.get_node_attributes(G, 'label')

            sq_nodes = [n for n in G.nodes() if nodes_info[n]["isSquare"]]
            rd_nodes = [n for n in G.nodes() if n not in sq_nodes]

            nx.draw_networkx_nodes(G, pos_dict,
                nodelist=[n for n in sq_nodes if n in highlighted_nodes],
                node_size=3000, node_color='#ffcccc', edgecolors='black', linewidths=1.5, node_shape='s')
            nx.draw_networkx_nodes(G, pos_dict,
                nodelist=[n for n in sq_nodes if n not in highlighted_nodes],
                node_size=3000, node_color='white', edgecolors='black', linewidths=1.5, node_shape='s')
            nx.draw_networkx_nodes(G, pos_dict,
                nodelist=[n for n in rd_nodes if n in highlighted_nodes],
                node_size=3000, node_color='#ffcccc', edgecolors='black', linewidths=1.5, node_shape='o')
            nx.draw_networkx_nodes(G, pos_dict,
                nodelist=[n for n in rd_nodes if n not in highlighted_nodes],
                node_size=3000, node_color='white', edgecolors='black', linewidths=1.5, node_shape='o')

            nx.draw_networkx_edges(G, pos_dict, edgelist=highlighted_edges, width=4, edge_color='red')
            nx.draw_networkx_edges(G, pos_dict, edgelist=regular_edges, width=1, edge_color='black')

            nx.draw_networkx_labels(G, pos_dict, labels=node_labels, font_size=11,
                font_family='sans-serif', font_weight='bold',
                bbox=dict(boxstyle='round,pad=0.28', facecolor='white', alpha=0.7, edgecolor='lightgray'))

            plt.gcf().suptitle(f"Statistical Decision Path: {test_name}", fontsize=15, y=0.98)

            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch
            legend_elements = [
                Line2D([0], [0], color='red', lw=4, label='Taken path'),
                Patch(facecolor='#ffcccc', edgecolor='black', label='Steps performed'),
                Patch(facecolor='white', edgecolor='black', label='Alternative steps'),
            ]
            plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.99),
                       fontsize=10, frameon=True, facecolor='white', edgecolor='black',
                       framealpha=0.9, shadow=True)

            plt.axis('off')
            plt.tight_layout(rect=[0, 0, 1, 0.97])

            if output_path:
                out_file = f"{output_path}.png"
                plt.savefig(out_file, format="png", dpi=200, transparent=False,
                            facecolor='white', bbox_inches='tight')
                plt.close('all')
                return out_file
            else:
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    plt.savefig(tmp.name, format="png", dpi=200, transparent=False,
                                facecolor='white', bbox_inches='tight')
                    path = tmp.name
                plt.close('all')
                return path

        except Exception as e:
            print(f"Error generating association decision tree: {e}")
            import traceback
            traceback.print_exc()
            return None

    @staticmethod
    def _highlight_posthoc_path(results, highlighted):
        """
        Helper method to determine which post-hoc test path to highlight based on the results.
        This centralizes the logic to avoid duplication and conflicts.
        
        For One-Way ANOVA: Shows multiple options (user can choose)
        For Two-Way ANOVA, RM ANOVA, Mixed ANOVA: Shows the actually performed test
        """
        test_name = results.get("test_name", results.get("test", "")).lower()
        posthoc_test = results.get("posthoc_test")
        
        # Check if this is a One-Way ANOVA where users should see options
        is_one_way_anova = ("one-way" in test_name or 
                           (("anova" in test_name or "one way" in test_name) and 
                            "two-way" not in test_name and "two way" not in test_name and
                            "rm" not in test_name and "repeated" not in test_name and
                            "mixed" not in test_name))
        
        # Check if this is an advanced ANOVA (Two-Way, Mixed, RM)
        is_advanced_anova = ("two-way" in test_name or "two way" in test_name or 
                            "mixed" in test_name or "rm" in test_name or "repeated" in test_name)
        
        if is_one_way_anova and not posthoc_test:
            # For One-Way ANOVA with no specific post-hoc performed: show all options (including Dunnett)
            print(f"DEBUG TREE: One-Way ANOVA detected - showing all post-hoc options for user choice")
            highlighted.add(('O1_PH', 'P1_PH_TK'))  # Tukey
            highlighted.add(('O1_PH', 'P1_PH_DN'))  # Dunnett  
            highlighted.add(('O1_PH', 'P1_PH_SD'))  # Holm-Bonferroni
            return
        elif is_advanced_anova and not posthoc_test:
            # For Advanced ANOVAs with no specific post-hoc performed: show only Tukey and Pairwise (no Dunnett)
            print(f"DEBUG TREE: Advanced ANOVA detected - showing only Tukey and Pairwise post-hoc options")
            highlighted.add(('O1_PH', 'P1_PH_TK'))  # Tukey
            highlighted.add(('O1_PH', 'P1_PH_SD'))  # Holm-Bonferroni (Pairwise t-tests)
            return
        
        # For specific tests or when a post-hoc was actually performed: show the specific path
        if posthoc_test:
            print(f"DEBUG TREE: Post-hoc test detected: '{posthoc_test}'")
            if "tukey" in posthoc_test.lower():
                print(f"DEBUG TREE: Highlighting Tukey path")
                highlighted.add(('O1_PH', 'P1_PH_TK'))
            elif "dunnett" in posthoc_test.lower() and "t3" not in posthoc_test.lower():
                print(f"DEBUG TREE: Highlighting Dunnett path")
                highlighted.add(('O1_PH', 'P1_PH_DN'))
            elif ("holm" in posthoc_test.lower() or "sidak" in posthoc_test.lower() or 
                  "pairwise t-test" in posthoc_test.lower() or "pairwise" in posthoc_test.lower()):
                print(f"DEBUG TREE: Highlighting Holm-Bonferroni path for posthoc: '{posthoc_test}'")
                highlighted.add(('O1_PH', 'P1_PH_SD'))
            # Handle non-parametric post-hoc tests
            elif "mann-whitney" in posthoc_test.lower():
                print(f"DEBUG TREE: Highlighting Pairwise Mann-Whitney-U path")
                highlighted.add(('L2_PH', 'M2_PH_MWU'))
            elif "dunn" in posthoc_test.lower():
                print(f"DEBUG TREE: Highlighting Dunn test path")
                highlighted.add(('L2_PH', 'M2_PH_DU'))
            elif "wilcoxon" in posthoc_test.lower():
                print(f"DEBUG TREE: Highlighting Wilcoxon post-hoc path")
                highlighted.add(('L2_PH', 'NP_PH_WILC'))
            else:
                print(f"DEBUG TREE: Unknown post-hoc test '{posthoc_test}', defaulting to Holm-Bonferroni")
                highlighted.add(('O1_PH', 'P1_PH_SD'))
        else:
            # Check for pairwise comparisons to infer post-hoc test type
            pairwise_comps = results.get("pairwise_comparisons", [])
            if pairwise_comps:
                # Try to infer from the test names in pairwise comparisons
                first_comp = pairwise_comps[0]
                test_name_in_comp = first_comp.get("test", "").lower()
                
                # Check both "corrected" and "correction_method" fields for the correction method
                corrected_info = first_comp.get("corrected", "")
                correction_method = first_comp.get("correction_method", "")
                correction_field = first_comp.get("correction", "")  # Also check "correction" field
                corrected_method = str(corrected_info).lower() if corrected_info else ""
                correction_method_str = str(correction_method).lower() if correction_method else ""
                correction_field_str = str(correction_field).lower() if correction_field else ""
                
                print(f"DEBUG TREE: No explicit posthoc_test, inferring from pairwise test: '{test_name_in_comp}' with correction: '{corrected_method}' / '{correction_method_str}' / '{correction_field_str}'")
                
                if ("holm" in test_name_in_comp or "sidak" in test_name_in_comp or 
                    "holm" in corrected_method or "sidak" in corrected_method or 
                    "holm" in correction_method_str or "sidak" in correction_method_str or
                    "holm" in correction_field_str or "sidak" in correction_field_str):
                    print(f"DEBUG TREE: Inferred Holm-Bonferroni from pairwise test")
                    highlighted.add(('O1_PH', 'P1_PH_SD'))
                elif ("tukey" in test_name_in_comp or 
                      "tukey" in correction_field_str):
                    print(f"DEBUG TREE: Inferred Tukey from pairwise test")
                    highlighted.add(('O1_PH', 'P1_PH_TK'))
                elif "dunnett" in test_name_in_comp:
                    print(f"DEBUG TREE: Inferred Dunnett from pairwise test")
                    highlighted.add(('O1_PH', 'P1_PH_DN'))
                elif ("pairwise" in test_name_in_comp and ("holm" in corrected_method or "sidak" in corrected_method or 
                                                        "holm" in correction_method_str or "sidak" in correction_method_str)):
                    print(f"DEBUG TREE: Inferred Holm-Bonferroni from pairwise test with correction method")
                    highlighted.add(('O1_PH', 'P1_PH_SD'))
                else:
                    print(f"DEBUG TREE: Unknown pairwise test type, showing options for choice")
                    if is_one_way_anova:
                        # Show all options for One-Way ANOVA (including Dunnett)
                        highlighted.add(('O1_PH', 'P1_PH_TK'))
                        highlighted.add(('O1_PH', 'P1_PH_DN'))
                        highlighted.add(('O1_PH', 'P1_PH_SD'))
                    elif is_advanced_anova:
                        # Show only Tukey and Pairwise for Advanced ANOVAs (no Dunnett)
                        highlighted.add(('O1_PH', 'P1_PH_TK'))
                        highlighted.add(('O1_PH', 'P1_PH_SD'))
                    else:
                        highlighted.add(('O1_PH', 'P1_PH_TK'))  # Default to Tukey
            else:
                print(f"DEBUG TREE: No post-hoc info available")
                if is_one_way_anova:
                    print(f"DEBUG TREE: One-Way ANOVA - showing all post-hoc options for user choice")
                    highlighted.add(('O1_PH', 'P1_PH_TK'))  # Tukey
                    highlighted.add(('O1_PH', 'P1_PH_DN'))  # Dunnett
                    highlighted.add(('O1_PH', 'P1_PH_SD'))  # Holm-Bonferroni
                elif is_advanced_anova:
                    print(f"DEBUG TREE: Advanced ANOVA - showing only Tukey and Pairwise post-hoc options")
                    highlighted.add(('O1_PH', 'P1_PH_TK'))  # Tukey
                    highlighted.add(('O1_PH', 'P1_PH_SD'))  # Holm-Bonferroni (Pairwise t-tests)
                else:
                    print(f"DEBUG TREE: Using default Tukey for other test types")
                    highlighted.add(('O1_PH', 'P1_PH_TK'))

    @staticmethod
    def _highlight_rm_posthoc_path(results, highlighted):
        """
        Helper method to determine which RM ANOVA post-hoc test path to highlight.
        """
        posthoc_test = results.get("posthoc_test")
        
        if posthoc_test:
            print(f"DEBUG TREE: RM ANOVA Post-hoc test detected: '{posthoc_test}'")
            if "tukey" in posthoc_test.lower() and "rm" in posthoc_test.lower():
                print(f"DEBUG TREE: Highlighting RM Tukey path")
                highlighted.add(('O1_RM_PH', 'P1_RM_TK'))
            elif ("paired" in posthoc_test.lower() or 
                  "holm" in posthoc_test.lower() or 
                  "sidak" in posthoc_test.lower()):
                print(f"DEBUG TREE: Highlighting RM Paired t-tests path")
                highlighted.add(('O1_RM_PH', 'P1_RM_PAIRED'))
            else:
                print(f"DEBUG TREE: Unknown RM post-hoc test, defaulting to Tukey RM")
                highlighted.add(('O1_RM_PH', 'P1_RM_TK'))
        else:
            # Check pairwise comparisons for RM-specific tests
            pairwise_comps = results.get("pairwise_comparisons", [])
            if pairwise_comps:
                first_comp = pairwise_comps[0]
                test_name_in_comp = first_comp.get("test", "").lower()
                
                if "paired" in test_name_in_comp or "dependent" in test_name_in_comp:
                    print(f"DEBUG TREE: Inferred RM Paired t-tests from pairwise test")
                    highlighted.add(('O1_RM_PH', 'P1_RM_PAIRED'))
                elif "tukey" in test_name_in_comp:
                    print(f"DEBUG TREE: Inferred RM Tukey from pairwise test")
                    highlighted.add(('O1_RM_PH', 'P1_RM_TK'))
                else:
                    print(f"DEBUG TREE: Default RM Tukey for unknown pairwise test")
                    highlighted.add(('O1_RM_PH', 'P1_RM_TK'))
            else:
                print(f"DEBUG TREE: No RM post-hoc info available, showing both options")
                highlighted.add(('O1_RM_PH', 'P1_RM_TK'))
                highlighted.add(('O1_RM_PH', 'P1_RM_PAIRED'))

    @staticmethod
    def _highlight_mixed_posthoc_path(results, highlighted):
        """
        Helper method to determine which Mixed ANOVA post-hoc test path to highlight.
        """
        posthoc_test = results.get("posthoc_test")
        
        if posthoc_test:
            print(f"DEBUG TREE: Mixed ANOVA Post-hoc test detected: '{posthoc_test}'")
            if "mixed" in posthoc_test.lower() and "tukey" in posthoc_test.lower():
                print(f"DEBUG TREE: Highlighting Mixed Tukey path")
                highlighted.add(('O1_MIX_PH', 'P1_MIX_TK'))
            elif "between" in posthoc_test.lower():
                print(f"DEBUG TREE: Highlighting Between-subjects comparisons path")
                highlighted.add(('O1_MIX_PH', 'P1_MIX_BETWEEN'))
            elif "within" in posthoc_test.lower():
                print(f"DEBUG TREE: Highlighting Within-subjects comparisons path")
                highlighted.add(('O1_MIX_PH', 'P1_MIX_WITHIN'))
            else:
                print(f"DEBUG TREE: Unknown Mixed post-hoc test, defaulting to Mixed Tukey")
                highlighted.add(('O1_MIX_PH', 'P1_MIX_TK'))
        else:
            # Check pairwise comparisons for Mixed-specific tests
            pairwise_comps = results.get("pairwise_comparisons", [])
            if pairwise_comps:
                first_comp = pairwise_comps[0]
                test_name_in_comp = first_comp.get("test", "").lower()
                
                if "between" in test_name_in_comp:
                    print(f"DEBUG TREE: Inferred Between-subjects from pairwise test")
                    highlighted.add(('O1_MIX_PH', 'P1_MIX_BETWEEN'))
                elif "within" in test_name_in_comp or "paired" in test_name_in_comp:
                    print(f"DEBUG TREE: Inferred Within-subjects from pairwise test")
                    highlighted.add(('O1_MIX_PH', 'P1_MIX_WITHIN'))
                elif "mixed" in test_name_in_comp and "tukey" in test_name_in_comp:
                    print(f"DEBUG TREE: Inferred Mixed Tukey from pairwise test")
                    highlighted.add(('O1_MIX_PH', 'P1_MIX_TK'))
                else:
                    print(f"DEBUG TREE: Default Mixed Tukey for unknown pairwise test")
                    highlighted.add(('O1_MIX_PH', 'P1_MIX_TK'))
            else:
                print(f"DEBUG TREE: No Mixed post-hoc info available, showing all options")
                highlighted.add(('O1_MIX_PH', 'P1_MIX_TK'))
                highlighted.add(('O1_MIX_PH', 'P1_MIX_BETWEEN'))
                highlighted.add(('O1_MIX_PH', 'P1_MIX_WITHIN'))

def test_decision_tree_visualization():
    # Beispielhafte Ergebnisse für einen One-Way-ANOVA mit signifikantem Ergebnis und Tukey-Posthoc
    results = {
        "test": "One-way ANOVA",
        "test_recommendation": "parametric",
        "transformation": "None",
        "p_value": 0.002,
        "alpha": 0.05,
        "groups": ["A", "B", "C"],
        "normality_tests": {
            "A": {"is_normal": True},
            "B": {"is_normal": True},
            "C": {"is_normal": True},
            "all_data": {"is_normal": True}
        },
        "variance_test": {"equal_variance": True},
        "posthoc_test": "Tukey",
        "pairwise_comparisons": [
            {"groups": ("A", "B"), "p_value": 0.01, "test": "Tukey"},
            {"groups": ("A", "C"), "p_value": 0.03, "test": "Tukey"},
            {"groups": ("B", "C"), "p_value": 0.20, "test": "Tukey"}
        ]
    }
    # Generiere und speichere den Entscheidungsbaum
    output_path = DecisionTreeVisualizer.visualize(results, output_path="decision_tree_example")
    print(f"Decision tree saved to: {output_path}")

def test_rm_anova_decision_tree():
    # NEW: Beispielhafte Ergebnisse für RM ANOVA mit Sphärizitätskorrektur
    results = {
        "test": "Repeated Measures ANOVA",
        "test_recommendation": "parametric",
        "transformation": "None",
        "p_value": 0.01,
        "alpha": 0.05,
        "groups": ["Time1", "Time2", "Time3", "Time4"],
        "normality_tests": {
            "Time1": {"is_normal": True},
            "Time2": {"is_normal": True},
            "Time3": {"is_normal": True},
            "Time4": {"is_normal": True},
            "all_data": {"is_normal": True}
        },
        "variance_test": {"equal_variance": True},
        "sphericity_test": {
            "test_name": "Mauchly's Test for Sphericity",
            "W": 0.65,
            "p_value": 0.03,
            "sphericity_assumed": False
        },
        "correction_used": "Greenhouse-Geisser (ε = 0.72 ≤ 0.75)",
        "corrected_p_value": 0.018,
        "posthoc_test": "Paired t-tests (Holm-Bonferroni corrected)",
        "pairwise_comparisons": [
            {"groups": ("Time1", "Time2"), "p_value": 0.02, "test": "Paired t-test", "correction": "Holm-Bonferroni"},
            {"groups": ("Time1", "Time3"), "p_value": 0.005, "test": "Paired t-test", "correction": "Holm-Bonferroni"},
            {"groups": ("Time2", "Time3"), "p_value": 0.15, "test": "Paired t-test", "correction": "Holm-Bonferroni"}
        ]
    }
    # Generiere und speichere den Entscheidungsbaum
    output_path = DecisionTreeVisualizer.visualize(results, output_path="rm_anova_decision_tree")
    print(f"RM ANOVA Decision tree saved to: {output_path}")

def test_mixed_anova_decision_tree():
    # NEW: Beispielhafte Ergebnisse für Mixed ANOVA mit Within-Factor Korrektur
    results = {
        "test": "Mixed ANOVA",
        "test_recommendation": "parametric",
        "transformation": "None",
        "p_value": 0.008,
        "alpha": 0.05,
        "groups": ["Group_A_Time1", "Group_A_Time2", "Group_B_Time1", "Group_B_Time2"],
        "normality_tests": {
            "Group_A_Time1": {"is_normal": True},
            "Group_A_Time2": {"is_normal": True},
            "Group_B_Time1": {"is_normal": True},
            "Group_B_Time2": {"is_normal": True},
            "all_data": {"is_normal": True}
        },
        "variance_test": {"equal_variance": True},
        "within_sphericity_test": {
            "test_name": "Mauchly's Test for Sphericity (Within-Factor)",
            "factor": "Time",
            "W": 0.82,
            "p_value": 0.04,
            "sphericity_assumed": False
        },
        "within_correction_used": "Huynh-Feldt (ε = 0.88 > 0.75)",
        "within_corrected_p_value": 0.012,
        "posthoc_test": "Mixed Tukey (Between/Within)",
        "pairwise_comparisons": [
            {"groups": ("Group_A", "Group_B"), "p_value": 0.01, "test": "Between-subjects Tukey"},
            {"groups": ("Time1", "Time2"), "p_value": 0.03, "test": "Within-subjects Tukey"}
        ]
    }
    # Generiere und speichere den Entscheidungsbaum
    output_path = DecisionTreeVisualizer.visualize(results, output_path="mixed_anova_decision_tree")
    print(f"Mixed ANOVA Decision tree saved to: {output_path}")

# Zum Testen einfach aufrufen:
if __name__ == "__main__":
    test_decision_tree_visualization()
    test_rm_anova_decision_tree()  # NEW
    test_mixed_anova_decision_tree()  # NEW
    