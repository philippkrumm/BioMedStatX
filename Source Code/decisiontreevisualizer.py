import matplotlib.pyplot as plt
import networkx as nx
import tempfile
import os


class DecisionTreeVisualizer:
    """
    Creates visual decision trees for statistical test workflows with the actual path highlighted.
    Uses networkx and matplotlib to generate a directed graph showing the decision-making process.
    """
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
            # Apply a clean style for better aesthetics
            plt.style.use('seaborn-v0_8-whitegrid')

            # Extract key information from results
            test_name = results.get("test", "")
            test_type = results.get("test_recommendation", results.get("test_type", ""))
            transformation = results.get("transformation", "None")
            p_value = results.get("p_value", None)
            
            # Get test_info for more detailed analysis
            test_info = results.get("test_info", {})
            normality_tests = test_info.get("normality_tests", results.get("normality_tests", {}))
            variance_test = test_info.get("variance_test", results.get("variance_test", {}))
            sphericity_test = results.get("sphericity_test", {})
            posthoc_test = results.get("posthoc_test", None)
            
            # BETTER DETECTION OF TEST TYPE FOR T-TESTS
            # This needs to be at the beginning of the visualize method
            dependence_type = "independent"  # Default
            
            # Check for explicit indicators in the parameters
            dependent_param = results.get("dependent", None)
            print(f"DEBUG TREE: dependent_param={dependent_param} (type={type(dependent_param)})")
            if isinstance(dependent_param, bool):
                dependence_type = "dependent" if dependent_param else "independent"
            elif dependent_param is not None:
                # Fallback: try to interpret string or int
                if str(dependent_param).lower() in ("true", "1"):
                    dependence_type = "dependent"
                else:
                    dependence_type = "independent"
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

            has_sphericity = sphericity_test.get("has_sphericity", None)
            was_transformed = transformation != "None"
            
            # Define auto_switched flag here
            auto_switched = False
            # Check for auto-switch in analysis log
            if results.get("analysis_log", ""):
                if "Switching to nonparametric" in results.get("analysis_log", ""):
                    auto_switched = True
            # Check test name for indicators
            if test_name.lower().startswith("nonparametric_"):
                auto_switched = True

            # Determine number of groups
            n_groups = 0
            if "groups" in results:
                n_groups = len(results.get("groups", []))
            elif "descriptive" in results:
                n_groups = len(results.get("descriptive", {}))
            else:
                # Fallback: infer from test name
                if "two" in test_name.lower():
                    # For two-way ANOVA, we need to treat this as multiple groups
                    if "anova" in test_name.lower() or "two_way" in test_name.lower():
                        n_groups = 3  # Treat two-way ANOVA as always having multiple groups
                    else:
                        n_groups = 2
                elif "one-way" in test_name.lower() or "anova" in test_name.lower():
                    n_groups = 3  # Assume multiple groups for ANOVA
                else:
                    n_groups = 2

            # Check if non-parametric alternatives are disabled
            nonparametric_disabled = False
            if "error" in results and results.get("error"):
                error_msg = str(results.get("error", ""))
                if "nonparametric alternatives are" in error_msg.lower() and "disabled" in error_msg.lower():
                    nonparametric_disabled = True
                    print(f"DEBUG TREE: Detected disabled non-parametric alternatives")

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
                k1_m_sph_label = "Sphericity not required\n(2 levels)"
            elif has_sphericity is True:
                k1_m_sph_label = "Has Sphericity"
            elif has_sphericity is False:
                k1_m_sph_label = "No Sphericity"
            else:
                k1_m_sph_label = "Sphericity\nCheck"

            # Define nodes with positions and labels (keeping your existing positions)
            nodes_info = {
                # Common path
                'A': {"label": "Start", "pos": (0, 14)},
                'B': {"label": f"Check Assumptions\nShapiro-Wilk: {is_normal}\nLevene: {has_equal_variance}", "pos": (0, 12.5)},
                'C': {"label": f"Assumptions{': ' + ('Met' if is_normal and has_equal_variance else 'Not Met')}", "pos": (0, 11)},

                # Transformation branch point
                'D1': {"label": f"No Transformation\nNeeded", "pos": (-2, 9.5)},
                'D2': {"label": f"Apply Transformation\n{transformation}", "pos": (2, 9.5)},
                'E': {"label": "Re-check Assumptions", "pos": (2, 8)},

                # Test recommendation
                'F': {"label": f"{test_recommendation_label}", "pos": (0, 6.5)},

                # Welch tests - direct branches from test recommendation
                'WELCH_T_TEST': {"label": "Welch's t-test\n(2 groups)", "pos": (-1.5, 4.5)},
                'WELCH_ANOVA': {"label": "Welch-ANOVA\n(>2 groups)", "pos": (1.5, 4.5)},
                'WELCH_DUNNETT_T3': {"label": "Dunnett T3\nPost-hoc", "pos": (1.5, 3)},

                # Parametric branch
                'G1': {"label": "Parametric Test", "pos": (-5, 5)},
                'H1': {"label": "Group Structure", "pos": (-5, 4)},
                'I1_2': {"label": "Two Groups", "pos": (-7.5, 3)},
                'I1_M': {"label": "Multiple Groups", "pos": (-3, 3)},

                # Parametric - Two Groups
                'J1_INDEP': {"label": "Independent\nSamples", "pos": (-8.5, 2)},
                'J1_DEP': {"label": "Dependent\nSamples", "pos": (-6.5, 2)},
                'K1_2_IND': {"label": "Independent t-test", "pos": (-8.5, 1)},
                'K1_2_DEP': {"label": "Paired t-test", "pos": (-6.5, 1)},

                # Parametric - Multiple groups
                'J1_M_SPH': {"label": k1_m_sph_label, "pos": (-3, 2)},
                'K1_M_SPH': {"label": f"{'Has' if has_sphericity else 'No'} Sphericity", "pos": (-3, 1)},

                # ANOVA types (FIXED: Better organization)
                'L1_M_IND': {"label": "Independent\nGroups", "pos": (-4, 0)},
                'L1_M_DEP': {"label": "Repeated\nMeasures", "pos": (-2.5, 0)},
                'L1_M_MIX': {"label": "Mixed\nDesign", "pos": (-1, 0)},

                # Specific ANOVA tests
                'M1_M_IND_ONE': {"label": "One-way ANOVA", "pos": (-5, -1)},
                'M1_M_IND_TWO': {"label": "Two-way ANOVA", "pos": (-3, -1)},
                'M1_M_DEP': {"label": "RM ANOVA", "pos": (-1.5, -1)},
                'M1_M_MIX': {"label": "Mixed ANOVA", "pos": (0, -1)},

                'N1_CORR': {"label": "Apply Sphericity\nCorrection", "pos": (-1.5, 1)},

                # Parametric post-hoc tests (FIXED: Added more options)
                'O1_PH': {"label": "Post-hoc Tests", "pos": (-3, -3)},
                'P1_PH_TK': {"label": "Tukey HSD", "pos": (-5, -4)},
                'P1_PH_DN': {"label": "Dunnett Test", "pos": (-3, -4)},
                'P1_PH_SD': {"label": "Pairwise t-test\n(Holm-Sidak-corrected)", "pos": (-6.5, -4)},

                # Non-parametric branch
                'G2': {"label": "Non-parametric Test", "pos": (5, 5)},
                'H2': {"label": "Group Structure", "pos": (5, 4)},
                'I2_2': {"label": "Two Groups", "pos": (3, 3)},
                'I2_M': {"label": "Multiple Groups", "pos": (7.75, 3)},

                # Non-parametric - Two groups
                'J2_INDEP': {"label": "Independent\nSamples", "pos": (2, 2)},
                'J2_DEP': {"label": "Dependent\nSamples", "pos": (4, 2)},
                'K2_2_IND': {"label": "Mann-Whitney U", "pos": (2, 1)},
                'K2_2_DEP': {"label": "Wilcoxon\nSigned-Rank", "pos": (4, 1)},

                # Non-parametric - Multiple groups (FIXED: Better structure)
                'J2_M_INDEP': {"label": "Independent\nSamples", "pos": (6.25, 2)},
                'J2_M_DEP': {"label": "Dependent\nSamples", "pos": (9.25, 2)},
                
                # Tests under independent samples
                'K2_M_IND': {"label": "Kruskal-Wallis", "pos": (5.5, 1)},
                'NP_M_IND': {"label": "Non-parametric\nTwo-Way ANOVA", "pos": (7, 1)},
                
                # Tests under dependent samples
                'NP_M_DEP': {"label": "Non-parametric\nRM ANOVA", "pos": (8.75, 1)},
                'NP_M_MIX': {"label": "Non-parametric\nMixed ANOVA", "pos": (10.24, 1)},
                
                # Post-hoc nodes for non-parametric branch
                'L2_PH': {"label": "Post-hoc Tests", "pos": (7.75, -3)},
                'M2_PH_DU': {"label": "Dunn Test", "pos": (6.25, -4)},
                'M2_PH_MWU': {"label": "Pairwise Mann-Whitney-U\n(Sidak-corrected)", "pos": (9.25, -4)},
                'NP_PH_WILC': {"label": "Pairwise Wilcoxon\n(Bonferroni-corrected)", "pos": (8, -4)},
            }

            # Add nodes to graph
            for node_id, info in nodes_info.items():
                G.add_node(node_id, label=info["label"], pos=info["pos"])

            # Define edges with more detailed paths (FIXED: Added missing connections)
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

                # Parametric - Multiple groups
                ('I1_M', 'J1_M_SPH'),
                ('J1_M_SPH', 'K1_M_SPH'),
                ('K1_M_SPH', 'L1_M_IND'),
                ('K1_M_SPH', 'L1_M_DEP'),
                ('K1_M_SPH', 'L1_M_MIX'),
                ('K1_M_SPH', 'N1_CORR'),  # Sphericity correction
                
                # ANOVA connections
                ('L1_M_IND', 'M1_M_IND_ONE'),  # Independent Groups -> One-way ANOVA
                ('L1_M_IND', 'M1_M_IND_TWO'),  # Independent Groups -> Two-way ANOVA
                ('L1_M_DEP', 'M1_M_DEP'),      # Repeated Measures -> RM ANOVA
                ('L1_M_MIX', 'M1_M_MIX'),      # Mixed Design -> Mixed ANOVA

                # Parametric - Post-hoc tests (FIXED: Better connections)
                ('M1_M_IND_ONE', 'O1_PH'),  # One-way ANOVA -> Post-hoc
                ('M1_M_IND_TWO', 'O1_PH'),  # Two-way ANOVA -> Post-hoc
                ('M1_M_DEP', 'O1_PH'),      # RM ANOVA -> Post-hoc
                ('M1_M_MIX', 'O1_PH'),      # Mixed ANOVA -> Post-hoc
                
                # Post-hoc test types
                ('O1_PH', 'P1_PH_TK'),     # Tukey
                ('O1_PH', 'P1_PH_DN'),     # Dunnett
                ('O1_PH', 'P1_PH_SD'),     # Holm-Sidak

                # Non-parametric branch - Group structure
                ('G2', 'H2'),
                ('H2', 'I2_2'),  # Two groups
                ('H2', 'I2_M'),  # Multiple groups

                # Non-parametric - Two groups
                ('I2_2', 'J2_INDEP'),
                ('I2_2', 'J2_DEP'),
                ('J2_INDEP', 'K2_2_IND'),  # Mann-Whitney U
                ('J2_DEP', 'K2_2_DEP'),    # Wilcoxon

                # Non-parametric - Multiple groups (FIXED: Better structure)
                ('I2_M', 'J2_M_INDEP'),    # Multiple groups -> Independent samples
                ('I2_M', 'J2_M_DEP'),      # Multiple groups -> Dependent samples
                
                # Independent samples tests
                ('J2_M_INDEP', 'K2_M_IND'),    # Independent -> Kruskal-Wallis
                ('J2_M_INDEP', 'NP_M_IND'),    # Independent -> Non-parametric Two-Way ANOVA
                
                # Dependent samples tests (FIXED: Added Friedman)
                ('J2_M_DEP', 'NP_M_DEP'),            # Dependent -> Non-parametric RM ANOVA
                ('J2_M_DEP', 'NP_M_MIX'),            # Dependent -> Non-parametric Mixed ANOVA

                # Post-hoc connections for non-parametric branch
                ('K2_M_IND', 'L2_PH'),             # Kruskal-Wallis -> Post-hoc
                ('NP_M_IND', 'L2_PH'),             # Non-parametric Two-Way ANOVA -> Post-hoc   # Friedman -> Post-hoc
                ('NP_M_DEP', 'L2_PH'),             # Non-parametric RM ANOVA -> Post-hoc
                ('NP_M_MIX', 'L2_PH'),             # Non-parametric Mixed ANOVA -> Post-hoc

                # From central post-hoc node to specific tests
                ('L2_PH', 'M2_PH_DU'),        # Post-hoc -> Dunn Test
                ('L2_PH', 'M2_PH_MWU'),  # Post-hoc -> Pairwise Mann-Whitney-U
                ('L2_PH', 'NP_PH_WILC'),  # Post-hoc -> Pairwise Wilcoxon
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
                    
                    # Multiple groups - determine if dependent or independent
                    if dependence_type == "dependent" or "rm" in test_name.lower() or "repeated" in test_name.lower() or "mixed" in test_name.lower():
                        highlighted.add(('I2_M', 'J2_M_DEP'))
                        
                        # Better explicit detection for dependent samples
                        if "mixed" in test_name.lower():
                            print(f"DEBUG TREE: Non-parametric Mixed ANOVA path: {test_name}")
                            highlighted.add(('J2_M_DEP', 'NP_M_MIX'))
                            highlighted.add(('NP_M_MIX', 'L2_PH'))
                            highlighted.add(('L2_PH', 'NP_PH_WILC'))  # Wilcoxon for mixed
                        elif "rm" in test_name.lower() or "repeated" in test_name.lower():
                            print(f"DEBUG TREE: Non-parametric RM ANOVA path: {test_name}")
                            highlighted.add(('J2_M_DEP', 'NP_M_DEP'))
                            highlighted.add(('NP_M_DEP', 'L2_PH'))
                            highlighted.add(('L2_PH', 'NP_PH_WILC'))  # Wilcoxon for RM
                        else:
                            print(f"DEBUG TREE: Non-parametric dependent path (default): {test_name}")
                            highlighted.add(('J2_M_DEP', 'NP_M_DEP'))  # Default to RM ANOVA
                            highlighted.add(('NP_M_DEP', 'L2_PH'))
                            highlighted.add(('L2_PH', 'NP_PH_WILC'))  # Wilcoxon post-hoc
                    else:
                        highlighted.add(('I2_M', 'J2_M_INDEP'))
                        
                        # Better detection for independent samples non-parametric tests
                        if "two-way" in test_name.lower() or "two way" in test_name.lower() or "two_way" in test_name.lower():
                            print(f"DEBUG TREE: Non-parametric Two-Way ANOVA path: {test_name}")
                            highlighted.add(('J2_M_INDEP', 'NP_M_IND'))  # Non-parametric Two-Way ANOVA
                            
                            # Only show post-hoc path if non-parametric test is available
                            if not nonparametric_disabled and "required but not available" not in test_name:
                                highlighted.add(('NP_M_IND', 'L2_PH'))
                                highlighted.add(('L2_PH', 'NP_PH_MWU'))  # Mann-Whitney U for Two-Way
                            else:
                                # Don't highlight post-hoc path if non-parametric alternative is not available
                                print(f"DEBUG TREE: Non-parametric Two-Way post-hoc not highlighted - alternative not available")
                                
                        else:
                            print(f"DEBUG TREE: Non-parametric One-Way ANOVA path (Kruskal-Wallis): {test_name}")
                            # Kruskal-Wallis for general independent samples (One-Way equivalent)
                            highlighted.add(('J2_M_INDEP', 'K2_M_IND'))
                            alpha = results.get("alpha", 0.05)
                            if p_value is not None and p_value < alpha:
                                highlighted.add(('K2_M_IND', 'L2_PH'))
                                highlighted.add(('L2_PH', 'M2_PH_DU'))  # Dunn test for Kruskal-Wallis
                        
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
                        
                    is_ttest = "t-test" in test_name.lower() or "t test" in test_name.lower()
                    if is_ttest:
                        print(f"DEBUG TREE: Detected t-test, skipping all post-hoc path highlighting")
                    else:
                        # Highlight post-hoc path for two-group parametric tests
                        highlighted.add(('K1_2_IND', 'O1_PH'))
                        highlighted.add(('K1_2_DEP', 'O1_PH'))
                else:
                    highlighted.add(('H1', 'I1_M'))
                    
                    # Better logic for advanced ANOVA detection with explicit prioritization
                    if "repeated" in test_name.lower() or ("rm" in test_name.lower() and "anova" in test_name.lower()):
                        # Repeated Measures ANOVA path
                        print(f"DEBUG TREE: Detected RM ANOVA: {test_name}")
                        highlighted.add(('I1_M', 'J1_M_SPH'))
                        highlighted.add(('J1_M_SPH', 'K1_M_SPH'))
                        highlighted.add(('K1_M_SPH', 'L1_M_DEP'))
                        highlighted.add(('L1_M_DEP', 'M1_M_DEP'))
                        # Only add post-hoc if ANOVA is significant:
                        alpha = results.get("alpha", 0.05)
                        if p_value is not None and p_value < alpha:
                            highlighted.add(('M1_M_DEP', 'O1_PH'))
                            
                    elif "mixed" in test_name.lower():
                        # Mixed ANOVA path
                        print(f"DEBUG TREE: Detected Mixed ANOVA: {test_name}")
                        highlighted.add(('I1_M', 'J1_M_SPH'))
                        highlighted.add(('J1_M_SPH', 'K1_M_SPH'))
                        highlighted.add(('K1_M_SPH', 'L1_M_MIX'))
                        highlighted.add(('L1_M_MIX', 'M1_M_MIX'))
                        # Only add post-hoc if Mixed-ANOVA is significant:
                        alpha = results.get("alpha", 0.05)
                        if p_value is not None and p_value < alpha:
                            highlighted.add(('M1_M_MIX', 'O1_PH'))
                            
                    elif "two-way" in test_name.lower() or "two way" in test_name.lower():
                        # Two-Way ANOVA path (explicit detection)
                        print(f"DEBUG TREE: Detected Two-Way ANOVA: {test_name}")
                        highlighted.add(('I1_M', 'J1_M_SPH'))
                        highlighted.add(('J1_M_SPH', 'K1_M_SPH'))
                        highlighted.add(('K1_M_SPH', 'L1_M_IND'))
                        highlighted.add(('L1_M_IND', 'M1_M_IND_TWO'))
                        alpha = results.get("alpha", 0.05)
                        if p_value is not None and p_value < alpha:
                            highlighted.add(('M1_M_IND_TWO', 'O1_PH'))
                            
                    else:
                        # One-Way ANOVA path (default for unspecified multiple group parametric tests)
                        print(f"DEBUG TREE: Detected One-Way ANOVA (default): {test_name}")
                        highlighted.add(('I1_M', 'J1_M_SPH'))
                        highlighted.add(('J1_M_SPH', 'K1_M_SPH'))
                        highlighted.add(('K1_M_SPH', 'L1_M_IND'))
                        highlighted.add(('L1_M_IND', 'M1_M_IND_ONE'))
                        alpha = results.get("alpha", 0.05)
                        if p_value is not None and p_value < alpha:
                            highlighted.add(('M1_M_IND_ONE', 'O1_PH'))

                    # Post-hoc test types only if ANOVA was significant:
                    alpha = results.get("alpha", 0.05)
                    if p_value is not None and p_value < alpha:
                        # Call helper function to determine which post-hoc path to highlight
                        DecisionTreeVisualizer._highlight_posthoc_path(results, highlighted)
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
            plt.figure(figsize=(24, 20))
            node_labels = nx.get_node_attributes(G, 'label')

            # --- Shape logic for nodes ---
            always_square_labels = {
                "Start", "Check Assumptions", "Assumptions: Met", "Assumptions: Not Met",
                "Parametric Test", "Non-parametric Test", "Group Structure",
                "Two Groups", "Multiple Groups", "Independent Samples",
                "Dependent Samples", "Sphericity Check", "Repeated Measures",
                "Mixed Design", "Post-hoc Tests",
                "Independent Groups"  # <-- add this line
            }

            def is_always_square(node_id):
                label = nodes_info[node_id]["label"].replace('\n', ' ').replace(':', ': ').replace('  ', ' ')
                for square_label in always_square_labels:
                    if square_label in label:
                        return True
                return False

            # Draw nodes
            square_nodes = [n for n in G.nodes() if is_always_square(n)]
            round_nodes = [n for n in G.nodes() if n not in square_nodes]

            # Highlighted nodes for color
            highlighted_nodes = set()
            for u, v in highlighted:
                highlighted_nodes.add(u)
                highlighted_nodes.add(v)

            # Draw all nodes: highlighted ones in color, others in white
            nx.draw_networkx_nodes(G, pos, nodelist=[n for n in square_nodes if n in highlighted_nodes], node_size=3500,
                                    node_color='#ffcccc', edgecolors='black', linewidths=1.5, node_shape='s')
            nx.draw_networkx_nodes(G, pos, nodelist=[n for n in square_nodes if n not in highlighted_nodes], node_size=3500,
                                    node_color='white', edgecolors='black', linewidths=1.5, node_shape='s')
            nx.draw_networkx_nodes(G, pos, nodelist=[n for n in round_nodes if n in highlighted_nodes], node_size=3500,
                                    node_color='#ffcccc', edgecolors='black', linewidths=1.5, node_shape='o')
            nx.draw_networkx_nodes(G, pos, nodelist=[n for n in round_nodes if n not in highlighted_nodes], node_size=3500,
                                    node_color='white', edgecolors='black', linewidths=1.5, node_shape='o')

            # Draw all edges with different styles
            nx.draw_networkx_edges(G, pos, edgelist=highlighted_edges, width=4, edge_color='red')
            nx.draw_networkx_edges(G, pos, edgelist=transformation_edges, width=4, edge_color='blue', style='dashed')
            nx.draw_networkx_edges(G, pos, edgelist=check_edges, width=2, edge_color='gray', style='dotted')
            nx.draw_networkx_edges(G, pos, edgelist=regular_edges, width=1, edge_color='black', style='solid')

            # Draw node labels with background boxes
            nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=14,
                    font_family='sans-serif', font_weight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
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
                    fontsize=14, frameon=True, facecolor='white', edgecolor='black',
                    framealpha=0.9, shadow=True)

            # Remove axis
            plt.axis('off')
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            
            # Set figure size early and maintain it
            fig = plt.gcf()
            fig.set_size_inches(24, 20, forward=True)
            
            print(f"DEBUG: About to save figure to {'temp file' if not output_path else output_path}")
            print(f"DEBUG: Using matplotlib backend: {plt.get_backend()}")
            print(f"DEBUG: Graph has {len(G.nodes())} nodes and {len(G.edges())} edges")
            print(f"DEBUG: Figure size: {plt.gcf().get_size_inches()}")
            print(f"DEBUG: Number of highlighted edges: {len(highlighted_edges)}")

            # Save the image if path provided
            if output_path:
                output_file = f"{output_path}.png"
                plt.savefig(output_file, format="png", dpi=300, transparent=False, 
                        facecolor='white', bbox_inches='tight')
                plt.close('all')
                
                print(f"DEBUG: Image saved as {output_file}")
                return output_file
            else:
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    plt.savefig(tmp.name, format="png", dpi=300, transparent=False,
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
            temp_dir = tempfile.gettempdir()
            temp_filename = f"decision_tree_{int(time.time())}.png"
            temp_path = os.path.join(temp_dir, temp_filename)

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
    def _highlight_posthoc_path(results, highlighted):
        """
        Helper method to determine which post-hoc test path to highlight based on the results.
        This centralizes the logic to avoid duplication and conflicts.
        
        For One-Way ANOVA: Shows multiple options (user can choose)
        For Two-Way ANOVA, RM ANOVA, Mixed ANOVA: Shows the actually performed test
        """
        test_name = results.get("test", "").lower()
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
            highlighted.add(('O1_PH', 'P1_PH_SD'))  # Holm-Sidak
            return
        elif is_advanced_anova and not posthoc_test:
            # For Advanced ANOVAs with no specific post-hoc performed: show only Tukey and Pairwise (no Dunnett)
            print(f"DEBUG TREE: Advanced ANOVA detected - showing only Tukey and Pairwise post-hoc options")
            highlighted.add(('O1_PH', 'P1_PH_TK'))  # Tukey
            highlighted.add(('O1_PH', 'P1_PH_SD'))  # Holm-Sidak (Pairwise t-tests)
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
                print(f"DEBUG TREE: Highlighting Holm-Sidak path for posthoc: '{posthoc_test}'")
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
                print(f"DEBUG TREE: Unknown post-hoc test '{posthoc_test}', defaulting to Holm-Sidak")
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
                    print(f"DEBUG TREE: Inferred Holm-Sidak from pairwise test")
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
                    print(f"DEBUG TREE: Inferred Holm-Sidak from pairwise test with correction method")
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
                    highlighted.add(('O1_PH', 'P1_PH_SD'))  # Holm-Sidak
                elif is_advanced_anova:
                    print(f"DEBUG TREE: Advanced ANOVA - showing only Tukey and Pairwise post-hoc options")
                    highlighted.add(('O1_PH', 'P1_PH_TK'))  # Tukey
                    highlighted.add(('O1_PH', 'P1_PH_SD'))  # Holm-Sidak (Pairwise t-tests)
                else:
                    print(f"DEBUG TREE: Using default Tukey for other test types")
                    highlighted.add(('O1_PH', 'P1_PH_TK'))

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

# Zum Testen einfach aufrufen:
if __name__ == "__main__":
    test_decision_tree_visualization()
    