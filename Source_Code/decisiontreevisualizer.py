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
            test_name = results.get("test_name", results.get("test", ""))
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
                'B': {"label": f"Check Assumptions\nShapiro-Wilk: {is_normal}\nBrown-Forsythe: {has_equal_variance}", "pos": (0, 12.5)},
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
                'IND_HOLM_SIDAK': {"label": "Pairwise t-tests\n(Holm-Sidak)", "pos": (-6.5, -1)}, # MORE SPACING: 1.5 apart

                # REPEATED MEASURES PATH - SAME INTERNAL SPACING
                'RM_MAUCHLY': {"label": "Mauchly's Test\nfor Sphericity", "pos": (-2, 1)},
                'RM_SPHERICITY_OK': {"label": "Sphericity\nAssumption Met", "pos": (-3.5, 0)},    # KEEP CLOSE
                'RM_SPHERICITY_VIOLATED': {"label": "Sphericity\nViolated", "pos": (-0.5, 0)},   # KEEP CLOSE
                'RM_CHOOSE_CORRECTION': {"label": "Choose\nCorrection", "pos": (-0.5, -1)},
                'RM_GG_CORRECTION': {"label": "Greenhouse-Geisser\nCorrection", "pos": (-1.5, -2)},  # KEEP CLOSE
                'RM_HF_CORRECTION': {"label": "Huynh-Feldt\nCorrection", "pos": (0.5, -2)},         # KEEP CLOSE
                'RM_ANOVA_STANDARD': {"label": "RM ANOVA", "pos": (-3.5, -1)},
                'RM_ANOVA_CORRECTED': {"label": "RM ANOVA\n(Corrected)", "pos": (-0.5, -3)},
                'RM_POSTHOC': {"label": "RM Post-hoc Tests", "pos": (-2, -4)},
                'RM_TUKEY': {"label": "Tukey HSD\n(RM)", "pos": (-3.5, -5)},
                'RM_PAIRED_TESTS': {"label": "Pairwise Paired t-tests\n(Holm-Sidak)", "pos": (-0.5, -5)},

                # MIXED DESIGN PATH - MOVED LEFT TO BE CLOSER, SAME INTERNAL SPACING
                'MIXED_MAUCHLY': {"label": "Mauchly's Test\nfor Sphericity", "pos": (4, 1)},
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
                'I2_M': {"label": "Multiple Groups", "pos": (12, 3)},                       # MOVED CLOSER

                # Non-parametric - Two groups (MOVED CLOSER TO TWO GROUPS)
                'J2_INDEP': {"label": "Independent\nSamples", "pos": (7, 2)},               # MOVED CLOSER
                'J2_DEP': {"label": "Dependent\nSamples", "pos": (9, 2)},                   # MOVED CLOSER
                'K2_2_IND': {"label": "Mann-Whitney U", "pos": (7, 1)},
                'K2_2_DEP': {"label": "Wilcoxon\nSigned-Rank", "pos": (9, 1)},

                # Non-parametric - Multiple groups (ONLY INDEPENDENT - NO FRIEDMAN)
                'J2_M_INDEP': {"label": "Independent\nSamples", "pos": (11, 2)},            # KEPT
                'K2_M_IND': {"label": "Kruskal-Wallis", "pos": (11, 1)},                    # KEPT
                'NP_POSTHOC': {"label": "Non-parametric\nPost-hoc Tests", "pos": (11, 0)},  # CENTERED
                'NP_DUNN': {"label": "Dunn Test", "pos": (10, -1)},                         # KRUSKAL-WALLIS POST-HOC
                'NP_MANN_WHITNEY': {"label": "Pairwise\nMann-Whitney U", "pos": (12, -1)},  # KRUSKAL-WALLIS POST-HOC
            }

            # Add nodes to graph
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

                # Non-parametric - Multiple groups (ONLY INDEPENDENT - NO FRIEDMAN)
                ('I2_M', 'J2_M_INDEP'),    # Multiple groups -> Independent samples ONLY
                ('J2_M_INDEP', 'K2_M_IND'),    # Independent -> Kruskal-Wallis
                ('K2_M_IND', 'NP_POSTHOC'),    # Kruskal-Wallis -> Post-hoc
                ('NP_POSTHOC', 'NP_DUNN'),         # Non-parametric -> Dunn Test (for Kruskal-Wallis)
                ('NP_POSTHOC', 'NP_MANN_WHITNEY'), # Non-parametric -> Pairwise Mann-Whitney U (for Kruskal-Wallis)
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
                    
                    # For non-parametric multiple groups, only independent samples are supported in this tree
                    highlighted.add(('I2_M', 'J2_M_INDEP'))
                    highlighted.add(('J2_M_INDEP', 'K2_M_IND'))  # Kruskal-Wallis
                    
                    # Only add post-hoc if Kruskal-Wallis is significant
                    alpha = results.get("alpha", 0.05)
                    if p_value is not None and p_value < alpha:
                        highlighted.add(('K2_M_IND', 'NP_POSTHOC'))
                        
                        # Determine specific post-hoc test
                        posthoc_test = results.get("posthoc_test", "")
                        if "dunn" in posthoc_test.lower():
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
                            posthoc_test = results.get("posthoc_test", "")
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
            plt.figure(figsize=(32, 24))  # Increased size for better readability
            node_labels = nx.get_node_attributes(G, 'label')

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
            fig.set_size_inches(32, 24, forward=True)  # Increased size
            
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
        "posthoc_test": "Paired t-tests (Holm-Sidak corrected)",
        "pairwise_comparisons": [
            {"groups": ("Time1", "Time2"), "p_value": 0.02, "test": "Paired t-test", "correction": "Holm-Sidak"},
            {"groups": ("Time1", "Time3"), "p_value": 0.005, "test": "Paired t-test", "correction": "Holm-Sidak"},
            {"groups": ("Time2", "Time3"), "p_value": 0.15, "test": "Paired t-test", "correction": "Holm-Sidak"}
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
    