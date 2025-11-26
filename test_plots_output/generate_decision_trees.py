#!/usr/bin/env python3
"""
Script to generate a decision tree visualization using the DecisionTreeVisualizer class
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from decisiontreevisualizer import DecisionTreeVisualizer

def create_sample_decision_tree():
    """Create a sample decision tree with typical ANOVA results"""
    
    # Example results for a One-way ANOVA with significant result and Tukey post-hoc
    results = {
        "test_name": "One-way ANOVA",
        "test": "One-way ANOVA", 
        "test_recommendation": "parametric",
        "recommendation": "parametric",
        "transformation": "None",
        "p_value": 0.002,
        "alpha": 0.05,
        "groups": ["Group A", "Group B", "Group C"],
        "n_groups": 3,
        "raw_data": {
            "Group A": [1, 2, 3, 4, 5],
            "Group B": [3, 4, 5, 6, 7], 
            "Group C": [5, 6, 7, 8, 9]
        },
        "descriptive_stats": {
            "groups": ["Group A", "Group B", "Group C"],
            "means": {"Group A": 3.0, "Group B": 5.0, "Group C": 7.0}
        },
        "normality_tests": {
            "Group A": {"is_normal": True, "p_value": 0.8},
            "Group B": {"is_normal": True, "p_value": 0.7},
            "Group C": {"is_normal": True, "p_value": 0.9},
            "all_data": {"is_normal": True, "p_value": 0.6}
        },
        "variance_test": {"equal_variance": True, "p_value": 0.4},
        "posthoc_test": "Tukey HSD",
        "pairwise_comparisons": [
            {"groups": ("Group A", "Group B"), "p_value": 0.01, "test": "Tukey HSD"},
            {"groups": ("Group A", "Group C"), "p_value": 0.001, "test": "Tukey HSD"},
            {"groups": ("Group B", "Group C"), "p_value": 0.20, "test": "Tukey HSD"}
        ]
    }
    
    print("Creating One-way ANOVA decision tree...")
    output_path = DecisionTreeVisualizer.visualize(results, output_path="one_way_anova_decision_tree")
    print(f"One-way ANOVA decision tree saved to: {output_path}")
    return output_path

def create_rm_anova_decision_tree():
    """Create a RM ANOVA decision tree with sphericity correction"""
    
    results = {
        "test_name": "Repeated Measures ANOVA",
        "test": "Repeated Measures ANOVA",
        "test_recommendation": "parametric", 
        "recommendation": "parametric",
        "transformation": "None",
        "p_value": 0.01,
        "alpha": 0.05,
        "groups": ["Time1", "Time2", "Time3", "Time4"],
        "n_groups": 4,
        "raw_data": {
            "Time1": [10, 12, 11, 13, 9],
            "Time2": [12, 14, 13, 15, 11],
            "Time3": [15, 17, 16, 18, 14], 
            "Time4": [18, 20, 19, 21, 17]
        },
        "descriptive_stats": {
            "groups": ["Time1", "Time2", "Time3", "Time4"],
            "means": {"Time1": 11.0, "Time2": 13.0, "Time3": 16.0, "Time4": 19.0}
        },
        "normality_tests": {
            "Time1": {"is_normal": True, "p_value": 0.7},
            "Time2": {"is_normal": True, "p_value": 0.8},
            "Time3": {"is_normal": True, "p_value": 0.6},
            "Time4": {"is_normal": True, "p_value": 0.9},
            "all_data": {"is_normal": True, "p_value": 0.5}
        },
        "variance_test": {"equal_variance": True, "p_value": 0.3},
        "sphericity_test": {
            "test_name": "Mauchly's Test for Sphericity",
            "W": 0.65,
            "p_value": 0.03,
            "has_sphericity": False,
            "sphericity_assumed": False
        },
        "correction_used": "Greenhouse-Geisser (ε = 0.72 ≤ 0.75)",
        "corrected_p_value": 0.018,
        "posthoc_test": "Pairwise Paired t-tests (Holm-Sidak corrected)",
        "pairwise_comparisons": [
            {"groups": ("Time1", "Time2"), "p_value": 0.02, "test": "Paired t-test", "correction": "Holm-Sidak"},
            {"groups": ("Time1", "Time3"), "p_value": 0.005, "test": "Paired t-test", "correction": "Holm-Sidak"},
            {"groups": ("Time2", "Time3"), "p_value": 0.15, "test": "Paired t-test", "correction": "Holm-Sidak"},
            {"groups": ("Time3", "Time4"), "p_value": 0.08, "test": "Paired t-test", "correction": "Holm-Sidak"}
        ]
    }
    
    print("Creating RM ANOVA decision tree...")
    output_path = DecisionTreeVisualizer.visualize(results, output_path="rm_anova_decision_tree")
    print(f"RM ANOVA decision tree saved to: {output_path}")
    return output_path

def create_welch_anova_decision_tree():
    """Create a Welch ANOVA decision tree for unequal variances"""
    
    results = {
        "test_name": "Welch's ANOVA",
        "test": "Welch's ANOVA", 
        "test_recommendation": "parametric",
        "recommendation": "parametric",
        "transformation": "None",
        "p_value": 0.008,
        "alpha": 0.05,
        "groups": ["Group A", "Group B", "Group C"],
        "n_groups": 3,
        "raw_data": {
            "Group A": [1, 2, 3, 4, 5],
            "Group B": [10, 12, 14, 16, 18],
            "Group C": [25, 30, 35, 40, 45]
        },
        "descriptive_stats": {
            "groups": ["Group A", "Group B", "Group C"],
            "means": {"Group A": 3.0, "Group B": 14.0, "Group C": 35.0}
        },
        "normality_tests": {
            "Group A": {"is_normal": True, "p_value": 0.6},
            "Group B": {"is_normal": True, "p_value": 0.7},
            "Group C": {"is_normal": True, "p_value": 0.8},
            "all_data": {"is_normal": True, "p_value": 0.5}
        },
        "variance_test": {"equal_variance": False, "p_value": 0.01},  # Unequal variances!
        "posthoc_test": "Dunnett T3",
        "pairwise_comparisons": [
            {"groups": ("Group A", "Group B"), "p_value": 0.005, "test": "Dunnett T3"},
            {"groups": ("Group A", "Group C"), "p_value": 0.001, "test": "Dunnett T3"}, 
            {"groups": ("Group B", "Group C"), "p_value": 0.01, "test": "Dunnett T3"}
        ]
    }
    
    print("Creating Welch ANOVA decision tree...")
    output_path = DecisionTreeVisualizer.visualize(results, output_path="welch_anova_decision_tree")
    print(f"Welch ANOVA decision tree saved to: {output_path}")
    return output_path

def create_nonparametric_decision_tree():
    """Create a non-parametric decision tree"""
    
    results = {
        "test_name": "Kruskal-Wallis",
        "test": "Kruskal-Wallis",
        "test_recommendation": "non-parametric",
        "recommendation": "non_parametric", 
        "transformation": "None",
        "p_value": 0.003,
        "alpha": 0.05,
        "groups": ["Group A", "Group B", "Group C"],
        "n_groups": 3,
        "raw_data": {
            "Group A": [1, 3, 5, 20, 25],  # Skewed data
            "Group B": [2, 4, 6, 30, 35],
            "Group C": [10, 12, 15, 40, 50]
        },
        "descriptive_stats": {
            "groups": ["Group A", "Group B", "Group C"],
            "means": {"Group A": 10.8, "Group B": 15.4, "Group C": 25.4}
        },
        "normality_tests": {
            "Group A": {"is_normal": False, "p_value": 0.02},  # Not normal
            "Group B": {"is_normal": False, "p_value": 0.03},
            "Group C": {"is_normal": False, "p_value": 0.01},
            "all_data": {"is_normal": False, "p_value": 0.005}
        },
        "variance_test": {"equal_variance": False, "p_value": 0.02},
        "posthoc_test": "Dunn Test",
        "pairwise_comparisons": [
            {"groups": ("Group A", "Group B"), "p_value": 0.08, "test": "Dunn"},
            {"groups": ("Group A", "Group C"), "p_value": 0.002, "test": "Dunn"},
            {"groups": ("Group B", "Group C"), "p_value": 0.01, "test": "Dunn"}
        ]
    }
    
    print("Creating Kruskal-Wallis decision tree...")
    output_path = DecisionTreeVisualizer.visualize(results, output_path="kruskal_wallis_decision_tree")
    print(f"Kruskal-Wallis decision tree saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    print("=== Decision Tree Visualizations ===")
    print()
    
    # Create different types of decision trees
    paths = []
    
    try:
        paths.append(create_sample_decision_tree())
        print()
        paths.append(create_rm_anova_decision_tree())
        print()
        paths.append(create_welch_anova_decision_tree())
        print()
        paths.append(create_nonparametric_decision_tree())
        print()
        
        print("=== Summary ===")
        print("Generated decision trees:")
        for i, path in enumerate(paths, 1):
            if path:
                print(f"{i}. {path}")
            else:
                print(f"{i}. Failed to generate")
                
    except Exception as e:
        print(f"Error creating decision trees: {e}")
        import traceback
        traceback.print_exc()