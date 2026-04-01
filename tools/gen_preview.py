"""
Generate test_report_preview.html using real-looking WT/ntc/keap1 data.
Run from: BioMedStatX/ directory
    python tools/gen_preview.py
"""
import sys
import os

os.environ.setdefault("MPLBACKEND", "Agg")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from html_exporter import HTMLExporter

results = {
    "test_name": "One-Way ANOVA",
    "test": "One-Way ANOVA",
    "model_type": "ANOVA",
    "test_recommendation": "parametric",
    "recommendation": "parametric",
    "transformation": "None",
    "p_value": 0.000056,
    "f_statistic": 16.94,
    "effect_size": 0.6174,
    "effect_size_type": "eta_squared",
    "alpha": 0.05,
    "dependent": False,
    "groups": ["WT", "ntc", "keap1"],
    "raw_data": {
        "WT":    [2.1, 1.9, 2.3, 2.0, 2.2, 1.8, 2.4, 2.1],
        "ntc":   [2.0, 2.2, 1.9, 2.1, 2.3, 2.0, 1.8, 2.2],
        "keap1": [4.8, 5.2, 4.5, 5.0, 4.9, 5.3, 4.7, 5.1],
    },
    "descriptive_stats": {
        "means": {"WT": 2.1, "ntc": 2.0625, "keap1": 4.9375},
        "std":   {"WT": 0.19, "ntc": 0.17, "keap1": 0.26},
        "n":     {"WT": 8, "ntc": 8, "keap1": 8},
        "groups": ["WT", "ntc", "keap1"],
    },
    "normality_tests": {
        "WT":    {"statistic": 0.961, "p_value": 0.812, "is_normal": True},
        "ntc":   {"statistic": 0.955, "p_value": 0.756, "is_normal": True},
        "keap1": {"statistic": 0.967, "p_value": 0.873, "is_normal": True},
    },
    "variance_test": {
        "statistic": 1.23,
        "p_value": 0.31,
        "equal_variance": True,
        "test_name": "Levene",
    },
    "pairwise_comparisons": [
        {
            "comparison": "WT vs ntc",
            "group1": "WT",
            "group2": "ntc",
            "p_value": 0.847,
            "significant": False,
            "mean_diff": 0.0375,
        },
        {
            "comparison": "WT vs keap1",
            "group1": "WT",
            "group2": "keap1",
            "p_value": 0.00012,
            "significant": True,
            "mean_diff": -2.8375,
        },
        {
            "comparison": "ntc vs keap1",
            "group1": "ntc",
            "group2": "keap1",
            "p_value": 0.00018,
            "significant": True,
            "mean_diff": -2.875,
        },
    ],
    "posthoc_test": "Tukey HSD",
    "test_info": {
        "normality_tests": {
            "WT":    {"statistic": 0.961, "p_value": 0.812, "is_normal": True},
            "ntc":   {"statistic": 0.955, "p_value": 0.756, "is_normal": True},
            "keap1": {"statistic": 0.967, "p_value": 0.873, "is_normal": True},
        },
        "variance_test": {
            "statistic": 1.23,
            "p_value": 0.31,
            "equal_variance": True,
        },
    },
}

out = HTMLExporter.export_results_to_html(
    results,
    os.path.join(os.path.dirname(__file__), "..", "docs", "test_report_preview.html"),
    analysis_log=[
        "One-Way ANOVA performed (parametric, independent groups).",
        "Shapiro-Wilk normality: all groups passed.",
        "Levene variance test: equal variances (p=0.31).",
        "Post-hoc: Tukey HSD — WT vs keap1 p<0.001, ntc vs keap1 p<0.001.",
    ],
)
print("Written to:", out)
