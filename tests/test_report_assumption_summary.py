"""Sphericity-correction note in the HTML assumption summary must read the
epsilon value from where statisticaltester.py actually writes it
(results["correction_used"] top-level, results["sphericity_corrections"][...]
["epsilon"] nested) — not from results["sphericity_test"], which never
contains a correction/epsilon key.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from export.report_summaries import _SummariesMixin


def _rm_anova_results_with_sphericity_violation():
    """Shaped exactly like StatisticalTester._perform_comprehensive_sphericity_test
    + _apply_sphericity_corrections actually produce it (statisticaltester.py:2624-2874).
    """
    return {
        "model_type": "RMANOVA",
        "sphericity_test": {
            "test_name": "Mauchly's Test for Sphericity",
            "W": 0.72,
            "p_value": 0.01,
            "sphericity_assumed": False,
            "d": 2,
            "interpretation": "Sphericity violated",
        },
        "sphericity_corrections": {
            "needed": True,
            "greenhouse_geisser": {
                "epsilon": 0.6543,
                "corrected_df1": 1.31,
                "corrected_df2": 13.1,
                "p_value": 0.02,
                "conservative": True,
                "description": "Conservative correction for sphericity violation",
            },
        },
        "corrected_p_value": 0.02,
        "correction_used": "Greenhouse-Geisser (ε = 0.654)",
        "final_p_value": 0.02,
    }


def test_sphericity_note_includes_epsilon_from_real_backend_shape():
    result = _SummariesMixin._build_assumption_summary(
        _rm_anova_results_with_sphericity_violation()
    )
    note = result["sphericity_correction_note"]
    assert note is not None
    assert "Greenhouse-Geisser" in note
    assert "0.6543" in note, (
        f"epsilon missing from note (key-path bug not fixed): {note!r}"
    )
