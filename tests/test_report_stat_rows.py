"""RTE row-label extraction in the Brunner-Langer/ATS branch of
_build_statistical_rows must surface a missing between_group/within_level key
loudly (log warning) instead of silently substituting a blank label — matches
this session's "surface the failure" paradigm from Sprint 1/2.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from export.report_stat_rows import _StatRowsMixin


def _brunner_langer_results(rte_rows):
    return {
        "model_type": "BrunnerLangerATS",
        "RTE": pd.DataFrame(rte_rows),
    }


def test_rte_row_with_missing_key_logs_warning_not_silent_blank(caplog):
    results = _brunner_langer_results([
        {"within_level": "T0", "RTE": 0.62, "n": 12},  # missing between_group
    ])
    with caplog.at_level("WARNING"):
        rows = _StatRowsMixin._build_statistical_rows(results)
    rte_row_labels = [r["label"] for r in rows if r["label"].startswith("RTE:")]
    assert len(rte_row_labels) == 1
    assert any("missing" in rec.message.lower() for rec in caplog.records), (
        "a missing RTE key must be logged loudly, not silently substituted"
    )


def test_rte_row_with_all_keys_present_renders_normally():
    results = _brunner_langer_results([
        {"between_group": "drug", "within_level": "T0", "RTE": 0.62, "n": 12},
    ])
    rows = _StatRowsMixin._build_statistical_rows(results)
    rte_row_labels = [r["label"] for r in rows if r["label"].startswith("RTE:")]
    assert rte_row_labels == ["RTE: drug / T0"]
