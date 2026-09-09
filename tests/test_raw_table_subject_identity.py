"""The raw-data table must identify observations, not merely number them.

It used to print a per-group row number. In an independent design that is
import order and says nothing. In a repeated-measures design it says something
false: two rows carrying the same number read as the same subject, and the
extraction does not guarantee that -- values are filtered per level in whatever
order the frame happens to hold. Where the design has subjects, the subject id
is carried through and printed instead.
"""

import pandas as pd

from analysis.statisticaltester import StatisticalTester
from export.report_summaries import _SummariesMixin

# Three subjects measured twice. The second block is stored in a different
# order, which is legal input and exactly what breaks a positional pairing.
SHUFFLED = pd.DataFrame([
    {"Subject": "S1", "Time": "0h", "Y": 10},
    {"Subject": "S2", "Time": "0h", "Y": 20},
    {"Subject": "S3", "Time": "0h", "Y": 30},
    {"Subject": "S3", "Time": "24h", "Y": 33},
    {"Subject": "S1", "Time": "24h", "Y": 11},
    {"Subject": "S2", "Time": "24h", "Y": 22},
])


def _table_for(df):
    raw, subjects = StatisticalTester._extract_raw_data_rm_anova(
        df, "Y", None, ["Time"], "Subject")
    return _SummariesMixin._build_raw_data_table(
        {"raw_data": raw, "raw_data_subjects": subjects})


def test_each_value_is_labelled_with_the_subject_it_came_from():
    table = _table_for(SHUFFLED)
    assert table["has_subjects"]
    by_subject = {}
    for row in table["rows"]:
        by_subject.setdefault(row["subject"], []).append(float(row["raw_value"]))
    # The real pairing, which a row number would have got wrong for all three.
    assert by_subject["S1"] == [10.0, 11.0]
    assert by_subject["S2"] == [20.0, 22.0]
    assert by_subject["S3"] == [30.0, 33.0]


def test_the_label_follows_the_value_not_the_position():
    """Row two of the second level is S1, not the second subject listed.

    Keyed by the bare level: every other extraction of a repeated-measures
    design uses it, and this extractor's "factor=level" was the lone outlier
    that stopped the two halves of the table pairing at all.
    """
    table = _table_for(SHUFFLED)
    second_level = [r for r in table["rows"] if r["group"] == "24h"]
    assert [r["subject"] for r in second_level] == ["S3", "S1", "S2"]
    assert [float(r["raw_value"]) for r in second_level] == [33.0, 11.0, 22.0]


def test_an_independent_design_gets_no_subject_column():
    table = _SummariesMixin._build_raw_data_table({"raw_data": {"A": [1, 2], "B": [3, 4]}})
    assert not table["has_subjects"]
    assert all("subject" not in row for row in table["rows"])
    # And no row number sneaks back in as a replacement.
    assert all("index" not in row for row in table["rows"])


def test_column_mode_keeps_its_row_number():
    """For correlation and regression a row genuinely is one observation."""
    table = _SummariesMixin._build_raw_data_table(
        {"raw_data_columns": {"x": [1.0, 2.0], "y": [3.0, 4.0]}})
    assert table["column_mode"]
    assert [row["index"] for row in table["rows"]] == [1, 2]


def test_a_between_only_extractor_reports_no_subjects():
    df = pd.DataFrame([
        {"FA": "A1", "FB": "B1", "Y": 1.0},
        {"FA": "A2", "FB": "B1", "Y": 2.0},
    ])
    raw = StatisticalTester._extract_raw_data_two_way_anova(df, "Y", ["FA", "FB"], None, None)
    assert isinstance(raw, dict), "between-only designs keep the plain mapping"
