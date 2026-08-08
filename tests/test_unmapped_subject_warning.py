"""Guards the advisory warning for a mixed design that silently ran as Two-Way.

When two factors are mapped without a Subject ID, but the sheet contains a column
that looks like repeated-measures subject IDs (each subject constant on one factor,
spanning multiple levels of the other), the analysis runs as a between-subjects
Two-Way ANOVA -- ignoring the within-subject structure. The pipeline now detects
that column and warns the user to map it as Subject ID for a Mixed ANOVA.
"""
import pandas as pd

from autopilot.statistical_analyzer_autopilot_pipeline import (
    _detect_unmapped_repeated_subject,
    _ap_build_analysis_context,
)


def _mixed_df():
    # 4 subjects per between group, each measured at two timepoints (within).
    rows = []
    for grp, subs in (("WT", ["S1", "S2", "S3", "S4"]), ("KO", ["S5", "S6", "S7", "S8"])):
        for s in subs:
            for t in ("0h", "2h"):
                rows.append({"Subject": s, "BetweenGrp": grp, "Timepoint": t,
                             "Value": 10.0 + len(rows) * 0.3})
    return pd.DataFrame(rows)


# --- pure detector ----------------------------------------------------------

def test_detects_unmapped_subject_and_within_factor():
    df = _mixed_df()
    hit = _detect_unmapped_repeated_subject(
        df, "BetweenGrp", "Timepoint",
        mapped_cols={"Value", "BetweenGrp", "Timepoint"},
    )
    assert hit == ("Subject", "Timepoint")


def test_no_subject_column_returns_none():
    # Plain between-subjects two-way: no ID-like column present.
    df = _mixed_df().drop(columns=["Subject"])
    hit = _detect_unmapped_repeated_subject(
        df, "BetweenGrp", "Timepoint",
        mapped_cols={"Value", "BetweenGrp", "Timepoint"},
    )
    assert hit is None


def test_ordinary_categorical_column_is_not_flagged():
    # An unmapped 2-value column whose values cut across BOTH factors (neither
    # nested) must NOT be mistaken for subject IDs.
    df = _mixed_df()
    df["Batch"] = ["A", "B"] * (len(df) // 2)  # alternates, nests nothing
    hit = _detect_unmapped_repeated_subject(
        df, "BetweenGrp", "Timepoint",
        mapped_cols={"Value", "BetweenGrp", "Timepoint", "Subject"},  # Subject mapped away
    )
    assert hit is None


# --- wiring into the context builder ---------------------------------------

class _FakeBucket:
    def __init__(self, columns=None):
        self._columns = list(columns or [])

    def get_assigned_columns(self):
        return list(self._columns)


class _FakeToggle:
    def isChecked(self):
        return False


class _FakeApp:
    def __init__(self, df):
        self.df = df
        self.dv_bucket = _FakeBucket(["Value"])
        self.factor1_bucket = _FakeBucket(["BetweenGrp"])
        self.factor2_bucket = _FakeBucket(["Timepoint"])
        self.subject_bucket = _FakeBucket([])          # Subject deliberately unmapped
        self.covariates_bucket = _FakeBucket([])
        self.multi_mode_button = _FakeToggle()
        self.analysis_selected_groups = []
        self.warn_calls = []

    def _warn_unmapped_subject(self, subject_col, within_factor):
        self.warn_calls.append((subject_col, within_factor))


def test_two_way_without_mapped_subject_warns():
    app = _FakeApp(_mixed_df())
    ctx = _ap_build_analysis_context(app)
    assert ctx["inferred_test"] == "two_way_anova"
    assert app.warn_calls == [("Subject", "Timepoint")]


def test_two_way_without_id_column_does_not_warn():
    app = _FakeApp(_mixed_df().drop(columns=["Subject"]))
    ctx = _ap_build_analysis_context(app)
    assert ctx["inferred_test"] == "two_way_anova"
    assert app.warn_calls == []
