"""A measurement column must not be mistaken for a subject ID.

``_looks_like_subject`` matched its keywords as bare substrings, and one of them
-- ``"id"`` -- is two characters long. It therefore hid inside a long list of
perfectly ordinary biomedical measurement names: lipid, peptid, humidity,
acidity, oxidation, rigidity, solid, fluid.

Two things followed from that, and neither told the user anything.

The column was struck from the numeric candidates in
``_ap_apply_mapping_heuristics`` before the DV was chosen, while the
"does this subject span more than one factor level" guard further down
correctly refused to assign it as a subject. The column then belonged nowhere:
not the DV bucket, not a factor, not the subject -- it simply vanished from the
mapping, and the user was left with an empty DV bucket and no explanation.

And ``_detect_wide_format`` requires exactly one subject candidate, so a wide
paired table whose measurement columns happened to carry such a name produced
several candidates, failed the check, and was never pivoted -- costing the user
the paired analysis entirely.

The long keywords stay substring matches; no ordinary column name contains
"subject" or "patient" by accident. Only "id" is matched as a word.
"""

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from autopilot.statistical_analyzer_autopilot_ui import (
    _detect_wide_format,
    _looks_like_subject,
)

# Names a lab actually gives a measurement column. Every one of them contains
# the letters "id"; none of them is a subject identifier.
MEASUREMENT_NAMES = [
    "Lipid", "Peptid", "Humidity", "Acidity", "Oxidation",
    "Rigidity", "Solid_fraction", "Fluid_volume", "Lipid_Pre",
]

# Names that genuinely identify a subject, in the spellings the app has to keep
# recognising: separators, camelCase, bare, German.
SUBJECT_NAMES = [
    "ID", "id", "PatientID", "Patient_ID", "Subject_Id", "SubjectID",
    "Tier-ID", "ProbandenID", "Mouse ID", "id_nr", "Animal_ID", "SampleID",
]


@pytest.fixture
def measurements():
    """High-uniqueness numeric values -- the shape that made the old code
    accept these columns once the name had matched."""
    return pd.Series(np.random.default_rng(3).normal(10, 2, 24))


@pytest.fixture
def identifiers():
    return pd.Series([f"S{i:02d}" for i in range(24)])


@pytest.mark.parametrize("name", MEASUREMENT_NAMES)
def test_a_measurement_column_is_not_a_subject_id(name, measurements):
    assert _looks_like_subject(name, measurements) is False


@pytest.mark.parametrize("name", SUBJECT_NAMES)
def test_a_real_identifier_is_still_recognised(name, identifiers):
    assert _looks_like_subject(name, identifiers) is True


def test_the_long_keywords_are_still_matched_as_substrings(identifiers):
    """Only "id" is short enough to collide, so only "id" gets the stricter
    rule. Run-together spellings of the long keywords must keep working."""
    for name in ("subjectnr", "patientnummer", "animalno", "mousetag", "subjekt"):
        assert _looks_like_subject(name, identifiers) is True, name


def _wide(pre, post, n=10):
    rng = np.random.default_rng(9)
    return pd.DataFrame({
        "Subject": [f"S{i:02d}" for i in range(n)],
        pre: rng.normal(10, 2, n),
        post: rng.normal(13, 2, n),
    })


@pytest.mark.parametrize("pre,post", [
    ("Pre", "Post"),
    ("Baseline", "Followup"),
    ("Lipid_Pre", "Lipid_Post"),
    ("Peptid_T0", "Peptid_T1"),
])
def test_a_wide_paired_table_is_pivoted_whatever_the_columns_are_called(pre, post):
    detected = _detect_wide_format(_wide(pre, post))

    assert detected is not None, f"{pre}/{post} was not recognised as wide format"
    assert detected["subject_col"] == "Subject"
    assert sorted(detected["value_cols"]) == sorted([pre, post])


# --- the consequence the user actually sees -------------------------------------


@pytest.fixture(scope="module")
def qapp():
    from PyQt5.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def window(qapp):
    from PyQt5.QtCore import QSettings
    from autopilot.statistical_analyzer_autopilot_pipeline import _current_app_version

    # Pin onboarding so the first-run welcome modal is not offered offscreen.
    QSettings("BioMedStatX", "BioMedStatX").setValue(
        "onboarding/completed_version", _current_app_version())

    from analysis.statistical_analyzer import StatisticalAnalyzerApp
    win = StatisticalAnalyzerApp()
    yield win
    win.close()


@pytest.mark.parametrize("dv_name", ["Value", "Lipid", "Peptid", "Humidity"])
def test_the_measurement_column_reaches_the_dv_bucket(window, tmp_path, dv_name):
    """End to end through the real window: load a two-column sheet and ask what
    the mapping heuristics did with it."""
    rng = np.random.default_rng(5)
    df = pd.DataFrame({
        "Group": ["Ctrl"] * 12 + ["Treat"] * 12,
        dv_name: np.concatenate([rng.normal(10, 2, 12), rng.normal(13, 2, 12)]),
    })
    book = tmp_path / f"{dv_name}.xlsx"
    with pd.ExcelWriter(book) as writer:
        df.to_excel(writer, sheet_name="Sheet1", index=False)

    window.file_path = str(book)
    window.load_file()

    assert window.dv_bucket.get_assigned_columns() == [dv_name], \
        f"{dv_name} did not reach the DV bucket"
    assert window.factor1_bucket.get_assigned_columns() == ["Group"]
    assert window.subject_bucket.get_assigned_columns() == []
