"""The group axis must be ranked, not alphabetical.

``natural_order`` has existed and been correct for a long time, and six modules
use it. ``analysis_core`` applied it only under ``if not groups_to_use:`` -- and
the window always supplies groups, from ``_build_analysis_context``, where the
levels come out of ``_sorted_unique``, a plain ``sorted(key=str)``. So on the one
path a real user walks the ranking was skipped and the axis came out in
alphabetical order.

What that looks like in a report: ``KO`` drawn before ``WT``, so the control is
not the reference bar; and a timecourse rendered ``D0, D14, D21, D7``, with day
seven after day twenty-one. Both cases below are the ones the fuzzer produced.

The two cases are checked through the real window -- load a file, let the
mapping run, press the button, read the axis out of the report that lands on
disk -- because the window is exactly the caller that used to defeat the
ranking.
"""

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def qapp():
    from PyQt5.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def window(qapp):
    from PyQt5.QtCore import QSettings
    from PyQt5.QtWidgets import QMessageBox
    from autopilot.statistical_analyzer_autopilot_pipeline import _current_app_version

    for name in ("warning", "critical", "information", "question"):
        setattr(QMessageBox, name, staticmethod(lambda *a, **k: QMessageBox.Ok))
    QSettings("BioMedStatX", "BioMedStatX").setValue(
        "onboarding/completed_version", _current_app_version())

    from analysis.statistical_analyzer import StatisticalAnalyzerApp
    win = StatisticalAnalyzerApp()
    yield win
    win.close()


@pytest.fixture
def neutralized_dialogs(monkeypatch):
    """Answer the analysis dialogs, and put them back afterwards.

    ``_neutralize_dialogs`` patches ``QDialog.exec_`` and the ``UIDialogManager``
    statics in place with no teardown. That is right inside a fuzz worker, which
    exits moments later, and a leak inside a shared pytest process: left behind,
    this file would be deciding how unrelated tests answer their dialogs. Each
    attribute is restored through monkeypatch, so the neutralization lasts
    exactly one test.
    """
    from PyQt5.QtWidgets import QDialog
    from analysis.statisticaltester import UIDialogManager
    from ui.dialogs import comparison_selection_dialog as csd

    for attr in ("exec_", "exec"):
        monkeypatch.setattr(QDialog, attr, getattr(QDialog, attr), raising=False)
    for attr in ("select_transformation_dialog", "select_posthoc_test_dialog",
                 "select_nonparametric_posthoc_dialog",
                 "select_control_group_dialog", "select_custom_pairs_dialog"):
        monkeypatch.setattr(UIDialogManager, attr,
                            getattr(UIDialogManager, attr), raising=False)
    monkeypatch.setattr(csd, "ComparisonSelectionDialog",
                        csd.ComparisonSelectionDialog, raising=False)

    from fuzzing._worker import _neutralize_dialogs
    _neutralize_dialogs(5)


def _axis_order_after_running(window, tmp_path, levels, monkeypatch):
    """Load a sheet with these levels, run the analysis, return the axis order."""
    from PyQt5.QtWidgets import QFileDialog
    from fuzzing.html_oracles import load_report

    rng = np.random.default_rng(17)
    frame = pd.DataFrame({
        "Group": sum(([lv] * 8 for lv in levels), []),
        "Value": np.concatenate([rng.normal(10 + 2 * i, 1.5, 8)
                                 for i in range(len(levels))]),
    })
    book = tmp_path / "levels.xlsx"
    with pd.ExcelWriter(book) as writer:
        frame.to_excel(writer, sheet_name="Sheet1", index=False)

    report = tmp_path / "report.html"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(report), "")))

    window.file_path = str(book)
    window.load_file()
    assert window.start_analysis_button.isEnabled(), window.mapping_feedback_label.text()
    window.determine_and_run_test()

    assert report.exists(), "the analysis wrote no report"
    return load_report(str(report)).order


def test_the_control_group_is_drawn_first(window, tmp_path, monkeypatch,
                                         neutralized_dialogs):
    """WT is the reference; alphabetical order puts KO in front of it."""
    order = _axis_order_after_running(window, tmp_path, ["WT", "KO"], monkeypatch)

    assert order == ["WT", "KO"]


def test_a_timecourse_is_ordered_by_time_not_by_alphabet(window, tmp_path, monkeypatch,
                                                        neutralized_dialogs):
    """D7 belongs between D0 and D14, not after D21."""
    order = _axis_order_after_running(
        window, tmp_path, ["D0", "D7", "D14", "D21"], monkeypatch)

    assert order == ["D0", "D7", "D14", "D21"]


def _two_factor_axis_order(window, tmp_path, monkeypatch, genotypes, times):
    """Same journey, but a two-factor design, whose cells reach the ranking joined."""
    from PyQt5.QtWidgets import QFileDialog
    from fuzzing.html_oracles import load_report

    rng = np.random.default_rng(11)
    rows = []
    for gi, genotype in enumerate(genotypes):
        for ti, time in enumerate(times):
            for _ in range(6):
                rows.append({"Genotype": genotype, "Time": time,
                             "Value": float(rng.normal(10 + gi * 2 + ti * 1.5, 1.0))})
    book = tmp_path / "twofactor.xlsx"
    with pd.ExcelWriter(book) as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="Sheet1", index=False)

    report = tmp_path / "report_two.html"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(report), "")))

    window.file_path = str(book)
    window.load_file()
    assert window.start_analysis_button.isEnabled(), window.mapping_feedback_label.text()
    window.determine_and_run_test()

    assert report.exists(), "the two-factor analysis wrote no report"
    return load_report(str(report)).order


def test_two_factor_cells_keep_the_control_first(window, tmp_path, monkeypatch,
                                                 neutralized_dialogs):
    """The join must not cost the ranking its rules.

    A two-factor design arrives here as one label per cell -- "Genotype=WT,
    Time=D0" -- and ranking those strings is not ranking the design. The numeric
    half survives the join, because natural_order reads the digits inside the
    string, so D7 still precedes D21. Control-first does not: "Genotype=KO, ..."
    sorts before "Genotype=WT, ...", and the reference cell stops leading the
    axis. Where that looks right -- Control before Treated -- it is alphabetical
    luck rather than the rule, which is why the case here uses WT/KO.
    """
    order = _two_factor_axis_order(window, tmp_path, monkeypatch,
                                   ["WT", "KO"], ["D0", "D7", "D14", "D21"])

    assert order == [
        "Genotype=WT, Time=D0", "Genotype=WT, Time=D7",
        "Genotype=WT, Time=D14", "Genotype=WT, Time=D21",
        "Genotype=KO, Time=D0", "Genotype=KO, Time=D7",
        "Genotype=KO, Time=D14", "Genotype=KO, Time=D21",
    ]


def test_the_ranking_reorders_the_selection_without_widening_it():
    """`groups` is a selection as well as an order.

    Ranking must reorder exactly what the caller passed. Replacing it with every
    level found in the frame would quietly pull back in the groups a user had
    deselected.
    """
    from core.level_order import natural_order

    selected = ["D21", "D0"]
    ranked = [str(v) for v in natural_order(selected)]

    assert ranked == ["D0", "D21"]
    assert sorted(ranked) == sorted(selected), "the selection changed membership"


def test_ranking_is_idempotent_so_a_correct_caller_is_never_disturbed():
    """Applying it unconditionally is only safe because re-ranking changes
    nothing -- a caller that already handed over a ranked list is untouched."""
    from core.level_order import natural_order

    for labels in (["WT", "KO"], ["D0", "D7", "D14", "D21"],
                   ["Vehicle", "HighDose", "LowDose"], ["Ctrl"], []):
        once = [str(v) for v in natural_order(labels)]
        twice = [str(v) for v in natural_order(once)]
        assert twice == once, labels
