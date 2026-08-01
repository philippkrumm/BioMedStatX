"""The main window must fit common small/scaled Windows viewports without a
whole-window scroll. The autopilot layout wraps its tall columns (mapping,
cockpit) in scroll areas with an Ignored vertical policy so the columns scroll
internally instead of inflating the window's preferred height. This guards
against a regression (a re-added minimum height, a removed size policy, or the
stale hardcoded setGeometry) silently bringing the outer scroll back.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtWidgets import QApplication, QScrollArea


@pytest.fixture(scope="module")
def qapp():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.mark.parametrize("w,h", [(1366, 768), (1280, 720)])
def test_no_whole_window_scroll_on_small_screens(qapp, w, h):
    # Pin the onboarding version to the current one so the first-run welcome box
    # (modal, would block offscreen) is not offered during construction.
    from autopilot.statistical_analyzer_autopilot_pipeline import _current_app_version
    QSettings("BioMedStatX", "BioMedStatX").setValue(
        "onboarding/completed_version", _current_app_version())

    from analysis.statistical_analyzer import StatisticalAnalyzerApp
    win = StatisticalAnalyzerApp()
    try:
        win.showNormal()
        win.resize(w, h)
        win.show()
        for _ in range(8):
            qapp.processEvents()

        root_scroll = next(
            (sa for sa in win.findChildren(QScrollArea)
             if sa.widget() is not None and sa.widget().objectName() == "autoPilotRoot"),
            None,
        )
        assert root_scroll is not None, "central autoPilotRoot scroll area not found"
        overflow = root_scroll.verticalScrollBar().maximum()
        assert overflow == 0, f"whole-window scroll reappeared at {w}x{h}: overflow {overflow}px"
    finally:
        win.close()
