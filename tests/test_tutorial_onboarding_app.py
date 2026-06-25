# tests/test_tutorial_onboarding_app.py
import pytest
from PyQt5.QtWidgets import QApplication
from analysis.statistical_analyzer import StatisticalAnalyzerApp


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def app(qapp):
    w = StatisticalAnalyzerApp()
    yield w
    w.close()


def test_tour_steps_titles_and_count(app):
    steps = app._build_tour_steps()
    titles = [s.title for s in steps]
    assert titles == [
        "Bring In Your Data",
        "Meet Your Variables",
        "Shape Your Analysis",
        "Define and Compute",
        "Full Statistical Transparency",
        "Your Results at a Glance",
        "Help Is Always One Click Away",
    ]


def test_start_tutorial_is_reentrant(app):
    app.start_tutorial()
    assert app._tour_active is True
    first = app._tour_overlay
    app.start_tutorial()             # second call must not create a 2nd overlay
    assert app._tour_overlay is first
    app._tour_overlay.close_tour()
    assert app._tour_active is False


def test_export_template_copies_file(app, tmp_path, monkeypatch):
    from PyQt5.QtWidgets import QFileDialog
    from PyQt5.QtGui import QDesktopServices
    target = tmp_path / "out.xlsx"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(target), "")))
    opened = {"url": None}
    monkeypatch.setattr(QDesktopServices, "openUrl",
                        staticmethod(lambda url: opened.__setitem__("url", url)))
    app.export_example_template()
    assert target.exists() and target.stat().st_size > 0
    assert opened["url"] is not None


def test_export_template_cancelled_is_noop(app, tmp_path, monkeypatch):
    from PyQt5.QtWidgets import QFileDialog
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    app.export_example_template()  # no exception, nothing written


def test_help_menu_has_tour_and_template_actions(app):
    assert hasattr(app, "help_menu")
    texts = [a.text() for a in app.help_menu.actions()]
    assert any("Interactive Tour" in t for t in texts)
    assert any("Example Template" in t for t in texts)
