"""_install_global_excepthook's inner _excepthook calls logger.info(msg, file=sys.stderr) -
file= isn't a valid logging kwarg, so this raises TypeError before the QMessageBox.critical
dialog call ever runs. The crash dialog has never fired for any uncaught exception.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest


@pytest.fixture(autouse=True)
def _qapp():
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_global_excepthook_shows_the_crash_dialog(monkeypatch, tmp_path):
    import analysis.statistical_analyzer as sa_module

    shown = []
    monkeypatch.setattr(
        sa_module.QMessageBox, "critical",
        staticmethod(lambda *a, **k: shown.append((a, k))), raising=False
    )
    # Avoid writing to the repo's real crash_log.txt during the test - the
    # write is already wrapped in its own try/except that silently passes on
    # any failure, so forcing it to fail here is a safe, realistic way to
    # isolate the test from disk state without changing what's under test.
    monkeypatch.setattr("builtins.open", lambda *a, **k: (_ for _ in ()).throw(OSError("no log in test")))

    sa_module._install_global_excepthook()
    hook = sys.excepthook
    assert hook is not sys.__excepthook__, "excepthook was not installed"

    try:
        raise ValueError("synthetic test exception")
    except ValueError:
        exc_type, exc_value, exc_tb = sys.exc_info()
        hook(exc_type, exc_value, exc_tb)

    assert len(shown) == 1, (
        "QMessageBox.critical was never called - the excepthook itself likely "
        "raised (e.g. the file=sys.stderr TypeError) before reaching it"
    )
