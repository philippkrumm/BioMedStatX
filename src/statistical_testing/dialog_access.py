"""Single resolver for the UI dialog manager used by the testing engines.

`assumption_checks` and `posthoc_fallback` each carried an identical copy of
this indirection. It exists so the engines pick up a dialog manager that tests
monkeypatch on `analysis.statisticaltester`, rather than binding the original
class at import time.
"""
from analysis.stats_functions import UIDialogManager


def get_ui_dialog_manager():
    """Resolve dialog manager through statisticaltester to honor test-time monkeypatches."""
    try:
        from analysis.statisticaltester import UIDialogManager as patched_dialog_manager
        return patched_dialog_manager
    except Exception:
        return UIDialogManager
