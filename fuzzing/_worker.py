"""Fuzzer worker — runs ONE seed in an isolated process.

Isolation is the whole point: C-level segfaults in NumPy/SciPy/pingouin on a
malformed tensor terminate the process with no Python traceback. Running each
case in a child means a segfault kills only the child; the orchestrator records
the offending seed (returncode = negative signal) and moves on.

Exit codes:
  0  graceful (valid result or clean block/error) and oracles passed
  2  oracle violation (silent statistical failure)
  3  uncaught Python exception escaped analyze()
  <0 (signal) crash/segfault — observed by the orchestrator, not set here
"""
import json
import os
import sys
import tempfile

# Headless + non-GUI plotting BEFORE any heavy import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, os.path.join(_ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _neutralize_dialogs():
    """Replace every interactive Qt dialog with a non-interactive default so an
    assumption-violating or significant case doesn't block on a modal dialog."""
    # Catch-all: any modal dialog returns immediately (Rejected) instead of
    # blocking the event-less headless process forever. This covers dialogs that
    # are constructed directly (e.g. ComparisonSelectionDialog.exec_()), not only
    # those routed through UIDialogManager.
    try:
        from PyQt5.QtWidgets import QDialog
        QDialog.exec_ = lambda self, *a, **k: 0
        QDialog.exec = lambda self, *a, **k: 0
    except Exception:
        pass
    try:
        from analysis.statisticaltester import UIDialogManager
    except Exception:
        return
    UIDialogManager.select_transformation_dialog = staticmethod(lambda *a, **k: "log10")
    # "tukey" uses all comparisons without a follow-up custom-pairs modal.
    UIDialogManager.select_posthoc_test_dialog = staticmethod(lambda *a, **k: "tukey")
    for name in ("select_nonparametric_posthoc_dialog",
                 "select_control_group_dialog", "select_custom_pairs_dialog"):
        setattr(UIDialogManager, name, staticmethod(lambda *a, **k: None))


def main(seed: int) -> int:
    from fuzzing.generators import build_case, case_to_analyze_kwargs
    from fuzzing.oracles import check_result

    _neutralize_dialogs()

    case = build_case(seed)
    verdict = {"seed": seed, "test": case.test_label, "mutations": case.mutations}

    with tempfile.TemporaryDirectory() as tmp:
        import pandas as pd
        dummy = os.path.join(tmp, "dummy.xlsx")
        pd.DataFrame({"a": [1]}).to_excel(dummy, index=False)
        kwargs = case_to_analyze_kwargs(case, dummy, os.path.join(tmp, "out"))

        from analysis.analysis_core import AnalysisManager
        try:
            result = AnalysisManager.analyze(**kwargs)
        except Exception as exc:  # pragma: no cover - this is a finding
            import traceback
            verdict["status"] = "exception"
            verdict["error"] = f"{type(exc).__name__}: {exc}"
            verdict["traceback"] = traceback.format_exc()[-1500:]
            print("__FUZZ__" + json.dumps(verdict))
            return 3

    violations = check_result(result)
    if violations:
        verdict["status"] = "oracle_violation"
        verdict["violations"] = violations
        verdict["test_label_result"] = result.get("test") if isinstance(result, dict) else None
        print("__FUZZ__" + json.dumps(verdict))
        return 2

    verdict["status"] = "ok"
    verdict["blocked"] = bool(result.get("blocked")) if isinstance(result, dict) else None
    print("__FUZZ__" + json.dumps(verdict))
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1])))
