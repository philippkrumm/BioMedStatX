"""Import fuzzer worker — runs ONE seed in an isolated process.

Unlike the analysis worker this one drives the real window. There is no way to
test the mapping otherwise: the buckets, the heuristics and the wide-format
pivot all live on ``StatisticalAnalyzerApp``, and a harness that reproduced them
would be testing its own copy. The window builds headless under
``QT_QPA_PLATFORM=offscreen``.

Exactly two things are stubbed, and both are things a user answers rather than
things the app computes: the CSV number-format dialog (the fuzzed answer is the
case's ``declared_format``, correct or not) and the message boxes, which would
otherwise block on any load error. The file dialog needs no stub -- ``browse``
only sets ``file_path`` and calls ``load_file``, so the worker does the same.

Exit codes:
  0  the file was handled and every applicable oracle passed
  2  oracle violation
  3  uncaught Python exception escaped the load/map path
  <0 (signal) crash — observed by the orchestrator, not set here
"""
import json
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, os.path.join(_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _neutralize(declared_format):
    """Answer the two dialogs a user would answer, and nothing else."""
    from PyQt5.QtWidgets import QMessageBox

    # Recorded, not merely swallowed: an app that gives up on a file without
    # telling anyone is the failure this layer is most likely to hide.
    seen = []
    for name in ("warning", "critical", "information", "question"):
        def _record(*args, _n=name, **kwargs):
            seen.append({"kind": _n,
                         "title": str(args[1]) if len(args) > 1 else "",
                         "text": str(args[2]) if len(args) > 2 else ""})
            return QMessageBox.Ok
        setattr(QMessageBox, name, staticmethod(_record))

    from autopilot import statistical_analyzer_autopilot_pipeline as pipeline
    pipeline.AutopilotMixin._prompt_csv_format = (
        lambda self, _f=declared_format: dict(_f) if _f else None
    )
    return seen


def _snapshot(window, case, messages):
    import pandas as pd

    frame = getattr(window, "df", None)
    state = {
        "loaded": frame is not None and not frame.empty,
        "columns": [], "dtypes": {}, "n_rows": 0,
        "dv_is_numeric": False, "dv_values": None, "factor_levels": None,
        "wide_pivoted": bool(getattr(window, "_wide_format_info", None)),
        "dv_bucket": window.dv_bucket.get_assigned_columns(),
        "factor1_bucket": window.factor1_bucket.get_assigned_columns(),
        "factor2_bucket": window.factor2_bucket.get_assigned_columns(),
        "subject_bucket": window.subject_bucket.get_assigned_columns(),
        "mapping_feedback": window.mapping_feedback_label.text(),
        "start_enabled": bool(window.start_analysis_button.isEnabled()),
        "messages": list(messages),
    }
    if frame is None:
        return state

    state["columns"] = [str(c) for c in frame.columns]
    state["dtypes"] = {str(c): str(frame[c].dtype) for c in frame.columns}
    state["n_rows"] = int(len(frame))

    if case.dv_name in frame.columns:
        column = frame[case.dv_name]
        state["dv_is_numeric"] = bool(pd.api.types.is_numeric_dtype(column))
        if state["dv_is_numeric"]:
            state["dv_values"] = [None if pd.isna(v) else float(v) for v in column]
    if case.factor_name in frame.columns:
        state["factor_levels"] = [
            "" if pd.isna(v) else str(v) for v in frame[case.factor_name]
        ]
    return state


def main(seed: int) -> int:
    from PyQt5.QtCore import QSettings
    from PyQt5.QtWidgets import QApplication

    verdict = {"seed": seed}
    with tempfile.TemporaryDirectory() as work_dir:
        from fuzzing.import_generators import build_case
        from fuzzing.import_oracles import check_import

        case = build_case(seed, work_dir)
        verdict.update({
            "file_format": case.file_format,
            "mutations": case.mutations,
            "header_row": case.header_row,
            "faithful_expected": case.expect_faithful_read,
            "bytes": os.path.getsize(case.file_path),
        })

        app = QApplication.instance() or QApplication([])
        messages = _neutralize(case.declared_format)

        from autopilot.statistical_analyzer_autopilot_pipeline import _current_app_version
        QSettings("BioMedStatX", "BioMedStatX").setValue(
            "onboarding/completed_version", _current_app_version())

        from analysis.statistical_analyzer import StatisticalAnalyzerApp
        window = StatisticalAnalyzerApp()
        try:
            window.file_path = case.file_path
            try:
                window.load_file()
            except Exception as exc:
                verdict.update({"status": "exception",
                                "error": f"{type(exc).__name__}: {exc}"})
                print("__IMPORT_FUZZ__" + json.dumps(verdict, default=str))
                return 3

            state = _snapshot(window, case, messages)
            violations, fired = check_import(state, case)
            verdict.update({
                "oracles_fired": fired,
                "loaded": state["loaded"],
                "columns_imported": len(state["columns"]),
                "dv_numeric": state["dv_is_numeric"],
                "start_enabled": state["start_enabled"],
                "mapping_feedback": state["mapping_feedback"],
                "wide_pivoted": state["wide_pivoted"],
                "messages": state["messages"],
            })
            if violations:
                verdict.update({"status": "violation", "violations": violations})
                print("__IMPORT_FUZZ__" + json.dumps(verdict, default=str))
                return 2

            verdict["status"] = "ok"
            print("__IMPORT_FUZZ__" + json.dumps(verdict, default=str))
            return 0
        finally:
            window.close()
            app.processEvents()


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1])))
