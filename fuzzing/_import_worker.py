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


def _neutralize(declared_format, seed, report_path):
    """Answer the dialogs a user would answer, and nothing else.

    Everything here is a *user decision*, never something the app computes. The
    CSV number format is the case's declaration, right or wrong. The save
    location is where the report goes. The post-hoc, transformation, control
    group and pair dialogs behind the analysis are answered by the analysis
    fuzzer's own neutralizer, reused verbatim rather than rebuilt -- a second
    copy of that logic would drift from the first and test itself.
    """
    from PyQt5.QtWidgets import QFileDialog, QMessageBox

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

    # "Start Auto Analysis" opens a Save Analysis Report dialog before anything
    # runs, and returns without analysing if it comes back empty. Headless it
    # would simply block -- the first thing to hang, before any of the post-hoc
    # dialogs are even reached.
    QFileDialog.getSaveFileName = staticmethod(
        lambda *a, _p=report_path, **k: (_p, "HTML Report (*.html)")
    )

    from autopilot import statistical_analyzer_autopilot_pipeline as pipeline
    pipeline.AutopilotMixin._prompt_csv_format = (
        lambda self, _f=declared_format: dict(_f) if _f else None
    )

    from fuzzing._worker import _neutralize_dialogs
    _neutralize_dialogs(seed)
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


def _run_the_analysis(window, report_dir):
    """Press the button the worker used to only look at.

    This is the seam: until now the import fuzzer stopped at the mapping and the
    analysis fuzzer started from a DataFrame it had built itself, so the join --
    a real mapping actually driving a real analysis -- ran nowhere, in neither
    fuzzer and not in ``validation/test_all_paths.py`` either, which reads a real
    file but hand-builds the context the UI would have produced.

    ``determine_and_run_test`` is a plain synchronous slot: no QThread, no
    QRunnable, no thread pool anywhere in the autopilot. It returns only once
    the analysis is done and the report is written, so there is nothing to poll.
    """
    from fuzzing._worker import _check_one, _locate_reports, _report_was_expected

    outcome = {"analysis_ran": True, "reports": [], "report_oracles": [],
               "violations": []}
    try:
        window.determine_and_run_test()
    except Exception as exc:
        outcome["analysis_error"] = f"{type(exc).__name__}: {exc}"
        outcome["violations"].append(
            f"the mapping was accepted but the analysis raised {type(exc).__name__}: {exc}"
        )
        return outcome

    result = getattr(window, "current_analysis_result", None)
    outcome["result_test"] = (result or {}).get("test") if isinstance(result, dict) else None
    outcome["blocked"] = bool((result or {}).get("blocked")) if isinstance(result, dict) else False

    reports = _locate_reports(report_dir)
    outcome["reports"] = [os.path.basename(p) for p in reports]

    if not isinstance(result, dict):
        # The button was enabled, so the app promised the design was runnable.
        # Coming back with no result at all is that promise broken.
        outcome["violations"].append(
            "the mapping was presented as ready but the analysis produced no result")
        return outcome

    if not reports and _report_was_expected(result):
        outcome["violations"].append(
            "the analysis produced a result but wrote no report")
        return outcome

    fired = []
    for path in reports:
        violations, names = _check_one(path, result)
        fired.extend(names)
        outcome["violations"].extend(
            f"[{os.path.basename(path)}] {v}" for v in violations)
    outcome["report_oracles"] = sorted(set(fired))
    return outcome


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

        report_dir = os.path.join(work_dir, "out")
        os.makedirs(report_dir, exist_ok=True)

        app = QApplication.instance() or QApplication([])
        messages = _neutralize(case.declared_format, seed,
                               os.path.join(report_dir, "report.html"))

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
            # Only when the app itself says the design is runnable. A file it
            # correctly refused has nothing to analyse, and pressing the button
            # anyway would test a path no user can reach.
            if state["start_enabled"]:
                outcome = _run_the_analysis(window, report_dir)
                violations.extend(outcome.pop("violations"))
                fired.extend(outcome.pop("report_oracles"))
                verdict.update(outcome)
                verdict["oracles_fired"] = sorted(set(fired))
            else:
                verdict["analysis_ran"] = False

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
