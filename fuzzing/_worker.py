"""Fuzzer worker — runs ONE seed in an isolated process.

Isolation is the whole point: C-level segfaults in NumPy/SciPy/pingouin on a
malformed tensor terminate the process with no Python traceback. Running each
case in a child means a segfault kills only the child; the orchestrator records
the offending seed (returncode = negative signal) and moves on.

The exported HTML report is checked before the working directory goes away.
It used to be written into a TemporaryDirectory and deleted immediately after
``analyze()`` returned, so the artefact the user actually receives was produced
and destroyed without anything ever looking at it. A failing report is copied
out so the seed can be inspected rather than only re-run.

Exit codes:
  0  graceful (valid result or clean block/error) and oracles passed
  2  oracle violation (silent statistical failure, in the result or the report)
  3  uncaught Python exception escaped analyze()
  <0 (signal) crash/segfault — observed by the orchestrator, not set here
"""
import glob
import json
import os
import shutil
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


_TRANSFORMS = ["log10", "sqrt", "box_cox", None]
_POSTHOC_PARAMETRIC = ["tukey", "games_howell", "dunnett"]
_POSTHOC_NONPARAM = ["dunn", None]


def _neutralize_dialogs(seed: int):
    """Replace every interactive Qt dialog with a randomized-but-deterministic default.

    Randomizing transform and posthoc ensures every method is exercised across seeds
    instead of always hitting the same code path.
    """
    import numpy as np
    rng = np.random.default_rng(seed + 0xDEAD)
    chosen_transform = _TRANSFORMS[int(rng.integers(0, len(_TRANSFORMS)))]
    chosen_posthoc = _POSTHOC_PARAMETRIC[int(rng.integers(0, len(_POSTHOC_PARAMETRIC)))]
    chosen_nonparam = _POSTHOC_NONPARAM[int(rng.integers(0, len(_POSTHOC_NONPARAM)))]

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

    UIDialogManager.select_transformation_dialog = staticmethod(
        lambda *a, _t=chosen_transform, **k: _t
    )
    UIDialogManager.select_posthoc_test_dialog = staticmethod(
        lambda *a, _p=chosen_posthoc, **k: _p
    )
    UIDialogManager.select_nonparametric_posthoc_dialog = staticmethod(
        lambda *a, _np=chosen_nonparam, **k: _np
    )
    # Dunnett needs a control group; return the first group if available
    def _control_group_dialog(*a, groups=None, **k):
        if groups:
            return groups[0]
        return None
    UIDialogManager.select_control_group_dialog = staticmethod(_control_group_dialog)
    UIDialogManager.select_custom_pairs_dialog = staticmethod(lambda *a, **k: [])


def _locate_report(directory: str) -> str:
    """The written HTML report, or "" if the run produced none."""
    reports = sorted(glob.glob(os.path.join(directory, "*.html")))
    return reports[0] if reports else ""


def _report_was_expected(result) -> bool:
    """A run that got as far as a result should have written its report.

    Blocked, cancelled and errored runs unwind before the export step, which is
    correct behaviour rather than a missing artefact.
    """
    if not isinstance(result, dict):
        return False
    return not (result.get("blocked") is True
                or result.get("cancelled") is True
                or result.get("error"))


def main(seed: int, keep_dir: str = "") -> int:
    from fuzzing.generators import build_case, case_to_analyze_kwargs
    from fuzzing.html_oracles import check_report
    from fuzzing.oracles import check_result

    _neutralize_dialogs(seed)

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

        # The report is still on disk here; the temp directory closes below.
        report_path = _locate_report(tmp)
        fired = []
        if report_path:
            report_violations, fired = check_report(report_path, result)
            violations += report_violations
        elif _report_was_expected(result):
            violations.append("analysis produced a result but wrote no HTML report")
        verdict["oracles_fired"] = fired
        verdict["report_written"] = bool(report_path)
        # Recorded so the coverage summary can say which branches a run actually
        # reached, rather than only how many seeds were green.
        if isinstance(result, dict):
            verdict["posthoc"] = result.get("posthoc_test")
            verdict["had_resolution"] = bool(
                result.get("p_value_resolution")
                or any(isinstance(c, dict) and c.get("p_value_resolution")
                       for c in (result.get("pairwise_comparisons") or [])))

        if violations and keep_dir and report_path:
            try:
                os.makedirs(keep_dir, exist_ok=True)
                kept = os.path.join(keep_dir, f"seed_{seed}.html")
                shutil.copyfile(report_path, kept)
                verdict["report_kept"] = kept
            except Exception as exc:  # keeping the evidence must not mask the finding
                verdict["report_keep_error"] = f"{type(exc).__name__}: {exc}"

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
    sys.exit(main(int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else ""))
