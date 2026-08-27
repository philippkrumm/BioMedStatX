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


# What the transformation dialog actually returns. The list used to read
# ["log10", "sqrt", "box_cox", None] -- but the dialog offers log10, boxcox and
# arcsin_sqrt (stats_functions.py:805), so "sqrt" and "box_cox" were answers no
# user can give. Both fall through every apply-branch and transform nothing, so
# three draws in four produced an untransformed run and the whole branch was
# fuzzed at a quarter of its apparent rate -- with boxcox and arcsin_sqrt, two
# of the three real transformations, never exercised at all. Same mistake as
# the "tukey" answer described in _posthoc_options below.
#
# "skip" is the explicit "continue without transforming" choice and None is
# Cancel; both are real answers and are kept.
_TRANSFORMS = ["log10", "boxcox", "arcsin_sqrt", "skip", None]
_POSTHOC_NONPARAM = ["dunn", None]


def _posthoc_options(progress_text):
    """The choices the real dialog would have offered for this design.

    Mirrors UIDialogManager.select_posthoc_test_dialog rather than guessing.
    The fuzzer used to answer "tukey" for every parametric design, which the
    one-way dialog does not offer at all -- a studentized-range post-hoc has no
    place after the Welch ANOVA that runs there -- so one-way post-hoc fuzzing
    drove a value no user can produce while Games-Howell and the pair dialog
    went under-exercised. RM and mixed reach their EMM/multivariate-t option
    only through this list, and with it the estimated p-values whose resolution
    the report has to respect.
    """
    text = str(progress_text or "")
    if "two_way_anova" in text:
        return ["tukey", "paired_custom"]
    if "mixed_anova" in text or "repeated_measures_anova" in text:
        return ["paired_custom", "emm_mvt"]
    return ["games_howell", "dunnett", "paired_custom"]


def _neutralize_dialogs(seed: int):
    """Replace every interactive Qt dialog with a randomized-but-deterministic default.

    Randomizing the answers ensures every branch is exercised across seeds
    instead of always hitting the same code path. The choices are drawn from
    what the real dialog offers, so a fuzzed run stays a run a user could have
    had.
    """
    import numpy as np
    rng = np.random.default_rng(seed + 0xDEAD)
    chosen_transform = _TRANSFORMS[int(rng.integers(0, len(_TRANSFORMS)))]
    chosen_nonparam = _POSTHOC_NONPARAM[int(rng.integers(0, len(_POSTHOC_NONPARAM)))]
    # Whether this user ticks every box in the pair dialog. Both answers matter:
    # all pairs is a complete comparison matrix and lets compact letters through,
    # a subset is exactly the case the letter gate has to refuse.
    takes_all_pairs = bool(rng.integers(0, 2))

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

    def _posthoc_dialog(parent=None, progress_text=None, column_name=None, default_method=None):
        options = _posthoc_options(progress_text)
        return options[int(rng.integers(0, len(options)))]

    UIDialogManager.select_posthoc_test_dialog = staticmethod(_posthoc_dialog)
    UIDialogManager.select_nonparametric_posthoc_dialog = staticmethod(
        lambda *a, _np=chosen_nonparam, **k: _np
    )

    def _control_group_dialog(*args, **kwargs):
        """Dunnett needs a control; pick one the way a user would.

        The real dialog is ``select_control_group_dialog(groups, parent=None)``
        and is called both positionally and by keyword. The stub only read the
        keyword, so every positional call answered None -- Dunnett then fell
        back to the all-pairs default and the fuzzer never actually ran it.
        """
        groups = kwargs.get("groups")
        if groups is None and args:
            groups = args[0]
        groups = list(groups or [])
        return groups[int(rng.integers(0, len(groups)))] if groups else None
    UIDialogManager.select_control_group_dialog = staticmethod(_control_group_dialog)

    # Only the arcsin_sqrt answer reaches this dialog, and until that answer was
    # a real option nothing here had to stand in for it -- so the run built a
    # QDialog with no QApplication and the process aborted with "Must construct
    # a QApplication before a QWidget", six seeds in two hundred. The real
    # dialog offers "proportion" (0-1) and "percent" (0-100) and returns None on
    # cancel; all three are answers a user gives, and all three matter, because
    # out-of-domain values are hard-rejected downstream and the refusal is the
    # path worth exercising.
    _arcsin_domain = ["proportion", "percent", None][int(rng.integers(0, 3))]
    UIDialogManager.select_arcsin_domain_type = staticmethod(
        lambda *a, _d=_arcsin_domain, **k: _d
    )

    def _custom_pairs_dialog(groups, parent=None):
        """A user who actually ticks boxes, rather than one who ticks none.

        Returning an empty list meant every seed that reached the pair dialog
        produced zero comparisons, so the branch ran without ever testing
        anything.
        """
        names = [g for g in (groups or [])]
        all_pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
        if not all_pairs or takes_all_pairs:
            return all_pairs
        count = int(rng.integers(1, len(all_pairs) + 1))
        picked = rng.choice(len(all_pairs), size=count, replace=False)
        return [all_pairs[int(i)] for i in sorted(picked)]

    UIDialogManager.select_custom_pairs_dialog = staticmethod(_custom_pairs_dialog)

    # Advanced designs do not reach the pair selection through UIDialogManager:
    # analysis_core builds a ComparisonSelectionDialog directly. With only
    # QDialog.exec_ stubbed that dialog reported "Rejected", which the pipeline
    # correctly reads as a user cancelling -- so every repeated-measures, mixed
    # and two-way run that chose paired_custom aborted, and the branch was never
    # tested. A stand-in that answers like a user who picked pairs restores it.
    try:
        from ui.dialogs import comparison_selection_dialog as _csd_module

        class _FuzzComparisonDialog:
            Accepted = 1

            def __init__(self, all_pairs, checked_by_default=False, *args, **kwargs):
                self._all_pairs = list(all_pairs or [])

            def exec_(self):
                return self.Accepted

            def get_selected_comparisons(self):
                if not self._all_pairs or takes_all_pairs:
                    return self._all_pairs
                count = int(rng.integers(1, len(self._all_pairs) + 1))
                picked = rng.choice(len(self._all_pairs), size=count, replace=False)
                return [self._all_pairs[int(i)] for i in sorted(picked)]

        _csd_module.ComparisonSelectionDialog = _FuzzComparisonDialog
    except Exception:
        pass


def _locate_reports(directory: str) -> list:
    """Every HTML report the run wrote.

    A multi-dataset run writes one report per dataset plus the combined
    overview. Checking only the first of the sorted list -- which is what this
    did -- would have silently ignored the rest.
    """
    return sorted(glob.glob(os.path.join(directory, "*.html")))


def _dataset_for(path: str, result: dict):
    """Match a per-dataset report file back to the dataset it was rendered for.

    Returns ``(dataset_name, sub_result_or_None)``, or ``None`` when the file
    belongs to no dataset of this run at all. A dataset whose analysis errored
    still leaves its own report behind, but never reaches ``results`` -- so a
    missing sub-result means "failed", not "orphaned", and the two must not be
    reported as the same thing.

    The file name is built from the dataset name, but the suffix differs by
    export branch (``_results.html`` for the clinical path, ``.html`` for the
    standard one), so the name is looked for inside the basename rather than
    matched against a fixed pattern. The longest match wins, so DS1 does not
    claim a file belonging to DS10.
    """
    name = os.path.basename(path)
    per_dataset = result.get("results") or {}
    failed = result.get("failed_datasets") or {}
    best = None
    for dataset_name in list(per_dataset) + list(failed):
        token = str(dataset_name)
        if token and token in name and (best is None or len(token) > len(best[0])):
            best = (token, per_dataset.get(dataset_name))
    return best


def _check_one(path: str, result: dict):
    """Route one file to the checks that apply to it."""
    from fuzzing.html_oracles import (check_multi_report, check_report,
                                      check_report_without_result)

    if result.get("type") == "multi_dataset_analysis":
        if os.path.basename(path).endswith("_combined_results_report.html"):
            return check_multi_report(path, result)
        match = _dataset_for(path, result)
        if match is None:
            return ([f"report {os.path.basename(path)} belongs to no dataset in the result"], [])
        _, sub = match
        if sub is None:
            # This dataset's analysis errored, so there is no result to compare
            # against -- but the report it left behind is still a report, and
            # what it asserts on its own can be checked.
            return check_report_without_result(path)
        return check_report(path, sub)
    return check_report(path, result)


def _report_was_expected(result) -> bool:
    """A run that got as far as a result should have written its report.

    Blocked, cancelled and errored runs unwind before the export step, which is
    correct behaviour rather than a missing artefact.
    """
    if not isinstance(result, dict):
        return False
    # A multi-dataset run reports per dataset: if none of them survived there is
    # nothing to combine and no report is written, which is correct.
    if result.get("type") == "multi_dataset_analysis":
        return bool(result.get("successful_datasets"))
    return not (result.get("blocked") is True
                or result.get("cancelled") is True
                or result.get("error"))


def main(seed: int, keep_dir: str = "") -> int:
    from fuzzing.generators import build_case, case_to_analyze_kwargs
    from fuzzing.html_oracles import report_stats
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

        # The reports are still on disk here; the temp directory closes below.
        reports = _locate_reports(tmp)
        fired = []
        stats = {"bytes": 0, "empty_states": 0, "figures": 0}
        for report_path in reports:
            report_violations, report_fired = _check_one(report_path, result)
            violations += report_violations
            for name in report_fired:
                if name not in fired:
                    fired.append(name)
            for key, value in (report_stats(report_path) or {}).items():
                stats[key] = stats.get(key, 0) + value
        if not reports and _report_was_expected(result):
            violations.append("analysis produced a result but wrote no HTML report")
        verdict["oracles_fired"] = fired
        verdict["report_written"] = bool(reports)
        verdict["reports_written"] = len(reports)
        verdict["report_stats"] = stats
        verdict["datasets"] = case.datasets
        # Recorded so the coverage summary can say which branches a run actually
        # reached, rather than only how many seeds were green.
        if isinstance(result, dict):
            verdict["posthoc"] = result.get("posthoc_test")
            verdict["had_resolution"] = bool(
                result.get("p_value_resolution")
                or any(isinstance(c, dict) and c.get("p_value_resolution")
                       for c in (result.get("pairwise_comparisons") or [])))

        # Keep every report of a failing seed, not the last one the loop happened
        # to leave behind in its variable -- a multi-dataset run writes several,
        # and which of them carries the finding is the question being asked.
        if violations and keep_dir and reports:
            try:
                os.makedirs(keep_dir, exist_ok=True)
                kept = []
                for index, report_path in enumerate(reports):
                    suffix = "" if len(reports) == 1 else f"_{index}"
                    target = os.path.join(keep_dir, f"seed_{seed}{suffix}.html")
                    shutil.copyfile(report_path, target)
                    kept.append(target)
                verdict["reports_kept"] = kept
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
