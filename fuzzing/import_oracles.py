"""Invariants for the import and mapping layers.

The question this layer has to answer is not "did a DataFrame come out" but "was
the file understood". Every check below therefore compares the app's state
against the ground truth the generator recorded when it wrote the file, and each
states its own precondition, so an oracle that had no occasion to apply is
reported as not fired rather than as passed.

The split that matters is ``case.expect_faithful_read``. When the file is one
the app can be expected to parse -- header on the first row, and, for a CSV, the
number format the user declared is the one it was written in -- the read must
reproduce the file exactly. When it is not, the app is allowed to misread; what
it is not allowed to do is present a complete, ready-looking mapping built on a
misread, because that is the one failure the user cannot see.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

# Wording the app uses when it refuses to proceed. Presence of any of these in
# the mapping feedback means the user was told something is missing.
_REFUSAL_MARKERS = (
    "assign at least one measurement column",
    "assign at least one factor column",
    "assign factor 1",
    "single mode requires exactly one measurement column",
    "multi mode requires at least two measurement columns",
    "at most two factor columns",
    "only one subject-id column",
    "load a file",
)


def _is_long_ground_truth(case) -> bool:
    """Do ``case.values`` / ``case.levels`` describe the frame the app holds?

    Only for a long-format file. A wide case records its truth as subjects,
    conditions and a value grid instead, and a wide file that failed to pivot
    (itself a finding) leaves the app holding a frame that neither description
    fits -- so the long checks must decline rather than compare against empty
    lists and report a phantom mismatch.
    """
    return case.layout == "long"


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite(values) -> List[float]:
    return [float(v) for v in (values or []) if _is_number(v) and math.isfinite(v)]


def _values_match(written: List[float], imported: List[float], tol=1e-6) -> bool:
    if len(written) != len(imported):
        return False
    return all(abs(a - b) <= tol * max(1.0, abs(a))
               for a, b in zip(sorted(written), sorted(imported)))


# --- the file was read faithfully -----------------------------------------------


def _oracle_file_loaded(state, case, violations) -> bool:
    # A file the app is meant to decline is not a file it is meant to load.
    # ``expect_faithful_read`` is about parsing -- header position and number
    # format -- and says nothing about a blank subject ID, which is perfectly
    # parseable and still has to be refused.
    if not case.expect_faithful_read or case.expect_refusal:
        return False
    if not state.get("loaded"):
        violations.append(
            f"a {case.file_format} file the app is expected to parse did not load at all"
        )
        return True
    if not state.get("n_rows"):
        violations.append("the file loaded but the frame has no rows")
    return True


def _oracle_shape_survives(state, case, violations) -> bool:
    # n_rows/n_cols describe the file on disk in either layout, so this one
    # stays useful for a wide file too -- and it is what says *how* a wide file
    # that failed to pivot still looks, next to the pivot oracle saying that it
    # did not.
    if not (case.expect_faithful_read and state.get("loaded")):
        return False
    if state.get("wide_pivoted"):
        return False        # a pivot legitimately reshapes the frame
    if state.get("n_rows") != case.n_rows:
        violations.append(
            f"the file has {case.n_rows} data rows but {state.get('n_rows')} were imported"
        )
    columns = state.get("columns") or []
    if len(columns) != case.n_cols:
        violations.append(
            f"the file has {case.n_cols} columns but {len(columns)} were imported: {columns}"
        )
    return True


def _oracle_dv_is_numeric(state, case, violations) -> bool:
    """A column of numbers must arrive as numbers.

    This is the whole point of declaring the number format: a value that reaches
    the app as text is not merely mistyped, it is invisible to the mapping --
    ``is_numeric_dtype`` gates which columns can become the dependent variable.
    """
    if not (_is_long_ground_truth(case) and case.expect_faithful_read
            and state.get("loaded")):
        return False
    dtypes = state.get("dtypes") or {}
    if case.dv_name not in dtypes:
        violations.append(
            f"the measurement column {case.dv_name!r} is not among the imported "
            f"columns {list(dtypes)}"
        )
        return True
    if not state.get("dv_is_numeric"):
        violations.append(
            f"{case.dv_name!r} holds numbers in the file but was imported as "
            f"{dtypes[case.dv_name]} -- it cannot be chosen as a measurement column"
        )
    return True


def _oracle_values_survive(state, case, violations) -> bool:
    if not (_is_long_ground_truth(case) and case.expect_faithful_read
            and state.get("loaded") and state.get("dv_is_numeric")
            and not state.get("wide_pivoted")):
        return False
    imported = _finite(state.get("dv_values"))
    written = _finite(case.values)
    if len(imported) != len(written):
        violations.append(
            f"{case.dv_name!r}: {len(written)} numbers were written, {len(imported)} "
            f"survived import -- {len(written) - len(imported)} became missing"
        )
        return True
    if not _values_match(written, imported):
        # The dangerous shape: same count, different numbers. A thousands
        # separator read as a decimal point is off by a factor of 1000 and
        # otherwise looks perfectly plausible.
        sample = [(w, i) for w, i in zip(sorted(written), sorted(imported)) if abs(w - i) > 1e-6][:3]
        violations.append(
            f"{case.dv_name!r}: the values changed during import, e.g. "
            + ", ".join(f"{w} -> {i}" for w, i in sample)
        )
    return True


def _oracle_levels_survive(state, case, violations) -> bool:
    if not (_is_long_ground_truth(case) and case.expect_faithful_read
            and state.get("loaded") and not state.get("wide_pivoted")):
        return False
    imported = [str(v) for v in (state.get("factor_levels") or []) if str(v).strip()]
    written = [v for v in case.levels if v]
    if sorted(imported) != sorted(written):
        missing = sorted(set(written) - set(imported))
        extra = sorted(set(imported) - set(written))
        violations.append(
            f"{case.factor_name!r}: group labels changed during import"
            + (f"; missing {missing}" if missing else "")
            + (f"; unexpected {extra}" if extra else "")
            + (f"; {len(written)} written vs {len(imported)} imported"
               if not missing and not extra else "")
        )
    return True


def _oracle_no_phantom_levels(state, case, violations) -> bool:
    """A blank cell must not become a group.

    Merged cells leave real blanks behind. Those rows have no label and the app
    cannot invent one -- but it must not manufacture a level out of them either,
    which is what a stringified NaN or an unstripped label would do.
    """
    if not state.get("loaded"):
        return False
    imported = {str(v) for v in (state.get("factor_levels") or [])}
    if not imported:
        return False
    written = set(case.levels) - {""}
    phantom = sorted(
        level for level in imported
        if level.strip() and level not in written
        and level.lower() in {"nan", "none", "<na>", "null"}
    )
    if phantom:
        violations.append(
            f"{case.factor_name!r}: blank cells became the group label(s) {phantom}"
        )
    return True


# --- the file could not be read faithfully, and the app must say so ---------------


# --- the wide layout was recognised and melted -----------------------------------


def _oracle_wide_is_pivoted(state, case, violations) -> bool:
    """A wide file must be melted on load, and a long one must not be.

    Both halves matter. A wide file left unpivoted reaches the mapping as one
    column per condition, so the user is asked to pick a "measurement" from four
    equally plausible ones; a long file wrongly pivoted invents a subject
    structure the data never had. The detector's own preconditions are in
    ``_detect_wide_format`` -- one subject-like column, two to eight numeric
    columns, unique subjects -- and the generator only builds files that meet
    them, so "not pivoted" here is a defect and not a taste.
    """
    if not state.get("loaded"):
        return False
    if case.layout == "long":
        if state.get("wide_pivoted"):
            violations.append(
                f"a long-format file was pivoted as if it were wide "
                f"(subject column '{state.get('pivot_subject_col')}')")
        return True
    if not case.expect_pivot:
        return False
    if not state.get("wide_pivoted"):
        violations.append(
            f"wide file with subject column '{case.subject_name}' and conditions "
            f"{case.condition_names} was not pivoted; the frame still holds "
            f"{state.get('columns')}")
    return True


def _oracle_pivot_keeps_every_value(state, case, violations) -> bool:
    """Melting rearranges the numbers; it must not lose or change any."""
    if not (case.expect_pivot and state.get("wide_pivoted")):
        return False
    imported = state.get("long_values")
    if imported is None:
        violations.append("the pivoted frame has no numeric Value column")
        return True
    written = _finite(case.melted_values)
    got = _finite(imported)
    if not _values_match(written, got):
        violations.append(
            f"the file holds {len(written)} measurements, the melted frame "
            f"{len(got)}; first written {written[:3]} vs imported {got[:3]}")
    return True


def _oracle_pivot_keeps_every_subject(state, case, violations) -> bool:
    """Every subject survives the melt, once per condition."""
    if not (case.expect_pivot and state.get("wide_pivoted")):
        return False
    subjects = state.get("long_subjects")
    if subjects is None:
        violations.append("the pivoted frame has no subject column")
        return True
    expected = sorted(case.subject_ids)
    seen = sorted(set(subjects))
    if seen != sorted(set(expected)):
        violations.append(
            f"the file has subjects {expected[:5]}... but the melted frame has "
            f"{seen[:5]}...")
        return True
    per_subject = {s: subjects.count(s) for s in set(subjects)}
    wrong = {s: n for s, n in per_subject.items() if n != len(case.condition_names)}
    if wrong:
        violations.append(
            f"every subject should appear once per condition "
            f"({len(case.condition_names)}x); these do not: "
            f"{dict(list(wrong.items())[:3])}")
    return True


def _oracle_conditions_are_the_columns(state, case, violations) -> bool:
    """The Condition levels are the wide file's value-column headers.

    This is where a BOM or an umlaut header shows up if it was mishandled: the
    condition names come straight from the header row, so a mangled encoding
    turns into a level the user never wrote.
    """
    if not (case.expect_pivot and state.get("wide_pivoted")):
        return False
    conditions = state.get("long_conditions")
    if conditions is None:
        return False
    seen = sorted(set(conditions))
    expected = sorted(case.condition_names)
    if seen != expected:
        violations.append(
            f"the file's condition columns are {expected} but the melted frame "
            f"has conditions {seen}")
    return True


def _oracle_wide_feedback_matches_design(state, case, violations) -> bool:
    """What the user is told about the pivot must match what was built.

    The line under the mapping is the only place the pivot is explained, and it
    is read before the analysis runs. It used to name a paired t-test for every
    wide file, so a four-timepoint study was told "paired t-test" and then given
    a repeated-measures ANOVA. Two conditions really are a paired design; three
    or more are not, whatever test the decision logic later picks.
    """
    if not (case.expect_pivot and state.get("wide_pivoted")):
        return False
    feedback = (state.get("mapping_feedback") or "").lower()
    if not feedback:
        violations.append("the frame was pivoted but the mapping said nothing about it")
        return True
    n_conditions = len(case.condition_names)
    if n_conditions > 2 and "paired" in feedback and "repeated" not in feedback:
        violations.append(
            f"{n_conditions} conditions were pivoted but the user is told this is a "
            f"paired design: {state.get('mapping_feedback')!r}")
    if n_conditions == 2 and "repeated-measures" in feedback:
        violations.append(
            f"two conditions were pivoted but the user is told this is a "
            f"repeated-measures design: {state.get('mapping_feedback')!r}")
    return True


def _oracle_missing_subject_is_refused(state, case, violations) -> bool:
    """A blank subject cell must stop the load out loud.

    The pivot refuses it on purpose: pandas drops NaN keys in the groupbys that
    decide repeated-measures structure, so a subject with no ID would vanish
    from the balance check without a word. What the oracle asks is that the
    refusal reached the user -- a frame silently short a few rows is exactly
    the failure the guard exists to prevent.
    """
    if not case.expect_refusal:
        return False
    if state.get("loaded") and state.get("wide_pivoted"):
        violations.append(
            "a wide file with a blank subject ID was pivoted anyway; rows "
            "without an ID silently leave every subject-keyed groupby")
        return True
    if not state.get("messages"):
        violations.append(
            "a wide file with a blank subject ID was declined without telling "
            "the user anything")
    return True


def _oracle_broken_import_fails_visibly(state, case, violations) -> bool:
    """The one failure the user cannot catch on their own.

    A file with notes above the header, or a CSV whose number format the user
    declared wrongly, may legitimately come out wrong -- there is deliberately no
    autodetect, so the declaration is the user's and the dialog's preview is the
    app's answer. What must not happen is that a misread turns into a mapping
    that looks finished: correct-looking buckets, the analysis button live, and
    numbers that are plausible and wrong.
    """
    if case.expect_faithful_read:
        return False
    if not state.get("loaded"):
        return True          # giving up on the file is the most visible failure
                             # there is; `load_failure_is_announced` checks the
                             # user was actually told.

    imported = _finite(state.get("dv_values"))
    written = _finite(case.values)
    read_was_faithful = state.get("dv_is_numeric") and _values_match(written, imported)
    if read_was_faithful:
        return True          # fired, and the app got it right anyway

    # An empty measurement bucket is NOT on its own a refusal: an app that
    # assigns nothing and still lights the analysis button up is the exact
    # contradiction this oracle exists for, and counting the empty bucket as
    # "refused" let a mutant that did precisely that walk straight past.
    feedback = str(state.get("mapping_feedback") or "").lower()
    refused = (
        not state.get("start_enabled")
        or any(marker in feedback for marker in _REFUSAL_MARKERS)
    )
    if not refused:
        violations.append(
            f"the import did not reproduce the file ({case.mutations}) yet the mapping "
            f"is presented as ready: dv={state.get('dv_bucket')}, "
            f"factor1={state.get('factor1_bucket')}, button enabled, "
            f"feedback {state.get('mapping_feedback')!r}"
        )
    return True


# --- the mapping put the columns where they belong --------------------------------


def _oracle_dv_reaches_bucket(state, case, violations) -> bool:
    if not (_is_long_ground_truth(case) and case.expect_faithful_read
            and state.get("loaded") and state.get("dv_is_numeric")
            and not state.get("wide_pivoted")):
        return False
    if state.get("dv_bucket") != [case.dv_name]:
        violations.append(
            f"the measurement column {case.dv_name!r} was imported as numbers but the "
            f"mapping put {state.get('dv_bucket')} in the measurement bucket"
        )
    return True


def _oracle_factor_reaches_bucket(state, case, violations) -> bool:
    if not (_is_long_ground_truth(case) and case.expect_faithful_read
            and state.get("loaded") and not state.get("wide_pivoted")):
        return False
    if len(case.distinct_levels) < 2:
        return False
    if state.get("factor1_bucket") != [case.factor_name]:
        violations.append(
            f"the group column {case.factor_name!r} carries "
            f"{len(case.distinct_levels)} levels but the mapping put "
            f"{state.get('factor1_bucket')} in Factor 1"
        )
    return True


def _oracle_measurement_is_not_a_subject(state, case, violations) -> bool:
    """A column of measurements must never be filed as the subject identifier."""
    if not (_is_long_ground_truth(case) and case.expect_faithful_read
            and state.get("loaded")):
        return False
    if case.dv_name in (state.get("subject_bucket") or []):
        violations.append(
            f"the measurement column {case.dv_name!r} was filed as a subject ID"
        )
    return True


def _oracle_load_failure_is_announced(state, case, violations) -> bool:
    """Giving up on a file silently is worse than failing to read it.

    ``_ap_load_file`` wraps the whole load in one ``except`` that sets
    ``self.df = None``. If the message box that is supposed to accompany that
    ever stops appearing, the user is left staring at the previous file's
    preview with no indication that the new one was refused -- and this is the
    seed shape that fires no other oracle at all, so nothing else would notice.
    """
    if state.get("loaded"):
        return False
    if not state.get("messages"):
        violations.append(
            f"the {case.file_format} file did not load and the user was told nothing: "
            f"feedback is {state.get('mapping_feedback')!r}"
        )
    return True


ORACLES = (
    ("file_loaded", _oracle_file_loaded),
    ("load_failure_is_announced", _oracle_load_failure_is_announced),
    ("wide_is_pivoted", _oracle_wide_is_pivoted),
    ("pivot_keeps_every_value", _oracle_pivot_keeps_every_value),
    ("pivot_keeps_every_subject", _oracle_pivot_keeps_every_subject),
    ("conditions_are_the_columns", _oracle_conditions_are_the_columns),
    ("missing_subject_is_refused", _oracle_missing_subject_is_refused),
    ("wide_feedback_matches_design", _oracle_wide_feedback_matches_design),
    ("shape_survives", _oracle_shape_survives),
    ("dv_is_numeric", _oracle_dv_is_numeric),
    ("values_survive", _oracle_values_survive),
    ("levels_survive", _oracle_levels_survive),
    ("no_phantom_levels", _oracle_no_phantom_levels),
    ("broken_import_fails_visibly", _oracle_broken_import_fails_visibly),
    ("dv_reaches_bucket", _oracle_dv_reaches_bucket),
    ("factor_reaches_bucket", _oracle_factor_reaches_bucket),
    ("measurement_is_not_a_subject", _oracle_measurement_is_not_a_subject),
)


def check_import(state: Dict[str, Any], case) -> Tuple[List[str], List[str]]:
    """Returns (violations, names of oracles that actually applied)."""
    violations: List[str] = []
    fired: List[str] = []
    for name, oracle in ORACLES:
        before = len(violations)
        try:
            if oracle(state, case, violations):
                fired.append(name)
        except Exception as exc:
            violations.append(f"oracle {name} raised {type(exc).__name__}: {exc}")
        # Tag each violation with the oracle that raised it: without this a
        # failing run says what is wrong but not which check noticed, and a
        # mutation test cannot tell "the right oracle bit" from "something did".
        for index in range(before, len(violations)):
            violations[index] = f"[{name}] {violations[index]}"
    return violations, fired
