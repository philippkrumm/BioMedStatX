"""Oracles for the Result Cockpit -- the panel the user reads before the report.

Three defects were found in this panel in one sitting, all by eye, all in the
shipped 2.0 build: the design card announced the model that was PLANNED rather
than the one that ran, two design labels named tests this program never
performs, and a fit that produced nothing printed ``p = nan`` as though it were
a number to act on. None of the three fuzzers saw any of it, because all three
read the exported HTML and the cockpit is not in the HTML. A surface nothing
asks about is a surface where the next defect also survives to a release.

These checks read the summary dict the widget is handed -- built by the
product's own ``_build_result_summary``, never re-assembled here -- and hold
each claim against the result it was built from. The questions are the ones a
reader would ask of the panel:

* is every number on it a number
* does the model named there match the model that ran
* do the printed p-value, effect size, N and groups survive a round trip back
  to the result dict
* is the post-hoc sentence TRUE -- not merely present
* do the two validity cards agree with the assumption tests behind them

Each oracle returns whether it fired, separately from whether it passed. An
oracle whose precondition is never met adds nothing but the appearance of
coverage.

Known gap, stated rather than papered over: a blocked or cancelled run never
reaches these cards at all -- ``_ap_handle_blocked_result`` and
``_ap_handle_cancelled_result`` return first, and the widget blanks every card
itself. Those two states are widget behaviour, not formatter behaviour, and
nothing here can see them.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_real_number(x: Any) -> bool:
    return _is_number(x) and math.isfinite(x)


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _line(card: str, prefix: str) -> Optional[str]:
    """The value on the ``prefix:`` line of a multi-line card."""
    for line in _text(card).split("\n"):
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return None


def build_summary(context: Any, results: Any) -> Dict[str, str]:
    """What the cockpit is about to claim, built by the product's own code.

    Deliberately routed through ``AutopilotMixin._build_result_summary`` rather
    than calling the eight formatters in a list here: a second assembly of the
    same panel is free to drift from the real one, and then the oracle checks a
    cockpit nobody sees. The window itself is never constructed -- the mixin
    holds these as plain functions and none of them touches a widget.
    """
    from autopilot.statistical_analyzer_autopilot_pipeline import AutopilotMixin

    reader = object.__new__(type("_CockpitReader", (AutopilotMixin,), {}))
    return reader._build_result_summary(context or {}, results or {}, subtitle="")


# --- what the panel must never print -------------------------------------------
#
# Scanned per SEGMENT rather than over the whole panel: a group label really can
# be the string "nan" (a row with a missing label lands in one), and that is
# data the panel is correctly repeating, not a number it computed. Only the
# segments that carry computed numbers are read.

_NOT_A_NUMBER = re.compile(r"(?<![0-9A-Za-z_])([+-]?(?:nan|inf(?:inity)?))(?![0-9A-Za-z_])",
                           re.IGNORECASE)


def _numeric_segments(summary: Dict[str, str]) -> List[Tuple[str, str]]:
    segments = [
        ("Normality", _text(summary.get("metric_normality"))),
        ("Variance", _text(summary.get("metric_variance"))),
        ("Main test", _text(summary.get("inference_main_test"))),
        ("Effect size", _text(summary.get("inference_effect_size"))),
        ("Model", _line(_text(summary.get("context_design")), "Model:") or ""),
        ("Sample size", _line(_text(summary.get("context_sample_overview")), "Sample size (N):") or ""),
        ("Post-hoc", _line(_text(summary.get("context_analysis_scope")), "Post-hoc:") or ""),
    ]
    return [(name, value) for name, value in segments if value]


def _oracle_no_number_that_is_not_one(summary, context, results, violations) -> bool:
    """A value the reader would act on must be a value.

    ``p_value is None`` was the only guard the main-test line had, and NaN is
    not None: ``nan < 0.0001`` is False, so the card fell through to the
    ordinary format and printed "p = nan". The report already had a third state
    for this; the cockpit printed it as a number.
    """
    for name, value in _numeric_segments(summary):
        found = _NOT_A_NUMBER.search(value)
        if found:
            violations.append(
                f"cockpit {name} card prints {found.group(1)!r} as a value: {value!r}")
    return True


# --- the model named on the card -----------------------------------------------

# The placeholders the result carries where nothing was fitted. Mirrors
# _NOT_A_MODEL_NAME in the pipeline; a card falling back to the design label on
# one of these is correct, not a finding.
_NOT_A_MODEL_NAME = frozenset({"", "not performed", "none", "n/a"})


def _performed_test(results: dict) -> str:
    name = results.get("test")
    name = name.strip() if isinstance(name, str) else ""
    return "" if name.lower() in _NOT_A_MODEL_NAME else name


def _oracle_design_names_what_ran(summary, context, results, violations) -> bool:
    """The design card names the model that RAN, not the one that was planned.

    ``inferred_test`` is read off the shape of the data, before the assumption
    checks look at the numbers. Where those checks switch the analysis -- and
    for independent groups the router picks Welch unconditionally, so they
    always do -- the card announced a model that never ran, beside a results
    section and a post-hoc naming the real one.
    """
    performed = _performed_test(results)
    if not performed:
        return False
    shown = _line(summary.get("context_design"), "Model:")
    if shown is None:
        violations.append("cockpit design card has no 'Model:' line at all")
        return True
    if shown != performed:
        violations.append(
            f"cockpit names the model {shown!r} but the analysis that ran was {performed!r}")
    return True


# --- round trips: what is printed must survive parsing back to the result ------

_P_PATTERN = re.compile(r";\s*p\s*(<|=)\s*([0-9.eE+-]+)\s*$")


def _oracle_p_value_round_trips(summary, context, results, violations) -> bool:
    """The p-value on the card is the p-value in the result, to the digit.

    The card is the shortest path between a fit and a decision somebody makes,
    and it reformats the number itself rather than reusing the report's
    formatter -- so a wrong field, a lost sign or a truncation is invisible
    everywhere else.
    """
    card = _text(summary.get("inference_main_test"))
    if not card:
        return False

    expected_name = (results.get("tested_against") or results.get("final_test_label")
                     or results.get("test") or "Not available")
    shown_name = card.split(";")[0].strip()
    if str(expected_name).strip() != shown_name:
        violations.append(
            f"cockpit main-test card names {shown_name!r}; the result says {expected_name!r}")

    p = results.get("p_value")
    match = _P_PATTERN.search(card)

    if p is None:
        if "p = N/A" not in card:
            violations.append(f"result carries no p-value but the card says {card!r}")
        return True
    if not _is_real_number(p):
        # Caught as a printed non-number by its own oracle; here the claim is
        # only that no numeric p is asserted for a fit that produced none.
        if match:
            violations.append(
                f"result p_value is {p!r} but the card asserts p {match.group(1)} {match.group(2)}")
        return True

    if match is None:
        violations.append(f"result has p={p!r} but the card states no p-value: {card!r}")
        return True
    operator, number = match.group(1), match.group(2)
    try:
        shown = float(number)
    except ValueError:
        violations.append(f"cockpit p-value {number!r} does not parse as a number")
        return True
    if operator == "<":
        if not (p < shown):
            violations.append(f"card claims p < {shown} but the result p_value is {p!r}")
    elif abs(shown - round(float(p), 4)) > 1e-9:
        violations.append(f"card prints p = {shown} for a result p_value of {p!r}")
    return True


_EFFECT_PATTERN = re.compile(r"^(.*?)\s*=\s*(-?[0-9.eE+-]+)$")


def _oracle_effect_size_round_trips(summary, context, results, violations) -> bool:
    """The effect size on the card is the one in the result, under its own name."""
    if results.get("model_type") == "LogisticRegression":
        return False  # that card shows an odds ratio, checked by its own path
    card = _text(summary.get("inference_effect_size"))
    if not card:
        return False

    effect = results.get("effect_size")
    if not _is_real_number(effect):
        if card != "Not available":
            violations.append(
                f"result effect_size is {effect!r} but the card says {card!r}")
        return True

    match = _EFFECT_PATTERN.match(card)
    if match is None:
        violations.append(f"result has effect_size={effect!r} but the card reads {card!r}")
        return True
    label, number = match.group(1).strip(), match.group(2)
    try:
        shown = float(number)
    except ValueError:
        violations.append(f"cockpit effect size {number!r} does not parse as a number")
        return True
    if abs(shown - round(float(effect), 4)) > 1e-9:
        violations.append(f"card prints an effect size of {shown} for a result of {effect!r}")

    # An effect size whose KIND is dropped is a number without units: eta² and
    # Cohen's d are read against entirely different thresholds.
    if results.get("effect_size_type") and label == "Effect size":
        violations.append(
            f"result reports effect_size_type={results['effect_size_type']!r} but the "
            "card names no kind")
    return True


def _oracle_sample_size_round_trips(summary, context, results, violations) -> bool:
    """The N on the card is the N of the analysis."""
    shown = _line(summary.get("context_sample_overview"), "Sample size (N):")
    if shown is None:
        return False

    # The same four keys the card reads, held against the RESULT rather than
    # against the formatter: a linear or logistic regression records its count
    # only as `n_observations`, and an oracle that knew three keys would have
    # reported the fix for that as the defect.
    expected = results.get("n_total")
    if expected is None:
        expected = results.get("n")
    if expected is None:
        expected = results.get("n_observations")
    raw = results.get("raw_data")
    observations = None
    if isinstance(raw, dict) and raw:
        try:
            observations = sum(len(values) for values in raw.values()
                               if hasattr(values, "__len__"))
        except TypeError:
            observations = None
    if expected is None:
        expected = observations

    if expected is None:
        if shown != "N/A":
            violations.append(f"result carries no sample size but the card says {shown!r}")
        return True
    if shown != str(expected):
        violations.append(f"card shows N = {shown!r}; the result says {expected!r}")
        return True

    # Held against the data itself, not only against the field that produced the
    # text -- otherwise this asks whether the formatter formatted. Only where a
    # row IS an observation: a repeated-measures N counts subjects, and the raw
    # groups then hold more rows than there are subjects, correctly.
    if (observations is not None and _is_number(expected)
            and not context.get("subject_column") and observations != expected):
        violations.append(
            f"card shows N = {expected} but the analysed data holds {observations} values")
    return True


def _oracle_groups_round_trip(summary, context, results, violations) -> bool:
    """The groups listed are the groups analysed, and a truncation says how many it hid."""
    shown = _line(summary.get("context_sample_overview"), "Groups:")
    if shown is None or shown == "All available groups":
        return False
    expected = (results.get("selected_groups") or context.get("selected_groups")
                or results.get("groups") or [])
    expected = [str(group) for group in expected]
    if not expected:
        return False

    # A label carrying a newline breaks the card into more lines than it has
    # fields, so the listing cannot be read back at all. The panel is a QLabel
    # and a line break in the data renders as a line break -- ugly, but not a
    # false claim about the analysis, which is what this oracle is for. The
    # import layer strips surrounding whitespace off every label; what survives
    # is inside the text.
    if any(_CONTROL.search(name) for name in expected):
        return False

    hidden = re.search(r"\(\+(\d+) more\)$", shown)
    listed = re.sub(r"\s*\(\+\d+ more\)$", "", shown)

    if hidden and int(hidden.group(1)) != len(expected) - min(len(expected), 6):
        violations.append(
            f"card hides +{hidden.group(1)} groups of {len(expected)} analysed")

    for name in expected[:6]:
        if name not in listed:
            violations.append(f"card omits the group {name!r} the analysis ran on")

    # The card joins group names with ", " and a two-factor design names its
    # groups by CELL -- "FacA=A0, FacB=B0" -- so the separator appears inside
    # the names themselves and the listing cannot be split back apart. Counting
    # the pieces there measured the commas, not the groups. Where no name
    # carries one, the stronger question is still worth asking: is anything
    # listed that was not analysed?
    if not any("," in name for name in expected):
        names = [part.strip() for part in listed.split(",") if part.strip()]
        if not hidden and len(names) != len(expected):
            violations.append(
                f"card lists {len(names)} groups; the analysis ran on {len(expected)}")
        for name in names:
            if name not in expected:
                violations.append(
                    f"card lists a group {name!r} the analysis did not run on")
    return True


# --- the post-hoc sentence must be true, not merely present --------------------

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


_TTEST_CLAIM = "No post-hoc applicable for t-tests (two groups only)."
_NOT_SIGNIFICANT_CLAIM = "No post-hoc required because the omnibus test was not significant."
_NOT_STORED_CLAIM = "Significant omnibus result, but no post-hoc result was stored."
_NONE_CLAIM = "No post-hoc performed."


def _oracle_posthoc_line_is_true(summary, context, results, violations) -> bool:
    """Four sentences, each asserting something checkable about the run.

    The heading over a comparison table has already been wrong once in this
    program -- it said Tukey HSD over Holm-corrected t-tests -- and this line is
    the same claim one panel earlier.
    """
    shown = _line(summary.get("context_analysis_scope"), "Post-hoc:")
    if shown is None:
        return False
    groups = results.get("groups") or results.get("selected_groups") or []
    p = results.get("p_value")

    if shown == _TTEST_CLAIM:
        if len(groups) > 2:
            violations.append(
                f"card says post-hoc is inapplicable because there are two groups, but "
                f"the analysis ran on {len(groups)}")
        return True
    if shown == _NOT_SIGNIFICANT_CLAIM:
        if not _is_real_number(p):
            violations.append(
                f"card explains the omnibus test was not significant, but p_value is {p!r}")
        elif p < 0.05:
            violations.append(f"card calls the omnibus test not significant at p={p!r}")
        return True
    if shown == _NOT_STORED_CLAIM:
        if results.get("posthoc_test"):
            violations.append(
                f"card says no post-hoc was stored while the result carries "
                f"{results['posthoc_test']!r}")
        return True
    if shown == _NONE_CLAIM:
        if results.get("posthoc_test"):
            violations.append(
                f"card says no post-hoc was performed while the result carries "
                f"{results['posthoc_test']!r}")
        return True

    named = shown[:-1] if shown.endswith(".") else shown
    performed = results.get("posthoc_test")
    if not performed:
        violations.append(f"card names the post-hoc {named!r} but the result carries none")
    elif str(performed).strip() != named:
        violations.append(
            f"card names the post-hoc {named!r}; the result says {performed!r}")
    return True


# --- the two validity cards ----------------------------------------------------
#
# Held against the assumption tests WITHOUT re-implementing which of them the
# formatter picks. Reproducing that choice here would put two copies of the
# selection rule in the repository, free to drift, and the oracle would then be
# checking its own copy. The claim made instead is weaker and cannot drift: where
# every available answer points one way, the card must not say the other.

def _verdicts(container: Any, key: str) -> List[bool]:
    found = []
    if isinstance(container, dict):
        for value in container.values():
            if isinstance(value, dict) and value.get(key) is not None:
                found.append(bool(value.get(key)))
        if container.get(key) is not None:
            found.append(bool(container.get(key)))
    return found


def _oracle_assumption_cards_agree(summary, context, results, violations) -> bool:
    """"OK" where every normality test failed is a claim about nothing."""
    fired = False

    normality = _text(summary.get("metric_normality"))
    verdicts = _verdicts(results.get("normality_tests"), "is_normal")
    if verdicts and normality not in ("Not available", ""):
        fired = True
        if normality.startswith("OK") and not any(verdicts):
            violations.append(
                f"normality card says {normality!r} while every normality test failed")
        if normality.startswith("Violated") and all(verdicts):
            violations.append(
                f"normality card says {normality!r} while every normality test passed")

    variance = _text(summary.get("metric_variance"))
    equal = _verdicts(results.get("variance_test"), "equal_variance")
    if equal and variance not in ("Not available", ""):
        fired = True
        if variance == "Homogeneous" and not any(equal):
            violations.append("variance card says Homogeneous while every check said otherwise")
        if variance == "Heterogeneous" and all(equal):
            violations.append("variance card says Heterogeneous while every check said otherwise")

    # "(after transformation)" is a claim that one was applied.
    transformation = str(results.get("transformation") or "").lower()
    applied = bool(transformation and transformation not in ("none", "no further"))
    if "after transformation" in normality and not applied:
        fired = True
        violations.append(
            f"normality card credits a transformation; the result records {results.get('transformation')!r}")
    return fired


# --- the seam: the panel and the page must name the same analysis ---------------

def _oracle_cockpit_agrees_with_report(summary, context, results, violations) -> bool:
    """The model on the card must be the model the report names.

    Both are rendered from the same result, by different code, and the defect
    that started this file was exactly a disagreement between them: the panel
    said One-Way ANOVA while every line of the report said Welch's. Neither side
    could see it alone.
    """
    report_text = results.get("_report_text")
    if not report_text:
        return False
    shown = _line(summary.get("context_design"), "Model:")
    if not shown:
        return False
    if shown not in report_text:
        violations.append(
            f"cockpit names the model {shown!r}, which appears nowhere in the report")
    return True


ORACLES = (
    ("cockpit_no_non_number", _oracle_no_number_that_is_not_one),
    ("cockpit_design_names_what_ran", _oracle_design_names_what_ran),
    ("cockpit_p_round_trips", _oracle_p_value_round_trips),
    ("cockpit_effect_round_trips", _oracle_effect_size_round_trips),
    ("cockpit_n_round_trips", _oracle_sample_size_round_trips),
    ("cockpit_groups_round_trip", _oracle_groups_round_trip),
    ("cockpit_posthoc_line_true", _oracle_posthoc_line_is_true),
    ("cockpit_assumptions_agree", _oracle_assumption_cards_agree),
    ("cockpit_agrees_with_report", _oracle_cockpit_agrees_with_report),
)


def cockpit_target(result: Any, context: Any):
    """The (context, result) pair the cockpit would actually have rendered.

    A blocked, cancelled or errored run returns before the cards are built --
    the widget blanks them itself -- so there is nothing here to check and
    checking anyway would invent findings about a panel the user never saw.

    A multi-dataset run renders the LEAD dataset and only the lead, which is
    what the window does: the other datasets reach the reader through the
    combined report, not through this panel. Judging all of them would file the
    same defect once per dataset and describe a surface that was never shown.
    """
    if not isinstance(result, dict) or not isinstance(context, dict):
        return None
    if result.get("blocked") is True or result.get("cancelled") is True or result.get("error"):
        return None

    if result.get("type") == "multi_dataset_analysis":
        per_dataset = result.get("results") or {}
        if not isinstance(per_dataset, dict) or not per_dataset:
            return None
        lead_name = next(iter(per_dataset))
        lead = per_dataset[lead_name]
        if (not isinstance(lead, dict) or lead.get("blocked") is True
                or lead.get("cancelled") is True):
            # A cancelled dataset lands in ``results`` and is counted among the
            # successful ones by the multi-dataset wrapper, so "it is in the
            # results dict" is not evidence that anything ran. The window aborts
            # the whole batch on a cancel and never renders the panel at all.
            return None
        lead_context = dict(context)
        lead_context["current_dv"] = lead_name
        return lead_context, lead
    return context, result


def check_cockpit(context: Any, results: Any, report_text: str = "") -> Tuple[List[str], List[str]]:
    """Check every claim the panel makes. Returns ``(violations, oracles_fired)``."""
    violations: List[str] = []
    fired: List[str] = []
    if not isinstance(results, dict):
        return [f"cockpit result is not a dict: {type(results).__name__}"], []

    try:
        summary = build_summary(context, results)
    except Exception as exc:
        return [f"the cockpit summary could not be built at all: "
                f"{type(exc).__name__}: {exc}"], []

    if report_text:
        results = dict(results)
        results["_report_text"] = report_text

    for name, oracle in ORACLES:
        try:
            if oracle(summary, context or {}, results, violations):
                fired.append(name)
        except Exception as exc:  # an oracle that throws is a finding about itself
            violations.append(f"cockpit oracle {name} raised {type(exc).__name__}: {exc}")
    return violations, fired
