"""Properties of a written report, checkable without a browser.

The exported HTML carries its own machine-readable payloads -- ``pd-data-*``
script tags plus the embedded Plotly figures -- so the invariants that matter
can be verified by parsing the file, at a cost that survives hundreds of runs.
That is why this lives in ``src`` rather than in ``fuzzing``: the same checks
serve the fuzzer and the export path itself, and two copies of a check is the
failure this repository keeps paying for.

Living in ``src`` is not the same as running for everyone. The export path runs
these only when ``BIOMEDSTATX_SELFCHECK=1`` is set before launch -- see
``selfcheck_enabled`` below for why an installed copy is left alone.

What is here is exactly the report/export layer: the structure of the file, the
numbers it renders, the significance layer it draws, the axis order and the
font. Nothing here opens a browser (the rendered figure is the visual fuzzer's
business) and nothing here inspects live window state (that is the import
fuzzer's, and it would mean reaching into the import path rather than reading
its result).

Each check takes ``(report, result, violations)``, appends to ``violations`` and
returns whether it applied at all. False means "precondition absent", not
"passed" -- the distinction is what makes a coverage summary honest, and it is
also what lets the sidecar say "not applicable" instead of a silent pass.
"""
from __future__ import annotations

import html as html_module
import json
import math
import os
import re
from typing import Any, Dict, List

# Sections every report carries unconditionally, on both the standard and the
# clinical export path -- they write through different call sites and were
# checked against both.
REQUIRED_SECTIONS = (
    "hd-results", "hd-assumptions", "hd-charts",
    "hd-decision", "hd-descriptive", "hd-methods", "hd-raw",
)

# The pairwise section is conditional in the template: no comparisons, no table.
# That is correct for a blocked run or an omnibus without a post-hoc, so it is
# demanded only when the result actually holds comparisons.
PAIRWISE_SECTION = "hd-pairwise"

# Payloads the interactive figure builder reads. The builder itself only exists
# for designs with plottable groups (correlation and regression have none), so a
# missing payload counts only where the builder is present -- and whether it
# should have been present is its own check.
REQUIRED_PAYLOADS = (
    "pd-data-plot", "pd-data-order", "pd-data-pairs", "pd-data-stats",
    "pd-data-style", "pd-data-paired-lines",
)

GROUP_CHART_DIV = "biomedstatx-group-chart"

_PAYLOAD_RE = re.compile(
    r'<script[^>]*id="(pd-data-[a-z-]+)"[^>]*>(.*?)</script>', re.S)
_NEWPLOT_RE = re.compile(r"Plotly\.newPlot\(\s*")
_LETTER_RE = re.compile(r"^<b>([a-zA-Z]+)</b>$")
_DECODER = json.JSONDecoder()


class _Report:
    """The parsed report: payloads, the group figure, and the raw text."""

    def __init__(self, path: str):
        self.path = path
        self.text = ""
        # The same file with entities resolved. Structure is read from ``text``
        # (a payload must parse as written); rendered strings are looked for
        # here, because "p < 0.001" reaches the file as "p &lt; 0.001" and a
        # literal search of the source would miss every bound in the report.
        self.rendered = ""
        self.payloads: Dict[str, Any] = {}
        self.unparseable: List[str] = []
        self.figure: Dict[str, Any] = {}
        self.sig_mode = ""

    @property
    def order(self) -> List[str]:
        value = self.payloads.get("pd-data-order")
        return [str(v) for v in value] if isinstance(value, list) else []

    @property
    def pairs(self) -> List[dict]:
        value = self.payloads.get("pd-data-pairs")
        return [p for p in value if isinstance(p, dict)] if isinstance(value, list) else []


def _decode_at(text: str, index: int):
    """Decode one JSON value starting at ``index``, skipping leading whitespace."""
    while index < len(text) and text[index] in " \t\r\n,":
        index += 1
    return _DECODER.raw_decode(text, index)


def _extract_figure(text: str, div_id: str) -> Dict[str, Any]:
    """Pull ``{"data": [...], "layout": {...}}`` out of a Plotly.newPlot call.

    Plotly writes its arguments as JSON literals, so the decoder can read them
    straight out of the script rather than a regex guessing where the braces
    balance -- the annotations carry HTML with braces of their own.
    """
    for match in _NEWPLOT_RE.finditer(text):
        try:
            name, end = _decode_at(text, match.end())
        except ValueError:
            continue
        if name != div_id:
            continue
        try:
            data, end = _decode_at(text, end)
            layout, _ = _decode_at(text, end)
        except ValueError:
            return {}
        return {"data": data, "layout": layout}
    return {}


def load_report(path: str) -> _Report:
    report = _Report(path)
    with open(path, encoding="utf-8", errors="replace") as fh:
        report.text = fh.read()
    report.rendered = html_module.unescape(report.text)

    for pid, body in _PAYLOAD_RE.findall(report.text):
        try:
            report.payloads[pid] = json.loads(body.strip())
        except Exception:
            report.unparseable.append(pid)

    report.figure = _extract_figure(report.text, GROUP_CHART_DIV)
    mode = re.search(r'const SIG_MODE="([a-z]*)"', report.text)
    report.sig_mode = mode.group(1) if mode else ""
    return report


def _letters_in_figure(report: _Report) -> Dict[int, str]:
    """Server-rendered compact letters, keyed by the group's x position.

    Bracket stars are annotations too; only alphabetic text is a letter, so the
    two layers stay distinguishable without depending on the mode flag.
    """
    annotations = (report.figure.get("layout") or {}).get("annotations") or []
    letters: Dict[int, str] = {}
    for note in annotations:
        if not isinstance(note, dict):
            continue
        found = _LETTER_RE.match(str(note.get("text", "")))
        x = note.get("x")
        if found and isinstance(x, (int, float)) and float(x).is_integer():
            letters[int(x)] = found.group(1)
    return letters


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _usable_resolution(value: Any) -> bool:
    return _is_number(value) and math.isfinite(value) and 0.0 < value < 1.0


# --- individual oracles -------------------------------------------------------
# Each takes (report, result, violations) and returns True if it applied. A
# False means "precondition absent", not "passed" -- the difference is what the
# coverage summary is for.


def _plottable_groups(result) -> int:
    """Groups the report could draw: at least one value that is a real number.

    Mirrors the exporter's own admission rule, which is what decides whether the
    figure builder is emitted at all.
    """
    raw = result.get("raw_data") or result.get("samples") or {}
    if not isinstance(raw, dict):
        return 0
    count = 0
    for values in raw.values():
        try:
            numeric = [float(v) for v in values
                       if _is_number(v) or (isinstance(v, str) and v.strip())]
        except (TypeError, ValueError):
            continue
        if any(math.isfinite(v) for v in numeric):
            count += 1
    return count


def _oracle_payloads_parse(report, result, violations) -> bool:
    for pid in report.unparseable:
        violations.append(f"payload {pid} is not valid JSON")
    if not report.payloads and not report.unparseable:
        return False  # no figure builder in this report; its own oracle covers that
    for pid in REQUIRED_PAYLOADS:
        if pid not in report.payloads and pid not in report.unparseable:
            violations.append(f"payload {pid} missing from the report")
    return True



def _oracle_sections_present(report, result, violations) -> bool:
    for section in REQUIRED_SECTIONS:
        if f'id="{section}"' not in report.text:
            violations.append(f"section {section} missing from the report")

    comparisons = [c for c in (result.get("pairwise_comparisons") or [])
                   if isinstance(c, dict) and _is_number(c.get("p_value"))]
    if comparisons and f'id="{PAIRWISE_SECTION}"' not in report.text:
        violations.append(
            f"result holds {len(comparisons)} pairwise comparison(s) but the "
            "report has no pairwise section"
        )
    return True


def _oracle_result_number_is_rendered(report, result, violations) -> bool:
    """The headline p-value must appear as text, not as an N/A placeholder."""
    from export.report_formatting import _FormattingMixin

    p = result.get("p_value")
    if result.get("blocked") is True or not _is_number(p) or not math.isfinite(p):
        return False
    shown = _FormattingMixin._format_p_value(p, result.get("p_value_resolution"))
    if shown not in report.rendered:
        violations.append(
            f"p_value {p!r} is finite but its rendering {shown!r} is absent from the report"
        )
    return True


def _oracle_p_precision_capped(report, result, violations) -> bool:
    """A simulated p-value must not be printed with precision it does not have.

    The engines that estimate rather than derive their p-values declare a
    resolution. Where the estimate sits at or below it, the report has to show a
    bound; printing the figure would be a measurement the method never made.

    Scoped deliberately. Searching the whole document for the *unbounded*
    rendering finds any number that happens to round to the same three
    significant figures -- in a two-level repeated-measures design the analytic
    omnibus p and the estimated contrast are the same quantity, so the correct
    full display of the former read as the forbidden display of the latter. The
    pairwise side is therefore checked row by row, against the row builder the
    report itself renders from.
    """
    from export.report_formatting import _FormattingMixin
    from export.report_stat_rows import _StatRowsMixin

    fired = False

    resolution = result.get("p_value_resolution")
    p = result.get("p_value")
    if (_usable_resolution(resolution) and _is_number(p) and math.isfinite(p)
            and p <= resolution):
        fired = True
        shown = _FormattingMixin._format_p_value(p, resolution)
        if shown not in report.rendered:
            violations.append(
                f"the headline p={p!r} is at or below its resolution {resolution!r} "
                f"but the bound {shown!r} is not in the report"
            )

    comparisons = [c for c in (result.get("pairwise_comparisons") or [])
                   if isinstance(c, dict)]
    bounded = [c for c in comparisons
               if _usable_resolution(c.get("p_value_resolution"))
               and _is_number(c.get("p_value")) and math.isfinite(c["p_value"])
               and c["p_value"] <= c["p_value_resolution"]]
    if not bounded:
        return fired

    fired = True
    try:
        rows = _StatRowsMixin._build_pairwise_rows(result) or []
    except Exception as exc:
        violations.append(f"pairwise rows could not be built: {type(exc).__name__}: {exc}")
        return fired

    by_pair = {}
    for row in rows:
        key = tuple(sorted((str(row.get("group1")), str(row.get("group2")))))
        by_pair[key] = row

    for comparison in bounded:
        key = tuple(sorted((str(comparison.get("group1")), str(comparison.get("group2")))))
        row = by_pair.get(key)
        if row is None:
            violations.append(f"comparison {key} carries a resolution but has no report row")
            continue
        expected = _FormattingMixin._format_p_value(
            comparison["p_value"], comparison["p_value_resolution"])
        if str(row.get("p_value")) != expected:
            violations.append(
                f"comparison {key} p={comparison['p_value']!r} is unresolvable at "
                f"{comparison['p_value_resolution']!r} but its row shows "
                f"{row.get('p_value')!r} instead of {expected!r}"
            )
        elif expected not in report.rendered:
            violations.append(
                f"comparison {key} is bounded to {expected!r} but that never "
                "reaches the report"
            )
    return fired


def _oracle_letters_gate_is_complete(report, result, violations) -> bool:
    """Letters may only be drawn from a complete comparison matrix.

    A letter makes a statement about every other group at once. Given a partial
    set -- a many-to-one post-hoc, or a hand-picked selection from the pair
    dialog -- the untested pairs would silently read as "not different".
    """
    if report.sig_mode != "letters":
        return False
    k = len(report.order)
    required = k * (k - 1) // 2
    tested = {tuple(sorted((str(p.get("group1")), str(p.get("group2")))))
              for p in report.pairs
              if str(p.get("group1")) in report.order and str(p.get("group2")) in report.order}
    if len(tested) < required:
        violations.append(
            f"letters drawn for {k} groups from {len(tested)} comparisons, "
            f"{required} required"
        )
    return True


def _oracle_letters_match_the_pairwise_table(report, result, violations) -> bool:
    """Shared letter must mean "not significantly different", both ways round.

    Groups that differ must not share a letter -- the failure that made the
    superseded implementation hide a real difference -- and groups that do not
    differ must share one, or the plot invents a difference the test never found.
    """
    letters = _letters_in_figure(report)
    if not letters or report.sig_mode != "letters":
        return False

    order = report.order
    missing = [g for i, g in enumerate(order) if i not in letters]
    if missing:
        violations.append(f"groups without a letter: {missing}")

    for pair in report.pairs:
        g1, g2 = str(pair.get("group1")), str(pair.get("group2"))
        if g1 not in order or g2 not in order:
            continue
        a = set(letters.get(order.index(g1), ""))
        b = set(letters.get(order.index(g2), ""))
        if not a or not b:
            continue
        if pair.get("significant") and (a & b):
            violations.append(
                f"{g1} vs {g2} is significant but they share letter(s) {sorted(a & b)}"
            )
        if not pair.get("significant") and not (a & b):
            violations.append(
                f"{g1} vs {g2} is not significant but shares no letter "
                f"({sorted(a)} vs {sorted(b)})"
            )
    return True


def _oracle_brackets_mode_has_no_letters(report, result, violations) -> bool:
    """The converse gate: no letters where the matrix was incomplete."""
    if report.sig_mode != "brackets":
        return False
    letters = _letters_in_figure(report)
    if letters:
        violations.append(
            f"significance mode is 'brackets' but the chart carries letters {letters}"
        )
    return True



def _oracle_axis_order_is_ranked(report, result, violations) -> bool:
    """The axis order must be the ranked one, not whatever the labels sort to.

    Ranking is idempotent, so re-ranking the order the report shows has to
    return it unchanged. A renderer that sorted on its own shows up here as a
    disagreement rather than as a plot nobody looked at.
    """
    from core.level_order import natural_order

    order = report.order
    if len(order) < 2:
        return False
    ranked = [str(v) for v in natural_order(order)]
    if ranked != order:
        violations.append(f"axis order {order} is not the ranked order {ranked}")
    return True


def _oracle_one_plot_font(report, result, violations) -> bool:
    """One font family across every figure in the report.

    The charts are built by two renderers and several plot-type branches; a
    second family creeping into one of them is the visible half of the drift the
    duplicated layers keep producing.
    """
    from visualization import style_tokens

    families = set()
    for match in re.finditer(r'"family":\s*("(?:[^"\\]|\\.)*")', report.text):
        try:
            families.add(json.loads(match.group(1)))
        except Exception:
            continue
    if not families:
        return False
    strays = sorted(f for f in families if f != style_tokens.FONT_FAMILY_STACK)
    if strays:
        violations.append(f"figures use font families besides the report stack: {strays}")
    return True




# The report/export checks, in the order a reader meets their subject: the file
# parses, it has its sections, the numbers reach the page, the significance
# layer is honest, and the axis and font are what they should be.
REPORT_CHECKS = (
    ("payloads_parse", _oracle_payloads_parse),
    ("sections_present", _oracle_sections_present),
    ("result_number_rendered", _oracle_result_number_is_rendered),
    ("p_precision_capped", _oracle_p_precision_capped),
    ("letters_gate_complete", _oracle_letters_gate_is_complete),
    ("letters_match_pairs", _oracle_letters_match_the_pairwise_table),
    ("brackets_have_no_letters", _oracle_brackets_mode_has_no_letters),
    ("axis_order_ranked", _oracle_axis_order_is_ranked),
    ("one_plot_font", _oracle_one_plot_font),
)

# What each check is about, in words that name a property of the report and
# never a value from the data. The sidecar prints these; the fuzzer prints the
# violation text instead, because there the data is the point.
CHECK_SUBJECTS = {
    "payloads_parse": "figure-builder payloads present and valid JSON",
    "sections_present": "every report section rendered",
    "result_number_rendered": "the headline number reached the page",
    "p_precision_capped": "estimated p-values printed within their resolution",
    "letters_gate_complete": "compact letters drawn only from a complete comparison matrix",
    "letters_match_pairs": "compact letters agree with the pairwise table",
    "brackets_have_no_letters": "no letters left on a chart drawn in bracket mode",
    "axis_order_ranked": "group axis in ranked order",
    "one_plot_font": "a single font family across the figures",
}


# --- the sidecar ----------------------------------------------------------------

# The sidecar is a developer's instrument, and it is off unless asked for.
#
# There is no channel back from an installed copy. A sidecar written on a
# researcher's disk describes a report to someone who did not write the checks,
# does not know what "letters_gate_complete" means, and will never send the file
# to anyone who does -- so the one thing it reliably produces is a file
# appearing next to somebody's unpublished experimental data without their
# having asked for it. No visible UI is not the same as asked for.
#
# Set BIOMEDSTATX_SELFCHECK=1 before launching to turn it on. An environment
# variable set before start is a deliberate act; double-clicking the app in the
# Finder can never be one. This is also what makes the checks usable against
# real research data rather than only against generated cases -- the fuzzers
# invent their data, and a real repeated-measures design has shapes no generator
# thinks of.
SELFCHECK_ENV_VAR = "BIOMEDSTATX_SELFCHECK"
_ENABLED_VALUES = frozenset({"1", "true", "yes", "on"})

# Written next to the report, and only when a check actually failed. A file
# beside every export would be noise; a file that appears only when something is
# off is a signal. It never touches the report, never blocks the export, and
# never surfaces in the UI on its own.
SIDECAR_SUFFIX = "_selfcheck.txt"


def selfcheck_enabled() -> bool:
    """Whether the export path should check the report it has just written."""
    return os.environ.get(SELFCHECK_ENV_VAR, "").strip().lower() in _ENABLED_VALUES


def run_report_checks(path: str, results: Any):
    """Run the report/export checks against a written report.

    Returns ``(check_name, verdict, count)`` per check, where verdict is one of
    ``pass`` / ``fail`` / ``n-a`` / ``error``. "n-a" is not a pass: a check whose
    precondition is absent has said nothing, and reporting that as a pass is how
    a green run stops meaning anything.
    """
    report = load_report(path)
    outcome = []
    for name, check in REPORT_CHECKS:
        violations = []
        try:
            applied = check(report, results if isinstance(results, dict) else {}, violations)
        except Exception:
            # A check that cannot run is a fact about the check, not about the
            # report, and it must not take the export down with it.
            outcome.append((name, "error", 0))
            continue
        if not applied:
            outcome.append((name, "n-a", 0))
        elif violations:
            outcome.append((name, "fail", len(violations)))
        else:
            outcome.append((name, "pass", 0))
    return outcome


def _sidecar_text(report_path: str, outcome) -> str:
    """Flags and counts only.

    Deliberately free of values from the data: this file describes properties of
    the report, it is not a second export of what the report contains. A count
    and the name of the property is enough to know where to look; the report
    itself is right next to it.
    """
    failed = [row for row in outcome if row[1] in ("fail", "error")]
    lines = [
        "BioMedStatX report self-check",
        f"report: {os.path.basename(report_path)}",
        "",
        f"{len(failed)} of {len(outcome)} checks did not pass. This file is "
        "informational: the report beside it was written normally and is "
        "unaffected.",
        "",
    ]
    width = max(len(name) for name, _, _ in outcome)
    for name, verdict, count in outcome:
        subject = CHECK_SUBJECTS.get(name, "")
        suffix = f"  ({count} finding{'s' if count != 1 else ''})" if count else ""
        lines.append(f"  {name:<{width}}  {verdict:<5}{suffix}  {subject}")
    lines.append("")
    return "\n".join(lines)


def write_sidecar(report_path: str, results: Any):
    """Check a written report; write a sidecar only if something did not pass.

    Returns the sidecar path, or None when the self-check is not enabled, when
    everything passed, or when the check itself could not run. Never raises:
    this runs after the export has already succeeded, and a diagnostic that can
    break the thing it diagnoses is worse than no diagnostic at all.

    The gate is checked here rather than at the call site so that it holds for
    every caller, present and future -- a second call site added later would
    otherwise be silently ungated. It is checked first so that "off" costs
    nothing: not the read, not the parse.
    """
    import logging

    if not selfcheck_enabled():
        return None

    log = logging.getLogger(__name__)
    try:
        outcome = run_report_checks(report_path, results)
    except Exception as exc:
        log.debug("report self-check could not run for %r: %s", report_path, exc)
        return None

    failed = [row for row in outcome if row[1] in ("fail", "error")]
    if not failed:
        return None

    sidecar = os.path.splitext(report_path)[0] + SIDECAR_SUFFIX
    try:
        with open(sidecar, "w", encoding="utf-8") as handle:
            handle.write(_sidecar_text(report_path, outcome))
    except Exception as exc:
        log.debug("could not write report self-check sidecar %r: %s", sidecar, exc)
        return None

    log.warning("report self-check: %s did not pass for %s (see %s)",
                ", ".join(name for name, _, _ in failed),
                os.path.basename(report_path), os.path.basename(sidecar))
    return sidecar
