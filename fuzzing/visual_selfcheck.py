"""Prove the browser oracles can fail.

An oracle that fires on every seed and passes on every seed has told you
nothing, and this repo has now produced four separate ways for a check to be
green without checking anything. The browser oracles are the most exposed of
all of them: they judge a rendered page through two layers of indirection, so
"no violations" is cheap to achieve by accident -- a selector that matches
nothing reports no overflow, and a payload read from the same place the
renderer reads it moves with the renderer.

So each oracle here gets a negative control: a deliberate break in a real
report that it is supposed to catch. The run fails if the baseline is dirty or
if any mutant slips through.

Three of the mutations are only correct because a weaker version was tried
first and proved to be no mutation at all -- one instance of each failure class
this repo has already paid for:

  * turning ``automargin`` off does not clip anything -- Plotly's own
    autoexpand still fits the tick labels -- so the mutation restores the bug
    that actually shipped: a fixed pixel margin that cannot grow;
  * scrambling the ``pd-data-order`` payload changes nothing observable,
    because the designer reads that payload; both sides move together, which
    is the chained-to-its-own-source class. The renderer has to be made to
    re-sort while the payload stays ranked;
  * relaxing the bracket builder's own gate is unreachable in letters mode --
    the dispatcher returns before that builder is called -- so the mutant
    equals the original. Both the dispatch and the gate have to move.

Three controls are not about an oracle at all. "fixed top margin", "refusal
keeps the old figure" and "raincloud log without the guard" each restore a
product bug this fuzzer found on its first real batches, so the run fails if
any of those fixes is quietly reverted.

Two reference reports are built, because no single one carries every property a
control needs: the four-dose report is all-positive and forest-able, the ANCOVA
(seed 46 of the analysis generator) runs negative and cannot draw a forest plot
at all.

Usage:
    python -m fuzzing.visual_selfcheck
"""
from __future__ import annotations

import glob
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


def _build_ancova_report(outdir: str) -> str:
    """A design the fuzzer already produced, kept for what it cannot do.

    Seed 46 of the analysis generator is an ANCOVA whose values run negative and
    whose post-hoc carries no effect sizes, so it has both properties the two
    newest fixes needed: a forest plot that cannot be drawn, and a value range
    that is illegal on a log axis. The four-dose reference below has neither --
    it is all-positive and forest-able -- so a single report cannot carry every
    control.
    """
    from fuzzing._visual_worker import _build_reports
    _, _, reports = _build_reports(46, outdir)
    if not reports:
        raise SystemExit("self-check could not build the ANCOVA reference report")
    return reports[0]


def _build_reference_report(outdir: str) -> str:
    """Four dose groups with an all-pairs post-hoc.

    Four is the smallest size at which compact letters are offered, and the
    labels are chosen so alphabetical order (Dose1, Dose10, Dose2, Vehicle) and
    the ranked order (Vehicle first, then 1 < 2 < 10) disagree -- otherwise the
    ordering control would pass no matter what the renderer did.
    """
    import numpy as np
    import pandas as pd

    from fuzzing._worker import _neutralize_dialogs
    _neutralize_dialogs(4)
    from analysis.statisticaltester import UIDialogManager
    UIDialogManager.select_transformation_dialog = staticmethod(lambda *a, **k: "skip")
    UIDialogManager.select_posthoc_test_dialog = staticmethod(lambda *a, **k: "games_howell")
    UIDialogManager.select_custom_pairs_dialog = staticmethod(
        lambda groups, parent=None: [(a, b) for i, a in enumerate(groups)
                                     for b in groups[i + 1:]])

    rng = np.random.default_rng(4)
    levels = ["Vehicle", "Dose1", "Dose2", "Dose10"]
    frame = pd.DataFrame({
        "Group": sum(([lv] * 9 for lv in levels), []),
        "Value": np.concatenate([rng.normal(10 + 3.0 * i, 1.1, 9)
                                 for i in range(len(levels))]),
    })
    dummy = os.path.join(outdir, "dummy.xlsx")
    pd.DataFrame({"a": [1]}).to_excel(dummy, index=False)

    from analysis.analysis_core import AnalysisManager
    AnalysisManager.analyze(
        file_path=dummy, file_name=os.path.join(outdir, "out"),
        value_cols=["Value"], group_col="Group", groups=levels,
        analysis_context={"injected_df": frame, "factor_columns": ["Group"],
                          "dv_columns": ["Value"], "group_labels": levels,
                          "mode": "single", "inferred_test": "one_way_anova"},
        colors=["#4C72B0", "#DD8452", "#55A868", "#C44E52"],
        hatches=["", "", "", ""], skip_plots=True)
    written = sorted(glob.glob(os.path.join(outdir, "*.html")))
    if not written:
        raise SystemExit("self-check could not build a reference report")
    return written[0]


# --- the mutations ----------------------------------------------------------

def _mut_baseline(text):
    return text


def _mut_script_throws(text):
    return text.replace(
        "</body>",
        "<script>window.setTimeout(function () { "
        "throw new Error('selfcheck-injected failure'); }, 200);</script></body>")


def _mut_fixed_margin(text):
    anchor_flag = "layout[axisKey].automargin = true;"
    anchor_margin = "margin: { l: 64,"
    _require(text, anchor_flag, anchor_margin)
    text = text.replace(anchor_flag, "layout[axisKey].automargin = false;")
    return text.replace(anchor_margin, "margin: { autoexpand: false, l: 2,")


def _mut_renderer_resorts(text):
    anchor = 'var groupOrder = parseJsonNode("pd-data-order", []);'
    _require(text, anchor)
    return text.replace(
        anchor, 'var groupOrder = parseJsonNode("pd-data-order", []).slice().sort();')


def _mut_brackets_in_letters_mode(text):
    dispatch = 'if (state.significanceMode === "letters") return buildLetters(yMin, yMax, idxMap);'
    gate = 'if (state.significanceMode !== "brackets" || !state.visiblePairIds.length) {'
    _require(text, dispatch, gate)
    text = text.replace(dispatch, "if (false) return buildLetters(yMin, yMax, idxMap);")
    return text.replace(gate, "if (!state.visiblePairIds.length) {")


def _mut_fixed_top_margin(text):
    """The pre-fix top margin: fixed at 58 regardless of the title size.

    This is the product bug the visual fuzzer found on its first real batch --
    at the largest title size the control itself offers, the title was clipped
    by the top edge -- so the control here does double duty: it proves the
    clipping oracle sees titles as well as tick labels, and it is the guard
    that the margin formula is not quietly reverted.
    """
    anchor = "t: Math.max(58, Math.round(state.titleSize * 2.2) + 22),"
    _require(text, anchor)
    return text.replace(anchor, "t: 58,")


def _mut_figure_without_traces(text):
    anchor = 'Plotly.react("pd-plot", traces, layout, {'
    _require(text, anchor)
    return text.replace(anchor, 'traces = []; Plotly.react("pd-plot", traces, layout, {')


def _mut_refusal_keeps_old_figure(text):
    """The pre-fix refusal: warn, and leave the previous chart on screen.

    Picking a plot type that cannot be drawn then leaves the reader looking at
    the previous figure -- with its significance letters, at their old
    coordinates -- while the significance control has switched itself to
    "none". That mismatch is what the mode oracle sees.
    """
    anchor = '      Plotly.react("pd-plot", [], {'
    _require(text, anchor)
    return text.replace(anchor, "      if (true) { return; }\n" + anchor)


def _mut_raincloud_log_without_guard(text):
    """The pre-fix raincloud: log applied to values that include zero or less."""
    anchor = "var useLogValues = state.logY && logValuesUsable;"
    _require(text, anchor)
    return text.replace(anchor, "var useLogValues = state.logY;")


def _require(text, *anchors):
    """A mutation whose anchor is absent is not a mutation -- it is the original."""
    for anchor in anchors:
        if anchor not in text:
            raise SystemExit(f"self-check anchor missing, the mutant would equal the "
                             f"original: {anchor[:70]!r}")


# (label, mutation, oracle that must catch it, extra page setup)
# The setup runs after load and before the snapshot: most controls need a long
# axis title and rotated ticks to have anything to clip, the title control needs
# the title turned up to the maximum the UI offers.
_LONG_LABEL = "Concentration of a very long analyte name in plasma [nmol/L]"
_SETUP_LONG_AXIS = (("style", "#pd-y-label", _LONG_LABEL), ("axes", "#pd-x-tick-angle", "90"))
_SETUP_BIG_TITLE = (("style", "#pd-title", "IL-6"), ("style", "#pd-title-size", "42"))

# Two reports, because no single one has every property a control needs: the
# four-dose reference is all-positive and forest-able, the ANCOVA runs negative
# and cannot draw a forest plot at all.
_SETUP_FOREST = (("plot", "#pd-plot-type", "Forest"),)
_SETUP_RAINCLOUD_LOG = (("plot", "#pd-plot-type", "Raincloud"),
                        ("axes", "#pd-log-y", "true"))

# (label, mutation, oracle that must catch it, page setup, reference report)
CONTROLS = (
    ("baseline", _mut_baseline, None, _SETUP_LONG_AXIS, "letters"),
    ("baseline, title at max size", _mut_baseline, None, _SETUP_BIG_TITLE, "letters"),
    ("baseline, refused forest", _mut_baseline, None, _SETUP_FOREST, "ancova"),
    ("baseline, raincloud on log", _mut_baseline, None, _SETUP_RAINCLOUD_LOG, "ancova"),
    ("script throws", _mut_script_throws, "no_script_error", _SETUP_LONG_AXIS, "letters"),
    ("fixed margin, no automargin", _mut_fixed_margin, "labels_not_clipped",
     _SETUP_LONG_AXIS, "letters"),
    ("fixed top margin (pre-fix)", _mut_fixed_top_margin, "labels_not_clipped",
     _SETUP_BIG_TITLE, "letters"),
    ("renderer re-sorts the groups", _mut_renderer_resorts, "designer_keeps_report_order",
     _SETUP_LONG_AXIS, "letters"),
    ("brackets drawn in letters mode", _mut_brackets_in_letters_mode,
     "significance_matches_mode", _SETUP_LONG_AXIS, "letters"),
    ("figure built with no traces", _mut_figure_without_traces,
     "designer_live_when_plottable", _SETUP_LONG_AXIS, "letters"),
    ("refusal keeps the old figure (pre-fix)", _mut_refusal_keeps_old_figure,
     "significance_matches_mode", _SETUP_FOREST, "ancova"),
    ("raincloud log without the guard (pre-fix)", _mut_raincloud_log_without_guard,
     "no_script_error", _SETUP_RAINCLOUD_LOG, "ancova"),
)


def main() -> int:
    from playwright.sync_api import sync_playwright

    from fuzzing._visual_worker import _INSTRUMENT_JS, _SNAPSHOT_JS
    from fuzzing.visual_oracles import ORACLES, check_download

    oracles = dict(ORACLES)
    workdir = tempfile.mkdtemp(prefix="bmsx_selfcheck_")
    ancova_dir = os.path.join(workdir, "ancova")
    os.makedirs(ancova_dir, exist_ok=True)
    sources = {
        "letters": open(_build_reference_report(workdir), encoding="utf-8").read(),
        "ancova": open(_build_ancova_report(ancova_dir), encoding="utf-8").read(),
    }

    failures = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        context = browser.new_context(viewport={"width": 1440, "height": 1000})
        for label, mutate, expected, setup, source in CONTROLS:
            path = os.path.join(workdir, label.replace(" ", "_").replace(",", "")
                                .replace("(", "").replace(")", "") + ".html")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(mutate(sources[source]))

            page = context.new_page()
            console, page_errors = [], []
            page.on("console", lambda m: console.append(f"{m.type}: {m.text}"[:160])
                    if m.type == "error" else None)
            page.on("pageerror", lambda e: page_errors.append(str(e)[:200]))
            page.goto("file://" + path, wait_until="load")
            page.wait_for_timeout(1500)
            page.evaluate(_INSTRUMENT_JS)
            for tab, selector, value in setup:
                try:
                    page.click(f"#pd-tab-bar [data-tab='{tab}']")
                    kind = page.eval_on_selector(
                        selector, "el => el.tagName === 'SELECT' ? 'select' : el.type")
                    if kind == "select":
                        page.select_option(selector, value)
                    elif kind == "checkbox":
                        page.set_checked(selector, value == "true")
                    else:
                        page.fill(selector, value)
                    page.wait_for_timeout(400)
                except Exception as exc:
                    failures.append(f"could not set up '{label}': {selector} "
                                    f"{type(exc).__name__}")
            page.wait_for_timeout(900)

            snap = page.evaluate(_SNAPSHOT_JS)
            snap.update({"stage": label, "console_errors": console, "page_errors": page_errors})
            page.close()

            if expected is None:
                from fuzzing.visual_oracles import check_stage
                violations, _ = check_stage(snap)
                status = "clean" if not violations else "; ".join(v[:90] for v in violations[:2])
                if violations:
                    failures.append(f"baseline is not clean: {status}")
                print(f"{'ok  ' if not violations else 'FAIL'}  {label:32} {status}")
                continue

            found = []
            oracles[expected](snap, found)
            if not found:
                failures.append(f"{expected} did not catch '{label}'")
            print(f"{'ok  ' if found else 'FAIL'}  {label:32} "
                  f"{expected}: {found[0][:88] if found else 'SILENT'}")
        browser.close()

    # The export oracle needs no browser: it judges bytes, so its controls are
    # files rather than pages.
    bytes_dir = tempfile.mkdtemp()
    stub = os.path.join(bytes_dir, "stub.svg")
    open(stub, "w").write("<svg/>")
    wrong = os.path.join(bytes_dir, "wrong.png")
    open(wrong, "wb").write(b"not a png" * 900)
    for label, args in (("missing file", ("png", os.path.join(bytes_dir, "absent.png"))),
                        ("stub svg", ("svg", stub)),
                        ("png that is not a png", ("png", wrong)),
                        ("download failed", ("svg", "", "TimeoutError: no download"))):
        violations, _ = check_download(*args)
        if not violations:
            failures.append(f"check_download did not catch '{label}'")
        print(f"{'ok  ' if violations else 'FAIL'}  {label:32} "
              f"{violations[0][:88] if violations else 'SILENT'}")

    print()
    if failures:
        for line in failures:
            print("  FAILED CONTROL:", line)
        return 1
    mutations = sum(1 for entry in CONTROLS if entry[2])
    print(f"  every browser oracle caught its negative control "
          f"({mutations} page mutations + 4 export cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
