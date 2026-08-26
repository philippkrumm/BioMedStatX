"""Visual fuzzer worker — one seed, from analysis to a rendered, driven figure.

The seed produces a real report through the same case generator the analysis
fuzzer uses, and the report is then opened in headless Chromium and *used*:
plot types switched, controls changed, presets pressed, the figure exported.

Why a browser at all. The exported HTML is the product a reader receives, and
everything about it that can break after export -- a script that fails to
parse, a figure that renders no trace, a label clipped by its container, a
Download button that yields an empty file -- is invisible to any check that
reads the file as text. That gap shipped a broken figure builder once already.

Isolation is the same bargain as the other workers: the orchestrator runs this
module as a child process per seed, so a segfault in the analysis half or a
browser that wedges costs one seed.

Exit codes:
  0  everything rendered and every oracle passed
  2  oracle violation
  3  uncaught Python exception
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, os.path.join(_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# How many reports of a multi-dataset seed get the full browser treatment.
# Every report is still opened and load-checked; the interaction plan is the
# expensive part and three of them is enough to keep a seed under a minute.
_MAX_DRIVEN = 3
_SETTLE_MS = 120
_REPLOT_TIMEOUT_MS = 1500
# A control that cannot be operated is a finding, but it must not cost the
# default half-minute of actionability waiting to say so.
_ACTION_TIMEOUT_MS = 4000


# --- what the page reports about itself -------------------------------------

_SURFACE_JS = """() => {
  const panel = document.getElementById('plot-designer-panel');
  const controls = [];
  if (panel) {
    panel.querySelectorAll('input, select').forEach(el => {
      if (!el.id) return;
      const owner = el.closest('[data-tab-panel]');
      controls.push({
        id: el.id,
        tag: el.tagName.toLowerCase(),
        type: (el.type || '').toLowerCase(),
        options: el.tagName.toLowerCase() === 'select'
          ? Array.from(el.options).filter(o => !o.disabled).map(o => o.value) : null,
        min: el.min === '' ? null : el.min,
        max: el.max === '' ? null : el.max,
        tab: owner ? owner.getAttribute('data-tab-panel') : null
      });
    });
  }
  return {
    controls: controls,
    plot_types: Array.from(document.querySelectorAll('#pd-plot-type option')).map(o => o.value),
    has_designer: !!panel
  };
}"""

# Everything an oracle needs, read off the rendered page in one round trip.
_SNAPSHOT_JS = """() => {
  const payload = (id) => {
    const n = document.getElementById(id);
    if (!n) return null;
    try { return JSON.parse(n.textContent); } catch (e) { return 'UNPARSEABLE'; }
  };
  const figures = Array.from(document.querySelectorAll('.js-plotly-plot')).map(d => {
    const r = d.getBoundingClientRect();
    return {id: d.id, traces: (d.data || []).length,
            svgs: d.querySelectorAll('svg.main-svg').length,
            w: Math.round(r.width), h: Math.round(r.height)};
  });

  const d = document.getElementById('pd-plot');
  let pd = null;
  if (d) {
    const box = d.getBoundingClientRect();
    // Text Plotly draws for the reader. The modebar and hover layers are left
    // out: they are chrome, and they legitimately sit outside the axes.
    const sel = '.xtick text, .ytick text, .g-xtitle, .g-ytitle, .gtitle,'
              + ' .infolayer .legend text, .annotation-text';
    const overflow = [];
    d.querySelectorAll(sel).forEach(t => {
      const r = t.getBoundingClientRect();
      if (!r.width && !r.height) return;
      const over = Math.max(box.left - r.left, r.right - box.right,
                            box.top - r.top, r.bottom - box.bottom);
      // Plotly leaves the tick <text> itself unclassed; the class that names
      // the layer sits on the parent, and without it every finding would read
      // "(None)" and say nothing about which part of the figure overflowed.
      if (over > 1) overflow.push({
        text: (t.textContent || '').slice(0, 30),
        cls: t.getAttribute('class')
             || (t.parentNode && t.parentNode.getAttribute
                 ? t.parentNode.getAttribute('class') : null),
        over: Math.round(over)});
    });

    const layout = d.layout || {};
    const shapes = layout.shapes || [];
    const strip = (s) => String(s || '').replace(/<[^>]*>/g, '').trim();
    const anns = (layout.annotations || []).map(a => strip(a.text));
    const ticks = Array.from(d.querySelectorAll('.xtick text'))
      .map(t => ({t: t.textContent, x: t.getBoundingClientRect().left}))
      .sort((a, b) => a.x - b.x).map(o => o.t);
    // Forest and the horizontal layouts put the groups on the y axis and a
    // numeric scale on x, so the order oracle has to be told which axis is
    // carrying the categories rather than assuming it is always x.
    const ticksY = Array.from(d.querySelectorAll('.ytick text'))
      .map(t => ({t: t.textContent, y: t.getBoundingClientRect().top}))
      .sort((a, b) => a.y - b.y).map(o => o.t);
    const mode = document.getElementById('pd-significance-mode');
    pd = {
      traces: (d.data || []).length,
      w: Math.round(box.width), h: Math.round(box.height),
      overflow: overflow,
      // Only the significance brackets are drawn in data coordinates in this
      // colour; reference lines and the forest zero line use "paper" and a
      // lighter alpha, so a user's reference line is never counted as a bracket.
      bracket_shapes: shapes.filter(s => ((s.line || {}).color === 'rgba(22,49,58,0.65)')
                                          && s.xref === 'x' && s.yref === 'y').length,
      other_shapes: shapes.length,
      letter_annotations: anns.filter(t => /^[a-z]+$/.test(t) && t !== 'ns').length,
      annotations: anns.length,
      categories: ticks,
      categories_y: ticksY,
      sig_mode: mode ? mode.value : null,
      sig_disabled: mode ? !!mode.disabled : true,
      warning: (document.getElementById('pd-warning') || {}).textContent || ''
    };
  }
  return {figures: figures, pd: pd,
          designer: !!document.getElementById('plot-designer-panel'),
          has_plot_payload: !!payload('pd-data-plot'),
          payload_order: payload('pd-data-order'),
          replots: window.__bmsx_replots || 0};
}"""

# Counts completed redraws so an action can be waited on precisely instead of
# guessed at with a sleep.
_INSTRUMENT_JS = """() => {
  const d = document.getElementById('pd-plot');
  window.__bmsx_replots = 0;
  if (d && typeof d.on === 'function') {
    d.on('plotly_afterplot', () => { window.__bmsx_replots++; });
  }
}"""


def _build_reports(seed: int, workdir: str):
    """Run the seed's analysis and return the HTML files it wrote."""
    from fuzzing._worker import _neutralize_dialogs
    from fuzzing.generators import build_case, case_to_analyze_kwargs

    _neutralize_dialogs(seed)
    case = build_case(seed)

    import pandas as pd
    dummy = os.path.join(workdir, "dummy.xlsx")
    pd.DataFrame({"a": [1]}).to_excel(dummy, index=False)
    kwargs = case_to_analyze_kwargs(case, dummy, os.path.join(workdir, "out"))

    from analysis.analysis_core import AnalysisManager
    result = AnalysisManager.analyze(**kwargs)
    return case, result, sorted(glob.glob(os.path.join(workdir, "*.html")))


class _PageRun:
    """One report in one tab: load, drive, check."""

    def __init__(self, page, path, seed, drive: bool):
        self.page = page
        self.path = path
        self.seed = seed
        self.drive = drive
        self.console = []
        self.page_errors = []
        self._seen = 0
        self.violations = []
        self.fired = []
        self.warnings = []
        self.stages = 0
        # Which significance form was actually on screen. Firing counts cannot
        # answer that: the mode oracle fires just as happily on a figure that
        # never left brackets, and "letters were never rendered" is exactly the
        # kind of gap a green run hides.
        self.modes_seen = []
        # Set once the user has pressed move-up / move-down: from then on the
        # axis is allowed to differ from the report's order, because differing
        # is what the button does.
        self.order_user_modified = False

    # -- plumbing ----------------------------------------------------------
    def _drain(self):
        """Errors that appeared since the last stage, so each stage owns its own."""
        fresh = self.console[self._seen:]
        self._seen = len(self.console)
        errs, self.page_errors = self.page_errors, []
        return fresh, errs

    def _snapshot(self, stage):
        from fuzzing.visual_oracles import check_stage
        snap = self.page.evaluate(_SNAPSHOT_JS)
        fresh_console, fresh_errors = self._drain()
        snap["stage"] = stage
        snap["order_user_modified"] = self.order_user_modified
        snap["console_errors"] = fresh_console
        snap["page_errors"] = fresh_errors
        violations, fired = check_stage(snap)
        self.violations += violations
        for name in fired:
            if name not in self.fired:
                self.fired.append(name)
        for text in snap.get("warnings_seen") or []:
            if text not in self.warnings:
                self.warnings.append(text)
        pd = snap.get("pd") or {}
        if pd.get("traces") and pd.get("sig_mode"):
            drawn = pd["sig_mode"]
            # What the select says and what is drawn can differ when the layer
            # is gated, so the mode is recorded together with whether anything
            # of it reached the figure.
            if drawn == "letters" and not pd.get("letter_annotations"):
                drawn = "letters(empty)"
            elif drawn == "brackets" and not pd.get("bracket_shapes"):
                drawn = "brackets(empty)"
            if drawn not in self.modes_seen:
                self.modes_seen.append(drawn)
        self.stages += 1
        return snap

    def _operable(self, selector) -> bool:
        """Can a user work this control right now?

        A control the UI has deliberately switched off, or hidden along with its
        section, is not a target: the gates that disable letters, or hide the
        error-bar rows for a plot type that has none, are behaviour under test
        elsewhere and a user cannot reach past them either. Playwright's own
        visibility check is asked as well as the disabled flag, because an
        element can be present, enabled and still have no box.
        """
        if self.page.eval_on_selector(selector, "el => !!el.disabled"):
            return False
        # ``.first`` because the group move buttons exist once per group and a
        # bare locator is strict about that, while every other call here --
        # query_selector, eval_on_selector, click -- already acts on the first
        # match. Fuzzing the first group's arrow is the same action either way.
        return self.page.locator(selector).first.is_visible()

    def _act(self, fn):
        """Perform an action and wait for the redraw it should have caused.

        Not every change forces a redraw (setting a select to the value it
        already had does not), so a timeout here is normal and not a finding --
        the oracles judge the resulting figure, not the event.
        """
        before = self.page.evaluate("() => window.__bmsx_replots || 0")
        fn()
        try:
            self.page.wait_for_function(
                "before => (window.__bmsx_replots || 0) > before",
                arg=before, timeout=_REPLOT_TIMEOUT_MS)
        except Exception:
            pass
        self.page.wait_for_timeout(_SETTLE_MS)

    def _reveal(self, selector):
        """Open everything between the panel and this control, as a user must.

        Two layers hide a control: the tab it lives on, and -- for the
        reference-line block -- a collapsed ``<details>``. Both were reported as
        findings before they were handled: an unopened ``<details>`` leaves its
        inputs in the DOM and enabled, so the check for ``disabled`` waved them
        through and the click then sat in Playwright's actionability wait for
        the full timeout. A user opens the disclosure; so does this.
        """
        tab = self.page.eval_on_selector(
            selector,
            "el => { const p = el.closest('[data-tab-panel]');"
            " return p ? p.getAttribute('data-tab-panel') : null; }")
        if tab:
            self.page.click(f"#pd-tab-bar [data-tab='{tab}']")
        self.page.eval_on_selector(
            selector,
            "el => { let d = el.closest('details');"
            " while (d) { d.open = true; d = d.parentElement"
            "   ? d.parentElement.closest('details') : null; } }")

    # -- the run -----------------------------------------------------------
    def run(self):
        page = self.page
        page.on("console", lambda m: self.console.append(f"{m.type}: {m.text}"[:200])
                if m.type == "error" else None)
        page.on("pageerror", lambda e: self.page_errors.append(str(e)[:300]))
        page.goto("file://" + self.path, wait_until="load")
        page.wait_for_timeout(1200)
        page.evaluate(_INSTRUMENT_JS)
        snap = self._snapshot("load")

        if not (self.drive and snap.get("designer") and (snap.get("pd") or {}).get("traces")):
            return

        from fuzzing.visual_generators import build_plan
        surface = page.evaluate(_SURFACE_JS)
        plan = build_plan(self.seed, surface)
        self.plan_labels = plan.labels

        for plot_type in plan.plot_types:
            self._reveal("#pd-plot-type")
            self._act(lambda t=plot_type: page.select_option("#pd-plot-type", t))
            self._snapshot(f"type:{plot_type}")

        for index, step in enumerate(plan.steps):
            try:
                self._apply(step)
            except Exception as exc:
                # A control that cannot be operated is a finding about the UI --
                # a disabled-but-offered control, or one hidden by its own tab.
                name = step.get("control") or step.get("action")
                self.violations.append(
                    f"[step:{index}] could not operate '{name}': {type(exc).__name__}: "
                    f"{str(exc).splitlines()[0][:120]}")
                continue
            self._snapshot(f"step:{index}:{step.get('control') or step.get('action')}")

        self._download(plan.downloads)

    def _apply(self, step):
        page = self.page
        if step["kind"] == "click":
            selector = _BUTTONS[step["action"]]
            if page.query_selector(selector) is None:
                return
            self._reveal(selector)
            # Same rule as for the inputs: the group move buttons only exist
            # while the group section is shown, and clicking one that is hidden
            # would sit in Playwright's actionability wait until it times out
            # and then be reported as a finding about the product.
            if not self._operable(selector):
                return
            self._act(lambda s=selector: page.click(s, timeout=_ACTION_TIMEOUT_MS))
            if step["action"] in ("group_up", "group_down"):
                self.order_user_modified = True
            return

        selector = "#" + step["control"]
        node = page.query_selector(selector)
        if node is None:
            return
        self._reveal(selector)
        # A control the UI has deliberately switched off, or hidden along with
        # its whole section, is not a target: the gates that disable letters or
        # hide the error-bar rows for a plot type that has none are behaviour
        # under test elsewhere, and a user cannot reach past them either.
        if not self._operable(selector):
            return
        value = step["value"]
        if step.get("type") == "checkbox":
            self._act(lambda: page.set_checked(selector, bool(value), timeout=_ACTION_TIMEOUT_MS))
        elif step.get("tag") == "select":
            options = page.eval_on_selector_all(
                selector + " option", "os => os.filter(o => !o.disabled).map(o => o.value)")
            if value not in options:
                return
            self._act(lambda: page.select_option(selector, value,
                                                 timeout=_ACTION_TIMEOUT_MS))
        elif step.get("type") == "range":
            # A slider has no text to fill. Setting .value and dispatching the
            # events is what dragging the thumb does, and the designer listens
            # for exactly those two.
            self._act(lambda: page.eval_on_selector(
                selector,
                "(el, v) => { el.value = v;"
                " el.dispatchEvent(new Event('input', {bubbles: true}));"
                " el.dispatchEvent(new Event('change', {bubbles: true})); }",
                str(value)))
        else:
            self._act(lambda: page.fill(selector, str(value), timeout=_ACTION_TIMEOUT_MS))

    def _download(self, formats):
        from fuzzing.visual_oracles import check_download
        page = self.page
        for fmt in formats:
            button = f"#pd-download-{fmt}"
            if page.query_selector(button) is None:
                continue
            self._reveal(button)
            target, error = "", ""
            try:
                with page.expect_download(timeout=20000) as download:
                    page.click(button)
                got = download.value
                target = os.path.join(tempfile.mkdtemp(), got.suggested_filename)
                got.save_as(target)
            except Exception as exc:
                error = f"{type(exc).__name__}: {str(exc).splitlines()[0][:120]}"
            violations, fired = check_download(fmt, target, error)
            self.violations += violations
            for name in fired:
                if name not in self.fired:
                    self.fired.append(name)


_BUTTONS = {
    "preset_publication": ".pd-preset-row [data-preset='publication']",
    "preset_poster": ".pd-preset-row [data-preset='poster']",
    "preset_talk": ".pd-preset-row [data-preset='talk']",
    "legend_inside": "#pd-legend-preset-inside-top-right",
    "legend_outside": "#pd-legend-preset-outside-right",
    "legend_bottom": "#pd-legend-preset-bottom-horizontal",
    "group_up": "#pd-node-label-controls button[aria-label='Move up']",
    "group_down": "#pd-node-label-controls button[aria-label='Move down']",
}


def main(seed: int, keep_dir: str = "") -> int:
    verdict = {"seed": seed}
    with tempfile.TemporaryDirectory() as tmp:
        try:
            case, result, reports = _build_reports(seed, tmp)
        except Exception as exc:
            import traceback
            verdict.update({"status": "exception", "phase": "analysis",
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc()[-1200:]})
            print("__FUZZ__" + json.dumps(verdict))
            return 3

        verdict["test"] = case.test_label
        verdict["mutations"] = case.mutations
        verdict["reports"] = len(reports)
        if not reports:
            verdict["status"] = "ok"
            verdict["note"] = "no report written (blocked or errored run)"
            print("__FUZZ__" + json.dumps(verdict))
            return 0

        violations, fired, warnings, labels, modes = [], [], [], [], []
        stages = 0
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            context = browser.new_context(viewport={"width": 1440, "height": 1000},
                                          accept_downloads=True)
            try:
                for index, path in enumerate(reports):
                    page = context.new_page()
                    run = _PageRun(page, path, seed, drive=index < _MAX_DRIVEN)
                    try:
                        run.run()
                    finally:
                        page.close()
                    violations += [f"{os.path.basename(path)} {v}" for v in run.violations]
                    for name in run.fired:
                        if name not in fired:
                            fired.append(name)
                    warnings += [w for w in run.warnings if w not in warnings]
                    modes += [m for m in run.modes_seen if m not in modes]
                    labels += [x for x in getattr(run, "plan_labels", []) if x not in labels]
                    stages += run.stages
            finally:
                browser.close()

        verdict.update({"oracles_fired": fired, "stages": stages, "plan": labels,
                        "designer_warnings": warnings, "significance_rendered": modes})

        if violations and keep_dir:
            try:
                os.makedirs(keep_dir, exist_ok=True)
                kept = []
                for index, path in enumerate(reports[:_MAX_DRIVEN]):
                    suffix = "" if len(reports) == 1 else f"_{index}"
                    target = os.path.join(keep_dir, f"visual_seed_{seed}{suffix}.html")
                    shutil.copyfile(path, target)
                    kept.append(target)
                verdict["reports_kept"] = kept
            except Exception as exc:
                verdict["report_keep_error"] = f"{type(exc).__name__}: {exc}"

    if violations:
        verdict["status"] = "oracle_violation"
        verdict["violations"] = violations[:20]
        verdict["violation_count"] = len(violations)
        print("__FUZZ__" + json.dumps(verdict))
        return 2

    verdict["status"] = "ok"
    print("__FUZZ__" + json.dumps(verdict))
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else ""))
