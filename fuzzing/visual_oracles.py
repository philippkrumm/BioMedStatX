"""Invariants for the rendered figure, checked in a real browser.

``html_oracles`` reads the exported file as text: it can tell that a Plotly
payload parses and that the designer markup is present. It cannot tell whether
anything was actually drawn. A report whose plot_designer.js fails to parse
still contains a perfectly valid payload and a perfectly valid ``<script>`` tag
-- that exact bug shipped once and was found by clicking, not by a check.

Everything here therefore asks the *rendered* page: how many traces the figure
carries, whether a tick label sticks out of its container, which significance
layer is on screen, and whether pressing Download produces bytes.

Each oracle takes a stage snapshot (see ``_visual_worker.snapshot``), appends to
``violations``, and returns whether it was in a position to judge at all. An
oracle that never fires proves nothing, so the orchestrator prints the firing
counts and names the silent ones.
"""
from __future__ import annotations

# Note on `bracket_shapes` in the snapshot: Plotly draws several kinds of line
# shape, and only the significance brackets use data coordinates on both axes
# together with the bracket colour. Reference lines and the forest zero line go
# against "paper" in a lighter alpha. That discrimination happens where the
# layout is read -- in `_visual_worker._SNAPSHOT_JS` -- so the literal lives in
# one place rather than being copied here to drift.


def _oracle_no_script_error(snap, violations) -> bool:
    """The page must not throw.

    An uncaught error in plot_designer.js leaves the surrounding report intact
    and the figure silently stale, which is exactly how it stayed unnoticed the
    last time.
    """
    for message in snap.get("page_errors") or []:
        violations.append(f"[{snap['stage']}] uncaught JS error: {message}")
    for message in snap.get("console_errors") or []:
        violations.append(f"[{snap['stage']}] console error: {message}")
    return True


def _refusal_is_announced(snap) -> bool:
    """Is the designer's empty canvas an announced refusal rather than a failure?

    Some plot types cannot be drawn for some designs -- a forest plot needs
    effect sizes per comparison. Refusing is correct; refusing *silently* is
    not, and leaving the previous figure standing is worse still, because the
    plot-type control then names a chart the reader is not looking at. So an
    empty designer canvas is accepted exactly when a warning explains it.
    """
    return bool(((snap.get("pd") or {}).get("warning") or "").strip())


def _oracle_figures_render(snap, violations) -> bool:
    """Every chart in the report has to be drawn, not merely declared."""
    figures = snap.get("figures") or []
    if not figures:
        return False
    excused = _refusal_is_announced(snap)
    for fig in figures:
        if not fig.get("traces") and not (fig.get("id") == "pd-plot" and excused):
            violations.append(f"[{snap['stage']}] figure {fig.get('id')} rendered no trace")
        if not fig.get("svgs"):
            violations.append(f"[{snap['stage']}] figure {fig.get('id')} produced no SVG")
        if not (fig.get("w") and fig.get("h")):
            violations.append(f"[{snap['stage']}] figure {fig.get('id')} has zero size "
                              f"({fig.get('w')}x{fig.get('h')})")
    return True


def _oracle_designer_live_when_plottable(snap, violations) -> bool:
    """A report carrying plot data must open a *working* figure builder.

    The text-level oracle checks the panel is in the file. This one checks the
    figure inside it exists, which is the part a broken script kills.
    """
    if not snap.get("has_plot_payload"):
        return False
    if not snap.get("designer"):
        violations.append(f"[{snap['stage']}] report carries plot data but no designer panel")
        return True
    pd = snap.get("pd") or {}
    if not pd.get("traces") and not _refusal_is_announced(snap):
        violations.append(f"[{snap['stage']}] designer is present but its figure has no trace, "
                          f"and nothing on the page says why")
    return True


def _oracle_labels_not_clipped(snap, violations) -> bool:
    """No axis label, title or legend entry may stick out of the plot container.

    The figure builder draws into a fixed-size box. Long axis titles and rotated
    group labels are what overflow it, and a hand-tuned pixel margin never fits
    arbitrary text -- automargin does, which is why this is worth watching from
    the outside rather than trusting the flag is still set.
    """
    pd = snap.get("pd") or {}
    if not pd.get("traces"):
        return False
    overflow = pd.get("overflow") or []
    for item in overflow[:4]:
        violations.append(
            f"[{snap['stage']}] '{item.get('text')}' ({item.get('cls')}) overflows the plot "
            f"container by {item.get('over')}px")
    return True


def _oracle_significance_matches_mode(snap, violations) -> bool:
    """What is drawn has to be the significance form the control says.

    Letters and brackets are mutually exclusive claims about the same
    comparisons: drawing both, or drawing brackets while the select reads
    "letters", means the reader is looking at a layer nobody chose.
    """
    pd = snap.get("pd") or {}
    if not pd.get("traces") or pd.get("sig_disabled"):
        return False
    mode = pd.get("sig_mode")
    brackets = pd.get("bracket_shapes") or 0
    letters = pd.get("letter_annotations") or 0
    if mode == "none" and (brackets or letters):
        violations.append(f"[{snap['stage']}] significance is off but the figure shows "
                          f"{brackets} bracket lines and {letters} letters")
    if mode == "letters" and brackets:
        violations.append(f"[{snap['stage']}] letters mode still draws {brackets} bracket lines")
    return True


def _oracle_designer_keeps_report_order(snap, violations) -> bool:
    """The builder must open on the order the report decided, not re-sort it.

    The group axis is ranked once, centrally, so the control group comes first
    and D7 precedes D21. If the figure builder rebuilt that order from the
    payload keys it would quietly undo the ranking for the one figure a user
    actually exports.
    """
    order = [str(x) for x in (snap.get("payload_order") or [])]
    pd = snap.get("pd") or {}
    if not order:
        return False

    # Which axis carries the groups depends on the plot type -- Forest and the
    # horizontal layouts put them on y and a numeric scale on x -- so the axis
    # is identified by its labels rather than assumed. An axis that carries
    # something other than the groups (a numeric effect-size scale) is simply
    # not this oracle's business.
    wanted = set(order)
    on_x = [str(x) for x in (pd.get("categories") or [])]
    on_y = [str(x) for x in (pd.get("categories_y") or [])]
    if wanted <= set(on_x):
        drawn, axis = on_x, "x"
    elif wanted <= set(on_y):
        drawn, axis = on_y, "y"
    else:
        # Neither axis carries the whole group set. Partly is the interesting
        # case: a figure showing three of four groups has dropped one, and
        # returning False there would file a missing group as "not my business".
        # No overlap at all is a genuinely different axis -- a regression
        # scatter, an effect-size scale -- and this oracle has nothing to say.
        present = wanted & (set(on_x) | set(on_y))
        if present and present != wanted:
            violations.append(
                f"[{snap['stage']}] the figure shows {sorted(present)} but the report's "
                f"groups are {order} -- {sorted(wanted - present)} reached no axis")
            return True
        return False

    # The figure builder offers move-up / move-down per group, so once the user
    # has pressed one, an axis that differs from the report is the feature
    # working. Claiming a violation there would be the oracle asserting that a
    # control it just operated should have had no effect. What survives a
    # deliberate reorder is membership: moving a group must not lose one.
    if snap.get("order_user_modified"):
        if set(drawn) != wanted:
            violations.append(f"[{snap['stage']}] reordering the groups changed which groups "
                              f"are on the {axis} axis: {drawn} vs {order}")
        return True

    # Plotly numbers a categorical y axis from the bottom, so a horizontal
    # figure reads top-to-bottom in reverse. Which of the two a reader should
    # see is a design decision, not this oracle's to make -- both are accepted,
    # and an arbitrary re-sort (alphabetical, insertion order) still fails.
    acceptable = [order] if axis == "x" else [order, list(reversed(order))]
    if drawn not in acceptable:
        violations.append(f"[{snap['stage']}] designer {axis}-axis order {drawn} does not match "
                          f"the report's group order {order}")
    return True


def _oracle_designer_warning_text(snap, violations) -> bool:
    """The builder's own warning line is a finding when it appears unprompted.

    It exists for honest refusals ("letters need a complete comparison matrix"),
    so its text is reported rather than treated as a failure -- the orchestrator
    collects them and a new wording shows up as a new string.
    """
    pd = snap.get("pd") or {}
    warning = (pd.get("warning") or "").strip()
    if not warning:
        return False
    snap.setdefault("warnings_seen", []).append(warning[:120])
    return True


ORACLES = (
    ("no_script_error", _oracle_no_script_error),
    ("figures_render", _oracle_figures_render),
    ("designer_live_when_plottable", _oracle_designer_live_when_plottable),
    ("labels_not_clipped", _oracle_labels_not_clipped),
    ("significance_matches_mode", _oracle_significance_matches_mode),
    ("designer_keeps_report_order", _oracle_designer_keeps_report_order),
    ("designer_warning_text", _oracle_designer_warning_text),
)


def check_stage(snap: dict):
    """Run every stage oracle against one snapshot."""
    violations, fired = [], []
    for name, oracle in ORACLES:
        try:
            if oracle(snap, violations):
                fired.append(name)
        except Exception as exc:      # a broken oracle is a finding about the oracle
            violations.append(f"[{snap.get('stage')}] oracle {name} raised "
                              f"{type(exc).__name__}: {exc}")
    return violations, fired


# --- the exported file ------------------------------------------------------

_MAGIC = {"png": b"\x89PNG\r\n\x1a\n", "svg": b"<svg"}
# A figure that renders but exports a stub is still a broken export. These are
# floors, not targets: an SVG with one empty <svg/> element is ~120 bytes and a
# blank PNG canvas a few hundred.
_MIN_BYTES = {"png": 4000, "svg": 1200}


def check_download(fmt: str, path: str, error: str = ""):
    """Pressing Download has to produce a file that is really that format."""
    violations, fired = [], [f"download_{fmt}"]
    if error:
        violations.append(f"[export] downloading {fmt} failed: {error}")
        return violations, fired
    import os
    if not path or not os.path.exists(path):
        violations.append(f"[export] downloading {fmt} produced no file")
        return violations, fired
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        head = fh.read(16)
    magic = _MAGIC[fmt]
    if not head.startswith(magic):
        violations.append(f"[export] {fmt} download does not start with {magic!r} "
                          f"(got {head[:8]!r})")
    if size < _MIN_BYTES[fmt]:
        violations.append(f"[export] {fmt} download is {size} bytes, which is an empty canvas "
                          f"rather than a figure")
    return violations, fired
