"""What a user does to the figure *after* the report exists.

The other two fuzzers stop where the HTML is written. Everything a reader then
does -- open the file, look at the chart, switch it to a violin, rename the
axis, move the legend, press Download -- happens in a browser and was checked
by nothing. The two symptoms the user named ("der Plot sieht komisch aus", "die
Formatierung vom Plot funktioniert nicht") both live on that side of the line.

The action plan is not a hardcoded list of control ids. It is drawn from the
controls the page actually offers, discovered at load time, so a control added
to plot_designer.html is fuzzed the day it appears rather than the day someone
remembers to add it here. What this module owns is the *choice*: which of the
offered controls this seed pokes, and with what value.

Values are picked to be plausible-but-hostile, in the shape a real user
produces: a y-axis title long enough to need a margin, an empty label, a Greek
letter, a y-minimum above the y-maximum. Nothing here invents a value the UI
could not have accepted -- the range and the option list come from the element.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Text a user really types into a label field. The long one exists to force a
# margin (the axis-label clipping bug came back five times), the empty one to
# check a blank title does not take the layout down with it, and the maths and
# unicode ones because biomedical labels are full of both.
NASTY_TEXT = (
    "Concentration of a very long analyte name in plasma [nmol/L]",
    "",
    "α-Synuclein (µg·mL⁻¹)",
    "$\\Delta$F/F$_0$",
    "Tumour volume\nafter 21 days",
    "IL-6",
)

# Out-of-range and inverted numbers a user can type into a spin box. y-min above
# y-max is the interesting one: an axis with an impossible range must degrade to
# something readable rather than render an empty canvas.
NUMBER_STRATEGIES = ("low", "high", "middle", "over_max", "under_min", "zero")

BUTTON_ACTIONS = (
    "preset_publication", "preset_poster", "preset_talk",
    "legend_inside", "legend_outside", "legend_bottom",
    "group_up", "group_down",
)


@dataclass
class VisualPlan:
    seed: int
    plot_types: list = field(default_factory=list)
    steps: list = field(default_factory=list)     # [{"kind": ..., ...}, ...]
    downloads: list = field(default_factory=list)

    @property
    def labels(self) -> list:
        """Names for coverage reporting -- what this seed actually exercised."""
        names = [f"type:{t}" for t in self.plot_types]
        names += [s.get("control") or s.get("action") for s in self.steps]
        names += [f"download:{f}" for f in self.downloads]
        return [n for n in names if n]


def _number_for(control, strategy, rng):
    lo = control.get("min")
    hi = control.get("max")
    lo = float(lo) if lo not in (None, "") else None
    hi = float(hi) if hi not in (None, "") else None
    if strategy == "zero":
        return 0
    if lo is not None and strategy == "low":
        return lo
    if hi is not None and strategy == "high":
        return hi
    if lo is not None and hi is not None and strategy == "middle":
        return round(lo + (hi - lo) * float(rng.random()), 3)
    if hi is not None and strategy == "over_max":
        return hi + abs(hi) + 1
    if lo is not None and strategy == "under_min":
        return lo - abs(lo) - 1
    # An unbounded field (y-min, y-max, legend x/y) -- stay in a range a user
    # could plausibly type rather than generating an astronomical number that
    # tests Plotly's float handling instead of ours.
    return round(float(rng.uniform(-50, 50)), 3)


def _value_for(control, rng):
    kind = control.get("type")
    if kind == "checkbox":
        return bool(rng.integers(0, 2))
    if control.get("tag") == "select":
        options = [o for o in (control.get("options") or []) if o is not None]
        if not options:
            return None
        return options[int(rng.integers(0, len(options)))]
    if kind in ("number", "range"):
        strategy = NUMBER_STRATEGIES[int(rng.integers(0, len(NUMBER_STRATEGIES)))]
        return _number_for(control, strategy, rng)
    if kind == "color":
        return "#%02x%02x%02x" % tuple(int(rng.integers(0, 256)) for _ in range(3))
    return NASTY_TEXT[int(rng.integers(0, len(NASTY_TEXT)))]


def build_plan(seed: int, surface: dict) -> VisualPlan:
    """Decide what this seed does to the figure.

    ``surface`` is what the page reported about itself: the controls it offers
    (id, tag, type, option list, bounds, owning tab), the buttons, and the plot
    types in the dropdown. Given the same surface and seed the plan is
    identical, so a finding replays exactly.
    """
    import numpy as np
    rng = np.random.default_rng(seed + 0x9115)

    types = list(surface.get("plot_types") or [])
    # One to three plot types per seed rather than all six: a seed stays cheap,
    # and the orchestrator reports which types were reached across the run, so
    # a type that is never exercised is visible instead of assumed.
    picked_types = []
    if types:
        count = int(rng.integers(1, min(3, len(types)) + 1))
        order = rng.permutation(len(types))[:count]
        picked_types = [types[int(i)] for i in order]

    controls = [c for c in (surface.get("controls") or []) if c.get("id")]
    controls.sort(key=lambda c: c["id"])          # discovery order is not stable
    steps = []
    if controls:
        count = int(rng.integers(3, min(12, len(controls)) + 1))
        chosen = rng.permutation(len(controls))[:count]
        for index in chosen:
            control = controls[int(index)]
            value = _value_for(control, rng)
            if value is None:
                continue
            steps.append({"kind": "set", "control": control["id"], "tab": control.get("tab"),
                          "tag": control.get("tag"), "type": control.get("type"),
                          "value": value})

    for action in BUTTON_ACTIONS:
        if int(rng.integers(0, 3)) == 0:          # roughly a third of the buttons
            steps.append({"kind": "click", "action": action})

    rng.shuffle(steps)

    # Both formats every time. The export is the last thing a user does and the
    # one artefact that leaves the report, so it is not worth sampling.
    downloads = ["svg", "png"]
    return VisualPlan(seed=seed, plot_types=picked_types, steps=steps, downloads=downloads)
