"""Guards for the compact letter display and its completeness gate.

Two things are worth locking down here. The letter algorithm itself has a known
failure mode -- an intransitive pattern used to collapse onto one letter and
hide a real difference -- and the layer now exists twice, in Python for the
static report and in JavaScript for the interactive figure builder. A duplicated
renderer that drifts is exactly how the axis-label clipping kept coming back, so
the parity test below runs the JS implementation and demands the same letters.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from analysis.compact_letters import letters_from_pairs, letters_supported
from export.report_charts import _ChartsMixin

_JS_SOURCE = Path(__file__).resolve().parents[1] / "src" / "templates" / "plot_designer.js"

GROUPS = ["A", "B", "C", "D"]

# A~B and B~C, but A#C: the intransitive pattern. A and C must NOT share a
# letter -- the superseded star+absorb implementation merged them and silently
# hid the A-C difference.
INTRANSITIVE = [
    {"group1": "A", "group2": "B", "significant": False},
    {"group1": "A", "group2": "C", "significant": True},
    {"group1": "A", "group2": "D", "significant": True},
    {"group1": "B", "group2": "C", "significant": False},
    {"group1": "B", "group2": "D", "significant": True},
    {"group1": "C", "group2": "D", "significant": True},
]

TOPS = {"A": 11.6, "B": 13.9, "C": 16.2, "D": 23.6}


def _pairs(results):
    return [{"group1": g1, "group2": g2, "significant": sig}
            for g1, g2, sig in results]


def test_intransitive_pattern_keeps_the_real_difference_visible():
    letters = letters_from_pairs(GROUPS, INTRANSITIVE, sort_by=TOPS)
    shared = set(letters["A"]) & set(letters["C"])
    assert not shared, f"A and C are significantly different but share {shared}"
    # The non-significant neighbours do share, which is the whole point.
    assert set(letters["A"]) & set(letters["B"])
    assert set(letters["B"]) & set(letters["C"])
    # D differs from everything, so its letter is its own.
    for other in ("A", "B", "C"):
        assert not (set(letters["D"]) & set(letters[other]))


def test_every_group_always_receives_a_letter():
    letters = letters_from_pairs(GROUPS, INTRANSITIVE, sort_by=TOPS)
    assert all(letters[g] for g in GROUPS)


def test_gate_accepts_a_complete_comparison_set():
    supported, reason = letters_supported(GROUPS, INTRANSITIVE)
    assert supported
    assert reason == ""


def test_gate_refuses_a_many_to_one_post_hoc():
    dunnett = [p for p in INTRANSITIVE if "A" in (p["group1"], p["group2"])]
    supported, reason = letters_supported(GROUPS, dunnett)
    assert not supported
    assert "3" in reason and "6" in reason


def test_gate_is_structural_not_a_list_of_test_names():
    """A hand-picked pair set is judged by what it covers, nothing else.

    ``paired_custom`` lets the user choose the comparisons in a dialog, so no
    post-hoc name could ever settle this. Picking all of them is a legitimate
    all-pairs result; picking some is not.
    """
    assert letters_supported(GROUPS, INTRANSITIVE)[0]
    assert not letters_supported(GROUPS, INTRANSITIVE[:-1])[0]


def test_gate_needs_at_least_two_groups():
    assert not letters_supported(["A"], [])[0]


@pytest.mark.parametrize("groups, pairs, expected", [
    # Complete but small: brackets stay readable, and k=2 must fall through the
    # same way k=3 does rather than needing its own case.
    (["A", "B"], [{"group1": "A", "group2": "B", "significant": True}], "brackets"),
    (["A", "B", "C"], _pairs([("A", "B", True), ("A", "C", True), ("B", "C", False)]), "brackets"),
    # Complete and crowded: 6 brackets, letters win.
    (GROUPS, INTRANSITIVE, "letters"),
    # Incomplete at any size.
    (GROUPS, [p for p in INTRANSITIVE if "A" in (p["group1"], p["group2"])], "brackets"),
])
def test_default_mode_rule(groups, pairs, expected):
    assert _ChartsMixin._significance_mode(groups, pairs) == expected


def test_static_and_interactive_agree_letter_for_letter():
    """The Python and JavaScript implementations must not drift apart."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available; JS parity cannot be checked")

    source = _JS_SOURCE.read_text()
    start = source.index("  function lettersSupported(pairs) {")
    end = source.index("  // Letters are computed from the FULL comparison matrix")
    harness = (
        "var groupOrder = " + json.dumps(GROUPS) + ";\n"
        + source[start:end]
        + "var pairs = " + json.dumps(INTRANSITIVE) + ";\n"
        + "var tops = " + json.dumps(TOPS) + ";\n"
        + "console.log(JSON.stringify({letters: compactLetters(pairs, tops),"
          " supported: lettersSupported(pairs).ok,"
          " mode: defaultSignificanceMode(pairs)}));\n"
    )
    proc = subprocess.run([node, "-e", harness], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    js = json.loads(proc.stdout)

    assert js["letters"] == letters_from_pairs(GROUPS, INTRANSITIVE, sort_by=TOPS)
    assert js["supported"] is letters_supported(GROUPS, INTRANSITIVE)[0]
    assert js["mode"] == _ChartsMixin._significance_mode(GROUPS, INTRANSITIVE)


def test_pair_helper_is_the_single_source_for_stars():
    """Both layers read stars from the shared row builder, never re-derive them."""
    results = {"pairwise_comparisons": [
        {"group1": "A", "group2": "B", "p_value": 0.0004, "significant": True},
        {"group1": "A", "group2": "C", "p_value": 0.03, "significant": True},
        {"group1": "B", "group2": "C", "p_value": 0.4, "significant": False},
        # Off-plot pair: must be dropped rather than misplaced.
        {"group1": "Z", "group2": "A", "p_value": 0.01, "significant": True},
    ]}
    pairs = _ChartsMixin._pairs_for_plot(results, ["A", "B", "C"])
    assert [(p["group1"], p["group2"], p["stars"]) for p in pairs] == [
        ("A", "B", "***"), ("A", "C", "*"), ("B", "C", ""),
    ]
    assert all(p["i1"] < p["i2"] for p in pairs)


def test_kde_headroom_rule_has_a_single_owner():
    """The buffer that keeps annotations out of a violin's KDE tail lives once.

    A KDE overshoots the data extremes, so anything placed at the data maximum
    lands inside the visible tip. That rule was written out three times -- for
    the vertical violin baseline, inline in the raincloud bracket branch, and
    then missed entirely when the letter layer was added, which put the letters
    inside the cloud. It now belongs to violinHeadroom() alone; a second literal
    copy means someone is drifting again.
    """
    source = _JS_SOURCE.read_text()
    assert source.count("0.30") == 1, "the 30% KDE buffer was copied again"
    assert source.count("function violinHeadroom(") == 1
    # Both annotation layers must read it rather than compute their own.
    assert source.count("violinHeadroom(") >= 4  # definition + 3 call sites


def test_raincloud_is_covered_by_the_kde_buffer():
    """Raincloud draws a violin too, along x; leaving it out was the bug."""
    source = _JS_SOURCE.read_text()
    guard = source[source.index("function violinHeadroom("):]
    guard = guard[:guard.index("\n  }")]
    assert '"Violin"' in guard and '"Raincloud"' in guard
