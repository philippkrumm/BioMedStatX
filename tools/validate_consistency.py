"""
Consistency validator for BioMedStatX.
Run: python tools/validate_consistency.py
Exit code 0 = all checks passed. Non-zero = failures present (warnings alone do not fail).

Checks:
  1. DecisionTreeVisualizer.get_tree_json() — all node IDs referenced in edges
     must exist in nodes_info (catches orphan edge endpoints like the IND_FDR bug).
  2. FlowchartVisualizer._build_topology() — same orphan-edge check across all
     model-type branches (both renderers share this code, so one check covers both).
  3. HTML template field coverage — required context keys referenced in both templates.
  4. (Warning only) GUI button labels absent from docs/HowTo.md.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAIL = False


def fail(msg: str) -> None:
    global FAIL
    FAIL = True
    print(f"  FAIL  {msg}")


def warn(msg: str) -> None:
    print(f"  WARN  {msg}")


def ok(msg: str) -> None:
    print(f"  OK    {msg}")


# ---------------------------------------------------------------------------
# Helpers — parse node IDs and edge tuples from Python source via regex
# ---------------------------------------------------------------------------
# Matches:  'SOME_ID': {  or  "SOME_ID": {
NODE_ID_IN_DICT = re.compile(r"""['"]((?:[A-Z][A-Z0-9_]*)|(?:[A-Z][A-Z0-9_]*))['"]\s*:\s*\{""")
# Matches edge tuples with single or double quotes:  ('SRC', 'TGT')  ("SRC", "TGT")
EDGE_PAIR = re.compile(r"""\(\s*['"]([A-Z][A-Z0-9_]*)['\"]\s*,\s*['"]([A-Z][A-Z0-9_]*)['\"]\s*\)""")


def extract_node_ids(src: str) -> set[str]:
    return set(NODE_ID_IN_DICT.findall(src))


def extract_edge_pairs(src: str) -> list[tuple[str, str]]:
    return EDGE_PAIR.findall(src)


def find_orphan_edges(
    node_ids: set[str],
    edges: list[tuple[str, str]],
    context: str,
) -> list[str]:
    orphans = []
    for src, tgt in edges:
        if src not in node_ids:
            orphans.append(f"{context}: edge source '{src}' has no node in nodes_info")
        if tgt not in node_ids:
            orphans.append(f"{context}: edge target '{tgt}' has no node in nodes_info")
    return orphans


# ---------------------------------------------------------------------------
# Check 1 — DecisionTreeVisualizer.get_tree_json() orphan edges
# ---------------------------------------------------------------------------
def check_decision_tree_visualizer() -> None:
    print("\n[1] DecisionTreeVisualizer — orphan edge check")

    path = ROOT / "src" / "visualization" / "decisiontreevisualizer.py"
    if not path.exists():
        fail(f"Not found: {path}")
        return

    src = path.read_text(encoding="utf-8")

    # Isolate the get_tree_json method body (everything from its def line onward
    # until the next same-level def or end of class)
    method_src = _slice_method(src, "get_tree_json")
    if not method_src:
        warn("Could not isolate get_tree_json() — checking full file")
        method_src = src

    node_ids = extract_node_ids(method_src)
    edges    = extract_edge_pairs(method_src)

    if not node_ids:
        warn("No node IDs extracted from decisiontreevisualizer.get_tree_json() — check regex")
        return
    if not edges:
        warn("No edge pairs extracted from decisiontreevisualizer.get_tree_json() — check regex")
        return

    ok(f"Found {len(node_ids)} nodes, {len(edges)} edge pairs in get_tree_json()")
    orphans = find_orphan_edges(node_ids, edges, "DecisionTreeVisualizer.get_tree_json")
    if orphans:
        for msg in orphans:
            fail(msg)
    else:
        ok("No orphan edges — all edge endpoints have a node")


# ---------------------------------------------------------------------------
# Check 2 — FlowchartVisualizer._build_topology() orphan edges
# ---------------------------------------------------------------------------
def check_flowchart_visualizer() -> None:
    print("\n[2] FlowchartVisualizer — orphan edge check in _build_topology()")

    path = ROOT / "src" / "visualization" / "flowchartvisualizer.py"
    if not path.exists():
        fail(f"Not found: {path}")
        return

    src = path.read_text(encoding="utf-8")

    method_src = _slice_method(src, "_build_topology")
    if not method_src:
        warn("Could not isolate _build_topology() — checking full file")
        method_src = src

    node_ids = extract_node_ids(method_src)
    edges    = extract_edge_pairs(method_src)

    if not node_ids:
        warn("No node IDs extracted from _build_topology() — check regex")
        return
    if not edges:
        warn("No edge pairs extracted from _build_topology() — check regex")
        return

    ok(f"Found {len(node_ids)} nodes, {len(edges)} edge pairs in _build_topology()")
    orphans = find_orphan_edges(node_ids, edges, "FlowchartVisualizer._build_topology")
    if orphans:
        for msg in orphans:
            fail(msg)
    else:
        ok("No orphan edges — all edge endpoints have a node")


def _slice_method(src: str, method_name: str) -> str:
    """Extract source lines for a method by indentation heuristic."""
    lines = src.splitlines()
    start = None
    base_indent = None
    for i, line in enumerate(lines):
        m = re.match(r'^(\s*)def\s+' + re.escape(method_name) + r'\b', line)
        if m:
            start = i
            base_indent = len(m.group(1))
            break

    if start is None:
        return ""

    collected = []
    for line in lines[start + 1:]:
        stripped = line.lstrip()
        if stripped and not stripped.startswith("#"):
            indent = len(line) - len(line.lstrip())
            if indent <= base_indent and re.match(r'\s*def\s+', line):
                break
        collected.append(line)

    return "\n".join(collected)


# ---------------------------------------------------------------------------
# Check 3 — HTML template context key coverage
# ---------------------------------------------------------------------------
# These are Jinja2 context paths that MUST appear in both report templates.
# Add to this list when you add a new top-level section to the report context builder.
REQUIRED_TEMPLATE_CONTEXT_KEYS = [
    "context.hero",
    "context.decision_tree_json",
    "context.statistical_rows",
    "context.assumptions",
    "context.descriptive",
    "context.pairwise_rows",
    "context.methods_text",
    "context.chart_blocks",
    "context.raw_data_table",
    "context.info_texts",
]


def check_html_templates() -> None:
    print("\n[3] HTML template context key coverage")

    templates = [
        ROOT / "src" / "templates" / "report_single.html.j2",
        ROOT / "src" / "templates" / "report_multi.html.j2",
    ]

    for tpl in templates:
        if not tpl.exists():
            fail(f"Template not found: {tpl.name}")
            continue

        src = tpl.read_text(encoding="utf-8")
        missing = [k for k in REQUIRED_TEMPLATE_CONTEXT_KEYS if k not in src]

        if missing:
            for k in missing:
                fail(f"{tpl.name}: required context key '{k}' not referenced")
        else:
            ok(f"{tpl.name}: all {len(REQUIRED_TEMPLATE_CONTEXT_KEYS)} required context keys present")


# ---------------------------------------------------------------------------
# Check 4 — GUI button labels in HowTo.md (warning only)
# ---------------------------------------------------------------------------
BUTTON_PATTERNS = [
    re.compile(r'QPushButton\s*\(\s*["\']([^"\']{4,})["\']'),
    re.compile(r'setText\s*\(\s*["\']([^"\']{4,})["\']'),
    re.compile(r'QAction\s*\(\s*["\']([^"\']{4,})["\']'),
]
# Filter out labels that are clearly not user-facing or too long (tooltips etc.)
BUTTON_IGNORE = re.compile(
    r'^\s*$|^&|debug|Dev|_|^\d|^OK$|^Cancel$|^Close$|^Apply$|^Yes$|^No$|^None$|\.{3}$|\n',
    re.IGNORECASE
)


def check_howto_gui_coverage() -> None:
    print("\n[4] GUI button labels in docs/HowTo.md (warning only)")

    howto = ROOT / "docs" / "HowTo.md"
    if not howto.exists():
        warn("docs/HowTo.md not found — skipping GUI coverage check")
        return

    howto_src = howto.read_text(encoding="utf-8").lower()

    ui_dirs = [ROOT / "src" / "ui", ROOT / "src" / "autopilot"]
    ui_files = []
    for d in ui_dirs:
        if d.exists():
            ui_files.extend(d.rglob("*.py"))

    labels: set[str] = set()
    for ui_file in ui_files:
        src = ui_file.read_text(encoding="utf-8", errors="ignore")
        for pat in BUTTON_PATTERNS:
            for m in pat.finditer(src):
                label = m.group(1).strip()
                # Skip long strings (likely error messages / tooltips, not button labels)
                if len(label) > 60:
                    continue
                if not BUTTON_IGNORE.search(label):
                    labels.add(label)

    missing = [lb for lb in sorted(labels) if lb.lower() not in howto_src]

    if missing:
        for label in missing[:15]:
            warn(f"UI label '{label}' not mentioned in docs/HowTo.md")
        if len(missing) > 15:
            warn(f"... and {len(missing) - 15} more (total {len(missing)} unlisted labels)")
    else:
        ok(f"All {len(labels)} UI labels found in HowTo.md")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    print("BioMedStatX consistency validator")
    print("=" * 50)

    check_decision_tree_visualizer()
    check_flowchart_visualizer()
    check_html_templates()
    check_howto_gui_coverage()

    print("\n" + "=" * 50)
    if FAIL:
        print("RESULT: FAILURES detected — fix the issues above before committing.")
        return 1
    else:
        print("RESULT: All checks passed.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
