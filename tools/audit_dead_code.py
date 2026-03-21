"""
audit_dead_code.py
==================
BioMedStatX codebase dead-code audit.

Schicht 1 des Zwei-Schichten-Audit-Systems:
- ruff: ungenutzte Imports (F401), Variablen (F841), auskommentierter Code (ERA001),
         hohe Komplexität (PLR0912, PLR0915)
- vulture: ungenutzte Funktionen/Klassen/Variablen (min-confidence 80)
- AST-Scan: doppelte Funktionsnamen cross-file (Namespace Collisions)
- Regex-Scan: # from ..., # DISABLED, silent try/except
- Import-Herkunft: Nicht-stdlib-Imports außerhalb von lazy_imports.py

Run:  python audit_dead_code.py
      python audit_dead_code.py --report   (schreibt audit_report.md)
"""

import argparse
import ast
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SOURCE_DIR = Path(__file__).parent / "Source_Code"
VULTURE_ALLOWLIST = Path(__file__).parent / "vulture_allowlist.py"
REPORT_FILE = Path(__file__).parent / "audit_report.md"

# Ruff rules: F401=unused import, F841=unused var, ERA001=commented-out code,
#             PLR0912=too many branches, PLR0915=too many statements
RUFF_RULES = "F401,F841,ERA001,PLR0912,PLR0915"

# statisticaltester.py is ~5700 lines — only check imports there, not complexity
LARGE_FILE_RUFF_RULES = "F401"
LARGE_FILE_THRESHOLD_LINES = 3000

# Standard library module names (subset relevant to this project)
STDLIB_MODULES = {
    "os", "sys", "re", "ast", "math", "time", "json", "csv", "copy",
    "io", "pathlib", "typing", "collections", "itertools", "functools",
    "warnings", "logging", "subprocess", "threading", "datetime",
    "traceback", "inspect", "importlib", "abc", "enum", "dataclasses",
    "contextlib", "textwrap", "struct", "hashlib", "random", "string",
    "shutil", "tempfile", "glob", "fnmatch", "platform", "argparse",
}

SEPARATOR = "=" * 70


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def header(title: str) -> str:
    return f"\n{SEPARATOR}\n  {title}\n{SEPARATOR}"


def run_tool(cmd: list[str]) -> tuple[str, str, int]:
    """Run external tool, return (stdout, stderr, returncode)."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode


def check_tool(name: str, module_flag: str = None) -> bool:
    """Check if a tool is available."""
    cmd = [sys.executable, "-m", name, "--version"] if module_flag else [name, "--version"]
    try:
        r = subprocess.run(cmd, capture_output=True)
        return r.returncode == 0
    except FileNotFoundError:
        return False


def py_files() -> list[Path]:
    return sorted(SOURCE_DIR.glob("*.py"))


def line_count(path: Path) -> int:
    try:
        return sum(1 for _ in path.open(encoding="utf-8", errors="ignore"))
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Check 1: ruff
# ---------------------------------------------------------------------------
def run_ruff() -> list[str]:
    findings = []
    if not check_tool("ruff"):
        findings.append("  [SKIP] ruff nicht installiert. Installieren: pip install ruff")
        return findings

    for f in py_files():
        n_lines = line_count(f)
        rules = LARGE_FILE_RUFF_RULES if n_lines >= LARGE_FILE_THRESHOLD_LINES else RUFF_RULES
        stdout, _, _ = run_tool(["ruff", "check", str(f), f"--select={rules}", "--output-format=text"])
        if stdout.strip():
            findings.append(f"\n  --- {f.name} ({n_lines} Zeilen) ---")
            for line in stdout.strip().splitlines():
                findings.append(f"  {line}")

    return findings if findings else ["  Keine ruff-Findings."]


# ---------------------------------------------------------------------------
# Check 2: vulture
# ---------------------------------------------------------------------------
def run_vulture() -> list[str]:
    findings = []
    if not check_tool("vulture"):
        findings.append("  [SKIP] vulture nicht installiert. Installieren: pip install vulture")
        return findings

    cmd = ["vulture", str(SOURCE_DIR), "--min-confidence", "80"]
    if VULTURE_ALLOWLIST.exists():
        cmd.append(str(VULTURE_ALLOWLIST))

    stdout, _, _ = run_tool(cmd)
    if stdout.strip():
        for line in stdout.strip().splitlines():
            findings.append(f"  {line}")
    else:
        findings.append("  Keine vulture-Findings (min-confidence 80).")

    return findings


# ---------------------------------------------------------------------------
# Check 3: AST cross-file namespace collisions
# Only flags MODULE-LEVEL functions/classes, not methods or nested functions.
# ---------------------------------------------------------------------------
def _get_toplevel_names(tree: ast.Module) -> list[tuple[str, int, str]]:
    """
    Returns (name, lineno, kind) for all top-level defs/classes in a module.
    Skips methods (inside ClassDef) and nested functions (inside FunctionDef).
    """
    results = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            results.append((node.name, node.lineno, "def"))
        elif isinstance(node, ast.ClassDef):
            results.append((node.name, node.lineno, "class"))
    return results


def run_ast_collision_check() -> list[str]:
    findings = []
    func_map: dict[str, list[str]] = defaultdict(list)  # name -> [file:line, ...]

    for f in py_files():
        try:
            tree = ast.parse(f.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError as e:
            findings.append(f"  [ERROR] Konnte {f.name} nicht parsen: {e}")
            continue

        for name, lineno, _ in _get_toplevel_names(tree):
            func_map[name].append(f"{f.name}:{lineno}")

    # Only flag names that appear in MORE THAN ONE FILE
    collisions = {
        name: locs for name, locs in func_map.items()
        if len(set(loc.split(":")[0] for loc in locs)) > 1
    }

    if collisions:
        findings.append(f"  {len(collisions)} cross-file Kollisionen gefunden:\n")
        for name, locs in sorted(collisions.items()):
            findings.append(f"  def/class '{name}' in:")
            for loc in locs:
                findings.append(f"    - {loc}")
            findings.append("")
    else:
        findings.append("  Keine cross-file Namespace-Kollisionen (module-level).")

    return findings


# ---------------------------------------------------------------------------
# Check 4: Regex-Scan (project-specific patterns)
# ---------------------------------------------------------------------------
REGEX_PATTERNS = [
    (r"^[ \t]*#\s*from\s+\w+\s+import", "Auskommentierter Import"),
    (r"#\s*DISABLED", "DISABLED-Marker"),
    (r"#\s*TODO|#\s*FIXME|#\s*HACK|#\s*XXX", "TODO/FIXME/HACK"),
    (r"except\s*[^:]*:\s*pass\s*$", "Silent except: pass (kein Logging)"),
    (r"if\s+False\s*:", "if False: (toter Block)"),
    (r"if\s+0\s*:", "if 0: (toter Block)"),
]


def run_regex_scan() -> list[str]:
    findings = []
    total = 0

    for f in py_files():
        file_findings = []
        lines = f.read_text(encoding="utf-8", errors="ignore").splitlines()
        for lineno, line in enumerate(lines, 1):
            for pattern, label in REGEX_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    file_findings.append(f"    Zeile {lineno:4d} [{label}]: {line.rstrip()}")
                    total += 1
                    break

        if file_findings:
            findings.append(f"\n  --- {f.name} ---")
            findings.extend(file_findings)

    if total == 0:
        findings.append("  Keine Regex-Findings.")
    else:
        findings.insert(0, f"  {total} Findings gesamt:")

    return findings


# ---------------------------------------------------------------------------
# Check 5: Import-Herkunft (non-stdlib imports outside lazy_imports.py)
# ---------------------------------------------------------------------------
INTERNAL_MODULES = {f.stem for f in py_files()}


def run_import_source_check() -> list[str]:
    findings = []

    for f in py_files():
        if f.name == "lazy_imports.py":
            continue  # lazy_imports.py darf alles importieren

        try:
            tree = ast.parse(f.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top not in STDLIB_MODULES and top not in INTERNAL_MODULES:
                        findings.append(
                            f"  {f.name}:{node.lineno}  import {alias.name}"
                            f"  → Kandidat für lazy_imports.py"
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    top = node.module.split(".")[0]
                    if top not in STDLIB_MODULES and top not in INTERNAL_MODULES:
                        names = ", ".join(a.name for a in node.names)
                        findings.append(
                            f"  {f.name}:{node.lineno}  from {node.module} import {names}"
                            f"  → Kandidat für lazy_imports.py"
                        )

    return findings if findings else ["  Alle externen Imports gehen durch lazy_imports.py oder sind stdlib."]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_report() -> str:
    sections = []

    sections.append("# BioMedStatX — Dead-Code Audit Report\n")
    sections.append(f"Source: `{SOURCE_DIR}`\n")
    sections.append(f"Dateien geprüft: {len(py_files())}\n")

    sections.append(header("1. ruff (ungenutzte Imports, Variablen, auskommentierter Code, Komplexität)"))
    sections.extend(run_ruff())

    sections.append(header("2. vulture (ungenutzte Funktionen / Klassen / Variablen)"))
    sections.extend(run_vulture())

    sections.append(header("3. AST Cross-File Namespace Collisions (doppelte def/class Namen)"))
    sections.extend(run_ast_collision_check())

    sections.append(header("4. Regex-Scan (projektspezifische tote-Code-Muster)"))
    sections.extend(run_regex_scan())

    sections.append(header("5. Import-Herkunft (externe Imports außerhalb lazy_imports.py)"))
    sections.extend(run_import_source_check())

    sections.append(f"\n{SEPARATOR}\n  Audit abgeschlossen.\n{SEPARATOR}\n")

    return "\n".join(str(s) for s in sections)


def main():
    parser = argparse.ArgumentParser(description="BioMedStatX Dead-Code Audit")
    parser.add_argument("--report", action="store_true", help="Schreibe audit_report.md")
    args = parser.parse_args()

    report = build_report()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(report)

    if args.report:
        REPORT_FILE.write_text(report, encoding="utf-8")
        print(f"\nReport gespeichert: {REPORT_FILE}")


if __name__ == "__main__":
    main()
