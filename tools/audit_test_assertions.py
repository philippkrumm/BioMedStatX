"""Find assertions that cannot fail.

Four ways a check can be green without checking anything have turned up in this
repo, and every one of them cost real confidence before it was noticed:

  A  dead mutation      -- a test mutates source with ``str.replace`` and the
                           anchor is not in the file, so the "mutant" is the
                           original and the test passes by doing nothing.
  B  phantom key        -- an assertion reads a field no code anywhere produces,
                           so it compares None against None forever.
  C  chained to source  -- the expected value is imported from the very module
                           being mutated, so both sides move together.
  D  weak predicate     -- a bare truthiness check that a fallback value also
                           satisfies.

WHAT THIS TOOL DOES NOT DO. It is a *locator*, not a verdict. Every hit needs
the region read before it is called a defect -- a key absent from src/ may be
supplied by a library, a constant compared against its own module may be exactly
the right thing to assert. Equally, a clean run does not mean the suite is
sound: nothing here detects an oracle whose fixture data happens to make it
vacuous (G1/G2/G3 ordering), which is the same family and is invisible to static
analysis.

Usage:
    python tools/audit_test_assertions.py            # A, B, C
    python tools/audit_test_assertions.py --weak     # also D (noisy, advisory)
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
_TEST_DIRS = ("tests", "validation", "fuzzing")

# Keys that are pytest/pandas/plotly vocabulary rather than product fields.
_KEY_ALLOWLIST = {
    "__main__", "__file__", "__name__", "text", "data", "layout", "name",
    "type", "x", "y", "PATH", "PYTHONPATH", "QT_QPA_PLATFORM", "MPLBACKEND",
}


def _iter_py(root):
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git", "node_modules")]
        for f in files:
            if f.endswith(".py"):
                yield os.path.join(base, f)


def _parse(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return ast.parse(fh.read(), filename=path), fh
    except Exception:
        return None, None


def _src_string_literals():
    """Every string a producer can emit.

    src/*.py is the product, but two other places genuinely produce keys and
    leaving them out manufactures false hits: the Jinja templates spell the
    ``pd-data-*`` payload ids that only ever appear in HTML, and the fuzz
    workers build their own snapshot dicts that the fuzz oracles then read.
    Templates are scanned as raw text -- they are not Python.
    """
    seen = set()
    for path in _iter_py(_SRC) if True else ():
        tree, _ = _parse(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                seen.add(node.value)

    for worker in ("fuzzing/_worker.py", "fuzzing/_import_worker.py"):
        tree, _ = _parse(os.path.join(_ROOT, worker))
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                seen.add(node.value)

    tpl_dir = os.path.join(_SRC, "templates")
    if os.path.isdir(tpl_dir):
        for base, _, files in os.walk(tpl_dir):
            for f in files:
                try:
                    with open(os.path.join(base, f), encoding="utf-8") as fh:
                        text = fh.read()
                except Exception:
                    continue
                for token in re.findall(r"[A-Za-z_][\w.\-]{2,}", text):
                    seen.add(token)
    return seen


def _enclosing_functions(tree):
    """Map each node to the function that contains it."""
    owner = {}
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for node in ast.walk(fn):
                owner.setdefault(node, fn)
    return owner


# --- A: mutation without an anchor check -----------------------------------------

def check_dead_mutation(path, tree, findings):
    owner = _enclosing_functions(tree)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "replace"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            continue
        anchor = node.args[0].value
        if len(anchor) < 8:            # short replaces are formatting, not mutation
            continue
        fn = owner.get(node)
        if fn is None:
            continue
        # Is the anchor guarded anywhere in the same function?
        guarded = False
        for inner in ast.walk(fn):
            if isinstance(inner, ast.Compare) and isinstance(inner.ops[0], ast.In):
                left = inner.left
                if isinstance(left, ast.Constant) and left.value == anchor:
                    guarded = True
            if isinstance(inner, ast.Assert):
                for c in ast.walk(inner):
                    if isinstance(c, ast.Constant) and c.value == anchor:
                        guarded = True
        if not guarded:
            findings["A"].append((path, node.lineno, fn.name, anchor[:60]))


# --- B: assertion on a key no product code produces -------------------------------

# Reading a key off `result` is a claim about the product. Reading one off a
# formula string, a golden fixture or a Plotly layout is not -- those keys are
# produced by libraries and JSON files that src/ never spells out, and flagging
# them buries the real hits under a hundred of them.
_RESULT_NAMES = {"result", "results", "res", "sub", "outcome", "verdict",
                 "summary", "payload", "state", "report", "context", "ctx",
                 "posthoc", "posthoc_results", "entry", "card"}


def _reads_a_product_result(node) -> bool:
    target = node.func.value if isinstance(node, ast.Call) else node.value
    while isinstance(target, (ast.Subscript, ast.Attribute)):
        target = target.value
    return isinstance(target, ast.Name) and target.id in _RESULT_NAMES


def check_phantom_keys(path, tree, src_literals, findings):
    local_keys = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for k in node.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    local_keys.add(k.value)
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
            sl = node.slice
            if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                local_keys.add(sl.value)
        # kwargs like foo(bar="x") often name fields too
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg:
                    local_keys.add(kw.arg)

    read_keys = defaultdict(list)
    for node in ast.walk(tree):
        key = None
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get" and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            key = node.args[0].value if _reads_a_product_result(node) else None
        elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
            sl = node.slice
            if (isinstance(sl, ast.Constant) and isinstance(sl.value, str)
                    and _reads_a_product_result(node)):
                key = sl.value
        if key and key not in _KEY_ALLOWLIST and len(key) > 2:
            read_keys[key].append(node.lineno)

    for key, lines in sorted(read_keys.items()):
        if key in src_literals or key in local_keys:
            continue
        findings["B"].append((path, lines[0], key, len(lines)))


# --- C: expected value imported from the module under test -------------------------

def check_chained_to_source(path, tree, findings):
    imported = {}          # name -> module
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                name = alias.asname or alias.name
                if name.isupper() and len(name) > 3:
                    imported[name] = node.module
    if not imported:
        return
    # The circular case is not "a test mentions a constant" -- that is often
    # exactly right. It is "the test moves the constant (or its module) and then
    # compares against it", so both sides shift together and the check holds no
    # matter what the code does.
    patched = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr in ("setattr", "setitem"):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    patched.add(arg.value)
                if isinstance(arg, ast.Name):
                    patched.add(arg.id)
                if isinstance(arg, ast.Attribute):
                    patched.add(arg.attr)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue
        for inner in ast.walk(node.test):
            if isinstance(inner, ast.Name) and inner.id in imported:
                mark = "PATCHED IN THIS FILE" if inner.id in patched else "compared only"
                findings["C"].append((path, node.lineno, inner.id,
                                      f"{imported[inner.id]}  [{mark}]"))


# --- D: bare truthiness (advisory) -------------------------------------------------

def check_weak_predicates(path, tree, findings):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue
        test = node.test
        if isinstance(test, (ast.Name, ast.Attribute)):
            findings["D"].append((path, node.lineno, ast.unparse(test)[:60], "bare truthiness"))
        elif (isinstance(test, ast.Compare) and len(test.ops) == 1
              and isinstance(test.ops[0], ast.IsNot)
              and isinstance(test.comparators[0], ast.Constant)
              and test.comparators[0].value is None):
            findings["D"].append((path, node.lineno, ast.unparse(test)[:60], "is not None"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weak", action="store_true", help="include class D (noisy)")
    args = ap.parse_args()

    src_literals = _src_string_literals()
    findings = defaultdict(list)

    for directory in _TEST_DIRS:
        root = os.path.join(_ROOT, directory)
        if not os.path.isdir(root):
            continue
        for path in _iter_py(root):
            tree, _ = _parse(path)
            if tree is None:
                continue
            check_dead_mutation(path, tree, findings)
            check_phantom_keys(path, tree, src_literals, findings)
            check_chained_to_source(path, tree, findings)
            if args.weak:
                check_weak_predicates(path, tree, findings)

    titles = {
        "A": "MUTATION WITHOUT AN ANCHOR CHECK  (the mutant may equal the original)",
        "B": "KEY READ IN A TEST THAT NO src/ CODE EVER PRODUCES",
        "C": "EXPECTED VALUE IMPORTED FROM THE MODULE UNDER TEST (advisory)",
        "D": "BARE TRUTHINESS / is-not-None (advisory)",
    }
    total = 0
    for cls in ("A", "B", "C", "D"):
        hits = findings.get(cls) or []
        if cls == "D" and not args.weak:
            continue
        print(f"\n=== {cls}. {titles[cls]} — {len(hits)} ===")
        for item in hits[:40]:
            rel = os.path.relpath(item[0], _ROOT)
            print(f"  {rel}:{item[1]}  {item[2]}  {item[3]}")
        if len(hits) > 40:
            print(f"  ... and {len(hits) - 40} more")
        total += 0 if cls in ("C", "D") else len(hits)

    print("\n" + "=" * 60)
    print("Locator only. Read the region before calling any hit a defect —")
    print("and a clean run does not prove the suite sound: a fixture that makes")
    print("an oracle vacuous is the same family and is invisible here.")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
