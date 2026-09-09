"""Visual fuzzer orchestrator.

Runs N seeds, each in its own process (see _visual_worker.py), where a seed
means: produce a real report, open it in headless Chromium, and use the figure
builder the way a reader does -- switch the plot type, rename an axis, move the
legend, press a preset, export the figure.

Coverage is reported for the same reason as in the other two fuzzers: "200 OK"
says nothing on its own. A run that never reached a Violin, never pressed a
preset and never got a letters display is green because it looked away, not
because the product is sound. Plot types, actions and per-oracle firing counts
are printed, and oracles that never fired are named.

Usage:
    python -m fuzzing.run_visual_fuzzer --count 60
    python -m fuzzing.run_visual_fuzzer --count 60 --start 500 --timeout 180
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


def _classify(seed: int, proc: subprocess.CompletedProcess) -> dict:
    rc = proc.returncode
    record = {"seed": seed, "returncode": rc}
    parsed = None
    for line in reversed((proc.stdout or "").splitlines()):
        if line.startswith("__FUZZ__"):
            try:
                parsed = json.loads(line[len("__FUZZ__"):])
            except Exception:
                parsed = None
            break

    if rc < 0:
        record["category"] = "CRASH_SIGNAL"
        record["signal"] = -rc
        record["stderr_tail"] = (proc.stderr or "")[-600:]
    elif rc == 2:
        record["category"] = "ORACLE_VIOLATION"
    elif rc == 3:
        record["category"] = "EXCEPTION"
    elif rc == 0:
        record["category"] = "OK"
    else:
        record["category"] = "UNKNOWN_RC"
        record["stderr_tail"] = (proc.stderr or "")[-600:]
    record.update(parsed or {})
    return record


def _coverage(records, oracle_names):
    oracles = Counter({name: 0 for name in oracle_names})
    plot_types = Counter()
    actions = Counter()
    designs = Counter()
    warnings = Counter()
    modes = Counter()
    for rec in records:
        for mode in rec.get("significance_rendered") or []:
            modes[mode] += 1
        for name in rec.get("oracles_fired") or []:
            oracles[name] += 1
        for label in rec.get("plan") or []:
            if str(label).startswith("type:"):
                plot_types[label[5:]] += 1
            elif str(label).startswith("download:"):
                actions[label] += 1
            else:
                actions[label] += 1
        if rec.get("test"):
            designs[rec["test"]] += 1
        for text in rec.get("designer_warnings") or []:
            warnings[text] += 1
    driven = [r for r in records if r.get("stages")]
    return {"oracles_fired": dict(oracles), "plot_types": dict(plot_types),
            "actions": dict(actions), "designs": dict(designs),
            "designer_warnings": dict(warnings),
            "significance_rendered": dict(modes),
            # A seed whose analysis was blocked writes no report and drives no
            # browser. Counting those as covered would be the same lie as
            # counting a blocked run as a tested one.
            "seeds_with_a_report": sum(1 for r in records if r.get("reports")),
            "seeds_driven": len(driven),
            "stages": sum(r.get("stages") or 0 for r in records)}


def main() -> int:
    from fuzzing.visual_generators import BUTTON_ACTIONS
    from fuzzing.visual_oracles import ORACLES

    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=60)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--report", default=os.path.join(_HERE, "visual_fuzz_report.json"))
    ap.add_argument("--keep-dir", default=os.path.join(_HERE, "failures"))
    args = ap.parse_args()

    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", MPLBACKEND="Agg")
    oracle_names = [name for name, _ in ORACLES] + ["download_svg", "download_png"]
    findings, records = [], []
    counts = Counter()
    t0 = time.time()

    for i in range(args.count):
        seed = args.start + i
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "fuzzing._visual_worker", str(seed), args.keep_dir],
                cwd=_ROOT, env=env, capture_output=True, text=True, timeout=args.timeout)
            record = _classify(seed, proc)
        except subprocess.TimeoutExpired:
            record = {"seed": seed, "category": "TIMEOUT"}

        counts[record["category"]] += 1
        records.append(record)
        if record["category"] != "OK":
            findings.append(record)
            print(f"[{seed}] {record['category']} :: test={record.get('test')} "
                  f"muts={record.get('mutations')}")
            for violation in (record.get("violations") or [])[:4]:
                print(f"         {violation}")
            if record.get("error"):
                print(f"         {record['error']}")

    elapsed = time.time() - t0
    coverage = _coverage(records, oracle_names)
    never_fired = [n for n in oracle_names if not coverage["oracles_fired"].get(n)]
    unseen_actions = [a for a in BUTTON_ACTIONS if not coverage["actions"].get(a)]

    summary = {"count": args.count, "start": args.start, "elapsed_sec": round(elapsed, 1),
               "categories": dict(counts), "coverage": coverage,
               "never_fired_oracles": never_fired, "unseen_actions": unseen_actions,
               "seeds": [{k: r.get(k) for k in ("seed", "category", "test", "mutations",
                                                "reports", "stages", "oracles_fired", "plan",
                                                "significance_rendered")}
                         for r in records],
               "findings": findings}
    with open(args.report, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print("\n=== VISUAL FUZZ SUMMARY ===")
    for cat, n in counts.most_common():
        print(f"  {cat:18} {n}")
    print(f"  seeds that produced a report: {coverage['seeds_with_a_report']}/{args.count}"
          f"   driven in the browser: {coverage['seeds_driven']}"
          f"   stages checked: {coverage['stages']}")

    print("\n--- coverage ---")
    print("  plot types: " + (", ".join(f"{k}={v}" for k, v in sorted(coverage["plot_types"].items()))
                              or "none reached"))
    print("  oracles:    " + ", ".join(f"{k}={v}" for k, v in sorted(coverage["oracles_fired"].items())))
    print("  significance on screen: "
          + (", ".join(f"{k}={v}" for k, v in sorted(coverage["significance_rendered"].items()))
             or "never rendered"))
    top = sorted(coverage["actions"].items(), key=lambda kv: -kv[1])[:12]
    print("  actions:    " + (", ".join(f"{k}={v}" for k, v in top) or "none"))
    if coverage["designer_warnings"]:
        print("  designer said:")
        for text, n in sorted(coverage["designer_warnings"].items(), key=lambda kv: -kv[1])[:6]:
            print(f"      {n:>4}x  {text}")
    for label, missing in (("actions", unseen_actions), ("oracles NEVER FIRED", never_fired)):
        if missing:
            print(f"  untouched {label}: {', '.join(missing)}")

    print(f"\n  elapsed {elapsed:.1f}s  report -> {args.report}")
    return 0 if set(counts) <= {"OK"} else 1


if __name__ == "__main__":
    sys.exit(main())
