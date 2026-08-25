"""Fuzzer orchestrator.

Runs N seeds, each in an isolated subprocess (see _worker.py), and classifies
every outcome. Crashes are reproducible: re-run a failing seed with
`python -m fuzzing._worker <seed>`.

The summary reports coverage, not only outcomes. "300 OK" says nothing about
what was tested: a run can be entirely green because every seed took the same
branch, and the oracles that never fired contributed only the appearance of
safety. Designs, mutations and the per-oracle firing counts are therefore
printed and stored alongside the findings, with oracles that never fired called
out by name.

Usage:
    python -m fuzzing.run_fuzzer --count 300
    python -m fuzzing.run_fuzzer --count 300 --start 1000 --timeout 60
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
    stdout = (proc.stdout or "").strip()
    record = {"seed": seed, "returncode": rc}
    # Worker prints exactly one JSON line on a clean run / handled finding.
    parsed = None
    for line in reversed(stdout.splitlines()):
        if line.startswith("__FUZZ__"):
            try:
                parsed = json.loads(line[len("__FUZZ__"):])
            except Exception:
                parsed = None
            break

    if rc < 0:  # killed by signal — segfault / abort
        record["category"] = "CRASH_SIGNAL"
        record["signal"] = -rc
        record["stderr_tail"] = (proc.stderr or "")[-600:]
    elif rc == 2:
        record["category"] = "ORACLE_VIOLATION"
        record.update(parsed or {})
    elif rc == 3:
        record["category"] = "EXCEPTION"
        record.update(parsed or {})
    elif rc == 0:
        record["category"] = "OK"
        record.update(parsed or {})
    else:
        record["category"] = "UNKNOWN_RC"
        record["stderr_tail"] = (proc.stderr or "")[-600:]
    return record


def _coverage(records: list, oracle_names: list) -> dict:
    designs = Counter()
    mutations = Counter()
    oracles = Counter({name: 0 for name in oracle_names})
    for rec in records:
        if rec.get("test"):
            designs[rec["test"]] += 1
        for mut in rec.get("mutations") or []:
            mutations[mut] += 1
        for name in rec.get("oracles_fired") or []:
            oracles[name] += 1
    posthocs = Counter(rec["posthoc"] for rec in records if rec.get("posthoc"))
    return {"designs": dict(designs), "mutations": dict(mutations),
            "oracles_fired": dict(oracles), "posthoc_tests": dict(posthocs),
            "estimated_p_values": sum(1 for r in records if r.get("had_resolution")),
            "reports_written": sum(1 for r in records if r.get("report_written"))}


def main() -> int:
    from fuzzing.generators import MUTATIONS, TEST_TYPES
    from fuzzing.html_oracles import ORACLES

    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--report", default=os.path.join(_HERE, "fuzz_report.json"))
    ap.add_argument("--keep-dir", default=os.path.join(_HERE, "failures"),
                    help="where reports belonging to a finding are copied")
    args = ap.parse_args()

    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", MPLBACKEND="Agg")
    oracle_names = [name for name, _ in ORACLES]
    findings = []
    records = []
    counts = Counter()
    t0 = time.time()

    for i in range(args.count):
        seed = args.start + i
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "fuzzing._worker", str(seed), args.keep_dir],
                cwd=_ROOT, env=env, capture_output=True, text=True, timeout=args.timeout,
            )
            record = _classify(seed, proc)
        except subprocess.TimeoutExpired:
            record = {"seed": seed, "category": "TIMEOUT"}

        cat = record["category"]
        counts[cat] += 1
        records.append(record)
        if cat not in ("OK",):
            findings.append(record)
            print(f"[{seed}] {cat} :: test={record.get('test')} muts={record.get('mutations')}")
            for violation in (record.get("violations") or [])[:4]:
                print(f"         {violation}")

    elapsed = time.time() - t0
    coverage = _coverage(records, oracle_names)
    never_fired = [n for n in oracle_names if not coverage["oracles_fired"].get(n)]
    unseen_designs = [d for d in TEST_TYPES if d not in coverage["designs"]]
    unseen_mutations = [m for m in MUTATIONS if m not in coverage["mutations"]]

    summary = {"count": args.count, "start": args.start, "elapsed_sec": round(elapsed, 1),
               "categories": dict(counts), "coverage": coverage,
               "never_fired_oracles": never_fired,
               "unseen_designs": unseen_designs, "unseen_mutations": unseen_mutations,
               "seeds": [{k: r.get(k) for k in ("seed", "category", "test", "mutations",
                                                "report_written", "oracles_fired",
                                                "posthoc", "had_resolution")}
                         for r in records],
               "findings": findings}
    with open(args.report, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print("\n=== FUZZ SUMMARY ===")
    for cat, n in counts.most_common():
        print(f"  {cat:18} {n}")
    print(f"  reports written    {coverage['reports_written']}/{args.count}")

    print("\n--- coverage ---")
    print("  designs:   " + ", ".join(f"{k}={v}" for k, v in
                                      sorted(coverage["designs"].items())))
    print("  mutations: " + ", ".join(f"{k}={v}" for k, v in
                                      sorted(coverage["mutations"].items())))
    print("  oracles:   " + ", ".join(f"{k}={v}" for k, v in
                                      sorted(coverage["oracles_fired"].items())))
    print("  post-hoc:  " + (", ".join(f"{k}={v}" for k, v in
                                       sorted(coverage["posthoc_tests"].items())) or "none reached"))
    print(f"  results with an estimated p-value: {coverage['estimated_p_values']}")
    for label, missing in (("designs", unseen_designs), ("mutations", unseen_mutations),
                           ("oracles NEVER FIRED", never_fired)):
        if missing:
            print(f"  untouched {label}: {', '.join(missing)}")

    print(f"\n  elapsed {elapsed:.1f}s  report -> {args.report}")
    # Non-zero exit if any non-OK finding (useful in CI).
    return 0 if set(counts) <= {"OK"} else 1


if __name__ == "__main__":
    sys.exit(main())
