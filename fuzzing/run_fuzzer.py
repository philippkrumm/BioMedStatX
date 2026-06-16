"""Fuzzer orchestrator.

Runs N seeds, each in an isolated subprocess (see _worker.py), and classifies
every outcome. Crashes are reproducible: re-run a failing seed with
`python -m fuzzing._worker <seed>`.

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--report", default=os.path.join(_HERE, "fuzz_report.json"))
    args = ap.parse_args()

    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", MPLBACKEND="Agg")
    findings = []
    counts = Counter()
    t0 = time.time()

    for i in range(args.count):
        seed = args.start + i
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "fuzzing._worker", str(seed)],
                cwd=_ROOT, env=env, capture_output=True, text=True, timeout=args.timeout,
            )
            record = _classify(seed, proc)
        except subprocess.TimeoutExpired:
            record = {"seed": seed, "category": "TIMEOUT"}

        cat = record["category"]
        counts[cat] += 1
        if cat not in ("OK",):
            findings.append(record)
            print(f"[{seed}] {cat} :: test={record.get('test')} muts={record.get('mutations')}")

    elapsed = time.time() - t0
    summary = {"count": args.count, "start": args.start, "elapsed_sec": round(elapsed, 1),
               "categories": dict(counts), "findings": findings}
    with open(args.report, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print("\n=== FUZZ SUMMARY ===")
    for cat, n in counts.most_common():
        print(f"  {cat:18} {n}")
    print(f"  elapsed {elapsed:.1f}s  report -> {args.report}")
    # Non-zero exit if any non-OK finding (useful in CI).
    return 0 if set(counts) <= {"OK"} else 1


if __name__ == "__main__":
    sys.exit(main())
