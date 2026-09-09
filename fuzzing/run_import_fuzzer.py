"""Import/mapping fuzzer orchestrator.

Runs N seeds, each in an isolated subprocess (see ``_import_worker.py``), where
a seed is a real ``.csv`` or ``.xlsx`` written to disk and then opened by the
real window. Reproduce a seed with ``python -m fuzzing._import_worker <seed>``.

The same rule as the analysis fuzzer applies to the summary, and it bit here
immediately: a seed whose file failed to load fired *no oracle at all* and still
counted as OK, which is precisely the shape "300 OK" is designed to hide. Firing
counts are therefore printed per oracle, seeds that fired nothing are counted
and named, and the split between files the app is expected to parse and files it
is only expected to refuse visibly is reported separately -- a run made entirely
of the latter would be green without having checked a single value.

Usage:
    python -m fuzzing.run_import_fuzzer --count 200
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
        if line.startswith("__IMPORT_FUZZ__"):
            try:
                parsed = json.loads(line[len("__IMPORT_FUZZ__"):])
            except Exception:
                parsed = None
            break

    if rc < 0:
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
    formats = Counter()
    mutations = Counter()
    oracles = Counter({name: 0 for name in oracle_names})
    for rec in records:
        if rec.get("file_format"):
            formats[rec["file_format"]] += 1
        for mut in rec.get("mutations") or []:
            mutations[mut] += 1
        for name in rec.get("oracles_fired") or []:
            oracles[name] += 1

    handled = [r for r in records if r.get("status")]
    # The distinction the firing counts alone cannot make: a file the app is
    # expected to read exactly is where the value-level checks live. A run made
    # only of files it may legitimately refuse checks nothing about parsing.
    faithful = [r for r in handled if r.get("faithful_expected")]
    return {
        "file_formats": dict(formats),
        "mutations": dict(mutations),
        "oracles_fired": dict(oracles),
        "files_expected_to_parse": len(faithful),
        "files_only_expected_to_refuse": len(handled) - len(faithful),
        "loaded": sum(1 for r in handled if r.get("loaded")),
        "dv_numeric": sum(1 for r in handled if r.get("dv_numeric")),
        "mapping_ready": sum(1 for r in handled if r.get("start_enabled")),
        "wide_pivoted": sum(1 for r in handled if r.get("wide_pivoted")),
        # Runs the fuzzer cancelled itself, by answering a mid-analysis dialog
        # with cancel. Counted because a run that ends this way exercises the
        # abort path and NOT the analysis, so a rise here is coverage quietly
        # draining away rather than the app getting better.
        "cancelled_by_the_fuzzer": sum(1 for r in handled if r.get("cancelled")),
        "seeds_firing_nothing": [r["seed"] for r in handled
                                 if not (r.get("oracles_fired") or [])],
    }


def main() -> int:
    from fuzzing.html_oracles import MULTI_ORACLES, ORACLES as REPORT_ORACLES
    from fuzzing.import_generators import FILE_FORMATS, MUTATIONS
    from fuzzing.import_oracles import ORACLES

    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--report", default=os.path.join(_HERE, "import_fuzz_report.json"))
    args = ap.parse_args()

    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", MPLBACKEND="Agg")
    # The chain runs on past the mapping now, so the report oracles are part
    # of this run's coverage too -- they fire here without any change of their
    # own, which is the whole point of joining the two spans.
    oracle_names = ([name for name, _ in ORACLES]
                    + [name for name, _ in REPORT_ORACLES]
                    + [name for name, _ in MULTI_ORACLES])
    findings, records = [], []
    counts = Counter()
    t0 = time.time()

    for i in range(args.count):
        seed = args.start + i
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "fuzzing._import_worker", str(seed)],
                cwd=_ROOT, env=env, capture_output=True, text=True, timeout=args.timeout,
            )
            record = _classify(seed, proc)
        except subprocess.TimeoutExpired:
            record = {"seed": seed, "category": "TIMEOUT"}

        cat = record["category"]
        counts[cat] += 1
        records.append(record)
        if cat != "OK":
            findings.append(record)
            print(f"[{seed}] {cat} :: {record.get('file_format')} "
                  f"muts={record.get('mutations')}")
            for violation in (record.get("violations") or [])[:4]:
                print(f"         {violation}")

    elapsed = time.time() - t0
    coverage = _coverage(records, oracle_names)
    never_fired = [n for n in oracle_names if not coverage["oracles_fired"].get(n)]
    unseen_formats = [f for f in FILE_FORMATS if f not in coverage["file_formats"]]
    unseen_mutations = [m for m in MUTATIONS if m not in coverage["mutations"]]

    summary = {"count": args.count, "start": args.start, "elapsed_sec": round(elapsed, 1),
               "categories": dict(counts), "coverage": coverage,
               "never_fired_oracles": never_fired,
               "unseen_file_formats": unseen_formats,
               "unseen_mutations": unseen_mutations,
               "seeds": [{k: r.get(k) for k in
                          ("seed", "category", "file_format", "mutations", "header_row",
                           "faithful_expected", "loaded", "dv_numeric", "start_enabled",
                           "wide_pivoted", "cancelled", "oracles_fired")}
                         for r in records],
               "findings": findings}
    with open(args.report, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print("\n=== IMPORT FUZZ SUMMARY ===")
    for cat, n in counts.most_common():
        print(f"  {cat:18} {n}")
    print(f"  files the app must parse exactly : {coverage['files_expected_to_parse']}")
    print(f"  files it may only refuse visibly : {coverage['files_only_expected_to_refuse']}")
    print(f"  loaded {coverage['loaded']}  |  measurement column numeric "
          f"{coverage['dv_numeric']}  |  mapping ready {coverage['mapping_ready']}"
          f"  |  wide-pivoted {coverage['wide_pivoted']}"
          f"  |  cancelled by the fuzzer {coverage['cancelled_by_the_fuzzer']}")

    print("\n--- coverage ---")
    print("  formats:   " + ", ".join(f"{k}={v}" for k, v in sorted(coverage["file_formats"].items())))
    print("  mutations: " + (", ".join(f"{k}={v}" for k, v in sorted(coverage["mutations"].items()))
                             or "none (every seed clean)"))
    print("  oracles:   " + ", ".join(f"{k}={v}" for k, v in sorted(coverage["oracles_fired"].items())))
    for label, missing in (("file formats", unseen_formats),
                           ("mutations", unseen_mutations),
                           ("oracles NEVER FIRED", never_fired)):
        if missing:
            print(f"  untouched {label}: {', '.join(missing)}")
    blind = coverage["seeds_firing_nothing"]
    if blind:
        print(f"  seeds that fired NO oracle at all ({len(blind)}): "
              f"{', '.join(str(s) for s in blind[:15])}"
              + (" ..." if len(blind) > 15 else ""))

    print(f"\n  elapsed {elapsed:.1f}s  report -> {args.report}")
    return 0 if not findings else 1


if __name__ == "__main__":
    sys.exit(main())
