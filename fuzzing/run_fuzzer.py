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
from concurrent.futures import ThreadPoolExecutor

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
    written = [r for r in records if r.get("report_written")]
    # A blocked run still writes a report, honestly, with placeholders -- and
    # almost nothing to check. Substance separates "thin because rightly gated"
    # from "thin because the checks did not apply"; the firing count cannot.
    substantial = [r for r in written
                   if (r.get("report_stats") or {}).get("figures")]
    return {"designs": dict(designs), "mutations": dict(mutations),
            "oracles_fired": dict(oracles), "posthoc_tests": dict(posthocs),
            "estimated_p_values": sum(1 for r in records if r.get("had_resolution")),
            "reports_written": len(written),
            "report_files": sum(r.get("reports_written") or 0 for r in records),
            "multi_dataset_runs": sum(1 for r in records if (r.get("datasets") or 1) > 1),
            "reports_with_a_figure": len(substantial),
            "empty_state_blocks": sum((r.get("report_stats") or {}).get("empty_states", 0)
                                      for r in written)}


def _calibration(records, alpha=0.05):
    """Do the answers track the effects the data were built with?

    A statement about a RUN, never about a seed. One null term coming back
    significant is exactly what alpha promises; one strong effect being missed is
    what finite power means. Only the rates are evidence.

    Counted per TERM, not per seed, and each term is held against the p-value for
    that same term. Holding a built main effect against the headline measures the
    wrong thing: for a mixed design the headline is the INTERACTION, and these
    designs carry no interaction at all, so the first version of this reported 7%
    "power" while actually reporting the interaction's type-I error.

    Restricted to unmutated, single-dataset seeds. A mutation exists to break an
    assumption -- NaNs, outliers, unequal variances -- so a rejection rate over
    mutated data says nothing about calibration, and a blocked run has no
    p-values to judge. Designs that draw no effect contribute nothing rather than
    counting as null: correlation and regression are always built with one and
    LMM never is, so including them would measure the generator.
    """
    buckets = {"null": [0, 0], "effect": [0, 0]}
    per_design = {}
    unmatched = set()
    not_marginal = 0
    for record in records:
        if record.get("category") != "OK" or record.get("blocked"):
            continue
        if record.get("mutations") != ["none"] or record.get("datasets", 1) != 1:
            continue
        truth = record.get("truth") or {}
        reported = record.get("term_p_values") or {}
        if not truth or not reported:
            continue
        design = per_design.setdefault(
            record.get("test"), {"null": [0, 0], "effect": [0, 0]})
        # A main effect is not judged at all when the design carries an
        # interaction. The truth records a COEFFICIENT; the ANOVA tests the
        # MARGINAL means, and an interaction moves those even where the main
        # coefficient is zero -- so such a term is not a null term and rejecting
        # it is not an error. Measured on 2500 seeds: two-way main effects were
        # rejected 37% of the time where an interaction was present, 6.8% where
        # the design was purely null. The 8.5% that looked like an engine
        # problem was this, and the engine was right.
        has_interaction = any(":" in str(k) and float(v) != 0.0
                              for k, v in truth.items())
        for term, size in truth.items():
            p_value = reported.get(term)
            if not isinstance(p_value, (int, float)):
                unmatched.add(f"{record.get('test')}:{term}")
                continue
            if has_interaction and ":" not in str(term):
                not_marginal += 1
                continue
            kind = "null" if float(size) == 0.0 else "effect"
            for target in (buckets[kind], design[kind]):
                target[0] += 1
                target[1] += int(p_value < alpha)

    def _rate(bucket):
        return (bucket[1] / bucket[0]) if bucket[0] else None

    return {
        "alpha": alpha,
        "null_terms": buckets["null"][0],
        "null_rejected": buckets["null"][1],
        "null_rate": _rate(buckets["null"]),
        "effect_terms": buckets["effect"][0],
        "effect_found": buckets["effect"][1],
        "power": _rate(buckets["effect"]),
        "per_design": per_design,
        # Terms the generator built but the result never reported under that
        # name. Surfaced rather than dropped: silently skipping them is how a
        # calibration ends up measuring three terms and calling it a run.
        "unmatched_terms": sorted(unmatched),
        # Main-effect terms skipped because the design carries an interaction,
        # so the coefficient the generator drew says nothing about the marginal
        # means the test is about. Reported rather than silently dropped.
        "not_marginal_terms": not_marginal,
    }


def _calibration_verdict(calibration, min_terms=60):
    """What the rates say, and whether they say anything yet.

    The bound is deliberately loose. A few hundred null terms estimate a 5% rate
    to about a percentage point either way, several designs share the bucket, and
    the terms within a seed are not independent, so only a gross departure is
    evidence. Below min_terms the honest answer is that the run is too small --
    reporting a rate from twenty terms would repeat the mistake of judging one.
    """
    total = calibration["null_terms"]
    rate = calibration["null_rate"]
    if total < min_terms or rate is None:
        return "too few null terms to judge (%d)" % total
    if rate > 0.15:
        return ("REJECTS THE NULL TOO OFTEN: %.1f%% of %d null terms at alpha=%.2f"
                % (100 * rate, total, calibration["alpha"]))
    if rate < 0.005:
        return ("rejects the null implausibly rarely: %.1f%% of %d null terms"
                % (100 * rate, total))
    return "consistent with alpha (%.1f%% of %d null terms)" % (100 * rate, total)


def _seeds_to_run(start, count, designs):
    """The seeds this run actually spawns a worker for.

    Without a filter, simply start..start+count. With one, seeds whose design
    the run cannot learn from are skipped BEFORE the subprocess -- the design is
    the generator's first draw, so it can be read for a seed without building
    anything. `count` still counts seeds actually run, so every denominator in
    the summary keeps meaning what it says; the span scanned is reported
    separately.
    """
    from fuzzing.generators import design_for_seed
    if not designs:
        yield from range(start, start + count)
        return
    seed = start
    while count > 0:
        if design_for_seed(seed) in designs:
            yield seed
            count -= 1
        seed += 1


_HISTORY = os.path.join(_HERE, "history.jsonl")


def _record_run(summary, oracle_names, path=_HISTORY):
    """One line per run, so "is the discovery rate falling?" can be measured.

    The question the fuzzer is ultimately judged by is whether a week of running
    it still turns up findings, and that is a trend across runs -- not something
    any single run can answer and not something worth trusting to memory.

    The number of oracles is recorded beside the rate, because the easiest way
    to make a discovery rate fall is to stop looking. A rate that drops while
    the oracle count drops with it says nothing about the product. So does one
    compared across different design filters, which is why the filter is kept
    too.
    """
    entry = {
        "when": time.strftime("%Y-%m-%d %H:%M"),
        "start": summary["start"], "count": summary["count"],
        "last_seed": summary["last_seed"], "designs": summary["designs_filter"],
        "elapsed_sec": summary["elapsed_sec"],
        "findings": len(summary["findings"]),
        "categories": summary["categories"],
        "oracles": len(oracle_names),
        "oracles_that_fired": sum(1 for v in summary["coverage"]["oracles_fired"].values() if v),
        "calibration": {k: summary["calibration"][k]
                        for k in ("null_terms", "null_rejected", "null_rate",
                                  "effect_terms", "effect_found", "power")},
    }
    with open(path, "a") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry


def _print_history(path=_HISTORY):
    if not os.path.exists(path):
        print("no runs recorded yet (%s)" % path)
        return 0
    # Per TEN THOUSAND, not per hundred. A 20000-seed run with two findings is
    # 0.01 per 100, which "%.1f" prints as 0.0 -- so the column that exists to
    # show the rate falling showed every real rate as zero, and a genuinely
    # clean run looked identical to one with findings.
    print("  when              seeds  findings  per 10k  oracles  designs")
    for line in open(path):
        try:
            e = json.loads(line)
        except Exception:
            continue
        rate = 10000.0 * e["findings"] / e["count"] if e["count"] else 0.0
        print("  %-16s  %5d  %8d  %7.2f  %3d/%-3d  %s"
              % (e["when"], e["count"], e["findings"], rate,
                 e["oracles_that_fired"], e["oracles"],
                 ",".join(e["designs"]) or "all"))
    print("\n  A falling rate is only evidence while the oracle count holds and the")
    print("  design filter matches -- fewer findings from fewer questions is not")
    print("  progress.")
    return 0


_SEED_FIELDS = ("seed", "category", "test", "mutations", "report_written",
                "reports_written", "oracles_fired", "posthoc", "had_resolution",
                "datasets", "report_stats", "truth", "p_value", "term_p_values",
                "effect_size", "effect_size_type")


def _run_one(seed, args, env):
    """One seed in its own process. No shared state, so seeds are independent."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "fuzzing._worker", str(seed), args.keep_dir],
            cwd=_ROOT, env=env, capture_output=True, text=True, timeout=args.timeout,
        )
        return _classify(seed, proc)
    except subprocess.TimeoutExpired:
        return {"seed": seed, "category": "TIMEOUT"}


def _default_jobs():
    """Workers to run at once, leaving the machine usable.

    Capped so every worker still gets a core of its own: the per-seed timeout is
    wall-clock, so oversubscribing would turn slow-because-queued into a TIMEOUT
    finding that says nothing about the product.
    """
    cores = os.cpu_count() or 2
    return max(1, min(8, cores - 2))


def _summary(records, findings, counts, oracle_names, args, only_designs,
             last_seed, elapsed, complete):
    from fuzzing.generators import MUTATIONS, TEST_TYPES

    coverage = _coverage(records, oracle_names)
    return {"count": args.count, "seeds_run": len(records), "complete": complete,
            "start": args.start, "elapsed_sec": round(elapsed, 1),
            "designs_filter": sorted(only_designs), "last_seed": last_seed,
            "categories": dict(counts), "coverage": coverage,
            "never_fired_oracles": [n for n in oracle_names
                                    if not coverage["oracles_fired"].get(n)],
            "unseen_designs": [d for d in TEST_TYPES if d not in coverage["designs"]],
            "unseen_mutations": [m for m in MUTATIONS if m not in coverage["mutations"]],
            "seeds": [{k: r.get(k) for k in _SEED_FIELDS} for r in records],
            "calibration": _calibration(records),
            "findings": findings}


def _write_report(summary, path):
    """Written to a temporary file and moved into place.

    A long run is otherwise all-or-nothing: a 2000-seed run killed near the end
    left nothing at all behind, having spent forty minutes. It is now written
    every _FLUSH_EVERY seeds, and a reader can tell a partial report from a
    finished one by its "complete" flag rather than by guessing from the count.
    """
    tmp = path + ".partial"
    with open(tmp, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    os.replace(tmp, path)


_FLUSH_EVERY = 100


def main() -> int:
    from fuzzing.generators import TEST_TYPES
    from fuzzing.html_oracles import MULTI_ORACLES, ORACLES

    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--jobs", type=int, default=0,
                    help="seeds to run at once (default: cores - 2, capped at 8)")
    ap.add_argument("--report", default=os.path.join(_HERE, "fuzz_report.json"))
    ap.add_argument("--keep-dir", default=os.path.join(_HERE, "failures"),
                    help="where reports belonging to a finding are copied")
    ap.add_argument("--no-history", action="store_true",
                    help="do not append this run to the history. For test runs: "
                         "a four-seed run in the trend is not a data point, it "
                         "is noise in the one measurement the fuzzer is judged by.")
    ap.add_argument("--history", action="store_true",
                    help="print the recorded runs and their finding rate, then exit")
    ap.add_argument("--designs", default="",
                    help="comma-separated designs to run; other seeds are "
                         "skipped without being spawned. --count still means "
                         "seeds run, so the run walks further up the seed space.")
    args = ap.parse_args()
    if args.history:
        return _print_history()
    if args.jobs < 1:
        args.jobs = _default_jobs()

    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", MPLBACKEND="Agg")
    oracle_names = [name for name, _ in ORACLES] + [name for name, _ in MULTI_ORACLES]
    findings = []
    records = []
    counts = Counter()
    t0 = time.time()

    only_designs = {d.strip() for d in args.designs.split(",") if d.strip()}
    if only_designs - set(TEST_TYPES):
        print("unknown design(s): %s" % ", ".join(sorted(only_designs - set(TEST_TYPES))))
        return 2
    last_seed = args.start
    seeds = list(_seeds_to_run(args.start, args.count, only_designs))
    print("running %d seeds%s on %d worker%s"
          % (len(seeds),
             " (%s)" % ",".join(sorted(only_designs)) if only_designs else "",
             args.jobs, "" if args.jobs == 1 else "s"))

    # Every seed already ran in its own process and shared nothing with the
    # others, so this was one core of fourteen for no reason: 2500 seeds at
    # 1.5 s each took an hour of wall clock while the machine idled. Results are
    # unchanged -- a seed is deterministic and independent of every other -- and
    # pool.map yields in seed order, so findings still print in the order a
    # reader would reproduce them in.
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for record in pool.map(lambda s: _run_one(s, args, env), seeds):
            seed = record["seed"]
            last_seed = max(last_seed, seed)
            cat = record["category"]
            counts[cat] += 1
            records.append(record)
            if cat not in ("OK",):
                findings.append(record)
                print(f"[{seed}] {cat} :: test={record.get('test')} "
                      f"muts={record.get('mutations')}")
                for violation in (record.get("violations") or [])[:4]:
                    print(f"         {violation}")
            if len(records) % _FLUSH_EVERY == 0:
                _write_report(_summary(records, findings, counts, oracle_names,
                                       args, only_designs, last_seed,
                                       time.time() - t0, complete=False),
                              args.report)

    elapsed = time.time() - t0

    summary = _summary(records, findings, counts, oracle_names, args,
                       only_designs, last_seed, elapsed, complete=True)
    _write_report(summary, args.report)
    if not args.no_history:
        _record_run(summary, oracle_names)
    coverage = summary["coverage"]
    never_fired = summary["never_fired_oracles"]
    unseen_designs = summary["unseen_designs"]
    unseen_mutations = summary["unseen_mutations"]

    print("\n=== FUZZ SUMMARY ===")
    for cat, n in counts.most_common():
        print(f"  {cat:18} {n}")
    print(f"  reports written    {coverage['reports_written']}/{args.count}"
          f"  ({coverage['report_files']} files, {coverage['multi_dataset_runs']} multi-dataset runs)")
    print(f"  of those, with an actual figure: {coverage['reports_with_a_figure']}"
          f"  (empty-state blocks total: {coverage['empty_state_blocks']})")

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

    calibration = _calibration(records)
    verdict = _calibration_verdict(calibration)
    print("\n--- calibration (clean single-dataset seeds, one row per term) ---")
    print("  null terms  : %d of %d called significant at alpha=%.2f -> %s"
          % (calibration["null_rejected"], calibration["null_terms"],
             calibration["alpha"], verdict))
    if calibration["power"] is not None:
        print("  real effects: %d of %d found (%.0f%%)"
              % (calibration["effect_found"], calibration["effect_terms"],
                 100 * calibration["power"]))
    for design, both in sorted(calibration["per_design"].items()):
        print("    %-16s null %d/%-4d effect %d/%d"
              % (design, both["null"][1], both["null"][0],
                 both["effect"][1], both["effect"][0]))
    if calibration.get("not_marginal_terms"):
        print("    main effects not judged (design carries an interaction): %d"
              % calibration["not_marginal_terms"])
    if calibration["unmatched_terms"]:
        print("    terms built but never reported under that name: %s"
              % ", ".join(calibration["unmatched_terms"]))

    for label, missing in (("designs", unseen_designs), ("mutations", unseen_mutations),
                           ("oracles NEVER FIRED", never_fired)):
        if missing:
            print(f"  untouched {label}: {', '.join(missing)}")

    print(f"\n  elapsed {elapsed:.1f}s  report -> {args.report}")
    # Non-zero exit if any non-OK finding (useful in CI).
    return 0 if set(counts) <= {"OK"} else 1


if __name__ == "__main__":
    sys.exit(main())
