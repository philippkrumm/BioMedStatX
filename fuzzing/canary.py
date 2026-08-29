"""Does the fuzzer still catch the bugs it once caught?

A falling discovery rate is the thing this project wants to see before calling a
release ready -- and it is also exactly what a broken fuzzer produces. Zero
findings means "nothing left to find" and "nothing is being asked" equally well,
and the second is the cheaper way to get there.

So the rate is only evidence alongside this: take a defect the fuzzer really did
find, put it back, and require the fuzzer to find it again. A quiet run next to a
caught canary is evidence. A quiet run next to a missed canary means the
instrument died and the trend was measuring nothing.

The defect goes back by REVERTING its own fix in a throwaway worktree, not by
editing a string in place. A mutation applied by search-and-replace that matches
nothing looks exactly like a mutation the fuzzer failed to catch, and this
repository has already produced that false result more than once.

    python -m fuzzing.canary            # every canary
    python -m fuzzing.canary --only negative-F
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# Each entry is a defect the fuzzer found for real, the commit that fixed it,
# and the words its finding carried. The seed window is deliberately a RANGE
# rather than the seed that originally caught it: the generator has changed
# since, so a fixed seed no longer builds the same data, and pinning one would
# make this fail for a reason that has nothing to do with the product.
CANARIES = [
    {
        "name": "negative-F",
        "what": "an incomplete factorial reported F = -3.07 as an ordinary result",
        "fix": "6fc3b41",
        "expect": "is negative",
        "start": 1300, "count": 300, "designs": "two_way_anova",
    },
    {
        "name": "transformed-pairing",
        "what": "the Transformed column printed against a different extraction",
        # 63e2cbe, not the earlier 4499ef3: a later commit reworked the same
        # code, so reverting the first one conflicts. The canary follows the
        # commit that holds the guard TODAY.
        "fix": "63e2cbe",
        "expect": "transformed column does not follow the raw one",
        "start": 50000, "count": 400, "designs": "two_way_anova,rm_anova,mixed_anova",
    },
]


def _run(cmd, cwd, **kw):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, **kw)


def _findings_with(report_path, phrase):
    """Every finding whose violations mention the phrase."""
    with open(report_path) as fh:
        report = json.load(fh)
    hits = []
    for finding in report.get("findings") or []:
        for violation in finding.get("violations") or []:
            if phrase.lower() in violation.lower():
                hits.append((finding.get("seed"), violation))
                break
    return hits, report


def _check_one(canary, timeout, jobs):
    """Put the defect back in a worktree of its own and look for its finding."""
    with tempfile.TemporaryDirectory() as tmp:
        tree = os.path.join(tmp, "tree")
        add = _run(["git", "worktree", "add", "--detach", tree, "HEAD"], _ROOT)
        if add.returncode != 0:
            return {"status": "SETUP FAILED", "detail": add.stderr.strip()[-300:]}
        try:
            revert = _run(["git", "revert", "--no-commit", canary["fix"]], tree)
            if revert.returncode != 0:
                # Worth saying plainly: a conflict means the canary needs
                # rewriting, not that the fuzzer failed.
                return {"status": "REVERT FAILED",
                        "detail": (revert.stderr or revert.stdout).strip()[-300:]}

            report = os.path.join(tmp, "canary.json")
            cmd = [sys.executable, "-m", "fuzzing.run_fuzzer",
                   "--count", str(canary["count"]), "--start", str(canary["start"]),
                   "--timeout", str(timeout), "--no-history",
                   "--report", report, "--keep-dir", os.path.join(tmp, "keep")]
            if canary.get("designs"):
                cmd += ["--designs", canary["designs"]]
            if jobs:
                cmd += ["--jobs", str(jobs)]
            env = dict(os.environ, QT_QPA_PLATFORM="offscreen", MPLBACKEND="Agg")
            run = _run(cmd, tree, env=env)
            if not os.path.exists(report):
                return {"status": "RUN FAILED",
                        "detail": (run.stderr or run.stdout).strip()[-400:]}

            hits, full = _findings_with(report, canary["expect"])
            return {"status": "caught" if hits else "MISSED",
                    "hits": len(hits), "seeds_run": full.get("seeds_run"),
                    "all_findings": len(full.get("findings") or []),
                    "example": hits[0][1][:160] if hits else None}
        finally:
            _run(["git", "worktree", "remove", "--force", tree], _ROOT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="run just this canary by name")
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--jobs", type=int, default=0)
    args = ap.parse_args()

    wanted = [c for c in CANARIES if not args.only or c["name"] == args.only]
    if not wanted:
        print("no canary named %r; have: %s"
              % (args.only, ", ".join(c["name"] for c in CANARIES)))
        return 2

    print("putting %d defect%s back and asking the fuzzer to find them again\n"
          % (len(wanted), "" if len(wanted) == 1 else "s"))
    results = []
    for canary in wanted:
        print("--- %s: %s" % (canary["name"], canary["what"]))
        print("    reverting %s, %d seeds from %d"
              % (canary["fix"], canary["count"], canary["start"]))
        t0 = time.time()
        outcome = _check_one(canary, args.timeout, args.jobs)
        outcome["name"] = canary["name"]
        results.append(outcome)
        if outcome["status"] == "caught":
            print("    caught: %d of %d findings carried it (%d seeds, %.0fs)"
                  % (outcome["hits"], outcome["all_findings"],
                     outcome["seeds_run"], time.time() - t0))
            print("    %s" % outcome["example"])
        elif outcome["status"] == "MISSED":
            print("    MISSED: the defect was back and the fuzzer did not report it")
            print("    (%d seeds, %d other findings) -- the trend is measuring nothing"
                  % (outcome["seeds_run"], outcome["all_findings"]))
        else:
            print("    %s: %s" % (outcome["status"], outcome.get("detail")))
        print()

    caught = sum(1 for r in results if r["status"] == "caught")
    print("=== %d of %d canaries caught ===" % (caught, len(results)))
    for r in results:
        if r["status"] != "caught":
            print("    %-22s %s" % (r["name"], r["status"]))
    return 0 if caught == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
