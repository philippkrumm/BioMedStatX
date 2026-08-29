"""Running seeds side by side must not change what a run finds.

The runner drove one seed at a time, which was one core of fourteen: a
2500-seed run took an hour of wall clock at 1.5 s a seed while the machine
idled. Each seed already ran in its own process and shared nothing with the
others, so the only thing that had to be shown is the thing tested here -- that
the answers are the same either way, and that a slower machine cannot turn a
queued seed into a TIMEOUT finding.
"""
import os
import subprocess
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from fuzzing.run_fuzzer import _default_jobs, _run_one


@pytest.mark.parametrize("cores,expected", [
    (1, 1),    # a single core still runs
    (2, 1),
    (4, 2),
    (14, 8),   # capped: every worker keeps a core, or the timeout lies
    (128, 8),
])
def test_the_default_leaves_the_machine_headroom(monkeypatch, cores, expected):
    monkeypatch.setattr(os, "cpu_count", lambda: cores)
    assert _default_jobs() == expected


def test_a_seed_that_runs_too_long_is_a_timeout_not_a_crash(monkeypatch):
    """The per-seed budget is wall clock, so this is the one thing oversubscribing
    could corrupt -- a queued seed reported as a product finding."""
    def _too_slow(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="worker", timeout=1)

    monkeypatch.setattr(subprocess, "run", _too_slow)
    args = types.SimpleNamespace(keep_dir="", timeout=1)
    record = _run_one(4242, args, {})
    assert record == {"seed": 4242, "category": "TIMEOUT"}


@pytest.mark.slow
def test_the_same_seeds_give_the_same_records_at_any_job_count(tmp_path):
    """Runs real seeds, because the claim is about the real thing.

    A stub could only prove the bookkeeping is order-independent; what has to
    hold is that eight concurrent analyses produce the same records as one.
    """
    import json

    reports = {}
    for jobs in (1, 4):
        report = tmp_path / f"r{jobs}.json"
        subprocess.run(
            [sys.executable, "-m", "fuzzing.run_fuzzer", "--count", "4",
             "--start", "61000", "--jobs", str(jobs),
             "--designs", "rm_anova,mixed_anova,two_way_anova", "--no-history",
             "--report", str(report), "--keep-dir", str(tmp_path / "keep")],
            cwd=os.path.join(os.path.dirname(__file__), ".."),
            env=dict(os.environ, QT_QPA_PLATFORM="offscreen", MPLBACKEND="Agg"),
            capture_output=True, text=True, timeout=600,
        )
        reports[jobs] = json.loads(report.read_text())

    one, many = reports[1], reports[4]
    assert one["categories"] == many["categories"]
    assert one["calibration"] == many["calibration"]

    by_seed = {j: {s["seed"]: s for s in r["seeds"]} for j, r in reports.items()}
    assert set(by_seed[1]) == set(by_seed[4])
    for seed in by_seed[1]:
        for field in ("category", "test", "mutations", "p_value",
                      "term_p_values", "oracles_fired", "posthoc"):
            assert by_seed[1][seed].get(field) == by_seed[4][seed].get(field), (
                f"seed {seed} differs in {field} between 1 and 4 workers")
