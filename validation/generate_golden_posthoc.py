"""Generate frozen golden-reference values for the pairwise post-hoc tests:
Tukey HSD, Games-Howell, and Dunn (raw).

Run manually (not at test time):  python validation/generate_golden_posthoc.py

Oracles (chosen to match the app's actual implementation, verified per method on
a probe dataset before freezing -- max |dp|: Tukey 1.7e-5 [statsmodels rounds
p-adj to 4 dp], Games-Howell 4e-9, Dunn-raw 4e-10):

  Tukey HSD      -> Base R stats::TukeyHSD(aov(...))   [r_templates/tukey.R]
                   canonical target of statsmodels.pairwise_tukeyhsd.
  Games-Howell   -> PMCMRplus::gamesHowellTest         [r_templates/games_howell.R]
                   same sqrt(2)*t / studentized-range / Welch-df formula the app
                   hand-rolls. All golden groups have n>=2 so the app's
                   k=#(n>=2) equals R's total group count; the n<2 exclusion is a
                   separate reachability guard test, not a numeric case.
  Dunn (raw)     -> PMCMRplus::kwAllPairsDunnTest(p.adjust="none")  [r_templates/dunn.R]
                   validates ONLY the rank-based tie-corrected raw p-value. The
                   app's Holm-Sidak multiplicity step is NOT validated against R
                   (no bit-identical R equivalent -> second-oracle risk); it is a
                   statsmodels unit, and the raw->adjusted seam is a wiring test.

Outputs three JSONs into tests/golden/, consumed by tests/test_golden_tukey.py,
test_golden_games_howell.py, test_golden_dunn.py. Datasets are embedded so the
tests never re-run R.
"""
import json
import os
import shutil
import subprocess
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATES = os.path.join(HERE, "r_templates")
GOLDEN = os.path.join(os.path.dirname(HERE), "tests", "golden")


def _find_rscript():
    rscript = shutil.which("Rscript")
    if rscript:
        return rscript
    for candidate in ("/opt/homebrew/bin/Rscript", "/usr/local/bin/Rscript", "/usr/bin/Rscript"):
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError("Rscript not found on PATH. Install R or add R/bin to PATH.")


def _run_template(rscript, template, samples):
    """Write samples as a long Group,Value CSV, run the R template, return the
    list of whitespace-split output lines (each line ends in the numeric value)."""
    with tempfile.TemporaryDirectory() as tmp:
        csv = os.path.join(tmp, "d.csv")
        lines = ["Group,Value"]
        for g, vals in samples.items():
            for v in vals:
                lines.append(f"{g},{v}")
        with open(csv, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        out = subprocess.run([rscript, os.path.join(TEMPLATES, template), csv],
                             capture_output=True, text=True, check=True)
        return [ln.split() for ln in out.stdout.strip().splitlines() if ln.strip()]


def _norm(g1, g2):
    return sorted([str(g1), str(g2)])


# ---------------- datasets ----------------

def _tukey_datasets():
    cases = {}
    rng = np.random.default_rng(31)
    cases["tukey_3bal"] = {g: list(rng.normal(m, 2.0, 8))
                           for g, m in {"A": 10, "B": 13, "C": 16}.items()}
    rng = np.random.default_rng(32)
    cases["tukey_4unbal"] = {"A": list(rng.normal(5, 1.5, 7)), "B": list(rng.normal(6, 1.5, 9)),
                             "C": list(rng.normal(9, 1.5, 8)), "D": list(rng.normal(6, 1.5, 10))}
    rng = np.random.default_rng(33)
    cases["tukey_null"] = {g: list(rng.normal(5, 1.5, 10)) for g in ("A", "B", "C")}
    return cases


def _games_howell_datasets():
    cases = {}
    rng = np.random.default_rng(41)
    cases["gh_3het"] = {"A": list(rng.normal(10, 1.0, 8)), "B": list(rng.normal(13, 3.0, 9)),
                        "C": list(rng.normal(16, 5.0, 7))}
    rng = np.random.default_rng(42)
    cases["gh_4unbal"] = {"A": list(rng.normal(5, 1.0, 6)), "B": list(rng.normal(7, 4.0, 11)),
                          "C": list(rng.normal(9, 2.0, 8)), "D": list(rng.normal(6, 6.0, 9))}
    rng = np.random.default_rng(43)
    cases["gh_null"] = {"A": list(rng.normal(5, 1.0, 9)), "B": list(rng.normal(5, 4.0, 10)),
                        "C": list(rng.normal(5, 2.0, 8))}
    return cases


def _dunn_datasets():
    cases = {}
    rng = np.random.default_rng(51)
    cases["dunn_3skew"] = {"A": list(rng.lognormal(0.0, 0.5, 9)),
                           "B": list(rng.lognormal(0.6, 0.5, 10)),
                           "C": list(rng.lognormal(1.2, 0.5, 8))}
    rng = np.random.default_rng(52)
    cases["dunn_4"] = {"A": list(rng.lognormal(0.0, 0.6, 8)), "B": list(rng.lognormal(0.4, 0.6, 9)),
                       "C": list(rng.lognormal(0.9, 0.6, 10)), "D": list(rng.lognormal(0.5, 0.6, 7))}
    rng = np.random.default_rng(53)
    cases["dunn_null"] = {g: list(rng.lognormal(0.0, 0.6, 10)) for g in ("A", "B", "C")}
    return cases


# ---------------- build + freeze ----------------

def _floatify(samples):
    return {g: [float(v) for v in vals] for g, vals in samples.items()}


def _build(rscript, datasets, template, value_keys):
    """value_keys: list of (json_key, output_column_index) for the numeric fields
    after the two group labels on each R output line. Tukey lines are
    '<g2>-<g1> diff padj' (pair labels fused by '-'); the others are 'g1 g2 v'."""
    out_cases = []
    fused = template == "tukey.R"
    for cid, samples in datasets.items():
        samples = _floatify(samples)
        rows = _run_template(rscript, template, samples)
        comparisons = []
        for row in rows:
            if fused:
                g2, g1 = row[0].split("-")
                nums = row[1:]
            else:
                g1, g2 = row[0], row[1]
                nums = row[2:]
            comp = {"groups": _norm(g1, g2)}
            for key, idx in value_keys:
                comp[key] = float(nums[idx])
            comparisons.append(comp)
        out_cases.append({"id": cid, "groups": sorted(samples.keys()),
                          "samples": samples, "comparisons": comparisons})
    return out_cases


def main():
    rscript = _find_rscript()
    os.makedirs(GOLDEN, exist_ok=True)

    specs = [
        ("references_tukey.json", "Base R stats::TukeyHSD(aov)", _tukey_datasets(), "tukey.R",
         [("diff", 0), ("p_value", 1)], {"p_value": 1e-4, "diff": 1e-4}),
        ("references_games_howell.json", "PMCMRplus::gamesHowellTest", _games_howell_datasets(),
         "games_howell.R", [("p_value", 0)], {"p_value": 1e-6}),
        ("references_dunn.json", "PMCMRplus::kwAllPairsDunnTest(p.adjust='none')", _dunn_datasets(),
         "dunn.R", [("raw_p", 0)], {"raw_p": 1e-6}),
    ]

    for fname, oracle, datasets, template, value_keys, tol in specs:
        cases = _build(rscript, datasets, template, value_keys)
        data = {"schema_version": 1, "oracle": oracle, "tol": tol, "cases": cases}
        path = os.path.join(GOLDEN, fname)
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)
        npairs = sum(len(c["comparisons"]) for c in cases)
        print(f"  {fname:34s} {len(cases)} cases, {npairs} pairs  ({oracle})")

    print("Wrote post-hoc golden references -> tests/golden/")


if __name__ == "__main__":
    main()
