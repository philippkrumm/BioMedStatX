"""Generate frozen golden-reference values for the Friedman test.

Run manually (not at test time):  python validation/generate_golden_friedman.py

Oracle: R's friedman.test via validation/r_templates/friedman.R.

Verified equivalence (R base friedman.test vs scipy.friedmanchisquare, which the
app calls): the earlier assumption "R base has no tie correction" was FALSIFIED
by direct measurement -- R applies the same tie correction as scipy:
  continuous no-ties  : |d chi2| = 0.0
  tied integers       : |d chi2| = 2.3e-08
  heavy within-row ties: |d chi2| = 2.3e-07
So any data (tied or continuous) validates cleanly against R; friedman.R needed
no change. Tolerance 1e-4 covers the CSV round-trip noise.

The output JSON (tests/golden/references_friedman.json) is consumed by
tests/test_golden_friedman.py, which runs perform_friedman_test and compares the
Chi2 statistic and p-value to these frozen R values. It also checks the post-hoc
comparison-count structure -- the one complementary assertion absorbed from the
now-removed standalone validation/validate_friedman.py.
"""
import json
import os
import shutil
import subprocess
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "r_templates", "friedman.R")
OUT = os.path.join(os.path.dirname(HERE), "tests", "golden", "references_friedman.json")


def _find_rscript():
    rscript = shutil.which("Rscript")
    if rscript:
        return rscript
    for candidate in ("/opt/homebrew/bin/Rscript", "/usr/local/bin/Rscript", "/usr/bin/Rscript"):
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError("Rscript not found on PATH. Install R or add R/bin to PATH.")


def _r_friedman(rscript, matrix):
    """Run the committed friedman.R template on a long CSV; return (chi2, p)."""
    n = len(matrix)
    k = len(matrix[0])
    with tempfile.TemporaryDirectory() as tmp:
        csv = os.path.join(tmp, "d.csv")
        lines = ["Subject,Group,Value"]
        for i in range(n):
            for j in range(k):
                lines.append(f"S{i + 1},T{j + 1},{matrix[i][j]}")
        with open(csv, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        out = subprocess.run([rscript, TEMPLATE, csv],
                             capture_output=True, text=True, check=True)
        line = [ln for ln in out.stdout.strip().splitlines() if ln.strip()][-1]
        # template order: cat(res$p.value, res$statistic)
        p_str, chi2_str = line.split()
        return float(chi2_str), float(p_str)


def build_datasets():
    """(id, matrix) with rows=subjects, cols=timepoints. Mirrors the cases the
    now-removed validate_friedman.py exercised, but validated against R:
    standard/k=3-min/large-n/ties/null. friedman_ties deliberately runs the
    verified tie path."""
    cases = []

    rng = np.random.default_rng(0)
    cases.append(("friedman_std",
                  [[float(5 + 2 * t + rng.normal(0, 0.5)) for t in range(4)] for _ in range(10)]))

    rng = np.random.default_rng(1)
    cases.append(("friedman_k3min",
                  [[float(rng.normal(5 + 2 * t, 1)) for t in range(3)] for _ in range(12)]))

    rng = np.random.default_rng(2)
    cases.append(("friedman_large",
                  [[float(rng.normal(t * 2, 1)) for t in range(3)] for _ in range(50)]))

    cases.append(("friedman_ties",
                  [[1, 2, 3], [1, 3, 2], [2, 1, 3], [3, 1, 2],
                   [2, 3, 1], [1, 2, 3], [2, 2, 3], [3, 2, 1]]))

    rng = np.random.default_rng(42)
    cases.append(("friedman_null",
                  [[float(rng.normal(5, 1)) for _ in range(4)] for _ in range(15)]))

    return cases


def main():
    rscript = _find_rscript()
    out_cases = []
    for cid, matrix in build_datasets():
        matrix = [[float(v) for v in row] for row in matrix]
        n = len(matrix)
        k = len(matrix[0])
        chi2, p_val = _r_friedman(rscript, matrix)
        out_cases.append({
            "id": cid,
            "n_subjects": n,
            "n_timepoints": k,
            "matrix": matrix,
            "expected": {
                "statistic": chi2,
                "p_value": p_val,
                "expected_comparisons": k * (k - 1) // 2,
                "tol": {"statistic": 1e-4, "p_value": 1e-4},
            },
            "reference_source": "R friedman.test via r_templates/friedman.R",
        })
        print(f"  {cid:16s} n={n:3d} k={k}  chi2={chi2:.6f}  p={p_val:.6g}")

    data = {"schema_version": 1,
            "oracle": "R 4.x friedman.test via r_templates/friedman.R",
            "cases": out_cases}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"Wrote {len(out_cases)} Friedman golden cases -> {OUT}")


if __name__ == "__main__":
    main()
