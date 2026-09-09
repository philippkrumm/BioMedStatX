"""Generate frozen golden-reference values for correlation (Pearson / Spearman).

Run manually (not at test time):  python validation/generate_golden_correlation.py

Oracle: R's cor.test via validation/r_templates/correlation.R. That template is
invoked with exact=FALSE so R uses the large-sample t-approximation for Spearman
(p = pt(r/sqrt((1-r^2)/(n-2)), n-2)) -- exactly what the app computes through
scipy.spearmanr / its explicit t-recompute. R's default (exact permutation for
small n without ties) would diverge from the app; the golden must validate the
app's actual method, not a different one. Pearson is t-based in both and matches
regardless.

Proven alignment (R vs scipy/app, this dataset design):
  spearman small-n no-ties : R exact diverges 4e-3, exact=FALSE matches 5e-9
  spearman small-n WITH ties: app == R  rho 1e-8, p 1e-13

The output JSON (tests/golden/references_correlation.json) is consumed by
tests/test_golden_correlation.py, which runs CorrelationModel().fit(method=...)
and compares r / p to these frozen R values. Datasets are embedded so the test
is self-contained and never re-runs R.

Note: the pure constant-predictor case (Wave-2 B4) yields r = NaN in both R and
scipy, so it is not a numeric golden comparison -- that guard is covered by the
dedicated correlation tests, not here.
"""
import json
import os
import shutil
import subprocess
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "r_templates", "correlation.R")
OUT = os.path.join(os.path.dirname(HERE), "tests", "golden", "references_correlation.json")


def _find_rscript():
    rscript = shutil.which("Rscript")
    if rscript:
        return rscript
    for candidate in ("/opt/homebrew/bin/Rscript", "/usr/local/bin/Rscript", "/usr/bin/Rscript"):
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError("Rscript not found on PATH. Install R or add R/bin to PATH.")


def _r_cor(rscript, x, y, method):
    """Run the committed correlation.R template; return (p_value, r)."""
    with tempfile.TemporaryDirectory() as tmp:
        csv = os.path.join(tmp, "d.csv")
        with open(csv, "w") as fh:
            fh.write("X,Y\n" + "\n".join(f"{a},{b}" for a, b in zip(x, y)) + "\n")
        out = subprocess.run([rscript, TEMPLATE, csv, method],
                             capture_output=True, text=True, check=True)
        # template's last non-empty line: "p_value r_estimate"
        line = [ln for ln in out.stdout.strip().splitlines() if ln.strip()][-1]
        p_str, r_str = line.split()
        return float(p_str), float(r_str)


def build_datasets():
    """Deterministic (x, y, method) cases. spearman_smalln is the case where R's
    exact default would have diverged -> it proves the exact=FALSE oracle choice;
    spearman_ties exercises the tie path that Friedman flagged as must-verify."""
    cases = []

    rng = np.random.default_rng(20240727)
    x = rng.normal(0, 1, 30)
    y = 0.7 * x + rng.normal(0, 0.8, 30)
    cases.append(("pearson_pos", "pearson", x, y))

    rng = np.random.default_rng(101)
    x = rng.normal(5, 2, 50)
    y = -0.5 * x + rng.normal(0, 1.5, 50)
    cases.append(("pearson_neg", "pearson", x, y))

    rng = np.random.default_rng(202)
    x = rng.normal(0, 1, 25)
    y = 0.1 * x + rng.normal(0, 1, 25)
    cases.append(("pearson_weak", "pearson", x, y))

    rng = np.random.default_rng(303)
    x = rng.normal(0, 1, 30)
    y = np.sign(x) * np.abs(x) ** 1.5 + rng.normal(0, 0.3, 30)  # monotone non-linear
    cases.append(("spearman_mono", "spearman", x, y))

    rng = np.random.default_rng(404)
    x = rng.normal(0, 1, 40)
    y = -np.exp(x * 0.5) + rng.normal(0, 0.2, 40)  # monotone decreasing
    cases.append(("spearman_neg", "spearman", x, y))

    rng = np.random.default_rng(505)
    x = rng.normal(0, 1, 12)
    y = 0.6 * x + rng.normal(0, 1, 12)  # small n -> R exact would diverge
    cases.append(("spearman_smalln", "spearman", x, y))

    # heavy tied ranks in both variables, small n
    x = np.array([1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6], dtype=float)
    y = np.array([2, 1, 2, 3, 3, 4, 4, 5, 4, 6, 5, 6], dtype=float)
    cases.append(("spearman_ties", "spearman", x, y))

    return cases


def main():
    rscript = _find_rscript()
    out_cases = []
    for cid, method, x, y in build_datasets():
        x = [float(v) for v in x]
        y = [float(v) for v in y]
        p_val, r_val = _r_cor(rscript, x, y, method)
        out_cases.append({
            "id": cid,
            "method": method,
            "n": len(x),
            "x": x,
            "y": y,
            "expected": {
                "r": r_val,
                "p_value": p_val,
                "tol": {"r": 1e-6, "p_value": 1e-6},
            },
            "reference_source": f"R cor.test(method='{method}', exact=FALSE) via r_templates/correlation.R",
        })
        print(f"  {cid:16s} {method:9s} n={len(x):3d}  r={r_val:+.6f}  p={p_val:.6g}")

    data = {"schema_version": 1,
            "oracle": "R 4.x cor.test (exact=FALSE) via r_templates/correlation.R",
            "cases": out_cases}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"Wrote {len(out_cases)} correlation golden cases -> {OUT}")


if __name__ == "__main__":
    main()
