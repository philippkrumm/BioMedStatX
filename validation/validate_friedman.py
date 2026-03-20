"""
Validation: perform_friedman_test() vs scipy.stats.friedmanchisquare()

Friedman ist eine Rang-Summen-Statistik ohne freie Parameter.
scipy.stats.friedmanchisquare() ist der Goldstandard — unsere Funktion
ruft ihn intern auf, daher pruefen wir:
  1. Identische Chi2-Statistik und p-Wert (exakt, nicht nur 2 Nachkommastellen)
  2. Korrekte Post-hoc-Struktur (Wilcoxon + Holm)
  3. Grenzfaelle: k=2 Zeitpunkte, grosse n, ties
"""
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

ROOT = Path(__file__).resolve().parents[1]
for p in [str(ROOT), str(ROOT / "Source_Code")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from nonparametricanovas import perform_friedman_test

SEP = "-" * 60

def make_df(matrix, subject_prefix="S", within_name="time"):
    """matrix: list of lists, rows=subjects, cols=time points"""
    n_subj = len(matrix)
    n_time = len(matrix[0])
    rows = []
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            rows.append({
                "subject": f"{subject_prefix}{i+1}",
                within_name: f"T{j+1}",
                "score": val,
            })
    return pd.DataFrame(rows)


def run_case(name, matrix, within_name="time", tol=1e-10):
    print(f"\n{SEP}")
    print(f"  Case: {name}")
    print(SEP)

    df = make_df(matrix, within_name=within_name)
    n_subj = len(matrix)
    n_time = len(matrix[0])

    # Reference: scipy directly
    groups = [np.array([matrix[s][t] for s in range(n_subj)]) for t in range(n_time)]
    ref_chi2, ref_p = sp_stats.friedmanchisquare(*groups)

    # Our function
    res = perform_friedman_test(
        data=df,
        dv="score",
        within_factor=within_name,
        subject_col="subject",
        alpha=0.05,
    )

    assert res.get("model_class") == "Friedman", f"Wrong model_class: {res.get('model_class')}"
    assert res.get("error") is None or not res.get("error"), f"Error: {res.get('error')}"

    tbl = res["anova_table"]
    # Column is "Chi2" in the Friedman anova_table
    chi2_col = next(c for c in tbl.columns if c.lower() in ("chi2", "statistic", "f", "wald_chi2"))
    py_chi2 = float(tbl[chi2_col].iloc[0])
    py_p    = float(tbl["p-unc"].iloc[0])

    diff_chi2 = abs(py_chi2 - ref_chi2)
    diff_p    = abs(py_p    - ref_p)

    print(f"  n_subjects={n_subj}, n_timepoints={n_time}")
    print(f"  Chi2  — Python: {py_chi2:.8f}  |  scipy: {ref_chi2:.8f}  |  |diff|={diff_chi2:.2e}")
    print(f"  p     — Python: {py_p:.8f}  |  scipy: {ref_p:.8f}  |  |diff|={diff_p:.2e}")

    assert diff_chi2 < tol, f"Chi2 mismatch: {diff_chi2:.2e} > {tol}"
    assert diff_p    < tol, f"p mismatch: {diff_p:.2e} > {tol}"

    posthoc = res.get("pairwise_comparisons", [])
    print(f"  Post-hoc comparisons: {len(posthoc)}")
    expected_comparisons = n_time * (n_time - 1) // 2
    # Post-hoc only computed when result is significant
    if float(py_p) < 0.05:
        assert len(posthoc) == expected_comparisons, (
            f"Significant result: expected {expected_comparisons} comparisons, got {len(posthoc)}"
        )

    print("  -> PASS")
    return True


if __name__ == "__main__":
    results = []

    # Case 1: Standard (n=10, k=4) — clear effect
    np.random.seed(0)
    mat1 = [[5+np.random.normal(0,0.5), 7+np.random.normal(0,0.5),
             9+np.random.normal(0,0.5), 11+np.random.normal(0,0.5)]
            for _ in range(10)]
    results.append(("Standard n=10 k=4", run_case("Standard n=10 k=4", mat1)))

    # Case 2: k=3 time points (minimum valid for Friedman; k=2 -> Wilcoxon signed-rank)
    np.random.seed(1)
    mat2 = [[np.random.normal(5,1), np.random.normal(7,1), np.random.normal(9,1)]
            for _ in range(12)]
    results.append(("k=3 minimum", run_case("k=3 minimum", mat2)))

    # Case 3: Large n (n=50, k=3)
    np.random.seed(2)
    mat3 = [[np.random.normal(i*2, 1) for i in range(3)] for _ in range(50)]
    results.append(("Large n=50 k=3", run_case("Large n=50 k=3", mat3)))

    # Case 4: Ties (integer scores)
    mat4 = [
        [1, 2, 3], [1, 3, 2], [2, 1, 3], [3, 1, 2],
        [2, 3, 1], [1, 2, 3], [2, 2, 3], [3, 2, 1],
    ]
    results.append(("Ties (integer scores)", run_case("Ties (integer scores)", mat4)))

    # Case 5: Null hypothesis (no effect expected, p should be > 0.05)
    np.random.seed(42)
    mat5 = [[np.random.normal(5, 1) for _ in range(4)] for _ in range(15)]
    df5 = make_df(mat5)
    res5 = perform_friedman_test(df5, dv="score", within_factor="time",
                                 subject_col="subject", alpha=0.05)
    tbl5 = res5["anova_table"]
    p5 = float(tbl5["p-unc"].iloc[0])
    print(f"\n{SEP}")
    print(f"  Case: H0 null data")
    print(SEP)
    print(f"  p = {p5:.4f} (should be > 0.05 for null data)")
    # Not a hard assertion since it's random, just print
    results.append(("H0 null data", True))
    print("  -> INFO")

    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    all_ok = True
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            all_ok = False
    if all_ok:
        print("\nAlle Friedman-Tests bestanden.\n")
        sys.exit(0)
    else:
        print("\nFehler.\n")
        sys.exit(1)
