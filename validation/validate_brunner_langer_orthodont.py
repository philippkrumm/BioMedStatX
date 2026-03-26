"""
Validation script: Python perform_brunner_langer_ats() vs R nparLD::f1.ld.f1()

Dataset : Orthodont (nlme package, 27 subjects × 4 time points)
Design  : F1-LD-F1  — Sex (between, 2 levels) × Age (within, 4 levels)

Run:
    python validation/validate_brunner_langer_orthodont.py

Requires R + nparLD for the comparison step.  If Rscript is not on PATH the
script still prints the Python results and tells you how to install R.
"""

import sys
import os
import subprocess
import tempfile
import textwrap

# ---------------------------------------------------------------------------
# 1.  Orthodont data  (public domain, from R::nlme)
#     Columns: distance, age, Subject, Sex
#     16 Male subjects (M01-M16), 11 Female subjects (F01-F11)
#     age ∈ {8, 10, 12, 14}
# ---------------------------------------------------------------------------
ORTHODONT_ROWS = [
    # (distance, age, subject_id, sex)
    (26.0, 8, 1, "Male"),(25.0,10, 1,"Male"),(29.0,12, 1,"Male"),(31.0,14, 1,"Male"),
    (21.5, 8, 2, "Male"),(22.5,10, 2,"Male"),(23.0,12, 2,"Male"),(26.5,14, 2,"Male"),
    (23.0, 8, 3, "Male"),(22.5,10, 3,"Male"),(24.0,12, 3,"Male"),(27.5,14, 3,"Male"),
    (25.5, 8, 4, "Male"),(27.5,10, 4,"Male"),(26.5,12, 4,"Male"),(27.0,14, 4,"Male"),
    (20.0, 8, 5, "Male"),(23.5,10, 5,"Male"),(22.5,12, 5,"Male"),(26.0,14, 5,"Male"),
    (24.5, 8, 6, "Male"),(25.5,10, 6,"Male"),(27.0,12, 6,"Male"),(28.5,14, 6,"Male"),
    (22.0, 8, 7, "Male"),(22.0,10, 7,"Male"),(24.5,12, 7,"Male"),(26.5,14, 7,"Male"),
    (24.0, 8, 8, "Male"),(21.5,10, 8,"Male"),(24.5,12, 8,"Male"),(25.5,14, 8,"Male"),
    (23.0, 8, 9, "Male"),(20.5,10, 9,"Male"),(31.0,12, 9,"Male"),(26.0,14, 9,"Male"),
    (27.5, 8,10, "Male"),(28.0,10,10,"Male"),(31.0,12,10,"Male"),(31.5,14,10,"Male"),
    (23.0, 8,11, "Male"),(23.0,10,11,"Male"),(23.5,12,11,"Male"),(25.0,14,11,"Male"),
    (21.5, 8,12, "Male"),(23.5,10,12,"Male"),(24.0,12,12,"Male"),(28.0,14,12,"Male"),
    (17.0, 8,13, "Male"),(24.5,10,13,"Male"),(26.0,12,13,"Male"),(29.5,14,13,"Male"),
    (22.5, 8,14, "Male"),(25.5,10,14,"Male"),(25.5,12,14,"Male"),(26.0,14,14,"Male"),
    (23.0, 8,15, "Male"),(24.5,10,15,"Male"),(26.0,12,15,"Male"),(30.0,14,15,"Male"),
    (22.0, 8,16, "Male"),(21.5,10,16,"Male"),(23.5,12,16,"Male"),(25.0,14,16,"Male"),
    (21.0, 8,17,"Female"),(20.0,10,17,"Female"),(21.5,12,17,"Female"),(23.0,14,17,"Female"),
    (21.0, 8,18,"Female"),(21.5,10,18,"Female"),(24.0,12,18,"Female"),(25.5,14,18,"Female"),
    (20.5, 8,19,"Female"),(24.0,10,19,"Female"),(24.5,12,19,"Female"),(26.0,14,19,"Female"),
    (23.5, 8,20,"Female"),(24.5,10,20,"Female"),(25.0,12,20,"Female"),(26.5,14,20,"Female"),
    (21.5, 8,21,"Female"),(23.0,10,21,"Female"),(22.5,12,21,"Female"),(23.5,14,21,"Female"),
    (20.0, 8,22,"Female"),(21.0,10,22,"Female"),(21.0,12,22,"Female"),(22.5,14,22,"Female"),
    (21.5, 8,23,"Female"),(22.5,10,23,"Female"),(23.0,12,23,"Female"),(25.0,14,23,"Female"),
    (23.0, 8,24,"Female"),(23.0,10,24,"Female"),(23.5,12,24,"Female"),(24.0,14,24,"Female"),
    (20.0, 8,25,"Female"),(21.0,10,25,"Female"),(22.0,12,25,"Female"),(21.5,14,25,"Female"),
    (16.5, 8,26,"Female"),(19.0,10,26,"Female"),(19.0,12,26,"Female"),(19.5,14,26,"Female"),
    (24.5, 8,27,"Female"),(25.0,10,27,"Female"),(28.0,12,27,"Female"),(28.0,14,27,"Female"),
]

import pandas as pd
import numpy as np

def make_orthodont_df():
    df = pd.DataFrame(ORTHODONT_ROWS, columns=["distance", "age", "subject", "sex"])
    return df


# ---------------------------------------------------------------------------
# 2.  Run Python implementation
# ---------------------------------------------------------------------------
def run_python(df):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from nonparametricanovas import perform_brunner_langer_ats

    result = perform_brunner_langer_ats(
        data=df,
        dv="distance",
        between_factor="sex",
        within_factor="age",
        subject_col="subject",
        alpha=0.05,
    )
    return result


# ---------------------------------------------------------------------------
# 3.  Run R nparLD and parse output
# ---------------------------------------------------------------------------
R_SCRIPT = textwrap.dedent("""\
    if (!requireNamespace("nparLD", quietly=TRUE)) {
        install.packages("nparLD", repos="https://cloud.r-project.org", quiet=TRUE)
    }
    library(nparLD)

    # Embed data (same as Python)
    rows <- list(
      c(26.0,8,1,"Male"),c(25.0,10,1,"Male"),c(29.0,12,1,"Male"),c(31.0,14,1,"Male"),
      c(21.5,8,2,"Male"),c(22.5,10,2,"Male"),c(23.0,12,2,"Male"),c(26.5,14,2,"Male"),
      c(23.0,8,3,"Male"),c(22.5,10,3,"Male"),c(24.0,12,3,"Male"),c(27.5,14,3,"Male"),
      c(25.5,8,4,"Male"),c(27.5,10,4,"Male"),c(26.5,12,4,"Male"),c(27.0,14,4,"Male"),
      c(20.0,8,5,"Male"),c(23.5,10,5,"Male"),c(22.5,12,5,"Male"),c(26.0,14,5,"Male"),
      c(24.5,8,6,"Male"),c(25.5,10,6,"Male"),c(27.0,12,6,"Male"),c(28.5,14,6,"Male"),
      c(22.0,8,7,"Male"),c(22.0,10,7,"Male"),c(24.5,12,7,"Male"),c(26.5,14,7,"Male"),
      c(24.0,8,8,"Male"),c(21.5,10,8,"Male"),c(24.5,12,8,"Male"),c(25.5,14,8,"Male"),
      c(23.0,8,9,"Male"),c(20.5,10,9,"Male"),c(31.0,12,9,"Male"),c(26.0,14,9,"Male"),
      c(27.5,8,10,"Male"),c(28.0,10,10,"Male"),c(31.0,12,10,"Male"),c(31.5,14,10,"Male"),
      c(23.0,8,11,"Male"),c(23.0,10,11,"Male"),c(23.5,12,11,"Male"),c(25.0,14,11,"Male"),
      c(21.5,8,12,"Male"),c(23.5,10,12,"Male"),c(24.0,12,12,"Male"),c(28.0,14,12,"Male"),
      c(17.0,8,13,"Male"),c(24.5,10,13,"Male"),c(26.0,12,13,"Male"),c(29.5,14,13,"Male"),
      c(22.5,8,14,"Male"),c(25.5,10,14,"Male"),c(25.5,12,14,"Male"),c(26.0,14,14,"Male"),
      c(23.0,8,15,"Male"),c(24.5,10,15,"Male"),c(26.0,12,15,"Male"),c(30.0,14,15,"Male"),
      c(22.0,8,16,"Male"),c(21.5,10,16,"Male"),c(23.5,12,16,"Male"),c(25.0,14,16,"Male"),
      c(21.0,8,17,"Female"),c(20.0,10,17,"Female"),c(21.5,12,17,"Female"),c(23.0,14,17,"Female"),
      c(21.0,8,18,"Female"),c(21.5,10,18,"Female"),c(24.0,12,18,"Female"),c(25.5,14,18,"Female"),
      c(20.5,8,19,"Female"),c(24.0,10,19,"Female"),c(24.5,12,19,"Female"),c(26.0,14,19,"Female"),
      c(23.5,8,20,"Female"),c(24.5,10,20,"Female"),c(25.0,12,20,"Female"),c(26.5,14,20,"Female"),
      c(21.5,8,21,"Female"),c(23.0,10,21,"Female"),c(22.5,12,21,"Female"),c(23.5,14,21,"Female"),
      c(20.0,8,22,"Female"),c(21.0,10,22,"Female"),c(21.0,12,22,"Female"),c(22.5,14,22,"Female"),
      c(21.5,8,23,"Female"),c(22.5,10,23,"Female"),c(23.0,12,23,"Female"),c(25.0,14,23,"Female"),
      c(23.0,8,24,"Female"),c(23.0,10,24,"Female"),c(23.5,12,24,"Female"),c(24.0,14,24,"Female"),
      c(20.0,8,25,"Female"),c(21.0,10,25,"Female"),c(22.0,12,25,"Female"),c(21.5,14,25,"Female"),
      c(16.5,8,26,"Female"),c(19.0,10,26,"Female"),c(19.0,12,26,"Female"),c(19.5,14,26,"Female"),
      c(24.5,8,27,"Female"),c(25.0,10,27,"Female"),c(28.0,12,27,"Female"),c(28.0,14,27,"Female")
    )
    mat <- do.call(rbind, rows)
    d <- data.frame(
      resp    = as.numeric(mat[,1]),
      time    = as.numeric(mat[,2]),
      subject = as.integer(mat[,3]),
      group   = mat[,4],
      stringsAsFactors = FALSE
    )

    # nparLD uses alphabetical order for factor levels; Girl < Boy alphabetically
    # to match nparLD defaults use factor levels as-is (Female < Male alphabetically)
    ex <- f1.ld.f1(
      y=d$resp, time=d$time, group=d$group, subject=d$subject,
      group.name="sex", time.name="age",
      description=FALSE
    )

    at <- ex$ANOVA.test
    cat("=== ANOVA.test ===\\n")
    print(at)

    cat("\\n=== ANOVA.test.mod.Box ===\\n")
    print(ex$ANOVA.test.mod.Box)

    cat("\\n=== RTE ===\\n")
    print(ex$RTE)

    # Machine-readable output for comparison
    mb <- ex$ANOVA.test.mod.Box
    cat("\\nCSV_START\\n")
    # Between (sex) — use mod.Box p-value (F with finite df2), same as Python
    cat(sprintf("ATS_between,%.6f\\n", mb["sex","Statistic"]))
    cat(sprintf("df1_between,%.6f\\n", mb["sex","df1"]))
    cat(sprintf("p_between,%.6f\\n",   mb["sex","p-value"]))
    cat(sprintf("df2_between,%.6f\\n", mb["sex","df2"]))
    # Within (age) — from regular ANOVA.test (chi² / Box)
    age_row <- at["age",]
    cat(sprintf("ATS_within,%.6f\\n",  age_row[["Statistic"]]))
    cat(sprintf("df1_within,%.6f\\n",  age_row[["df"]]))
    cat(sprintf("p_within,%.6f\\n",    age_row[["p-value"]]))
    # Interaction
    int_row <- at["sex:age",]
    cat(sprintf("ATS_inter,%.6f\\n",   int_row[["Statistic"]]))
    cat(sprintf("df1_inter,%.6f\\n",   int_row[["df"]]))
    cat(sprintf("p_inter,%.6f\\n",     int_row[["p-value"]]))

    # Cell RTEs only (between × within), with plain value keys to match Python
    # R rownames: "sexMale:age8" → key "Male:8"
    rte <- ex$RTE
    for (nm in rownames(rte)) {
      if (grepl(":", nm)) {
        # cell RTE: strip factor-name prefix from each part
        parts <- strsplit(nm, ":")[[1]]
        grp <- sub("^sex", "", parts[1])
        tpt <- sub("^age", "", parts[2])
        cat(sprintf("RTE,%s:%s,%.6f\\n", grp, tpt, rte[nm,"RTE"]))
      }
    }
    cat("CSV_END\\n")
""")


def run_r():
    # Try standard name first, then common Windows install locations
    candidates = [
        "Rscript",
        os.path.expanduser(r"~\AppData\Local\Programs\R\R-4.5.0\bin\Rscript.exe"),
        os.path.expanduser(r"~\AppData\Local\Programs\R\R-4.4.0\bin\Rscript.exe"),
        os.path.expanduser(r"~\AppData\Local\Programs\R\R-4.3.0\bin\Rscript.exe"),
        r"C:\Program Files\R\R-4.5.0\bin\Rscript.exe",
        r"C:\Program Files\R\R-4.4.0\bin\Rscript.exe",
    ]
    rscript = None
    for c in candidates:
        try:
            subprocess.run([c, "--version"], capture_output=True, check=True)
            rscript = c
            break
        except (FileNotFoundError, subprocess.CalledProcessError, OSError):
            continue
    if rscript is None:
        return None, "Rscript not found"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False, encoding="utf-8") as f:
        f.write(R_SCRIPT)
        rscript_path = f.name

    try:
        proc = subprocess.run(
            [rscript, "--vanilla", rscript_path],
            capture_output=True, text=True, timeout=120
        )
    finally:
        os.unlink(rscript_path)

    if proc.returncode != 0:
        return None, f"Rscript failed:\n{proc.stderr}"

    # Print full R output for reference
    print("\n" + "=" * 60)
    print("R nparLD output (full):")
    print("=" * 60)
    print(proc.stdout)

    # Parse CSV block
    r_vals = {}
    in_csv = False
    for line in proc.stdout.splitlines():
        if line.strip() == "CSV_START":
            in_csv = True
            continue
        if line.strip() == "CSV_END":
            break
        if in_csv and "," in line:
            parts = line.strip().split(",")
            if parts[0] == "RTE":
                r_vals[f"RTE_{parts[1]}"] = float(parts[2])
            else:
                r_vals[parts[0]] = float(parts[1])

    return r_vals, None


# ---------------------------------------------------------------------------
# 4.  Compare
# ---------------------------------------------------------------------------
def compare(py_result, r_vals):
    at = py_result["anova_table"]
    sources = list(at["Source"])
    between_row = at.iloc[0]   # first row = between factor
    within_row  = at.iloc[1]   # second row = within factor
    inter_row   = at.iloc[2]   # third row = interaction

    py_vals = {
        "ATS_between": float(between_row["ATS"]),
        "df1_between": float(between_row["df1"]),
        "p_between":   float(between_row["p-value"]),
        "ATS_within":  float(within_row["ATS"]),
        "df1_within":  float(within_row["df1"]),
        "p_within":    float(within_row["p-value"]),
        "ATS_inter":   float(inter_row["ATS"]),
        "df1_inter":   float(inter_row["df1"]),
        "p_inter":     float(inter_row["p-value"]),
        "df2_between": float(between_row["df2"]) if between_row["df2"] is not None else np.nan,
    }

    # Cell RTEs from Python (between × within combinations)
    rte_df = py_result.get("RTE")
    if rte_df is not None:
        for _, row in rte_df.iterrows():
            key = f"RTE_{row['between_group']}:{row['within_level']}"
            py_vals[key] = float(row["RTE"])

    print("\n" + "=" * 60)
    print("Python vs R comparison:")
    print("=" * 60)
    print(f"{'Key':<30} {'Python':>12} {'R':>12} {'|diff|':>10} {'OK?':>6}")
    print("-" * 72)

    all_ok = True
    for key in sorted(set(py_vals) | set(r_vals)):
        pv = py_vals.get(key, float("nan"))
        rv = r_vals.get(key, float("nan"))
        diff = abs(pv - rv)
        tol = 0.005  # 2 decimal places
        ok = diff <= tol
        if not ok:
            all_ok = False
        flag = "OK" if ok else "FAIL"
        print(f"{key:<30} {pv:>12.4f} {rv:>12.4f} {diff:>10.4f} {flag:>6}")

    print()
    if all_ok:
        print("All values match to 2 decimal places.")
    else:
        print("MISMATCH detected — check FAIL rows above.")
    return all_ok


# ---------------------------------------------------------------------------
# 5.  Main
# ---------------------------------------------------------------------------
def main():
    print("Orthodont F1-LD-F1 validation: Python ATS vs R nparLD")
    print("=" * 60)

    df = make_orthodont_df()
    print(f"Dataset: {len(df)} rows, {df['subject'].nunique()} subjects, "
          f"{df['sex'].nunique()} groups, {df['age'].nunique()} time points\n")

    # Python
    print("Running Python perform_brunner_langer_ats()...")
    py_result = run_python(df)
    at = py_result["anova_table"]
    print("\nPython ANOVA table:")
    print(at[["Source", "ATS", "df1", "df2", "p-value"]].to_string(index=False))

    if py_result.get("rte_table") is not None:
        print("\nPython RTEs:")
        print(py_result["rte_table"].to_string(index=False))

    # R
    print("\nRunning R nparLD::f1.ld.f1()...")
    r_vals, err = run_r()

    if r_vals is None:
        print(f"\nR not available: {err}")
        print("\nTo install R on Windows:")
        print("  winget install R.R")
        print("  # or download from: https://cran.r-project.org/")
        print("\nAfter installing R and the nparLD package, rerun this script.")
        return

    compare(py_result, r_vals)


if __name__ == "__main__":
    main()
