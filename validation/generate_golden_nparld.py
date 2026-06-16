"""Generate the frozen nparLD (Brunner-Langer ATS) golden reference.

Run manually:  python validation/generate_golden_nparld.py   (requires R + nparLD)

The Brunner-Langer ANOVA-Type Statistic has no reliable Python reference, so R's
nparLD is the canonical oracle. This script builds a deterministic mixed design
(1 between, 1 within), runs nparLD via Rscript, and freezes the dataset together
with the canonical ATS / df / p per effect into tests/golden/references_nparld.json.
The gated test (tests/test_golden_nparld.py) then runs the APP's
perform_brunner_langer_ats on the SAME frozen data and compares — no R at test time.

Note: the app uses a Satterthwaite finite df2 (F-test) for the BETWEEN (whole-plot)
factor, deliberately deviating from nparLD's chi-square p there. So between-p is
frozen but flagged compare_p=False; ATS + df1 match nparLD for all effects, and p
matches for the within and interaction effects.
"""
import json
import os
import subprocess
import tempfile

import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "tests", "golden", "references_nparld.json")

R_SCRIPT = r'''
suppressMessages({library(nparLD); library(jsonlite)})
args <- commandArgs(trailingOnly=TRUE)
d <- read.csv(args[1], stringsAsFactors=TRUE)
f <- nparLD(y ~ grp*time, data=d, subject="subj", description=FALSE)
a <- f$ANOVA.test
rows <- lapply(seq_len(nrow(a)), function(i) {
  list(source=rownames(a)[i], ATS=unname(a[i,"Statistic"]),
       df1=unname(a[i,"df"]), p=unname(a[i,"p-value"]))
})
cat(toJSON(rows, auto_unbox=TRUE, digits=10))
'''


def build_dataset():
    rng = np.random.default_rng(42)
    rows = []
    for g in ("G1", "G2"):
        for s in range(10):
            sid = f"{g}_S{s}"
            base = rng.normal(0.0 if g == "G1" else 0.8, 1.0)
            for ti, lvl in enumerate(("T1", "T2", "T3")):
                rows.append({"subj": sid, "grp": g, "time": lvl,
                             "y": float(base + ti * 0.5 + rng.normal(0, 1))})
    return pd.DataFrame(rows)


def main():
    df = build_dataset()
    with tempfile.TemporaryDirectory() as tmp:
        csv = os.path.join(tmp, "bl.csv")
        df.to_csv(csv, index=False)
        rscript_path = os.path.join(tmp, "bl.R")
        with open(rscript_path, "w") as fh:
            fh.write(R_SCRIPT)
        out = subprocess.run(["Rscript", rscript_path, csv], capture_output=True, text=True, check=True)
        r_effects = json.loads(out.stdout.strip().splitlines()[-1])

    # nparLD source label -> app anova_table "Source"; compare_p only off-between.
    src_map = {"grp": "grp", "time": "time", "grp:time": "grp:time"}
    compare_p = {"grp": False, "time": True, "grp:time": True}  # between p deviates (Satterthwaite)
    effects = []
    for e in r_effects:
        src = src_map.get(e["source"], e["source"])
        effects.append({
            "source": src,
            "ATS": float(e["ATS"]),
            "df1": float(e["df1"]),
            "p": float(e["p"]),
            "compare_p": compare_p.get(src, True),
            "tol": {"ATS": 1e-3, "df1": 1e-3, "p": 1e-3},
        })

    data = {
        "schema_version": 1,
        "oracle": "R nparLD (Brunner-Langer ATS)",
        "design": {"dv": "y", "between": "grp", "within": "time", "subject": "subj"},
        "data": df.to_dict(orient="records"),
        "effects": effects,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"Wrote nparLD golden ({len(effects)} effects) -> {OUT}")


if __name__ == "__main__":
    main()
