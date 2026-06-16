"""Generate frozen golden-reference values for advanced R integration.

Run manually: python validation/generate_golden_r_advanced.py (requires R, car, afex, emmeans)

Generates:
1. Two-way Type-II ANOVA via `car`
2. RM / Mixed ANOVA via `afex`
3. ANCOVA via `emmeans`

Freezes into tests/golden/references_r_advanced.json.
"""
import json
import os
import subprocess
import tempfile

import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "tests", "golden", "references_r_advanced.json")

R_SCRIPT = r'''
suppressMessages({
  library(jsonlite)
  library(car)
  library(afex)
  library(emmeans)
})

args <- commandArgs(trailingOnly=TRUE)
d <- read.csv(args[1], stringsAsFactors=TRUE)
d$subj <- as.factor(d$subj)

results <- list()

# 1. Two-way Type-II ANOVA (car)
# Model: y ~ groupA * groupB
m_car <- lm(y_car ~ groupA * groupB, data=d)
a_car <- Anova(m_car, type=2)
# Extract F, df, p
car_eff <- list()
for(rn in rownames(a_car)) {
    if(rn != "Residuals") {
        car_eff[[rn]] <- list(
            F = unname(a_car[rn, "F value"]),
            df1 = unname(a_car[rn, "Df"]),
            df2 = unname(a_car["Residuals", "Df"]),
            p = unname(a_car[rn, "Pr(>F)"])
        )
    }
}
results$car_type2 <- car_eff

# 2. Mixed ANOVA (afex)
# Model: y ~ groupA * time + Error(subj/time)
# afex::aov_ez
a_afex <- aov_ez(id="subj", dv="y_mixed", data=d, between="groupA", within="time",
                 type=3, print.formula=FALSE)
# Extract F, df, p and Sphericity corrections
afex_table <- a_afex$anova_table
afex_eff <- list()
for(rn in rownames(afex_table)) {
    afex_eff[[rn]] <- list(
        F = unname(afex_table[rn, "F"]),
        df1 = unname(afex_table[rn, "num Df"]),
        df2 = unname(afex_table[rn, "den Df"]),
        p = unname(afex_table[rn, "Pr(>F)"])
    )
    # If Sphericity corrections are available (for within factors)
    if("MSE" %in% colnames(afex_table)) {
        # afex adds columns for GG/HF if requested, but we just want the uncorrected p for golden testing,
        # or we can test the corrections. The default is GG corrected p-values if sphericity is violated.
        # Let's extract uncorrected and GG epsilon.
    }
}
results$afex_mixed <- afex_eff

# 3. ANCOVA & emmeans
# Model: y ~ groupA + covar
m_ancova <- lm(y_ancova ~ groupA + covar, data=d)
a_ancova <- Anova(m_ancova, type=2)
# emmeans
emm <- emmeans(m_ancova, "groupA")
emm_df <- as.data.frame(emm)
pairs <- as.data.frame(pairs(emm))

ancova_eff <- list()
for(rn in rownames(a_ancova)) {
    if(rn != "Residuals") {
        ancova_eff[[rn]] <- list(
            F = unname(a_ancova[rn, "F value"]),
            df1 = unname(a_ancova[rn, "Df"]),
            df2 = unname(a_ancova["Residuals", "Df"]),
            p = unname(a_ancova[rn, "Pr(>F)"])
        )
    }
}
results$ancova_main <- ancova_eff

emm_list <- list()
for(i in 1:nrow(emm_df)) {
    emm_list[[as.character(emm_df[i, "groupA"])]] <- list(
        emmean = unname(emm_df[i, "emmean"]),
        SE = unname(emm_df[i, "SE"]),
        df = unname(emm_df[i, "df"])
    )
}
results$ancova_emmeans <- emm_list

cat(toJSON(results, auto_unbox=TRUE, digits=10))
'''


def build_dataset():
    rng = np.random.default_rng(123)
    rows = []
    for s in range(1, 21):
        groupA = "Trt" if s <= 10 else "Ctrl"
        groupB = "Low" if s % 2 == 0 else "High"
        covar = rng.normal(50, 10)
        
        # Responses for different tests to ensure signals
        y_car = 10 + (5 if groupA == "Trt" else 0) + (3 if groupB == "High" else 0) + rng.normal(0, 2)
        y_ancova = 10 + (4 if groupA == "Trt" else 0) + 0.5 * covar + rng.normal(0, 2)
        
        for ti, time in enumerate(["T1", "T2", "T3"]):
            y_mixed = 10 + (2 if groupA == "Trt" else 0) + ti * 2 + (ti * 1.5 if groupA == "Trt" else 0) + rng.normal(0, 1) + rng.normal(0, 0.5) # subj random effect
            rows.append({
                "subj": f"S{s}",
                "groupA": groupA,
                "groupB": groupB,
                "covar": float(covar),
                "time": time,
                "y_car": float(y_car),
                "y_ancova": float(y_ancova),
                "y_mixed": float(y_mixed)
            })
    return pd.DataFrame(rows)


def main():
    df = build_dataset()
    with tempfile.TemporaryDirectory() as tmp:
        csv = os.path.join(tmp, "data.csv")
        df.to_csv(csv, index=False)
        rscript_path = os.path.join(tmp, "script.R")
        with open(rscript_path, "w") as fh:
            fh.write(R_SCRIPT)
        
        try:
            out = subprocess.run(["Rscript", rscript_path, csv], capture_output=True, text=True, check=True)
            results = json.loads(out.stdout.strip().splitlines()[-1])
        except subprocess.CalledProcessError as e:
            print("Rscript failed!")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
            raise
        except json.JSONDecodeError as e:
            print("JSON decoding failed. R output was:")
            print(out.stdout)
            raise

    data = {
        "schema_version": 1,
        "oracle": "R (car, afex, emmeans)",
        "data": df.to_dict(orient="records"),
        "results": results
    }
    
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"Wrote advanced R golden -> {OUT}")


if __name__ == "__main__":
    main()
