"""Generate frozen golden-reference values for advanced R integration.

Run manually: python validation/generate_golden_r_advanced.py (requires R, car, afex, emmeans)

Generates:
1. Two-way Type-III ANOVA via `car` (sum contrasts, matches Python anova_lm typ=3)
2. RM / Mixed ANOVA via `afex`
3. ANCOVA via `emmeans` (Type-III SS, matches Python anova_lm typ=3)

Freezes into tests/golden/references_r_advanced.json.
"""
import json
import os
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "tests", "golden", "references_r_advanced.json")

R_SCRIPT = r'''
options(OutDec=".", scipen=999)
suppressMessages({
  library(jsonlite)
  library(car)
  library(afex)
  library(emmeans)
  library(lme4)
  library(logistf)
})

args <- commandArgs(trailingOnly=TRUE)
d <- read.csv(args[1], stringsAsFactors=TRUE)
d$subj <- as.factor(d$subj)

# Sum contrasts to match Python's C(factor, Sum) / anova_lm(typ=3)
options(contrasts=c("contr.sum", "contr.poly"))

results <- list()

# 1. Two-way Type-III ANOVA (car, sum contrasts)
# Model: y ~ groupA * groupB
m_car <- lm(y_car ~ groupA * groupB, data=d)
a_car <- Anova(m_car, type=3)
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
results$car_type2 <- car_eff  # key kept as car_type2 for backward compat with test references

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

# 3. ANCOVA & emmeans (Type-III SS, sum contrasts set globally above)
# Model: y ~ groupA + covar
m_ancova <- lm(y_ancova ~ groupA + covar, data=d)
a_ancova <- Anova(m_ancova, type=3)
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

# 4. LMM (lme4)
# Model: y_mixed ~ groupA * time + covar + (1|subj)
# REML=TRUE is default for lmer
m_lmer <- lmer(y_mixed ~ groupA * time + covar + (1|subj), data=d, REML=TRUE)
lmer_sum <- summary(m_lmer)
lmer_coefs <- lmer_sum$coefficients
lmer_eff <- list()
for(rn in rownames(lmer_coefs)) {
    lmer_eff[[rn]] <- list(
        Estimate = unname(lmer_coefs[rn, "Estimate"]),
        SE = unname(lmer_coefs[rn, "Std. Error"])
    )
}
results$lme4_lmm <- lmer_eff

# Create a dataset without repeated measures for Logistic Regression
d_subj <- d[!duplicated(d$subj), ]

# 5. Logistic Regression (Standard)
# Model: y_logit_std ~ groupA + covar
m_glm <- glm(y_logit_std ~ groupA + covar, data=d_subj, family=binomial)
glm_sum <- summary(m_glm)
glm_coefs <- glm_sum$coefficients
glm_eff <- list()
for(rn in rownames(glm_coefs)) {
    glm_eff[[rn]] <- list(
        Estimate = unname(glm_coefs[rn, "Estimate"]),
        SE = unname(glm_coefs[rn, "Std. Error"]),
        p = unname(glm_coefs[rn, "Pr(>|z|)"])
    )
}
results$glm_logistic <- glm_eff

# 6. Logistic Regression (Firth)
# Model: y_logit_sep ~ groupA + covar
m_logistf <- logistf(y_logit_sep ~ groupA + covar, data=d_subj)
logistf_eff <- list()
for(rn in names(m_logistf$coefficients)) {
    logistf_eff[[rn]] <- list(
        Estimate = unname(m_logistf$coefficients[rn]),
        SE = unname(sqrt(diag(m_logistf$var))[rn]),
        p = unname(m_logistf$prob[rn])
    )
}
results$logistf_firth <- logistf_eff

cat(toJSON(results, auto_unbox=TRUE, digits=10))
'''


def build_dataset():
    rng = np.random.default_rng(123)
    rows = []
    for s in range(1, 21):
        groupA = "Trt" if s <= 10 else "Ctrl"
        groupB = "Low" if s % 2 == 0 else "High"
        covar = rng.normal(50, 10)
        
        y_car = 10 + (5 if groupA == "Trt" else 0) + (3 if groupB == "High" else 0) + rng.normal(0, 2)
        y_ancova = 10 + (4 if groupA == "Trt" else 0) + 0.5 * covar + rng.normal(0, 2)
        y_logit_std = 1 if (covar > 50 and rng.random() > 0.3) else 0
        y_logit_sep = 1 if groupA == "Trt" else 0  # Perfect separation for Firth
        
        for ti, time in enumerate(["T1", "T2", "T3"]):
            y_mixed = 10 + (2 if groupA == "Trt" else 0) + ti * 2 + (ti * 1.5 if groupA == "Trt" else 0) + 0.5 * covar + rng.normal(0, 1) + rng.normal(0, 0.5) # subj random effect
            rows.append({
                "subj": f"S{s}",
                "groupA": groupA,
                "groupB": groupB,
                "covar": float(covar),
                "time": time,
                "y_car": float(y_car),
                "y_ancova": float(y_ancova),
                "y_mixed": float(y_mixed),
                "y_logit_std": int(y_logit_std),
                "y_logit_sep": int(y_logit_sep)
            })
    return pd.DataFrame(rows)


def _find_rscript():
    rscript = shutil.which("Rscript")
    if rscript:
        return rscript
    for candidate in [
        r"C:\Program Files\R\R-4.4.1\bin\Rscript.exe",
        r"C:\Program Files\R\R-4.4.0\bin\Rscript.exe",
        r"C:\Program Files\R\R-4.3.3\bin\Rscript.exe",
        "/usr/local/bin/Rscript",
        "/usr/bin/Rscript",
    ]:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError("Rscript not found on PATH. Install R or add R/bin to PATH.")


def _parse_json_from_r_output(stdout: str) -> dict:
    """Extract JSON from R output, skipping any warning/message lines before or after."""
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") or line.startswith("["):
            return json.loads(line)
    raise ValueError(f"No JSON found in R output:\n{stdout}")


def main():
    df = build_dataset()
    rscript = _find_rscript()
    with tempfile.TemporaryDirectory() as tmp:
        csv = os.path.join(tmp, "data.csv")
        df.to_csv(csv, index=False)
        rscript_path = os.path.join(tmp, "script.R")
        with open(rscript_path, "w") as fh:
            fh.write(R_SCRIPT)

        try:
            out = subprocess.run([rscript, rscript_path, csv], capture_output=True, text=True, check=True)
            results = _parse_json_from_r_output(out.stdout)
        except subprocess.CalledProcessError as e:
            print("Rscript failed!")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
            raise
        except (json.JSONDecodeError, ValueError) as e:
            print(f"JSON decoding failed: {e}\nR output was:")
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
