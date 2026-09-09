"""Golden references for Feature-B's HETEROGENEOUS-variance branches.

Run manually: python validation/generate_golden_feature_b_het.py   (requires R + afex + emmeans)

Why a second dataset: the canonical seed-123 set never triggers Welch. All three of
its between contrasts are homogeneous (Levene p = 0.63 / 0.78 / 0.36), so
_indep_pair's Welch/Satterthwaite branch — including the hand-written Satterthwaite
df — has never been exercised. Its within pairs are also borderline (p ~= 0.035),
which is a fragile thing to pin numerically.

Seed 456. Two purpose-built outcomes on the same groupA(Ctrl/Trt) x time(T1..T3)
x subj skeleton:

  y_bet_het   PRIMARY. Raw-value variance deliberately unequal between groups
              (Ctrl sd=1, Trt sd=5 -> variance ratio 25) so Levene on the raw
              values is decisively < 0.05 at every timepoint and the app must
              route to Welch. NOTE: drawn independently per subject-time, i.e. it
              carries no repeated-measures structure — it is only ever used for
              the BETWEEN contrasts.

  y_win_mix   SECONDARY. Built from per-subject increments so the DIFFERENCE
              variances are controlled directly:
                T2 = T1 + d12,  d12 ~ N(1.5, sd=4.0) for Trt vs sd=0.8 for Ctrl
                     -> T1-T2 difference variances decisively heterogeneous
                T3 = T2 + d23,  d23 ~ N(1.0, sd=1.0) for both groups
                     -> T2-T3 difference variances decisively homogeneous
              Gives one unambiguous isolated pair and one unambiguous pooled pair,
              instead of seed-123's borderline p ~= 0.035.

Freezes into tests/golden/references_feature_b_het.json.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_golden_r_advanced import _find_rscript, _parse_json_from_r_output

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "tests", "golden", "references_feature_b_het.json")

TIMES = ["T1", "T2", "T3"]


def build_het_dataset():
    rng = np.random.default_rng(456)
    rows = []
    for s in range(1, 21):
        groupA = "Trt" if s <= 10 else "Ctrl"
        subj_re = rng.normal(0, 0.5)
        base = 10.0 + (2.0 if groupA == "Trt" else 0.0)

        # Ratio deliberately extreme (variance ratio ~256): at sd 5 vs 1 the realized
        # sample sds still landed at Levene p=0.08 for one timepoint (Student, not
        # Welch) — too weak to pin the Welch branch reliably.
        sd_bet = 8.0 if groupA == "Trt" else 0.5     # between-group raw heterogeneity
        sd_d12 = 4.0 if groupA == "Trt" else 0.8     # T1-T2 diff heterogeneity
        sd_d23 = 1.0                                  # T2-T3 diff homogeneity

        t1 = base + subj_re + rng.normal(0, 1.0)
        d12 = rng.normal(1.5, sd_d12)
        d23 = rng.normal(1.0, sd_d23)
        y_win = {"T1": t1, "T2": t1 + d12, "T3": t1 + d12 + d23}

        for time in TIMES:
            rows.append({
                "subj": f"S{s}", "groupA": groupA, "time": time,
                "y_bet_het": float(base + rng.normal(0, sd_bet)),
                "y_win_mix": float(y_win[time]),
            })
    return pd.DataFrame(rows)


R_SCRIPT = r'''
options(OutDec=".", scipen=999)
suppressMessages({ library(jsonlite); library(afex); library(emmeans) })
args <- commandArgs(trailingOnly=TRUE)
d <- read.csv(args[1], stringsAsFactors=TRUE)
d$subj <- as.factor(d$subj)
d$time <- factor(d$time, levels=c("T1","T2","T3"))
d$groupA <- factor(d$groupA, levels=c("Ctrl","Trt"))
WPAIRS <- list(c("T1","T2"), c("T1","T3"), c("T2","T3"))

# ---- PRIMARY: between contrasts on y_bet_het, Welch AND Student ----
welch <- list(); student <- list()
for (w in c("T1","T2","T3")) {
  a <- d$y_bet_het[d$groupA=="Ctrl" & d$time==w]
  b <- d$y_bet_het[d$groupA=="Trt"  & d$time==w]
  tw <- t.test(a, b, var.equal=FALSE)   # Welch-Satterthwaite
  ts <- t.test(a, b, var.equal=TRUE)    # Student, pooled variance
  key <- paste0("Ctrl:",w,"|Trt:",w)
  welch[[key]]   <- list(t=unname(tw$statistic), df=unname(tw$parameter), p=unname(tw$p.value))
  student[[key]] <- list(t=unname(ts$statistic), df=unname(ts$parameter), p=unname(ts$p.value))
}

# ---- SECONDARY: within contrasts on y_win_mix, isolated AND emmeans-pooled ----
iso_within <- list()
for (g in c("Ctrl","Trt")) {
  sub <- d[d$groupA==g,]
  for (pr in WPAIRS) {
    w1 <- pr[1]; w2 <- pr[2]
    a <- sub[sub$time==w1, c("subj","y_win_mix")]
    b <- sub[sub$time==w2, c("subj","y_win_mix")]
    m <- merge(a, b, by="subj")
    tt <- t.test(m$y_win_mix.x, m$y_win_mix.y, paired=TRUE)
    key <- paste0(g,":",w1,"|",g,":",w2)
    iso_within[[key]] <- list(t=unname(tt$statistic), df=unname(tt$parameter),
                              p=unname(tt$p.value), n=nrow(m))
  }
}
m_afex <- aov_ez(id="subj", dv="y_win_mix", data=d, between="groupA", within="time",
                 type=3, print.formula=FALSE)
emm_w <- as.data.frame(pairs(emmeans(m_afex, ~ time | groupA)))
emm_within <- list()
for (i in 1:nrow(emm_w)) {
  key <- paste0(as.character(emm_w[i,"groupA"]), ":", gsub(" ", "", as.character(emm_w[i,"contrast"])))
  emm_within[[key]] <- list(t_ratio=unname(emm_w[i,"t.ratio"]), df=unname(emm_w[i,"df"]),
                            p=unname(emm_w[i,"p.value"]))
}

out <- list(
  assumptions = list(
    welch = "base-R t.test(var.equal=FALSE): Welch-Satterthwaite df. Matches scipy ttest_ind(equal_var=False).",
    student = "base-R t.test(var.equal=TRUE): pooled variance, df=n1+n2-2.",
    iso_within = "base-R t.test(paired=TRUE) per group, df=n-1 (isolated error term).",
    emmeans_pooled_within = "afex::aov_ez(type=3) + pairs(emmeans(~time|groupA)): pooled residual error."
  ),
  between_y_bet_het = list(welch=welch, student=student),
  within_y_win_mix = list(isolated=iso_within, emmeans_pooled=emm_within)
)
cat(toJSON(out, auto_unbox=TRUE, digits=12))
'''


def main():
    df = build_het_dataset()
    rscript = _find_rscript()
    with tempfile.TemporaryDirectory() as tmp:
        csv = os.path.join(tmp, "data.csv")
        df.to_csv(csv, index=False)
        rs = os.path.join(tmp, "script.R")
        with open(rs, "w") as fh:
            fh.write(R_SCRIPT)
        try:
            res = subprocess.run([rscript, rs, csv], capture_output=True, text=True, check=True)
            results = _parse_json_from_r_output(res.stdout)
        except subprocess.CalledProcessError as e:
            print("Rscript failed!\nSTDOUT:", e.stdout, "\nSTDERR:", e.stderr)
            raise

    data = {
        "schema_version": 1,
        "oracle": "R base t.test (Welch + Student + paired) & afex/emmeans (pooled within)",
        "dataset": {
            "generator": "validation/generate_golden_feature_b_het.py build_het_dataset() seed 456",
            "y_bet_het": "between-heterogeneous raw values (Ctrl sd=1, Trt sd=5); BETWEEN contrasts only",
            "y_win_mix": "T1-T2 difference variances heterogeneous (sd 0.8 vs 4.0); T2-T3 homogeneous (sd 1.0)",
        },
        "data": df.to_dict(orient="records"),
        "results": results,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(data, fh, indent=2)
    print("Wrote heterogeneous Feature-B golden ->", OUT)


if __name__ == "__main__":
    main()
