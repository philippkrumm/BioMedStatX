"""Generate frozen golden-reference values for the Feature-B Mixed/RM default
post-hoc (effect-driven simple main effects in src/analysis/mixed_simple_effects.py).

Run manually: python validation/generate_golden_feature_b.py   (requires R + afex + emmeans)

Reuses the canonical seed-123 dataset (validation/generate_golden_r_advanced.build_dataset)
so this golden is consistent with references_r_advanced.json. Adds a second outcome
`y_mixed_noia` = y_mixed WITHOUT the groupA x time interaction term. That term is
`(ti * 1.5 if groupA == "Trt")`, so y_mixed_noia is derived ARITHMETICALLY from the
existing column (y_mixed_noia = y_mixed - 1.5*ti for Trt) — this does NOT draw from the
RNG and therefore does NOT shift the stream / does NOT alter references_r_advanced.json.

TWO reference conventions are emitted, because the app and emmeans make DIFFERENT
error-term assumptions (documented in the JSON under "assumptions"):

  * "isolated"        base-R t.test per pair  (within: paired, df=n-1 ;
                      between: var.equal=TRUE, df=n1+n2-2). This MATCHES the app
                      (scipy ttest_rel / ttest_ind(equal_var=True)).
  * "emmeans_pooled"  afex::aov_ez mixed model + emmeans simple effects with a
                      POOLED residual error term and multivariate/Satterthwaite df.
                      Textbook alternative; does NOT match the app's isolated tests.

Freezes into tests/golden/references_feature_b.json.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

# reuse the exact canonical seed-123 dataset builder + rscript finder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_golden_r_advanced import build_dataset, _find_rscript, _parse_json_from_r_output

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "tests", "golden", "references_feature_b.json")

_TI = {"T1": 0, "T2": 1, "T3": 2}


def build_feature_b_dataset():
    """Canonical seed-123 df + y_mixed_noia (interaction term removed, stream-safe)."""
    df = build_dataset()
    df["y_mixed_noia"] = [
        row.y_mixed - (1.5 * _TI[row.time] if row.groupA == "Trt" else 0.0)
        for row in df.itertuples()
    ]
    return df


R_SCRIPT = r'''
options(OutDec=".", scipen=999)
suppressMessages({
  library(jsonlite)
  library(afex)
  library(emmeans)
})

args <- commandArgs(trailingOnly=TRUE)
d <- read.csv(args[1], stringsAsFactors=TRUE)
d$subj <- as.factor(d$subj)
d$time <- factor(d$time, levels=c("T1","T2","T3"))
d$groupA <- factor(d$groupA, levels=c("Ctrl","Trt"))

WPAIRS <- list(c("T1","T2"), c("T1","T3"), c("T2","T3"))

# ---- 1) ISOLATED per-pair t-tests on y_mixed (matches the app) ----
iso_within <- list()
for (g in c("Ctrl","Trt")) {
  sub <- d[d$groupA==g,]
  for (pr in WPAIRS) {
    w1 <- pr[1]; w2 <- pr[2]
    a <- sub[sub$time==w1, c("subj","y_mixed")]
    b <- sub[sub$time==w2, c("subj","y_mixed")]
    m <- merge(a, b, by="subj")          # align on subject
    tt <- t.test(m$y_mixed.x, m$y_mixed.y, paired=TRUE)
    key <- paste0(g,":",w1,"|",g,":",w2)
    iso_within[[key]] <- list(t=unname(tt$statistic), df=unname(tt$parameter),
                              p=unname(tt$p.value), n=nrow(m))
  }
}
iso_between <- list()
for (w in c("T1","T2","T3")) {
  a <- d$y_mixed[d$groupA=="Ctrl" & d$time==w]
  b <- d$y_mixed[d$groupA=="Trt"  & d$time==w]
  tt <- t.test(a, b, var.equal=TRUE)     # pooled-variance two-sample = scipy ttest_ind(equal_var=TRUE)
  key <- paste0("Ctrl:",w,"|Trt:",w)
  iso_between[[key]] <- list(t=unname(tt$statistic), df=unname(tt$parameter),
                             p=unname(tt$p.value), n1=length(a), n2=length(b))
}

# ---- 2) EMMEANS POOLED simple effects on y_mixed (documented alternative) ----
m_afex <- aov_ez(id="subj", dv="y_mixed", data=d, between="groupA", within="time",
                 type=3, print.formula=FALSE)
emm_w <- as.data.frame(pairs(emmeans(m_afex, ~ time | groupA)))
emm_within <- list()
for (i in 1:nrow(emm_w)) {
  key <- paste0(as.character(emm_w[i,"groupA"]), ":", gsub(" ", "", as.character(emm_w[i,"contrast"])))
  emm_within[[key]] <- list(t_ratio=unname(emm_w[i,"t.ratio"]), df=unname(emm_w[i,"df"]),
                            p=unname(emm_w[i,"p.value"]), estimate=unname(emm_w[i,"estimate"]))
}
emm_b <- as.data.frame(pairs(emmeans(m_afex, ~ groupA | time)))
emm_between <- list()
for (i in 1:nrow(emm_b)) {
  key <- paste0(as.character(emm_b[i,"time"]), ":", gsub(" ", "", as.character(emm_b[i,"contrast"])))
  emm_between[[key]] <- list(t_ratio=unname(emm_b[i,"t.ratio"]), df=unname(emm_b[i,"df"]),
                             p=unname(emm_b[i,"p.value"]), estimate=unname(emm_b[i,"estimate"]))
}

# ---- 3) RM within-only (all subjects, ignore groupA): paired t on y_mixed ----
rm_within <- list()
for (pr in WPAIRS) {
  w1 <- pr[1]; w2 <- pr[2]
  a <- d[d$time==w1, c("subj","y_mixed")]
  b <- d[d$time==w2, c("subj","y_mixed")]
  m <- merge(a, b, by="subj")
  tt <- t.test(m$y_mixed.x, m$y_mixed.y, paired=TRUE)
  key <- paste0(w1,"|",w2)
  rm_within[[key]] <- list(t=unname(tt$statistic), df=unname(tt$parameter),
                           p=unname(tt$p.value), n=nrow(m))
}

out <- list(
  assumptions = list(
    isolated = paste0("base-R t.test per pair on y_mixed. within: t.test(paired=TRUE), df=n-1; ",
                      "between: t.test(var.equal=TRUE) pooled variance, df=n1+n2-2. Isolated/separate ",
                      "error per pair. MATCHES the app (scipy.stats ttest_rel / ttest_ind(equal_var=True))."),
    emmeans_pooled = paste0("afex::aov_ez(type=3) mixed model on y_mixed; emmeans simple effects via ",
                      "pairs(emmeans(~ time|groupA)) and pairs(emmeans(~ groupA|time)). POOLED residual ",
                      "error term, multivariate/Satterthwaite df. Textbook alternative; does NOT match ",
                      "the app's isolated per-pair tests (different SE/df/p)."),
    rm_within = "base-R t.test(paired=TRUE) on y_mixed collapsed across groupA (all subjects)."
  ),
  isolated = list(within=iso_within, between=iso_between),
  emmeans_pooled = list(within=emm_within, between=emm_between),
  rm_within = rm_within
)
cat(toJSON(out, auto_unbox=TRUE, digits=12))
'''


def main():
    df = build_feature_b_dataset()
    rscript = _find_rscript()
    with tempfile.TemporaryDirectory() as tmp:
        csv = os.path.join(tmp, "data.csv")
        df.to_csv(csv, index=False)
        rscript_path = os.path.join(tmp, "script.R")
        with open(rscript_path, "w") as fh:
            fh.write(R_SCRIPT)
        try:
            res = subprocess.run([rscript, rscript_path, csv], capture_output=True, text=True, check=True)
            results = _parse_json_from_r_output(res.stdout)
        except subprocess.CalledProcessError as e:
            print("Rscript failed!\nSTDOUT:", e.stdout, "\nSTDERR:", e.stderr)
            raise

    data = {
        "schema_version": 1,
        "oracle": "R base t.test (isolated per-pair, app-matching) + afex/emmeans (pooled, alternative)",
        "dataset": {
            "generator": "validation/generate_golden_r_advanced.build_dataset() seed 123 + derived y_mixed_noia",
            "dv": "y_mixed", "between": "groupA (Ctrl/Trt)", "within": "time (T1/T2/T3)", "subject": "subj",
            "note": "y_mixed has a groupA x time interaction; y_mixed_noia is the same outcome with the "
                    "interaction term removed (for the interaction-n.s. gating branch).",
        },
        "data": df.to_dict(orient="records"),
        "results": results,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(data, fh, indent=2)
    print("Wrote Feature-B golden ->", OUT)


if __name__ == "__main__":
    main()
