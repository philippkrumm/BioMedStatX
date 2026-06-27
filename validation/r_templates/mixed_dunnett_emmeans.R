# Ground-truth generator: Mixed-ANOVA Dunnett-type post-hoc.
#
# Contrast family (decided 2026-06-27): treatment-vs-control AT EACH within
# level -- within every Time, each treatment Group vs the Ctrl group, the whole
# (G-1) x W family jointly corrected with the exact multivariate-t (adjust="mvt").
#
#   emmeans(fit, ~ Group | Time, model = "univariate")
#   contrast(emm, method = "trt.vs.ctrl", adjust = "mvt")
#
# model = "univariate" forces the classical split-plot error strata (not the
# multivariate/mlm reformulation), so the degrees of freedom are reproducible in
# Python without Satterthwaite/Kenward-Roger.
#
# Usage:  Rscript mixed_dunnett_emmeans.R <dataset.csv> > references_mixed_dunnett_emmeans.json
#
# Output: JSON with one object per contrast (Time, contrast, estimate, SE, df,
# t_ratio, p_value). p_value is the mvt-adjusted simultaneous p-value.

suppressWarnings(suppressMessages({
  library(afex)
  library(emmeans)
}))

args <- commandArgs(trailingOnly = TRUE)
csv_path <- args[1]

df <- read.csv(csv_path, stringsAsFactors = FALSE)
# Ctrl must be the reference (first) level so trt.vs.ctrl uses it.
df$Group   <- factor(df$Group, levels = c("Ctrl", "TrtA", "TrtB"))
df$Time    <- factor(df$Time)
df$Subject <- factor(df$Subject)

fit <- aov_ez(id = "Subject", dv = "Value", data = df,
              between = "Group", within = "Time", type = 3,
              include_aov = TRUE)  # keep the aov strata so model="univariate" works

emm <- emmeans(fit, ~ Group | Time, model = "univariate")
ctr <- contrast(emm, method = "trt.vs.ctrl", adjust = "mvt")
cs  <- as.data.frame(summary(ctr))

# --- emit JSON manually (no jsonlite dependency) ---
jnum <- function(x) ifelse(is.na(x), "null", formatC(x, format = "e", digits = 12))
jstr <- function(x) paste0('"', gsub('"', '\\\\"', x), '"')

rows <- character(nrow(cs))
for (i in seq_len(nrow(cs))) {
  rows[i] <- paste0(
    "    {",
    '"Time": ',     jstr(as.character(cs$Time[i])),     ", ",
    '"contrast": ', jstr(as.character(cs$contrast[i])), ", ",
    '"estimate": ', jnum(cs$estimate[i]),  ", ",
    '"SE": ',       jnum(cs$SE[i]),        ", ",
    '"df": ',       jnum(cs$df[i]),        ", ",
    '"t_ratio": ',  jnum(cs$t.ratio[i]),   ", ",
    '"p_value": ',  jnum(cs$p.value[i]),
    "}"
  )
}

cat("{\n")
cat('  "dataset": ', jstr(csv_path), ",\n", sep = "")
cat('  "method": "emmeans(~ Group | Time, model=univariate); contrast(trt.vs.ctrl, adjust=mvt)",\n')
cat('  "family": "treatment-vs-control at each within level",\n')
cat('  "adjust": "mvt",\n')
cat('  "r_version": ', jstr(R.version.string), ",\n", sep = "")
cat('  "afex_version": ', jstr(as.character(packageVersion("afex"))), ",\n", sep = "")
cat('  "emmeans_version": ', jstr(as.character(packageVersion("emmeans"))), ",\n", sep = "")
cat('  "contrasts": [\n')
cat(paste(rows, collapse = ",\n"), "\n")
cat("  ]\n")
cat("}\n")
