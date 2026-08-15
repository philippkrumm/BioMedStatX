args <- commandArgs(trailingOnly=TRUE)
csv_path <- args[1]
df <- read.csv(csv_path, stringsAsFactors=FALSE)
df <- df[!is.na(df$Value), ]
g1 <- df$Value[df$Group == unique(df$Group)[1]]
g2 <- df$Value[df$Group == unique(df$Group)[2]]
# Welch t-test — matches the app's unconditional Welch default for two groups
# (decision_logic.select_comparison_test always returns welch_ttest, never student).
# var.equal=TRUE would only agree with Python on p/statistic by coincidence when
# the synthetic groups happen to have near-equal variance; Welch makes it a real check.
res <- t.test(g1, g2, var.equal=FALSE)
# Effect size via effsize::cohen.d(hedges.correction=TRUE) — an INDEPENDENT
# implementation of pooled-SD Hedges' g (NOT a hand-copy of Python's J factor),
# matching the app's Welch-branch effect size (pingouin 'hedges' convention).
# Requires the CRAN 'effsize' package; if absent, R exits non-zero and the
# validator skips the cross-check gracefully.
suppressMessages(library(effsize))
d <- cohen.d(g1, g2, hedges.correction=TRUE)$estimate
cat(res$p.value, res$statistic, d, "\n")
