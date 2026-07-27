# Tukey HSD -- Base R stats::TukeyHSD on a one-way aov.
# This is the canonical implementation that statsmodels.pairwise_tukeyhsd (the
# app's Tukey path) was built to reproduce: pooled-variance studentized-range
# HSD with simultaneous confidence intervals. No extra packages.
#
# Input : long CSV with columns Group, Value.
# Output: one line per pair -- "<g2>-<g1> <diff> <p_adj>"
#         (diff = mean(g2) - mean(g1); the app compares the unordered pair, so
#          sign is matched on |diff|).
args <- commandArgs(trailingOnly=TRUE)
df <- read.csv(args[1], stringsAsFactors=FALSE)
df$Group <- factor(df$Group)
th <- TukeyHSD(aov(Value ~ Group, data=df))$Group
for (rn in rownames(th)) cat(rn, th[rn, "diff"], th[rn, "p adj"], "\n")
