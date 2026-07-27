# Dunn's test (RAW, unadjusted) -- PMCMRplus::kwAllPairsDunnTest, p.adjust="none".
# Validates only the rank-based, tie-corrected z-statistic / raw p-value -- the
# complex hand-written layer the app takes from scikit_posthocs.posthoc_dunn
# (p_adjust=None). The multiplicity correction is deliberately NOT taken from R:
# the app applies Holm-Sidak via statsmodels.multipletests, and R packages do not
# offer a bit-identical Holm-Sidak, so validating it here would introduce a
# second-oracle assumption. Holm-Sidak is checked separately as a statsmodels
# unit, and the seam between the two (raw p -> adjusted p lands on the right pair)
# is checked by a dedicated wiring test.
#
# Input : long CSV with columns Group, Value.
# Output: one line per pair -- "<g1> <g2> <raw_p>" (lower-triangle, each pair once).
args <- commandArgs(trailingOnly=TRUE)
df <- read.csv(args[1], stringsAsFactors=FALSE)
df$Group <- factor(df$Group)
suppressMessages(library(PMCMRplus))
pm <- kwAllPairsDunnTest(Value ~ Group, data=df, p.adjust.method="none")$p.value
for (i in rownames(pm)) for (j in colnames(pm))
    if (!is.na(pm[i, j])) cat(i, j, pm[i, j], "\n")
