# Games-Howell -- PMCMRplus::gamesHowellTest.
# Robust post-hoc for unequal variances / unequal n: sqrt(2)*|t| against the
# studentized-range distribution with Welch-Satterthwaite df. This is exactly
# the classic formula the app hand-rolls in posthoc_core.GamesHowellTest.
#
# NOTE the app uses k = number of groups with n>=2 for the studentized-range
# nmeans, while this test uses the total group count. They agree when every
# group has n>=2 -- the golden datasets all satisfy that; the n<2 exclusion path
# is checked by a separate reachability guard test, not a numeric golden case.
#
# Input : long CSV with columns Group, Value.
# Output: one line per pair -- "<g1> <g2> <p_value>" (lower-triangle of the
#         p-value matrix, each unordered pair once).
args <- commandArgs(trailingOnly=TRUE)
df <- read.csv(args[1], stringsAsFactors=FALSE)
df$Group <- factor(df$Group)
suppressMessages(library(PMCMRplus))
pm <- gamesHowellTest(Value ~ Group, data=df)$p.value
for (i in rownames(pm)) for (j in colnames(pm))
    if (!is.na(pm[i, j])) cat(i, j, pm[i, j], "\n")
