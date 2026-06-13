args <- commandArgs(trailingOnly=TRUE)
csv_path <- args[1]
df <- read.csv(csv_path, stringsAsFactors=FALSE)
grps <- unique(df$Group)
df_wide <- reshape(df, idvar="Subject", timevar="Group", direction="wide")
before <- df_wide[, paste0("Value.", grps[1])]
after  <- df_wide[, paste0("Value.", grps[2])]
res <- wilcox.test(before, after, paired=TRUE, exact=FALSE)
cat(res$p.value, res$statistic, "\n")
