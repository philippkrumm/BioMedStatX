# Friedman test — pivot long→wide then call friedman.test
args <- commandArgs(trailingOnly=TRUE)
csv_path <- args[1]
df <- read.csv(csv_path, stringsAsFactors=FALSE)
df$Subject <- as.factor(df$Subject)
df$Group   <- as.factor(df$Group)
df_wide <- reshape(df[, c("Subject","Group","Value")],
                   idvar="Subject", timevar="Group", direction="wide")
m <- as.matrix(df_wide[, -1])
res <- friedman.test(m)
cat(res$p.value, res$statistic, "\n")
