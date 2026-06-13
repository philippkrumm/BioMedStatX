args <- commandArgs(trailingOnly=TRUE)
csv_path <- args[1]
df <- read.csv(csv_path, stringsAsFactors=FALSE)
df <- df[!is.na(df$Value), ]
g1 <- df$Value[df$Group == unique(df$Group)[1]]
g2 <- df$Value[df$Group == unique(df$Group)[2]]
res <- wilcox.test(g1, g2, exact=FALSE)
cat(res$p.value, res$statistic, "\n")
