args <- commandArgs(trailingOnly=TRUE)
csv_path <- args[1]
df <- read.csv(csv_path, stringsAsFactors=FALSE)
df <- df[!is.na(df$Value), ]
g1 <- df$Value[df$Group == unique(df$Group)[1]]
g2 <- df$Value[df$Group == unique(df$Group)[2]]
res <- t.test(g1, g2, var.equal=TRUE)
# Cohen's d with pooled SD (matches Python: statisticaltester.py pooled formula)
s_pooled <- sqrt(((length(g1)-1)*var(g1) + (length(g2)-1)*var(g2)) /
                  (length(g1)+length(g2)-2))
d <- (mean(g1) - mean(g2)) / s_pooled
cat(res$p.value, res$statistic, d, "\n")
