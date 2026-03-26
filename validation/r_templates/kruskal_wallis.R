args <- commandArgs(trailingOnly=TRUE)
csv_path <- args[1]
df <- read.csv(csv_path, stringsAsFactors=FALSE)
df <- df[!is.na(df$Value), ]
df$Group <- as.factor(df$Group)
res <- kruskal.test(Value ~ Group, data=df)
cat(res$p.value, res$statistic, "\n")
