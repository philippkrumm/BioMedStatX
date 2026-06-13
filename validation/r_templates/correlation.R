args <- commandArgs(trailingOnly=TRUE)
csv_path <- args[1]
# Optional 2nd arg: correlation method ("pearson" or "spearman")
method <- if (length(args) >= 2) args[2] else "pearson"
df <- read.csv(csv_path, stringsAsFactors=FALSE)
df <- df[!is.na(df$X) & !is.na(df$Y), ]
res <- cor.test(df$X, df$Y, method=method)
# Output: p_value r_statistic
cat(res$p.value, as.numeric(res$estimate), "\n")
