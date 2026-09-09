args <- commandArgs(trailingOnly=TRUE)
csv_path <- args[1]
# Optional 2nd arg: correlation method ("pearson" or "spearman")
method <- if (length(args) >= 2) args[2] else "pearson"
df <- read.csv(csv_path, stringsAsFactors=FALSE)
df <- df[!is.na(df$X) & !is.na(df$Y), ]
# exact=FALSE forces R's large-sample t-approximation for Spearman
# (p = pt(r/sqrt((1-r^2)/(n-2)), n-2)), which is exactly what the app computes
# via scipy.spearmanr / its explicit t-recompute. R's default (exact permutation
# for small n without ties) would diverge from the app -- the golden must test
# the app's actual method, not a different one. Ignored for Pearson (already t).
res <- cor.test(df$X, df$Y, method=method, exact=FALSE)
# Output: p_value r_statistic
cat(res$p.value, as.numeric(res$estimate), "\n")
