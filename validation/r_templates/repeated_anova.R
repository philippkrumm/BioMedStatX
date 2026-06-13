# Repeated Measures ANOVA via aov() with Error term
args <- commandArgs(trailingOnly=TRUE)
csv_path <- args[1]
df <- read.csv(csv_path, stringsAsFactors=FALSE)
df$Subject <- as.factor(df$Subject)
df$Group   <- as.factor(df$Group)
m <- aov(Value ~ Group + Error(Subject/Group), data=df)
s <- summary(m)
# Extract F and p from the within-subject stratum
within <- s[[paste0("Error: Subject:Group")]]
if (is.null(within)) {
  within <- s[[length(s)]]
}
tbl <- within[[1]]
p_val <- tbl[["Pr(>F)"]][1]
f_val <- tbl[["F value"]][1]
cat(p_val, f_val, "\n")
