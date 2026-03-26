args <- commandArgs(trailingOnly=TRUE)
csv_path <- args[1]
df <- read.csv(csv_path, stringsAsFactors=FALSE)
df <- df[!is.na(df$Value), ]
df$Group <- as.factor(df$Group)
m <- aov(Value ~ Group, data=df)
s <- summary(m)[[1]]
p_val <- s[["Pr(>F)"]][1]
f_val <- s[["F value"]][1]
cat(p_val, f_val, "\n")
