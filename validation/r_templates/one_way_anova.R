args <- commandArgs(trailingOnly=TRUE)
csv_path <- args[1]
df <- read.csv(csv_path, stringsAsFactors=FALSE)
df <- df[!is.na(df$Value), ]
df$Group <- as.factor(df$Group)
m <- aov(Value ~ Group, data=df)
s <- summary(m)[[1]]
p_val <- s[["Pr(>F)"]][1]
f_val <- s[["F value"]][1]
# eta-squared = SS_between / SS_total; Cohen's f = sqrt(eta2 / (1 - eta2))
ss     <- s[["Sum Sq"]]
eta_sq <- ss[1] / sum(ss)
cohens_f <- sqrt(eta_sq / (1 - eta_sq))
# Tukey HSD post-hoc — 3 pairs for 3-group fixture (lexicographic order)
tukey   <- TukeyHSD(m)$Group
p_tukey <- tukey[, "p adj"]
cat(p_val, f_val, eta_sq, cohens_f, p_tukey[1], p_tukey[2], p_tukey[3], "\n")
