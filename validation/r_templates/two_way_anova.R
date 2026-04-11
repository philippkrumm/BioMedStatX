args <- commandArgs(trailingOnly=TRUE)
csv_path <- args[1]
df <- read.csv(csv_path, stringsAsFactors=FALSE)
df <- df[!is.na(df$Value), ]
df$FactorA <- as.factor(df$FactorA)
df$FactorB <- as.factor(df$FactorB)
m <- aov(Value ~ FactorA * FactorB, data=df)
s <- summary(m)[[1]]
# Partial eta-squared per effect: SS_effect / (SS_effect + SS_residual)
ss     <- s[["Sum Sq"]]
ss_res <- ss[4]
peta   <- ss[1:3] / (ss[1:3] + ss_res)
# Output: p_A p_B p_AB F_A F_B F_AB peta_A peta_B peta_AB
cat(s[["Pr(>F)"]][1], s[["Pr(>F)"]][2], s[["Pr(>F)"]][3],
    s[["F value"]][1], s[["F value"]][2], s[["F value"]][3],
    peta[1], peta[2], peta[3], "\n")
