args <- commandArgs(trailingOnly=TRUE)
csv_path <- args[1]
df <- read.csv(csv_path, stringsAsFactors=FALSE)
df <- df[!is.na(df$Value), ]
df$FactorA <- as.factor(df$FactorA)
df$FactorB <- as.factor(df$FactorB)
m <- aov(Value ~ FactorA * FactorB, data=df)
s <- summary(m)[[1]]
# Output: p_FactorA p_FactorB p_Interaction F_FactorA F_FactorB F_Interaction
cat(s[["Pr(>F)"]][1], s[["Pr(>F)"]][2], s[["Pr(>F)"]][3],
    s[["F value"]][1], s[["F value"]][2], s[["F value"]][3], "\n")
