args <- commandArgs(trailingOnly=TRUE)
csv_path <- args[1]
df <- read.csv(csv_path, stringsAsFactors=FALSE)
df <- df[!is.na(df$X) & !is.na(df$Y), ]
m <- lm(Y ~ X, data=df)
s <- summary(m)
# Slope row: X (intercept is row 1, X is row 2)
p_slope <- s$coefficients["X", "Pr(>|t|)"]
t_slope <- s$coefficients["X", "t value"]
r_sq    <- s$r.squared
# Output: p_value t_statistic r_squared
cat(p_slope, t_slope, r_sq, "\n")
