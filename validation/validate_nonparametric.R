# validate_nonparametric.R
# This script is used to validate BioMedStatX's nonparametric fallback implementations
# for Two-Way ANOVA (Freedman-Lane) and Mixed ANOVA (Brunner-Langer).

# Required packages
if (!requireNamespace("nparLD", quietly = TRUE)) install.packages("nparLD")
if (!requireNamespace("lmPerm", quietly = TRUE)) install.packages("lmPerm")
if (!requireNamespace("lme4", quietly = TRUE)) install.packages("lme4")

library(nparLD)
library(lmPerm)
library(lme4)

cat("==============================================\n")
cat("Validation of Nonparametric Fallbacks\n")
cat("==============================================\n\n")

# 1. Validation for Brunner-Langer (Mixed ANOVA Fallback: F1-LD-F1)
cat("--- 1. Brunner-Langer (F1-LD-F1) Validation ---\n")
# Generate a dummy dataset with 1 Between factor and 1 Within factor
set.seed(42)
n_subjects <- 20
df_mixed <- data.frame(
  Subject = rep(1:n_subjects, each=3),
  Between = factor(rep(rep(c("A", "B"), each=n_subjects/2), each=3)),
  Within = factor(rep(c("T1", "T2", "T3"), times=n_subjects)),
  Value = rnorm(n_subjects * 3, mean=50, sd=10)
)
# Add some artificial effect
df_mixed$Value <- df_mixed$Value + ifelse(df_mixed$Between=="B", 5, 0)
df_mixed$Value <- df_mixed$Value + ifelse(df_mixed$Within=="T3", 10, 0)
df_mixed$Value <- df_mixed$Value + ifelse(df_mixed$Between=="B" & df_mixed$Within=="T3", 15, 0)

# Run nparLD
ex.f1f1 <- nparLD(Value ~ Between * Within, data=df_mixed, subject="Subject", description=FALSE)
cat("Reference ATS Output (nparLD):\n")
print(ex.f1f1$ANOVA.test)

cat("\n==============================================\n")

# 2. Validation for Freedman-Lane (Two-Way ANOVA Fallback)
cat("--- 2. Freedman-Lane (Permutation Two-Way) Validation ---\n")
# Generate a dummy dataset with 2 Between factors
df_twoway <- data.frame(
  Factor1 = factor(rep(c("Low", "High"), each=20)),
  Factor2 = factor(rep(rep(c("Control", "Treatment"), each=10), times=2)),
  Value = rnorm(40, mean=100, sd=15)
)
# Add some artificial effect
df_twoway$Value <- df_twoway$Value + ifelse(df_twoway$Factor1=="High", 20, 0)
df_twoway$Value <- df_twoway$Value + ifelse(df_twoway$Factor2=="Treatment", 10, 0)

# Run lmPerm
# Note: lmPerm's aovp uses permutation tests but might differ slightly from exact Freedman-Lane depending on parameterization.
# It serves as a good benchmark for nonparametric F-tests.
set.seed(42)
fit_perm <- aovp(Value ~ Factor1 * Factor2, data=df_twoway, maxIter=5000)
cat("Reference Permutation ANOVA Output (lmPerm):\n")
print(summary(fit_perm))

cat("\n==============================================\n")
cat("Run BioMedStatX's corresponding nonparametric fallbacks on this data to compare p-values.\n")
