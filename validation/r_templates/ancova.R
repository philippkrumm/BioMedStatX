args <- commandArgs(trailingOnly = TRUE)
csv_path <- args[1]
df <- read.csv(csv_path, stringsAsFactors = FALSE)
df <- df[!is.na(df$Value) & !is.na(df$Covariate), ]
df$Group <- as.factor(df$Group)

# Type II SS via sequential model comparison — matches Python statsmodels anova_lm(typ=2)
# SS_Group(II)     = RSS(~Covariate)   - RSS(~Group + Covariate)
# SS_Covariate(II) = RSS(~Group)       - RSS(~Group + Covariate)
m_full   <- lm(Value ~ Group + Covariate, data = df)
m_no_grp <- lm(Value ~ Covariate,         data = df)
m_no_cov <- lm(Value ~ Group,             data = df)

ss_group <- sum(residuals(m_no_grp)^2) - sum(residuals(m_full)^2)
ss_cov   <- sum(residuals(m_no_cov)^2) - sum(residuals(m_full)^2)
ss_resid <- sum(residuals(m_full)^2)
df_resid <- df.residual(m_full)

f_group <- (ss_group / 1) / (ss_resid / df_resid)
f_cov   <- (ss_cov   / 1) / (ss_resid / df_resid)
p_group <- pf(f_group, 1, df_resid, lower.tail = FALSE)
p_cov   <- pf(f_cov,   1, df_resid, lower.tail = FALSE)

# eta-squared = SS_group / (SS_group + SS_cov + SS_resid)
eta_sq <- ss_group / (ss_group + ss_cov + ss_resid)

# Output: p_group F_group eta_sq p_covariate
cat(p_group, f_group, eta_sq, p_cov, "\n")
