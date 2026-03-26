# Mixed ANOVA via ez package or nlme as fallback
args <- commandArgs(trailingOnly=TRUE)
csv_path <- args[1]
df <- read.csv(csv_path, stringsAsFactors=FALSE)
df$Subject <- as.factor(df$Subject)
df$Group   <- as.factor(df$Group)
df$Time    <- as.factor(df$Time)

if (requireNamespace("ez", quietly=TRUE)) {
  library(ez)
  suppressWarnings({
    m <- ezANOVA(data=df, dv=Value, wid=Subject,
                 between=Group, within=Time, type=3, detailed=FALSE)
  })
  tbl <- m$ANOVA
  # Rows: Group (between), Time (within), Group:Time (interaction)
  p_between <- tbl$p[tbl$Effect == "Group"]
  p_within  <- tbl$p[tbl$Effect == "Time"]
  f_between <- tbl$F[tbl$Effect == "Group"]
  f_within  <- tbl$F[tbl$Effect == "Time"]
  cat(p_between, p_within, f_between, f_within, "\n")
} else {
  # Fallback: lme4 + lmerTest
  if (!requireNamespace("lmerTest", quietly=TRUE)) {
    cat("NA NA NA NA\n")
    quit()
  }
  library(lmerTest)
  m <- lmer(Value ~ Group * Time + (1|Subject), data=df)
  a <- anova(m)
  cat(a["Group","Pr(>F)"], a["Time","Pr(>F)"],
      a["Group","F value"], a["Time","F value"], "\n")
}
