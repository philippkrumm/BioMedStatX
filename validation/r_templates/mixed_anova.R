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
    # detailed=TRUE exposes pes (partial eta-squared) column
    m <- ezANOVA(data=df, dv=Value, wid=Subject,
                 between=Group, within=Time, type=3, detailed=TRUE)
  })
  tbl <- m$ANOVA
  # Rows: Group (between), Time (within), Group:Time (interaction)
  p_between     <- tbl$p[tbl$Effect == "Group"]
  p_within      <- tbl$p[tbl$Effect == "Time"]
  p_interaction <- tbl$p[tbl$Effect == "Group:Time"]
  f_between     <- tbl$F[tbl$Effect == "Group"]
  f_within      <- tbl$F[tbl$Effect == "Time"]
  f_interaction <- tbl$F[tbl$Effect == "Group:Time"]
  # pes = partial eta-squared (available with detailed=TRUE)
  peta_between     <- tbl$pes[tbl$Effect == "Group"]
  peta_within      <- tbl$pes[tbl$Effect == "Time"]
  peta_interaction <- tbl$pes[tbl$Effect == "Group:Time"]
  # Output: p_between p_within p_interaction F_between F_within F_interaction peta x3
  cat(p_between, p_within, p_interaction,
      f_between, f_within, f_interaction,
      peta_between, peta_within, peta_interaction, "\n")
} else {
  # Fallback: lme4 + lmerTest (interaction effect available, pes not)
  if (!requireNamespace("lmerTest", quietly=TRUE)) {
    cat("NA NA NA NA NA NA NA NA NA\n")
    quit()
  }
  library(lmerTest)
  m <- lmer(Value ~ Group * Time + (1|Subject), data=df)
  a <- anova(m)
  cat(a["Group","Pr(>F)"], a["Time","Pr(>F)"], a["Group:Time","Pr(>F)"],
      a["Group","F value"], a["Time","F value"], a["Group:Time","F value"],
      NA, NA, NA, "\n")
}
