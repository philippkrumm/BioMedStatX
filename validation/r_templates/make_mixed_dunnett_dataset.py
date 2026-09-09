"""Deterministic synthetic split-plot dataset for the Mixed-ANOVA Dunnett-type
(treatment-vs-control at each within level) golden reference.

Both the R ground-truth generator (mixed_dunnett_emmeans.R) and the future
Python EMM/mvt implementation read THIS exact CSV, so the comparison is on a
frozen dataset. Balanced split-plot -> classical afex::aov_ez strata, so the
degrees of freedom stay matchable without Satterthwaite/Kenward-Roger.

Design:
  between factor Group : Ctrl, TrtA, TrtB   (Ctrl is the reference)
  within  factor Time  : T1, T2, T3
  Subject              : 8 per group, nested in Group (24 subjects, 72 rows)
"""
import numpy as np
import pandas as pd

OUT = "tests/golden/mixed_dunnett_emmeans_dataset.csv"

GROUPS = ["Ctrl", "TrtA", "TrtB"]
TIMES = ["T1", "T2", "T3"]
N_PER_GROUP = 8

# Fixed true cell means (Group x Time): a real, non-trivial interaction so the
# treatment-vs-control contrasts differ across timepoints.
CELL_MEAN = {
    ("Ctrl", "T1"): 10.0, ("Ctrl", "T2"): 10.5, ("Ctrl", "T3"): 11.0,
    ("TrtA", "T1"): 10.2, ("TrtA", "T2"): 12.5, ("TrtA", "T3"): 15.0,
    ("TrtB", "T1"):  9.8, ("TrtB", "T2"): 11.0, ("TrtB", "T3"): 12.0,
}
SUBJECT_SD = 1.2   # between-subject random intercept
NOISE_SD = 0.8     # within-subject residual


def main():
    rng = np.random.default_rng(20260627)
    rows = []
    sid = 0
    for group in GROUPS:
        for _ in range(N_PER_GROUP):
            sid += 1
            subject = f"S{sid:02d}"
            subj_intercept = rng.normal(0.0, SUBJECT_SD)
            for time in TIMES:
                value = CELL_MEAN[(group, time)] + subj_intercept + rng.normal(0.0, NOISE_SD)
                rows.append({
                    "Subject": subject,
                    "Group": group,
                    "Time": time,
                    "Value": round(float(value), 6),
                })
    df = pd.DataFrame(rows, columns=["Subject", "Group", "Time", "Value"])
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(df)} rows, {df['Subject'].nunique()} subjects")


if __name__ == "__main__":
    main()
