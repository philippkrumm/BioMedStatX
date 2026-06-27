"""Estimated-marginal-means + multivariate-t post-hoc for balanced split-plot
(Mixed ANOVA) designs. Closed-form reproduction of R afex::aov_ez +
emmeans(model="univariate") + contrast(trt.vs.ctrl, adjust="mvt").

Pure numeric (numpy/pandas/scipy); no PyQt, no model refit, no R at runtime.
Only balanced, complete designs are supported; callers must fall back to the
isolated-t-test path on UnsupportedDesignError.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class UnsupportedDesignError(ValueError):
    """Raised when the design is not a balanced, complete split-plot."""


@dataclass
class SplitPlotStrata:
    G: int
    W: int
    n_per_group: int
    df_sg: int
    df_res: int
    ms_sg: float
    ms_res: float
    cell_means: pd.DataFrame          # index=between level, columns=within level
    group_levels: list
    within_levels: list


def split_plot_strata(df: pd.DataFrame, dv: str, subject: str,
                      between: str, within: str) -> SplitPlotStrata:
    data = df[[subject, between, within, dv]].dropna()
    group_levels = list(pd.unique(data[between]))
    within_levels = list(pd.unique(data[within]))
    G, W = len(group_levels), len(within_levels)

    # Each subject must appear exactly once per within level, in one group.
    counts = data.groupby([subject, within]).size()
    if not (counts == 1).all():
        raise UnsupportedDesignError("subjects must have exactly one row per within level")
    subj_group = data.groupby(subject)[between].nunique()
    if not (subj_group == 1).all():
        raise UnsupportedDesignError("each subject must belong to one between group")

    subjects_per_group = data.groupby(between)[subject].nunique()
    if subjects_per_group.nunique() != 1:
        raise UnsupportedDesignError("unbalanced: groups differ in subject count")
    n = int(subjects_per_group.iloc[0])

    if len(data) != G * W * n:
        raise UnsupportedDesignError("incomplete cells in the design")

    group_mean = data.groupby(between)[dv].mean()
    subj_mean = data.groupby(subject)[dv].mean()
    subj_to_group = data.groupby(subject)[between].first()
    cell_means = data.groupby([between, within])[dv].mean().unstack()

    ss_sg = 0.0
    for subj, m in subj_mean.items():
        ss_sg += W * (m - group_mean[subj_to_group[subj]]) ** 2
    df_sg = G * (n - 1)
    ms_sg = ss_sg / df_sg

    ss_res = 0.0
    for _, r in data.iterrows():
        e = (r[dv] - subj_mean[r[subject]]
             - cell_means.loc[r[between], r[within]] + group_mean[r[between]])
        ss_res += e * e
    df_res = G * (n - 1) * (W - 1)
    ms_res = ss_res / df_res

    return SplitPlotStrata(G=G, W=W, n_per_group=n, df_sg=df_sg, df_res=df_res,
                           ms_sg=ms_sg, ms_res=ms_res, cell_means=cell_means,
                           group_levels=group_levels, within_levels=within_levels)
