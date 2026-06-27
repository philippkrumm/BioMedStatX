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
from scipy.stats import multivariate_t, t as student_t


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


def variance_components(s: SplitPlotStrata) -> tuple[float, float]:
    """Return (sig_s2, sig_w2): between-subject and within-subject variances."""
    sig_w2 = s.ms_res
    sig_s2 = (s.ms_sg - s.ms_res) / s.W
    return sig_s2, sig_w2


def contrast_se_df(s: SplitPlotStrata) -> tuple[float, float]:
    """Constant contrast SE and the Satterthwaite df for a between-group
    contrast at a fixed within level."""
    sig_s2, sig_w2 = variance_components(s)
    var_t = sig_s2 + sig_w2                      # per-obs variance at a fixed level
    se = float(np.sqrt(2.0 * var_t / s.n_per_group))

    c1 = 1.0 / s.W
    c2 = (s.W - 1.0) / s.W
    num = (c1 * s.ms_sg + c2 * s.ms_res) ** 2
    den = (c1 * s.ms_sg) ** 2 / s.df_sg + (c2 * s.ms_res) ** 2 / s.df_res
    df = float(num / den)
    return se, df


def _mvt_adjusted_p(t_values: list[float], df: float) -> list[float]:
    """Single-step two-sided multivariate-t adjusted p-values for a family that
    shares one control (balanced => equicorrelation 0.5). Family size k = len."""
    k = len(t_values)
    if k == 1:
        return [float(2.0 * student_t.sf(abs(t_values[0]), df))]
    R = np.full((k, k), 0.5)
    np.fill_diagonal(R, 1.0)
    rv = multivariate_t(loc=np.zeros(k), shape=R, df=df, allow_singular=True)
    out = []
    for tv in t_values:
        c = abs(float(tv))
        p_all = float(rv.cdf(np.full(k, c), lower_limit=np.full(k, -c)))
        out.append(float(min(1.0, max(0.0, 1.0 - p_all))))
    return out


@dataclass
class RMStrata:
    W: int
    n: int
    df_res: int
    ms_res: float
    cell_means: pd.Series             # index = within level
    within_levels: list


def rm_strata(df: pd.DataFrame, dv: str, subject: str, within: str) -> RMStrata:
    """Single within-subject error stratum for a balanced, complete one-way
    repeated-measures design. Raises UnsupportedDesignError otherwise."""
    data = df[[subject, within, dv]].dropna()
    within_levels = list(pd.unique(data[within]))
    W = len(within_levels)

    counts = data.groupby([subject, within]).size()
    if not (counts == 1).all():
        raise UnsupportedDesignError("subjects must have exactly one row per within level")
    n_per = data.groupby(within)[subject].nunique()
    if n_per.nunique() != 1:
        raise UnsupportedDesignError("unbalanced: within levels differ in subject count")
    n = int(n_per.iloc[0])
    if len(data) != W * n:
        raise UnsupportedDesignError("incomplete cells in the design")

    grand = data[dv].mean()
    subj_mean = data.groupby(subject)[dv].mean()
    cell_means = data.groupby(within)[dv].mean()

    ss_res = 0.0
    for _, r in data.iterrows():
        e = r[dv] - subj_mean[r[subject]] - cell_means[r[within]] + grand
        ss_res += e * e
    df_res = (n - 1) * (W - 1)
    ms_res = ss_res / df_res
    return RMStrata(W=W, n=n, df_res=df_res, ms_res=ms_res,
                    cell_means=cell_means, within_levels=within_levels)


def rm_dunnett_emm_mvt(df: pd.DataFrame, dv: str, subject: str, within: str,
                       control_level, alpha: float = 0.05) -> list[dict]:
    """Level-vs-baseline EMM contrasts for a pure one-way RM ANOVA, jointly
    multivariate-t adjusted. Reproduces R emmeans on a univariate aov model:
    ``emmeans(~within) |> contrast("trt.vs.ctrl", adjust="mvt")``.

    Under the univariate RM model (compound symmetry / sphericity) every
    vs-baseline contrast shares the pooled within-subject error, giving
    Var=2*MS_res/n, Cov=MS_res/n => equicorrelation 0.5 — the same structure as
    independent-groups Dunnett, so the closed-form mvt in ``_mvt_adjusted_p``
    applies exactly. Callers must fall back to the isolated paired-t path on
    UnsupportedDesignError (unbalanced/incomplete designs).
    """
    s = rm_strata(df, dv=dv, subject=subject, within=within)
    if control_level not in s.within_levels:
        raise UnsupportedDesignError(f"control level {control_level!r} not present")

    se = float(np.sqrt(2.0 * s.ms_res / s.n))
    treatments = [lvl for lvl in s.within_levels if lvl != control_level]

    t_values, rows = [], []
    for lvl in treatments:
        est = float(s.cell_means[lvl] - s.cell_means[control_level])
        tval = est / se if se > 0 else 0.0
        t_values.append(tval)
        rows.append({"level": lvl, "control": control_level, "estimate": est,
                     "se": se, "df": float(s.df_res), "t": tval})

    p_adj = _mvt_adjusted_p(t_values, float(s.df_res))
    results: list[dict] = []
    for row, p in zip(rows, p_adj):
        row["p_value"] = p
        row["significant"] = bool(p < alpha)
        results.append(row)
    return results


def mixed_dunnett_emm_mvt(df: pd.DataFrame, dv: str, subject: str, between: str,
                          within: str, control_group, alpha: float = 0.05) -> list[dict]:
    """Treatment-vs-control EMM contrasts at each within level, mvt-adjusted
    within each within level. Raises UnsupportedDesignError for designs the
    closed form does not cover (callers must then fall back to isolated t-tests).
    """
    s = split_plot_strata(df, dv=dv, subject=subject, between=between, within=within)
    if control_group not in s.group_levels:
        raise UnsupportedDesignError(f"control group {control_group!r} not present")

    se, ddf = contrast_se_df(s)
    treatments = [g for g in s.group_levels if g != control_group]

    results: list[dict] = []
    for level in s.within_levels:
        t_values, rows = [], []
        for trt in treatments:
            est = float(s.cell_means.loc[trt, level] - s.cell_means.loc[control_group, level])
            tval = est / se
            t_values.append(tval)
            rows.append({"within_level": level, "treatment": trt,
                         "control": control_group, "estimate": est,
                         "se": se, "df": ddf, "t": tval})
        p_adj = _mvt_adjusted_p(t_values, ddf)
        for row, p in zip(rows, p_adj):
            row["p_value"] = p
            row["significant"] = bool(p < alpha)
            results.append(row)
    return results
