"""Effect-driven post-hoc contrasts for a Mixed ANOVA.

Follows the standard textbook decision after a significant Mixed ANOVA:

* interaction significant  -> simple main effects
    - within-factor pairwise *within each between-group* (paired t)
    - between-factor pairwise *at each within-level* (independent t)
    (NO cross-cell comparisons like GroupA:t1 vs GroupB:t2 — those confound both
     factors and, folded into one all-pairwise family, only dilute the FWER
     correction and cost power on the meaningful contrasts.)
* interaction n.s., within main effect significant  -> marginal within means
    (within-factor pairwise, collapsed across groups; paired t)
* interaction n.s., between main effect significant  -> marginal between means
    (between-factor pairwise, collapsed across the within factor; independent t)

ERROR TERM IS ASSUMPTION-DRIVEN, not hard-wired (consistent with the rest of the
app, which tests an assumption and then picks the method — e.g. Levene -> Welch
vs. Student, Mauchly -> Greenhouse-Geisser):

* within contrasts: Levene on the *within-subject differences* per between-group
  (Ctrl_diffs vs Trt_diffs for that pair). The variance of the DIFFERENCES — not
  of the raw cell values — is what a pooled paired error term rests on; the raw
  between-subject variance cancels inside a paired difference.
    - homogeneous -> POOLED error across groups (emmeans convention):
        s2_pooled = sum((n_g-1) * var(d_g)) / sum(n_g-1),  df = sum(n_g-1)
        t_g       = mean(d_g) / sqrt(s2_pooled / n_g)
    - heterogeneous -> ISOLATED error (paired t per group, df = n_g-1)
  Pooling is not uniformly "more power": it helps the group with the larger
  difference variance and penalises the one with the smaller, which is exactly
  why the choice is gated on Levene instead of being fixed.
* between contrasts: Levene on the raw group values -> Student (pooled variance)
  vs. Welch (separate variances), the same routing the rest of the app uses.

Every comparison carries `error_term` ("pooled"/"isolated") AND the underlying
`variance_check` (Levene statistic + p) so the choice is auditable.

Holm-Šidák is applied *per effect family* so unrelated families don't dilute
each other. Ordering of levels is human-friendly (see core.level_order).

Pure numeric (numpy/pandas/scipy/statsmodels); returns plain comparison dicts.
"""
from itertools import combinations

import numpy as np
import pandas as pd

from core.level_order import natural_order

_LEVENE_LABEL = "Levene (Brown-Forsythe, center=median)"


def _holm_sidak(pvals):
    from core.lazy_imports import get_statsmodels_multitest
    multipletests = get_statsmodels_multitest()
    if not pvals:
        return []
    return list(multipletests(pvals, method="holm-sidak")[1])


def _levene_check(arrays, labels, dv, factor_name, basis):
    """Variance-homogeneity check reusing the app's single Levene implementation.

    Delegates to MixedAnovaAssumptionEngine._perform_levene_test, which is the
    same Brown-Forsythe (center='median') / p > 0.05 convention used by the
    Welch-vs-Student routing. Returns a dict that always carries the statistic
    and p-value so the downstream choice can be audited.
    """
    base = {"test": _LEVENE_LABEL, "basis": basis,
            "statistic": None, "p_value": None, "equal_variance": False}
    usable, used_labels = [], []
    for arr, lab in zip(arrays, labels):
        if arr is None:
            continue
        a = np.asarray(arr, dtype=float)
        if a.size >= 3 and np.ptp(a) > 0:
            usable.append(a)
            used_labels.append(str(lab))
    if len(usable) < 2:
        base["note"] = ("fewer than two testable groups (need n>=3 and non-constant "
                        "values); defaulted to separate/isolated error terms")
        return base
    from statistical_testing.mixed_assumptions import MixedAnovaAssumptionEngine
    res = MixedAnovaAssumptionEngine._perform_levene_test(usable, used_labels, dv, factor_name)
    if res.get("error") or res.get("p_value") is None:
        base["note"] = res.get("error", "Levene unavailable; defaulted to isolated error terms")
        return base
    base["statistic"] = float(res["statistic"])
    base["p_value"] = float(res["p_value"])
    base["equal_variance"] = bool(res.get("assumption_met"))
    return base


def _cohen_d_paired(d):
    d = np.asarray(d, float)
    sd = np.std(d, ddof=1)
    return float(np.mean(d) / sd) if sd > 0 else 0.0


def _cohen_d_indep(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    sp = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    return float((np.mean(a) - np.mean(b)) / sp) if sp > 0 else 0.0


def _subject_diffs(df, dv, subject, within, w1, w2):
    """Per-subject within-pair differences (w1 - w2) on the subjects present in both."""
    d1 = df[df[within] == w1][[subject, dv]].set_index(subject)[dv]
    d2 = df[df[within] == w2][[subject, dv]].set_index(subject)[dv]
    common = d1.index.intersection(d2.index)
    if len(common) < 3:
        return None
    return d1.loc[common].to_numpy(dtype=float) - d2.loc[common].to_numpy(dtype=float)


def _within_comp(group_desc, w1, w2, diffs, error_term, vcheck, s2_pooled=None, df_pooled=None):
    """Build one within comparison, either on a pooled or an isolated error term."""
    from scipy.stats import ttest_1samp, t as _t
    n = int(len(diffs))
    mean_d = float(np.mean(diffs))
    if error_term == "pooled" and s2_pooled and s2_pooled > 0:
        stat = mean_d / np.sqrt(s2_pooled / n)
        dfree = int(df_pooled)
        p = float(2.0 * _t.sf(abs(stat), dfree))
        test_name = "Paired t-test (pooled error)"
    else:
        error_term = "isolated"
        res = ttest_1samp(diffs, 0.0)   # identical to ttest_rel(a, b)
        stat, p = float(res[0]), float(res[1])
        dfree = n - 1
        test_name = "Paired t-test"
    return {
        "group1": f"{group_desc}{w1}", "group2": f"{group_desc}{w2}",
        "test": test_name, "comparison_type": "within_subject",
        "p_val": float(p), "statistic": float(stat), "df": dfree,
        "effect_size": _cohen_d_paired(diffs), "n_pairs": n,
        "error_term": error_term, "variance_check": vcheck,
    }


def _within_family(data, dv, subject, between, within, groups, levels, group_desc):
    """Within contrasts for every (group, level-pair), with a per-pair pooled/isolated
    decision driven by Levene on the subject differences across the between-groups."""
    comps = []
    for w1, w2 in combinations(levels, 2):
        per_group = {}
        for g in groups:
            sub = data if between is None else data[data[between] == g]
            d = _subject_diffs(sub, dv, subject, within, w1, w2)
            if d is not None:
                per_group[g] = d
        if not per_group:
            continue
        if between is None or len(per_group) < 2:
            vcheck = {"test": _LEVENE_LABEL,
                      "basis": "within-subject differences",
                      "statistic": None, "p_value": None, "equal_variance": False,
                      "note": "single group of differences; no between-group variance "
                              "comparison applicable -> isolated error term"}
            pooled = False
        else:
            vcheck = _levene_check(
                list(per_group.values()), list(per_group.keys()), dv, between,
                f"within-subject differences ({w1} - {w2}) per {between} group")
            pooled = bool(vcheck["equal_variance"])
        s2p = dfp = None
        if pooled:
            dfp = int(sum(len(d) - 1 for d in per_group.values()))
            if dfp > 0:
                s2p = float(sum((len(d) - 1) * np.var(d, ddof=1) for d in per_group.values()) / dfp)
            pooled = bool(s2p and s2p > 0)
        for g, d in per_group.items():
            desc = group_desc(g)
            comps.append(_within_comp(desc, w1, w2, d,
                                      "pooled" if pooled else "isolated", vcheck, s2p, dfp))
    return comps


def _indep_pair(df, dv, between, level_desc, g1, g2, alpha):
    """Between contrast; Levene on the raw group values routes Student vs. Welch."""
    from scipy.stats import ttest_ind
    a = df[df[between] == g1][dv].to_numpy(dtype=float)
    b = df[df[between] == g2][dv].to_numpy(dtype=float)
    if len(a) < 2 or len(b) < 2:
        return None
    vcheck = _levene_check([a, b], [g1, g2], dv, between, "raw group values")
    equal_var = bool(vcheck["equal_variance"])
    t, p = ttest_ind(a, b, equal_var=equal_var)
    n1, n2 = len(a), len(b)
    if equal_var:
        dfree = float(n1 + n2 - 2)
        test_name = "Independent t-test (Student, pooled variance)"
    else:
        s1, s2 = np.var(a, ddof=1) / n1, np.var(b, ddof=1) / n2
        denom = (s1 ** 2) / (n1 - 1) + (s2 ** 2) / (n2 - 1)
        dfree = float(((s1 + s2) ** 2) / denom) if denom > 0 else float(n1 + n2 - 2)
        test_name = "Welch's t-test (separate variances)"
    return {
        "group1": f"{g1}{level_desc}", "group2": f"{g2}{level_desc}",
        "test": test_name, "comparison_type": "between_subject",
        "p_val": float(p), "statistic": float(t), "df": dfree,
        "effect_size": _cohen_d_indep(a, b), "n_pairs": int(min(n1, n2)),
        "error_term": "pooled" if equal_var else "isolated", "variance_check": vcheck,
    }


def _finalize_family(comps, family, alpha):
    """Apply Holm-Šidák across one effect family and attach fields."""
    if not comps:
        return []
    adj = _holm_sidak([c["p_val"] for c in comps])
    out = []
    for c, p_corr in zip(comps, adj):
        out.append({
            "group1": c["group1"], "group2": c["group2"],
            "test": c["test"], "p_value": float(p_corr), "statistic": c["statistic"],
            "p_value_raw": float(c["p_val"]), "df": c["df"],
            "corrected": True, "correction_method": "Holm-Šidák",
            "effect_size": c["effect_size"], "effect_size_type": "cohen_d",
            "significant": bool(p_corr < alpha),
            "comparison_type": c["comparison_type"], "family": family,
            "n_pairs": c["n_pairs"],
            "error_term": c["error_term"], "variance_check": c["variance_check"],
        })
    return out


def simple_main_effects(df, dv, subject, between, within, alpha=0.05):
    """Interaction-significant case: within-per-group + between-per-within-level.

    Two families, each Holm-Šidák-corrected on its own. The within family's error
    term (pooled vs isolated) is decided per level-pair by Levene on the subject
    differences; the between family routes Student vs Welch by Levene on the raw
    values.
    """
    data = df[[subject, between, within, dv]].dropna()
    groups = natural_order(data[between].unique())
    levels = natural_order(data[within].unique())

    within_comps = _within_family(data, dv, subject, between, within, groups, levels,
                                  group_desc=lambda g: f"{g}:")

    between_comps = []
    for w in levels:
        sub = data[data[within] == w]
        for g1, g2 in combinations(groups, 2):
            c = _indep_pair(sub, dv, between, f":{w}", g1, g2, alpha)
            if c:
                between_comps.append(c)

    return (_finalize_family(within_comps, "within_simple_effect", alpha)
            + _finalize_family(between_comps, "between_simple_effect", alpha))


def marginal_within(df, dv, subject, within, alpha=0.05):
    """Interaction n.s., within main effect significant: within-factor pairwise on
    marginal means (collapsed across groups), paired t across all subjects.

    Collapsed across the between factor there is only one set of differences, so
    no between-group variance comparison applies -> isolated error term.
    """
    data = df[[subject, within, dv]].dropna()
    levels = natural_order(data[within].unique())
    comps = _within_family(data, dv, subject, None, within, [None], levels,
                           group_desc=lambda g: "")
    return _finalize_family(comps, "within_marginal", alpha)


def marginal_between(df, dv, subject, between, within, alpha=0.05):
    """Interaction n.s., between main effect significant: between-factor pairwise on
    marginal means (per-subject mean collapsed across the within factor)."""
    data = df[[subject, between, within, dv]].dropna()
    subj_means = data.groupby([subject, between], observed=True)[dv].mean().reset_index()
    groups = natural_order(subj_means[between].unique())
    comps = []
    for g1, g2 in combinations(groups, 2):
        c = _indep_pair(subj_means, dv, between, "", g1, g2, alpha)
        if c:
            comps.append(c)
    return _finalize_family(comps, "between_marginal", alpha)


def mixed_effect_driven_posthoc(df, dv, subject, between, within, alpha=0.05,
                                interaction_p=None, within_p=None, between_p=None):
    """Route to the right contrast family based on which omnibus effects are
    significant. Falls back to simple main effects when effect p-values are not
    supplied (still the meaningful contrasts, never cross-cells).

    Returns (comparisons, mode) where mode is one of
    'simple_main_effects' | 'marginal_within' | 'marginal_between' | 'none'.
    """
    def _sig(p):
        return p is not None and p < alpha

    if interaction_p is None and within_p is None and between_p is None:
        return simple_main_effects(df, dv, subject, between, within, alpha), "simple_main_effects"

    if _sig(interaction_p):
        return simple_main_effects(df, dv, subject, between, within, alpha), "simple_main_effects"
    if _sig(within_p):
        return marginal_within(df, dv, subject, within, alpha), "marginal_within"
    if _sig(between_p):
        return marginal_between(df, dv, subject, between, within, alpha), "marginal_between"
    return [], "none"
