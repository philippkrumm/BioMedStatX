"""Seed-based deterministic dataset generator for the BioMedStatX fuzzer.

Every case is a pure function of its integer seed, so any crash the orchestrator
finds is reproducible by re-running that one seed. The generator builds a clean
design for a randomly chosen test type, then layers statistical / parser
mutations on top (skew, heteroscedasticity, zero variance, NaN/Inf, huge values,
collinear covariates, unicode/control chars in labels, comma decimals, tiny
groups, ...).

`build_case(seed)` returns a `FuzzCase`; `case_to_analyze_kwargs(case)` turns it
into the kwargs dict for `AnalysisManager.analyze`. The DataFrame travels inside
`analysis_context["injected_df"]` — the documented single-source-of-truth path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Test designs the fuzzer drives.
TEST_TYPES = [
    "oneway", "ttest", "rm_anova", "two_way_anova", "mixed_anova", "ancova",
    "correlation", "regression", "firth_logistic", "lmm",
]

MUTATIONS = [
    "none", "nan_scatter", "nan_group", "inf", "zero_variance_group",
    "huge_values", "outlier_10sigma", "unicode_labels", "control_chars",
    "comma_decimals", "tiny_groups", "high_cardinality", "collinear_covariate",
    "heavy_skew", "heteroscedastic", "all_constant",
    # rank-deficiency / structural mutations (target (X^T X)^-1 singularity)
    "empty_factor_cell", "cross_level_missing",
]

# Simple categorical palette so plotting (when enabled) gets valid colors/hatches.
_PALETTE = ["#0f766e", "#d97706", "#0369a1", "#be123c", "#7e22ce", "#65a30d", "#0891b2", "#475569"]
_HATCHES = ["", "/", "\\", "x", ".", "o", "+", "*"]


@dataclass
class FuzzCase:
    seed: int
    test_label: str
    df: pd.DataFrame
    mutations: List[str]
    analyze_kwargs: Dict[str, Any] = field(default_factory=dict)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _base_design(rng: np.random.Generator, test_label: str):
    """Return (df, context, kwargs) for a clean design of the chosen type."""
    n_groups = int(rng.integers(2, 5)) if test_label != "ttest" else 2
    n_per = int(rng.integers(3, 12))
    group_names = [f"G{i+1}" for i in range(n_groups)]

    rows = []
    if test_label in ("oneway", "ttest"):
        for gi, g in enumerate(group_names):
            for _ in range(n_per):
                rows.append({"Grp": g, "Val": float(rng.normal(gi, 1.0))})
        df = pd.DataFrame(rows)
        ctx = {"factor_columns": ["Grp"], "dv_columns": ["Val"],
               "group_labels": group_names, "mode": "single", "inferred_test": "one_way_anova"}
        kwargs = {"group_col": "Grp", "groups": group_names, "value_cols": ["Val"],
                  "dependent": bool(rng.integers(0, 2)) if test_label == "ttest" else False}

    elif test_label in ("rm_anova", "mixed_anova"):
        n_subj = int(rng.integers(4, 10))
        within_levels = [f"T{i}" for i in range(int(rng.integers(2, 4)))]
        between = [f"B{i%2}" for i in range(n_subj)]
        for s in range(n_subj):
            for lvl in within_levels:
                rows.append({
                    "Subject": f"S{s}", "Time": lvl, "Between": between[s],
                    "Val": float(rng.normal(0, 1)),
                })
        df = pd.DataFrame(rows)
        if test_label == "rm_anova":
            ctx = {"factor_columns": ["Time"], "dv_columns": ["Val"],
                   "group_labels": within_levels, "subject_column": "Subject", "mode": "single",
                   "inferred_test": "repeated_measures_anova", "within_factors": ["Time"]}
            kwargs = {"group_col": "Time", "groups": within_levels, "value_cols": ["Val"],
                      "dependent": True, "subject_column": "Subject"}
        else:
            ctx = {"factor_columns": ["Between"], "dv_columns": ["Val"],
                   "group_labels": sorted(set(between)), "subject_column": "Subject", "mode": "single",
                   "inferred_test": "mixed_anova", "between_factors": ["Between"], "within_factors": ["Time"]}
            kwargs = {"group_col": "Between", "groups": sorted(set(between)), "value_cols": ["Val"],
                      "dependent": True, "subject_column": "Subject"}

    elif test_label == "two_way_anova":
        fa = [f"A{i}" for i in range(2)]
        fb = [f"B{i}" for i in range(2)]
        for a in fa:
            for b in fb:
                for _ in range(n_per):
                    rows.append({"FacA": a, "FacB": b, "Val": float(rng.normal(0, 1))})
        df = pd.DataFrame(rows)
        ctx = {"factor_columns": ["FacA", "FacB"], "dv_columns": ["Val"],
               "group_labels": fa, "mode": "single", "inferred_test": "two_way_anova"}
        kwargs = {"group_col": "FacA", "groups": fa, "value_cols": ["Val"],
                  "dependent": False}

    elif test_label == "ancova":
        for gi, g in enumerate(group_names):
            for _ in range(n_per):
                cov = float(rng.normal(0, 1))
                rows.append({"Grp": g, "Cov": cov, "Val": float(gi + 0.5 * cov + rng.normal(0, 1))})
        df = pd.DataFrame(rows)
        ctx = {"factor_columns": ["Grp"], "dv_columns": ["Val"],
               "group_labels": group_names, "covariates": ["Cov"], "mode": "single",
               "inferred_test": "ancova"}
        kwargs = {"group_col": "Grp", "groups": group_names, "value_cols": ["Val"],
                  "dependent": False, "covariates": ["Cov"]}

    elif test_label == "correlation":
        n = int(rng.integers(10, 30))
        x = rng.normal(0, 1, size=n)
        y = 0.6 * x + rng.normal(0, 0.8, size=n)
        df = pd.DataFrame({"Grp": ["Sample"] * n, "X": x.tolist(), "Y": y.tolist()})
        ctx = {"factor_columns": ["Grp"], "dv_columns": ["Y"], "group_labels": ["Sample"],
               "x_variable": "X", "inferred_test": "correlation", "mode": "single"}
        kwargs = {"group_col": "Grp", "groups": ["Sample"], "value_cols": ["Y"],
                  "dependent": False}

    elif test_label == "regression":
        n = int(rng.integers(10, 30))
        x = rng.uniform(0, 10, size=n)
        y = 2.0 * x + rng.normal(0, 2.0, size=n)
        df = pd.DataFrame({"Grp": ["Sample"] * n, "X": x.tolist(), "Y": y.tolist()})
        ctx = {"factor_columns": ["Grp"], "dv_columns": ["Y"], "group_labels": ["Sample"],
               "x_variable": "X", "inferred_test": "linear_regression", "mode": "single"}
        kwargs = {"group_col": "Grp", "groups": ["Sample"], "value_cols": ["Y"],
                  "dependent": False}

    elif test_label == "firth_logistic":
        n_subj = int(rng.integers(10, 25))
        group_names_2 = ["A", "B"]
        n_half = n_subj // 2
        grp = ["A"] * n_half + ["B"] * (n_subj - n_half)
        # Near-separation: group A → mostly 1, group B → mostly 0
        outcome = ([1] * (n_half - 1) + [0]) + ([0] * (n_subj - n_half - 1) + [1])
        cov = rng.normal(0, 1, size=n_subj).tolist()
        df = pd.DataFrame({"Grp": grp, "Cov": cov, "Outcome": outcome})
        ctx = {"factor_columns": ["Grp"], "dv_columns": ["Outcome"], "group_labels": group_names_2,
               "covariates": ["Cov"], "inferred_test": "logistic_regression", "mode": "single"}
        kwargs = {"group_col": "Grp", "groups": group_names_2, "value_cols": ["Outcome"],
                  "dependent": False, "covariates": ["Cov"]}

    else:  # lmm
        n_subj = int(rng.integers(6, 15))
        within_levels = [f"T{i}" for i in range(int(rng.integers(2, 4)))]
        between_grps = [f"B{i % 2}" for i in range(n_subj)]
        for s in range(n_subj):
            re = float(rng.normal(0, 1))
            for lvl in within_levels:
                rows.append({
                    "Subject": f"S{s}", "Time": lvl, "Between": between_grps[s],
                    "Val": float(rng.normal(0, 1) + re),
                })
        df = pd.DataFrame(rows)
        ctx = {"factor_columns": ["Between"], "dv_columns": ["Val"],
               "group_labels": list(dict.fromkeys(between_grps)),
               "subject_column": "Subject", "between_factors": ["Between"],
               "within_factors": ["Time"], "inferred_test": "lmm", "mode": "single"}
        kwargs = {"group_col": "Between", "groups": list(dict.fromkeys(between_grps)),
                  "value_cols": ["Val"], "dependent": True, "subject_column": "Subject"}

    return df, ctx, kwargs


# Mutations that write numeric values into the DV column. Before any of them we
# normalize the column to float so the mutation is order-independent (a prior
# string-producing mutation like comma_decimals must not break a later one).
_NUMERIC_MUTS = {
    "nan_scatter", "nan_group", "inf", "zero_variance_group", "all_constant",
    "huge_values", "outlier_10sigma", "heavy_skew", "heteroscedastic",
}


def _apply_mutation(df: pd.DataFrame, mut: str, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    val = "Val"
    groups_col = df.columns[0]
    if mut in _NUMERIC_MUTS:
        df[val] = pd.to_numeric(df[val], errors="coerce").astype(float)
    if mut == "nan_scatter":
        idx = rng.choice(df.index, size=max(1, len(df) // 4), replace=False)
        df.loc[idx, val] = np.nan
    elif mut == "nan_group":
        g = rng.choice(df[groups_col].unique())
        df.loc[df[groups_col] == g, val] = np.nan
    elif mut == "inf":
        idx = rng.choice(df.index, size=max(1, len(df) // 8), replace=False)
        df.loc[idx, val] = np.inf * rng.choice([1, -1])
    elif mut == "zero_variance_group":
        g = rng.choice(df[groups_col].unique())
        df.loc[df[groups_col] == g, val] = 42.0
    elif mut == "all_constant":
        df[val] = 7.0
    elif mut == "huge_values":
        df[val] = pd.to_numeric(df[val], errors="coerce") * 1e160
    elif mut == "outlier_10sigma":
        idx = rng.choice(df.index, size=1)
        col = pd.to_numeric(df[val], errors="coerce")
        df.loc[idx, val] = float((col.std() if col.notna().any() else 1.0) * 50 + 1e6)
    elif mut == "unicode_labels":
        df[groups_col] = df[groups_col].astype(str) + rng.choice(["​", " ", "\U0001F9EA"])
    elif mut == "control_chars":
        df[groups_col] = df[groups_col].astype(str) + "\t\n"
    elif mut == "comma_decimals":
        df[val] = df[val].apply(lambda x: str(x).replace(".", ",") if pd.notna(x) else x)
    elif mut == "tiny_groups":
        # keep only 1-2 rows per group
        keep = df.groupby(groups_col, group_keys=False).apply(
            lambda s: s.head(int(rng.integers(1, 3))))
        df = keep.reset_index(drop=True)
    elif mut == "high_cardinality":
        df[groups_col] = [f"u{i}" for i in range(len(df))]
    elif mut == "collinear_covariate" and "Cov" in df.columns:
        # TRUE rank deficiency: make the covariate constant within each factor
        # level, i.e. perfectly collinear with the group dummies -> singular X^T X.
        codes = {g: float(i) for i, g in enumerate(df[groups_col].unique())}
        df["Cov"] = df[groups_col].map(codes).astype(float)
    elif mut == "empty_factor_cell" and {"FacA", "FacB"}.issubset(df.columns):
        # Drop an entire A×B cell -> unbalanced/rank-deficient two-way design.
        a = rng.choice(df["FacA"].unique())
        b = rng.choice(df["FacB"].unique())
        df = df[~((df["FacA"] == a) & (df["FacB"] == b))].reset_index(drop=True)
    elif mut == "cross_level_missing" and {"Subject", "Time"}.issubset(df.columns):
        # Remove random subject×time observations -> unbalanced repeated measures.
        drop = rng.choice(df.index, size=max(1, len(df) // 3), replace=False)
        df = df.drop(index=drop).reset_index(drop=True)
    elif mut == "heavy_skew":
        df[val] = np.exp(pd.to_numeric(df[val], errors="coerce") * 3)
    elif mut == "heteroscedastic":
        col = pd.to_numeric(df[val], errors="coerce")
        for gi, g in enumerate(df[groups_col].unique()):
            mask = df[groups_col] == g
            df.loc[mask, val] = col[mask] * (10 ** gi)
    return df


def build_case(seed: int) -> FuzzCase:
    rng = _rng(seed)
    test_label = TEST_TYPES[int(rng.integers(0, len(TEST_TYPES)))]
    df, ctx, kwargs = _base_design(rng, test_label)

    n_mut = int(rng.integers(0, 4))
    muts = list(rng.choice(MUTATIONS, size=n_mut, replace=False)) if n_mut else ["none"]
    for m in muts:
        df = _apply_mutation(df, m, rng)

    ctx["injected_df"] = df
    kwargs["analysis_context"] = ctx
    # Exercise the full pipeline including matplotlib plot rendering (Agg backend)
    # and the HTML/Excel export — that's where C-level rendering faults can hide.
    kwargs.setdefault("save_plot", True)
    kwargs.setdefault("skip_plots", False)
    kwargs.setdefault("plot_type", "Bar")
    return FuzzCase(seed=seed, test_label=test_label, df=df, mutations=[str(m) for m in muts],
                    analyze_kwargs=kwargs)


def case_to_analyze_kwargs(case: FuzzCase, file_path: str, output_base: str) -> Dict[str, Any]:
    kw = dict(case.analyze_kwargs)
    kw["file_path"] = file_path
    kw["file_name"] = output_base
    n = max(1, len(kw.get("groups") or []))
    kw.setdefault("colors", [_PALETTE[i % len(_PALETTE)] for i in range(n)])
    kw.setdefault("hatches", [_HATCHES[i % len(_HATCHES)] for i in range(n)])
    return kw
