"""Generate frozen golden-reference values for the gated correctness suite.

Run manually (not at test time):  python validation/generate_golden.py

References are computed with scipy / pingouin (v1 oracle — themselves validated
against R). The output JSON (tests/golden/references.json) is consumed by
tests/test_golden_core.py, which runs the APP and compares to these frozen
values. Keeping the oracle out of the test run makes the gate fast and free of
heavy/optional dependencies, and freezes the numbers so any regression in the
app's routing, corrections, or degrees of freedom is caught.

Schema is source-agnostic so later R dumps (nparLD ATS, emmeans marginal means)
slot in without changing the test logic: every numeric field is optional and
compared only when present.
"""
import json
import os

import numpy as np
import scipy.stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "tests", "golden", "references.json")

MIN_N_SMALL = 20  # mirror validators.MIN_N_SMALL for MWU exact/asymptotic choice


def _ds():
    """Deterministic datasets keyed by name."""
    rng = np.random.default_rng(20240617)
    return {
        "two_a": list(rng.normal(0.0, 1.0, 12)),
        "two_b": list(rng.normal(1.3, 1.4, 12)),
        "paired_a": list(rng.normal(5.0, 1.0, 14)),
        "paired_b": None,  # built below as a + shift + noise
        "g3_a": list(rng.normal(0.0, 1.0, 11)),
        "g3_b": list(rng.normal(1.0, 1.2, 10)),
        "g3_c": list(rng.normal(2.2, 0.9, 9)),
    }


def build_cases():
    d = _ds()
    rng = np.random.default_rng(7)
    a = np.array(d["paired_a"])
    b = list(a + 0.8 + rng.normal(0, 0.6, a.size))  # correlated pair, real diff
    d["paired_b"] = b

    cases = []

    # Welch t-test (independent) — app's robust default for 2 groups
    s, p = st.ttest_ind(d["two_a"], d["two_b"], equal_var=False)
    cases.append({
        "id": "welch_ttest_independent",
        "invocation": {"dependent": False, "test_recommendation": "welch"},
        "expected_test_contains": "welch",
        "samples": {"A": d["two_a"], "B": d["two_b"]},
        "expected": {"statistic": float(s), "p_value": float(p),
                     "tol": {"statistic": 1e-6, "p_value": 1e-6}},
        "reference_source": "scipy.stats.ttest_ind(equal_var=False)",
    })

    # Mann-Whitney U
    n = len(d["two_a"]) + len(d["two_b"])
    method = "exact" if n < MIN_N_SMALL else "asymptotic"
    s, p = st.mannwhitneyu(d["two_a"], d["two_b"], alternative="two-sided", method=method)
    cases.append({
        "id": "mann_whitney_u",
        "invocation": {"dependent": False, "test_recommendation": "non_parametric"},
        "expected_test_contains": "mann-whitney",
        "samples": {"A": d["two_a"], "B": d["two_b"]},
        "expected": {"statistic": float(s), "p_value": float(p),
                     "tol": {"statistic": 1e-6, "p_value": 1e-4}},
        "reference_source": f"scipy.stats.mannwhitneyu(method={method})",
    })

    # Paired t-test
    s, p = st.ttest_rel(d["paired_a"], d["paired_b"])
    cases.append({
        "id": "paired_ttest",
        "invocation": {"dependent": True, "test_recommendation": "parametric"},
        "expected_test_contains": "paired",
        "samples": {"A": d["paired_a"], "B": d["paired_b"]},
        "expected": {"statistic": float(s), "p_value": float(p),
                     "tol": {"statistic": 1e-6, "p_value": 1e-6}},
        "reference_source": "scipy.stats.ttest_rel",
    })

    # Wilcoxon signed-rank
    nlen = len(d["paired_a"])
    wmethod = "exact" if nlen <= 25 else "approx"
    s, p = st.wilcoxon(d["paired_a"], d["paired_b"], zero_method="pratt", method=wmethod)
    cases.append({
        "id": "wilcoxon_signed_rank",
        "invocation": {"dependent": True, "test_recommendation": "non_parametric"},
        "expected_test_contains": "wilcoxon",
        "samples": {"A": d["paired_a"], "B": d["paired_b"]},
        "expected": {"statistic": float(s), "p_value": float(p),
                     "tol": {"statistic": 1e-6, "p_value": 1e-4}},
        "reference_source": "scipy.stats.wilcoxon(zero_method=pratt)",
    })

    # One-way ANOVA
    groups3 = [d["g3_a"], d["g3_b"], d["g3_c"]]
    s, p = st.f_oneway(*groups3)
    k = len(groups3)
    ntot = sum(len(g) for g in groups3)
    cases.append({
        "id": "one_way_anova",
        "invocation": {"dependent": False, "test_recommendation": "parametric"},
        "expected_test_contains": "anova",
        "samples": {"A": d["g3_a"], "B": d["g3_b"], "C": d["g3_c"]},
        "expected": {"statistic": float(s), "p_value": float(p),
                     "df_num": float(k - 1), "df_den": float(ntot - k),
                     "tol": {"statistic": 1e-5, "p_value": 1e-5}},
        "reference_source": "scipy.stats.f_oneway",
    })

    # Welch ANOVA (pingouin)
    try:
        import pandas as pd
        import pingouin as pg
        long = pd.DataFrame(
            {"Value": d["g3_a"] + d["g3_b"] + d["g3_c"],
             "Group": ["A"] * len(d["g3_a"]) + ["B"] * len(d["g3_b"]) + ["C"] * len(d["g3_c"])})
        wa = pg.welch_anova(data=long, dv="Value", between="Group")
        p_col = "p_unc" if "p_unc" in wa.columns else "p-unc"
        cases.append({
            "id": "welch_anova",
            "invocation": {"dependent": False, "test_recommendation": "welch"},
            "expected_test_contains": "welch",
            "samples": {"A": d["g3_a"], "B": d["g3_b"], "C": d["g3_c"]},
            "expected": {"statistic": float(wa["F"].iloc[0]), "p_value": float(wa[p_col].iloc[0]),
                         "df_num": float(wa["ddof1"].iloc[0]), "df_den": float(wa["ddof2"].iloc[0]),
                         "tol": {"statistic": 1e-4, "p_value": 1e-4}},
            "reference_source": "pingouin.welch_anova",
        })
    except Exception as exc:
        print(f"WARNING: skipping welch_anova reference ({exc})")

    # Kruskal-Wallis
    s, p = st.kruskal(*groups3)
    cases.append({
        "id": "kruskal_wallis",
        "invocation": {"dependent": False, "test_recommendation": "non_parametric"},
        "expected_test_contains": "kruskal",
        "samples": {"A": d["g3_a"], "B": d["g3_b"], "C": d["g3_c"]},
        "expected": {"statistic": float(s), "p_value": float(p),
                     "tol": {"statistic": 1e-5, "p_value": 1e-5}},
        "reference_source": "scipy.stats.kruskal",
    })

    return {"schema_version": 1, "oracle": "scipy/pingouin (v1)", "cases": cases}


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    data = build_cases()
    with open(OUT, "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"Wrote {len(data['cases'])} golden cases -> {OUT}")


if __name__ == "__main__":
    main()
