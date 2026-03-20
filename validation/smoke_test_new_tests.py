"""
Smoke-Test für die drei neuen nonparametrischen Tests:
  - perform_friedman_test
  - perform_freedman_lane_test
  - perform_brunner_langer_ats

Kein GUI, kein R, keine externen Vergleichswerte.
Prüft nur: Import ok, Rückgabe-Struktur korrekt, keine Exceptions.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "Source_Code"
for p in [str(ROOT), str(SRC)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ──────────────────────────────────────────────
# Import
# ──────────────────────────────────────────────
print("Importiere nonparametricanovas …", end=" ", flush=True)
from nonparametricanovas import (
    perform_friedman_test,
    perform_freedman_lane_test,
    perform_brunner_langer_ats,
)
print("OK")


# ──────────────────────────────────────────────
# Hilfsfunktionen
# ──────────────────────────────────────────────
REQUIRED_KEYS = {"model_class", "anova_table", "factors"}

def _check_result(name, res):
    missing = REQUIRED_KEYS - res.keys()
    if missing:
        raise AssertionError(f"{name}: fehlende Keys {missing}")
    if "error" in res and res["error"]:
        raise AssertionError(f"{name}: Fehler im Ergebnis: {res['error']}")

    table = res["anova_table"]
    # anova_table kann ein DataFrame oder eine Liste sein
    if hasattr(table, "empty"):
        # pandas DataFrame
        if table is None or (hasattr(table, "empty") and table.empty):
            raise AssertionError(f"{name}: anova_table ist leer (DataFrame)")
        cols = [c.lower() for c in table.columns]
        has_p = any(c.startswith("p") for c in cols)
        if not has_p:
            raise AssertionError(f"{name}: ANOVA-Tabelle ohne p-Spalte: {list(table.columns)}")
        print(f"  model_class : {res.get('model_class')}")
        print(f"  ANOVA-Zeilen: {len(table)}")
        p_cols = [c for c in table.columns if c.lower().startswith("p")]
        print(f"  p-Spalten   : {p_cols} -> {table[p_cols].values.tolist()}")
    elif isinstance(table, list):
        if len(table) == 0:
            raise AssertionError(f"{name}: anova_table ist leer (Liste)")
        for row in table:
            keys = set(row.keys())
            has_p = any(k.lower().startswith("p") for k in keys)
            if not has_p:
                raise AssertionError(f"{name}: ANOVA-Zeile ohne p-Wert: {row}")
        print(f"  model_class : {res.get('model_class')}")
        print(f"  ANOVA-Zeilen: {len(table)}")
    else:
        raise AssertionError(f"{name}: anova_table hat unerwarteten Typ: {type(table)}")

    if res.get("pairwise_comparisons"):
        print(f"  Post-hoc    : {len(res['pairwise_comparisons'])} Vergleiche")


# ──────────────────────────────────────────────
# Test 1: Friedman
# ──────────────────────────────────────────────
def test_friedman():
    np.random.seed(0)
    n_subjects = 12
    df = pd.DataFrame({
        "subject": np.repeat([f"S{i}" for i in range(n_subjects)], 4),
        "time":    np.tile(["T1", "T2", "T3", "T4"], n_subjects),
        "score":   np.random.normal(loc=np.tile([5, 7, 6, 9], n_subjects), scale=1.0),
    })
    res = perform_friedman_test(
        data=df,
        dv="score",
        within_factor="time",
        subject_col="subject",
        alpha=0.05,
    )
    assert res.get("model_class") == "Friedman", f"Erwartet 'Friedman', erhalten: {res.get('model_class')}"
    _check_result("Friedman", res)


# ──────────────────────────────────────────────
# Test 2: Freedman-Lane Permutation
# ──────────────────────────────────────────────
def test_freedman_lane():
    np.random.seed(1)
    df = pd.DataFrame({
        "group":   np.repeat(["G1", "G2"], 20),
        "treat":   np.tile(["A", "B"], 20),
        "outcome": np.random.normal(size=40),
    })
    res = perform_freedman_lane_test(
        data=df,
        dv="outcome",
        factor_a="group",
        factor_b="treat",
        alpha=0.05,
        n_permutations=500,  # klein für Speed
        seed=42,
    )
    assert res.get("model_class") == "Freedman-Lane Permutation", (
        f"Erwartet 'Freedman-Lane Permutation', erhalten: {res.get('model_class')}"
    )
    _check_result("Freedman-Lane", res)

    # p-Werte müssen im [0, 1]-Intervall liegen
    tbl = res["anova_table"]
    p_cols = [c for c in tbl.columns if c.lower().startswith("p")]
    for col in p_cols:
        for val in tbl[col].dropna():
            assert 0.0 <= float(val) <= 1.0, f"Ungültiger p-Wert in {col}: {val}"


# ──────────────────────────────────────────────
# Test 3: Brunner-Langer ATS
# ──────────────────────────────────────────────
def test_brunner_langer():
    np.random.seed(2)
    subjects_m = [f"M{i}" for i in range(8)]
    subjects_f = [f"F{i}" for i in range(8)]
    times = ["T1", "T2", "T3"]
    rows = []
    for s in subjects_m:
        for t in times:
            rows.append({"subject": s, "sex": "Male", "age": t,
                          "distance": np.random.normal(24, 2)})
    for s in subjects_f:
        for t in times:
            rows.append({"subject": s, "sex": "Female", "age": t,
                          "distance": np.random.normal(21, 2)})
    df = pd.DataFrame(rows)

    res = perform_brunner_langer_ats(
        data=df,
        dv="distance",
        between_factor="sex",
        within_factor="age",
        subject_col="subject",
        alpha=0.05,
    )
    assert res.get("model_class") == "Brunner-Langer ATS", (
        f"Erwartet 'Brunner-Langer ATS', erhalten: {res.get('model_class')}"
    )
    _check_result("Brunner-Langer", res)

    # Muss Between/Within/Interaction enthalten (DataFrame)
    tbl = res["anova_table"]
    # Suche nach Effekt-Spalte: "Source", "effect", "Effect", oder Index
    source_col = next(
        (c for c in tbl.columns if c.lower() in ("source", "effect", "factor")), None
    )
    if source_col:
        labels = [str(v).lower() for v in tbl[source_col]]
    else:
        labels = [str(v).lower() for v in tbl.index]
    assert any("between" in l or "sex" in l for l in labels), \
        f"Kein Between-Effekt in ANOVA-Tabelle: {labels}"
    assert any("within" in l or "age" in l or "time" in l for l in labels), \
        f"Kein Within-Effekt in ANOVA-Tabelle: {labels}"


# -----------------------------------------------
# Runner
# -----------------------------------------------
TESTS = [
    ("Friedman",           test_friedman),
    ("Freedman-Lane",      test_freedman_lane),
    ("Brunner-Langer ATS", test_brunner_langer),
]

if __name__ == "__main__":
    run_results = []
    for name, fn in TESTS:
        print(f"\n{'-'*50}")
        print(f"  {name}")
        print(f"{'-'*50}")
        try:
            fn()
            print("  -> PASS")
            run_results.append((name, True, None))
        except Exception as e:
            print(f"  -> FAIL  {e}")
            run_results.append((name, False, str(e)))

    print(f"\n{'='*50}")
    print("  Ergebnis-Zusammenfassung")
    print(f"{'='*50}")
    all_passed = True
    for name, ok, err in run_results:
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name}")
        if err:
            print(f"         {err}")
            all_passed = False

    if all_passed:
        print("\nAlle Tests bestanden.\n")
        sys.exit(0)
    else:
        print("\nMindestens ein Test fehlgeschlagen.\n")
        sys.exit(1)

