import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SOURCE_CODE_DIR = REPO_ROOT / "Source_Code"
if str(SOURCE_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_CODE_DIR))

from Source_Code.resultsexporter import ResultsExporter
from Source_Code.statisticaltester import StatisticalTester
from Source_Code.nonparametricanovas import fallback_modern_models


def test_mixed_gee_workflow(tmp_path):
    np.random.seed(42)

    subjects = [f"S{i:02d}" for i in range(1, 25)]
    groups = ["WT", "KO"]
    times = ["T0", "T1", "T2"]

    rows = []
    for subject in subjects:
        genotype = np.random.choice(groups)
        trajectory = [5, 8, 12] if genotype == "WT" else [18, 12, 6]
        for idx, timepoint in enumerate(times):
            lam = trajectory[idx]
            value = np.random.poisson(lam)
            rows.append({
                "Sub": subject,
                "Gen": genotype,
                "Time": timepoint,
                "Val": value,
            })

    df = pd.DataFrame(rows)

    results = StatisticalTester.perform_advanced_test(
        df=df,
        test="mixed_anova",
        dv="Val",
        subject="Sub",
        between=["Gen"],
        within=["Time"],
        alpha=0.05,
        manual_transform="none",
        skip_excel=True,
    )

    assert results.get("fallback_model_used") is True, "Fallback auf modernes Modell wurde nicht ausgelöst."
    assert results.get("model_class") == "GEE", f"Erwartet GEE, erhalten: {results.get('model_class')}"
    assert results.get("model_family") in {"Poisson", "NegativeBinomial"}, (
        f"Erwartet Poisson/NegativeBinomial, erhalten: {results.get('model_family')}"
    )
    assert results.get("cov_struct_used") in {"Autoregressive", "Exchangeable"}, (
        f"Unerwartete Kovarianzstruktur: {results.get('cov_struct_used')}"
    )

    diagnostics = results.get("family_diagnostics", {})
    assert "zero_fraction" in diagnostics, "family_diagnostics.zero_fraction fehlt."
    assert "selection_reason" in diagnostics, "family_diagnostics.selection_reason fehlt."
    assert "pearson_phi" in diagnostics, "family_diagnostics.pearson_phi fehlt."

    pairwise = results.get("pairwise_comparisons", [])
    assert len(pairwise) > 0, "Keine Post-hoc-Vergleiche generiert."

    # Stable labels are required by GUI cards and exports.
    assert results.get("final_test_label"), "final_test_label fehlt im Mixed-Workflow."
    assert results.get("tested_against"), "tested_against fehlt im Mixed-Workflow."
    assert results.get("tested_against") == results.get("final_test_label"), (
        "tested_against und final_test_label sollten konsistent sein."
    )

    posthoc_test = str(results.get("posthoc_test") or "")
    labels = [str(comp.get("test", "")) for comp in pairwise]
    if "Marginal Effects Pairwise Comparisons (Mixed GEE-based)" in posthoc_test:
        assert any("Mixed: Between at fixed Within" in label for label in labels), "Between-Pass fehlt."
        assert any("Mixed: Within at fixed Between" in label for label in labels), "Within-Pass fehlt."
        assert all(comp.get("corrected") is True for comp in pairwise), "Marginaleffects-Vergleiche sollten korrigiert sein."
        assert all(comp.get("correction") == "holm-sidak" for comp in pairwise), (
            "Marginaleffects-Korrekturmethode sollte Holm-Sidak sein."
        )
    else:
        assert "Non-parametric pairwise tests" in posthoc_test, (
            f"Unerwarteter Post-hoc-Pfad: {posthoc_test}"
        )
        assert any("Dunn" in label for label in labels), "Between-Fallback via Dunn fehlt."
        assert any("Wilcoxon" in label for label in labels), "Within-Fallback via Wilcoxon fehlt."

    output_path = tmp_path / "validation_test_mixed.xlsx"

    ResultsExporter.export_results_to_excel(results, str(output_path))
    assert output_path.exists(), "Excel-Export fehlgeschlagen."

    print(f"Mixed GEE Smoke Test erfolgreich. Datei: {output_path}")


def test_gaussian_fallback_uses_robust_covariance():
    np.random.seed(123)

    rows = []
    for a_level in ["A1", "A2"]:
        for b_level in ["B1", "B2"]:
            base = 0.2 if a_level == "A1" else 0.5
            shift = -0.35 if b_level == "B1" else 0.15
            for _ in range(25):
                value = base + shift + np.random.normal(0, 0.2)
                rows.append({"A": a_level, "B": b_level, "Y": value})

    # Ensure non-positive values are present so Gamma fallback is not admissible.
    rows[0]["Y"] = 0.0
    rows[1]["Y"] = -0.1

    df = pd.DataFrame(rows)

    results = fallback_modern_models(
        data=df,
        dependent_var="Y",
        formula="Y ~ C(A) * C(B)",
        design_type="two_way",
        subject_col=None,
    )

    assert results.get("error") is None, f"Unerwarteter Fehler im Gaussian-Fallback: {results.get('error')}"
    assert results.get("model_class") == "GLM", f"Erwartet GLM, erhalten: {results.get('model_class')}"
    assert results.get("model_family") == "Gaussian", f"Erwartet Gaussian, erhalten: {results.get('model_family')}"
    assert results.get("covariance_estimator") == "sandwich_hc3", (
        f"Erwartet sandwich_hc3, erhalten: {results.get('covariance_estimator')}"
    )

    diagnostics = results.get("family_diagnostics", {})
    assert diagnostics.get("selection_reason"), "family_diagnostics.selection_reason fehlt im Gaussian-Fallback."


def test_two_group_paths_expose_stable_labels():
    groups = ["A", "B"]

    independent_samples = {
        "A": [2.1, 2.3, 2.2, 2.0, 2.4, 2.5],
        "B": [3.2, 3.1, 3.3, 3.5, 3.0, 3.4],
    }

    independent_result = StatisticalTester.perform_statistical_test(
        groups=groups,
        transformed_samples=independent_samples,
        original_samples=independent_samples,
        dependent=False,
        test_recommendation="parametric",
        alpha=0.05,
        test_info={"variance_test": {"equal_variance": True}},
    )

    assert independent_result.get("test") in {
        "t-test (independent)",
        "Welch's t-test (unequal variances)",
    }, f"Unerwarteter Testname: {independent_result.get('test')}"
    assert independent_result.get("final_test_label") == independent_result.get("test")
    assert independent_result.get("tested_against") == independent_result.get("final_test_label")

    paired_samples = {
        "A": [10.1, 9.8, 10.4, 9.9, 10.2, 10.0],
        "B": [9.4, 9.2, 9.8, 9.3, 9.7, 9.5],
    }

    paired_result = StatisticalTester.perform_statistical_test(
        groups=groups,
        transformed_samples=paired_samples,
        original_samples=paired_samples,
        dependent=True,
        test_recommendation="parametric",
        alpha=0.05,
        test_info={"variance_test": {"equal_variance": True}},
    )

    assert paired_result.get("test") == "Paired t-test", (
        f"Unerwarteter Testname für dependent=True: {paired_result.get('test')}"
    )
    assert paired_result.get("final_test_label") == paired_result.get("test")
    assert paired_result.get("tested_against") == paired_result.get("final_test_label")


if __name__ == "__main__":
    try:
        test_mixed_gee_workflow()
    except Exception as exc:
        print(f"Validierung fehlgeschlagen: {exc}")
        raise
