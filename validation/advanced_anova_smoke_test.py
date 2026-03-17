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


def test_mixed_gee_workflow():
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
    assert results.get("model_family") == "NegativeBinomial", (
        f"Erwartet NegativeBinomial, erhalten: {results.get('model_family')}"
    )

    pairwise = results.get("pairwise_comparisons", [])
    assert len(pairwise) > 0, "Keine Post-hoc-Vergleiche generiert."

    posthoc_test = str(results.get("posthoc_test") or "")
    labels = [str(comp.get("test", "")) for comp in pairwise]
    if "Marginal Effects Pairwise Comparisons (Mixed GEE-based)" in posthoc_test:
        assert any("Mixed: Between at fixed Within" in label for label in labels), "Between-Pass fehlt."
        assert any("Mixed: Within at fixed Between" in label for label in labels), "Within-Pass fehlt."
    else:
        assert "Non-parametric pairwise tests" in posthoc_test, (
            f"Unerwarteter Post-hoc-Pfad: {posthoc_test}"
        )
        assert any("Dunn" in label for label in labels), "Between-Fallback via Dunn fehlt."
        assert any("Wilcoxon" in label for label in labels), "Within-Fallback via Wilcoxon fehlt."

    output_path = REPO_ROOT / "validation_test_mixed.xlsx"
    if output_path.exists():
        output_path.unlink()

    ResultsExporter.export_results_to_excel(results, str(output_path))
    assert output_path.exists(), "Excel-Export fehlgeschlagen."

    print(f"Mixed GEE Smoke Test erfolgreich. Datei: {output_path}")


if __name__ == "__main__":
    try:
        test_mixed_gee_workflow()
    except Exception as exc:
        print(f"Validierung fehlgeschlagen: {exc}")
        raise
