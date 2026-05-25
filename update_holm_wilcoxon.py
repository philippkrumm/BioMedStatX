import os
import glob
import re

files_to_check = [
    "src/posthoc_core.py",
    "src/statisticaltester.py",
    "src/decisiontreevisualizer.py",
    "src/resultsexporter.py",
    "src/statistical_testing/posthoc_fallback.py",
    "src/stats_functions.py",
    "src/nonparametricanovas.py",
    "src/statistical_testing/validators.py",
    "src/statistical_testing/engines/posthoc_engine.py",
    "src/ui_dialogs/posthoc_dialog.py"
]

def replace_holm(content):
    content = content.replace("Holm-Sidak", "Holm-Bonferroni")
    content = content.replace("'holm-sidak'", "'holm'")
    content = content.replace('"holm-sidak"', '"holm"')
    content = content.replace('"holm_sidak"', '"holm"')
    content = content.replace("'holm_sidak'", "'holm'")
    content = content.replace("_holm_sidak_correction", "_holm_correction")
    return content

for fpath in files_to_check:
    full_path = "/Users/philippkrumm/Documents/BioMedStatX/" + fpath
    if os.path.exists(full_path):
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
        new_content = replace_holm(content)
        if new_content != content:
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            print(f"Updated {fpath}")

# Now update Wilcoxon in posthoc_core.py
ph_path = "/Users/philippkrumm/Documents/BioMedStatX/src/posthoc_core.py"
with open(ph_path, "r", encoding="utf-8") as f:
    ph_content = f.read()

wilcoxon_old = """            else:
                wstat, p = stats.wilcoxon(x, y)
                stats_list.append(wstat)"""

wilcoxon_new = """            else:
                import warnings
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    wstat, p = stats.wilcoxon(x, y, zero_method='pratt', exact=True if len(x) <= 25 else False)
                    if w:
                        for warn in w:
                            msg = f"Wilcoxon Warning: {str(warn.message)}"
                            if msg not in result.setdefault("warnings", []):
                                result["warnings"].append(msg)
                stats_list.append(wstat)"""

if wilcoxon_old in ph_content:
    ph_content = ph_content.replace(wilcoxon_old, wilcoxon_new)
    with open(ph_path, "w", encoding="utf-8") as f:
        f.write(ph_content)
    print("Updated Wilcoxon in posthoc_core.py")

# Update Wilcoxon in nonparametricanovas.py
np_path = "/Users/philippkrumm/Documents/BioMedStatX/src/nonparametricanovas.py"
with open(np_path, "r", encoding="utf-8") as f:
    np_content = f.read()

np_old_sig = "def _wilcoxon_posthoc_comp(arr1, arr2, label1, label2, alpha):"
np_new_sig = "def _wilcoxon_posthoc_comp(arr1, arr2, label1, label2, alpha, warnings_list=None):"

np_old_w = """    try:
        stat, p_raw = sp_stats.wilcoxon(diffs, alternative='two-sided', zero_method='wilcox')
    except Exception:"""

np_new_w = """    try:
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            stat, p_raw = sp_stats.wilcoxon(diffs, alternative='two-sided', zero_method='pratt', exact=True if len(diffs) <= 25 else False)
            if w and warnings_list is not None:
                for warn in w:
                    msg = f"Wilcoxon Warning ({label1} vs {label2}): {str(warn.message)}"
                    if msg not in warnings_list:
                        warnings_list.append(msg)
    except Exception:"""

np_old_call = """                    comp = _wilcoxon_posthoc_comp(
                        wide[c1].values, wide[c2].values,
                        f"{within_factor}={c1}", f"{within_factor}={c2}", alpha
                    )"""

np_new_call = """                    comp = _wilcoxon_posthoc_comp(
                        wide[c1].values, wide[c2].values,
                        f"{within_factor}={c1}", f"{within_factor}={c2}", alpha, warnings_list
                    )"""

if np_old_sig in np_content:
    np_content = np_content.replace(np_old_sig, np_new_sig)
    np_content = np_content.replace(np_old_w, np_new_w)
    np_content = np_content.replace(np_old_call, np_new_call)
    with open(np_path, "w", encoding="utf-8") as f:
        f.write(np_content)
    print("Updated Wilcoxon in nonparametricanovas.py")

