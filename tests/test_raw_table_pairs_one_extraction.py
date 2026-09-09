"""The raw data vault prints raw and transformed side by side. They must pair.

``_build_raw_data_table`` emits one row per index: ``raw_data[g][i]`` next to
``raw_data_transformed[g][i]``. That row is a claim -- "this measurement became
that value" -- and it only holds if both lists came out of the same extraction.

They did not. The advanced pipeline builds both from its own samples and stores
them together, correctly paired; ``AnalysisManager.analyze`` then overwrote the
raw half with a separately-extracted copy of the same data in a different order.
Same values, same length, same groups, so every downstream summary (means, SD,
Q-Q plot, distribution charts) stayed right -- only the row-wise pairing was
wrong, which is the one thing nothing looked at.

Measured on a repeated-measures run with a real log10: 24 of 28 printed rows
showed one subject's raw value beside another subject's transformed value. A
raw value below 1 was printed next to a positive log10, which is arithmetically
impossible and still looked plausible enough to read past.

The oracle here is arithmetic, not a re-ask of the gate that produced the data:
log10 of the printed raw value must equal the printed transformed value.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

# Root conftest.py puts src/ on sys.path and forces headless Qt.
import analysis.stats_functions as stats_functions_module
import analysis.statisticaltester as statisticaltester_module
from analysis.analysis_core import AnalysisManager
from export.report_summaries import _SummariesMixin


class _Dialogs:
    """Every dialog this run can reach, answered the way a user would.

    Installed over the module attribute rather than onto the imported class,
    because that is what the product looks up: statistical_testing.dialog_access
    resolves the manager through ``analysis.statisticaltester.UIDialogManager``
    at call time. validation/conftest.py replaces exactly that attribute with a
    session-scoped mock answering "skip", so patching the class this module
    imported would set attributes on an object nothing consults -- the run would
    silently take the untransformed route and land on a Friedman test, and the
    checks below would have no transformed column to judge.
    """

    @staticmethod
    def select_transformation_dialog(*args, **kwargs):
        return "log10"

    @staticmethod
    def select_posthoc_test_dialog(*args, **kwargs):
        return "emm_mvt"

    @staticmethod
    def select_nonparametric_posthoc_dialog(*args, **kwargs):
        return None

    # A significant omnibus walks into the post-hoc branch, which builds real Qt
    # dialogs. Without a QApplication that aborts the interpreter rather than
    # raising, so every dialog reachable from here needs an answer, not just the
    # ones this design is expected to use.
    @staticmethod
    def select_control_group_dialog(*args, **kwargs):
        return None

    @staticmethod
    def select_custom_pairs_dialog(groups=None, parent=None):
        return []


def _lognormal_rm_frame(n_subjects: int = 16, levels: int = 2) -> pd.DataFrame:
    """Right-skewed, strictly positive repeated measures.

    Skew so the normality check fails and the transformation dialog is actually
    offered -- on normal data no transformation is proposed and the column under
    test never appears. Strictly positive so log10 is defined for every value,
    which is what makes the arithmetic oracle below usable.
    """
    rng = np.random.default_rng(20260827)
    offsets = rng.normal(0, 1.0, size=n_subjects)
    rows = []
    for s in range(n_subjects):
        for li in range(levels):
            value = math.exp(0.8 * li + offsets[s] + float(rng.normal(0, 1)))
            rows.append({"Subject": f"S{s}", "Time": f"T{li}", "Val": value})
    return pd.DataFrame(rows)


@pytest.fixture
def rm_result(tmp_path, monkeypatch):
    """One repeated-measures analysis with log10 actually applied."""
    mp = pytest.MonkeyPatch()
    try:
        # Patched with undo rather than assigned, so nothing leaks onward.
        mp.setattr(statisticaltester_module, "UIDialogManager", _Dialogs)
        mp.setattr(stats_functions_module, "UIDialogManager", _Dialogs)
        from ui.dialogs import comparison_selection_dialog as csd

        class _AllPairs:
            Accepted = 1

            def __init__(self, all_pairs, *args, **kwargs):
                self._pairs = list(all_pairs or [])

            def exec_(self):
                return self.Accepted

            def get_selected_comparisons(self):
                return self._pairs

        mp.setattr(csd, "ComparisonSelectionDialog", _AllPairs, raising=False)
        try:
            from PyQt5.QtWidgets import QDialog
            mp.setattr(QDialog, "exec_", lambda self, *a, **k: 0, raising=False)
        except Exception:
            pass

        levels = ["T0", "T1"]
        df = _lognormal_rm_frame()
        context = {
            "factor_columns": ["Time"],
            "dv_columns": ["Val"],
            "group_labels": levels,
            "subject_column": "Subject",
            "mode": "single",
            "inferred_test": "repeated_measures_anova",
            "within_factors": ["Time"],
            "injected_df": df,
        }
        dummy = tmp_path / "dummy.xlsx"
        pd.DataFrame({"a": [1]}).to_excel(dummy, index=False)
        result = AnalysisManager.analyze(
            file_path=str(dummy),
            file_name=str(tmp_path / "out"),
            group_col="Time",
            groups=levels,
            value_cols=["Val"],
            dependent=True,
            subject_column="Subject",
            analysis_context=context,
        )
    finally:
        mp.undo()

    # Not a skip. If the fixture stops reaching the transformation the tests
    # below would pass by testing nothing, which is the failure this file
    # exists to prevent -- so say so loudly instead.
    assert result.get("raw_data_transformed") or result.get("transformed_data"), (
        "the fixture no longer reaches the transformation branch, so nothing "
        "below is actually checking a paired column; adjust the design until "
        "log10 is applied again"
    )
    return result


def test_each_printed_row_pairs_a_value_with_its_own_transform(rm_result):
    raw = rm_result.get("raw_data") or {}
    transformed = (rm_result.get("raw_data_transformed")
                   or rm_result.get("transformed_data") or {})
    assert raw, "the run produced no raw data to check"

    for group, values in raw.items():
        got = [float(v) for v in transformed.get(group, [])]
        originals = [float(v) for v in values]
        assert len(got) == len(originals), (
            f"group {group}: {len(originals)} raw values but {len(got)} "
            f"transformed ones -- the table would pad the difference with N/A"
        )
        for index, (original, actual) in enumerate(zip(originals, got)):
            assert original > 0, "fixture must stay strictly positive for log10"
            assert actual == pytest.approx(math.log10(original), abs=1e-9), (
                f"group {group} row {index}: the report prints {original:.6g} "
                f"next to {actual:.6g}, but log10({original:.6g}) is "
                f"{math.log10(original):.6g}"
            )


def test_the_rendered_table_never_shows_an_impossible_log(rm_result):
    """Read the product's own table, not the dict it was built from.

    A raw value below 1 has a negative log10. A row that pairs one with a
    positive transformed value is wrong on arithmetic alone, with no reference
    to how either number was produced.
    """
    table = _SummariesMixin._build_raw_data_table(rm_result)
    assert table["has_transformed"], "no transformed column was rendered to check"

    impossible = []
    for row in table["rows"]:
        raw_text = row["raw_value"]
        transformed_text = row["transformed_value"]
        if "N/A" in (raw_text, transformed_text):
            continue
        raw_value = float(raw_text)
        transformed_value = float(transformed_text)
        if (raw_value < 1.0) != (transformed_value < 0.0):
            impossible.append((raw_text, transformed_text))

    assert not impossible, (
        f"{len(impossible)} of {len(table['rows'])} printed rows pair a raw "
        f"value with a log10 of the wrong sign, e.g. {impossible[:3]}"
    )


def test_the_pair_survives_as_a_permutation_of_the_right_answers(rm_result):
    """Guards against a fix that silences the column instead of ordering it.

    The wrong ordering held all the correct values, just in the wrong rows. A
    "fix" that dropped or blanked the transformed column would pass the two
    tests above while telling the user less than before, so assert the values
    are all still there.
    """
    raw = rm_result.get("raw_data") or {}
    transformed = (rm_result.get("raw_data_transformed")
                   or rm_result.get("transformed_data") or {})
    for group, values in raw.items():
        expected = sorted(math.log10(float(v)) for v in values)
        got = sorted(float(v) for v in transformed.get(group, []))
        assert got == pytest.approx(expected, abs=1e-9), (
            f"group {group}: the transformed column no longer carries every "
            f"transformed value"
        )
