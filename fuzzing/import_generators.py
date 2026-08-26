"""Build real spreadsheet files, the way a lab actually exports them.

The analysis fuzzer injects a clean DataFrame straight into ``analyze()`` via
``analysis_context["injected_df"]``, so the import layer (file -> DataFrame) and
the mapping layer (DataFrame -> analysis_context) are never exercised: the
context it hands over is one it built itself, which makes a wrong mapping
impossible by construction. This module produces the other thing -- an actual
``.csv`` or ``.xlsx`` on disk -- and carries the ground truth of what it wrote,
so the question can be "was the file understood", not merely "did a DataFrame
come out".

Each case records what is *in the file*, not what its author meant. A merged
group cell really does leave the following rows blank; that is data the file no
longer contains, and the app cannot invent it. What the app owes the user there
is a faithful read and no phantom group -- which is exactly what the ground
truth lets us ask.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

MUTATIONS = (
    "german_numbers",
    "notes_above_header",
    "merged_group_cells",
    "wrong_format_declared",
)

FILE_FORMATS = ("csv", "xlsx")

# Measurement names a lab actually uses. Several carry the letters "id", which
# used to be read as a subject identifier and cost the column its place in the
# mapping entirely -- keeping them here holds that fix under continuous pressure
# rather than only in its own unit test.
DV_NAMES = ("Value", "Expression", "Concentration", "Lipid", "Peptid",
            "Absorbance", "Humidity", "Cell_count")
FACTOR_NAMES = ("Group", "Treatment", "Condition", "Genotype")
LEVEL_POOLS = (
    ("Ctrl", "Treated"),
    ("WT", "KO"),
    ("Vehicle", "LowDose", "HighDose"),
    ("D0", "D7", "D14", "D21"),
)

_NOTE_LINES = (
    "Experiment 47 - plate B",
    "Exported by LabExport 3.2",
    "Operator: n/a",
)


@dataclass
class ImportCase:
    seed: int
    file_path: str
    file_format: str
    mutations: List[str]

    # --- ground truth: what the file on disk actually contains -----------------
    dv_name: str
    factor_name: str
    levels: List[str]                 # per row, "" where the file has a blank
    values: List[float]               # per row, in row order
    n_rows: int
    n_cols: int

    # --- what the user will answer in the CSV number-format dialog -------------
    declared_format: Optional[Dict[str, Any]] = None
    written_format: Optional[Dict[str, Any]] = None

    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def header_row(self) -> int:
        return int(self.extra.get("header_row", 0))

    @property
    def format_declared_matches(self) -> bool:
        """Did the user declare the format the file was actually written in?"""
        if self.file_format != "csv":
            return True          # Excel stores numbers as floats; nothing to declare
        return self.declared_format == self.written_format

    @property
    def expect_faithful_read(self) -> bool:
        """Whether the app is expected to reproduce the file exactly.

        When this is False the app is *allowed* to misread -- the file is not
        one it can be expected to parse blind -- but it still owes the user a
        visible failure rather than a plausible wrong answer.
        """
        return self.header_row == 0 and self.format_declared_matches

    @property
    def distinct_levels(self) -> List[str]:
        seen = []
        for level in self.levels:
            if level and level not in seen:
                seen.append(level)
        return seen


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed + 0x10A9)


def _format_de(value: float) -> str:
    """1234.56 -> '1.234,56' — dot thousands, comma decimal, as Excel-DE writes."""
    whole, _, frac = f"{value:,.2f}".partition(".")
    return whole.replace(",", ".") + "," + frac


def _base_frame(rng: np.random.Generator, big_numbers: bool):
    dv_name = DV_NAMES[int(rng.integers(0, len(DV_NAMES)))]
    factor_name = FACTOR_NAMES[int(rng.integers(0, len(FACTOR_NAMES)))]
    levels = list(LEVEL_POOLS[int(rng.integers(0, len(LEVEL_POOLS)))])
    n_per = int(rng.integers(4, 12))

    # Values above 1000 are what makes a thousands separator appear at all; a
    # generator that only ever emits 12.5 would exercise the decimal mark and
    # silently skip the grouping half of the contract.
    centre, spread = (4200.0, 900.0) if big_numbers else (12.0, 3.0)

    rows_level, rows_value = [], []
    for index, level in enumerate(levels):
        for _ in range(n_per):
            rows_level.append(level)
            rows_value.append(round(float(rng.normal(centre + index * spread * 0.6, spread * 0.25)), 2))
    return dv_name, factor_name, rows_level, rows_value


def _write_csv(path, dv_name, factor_name, levels, values, fmt, notes):
    sep = fmt["sep"]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        for note in notes:
            handle.write(note + "\n")
        handle.write(sep.join([factor_name, dv_name]) + "\n")
        for level, value in zip(levels, values):
            if fmt["decimal"] == ",":
                cell = _format_de(value) if fmt["thousands"] == "." else f"{value:.2f}".replace(".", ",")
            else:
                cell = f"{value:.2f}"
            handle.write(sep.join([level, cell]) + "\n")


def _write_xlsx(path, dv_name, factor_name, levels, values, notes, merge_runs):
    frame = pd.DataFrame({factor_name: levels, dv_name: values})
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="Data", index=False,
                       startrow=len(notes))
        sheet = writer.sheets["Data"]
        for offset, note in enumerate(notes):
            sheet.cell(row=offset + 1, column=1, value=note)
        if merge_runs:
            # A real merged cell: the label is stored once and the rows beneath
            # it are genuinely empty in the file. Reading it back gives exactly
            # the blanks the generator recorded as ground truth.
            first_data_row = len(notes) + 2      # 1-based, past the header
            start = 0
            for index in range(1, len(levels) + 1):
                if index == len(levels) or levels[index] != levels[start]:
                    if index - start > 1:
                        sheet.merge_cells(start_row=first_data_row + start, start_column=1,
                                          end_row=first_data_row + index - 1, end_column=1)
                    start = index


def build_case(seed: int, out_dir: str) -> ImportCase:
    from core.csv_import import CSV_FORMAT_PRESETS

    rng = _rng(seed)
    mutations: List[str] = []

    # One mutation in three seeds is none at all: a clean file is the control
    # that proves the harness itself is not the thing failing.
    n_mut = int(rng.integers(0, 3))
    order = rng.permutation(len(MUTATIONS))
    candidate = [MUTATIONS[i] for i in order[:n_mut]]

    file_format = FILE_FORMATS[int(rng.integers(0, len(FILE_FORMATS)))]
    if "german_numbers" in candidate or "wrong_format_declared" in candidate:
        file_format = "csv"          # both are number-format mutations
    if "merged_group_cells" in candidate:
        file_format = "xlsx"         # merged cells do not exist in a CSV
        candidate = [m for m in candidate
                     if m not in ("german_numbers", "wrong_format_declared")]

    german = "german_numbers" in candidate
    dv_name, factor_name, levels, values = _base_frame(rng, big_numbers=german)

    notes = []
    if "notes_above_header" in candidate:
        notes = list(_NOTE_LINES[:int(rng.integers(1, 4))])
        mutations.append("notes_above_header")

    written_format = declared_format = None
    if file_format == "csv":
        written_format = dict(CSV_FORMAT_PRESETS["german" if german else "international"])
        written_format.pop("label", None)
        declared_format = dict(written_format)
        if german:
            mutations.append("german_numbers")
        if "wrong_format_declared" in candidate:
            # The user reaches for the other preset. This is the realistic
            # mistake: there is deliberately no autodetect, so the declaration
            # is the user's, and the dialog's preview is the app's answer to it.
            other = "international" if german else "german"
            declared_format = dict(CSV_FORMAT_PRESETS[other])
            declared_format.pop("label", None)
            mutations.append("wrong_format_declared")

    merge_runs = "merged_group_cells" in candidate
    if merge_runs:
        mutations.append("merged_group_cells")

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"seed{seed}.{file_format}")
    if file_format == "csv":
        _write_csv(path, dv_name, factor_name, levels, values, written_format, notes)
        file_levels = list(levels)
    else:
        _write_xlsx(path, dv_name, factor_name, levels, values, notes, merge_runs)
        # After merging, only the first row of each run still holds a label.
        file_levels = list(levels)
        if merge_runs:
            for index in range(1, len(levels)):
                if levels[index] == levels[index - 1]:
                    file_levels[index] = ""

    return ImportCase(
        seed=seed, file_path=path, file_format=file_format, mutations=mutations,
        dv_name=dv_name, factor_name=factor_name,
        levels=file_levels, values=list(values),
        n_rows=len(values), n_cols=2,
        declared_format=declared_format, written_format=written_format,
        extra={"header_row": len(notes), "notes": notes, "merged": merge_runs},
    )
