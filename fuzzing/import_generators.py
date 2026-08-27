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
    # Second wave. A BOM and umlauts are not exotic here -- they are what
    # Excel-DE writes by default and what a German lab names its columns -- and
    # unlike the four above they are NOT allowed to make the read unfaithful:
    # the app is expected to handle both, so a failure is a finding rather than
    # a legitimate refusal.
    "bom_prefix",
    "umlaut_headers",
    # Wide layout only: a subject cell left empty. The pivot refuses it on
    # purpose (missing IDs make pandas drop rows from every groupby that
    # decides repeated-measures structure), so what is checked is that the
    # refusal reaches the user rather than the frame quietly losing rows.
    "missing_subject_id",
)

FILE_FORMATS = ("csv", "xlsx")

# Long format is one measurement per row with a group column; wide format is one
# SUBJECT per row with a column per condition, which the app melts on load. The
# whole pivot path -- detection, the subject heuristic behind it, and the melt --
# had no file-level coverage at all: every run reported "wide-pivoted 0".
LAYOUTS = ("long", "wide")

# Measurement names a lab actually uses. Several carry the letters "id", which
# used to be read as a subject identifier and cost the column its place in the
# mapping entirely -- keeping them here holds that fix under continuous pressure
# rather than only in its own unit test.
DV_NAMES = ("Value", "Expression", "Concentration", "Lipid", "Peptid",
            "Absorbance", "Humidity", "Cell_count")
FACTOR_NAMES = ("Group", "Treatment", "Condition", "Genotype")

# Names the subject heuristic is expected to accept. "Sample ID" is the
# interesting one: "sample" alone is a weak keyword deliberately held to a
# stricter uniqueness rule, and it only passes here because "ID" stands as its
# own word -- the exact distinction the substring fix introduced.
SUBJECT_NAMES = ("Subject", "SubjectID", "Patient ID", "Animal", "Mouse", "Sample ID")

# Condition headers for a wide file. None of them may look subject-like, or the
# detector would find two candidates and refuse -- which would make the case a
# test of the generator rather than of the product.
CONDITION_POOLS = (
    ("Pre", "Post"),
    ("T0", "T24", "T48"),
    ("Baseline", "Week4", "Week12"),
    ("Left", "Right"),
    ("Day0", "Day7", "Day14", "Day28"),
)

# German column names, for the umlaut mutation. Nothing about them should be
# hard -- that is the point of checking.
UMLAUT_DV_NAMES = ("Messwert", "Größe", "Zellzähl", "Trübung")
UMLAUT_FACTOR_NAMES = ("Behandlung", "Gruppe", "Zustand")
UMLAUT_LEVEL_POOLS = (
    ("Kontrolle", "Behandelt"),
    ("Ohne Zusätze", "Mit Zusätzen"),
    ("früh", "spät"),
)

# Wide equivalents. Every subject name here still has to be recognised as one --
# none of them carries a strong keyword, so all three rely on "ID" standing as
# its own word, which is exactly the rule the substring fix introduced.
UMLAUT_SUBJECT_NAMES = ("Präparat ID", "Proband ID", "Tier-ID")
UMLAUT_CONDITION_POOLS = (
    ("Vorher", "Nachher"),
    ("früh", "spät", "später"),
)
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

    # --- wide layout: one subject per row, one column per condition ------------
    layout: str = "long"
    subject_name: Optional[str] = None
    subject_ids: List[str] = field(default_factory=list)
    condition_names: List[str] = field(default_factory=list)
    # Per condition, per subject -- the same nesting pd.melt produces, so the
    # expected long frame can be written down rather than recomputed.
    wide_values: List[List[float]] = field(default_factory=list)

    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def expect_pivot(self) -> bool:
        """Whether the app must melt this file into long format on load.

        Only a wide file that is also readable at all: a header three rows down,
        or a number format the user declared wrongly, gives the detector a frame
        that is not the one on disk, and refusing to pivot that is correct.
        """
        return (self.layout == "wide" and self.expect_faithful_read
                and not self.expect_refusal)

    @property
    def expect_refusal(self) -> bool:
        """A file the app must decline out loud rather than read in part.

        Only when the file is otherwise readable. A blank subject cell in a file
        whose header sits three rows down, or whose number format the user
        declared wrongly, never reaches the subject guard at all: the detector
        declines the misread frame first, the file loads as ordinary data, and
        the mapping refuses it in its own words. Demanding a message box there
        was this oracle asking the app to answer a question it was never asked.
        """
        return "missing_subject_id" in self.mutations and self.expect_faithful_read

    @property
    def melted_conditions(self) -> List[str]:
        """The Condition column pd.melt produces, in row order."""
        return [name for name, column in zip(self.condition_names, self.wide_values)
                for _ in column]

    @property
    def melted_values(self) -> List[float]:
        return [value for column in self.wide_values for value in column]

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

        A BOM and umlaut headers deliberately do NOT relax this. Both are what
        an ordinary German lab export contains, the app is expected to handle
        them, and a case that excused itself from the value checks would be
        testing nothing.
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


def _base_frame(rng: np.random.Generator, big_numbers: bool, umlauts: bool = False):
    dv_pool = UMLAUT_DV_NAMES if umlauts else DV_NAMES
    factor_pool = UMLAUT_FACTOR_NAMES if umlauts else FACTOR_NAMES
    level_pools = UMLAUT_LEVEL_POOLS if umlauts else LEVEL_POOLS
    dv_name = dv_pool[int(rng.integers(0, len(dv_pool)))]
    factor_name = factor_pool[int(rng.integers(0, len(factor_pool)))]
    levels = list(level_pools[int(rng.integers(0, len(level_pools)))])
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


def _cell(value, fmt):
    """One number, written the way the declared format says it should be."""
    if fmt["decimal"] == ",":
        return _format_de(value) if fmt["thousands"] == "." else f"{value:.2f}".replace(".", ",")
    return f"{value:.2f}"


def _encoding(bom: bool) -> str:
    # utf-8-sig is what Excel writes when it saves a CSV, so the BOM is not an
    # exotic case -- it is the default one, and it lands on the first header
    # cell where a naive read turns "Group" into "\ufeffGroup".
    return "utf-8-sig" if bom else "utf-8"


def _write_wide_csv(path, subject_name, subject_ids, condition_names, wide_values,
                    fmt, notes, bom):
    sep = fmt["sep"]
    with open(path, "w", encoding=_encoding(bom), newline="") as handle:
        for note in notes:
            handle.write(note + "\n")
        handle.write(sep.join([subject_name] + list(condition_names)) + "\n")
        for row, subject in enumerate(subject_ids):
            cells = [subject] + [_cell(column[row], fmt) for column in wide_values]
            handle.write(sep.join(cells) + "\n")


def _write_wide_xlsx(path, subject_name, subject_ids, condition_names, wide_values, notes):
    data = {subject_name: subject_ids}
    for name, column in zip(condition_names, wide_values):
        data[name] = column
    frame = pd.DataFrame(data)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="Data", index=False, startrow=len(notes))
        sheet = writer.sheets["Data"]
        for offset, note in enumerate(notes):
            sheet.cell(row=offset + 1, column=1, value=note)


def _write_csv(path, dv_name, factor_name, levels, values, fmt, notes, bom=False):
    sep = fmt["sep"]
    with open(path, "w", encoding=_encoding(bom), newline="") as handle:
        for note in notes:
            handle.write(note + "\n")
        handle.write(sep.join([factor_name, dv_name]) + "\n")
        for level, value in zip(levels, values):
            handle.write(sep.join([level, _cell(value, fmt)]) + "\n")


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


def _subject_ids(rng, n):
    """IDs a lab writes: S01.., M07.., or plain integers as text."""
    style = int(rng.integers(0, 3))
    if style == 0:
        return [f"S{i + 1:02d}" for i in range(n)]
    if style == 1:
        return [f"M{i + 1:02d}" for i in range(n)]
    return [str(i + 1) for i in range(n)]


def _wide_frame(rng, umlauts: bool, big_numbers: bool):
    names = UMLAUT_SUBJECT_NAMES if umlauts else SUBJECT_NAMES
    pools = UMLAUT_CONDITION_POOLS if umlauts else CONDITION_POOLS
    subject_name = names[int(rng.integers(0, len(names)))]
    conditions = list(pools[int(rng.integers(0, len(pools)))])

    # At least three rows, or the detector declines on size alone and the case
    # would prove nothing about the heuristic it was written for.
    n_subjects = int(rng.integers(4, 14))
    subject_ids = _subject_ids(rng, n_subjects)

    centre, spread = (4200.0, 900.0) if big_numbers else (12.0, 3.0)
    wide_values = []
    for index in range(len(conditions)):
        wide_values.append([round(float(rng.normal(centre + index * spread * 0.5,
                                                   spread * 0.25)), 2)
                            for _ in range(n_subjects)])
    return subject_name, subject_ids, conditions, wide_values


def build_case(seed: int, out_dir: str) -> ImportCase:
    from core.csv_import import CSV_FORMAT_PRESETS

    rng = _rng(seed)
    mutations: List[str] = []

    # Roughly one seed in three is a wide file. The pivot path -- detection, the
    # subject heuristic behind it, and the melt -- had no file-level coverage at
    # all before this, and every run reported "wide-pivoted 0".
    layout = "wide" if int(rng.integers(0, 3)) == 0 else "long"

    # A mutation that cannot exist in this layout is not offered: merged group
    # cells need a group column, and a missing subject ID needs a subject one.
    allowed = [m for m in MUTATIONS
               if m != ("missing_subject_id" if layout == "long" else "merged_group_cells")]

    # One seed in three carries no mutation at all: a clean file is the control
    # that proves the harness itself is not the thing failing.
    n_mut = int(rng.integers(0, 3))
    order = rng.permutation(len(allowed))
    candidate = [allowed[i] for i in order[:n_mut]]

    file_format = FILE_FORMATS[int(rng.integers(0, len(FILE_FORMATS)))]
    if any(m in candidate for m in ("german_numbers", "wrong_format_declared", "bom_prefix")):
        file_format = "csv"          # all three are text-file properties
    if "merged_group_cells" in candidate:
        file_format = "xlsx"         # merged cells do not exist in a CSV
        candidate = [m for m in candidate
                     if m not in ("german_numbers", "wrong_format_declared", "bom_prefix")]

    german = "german_numbers" in candidate
    umlauts = "umlaut_headers" in candidate
    bom = "bom_prefix" in candidate

    subject_name = None
    subject_ids: List[str] = []
    conditions: List[str] = []
    wide_values: List[List[float]] = []
    dv_name = factor_name = ""
    levels: List[str] = []
    values: List[float] = []

    if layout == "wide":
        subject_name, subject_ids, conditions, wide_values = _wide_frame(
            rng, umlauts=umlauts, big_numbers=german)
    else:
        dv_name, factor_name, levels, values = _base_frame(
            rng, big_numbers=german, umlauts=umlauts)

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
    if bom and file_format == "csv":
        mutations.append("bom_prefix")
    if umlauts:
        mutations.append("umlaut_headers")

    merge_runs = "merged_group_cells" in candidate
    if merge_runs:
        mutations.append("merged_group_cells")

    if layout == "wide" and "missing_subject_id" in candidate:
        # One subject cell left blank, never the first: the header row still
        # names the column, so the file reads fine and only the pivot's own
        # guard can object.
        subject_ids = list(subject_ids)
        subject_ids[int(rng.integers(1, len(subject_ids)))] = ""
        mutations.append("missing_subject_id")

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"seed{seed}.{file_format}")
    file_levels = list(levels)

    if layout == "wide":
        if file_format == "csv":
            _write_wide_csv(path, subject_name, subject_ids, conditions, wide_values,
                            written_format, notes, bom)
        else:
            _write_wide_xlsx(path, subject_name, subject_ids, conditions, wide_values, notes)
        n_rows, n_cols = len(subject_ids), 1 + len(conditions)
    else:
        if file_format == "csv":
            _write_csv(path, dv_name, factor_name, levels, values, written_format, notes, bom)
        else:
            _write_xlsx(path, dv_name, factor_name, levels, values, notes, merge_runs)
            # After merging, only the first row of each run still holds a label.
            if merge_runs:
                for index in range(1, len(levels)):
                    if levels[index] == levels[index - 1]:
                        file_levels[index] = ""
        n_rows, n_cols = len(values), 2

    return ImportCase(
        seed=seed, file_path=path, file_format=file_format, mutations=mutations,
        dv_name=dv_name, factor_name=factor_name,
        levels=file_levels, values=list(values),
        n_rows=n_rows, n_cols=n_cols,
        declared_format=declared_format, written_format=written_format,
        layout=layout, subject_name=subject_name, subject_ids=list(subject_ids),
        condition_names=list(conditions), wide_values=[list(c) for c in wide_values],
        extra={"header_row": len(notes), "notes": notes, "merged": merge_runs,
               "bom": bom, "umlauts": umlauts},
    )
