# Tier B8: Plot Aesthetics Dialog Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Swap 11 WCAG-failing journal-palette preset colors for same-hue, legible variants
(A2/GD8), and stop `get_config()` from silently dropping the user's filename choice behind a
modal that can fire repeatedly during live preview (A3/GD12).

**Architecture:** Two independent, small fixes in one file
(`src/ui/dialogs/plot_aesthetics_dialog.py`). A2 is a pure data swap (same shape as this
session's `DEFAULT_COLORS` fix in `tests/test_default_colors_contrast.py`). A3 replaces a
blocking `QMessageBox.warning` + early-return with inline sanitization. No shared code between
them — implement and commit as two separate steps.

**Tech Stack:** Python, PyQt5, pytest.

---

### Task 1: A2 (GD8) — swap 11 WCAG-failing journal-palette colors

**Files:**
- Modify: `src/ui/dialogs/plot_aesthetics_dialog.py:388-393` (`ColorsTab.__init__`,
  `self.journal_palettes`)
- Test: `tests/test_journal_palette_contrast.py`

`ColorsTab.journal_palettes` bundles 29 hex values across 4 journal-styled presets
(Nature/Science/NEJM/Lancet). 11 fail WCAG's 3:1 non-text contrast floor against a white plot
background, some badly (`#FFDC91` at 1.32:1). Per
`docs/superpowers/specs/2026-07-08-a2-a3-plot-aesthetics-fixes-design.md`, swap only the 11
failing entries for same-hue, darker/more-saturated variants (all ≥4.5:1) — the 18 passing
entries are untouched.

- [ ] **Step 1: Write the failing test**

```python
"""ColorsTab.journal_palettes must clear WCAG's 3:1 non-text contrast floor against a white
plot background - 11 of 29 bundled hex values (Nature/Science/NEJM/Lancet) currently fail it,
some badly (#FFDC91 at 1.32:1), despite being presented as publication-ready presets.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from PyQt5.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

from ui.dialogs.plot_aesthetics_dialog import ColorsTab


def _relative_luminance(hex_color):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    def lin(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = lin(r), lin(g), lin(b)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast_ratio(hex1, hex2):
    l1, l2 = _relative_luminance(hex1), _relative_luminance(hex2)
    l1, l2 = max(l1, l2), min(l1, l2)
    return (l1 + 0.05) / (l2 + 0.05)


def test_journal_palettes_all_clear_wcag_non_text_floor_against_white():
    tab = ColorsTab()
    try:
        failing = [
            (journal, c, round(_contrast_ratio(c, "#FFFFFF"), 2))
            for journal, colors in tab.journal_palettes.items()
            for c in colors
            if _contrast_ratio(c, "#FFFFFF") < 3.0
        ]
        assert not failing, f"colors failing WCAG's 3:1 non-text floor against white: {failing}"
    finally:
        tab.deleteLater()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/test_journal_palette_contrast.py -v`
Expected: FAIL — `failing` contains 11 entries: `('Nature', '#F28E2B', 2.42)`,
`('Nature', '#76B7B2', 2.29)`, `('Nature', '#EDC948', 1.61)`, `('Nature', '#FF9DA7', 1.98)`,
`('Science', '#56B4E9', 2.31)`, `('Science', '#E69F00', 2.25)`, `('Science', '#999999', 2.85)`,
`('NEJM', '#E18727', 2.73)`, `('NEJM', '#FFDC91', 1.32)`, `('Lancet', '#42B540', 2.65)`,
`('Lancet', '#FDAF91', 1.79)`.

- [ ] **Step 3: Swap the 11 failing colors**

Read the current dict fresh (`grep -n "journal_palettes = " src/ui/dialogs/plot_aesthetics_dialog.py`),
then change:

```python
        self.journal_palettes = {
            'Nature': ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7'],
            'Science': ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#56B4E9', '#E69F00', '#999999'],
            'NEJM': ['#BC3C29', '#0072B5', '#E18727', '#20854E', '#7876B1', '#6F99AD', '#FFDC91'],
            'Lancet': ['#00468B', '#ED0000', '#42B540', '#0099B4', '#925E9F', '#FDAF91', '#AD002A']
        }
```

to:

```python
        self.journal_palettes = {
            'Nature': ['#4E79A7', '#B75C03', '#E15759', '#3F817C', '#59A14F', '#907207', '#B07AA1', '#EB0018'],
            'Science': ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#0F7CBA', '#9C6C00', '#767676'],
            'NEJM': ['#BC3C29', '#0072B5', '#B16310', '#20854E', '#7876B1', '#6F99AD', '#9C6A00'],
            'Lancet': ['#00468B', '#ED0000', '#2C882A', '#0099B4', '#925E9F', '#D73C00', '#AD002A']
        }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/test_journal_palette_contrast.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_journal_palette_contrast.py src/ui/dialogs/plot_aesthetics_dialog.py
git commit -m "fix(gui): swap 11 WCAG-failing journal-palette colors for legible variants"
```

---

### Task 2: A3 (GD12) — sanitize invalid filenames inline instead of dropping them behind a modal

**Files:**
- Modify: `src/ui/dialogs/plot_aesthetics_dialog.py:1868-1875` (`PlotAestheticsDialog.get_config`)
- Test: `tests/test_plot_aesthetics_filename_sanitization.py`

`get_config()` runs on every live-preview tick (`update_preview_immediately` line 1757,
`_do_update_preview` line 1803 — both fire on any settings change in any tab), not just at
final dialog acceptance (line 1915). Its invalid-filename branch currently shows a blocking
`QMessageBox.warning` and returns a config dict missing `file_name`, `create_plot`, and
`dependent` — meaning the modal can fire repeatedly, once per unrelated settings change, for as
long as an invalid character sits in the filename field. Per
`docs/superpowers/specs/2026-07-08-a2-a3-plot-aesthetics-fixes-design.md`, replace this with
inline sanitization: no modal, ever.

- [ ] **Step 1: Write the failing test**

```python
"""get_config()'s invalid-filename branch shows a blocking QMessageBox.warning and returns a
config dict missing file_name/create_plot/dependent - and because get_config() also runs on
every live-preview tick (not just final dialog acceptance), this modal can fire repeatedly for
as long as an invalid character sits in the field. Fix: sanitize inline, no modal, ever.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from PyQt5.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

from ui.dialogs import plot_aesthetics_dialog
from ui.dialogs.plot_aesthetics_dialog import PlotAestheticsDialog


def test_invalid_filename_is_sanitized_not_dropped(monkeypatch):
    warned = []
    monkeypatch.setattr(plot_aesthetics_dialog.QMessageBox, "warning",
                         lambda *a, **k: warned.append(True))

    groups = ["A", "B"]
    samples = {"A": [1.0, 2.0], "B": [3.0, 4.0]}
    dialog = PlotAestheticsDialog(groups=groups, samples=samples, show_export_controls=True)
    try:
        dialog.file_name_edit.setText("bad<>name")
        config = dialog.get_config()

        assert warned == [], "no modal should ever be shown for an invalid filename"
        assert config["file_name"] == "bad__name"
        assert dialog.file_name_edit.text() == "bad__name", (
            "the field should visibly reflect what will actually be used, not silently differ"
        )
        assert config["create_plot"] is True
        assert config["dependent"] is False
    finally:
        dialog.close()


def test_valid_filename_passes_through_unchanged(monkeypatch):
    warned = []
    monkeypatch.setattr(plot_aesthetics_dialog.QMessageBox, "warning",
                         lambda *a, **k: warned.append(True))

    groups = ["A", "B"]
    samples = {"A": [1.0, 2.0], "B": [3.0, 4.0]}
    dialog = PlotAestheticsDialog(groups=groups, samples=samples, show_export_controls=True)
    try:
        dialog.file_name_edit.setText("good_name-1")
        config = dialog.get_config()

        assert warned == []
        assert config["file_name"] == "good_name-1"
        assert dialog.file_name_edit.text() == "good_name-1"
    finally:
        dialog.close()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/test_plot_aesthetics_filename_sanitization.py -v`
Expected: `test_invalid_filename_is_sanitized_not_dropped` FAILS — `warned == [True]` (the modal
fired) and/or `config["file_name"]` raises `KeyError` (the early-return path never sets it).
`test_valid_filename_passes_through_unchanged` PASSES already (valid filenames already work).

- [ ] **Step 3: Fix — sanitize inline, no modal**

Read the current block fresh (`grep -n "Invalid filename" src/ui/dialogs/plot_aesthetics_dialog.py`),
then change:

```python
        if hasattr(self, 'file_name_edit') and self.file_name_edit is not None:
            raw_name = self.file_name_edit.text().strip()
            if raw_name and _re.search(r'[<>:"/\\|?*]', raw_name):
                QMessageBox.warning(self, "Invalid filename",
                    'File name contains invalid characters: < > : " / \\ | ? *\n'
                    'Please use only letters, digits, spaces, hyphens, or underscores.')
                return config
            config['file_name'] = raw_name or None
```

to:

```python
        if hasattr(self, 'file_name_edit') and self.file_name_edit is not None:
            raw_name = self.file_name_edit.text().strip()
            if raw_name and _re.search(r'[<>:"/\\|?*]', raw_name):
                sanitized_name = _re.sub(r'[<>:"/\\|?*]', '_', raw_name)
                self.file_name_edit.setText(sanitized_name)
                raw_name = sanitized_name
            config['file_name'] = raw_name or None
```

(`_re` is the module's existing `import re as _re` at line 7 — no new import needed.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/test_plot_aesthetics_filename_sanitization.py -v`
Expected: PASS (both cases).

- [ ] **Step 5: Check whether `QMessageBox` is still used elsewhere in the file**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && grep -n "QMessageBox" src/ui/dialogs/plot_aesthetics_dialog.py`
Expected: at least one other reference (the import line itself, plus any other dialog/warning
in this file). Do NOT remove the `QMessageBox` import — this file has multiple classes and
other call sites are expected to still use it.

- [ ] **Step 6: Run the full test suite**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/ -q --tb=no`
Expected: same pass count as before this task plus the 2 new tests in this file plus the 1 new
test from Task 1 (the 1 pre-existing unrelated `test_convergence.py::test_convergence_keys`
failure is expected). Pay particular attention to
`tests/test_plot_aesthetics_log_gating.py` (existing tests that construct
`PlotAestheticsDialog` directly) still passing.

- [ ] **Step 7: Commit**

```bash
git add tests/test_plot_aesthetics_filename_sanitization.py src/ui/dialogs/plot_aesthetics_dialog.py
git commit -m "fix(gui): sanitize invalid plot filenames inline instead of dropping them behind a modal"
```

---

## Self-review notes

- **Spec coverage:** A2/GD8 (Task 1), A3/GD12 (Task 2) — both findings assigned to this package
  are covered, matching `docs/superpowers/specs/2026-07-08-a2-a3-plot-aesthetics-fixes-design.md`
  exactly (same 11 hex swaps, same sanitize-inline shape).
- **A2's test constructs a real `ColorsTab`**, not a hand-copied dict, so it verifies the
  actual shipped `journal_palettes` value rather than a duplicate the test maintains itself
  (which could silently drift from the real dict and pass for the wrong reason).
- **A3's test uses the real `PlotAestheticsDialog`** (same construction pattern as the existing
  `tests/test_plot_aesthetics_log_gating.py`), not a hand-built fake — `get_config()` reads from
  8 different tabs' `get_settings()`, so a fake would need to shadow all 8 and risks drifting
  from the real dialog's structure. Mocking only `QMessageBox.warning` isolates the one
  behavior under test (no modal) without needing to fake the rest of the dialog.
- **Task 2's Step 5 explicitly checks `QMessageBox` isn't removed** — this file has many
  classes/dialogs, and an engineer following this plan out of order or over-aggressively
  cleaning up could otherwise break an unrelated warning dialog elsewhere in the same file.
