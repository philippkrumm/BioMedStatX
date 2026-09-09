# AUDIT: docs/HowTo.md — GUI Parity @ fce568d

**Date:** 2026-08-17 · **Branch:** `feature/advanced-stats-automation` · **Scope:** `docs/HowTo.md`
(559 lines, 22 sections) against the live GUI source. **Report-only; no fixes in this pass.**
**Method:** every concrete GUI claim (button text, dialog name, menu entry, status message, tab
label) verified by a **region-read of the constructing code**, not a keyword grep — a grep that
misses the *claimed* string does not prove the control is absent (it may exist under a different
string), and a grep that hits does not prove the surrounding claim is right. This is the GD4 lesson
from the Round-3 audit applied end to end.

`HowTo.md` had never been in the scope of a parity audit before. Extra attention went to the
sections written **before** today's Filter-Bucket addition (`68f89e2`), since those had the longest
window to drift.

## Verdict

`HowTo.md` is largely accurate — buttons, menus, the six mapping buckets, the five Plot-Designer
tabs, the mapping status strings, and the statistical-methods reference all check out against source.
**One section (6, Export Settings) describes controls that do not exist** and is the one real
defect. Two smaller quoted-label mismatches and one "representative-not-complete" list round out the
findings. Nothing blocks the release; §6 is worth a rewrite because it tells a first-time user to
look for a field and a button that aren't there.

## What I mechanically verified (not eyeballed)

| Claim (HowTo) | Verified against | Result |
|---|---|---|
| §2 "Load Data File" button | `pipeline:157` `QPushButton("Load Data File")` | ✓ exact |
| §2 "Worksheet" / "Table Preview" labels | `pipeline:177,203` | ✓ exact |
| §2 "Select Data Ranges…" | `pipeline:185` `QPushButton("Select Data Ranges...")` | ✓ (ellipsis vs `...`, cosmetic) |
| §3 six buckets (Dep. Var., Factor 1, Factor 2, Subject ID, Covariates, Filter) | bucket construction in `pipeline` | ✓ |
| §3 "Select Groups For Analysis" | `pipeline:348` | ✓ exact |
| §3 five mapping status strings | `pipeline:822,843,855,908,1891` | ✓ all five exact — but list is a **subset** (see HT4) |
| §5/§14 "Start Auto Analysis" | `pipeline:435` | ✓ exact |
| §6 "output file name" field + "Browse…" beside it | grep for a filename `QLineEdit`/`QPushButton("Browse")` on the main window; `getSaveFileName` at `pipeline:1880` | **✗ neither exists** — see HT1 |
| §8 "Select All / Deselect All / Select None" on group + pairwise dialogs | read `comparison_selection_dialog.py:62-72`, `statistical_analyzer_dialogs.py:90-101,484` | **✗ "Select None" is outlier-dialog-only** — see HT2 |
| §9 Plot-Designer five tabs (Plot/Axes/Style/Stats/Export) | `templates/plot_designer.html:92-96` | ✓ exact, all five |
| §11 "Maximize" button | `statistical_analyzer_autopilot_ui.py:866` | ✓ exact |
| §13 "Detect Outliers" menu; Grubbs default; n<8 caution | `statistical_analyzer_dialogs.py:525,590` | ✓ |
| §15 "Too few observations after filter" | `analysis_core.py:771` | ✓ exact |
| §16 regression-toggle checkbox label | `pipeline:371` `QCheckBox("Analyze as Linear Regression (Y = a + bX)")` | **✗ spelling** — see HT3 |
| §16/§17 residual diagnostics (Shapiro/Breusch-Pagan/Ramsey RESET) | `correlation_models.py:807,821,834` | ✓ all three run |
| §18 "Exploratory Correlation Matrix" menu | grep `-- src/` | ✓ present |

## Findings

### MEDIUM

**HT1 — §6 "Export Settings" describes a file-name field and a "Browse…" button that do not exist.**
`docs/HowTo.md:138-140`. §6 says: *"Set the output file name before or after analysis. Browse…,
next to the file-name field, opens a save dialog…"*. The main window has **no** persistent
output-file-name `QLineEdit` and **no** `QPushButton("Browse")` (the only "Browse..." button in the
codebase is the outlier dialog's, `statistical_analyzer_dialogs.py:569`). The output folder and base
name are instead chosen through a **save dialog** — `QFileDialog.getSaveFileName` at
`pipeline:1880`, with the directory derived from the chosen path (`output_dir =
os.path.dirname(ap_file_path)`, `:1888`) — which appears as part of the export/analysis flow, not as
a standing field-plus-button. **Impact:** a first-time user following §6 looks for a control that
isn't there; this is a pre-Filter-Bucket section that drifted after the export UI changed. **Fix:**
rewrite §6 to describe the save dialog (when it appears, what it sets) and drop the "file-name field
+ Browse… button" description.

### LOW

**HT2 — §8 names a "Select None" button the group and pairwise dialogs don't have.**
`docs/HowTo.md:182`. It states both *"the group-selection dialog and the pairwise-comparison dialog…
provide Select All, Deselect All, and Select None buttons."* Read from source: `GroupSelectionDialog`
provides **Select All + Deselect All** only (`statistical_analyzer_dialogs.py:92-93`), and
`ComparisonSelectionDialog` provides **Select All + Deselect All** only
(`comparison_selection_dialog.py:64-65`). "Select None" exists **only** on the outlier/dataset dialog
(`statistical_analyzer_dialogs.py:484`). **Impact:** minor — "Deselect All" does the same job, but
the sentence lists a third button that isn't on those two dialogs. **Fix:** drop "and Select None"
from the §8 sentence (or reword to "Select All / Deselect All").

**HT3 — §16 quotes the regression toggle with British spelling; the button is American.**
`docs/HowTo.md:342` quotes *"Analyse as Linear Regression (Y = a + bX)"*. The actual checkbox is
`QCheckBox("Analyze as Linear Regression (Y = a + bX)")` (`pipeline:371`; `help_content.py:458` also
uses "Analyze"). **Impact:** trivial — a one-letter mismatch in an explicitly-quoted control label
(the surrounding prose's British spelling is fine; only the *quoted button text* must match the UI).
**Fix:** change the quoted label to "Analyze".

### INFORMATIONAL

**HT4 — §3's "literal" status-message list is a representative subset, not the complete set.**
`docs/HowTo.md:100-106` introduces five messages with *"The messages are literal:"*. All five are
exact matches in source, but the mapping feedback label emits ~10 distinct strings (e.g.
`pipeline:836` "Assign at least one measurement column.", `:841` "Assign Factor 1 (group column, for
example WT/KO)…", `:847` "Single mode requires exactly one measurement column.", `:859` "Only one
subject-ID column is supported."). Not a defect — the quoted five are correct and a guide need not
enumerate every state — but "literal" could be read as "complete". **Optional:** add "among others"
or list the remaining measurement/subject-ID messages.

## Strengths (verified)

- **Every button, menu entry, and tab label the guide names is real and exactly spelled**, bar the
  two label issues above: Load Data File, Select Data Ranges…, Select Groups For Analysis, Start Auto
  Analysis, Maximize, Worksheet, Table Preview, the Detect Outliers and Exploratory Correlation
  Matrix menus, and the five Plot-Designer tabs all match source.
- **The mapping status strings are quoted verbatim** — all five in §3 are byte-for-byte the strings
  in `pipeline`, a good sign the guide was written against the running app, not from memory.
- **The freshly-added §15 (Filter Bucket) is accurate**, including the exact "Too few observations
  after filter" stop message (`analysis_core.py:771`) — today's `68f89e2` "verified GUI details"
  claim holds up.
- **The statistical-methods sections are sound.** §16/§17's residual-diagnostics claim (Shapiro-Wilk,
  Breusch-Pagan, Ramsey RESET) matches the code that runs all three (`correlation_models.py:807-834`),
  and §13's Grubbs-is-default / Modified-Z small-n caution matches the outlier dialog.

## Recommended remediation order

1. **HT1 (§6 rewrite)** — the one section describing controls that don't exist; highest reader impact.
2. **HT2 (§8 "Select None")** — delete three words.
3. **HT3 (§16 "Analyse"→"Analyze")** — one letter in a quoted label.
4. **HT4 (§3 list framing)** — optional wording softening.

All are documentation edits; none touches code. HT1 is the only one a user would actually trip over.
