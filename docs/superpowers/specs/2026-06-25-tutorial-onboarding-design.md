# In-App Onboarding Tour — Design Spec

Date: 2026-06-25
Status: Approved for planning
Author: Philipp Krumm (with Claude)

## 1. Goal

Give first-time users a 60-second guided orientation of the BioMedStatX
main window so they understand what each panel does and how the workflow
flows, without reading a manual or watching a video. The tour removes fear
of the bare GUI. It does not teach statistics and does not enter data.

## 2. Scope

In scope:
- A passive, linear "Next"-button guided tour (7 steps) drawn as an overlay
  over the live main window.
- First-run detection: a one-time welcome gate offers the tour on the very
  first launch. After that, the tour is on demand only.
- Re-run entry point in the existing Help menu.
- A "Save Example Template" action that copies the bundled Excel template to
  a user-chosen location and reveals it in Finder/Explorer.

Out of scope (explicit non-goals):
- No interactive/action-gated tour (no "click this real button to advance").
  Stability over muscle memory: the analysis runs synchronously on the GUI
  thread, so a forced-action path risks half-built state.
- No sample-data auto-loading during the tour. The tour runs over the live
  (initially empty) app and explains where things will appear.
- No changes to the statistics engine, analysis pipeline, or report output.
- Visual styling (QSS, colors, animation curves) is specified at a high level
  here and finalized during implementation with the frontend-design skill.

## 3. UX flow

### First launch
1. App starts as today.
2. After the main window is shown, check `QSettings`. If the stored
   `onboarding/completed_version` does not match the current app version
   (empty on a fresh machine), show a small modal welcome dialog:
   "New here? Take a 60-second tour." with `[Start tour]` and `[Maybe later]`.
3. `[Start tour]` launches the overlay tour. Either choice writes the current
   version to `QSettings` so the gate never auto-shows again for this version.

### Subsequent launches
- No auto prompt. The tour is reachable from `Help -> Interactive Tour`.
- A future major UI version can bump the stored-version comparison to
  re-offer the tour once ("what's new" style). Not required for v1.

### During the tour (passive model)
- A full-window overlay dims the app and cuts a spotlight hole around the
  current target. A bubble shows the step title, body, optional tip, a
  progress indicator, and `Back` / `Next` / `Skip tour`.
- The overlay intercepts all mouse and keyboard input. `Esc` exits. `Enter`
  and `Space` advance. Background widgets cannot be triggered.
- The final step points at the Help menu with a gentle pulse animation and
  tells the user they can restart the tour and find more help there.

## 4. Tour content (7 steps)

Each step targets real widgets (see Section 6 for exact references). Body is
2-3 sentences, friendly but professional, English. One tip line per step.
Titles use the wording below.

1. Bring In Your Data — targets: Load Data File button, Worksheet dropdown,
   Select Data Ranges button.
2. Meet Your Variables — targets: preview table, column cards area.
3. Shape Your Analysis — targets: the six mapping buckets.
4. Define and Compute — targets: Select Groups button, Start Auto Analysis.
5. Full Statistical Transparency — target: Decision Tree panel.
6. Your Results at a Glance — target: Result Cockpit.
7. Help Is Always One Click Away — target: Help menu in the menu bar (pulse).

Full final copy lives in Appendix A.

## 5. Architecture

New module: `src/ui/components/tutorial_overlay.py` (matches the existing
`ui/components` layout). Three units:

### `TourStep` (dataclass)
- `title: str`
- `body: str`
- `tip: str | None`
- `resolve_rect: Callable[[], QRect | None]` — returns the spotlight rect in
  the host window's coordinate space, or `None` for a centered bubble with no
  spotlight (used as a fallback when a target is missing/hidden).
- `placement: Literal["auto","above","below","left","right"]` (default auto)
- `pre_show: Callable[[], None] | None` — optional hook run before measuring,
  e.g. `scroll_area.ensureWidgetVisible(widget)`.

Helper constructors keep call sites clean:
- `from_widgets(*widgets)` -> resolver returning the union of the widgets'
  rects mapped into window coordinates, skipping any that are `None`,
  not visible, or zero-size.
- `from_menu_action(menubar, action)` -> resolver using
  `menubar.actionGeometry(action)` for the Help-menu step.

### `TutorialOverlay(QWidget)`
- Child of the `QMainWindow`, resized to the full window rect and raised on
  top (covers the central scroll area and the menu bar region).
- `paintEvent`: fills a dark translucent layer, then punches the spotlight
  using `QPainter.CompositionMode_DestinationOut` (antialiased, rounded
  corners, optional accent ring). No `QRegion.setMask` (pixelated on HiDPI).
- Owns the bubble (a child frame) and positions it next to the spotlight,
  flipping side if it would clip the window.
- Input capture: `setFocusPolicy(StrongFocus)`, `setFocus()`,
  `grabKeyboard()`, plus an app-level `eventFilter` that swallows key events
  except `Esc`/`Enter`/`Space`/Tab within the bubble. Releases the grab on
  close.
- Re-anchoring: recompute spotlight + bubble on the window `resizeEvent`, on
  `QSplitter.splitterMoved`, and on the scroll area's scrollbar
  `valueChanged`. All geometry reads are deferred via `QTimer.singleShot(0,
  ...)` after `pre_show`, so pending layout/scroll/paint settle first.

### `TutorialController` (mixin function group)
- Holds the ordered `list[TourStep]`, current index, and `_tour_active` flag.
- `start_tutorial()`, `_advance()`, `_back()`, `_finish()`.
- Bound on `AutopilotMixin` as `_ap_*` functions, consistent with the rest of
  the UI pipeline. `start_tutorial` is the single entry point used by both the
  welcome gate and the Help menu action.

## 6. Widget targeting

Targets are addressed by direct attribute reference (robust, no string
lookups), with three small promotions/additions to expose them:

| Step | Target attribute(s) | Current state | Change |
|------|--------------------|---------------|--------|
| 1 | `auto_sheet_combo`, `range_select_btn` | already `self.` | none |
| 1 | Load Data File button | local `browse_button` (pipeline:148) | promote to `self.browse_button` |
| 2 | `preview_table`, `header_cards_widget` | already `self.` | none |
| 3 | `dv_bucket`, `factor1_bucket`, `factor2_bucket`, `subject_bucket`, `covariates_bucket`, `filter_bucket` | already `self.` | wrap in a container `self.mapping_panel` for one clean spotlight, or union their rects |
| 4 | `analysis_group_button`, `start_analysis_button` | already `self.` | none (union rect) |
| 5 | `decision_tree_panel` | already `self.` | none |
| 6 | `result_cockpit` | already `self.` | none |
| 7 | Help menu action | local in `create_menu` (statistical_analyzer.py:191) | store `self.help_menu` / its `menuAction()` |

Reuse the existing `from_widgets` union-rect logic for multi-target steps
(1 and 4). The mapping step (3) prefers a wrapping container so the spotlight
is a single tidy rectangle rather than a ragged union.

## 7. Edge-case guards (verified against the code)

1. Layout settle: every geometry read happens inside `QTimer.singleShot(0,
   ...)` after `pre_show` (which may call `ensureWidgetVisible`), because the
   main UI lives inside a `QScrollArea` and freshly shown/populated targets
   have dirty geometry until the event loop finalizes layout and paint.
2. Keyboard capture: `grabKeyboard()` plus an app-level `eventFilter` so a
   stray Space/Enter cannot trigger a focused background button (e.g.
   `start_analysis_button`). Released on tour close.
3. Spotlight rendering: `CompositionMode_DestinationOut` / `QPainterPath` with
   antialiasing for clean rounded edges on HiDPI. Never `QRegion.setMask`.
4. Missing/hidden targets and re-entrancy:
   - If a step's resolver returns `None`, or the target is not visible or has
     zero size (true on first run: `preview_table` is hidden until a file is
     loaded, cards/buckets/cockpit are empty), the overlay shows a centered
     bubble with explanatory text and no spotlight, then continues.
   - `start_tutorial()` returns immediately if `_tour_active` is already true,
     preventing a second overlay when re-launched from the Help menu.

Note: the original review's "guard against starting the tour during a
background analysis thread" does not apply. The only `QThread` in the codebase
is the update checker (`src/core/updater.py`); analysis runs synchronously on
the GUI thread, so the menu is inert while it computes and the race cannot
occur. No `is_processing` guard is added.

## 8. First-run persistence

`QSettings("BioMedStatX", "BioMedStatX")`, key `onboarding/completed_version`.

- In `StatisticalAnalyzerApp.__init__`, after `init_ui` and window show:
  read the key; if it differs from the current app version, schedule the
  welcome dialog via `QTimer.singleShot(400, self._maybe_offer_tour)`.
- Both welcome-dialog choices, and tour completion/skip, write the current
  version back.

The app version is read from the same constant the updater already uses, so
there is one source of truth.

## 9. Example-template export feature

The bundled template (`assets/BioMedStatX_Excel_Template.xlsx`, 8 long-format
sheets covering t-test, ANOVA, repeated t-test, RM one-factor, two-way ANOVA,
mixed ANOVA, multi-dataset, Kruskal) ships with the app. The build already
globs the whole `assets/` folder (`datas=[("assets/", "assets"), ...]` in
`BioMedStatX.spec`), so no build change is needed.

Users must never browse into the bundled copy. The app is built `onedir`
(`exclude_binaries=True`, `COLLECT`), so assets live in a persistent
`_internal/assets/` folder; on macOS that path is inside the signed `.app`
bundle. Editing it in place is invisible and breaks signing/updates. Instead,
provide an export action:

```python
def _ap_export_example_template(self):
    src = _resource_path("assets/BioMedStatX_Excel_Template.xlsx")  # existing helper
    if not os.path.exists(src):
        QMessageBox.critical(self, "Error", "Internal template asset missing.")
        return
    default = os.path.join(os.path.expanduser("~"), "Desktop",
                           "BioMedStatX_Template.xlsx")
    target, _ = QFileDialog.getSaveFileName(
        self, "Save Example Template As", default, "Excel Worksheets (*.xlsx)")
    if not target:
        return
    try:
        shutil.copy2(src, target)
        QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(target)))
    except OSError as e:
        QMessageBox.critical(self, "Export Error", f"Could not write file:\n{e}")
```

- Reuses the existing `_resource_path()` helper (handles `sys._MEIPASS`); no
  new path function.
- `QFileDialog` + `QDesktopServices.openUrl` are cross-platform: native save
  sheet and Finder/Explorer on both macOS and Windows, no OS-specific code.

Placement (a permanent affordance, discoverable after the tour too):
- `Help -> Save Example Template...`
- A small secondary link/button in the left panel next to "Load Data File".
- The tour's Step 1 only points at this; it does not own the action.

## 10. Styling (finalized during implementation)

Match the existing dark dashboard palette (`assets/BioMedStatX_2_0.qss`).
Reuse `_apply_elevation()` / `QGraphicsDropShadowEffect` for the bubble.
Spotlight gets a soft accent ring; the final Help-menu step pulses gently.
Concrete QSS and animation values are produced with the frontend-design skill
in the implementation phase, not here.

## 11. Files touched

- New: `src/ui/components/tutorial_overlay.py`
- Edit `src/autopilot/statistical_analyzer_autopilot_pipeline.py`: promote
  `self.browse_button`; optional `self.mapping_panel` container; bind
  `_ap_start_tutorial`, `_ap_export_example_template`, `_ap_maybe_offer_tour`.
- Edit `src/analysis/statistical_analyzer.py`: `create_menu` adds
  "Interactive Tour" and "Save Example Template..."; store `self.help_menu`;
  `__init__` first-run check.
- New asset already in place: `assets/BioMedStatX_Excel_Template.xlsx`.
- No change to `BioMedStatX.spec`.

## 12. Testing

- Unit/headless: `from_widgets` union and the missing/hidden/zero-size
  fallback to a centered bubble; `_tour_active` re-entrancy guard;
  `QSettings` gate logic (fresh vs stored version).
- Manual: run the full 7-step tour on an empty app and on a loaded file;
  resize and drag the splitter mid-tour to confirm re-anchoring; confirm
  Space/Enter do not trigger background buttons; confirm the template export
  writes the file and opens the right folder on macOS (and Windows if
  available).

## 13. Follow-ups (not in this spec)

- Optional: extend the Getting Started Help recipe with a paired/repeated and
  a covariate example. The current recipe is accurate; this is additive only.
- Optional: a "what's new" re-trigger on a future major UI version.

---

## Appendix A — Final bubble copy

Step 1 — Bring In Your Data
Start by loading your experimental or clinical data. Open any Excel or CSV
file here, then pick the relevant sheet from the worksheet dropdown. If a
sheet is cluttered, use Select Data Ranges to capture only the cells you need.
Tip: New to the layout? A ready-made Excel template ships with BioMedStatX,
with one tab per design (t-test, ANOVA, repeated measures, and more), each
already in the long format the app expects.

Step 2 — Meet Your Variables
A live preview of your table appears as soon as the file loads, and each
column header becomes a draggable card here. These cards are the building
blocks you route into your analysis.
Tip: Every card shows its detected type (Numeric, Categorical, or Datetime)
right on the card, so you can check each column at a glance.

Step 3 — Shape Your Analysis
This is where you define your study design. Drag the variable cards into the
buckets to assign their roles: Dependent Variable, Factor 1 and 2, an optional
Subject ID for repeated measures, and Covariates. As you map, the status line
below the buckets shows what each assignment means and what is still missing,
and the Start button activates once your mapping is complete and consistent.
Tip: Placed a card in the wrong bucket? Click the small x on the chip to
return it.

Step 4 — Define and Compute
Use the group selector to set your comparison cohorts, such as Treatment
versus Control, then start the analysis. BioMedStatX checks normality and
distribution to select the appropriate statistical test for your data.
Tip: Assumption testing runs automatically in the background, so the method
comes from your data rather than a guess.

Step 5 — Full Statistical Transparency
BioMedStatX is not a black box. The decision tree traces every step the engine
took, so you can see why a given test (an ANOVA, a Mann-Whitney U, and so on)
was selected for your dataset.
Tip: Hover over any node to read its full label, or use Maximize to study the
whole path full-screen.

Step 6 — Your Results at a Glance
The cockpit gathers your test statistics, p-values, effect sizes, and a
written summary you can drop straight into a manuscript, all in one view.
Tip: Use Open Output Folder for the full HTML report, including tables, plots,
and the complete method trace.

Step 7 — Help Is Always One Click Away
That is the whole workflow, from loading a file to a finished result. You can
restart this tour anytime from the Help menu, which also holds the Getting
Started guide and recipe-based help for every design.
Tip: The Help menu links to dedicated guides for paired samples, advanced
ANOVA, and correlation and regression.
