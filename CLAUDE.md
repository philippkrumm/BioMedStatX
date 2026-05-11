# lean-ctx — Context Engineering Layer

PREFER lean-ctx MCP tools over native equivalents for token savings:

| PREFER | OVER | Why |
|--------|------|-----|
| `ctx_read(path)` | Read / cat / head / tail | Session caching, 8 compression modes, re-reads cost ~13 tokens |
| `ctx_shell(command)` | Bash (shell commands) | Pattern-based compression for git, npm, cargo, docker, tsc |
| `ctx_search(pattern, path)` | Grep / rg | Compact context, token-efficient results |
| `ctx_tree(path, depth)` | ls / find | Compact directory maps with file counts |

## ctx_read Modes

- `full` — cached read (use for files you will edit)
- `map` — deps + API signatures (use for context-only files)
- `signatures` — API surface only
- `diff` — changed lines only (after edits)
- `aggressive` — syntax stripped
- `entropy` — Shannon + Jaccard filtering
- `lines:N-M` — specific range

## File Editing

Use native Edit/StrReplace when available. If Edit requires Read and Read is unavailable,
use `ctx_edit(path, old_string, new_string)` — it reads, replaces, and writes in one MCP call.
NEVER loop trying to make Edit work. If it fails, switch to ctx_edit immediately.
Write, Delete have no lean-ctx equivalent — use them normally.

# BioMedStatX Architecture (read before editing UI / analysis flow)

## Entry point + AutopilotMixin

`src/statistical_analyzer.py` is the runtime entry (`__main__`). The main
class is declared as:

```python
class StatisticalAnalyzerApp(AutopilotMixin, QMainWindow):
```

`AutopilotMixin` lives in `src/statistical_analyzer_autopilot_pipeline.py` and
binds ~40 module-level `_ap_*` functions as class attributes. Methods like
`init_ui`, `browse_file`, `load_file`, `load_sheet`,
`reset_application_state`, `determine_and_run_test`,
`configure_plot_from_result` resolve through the mixin to the corresponding
`_ap_*` implementations.

**Legacy note.** Earlier versions used a runtime `attach_autopilot_methods()`
monkey-patch instead of a mixin. The shim still exists for backwards
compatibility but is deprecated and emits a `DeprecationWarning`. New code
should rely on the mixin.

**Rule:** when looking for UI / mapping / analysis behavior, search
`statistical_analyzer_autopilot_pipeline.py` first. The methods that live
directly in `statistical_analyzer.py` are: `__init__`,
`_position_debug_console`, `create_menu`, the `show_*_help` dialogs,
`show_analysis_success_dialog`, `closeEvent`, `run_outlier_detection`,
`setup_updater`, `check_for_updates`. Everything else comes from the mixin.

## Excel / CSV loading

User clicks "Browse..." → `_ap_browse_file` (pipeline:822) → `_ap_load_file`
(pipeline:832): `pd.read_csv()` for `.csv`, `pd.ExcelFile().sheet_names` +
`pd.read_excel(sheet_name=…)` for `.xlsx`/`.xls`. Sheet switcher: `_ap_load_sheet`
(pipeline:875). Loaded DataFrame stored as `self.df`.

`_ap_maybe_pivot()` auto-detects wide-format paired data and melts to long.

## Analysis pipeline

`_ap_determine_and_run_test` builds context via
`_ap_build_analysis_context` and calls `_ap_execute_single_analysis` →
`AnalysisManager.analyze()` in `src/analysis_core.py`.

**Single source of truth:** the autopilot pipeline always injects the
in-memory DataFrame as `analysis_context["injected_df"] = self.df`, so
`AnalysisManager` does not re-read the file from disk. `DataImporter` in
`stats_functions.py` is dead code in the autopilot path (kept only as a
CLI-style entry point).

For advanced designs (RM ANOVA, Mixed ANOVA, Two-Way), control flows through
`src/statistical_testing/advanced_pipeline.py` →
`statistical_testing/engines/advanced_posthoc.py`. The "paired_custom" post-hoc
branch reaches `src/comparison_selection_dialog.py` via a UI dialog.

## Don't add legacy fallbacks

The classic plot workflow, multi-dataset button, and old `init_ui` body have
already been removed. Don't reintroduce them. Method bindings now live on
`AutopilotMixin` — if you want to change UI / mapping / analysis behavior,
edit the corresponding `_ap_*` function in
`statistical_analyzer_autopilot_pipeline.py`, NOT a redeclaration in
`statistical_analyzer.py`.
