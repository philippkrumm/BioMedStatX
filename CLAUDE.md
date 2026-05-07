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

## Entry point + monkey-patch

`src/statistical_analyzer.py` is the runtime entry (`__main__`). It defines
`StatisticalAnalyzerApp(QMainWindow)`, but **most of its methods are replaced at
runtime** by `attach_autopilot_methods()` in
`src/statistical_analyzer_autopilot_pipeline.py:1737`.

The patch overrides ~36 methods on the class object before `StatisticalAnalyzerApp()`
is instantiated. So `self.init_ui()`, `self.browse_file()`, `self.load_file()`,
`self.load_sheet()`, `self.reset_application_state()`, `self.determine_and_run_test()`,
`self.configure_plot_from_result()`, etc. resolve to the `_ap_*` functions in the
autopilot pipeline file, **not** to anything defined in `statistical_analyzer.py`.

**Rule:** when looking for UI / mapping / analysis behavior, search
`statistical_analyzer_autopilot_pipeline.py` first. The methods that remain in
`statistical_analyzer.py` are: `__init__`, `create_menu`, the `show_*_help`
dialogs, `show_analysis_success_dialog`, `closeEvent`, `run_outlier_detection`,
`setup_updater`, `check_for_updates`. Everything else lives in the pipeline.

## Excel / CSV loading

User clicks "Browse..." → `_ap_browse_file` (pipeline:822) → `_ap_load_file`
(pipeline:832): `pd.read_csv()` for `.csv`, `pd.ExcelFile().sheet_names` +
`pd.read_excel(sheet_name=…)` for `.xlsx`/`.xls`. Sheet switcher: `_ap_load_sheet`
(pipeline:875). Loaded DataFrame stored as `self.df`.

`_ap_maybe_pivot()` auto-detects wide-format paired data and melts to long.

## Analysis pipeline

`_ap_determine_and_run_test` (pipeline:1516) builds context via
`_ap_build_analysis_context` and calls `_ap_execute_single_analysis` →
`AnalysisManager.analyze()` in `src/analysis_core.py`. `AnalysisManager` re-reads
the file via `DataImporter.import_data()` in `src/stats_functions.py`. That is
the only live caller of `DataImporter`.

For advanced designs (RM ANOVA, Mixed ANOVA, Two-Way), control flows through
`src/statistical_testing/advanced_pipeline.py` →
`statistical_testing/engines/advanced_posthoc.py`. The "paired_custom" post-hoc
branch reaches `src/comparison_selection_dialog.py` via a UI dialog.

## Don't add legacy fallbacks

The classic plot workflow, multi-dataset button, and old `init_ui` body have
already been removed. Don't reintroduce them. If a method exists in
`statistical_analyzer.py` AND in `attach_autopilot_methods`'s assignment list
(pipeline:1737), the original is dead by definition — don't edit it.
