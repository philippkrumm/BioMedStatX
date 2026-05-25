import os
import re

# Map old module names to new package modules
module_map = {
    'lazy_imports': 'core.lazy_imports',
    'logger_config': 'core.logger_config',
    'methodology_trace': 'core.methodology_trace',
    'updater': 'core.updater',
    'help_content': 'core.help_content',
    
    'analysis_core': 'analysis.analysis_core',
    'statisticaltester': 'analysis.statisticaltester',
    'statistical_analyzer': 'analysis.statistical_analyzer',
    'stats_functions': 'analysis.stats_functions',
    'nonparametricanovas': 'analysis.nonparametricanovas',
    'posthoc_core': 'analysis.posthoc_core',
    'outlier_core': 'analysis.outlier_core',
    'effect_sizes': 'analysis.effect_sizes',
    'clinical_models': 'analysis.clinical_models',
    'correlation_models': 'analysis.correlation_models',
    
    'statistical_analyzer_autopilot_pipeline': 'autopilot.statistical_analyzer_autopilot_pipeline',
    'statistical_analyzer_autopilot_ui': 'autopilot.statistical_analyzer_autopilot_ui',
    
    'resultsexporter': 'export.resultsexporter',
    'html_exporter': 'export.html_exporter',
    'export_dispatcher': 'export.export_dispatcher',
    'report_methods': 'export.report_methods',
    'report_tooltips': 'export.report_tooltips',
    
    'datavisualizer': 'visualization.datavisualizer',
    'decisiontreevisualizer': 'visualization.decisiontreevisualizer',
    'plot_preview': 'visualization.plot_preview',
    
    'plot_aesthetics_dialog': 'ui.dialogs.plot_aesthetics_dialog',
    'comparison_selection_dialog': 'ui.dialogs.comparison_selection_dialog',
    'statistical_analyzer_dialogs': 'ui.dialogs.statistical_analyzer_dialogs',
}

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content

    for old_mod, new_mod in module_map.items():
        # Replace 'from module import X'
        # Matches 'from module import ' or 'from module\nimport ' or similar
        # Since module names are exact, we use word boundaries
        pattern_from = rf'\bfrom\s+{old_mod}\b'
        content = re.sub(pattern_from, f'from {new_mod}', content)

        # Replace 'import module' -> 'import new_mod as module' to avoid breaking code that uses 'module.func()'
        # Note: sometimes they do 'import module1, module2' which is harder to regex.
        # Let's handle 'import module' and 'import module as mod'
        pattern_import = rf'\bimport\s+{old_mod}\b(?!\s+as)'
        content = re.sub(pattern_import, f'import {new_mod} as {old_mod}', content)

    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated {filepath}")

def main():
    search_dirs = ['src', 'tests']
    for sdir in search_dirs:
        for root, _, files in os.walk(sdir):
            for file in files:
                if file.endswith('.py'):
                    process_file(os.path.join(root, file))

if __name__ == '__main__':
    main()
