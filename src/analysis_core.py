import os
from datetime import datetime

import numpy as np
import pandas as pd

from lazy_imports import get_scipy_stats, get_statsmodels_multitest, get_matplotlib_pyplot


def get_results_exporter():
    from resultsexporter import ResultsExporter
    return ResultsExporter


def get_export_dispatcher():
    from export_dispatcher import ExportDispatcher
    return ExportDispatcher


def get_data_visualizer():
    from datavisualizer import DataVisualizer
    return DataVisualizer


def get_statistical_tester():
    from statisticaltester import StatisticalTester
    return StatisticalTester


def safe_format(val, fmt="{:.4f}", none_text="N/A"):
    if isinstance(val, (float, int)):
        try:
            return fmt.format(val)
        except Exception:
            return str(val)
    if val is None:
        return none_text
    return str(val)


def _get_stats_functions_deps():
    from stats_functions import DataImporter, UIDialogManager, PostHocAnalyzer, PostHocStatistics

    return DataImporter, UIDialogManager, PostHocAnalyzer, PostHocStatistics


class _DataImporterProxy:
    @staticmethod
    def import_data(*args, **kwargs):
        DataImporter, _, _, _ = _get_stats_functions_deps()
        return DataImporter.import_data(*args, **kwargs)


class _UIDialogManagerProxy:
    @staticmethod
    def select_posthoc_test_dialog(*args, **kwargs):
        _, UIDialogManager, _, _ = _get_stats_functions_deps()
        return UIDialogManager.select_posthoc_test_dialog(*args, **kwargs)

    @staticmethod
    def select_control_group_dialog(*args, **kwargs):
        _, UIDialogManager, _, _ = _get_stats_functions_deps()
        return UIDialogManager.select_control_group_dialog(*args, **kwargs)

    @staticmethod
    def select_custom_pairs_dialog(*args, **kwargs):
        _, UIDialogManager, _, _ = _get_stats_functions_deps()
        return UIDialogManager.select_custom_pairs_dialog(*args, **kwargs)


class _PostHocAnalyzerProxy:
    @staticmethod
    def add_comparison(*args, **kwargs):
        _, _, PostHocAnalyzer, _ = _get_stats_functions_deps()
        return PostHocAnalyzer.add_comparison(*args, **kwargs)


class _PostHocStatisticsProxy:
    @staticmethod
    def calculate_ci_mean_diff(*args, **kwargs):
        _, _, _, PostHocStatistics = _get_stats_functions_deps()
        return PostHocStatistics.calculate_ci_mean_diff(*args, **kwargs)

    @staticmethod
    def calculate_cohens_d(*args, **kwargs):
        _, _, _, PostHocStatistics = _get_stats_functions_deps()
        return PostHocStatistics.calculate_cohens_d(*args, **kwargs)


DataImporter = _DataImporterProxy
UIDialogManager = _UIDialogManagerProxy
PostHocAnalyzer = _PostHocAnalyzerProxy
PostHocStatistics = _PostHocStatisticsProxy


class DatasetSelector:
    """Helper class to manage dataset selection in the UI"""
    
    @staticmethod
    def get_available_datasets(file_path, sheet_name=None):
        """
        Get all available datasets (sheets) from an Excel file
        
        Returns:
        --------
        dict: {sheet_name: preview_info}
        """
        try:
            if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                # Get all sheet names
                xl_file = pd.ExcelFile(file_path)
                datasets = {}
                
                for sheet in xl_file.sheet_names:
                    try:
                        # Get a preview of each sheet
                        df_preview = pd.read_excel(file_path, sheet_name=sheet, nrows=5)
                        datasets[sheet] = {
                            'columns': df_preview.columns.tolist(),
                            'shape': f"{len(pd.read_excel(file_path, sheet_name=sheet))} rows",
                            'preview': df_preview.head(3).to_dict('records')
                        }
                    except Exception as e:
                        datasets[sheet] = {'error': str(e)}
                
                return datasets
            else:
                # For CSV files, return single dataset
                df_preview = pd.read_csv(file_path, nrows=5)
                return {
                    'CSV Data': {
                        'columns': df_preview.columns.tolist(),
                        'shape': f"{len(pd.read_csv(file_path))} rows",
                        'preview': df_preview.head(3).to_dict('records')
                    }
                }
        except Exception as e:
            return {'Error': {'error': str(e)}}

# Modified AnalysisManager.analyze function
class AnalysisManager:
    @staticmethod
    def analyze(file_path, group_col, groups, sheet_name=0, value_cols=None, 
                selected_datasets=None, combine_columns=False, width=12, height=10, 
                dependent=False, compare=None, colors=None, hatches=None,
                title=None, x_label=None, y_label=None, file_name=None, 
                save_plot=True, skip_plots=False, error_type="sd", skip_excel=False, 
                dataset_name=None, additional_factors=None, show_individual_lines=True, 
                **kwargs):
        
        print("DEBUG ANALYZE: AnalysisManager.analyze called")
        print(f"DEBUG ANALYZE: Current working directory: {os.getcwd()}")
        print(f"DEBUG ANALYZE: file_path = {file_path}")
        print(f"DEBUG ANALYZE: file_name = {file_name}")
        print(f"DEBUG ANALYZE: save_plot = {save_plot}, skip_plots = {skip_plots}, skip_excel = {skip_excel}")
        # Single dataset analysis (existing functionality)
        if selected_datasets is None or len(selected_datasets) <= 1:
            # Use existing single dataset logic
            actual_sheet = selected_datasets[0] if selected_datasets else sheet_name
            return AnalysisManager._analyze_single_dataset(
                file_path, group_col, groups, actual_sheet, value_cols, 
                combine_columns, width, height, dependent, compare, colors, hatches,
                title, x_label, y_label, file_name, save_plot, skip_plots, 
                error_type, skip_excel, dataset_name, additional_factors, 
                show_individual_lines, **kwargs
            )
        
        # Multiple dataset analysis
        else:
            return AnalysisManager._analyze_multiple_datasets(
                file_path, group_col, groups, selected_datasets, value_cols,
                combine_columns, width, height, dependent, compare, colors, hatches,
                title, x_label, y_label, file_name, save_plot, skip_plots,
                error_type, skip_excel, additional_factors, show_individual_lines, **kwargs
            )

    @staticmethod
    def _load_dataframe(file_path, sheet_name=0):
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        return pd.read_excel(file_path, sheet_name=sheet_name)

    @staticmethod
    def _prepare_contextual_inputs(file_path, sheet_name, group_col, groups, value_cols,
                                   combine_columns, dependent, additional_factors, kwargs):
        analysis_context = kwargs.get("analysis_context")
        if not analysis_context:
            samples, df = DataImporter.import_data(
                file_path,
                sheet_name=sheet_name,
                group_col=group_col,
                value_cols=value_cols,
                combine_columns=combine_columns
            )
            filtered_samples = {g: samples[g] for g in groups if g in samples}
            return {
                "df": df,
                "samples": samples,
                "filtered_samples": filtered_samples,
                "groups": groups,
                "group_col": group_col,
                "value_cols": value_cols,
                "dependent": dependent,
                "additional_factors": additional_factors,
            }

        injected_df = analysis_context.get("injected_df")
        if injected_df is not None:
            df = injected_df.copy()
        else:
            df = AnalysisManager._load_dataframe(file_path, sheet_name=sheet_name).copy()
        context_value_cols = analysis_context.get("dv_columns") or value_cols
        if not context_value_cols:
            raise ValueError("Auto-pilot analysis requires at least one dependent variable.")

        factor_columns = analysis_context.get("factor_columns", [])
        if not factor_columns:
            raise ValueError("Auto-pilot analysis requires at least one factor column.")

        display_group_col = factor_columns[0]
        groups_to_use = analysis_context.get("group_labels") or groups
        working_df = df.copy()
        filter_spec = analysis_context.get("filter")
        if filter_spec:
            filter_col, filter_val = filter_spec
            if filter_col in working_df.columns:
                working_df = working_df[working_df[filter_col] == filter_val]

        selected_group_column = analysis_context.get("selected_group_column")
        selected_groups = analysis_context.get("selected_groups") or []
        if selected_group_column and selected_groups and selected_group_column in working_df.columns:
            working_df = working_df[working_df[selected_group_column].isin(selected_groups)]

        if len(factor_columns) == 2:
            factor_a, factor_b = factor_columns
            display_group_col = "__AUTO_GROUP__"
            working_df[display_group_col] = working_df.apply(
                lambda row: f"{factor_a}={row[factor_a]}, {factor_b}={row[factor_b]}",
                axis=1
            )
            if not groups_to_use:
                groups_to_use = sorted(working_df[display_group_col].dropna().unique(), key=lambda item: str(item))
        else:
            if not groups_to_use:
                groups_to_use = sorted(working_df[display_group_col].dropna().unique(), key=lambda item: str(item))

        primary_dv = context_value_cols[0]
        samples = {}
        for group_name in groups_to_use:
            subset = working_df[working_df[display_group_col] == group_name]
            samples[group_name] = subset[primary_dv].dropna().tolist()

        local_kwargs = dict(kwargs)
        inferred_test = analysis_context.get("inferred_test")
        if inferred_test in {"two_way_anova", "mixed_anova", "repeated_measures_anova",
                             "ancova", "two_way_ancova", "lmm", "logistic_regression",
                             "beta_regression", "correlation", "linear_regression"}:
            local_kwargs["test"] = inferred_test
        if analysis_context.get("subject_column"):
            local_kwargs["subject_column"] = analysis_context.get("subject_column")
        if analysis_context.get("covariates"):
            local_kwargs["covariates"] = analysis_context.get("covariates")

        resolved_additional_factors = additional_factors
        if inferred_test == "two_way_anova":
            resolved_additional_factors = factor_columns[:2]
        elif inferred_test == "mixed_anova":
            resolved_additional_factors = [
                *(analysis_context.get("between_factors") or []),
                *(analysis_context.get("within_factors") or []),
            ]
        elif inferred_test == "repeated_measures_anova":
            resolved_additional_factors = analysis_context.get("within_factors") or factor_columns[:1]

        if resolved_additional_factors is not None:
            local_kwargs["additional_factors"] = resolved_additional_factors
        local_kwargs["analysis_context"] = analysis_context

        return {
            "df": working_df,
            "samples": samples,
            "filtered_samples": samples,
            "groups": list(groups_to_use),
            "group_col": display_group_col,
            "value_cols": context_value_cols,
            "dependent": analysis_context.get("dependent", dependent),
            "additional_factors": resolved_additional_factors,
            "kwargs": local_kwargs,
        }
            
    @staticmethod
    def _analyze_multiple_datasets(file_path, group_col, groups, selected_datasets, value_cols,
                                  combine_columns, width, height, dependent, compare, colors, hatches,
                                  title, x_label, y_label, file_name, save_plot, skip_plots,
                                  error_type, skip_excel, additional_factors, show_individual_lines, **kwargs):
        """
        Multiple dataset analysis with unified Excel output
        """
        all_results = {}
        failed_datasets = {}
        
        print(f"Starting analysis of {len(selected_datasets)} datasets...")
        
        # Analyze each selected dataset
        for i, dataset_name in enumerate(selected_datasets):
            print(f"Analyzing dataset {i+1}/{len(selected_datasets)}: {dataset_name}")
            
            try:
                # Analyze single dataset
                result = AnalysisManager._analyze_single_dataset(
                    file_path=file_path,
                    group_col=group_col,
                    groups=groups,
                    sheet_name=dataset_name,
                    value_cols=value_cols,
                    combine_columns=combine_columns,
                    width=width,
                    height=height,
                    dependent=dependent,
                    compare=compare,
                    colors=colors,
                    hatches=hatches,
                    title=f"{title} - {dataset_name}" if title else dataset_name,
                    x_label=x_label,
                    y_label=y_label,
                    file_name=f"{file_name}_{dataset_name}" if file_name else dataset_name,
                    save_plot=save_plot,
                    skip_plots=skip_plots,
                    error_type=error_type,
                    skip_excel=True,  # Skip individual Excel files
                    dataset_name=dataset_name,
                    additional_factors=additional_factors,
                    show_individual_lines=show_individual_lines,
                    dialog_progress=f"({i+1}/{len(selected_datasets)})",
                    dialog_column=dataset_name,
                    **kwargs
                )
                
                if "error" in result:
                    failed_datasets[dataset_name] = result["error"]
                    print(f"ERROR analyzing {dataset_name}: {result['error']}")
                else:
                    all_results[dataset_name] = result
                    print(f"Successfully analyzed {dataset_name}")
                    
            except Exception as e:
                error_msg = f"Exception during analysis: {str(e)}"
                failed_datasets[dataset_name] = error_msg
                print(f"ERROR analyzing {dataset_name}: {error_msg}")
        
        # Apply FDR correction (Benjamini-Hochberg) across all primary p-values
        if len(all_results) >= 2:
            try:
                multipletests = get_statsmodels_multitest()
                dataset_names_ordered = list(all_results.keys())
                raw_ps = [all_results[n].get("p_value") for n in dataset_names_ordered]
                valid_indices = [i for i, p in enumerate(raw_ps) if isinstance(p, (float, int))]
                if len(valid_indices) >= 2:
                    valid_ps = [raw_ps[i] for i in valid_indices]
                    _, p_adj, _, _ = multipletests(valid_ps, method='fdr_bh')
                    for rank, ds_idx in enumerate(valid_indices):
                        all_results[dataset_names_ordered[ds_idx]]["p_value_fdr"] = float(p_adj[rank])
                    print(f"FDR correction applied across {len(valid_indices)} datasets.")
            except Exception as e:
                print(f"Warning: FDR correction failed: {str(e)}")

        # Create combined Excel output
        if all_results:
            base_name = file_name if file_name else "multi_dataset_analysis"
            excel_path = f"{base_name}_combined_results.xlsx"
            
            try:
                ExportDispatcher = get_export_dispatcher()
                export_result = ExportDispatcher.export_multi_dataset_results(all_results, excel_path)
                if export_result.get("warning"):
                    print(f"WARNING: {export_result['warning']}")
                print(f"Combined results saved to: {excel_path}")
            except Exception as e:
                print(f"Error creating combined Excel file: {str(e)}")
        
        # Return summary
        return {
            "type": "multi_dataset_analysis",
            "successful_datasets": list(all_results.keys()),
            "failed_datasets": failed_datasets,
            "results": all_results,
            "combined_excel": excel_path if all_results else None,
            "summary": {
                "total_datasets": len(selected_datasets),
                "successful": len(all_results),
                "failed": len(failed_datasets),
                "success_rate": f"{len(all_results)/len(selected_datasets)*100:.1f}%"
            }
        }
            
    @staticmethod
    def _analyze_single_dataset(file_path, group_col, groups, sheet_name, value_cols, 
                               combine_columns, width, height, dependent, compare, colors, hatches,
                               title, x_label, y_label, file_name, save_plot, skip_plots, 
                               error_type, skip_excel, dataset_name, additional_factors, 
                               show_individual_lines, **kwargs):
        
        # Get classes lazily to avoid circular imports
        ResultsExporter = get_results_exporter()
        StatisticalTester = get_statistical_tester()
        DataVisualizer = get_data_visualizer()
        
        # CRITICAL FIX: Ensure additional_factors is available in kwargs
        # since the advanced test logic looks for it there
        if additional_factors is not None and 'additional_factors' not in kwargs:
            kwargs['additional_factors'] = additional_factors
        
        # Basic parameter validation
        if not file_path or not os.path.exists(file_path):
            raise ValueError("Please specify a valid file")
        if not groups and not kwargs.get("analysis_context"):
            raise ValueError("Please specify at least one group")
        if not group_col:
            raise ValueError("Please specify a valid group column")
        if error_type not in ["sd", "se"]:
            raise ValueError("Error bar type must be 'sd' or 'se'")

        analysis_log = f"Analysis Report\n"
        analysis_log += f"Date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        analysis_log += f"File: {file_path}\n"
        analysis_log += f"Worksheet: {sheet_name}\n"
        analysis_log += f"Group column: {group_col}\n"
        analysis_log += f"Value column(s): {', '.join(value_cols) if value_cols else 'All numeric columns'}\n"
        analysis_log += f"Groups to analyze: {', '.join(map(str, groups))}\n"
        analysis_log += f"Dependent samples: {'Yes' if dependent else 'No'}\n"
        analysis_log += f"Error bar type: {'SEM (standard error)' if error_type == 'se' else 'SD (standard deviation)'}\n"

        if compare:
            compare_str = ", ".join([f"{g1} vs {g2}" for g1, g2 in compare])
            analysis_log += f"Specific comparisons: {compare_str}\n"

        analysis_log += "\n--- ANALYSIS ---\n\n"

        try:
            prepared_inputs = AnalysisManager._prepare_contextual_inputs(
                file_path=file_path,
                sheet_name=sheet_name,
                group_col=group_col,
                groups=groups,
                value_cols=value_cols,
                combine_columns=combine_columns,
                dependent=dependent,
                additional_factors=additional_factors,
                kwargs=kwargs
            )
            samples = prepared_inputs["samples"]
            df = prepared_inputs["df"]
            filtered_samples = prepared_inputs["filtered_samples"]
            groups = prepared_inputs["groups"]
            group_col = prepared_inputs["group_col"]
            value_cols = prepared_inputs["value_cols"]
            dependent = prepared_inputs["dependent"]
            additional_factors = prepared_inputs["additional_factors"]
            kwargs = prepared_inputs.get("kwargs", kwargs)

            # Validations and logging
            if not filtered_samples:
                raise ValueError(f"None of the specified groups were found in the data. Available groups: {list(samples.keys())}")

            for group, values in filtered_samples.items():
                if len(values) < 1:
                    raise ValueError(f"Group '{group}' contains no data.")

            analysis_log += f"Data imported successfully.\n"
            analysis_log += "Number of data points per group:\n"
            for group, values in filtered_samples.items():
                analysis_log += f"  {group}: {len(values)} data points\n"

            # Initialize the result dictionary (important: before first assignments!)
            results = {}

            # --- Clinical model dispatch (ANCOVA, LMM, Logistic Regression, Correlation, Linear Regression) ---
            if kwargs.get('test') in ('ancova', 'two_way_ancova', 'lmm', 'logistic_regression',
                                      'beta_regression', 'correlation', 'linear_regression'):
                from clinical_models import (ANCOVAModel, LinearMixedModel,
                                             LogisticRegressionModel, BetaRegressionModel,
                                             DataHealthScanner)

                clinical_test = kwargs['test']
                covariates = kwargs.get('covariates', [])
                analysis_context = kwargs.get('analysis_context', {})
                subject_column = kwargs.get('subject_column') or analysis_context.get('subject_column')

                # --- Data Health Scan (runs before model fit, non-blocking) ---
                _model_type_map = {
                    'ancova': 'ANCOVA', 'two_way_ancova': 'ANCOVA',
                    'lmm': 'LMM', 'logistic_regression': 'LogisticRegression',
                    'beta_regression': 'BetaRegression',
                }
                try:
                    _scanner = DataHealthScanner(
                        df=df,
                        model_type=_model_type_map[clinical_test],
                        dv=value_cols[0],
                        covariates=covariates,
                        factors=analysis_context.get('factor_columns', []),
                        subject_col=subject_column,
                    )
                    _health_report = _scanner.run()
                except Exception:
                    _health_report = {"warnings": [], "checks": {}}

                if clinical_test in ('ancova', 'two_way_ancova'):
                    model = ANCOVAModel()
                    between_factors = analysis_context.get('between_factors') or analysis_context.get('factor_columns', [])
                    model.fit(df, dv=value_cols[0], between_factors=between_factors, covariates=covariates)
                    test_results = model.as_results_dict()

                elif clinical_test == 'lmm':
                    model = LinearMixedModel()
                    fixed_effects = analysis_context.get('within_factors', []) + analysis_context.get('between_factors', [])
                    if not fixed_effects:
                        fixed_effects = analysis_context.get('factor_columns', [])
                    model.fit(df, dv=value_cols[0], fixed_effects=fixed_effects,
                              random_intercept=subject_column, covariates=covariates or None)
                    test_results = model.as_results_dict()

                elif clinical_test == 'logistic_regression':
                    model = LogisticRegressionModel()
                    predictors = analysis_context.get('factor_columns', [])
                    model.fit(df, dv=value_cols[0], predictors=predictors, covariates=covariates or None)
                    test_results = model.as_results_dict()

                elif clinical_test == 'beta_regression':
                    from methodology_trace import MethodologyTrace
                    _beta_trace = MethodologyTrace()
                    _beta_predictors = analysis_context.get('factor_columns', [])
                    _beta_n = int(df[value_cols[0]].dropna().count())
                    _beta_n_pred = analysis_context.get('beta_n_predictors') or max(1, len(covariates or []) + 1)
                    _beta_epv = analysis_context.get('beta_epv') or (_beta_n / _beta_n_pred)
                    _beta_bias = analysis_context.get('beta_bias_corrected', _beta_epv < 10)
                    _beta_sv = analysis_context.get('beta_sv_transformed', False)
                    _beta_n_unique = int(df[value_cols[0]].dropna().nunique())

                    if _beta_sv:
                        _beta_trace.add(1, "Data Transformation",
                            f"Boundary values (exact 0 or 1) were present in the outcome. "
                            f"Smithson-Verkuilen transformation applied: y_adj = (y × (n−1) + 0.5) / n. "
                            f"This pushes boundary values strictly inside (0,1) as required by Beta Regression.",
                            detail=f"n={_beta_n}")

                    if _beta_bias:
                        _beta_trace.add(2, "Test Selection",
                            f"Outcome detected as proportion (all values strictly in (0,1), "
                            f"{_beta_n_unique} unique values). "
                            f"EPV = {_beta_n} / {_beta_n_pred} = {_beta_epv:.1f} < 10. "
                            f"Bias-corrected Beta Regression applied to compensate for "
                            f"small sample bias (Peduzzi et al., 1996, adapted). "
                            f"Note: EPV rule was derived for logistic regression — "
                            f"interpretation should be cautious.",
                            detail=f"EPV={_beta_epv:.1f}, n={_beta_n}, predictors={_beta_n_pred}")
                    else:
                        _beta_trace.add(2, "Test Selection",
                            f"Outcome detected as proportion (all values strictly in (0,1), "
                            f"{_beta_n_unique} unique values). "
                            f"EPV = {_beta_n} / {_beta_n_pred} = {_beta_epv:.1f} ≥ 10. "
                            f"Standard Beta Regression applied.",
                            detail=f"EPV={_beta_epv:.1f}, n={_beta_n}, predictors={_beta_n_pred}")

                    model = BetaRegressionModel()
                    model.fit(df, dv=value_cols[0], predictors=_beta_predictors,
                              covariates=covariates or None, bias_corrected=_beta_bias)
                    test_results = model.as_results_dict()
                    test_results["methodology_trace"] = _beta_trace
                    test_results["sv_transformed"] = _beta_sv
                    test_results["epv"] = round(_beta_epv, 2)

                elif clinical_test in ('correlation', 'linear_regression'):
                    from correlation_models import (CorrelationModel, SimpleLinearRegressionModel,
                                                    RegressionHealthScanner)

                    # Apply optional filter (e.g. only On-Pump patients)
                    filter_spec = analysis_context.get('filter')
                    analysis_df = df.copy()
                    if filter_spec:
                        filter_col, filter_val = filter_spec
                        analysis_df = analysis_df[analysis_df[filter_col] == filter_val]
                        if len(analysis_df) < 5:
                            raise ValueError(
                                f"Too few observations after filter "
                                f"'{filter_col} = {filter_val}' (n={len(analysis_df)})."
                            )

                    x_col = analysis_context.get('x_variable') or analysis_context.get('factor_columns', [None])[0]
                    y_col = value_cols[0]

                    # Data Health Scan (replaces DataHealthScanner for regression)
                    try:
                        _scanner = RegressionHealthScanner(
                            analysis_df, x_col=x_col, y_col=y_col,
                            covariates=covariates or None,
                        )
                        _health_report = _scanner.run()
                    except Exception:
                        _health_report = {"warnings": [], "checks": {}}

                    if clinical_test == 'correlation':
                        model = CorrelationModel()
                        model.fit(
                            analysis_df,
                            x_col=x_col,
                            y_col=y_col,
                            method='auto',
                            x_transform=analysis_context.get('x_transform', 'none'),
                            y_transform=analysis_context.get('y_transform', 'none'),
                        )
                        test_results = model.as_results_dict()
                    else:  # linear_regression
                        model = SimpleLinearRegressionModel()
                        model.fit(analysis_df, x_col=x_col, y_col=y_col,
                                  covariates=covariates or None)
                        test_results = model.as_results_dict()

                # Attach health report to results (non-blocking)
                test_results["data_health"] = _health_report

                # Clinical models handle their own assumptions; skip normality/variance
                test_info = None
                test_recommendation = None
                transformed_samples = filtered_samples
                results.update(test_results)

                # Skip the rest of the standard flow (normality check, post-hoc, etc.)
                # Jump straight to export
                results['groups'] = groups
                # For continuous-variable models (correlation/regression/logistic), raw_data from
                # filtered_samples has no meaningful group structure — skip it to avoid the group
                # chart and descriptive table using X-values as bogus group labels.
                _no_group_raw = clinical_test in ('correlation', 'linear_regression', 'logistic_regression', 'beta_regression')
                if not _no_group_raw:
                    results['raw_data'] = {g: filtered_samples[g][:] for g in groups}
                results['selected_groups'] = analysis_context.get('selected_groups') or groups
                results['group_column'] = analysis_context.get('selected_group_column') or analysis_context.get('factor_columns', [None])[0]
                results['factor_columns'] = analysis_context.get('factor_columns', [])
                results['covariates'] = covariates
                results['dependent_variable'] = value_cols[0]
                if analysis_context.get('filter'):
                    filter_col, filter_val = analysis_context['filter']
                    results['filter_applied'] = f"{filter_col} = {filter_val}"

                analysis_log += f"\nClinical model: {test_results.get('test', clinical_test)}\n"
                p_value = test_results.get('p_value')
                if p_value is not None:
                    analysis_log += f"p-Value: {p_value:.6f}\n"

                if file_name:
                    file_base = file_name
                else:
                    file_base = "_".join(map(str, groups))
                excel_file = f"{file_base}_results.xlsx"

                if not skip_excel:
                    original_dir = os.getcwd()
                    export_result = {}
                    try:
                        output_dir = os.path.dirname(os.path.abspath(excel_file))
                        if output_dir:
                            os.makedirs(output_dir, exist_ok=True)
                        ExportDispatcher = get_export_dispatcher()
                        export_result = ExportDispatcher.export_analysis_results(results, excel_file, analysis_log)
                        if export_result.get("warning"):
                            print(f"WARNING: {export_result['warning']}")
                        print(f"Results exported to: {excel_file}")
                    except Exception as export_error:
                        print(f"Error exporting to Excel: {export_error}")
                    finally:
                        os.chdir(original_dir)

                results["analysis_log"] = analysis_log
                results["excel_file"] = export_result.get("excel_path", excel_file) if not skip_excel else excel_file
                return results

            # For advanced tests that use prepare_advanced_test, skip the normality check here
            # as it will be handled in the advanced test flow
            if kwargs.get('test') in ['mixed_anova', 'two_way_anova', 'repeated_measures_anova']:
                # Skip normality check - will be handled by prepare_advanced_test
                transformed_samples = None
                test_recommendation = None
                test_info = None
            else:
                # Determine model type based on parameters
                if len(groups) == 2:
                    model_type = "ttest"
                    formula = "Value ~ C(Group)"
                elif len(groups) > 2:
                    model_type = "oneway"
                    formula = "Value ~ C(Group)"
                else:
                    model_type = "oneway"
                    formula = "Value ~ C(Group)"
                    
                # Normality and variance check with dataset name
                transformed_samples, test_recommendation, test_info = StatisticalTester.check_normality_and_variance(
                    groups,
                    filtered_samples,
                    dataset_name=dataset_name,
                    progress_text=kwargs.get('dialog_progress', None),
                    column_name=kwargs.get('dialog_column', None),
                    formula=formula,
                    model_type=model_type
                )
            print(f"DEBUG: Test recommendation is '{test_recommendation}'")
            print(f"DEBUG: Test info transformation: '{test_info.get('transformation') if test_info else 'N/A'}'")

            # Write test recommendation to log (only if we have one)
            if test_recommendation:
                analysis_log += f"\nTest recommendation: {test_recommendation}\n"

            # For dependent samples, perform additional validation
            if dependent:
                validation = StatisticalTester.validate_dependent_data(filtered_samples, groups)
                if not validation["valid"]:
                    error_message = "Error validating dependent data:\n" + "\n".join(validation["messages"])
                    analysis_log += f"\n{error_message}\n"
                    if not kwargs.get('force_continue', False):
                        print(f"WARNING: {error_message}")
                        analysis_log += "\nAnalysis continues with warning, results may be unreliable."
                        

            # Perform the appropriate statistical test - only call ONCE
            if kwargs.get('test') == 'mixed_anova':
                additional_factors = kwargs.get('additional_factors', [])
                subject_column = kwargs.get('subject_column') or kwargs.get('analysis_context', {}).get('subject_column') or 'Subject'
                if len(additional_factors) >= 2:
                    between_factor, within_factor = additional_factors[0], additional_factors[1]
                else:
                    return {"error": "Mixed ANOVA requires two factors (between and within)"}
                # Step 3: Call prepare_advanced_test first
                prep = StatisticalTester.prepare_advanced_test(
                    df, 'mixed_anova', value_cols[0], subject_column, [between_factor], [within_factor]
                )
                if "error" in prep:
                    return prep  # or handle error

                # Step 4: Pass outputs to perform_advanced_test
                results = StatisticalTester.perform_advanced_test(
                    df=df,
                    test='mixed_anova',
                    dv=value_cols[0],
                    subject=subject_column,
                    between=[between_factor],
                    within=[within_factor],
                    alpha=0.05,
                    transformed_samples=prep["transformed_samples"],
                    recommendation=prep["recommendation"],
                    test_info=prep["test_info"],
                    transform_fn=None,
                    force_parametric=kwargs.get('force_parametric', False),
                    skip_excel=True,
                    file_name=file_name
                )
                # Get the transformation type from the test_info
                requested_transform = prep["test_info"].get("transformation", "None")
                print(f"DEBUG: Requested transformation: {requested_transform}")
                print(f"DEBUG: Applied transformation: {results.get('transformation')}")
                # For consistency with the rest of the code, assign results to test_results
                test_results = results
                # Also extract the test_info and other variables for the rest of the code
                test_info = prep["test_info"]
                test_recommendation = prep["recommendation"]
                transformed_samples = prep["transformed_samples"]
            elif kwargs.get('test') == 'two_way_anova':
                between_factors = kwargs.get('additional_factors', [])
                # Step 3: Call prepare_advanced_test first
                prep = StatisticalTester.prepare_advanced_test(
                    df, 'two_way_anova', value_cols[0], None, between_factors, None
                )
                if "error" in prep:
                    return prep  # or handle error

                # Step 4: Pass outputs to perform_advanced_test
                results = StatisticalTester.perform_advanced_test(
                    df=df,
                    test='two_way_anova',
                    dv=value_cols[0],
                    subject=None,
                    between=between_factors,
                    within=None,
                    alpha=0.05,
                    transformed_samples=prep["transformed_samples"],
                    recommendation=prep["recommendation"],
                    test_info=prep["test_info"],
                    transform_fn=None,
                    force_parametric=kwargs.get('force_parametric', False),
                    skip_excel=True,
                    file_name=file_name
                )
                # Get the transformation type from the test_info
                requested_transform = prep["test_info"].get("transformation", "None")
                print(f"DEBUG: Requested transformation: {requested_transform}")
                print(f"DEBUG: Applied transformation: {results.get('transformation')}")
                # For consistency with the rest of the code, assign results to test_results
                test_results = results
                # Also extract the test_info and other variables for the rest of the code
                test_info = prep["test_info"]
                test_recommendation = prep["recommendation"]
                transformed_samples = prep["transformed_samples"]
            elif kwargs.get('test') == 'repeated_measures_anova':
                additional_factors = kwargs.get('additional_factors', [])
                subject_column = kwargs.get('subject_column') or kwargs.get('analysis_context', {}).get('subject_column') or 'Subject'
                if len(additional_factors) >= 1:
                    within_factor = additional_factors[0]  # RM-ANOVA uses within factor
                else:
                    return {"error": "Repeated measures ANOVA requires at least one within factor"}
                # Step 3: Call prepare_advanced_test first
                prep = StatisticalTester.prepare_advanced_test(
                    df, 'repeated_measures_anova', value_cols[0], subject_column, None, [within_factor]
                )
                if "error" in prep:
                    return prep  # or handle error

                # Step 4: Pass outputs to perform_advanced_test
                results = StatisticalTester.perform_advanced_test(
                    df=df,
                    test='repeated_measures_anova',
                    dv=value_cols[0],
                    subject=subject_column,
                    between=None,
                    within=[within_factor],
                    alpha=0.05,
                    transformed_samples=prep["transformed_samples"],
                    recommendation=prep["recommendation"],
                    test_info=prep["test_info"],
                    transform_fn=None,
                    force_parametric=kwargs.get('force_parametric', False),
                    skip_excel=True,
                    file_name=file_name
                )
                # Get the transformation type from the test_info
                requested_transform = prep["test_info"].get("transformation", "None")
                print(f"DEBUG: Requested transformation: {requested_transform}")
                print(f"DEBUG: Applied transformation: {results.get('transformation')}")
                # For consistency with the rest of the code, assign results to test_results
                test_results = results
                # Also extract the test_info and other variables for the rest of the code
                test_info = prep["test_info"]
                test_recommendation = prep["recommendation"]
                transformed_samples = prep["transformed_samples"]
            else:
                # Standard path for simple tests
                test_results = StatisticalTester.perform_statistical_test(
                    groups, transformed_samples, filtered_samples,
                    dependent=dependent, test_recommendation=test_recommendation, test_info=test_info
                )            # Log before transformation (only for standard tests that went through normality checking)
            if test_info:
                analysis_log += "\nResults of assumption tests before transformation:\n"
                
                # Get residual normality from new structure
                pre_residual_norm = test_info.get("pre_transformation", {}).get("residuals_normality")
                if pre_residual_norm and pre_residual_norm.get("p_value") is not None:
                    analysis_log += f"Shapiro-Wilk test (model residuals normality): p = {pre_residual_norm['p_value']:.4f} - "
                    analysis_log += "Model residuals normally distributed\n" if pre_residual_norm.get('is_normal', False) else "Model residuals not normally distributed\n"
                else:
                    analysis_log += "Shapiro-Wilk test (model residuals): Test not performed (insufficient data)\n"

                # Get variance test from new structure
                pre_variance = test_info.get("pre_transformation", {}).get("variance")
                if pre_variance and pre_variance.get("p_value") is not None:
                    analysis_log += f"Brown-Forsythe test (variance homogeneity): p = {pre_variance['p_value']:.4f} - "
                    analysis_log += "Variances homogeneous\n" if pre_variance.get('equal_variance', False) else "Variances heterogeneous\n"
                else:
                    analysis_log += "Brown-Forsythe test: Not performed (insufficient data)\n"

                # Log transformation
                if test_info.get("transformation"):
                    analysis_log += f"\nTransformation: {test_info['transformation'].capitalize()} transformation performed.\n"
                    # Log after transformation
                    analysis_log += "Results of assumption tests after transformation:\n"
                    
                    # Get post-transformation residual normality
                    post_residual_norm = test_info.get("post_transformation", {}).get("residuals_normality")
                    if post_residual_norm and post_residual_norm.get("p_value") is not None:
                        analysis_log += f"Shapiro-Wilk test (transformed model residuals): p = {post_residual_norm['p_value']:.4f} - "
                        analysis_log += "Transformed model residuals normally distributed\n" if post_residual_norm.get('is_normal', False) else "Transformed model residuals not normally distributed\n"
                    else:
                        analysis_log += "Shapiro-Wilk test (transformed model residuals): Test not performed (insufficient data)\n"
                    
                    # Get post-transformation variance
                    post_variance = test_info.get("post_transformation", {}).get("variance")
                    if post_variance and post_variance.get("p_value") is not None:
                        analysis_log += f"Brown-Forsythe test (transformed data variance homogeneity): p = {post_variance['p_value']:.4f} - "
                        analysis_log += "Transformed data variances homogeneous\n" if post_variance.get('equal_variance', False) else "Transformed data variances heterogeneous\n"
                    else:
                        analysis_log += "Brown-Forsythe test (transformed data): Not performed (insufficient data)\n"
                else:
                    analysis_log += "\nTransformation: No transformation performed.\n"

            posthoc_results = None

            if test_results is not None and test_results.get('p_value') is not None and test_results['p_value'] < 0.05 and len(groups) > 2:
                # Significant result: perform post-hoc tests
                valid_groups = [g for g in groups if g in transformed_samples and len(transformed_samples[g]) > 0]
                print("DEBUG: valid_groups after filter:", valid_groups)
                print("DEBUG: transformed_samples:", {g: len(transformed_samples[g]) for g in transformed_samples})
                print("DEBUG: original_samples:", {g: len(filtered_samples[g]) for g in filtered_samples})

                test_name = test_results.get('test', '').lower()

                # Check if post-hoc tests have already been performed
                if not test_results.get('pairwise_comparisons'):
                    # Let the perform_refactored_posthoc_testing function handle dialog selection for all tests
                    if 'kruskal' in test_name or 'friedman' in test_name or test_recommendation == 'non_parametric':
                        print("DEBUG: Significant non-parametric test (section 2), calling perform_refactored_posthoc_testing without preset posthoc_choice")
                        posthoc_results = StatisticalTester.perform_refactored_posthoc_testing(
                            valid_groups,
                            transformed_samples,
                            test_recommendation,
                            alpha=0.05,
                            posthoc_choice=None,  # Let the function show the dialog
                            test_info=test_info,
                        )
                    else:
                        # Show dialog for parametric tests
                        posthoc_choice = UIDialogManager.select_posthoc_test_dialog(
                            progress_text=kwargs.get('dialog_progress', None),
                            column_name=kwargs.get('dialog_column', None)
                        )
                        if posthoc_choice and posthoc_choice != "none":
                            control_group = None
                            if posthoc_choice == "dunnett":
                                control_group = UIDialogManager.select_control_group_dialog(valid_groups)
                            elif posthoc_choice == "paired_custom":
                                # Handle paired custom directly here to avoid double dialog
                                pairs = UIDialogManager.select_custom_pairs_dialog(valid_groups)
                                if pairs:
                                    # Import required modules
                                    stats = get_scipy_stats()
                                    multipletests = get_statsmodels_multitest()
                                    
                                    # Paired t-tests for the selected pairs
                                    pvals, stats_list = [], []
                                    for g1, g2 in pairs:
                                        x, y = np.array(transformed_samples[g1]), np.array(transformed_samples[g2])
                                        tstat, p = stats.ttest_rel(x, y)
                                        stats_list.append(tstat)
                                        pvals.append(p)
                                        # Holm–Šidák correction
                                    reject, p_adj, _, _ = multipletests(pvals, alpha=0.05, method='holm-sidak')
                                    
                                    # Create results in the same format as other post-hoc tests
                                    posthoc_results = {
                                        "posthoc_test": "Custom paired t-tests (Holm-Sidak)",
                                        "pairwise_comparisons": [],
                                        "error": None
                                    }
                                    
                                    # Collect results
                                    for i, (g1, g2) in enumerate(pairs):
                                        ci = PostHocStatistics.calculate_ci_mean_diff(transformed_samples[g1], transformed_samples[g2], alpha=0.05, paired=True)
                                        d = PostHocStatistics.calculate_cohens_d(transformed_samples[g1], transformed_samples[g2], paired=True)
                                        PostHocAnalyzer.add_comparison(
                                            posthoc_results,
                                            group1=g1,
                                            group2=g2,
                                            test="Paired t-test (Holm-Sidak)",
                                            p_value=p_adj[i],
                                            statistic=stats_list[i],
                                            corrected=True,
                                            correction_method="Holm-Sidak",
                                            effect_size=d,
                                            effect_size_type="cohen_d",
                                            confidence_interval=ci,
                                            alpha=0.05
                                        )
                                else:
                                    posthoc_results = {
                                        "posthoc_test": "No pairs selected for custom paired t-tests",
                                        "pairwise_comparisons": [],
                                        "error": None
                                    }
                            else:
                                # For other parametric post-hoc tests, use the refactored function
                                posthoc_results = StatisticalTester.perform_refactored_posthoc_testing(
                                    valid_groups, transformed_samples, test_recommendation,
                                    alpha=0.05, posthoc_choice=posthoc_choice, control_group=control_group
                                )
                    # Process results uniformly - ONLY ONCE here!
                    if posthoc_results:                      
                        if posthoc_choice == "dunnett" and "control_group" in posthoc_results:
                            test_results["control_group"] = posthoc_results["control_group"]
                        if 'pairwise_comparisons' in posthoc_results:
                            import copy
                            test_results['pairwise_comparisons'] = copy.deepcopy(posthoc_results['pairwise_comparisons'])
      
                            print(f"DEBUG: Copied pairwise_comparisons from posthoc_results to test_results")
                            print(f"DEBUG: Same object? {posthoc_results.get('pairwise_comparisons', None) is test_results.get('pairwise_comparisons', None)}")

                        test_results["posthoc_test"] = posthoc_results.get("posthoc_test")

                        # Add debug print to verify
                        print(f"DEBUG: posthoc_results['pairwise_comparisons'] length: {len(posthoc_results.get('pairwise_comparisons', []))}")

                    # INSERT DEBUG OUTPUTS HERE
                    print(f"DEBUG: Pairwise comparisons after post-hoc: {len(test_results.get('pairwise_comparisons', []))}")
                    
            # After post-hoc processing, before test_results.update:
            print(f"DEBUG: Post-hoc results: {posthoc_results.keys() if posthoc_results else None}")
            if posthoc_results and 'error' in posthoc_results and posthoc_results['error']:
                print(f"DEBUG: Post-hoc ERROR: {posthoc_results['error']}")
            print(f"DEBUG: test_results pairwise_comparisons: {len(test_results.get('pairwise_comparisons', []))} items")        


            # Make sure normality and variance test results are explicitly set (only if available)
            # Convert new test_info structure to the expected format for Excel export
            if test_info:
                print(f"DEBUG TEST_INFO STRUCTURE: {test_info}")
                print(f"DEBUG TEST_INFO KEYS: {list(test_info.keys())}")
                if "pre_transformation" in test_info:
                    print(f"DEBUG PRE_TRANSFORMATION: {test_info['pre_transformation']}")
                
                # Convert new residuals-based test info to compatible format
                normality_tests_compat = {}
                variance_test_compat = {}
                
                # Pre-transformation residual normality
                if "pre_transformation" in test_info and "residuals_normality" in test_info["pre_transformation"]:
                    pre_norm = test_info["pre_transformation"]["residuals_normality"]
                    normality_tests_compat["model_residuals"] = {
                        "statistic": pre_norm.get("statistic"),
                        "p_value": pre_norm.get("p_value"),
                        "is_normal": pre_norm.get("is_normal", False),
                        "test_type": "Shapiro-Wilk (Model Residuals)"
                    }
                
                transformation_applied = bool(
                    test_info.get("transformation")
                    and str(test_info.get("transformation")).lower() not in ("none", "no further")
                )

                # Post-transformation residual normality
                if transformation_applied and "post_transformation" in test_info and "residuals_normality" in test_info["post_transformation"]:
                    post_norm = test_info["post_transformation"]["residuals_normality"]
                    normality_tests_compat["model_residuals_transformed"] = {
                        "statistic": post_norm.get("statistic"),
                        "p_value": post_norm.get("p_value"),
                        "is_normal": post_norm.get("is_normal", False),
                        "test_type": "Shapiro-Wilk (Transformed Model Residuals)"
                    }
                
                # Variance tests
                if "pre_transformation" in test_info and "variance" in test_info["pre_transformation"]:
                    pre_var = test_info["pre_transformation"]["variance"]
                    variance_test_compat.update({
                        "statistic": pre_var.get("statistic"),
                        "p_value": pre_var.get("p_value"),
                        "equal_variance": pre_var.get("equal_variance", False)
                    })
                    print(f"DEBUG VARIANCE_TEST_COMPAT: {variance_test_compat}")
                
                if transformation_applied and "post_transformation" in test_info and "variance" in test_info["post_transformation"]:
                    post_var = test_info["post_transformation"]["variance"]
                    variance_test_compat["transformed"] = {
                        "statistic": post_var.get("statistic"),
                        "p_value": post_var.get("p_value"),
                        "equal_variance": post_var.get("equal_variance", False)
                    }
                
                results["normality_tests"] = normality_tests_compat
                results["variance_test"] = variance_test_compat
                
                print(f"DEBUG FINAL normality_tests: {results['normality_tests']}")
                print(f"DEBUG FINAL variance_test: {results['variance_test']}")
                
                # Add test_info for complete information
                results["test_info"] = test_info
            else:
                print("DEBUG: test_info is None or empty!")

            # Make sure test_type/recommendation is set (only if available):
            if test_recommendation:
                results["recommendation"] = test_recommendation

            # Merge important transformation and test info into results
            results.update(test_results)
            
            # Store normality_tests and variance_test before they get overwritten
            preserved_normality = results.get("normality_tests", {})
            preserved_variance = results.get("variance_test", {})
            
            results.update({
                "transformed_samples": transformed_samples,
                "samples": filtered_samples,
                "transformation": test_info.get("transformation") if test_info else None,
                "test_type": test_recommendation
            })

            final_test_label = (
                results.get("final_test_label")
                or results.get("tested_against")
                or results.get("test")
                or test_results.get("test")
                or kwargs.get("test")
            )
            if final_test_label:
                results["final_test_label"] = final_test_label
                results["tested_against"] = final_test_label

            # Restore the correctly formatted test data (don't overwrite with empty data!)
            if preserved_normality:
                results["normality_tests"] = preserved_normality
            elif test_info and "normality_tests" in test_info:
                results["normality_tests"] = test_info["normality_tests"]
            else:
                results["normality_tests"] = {}
                
            if preserved_variance:
                results["variance_test"] = preserved_variance
            elif test_info and "variance_test" in test_info:
                results["variance_test"] = test_info["variance_test"]  
            else:
                results["variance_test"] = {}

            # Nach results.update(test_results):
            print(f"DEBUG: results pairwise_comparisons: {len(results.get('pairwise_comparisons', []))} items")            

            if test_info and "boxcox_lambda" in test_info:
                results["boxcox_lambda"] = test_info["boxcox_lambda"]

            analysis_log += f"\nTest performed: {results.get('test', 'Not specified')}\n"
            if 'p_value' in results:
                p_value = results['p_value']
                if isinstance(p_value, (float, int)):
                    analysis_log += f"p-Value: {p_value:.6f}\n"
                    analysis_log += f"Significance: {'Significant (p < 0.05)' if p_value < 0.05 else 'Not significant'}\n"
                else:
                    analysis_log += f"p-Value: {p_value}\n"
                    analysis_log += "Significance: Not determinable\n"

            # For Two-way/Mixed/RM-ANOVA: main effects and interactions in the log
            if "factors" in results:
                for factor in results["factors"]:
                    analysis_log += (
                        f"Main effect {factor['factor']}: "
                        f"F({factor['df1']}, {factor['df2']}) = {factor['F']:.3f}, "
                        f"p = {factor['p_value']:.4f}, "
                        f"Effect size: {factor.get('effect_size', 'N/A')}\n"
                    )
            if "interactions" in results:
                for inter in results["interactions"]:
                    analysis_log += (
                        f"Interaction {inter['factors'][0]} x {inter['factors'][1]}: "
                        f"F({inter['df1']}, {inter['df2']}) = {inter['F']:.3f}, "
                        f"p = {inter['p_value']:.4f}, "
                        f"Effect size: {inter.get('effect_size', 'N/A')}\n"
                    )

            # Post-hoc tests
            if 'pairwise_comparisons' in results and results['pairwise_comparisons']:
                posthoc_test = results.get("posthoc_test", None)
                if posthoc_test and "Tukey HSD" in posthoc_test:
                    posthoc_display = "Tukey HSD Test"
                elif posthoc_test and "Dunnett Test" in posthoc_test:
                    control_group = results.get("control_group", "")
                    posthoc_display = f"Dunnett Test (Control group: {control_group})"
                else:
                    posthoc_display = posthoc_test if posthoc_test else "No post-hoc test performed"

                analysis_log += f"\nPost‑hoc test: {posthoc_display}\n"
                analysis_log += "Pairwise comparisons:\n"
                for comp in results["pairwise_comparisons"]:
                    group1 = str(comp['group1'])
                    group2 = str(comp['group2'])
                    p_val = comp['p_value']
                    significant = comp['significant']
                    p_text = "p < 0.001" if isinstance(p_val, (float, int)) and p_val < 0.001 else safe_format(p_val, "p = {:.4f}")
                    sign_text = "significant" if significant else "not significant"
                    stars = "***" if significant and p_val < 0.001 else "**" if significant and p_val < 0.01 else "*" if significant else ""
                    analysis_log += f"  {group1} vs {group2}: {p_text}, {sign_text} {stars}\n"
            else:
                analysis_log += "\nNo pairwise comparisons were performed or calculated.\n"

            # Define file base based on custom name or groups
            if file_name:
                file_base = file_name
            else:
                file_base = "_".join(map(str, groups))

            excel_file = f"{file_base}_results.xlsx"
            
            results['groups'] = groups
            results['raw_data'] = {g: filtered_samples[g][:] for g in groups}
            if results.get('transformation', 'None') != 'None':
                results['raw_data_transformed'] = {g: transformed_samples[g][:] for g in groups}
            analysis_context = kwargs.get('analysis_context', {})
            results['selected_groups'] = analysis_context.get('selected_groups') or groups
            results['group_column'] = analysis_context.get('selected_group_column') or analysis_context.get('factor_columns', [None])[0] or group_col
            results['factor_columns'] = analysis_context.get('factor_columns', [])
            results['covariates'] = analysis_context.get('covariates', [])
            results['dependent_variable'] = value_cols[0]
            if analysis_context.get('filter'):
                filter_col, filter_val = analysis_context['filter']
                results['filter_applied'] = f"{filter_col} = {filter_val}"
            
            # DO NOT OVERWRITE! The variance_test and normality_tests are already set above
            # Keep the old format for backward compatibility if variance_test doesn't exist
            if "variance_test" not in results:
                results["variance_homogeneity_test"] = test_info.get("variance_test", {}) if test_info else {}    

            # Add debug statements before Excel export
            print("DEBUG: Assumption tests before Excel export:")
            print("  Normality tests:", results.get("normality_tests", {}))
            print("  Variance tests:", results.get("variance_test", {}))
            print("  Test recommendation:", test_recommendation)
                
            # Export to Excel
            if not skip_excel:
                original_dir = os.getcwd()
                print(f"DEBUG: Directory before Excel export: {original_dir}")
                
                # Use absolute path for Excel file
                excel_file = get_output_path(file_base, "xlsx") 
                
                ExportDispatcher = get_export_dispatcher()
                export_result = ExportDispatcher.export_analysis_results(results, excel_file, analysis_log)
                if export_result.get("warning"):
                    print(f"WARNING: {export_result['warning']}")
                excel_file = export_result.get("excel_path", excel_file)
                analysis_log += f"\nResults were saved to {excel_file}.\n"
                
                # Ensure we're back in the original directory
                if os.getcwd() != original_dir:
                    os.chdir(original_dir)
                    print(f"DEBUG: Restored original directory: {original_dir}")

            # Create the plot, if not skipped
            if not skip_plots:
                print(f"DEBUG: Current working directory before export: {os.getcwd()}")
                pairwise_comparisons = results.get('pairwise_comparisons', None)
                
                # Get plot type from kwargs, default to 'Bar'
                plot_type = kwargs.get('plot_type', 'Bar')
                print(f"DEBUG: Creating plot of type: {plot_type}")
                
                # Create a clean kwargs dict without parameters that plotting methods don't accept
                # Only exclude parameters that definitely don't exist in plot methods
                plot_kwargs = {k: v for k, v in kwargs.items() if k not in [
                    'plot_type', 'file_path', 'group_col', 'groups', 'sheet_name',
                    'value_cols', 'combine_columns', 'skip_plots', 'skip_excel',
                    'dependent', 'show_individual_lines', 'compare', 'additional_factors',
                    'dataset_name', 'dialog_column', 'dialog_progress',
                    # Parameters that don't exist in plot_bar method
                    'aspect',
                    'refline', 'panel_labels', 'value_annotations', 'significance_mode',
                    'embed_fonts', 'add_metadata',
                    # Legacy keys that are not supported by plotting signatures
                    'font_main', 'font_axis', 'axis_linewidth', 'gridline_width',
                    # Analysis metadata — not a plot parameter
                    'analysis_context',
                ]}
                
                # Choose the appropriate plot function based on plot_type
                if plot_type == "Bar":
                    plot_kwargs['show_points'] = plot_kwargs.get('show_points', True)
                    plot_kwargs['point_size'] = plot_kwargs.get('point_size', 80)
                    plot_kwargs['point_alpha'] = plot_kwargs.get('point_alpha', 0.8)
                    # Always pass colors to legend
                    fig, ax = DataVisualizer.plot_bar(
                        groups, filtered_samples, width=width, height=height,
                        colors=colors, hatches=hatches, compare=compare,
                        test_recommendation=test_recommendation,
                        x_label=x_label, y_label=y_label,
                        title=title, save_plot=save_plot, error_type=error_type,
                        pairwise_results=pairwise_comparisons,
                        file_name=file_base, legend_colors=colors, **plot_kwargs)
                elif plot_type == "Box":
                    fig, ax = DataVisualizer.plot_box(
                        groups, filtered_samples, width=width, height=height,
                        colors=colors, hatches=hatches,
                        test_recommendation=test_recommendation,
                        x_label=x_label, y_label=y_label,
                        title=title, save_plot=save_plot,
                        pairwise_results=pairwise_comparisons,
                        file_name=file_base, legend_colors=colors, **plot_kwargs)
                elif plot_type == "Violin":
                    fig, ax = DataVisualizer.plot_violin(
                        groups, filtered_samples, width=width, height=height,
                        colors=colors, hatches=hatches,
                        test_recommendation=test_recommendation,
                        x_label=x_label, y_label=y_label,
                        title=title, save_plot=save_plot,
                        pairwise_results=pairwise_comparisons,
                        file_name=file_base, legend_colors=colors, **plot_kwargs)
                elif plot_type == "Strip":
                    # Strip plot doesn't exist, fall back to box plot with points
                    plot_kwargs['show_points'] = plot_kwargs.get('show_points', True)
                    plot_kwargs['point_size'] = plot_kwargs.get('point_size', 80)
                    plot_kwargs['point_alpha'] = plot_kwargs.get('point_alpha', 0.8)
                    fig, ax = DataVisualizer.plot_box(
                        groups, filtered_samples, width=width, height=height,
                        colors=colors, hatches=hatches,
                        test_recommendation=test_recommendation,
                        x_label=x_label, y_label=y_label,
                        title=title, save_plot=save_plot,
                        pairwise_results=pairwise_comparisons,
                        file_name=file_base, legend_colors=colors, **plot_kwargs)
                elif plot_type == "Raincloud":
                    fig, ax = DataVisualizer.plot_raincloud(
                        groups, filtered_samples, width=width, height=height,
                        colors=colors, hatches=hatches,
                        test_recommendation=test_recommendation,
                        x_label=x_label, y_label=y_label,
                        title=title, save_plot=save_plot,
                        pairwise_results=pairwise_comparisons,
                        file_name=file_base, legend_colors=colors, **plot_kwargs)
                else:
                    # Fallback to bar plot for unknown plot types
                    print(f"WARNING: Unknown plot type '{plot_type}', falling back to Bar plot")
                    plot_kwargs['show_points'] = plot_kwargs.get('show_points', True)
                    plot_kwargs['point_size'] = plot_kwargs.get('point_size', 80)
                    plot_kwargs['point_alpha'] = plot_kwargs.get('point_alpha', 0.8)
                    fig, ax = DataVisualizer.plot_bar(
                        groups, filtered_samples, width=width, height=height,
                        colors=colors, hatches=hatches, compare=compare,
                        test_recommendation=test_recommendation,
                        x_label=x_label, y_label=y_label,
                        title=title, save_plot=save_plot, error_type=error_type,
                        pairwise_results=pairwise_comparisons,
                        file_name=file_base, legend_colors=colors, **plot_kwargs)
                analysis_log += f"\nPlots were saved as:\n"
                analysis_log += f"  {file_base}.pdf\n"
                analysis_log += f"  {file_base}.png\n"
                get_matplotlib_pyplot().close(fig)
                results["_file_paths"] = {
                    "excel": os.path.abspath(excel_file),
                    "pdf": os.path.abspath(f"{file_base}.pdf"),
                    "png": os.path.abspath(f"{file_base}.png")
                }
            else:
                results["_file_paths"] = {
                    "excel": os.path.abspath(excel_file)
                }
            # Special visualization for dependent data
            if dependent and not skip_plots:
                try:
                    line_fig, line_ax = DataVisualizer.plot_dependent_samples(
                        groups, filtered_samples, width=width, height=height,
                        colors=colors, title=f"{title} (dependent measurements)" if title else "Dependent measurements",
                        x_label=x_label, y_label=y_label,
                        save_plot=save_plot, file_name=file_base+"_lines",
                        show_individual=show_individual_lines
                    )
                    get_matplotlib_pyplot().close(line_fig)
                    line_plot_base = file_base+"_lines"
                    results["_file_paths"]["pdf_lines"] = os.path.abspath(f"{line_plot_base}.pdf")
                    results["_file_paths"]["png_lines"] = os.path.abspath(f"{line_plot_base}.png")
                    analysis_log += f"\nAdditional line plot for dependent data created:\n"
                    analysis_log += f"  {line_plot_base}.pdf\n"
                    analysis_log += f"  {line_plot_base}.png\n"
                except Exception as e:
                    analysis_log += f"\nError creating line plot for dependent data: {str(e)}\n"
                    print(f"Error creating line plot: {str(e)}")
            if results.get('transformation', 'None') != 'None':
                results['transformed_data'] = transformed_samples

            # About line 5647, just before "return results"
            params = {
                "file_path": file_path,
                "sheet_name": sheet_name,
                "group_col": group_col,
                "value_cols": value_cols,
                "groups": groups,
                "dependent": dependent,
                "error_type": error_type
            }
            # Build protocol
            def build_analysis_log(results, params):
                log = []
                log.append("ANALYSIS LOG")
                log.append('"This sheet documents the course of the statistical analysis and the decisions made. The log provides a chronological overview of the individual analysis steps, methods used, transformations, test selection, and special notes.\nEach paragraph describes a key step or decision in the analysis process."\n')
                log.append(f"Analysis report\nDate and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                if 'file_path' in params:
                    log.append(f"File: {params['file_path']}")
                if 'sheet_name' in params:
                    log.append(f"Worksheet: {params['sheet_name']}")
                if 'group_col' in params:
                    log.append(f"Group column: {params['group_col']}")
                if 'value_cols' in params:
                    log.append(f"Value column(s): {', '.join(params['value_cols'])}")
                if 'groups' in params:
                    log.append(f"Groups to analyze: {', '.join(params['groups'])}")
                if 'dependent' in params:
                    log.append(f"Dependent samples: {'Yes' if params['dependent'] else 'No'}")
                if 'error_type' in params:
                    log.append(f"Error bar type: {'SEM (standard error)' if params['error_type']=='se' else 'SD (standard deviation)'}")
                log.append("\n--- ANALYSIS ---\n")
                if results.get('import_status'):
                    log.append("Data imported successfully.")
                if 'group_sizes' in results:
                    log.append("Number of data points per group:")
                    for group, n in results['group_sizes'].items():
                        log.append(f"{group}: {n} data points")
                if 'test_recommendation' in results:
                    log.append(f"Test recommendation: {results['test_recommendation']}")
                if 'normality_p' in results:
                    log.append(f"Shapiro-Wilk test (normality): p = {results['normality_p']:.4f} - {'Normally distributed' if results['normality_p'] > 0.05 else 'Not normally distributed'}")
                if 'levene_p' in results:
                    log.append(f"Brown-Forsythe test (variance homogeneity): p = {results['levene_p']:.4f} - {'Variances homogeneous' if results['levene_p'] > 0.05 else 'Variances heterogeneous'}")
                if 'transformation' in results:
                    log.append(f"Transformation: {results['transformation'] if results['transformation'] else 'No transformation performed.'}")
                if 'test' in results:
                    log.append(f"Test performed: {results['test']}")
                if 'p_value' in results:
                    p_value = results['p_value']
                    if p_value is None:
                        log.append("p-Value: Not available (test may have failed)")
                        if 'error' in results and results['error']:
                            log.append(f"Error: {results['error']}")
                    elif isinstance(p_value, (float, int)):
                        log.append(f"p-Value: {p_value:.6f}")
                        log.append(f"Significance: {'Significant (p < 0.05)' if p_value < 0.05 else 'Not significant'}")
                    else:
                        log.append(f"p-Value: {p_value}")
                        log.append("Significance: Not determinable")
                if "factors" in results:
                    for factor in results["factors"]:
                        if not isinstance(factor, dict):
                            continue
                        log.append(
                            f"Main effect {factor['factor']}: F({factor['df1']}, {factor['df2']}) = {factor['F']:.3f}, "
                            f"p = {factor['p_value']:.4f}, Effect size: {factor.get('effect_size', 'N/A')}"
                        )
                if "interactions" in results:
                    for inter in results["interactions"]:
                        if not isinstance(inter, dict):
                            continue
                        inter_factors = inter.get('factors') if isinstance(inter.get('factors'), list) else []
                        inter_a = inter_factors[0] if len(inter_factors) > 0 else "Factor 1"
                        inter_b = inter_factors[1] if len(inter_factors) > 1 else "Factor 2"
                        log.append(
                            f"Interaction {inter_a} x {inter_b}: F({inter['df1']}, {inter['df2']}) = {inter['F']:.3f}, "
                            f"p = {inter['p_value']:.4f}, Effect size: {inter.get('effect_size', 'N/A')}"
                        )
                if 'pairwise_comparisons' in results and results['pairwise_comparisons']:
                    posthoc_test = results.get("posthoc_test", "Post-hoc test")
                    log.append(f"\nPost‑hoc test: {posthoc_test}")
                    log.append("Pairwise comparisons:")
                    for comp in results["pairwise_comparisons"]:
                        group1 = str(comp['group1'])
                        group2 = str(comp['group2'])
                        p_val = comp['p_value']
                        significant = comp['significant']
                        p_text = "p < 0.001" if isinstance(p_val, (float, int)) and p_val < 0.001 else f"p = {p_val:.4f}"
                        sign_text = "significant" if significant else "not significant"
                        stars = "***" if significant and p_val < 0.001 else "**" if significant and p_val < 0.01 else "*" if significant else ""
                        log.append(f"{group1} vs {group2}: {p_text}, {sign_text} {stars}")
                else:
                    log.append("\nNo pairwise comparisons were performed or calculated.")
                return "\n".join(log)
            analysis_log = build_analysis_log(results, params)
            results["analysis_log"] = analysis_log
            return results

        except Exception as e:
            error_message = str(e) if str(e) else "Unknown error occurred"
            analysis_log += f"\nERROR: {error_message}\n"
            print(f"Error during analysis: {error_message}")
            import traceback
            traceback.print_exc()
            return {"error": error_message, "analysis_log": analysis_log}       

def get_output_path(file_base, ext):
    """Get an absolute path to save output files on desktop."""
    if os.path.isabs(file_base):
        abs_path = os.path.abspath(f"{file_base}.{ext}")
        print(f"DEBUG: get_output_path returns absolute path from absolute base: {abs_path}")
        return abs_path

    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    if not os.path.isdir(desktop_path):
        # Fallback: current working directory
        desktop_path = os.getcwd()
    
    out_path = os.path.join(desktop_path, f"{file_base}.{ext}")
    abs_path = os.path.abspath(out_path)
    print(f"DEBUG: get_output_path returns absolute path: {abs_path}")
    return abs_path

