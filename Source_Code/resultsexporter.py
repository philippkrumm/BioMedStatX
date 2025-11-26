import xlsxwriter
import copy
import os
import pandas as pd

# Lazy import function to avoid circular imports
def get_assumption_visualizer():
    """Get AssumptionVisualizer class lazily"""
    from stats_functions import AssumptionVisualizer
    return AssumptionVisualizer
class ResultsExporter:
    """
    Advanced statistical results exporter with comprehensive Excel reporting capabilities.
    
    This class provides sophisticated Excel export functionality for various statistical analyses,
    with special emphasis on Repeated Measures ANOVA and Mixed ANOVA designs. The exporter
    creates comprehensive, user-friendly Excel reports with detailed explanations, assumptions
    testing, and practical recommendations.
    
    Key Features:
    ============
    - Multi-sheet Excel reports with professional formatting
    - Context-aware explanations for complex statistical concepts
    - Robust error handling and input validation
    - Comprehensive assumption testing for RM/Mixed ANOVA
    - Intelligent sphericity corrections with practical guidance
    - Visual indicators for assumption violations
    - Detailed pairwise comparisons and post-hoc analyses
    - Decision tree integration for method selection
    - Raw data preservation and analysis logs
    
    Specialized RM/Mixed ANOVA Support:
    ==================================
    - Enhanced sphericity testing with Mauchly's test
    - Greenhouse-Geisser and Huynh-Feldt corrections
    - Between-factor assumptions for Mixed ANOVA
    - Within-factor sphericity analysis
    - Interaction assumption testing
    - Automated correction recommendations based on epsilon values
    - Context-sensitive explanations for each assumption type
    
    Error Handling:
    ==============
    - Comprehensive input validation for all functions
    - Graceful degradation when data is missing or corrupted
    - Safe cell writing with fallback mechanisms
    - Detailed error reporting with practical solutions
    - Automatic cleanup of temporary files
    
    Usage Example:
    =============
    results = {
        'test': 'Repeated Measures ANOVA',
        'sphericity_test': {...},
        'sphericity_corrections': {...},
        # ... other results
    }
    
    ResultsExporter.export_results_to_excel(results, 'analysis_results.xlsx')
    
    Class Variables:
    ===============
    _temp_files : set
        Tracks temporary files for automatic cleanup
    """
    _temp_files = set()
    @staticmethod
    def export_results_to_excel(results, output_file, analysis_log=None):
        print(f"DEBUG: Current working directory before export: {os.getcwd()}")
        original_dir = os.getcwd()
        
        # Use absolute path for output file
        output_file = os.path.abspath(output_file)
        
        # Create a deep copy to prevent modifications during processing
        results_copy = copy.deepcopy(results)
        
        # Ensure pairwise_comparisons exists and is a list
        if 'pairwise_comparisons' not in results_copy:
            print("WARNING: No pairwise comparisons found, initializing empty list")
            results_copy['pairwise_comparisons'] = []
        elif not isinstance(results_copy['pairwise_comparisons'], list):
            print(f"WARNING: pairwise_comparisons is not a list, type: {type(results_copy['pairwise_comparisons'])}")
            results_copy['pairwise_comparisons'] = []
        
        print(f"DEBUG: Before Excel export - number of pairwise comparisons: {len(results_copy.get('pairwise_comparisons', []))}")
        
        # Initialize dataset_tree_paths for single dataset export
        dataset_tree_paths = {}
        
        workbook = xlsxwriter.Workbook(output_file, {'nan_inf_to_errors': True})
        fmt = ResultsExporter._get_excel_formats(workbook)

        ResultsExporter._write_summary_sheet(workbook, results, fmt)
        ResultsExporter._write_assumptions_sheet(workbook, results, fmt)
        ResultsExporter._write_results_sheet(workbook, results, fmt)
        ResultsExporter._write_descriptive_sheet(workbook, results, fmt)
        ResultsExporter._write_decision_tree_sheet(workbook, results, fmt)
        ResultsExporter._write_rawdata_sheet(workbook, results, fmt)
        ResultsExporter._write_pairwise_sheet(workbook, results, fmt)
        if analysis_log:
            ResultsExporter._write_analysislog_sheet(workbook, analysis_log, fmt)
            
        workbook.close()
        print(f"DEBUG: Excel export attempted to: {output_file}")
        print(f"DEBUG: Excel file exists after export: {os.path.exists(output_file)}")

        # Clean up all temporary decision tree files
        for dataset_name, tree_path in dataset_tree_paths.items():
            if tree_path and os.path.exists(tree_path):
                try:
                    os.remove(tree_path)
                    print(f"DEBUG MULTI: Cleaned up decision tree file for {dataset_name}: {tree_path}")
                except Exception as e:
                    print(f"DEBUG MULTI: Could not clean up {tree_path}: {str(e)}")

        # Clean up any other tracked temporary files
        if ResultsExporter._temp_files:
            print(f"DEBUG MULTI: Cleaning up {len(ResultsExporter._temp_files)} tracked temporary files")
            for temp_file in ResultsExporter._temp_files:
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                        print(f"DEBUG MULTI: Removed tracked temp file: {temp_file}")
                    except Exception as e:
                        print(f"DEBUG MULTI: Failed to remove temp file: {str(e)}")
            ResultsExporter._temp_files.clear()

    @staticmethod
    def export_multi_dataset_results(all_results, excel_path):
        print(f"DEBUG: Current working directory before export: {os.getcwd()}")
        print(f"DEBUG MULTI: export_multi_dataset_results called with excel_path='{excel_path}'")
        print("DEBUG MULTI: Received all_results with contents:")   
        for ds_name, results in all_results.items():
            print(f"  Dataset: {ds_name} → Keys in results: {list(results.keys())}")
            print(f"    p_value: {results.get('p_value')} | pairwise_comparisons: {len(results.get('pairwise_comparisons', []))}")
        
        """Exports the results of all dataset analyses into a shared Excel file."""
        # import os  # Already imported at top
        import time
        import xlsxwriter
        from decisiontreevisualizer import DecisionTreeVisualizer
        
        # Create a dictionary to track all decision tree images for this multi-dataset export
        dataset_tree_paths = {}
        
        # Generate all decision trees first, before creating the workbook
        for dataset_name, results in all_results.items():
            print(f"DEBUG MULTI: Pre-generating decision tree for {dataset_name}...")
            # Generate decision tree and track the file path
            tree_path = DecisionTreeVisualizer.generate_and_save_for_excel(results)
            if tree_path and os.path.exists(tree_path):
                print(f"DEBUG MULTI: Generated decision tree for {dataset_name}: {tree_path}")
                # Store in dictionary mapping dataset to file path
                dataset_tree_paths[dataset_name] = tree_path
            else:
                print(f"DEBUG MULTI: Warning - Failed to generate decision tree for {dataset_name}")
        
        # Now create the Excel workbook with all necessary formats
        workbook = xlsxwriter.Workbook(excel_path, {'nan_inf_to_errors': True})
        fmt = ResultsExporter._get_excel_formats(workbook)
        
        # DEBUG: Print available format keys
        print(f"DEBUG MULTI: Available format keys: {list(fmt.keys())}")
        
        # Create an overview sheet
        overview_sheet = workbook.add_worksheet("Overview")
        overview_sheet.set_column('A:A', 30)
        overview_sheet.set_column('B:E', 15)
        
        # Write overview headers
        overview_sheet.write(0, 0, "Dataset", fmt["header"])
        overview_sheet.write(0, 1, "Test", fmt["header"])
        overview_sheet.write(0, 2, "p-value", fmt["header"])
        overview_sheet.write(0, 3, "Significant", fmt["header"])
        overview_sheet.write(0, 4, "Transformation", fmt["header"])
        
        # For each dataset: write overview row with basic info
        row = 1
        for dataset_name, results in all_results.items():
            overview_sheet.write(row, 0, str(dataset_name), fmt["header"])
            overview_sheet.write(row, 1, str(results.get("test", "N/A")), fmt["cell"])
            
            p_value = results.get("p_value", None)
            if p_value is not None and isinstance(p_value, (float, int)):
                if p_value < 0.001:
                    overview_sheet.write(row, 2, "<0.001", fmt["cell"])
                else:
                    overview_sheet.write(row, 2, f"{p_value:.4f}", fmt["cell"])
            else:
                overview_sheet.write(row, 2, "N/A", fmt["cell"])
            
            is_significant = p_value is not None and isinstance(p_value, (float, int)) and p_value < 0.05
            sig_fmt = fmt["significant"] if is_significant else fmt["cell"]
            overview_sheet.write(row, 3, "Yes" if is_significant else "No", sig_fmt)
            
            transformation = results.get("transformation", "None")
            overview_sheet.write(row, 4, str(transformation), fmt["cell"])
            
            row += 1
            
        # Add detailed information for each dataset
        row += 2  # Add some space
        for dataset_name, results in all_results.items():
            # Dataset header
            overview_sheet.merge_range(f'A{row}:E{row}', f"DATASET: {dataset_name}", fmt["title"])
            row += 1
            
            # RAW DATA section
            overview_sheet.merge_range(f'A{row}:E{row}', "RAW DATA", fmt["section_header"])
            row += 1
            overview_sheet.write(row, 0, "These data are the basis of all calculations.", fmt["explanation"])
            row += 1  # FIX: Changed from row += 2 to row += 1 to prevent misalignment
            
            # Get raw data for this dataset
            raw_data = results.get("raw_data", results.get("original_data", {})) or {}
            print("DEBUG: raw_data keys:", list(raw_data.keys()))

            # Filtere evtl. "Group"-Key raus
            data_to_write = {k: v for k, v in raw_data.items() if str(k).lower() not in ["group", "sample", ""]}          

            row += 1  # die Zeile, in der gleich Group & Values stehen sollen

            overview_sheet.write(row, 0, "Group", fmt["header"])
            overview_sheet.write(row, 1, "Values", fmt["header"])
            row += 1
            for group_name, values in data_to_write.items():
                # 4) Gruppe in Spalte A
                overview_sheet.write(row, 0, group_name, fmt["cell"])
                # 5) Werte-String in Spalte B
                values_str = ", ".join([
                    f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
                    for v in values
                ])
                overview_sheet.write(row, 1, values_str, fmt["cell"])
                row += 1

            # Get raw data for this dataset and apply new alignment function
            raw_data = results.get("raw_data", results.get("original_data", {})) or {}

            print(f"DEBUG: Processing raw data for {dataset_name}")
            print(f"DEBUG: Raw data keys: {list(raw_data.keys())}")

            # TRANSFORMED DATA section for this dataset
            transformed_data = results.get("raw_data_transformed", results.get("transformed_data", {})) or {}
            transformation = results.get("transformation", "None")
            print("DEBUG: transformed_data keys:", list(transformed_data.keys()))
            # Only show transformed data if a transformation was performed
            if transformed_data and transformation and transformation.lower() != "none":
                print(f"DEBUG: Processing transformed data for {dataset_name}")
                print(f"DEBUG: Transformed data keys: {list(transformed_data.keys())}")
                
                # Use the same alignment function for transformed data
                transformed_to_write = transformed_data
                
                # Check if transformed data actually differ from raw data
                is_different = False
                if data_to_write and transformed_to_write:

                    if set(data_to_write.keys()) != set(transformed_to_write.keys()):
                        is_different = True
                    else:
                        for group in data_to_write:
                            if group in transformed_to_write:
 
                                raw_vals = data_to_write[group]
                                trans_vals = transformed_to_write[group]
                                if len(raw_vals) != len(trans_vals):
                                    is_different = True
                                    break

                                for r, t in zip(raw_vals, trans_vals):
                                    if abs(r - t) > 1e-10:
                                        is_different = True
                                        break
                                if is_different:
                                    break
                
                if is_different:
                    row += 1
                    transformed_to_write = {k: v for k, v in transformed_data.items() if str(k).lower() not in ["group", "sample", ""]}
                    overview_sheet.merge_range(f'A{row}:E{row}', "TRANSFORMED DATA", fmt["section_header"])
                    row += 1

                    overview_sheet.write(row, 0, "Group", fmt["header"])
                    overview_sheet.write(row, 1, "Values", fmt["header"])

                    row += 1
                    for group_name, values in transformed_to_write.items():
                        overview_sheet.write(row, 0, group_name, fmt["cell"])
                        values_str = ", ".join([
                            f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
                            for v in values
                        ])
                        overview_sheet.write(row, 1, values_str, fmt["cell"])
                        row += 1

            # PAIRWISE COMPARISONS section
            row += 2
            overview_sheet.merge_range(f'A{row}:E{row}', "PAIRWISE COMPARISONS", fmt["section_header"])
            row += 1
            
            # Headers for pairwise comparisons
            headers = ["Group 1", "Group 2", "Test", "p-Value", "Corrected"]
            for i, header in enumerate(headers):
                overview_sheet.write(row, i, header, fmt["header"])
            row += 1
            
            comps = results.get("pairwise_comparisons", [])
            if comps and len(comps) > 0:
                for comp in comps[:5]:  # Limit to first 5 comparisons to save space
                    group1 = str(comp.get('group1', 'N/A'))
                    group2 = str(comp.get('group2', 'N/A'))
                    test_name = comp.get('test', 'N/A')
                    pval = comp.get('p_value', None)
                    pval_str = "<0.001" if isinstance(pval, (float, int)) and pval < 0.001 else f"{pval:.4f}" if isinstance(pval, (float, int)) else "N/A"
                    corrected = "Yes" if comp.get('corrected', False) else "No"
                    
                    overview_sheet.write(row, 0, group1, fmt["cell"])
                    overview_sheet.write(row, 1, group2, fmt["cell"])
                    overview_sheet.write(row, 2, test_name, fmt["cell"])
                    overview_sheet.write(row, 3, pval_str, fmt["cell"])
                    overview_sheet.write(row, 4, corrected, fmt["cell"])
                    row += 1
                    
                if len(comps) > 5:
                    overview_sheet.merge_range(f'A{row}:E{row}', f"... and {len(comps) - 5} more comparisons (see {dataset_name}_Pairwise sheet)", fmt["explanation"])
                    row += 1
            else:
                message = "No pairwise comparisons performed or available."
                if p_value is not None and p_value >= results.get("alpha", 0.05) and len(results.get("groups", [])) > 2:
                    message = "No pairwise comparisons performed because the main test was not significant."
                
                overview_sheet.merge_range(f'A{row}:E{row}', message, fmt["cell"])
                row += 1
            
            # Add separator between datasets
            row += 3
        
        # For each dataset: create all detail sheets as in single analysis
        for dataset_name, results in all_results.items():
            # Use the pre-generated decision tree path
            pre_generated_tree = dataset_tree_paths.get(dataset_name)
            
            try:
                # Create all the detailed sheets for this dataset
                ResultsExporter._write_summary_sheet(workbook, results, fmt, f"{dataset_name}_Summary")
                ResultsExporter._write_assumptions_sheet(workbook, results, fmt, f"{dataset_name}_Assumptions")
                ResultsExporter._write_results_sheet(workbook, results, fmt, f"{dataset_name}_Results")
                ResultsExporter._write_descriptive_sheet(workbook, results, fmt, f"{dataset_name}_Descriptive")
                ResultsExporter._write_decision_tree_sheet(workbook, results, fmt, f"{dataset_name}_DecisionTree", pre_generated_tree)
                ResultsExporter._write_rawdata_sheet(workbook, results, fmt, f"{dataset_name}_RawData")
                ResultsExporter._write_pairwise_sheet(workbook, results, fmt, f"{dataset_name}_Pairwise")
                
                # Add analysis log if available
                if results.get("analysis_log"):
                    ResultsExporter._write_analysislog_sheet(workbook, results["analysis_log"], fmt, f"{dataset_name}_Log")
                    
            except Exception as e:
                print(f"DEBUG MULTI: Error creating sheets for {dataset_name}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Close the workbook to save changes
        workbook.close()
        print(f"DEBUG MULTI: Excel file created at {excel_path}")
        
        # Clean up all temporary decision tree files
        for dataset_name, tree_path in dataset_tree_paths.items():
            if tree_path and os.path.exists(tree_path):
                try:
                    os.remove(tree_path)
                    print(f"DEBUG MULTI: Cleaned up decision tree file for {dataset_name}")
                except Exception as e:
                    print(f"DEBUG MULTI: Could not clean up {tree_path}: {str(e)}")
        
        # Clean up any other tracked temporary files
        if hasattr(ResultsExporter, '_temp_files'):
            for temp_file in ResultsExporter._temp_files:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except Exception:
                        pass
            ResultsExporter._temp_files.clear()
        
        return excel_path

    @staticmethod
    def _get_excel_formats(workbook):
        return {
            "title": workbook.add_format({'bold': True, 'font_size': 14, 'align': 'center', 'valign': 'vcenter'}),
            "header": workbook.add_format({'bold': True, 'font_size': 12, 'align': 'center', 'bottom': 2}),
            "cell": workbook.add_format({'align': 'center', 'text_wrap': True}),
            "significant": workbook.add_format({'align': 'center', 'color': 'red', 'bold': True, 'text_wrap': True}),
            "explanation": workbook.add_format({'text_wrap': True, 'valign': 'top', 'font_color': '#1F4E78'}),
            "section_header": workbook.add_format({'bold': True, 'bg_color': '#B4C6E7', 'border': 1}),
            "section_header_center": workbook.add_format({'bold': True, 'bg_color': '#B4C6E7', 'border': 1, 'align': 'center'}),
            "effect_strong": workbook.add_format({'align': 'center', 'color': '#006400', 'bold': True, 'text_wrap': True}),
            "effect_medium": workbook.add_format({'align': 'center', 'color': '#FFA500', 'bold': True, 'text_wrap': True}),
            "effect_weak": workbook.add_format({'align': 'center', 'color': '#A52A2A', 'bold': True, 'text_wrap': True}),
            "key": workbook.add_format({'bold': True, 'align': 'right'}),
            "bold": workbook.add_format({'bold': True}),
            "recommended": workbook.add_format({'align': 'center', 'bg_color': '#E6F3FF', 'font_color': '#0066CC', 'bold': True, 'text_wrap': True})
        }
    
    @staticmethod
    def _validate_excel_inputs(workbook, results, fmt, ws, function_name="Excel function"):
        """
        Comprehensive input validation for Excel export functions.
        Returns (is_valid, error_message, row_offset)
        """
        error_messages = []
        
        # Basic parameter validation
        if workbook is None:
            error_messages.append("Workbook is None")
        if results is None:
            error_messages.append("Results data is None")
        if fmt is None:
            error_messages.append("Format dictionary is None")
        if ws is None:
            error_messages.append("Worksheet is None")
            
        # Results data validation
        if results is not None:
            if not isinstance(results, dict):
                error_messages.append(f"Results must be dictionary, got {type(results)}")
            elif len(results) == 0:
                error_messages.append("Results dictionary is empty")
                
        # Format validation
        if fmt is not None:
            required_formats = ["header", "cell", "significant", "section_header"]
            missing_formats = [f for f in required_formats if f not in fmt]
            if missing_formats:
                error_messages.append(f"Missing required formats: {', '.join(missing_formats)}")
        
        if error_messages:
            full_error = f"⚠️ VALIDATION ERROR in {function_name}: {'; '.join(error_messages)}"
            return False, full_error, 3
        
        return True, "", 0
    
    @staticmethod
    def _safe_write_cell(ws, row, col, value, cell_format, default_format=None):
        """
        Safely write a cell with error handling.
        Returns True if successful, False otherwise.
        """
        try:
            if cell_format is not None:
                ws.write(row, col, value, cell_format)
            elif default_format is not None:
                ws.write(row, col, value, default_format)
            else:
                ws.write(row, col, value)
            return True
        except Exception as e:
            try:
                # Fallback: write as plain text
                ws.write(row, col, f"ERROR: {str(value)}")
            except:
                pass
            return False

    @staticmethod
    def _write_anova_table(ws, anova_table, fmt, start_row=0):
        """
        Writes an ANOVA table (as DataFrame or dict) to the worksheet at the given row.
        Returns the next empty row after the table.
        """
        import pandas as pd
        if isinstance(anova_table, dict):
            anova_table = pd.DataFrame(anova_table)
        elif not isinstance(anova_table, pd.DataFrame):
            return start_row  # Nothing to write

        # Write header
        for col, colname in enumerate(anova_table.columns):
            ws.write(start_row, col, str(colname), fmt["header"])
        # Write rows
        for row_idx, (_, row) in enumerate(anova_table.iterrows()):
            for col, val in enumerate(row):
                ws.write(start_row + 1 + row_idx, col, val, fmt["cell"])
        return start_row + 1 + len(anova_table)

    @staticmethod
    def _write_summary_sheet(workbook, results, fmt, sheet_name="Summary"):
        ws = workbook.add_worksheet(sheet_name)
        # Set correct column widths: A=55, B-F=20
        ws.set_column(0, 0, 55)  # Column A
        ws.set_column(1, 5, 20)  # Columns B-F
        ws.set_row(0, 30)

        test_info = results.get("test", "Not specified")
        p_value = results.get("p_value", None)
        is_significant = p_value is not None and isinstance(p_value, (float, int)) and p_value < results.get("alpha", 0.05)
        significant_text = "Yes" if is_significant else "No"
        title = f"SUMMARY OF ANALYSIS - {test_info}"
        ws.merge_range('A1:F1', title, fmt["title"])

        # Key statement
        ws.merge_range('A3:F3', "KEY STATEMENT", fmt["section_header"])
        
        # Check if non-parametric alternative is needed
        if results.get("recommendation") == "non_parametric" and results.get("parametric_assumptions_violated", False):
            # Non-parametric alternative required
            conclusion = (
                f"ANALYSIS INCOMPLETE: Parametric assumptions could not be met even after data transformation. "
                f"A non-parametric alternative to {test_info.replace(' (required but not available)', '')} is required for this dataset. "
                f"The suggested approach is: {results.get('suggested_alternative', 'non-parametric statistical method')}. "
                f"Please consult with a statistician or use appropriate non-parametric software."
            )
        elif is_significant:
            effect_size_text = ""
            if "effect_size" in results and results["effect_size"] is not None:
                effect_size = results["effect_size"]
                effect_magnitude = ""
                if "effect_size_type" in results:
                    effect_type = results["effect_size_type"]
                    # Define magnitude based on effect size type
                    if effect_type.lower() == "cohen_d":
                        if abs(effect_size) < 0.2: effect_magnitude = "very small"
                        elif abs(effect_size) < 0.5: effect_magnitude = "small"
                        elif abs(effect_size) < 0.8: effect_magnitude = "medium"
                        else: effect_magnitude = "large"
                    elif effect_type.lower() in ["eta_squared", "partial_eta_squared", "epsilon_squared", "kendall_w", "r"]:
                        # Simplified thresholds for other effect sizes
                        if abs(effect_size) < 0.1: effect_magnitude = "very small"
                        elif abs(effect_size) < 0.3: effect_magnitude = "small"
                        elif abs(effect_size) < 0.5: effect_magnitude = "medium"
                        else: effect_magnitude = "large"
                effect_size_text = f" with a {effect_magnitude} effect (effect size: {effect_size:.3f})"
            p_val_text = "<0.001" if isinstance(p_value, (float, int)) and p_value < 0.001 else f"={p_value:.4f}" if isinstance(p_value, (float, int)) else "not available"
            conclusion = (
                f"The performed test ({test_info}) shows SIGNIFICANT differences "
                f"between the groups under investigation (p{p_val_text})"
                f"{effect_size_text}."
            )
        else:
            p_val_text = f"={p_value:.4f}" if isinstance(p_value, (float, int)) else "not available"
            conclusion = (
                f"The performed test ({test_info}) shows NO significant differences "
                f"between the groups under investigation (p{p_val_text})."
            )
        ws.merge_range('A4:F4', conclusion, fmt["cell"])
        ws.set_row(3, ResultsExporter.get_fixed_row_height("summary_conclusion"))

        # Key information
        row = 6
        ws.merge_range(f'A{row}:F{row}', "KEY INFORMATION", fmt["section_header"])
        row += 1

        key_value_pairs = [
            ("Test:", test_info),
        ]
        
        # Handle non-parametric recommendation case
        if results.get("recommendation") == "non_parametric" and results.get("parametric_assumptions_violated", False):
            key_value_pairs.extend([
                ("Status:", "INCOMPLETE - Non-parametric alternative required"),
                ("Reason:", "Parametric assumptions violated"),
                ("Recommended approach:", results.get("suggested_alternative", "Non-parametric method")),
                ("p-Value:", "Not available (test not performed)")
            ])
            
            # Add transformation information if available
            if results.get("transformation") and results["transformation"] not in ["none", "None", "Keine"]:
                key_value_pairs.append(("Transformation attempted:", results["transformation"]))
            
        else:
            # Normal case - show significance and p-value
            key_value_pairs.extend([
                ("Significant:", significant_text),
                ("p-Value:", f"{'<0.001' if p_value and isinstance(p_value, (float,int)) and p_value < 0.001 else f'={p_value:.4f}' if isinstance(p_value, (float,int)) else 'Not available'}")
            ])

        if "df1" in results and results["df1"] is not None and "df2" in results and results["df2"] is not None:
            key_value_pairs.append(("Degrees of freedom (numerator, denominator):", f"{results['df1']}, {results['df2']}"))
        elif "df" in results and results["df"] is not None: # For chi-square etc.
                key_value_pairs.append(("Degrees of freedom (df):", f"{results['df']}"))


        if "sphericity_test" in results:
            sphericity = results["sphericity_test"]
            if sphericity and sphericity.get("has_sphericity") is not None:
                sphericity_text = "Yes" if sphericity["has_sphericity"] else "No"
                key_value_pairs.append(("Sphericity (Mauchly's Test):", sphericity_text))
                if sphericity.get("p_value") is not None:
                    p_val_text = f"{sphericity['p_value']:.4f}" if sphericity["p_value"] >= 0.001 else "<0.001"
                    key_value_pairs.append(("  p-Value Sphericity:", p_val_text))
                if not sphericity["has_sphericity"] and "correction_used" in results:
                    key_value_pairs.append(("  Correction applied:", results["correction_used"]))


        stat_value = results.get("statistic")
        if stat_value is not None:
            stat_name = "Statistic"
            if "t-Test" in test_info: stat_name = "t-Statistic"
            elif "ANOVA" in test_info or "Welch" in test_info: stat_name = "F-Statistic"
            elif "Mann-Whitney" in test_info: stat_name = "U-Statistic"
            elif "Kruskal-Wallis" in test_info: stat_name = "H-Statistic"
            elif "Wilcoxon" in test_info: stat_name = "W-Statistic"
            elif "Friedman" in test_info: stat_name = "Chi²-Statistic"
            key_value_pairs.append((f"{stat_name}:", f"{stat_value:.4f}" if isinstance(stat_value, (float,int)) else str(stat_value)))


        if "effect_size" in results and results["effect_size"] is not None:
            effect_size = results["effect_size"]
            effect_type = results.get("effect_size_type", "")
            effect_desc = ""
            magnitude = ""
            format_to_use = fmt["cell"] # Default format

            if effect_type.lower() == "cohen_d":
                effect_desc = "Cohen's d"
                if abs(effect_size) < 0.2: magnitude = "very small"; format_to_use = fmt["effect_weak"]
                elif abs(effect_size) < 0.5: magnitude = "small"; format_to_use = fmt["effect_weak"]
                elif abs(effect_size) < 0.8: magnitude = "medium"; format_to_use = fmt["effect_medium"]
                else: magnitude = "large"; format_to_use = fmt["effect_strong"]
            elif effect_type.lower() in ["eta_squared", "partial_eta_squared", "omega_squared"]:
                effect_desc = effect_type.replace("_", " ").title()
                if abs(effect_size) < 0.01: magnitude = "very small"; format_to_use = fmt["effect_weak"]
                elif abs(effect_size) < 0.06: magnitude = "small"; format_to_use = fmt["effect_weak"]
                elif abs(effect_size) < 0.14: magnitude = "medium"; format_to_use = fmt["effect_medium"]
                else: magnitude = "large"; format_to_use = fmt["effect_strong"]
            elif effect_type.lower() == "epsilon_squared":
                effect_desc = "Epsilon²"
                if abs(effect_size) < 0.01: magnitude = "very small"; format_to_use = fmt["effect_weak"]
                elif abs(effect_size) < 0.08: magnitude = "small"; format_to_use = fmt["effect_weak"]
                elif abs(effect_size) < 0.26: magnitude = "medium"; format_to_use = fmt["effect_medium"]
                else: magnitude = "large"; format_to_use = fmt["effect_strong"]
            elif effect_type.lower() == "kendall_w":
                effect_desc = "Kendall's W"
                if abs(effect_size) < 0.1: magnitude = "very small"; format_to_use = fmt["effect_weak"]
                elif abs(effect_size) < 0.3: magnitude = "small"; format_to_use = fmt["effect_weak"]
                elif abs(effect_size) < 0.5: magnitude = "medium"; format_to_use = fmt["effect_medium"]
                else: magnitude = "large"; format_to_use = fmt["effect_strong"]
            elif effect_type.lower() == "r": # For Wilcoxon, Mann-Whitney U
                effect_desc = "r (rank correlation)"
                if abs(effect_size) < 0.1: magnitude = "very small"; format_to_use = fmt["effect_weak"]
                elif abs(effect_size) < 0.3: magnitude = "small"; format_to_use = fmt["effect_weak"]
                elif abs(effect_size) < 0.5: magnitude = "medium"; format_to_use = fmt["effect_medium"]
                else: magnitude = "large"; format_to_use = fmt["effect_strong"]
            else:
                effect_desc = effect_type if effect_type else "Effect size"
                magnitude = "not classified"

            key_value_pairs.append((f"{effect_desc}:", f"{effect_size:.4f} ({magnitude})"))
        else:
            format_to_use = fmt["cell"] # Ensure format_to_use is defined

        ci = results.get("confidence_interval", (None, None))
        ci_level = results.get("ci_level", 0.95) * 100

        ci_text = "Not calculated; see confidence intervals of pairwise comparisons (if available)."
        if ci is not None and isinstance(ci, (list, tuple)) and len(ci) == 2 and ci[0] is not None and ci[1] is not None:
            ci_text = f"[{ci[0]:.4f}, {ci[1]:.4f}]"
        elif test_info and ("ANOVA" in test_info or "Kruskal-Wallis" in test_info or "Friedman" in test_info) and len(results.get("groups",[])) > 2 :
                # For ANOVA-like tests with >2 groups, the main CI is often less informative than post-hoc CIs
            pass # ci_text remains the default message
        elif ci == (None, None) or ci is None : # Explicitly (None,None) or just None
            pass # ci_text remains the default message

        key_value_pairs.append((f"{ci_level:.0f}% Confidence interval:", ci_text))


        if "power" in results:
            power = results["power"]
            if power is not None:
                power_desc = "low" if power < 0.5 else "moderate" if power < 0.8 else "high"
                key_value_pairs.append(("Statistical power:", f"{power:.2f} ({power_desc})"))
            else:
                key_value_pairs.append(("Statistical power:", "Not calculated/available"))

        for key, value in key_value_pairs:
            ws.write(row, 0, key, fmt["key"])
            current_format = fmt["cell"]
            if key == "Significant:" and value == "Yes":
                current_format = fmt["significant"]
            elif key == "p-Value:" and is_significant:
                current_format = fmt["significant"]
            elif "Effect size" in key or "Cohen's d" in key or "Eta²" in key or "Epsilon²" in key or "Kendall's W" in key or "r (" in key:
                    # Use the format_to_use determined during effect size magnitude check
                current_format = format_to_use
            ws.write(row, 1, value, current_format)
            row += 1

        # Navigation
        row += 2
        ws.merge_range(f'A{row}:F{row}', "NAVIGATION TO DETAILED RESULTS", fmt["section_header"])
        row += 1
        nav_text = (
            "• Statistical results: Details on test and significance\n"
            "• Assumptions check: Tests for normality and variance homogeneity\n"
            "• Descriptive statistics: Metrics with confidence intervals for each group\n"
            "• Pairwise comparisons: Details on individual group differences with effect sizes and CIs\n"
            "• Raw data: The original measured values\n"
            "• Analysis log: Chronological sequence of the analysis\n"
            "• Hypotheses: Tested null and alternative hypotheses\n")
        
        # Use robust single cell for navigation text
        nav_wrap_fmt = workbook.add_format({
            'text_wrap': True, 
            'valign': 'top',
            'border': 1,
            'bg_color': '#F0F8FF'
        })
        ws.write(row, 0, nav_text, nav_wrap_fmt)
        nav_height = ResultsExporter.get_fixed_row_height("summary_navigation")
        ws.set_row(row, nav_height)

        # Update row position after navigation text
        row += 2

        # Post-hoc tests information
        posthoc_test = results.get("posthoc_test", None)
        if posthoc_test:
            row += 1
            ws.merge_range(f'A{row}:F{row}', "POST-HOC TESTS PERFORMED", fmt["section_header"])
            row += 1
            
            # Show the specific post-hoc test that was performed
            ws.write(row, 0, "Test performed:", fmt["key"])
            ws.write(row, 1, posthoc_test, fmt["cell"])
            row += 1
            
            # Add explanations for different post-hoc tests
            posthoc_explanations = {
                "Tukey HSD": "Tukey's Honestly Significant Difference test compares all possible pairs of groups while controlling the family-wise error rate. It's the most commonly used post-hoc test for ANOVA.",
                "Dunnett": "Dunnett's test compares all treatment groups against a single control group. It's more powerful than Tukey when you have a clear control condition.",
                "Custom paired t-tests (Holm-Sidak)": "User-selected group pairs are compared using paired t-tests with Holm-Sidak correction for multiple comparisons. This allows for focused comparisons of specific group pairs.",
                "Dunn": "Dunn's test is a non-parametric post-hoc test that compares all possible pairs after a significant Kruskal-Wallis test, using rank-based statistics with Holm-Sidak correction.",
                "Custom Mann-Whitney-U tests (Sidak)": "User-selected group pairs are compared using Mann-Whitney U tests with Sidak correction for multiple comparisons. This non-parametric approach is used when normality assumptions are violated.",
                "Dependent Post-hoc": "Specialized post-hoc tests for repeated measures designs, using either paired t-tests or Wilcoxon signed-rank tests depending on normality assumptions."
            }
            
            # Find the best matching explanation
            explanation = "See the 'Pairwise Comparisons' sheet for detailed results."
            for test_name, test_explanation in posthoc_explanations.items():
                if test_name.lower() in posthoc_test.lower():
                    explanation = test_explanation
                    break
            
            # Use robust single cell for post-hoc explanation
            explanation_wrap_fmt = workbook.add_format({
                'text_wrap': True, 
                'valign': 'top',
                'border': 1,
                'bg_color': '#F0F8FF'
            })
            ws.write(row, 0, explanation, explanation_wrap_fmt)
            explanation_height = ResultsExporter.get_fixed_row_height("summary_posthoc_info")
            ws.set_row(row, explanation_height)
            row += 2
            
            # Add general note about post-hoc tests
            general_note = (
                "Note: Post-hoc tests are only performed when the main test shows significant differences. "
                "They help identify which specific groups differ from each other while controlling for multiple comparisons."
            )
            
            # Use robust single cell for general note
            note_wrap_fmt = workbook.add_format({
                'text_wrap': True, 
                'valign': 'top',
                'border': 1,
                'bg_color': '#F0F8FF'
            })
            ws.write(row, 0, general_note, note_wrap_fmt)
            note_height = ResultsExporter.get_fixed_row_height("general_note")
            ws.set_row(row, note_height)
            row += 1

        anova_table = results.get("anova_table")
        if anova_table is not None:
            row += 2
            ws.merge_range(f'A{row}:F{row}', "ANOVA TABLE", fmt["section_header"])
            row += 1
            next_row = ResultsExporter._write_anova_table(ws, anova_table, fmt, start_row=row)
            row = next_row + 2  # Add space after the table
            
            # Add explanation section for ANOVA table
            ws.merge_range(f'A{row}:F{row}', "UNDERSTANDING THE ANOVA TABLE", fmt["section_header"])
            row += 1
            
            # Determine which type of ANOVA is being used
            test_name = results.get("test", "").lower()
            is_welch = "welch" in test_name
            is_rm = "repeated measures" in test_name or "rm anova" in test_name
            is_mixed = "mixed" in test_name
            is_two_way = "two-way" in test_name or "two way" in test_name
            
            # Introduction text based on ANOVA type
            if is_welch:
                intro_text = (
                    "This is a Welch's ANOVA table, which does not assume equal variances between groups. "
                    "Welch's ANOVA is more robust when the homogeneity of variance assumption is violated."
                )
            elif is_rm:
                intro_text = (
                    "This is a Repeated Measures ANOVA table, which analyzes differences between repeated measurements "
                    "on the same subjects, accounting for the dependency between observations."
                )
            elif is_mixed:
                intro_text = (
                    "This is a Mixed ANOVA table, which combines between-subjects factors (different groups) "
                    "and within-subjects factors (repeated measures) in the same analysis."
                )
            elif is_two_way:
                intro_text = (
                    "This is a Two-Way ANOVA table, which analyzes the effect of two independent variables (factors) "
                    "on one dependent variable, including their potential interaction."
                )
            else:
                intro_text = (
                    "This is a standard One-Way ANOVA table, which compares means across multiple groups "
                    "to determine if there are significant differences between them."
                )
            
            # Use robust single cell for ANOVA intro text
            intro_wrap_fmt = workbook.add_format({
                'text_wrap': True, 
                'valign': 'top',
                'border': 1,
                'bg_color': '#F0F8FF'
            })
            ws.write(row, 0, intro_text, intro_wrap_fmt)
            intro_anova_height = ResultsExporter.get_fixed_row_height("intro_anova_text")
            ws.set_row(row, intro_anova_height)
            row += 2
            
            # Column explanations
            ws.merge_range(f'A{row}:F{row}', "EXPLANATION OF ANOVA TABLE COLUMNS:", fmt["key"])
            row += 1
            
            explanations = [
                ("Source", "Identifies the component being analyzed:\n• Group/factor names indicate variation between groups/factors\n• Residual (or Error/Within) represents unexplained variation within groups\n• Interaction terms (in factorial designs) show how factors work together"),
                
                ("SS (Sum of Squares)", "Measures the total variation attributed to each source:\n• Higher values indicate more variation explained by that source\n• SS between groups shows variation due to group differences\n• SS within groups (residual) shows variation due to individual differences\n• The ratio of SS(between) to SS(total) provides an estimate of effect size"),
                
                ("DF (Degrees of Freedom)", "The number of values that are free to vary:\n• For between-groups: DF = number of groups - 1\n• For residuals (simple design): DF = total observations - number of groups\n• For residuals (factorial design): DF = total sample size - number of estimated parameters\n• For factors: DF = number of levels - 1\n• For interactions: DF = product of the individual factors' DFs"),
                
                ("MS (Mean Square)", "Average variation per degree of freedom (MS = SS/DF):\n• MS between represents average between-group variation\n• MS within (residual) represents average within-group variation (error variance)\n• The ratio MS(between)/MS(within) forms the F-statistic"),
                
                ("F", "The F-statistic (MS between / MS within):\n• Compares between-group variation to within-group variation\n• Larger F values suggest stronger group differences\n• F = 1 would indicate no difference between groups\n• The critical F value depends on the degrees of freedom and alpha level"),
                
                ("p-unc", "Probability value (significance level):\n• p < 0.05 is commonly used to indicate statistical significance, though this threshold is arbitrary\n• 'unc' indicates these are uncorrected p-values (not adjusted for multiple comparisons)\n• Small p-values suggest the observed differences are unlikely under the null hypothesis\n• For multiple tests, consider using corrected p-values to control error rates"),
                
                ("np2 (Partial Eta Squared)", "Effect size measure (proportion of variance explained):\n• 0.01 = small effect\n• 0.06 = medium effect\n• 0.14 = large effect\n• Higher values indicate stronger effects\n• Interpretation is context-dependent and varies between research fields\n• Consider field-specific benchmarks when interpreting effect sizes")
            ]
            
            explanation_heights = {
                "Source": "anova_source_explanation",
                "SS (Sum of Squares)": "anova_ss_explanation", 
                "DF (Degrees of Freedom)": "anova_df_explanation",
                "MS (Mean Square)": "anova_ms_explanation",
                "F": "anova_f_explanation",
                "p-unc": "anova_p_explanation", 
                "np2 (Partial Eta Squared)": "anova_np2_explanation"
            }
            
            for term, explanation in explanations:
                ws.write(row, 0, term, fmt["key"])
                ws.write(row, 1, explanation, fmt["explanation"])
                height_key = explanation_heights.get(term, "medium_text")
                ws.set_row(row, ResultsExporter.get_fixed_row_height(height_key))
                row += 1
            
            row += 1
            
            # How to interpret section
            ws.merge_range(f'A{row}:F{row}', "HOW TO INTERPRET THE RESULTS", fmt["key"])
            row += 1
            
            if is_two_way or is_mixed:
                interpret_text = (
                    "1. Check main effects: Look at p-values for each factor. If p < 0.05, that factor has a significant effect.\n\n"
                    "2. Check interaction: If the interaction p-value is < 0.05, the effect of one factor depends on the level of the other factor. "
                    "In this case, interpret main effects with caution and focus on pairwise comparisons.\n\n"
                    "3. Effect size (np2): Indicates the practical significance - how much variance is explained by each factor.\n\n"
                    "4. Post-hoc tests: For significant effects, examine post-hoc tests to identify which specific groups differ."
                )
            else:
                interpret_text = (
                    "1. Statistical significance: If the p-value for the between-groups factor is < 0.05, there are significant differences between at least some groups.\n\n"
                    "2. Effect size (np2): Indicates the practical significance - how much variance is explained by group differences.\n\n"
                    "3. F-statistic: Higher values indicate stronger evidence against the null hypothesis.\n\n"
                    "4. Post-hoc tests: For significant results, examine post-hoc tests to identify which specific groups differ from each other."
                )

            
            # Use robust single cell for interpretation text
            interpret_wrap_fmt = workbook.add_format({
                'text_wrap': True, 
                'valign': 'top',
                'border': 1,
                'bg_color': '#F0F8FF'
            })
            ws.write(row, 0, interpret_text, interpret_wrap_fmt)
            interpret_height = ResultsExporter.get_fixed_row_height("results_interpretation")
            ws.set_row(row, interpret_height)
            row += 1
    
    @staticmethod
    def _write_assumptions_sheet(workbook, results, fmt, sheet_name="Assumptions"):
        ws = workbook.add_worksheet(sheet_name)
        # Set column widths
        ws.set_column(0, 0, 55)   # Column A reduced from 80 to 55
        ws.set_column(1, 5, 20)   # Other columns smaller
        ws.set_row(0, 30)
        ws.merge_range('A1:F1', 'TEST ASSUMPTIONS CHECK', fmt["title"])

        # Introduction
        introduction = (
            "This sheet documents the tests for checking the assumptions for the statistical analysis. "
            "Depending on the type of test (parametric or non-parametric), different assumptions must be met."
        )
        
        # Use robust single cell instead of merge_range for text
        intro_wrap_fmt = workbook.add_format({
            'text_wrap': True, 
            'valign': 'top',
            'border': 1,
            'bg_color': '#F0F8FF'
        })
        ws.write(1, 0, introduction, intro_wrap_fmt)
        # Use fixed optimal height for introduction
        intro_height = ResultsExporter.get_fixed_row_height("assumptions_intro")
        ws.set_row(1, intro_height)

        row = 4

        # Test type and general notes with INTELLIGENT CONTEXT-AWARE EXPLANATIONS
        test_type = results.get("recommendation", results.get("test_type", "Not specified"))
        test_name = results.get("test", "")
        
        # Detect if this is RM or Mixed ANOVA for context-aware explanations
        is_rm_anova = "repeated measures" in test_name.lower() or "rm" in test_name.lower()
        is_mixed_anova = "mixed" in test_name.lower()
        has_sphericity = "sphericity_test" in results
        
        if test_type == "parametric":
            if is_rm_anova or is_mixed_anova:
                # ENHANCED EXPLANATION FOR RM/MIXED ANOVA
                assumptions_overview = (
                    f"🔬 SPECIAL ASSUMPTIONS FOR {test_name.upper()}:\n\n"
                    "Your analysis uses advanced statistical methods that require additional assumptions beyond basic ANOVA:\n\n"
                    
                    "📊 BASIC PARAMETRIC ASSUMPTIONS:\n"
                    "  • Normal distribution of model residuals\n"
                    "  • Independence of observations\n"
                    "  • Interval scale of the dependent variable\n\n"
                )
                
                if is_rm_anova:
                    assumptions_overview += (
                        "🔄 REPEATED MEASURES SPECIFIC ASSUMPTIONS:\n"
                        "  • SPHERICITY: Equal variances of differences between all condition pairs\n"
                        "    → Why important: RM ANOVA compares differences between conditions\n"
                        "    → When violated: F-test becomes too liberal (more Type I errors)\n"
                        "    → Solution: Greenhouse-Geisser or Huynh-Feldt corrections\n\n"
                        
                        "💡 WHAT ARE THESE CORRECTIONS?\n"
                        "  • Greenhouse-Geisser: Conservative correction (reduces degrees of freedom more)\n"
                        "  • Huynh-Feldt: Less conservative (closer to original degrees of freedom)\n"
                        "  • Automatic selection: ε > 0.75 → Huynh-Feldt, ε ≤ 0.75 → Greenhouse-Geisser\n\n"
                        
                        "🎯 PRACTICAL IMPACT:\n"
                        "If sphericity is violated, use corrected p-values instead of uncorrected ones!"
                    )
                
                elif is_mixed_anova:
                    assumptions_overview += (
                        "🔄🔀 MIXED ANOVA SPECIFIC ASSUMPTIONS:\n"
                        "  • BETWEEN-FACTOR: Homogeneity of variances (like regular ANOVA)\n"
                        "  • WITHIN-FACTOR: Sphericity (like Repeated Measures ANOVA)\n"
                        "  • INTERACTION: Both factors together must meet compound assumptions\n\n"
                        
                        "🧪 MULTIPLE ASSUMPTION TESTS:\n"
                        "  • Levene's Test: Checks if groups have equal variances\n"
                        "  • Brown-Forsythe: Robust version of Levene's test\n"
                        "  • Mauchly's Test: Checks sphericity for within-subject factor\n"
                        "  • Box's M Test: Checks if covariance patterns are similar across groups\n\n"
                        
                        "🎯 PRACTICAL IMPACT:\n"
                        "Mixed ANOVA is complex - multiple assumption violations may require different statistical approaches!"
                    )
                
                assumptions_overview += (
                    "\n\n✅ WHAT TO LOOK FOR IN RESULTS:\n"
                    "  • Green/Normal formatting = Assumptions met\n"
                    "  • Red/Warning formatting = Assumptions violated\n"
                    "  • Corrected p-values = Use these when corrections are applied\n"
                    "  • Recommendations = Follow these for best statistical practice"
                )
            else:
                # Standard parametric explanation
                assumptions_overview = (
                    "For parametric tests (such as t-test, ANOVA), the following assumptions apply:\n"
                    "  • Normal distribution of model residuals (NEW: tested on residuals, not raw group data)\n"
                    "  • Homogeneity of variances between groups\n"
                    "  • Independence of observations\n"
                    "  • Interval scale of the dependent variable\n\n"
                    "IMPORTANT CHANGE: Normality is now tested on model residuals rather than raw group data. "
                    "This provides a more accurate assessment of whether the statistical model assumptions are met. "
                    "Points 1 and 2 are tested statistically. Points 3 and 4 are ensured by the study design and data collection."
                )
        else:
            assumptions_overview = (
                "For non-parametric tests (such as Mann-Whitney U, Kruskal-Wallis), the following assumptions apply:\n"
                "  • Similar distribution shape in all groups (not necessarily normal distribution)\n"
                "  • Independence of observations\n"
                "  • At least ordinal scale of the dependent variable\n\n"
                "Assumption 1 is assessed visually. Points 2 and 3 are ensured by the study design and data collection."
            )
        ws.merge_range(f'A{row}:F{row}', "OVERVIEW OF ASSUMPTIONS", fmt["section_header"])
        row += 1
        
        # Use robust single cell for assumptions overview text - ONLY cell A gets formatting
        assumptions_wrap_fmt = workbook.add_format({
            'text_wrap': True, 
            'valign': 'top',
            'border': 1,
            'bg_color': '#F0F8FF'
        })
        ws.write(row, 0, assumptions_overview, assumptions_wrap_fmt)
        # Use fixed optimal height for assumptions overview
        overview_height = ResultsExporter.get_fixed_row_height("assumptions_overview")
        ws.set_row(row, overview_height)
        row += 2

        # Sphericity (only for Repeated Measures ANOVA)
        if "sphericity_test" in results:
            row += 1
            ws.write(row, 0, "Sphericity (Mauchly test):", fmt["section_header"])
            row += 1
            
            # Check if we have a two-level within factor (where sphericity is always met)
            has_two_levels = False
            
            # Method 1: Check ANOVA table for epsilon=1.0
            if "anova_table" in results and isinstance(results["anova_table"], pd.DataFrame):
                if "eps" in results["anova_table"].columns:
                    eps_values = results["anova_table"]["eps"].dropna()
                    if not eps_values.empty and (eps_values == 1.0).all():
                        has_two_levels = True
            
            # Method 2: Check within factor directly if available
            if "factors" in results:
                for factor in results["factors"]:
                    if factor.get("type") == "within":
                        within_factor = factor.get("factor")
                        if within_factor and "df1" in factor and factor["df1"] == 1:
                            has_two_levels = True
                            break
            
            if has_two_levels:
                # Special explanation for two-level within factors
                explanation = (
                    "With only two levels of the within-factor, sphericity is automatically satisfied mathematically.\n\n"
                    "Sphericity concerns the equality of variances of differences between all combinations of within-subject levels. "
                    "When there are only two levels, there is only one possible difference (level 1 - level 2), "
                    "so no comparison of different variances is possible and sphericity is perfectly met by definition.\n\n"
                    "Therefore, no sphericity test is necessary, and no corrections (Greenhouse-Geisser or Huynh-Feldt) are needed."
                )
                
                # Use robust single cell for sphericity explanation
                sphericity_wrap_fmt = workbook.add_format({
                    'text_wrap': True, 
                    'valign': 'top',
                    'border': 1,
                    'bg_color': '#F0F8FF'
                })
                ws.write(row, 0, explanation, sphericity_wrap_fmt)
                sphericity_height = ResultsExporter.get_fixed_row_height("sphericity_detail")
                ws.set_row(row, sphericity_height)
                row += 2
            else:
                # Regular sphericity test information
                sph_headers = ["Mauchly's W", "p-Value", "Sphericity assumed?", "Interpretation"]
                for i, h in enumerate(sph_headers):
                    ws.write(row, i, h, fmt["header"])
                row += 1
                
                sphericity = results["sphericity_test"]
                w_val = sphericity.get("W", "N/A")
                p_val = sphericity.get("p_value", "N/A")
                has_sphericity = sphericity.get("has_sphericity", None)
                sph_text = "Yes" if has_sphericity else "No" if has_sphericity is not None else "Indeterminable"
                
                interpretation = (
                    "No significant deviation from sphericity"
                    if has_sphericity else
                    "Significant deviation from sphericity, correction necessary"
                    if has_sphericity is not None else
                    "Sphericity could not be tested"
                )
                
                values = [
                    f"{w_val:.4f}" if isinstance(w_val, (float, int)) else w_val,
                    f"{p_val:.4f}" if isinstance(p_val, (float, int)) else p_val,
                    sph_text,
                    interpretation
                ]
                
                for col, val in enumerate(values):
                    ws.write(row, col, val, fmt["cell"])
                row += 1
    
        # Normality tests per group
        ws.write(row, 0, "Normality of Model Residuals (Shapiro-Wilk test):", fmt["section_header"])
        row += 1
        
        # Check if we have the new test_info structure with residual-based tests
        has_residual_tests = (
            "test_info" in results 
            and "pre_transformation" in results["test_info"]
            and "residuals_normality" in results["test_info"]["pre_transformation"]
        )
        
        # ALSO check if we have the new normality_tests structure with model_residuals
        has_new_normality_structure = (
            "normality_tests" in results 
            and isinstance(results["normality_tests"], dict)
            and "model_residuals" in results["normality_tests"]
        )
        
        if has_residual_tests or has_new_normality_structure:
            # NEW: Display residual-based normality tests
            norm_explanation = (
                "Note: Normality is tested on the residuals of the statistical model."
            )
            
            # Use robust single cell for normality explanation
            norm_wrap_fmt = workbook.add_format({
                'text_wrap': True, 
                'valign': 'top',
                'border': 1,
                'bg_color': '#F0F8FF'
            })
            ws.write(row, 0, norm_explanation, norm_wrap_fmt)
            norm_height = ResultsExporter.get_fixed_row_height("normality_detail")
            ws.set_row(row, norm_height)
            row += 2
            
            norm_headers = ["Data Type", "Shapiro-Wilk statistic", "p-Value", "Residuals normally distributed?", "Interpretation"]
            for i, h in enumerate(norm_headers):
                ws.write(row, i, h, fmt["header"])
            row += 1
            
            # Get test info from either test_info structure or compatible structure
            test_info = results.get("test_info", {})
            
            # Pre-transformation residual test
            if has_residual_tests and "pre_transformation" in test_info and "residuals_normality" in test_info["pre_transformation"]:
                pre_norm = test_info["pre_transformation"]["residuals_normality"]
                stat = pre_norm.get('statistic', 'N/A')
                p_val = pre_norm.get('p_value', 'N/A')
                is_normal = pre_norm.get('is_normal', False)
            elif has_new_normality_structure and "model_residuals" in results["normality_tests"]:
                pre_norm = results["normality_tests"]["model_residuals"]
                stat = pre_norm.get('statistic', 'N/A')
                p_val = pre_norm.get('p_value', 'N/A')
                is_normal = pre_norm.get('is_normal', False)
            else:
                stat = p_val = 'N/A'
                is_normal = False
                
            # Write pre-transformation row if we have data
            if stat != 'N/A' or p_val != 'N/A':
                normal_text = "Yes" if is_normal else "No"
                interpretation = (
                    "Model residuals show no significant deviation from normality"
                    if is_normal else
                    "Model residuals show significant deviation from normality"
                )
                values = [
                    "Original Data (Model Residuals)",
                    f"{stat:.4f}" if isinstance(stat, (float, int)) else stat,
                    f"{p_val:.4f}" if isinstance(p_val, (float, int)) else p_val,
                    normal_text,
                    interpretation
                ]
                for col, val in enumerate(values):
                    cell_fmt = fmt["significant"] if not is_normal else fmt["cell"]
                    ws.write(row, col, val, cell_fmt)
                row += 1
            
            # Post-transformation residual test (if transformation was applied)
            transformation = results.get("transformation", "None")
            if transformation and transformation != "None":
                # Try to get post-transformation data
                if (has_residual_tests and "post_transformation" in test_info and 
                    "residuals_normality" in test_info["post_transformation"]):
                    post_norm = test_info["post_transformation"]["residuals_normality"]
                    stat = post_norm.get('statistic', 'N/A')
                    p_val = post_norm.get('p_value', 'N/A')
                    is_normal = post_norm.get('is_normal', False)
                elif (has_new_normality_structure and "model_residuals_transformed" in results["normality_tests"]):
                    post_norm = results["normality_tests"]["model_residuals_transformed"]
                    stat = post_norm.get('statistic', 'N/A')
                    p_val = post_norm.get('p_value', 'N/A')
                    is_normal = post_norm.get('is_normal', False)
                else:
                    stat = p_val = 'N/A'
                    is_normal = False
                
                # Write post-transformation row if we have data
                if stat != 'N/A' or p_val != 'N/A':
                    normal_text = "Yes" if is_normal else "No"
                    interpretation = (
                        "Transformed model residuals show no significant deviation from normality"
                        if is_normal else
                        "Transformed model residuals show significant deviation from normality"
                    )
                    values = [
                        f"After {transformation} Transformation (Model Residuals)",
                        f"{stat:.4f}" if isinstance(stat, (float, int)) else stat,
                        f"{p_val:.4f}" if isinstance(p_val, (float, int)) else p_val,
                        normal_text,
                        interpretation
                    ]
                    for col, val in enumerate(values):
                        cell_fmt = fmt["significant"] if not is_normal else fmt["cell"]
                        ws.write(row, col, val, cell_fmt)
                    row += 1
                
        else:
            # FALLBACK: Display old-style group-based normality tests or model residuals
            normality_results = results.get("normality_tests", {})
            
            # Check if we have model_residuals structure even without test_info
            if "model_residuals" in normality_results:
                # Display as model residuals
                norm_headers = ["Data Type", "Shapiro-Wilk statistic", "p-Value", "Residuals normally distributed?", "Interpretation"]
                for i, h in enumerate(norm_headers):
                    ws.write(row, i, h, fmt["header"])
                row += 1
                
                # Original data (model residuals)
                model_res = normality_results["model_residuals"]
                stat = model_res.get('statistic', 'N/A')
                p_val = model_res.get('p_value', 'N/A')
                is_normal = model_res.get('is_normal', False)
                normal_text = "Yes" if is_normal else "No"
                interpretation = (
                    "Model residuals show no significant deviation from normality"
                    if is_normal else
                    "Model residuals show significant deviation from normality"
                )
                values = [
                    "Original Data (Model Residuals)",
                    f"{stat:.4f}" if isinstance(stat, (float, int)) else stat,
                    f"{p_val:.4f}" if isinstance(p_val, (float, int)) else p_val,
                    normal_text,
                    interpretation
                ]
                for col, val in enumerate(values):
                    cell_fmt = fmt["significant"] if not is_normal else fmt["cell"]
                    ws.write(row, col, val, cell_fmt)
                row += 1
                
                # Transformed data if available
                if "model_residuals_transformed" in normality_results:
                    model_res_trans = normality_results["model_residuals_transformed"]
                    stat = model_res_trans.get('statistic', 'N/A')
                    p_val = model_res_trans.get('p_value', 'N/A')
                    is_normal = model_res_trans.get('is_normal', False)
                    normal_text = "Yes" if is_normal else "No"
                    transformation = results.get("transformation", "Unknown")
                    interpretation = (
                        "Transformed model residuals show no significant deviation from normality"
                        if is_normal else
                        "Transformed model residuals show significant deviation from normality"
                    )
                    values = [
                        f"After {transformation} Transformation (Model Residuals)",
                        f"{stat:.4f}" if isinstance(stat, (float, int)) else stat,
                        f"{p_val:.4f}" if isinstance(p_val, (float, int)) else p_val,
                        normal_text,
                        interpretation
                    ]
                    for col, val in enumerate(values):
                        cell_fmt = fmt["significant"] if not is_normal else fmt["cell"]
                        ws.write(row, col, val, cell_fmt)
                    row += 1
            else:
                # Original old-style group-based normality tests
                norm_headers = ["Group", "Shapiro-Wilk statistic", "p-Value", "Normally distributed?", "Interpretation"]
                for i, h in enumerate(norm_headers):
                    ws.write(row, i, h, fmt["header"])
                row += 1
                
                for group, test_result in normality_results.items():
                    if group in ["all_data", "transformed_data", "model_residuals", "model_residuals_transformed"]:
                        continue  # Skip these special entries
                    stat = test_result.get('statistic', 'N/A')
                    p_val = test_result.get('p_value', 'N/A')
                    is_normal = (isinstance(p_val, (float, int)) and p_val > 0.05)
                    normal_text = "Yes" if is_normal else "No"
                    interpretation = (
                        "No significant deviation from normality"
                        if is_normal else
                        "Significant deviation from normality"
                    )
                    values = [
                        str(group),
                        f"{stat:.4f}" if isinstance(stat, (float, int)) else stat,
                        f"{p_val:.4f}" if isinstance(p_val, (float, int)) else p_val,
                        normal_text,
                        interpretation
                    ]
                    for col, val in enumerate(values):
                        cell_fmt = fmt["significant"] if not is_normal else fmt["cell"]
                        ws.write(row, col, val, cell_fmt)
                    row += 1
    
        row += 1

        # Note: The old "all_data" normality test is replaced by model residual tests above
        # This provides more accurate assumption testing for the statistical models

        # Homogeneity of variances (Brown-Forsythe-Test)
        ws.write(row, 0, "Homogeneity of variances (Brown-Forsythe-Test):", fmt["section_header"])
        row += 1
        var_headers = ["Data Type", "Brown-Forsythe statistic", "p-Value", "Variances equal?", "Interpretation"]
        for i, h in enumerate(var_headers):
            ws.write(row, i, h, fmt["header"])
        row += 1
        
        # Get variance test data - SIMPLIFIED APPROACH 
        variance_test = results.get("variance_test", {})
        transformation = results.get("transformation", "None")
        
        # Write original variance test data
        if variance_test and "statistic" in variance_test:
            stat = variance_test.get('statistic', 'N/A')
            p_val = variance_test.get('p_value', 'N/A')
            var_equal = variance_test.get('equal_variance', False)
            var_text = "Yes" if var_equal else "No"
            interpretation = (
                "No significant differences in variances"
                if var_equal else
                "Significant differences in variances"
            )
            
            values = [
                "Original Data",
                f"{stat:.4f}" if isinstance(stat, (float, int)) else str(stat),
                f"{p_val:.4f}" if isinstance(p_val, (float, int)) else str(p_val),
                var_text,
                interpretation
            ]
            
            for col, val in enumerate(values):
                cell_fmt = fmt["significant"] if not var_equal else fmt["cell"]
                ws.write(row, col, val, cell_fmt)
            row += 1
        else:
            # Fallback if no variance data
            ws.write(row, 0, "No variance test data available", fmt["cell"])
            row += 1
            
        # Write transformed variance test data if transformation was applied
        if transformation and transformation not in ["None", "No further"] and "transformed" in variance_test:
            trans_var = variance_test["transformed"]
            stat = trans_var.get('statistic', 'N/A')
            p_val = trans_var.get('p_value', 'N/A')
            var_equal = trans_var.get('equal_variance', False)
            var_text = "Yes" if var_equal else "No"
            interpretation = (
                "No significant differences in variances after transformation"
                if var_equal else
                "Significant differences in variances even after transformation"
            )
            
            values = [
                f"After {transformation} Transformation",
                f"{stat:.4f}" if isinstance(stat, (float, int)) else str(stat),
                f"{p_val:.4f}" if isinstance(p_val, (float, int)) else str(p_val),
                var_text,
                interpretation
            ]
            
            for col, val in enumerate(values):
                cell_fmt = fmt["significant"] if not var_equal else fmt["cell"]
                ws.write(row, col, val, cell_fmt)
            row += 1
        
        row += 1  # Add space after variance test section
        
        # ENHANCED ASSUMPTION TESTING FOR RM/MIXED ANOVA
        row = ResultsExporter._write_enhanced_assumption_tests(workbook, results, fmt, ws, row)
    
        # VISUAL EXAMINATION SECTION (for all cases)
        ws.merge_range(f'A{row}:F{row}', "VISUAL EXAMINATION OF ASSUMPTIONS", fmt["section_header"])
        row += 1
        
        visual_intro = (
            "📈📊 VISUAL ASSUMPTION CHECKING - Understanding What Statistical Tests Need\n\n"
            "Statistical tests like ANOVA and t-tests work best when your data meets certain mathematical requirements "
            "(called 'assumptions'). The plots below help you check these requirements visually:\n\n"
            
            "🔍 Q-Q PLOT: Checks if your data follows a normal distribution pattern\n"
            "📊 BOXPLOT: Checks if different groups have similar variability (variance)\n\n"
            
            "Each plot includes detailed explanations to help you understand what you're looking at and how to interpret the patterns."
        )
        
        # Use robust single cell for visual intro
        visual_wrap_fmt = workbook.add_format({
            'text_wrap': True, 
            'valign': 'top',
            'border': 1,
            'bg_color': '#F0F8FF'
        })
        ws.write(row, 0, visual_intro, visual_wrap_fmt)
        visual_height = ResultsExporter.get_fixed_row_height("visual_intro")
        ws.set_row(row, visual_height)
        row += 2  # Reduced space after introduction
        
        # Generate and insert visual plots - ROBUST SIDE-BY-SIDE LAYOUT
        try:
            AssumptionVisualizer = get_assumption_visualizer()
            plot_paths = AssumptionVisualizer.generate_assumption_plots(results)
            
            # Create side-by-side layout for both plots
            if plot_paths['normality_before'] and plot_paths['homoscedasticity_before']:
                wrap_fmt = workbook.add_format({
                    'text_wrap': True, 
                    'valign': 'top',
                    'border': 1,
                    'bg_color': '#F0F8FF'
                })
                
                LEFT_TOTAL = 55 * 6   # A..F (updated for new column width A=55)
                RIGHT_TOTAL = 28 * 7  # G..M (erweitert auf 7 Spalten für mehr Breite)
                
                # Single header for both plots (Merge ok für Überschrift)
                ws.merge_range(f'A{row}:M{row}', "📈📊 VISUAL ASSUMPTION EXAMINATION - Q-Q Plot & Boxplots Side by Side", fmt["section_header_center"])
                ws.set_row(row, 22)
                row += 1
                
                # Text content
                qq_text = (
                    "📊 UNDERSTANDING Q-Q PLOTS (Left Side) - Normality Check:\n\n"
                    "WHAT IS A Q-Q PLOT?\n"
                    "A Q-Q plot compares your data's distribution to a perfect normal distribution. "
                    "Each point represents one data value from your statistical model's residuals (prediction errors). "
                    "The X-axis shows where that value would fall in a perfect normal distribution, "
                    "and the Y-axis shows where it actually falls in your data.\n\n"
                    "HOW TO READ THE Q-Q PLOT:\n"
                    "• Red diagonal line = What a perfect normal distribution would look like\n"
                    "• Blue dots = Your actual model residuals (prediction errors)\n"
                    "• If dots follow the red line closely → Your data is normally distributed ✅\n"
                    "• If dots deviate from the red line → Your data is not normally distributed ⚠️"
                )
                
                box_text = (
                    "📦 UNDERSTANDING BOXPLOTS (Right Side) - Variance Equality Check:\n\n"
                    "WHAT IS A BOXPLOT?\n"
                    "A boxplot shows your data's distribution and spread. "
                    "Box = middle 50% of data, line = median, whiskers = normal range, dots = outliers.\n\n"
                    "HOW TO USE FOR VARIANCE TESTING:\n"
                    "Compare boxplots across groups - similar variability means equal variances.\n\n"
                    "PATTERNS:\n"
                    "✅ GOOD: All boxes roughly same height\n"
                    "⚠️ WARNING: Very different box heights\n"
                    "💡 REMEMBER: Compare spread, not central values"
                )
                
                ws.write(row, 0, qq_text, wrap_fmt)
                req_h_left = ResultsExporter.get_fixed_row_height("side_by_side_qq")
                
                ws.write(row, 7, box_text, wrap_fmt)
                req_h_right = ResultsExporter.get_fixed_row_height("side_by_side_box")
                
                ws.set_row(row, max(req_h_left, req_h_right))
                text_row = row
                row += 1
                
                ws.set_row(row, 60)
                image_row = row
                
                # Insert Q-Q plot
                try:
                    from PIL import Image
                    if os.path.exists(plot_paths['normality_before']):
                        ws.insert_image(image_row, 1, plot_paths['normality_before'], {
                            'x_scale': 0.7, 'y_scale': 0.7,
                            'object_position': 3, 'x_offset': 2, 'y_offset': 2
                        })
                        print(f"DEBUG: Successfully inserted Q-Q plot at row {image_row} with robust positioning")
                    else:
                        print(f"DEBUG: Q-Q plot file not found: {plot_paths['normality_before']}")
                        ws.write(image_row, 0, "Q-Q Plot could not be generated", fmt["explanation"])
                except Exception as e:
                    print(f"DEBUG: Error inserting Q-Q plot: {e}")
                    ws.write(image_row, 0, f"Q-Q Plot error: {str(e)}", fmt["explanation"])
                
                # Insert Boxplot
                try:
                    from PIL import Image
                    if os.path.exists(plot_paths['homoscedasticity_before']):
                        ws.insert_image(image_row, 8, plot_paths['homoscedasticity_before'], {
                            'x_scale': 0.7, 'y_scale': 0.7,
                            'object_position': 3, 'x_offset': 2, 'y_offset': 2
                        })
                        print(f"DEBUG: Successfully inserted Boxplot at row {image_row} with robust positioning")
                    else:
                        print(f"DEBUG: Boxplot file not found: {plot_paths['homoscedasticity_before']}")
                        ws.write(image_row, 7, "Boxplot could not be generated", fmt["explanation"])
                except Exception as e:
                    print(f"DEBUG: Error inserting Boxplot: {e}")
                    ws.write(image_row, 7, f"Boxplot error: {str(e)}", fmt["explanation"])
                
                row = image_row + 4
                print(f"DEBUG: ROBUST LAYOUT: Advanced to row {row} with decoupled images")
                
            elif plot_paths['normality_before']:
                # Fallback: Only Q-Q plot available - ROBUST LAYOUT
                ws.merge_range(f'A{row}:F{row}', "📈 NORMALITY EXAMINATION - Q-Q Plot of Model Residuals", fmt["key"])
                ws.set_row(row, 22)
                row += 1
                
                # Use robust text cell instead of merge_range
                wrap_fmt = workbook.add_format({
                    'text_wrap': True, 
                    'valign': 'top',
                    'border': 1,
                    'bg_color': '#F0F8FF'
                })
                
                normality_explanation = (
                    "📊 UNDERSTANDING Q-Q PLOTS (Quantile-Quantile Plots):\n\n"
                    "WHAT IS A Q-Q PLOT?\n"
                    "A Q-Q plot compares your data's distribution to a perfect normal distribution. Each point represents "
                    "one data value. The X-axis shows where that value would fall in a perfect normal distribution, "
                    "and the Y-axis shows where it actually falls in your data.\n\n"
                    
                    "HOW TO READ IT:\n"
                    "• Red diagonal line = What a perfect normal distribution would look like\n"
                    "• Blue dots = Your actual data points (model residuals)\n"
                    "• If dots follow the red line closely → Your data is normally distributed\n"
                    "• If dots deviate from the red line → Your data is not normally distributed\n\n"
                    
                    "PATTERN RECOGNITION:\n"
                    "✅ NORMAL RESIDUALS: Points form a straight line along the red diagonal line\n"
                    "⚠️ NON-NORMAL RESIDUALS: Points curve away from the red line (S-shape = skewed data, "
                    "upward curve at ends = too many outliers, downward curve = too few outliers)\n\n"
                    
                    "WHY MODEL RESIDUALS?\n"
                    "Statistical tests like ANOVA actually examine the prediction errors (residuals) from your statistical model, "
                    "not the raw group data. This gives a more accurate assessment of whether your test assumptions are met."
                )
                
                # Write text in single cell with fixed optimal height
                ws.write(row, 0, normality_explanation, wrap_fmt)
                req_height = ResultsExporter.get_fixed_row_height("qq_plot_explanation")
                ws.set_row(row, req_height)
                row += 1
                
                # Separate image row with robust positioning
                ws.set_row(row, 60)
                try:
                    from PIL import Image
                    if os.path.exists(plot_paths['normality_before']):
                        ws.insert_image(row, 0, plot_paths['normality_before'], {
                            'x_scale': 0.8, 'y_scale': 0.8,
                            'object_position': 3, 'x_offset': 2, 'y_offset': 2
                        })
                        row += 6  # Compact spacing for single plot
                    else:
                        ws.write(row, 0, "Q-Q Plot could not be generated", fmt["explanation"])
                        row += 2
                except Exception as e:
                    print(f"DEBUG: Error inserting normality plot: {e}")
                    ws.write(row, 0, f"Error displaying normality plot: {os.path.basename(plot_paths['normality_before'])}", fmt["explanation"])
                    row += 2
            
            elif plot_paths['homoscedasticity_before']:
                # Fallback: Only Boxplot available - ROBUST LAYOUT
                ws.merge_range(f'A{row}:F{row}', "📊 VARIANCE EQUALITY EXAMINATION - Boxplots", fmt["key"])
                ws.set_row(row, 22)
                row += 1
                
                # Use robust text cell instead of merge_range
                wrap_fmt = workbook.add_format({
                    'text_wrap': True, 
                    'valign': 'top',
                    'border': 1,
                    'bg_color': '#F0F8FF'
                })
                
                variance_explanation = (
                    "📦 UNDERSTANDING BOXPLOTS (Box-and-Whisker Plots):\n\n"
                    "WHAT IS A BOXPLOT?\n"
                    "A boxplot is a visual summary of your data's distribution. It shows the middle 50% of your data (the box), "
                    "the median (line inside the box), the range of typical values (whiskers), and any extreme values (dots).\n\n"
                    
                    "BOXPLOT COMPONENTS EXPLAINED:\n"
                    "• Box height = Middle 50% of your data (called the interquartile range or IQR)\n"
                    "• Horizontal line inside box = Median (the middle value when data is sorted)\n"
                    "• Whiskers (vertical lines) = Typical range of your data (usually 1.5 × IQR from the box edges)\n"
                    "• Dots beyond whiskers = Outliers (unusually high or low values)\n"
                    "• Notches (if present) = Confidence intervals around the median\n\n"
                    
                    "HOW TO USE BOXPLOTS FOR VARIANCE TESTING:\n"
                    "We compare boxplots across groups to see if they have similar variability (spread). "
                    "For statistical tests to work properly, all groups should have roughly equal variances.\n\n"
                    
                    "PATTERN RECOGNITION:\n"
                    "✅ EQUAL VARIANCES: All boxes are roughly the same height with similar whisker lengths\n"
                    "⚠️ UNEQUAL VARIANCES: Some boxes much taller/shorter than others, very different whisker lengths\n"
                    "💡 REMEMBER: We're comparing the spread/variability, not the central values (medians)"
                )
                
                # Write text in single cell with fixed optimal height
                ws.write(row, 0, variance_explanation, wrap_fmt)
                req_height = ResultsExporter.get_fixed_row_height("boxplot_explanation")
                ws.set_row(row, req_height)
                row += 1
                
                # Separate image row with robust positioning
                ws.set_row(row, 60)
                try:
                    from PIL import Image
                    if os.path.exists(plot_paths['homoscedasticity_before']):
                        ws.insert_image(row, 0, plot_paths['homoscedasticity_before'], {
                            'x_scale': 0.8, 'y_scale': 0.8,
                            'object_position': 3, 'x_offset': 2, 'y_offset': 2
                        })
                        row += 6  # Compact spacing for single plot
                    else:
                        ws.write(row, 0, "Boxplot could not be generated", fmt["explanation"])
                        row += 2
                except Exception as e:
                    print(f"DEBUG: Error inserting homoscedasticity plot: {e}")
                    ws.write(row, 0, f"Error displaying homoscedasticity plot: {os.path.basename(plot_paths['homoscedasticity_before'])}", fmt["explanation"])
                    row += 2
            
            
            # Comprehensive interpretation guidance - positioned right after plots - ROBUST LAYOUT
            # Split into logical sections for better readability
            
            # Section 1: Why test assumptions
            why_section = (
                "📚 COMPREHENSIVE INTERPRETATION GUIDE FOR BEGINNERS\n\n"
                "🎯 WHY DO WE TEST THESE ASSUMPTIONS?\n"
                "Statistical tests like ANOVA and t-tests are built on mathematical assumptions. When these assumptions "
                "are met, the tests give reliable and trustworthy results. When violated, the results may be misleading "
                "or completely wrong. The plots above help you visually check whether your data meets these requirements "
                "before trusting your statistical test results."
            )
            
            # Use robust text cell with automatic height calculation
            wrap_fmt = workbook.add_format({
                'text_wrap': True, 
                'valign': 'top',
                'border': 1,
                'bg_color': '#F0F8FF'
            })
            
            ws.write(row, 0, why_section, wrap_fmt)
            why_height = ResultsExporter.get_fixed_row_height("why_section")
            ws.set_row(row, why_height)
            row += 1
            
            # Section 2: Q-Q Plot interpretation
            qq_section = (
                "🔍 DETAILED Q-Q PLOT INTERPRETATION:\n"
                "Think of a Q-Q plot as comparing your data to an 'ideal' normal distribution:\n"
                "• Each blue dot represents one data point (specifically, a model residual/prediction error)\n"
                "• The red line shows where points would fall if your data were perfectly normally distributed\n"
                "• The closer the blue dots stick to the red line, the more normal your data distribution is\n\n"
                
                "What different Q-Q plot patterns mean:\n"
                "✅ EXCELLENT: Straight line along red diagonal = Normal distribution, assumptions met\n"
                "⚠️ CONCERN: S-shaped curve = Skewed data (more values bunched on one side)\n"
                "⚠️ CONCERN: Upward curve at ends = Heavy tails (more extreme values than normal)\n"
                "⚠️ CONCERN: Downward curve at ends = Light tails (fewer extreme values than normal)\n"
                "⚠️ CONCERN: Points scattered far from line = Not normally distributed at all"
            )
            
            ws.write(row, 0, qq_section, wrap_fmt)
            qq_height = ResultsExporter.get_fixed_row_height("qq_section")
            ws.set_row(row, qq_height)
            row += 1
            
            # Section 3: Boxplot interpretation
            boxplot_section = (
                "📊 DETAILED BOXPLOT INTERPRETATION:\n"
                "Boxplots summarize the spread and variability of your data for each group:\n"
                "• The 'box' contains the middle 50% of your data points for that group\n"
                "• A tall box = high variability in that group, short box = low variability\n"
                "• For equal variances assumption, all groups should have similarly-sized boxes\n"
                "• The line inside each box shows the median (middle value) for that group\n\n"
                
                "What different boxplot patterns mean:\n"
                "✅ EXCELLENT: Similar box heights across all groups = Equal variances, assumptions met\n"
                "⚠️ CONCERN: Very different box heights = Unequal variances between groups\n"
                "⚠️ CONCERN: Some groups with many outliers, others with none = Inconsistent data quality\n"
                "⚠️ CONCERN: Extremely different whisker lengths = Very unequal spreads"
            )
            
            ws.write(row, 0, boxplot_section, wrap_fmt)
            boxplot_height = ResultsExporter.get_fixed_row_height("boxplot_section")
            ws.set_row(row, boxplot_height)
            row += 1
            
            # Section 4: Practical decisions
            practical_section = (
                "🎯 MAKING PRACTICAL DECISIONS:\n"
                "• Both plots look GOOD (✅ patterns) → Your statistical test assumptions are met, results are reliable\n"
                "• Either plot shows CONCERNS (⚠️ patterns) → Consider data transformation or switch to non-parametric tests\n"
                "• When in doubt → Consult with a statistician about complex or borderline patterns\n"
                "• Remember: Visual inspection works together with the statistical test numbers above"
            )
            
            ws.write(row, 0, practical_section, wrap_fmt)
            practical_height = ResultsExporter.get_fixed_row_height("practical_section")
            ws.set_row(row, practical_height)
            row += 1
            
            # Section 5: Technical note
            technical_section = (
                "💡 TECHNICAL NOTE FOR ADVANCED USERS:\n"
                "We examine model residuals (prediction errors) rather than raw group data because statistical tests "
                "like ANOVA actually assess whether the model's prediction errors are normally distributed. This approach "
                "provides a more accurate evaluation of whether your chosen statistical test is appropriate for your data."
            )
            
            ws.write(row, 0, technical_section, wrap_fmt)
            technical_height = ResultsExporter.get_fixed_row_height("technical_section")
            ws.set_row(row, technical_height)
            row += 2  # Extra space after complete guide
            
        except Exception as e:
            print(f"DEBUG: Error generating assumption plots: {e}")
            ws.write(row, 0, "Error generating visual examination plots", fmt["explanation"])
            row += 2
    
        # Add a clear post-transformation section (after the transformation announcement)
        transformation = results.get("transformation", "None")
        if transformation and transformation != "None":
            ws.write(row, 0, f"Transformation applied: {transformation}", fmt["section_header"])
            trans_info = results.get("transformation_info", "")
            if trans_info:
                row += 1
                ws.merge_range(f'A{row}:F{row}', f"Details: {trans_info}", fmt["explanation"])
                ws.set_row(row, ResultsExporter.get_fixed_row_height("transformation_info"))
            # Show lambda value if Box-Cox transformation
            if transformation == "boxcox" and "boxcox_lambda" in results:
                row += 1
                lambda_val = results["boxcox_lambda"]
                ws.merge_range(f'A{row}:F{row}', f"Box-Cox Lambda (MLE): {lambda_val:.4f}", fmt["cell"])
            
            # IMPORTANT ADDITION: Note about data usage
            row += 1
            test_type = results.get("test_type", "")
            data_usage_note = (
                f"Note: {'Transformed data WAS used for statistical tests' if test_type == 'parametric' else 'Original (untransformed) data was used for statistical tests'} "
                f"based on test recommendation: {test_type}."
            )
            ws.merge_range(f'A{row}:F{row}', data_usage_note, fmt["explanation"])
            
            # Add post-transformation test results section
            row += 2
            ws.merge_range(f'A{row}:F{row}', "TESTS AFTER TRANSFORMATION", fmt["section_header"])
            row += 1
            
            # Normality tests after transformation
            if "normality_tests" in results and "transformed_data" in results["normality_tests"]:
                ws.write(row, 0, "Normality after transformation:", fmt["key"])
                row += 1
                norm_headers = ["Test", "Statistic", "p-Value", "Normally distributed?", "Interpretation"]
                for i, h in enumerate(norm_headers):
                    ws.write(row, i, h, fmt["header"])
                row += 1
                
                norm_result = results["normality_tests"]["transformed_data"]
                stat = norm_result.get('statistic', 'N/A')
                p_val = norm_result.get('p_value', 'N/A')
                is_normal = (isinstance(p_val, (float, int)) and p_val > 0.05)
                normal_text = "Yes" if is_normal else "No"
                interpretation = (
                    "No significant deviation from normality"
                    if is_normal else
                    "Significant deviation from normality"
                )
                
                values = [
                    "Shapiro-Wilk",
                    f"{stat:.4f}" if isinstance(stat, (float, int)) else stat,
                    f"{p_val:.4f}" if isinstance(p_val, (float, int)) else p_val,
                    normal_text,
                    interpretation
                ]
                
                for col, val in enumerate(values):
                    ws.write(row, col, val, fmt["cell"])
                row += 2
            
            # Homogeneity of variances after transformation
            if "variance_test" in results and "transformed" in results["variance_test"]:
                ws.write(row, 0, "Variance homogeneity after transformation:", fmt["key"])
                row += 1
                var_headers = ["Test", "Statistic", "p-Value", "Variances equal?", "Interpretation"]
                for i, h in enumerate(var_headers):
                    ws.write(row, i, h, fmt["header"])
                row += 1
                
                var_result = results["variance_test"]["transformed"]
                stat = var_result.get('statistic', 'N/A')
                p_val = var_result.get('p_value', 'N/A')
                var_equal = (isinstance(p_val, (float, int)) and p_val > 0.05)
                var_text = "Yes" if var_equal else "No"
                interpretation = (
                    "No significant differences in variances"
                    if var_equal else
                    "Significant differences in variances"
                )
                
                values = [
                    "Brown-Forsythe",
                    f"{stat:.4f}" if isinstance(stat, (float, int)) else stat,
                    f"{p_val:.4f}" if isinstance(p_val, (float, int)) else p_val,
                    var_text,
                    interpretation
                ]
                
                for col, val in enumerate(values):
                    ws.write(row, col, val, fmt["cell"])
                row += 2
            
            # VISUAL EXAMINATION AFTER TRANSFORMATION - SIDE BY SIDE LAYOUT
            try:
                AssumptionVisualizer = get_assumption_visualizer()
                plot_paths = AssumptionVisualizer.generate_assumption_plots(results)
                
                # Create side-by-side layout for after transformation plots
                if plot_paths['normality_after'] and plot_paths['homoscedasticity_after']:
                    ws.merge_range(f'A{row}:M{row}', f"🔄 AFTER {transformation.upper()} TRANSFORMATION - Q-Q Plot & Boxplots Comparison", fmt["section_header"])
                    row += 1
                    
                    # Brief comparison instruction
                    comparison_intro = (
                        "Compare these transformed plots with the original plots above. "
                        "✅ Success: Points closer to red line (left) + similar box heights (right) | "
                        "⚠️ Limited effect: Still curved lines or different box sizes"
                    )
                    ws.merge_range(f'A{row}:M{row}', comparison_intro, fmt["explanation"])
                    ws.set_row(row, ResultsExporter.get_fixed_row_height("comparison_intro"))
                    row += 1
                    
                    try:
                        from PIL import Image
                        
                        # Insert transformed Q-Q plot on the left (columns A:F)
                        # Use consistent scaling (0.7)
                        ws.insert_image(row, 0, plot_paths['normality_after'], 
                                      {'x_scale': 0.7, 'y_scale': 0.7,
                                       'object_position': 3, 'x_offset': 2, 'y_offset': 2})
                        
                        # Insert transformed Boxplot on the right (columns H:M)
                        # Use consistent scaling (0.7)
                        ws.insert_image(row, 7, plot_paths['homoscedasticity_after'], 
                                      {'x_scale': 0.7, 'y_scale': 0.7,
                                       'object_position': 3, 'x_offset': 2, 'y_offset': 2})
                        
                        # Calculate rows needed - use fixed scale factor of 0.7
                        with Image.open(plot_paths['normality_after']) as img1, Image.open(plot_paths['homoscedasticity_after']) as img2:
                            max_height = max(img1.size[1], img2.size[1])
                            scale_factor = 0.7  # Consistent with all other plots
                            image_rows = int((max_height * scale_factor) / 15) + 2
                            row += max(image_rows, 10)  # Compact layout
                            
                    except Exception as e:
                        print(f"DEBUG: Error inserting side-by-side transformed plots: {e}")
                        ws.write(row, 0, "Error displaying transformed assumption plots", fmt["explanation"])
                        row += 3
                        
                elif plot_paths['normality_after'] or plot_paths['homoscedasticity_after']:
                    # Fallback for single plots
                    ws.merge_range(f'A{row}:F{row}', f"� AFTER {transformation.upper()} TRANSFORMATION", fmt["section_header"])
                    row += 1
                    
                    if plot_paths['normality_after']:
                        ws.merge_range(f'A{row}:F{row}', f"📈 Q-Q Plot After {transformation} Transformation", fmt["key"])
                        row += 1
                        try:
                            from PIL import Image
                            # Use consistent scaling (0.7)
                            ws.insert_image(row, 0, plot_paths['normality_after'], 
                                          {'x_scale': 0.7, 'y_scale': 0.7,
                                           'object_position': 3, 'x_offset': 2, 'y_offset': 2})
                            # Use fixed row calculation based on consistent scale
                            row += 25  # Fixed spacing
                        except Exception as e:
                            print(f"DEBUG: Error inserting transformed normality plot: {e}")
                            ws.write(row, 0, "Error displaying transformed normality plot", fmt["explanation"])
                            row += 3
                    
                    if plot_paths['homoscedasticity_after']:
                        ws.merge_range(f'A{row}:F{row}', f"� Boxplots After {transformation} Transformation", fmt["key"])
                        row += 1
                        try:
                            from PIL import Image
                            # Use consistent scaling (0.7)
                            ws.insert_image(row, 0, plot_paths['homoscedasticity_after'], 
                                          {'x_scale': 0.7, 'y_scale': 0.7,
                                           'object_position': 3, 'x_offset': 2, 'y_offset': 2})
                            # Use fixed row calculation based on consistent scale
                            row += 25  # Fixed spacing
                        except Exception as e:
                            print(f"DEBUG: Error inserting transformed homoscedasticity plot: {e}")
                            ws.write(row, 0, "Error displaying transformed homoscedasticity plot", fmt["explanation"])
                            row += 3
                
            except Exception as e:
                print(f"DEBUG: Error generating after-transformation plots: {e}")
                ws.write(row, 0, "Error generating after-transformation visual plots", fmt["explanation"])
                row += 2
        
            # Summary text
            row += 1
            ws.merge_range(f'A{row}:F{row}', "SUMMARY OF ASSUMPTIONS CHECK", fmt["section_header"])
            row += 1
            
            # Generate enhanced summary based on residual tests
            summary = results.get("assumptions_summary", "")
            if not summary:
                # Check if we have residual-based tests to enhance summary
                has_residual_tests = (
                    "test_info" in results 
                    and "pre_transformation" in results["test_info"]
                    and "residuals_normality" in results["test_info"]["pre_transformation"]
                )
                
                if has_residual_tests:
                    test_info = results["test_info"]
                    pre_norm = test_info["pre_transformation"]["residuals_normality"]["is_normal"]
                    pre_var = test_info["pre_transformation"]["variance"]["equal_variance"]
                    transformation = results.get("transformation", "None")
                    
                    if transformation and transformation != "None":
                        post_norm = test_info.get("post_transformation", {}).get("residuals_normality", {}).get("is_normal", False)
                        post_var = test_info.get("post_transformation", {}).get("variance", {}).get("equal_variance", False)
                        summary = (
                            f"MODERN RESIDUAL-BASED ASSUMPTION TESTING:\n\n"
                            f"• Before transformation: Model residuals {'NORMAL' if pre_norm else 'NOT NORMAL'}, "
                            f"variances {'EQUAL' if pre_var else 'UNEQUAL'}\n"
                            f"• {transformation} transformation was applied\n"
                            f"• After transformation: Model residuals {'NORMAL' if post_norm else 'NOT NORMAL'}, "
                            f"variances {'EQUAL' if post_var else 'UNEQUAL'}\n\n"
                            f"IMPORTANT: This analysis uses model residuals (not raw group data) to test normality assumptions. "
                            f"This provides a more accurate assessment of whether the statistical model assumptions are met."
                        )
                    else:
                        summary = (
                            f"MODERN RESIDUAL-BASED ASSUMPTION TESTING:\n\n"
                            f"• Model residuals are {'NORMALLY DISTRIBUTED' if pre_norm else 'NOT NORMALLY DISTRIBUTED'}\n"
                            f"• Group variances are {'EQUAL (homogeneous)' if pre_var else 'UNEQUAL (heterogeneous)'}\n"
                            f"• No transformation was needed/applied\n\n"
                            f"IMPORTANT: This analysis uses model residuals (not raw group data) to test normality assumptions. "
                            f"This provides a more accurate assessment of whether the statistical model assumptions are met."
                        )
                else:
                    summary = (
                        "The assumptions were checked as documented above. "
                        "See test results for each group for details."
                    )
                    
            ws.merge_range(f'A{row}:F{row}', summary, fmt["explanation"])
            ws.set_row(row, ResultsExporter.get_fixed_row_height("results_summary"))
            row += 2
        
            # Decision tree
            ws.merge_range(f'A{row}:F{row}', "DECISION TREE FOR TEST SELECTION", fmt["section_header"])
            row += 1
            decision_tree = results.get("decision_tree_text", None)
            if not decision_tree:
                decision_tree = (
                    "1. Are the data in all groups normally distributed?\n"
                    "2. Are the variances between the groups equal?\n"
                    "→ If yes: Parametric test (e.g., t-test, ANOVA)\n"
                    "→ If no: Try transformation or non-parametric test (e.g., Mann-Whitney U, Kruskal-Wallis)"
                )
            ws.merge_range(f'A{row}:F{row+4}', decision_tree, fmt["explanation"])
            ws.set_row(row, ResultsExporter.get_fixed_row_height("decision_tree_text"))
            
            
    @staticmethod
    def _write_results_sheet(workbook, results, fmt, sheet_name="Statistical Results"):
        ws = workbook.add_worksheet(sheet_name)
        ws.set_column(0, 12, 22)
        ws.set_row(0, 30)
        ws.merge_range('A1:M1', 'STATISTICAL RESULTS', fmt["title"])

        # Introduction
        # Detect if this is a nonparametric permutation ANOVA
        is_perm = results.get("test_type", "").lower() == "non-parametric" or results.get("permutation_test", False)
        
        intro = (
            "This sheet contains the main results of the statistical analysis: "
            "test statistics, p-value, effect size, confidence interval, power, "
            "and – if relevant – alternative tests. "
        )
        
        # Add explanation about Freedman-Lane when permutation tests are used
        if is_perm:
            intro += (
                "For permutation-based nonparametric ANOVA, p-values are computed using the Freedman–Lane scheme."
            )
            
        ws.merge_range('A2:M2', intro, fmt["explanation"])
        ws.set_row(1, ResultsExporter.get_fixed_row_height("results_intro"))

        # Main result table
        row = 4
        
        # Define column headers (with permutation-specific headers when applicable)
        headers = [
            "Test", "Test statistic", 
            "Permutation p-value" if is_perm else "p-Value", 
            "Effect size", "Confidence interval", "Power", "Significant?",
            "Permutation Test" if is_perm else "",
            "Permutation Scheme" if is_perm else ""
        ]
        # Remove empty headers
        headers = [h for h in headers if h]
        
        for col, header in enumerate(headers):
            ws.write(row, col, header, fmt["header"])
        row += 1

        # Values
        test = results.get("test", "N/A")
        stat_val = (
            results.get("t_statistic") or results.get("u_statistic") or
            results.get("f_statistic") or results.get("h_statistic") or
            results.get("statistic", None)
        )
        p_val = results.get("p_value", None)
        # Special handling for non-parametric test effects
        if results.get("test_type") == "non-parametric" and "effects" in results:
            for effect in results.get("effects", []):
                if effect.get("name") and "within_effect" in effect.get("name", "").lower():
                    stat_val = effect.get("F")
                    p_val = effect.get("p")
                    print(f"DEBUG: Using effect data for non-parametric test: F={stat_val}, p={p_val}")
                    break
        effect_size = results.get("effect_size", None)
        ci = results.get("confidence_interval", None)
        power = results.get("power", None)
        is_significant = p_val is not None and p_val < 0.05

        stat_val_str = f"{stat_val:.4f}" if isinstance(stat_val, (float, int)) else (stat_val or "N/A")
        
        # Format p-value differently for permutation tests
        if is_perm:
            p_val_str = f"{p_val:.4f}" if isinstance(p_val, (float, int)) else (p_val or "N/A")
        else:
            p_val_str = (
                "p < 0.001" if isinstance(p_val, (float, int)) and p_val < 0.001 else
                f"p = {p_val:.4f}" if isinstance(p_val, (float, int)) else (p_val or "N/A")
            )
            
        effect_type = results.get("effect_size_type", "")
        if effect_size is not None:
            if effect_type == "cohen_d":
                if effect_size < 0.2: effect_str = f"{effect_size:.4f} (very small)"
                elif effect_size < 0.5: effect_str = f"{effect_size:.4f} (small)"
                elif effect_size < 0.8: effect_str = f"{effect_size:.4f} (medium)"
                else: effect_str = f"{effect_size:.4f} (large)"
            elif effect_type in ["eta_squared", "partial_eta_squared"]:
                if effect_size < 0.01: effect_str = f"{effect_size:.4f} (very small)"
                elif effect_size < 0.06: effect_str = f"{effect_size:.4f} (small)"
                elif effect_size < 0.14: effect_str = f"{effect_size:.4f} (medium)"
                else: effect_str = f"{effect_size:.4f} (large)"
            elif effect_type == "epsilon_squared":
                if effect_size < 0.01: effect_str = f"{effect_size:.4f} (very small)"
                elif effect_size < 0.08: effect_str = f"{effect_size:.4f} (small)"
                elif effect_size < 0.26: effect_str = f"{effect_size:.4f} (medium)"
                else: effect_str = f"{effect_size:.4f} (large)"
            else:
                effect_str = f"{effect_size:.4f}"
        else:
            effect_str = "N/A"
            
        if ci is not None and isinstance(ci, (tuple, list)) and len(ci) == 2 and ci[0] is not None and ci[1] is not None:
            ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]"
        else:
            ci_str = "N/A"
            
        power_str = f"{power:.2f}" if isinstance(power, (float, int)) else "N/A"
        sig_str = "Yes" if is_significant else "No"

        # Create list of values to write
        values = [test, stat_val_str, p_val_str, effect_str, ci_str, power_str, sig_str]
        
        # Add permutation-specific columns if needed
        if is_perm:
            values.extend(["Yes", perm_scheme := results.get("permutation_scheme", "Freedman–Lane")])

        # Write all values
        for col, val in enumerate(values):
            fmtx = fmt["significant"] if (col == 2 and is_significant) or (col == 6 and is_significant) else fmt["cell"]
            ws.write(row, col, val, fmtx)
        row += 2

        # Show sphericity corrections if present
        if "sphericity_corrections" in results:
            ws.merge_range(f'A{row}:F{row}', "CORRECTIONS FOR SPHERICITY VIOLATION", fmt["section_header"])
            row += 1
            
            # Show which correction was used, based on Girden (1992)
            if "correction_used" in results:
                ws.merge_range(f'A{row}:F{row}', f"Correction used: {results['correction_used']}", fmt["explanation"])
                row += 1
            
            corr_headers = ["Correction type", "Epsilon", "Corrected df1", "Corrected df2", "Corrected p-Value", "Significant?"]
            for col, header in enumerate(corr_headers):
                ws.write(row, col, header, fmt["header"])
            row += 1
            
            # Greenhouse-Geisser correction
            gg_corr = results["sphericity_corrections"]["greenhouse_geisser"]
            gg_p = gg_corr["p_value"]
            gg_sig = gg_p < results.get("alpha", 0.05) if isinstance(gg_p, (float, int)) else False
            ws.write(row, 0, "Greenhouse-Geisser", fmt["cell"])
            ws.write(row, 1, f"{gg_corr['epsilon']:.4f}", fmt["cell"])
            ws.write(row, 2, f"{gg_corr['df1']:.4f}", fmt["cell"])
            ws.write(row, 3, f"{gg_corr['df2']:.4f}", fmt["cell"])
            ws.write(row, 4, f"{gg_p:.4f}" if isinstance(gg_p, (float, int)) else "N/A", 
                    fmt["significant"] if gg_sig else fmt["cell"])
            ws.write(row, 5, "Yes" if gg_sig else "No",
                    fmt["significant"] if gg_sig else fmt["cell"])
            row += 1
            
            # Huynh-Feldt correction
            hf_corr = results["sphericity_corrections"]["huynh_feldt"]
            hf_p = hf_corr["p_value"]
            hf_sig = hf_p < results.get("alpha", 0.05) if isinstance(hf_p, (float, int)) else False
            ws.write(row, 0, "Huynh-Feldt", fmt["cell"])
            ws.write(row, 1, f"{hf_corr['epsilon']:.4f}", fmt["cell"])
            ws.write(row, 2, f"{hf_corr['df1']:.4f}", fmt["cell"])
            ws.write(row, 3, f"{hf_corr['df2']:.4f}", fmt["cell"])
            ws.write(row, 4, f"{hf_p:.4f}" if isinstance(hf_p, (float, int)) else "N/A", 
                    fmt["significant"] if hf_sig else fmt["cell"])
            ws.write(row, 5, "Yes" if hf_sig else "No",
                    fmt["significant"] if hf_sig else fmt["cell"])
            row += 2

        # Alternative tests
        alt_tests = results.get("alternative_tests", [])
        if alt_tests:
            ws.merge_range(f'A{row}:F{row}', "RESULTS OF ALTERNATIVE TESTS", fmt["section_header"])
            row += 1
            alt_headers = [
                "Test", "Test statistic", "p-Value", "Significant?", "Effect size", "Effect interpretation"
            ]
            for col, header in enumerate(alt_headers):
                ws.write(row, col, header, fmt["header"])
            row += 1
            for alt in alt_tests:
                test = alt.get("test", "")
                stat = alt.get("statistic", "N/A")
                p = alt.get("p_value", "N/A")
                eff = alt.get("effect_size", "N/A")
                eff_type = alt.get("effect_size_type", "")
                sig = p < 0.05 if isinstance(p, (float, int)) else False
                if eff != "N/A" and eff is not None:
                    if eff_type == "cohen_d":
                        if eff < 0.2: effint = "very small"
                        elif eff < 0.5: effint = "small"
                        elif eff < 0.8: effint = "medium"
                        else: effint = "large"
                    elif eff_type in ["eta_squared", "partial_eta_squared"]:
                        if eff < 0.01: effint = "very small"
                        elif eff < 0.06: effint = "small"
                        elif eff < 0.14: effint = "medium"
                        else: effint = "large"
                    else:
                        effint = ""
                else:
                    effint = ""
                vals = [
                    test,
                    f"{stat:.4f}" if isinstance(stat, (float, int)) else stat,
                    f"{p:.4f}" if isinstance(p, (float, int)) else p,
                    "Yes" if sig else "No",
                    f"{eff:.4f}" if isinstance(eff, (float, int)) else eff,
                    effint
                ]
                for col, val in enumerate(vals):
                    fmtx = fmt["significant"] if (col == 2 and sig) or (col == 3 and sig) else fmt["cell"]
                    ws.write(row, col, val, fmtx)
                row += 1
            row += 1

        # Interpretation
        ws.merge_range(f'A{row}:F{row}', "INTERPRETATION", fmt["section_header"])
        row += 1
        interpretation = (
            "The analysis shows a statistically significant difference between the groups."
            if is_significant else
            "The analysis shows no statistically significant difference between the groups."
        )
        ws.merge_range(f'A{row}:F{row}', interpretation, fmt["explanation"])
        ws.set_row(row, ResultsExporter.get_fixed_row_height("results_interpretation"))
        row += 2

        # Add a permutation explanation if applicable
        if is_perm:
            ws.merge_range(f'A{row}:F{row}', "ABOUT PERMUTATION TESTS", fmt["section_header"])
            row += 1
            perm_explanation = (
                "This analysis used a permutation-based approach with the Freedman–Lane scheme. "
                "In permutation tests, the data is repeatedly shuffled (permuted) to create a "
                "distribution of test statistics under the null hypothesis. The p-value represents "
                "the proportion of permuted datasets that produce a test statistic as extreme as "
                "or more extreme than the observed one. This approach is more robust when parametric "
                "assumptions are violated."
            )
            ws.merge_range(f'A{row}:F{row}', perm_explanation, fmt["explanation"])
            ws.set_row(row, ResultsExporter.get_fixed_row_height("permutation_explanation"))
            row += 2

        # Post-hoc tests information
        ws.merge_range(f'A{row}:F{row}', "AVAILABLE POST-HOC TESTS", fmt["section_header"])
        row += 1
        
        posthoc_info = (
            "This analysis software provides various post-hoc tests for different situations:\n\n"
            "PARAMETRIC POST-HOC TESTS (when normality assumptions are met):\n"
            "• Tukey HSD: Compares all possible pairs while controlling family-wise error rate\n"
            "• Dunnett Test: Compares all groups against a single control group\n"
            "• Custom Paired t-tests: User-selected pairs with Holm-Sidak correction\n\n"
            "NON-PARAMETRIC POST-HOC TESTS (when normality assumptions are violated):\n"
            "• Dunn Test: Rank-based comparisons of all pairs with Holm-Sidak correction\n"
            "• Custom Mann-Whitney-U Tests: User-selected pairs with Sidak correction\n\n"
            "REPEATED MEASURES POST-HOC TESTS (for dependent samples):\n"
            "• Dependent Post-hoc: Paired t-tests or Wilcoxon tests based on normality\n\n"
            "The appropriate test is automatically selected based on your data characteristics, "
            "or you can choose specific comparisons through the user interface."
        )
        ws.merge_range(f'A{row}:F{row}', posthoc_info, fmt["explanation"])
        ws.set_row(row, ResultsExporter.get_fixed_row_height("posthoc_info_detailed"))
        row += 2
        
        # Show which specific post-hoc test was performed, if any
        posthoc_test = results.get("posthoc_test", None)
        if posthoc_test:
            ws.merge_range(f'A{row}:F{row}', f"POST-HOC TEST PERFORMED: {posthoc_test}", fmt["section_header"])
            row += 1
            
            # Get number of pairwise comparisons
            pairwise_count = len(results.get("pairwise_comparisons", []))
            comparison_info = f"Number of pairwise comparisons: {pairwise_count}"
            if pairwise_count > 0:
                comparison_info += " (see 'Pairwise Comparisons' sheet for details)"
            else:
                comparison_info += " (no comparisons performed - main test not significant or error occurred)"
            
            ws.merge_range(f'A{row}:F{row}', comparison_info, fmt["explanation"])
            ws.set_row(row, ResultsExporter.get_fixed_row_height("comparison_info_detailed"))
    

    @staticmethod
    def _write_descriptive_sheet(workbook, results, fmt, sheet_name="Descriptive Statistics"):
        ws = workbook.add_worksheet(sheet_name)
        ws.set_column(0, 9, 20)
        ws.set_row(0, 28)
        ws.merge_range('A1:J1', 'DESCRIPTIVE STATISTICS', fmt["title"])

        # Introduction
        desc_explanation = (
            "This sheet contains summary statistics for each group:\n"
            "• n: Sample size of the group\n"
            "• Mean: Average of the values\n"
            "• 95% Confidence interval: Confidence interval for the mean\n"
            "• Median, standard deviation, standard error, minimum, maximum\n"
            "Transformed values are also shown if a transformation was performed."
        )
        ws.merge_range('A2:J2', desc_explanation, fmt["explanation"])
        ws.set_row(1, ResultsExporter.get_fixed_row_height("descriptive_intro"))

        # Header
        headers = [
            "Group", "n", "Mean", "95% CI Lower", "95% CI Upper",
            "Median", "Std. Dev.", "Std. Error", "Min", "Max"
        ]
        ws.set_row(3, 22)
        for col, header in enumerate(headers):
            ws.write(3, col, header, fmt["header"])

        # Original data
        desc = results.get('descriptive', results.get('descriptive_stats', {}))
        row = 4
        for group, grp in desc.items():
            n = grp.get('n', None)
            mean = grp.get('mean', None)
            median = grp.get('median', None)
            std = grp.get('std', None)
            stderr = grp.get('stderr', None)
            minv = grp.get('min', None)
            maxv = grp.get('max', None)
            
            # Calculate confidence interval if needed
            ci_lower = grp.get('ci_lower', None)
            ci_upper = grp.get('ci_upper', None)
            
            if ci_lower is None or ci_upper is None:
                try:
                    from scipy.stats import t
                    if n and n > 1 and stderr is not None:
                        ci_lower, ci_upper = t.interval(0.95, n - 1, loc=mean, scale=stderr)
                    else:
                        ci_lower, ci_upper = None, None
                except Exception:
                    ci_lower, ci_upper = None, None

            ws.write(row, 0, group, fmt["cell"])
            ws.write(row, 1, n if n is not None else "", fmt["cell"])
            ws.write(row, 2, f"{mean:.4f}" if mean is not None else "", fmt["cell"])
            ws.write(row, 3, f"{ci_lower:.4f}" if ci_lower is not None else "", fmt["cell"])
            ws.write(row, 4, f"{ci_upper:.4f}" if ci_upper is not None else "", fmt["cell"])
            ws.write(row, 5, f"{median:.4f}" if median is not None else "", fmt["cell"])
            ws.write(row, 6, f"{std:.4f}" if std is not None else "", fmt["cell"])
            ws.write(row, 7, f"{stderr:.4f}" if stderr is not None else "", fmt["cell"])
            ws.write(row, 8, f"{minv:.4f}" if minv is not None else "", fmt["cell"])
            ws.write(row, 9, f"{maxv:.4f}" if maxv is not None else "", fmt["cell"])
            row += 1

        # Transformed data, if present - Enhanced section
        desc_t = results.get('descriptive_transformed', {})
        transformation = results.get('transformation', 'None')
        
        # Show transformed data even if it wasn't used for tests
        if desc_t and transformation and transformation != 'None':
            row += 2
            header_text = "Descriptive Statistics (after transformation)"
            if results.get("test_type") != "parametric":
                header_text += " - Not used for statistical test"
            
            ws.merge_range(f'A{row}:J{row}', header_text, fmt["section_header"])
            row += 1
            
            # Add transformation method info
            transform_info = f"Transformation method: {transformation.capitalize()}"
            if transformation == "boxcox" and "boxcox_lambda" in results:
                transform_info += f", λ = {results['boxcox_lambda']:.4f}"
            ws.merge_range(f'A{row}:J{row}', transform_info, fmt["explanation"])
            row += 1
            
            # Column headers for transformed data
            for col, header in enumerate(headers):
                ws.write(row, col, header, fmt["header"])
            row += 1
            
            # Write transformed data
            for group, grp in desc_t.items():
                n = grp.get('n', None)
                mean = grp.get('mean', None)
                median = grp.get('median', None)
                std = grp.get('std', None)
                stderr = grp.get('stderr', None)
                minv = grp.get('min', None)
                maxv = grp.get('max', None)
                
                # Calculate confidence interval if needed
                ci_lower = grp.get('ci_lower', None)
                ci_upper = grp.get('ci_upper', None)
                
                if ci_lower is None or ci_upper is None:
                    try:
                        from scipy.stats import t
                        if n and n > 1 and stderr is not None:
                            ci_lower, ci_upper = t.interval(0.95, n - 1, loc=mean, scale=stderr)
                        else:
                            ci_lower, ci_upper = None, None
                    except Exception:
                        ci_lower, ci_upper = None, None
                        
                ws.write(row, 0, group, fmt["cell"])
                ws.write(row, 1, n if n is not None else "", fmt["cell"])
                ws.write(row, 2, f"{mean:.4f}" if mean is not None else "", fmt["cell"])
                ws.write(row, 3, f"{ci_lower:.4f}" if ci_lower is not None else "", fmt["cell"])
                ws.write(row, 4, f"{ci_upper:.4f}" if ci_upper is not None else "", fmt["cell"])
                ws.write(row, 5, f"{median:.4f}" if median is not None else "", fmt["cell"])
                ws.write(row, 6, f"{std:.4f}" if std is not None else "", fmt["cell"])
                ws.write(row, 7, f"{stderr:.4f}" if stderr is not None else "", fmt["cell"])
                ws.write(row, 8, f"{minv:.4f}" if minv is not None else "", fmt["cell"])
                ws.write(row, 9, f"{maxv:.4f}" if maxv is not None else "", fmt["cell"])
                row += 1
        
    @staticmethod
    def _write_pairwise_sheet(workbook, results, fmt, sheet_name="Pairwise Comparisons"):
        # RECONSTRUCTION SAFETY: If main list is empty but component lists exist, rebuild it
        if (not results.get('pairwise_comparisons') or len(results.get('pairwise_comparisons', [])) == 0):
            # Try to reconstruct from between and within comparisons
            all_comparisons = []
            
            if "between_pairwise_comparisons" in results and results["between_pairwise_comparisons"]:
                all_comparisons.extend(results["between_pairwise_comparisons"])
                
            if "within_pairwise_comparisons" in results and results["within_pairwise_comparisons"]:
                all_comparisons.extend(results["within_pairwise_comparisons"])
                
            if all_comparisons:
                # Use the reconstructed comparisons
                results["pairwise_comparisons"] = all_comparisons
                print(f"DEBUG: Reconstructed {len(all_comparisons)} pairwise comparisons for Excel export")
        
        print(f"DEBUG POSTHOC EXCEL: Number of pairwise comparisons when writing: {len(results.get('pairwise_comparisons', []))}")
        ws = workbook.add_worksheet(sheet_name)
        ws.set_column(0, 7, 22)  # Increased column count for CI
        ws.set_row(0, 28)
        posthoc_test_name = results.get("posthoc_test", "")
        title_text = 'RESULTS OF PAIRWISE COMPARISONS'
        if posthoc_test_name:
            title_text += f' – {posthoc_test_name}'
        ws.merge_range('A1:H1', title_text, fmt["title"])  # Increased merge range
    
        # Introduction
        pw_explanation = (
            "This sheet shows the results of the pairwise comparisons between the groups.\n"
            "• Group 1 & Group 2: The compared groups\n"
            "• Test: Test performed for the comparison\n"
            "• p-Value: (Corrected) significance value of the comparison\n"
            "• Corrected: Indicates whether a correction for multiple testing was applied\n"
            "• Significant: 'Yes' if p < Alpha (usually 0.05)\n"
            "• Effect size: Magnitude of the difference (e.g., Cohen's d, Hedges' g)\n"
            "• Classical Cohen limits (d = small≤0.2; medium≤0.5; large≤0.8)\n"
            "• 95% CI: Confidence interval for the difference between groups (if calculated)\n"
            "Interpretation of significance (typical): * p<0.05; ** p<0.01; *** p<0.001\n\n"
            "Available Post-hoc Tests:\n"
            "• Tukey HSD: Compares all pairs, controls family-wise error rate\n"
            "• Dunnett Test: Compares all groups to a control group\n"
            "• Custom Paired t-tests (Holm-Sidak): User-selected pairs with Holm-Sidak correction\n"
            "• Dunn Test: Non-parametric all pairwise comparisons with Holm-Sidak correction\n"
            "• Custom Mann-Whitney-U (Sidak): User-selected pairs with Sidak correction\n"
            "• Dependent Post-hoc: For repeated measures designs (paired t-tests or Wilcoxon)"
        )
        ws.merge_range('A2:H2', pw_explanation, fmt["explanation"])  # Text in Excel row 2
        ws.set_row(1, ResultsExporter.get_fixed_row_height("pairwise_intro"))  # Set height for row 2 (1-indexed)
    
        # Header
        headers = ["Group 1", "Group 2", "Test", "p-Value", "Corrected", "Significant", "Effect size", "95% CI Difference"]
        for col, header in enumerate(headers):
            ws.write(3, col, header, fmt["header"])
    
        # Data
        comps = results.get("pairwise_comparisons", [])
        if comps is None:  # Extra safety check
            comps = []
            print("WARNING: pairwise_comparisons was None, converted to empty list")
        
        print(f"DEBUG: comps type = {type(comps)}, content = {str(comps[:3]) if comps else 'empty'}")
        print(f"DEBUG: comps type = {type(comps)}, content = {comps[:3]}...")
        row = 4
    
        if len(comps) == 0:
            message = "No pairwise comparisons performed or available."
            if results.get("p_value") is not None and results.get("p_value") >= results.get("alpha", 0.05) and len(results.get("groups", [])) > 2:
                message = "No pairwise comparisons performed because the main test was not significant."
            elif results.get("error") and "Post-hoc" in results.get("error"):
                message = f"Error in post-hoc tests: {results.get('error')}"
    
            ws.merge_range(row, 0, row, len(headers)-1, message, fmt["cell"])
            return
    
        for comp_idx, comp in enumerate(comps):
            group1 = str(comp.get('group1', 'N/A'))
            group2 = str(comp.get('group2', 'N/A'))
            test_name = comp.get('test', posthoc_test_name or 'N/A')
            pval = comp.get('p_value', None)
    
            # Correction info
            corrected_info = "N/A"
            if comp.get('corrected') is True:
                corrected_info = comp.get('correction', 'Yes') if comp.get('correction') else 'Yes'
            elif comp.get('corrected') is False:
                corrected_info = "No"
    
            is_sign = comp.get('significant', False)
            if pval is not None and not isinstance(is_sign, bool):  # Fallback if 'significant' field is missing
                is_sign = pval < results.get("alpha", 0.05)
    
            effect_size_val = comp.get('effect_size', None)
            effect_size_type = comp.get('effect_size_type', '')
    
            pval_str = "N/A"
            if isinstance(pval, (float, int)):
                if pval < 0.001:
                    pval_str = "<0.001"
                else:
                    pval_str = f"{pval:.4f}"
    
            sign_str = "Yes" if is_sign else "No"
    
            eff_text = "N/A"
            eff_fmt = fmt["cell"]
            if isinstance(effect_size_val, (float, int)):
                magnitude = ""
                # Simplified magnitude for pairwise comparisons
                if effect_size_type.lower() in ["cohen_d", "hedges_g", "r"]:
                    if abs(effect_size_val) < 0.2:
                        magnitude = "very small"
                        eff_fmt = fmt["effect_weak"]
                    elif abs(effect_size_val) < 0.5:
                        magnitude = "small"
                        eff_fmt = fmt["effect_weak"]
                    elif abs(effect_size_val) < 0.8:
                        magnitude = "medium"
                        eff_fmt = fmt["effect_medium"]
                    else:
                        magnitude = "large"
                        eff_fmt = fmt["effect_strong"]
                eff_text = f"{effect_size_val:.3f}"
                if magnitude:
                    eff_text += f" ({magnitude})"
    
            ci_val = comp.get('confidence_interval', None)
            ci_str = "N/A"
            if ci_val and isinstance(ci_val, (tuple, list)) and len(ci_val) == 2 and ci_val[0] is not None and ci_val[1] is not None:
                ci_str = f"[{ci_val[0]:.3f}, {ci_val[1]:.3f}]"
    
            current_row_data = [group1, group2, test_name, pval_str, corrected_info, sign_str, eff_text, ci_str]
    
            for col, val_to_write in enumerate(current_row_data):
                current_fmt = fmt["cell"]
                if headers[col] == "p-Value" and is_sign:
                    current_fmt = fmt["significant"]
                elif headers[col] == "Significant" and is_sign:
                    current_fmt = fmt["significant"]
                elif headers[col] == "Effect size" and isinstance(effect_size_val, (float, int)):
                    current_fmt = eff_fmt  # Use pre-determined format for effect size
                ws.write(row + comp_idx, col, val_to_write, current_fmt)
    
    @staticmethod
    def _write_decision_tree_sheet(workbook, results, fmt, sheet_name="Decision Tree", pre_generated_tree=None):
        """Write decision tree sheet with visualization."""
        from decisiontreevisualizer import DecisionTreeVisualizer
        
        sheet = workbook.add_worksheet(sheet_name)
        sheet.set_column('A:A', 120)  # Wide column for the image
        
        # Write header
        sheet.write(0, 0, "Decision Tree Visualization", fmt["title"])
        sheet.write(1, 0, "Test Methodology: This decision tree shows the hypothesis workflow and statistical decisions.", fmt["explanation"])
        sheet.write(2, 0, "Highlighted path: The red path shows the decisions made for this specific analysis.", fmt["explanation"])
        
        # Use pre-generated path if provided, otherwise generate a new one
        image_path = pre_generated_tree
        if not image_path or not os.path.exists(image_path):
            print(f"DEBUG: No valid pre-generated tree, generating new one...")
            image_path = DecisionTreeVisualizer.generate_and_save_for_excel(results)
            # Track the newly generated file
            ResultsExporter.track_temp_file(image_path)
        
        # Insert the image if it exists
        if image_path and os.path.exists(image_path):
            print(f"Inserting decision tree image: {image_path}")
            
            # Get image dimensions to scale appropriately
            try:
                from PIL import Image
                with Image.open(image_path) as img:
                    width, height = img.size
                    aspect_ratio = width / height
                    print(f"DEBUG: Image dimensions: {width}x{height}, aspect ratio: {aspect_ratio:.2f}")
                    
                    # Scale to fit within Excel cell constraints
                    scale_factor = 0.75 if width > 4000 else 1.0
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                    print(f"DEBUG: Scale factor: {scale_factor}, resulting size: {new_width}x{new_height}")
                    
                    # Insert image at row 5
                    sheet.insert_image(5, 0, image_path, {'x_scale': scale_factor, 'y_scale': scale_factor})
                    print(f"Successfully inserted decision tree image at row 5")
                    
            except Exception as e:
                print(f"DEBUG: Error processing image dimensions: {e}")
                # Fallback: insert without scaling
                sheet.insert_image(5, 0, image_path)
            
            # Add image filename for reference
            sheet.write(3, 0, f"Image file: {os.path.basename(image_path)}", fmt["explanation"])
        else:
            sheet.write(5, 0, "Error: Failed to generate decision tree visualization.", fmt["explanation"])
        
        return image_path
    
    @staticmethod
    def _write_rawdata_sheet(workbook, results, fmt, sheet_name="Raw Data"):
        ws = workbook.add_worksheet(sheet_name)
        ws.set_column(0, 10, 15)
        
        # Title and description
        ws.merge_range('A1:K1', "RAW DATA", fmt["title"])
        ws.write('A3', "This sheet shows the original data and processing steps for each group.", fmt["explanation"])
        ws.write('A4', "These data are the basis of all calculations.", fmt["explanation"])
        
        # Check if this is a non-parametric test with special data storage
        if results.get("test_type") == "non-parametric":
            # Handle non-parametric test data
            original_data = results.get("original_data", {})
            aggregated_data = results.get("aggregated_data", {})
            ranked_data = results.get("ranked_data", {})
            
            if original_data or aggregated_data or ranked_data:
                row = 6
                
                # Original Data Section
                ws.merge_range(f'A{row}:K{row}', "ORIGINAL DATA (Before any processing)", fmt["section_header"])
                row += 1
                ws.write(row, 0, "Group", fmt["header"])
                ws.write(row, 1, "Original Values", fmt["header"])
                row += 1
                
                for group_name, values in original_data.items():
                    ws.write(row, 0, group_name, fmt["cell"])
                    if values:
                        values_str = ", ".join([f"{v:.4f}" if isinstance(v, (int, float)) else str(v) for v in values])
                        ws.write(row, 1, values_str, fmt["cell"])
                    else:
                        ws.write(row, 1, "No data", fmt["cell"])
                    row += 1
                
                row += 1
                
                # Aggregated Data Section (if different from original)
                if aggregated_data and any(original_data.get(k, []) != aggregated_data.get(k, []) for k in aggregated_data.keys()):
                    ws.merge_range(f'A{row}:K{row}', "AGGREGATED DATA (Means of replicates)", fmt["section_header"])
                    row += 1
                    ws.write(row, 0, "Group", fmt["header"])
                    ws.write(row, 1, "Aggregated Values", fmt["header"])
                    row += 1
                    
                    for group_name, values in aggregated_data.items():
                        ws.write(row, 0, group_name, fmt["cell"])
                        if values:
                            values_str = ", ".join([f"{v:.4f}" if isinstance(v, (int, float)) else str(v) for v in values])
                            ws.write(row, 1, values_str, fmt["cell"])
                        else:
                            ws.write(row, 1, "No data", fmt["cell"])
                        row += 1
                    
                    row += 1
                
                # Ranked Data Section
                if ranked_data:
                    ws.merge_range(f'A{row}:K{row}', "RANKED DATA (Used for statistical test)", fmt["section_header"])
                    row += 1
                    ws.write(row, 0, "Group", fmt["header"])
                    ws.write(row, 1, "Ranked Values", fmt["header"])
                    row += 1
                    
                    for group_name, values in ranked_data.items():
                        ws.write(row, 0, group_name, fmt["cell"])
                        if values:
                            values_str = ", ".join([f"{v:.2f}" if isinstance(v, (int, float)) else str(v) for v in values])
                            ws.write(row, 1, values_str, fmt["cell"])
                        else:
                            ws.write(row, 1, "No data", fmt["cell"])
                        row += 1
                    
                    # Add explanation
                    row += 2
                    explanation = results.get("data_explanation", {})
                    if explanation:
                        ws.merge_range(f'A{row}:K{row}', "DATA PROCESSING EXPLANATION", fmt["section_header"])
                        row += 1
                        for key, value in explanation.items():
                            ws.write(row, 0, key.replace("_", " ").title() + ":", fmt["key"])
                            ws.write(row, 1, str(value), fmt["explanation"])
                            row += 1
                    
                    return
        
        # Handle regular parametric test data or fallback
        raw_data = results.get("raw_data", {})
        transformed_data = results.get("raw_data_transformed", {})
        
        if not raw_data and not transformed_data:
            # Try to get data from descriptive statistics
            descriptive = results.get("descriptive", {})
            if descriptive:
                row = 6
                ws.write(row, 0, "Group", fmt["header"])
                ws.write(row, 1, "Sample Size", fmt["header"])
                ws.write(row, 2, "Mean", fmt["header"])
                ws.write(row, 3, "Std Dev", fmt["header"])
                row += 1
                
                for group_name, stats in descriptive.items():
                    ws.write(row, 0, group_name, fmt["cell"])
                    ws.write(row, 1, stats.get("n", "N/A"), fmt["cell"])
                    ws.write(row, 2, f"{stats.get('mean', 0):.4f}" if stats.get('mean') is not None else "N/A", fmt["cell"])
                    ws.write(row, 3, f"{stats.get('std', 0):.4f}" if stats.get('std') is not None else "N/A", fmt["cell"])
                    row += 1
            else:
                ws.write(6, 0, "Group", fmt["header"])
                ws.write(6, 1, "Original Value", fmt["header"])
                ws.write(7, 0, "No data available", fmt["cell"])
            return
        
        # Handle parametric test data
        row = 6
        if raw_data:
            ws.merge_range(f'A{row}:K{row}', "ORIGINAL DATA", fmt["section_header"])
            row += 1
            ws.write(row, 0, "Group", fmt["header"])
            ws.write(row, 1, "Original Values", fmt["header"])
            row += 1
            
            for group_name, values in raw_data.items():
                ws.write(row, 0, group_name, fmt["cell"])
                if values:
                    values_str = ", ".join([f"{v:.4f}" if isinstance(v, (int, float)) else str(v) for v in values])
                    ws.write(row, 1, values_str, fmt["cell"])
                else:
                    ws.write(row, 1, "No data", fmt["cell"])
                row += 1
            
            row += 1
        
        if transformed_data and transformed_data != raw_data:
            ws.merge_range(f'A{row}:K{row}', "TRANSFORMED DATA", fmt["section_header"])
            row += 1
            ws.write(row, 0, "Group", fmt["header"])
            ws.write(row, 1, "Transformed Values", fmt["header"])
            row += 1
            
            for group_name, values in transformed_data.items():
                ws.write(row, 0, group_name, fmt["cell"])
                if values:
                    values_str = ", ".join([f"{v:.4f}" if isinstance(v, (int, float)) else str(v) for v in values])
                    ws.write(row, 1, values_str, fmt["cell"])
                else:
                    ws.write(row, 1, "No data", fmt["cell"])
                row += 1
            
    @staticmethod
    def _write_analysislog_sheet(workbook, log, fmt, sheet_name="Analysis Log"):
        ws = workbook.add_worksheet(sheet_name)
        ws.set_column(0, 0, 80)
        ws.set_row(0, 28)
        ws.write('A1', 'ANALYSIS LOG', fmt["title"])

        # Introduction/Legend
        log_explanation = (
            "This sheet documents the course of the statistical analysis and the decisions made. "
            "The log provides a chronological overview of the individual analysis steps, "
            "methods used, transformations, test selection, and special notes.\n"
            "Each paragraph describes a key step or decision in the analysis process."
        )
        ws.write(1, 0, log_explanation, fmt["explanation"])
        ws.set_row(1, ResultsExporter.get_fixed_row_height("log_explanation"))

        row = 3

        # Apply structured formatting to all logs
        if isinstance(log, str):
            # Split the log into clear sections
            sections = {
                "header": [],
                "setup": [],
                "analysis": [],
                "results": [],
                "posthoc": []
            }

            current_section = "header"
            lines = log.split('\n')

            # Enhanced section detection for all log types
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Section detection - handle both advanced and basic test logs
                if "--- ANALYSE ---" in line or "--- ANALYSIS ---" in line:
                    current_section = "analysis"
                    continue
                elif "Testempfehlung:" in line or "Test recommendation:" in line:
                    current_section = "analysis"
                elif "Durchgeführter Test:" in line or "Test performed:" in line or "Two-Way ANOVA" in line or "t-test" in line or "ANOVA" in line:
                    current_section = "results"
                elif "paarweise Vergleiche" in line or "Post-hoc" in line or "Pairwise comparisons:" in line:
                    current_section = "posthoc"
                elif line.startswith("Datei:") or line.startswith("Arbeitsblatt:") or line.startswith("File:") or line.startswith("Worksheet:") or line.startswith("Group column:") or line.startswith("Value column"):
                    current_section = "setup"

                sections[current_section].append(line)

            # Write each section with consistent formatting
            # Header section
            for line in sections["header"]:
                ws.write(row, 0, line, fmt["cell"])
                row += 1
            row += 1  # Empty line after header

            # Setup section - file info, columns, groups
            if sections["setup"]:
                ws.write(row, 0, "DATASET INFORMATION", fmt["section_header"])
                row += 1
                for line in sections["setup"]:
                    ws.write(row, 0, line, fmt["cell"])
                    row += 1
                row += 1  # Empty line after setup

            # Analysis section - tests performed
            if sections["analysis"]:
                ws.write(row, 0, "ANALYSIS PREPARATION AND ASSUMPTIONS", fmt["section_header"])
                row += 1
                for line in sections["analysis"]:
                    ws.write(row, 0, line, fmt["cell"])
                    row += 1
                row += 1  # Empty line after analysis

            # Results section - main results
            if sections["results"]:
                ws.write(row, 0, "MAIN RESULTS", fmt["section_header"])
                row += 1
                
                # Format all results with bullet points for improved readability
                for line in sections["results"]:
                    if ":" in line and ("p =" in line or "p=" in line or "p <" in line or "p<" in line):
                        # This looks like a result line - add bullet point and highlight if significant
                        ws.write(row, 0, f"• {line}", fmt["significant"] if "significant" in line.lower() else fmt["cell"])
                    else:
                        ws.write(row, 0, line, fmt["cell"])
                    row += 1
                    
                row += 1  # Empty line after results

            # Post-hoc section
            if sections["posthoc"]:
                ws.write(row, 0, "POST-HOC ANALYSES", fmt["section_header"])
                row += 1
                
                # Add heading for post-hoc tests
                ws.write(row, 0, "Pairwise Comparisons (Post-hoc):", fmt["key"])
                row += 1
                
                # Format post-hoc results with bullet points for all tests
                for line in sections["posthoc"]:
                    if " vs " in line and ("p =" in line or "p=" in line or "p <" in line or "p<" in line):
                        # This is a comparison line - add bullet point and highlight if significant
                        is_significant = "significant" in line.lower() and "not significant" not in line.lower()
                        ws.write(row, 0, f"• {line}", fmt["significant"] if is_significant else fmt["cell"])
                    else:
                        ws.write(row, 0, line, fmt["cell"])
                    row += 1
                    
                row += 1  # Empty line after post-hoc

        elif isinstance(log, list):
            for entry in log:
                ws.write(row, 0, str(entry), fmt["cell"])
                row += 1
        else:
            ws.write(row, 0, str(log), fmt["cell"])

        # Add summary section for all tests when significant results are found
        if isinstance(log, str) and "significant" in log.lower() and "p < 0.05" in log:
            ws.write(row, 0, "SUMMARY OF RESULTS", fmt["section_header"])
            row += 1

            summary_text = (
                "The statistical analysis revealed significant effects. "
                "Please refer to the detailed results for the specific findings. "
                "Post-hoc tests were performed for more precise group comparisons where applicable."
            )
            ws.write(row, 0, summary_text, fmt["explanation"])
            ws.set_row(row, ResultsExporter.get_fixed_row_height("log_explanation"))
                
    @staticmethod
    def get_text_height(text, width):
        """
        Simplified wrapper that uses calc_robust_text_height for consistency.
        Maintains backward compatibility with existing code.
        """
        # Use the optimized calc_robust_text_height method
        return ResultsExporter.calc_robust_text_height(text, width, line_height_pts=15.0)
    
    @staticmethod
    def get_fixed_row_height(field_identifier: str, default_height: float = 25.0) -> float:
        """
        Get predefined optimal heights for specific Excel fields.
        Since text content is always the same, we can use fixed optimal heights.
        """
        fixed_heights = {
            # Summary sheet
            "summary_conclusion": 40.0,           # KEY STATEMENT conclusion text
            "summary_navigation": 120.0,          # NAVIGATION TO DETAILED RESULTS - reduced from 140
            "summary_posthoc_info": 100.0,        # POST-HOC TESTS PERFORMED - increased from 80
            
            # Assumptions sheet  
            "assumptions_intro": 60.0,            # Introduction text at top
            "assumptions_overview": 200.0,        # OVERVIEW OF ASSUMPTIONS - very long with bullet points
            "sphericity_explanation": 45.0,       # Sphericity test explanation
            "normality_explanation": 45.0,        # Normality test explanation
            "homogeneity_explanation": 45.0,      # Homogeneity test explanation
            "visual_intro": 210.0,                # VISUAL ASSUMPTION CHECKING - very long with emojis
            "qq_plot_explanation": 420.0,         # Q-Q plot detailed explanation - very detailed
            "boxplot_explanation": 420.0,         # Boxplot detailed explanation - very detailed
            "practical_advice": 160.0,            # Practical interpretation advice - very long
            "technical_details": 140.0,           # Technical details section - long
            
            # ANOVA explanations - these are the long detailed ones
            "anova_source_explanation": 140.0,    # Source column explanation - increased for full text
            "anova_ss_explanation": 120.0,        # SS explanation - increased for full text  
            "anova_df_explanation": 160.0,        # DF explanation - increased for full text
            "anova_ms_explanation": 120.0,        # MS explanation - increased for full text
            "anova_f_explanation": 140.0,         # F-statistic explanation - increased for full text
            "anova_p_explanation": 140.0,         # p-value explanation - increased for full text
            "anova_np2_explanation": 160.0,       # Partial Eta Squared - increased for full text
            
            # Results interpretation
            "results_interpretation": 180.0,      # HOW TO INTERPRET - significantly increased for 4 points
            
            # Additional specific text blocks that need fixed heights
            "general_note": 60.0,                  # General notes in summary
            "intro_anova_text": 70.0,             # ANOVA introduction text
            "sphericity_detail": 45.0,            # Sphericity test details
            "normality_detail": 45.0,             # Normality test details
            "side_by_side_qq": 260.0,             # Side-by-side Q-Q explanations - significantly increased for long text
            "side_by_side_box": 180.0,            # Side-by-side boxplot explanations - very long
            "why_section": 160.0,                 # Why sections in visual - long with explanations
            "qq_section": 180.0,                  # Q-Q plot sections - very detailed
            "boxplot_section": 180.0,             # Boxplot sections - very detailed
            "practical_section": 180.0,           # Practical advice sections - very long
            "technical_section": 160.0,           # Technical details sections - long
            "transformation_info": 50.0,          # Transformation information
            "comparison_intro": 100.0,            # Comparison introduction text
            "results_summary": 80.0,              # Results summary text
            "decision_tree_text": 90.0,           # Decision tree explanations
            "results_intro": 80.0,                # Results sheet introduction
            "permutation_explanation": 80.0,      # Permutation test explanations
            "posthoc_info_detailed": 50.0,        # Detailed post-hoc information (reduced)
            "comparison_info_detailed": 70.0,     # Detailed comparison information
            "descriptive_intro": 70.0,            # Descriptive statistics introduction
            "pairwise_intro": 140.0,              # Pairwise comparisons introduction - has many bullet points
            
            # Other sheets
            "descriptive_explanation": 80.0,      # Descriptive statistics explanation
            "pairwise_explanation": 100.0,        # Pairwise comparisons explanation
            "log_explanation": 40.0,              # Analysis log explanation
            
            # Default categories
            "short_text": 25.0,                   # Headers and short labels
            "medium_text": 45.0,                  # Medium explanations
            "long_text": 80.0,                    # Long explanations
            "bullet_list": 120.0,                 # Lists with multiple bullets
        }
        
        return fixed_heights.get(field_identifier, default_height)

    @staticmethod
    def calc_robust_text_height(text: str, total_char_width: int, line_height_pts: float = 15.0) -> float:

        import textwrap
        if not text:
            return 18.0
        
        # If total_char_width is very large (like 330), convert to reasonable estimate
        if total_char_width > 200:
            # Assume it's a merge_range A:F (55 + 5*20 = 155)
            total_char_width = 155
        elif total_char_width > 100 and total_char_width < 200:
            # Assume it's approximately correct for merge range
            pass  
        elif total_char_width < 30:
            # Too small, assume single column A
            total_char_width = 55
        
        # Character counting with intelligent text type detection
        char_count = len(text)
        line_breaks = text.count('\n')
        bullet_points = text.count('•')
        
        # Use textwrap to get more accurate line estimation
        wrapped_lines = textwrap.wrap(text.replace('\n', ' '), width=total_char_width)
        textwrap_lines = len(wrapped_lines)
        
        # Different estimation based on text characteristics
        if char_count < 80:
            # Very short text - minimal calculation
            estimated_lines = max(1, textwrap_lines + line_breaks)
            buffer_factor = 1.25  # Adequate buffer for short texts
        elif char_count < 200:
            # Medium text - conservative
            estimated_lines = max(1, textwrap_lines + line_breaks) 
            buffer_factor = 1.30  # Good buffer for medium text
        elif bullet_points > 3:
            # Bullet-heavy text - needs extra space
            estimated_lines = max(textwrap_lines, line_breaks + bullet_points)
            buffer_factor = 1.45  # Extra buffer for bullet formatting
        else:
            # Normal longer text - use wrapped lines with line breaks
            estimated_lines = max(2, textwrap_lines + line_breaks)
            buffer_factor = 1.35  # Good standard buffer
        
        # Calculate final height with adequate buffer
        height = max(30.0, estimated_lines * line_height_pts * buffer_factor)
        
        return height
    
    @staticmethod
    def track_temp_file(filepath):
        """Track a temporary file for later cleanup"""
        if filepath and os.path.exists(filepath):
            ResultsExporter._temp_files.add(filepath)
            print(f"DEBUG: Tracking temporary file: {filepath}")
        return filepath
    
    @staticmethod
    def cleanup_old_tree_files(max_age_hours=24):
        """Clean up old decision tree files."""
        import glob
        import time
        # import os  # Already imported at top
        import tempfile
        
        # First check temp directory for pattern-matching files
        temp_dir = tempfile.gettempdir()
        current_time = time.time()
        cleaned_count = 0
        
        # Find and delete old files
        for pattern in ["decision_tree_*.png", "tree_visualization_*.png"]:
            for file_path in glob.glob(os.path.join(temp_dir, pattern)):
                try:
                    file_age = current_time - os.path.getctime(file_path)
                    if file_age > max_age_hours * 3600:
                        os.remove(file_path)
                        cleaned_count += 1
                        print(f"Removed old decision tree: {file_path}")
                except Exception as e:
                    print(f"Error cleaning up file {file_path}: {str(e)}")
        
        # Also check legacy location (Documents/StatisticsTemp)
        docs_dir = os.path.join(os.path.expanduser("~"), "Documents", "StatisticsTemp")
        if os.path.exists(docs_dir):
            for pattern in ["decision_tree_*.png", "tree_visualization_*.png"]:
                for file_path in glob.glob(os.path.join(docs_dir, pattern)):
                    try:
                        file_age = current_time - os.path.getctime(file_path)
                        if file_age > max_age_hours * 3600:
                            os.remove(file_path)
                            cleaned_count += 1
                            print(f"Removed old decision tree from legacy location: {file_path}")
                    except Exception as e:
                        print(f"Error cleaning up legacy file {file_path}: {str(e)}")

    @staticmethod
    def _write_enhanced_assumption_tests(workbook, results, fmt, ws, start_row):
        """
        Writes enhanced assumption tests for RM/Mixed ANOVA including:
        - Sphericity tests with corrections
        - Between-factor assumptions (Mixed ANOVA)
        - Within-factor sphericity (Mixed ANOVA)
        - Interaction assumptions (Mixed ANOVA)
        
        Returns the next available row number.
        """
        try:
            row = start_row
            
            # Input validation
            if not isinstance(results, dict):
                ws.write(row, 0, "⚠️ ERROR: Invalid results data format", fmt["significant"])
                return row + 2
            
            if not all(param is not None for param in [workbook, fmt, ws]):
                raise ValueError("Missing required parameters: workbook, fmt, or ws cannot be None")
            
            # Check if this is a RM or Mixed ANOVA with enhanced assumption tests
            test_type = results.get("test", "")
            has_enhanced_tests = any([
                "sphericity_test" in results and isinstance(results["sphericity_test"], dict),
                "between_assumptions" in results,
                "within_sphericity_test" in results,
                "interaction_assumptions" in results
            ])
            
            if not has_enhanced_tests:
                return row  # No enhanced tests to display
            
            # Enhanced Assumption Tests Header
            ws.merge_range(f'A{row}:F{row}', "🔬 ENHANCED ASSUMPTION TESTING (RM/MIXED ANOVA)", fmt["section_header"])
            row += 1
            
            enhanced_intro = (
                "This section provides comprehensive assumption testing for Repeated Measures and Mixed ANOVA designs. "
                "These tests go beyond basic normality and variance checks to include specialized assumptions "
                "for within-subjects and between-subjects factors."
            )
            
            intro_wrap_fmt = workbook.add_format({
                'text_wrap': True, 
                'valign': 'top',
                'border': 1,
                'bg_color': '#E8F4FD'  # Light blue background
            })
            ws.write(row, 0, enhanced_intro, intro_wrap_fmt)
            ws.set_row(row, 40)
            row += 2
            
            # 1. Enhanced Sphericity Testing (RM ANOVA)
            if "sphericity_test" in results and isinstance(results["sphericity_test"], dict):
                try:
                    row = ResultsExporter._write_enhanced_sphericity_section(workbook, results, fmt, ws, row)
                except Exception as e:
                    ws.write(row, 0, f"⚠️ ERROR in sphericity section: {str(e)}", fmt["significant"])
                    row += 2
            
            # 2. Between-Factor Assumptions (Mixed ANOVA)
            if "between_assumptions" in results:
                try:
                    row = ResultsExporter._write_between_factor_assumptions(workbook, results, fmt, ws, row)
                except Exception as e:
                    ws.write(row, 0, f"⚠️ ERROR in between-factor assumptions: {str(e)}", fmt["significant"])
                    row += 2
            
            # 3. Within-Factor Sphericity (Mixed ANOVA)
            if "within_sphericity_test" in results:
                try:
                    row = ResultsExporter._write_within_factor_sphericity(workbook, results, fmt, ws, row)
                except Exception as e:
                    ws.write(row, 0, f"⚠️ ERROR in within-factor sphericity: {str(e)}", fmt["significant"])
                    row += 2
            
            # 4. Interaction Assumptions (Mixed ANOVA)
            if "interaction_assumptions" in results:
                try:
                    row = ResultsExporter._write_interaction_assumptions(workbook, results, fmt, ws, row)
                except Exception as e:
                    ws.write(row, 0, f"⚠️ ERROR in interaction assumptions: {str(e)}", fmt["significant"])
                    row += 2
            
            return row
            
        except Exception as e:
            # Global error handling
            error_msg = f"⚠️ CRITICAL ERROR in enhanced assumption tests: {str(e)}"
            try:
                ws.write(start_row, 0, error_msg, fmt["significant"])
                ws.write(start_row + 1, 0, "Please check your data and try again.", fmt["cell"])
            except:
                # If even writing the error fails, just return
                pass
            return start_row + 3
    
    @staticmethod
    def _write_enhanced_sphericity_section(workbook, results, fmt, ws, start_row):
        """
        Writes comprehensive sphericity testing results with user-friendly explanations and robust error handling.
        
        This function creates a detailed sphericity section in the Excel export that includes:
        - Clear explanation of what sphericity means and why it matters
        - Mauchly's test results with interpretation guidance
        - Visual indicators for assumption violations
        - Practical recommendations for next steps
        
        Args:
            workbook: xlsxwriter.Workbook object for creating Excel formats
            results (dict): Statistical analysis results containing sphericity test data
            fmt (dict): Dictionary of Excel formatting objects
            ws: xlsxwriter.Worksheet object for writing data
            start_row (int): Row number to start writing from
            
        Returns:
            int: Next available row number after writing sphericity section
            
        Expected results structure:
            results["sphericity_test"] = {
                "statistic": float,      # Mauchly's W statistic
                "p_value": float,        # p-value of the test
                "assumption_met": bool   # True if sphericity assumption is met
            }
            
        Example:
            # Write sphericity section starting at row 10
            next_row = _write_enhanced_sphericity_section(workbook, results, fmt, ws, 10)
        """
        row = start_row
        
        # Enhanced section header with context
        ws.write(row, 0, "� SPHERICITY ASSUMPTION (MAUCHLY'S TEST)", fmt["section_header"])
        row += 1
        
        sphericity_test = results["sphericity_test"]
        
        # Enhanced sphericity explanation with practical guidance
        sphericity_explanation = (
            "💡 WHAT IS SPHERICITY?\n"
            "Sphericity means that the variances of differences between ALL pairs of conditions are equal.\n"
            "This is crucial for Repeated Measures ANOVA because the test compares differences between conditions.\n\n"
            
            "🧪 MAUCHLY'S TEST INTERPRETATION:\n"
            "• H₀ (Null Hypothesis): Sphericity assumption is met (variances are equal)\n"
            "• H₁ (Alternative): Sphericity assumption is violated\n"
            "• p > 0.05: ✅ Assumption met → Use uncorrected ANOVA results\n"
            "• p ≤ 0.05: ⚠️ Assumption violated → Use corrected p-values instead\n\n"
            
            "🎯 PRACTICAL IMPACT:\n"
            "If sphericity is violated and you ignore it, your ANOVA results will be too liberal.\n"
            "This means you'll find 'significant' effects more often than you should (increased Type I error).\n"
            "Always check this assumption and use corrections when needed!"
        )
        
        try:
            explanation_fmt = workbook.add_format({
                'text_wrap': True, 
                'valign': 'top',
                'border': 1,
                'bg_color': '#F0F8FF',
                'font_size': 10
            })
            ws.write(row, 0, sphericity_explanation, explanation_fmt)
            ws.set_row(row, 80)  # Taller row for more content
            row += 2
            
            # Sphericity test results table with enhanced interpretation
            sph_headers = ["Test", "W Statistic", "Chi-Square", "df", "p-Value", "Result", "Practical Interpretation"]
            for i, header in enumerate(sph_headers):
                ws.write(row, i, header, fmt["header"])
            row += 1
            
            # Extract sphericity test data with validation
            test_name = sphericity_test.get("test_name", "Mauchly's Test for Sphericity")
            W = sphericity_test.get("W", "N/A")
            chi_square = sphericity_test.get("chi_square", "N/A")
            df = sphericity_test.get("df", "N/A")
            p_value = sphericity_test.get("p_value", "N/A")
            sphericity_met = sphericity_test.get("sphericity_assumed", None)
            
            # Data quality validation
            data_quality_issues = []
            if W == "N/A" or W is None:
                data_quality_issues.append("Missing W statistic")
            if p_value == "N/A" or p_value is None:
                data_quality_issues.append("Missing p-value")
            if df == "N/A" or df is None:
                data_quality_issues.append("Missing degrees of freedom")
            
            if data_quality_issues:
                ws.write(row, 0, f"⚠️ DATA QUALITY WARNING: {', '.join(data_quality_issues)}", fmt["significant"])
                row += 1
            
            # Enhanced interpretation based on results
            if sphericity_met is False:
                result_text = "⚠️ VIOLATED"
                practical_interpretation = "USE CORRECTED p-values! See corrections below."
            elif sphericity_met is True:
                result_text = "✅ ASSUMPTION MET"
                practical_interpretation = "Safe to use uncorrected ANOVA results."
            else:
                result_text = "Unknown"
                practical_interpretation = "Cannot determine - check data quality."
        
            # Format values with better presentation
            w_str = f"{W:.6f}" if isinstance(W, (float, int)) else str(W)
            chi_str = f"{chi_square:.4f}" if isinstance(chi_square, (float, int)) else str(chi_square)
            df_str = str(df) if df is not None else "N/A"
            p_str = f"{p_value:.6f}" if isinstance(p_value, (float, int)) else str(p_value)
            
            values = [test_name, w_str, chi_str, df_str, p_str, result_text, practical_interpretation]
            
            # Apply color-coded formatting based on sphericity result
            for col, val in enumerate(values):
                if sphericity_met is False and col >= 4:  # Highlight violations in red
                    ws.write(row, col, val, fmt["significant"])
                elif sphericity_met is True and col >= 4:  # Highlight success in normal format
                    ws.write(row, col, val, fmt["cell"])
                else:
                    ws.write(row, col, val, fmt["cell"])
            row += 2
            
            # Add practical guidance box
            if sphericity_met is False:
                guidance_text = (
                    "🚨 IMPORTANT: Sphericity is violated!\n"
                    "→ Do NOT interpret uncorrected ANOVA p-values\n"
                    "→ Use Greenhouse-Geisser or Huynh-Feldt corrected results instead\n"
                    "→ See correction table below for valid p-values"
                )
                guidance_fmt = workbook.add_format({
                    'text_wrap': True,
                    'valign': 'top', 
                    'border': 1,
                    'bg_color': '#FFE4E1',  # Light red background
                    'font_color': '#8B0000',  # Dark red text
                    'bold': True
                })
            else:
                guidance_text = (
                    "✅ GOOD NEWS: Sphericity assumption is met!\n"
                    "→ You can safely interpret uncorrected ANOVA results\n"
                    "→ Corrections are not necessary but provided for reference"
                )
                guidance_fmt = workbook.add_format({
                    'text_wrap': True,
                    'valign': 'top',
                    'border': 1, 
                    'bg_color': '#F0FFF0',  # Light green background
                    'font_color': '#006400'  # Dark green text
                })
            
            ws.write(row, 0, guidance_text, guidance_fmt)
            ws.set_row(row, 30)
            row += 2
            
            # Sphericity corrections if available
            if "sphericity_corrections" in results:
                row = ResultsExporter._write_sphericity_corrections(workbook, results, fmt, ws, row)
            
            return row
            
        except Exception as e:
            # Error handling for sphericity section
            error_msg = f"⚠️ ERROR in sphericity analysis: {str(e)}"
            try:
                ws.write(start_row, 0, error_msg, fmt["significant"])
                ws.write(start_row + 1, 0, "Sphericity test data may be incomplete or corrupted.", fmt["cell"])
            except:
                pass
            return start_row + 3
    
    @staticmethod
    def _write_sphericity_corrections(workbook, results, fmt, ws, start_row):
        """
        Writes sphericity correction results with comprehensive explanations and robust error handling.
        
        This function creates a detailed corrections section that helps users understand:
        - Why corrections are needed when sphericity is violated
        - The difference between Greenhouse-Geisser and Huynh-Feldt corrections
        - Which correction to use based on epsilon values
        - The corrected p-values they should report
        
        Args:
            workbook: xlsxwriter.Workbook object for creating Excel formats
            results (dict): Statistical analysis results containing correction data
            fmt (dict): Dictionary of Excel formatting objects
            ws: xlsxwriter.Worksheet object for writing data
            start_row (int): Row number to start writing from
            
        Returns:
            int: Next available row number after writing corrections section
            
        Expected results structure:
            results["sphericity_corrections"] = {
                "greenhouse_geisser": {
                    "epsilon": float,           # GG epsilon value
                    "corrected_p": float,       # GG corrected p-value
                    "corrected_f": float,       # GG corrected F-statistic
                    "df_num": float,           # Numerator degrees of freedom
                    "df_den": float            # Denominator degrees of freedom
                },
                "huynh_feldt": {
                    "epsilon": float,           # HF epsilon value
                    "corrected_p": float,       # HF corrected p-value
                    "corrected_f": float,       # HF corrected F-statistic
                    "df_num": float,           # Numerator degrees of freedom
                    "df_den": float            # Denominator degrees of freedom
                }
            }
            
        The function automatically determines which correction to recommend based on:
        - Greenhouse-Geisser: Recommended when epsilon ≤ 0.75 (more conservative)
        - Huynh-Feldt: Recommended when epsilon > 0.75 (less conservative)
        
        Example:
            # Write corrections section starting at row 15
            next_row = _write_sphericity_corrections(workbook, results, fmt, ws, 15)
        """
        try:
            row = start_row
            
            # Input validation
            if not isinstance(results, dict):
                ws.write(row, 0, "⚠️ ERROR: Invalid results format for corrections", fmt["significant"])
                return row + 2
                
            if "sphericity_corrections" not in results:
                ws.write(row, 0, "⚠️ WARNING: No sphericity corrections available", fmt["significant"])
                return row + 2
                
            corrections = results["sphericity_corrections"]
            if not isinstance(corrections, dict):
                ws.write(row, 0, "⚠️ ERROR: Invalid corrections data format", fmt["significant"])
                return row + 2
            
            ws.write(row, 0, "🔧 SPHERICITY CORRECTIONS - YOUR VALID P-VALUES", fmt["section_header"])
            row += 1
            
            # Enhanced corrections explanation with practical guidance
            corrections_explanation = (
                "🎯 WHY CORRECTIONS ARE NEEDED:\n"
                "When sphericity is violated, standard ANOVA p-values are invalid (too liberal).\n"
                "Corrections adjust the degrees of freedom to provide statistically valid results.\n\n"
                
                "🔧 AVAILABLE CORRECTIONS:\n\n"
                
                "GREENHOUSE-GEISSER CORRECTION:\n"
                "• More conservative (safer) correction\n"
                "• Reduces degrees of freedom more dramatically\n"
                "• Recommended when ε (epsilon) ≤ 0.75\n"
                "• Use this when you want to be extra cautious about Type I errors\n\n"
                
                "HUYNH-FELDT CORRECTION:\n"
                "• Less conservative correction\n"
                "• Closer to original degrees of freedom\n"
                "• Recommended when ε (epsilon) > 0.75\n"
                "• More powerful test (better at detecting real effects)\n\n"
                
                "🚨 WHICH CORRECTION TO USE?\n"
                "The system automatically selects the most appropriate correction based on epsilon value.\n"
                "Look for the 'RECOMMENDED' label in the results table below."
            )
            
            explanation_fmt = workbook.add_format({
                'text_wrap': True, 
                'valign': 'top',
                'border': 1,
                'bg_color': '#FFF8DC',  # Light yellow background
                'font_size': 10
            })
            ws.write(row, 0, corrections_explanation, explanation_fmt)
            ws.set_row(row, 100)  # Taller row for comprehensive explanation
            row += 2
            
            # Enhanced corrections table with recommendations
            correction_headers = ["Correction Type", "Epsilon (ε)", "Adjusted df", "F-statistic", "Corrected p-value", "Recommended?", "Interpretation"]
            for i, header in enumerate(correction_headers):
                ws.write(row, i, header, fmt["header"])
            row += 1
            
            # Greenhouse-Geisser correction
            gg_data = corrections.get("greenhouse_geisser", {})
            gg_epsilon = gg_data.get("epsilon", "N/A")
            gg_df = gg_data.get("df", "N/A")
            gg_f_stat = gg_data.get("F", "N/A")
            gg_p_value = gg_data.get("p_value", "N/A")
            
            # Huynh-Feldt correction
            hf_data = corrections.get("huynh_feldt", {})
            hf_epsilon = hf_data.get("epsilon", "N/A")
            hf_df = hf_data.get("df", "N/A")
            hf_f_stat = hf_data.get("F", "N/A")
            hf_p_value = hf_data.get("p_value", "N/A")
            
            # Determine which correction is recommended
            epsilon_val = gg_epsilon if isinstance(gg_epsilon, (int, float)) else None
            gg_recommended = epsilon_val is not None and epsilon_val <= 0.75
            hf_recommended = epsilon_val is not None and epsilon_val > 0.75
            
            # Write corrections data with error handling
            try:
                # Write Greenhouse-Geisser row
                gg_epsilon_str = f"{gg_epsilon:.4f}" if isinstance(gg_epsilon, (int, float)) else str(gg_epsilon)
                gg_df_str = f"{gg_df:.2f}" if isinstance(gg_df, (int, float)) else str(gg_df)
                gg_f_str = f"{gg_f_stat:.4f}" if isinstance(gg_f_stat, (int, float)) else str(gg_f_stat)
                gg_p_str = f"{gg_p_value:.6f}" if isinstance(gg_p_value, (int, float)) else str(gg_p_value)
                gg_rec_str = "⭐ YES (Conservative)" if gg_recommended else "No"
                gg_interp = "Use this p-value!" if gg_recommended else "Alternative option"
                
                gg_values = ["Greenhouse-Geisser", gg_epsilon_str, gg_df_str, gg_f_str, gg_p_str, gg_rec_str, gg_interp]
                
                for col, val in enumerate(gg_values):
                    cell_fmt = fmt["recommended"] if gg_recommended and col >= 4 else fmt["cell"]
                    ws.write(row, col, val, cell_fmt)
                row += 1
                
                # Write Huynh-Feldt row
                hf_epsilon_str = f"{hf_epsilon:.4f}" if isinstance(hf_epsilon, (int, float)) else str(hf_epsilon)
                hf_df_str = f"{hf_df:.2f}" if isinstance(hf_df, (int, float)) else str(hf_df)
                hf_f_str = f"{hf_f_stat:.4f}" if isinstance(hf_f_stat, (int, float)) else str(hf_f_stat)
                hf_p_str = f"{hf_p_value:.6f}" if isinstance(hf_p_value, (int, float)) else str(hf_p_value)
                hf_rec_str = "⭐ YES (Less Conservative)" if hf_recommended else "No"
                hf_interp = "Use this p-value!" if hf_recommended else "Alternative option"
                
                hf_values = ["Huynh-Feldt", hf_epsilon_str, hf_df_str, hf_f_str, hf_p_str, hf_rec_str, hf_interp]
                
                for col, val in enumerate(hf_values):
                    cell_fmt = fmt["recommended"] if hf_recommended and col >= 4 else fmt["cell"]
                    ws.write(row, col, val, cell_fmt)
                row += 2
                
            except Exception as table_error:
                ws.write(row, 0, f"⚠️ ERROR writing corrections table: {str(table_error)}", fmt["significant"])
                row += 2
            
            # Final recommendation box
            try:
                if gg_recommended:
                    final_rec = (
                        "📋 FINAL RECOMMENDATION: Use Greenhouse-Geisser Correction\n"
                        f"→ Your corrected p-value is: {gg_p_str}\n"
                        f"→ Epsilon = {gg_epsilon_str} (≤ 0.75, so conservative correction is appropriate)\n"
                        "→ This correction protects against Type I errors when sphericity is violated"
                    )
                    rec_fmt = workbook.add_format({
                        'text_wrap': True,
                        'valign': 'top',
                        'border': 1,
                        'bg_color': '#E6F3FF',  # Light blue
                        'font_color': '#0066CC',
                        'bold': True
                    })
                elif hf_recommended:
                    final_rec = (
                        "📋 FINAL RECOMMENDATION: Use Huynh-Feldt Correction\n"
                        f"→ Your corrected p-value is: {hf_p_str}\n"
                        f"→ Epsilon = {hf_epsilon_str} (> 0.75, so less conservative correction is appropriate)\n"
                        "→ This correction is less conservative while still controlling Type I errors"
                    )
                    rec_fmt = workbook.add_format({
                        'text_wrap': True,
                        'valign': 'top',
                        'border': 1,
                        'bg_color': '#E6F3FF',  # Light blue
                        'font_color': '#0066CC',
                        'bold': True
                    })
                else:
                    final_rec = (
                        "📋 RECOMMENDATION: Check epsilon values\n"
                        "→ Cannot determine optimal correction\n"
                        "→ Consider using the more conservative Greenhouse-Geisser correction\n"
                        "→ Consult with a statistician if unsure"
                    )
                    rec_fmt = workbook.add_format({
                        'text_wrap': True,
                        'valign': 'top',
                        'border': 1,
                        'bg_color': '#FFF0E6',  # Light orange
                        'font_color': '#CC6600'
                    })
                
                ws.write(row, 0, final_rec, rec_fmt)
                ws.set_row(row, 40)
                row += 2
                
            except Exception as rec_error:
                ws.write(row, 0, f"⚠️ ERROR writing recommendations: {str(rec_error)}", fmt["significant"])
                row += 2
            
            return row
            
        except Exception as e:
            # Global error handling for corrections section
            error_msg = f"⚠️ CRITICAL ERROR in sphericity corrections: {str(e)}"
            try:
                ws.write(start_row, 0, error_msg, fmt["significant"])
                ws.write(start_row + 1, 0, "Correction data may be incomplete or corrupted.", fmt["cell"])
            except:
                pass
            return start_row + 3
        
        explanation_fmt = workbook.add_format({
            'text_wrap': True, 
            'valign': 'top',
            'border': 1,
            'bg_color': '#FFF8DC',  # Light yellow background
            'font_size': 10
        })
        ws.write(row, 0, corrections_explanation, explanation_fmt)
        ws.set_row(row, 100)  # Taller row for comprehensive explanation
        row += 2
        
        # Enhanced corrections table with recommendations
        correction_headers = ["Correction Type", "Epsilon (ε)", "Adjusted df", "F-statistic", "Corrected p-value", "Recommended?", "Interpretation"]
        for i, header in enumerate(correction_headers):
            ws.write(row, i, header, fmt["header"])
        row += 1
        
        # Greenhouse-Geisser correction
        gg_data = corrections.get("greenhouse_geisser", {})
        gg_epsilon = gg_data.get("epsilon", "N/A")
        gg_df = gg_data.get("df", "N/A")
        gg_f_stat = gg_data.get("F", "N/A")
        gg_p_value = gg_data.get("p_value", "N/A")
        
        # Huynh-Feldt correction
        hf_data = corrections.get("huynh_feldt", {})
        hf_epsilon = hf_data.get("epsilon", "N/A")
        hf_df = hf_data.get("df", "N/A")
        hf_f_stat = hf_data.get("F", "N/A")
        hf_p_value = hf_data.get("p_value", "N/A")
        
        # Determine which correction is recommended
        epsilon_val = gg_epsilon if isinstance(gg_epsilon, (int, float)) else None
        gg_recommended = epsilon_val is not None and epsilon_val <= 0.75
        hf_recommended = epsilon_val is not None and epsilon_val > 0.75
        
        # Write Greenhouse-Geisser row
        gg_epsilon_str = f"{gg_epsilon:.4f}" if isinstance(gg_epsilon, (int, float)) else str(gg_epsilon)
        gg_df_str = f"{gg_df:.2f}" if isinstance(gg_df, (int, float)) else str(gg_df)
        gg_f_str = f"{gg_f_stat:.4f}" if isinstance(gg_f_stat, (int, float)) else str(gg_f_stat)
        gg_p_str = f"{gg_p_value:.6f}" if isinstance(gg_p_value, (int, float)) else str(gg_p_value)
        gg_rec_str = "⭐ YES (Conservative)" if gg_recommended else "No"
        gg_interp = "Use this p-value!" if gg_recommended else "Alternative option"
        
        gg_values = ["Greenhouse-Geisser", gg_epsilon_str, gg_df_str, gg_f_str, gg_p_str, gg_rec_str, gg_interp]
        
        for col, val in enumerate(gg_values):
            cell_fmt = fmt["recommended"] if gg_recommended and col >= 4 else fmt["cell"]
            ws.write(row, col, val, cell_fmt)
        row += 1
        
        # Write Huynh-Feldt row
        hf_epsilon_str = f"{hf_epsilon:.4f}" if isinstance(hf_epsilon, (int, float)) else str(hf_epsilon)
        hf_df_str = f"{hf_df:.2f}" if isinstance(hf_df, (int, float)) else str(hf_df)
        hf_f_str = f"{hf_f_stat:.4f}" if isinstance(hf_f_stat, (int, float)) else str(hf_f_stat)
        hf_p_str = f"{hf_p_value:.6f}" if isinstance(hf_p_value, (int, float)) else str(hf_p_value)
        hf_rec_str = "⭐ YES (Less Conservative)" if hf_recommended else "No"
        hf_interp = "Use this p-value!" if hf_recommended else "Alternative option"
        
        hf_values = ["Huynh-Feldt", hf_epsilon_str, hf_df_str, hf_f_str, hf_p_str, hf_rec_str, hf_interp]
        
        for col, val in enumerate(hf_values):
            cell_fmt = fmt["recommended"] if hf_recommended and col >= 4 else fmt["cell"]
            ws.write(row, col, val, cell_fmt)
        row += 2
        
        # Final recommendation box
        if gg_recommended:
            final_rec = (
                "📋 FINAL RECOMMENDATION: Use Greenhouse-Geisser Correction\n"
                f"→ Your corrected p-value is: {gg_p_str}\n"
                f"→ Epsilon = {gg_epsilon_str} (≤ 0.75, so conservative correction is appropriate)\n"
                "→ This correction protects against Type I errors when sphericity is violated"
            )
            rec_fmt = workbook.add_format({
                'text_wrap': True,
                'valign': 'top',
                'border': 1,
                'bg_color': '#E6F3FF',  # Light blue
                'font_color': '#0066CC',
                'bold': True
            })
        elif hf_recommended:
            final_rec = (
                "📋 FINAL RECOMMENDATION: Use Huynh-Feldt Correction\n"
                f"→ Your corrected p-value is: {hf_p_str}\n"
                f"→ Epsilon = {hf_epsilon_str} (> 0.75, so less conservative correction is appropriate)\n"
                "→ This correction is less conservative while still controlling Type I errors"
            )
            rec_fmt = workbook.add_format({
                'text_wrap': True,
                'valign': 'top',
                'border': 1,
                'bg_color': '#E6F3FF',  # Light blue
                'font_color': '#0066CC',
                'bold': True
            })
        else:
            final_rec = (
                "📋 RECOMMENDATION: Check epsilon values\n"
                "→ Cannot determine optimal correction\n"
                "→ Consider using the more conservative Greenhouse-Geisser correction\n"
                "→ Consult with a statistician if unsure"
            )
            rec_fmt = workbook.add_format({
                'text_wrap': True,
                'valign': 'top',
                'border': 1,
                'bg_color': '#FFF0E6',  # Light orange
                'font_color': '#CC6600'
            })
        
        ws.write(row, 0, final_rec, rec_fmt)
        ws.set_row(row, 40)
        row += 2
        
        return row

    @staticmethod
    def _write_between_factor_assumptions(workbook, results, fmt, ws, start_row):
        """
        Writes between-factor assumption test results for Mixed ANOVA with robust error handling.
        
        This function handles the special assumptions required for Mixed ANOVA designs,
        where between-subjects factors interact with within-subjects factors. It includes:
        - Homogeneity of variance tests for between-subjects factors
        - Independence of observations validation
        - Normality assessments for between-subjects groups
        - Box's M test for homogeneity of covariance matrices (when applicable)
        
        Args:
            workbook: xlsxwriter.Workbook object for creating Excel formats
            results (dict): Statistical analysis results containing between-factor assumptions
            fmt (dict): Dictionary of Excel formatting objects
            ws: xlsxwriter.Worksheet object for writing data
            start_row (int): Row number to start writing from
            
        Returns:
            int: Next available row number after writing between-factor assumptions section
            
        Expected results structure:
            results["between_assumptions"] = {
                "variance_tests": {
                    "levenes_test": {
                        "statistic": float,
                        "p_value": float,
                        "assumption_met": bool,
                        "interpretation": str
                    },
                    "bartletts_test": {
                        "statistic": float,
                        "p_value": float,
                        "assumption_met": bool,
                        "interpretation": str
                    }
                },
                "recommendations": [
                    "Recommendation 1",
                    "Recommendation 2"
                ]
            }
            
        Example:
            # Write between-factor assumptions starting at row 20
            next_row = _write_between_factor_assumptions(workbook, results, fmt, ws, 20)
        """
        try:
            # Input validation
            is_valid, error_msg, error_rows = ResultsExporter._validate_excel_inputs(
                workbook, results, fmt, ws, "_write_between_factor_assumptions"
            )
            if not is_valid:
                ws.write(start_row, 0, error_msg, fmt["significant"])
                return start_row + error_rows
            
            row = start_row
            
            # Check if between_assumptions data exists
            if "between_assumptions" not in results:
                ws.write(row, 0, "⚠️ WARNING: No between-factor assumptions data available", fmt["significant"])
                return row + 2
            
            between_assumptions = results["between_assumptions"]
            
            # Section header
            ws.write(row, 0, "🔵 BETWEEN-FACTOR ASSUMPTIONS (MIXED ANOVA)", fmt["section_header"])
            row += 1
            
            # Between-factor explanation
            between_explanation = (
                "BETWEEN-SUBJECTS FACTOR ASSUMPTIONS:\n"
                "For Mixed ANOVA, the between-subjects factor must meet homogeneity of variance assumptions.\n\n"
                "TESTS PERFORMED:\n"
                "• Levene's Test: Standard test for equal variances\n"
                "• Levene's Test: Standard test for equal variances\n"
                "• Brown-Forsythe Test: Robust alternative using medians\n"
                "• Bartlett's Test: Sensitive to normality violations\n"
                "• Welch's ANOVA: Robust alternative when variances are unequal"
            )
            
            explanation_fmt = workbook.add_format({
                'text_wrap': True, 
                'valign': 'top',
                'border': 1,
                'bg_color': '#E6F3FF'
            })
            ws.write(row, 0, between_explanation, explanation_fmt)
            ws.set_row(row, 55)
            row += 2
            
            # Variance tests table
            if "variance_tests" in between_assumptions:
                variance_tests = between_assumptions["variance_tests"]
                
                # Table headers
                var_headers = ["Test", "Statistic", "p-Value", "Assumption Met?", "Interpretation"]
                for i, header in enumerate(var_headers):
                    ws.write(row, i, header, fmt["header"])
                row += 1
                
                # Write each variance test
                for test_name, test_data in variance_tests.items():
                    if isinstance(test_data, dict):
                        statistic = test_data.get("statistic", "N/A")
                        p_value = test_data.get("p_value", "N/A")
                        assumption_met = test_data.get("assumption_met", None)
                        interpretation = test_data.get("interpretation", "No interpretation available")
                        
                        stat_str = f"{statistic:.4f}" if isinstance(statistic, (float, int)) else str(statistic)
                        p_str = f"{p_value:.4f}" if isinstance(p_value, (float, int)) else str(p_value)
                        assumption_str = "Yes" if assumption_met else "No" if assumption_met is not None else "Unknown"
                        
                        values = [test_name.replace("_", " ").title(), stat_str, p_str, assumption_str, interpretation]
                        
                        for col, val in enumerate(values):
                            if assumption_met is False and col >= 2:
                                ws.write(row, col, val, fmt["significant"])
                            else:
                                ws.write(row, col, val, fmt["cell"])
                        row += 1
                
                row += 1
            
            # Recommendations
            if "recommendations" in between_assumptions:
                ws.write(row, 0, "📋 RECOMMENDATIONS:", fmt["section_header"])
                row += 1
                
                recommendations = between_assumptions["recommendations"]
                if isinstance(recommendations, list):
                    for recommendation in recommendations:
                        ws.write(row, 0, f"• {recommendation}", fmt["cell"])
                        row += 1
                row += 1
            
            return row
            
        except Exception as e:
            # Error handling for between-factor assumptions
            error_msg = f"⚠️ ERROR in between-factor assumptions: {str(e)}"
            try:
                ws.write(start_row, 0, error_msg, fmt["significant"])
                ws.write(start_row + 1, 0, "Between-factor assumption data may be incomplete.", fmt["cell"])
            except:
                pass
            return start_row + 3
    
    @staticmethod
    def _write_within_factor_sphericity(workbook, results, fmt, ws, start_row):
        """
        Writes within-factor sphericity results for Mixed ANOVA with comprehensive explanations.
        
        In Mixed ANOVA designs, within-subjects factors must meet the sphericity assumption
        independently of between-subjects factors. This function provides detailed analysis
        of sphericity for the within-subjects portion of the Mixed ANOVA.
        
        Args:
            workbook: xlsxwriter.Workbook object for creating Excel formats
            results (dict): Statistical analysis results containing within-factor sphericity data
            fmt (dict): Dictionary of Excel formatting objects
            ws: xlsxwriter.Worksheet object for writing data
            start_row (int): Row number to start writing from
            
        Returns:
            int: Next available row number after writing within-factor sphericity section
            
        Expected results structure:
            results["within_sphericity"] = {
                "mauchly_test": {
                    "statistic": float,
                    "p_value": float,
                    "assumption_met": bool
                },
                "epsilon_values": {
                    "greenhouse_geisser": float,
                    "huynh_feldt": float
                }
            }
            
        Example:
            # Write within-factor sphericity starting at row 25
            next_row = _write_within_factor_sphericity(workbook, results, fmt, ws, 25)
        """
        row = start_row
        
        # Section header
        ws.write(row, 0, "🟡 WITHIN-FACTOR SPHERICITY (MIXED ANOVA)", fmt["section_header"])
        row += 1
        
        within_sphericity = results["within_sphericity_test"]
        
        # Within-factor explanation
        within_explanation = (
            "WITHIN-SUBJECTS FACTOR SPHERICITY:\n"
            "In Mixed ANOVA, the within-subjects factor must meet sphericity assumptions.\n"
            "This test examines whether the variance-covariance matrix has compound symmetry."
        )
        
        explanation_fmt = workbook.add_format({
            'text_wrap': True, 
            'valign': 'top',
            'border': 1,
            'bg_color': '#FFFACD'
        })
        ws.write(row, 0, within_explanation, explanation_fmt)
        ws.set_row(row, 35)
        row += 2
        
        # Within-factor sphericity table
        within_headers = ["Factor", "W Statistic", "p-Value", "Sphericity Met?", "Levels Tested", "Interpretation"]
        for i, header in enumerate(within_headers):
            ws.write(row, i, header, fmt["header"])
        row += 1
        
        # Extract data
        factor = within_sphericity.get("factor", "Within-Factor")
        W = within_sphericity.get("W", "N/A")
        p_value = within_sphericity.get("p_value", "N/A")
        sphericity_met = within_sphericity.get("sphericity_assumed", None)
        levels = within_sphericity.get("levels_tested", "N/A")
        interpretation = within_sphericity.get("interpretation", "No interpretation available")
        
        # Format values
        w_str = f"{W:.4f}" if isinstance(W, (float, int)) else str(W)
        p_str = f"{p_value:.4f}" if isinstance(p_value, (float, int)) else str(p_value)
        sphericity_str = "Yes" if sphericity_met else "No" if sphericity_met is not None else "Unknown"
        levels_str = str(levels)
        
        values = [factor, w_str, p_str, sphericity_str, levels_str, interpretation]
        
        for col, val in enumerate(values):
            if sphericity_met is False and col >= 2:
                ws.write(row, col, val, fmt["significant"])
            else:
                ws.write(row, col, val, fmt["cell"])
        row += 2
        
        # Within-factor corrections if available
        if "within_sphericity_corrections" in results:
            ws.write(row, 0, "🔧 WITHIN-FACTOR CORRECTIONS:", fmt["section_header"])
            row += 1
            
            corrections = results["within_sphericity_corrections"]
            if "main_effect" in corrections:
                main_effect = corrections["main_effect"]
                correction_used = main_effect.get("correction_used", "None")
                final_p = main_effect.get("final_p_value", "N/A")
                
                ws.write(row, 0, f"Main Effect Correction: {correction_used}", fmt["cell"])
                row += 1
                if isinstance(final_p, (float, int)):
                    ws.write(row, 0, f"Corrected p-value: {final_p:.4f}", fmt["highlight"])
                    row += 1
            row += 1
        
        return row
    
    @staticmethod
    def _write_interaction_assumptions(workbook, results, fmt, ws, start_row):
        """
        Writes interaction assumption test results for Mixed ANOVA with detailed explanations.
        
        Mixed ANOVA designs test interactions between within-subjects and between-subjects factors.
        These interactions require specific assumptions about homogeneity of covariance matrices
        and sphericity of interaction effects. This function provides comprehensive analysis
        of these complex interaction assumptions.
        
        Args:
            workbook: xlsxwriter.Workbook object for creating Excel formats
            results (dict): Statistical analysis results containing interaction assumptions data
            fmt (dict): Dictionary of Excel formatting objects
            ws: xlsxwriter.Worksheet object for writing data
            start_row (int): Row number to start writing from
            
        Returns:
            int: Next available row number after writing interaction assumptions section
            
        Expected results structure:
            results["interaction_assumptions"] = {
                "homogeneity_tests": {
                    "box_m_test": {
                        "statistic": float,
                        "p_value": float,
                        "assumption_met": bool,
                        "interpretation": str
                    }
                },
                "sphericity_interaction": {
                    "mauchly_test": {
                        "statistic": float,
                        "p_value": float,
                        "assumption_met": bool
                    }
                },
                "recommendations": [
                    "Interaction-specific recommendations"
                ]
            }
            
        Example:
            # Write interaction assumptions starting at row 30
            next_row = _write_interaction_assumptions(workbook, results, fmt, ws, 30)
        """
        row = start_row
        
        interaction_assumptions = results["interaction_assumptions"]
        
        # Section header
        ws.write(row, 0, "🔴 INTERACTION ASSUMPTIONS (MIXED ANOVA)", fmt["section_header"])
        row += 1
        
        # Interaction explanation
        interaction_explanation = (
            "INTERACTION EFFECT ASSUMPTIONS:\n"
            "Mixed ANOVA interactions require several specialized assumptions:\n"
            "• Sphericity for the interaction effect\n"
            "• Homogeneity across interaction cells\n"
            "• Similar covariance patterns across groups\n"
            "• Homogeneity of covariance matrices (Box's M test)"
        )
        
        explanation_fmt = workbook.add_format({
            'text_wrap': True, 
            'valign': 'top',
            'border': 1,
            'bg_color': '#FFE4E1'
        })
        ws.write(row, 0, interaction_explanation, explanation_fmt)
        ws.set_row(row, 45)
        row += 2
        
        # Interaction sphericity
        if "sphericity_tests" in interaction_assumptions:
            sphericity_tests = interaction_assumptions["sphericity_tests"]
            
            ws.write(row, 0, "Interaction Sphericity:", fmt["subsection_header"])
            row += 1
            
            sphericity_met = sphericity_tests.get("sphericity_assumed", None)
            interpretation = sphericity_tests.get("interpretation", "No interpretation available")
            
            ws.write(row, 0, f"Sphericity Met: {'Yes' if sphericity_met else 'No' if sphericity_met is not None else 'Unknown'}", 
                    fmt["significant"] if sphericity_met is False else fmt["cell"])
            row += 1
            ws.write(row, 0, f"Interpretation: {interpretation}", fmt["cell"])
            row += 2
        
        # Cell homogeneity
        if "cell_homogeneity" in interaction_assumptions:
            cell_homogeneity = interaction_assumptions["cell_homogeneity"]
            
            ws.write(row, 0, "Interaction Cell Homogeneity:", fmt["subsection_header"])
            row += 1
            
            if "levene_test" in cell_homogeneity:
                levene = cell_homogeneity["levene_test"]
                assumption_met = levene.get("assumption_met", None)
                p_value = levene.get("p_value", "N/A")
                
                p_str = f"{p_value:.4f}" if isinstance(p_value, (float, int)) else str(p_value)
                ws.write(row, 0, f"Levene's Test p-value: {p_str}", 
                        fmt["significant"] if assumption_met is False else fmt["cell"])
                row += 1
                
                ws.write(row, 0, f"Cell Homogeneity: {'Met' if assumption_met else 'Violated' if assumption_met is not None else 'Unknown'}", 
                        fmt["significant"] if assumption_met is False else fmt["cell"])
                row += 2
        
        # Overall recommendations
        if "overall_recommendations" in interaction_assumptions:
            ws.write(row, 0, "📋 INTERACTION RECOMMENDATIONS:", fmt["section_header"])
            row += 1
            
            recommendations = interaction_assumptions["overall_recommendations"]
            if isinstance(recommendations, list):
                for recommendation in recommendations:
                    ws.write(row, 0, f"• {recommendation}", fmt["cell"])
                    row += 1
            row += 1
        
        return row
        
        return cleaned_count
