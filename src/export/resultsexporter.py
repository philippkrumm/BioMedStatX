import xlsxwriter
import copy
import os
import pandas as pd

# Lazy import function to avoid circular imports
def get_assumption_visualizer():
    """Get AssumptionVisualizer class lazily"""
    from analysis.stats_functions import AssumptionVisualizer
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
    def export_results_to_excel(results, output_file, analysis_log=None, pre_generated_tree=None):
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

        # ── Sheet order (academic report layout) ────────────────────────────
        # 1. Cover
        ResultsExporter._write_summary_sheet(workbook, results, fmt)
        # 3. Statistical Results
        ResultsExporter._write_results_sheet(workbook, results, fmt)
        # 4. Descriptives
        ResultsExporter._write_descriptive_sheet(workbook, results, fmt)
        # 5. Pairwise — only if 3+ groups AND post-hoc was computed
        _groups = results.get("groups", [])
        _pairwise = results.get("pairwise_comparisons", [])
        if len(_groups) >= 3 and _pairwise:
            ResultsExporter._write_pairwise_sheet(workbook, results, fmt)
        # 6. Assumptions
        ResultsExporter._write_assumptions_sheet(workbook, results, fmt)
        # 7–12. Model-specific sheets
        model_type = results.get("model_type", "")
        if model_type == "ANCOVA":
            ResultsExporter._write_ancova_sheet(workbook, results, fmt)
        elif model_type == "LMM":
            ResultsExporter._write_lmm_sheet(workbook, results, fmt)
        elif model_type == "Correlation":
            ResultsExporter._write_correlation_sheet(workbook, results, fmt)
        elif model_type == "CorrelationMatrix":
            ResultsExporter._write_correlation_matrix_sheet(workbook, results, fmt)
        elif model_type == "LinearRegression":
            ResultsExporter._write_linear_regression_sheet(workbook, results, fmt)
        elif model_type == "LogisticRegression":
            ResultsExporter._write_logistic_regression_sheet(workbook, results, fmt)
        elif model_type == "BetaRegression":
            ResultsExporter._write_beta_regression_sheet(workbook, results, fmt)
        # 13. Decision Tree
        ResultsExporter._write_decision_tree_sheet(workbook, results, fmt, pre_generated_tree=pre_generated_tree)
        # 14. Methodology Log (replaces old Methodology + Analysis Log sheets)
        ResultsExporter._write_methodology_log_sheet(
            workbook,
            results,
            fmt,
            trace=results.get("methodology_trace"),
            analysis_log=analysis_log or results.get("analysis_log"),
        )
        # 15. Raw Data
        ResultsExporter._write_rawdata_sheet(workbook, results, fmt)
            
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
    def _write_assumption_summary_sheet(workbook, all_results, fmt):
        """Single-page summary of all assumption test results across datasets.

        Columns: Dataset | Test | Normality SW-W | SW-p | Normal? |
                 BF-F | BF-p | Equal var? | Sphericity W | Sph-p | Sph OK? |
                 Recommendation
        Red highlight = assumption violated. Green/neutral = met.
        """
        ws = workbook.add_worksheet("Assumption Summary")
        ws.set_column(0, 0, 28)   # Dataset
        ws.set_column(1, 1, 22)   # Test
        ws.set_column(2, 2, 14)   # SW-W
        ws.set_column(3, 3, 10)   # SW-p
        ws.set_column(4, 4, 12)   # Normal?
        ws.set_column(5, 5, 12)   # BF-F
        ws.set_column(6, 6, 10)   # BF-p
        ws.set_column(7, 7, 14)   # Equal var?
        ws.set_column(8, 8, 14)   # Mauchly-W
        ws.set_column(9, 9, 12)   # Sph-p
        ws.set_column(10, 10, 12) # Sphericity?
        ws.set_column(11, 11, 30) # Recommendation

        row = 0
        ws.merge_range(row, 0, row, 11, "ASSUMPTION TEST SUMMARY (all datasets)", fmt["title"])
        row += 1
        ws.merge_range(row, 0, row, 11,
            "SW = Shapiro-Wilk on model residuals  |  BF = Brown-Forsythe (variance homogeneity)  |  "
            "Sph = Mauchly sphericity (RM/Mixed ANOVA only)  |  Red = assumption violated",
            fmt.get("explanation", fmt["cell"]))
        row += 2

        headers = [
            "Dataset", "Statistical Test",
            "SW-W", "SW p-value", "Residuals Normal?",
            "BF-F", "BF p-value", "Equal Variance?",
            "Mauchly-W", "Sph. p-value", "Sphericity OK?",
            "Recommendation",
        ]
        for col, h in enumerate(headers):
            ws.write(row, col, h, fmt["header"])
        row += 1

        def _fmt_num(val):
            if isinstance(val, (float, int)):
                return "<0.001" if val < 0.001 else f"{val:.4f}"
            return "N/A"

        def _cell(ok):
            """Return (text, format_key) for a pass/fail cell."""
            if ok is True:
                return "Yes", "cell"
            if ok is False:
                return "No", "significant"
            return "N/A", "cell"

        for dataset_name, results in all_results.items():
            test_name = results.get("test", "N/A")
            test_info = results.get("test_info", {})
            pre = test_info.get("pre_transformation", {}) if test_info else {}
            post = test_info.get("post_transformation", {}) if test_info else {}
            transformation = results.get("transformation", "None")
            use_post = bool(transformation and transformation not in ("None", "No further") and post)

            # --- Normality (Shapiro-Wilk on residuals) ---
            norm_src = post.get("residuals_normality", {}) if use_post else pre.get("residuals_normality", {})
            if not norm_src:
                # fallback: new normality_tests structure
                nt = results.get("normality_tests", {})
                norm_src = nt.get("model_residuals_transformed" if use_post else "model_residuals", {})
            sw_w = norm_src.get("statistic")
            sw_p = norm_src.get("p_value")
            is_normal = norm_src.get("is_normal", (isinstance(sw_p, (float, int)) and sw_p > 0.05) if sw_p is not None else None)

            # --- Variance homogeneity (Brown-Forsythe) ---
            var_src = post.get("variance", {}) if use_post else pre.get("variance", {})
            if not var_src:
                vt = results.get("variance_test", {})
                var_src = vt.get("transformed", vt) if (use_post and "transformed" in vt) else vt
            bf_f = var_src.get("statistic")
            bf_p = var_src.get("p_value")
            equal_var = var_src.get("equal_variance", (isinstance(bf_p, (float, int)) and bf_p > 0.05) if bf_p is not None else None)

            # --- Sphericity (Mauchly, only RM/Mixed) ---
            sph = results.get("sphericity_test", {})
            sph_w = sph.get("W") if sph else None
            sph_p = sph.get("p_value") if sph else None
            sph_ok = sph.get("has_sphericity") if sph else None

            # --- Recommendation ---
            rec = results.get("recommendation", results.get("test_recommendation", results.get("test_type", "N/A")))

            norm_text, norm_fmt = _cell(is_normal)
            var_text, var_fmt = _cell(equal_var)
            sph_text, sph_fmt = _cell(sph_ok)

            ws.write(row, 0, str(dataset_name), fmt["cell"])
            ws.write(row, 1, str(test_name), fmt["cell"])
            ws.write(row, 2, _fmt_num(sw_w), fmt["cell"])
            ws.write(row, 3, _fmt_num(sw_p), fmt["sig_highlight"] if sw_p is not None and isinstance(sw_p, (float, int)) and sw_p < 0.05 else fmt["cell"])
            ws.write(row, 4, norm_text, fmt[norm_fmt])
            ws.write(row, 5, _fmt_num(bf_f), fmt["cell"])
            ws.write(row, 6, _fmt_num(bf_p), fmt["sig_highlight"] if bf_p is not None and isinstance(bf_p, (float, int)) and bf_p < 0.05 else fmt["cell"])
            ws.write(row, 7, var_text, fmt[var_fmt])
            ws.write(row, 8, _fmt_num(sph_w), fmt["cell"])
            ws.write(row, 9, _fmt_num(sph_p), fmt["sig_highlight"] if sph_p is not None and isinstance(sph_p, (float, int)) and sph_p < 0.05 else fmt["cell"])
            ws.write(row, 10, sph_text, fmt[sph_fmt])
            ws.write(row, 11, str(rec), fmt["cell"])
            row += 1

    @staticmethod
    def _write_assumptions_sheet(workbook, results, fmt, sheet_name="Assumptions"):
        """Write a single-analysis assumptions worksheet.

        This method is intentionally compact and defensive because result payloads
        vary heavily across analysis types. It surfaces the currently available
        normality, variance, sphericity, transformation, and assumption-related
        notes without failing when optional structures are missing.
        """
        ws = workbook.add_worksheet(sheet_name)
        ws.set_column(0, 0, 34)
        ws.set_column(1, 3, 18)
        row = 0

        ws.merge_range(row, 0, row, 3, "ASSUMPTION CHECKS", fmt["title"])
        row += 1
        ws.merge_range(
            row, 0, row, 3,
            "This sheet summarizes normality, variance, sphericity, and transformation-related diagnostics.",
            fmt.get("explanation", fmt["cell"])
        )
        row += 2

        transformation = results.get("transformation", "None")
        transformation_applied = bool(
            transformation and str(transformation).lower() not in ("none", "no further")
        )
        ws.write(row, 0, "Transformation", fmt["header"])
        ws.write(row, 1, str(transformation), fmt["cell"])
        row += 2

        headers = ["Check", "Statistic", "p-value", "Status"]
        for col, header in enumerate(headers):
            ws.write(row, col, header, fmt["header"])
        row += 1

        def _fmt_num(val):
            if isinstance(val, (float, int)):
                return "<0.001" if val < 0.001 else f"{val:.4f}"
            return "N/A"

        def _status_text(value):
            if value is True:
                return "Passed"
            if value is False:
                return "Flagged"
            return "N/A"

        def _status_fmt(value):
            if value is True:
                return fmt["cell"]
            if value is False:
                return fmt.get("sig_highlight", fmt["cell"])
            return fmt["cell"]

        test_info = results.get("test_info", {}) or {}
        normality_tests = results.get("normality_tests", {}) or {}
        if not normality_tests and test_info:
            phase = "post_transformation" if transformation_applied else "pre_transformation"
            residuals_normality = test_info.get(phase, {}).get("residuals_normality", {})
            if residuals_normality:
                normality_tests = {
                    "model_residuals_transformed" if transformation_applied else "model_residuals": residuals_normality
                }
        for label, payload in normality_tests.items():
            if not isinstance(payload, dict):
                continue
            if label == "model_residuals_transformed" and not transformation_applied:
                continue
            p_value = payload.get("p_value")
            status = payload.get("is_normal")
            ws.write(row, 0, f"Normality: {str(label)}", fmt["cell"])
            ws.write(row, 1, _fmt_num(payload.get("statistic")), fmt["cell"])
            ws.write(row, 2, _fmt_num(p_value), fmt.get("sig_highlight", fmt["cell"]) if isinstance(p_value, (float, int)) and p_value < 0.05 else fmt["cell"])
            ws.write(row, 3, _status_text(status), _status_fmt(status))
            row += 1

        variance_test = results.get("variance_test", {}) or {}
        if not variance_test and test_info:
            phase = "post_transformation" if transformation_applied else "pre_transformation"
            variance_test = test_info.get(phase, {}).get("variance", {}) or {}
        if isinstance(variance_test, dict) and variance_test:
            p_value = variance_test.get("p_value")
            status = variance_test.get("equal_variance")
            test_name = variance_test.get("test_name", "Brown-Forsythe")
            ws.write(row, 0, f"Variance homogeneity ({test_name})", fmt["cell"])
            ws.write(row, 1, _fmt_num(variance_test.get("statistic")), fmt["cell"])
            ws.write(row, 2, _fmt_num(p_value), fmt.get("sig_highlight", fmt["cell"]) if isinstance(p_value, (float, int)) and p_value < 0.05 else fmt["cell"])
            ws.write(row, 3, _status_text(status), _status_fmt(status))
            row += 1

            transformed = variance_test.get("transformed")
            if transformation_applied and isinstance(transformed, dict):
                p_value = transformed.get("p_value")
                status = transformed.get("equal_variance")
                transformed_test_name = transformed.get("test_name", test_name)
                ws.write(row, 0, f"Variance homogeneity ({transformed_test_name}, transformed)", fmt["cell"])
                ws.write(row, 1, _fmt_num(transformed.get("statistic")), fmt["cell"])
                ws.write(row, 2, _fmt_num(p_value), fmt.get("sig_highlight", fmt["cell"]) if isinstance(p_value, (float, int)) and p_value < 0.05 else fmt["cell"])
                ws.write(row, 3, _status_text(status), _status_fmt(status))
                row += 1

        sphericity_test = results.get("sphericity_test", {}) or {}
        if isinstance(sphericity_test, dict) and sphericity_test:
            p_value = sphericity_test.get("p_value")
            status = sphericity_test.get("sphericity_met")
            if status is None and isinstance(p_value, (float, int)):
                status = p_value >= 0.05
            ws.write(row, 0, "Sphericity", fmt["cell"])
            ws.write(row, 1, _fmt_num(sphericity_test.get("W", sphericity_test.get("statistic"))), fmt["cell"])
            ws.write(row, 2, _fmt_num(p_value), fmt.get("sig_highlight", fmt["cell"]) if isinstance(p_value, (float, int)) and p_value < 0.05 else fmt["cell"])
            ws.write(row, 3, _status_text(status), _status_fmt(status))
            row += 1

        if row == 4:
            ws.merge_range(row, 0, row, 3, "No structured assumption diagnostics available.", fmt.get("explanation", fmt["cell"]))
            row += 2

        note_candidates = [
            results.get("analysis_note"),
            results.get("posthoc_skip_reason"),
            results.get("design_note"),
        ]
        notes = [str(note) for note in note_candidates if note]
        if notes:
            ws.merge_range(row, 0, row, 3, "NOTES", fmt["section_header"])
            row += 1
            for note in notes:
                ws.merge_range(row, 0, row, 3, note, fmt.get("explanation", fmt["cell"]))
                row += 1

        _clinical_models = {'correlation', 'linear_regression', 'logistic_regression', 'beta_regression'}
        _is_clinical = results.get('model_type') in _clinical_models or results.get('test', '').lower().startswith(('correlation', 'linear regression', 'logistic', 'beta'))
        plot_paths = results.get("assumption_plot_paths")
        if not _is_clinical and (not isinstance(plot_paths, dict) or not any(plot_paths.values())):
            try:
                plot_paths = get_assumption_visualizer().generate_assumption_plots(results)
                if isinstance(plot_paths, dict):
                    results["assumption_plot_paths"] = plot_paths
            except Exception as exc:
                print(f"DEBUG: Failed to generate assumption plots for Excel export: {exc}")
                plot_paths = {}

        if isinstance(plot_paths, dict):
            if transformation_applied:
                image_specs = [
                    ("Q-Q plot", plot_paths.get("normality_after")),
                    ("Variance boxplots", plot_paths.get("homoscedasticity_after")),
                ]
            else:
                image_specs = [
                    ("Q-Q plot", plot_paths.get("normality_before")),
                    ("Variance boxplots", plot_paths.get("homoscedasticity_before")),
                ]
            image_specs = [(label, path) for label, path in image_specs if path and os.path.exists(path)]
            if image_specs:
                row += 1
                ws.merge_range(row, 0, row, 3, "VISUAL DIAGNOSTICS", fmt["section_header"])
                row += 1
                for label, image_path in image_specs:
                    ws.write(row, 0, label, fmt["key"])
                    ws.insert_image(row, 1, image_path, {
                        "x_scale": 0.45,
                        "y_scale": 0.45,
                        "object_position": 1,
                    })
                    row += 18

    @staticmethod
    def export_multi_dataset_results(all_results, excel_path):
        print(f"DEBUG: Current working directory before export: {os.getcwd()}")
        print(f"DEBUG MULTI: export_multi_dataset_results called with excel_path='{excel_path}'")
        print("DEBUG MULTI: Received all_results with contents:")   
        for ds_name, results in all_results.items():
            print(f"  Dataset: {ds_name} -> Keys in results: {list(results.keys())}")
            print(f"    p_value: {results.get('p_value')} | pairwise_comparisons: {len(results.get('pairwise_comparisons', []))}")
        
        """Exports the results of all dataset analyses into a shared Excel file."""
        # import os  # Already imported at top
        import time
        import xlsxwriter
        from visualization.decisiontreevisualizer import DecisionTreeVisualizer
        
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
        overview_sheet.set_column('F:F', 22)
        overview_sheet.set_column('G:G', 22)
        overview_sheet.write(0, 0, "Dataset", fmt["header"])
        overview_sheet.write(0, 1, "Test", fmt["header"])
        overview_sheet.write(0, 2, "p-value", fmt["header"])
        overview_sheet.write(0, 3, "Significant", fmt["header"])
        overview_sheet.write(0, 4, "Transformation", fmt["header"])
        overview_sheet.write(0, 5, "p-value (FDR-corrected)", fmt["header"])
        overview_sheet.write(0, 6, "Significant (FDR)", fmt["header"])

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
            sig_fmt = fmt["sig_highlight"] if is_significant else fmt["cell"]
            overview_sheet.write(row, 3, "Yes" if is_significant else "No", sig_fmt)

            transformation = results.get("transformation", "None")
            overview_sheet.write(row, 4, str(transformation), fmt["cell"])

            # FDR-corrected p-value
            p_fdr = results.get("p_value_fdr", None)
            if p_fdr is not None:
                fdr_str = "<0.001" if p_fdr < 0.001 else f"{p_fdr:.4f}"
                overview_sheet.write(row, 5, fdr_str, fmt["cell"])
                is_sig_fdr = p_fdr < 0.05
                fdr_sig_fmt = fmt["sig_highlight"] if is_sig_fdr else fmt["cell"]
                overview_sheet.write(row, 6, "Yes" if is_sig_fdr else "No", fdr_sig_fmt)
            else:
                overview_sheet.write(row, 5, "N/A", fmt["cell"])
                overview_sheet.write(row, 6, "N/A", fmt["cell"])

            row += 1
            
        # G3-FDR: document m (number of tests in the FDR family)
        n_valid_for_fdr = sum(
            1 for r in all_results.values()
            if isinstance(r.get("p_value"), (float, int))
        )
        overview_sheet.merge_range(
            row, 0, row, 6,
            f"Note: FDR correction (Benjamini-Hochberg) applied to m = {n_valid_for_fdr} tests.",
            fmt.get("explanation", fmt["cell"])
        )
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
        
        # Assumption Summary — second sheet, right after Overview
        try:
            ResultsExporter._write_assumption_summary_sheet(workbook, all_results, fmt)
        except Exception as e:
            print(f"DEBUG MULTI: Error creating assumption summary sheet: {str(e)}")

        # For each dataset: create all detail sheets as in single analysis
        for dataset_name, results in all_results.items():
            # Use the pre-generated decision tree path
            pre_generated_tree = dataset_tree_paths.get(dataset_name)
            
            def _sheet(suffix):
                """Return a sheet name <= 31 chars by truncating dataset_name if needed."""
                max_name = 31 - len(suffix)
                return dataset_name[:max_name] + suffix

            try:
                # Create all the detailed sheets for this dataset
                ResultsExporter._write_summary_sheet(workbook, results, fmt, _sheet("_Summary"))
                ResultsExporter._write_results_sheet(workbook, results, fmt, _sheet("_Results"))
                ResultsExporter._write_descriptive_sheet(workbook, results, fmt, _sheet("_Descriptive"))
                _groups = results.get("groups", [])
                _pairwise = results.get("pairwise_comparisons", [])
                if len(_groups) >= 3 and _pairwise:
                    ResultsExporter._write_pairwise_sheet(workbook, results, fmt, _sheet("_Pairwise"))
                ResultsExporter._write_assumptions_sheet(workbook, results, fmt, _sheet("_Assumptions"))
                ResultsExporter._write_decision_tree_sheet(workbook, results, fmt, _sheet("_DecisionTree"), pre_generated_tree)
                ResultsExporter._write_methodology_log_sheet(
                    workbook,
                    results,
                    fmt,
                    trace=results.get("methodology_trace"),
                    analysis_log=results.get("analysis_log"),
                    sheet_name=_sheet("_MethodLog"),
                )
                ResultsExporter._write_rawdata_sheet(workbook, results, fmt, _sheet("_RawData"))
                    
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
            "sig_highlight": workbook.add_format({'align': 'center', 'color': 'red', 'bold': True, 'text_wrap': True}),
            "explanation": workbook.add_format({'text_wrap': True, 'valign': 'top', 'font_color': '#1F4E78'}),
            "section_header": workbook.add_format({'bold': True, 'bg_color': '#B4C6E7', 'border': 1}),
            "section_header_center": workbook.add_format({'bold': True, 'bg_color': '#B4C6E7', 'border': 1, 'align': 'center'}),
            "effect_strong": workbook.add_format({'align': 'center', 'color': '#006400', 'bold': True, 'text_wrap': True}),
            "effect_med_text": workbook.add_format({'align': 'center', 'color': '#FFA500', 'bold': True, 'text_wrap': True}),
            "effect_weak": workbook.add_format({'align': 'center', 'color': '#A52A2A', 'bold': True, 'text_wrap': True}),
            "key": workbook.add_format({'bold': True, 'align': 'right'}),
            "bold": workbook.add_format({'bold': True}),
            "recommended": workbook.add_format({'align': 'center', 'bg_color': '#E6F3FF', 'font_color': '#0066CC', 'bold': True, 'text_wrap': True}),
            # ── New academic navy/grey palette ──────────────────────────────────────
            "navy_header":     workbook.add_format({'bold': True, 'font_size': 11, 'font_name': 'Calibri',
                                                    'bg_color': '#1F3864', 'font_color': '#FFFFFF', 'text_wrap': True}),
            "section_blue":    workbook.add_format({'bold': True, 'font_size': 10, 'font_name': 'Calibri',
                                                    'bg_color': '#2F5496', 'font_color': '#FFFFFF', 'text_wrap': True}),
            "subheader_light": workbook.add_format({'bold': True, 'font_size': 10, 'font_name': 'Calibri',
                                                    'bg_color': '#D6E4F7', 'font_color': '#1F3864', 'text_wrap': True}),
            "significant":     workbook.add_format({'font_size': 10, 'font_name': 'Calibri',
                                                    'bg_color': '#E2EFDA', 'font_color': '#375623', 'text_wrap': True}),
            "not_significant": workbook.add_format({'font_size': 10, 'font_name': 'Calibri',
                                                    'bg_color': '#F5F5F5', 'font_color': '#808080', 'text_wrap': True}),
            "warning":         workbook.add_format({'font_size': 10, 'font_name': 'Calibri',
                                                    'bg_color': '#FFF2CC', 'font_color': '#7F6000', 'text_wrap': True}),
            "critical":        workbook.add_format({'font_size': 10, 'font_name': 'Calibri',
                                                    'bg_color': '#FCE4D6', 'font_color': '#833C00', 'text_wrap': True}),
            "alternating":     workbook.add_format({'font_size': 10, 'font_name': 'Calibri',
                                                    'bg_color': '#F5F8FD', 'font_color': '#2F2F2F', 'text_wrap': True}),
            "number_4dp":      workbook.add_format({'font_size': 10, 'font_name': 'Calibri',
                                                    'align': 'right', 'num_format': '0.0000'}),
            "pvalue":          workbook.add_format({'font_size': 10, 'font_name': 'Calibri',
                                                    'align': 'right', 'num_format': '0.0000'}),
            "effect_low":      workbook.add_format({'font_size': 10, 'font_name': 'Calibri',
                                                    'bg_color': '#F5F5F5', 'font_color': '#808080', 'text_wrap': True}),
            "effect_medium":   workbook.add_format({'font_size': 10, 'font_name': 'Calibri',
                                                    'bg_color': '#FFF2CC', 'font_color': '#7F6000', 'text_wrap': True}),
            "effect_high":     workbook.add_format({'bold': True, 'font_size': 10, 'font_name': 'Calibri',
                                                    'bg_color': '#E2EFDA', 'font_color': '#375623', 'text_wrap': True}),
            "italic_grey":     workbook.add_format({'italic': True, 'font_size': 10, 'font_name': 'Calibri',
                                                    'font_color': '#808080', 'text_wrap': True}),
            "plain_language":  workbook.add_format({'font_size': 10, 'font_name': 'Calibri',
                                                    'bg_color': '#D6E4F7', 'font_color': '#1F3864', 'text_wrap': True}),
        }
    
    @staticmethod
    def _interpret_effect_size(metric: str, value: float):
        """Return (label, fmt_key) for an effect size value based on standard thresholds.

        Parameters
        ----------
        metric : str
            The effect_size_type string from the results dict.
        value : float
            The numeric effect size (absolute value is used).

        Returns
        -------
        tuple[str, str]
            (human-readable label, fmt dict key)
        """
        v = abs(value)
        m = metric.lower().strip()

        # Eta-squared family
        if m in ("eta_squared", "partial_eta_squared", "omega_squared"):
            if v < 0.01: return ("Negligible", "effect_low")
            if v < 0.06: return ("Small",      "effect_low")
            if v < 0.14: return ("Medium",     "effect_medium")
            return ("Large", "effect_high")

        # Cohen's d family (includes Hedges' g and RM/mixed variants)
        if m in ("cohen_d", "cohen_d_mixed", "cohen_d_rm",
                 "hedges_g", "hedges_g (undefined — zero pooled variance)"):
            if v < 0.20: return ("Negligible", "effect_low")
            if v < 0.50: return ("Small",      "effect_low")
            if v < 0.80: return ("Medium",     "effect_medium")
            return ("Large", "effect_high")

        # Correlation r
        if m in ("r", "r (spearman)"):
            if v < 0.10: return ("Negligible", "effect_low")
            if v < 0.30: return ("Small",      "effect_low")
            if v < 0.50: return ("Medium",     "effect_medium")
            return ("Large", "effect_high")

        # Cohen's f
        if m in ("cohen_f", "cohen's f", "cohen's f (approx.)"):
            if v < 0.10: return ("Negligible", "effect_low")
            if v < 0.25: return ("Small",      "effect_low")
            if v < 0.40: return ("Medium",     "effect_medium")
            return ("Large", "effect_high")

        # Epsilon-squared (Kruskal-Wallis analog)
        if m == "epsilon_squared":
            if v < 0.01: return ("Negligible", "effect_low")
            if v < 0.08: return ("Small",      "effect_low")
            if v < 0.26: return ("Medium",     "effect_medium")
            return ("Large", "effect_high")

        # Kendall's W (concordance coefficient)
        if m in ("kendall_w", "kendall's w"):
            if v < 0.10: return ("Negligible", "effect_low")
            if v < 0.30: return ("Small",      "effect_low")
            if v < 0.50: return ("Medium",     "effect_medium")
            return ("Large", "effect_high")

        # Intraclass correlation coefficient
        if m == "icc":
            if v < 0.10: return ("Negligible", "effect_low")
            if v < 0.40: return ("Small",      "effect_low")
            if v < 0.75: return ("Medium",     "effect_medium")
            return ("Large", "effect_high")

        # AUC, R-squared, and unknowns — no standard magnitude label
        return ("", "cell")

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
            required_formats = ["header", "cell", "sig_highlight", "section_header"]
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
    def _write_cover_sheet(workbook, results, fmt):
        """Professional cover sheet — first tab of every export."""
        from datetime import datetime

        ws = workbook.add_worksheet("Cover")
        ws.hide_gridlines(2)
        ws.set_column(0, 0, 30)   # A — labels
        ws.set_column(1, 1, 45)   # B — values

        # Helper: merge two columns and write with given format
        def merge_write(r, text, cell_fmt):
            ws.merge_range(r, 0, r, 1, text, cell_fmt)

        row = 3   # rows 0-2 are top margin

        # ── Main title ───────────────────────────────────────────────────────
        ws.set_row(row, 24)
        merge_write(row, "BioMedStatX 2.0 \u2014 Statistical Analysis Report", fmt["navy_header"])
        row += 2  # blank

        # ── Analysis Information ─────────────────────────────────────────────
        merge_write(row, "Analysis Information", fmt["section_blue"])
        row += 1

        model_type = results.get("model_type", results.get("test", "Unknown"))
        dep_var    = results.get("dependent_variable", results.get("value_column", "\u2014"))
        factors    = results.get("group_column", "")
        if results.get("factors"):
            factors = ", ".join(results["factors"]) if isinstance(results["factors"], list) else str(results["factors"])
        covariates = results.get("covariates", [])
        if isinstance(covariates, list):
            cov_text = ", ".join(str(c) for c in covariates) if covariates else "None"
        else:
            cov_text = str(covariates) if covariates else "None"
        filter_text = str(results.get("filter_applied", "None")) or "None"
        n_val = results.get("n_total", results.get("n", "\u2014"))

        info_rows = [
            ("Analysis Type:",       model_type),
            ("Dependent Variable:",  dep_var),
            ("Factor(s):",           factors or "\u2014"),
            ("Covariates:",          cov_text),
            ("Filter applied:",      filter_text),
            ("Sample size (N):",     str(n_val) if n_val is not None else "\u2014"),
        ]
        for label, value in info_rows:
            ws.write(row, 0, label, fmt["key"])
            ws.write(row, 1, str(value), fmt["cell"])
            row += 1
        row += 1  # blank

        # ── Results at a Glance ──────────────────────────────────────────────
        merge_write(row, "Results at a Glance", fmt["section_blue"])
        row += 1

        p_value = results.get("p_value")
        alpha   = results.get("alpha", 0.05)
        is_sig  = p_value is not None and isinstance(p_value, (float, int)) and p_value < alpha

        if p_value is not None and isinstance(p_value, (float, int)):
            p_text = "<0.001" if p_value < 0.001 else f"{p_value:.4f}"
        else:
            p_text = "Not available"

        p_fmt = fmt["significant"] if is_sig else fmt["critical"]
        ws.write(row, 0, "Main p-value:", fmt["key"])
        ws.write(row, 1, p_text, p_fmt)
        row += 1

        sig_fmt  = fmt["significant"] if is_sig else fmt["not_significant"]
        sig_text = "Yes" if is_sig else "No"
        ws.write(row, 0, "Significant:", fmt["key"])
        ws.write(row, 1, sig_text, sig_fmt)
        row += 1

        eff_val  = results.get("effect_size")
        eff_type = results.get("effect_size_type", "")
        if eff_val is not None and isinstance(eff_val, (float, int)):
            mag_label, eff_fmt_key = ResultsExporter._interpret_effect_size(eff_type, eff_val)
            eff_display = f"{eff_val:.4f}"
            if mag_label:
                eff_display += f" \u2014 {mag_label}"
            if eff_type:
                eff_display += f" ({eff_type})"
            eff_fmt = fmt.get(eff_fmt_key, fmt["cell"])
        else:
            eff_display = "Not available"
            eff_fmt = fmt["cell"]
        ws.write(row, 0, "Effect size:", fmt["key"])
        ws.write(row, 1, eff_display, eff_fmt)
        row += 1

        ws.write(row, 0, "Test used:", fmt["key"])
        ws.write(row, 1, str(results.get("test", "Not specified")), fmt["cell"])
        row += 2  # blank

        # ── Report Contents ──────────────────────────────────────────────────
        merge_write(row, "Report Contents", fmt["section_blue"])
        row += 1

        _sheet_desc = [
            ("Cover",                "This overview page"),
            ("Summary",              "Key statistics, effect size, confidence interval, power"),
            ("Statistical Results",  "Full test output, ANOVA table, omnibus results"),
            ("Descriptives",         "Group means, SDs, medians, quartiles"),
        ]
        _groups  = results.get("groups", [])
        _pairwise = results.get("pairwise_comparisons", [])
        if len(_groups) >= 3 and _pairwise:
            _sheet_desc.append(("Pairwise Comparisons", "Post-hoc test results with corrected p-values"))
        _sheet_desc.append(("Assumptions", "Normality, variance homogeneity, sphericity, diagnostic plots"))
        _mt = results.get("model_type", "")
        if _mt == "ANCOVA":
            _sheet_desc.append(("ANCOVA Details", "Adjusted means, regression slope homogeneity"))
        elif _mt == "LMM":
            _sheet_desc.append(("LMM Details", "Fixed and random effects, ICC, model fit indices"))
        elif _mt == "Correlation":
            _sheet_desc.append(("Correlation", "Pearson/Spearman coefficient, scatter plot"))
        elif _mt == "CorrelationMatrix":
            _sheet_desc.append(("Correlation Matrix", "Full pairwise correlation matrix by group"))
        elif _mt == "LinearRegression":
            _sheet_desc.append(("Linear Regression", "Model summary, coefficients, diagnostics"))
        elif _mt == "LogisticRegression":
            _sheet_desc.append(("Logistic Regression", "Odds ratios, Hosmer-Lemeshow, ROC/AUC"))
        elif _mt == "BetaRegression":
            _sheet_desc.append(("Beta Regression", "Coefficients, pseudo-R², dispersion (phi), diagnostics"))
        _sheet_desc += [
            ("Decision Tree",    "Visual flowchart of statistical decisions made"),
            ("Methodology Log",  "Step-by-step audit trail and suggested Methods section text"),
            ("Raw Data",         "Original data used in this analysis"),
        ]
        for sheet_name, desc in _sheet_desc:
            ws.write(row, 0, sheet_name, fmt["bold"])
            ws.write(row, 1, desc, fmt["cell"])
            row += 1
        row += 1

        # ── Footer ───────────────────────────────────────────────────────────
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        merge_write(row, f"Generated by BioMedStatX 2.0 \u2014 {now}", fmt["italic_grey"])

    @staticmethod
    def _write_summary_sheet(workbook, results, fmt, sheet_name="Summary"):
        from datetime import datetime

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

        model_type = results.get("model_type", results.get("test", "Unknown"))
        dep_var = results.get("dependent_variable", results.get("value_column", "—"))
        factors = results.get("group_column", "")
        factor_columns = results.get("factor_columns")
        if isinstance(factor_columns, list) and factor_columns:
            factors = ", ".join(map(str, factor_columns))
        elif isinstance(results.get("factors"), list) and results["factors"]:
            if all(isinstance(item, dict) for item in results["factors"]):
                factor_names = [str(item.get("factor", "")) for item in results["factors"] if item.get("factor")]
                if factor_names:
                    factors = ", ".join(factor_names)
            else:
                factors = ", ".join(map(str, results["factors"]))
        elif results.get("factors"):
            factors = str(results["factors"])
        covariates = results.get("covariates", [])
        if isinstance(covariates, list):
            cov_text = ", ".join(str(c) for c in covariates) if covariates else "None"
        else:
            cov_text = str(covariates) if covariates else "None"
        filter_text = str(results.get("filter_applied", "None")) or "None"
        selected_groups = results.get("selected_groups") or results.get("groups") or []
        selected_group_text = ", ".join(map(str, selected_groups)) if selected_groups else "None"
        n_val = results.get("n_total", results.get("n", "—"))

        ws.merge_range('A3:F3', "ANALYSIS OVERVIEW", fmt["section_header"])
        overview_rows = [
            ("Analysis Type:", model_type),
            ("Dependent Variable:", dep_var),
            ("Factor(s):", factors or "—"),
            ("Selected Groups:", selected_group_text),
            ("Covariates:", cov_text),
            ("Filter applied:", filter_text),
            ("Sample size (N):", str(n_val) if n_val is not None else "—"),
        ]
        row = 4
        for key, value in overview_rows:
            ws.write(row, 0, key, fmt["key"])
            ws.write(row, 1, str(value), fmt["cell"])
            row += 1

        ws.merge_range(f'A{row}:F{row}', "REPORT CONTENTS", fmt["section_header"])
        row += 1
        sheet_rows = [
            ("Summary", "Overview, key statement, key statistics, report navigation"),
            ("Statistical Results", "Full test output, ANOVA table, omnibus results"),
            ("Descriptive Statistics", "Group means, SDs, medians, quartiles"),
        ]
        _groups = results.get("groups", [])
        _pairwise = results.get("pairwise_comparisons", [])
        if len(_groups) >= 3 and _pairwise:
            sheet_rows.append(("Pairwise Comparisons", "Post-hoc test results with corrected p-values"))
        sheet_rows.append(("Assumptions", "Normality, variance homogeneity, sphericity, diagnostic plots"))
        _mt = results.get("model_type", "")
        if _mt == "ANCOVA":
            sheet_rows.append(("ANCOVA Details", "Adjusted means and slope homogeneity checks"))
        elif _mt == "LMM":
            sheet_rows.append(("LMM Details", "Fixed/random effects, ICC, model fit indices"))
        elif _mt == "Correlation":
            sheet_rows.append(("Correlation", "Coefficient, confidence interval, interpretation"))
        elif _mt == "CorrelationMatrix":
            sheet_rows.append(("Correlation Matrix", "Full pairwise correlation matrix"))
        elif _mt == "LinearRegression":
            sheet_rows.append(("Linear Regression", "Model summary, coefficients, diagnostics"))
        elif _mt == "LogisticRegression":
            sheet_rows.append(("Logistic Regression", "Odds ratios, calibration, ROC/AUC"))
        elif _mt == "BetaRegression":
            sheet_rows.append(("Beta Regression", "Coefficients, pseudo-R², dispersion (phi), diagnostics"))
        sheet_rows.extend([
            ("Decision Tree", "Visual flowchart of the applied decision path"),
            ("Methodology Log", "Audit trail and methods-text helper"),
            ("Raw Data", "Original data used in this analysis"),
        ])
        for sheet_name_label, desc in sheet_rows:
            ws.write(row, 0, sheet_name_label, fmt["bold"])
            ws.write(row, 1, desc, fmt["cell"])
            row += 1

        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        ws.merge_range(f'A{row}:F{row}', f"Generated by BioMedStatX 2.0 — {now}", fmt["italic_grey"])
        row += 2

        # Key statement
        ws.merge_range(f'A{row}:F{row}', "KEY STATEMENT", fmt["section_header"])
        
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
                    if effect_type.lower() in ["cohen_d", "hedges_g"]:
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
        row += 1
        ws.merge_range(f'A{row}:F{row}', conclusion, fmt["cell"])
        ws.set_row(row, ResultsExporter.get_fixed_row_height("summary_conclusion"))

        # Key information
        row += 2
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

            # Human-readable label for the effect size type
            _type_labels = {
                "cohen_d": "Cohen's d", "cohen_d_mixed": "Cohen's d", "cohen_d_rm": "Cohen's d",
                "hedges_g": "Hedges' g",
                "eta_squared": "Eta\u00b2", "partial_eta_squared": "Partial \u03b7\u00b2",
                "omega_squared": "Omega\u00b2", "epsilon_squared": "Epsilon\u00b2",
                "kendall_w": "Kendall's W", "kendall's w": "Kendall's W",
                "cohen_f": "Cohen's f", "cohen's f": "Cohen's f",
                "cohen's f (approx.)": "Cohen's f (approx.)",
                "r": "r (rank correlation)", "r (spearman)": "r (Spearman)",
                "icc": "ICC", "auc": "AUC", "r_squared": "R\u00b2",
            }
            effect_desc = _type_labels.get(effect_type.lower(), effect_type if effect_type else "Effect size")

            magnitude, fmt_key = ResultsExporter._interpret_effect_size(effect_type, effect_size)
            format_to_use = fmt.get(fmt_key, fmt["cell"])

            label_parts = [f"{effect_size:.4f}"]
            if magnitude:
                label_parts.append(magnitude)
            key_value_pairs.append((f"{effect_desc}:", " — ".join(label_parts)))
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

        if results.get("model_class"):
            key_value_pairs.append(("Model class:", str(results["model_class"])))

        if results.get("model_family"):
            key_value_pairs.append(("Model family:", str(results["model_family"])))

        if results.get("analysis_note"):
            key_value_pairs.append(("Analysis note:", str(results["analysis_note"])))

        if results.get("interaction_significant"):
            key_value_pairs.append((
                "Reporting priority:",
                "Interaction first; interpret main effects as averaged effects across the other factor."
            ))

        for key, value in key_value_pairs:
            ws.write(row, 0, key, fmt["key"])
            current_format = fmt["cell"]
            if key == "Significant:" and value == "Yes":
                current_format = fmt["sig_highlight"]
            elif key == "p-Value:" and is_significant:
                current_format = fmt["sig_highlight"]
            elif any(token in key for token in (
                    "Effect size", "Cohen's d", "Cohen's f", "Hedges'",
                    "Eta²", "Partial η²", "Omega²", "Epsilon²",
                    "Kendall's W", "r (rank", "r (Spearman)", "ICC", "AUC", "R²")):
                current_format = format_to_use
            ws.write(row, 1, value, current_format)
            row += 1

        # Analysis Warnings
        _warnings = results.get("warnings") or []
        if _warnings:
            row += 1
            ws.merge_range(f'A{row}:F{row}', "ANALYSIS WARNINGS", fmt["warning"])
            row += 1
            for _w in _warnings:
                ws.merge_range(f'A{row}:F{row}', str(_w), fmt["warning"])
                row += 1
            row += 1

        # Navigation
        row += 2
        ws.merge_range(f'A{row}:F{row}', "NAVIGATION TO DETAILED RESULTS", fmt["section_header"])
        row += 1
        nav_text = (
            "• Methodology: Model family, covariance choices, overdispersion metrics, and correction strategy\n"
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
                "Custom paired t-tests (Holm-Bonferroni)": "User-selected group pairs are compared using paired t-tests with Holm-Bonferroni correction for multiple comparisons. This allows for focused comparisons of specific group pairs.",
                "Dunn": "Dunn's test is a non-parametric post-hoc test that compares all possible pairs after a significant Kruskal-Wallis test, using rank-based statistics with Holm-Bonferroni correction.",
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

    @staticmethod
    def _write_methodology_log_sheet(workbook, results, fmt, trace=None, analysis_log=None, sheet_name="Methodology Log"):
        """Write the Methodology Log sheet — decision audit trail + suggested Methods paragraph.

        Parameters
        ----------
        trace : MethodologyTrace | None
            If None the sheet is still created but contains only a notice.
        """
        ws = workbook.add_worksheet(sheet_name)
        ws.set_column(0, 0, 6)    # Step
        ws.set_column(1, 1, 20)   # Category
        ws.set_column(2, 2, 60)   # Decision
        ws.set_column(3, 3, 25)   # Detail
        ws.freeze_panes(3, 0)

        row = 0

        # ── Title ────────────────────────────────────────────────────────────
        ws.merge_range(f'A{row+1}:D{row+1}',
                       "Methodology Log \u2014 Analysis Decision Trail", fmt["navy_header"])
        ws.set_row(row, 22)
        row += 1

        # ── Intro ─────────────────────────────────────────────────────────────
        intro = (
            "This log documents every automated decision made during the analysis. "
            "It can be used as a basis for the Methods section of a manuscript."
        )
        ws.merge_range(f'A{row+1}:D{row+1}', intro, fmt["italic_grey"])
        ws.set_row(row, 30)
        row += 1

        selected_groups = results.get("selected_groups") or results.get("groups") or []
        selected_groups_text = ", ".join(map(str, selected_groups)) if selected_groups else "All available groups"
        grouping_column = results.get("group_column") or results.get("group_col") or "Not specified"
        filter_applied = str(results.get("filter_applied") or "None")
        scope_text = (
            f"Grouping column: {grouping_column}\n"
            f"Selected groups: {selected_groups_text}\n"
            f"Filter applied: {filter_applied}"
        )
        ws.merge_range(f'A{row+1}:D{row+1}', "Analysis Scope", fmt["section_blue"])
        ws.set_row(row, 18)
        row += 1
        ws.merge_range(f'A{row+1}:D{row+1}', scope_text, fmt["plain_language"])
        ws.set_row(row, 58)
        row += 1

        # ── Column headers ────────────────────────────────────────────────────
        col_headers = ["Step", "Category", "Decision", "Detail"]
        for i, h in enumerate(col_headers):
            ws.write(row, i, h, fmt["subheader_light"])
        row += 1

        # ── Decision rows ─────────────────────────────────────────────────────
        _cat_fmt_map = {
            "normality":     "subheader_light",
            "assumption":    "warning",
            "test selection": "section_blue",
            "post-hoc":      "subheader_light",
            "correction":    "subheader_light",
            "data check":    "subheader_light",
        }

        steps = trace.to_list() if trace and hasattr(trace, "to_list") else []
        if steps:
            for i, step in enumerate(steps):
                row_bg = fmt["alternating"] if i % 2 == 0 else fmt["cell"]
                cat_key = _cat_fmt_map.get(step["category"].lower(), "cell")
                cat_fmt = fmt.get(cat_key, fmt["cell"])

                ws.write(row, 0, step["step"], row_bg)
                ws.write(row, 1, step["category"], cat_fmt)
                ws.write(row, 2, step["decision"], row_bg)
                ws.write(row, 3, step.get("detail", ""), fmt["italic_grey"])
                row += 1
        elif analysis_log:
            log_text = str(analysis_log).strip()
            ws.merge_range(f'A{row+1}:D{row+1}', "Analysis Log", fmt["section_blue"])
            ws.set_row(row, 18)
            row += 1
            ws.merge_range(f'A{row+1}:D{row+1}', log_text, fmt["plain_language"])
            ws.set_row(row, max(120, 15 * (log_text.count("\n") + 2)))
            row += 1
        else:
            ws.merge_range(f'A{row+1}:D{row+1}',
                           "No methodology trace available for this analysis.",
                           fmt["italic_grey"])
            row += 1

        row += 1

        # ── Suggested Methods Section ─────────────────────────────────────────
        ws.merge_range(f'A{row+1}:D{row+1}',
                       "Suggested Methods Section Text", fmt["section_blue"])
        ws.set_row(row, 18)
        row += 1

        if trace and hasattr(trace, "to_methods_paragraph"):
            methods_text = trace.to_methods_paragraph()
        elif analysis_log:
            methods_text = str(analysis_log).strip()
        else:
            methods_text = (
                "Statistical analysis was performed using BioMedStatX 2.0. "
                "The significance threshold was set at \u03b1\u202f=\u202f0.05."
            )
        ws.merge_range(f'A{row+1}:D{row+1}', methods_text, fmt["plain_language"])
        ws.set_row(row, max(60, 15 * (methods_text.count(".") + 1)))
        row += 1

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
            p_val_str = float(p_val) if isinstance(p_val, (float, int)) else (p_val or "N/A")
        else:
            p_val_str = float(p_val) if isinstance(p_val, (float, int)) else (p_val or "N/A")
            
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
            elif effect_type == "Kendall's W":
                if effect_size < 0.1: effect_str = f"{effect_size:.4f} (weak)"
                elif effect_size < 0.3: effect_str = f"{effect_size:.4f} (moderate)"
                else: effect_str = f"{effect_size:.4f} (strong)"
            elif effect_type in ["Cohen's f", "Cohen's f (approx.)"]:
                if effect_size < 0.1: effect_str = f"{effect_size:.4f} (very small)"
                elif effect_size < 0.25: effect_str = f"{effect_size:.4f} (small)"
                elif effect_size < 0.4: effect_str = f"{effect_size:.4f} (medium)"
                else: effect_str = f"{effect_size:.4f} (large)"
            elif effect_type == "r":
                if effect_size < 0.1: effect_str = f"{effect_size:.4f} (very small)"
                elif effect_size < 0.3: effect_str = f"{effect_size:.4f} (small)"
                elif effect_size < 0.5: effect_str = f"{effect_size:.4f} (medium)"
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
            fmtx = fmt["sig_highlight"] if (col == 2 and is_significant) or (col == 6 and is_significant) else fmt["cell"]
            ws.write(row, col, val, fmtx)
        row += 2

        # --- Factor / interaction breakdown (Friedman, Freedman-Lane, Brunner-Langer ATS) ---
        factors_list = [item for item in results.get("factors", []) if isinstance(item, dict)]
        interactions_list = [item for item in results.get("interactions", []) if isinstance(item, dict)]
        if factors_list or interactions_list:
            ws.merge_range(f'A{row+1}:G{row+1}', "EFFECTS BREAKDOWN (all factors & interactions)", fmt["section_header"])
            row += 2
            eff_hdrs = ["Source", "Statistic", "df1", "df2", "p-Value", "Effect size", "Effect size type"]
            for col, h in enumerate(eff_hdrs):
                ws.write(row, col, h, fmt["header"])
            row += 1

            def _fmt_num(v, decimals=4):
                return f"{v:.{decimals}f}" if isinstance(v, (float, int)) else ("N/A" if v is None else str(v))

            def _eff_interp(es, et):
                if es is None:
                    return "N/A"
                s = f"{es:.4f}"
                if et in ["Cohen's f", "Cohen's f (approx.)"]:
                    label = "very small" if es < 0.1 else ("small" if es < 0.25 else ("medium" if es < 0.4 else "large"))
                    return f"{s} ({label})"
                if et == "Kendall's W":
                    label = "weak" if es < 0.1 else ("moderate" if es < 0.3 else "strong")
                    return f"{s} ({label})"
                if et in ["eta_squared", "partial_eta_squared"]:
                    label = "very small" if es < 0.01 else ("small" if es < 0.06 else ("medium" if es < 0.14 else "large"))
                    return f"{s} ({label})"
                return s

            for f in factors_list:
                f_stat = f.get("F") or f.get("Wald_Chi2") or f.get("chi2")
                f_p = f.get("p_value")
                f_sig = isinstance(f_p, (float, int)) and f_p < results.get("alpha", 0.05)
                f_es = f.get("effect_size")
                f_et = f.get("effect_size_type") or ""
                row_vals = [
                    f.get("factor", "?"),
                    _fmt_num(f_stat),
                    _fmt_num(f.get("df1")),
                    _fmt_num(f.get("df2")),
                    _fmt_num(f_p),
                    _eff_interp(f_es, f_et),
                    f_et or "—",
                ]
                for col, val in enumerate(row_vals):
                    ws.write(row, col, val, fmt["sig_highlight"] if f_sig and col in (4,) else fmt["cell"])
                row += 1

            for inter in interactions_list:
                inter_name = " × ".join(inter.get("factors", ["?"])) if "factors" in inter else inter.get("factor", "Interaction")
                i_stat = inter.get("F") or inter.get("Wald_Chi2")
                i_p = inter.get("p_value")
                i_sig = isinstance(i_p, (float, int)) and i_p < results.get("alpha", 0.05)
                i_es = inter.get("effect_size")
                i_et = inter.get("effect_size_type") or ""
                row_vals = [
                    f"Interaction: {inter_name}",
                    _fmt_num(i_stat),
                    _fmt_num(inter.get("df1")),
                    _fmt_num(inter.get("df2")),
                    _fmt_num(i_p),
                    _eff_interp(i_es, i_et),
                    i_et or "—",
                ]
                for col, val in enumerate(row_vals):
                    ws.write(row, col, val, fmt["sig_highlight"] if i_sig and col in (4,) else fmt["cell"])
                row += 1
            row += 1

        # --- Relative Treatment Effects (Brunner-Langer ATS only) ---
        rte_df = results.get("RTE")
        if rte_df is not None and isinstance(rte_df, pd.DataFrame) and not rte_df.empty:
            ws.merge_range(f'A{row+1}:E{row+1}', "RELATIVE TREATMENT EFFECTS (RTE)", fmt["section_header"])
            row += 2
            ws.merge_range(f'A{row+1}:E{row+1}',
                "Relative Treatment Effects (RTE) measure the probability that a randomly chosen "
                "observation from one cell is smaller than from another. RTE = 0.5 means no effect; "
                "values above/below 0.5 indicate the direction of the effect.",
                fmt["explanation"])
            ws.set_row(row, 40)
            row += 2
            rte_cols = list(rte_df.columns)
            for col, h in enumerate(rte_cols):
                ws.write(row, col, str(h), fmt["header"])
            row += 1
            for _, rte_row in rte_df.iterrows():
                for col, h in enumerate(rte_cols):
                    val = rte_row[h]
                    if isinstance(val, float):
                        ws.write(row, col, f"{val:.4f}", fmt["cell"])
                    else:
                        ws.write(row, col, str(val) if val is not None else "N/A", fmt["cell"])
                row += 1
            row += 1

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
            gg_corr = results["sphericity_corrections"].get("greenhouse_geisser", {})
            gg_p = gg_corr.get("p_value")
            gg_epsilon = gg_corr.get("epsilon")
            gg_df1 = gg_corr.get("corrected_df1", gg_corr.get("df1"))
            gg_df2 = gg_corr.get("corrected_df2", gg_corr.get("df2"))
            gg_sig = gg_p < results.get("alpha", 0.05) if isinstance(gg_p, (float, int)) else False
            ws.write(row, 0, "Greenhouse-Geisser", fmt["cell"])
            ws.write(row, 1, f"{gg_epsilon:.4f}" if isinstance(gg_epsilon, (float, int)) else "N/A", fmt["cell"])
            ws.write(row, 2, f"{gg_df1:.4f}" if isinstance(gg_df1, (float, int)) else "N/A", fmt["cell"])
            ws.write(row, 3, f"{gg_df2:.4f}" if isinstance(gg_df2, (float, int)) else "N/A", fmt["cell"])
            ws.write(row, 4, f"{gg_p:.4f}" if isinstance(gg_p, (float, int)) else "N/A",
                    fmt["sig_highlight"] if gg_sig else fmt["cell"])
            ws.write(row, 5, "Yes" if gg_sig else "No",
                    fmt["sig_highlight"] if gg_sig else fmt["cell"])
            row += 1

            # Huynh-Feldt correction
            hf_corr = results["sphericity_corrections"].get("huynh_feldt", {})
            hf_p = hf_corr.get("p_value")
            hf_epsilon = hf_corr.get("epsilon")
            hf_df1 = hf_corr.get("corrected_df1", hf_corr.get("df1"))
            hf_df2 = hf_corr.get("corrected_df2", hf_corr.get("df2"))
            hf_sig = hf_p < results.get("alpha", 0.05) if isinstance(hf_p, (float, int)) else False
            ws.write(row, 0, "Huynh-Feldt", fmt["cell"])
            ws.write(row, 1, f"{hf_epsilon:.4f}" if isinstance(hf_epsilon, (float, int)) else "N/A", fmt["cell"])
            ws.write(row, 2, f"{hf_df1:.4f}" if isinstance(hf_df1, (float, int)) else "N/A", fmt["cell"])
            ws.write(row, 3, f"{hf_df2:.4f}" if isinstance(hf_df2, (float, int)) else "N/A", fmt["cell"])
            ws.write(row, 4, f"{hf_p:.4f}" if isinstance(hf_p, (float, int)) else "N/A",
                    fmt["sig_highlight"] if hf_sig else fmt["cell"])
            ws.write(row, 5, "Yes" if hf_sig else "No",
                    fmt["sig_highlight"] if hf_sig else fmt["cell"])
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
                    fmtx = fmt["sig_highlight"] if (col == 2 and sig) or (col == 3 and sig) else fmt["cell"]
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

        # --- ANOVA Table (one-way, two-way, RM, Mixed ANOVA) ---
        # Skipped for ANCOVA — that sheet renders its own ANOVA table.
        _anova_tbl = results.get("anova_table")
        _model_type = results.get("model_type", "")
        if _anova_tbl is not None and _model_type != "ANCOVA":
            _df = (pd.DataFrame(_anova_tbl) if isinstance(_anova_tbl, dict) else _anova_tbl)
            if isinstance(_df, pd.DataFrame) and not _df.empty:
                ws.merge_range(f'A{row}:F{row}', "ANOVA TABLE", fmt["section_header"])
                row += 1
                row = ResultsExporter._write_anova_table(ws, _df, fmt, start_row=row)
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
            "• Custom Paired t-tests: User-selected pairs with Holm-Bonferroni correction\n\n"
            "NON-PARAMETRIC POST-HOC TESTS (when normality assumptions are violated):\n"
            "• Dunn Test: Rank-based comparisons of all pairs with Holm-Bonferroni correction\n"
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
        from scipy.stats import t as _scipy_t
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
                    if n and n > 1 and stderr is not None:
                        ci_lower, ci_upper = _scipy_t.interval(0.95, n - 1, loc=mean, scale=stderr)
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
                        if n and n > 1 and stderr is not None:
                            ci_lower, ci_upper = _scipy_t.interval(0.95, n - 1, loc=mean, scale=stderr)
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

        # If post-hoc was explicitly skipped (non-significant main test), show reason and stop.
        if results.get("posthoc_skipped") and not results.get("pairwise_comparisons"):
            ws.merge_range('A3:H3',
                           results.get("posthoc_skip_reason",
                                       "Post-hoc tests were not performed for this analysis."),
                           fmt["not_significant"])
            return

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
            "• Custom Paired t-tests (Holm-Bonferroni): User-selected pairs with Holm-Bonferroni correction\n"
            "• Dunn Test: Non-parametric all pairwise comparisons with Holm-Bonferroni correction\n"
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
                        eff_fmt = fmt["effect_med_text"]
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
                    current_fmt = fmt["sig_highlight"]
                elif headers[col] == "Significant" and is_sign:
                    current_fmt = fmt["sig_highlight"]
                elif headers[col] == "Effect size" and isinstance(effect_size_val, (float, int)):
                    current_fmt = eff_fmt  # Use pre-determined format for effect size
                ws.write(row + comp_idx, col, val_to_write, current_fmt)
    
    @staticmethod
    def _write_decision_tree_sheet(workbook, results, fmt, sheet_name="Decision Tree", pre_generated_tree=None):
        """Write decision tree sheet with visualization."""
        from visualization.decisiontreevisualizer import DecisionTreeVisualizer
        
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
                    
                    # Scale to keep wide trees readable while avoiding oversize sheet rendering.
                    max_embed_width = 3800
                    max_embed_height = 2800
                    width_scale = max_embed_width / float(width) if width > 0 else 1.0
                    height_scale = max_embed_height / float(height) if height > 0 else 1.0
                    scale_factor = min(1.0, width_scale, height_scale)
                    scale_factor = max(0.5, scale_factor)
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
                ws.write(row, 0, "⚠️ ERROR: Invalid results data format", fmt["sig_highlight"])
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
                    ws.write(row, 0, f"⚠️ ERROR in sphericity section: {str(e)}", fmt["sig_highlight"])
                    row += 2
            
            # 2. Between-Factor Assumptions (Mixed ANOVA)
            if "between_assumptions" in results:
                try:
                    row = ResultsExporter._write_between_factor_assumptions(workbook, results, fmt, ws, row)
                except Exception as e:
                    ws.write(row, 0, f"⚠️ ERROR in between-factor assumptions: {str(e)}", fmt["sig_highlight"])
                    row += 2
            
            # 3. Within-Factor Sphericity (Mixed ANOVA)
            if "within_sphericity_test" in results:
                try:
                    row = ResultsExporter._write_within_factor_sphericity(workbook, results, fmt, ws, row)
                except Exception as e:
                    ws.write(row, 0, f"⚠️ ERROR in within-factor sphericity: {str(e)}", fmt["sig_highlight"])
                    row += 2
            
            # 4. Interaction Assumptions (Mixed ANOVA)
            if "interaction_assumptions" in results:
                try:
                    row = ResultsExporter._write_interaction_assumptions(workbook, results, fmt, ws, row)
                except Exception as e:
                    ws.write(row, 0, f"⚠️ ERROR in interaction assumptions: {str(e)}", fmt["sig_highlight"])
                    row += 2
            
            return row
            
        except Exception as e:
            # Global error handling
            error_msg = f"⚠️ CRITICAL ERROR in enhanced assumption tests: {str(e)}"
            try:
                ws.write(start_row, 0, error_msg, fmt["sig_highlight"])
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
                ws.write(row, 0, f"⚠️ DATA QUALITY WARNING: {', '.join(data_quality_issues)}", fmt["sig_highlight"])
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
                    ws.write(row, col, val, fmt["sig_highlight"])
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
                ws.write(start_row, 0, error_msg, fmt["sig_highlight"])
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
                ws.write(row, 0, "⚠️ ERROR: Invalid results format for corrections", fmt["sig_highlight"])
                return row + 2
                
            if "sphericity_corrections" not in results:
                ws.write(row, 0, "⚠️ WARNING: No sphericity corrections available", fmt["sig_highlight"])
                return row + 2
                
            corrections = results["sphericity_corrections"]
            if not isinstance(corrections, dict):
                ws.write(row, 0, "⚠️ ERROR: Invalid corrections data format", fmt["sig_highlight"])
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
                ws.write(row, 0, f"⚠️ ERROR writing corrections table: {str(table_error)}", fmt["sig_highlight"])
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
                ws.write(row, 0, f"⚠️ ERROR writing recommendations: {str(rec_error)}", fmt["sig_highlight"])
                row += 2
            
            return row
            
        except Exception as e:
            # Global error handling for corrections section
            error_msg = f"⚠️ CRITICAL ERROR in sphericity corrections: {str(e)}"
            try:
                ws.write(start_row, 0, error_msg, fmt["sig_highlight"])
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
                ws.write(start_row, 0, error_msg, fmt["sig_highlight"])
                return start_row + error_rows
            
            row = start_row
            
            # Check if between_assumptions data exists
            if "between_assumptions" not in results:
                ws.write(row, 0, "⚠️ WARNING: No between-factor assumptions data available", fmt["sig_highlight"])
                return row + 2
            
            between_assumptions = results["between_assumptions"]
            
            # Section header
            ws.write(row, 0, "BETWEEN-FACTOR ASSUMPTIONS (MIXED ANOVA)", fmt["section_header"])
            row += 1
            
            # Between-factor explanation
            between_explanation = (
                "BETWEEN-SUBJECTS FACTOR ASSUMPTIONS:\n"
                "For Mixed ANOVA, the between-subjects factor must meet homogeneity of variance assumptions.\n\n"
                "TESTS PERFORMED:\n"
                "• Levene's Test: Standard test for equal variances\n"
                "• Levene's Test (Brown-Forsythe): Robust variant using medians\n"
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
                                ws.write(row, col, val, fmt["sig_highlight"])
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
                ws.write(start_row, 0, error_msg, fmt["sig_highlight"])
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
                ws.write(row, col, val, fmt["sig_highlight"])
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
                    fmt["sig_highlight"] if sphericity_met is False else fmt["cell"])
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
                ws.write(row, 0, f"Levene's Test (Brown-Forsythe) p-value: {p_str}",
                        fmt["sig_highlight"] if assumption_met is False else fmt["cell"])
                row += 1
                
                ws.write(row, 0, f"Cell Homogeneity: {'Met' if assumption_met else 'Violated' if assumption_met is not None else 'Unknown'}", 
                        fmt["sig_highlight"] if assumption_met is False else fmt["cell"])
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

    # ========================================================================
    # Clinical Model Export Sheets
    # ========================================================================

    @staticmethod
    def _write_data_health_section(ws, row, results, fmt):
        """Write the Clinical Data Health warnings block into any clinical sheet."""
        health = results.get("data_health")
        if not health:
            return

        warnings = health.get("warnings", [])
        checks = health.get("checks", {})

        ws.write(row, 0, "Clinical Data Health Check", fmt["section_header"])
        row += 1

        if not warnings:
            ws.write(row, 0, "No data quality issues detected.", fmt["cell"])
            row += 1
        else:
            for w in warnings:
                ws.write(row, 0, f"⚠  {w}", fmt["sig_highlight"])
                row += 1

        row += 1

        # VIF detail table
        vif = checks.get("vif")
        if vif and isinstance(vif, dict) and "error" not in vif:
            ws.write(row, 0, "Variance Inflation Factors (VIF)", fmt["header"])
            row += 1
            for col, val in vif.items():
                ws.write(row, 0, col, fmt["cell"])
                ws.write(row, 1, val, fmt["cell"])
                ws.write(row, 2, "HIGH" if val > 10 else ("elevated" if val > 5 else "ok"),
                         fmt["sig_highlight"] if val > 10 else fmt["cell"])
                row += 1
            row += 1

        # Little's MCAR detail
        mcar = checks.get("mcar")
        if mcar and isinstance(mcar, dict) and "error" not in mcar and "note" not in mcar:
            ws.write(row, 0, "Little's MCAR Test", fmt["header"])
            row += 1
            ws.write(row, 0, "χ²", fmt["cell"])
            ws.write(row, 1, mcar.get("chi2"), fmt["cell"])
            row += 1
            ws.write(row, 0, "df", fmt["cell"])
            ws.write(row, 1, mcar.get("df"), fmt["cell"])
            row += 1
            ws.write(row, 0, "p-value", fmt["cell"])
            ws.write(row, 1, mcar.get("p_value"), fmt["cell"])
            row += 1
            interp = mcar.get("interpretation",
                               "Significant: non-random missing data (MAR/MNAR)" if
                               (mcar.get("p_value") or 1) < 0.05 else
                               "MCAR not rejected")
            ws.write(row, 0, "Interpretation", fmt["cell"])
            ws.write(row, 1, interp, fmt["cell"])
            row += 1

    @staticmethod
    def _write_ancova_sheet(workbook, results, fmt):
        ws = workbook.add_worksheet("ANCOVA Details")
        ws.set_column('A:A', 35)
        ws.set_column('B:G', 18)
        row = 0

        ws.write(row, 0, "ANCOVA Analysis Details", fmt["title"])
        row += 2

        # ANOVA Table (Type II SS)
        ws.write(row, 0, "ANOVA Table (Type II Sum of Squares)", fmt["section_header"])
        row += 1
        headers = ["Source", "Sum of Sq.", "df", "F", "p-value"]
        for c, h in enumerate(headers):
            ws.write(row, c, h, fmt["header"])
        row += 1
        for entry in results.get("anova_table", []):
            ws.write(row, 0, str(entry.get("source", "")), fmt["cell"])
            ws.write(row, 1, entry.get("sum_sq"), fmt["cell"])
            ws.write(row, 2, entry.get("df"), fmt["cell"])
            f_val = entry.get("F")
            ws.write(row, 3, f_val if f_val is not None else "N/A", fmt["cell"])
            p_val = entry.get("p_value")
            if p_val is not None and isinstance(p_val, (float, int)):
                cell_fmt = fmt["sig_highlight"] if p_val < 0.05 else fmt["cell"]
                ws.write(row, 4, p_val, cell_fmt)
            else:
                ws.write(row, 4, "N/A", fmt["cell"])
            row += 1
        row += 1

        # Covariate Effects
        cov_effects = results.get("covariate_effects", [])
        if cov_effects:
            ws.write(row, 0, "Covariate Effects", fmt["section_header"])
            row += 1
            cov_headers = ["Covariate", "Coefficient", "Std. Error", "t-value", "p-value", "CI Lower", "CI Upper"]
            for c, h in enumerate(cov_headers):
                ws.write(row, c, h, fmt["header"])
            row += 1
            for cov in cov_effects:
                ws.write(row, 0, str(cov.get("covariate", "")), fmt["cell"])
                ws.write(row, 1, cov.get("coefficient"), fmt["cell"])
                ws.write(row, 2, cov.get("std_err"), fmt["cell"])
                ws.write(row, 3, cov.get("t_value"), fmt["cell"])
                p_val = cov.get("p_value")
                if p_val is not None:
                    cell_fmt = fmt["sig_highlight"] if p_val < 0.05 else fmt["cell"]
                    ws.write(row, 4, p_val, cell_fmt)
                else:
                    ws.write(row, 4, "N/A", fmt["cell"])
                ws.write(row, 5, cov.get("ci_lower"), fmt["cell"])
                ws.write(row, 6, cov.get("ci_upper"), fmt["cell"])
                row += 1
            row += 1

        # Adjusted Means
        adj_means = results.get("adjusted_means", {})
        if adj_means:
            ws.write(row, 0, "Adjusted Means (Estimated Marginal Means)", fmt["section_header"])
            row += 1
            am_headers = ["Factor", "Level", "Adjusted Mean", "Raw Mean", "Raw SD", "n"]
            for c, h in enumerate(am_headers):
                ws.write(row, c, h, fmt["header"])
            row += 1
            for factor, levels in adj_means.items():
                for level, vals in levels.items():
                    ws.write(row, 0, str(factor), fmt["cell"])
                    ws.write(row, 1, str(level), fmt["cell"])
                    ws.write(row, 2, vals.get("adjusted_mean"), fmt["cell"])
                    ws.write(row, 3, vals.get("raw_mean"), fmt["cell"])
                    ws.write(row, 4, vals.get("raw_sd"), fmt["cell"])
                    ws.write(row, 5, vals.get("n"), fmt["cell"])
                    row += 1
            row += 1

        # Regression Slope Homogeneity Test
        slope_hom = results.get("slope_homogeneity", {})
        if slope_hom:
            ws.write(row, 0, "Homogeneity of Regression Slopes (ANCOVA Assumption)", fmt["section_header"])
            row += 1
            sh_headers = ["Interaction", "F", "p-value", "df", "Assumption Holds"]
            for c, h in enumerate(sh_headers):
                ws.write(row, c, h, fmt["header"])
            row += 1
            for interaction, vals in slope_hom.items():
                ws.write(row, 0, str(interaction), fmt["cell"])
                ws.write(row, 1, vals.get("F"), fmt["cell"])
                p_val = vals.get("p_value")
                if p_val is not None:
                    cell_fmt = fmt["sig_highlight"] if p_val < 0.05 else fmt["cell"]
                    ws.write(row, 2, p_val, cell_fmt)
                else:
                    ws.write(row, 2, "N/A", fmt["cell"])
                ws.write(row, 3, vals.get("df"), fmt["cell"])
                holds = vals.get("assumption_holds")
                ws.write(row, 4, "Yes" if holds else ("No - WARNING" if holds is False else "N/A"),
                         fmt["cell"] if holds else fmt["sig_highlight"])
                row += 1
            row += 1

        # Simple Slopes & Johnson-Neyman Analysis
        ssa = results.get("simple_slopes_analysis")
        if ssa:
            ws.write(row, 0, "Simple Slopes & Johnson-Neyman Analysis (Pick-a-Point)", fmt["section_header"])
            row += 1
            ws.write(row, 0, f"Moderator/Covariate: {ssa.get('covariate_name')}", fmt["cell"])
            row += 1
            ws.write(row, 0, f"Primary Factor: {ssa.get('factor_name')}", fmt["cell"])
            row += 2

            ws.write(row, 0, "Simple Slopes (Pick-a-Point)", fmt["section_header"])
            row += 1
            ss_headers = ["Covariate Level", "Covariate Value", "Effect Size / Coef", "Std. Error", "t-value", "p-value", "CI Lower", "CI Upper"]
            for c, h in enumerate(ss_headers):
                ws.write(row, c, h, fmt["header"])
            row += 1
            for slope in ssa.get("simple_slopes", []):
                ws.write(row, 0, str(slope.get("covariate_label", "")), fmt["cell"])
                ws.write(row, 1, slope.get("covariate_value"), fmt["cell"])
                ws.write(row, 2, slope.get("coefficient"), fmt["cell"])
                ws.write(row, 3, slope.get("std_err"), fmt["cell"])
                ws.write(row, 4, slope.get("t_value"), fmt["cell"])
                p_val = slope.get("p_value")
                if p_val is not None:
                    cell_fmt = fmt["sig_highlight"] if p_val < 0.05 else fmt["cell"]
                    ws.write(row, 5, p_val, cell_fmt)
                else:
                    ws.write(row, 5, "N/A", fmt["cell"])
                ws.write(row, 6, slope.get("ci_lower"), fmt["cell"])
                ws.write(row, 7, slope.get("ci_upper"), fmt["cell"])
                row += 1
            row += 2

            jn = ssa.get("johnson_neyman")
            if jn:
                ws.write(row, 0, "Johnson-Neyman Significance Regions", fmt["section_header"])
                row += 1
                roots = jn.get("roots", [])
                if len(roots) >= 2:
                    ws.write(row, 0, f"Critical Roots / Intervals", fmt["cell"])
                    ws.write(row, 1, f"[{roots[0]:.4f}, {roots[1]:.4f}]", fmt["cell"])
                    row += 1
                
                sig_regs = jn.get("significant_regions", [])
                if sig_regs:
                    ws.write(row, 0, "Significant Regions", fmt["cell"])
                    ws.write(row, 1, ", ".join(sig_regs), fmt["cell"])
                    row += 1
                else:
                    ws.write(row, 0, "Significant Regions", fmt["cell"])
                    ws.write(row, 1, "None in range", fmt["cell"])
                    row += 1
                ws.write(row, 0, "Covariate Range", fmt["cell"])
                ws.write(row, 1, f"[{jn.get('covariate_min', 0.0):.4f}, {jn.get('covariate_max', 0.0):.4f}]", fmt["cell"])
                row += 2

        # Model Fit
        ws.write(row, 0, "Model Fit", fmt["section_header"])
        row += 1
        ws.write(row, 0, "R-squared", fmt["cell"])
        ws.write(row, 1, results.get("r_squared"), fmt["cell"])
        row += 1
        ws.write(row, 0, "Adjusted R-squared", fmt["cell"])
        ws.write(row, 1, results.get("r_squared_adj"), fmt["cell"])
        row += 1
        ws.write(row, 0, "AIC", fmt["cell"])
        ws.write(row, 1, results.get("aic"), fmt["cell"])
        row += 1
        ws.write(row, 0, "N observations", fmt["cell"])
        ws.write(row, 1, results.get("n_observations"), fmt["cell"])
        row += 2
        ResultsExporter._write_data_health_section(ws, row, results, fmt)

    @staticmethod
    def _write_gee_glm_details_sheet(workbook, results, fmt):
        """GEE / GLM model parameter sheet — created when model_type is 'GEE' or 'GLM'."""
        ws = workbook.add_worksheet("GEE/GLM Details")
        ws.set_column('A:A', 40)
        ws.set_column('B:B', 35)
        row = 0

        ws.write(row, 0, "GEE / GLM Model Details", fmt["title"])
        row += 2

        _param_rows = [
            ("Model class",          results.get("model_class")),
            ("Model family",         results.get("model_family")),
            ("Link function",        results.get("model_link")),
            ("Covariance structure", results.get("cov_struct_used")),
            ("Covariance estimator", results.get("covariance_estimator")),
            ("Pearson phi",          results.get("pearson_phi")),
            ("Zero fraction",        results.get("zero_fraction")),
            ("Overdispersion ratio", results.get("overdispersion_ratio")),
        ]
        _fd = results.get("family_diagnostics") or {}
        if isinstance(_fd, dict) and _fd.get("selection_reason"):
            _param_rows.append(("Family selection reason", _fd["selection_reason"]))

        ws.write(row, 0, "Model Parameters", fmt["section_header"])
        row += 1
        for _label, _val in _param_rows:
            if _val is not None:
                ws.write(row, 0, _label, fmt["cell"])
                ws.write(row, 1,
                         f"{_val:.6f}" if isinstance(_val, float) else str(_val),
                         fmt["cell"])
                row += 1
        row += 1

        # Post-hoc path and multiplicity corrections
        pairwise = results.get("pairwise_comparisons") or []
        corrections = sorted({
            str(c.get("correction"))
            for c in pairwise
            if c.get("correction")
        })
        ws.write(row, 0, "Multiplicity corrections", fmt["cell"])
        ws.write(row, 1, ", ".join(corrections) if corrections else "None", fmt["cell"])
        row += 1

        _warnings = results.get("warnings") or []
        if _warnings:
            row += 1
            ws.write(row, 0, "Analysis Warnings", fmt["warning"])
            row += 1
            for _w in _warnings:
                ws.write(row, 0, str(_w), fmt["warning"])
                row += 1

    @staticmethod
    def _write_lmm_sheet(workbook, results, fmt):
        ws = workbook.add_worksheet("LMM Details")
        ws.set_column('A:A', 40)
        ws.set_column('B:G', 18)
        row = 0

        ws.write(row, 0, "Linear Mixed Model Details", fmt["title"])
        row += 2

        # Fixed Effects Table
        ws.write(row, 0, "Fixed Effects", fmt["section_header"])
        row += 1
        fe_headers = ["Parameter", "Coefficient", "Std. Error", "df", "t/z-value", "p-value", "CI Lower", "CI Upper"]
        for c, h in enumerate(fe_headers):
            ws.write(row, c, h, fmt["header"])
        row += 1
        for fe in results.get("fixed_effects_table", []):
            ws.write(row, 0, str(fe.get("parameter", "")), fmt["cell"])
            ws.write(row, 1, fe.get("coefficient"), fmt["cell"])
            ws.write(row, 2, fe.get("std_err"), fmt["cell"])
            df_val = fe.get("df")
            ws.write(row, 3, df_val if df_val is not None else "N/A", fmt["cell"])
            ws.write(row, 4, fe.get("z_value"), fmt["cell"])
            p_val = fe.get("p_value")
            if p_val is not None:
                cell_fmt = fmt["sig_highlight"] if p_val < 0.05 else fmt["cell"]
                ws.write(row, 5, p_val, cell_fmt)
            else:
                ws.write(row, 5, "N/A", fmt["cell"])
            ws.write(row, 6, fe.get("ci_lower"), fmt["cell"])
            ws.write(row, 7, fe.get("ci_upper"), fmt["cell"])
            row += 1
        row += 1

        # Random Effects
        ws.write(row, 0, "Random Effects", fmt["section_header"])
        row += 1
        ws.write(row, 0, "Random Intercept Variance", fmt["cell"])
        ws.write(row, 1, results.get("random_effects_variance"), fmt["cell"])
        row += 1
        ws.write(row, 0, "Residual Variance", fmt["cell"])
        ws.write(row, 1, results.get("residual_variance"), fmt["cell"])
        row += 1
        icc = results.get("icc")
        ws.write(row, 0, "ICC (Intraclass Correlation)", fmt["cell"])
        ws.write(row, 1, icc if icc is not None else "N/A", fmt["cell"])
        row += 2

        # Random Structure Selection & Degrees of Freedom
        ws.write(row, 0, "Random Structure & Degrees of Freedom Selection", fmt["section_header"])
        row += 1
        ws.write(row, 0, "Degrees of Freedom Method", fmt["cell"])
        ws.write(row, 1, results.get("df_method", "N/A"), fmt["cell"])
        row += 1
        ws.write(row, 0, "Random Structure Chosen", fmt["cell"])
        ws.write(row, 1, results.get("random_structure_chosen", "N/A"), fmt["cell"])
        row += 1
        if results.get("lrt_performed"):
            ws.write(row, 0, "Random Slope LRT Statistic", fmt["cell"])
            ws.write(row, 1, results.get("lrt_statistic"), fmt["cell"])
            row += 1
            lrt_p = results.get("lrt_p_value")
            ws.write(row, 0, "Random Slope LRT p-value", fmt["cell"])
            if lrt_p is not None:
                cell_fmt = fmt["sig_highlight"] if lrt_p < 0.05 else fmt["cell"]
                ws.write(row, 1, lrt_p, cell_fmt)
            else:
                ws.write(row, 1, "N/A", fmt["cell"])
            row += 1
        row += 2

        # Model Fit
        ws.write(row, 0, "Model Fit Statistics", fmt["section_header"])
        row += 1
        for key, label in [("aic", "AIC"), ("bic", "BIC"), ("log_likelihood", "Log-Likelihood")]:
            ws.write(row, 0, label, fmt["cell"])
            ws.write(row, 1, results.get(key), fmt["cell"])
            row += 1
        row += 1

        # Sample Info
        ws.write(row, 0, "Sample Information", fmt["section_header"])
        row += 1
        ws.write(row, 0, "Number of subjects", fmt["cell"])
        ws.write(row, 1, results.get("n_subjects"), fmt["cell"])
        row += 1
        ws.write(row, 0, "Total observations", fmt["cell"])
        ws.write(row, 1, results.get("n_observations"), fmt["cell"])
        row += 1
        ws.write(row, 0, "Converged", fmt["cell"])
        ws.write(row, 1, "Yes" if results.get("converged") else "No", fmt["cell"])
        row += 2

        # GEE/GLM Model Parameters (populated when model_class/family are present)
        _lmm_param_rows = [
            ("Model class",          results.get("model_class")),
            ("Model family",         results.get("model_family")),
            ("Link function",        results.get("model_link")),
            ("Covariance structure", results.get("cov_struct_used")),
            ("Covariance estimator", results.get("covariance_estimator")),
            ("Pearson phi",          results.get("pearson_phi")),
            ("Zero fraction",        results.get("zero_fraction")),
            ("Overdispersion ratio", results.get("overdispersion_ratio")),
        ]
        _fd = results.get("family_diagnostics") or {}
        if isinstance(_fd, dict) and _fd.get("selection_reason"):
            _lmm_param_rows.append(("Family selection reason", _fd["selection_reason"]))
        _has_params = any(v is not None for _, v in _lmm_param_rows)
        if _has_params:
            ws.write(row, 0, "Model Parameters", fmt["section_header"])
            row += 1
            for _label, _val in _lmm_param_rows:
                if _val is not None:
                    ws.write(row, 0, _label, fmt["cell"])
                    ws.write(row, 1,
                             f"{_val:.6f}" if isinstance(_val, float) else str(_val),
                             fmt["cell"])
                    row += 1
            row += 1

        # Interpretation note
        ws.write(row, 0, "Note", fmt["section_header"])
        row += 1
        ws.write(row, 0,
                 "Linear Mixed Models handle missing data via Maximum Likelihood estimation. "
                 "Unlike repeated-measures ANOVA, patients with missing visits are not dropped — "
                 "all available data points contribute to the model.",
                 fmt["cell"])
        row += 2
        ResultsExporter._write_data_health_section(ws, row, results, fmt)

    @staticmethod
    def _write_logistic_regression_sheet(workbook, results, fmt):
        ws = workbook.add_worksheet("Logistic Regression")
        ws.set_column('A:A', 35)
        ws.set_column('B:H', 16)
        row = 0

        ws.write(row, 0, "Logistic Regression Details", fmt["title"])
        row += 2

        # Odds Ratios
        or_table = results.get("odds_ratios", [])
        if or_table:
            ws.write(row, 0, "Odds Ratios", fmt["section_header"])
            row += 1
            or_headers = ["Parameter", "OR", "CI Lower", "CI Upper", "Coefficient", "Std. Error", "z-value", "p-value"]
            for c, h in enumerate(or_headers):
                ws.write(row, c, h, fmt["header"])
            row += 1
            for entry in or_table:
                ws.write(row, 0, str(entry.get("parameter", "")), fmt["cell"])
                ws.write(row, 1, entry.get("odds_ratio"), fmt["cell"])
                ws.write(row, 2, entry.get("ci_lower"), fmt["cell"])
                ws.write(row, 3, entry.get("ci_upper"), fmt["cell"])
                ws.write(row, 4, entry.get("coefficient"), fmt["cell"])
                ws.write(row, 5, entry.get("std_err"), fmt["cell"])
                ws.write(row, 6, entry.get("z_value"), fmt["cell"])
                p_val = entry.get("p_value")
                if p_val is not None:
                    cell_fmt = fmt["sig_highlight"] if p_val < 0.05 else fmt["cell"]
                    ws.write(row, 7, p_val, cell_fmt)
                else:
                    ws.write(row, 7, "N/A", fmt["cell"])
                row += 1
            row += 1

        # Hosmer-Lemeshow
        hl = results.get("hosmer_lemeshow", {})
        ws.write(row, 0, "Hosmer-Lemeshow Goodness-of-Fit Test", fmt["section_header"])
        row += 1
        ws.write(row, 0, "Chi-squared", fmt["cell"])
        ws.write(row, 1, hl.get("chi2"), fmt["cell"])
        row += 1
        ws.write(row, 0, "Degrees of freedom", fmt["cell"])
        ws.write(row, 1, hl.get("df"), fmt["cell"])
        row += 1
        hl_p = hl.get("p_value")
        ws.write(row, 0, "p-value", fmt["cell"])
        if hl_p is not None:
            cell_fmt = fmt["sig_highlight"] if hl_p < 0.05 else fmt["cell"]
            ws.write(row, 1, hl_p, cell_fmt)
        else:
            ws.write(row, 1, "N/A", fmt["cell"])
        row += 1
        ws.write(row, 0, "Interpretation", fmt["cell"])
        if hl_p is not None:
            ws.write(row, 1, "Good model fit (p > 0.05)" if hl_p > 0.05 else "Poor model fit (p < 0.05)", fmt["cell"])
        row += 2

        # Model Fit
        ws.write(row, 0, "Model Fit Statistics", fmt["section_header"])
        row += 1
        for key, label in [("pseudo_r_squared", "Pseudo R-squared (McFadden)"),
                           ("aic", "AIC"), ("bic", "BIC"),
                           ("log_likelihood", "Log-Likelihood"),
                           ("brier_score", "Brier Score"),
                           ("calibration_slope", "Calibration Slope"),
                           ("calibration_intercept", "Calibration Intercept"),
                           ("model_variant", "Model Variant")]:
            ws.write(row, 0, label, fmt["cell"])
            val = results.get(key)
            ws.write(row, 1, val if val is not None else "N/A", fmt["cell"])
            row += 1
        row += 1

        # AUC
        roc = results.get("roc_data", {})
        auc = roc.get("auc")
        ws.write(row, 0, "Discrimination (ROC Analysis)", fmt["section_header"])
        row += 1
        ws.write(row, 0, "AUC (Area Under ROC Curve)", fmt["cell"])
        ws.write(row, 1, auc if auc is not None else "N/A", fmt["cell"])
        row += 1
        if auc is not None:
            if auc >= 0.9:
                interp = "Excellent discrimination"
            elif auc >= 0.8:
                interp = "Good discrimination"
            elif auc >= 0.7:
                interp = "Acceptable discrimination"
            else:
                interp = "Poor discrimination"
            ws.write(row, 0, "Interpretation", fmt["cell"])
            ws.write(row, 1, interp, fmt["cell"])
        row += 1

        ws.write(row, 0, "N observations", fmt["cell"])
        ws.write(row, 1, results.get("n_observations"), fmt["cell"])
        row += 2
        ResultsExporter._write_data_health_section(ws, row, results, fmt)

    # ------------------------------------------------------------------
    # Beta Regression sheet
    # ------------------------------------------------------------------

    @staticmethod
    def _write_beta_regression_sheet(workbook, results, fmt):
        ws = workbook.add_worksheet("Beta Regression")
        ws.set_column('A:A', 38)
        ws.set_column('B:H', 16)
        row = 0

        ws.write(row, 0, "Beta Regression Details", fmt["title"])
        row += 1
        detection_note = results.get("detection_note")
        if detection_note:
            ws.write(row, 0, detection_note, fmt["italic_grey"])
        row += 2

        # Coefficients table
        coef_table = results.get("coefficients", [])
        if coef_table:
            ws.write(row, 0, "Coefficients (logit scale)", fmt["section_header"])
            row += 1
            headers = ["Parameter", "Coefficient", "CI Lower (95%)", "CI Upper (95%)", "Std. Error", "z-value", "p-value"]
            for c, h in enumerate(headers):
                ws.write(row, c, h, fmt["header"])
            row += 1
            for entry in coef_table:
                ws.write(row, 0, str(entry.get("parameter", "")), fmt["cell"])
                ws.write(row, 1, entry.get("coefficient"), fmt["cell"])
                ws.write(row, 2, entry.get("ci_lower"), fmt["cell"])
                ws.write(row, 3, entry.get("ci_upper"), fmt["cell"])
                ws.write(row, 4, entry.get("std_err"), fmt["cell"])
                ws.write(row, 5, entry.get("z_value"), fmt["cell"])
                p_val = entry.get("p_value")
                if p_val is not None:
                    cell_fmt = fmt["sig_highlight"] if p_val < 0.05 else fmt["cell"]
                    ws.write(row, 6, p_val, cell_fmt)
                else:
                    ws.write(row, 6, "N/A", fmt["cell"])
                row += 1
            row += 1

        # Model fit statistics
        ws.write(row, 0, "Model Fit Statistics", fmt["section_header"])
        row += 1
        phi = results.get("phi")
        for key, label in [
            ("pseudo_r_squared", "Pseudo R-squared (McFadden)"),
            ("aic", "AIC"),
            ("bic", "BIC"),
            ("log_likelihood", "Log-Likelihood"),
        ]:
            ws.write(row, 0, label, fmt["cell"])
            val = results.get(key)
            ws.write(row, 1, val if val is not None else "N/A", fmt["cell"])
            row += 1
        ws.write(row, 0, "Dispersion parameter (phi)", fmt["cell"])
        ws.write(row, 1, phi if phi is not None else "N/A", fmt["cell"])
        row += 1
        ws.write(row, 0, "Interpretation of phi", fmt["cell"])
        if phi is not None:
            ws.write(row, 1, "Higher phi = lower variance around the mean (more precise fit)", fmt["cell"])
        row += 2

        ws.write(row, 0, "N observations", fmt["cell"])
        ws.write(row, 1, results.get("n_observations"), fmt["cell"])
        row += 2
        ResultsExporter._write_data_health_section(ws, row, results, fmt)

    # ------------------------------------------------------------------
    # Correlation sheet
    # ------------------------------------------------------------------

    @staticmethod
    def _write_correlation_sheet(workbook, results, fmt):
        ws = workbook.add_worksheet("Korrelation")
        ws.set_column('A:A', 35)
        ws.set_column('B:D', 20)
        row = 0

        ws.write(row, 0, "Correlation Analysis", fmt["title"])
        row += 2

        fields = [
            ("Method", results.get("method", "N/A")),
            ("r (Correlation Coefficient)", results.get("r")),
            ("p-value", results.get("p_value")),
            ("95% CI Lower", results.get("ci_lower")),
            ("95% CI Upper", results.get("ci_upper")),
            ("n (Pairs)", results.get("n")),
            ("Interpretation", results.get("interpretation", "N/A")),
            ("X-Variable", results.get("x_variable", "N/A")),
            ("Y-Variable (Outcome)", results.get("y_variable", "N/A")),
        ]
        for label, val in fields:
            ws.write(row, 0, label, fmt["cell"])
            p_val = results.get("p_value")
            if label == "p-value" and p_val is not None:
                cell_fmt = fmt["sig_highlight"] if p_val < 0.05 else fmt["cell"]
                ws.write(row, 1, val if val is not None else "N/A", cell_fmt)
            else:
                ws.write(row, 1, val if val is not None else "N/A", fmt["cell"])
            row += 1

        row += 1

        # Normality check (only present when method='auto')
        nc = results.get("normality_check")
        if nc:
            ws.write(row, 0, "Normality Check (Shapiro-Wilk, auto method selection)", fmt["section_header"])
            row += 1
            ws.write(row, 0, "Variable", fmt["header"])
            ws.write(row, 1, "W Statistic", fmt["header"])
            ws.write(row, 2, "p-value", fmt["header"])
            ws.write(row, 3, "Normal (p > 0.05)?", fmt["header"])
            row += 1
            x_var = results.get("x_variable", "X")
            y_var = results.get("y_variable", "Y")
            for var_name in (x_var, y_var):
                entry = nc.get(var_name, {})
                ws.write(row, 0, var_name, fmt["cell"])
                ws.write(row, 1, entry.get("statistic", "N/A"), fmt["cell"])
                p = entry.get("p_value")
                p_fmt = fmt["sig_highlight"] if p is not None and p < 0.05 else fmt["cell"]
                ws.write(row, 2, p if p is not None else "N/A", p_fmt)
                ws.write(row, 3, "Yes" if entry.get("normal") else "No", fmt["cell"])
                row += 1
            conclusion = "Pearson" if nc.get("both_normal") else "Spearman"
            ws.write(row, 0, f"→ Both normal: {'Yes' if nc.get('both_normal') else 'No'} — {conclusion} selected", fmt["cell"])
            row += 2

        ResultsExporter._write_data_health_section(ws, row, results, fmt)

    # ------------------------------------------------------------------
    # Linear Regression sheet
    # ------------------------------------------------------------------

    @staticmethod
    def _write_linear_regression_sheet(workbook, results, fmt):
        ws = workbook.add_worksheet("Linear Regression")
        ws.set_column('A:A', 35)
        ws.set_column('B:H', 16)
        row = 0

        ws.write(row, 0, "Linear Regression (OLS)", fmt["title"])
        row += 2

        # Model summary
        ws.write(row, 0, "Model Summary", fmt["section_header"])
        row += 1
        summary_fields = [
            ("R² (Coefficient of Determination)", results.get("r_squared")),
            ("R² (adjusted)", results.get("r_squared_adj")),
            ("F-statistic", results.get("f_statistic")),
            ("F p-value", results.get("f_p_value")),
            ("AIC", results.get("aic")),
            ("BIC", results.get("bic")),
            ("Covariance Type", results.get("cov_type", "nonrobust")),
            ("N Observations", results.get("n_observations")),
        ]
        for label, val in summary_fields:
            ws.write(row, 0, label, fmt["cell"])
            ws.write(row, 1, val if val is not None else "N/A", fmt["cell"])
            row += 1
        row += 1

        # Coefficient table
        ws.write(row, 0, "Coefficient Table", fmt["section_header"])
        row += 1
        headers = ["Parameter", "Beta", "Std. Error", "t-value", "p-value", "CI Lower", "CI Upper"]
        for c, h in enumerate(headers):
            ws.write(row, c, h, fmt["header"])
        row += 1
        for coef in results.get("coefficient_table", []):
            ws.write(row, 0, str(coef.get("parameter", "")), fmt["cell"])
            ws.write(row, 1, coef.get("coefficient"), fmt["cell"])
            ws.write(row, 2, coef.get("std_err"), fmt["cell"])
            ws.write(row, 3, coef.get("t_value"), fmt["cell"])
            p_val = coef.get("p_value")
            if p_val is not None:
                cell_fmt = fmt["sig_highlight"] if p_val < 0.05 else fmt["cell"]
                ws.write(row, 4, p_val, cell_fmt)
            else:
                ws.write(row, 4, "N/A", fmt["cell"])
            ws.write(row, 5, coef.get("ci_lower"), fmt["cell"])
            ws.write(row, 6, coef.get("ci_upper"), fmt["cell"])
            row += 1
        row += 1

        # Diagnostics
        ws.write(row, 0, "Assumption Diagnostics", fmt["section_header"])
        row += 1
        diag_headers = ["Test", "Statistic", "p-value", "Assumption met?"]
        for c, h in enumerate(diag_headers):
            ws.write(row, c, h, fmt["header"])
        row += 1
        diag_order = [
            ("normality", "Shapiro-Wilk (Residuals)"),
            ("homoscedasticity", "Breusch-Pagan"),
            ("linearity", "Ramsey RESET"),
        ]
        for key, label in diag_order:
            d = results.get("diagnostics", {}).get(key, {})
            ws.write(row, 0, d.get("test", label), fmt["cell"])
            stat_val = d.get("statistic")
            ws.write(row, 1, stat_val if stat_val is not None else "N/A", fmt["cell"])
            p_d = d.get("p_value")
            if p_d is not None:
                cell_fmt = fmt["sig_highlight"] if not d.get("assumption_holds", True) else fmt["cell"]
                ws.write(row, 2, p_d, cell_fmt)
            else:
                ws.write(row, 2, d.get("error", "N/A"), fmt["cell"])
            holds = d.get("assumption_holds")
            ws.write(row, 3, ("Yes" if holds else "No — check") if holds is not None else "N/A", fmt["cell"])
            row += 1
        row += 1

        ResultsExporter._write_data_health_section(ws, row, results, fmt)

    # ------------------------------------------------------------------
    # Exploratory Correlation Matrix sheet
    # ------------------------------------------------------------------

    @staticmethod
    def _write_correlation_matrix_sheet(workbook, results, fmt):
        cols = results.get("variables", [])
        if not cols:
            return

        def _write_matrix(ws, start_row, label, matrix_dict, cols, highlight_sig=False, p_corrected=None):
            ws.write(start_row, 0, label, fmt["section_header"])
            start_row += 1
            for j, c in enumerate(cols):
                ws.write(start_row, j + 1, c, fmt["header"])
            start_row += 1
            for i, ri in enumerate(cols):
                ws.write(start_row + i, 0, ri, fmt["header"])
                for j, cj in enumerate(cols):
                    val = matrix_dict.get(ri, {}).get(cj)
                    if highlight_sig and p_corrected is not None:
                        p_val = p_corrected.get(ri, {}).get(cj)
                        cell_fmt = fmt["sig_highlight"] if (p_val is not None and p_val < 0.05) else fmt["cell"]
                    else:
                        cell_fmt = fmt["cell"]
                    ws.write(start_row + i, j + 1, val if val is not None else "N/A", cell_fmt)
            return start_row + len(cols) + 2

        ws = workbook.add_worksheet("Correlation Matrix")
        ws.set_column('A:A', 38)
        for j in range(len(cols) + 1):
            ws.set_column(j + 1, j + 1, 14)
        row = 0

        ws.write(row, 0, "Exploratory Correlation Matrix", fmt["title"])
        row += 1
        ws.write(row, 0,
                 f"Method: {results.get('method', 'N/A')} | "
                 f"Correction: {results.get('correction', 'none')} | "
                 f"Missing: {'pairwise' if results.get('pairwise_deletion') else 'listwise'}",
                 fmt["cell"])
        row += 2

        row = _write_matrix(ws, row, "r-Matrix (Correlation Coefficients)",
                            results.get("r_matrix", {}), cols,
                            highlight_sig=True, p_corrected=results.get("p_corrected_matrix"))
        row = _write_matrix(ws, row, "p-values (corrected)",
                            results.get("p_corrected_matrix", {}), cols, highlight_sig=True,
                            p_corrected=results.get("p_corrected_matrix"))
        row = _write_matrix(ws, row, "n per variable pair",
                            results.get("n_matrix", {}), cols)

        strata = results.get("strata", {})
        for grp, mats in strata.items():
            row = _write_matrix(ws, row, f"r-Matrix — Group: {grp}",
                                mats.get("r_matrix", {}), cols,
                                highlight_sig=True, p_corrected=mats.get("p_corrected_matrix"))
            row = _write_matrix(ws, row, f"p (corrected) — Group: {grp}",
                                mats.get("p_corrected_matrix", {}), cols, highlight_sig=True,
                                p_corrected=mats.get("p_corrected_matrix"))
