import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import t

from core.lazy_imports import get_seaborn

import logging
logger = logging.getLogger(__name__)

OUTLIER_IMPORTS_AVAILABLE = True
try:
    import matplotlib.pyplot as plt
except ImportError:
    OUTLIER_IMPORTS_AVAILABLE = False

class OutlierDetector:
    """
    Detect outliers in grouped data using Grubbs' Test or Dixon's Q-Test.
    Loads an Excel table with columns ['Group', 'Value'], 
    converts all values (German decimal numbers with comma) to float,
    performs Grubbs or Dixon tests iteratively or once for each group separately
    and marks found outliers.
    """

    def __init__(self, df, group_col, value_col):
        """
        Initializes the OutlierDetector with an already loaded DataFrame.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with the data
        group_col : str
            Name of the group column
        value_col : str
            Name of the values column
        """
        logger.debug(f"DEBUG: Initializing OutlierDetector with columns: group_col={group_col}, value_col={value_col}")
        logger.debug(f"DEBUG: DataFrame shape: {df.shape}, columns: {df.columns.tolist()}")
        
        if not OUTLIER_IMPORTS_AVAILABLE:
            raise ImportError("Required packages for outlier detection are not available. "
                              "Please install: pip install outliers pingouin openpyxl")
        
        # Create copy of DataFrame
        self.df = df.copy()
        self.group_col = group_col
        self.value_col = value_col
        self.active_test = None  # Track which test was run
        self.debug_log = []  # Initialize debug log

        # Ensure columns are present
        if group_col not in self.df.columns or value_col not in self.df.columns:
            raise ValueError(f"Columns '{group_col}' and '{value_col}' must be present in the DataFrame.")

        # Add initialization info to log
        self.debug_log.append("*** OUTLIER DETECTION INITIALIZATION ***")
        self.debug_log.append(f"DataFrame shape: {df.shape}")
        self.debug_log.append(f"Group column: {group_col}")
        self.debug_log.append(f"Value column: {value_col}")
        self.debug_log.append(f"Available columns: {df.columns.tolist()}")

        # Convert values column to float (if necessary)
        self._convert_values_to_float()
        
        # Show group statistics after initialization
        self.debug_log.append("\n=== GROUP STATISTICS ===")
        self.debug_log.append(f"Number of groups in data: {self.df[group_col].nunique()}")
        
        for group_name, group_data in self.df.groupby(group_col):
            group_stats = self._calculate_group_statistics(group_name, group_data)
            self.debug_log.extend(group_stats)

    def _convert_values_to_float(self):
        """
        Converts the values column to float.
        Handles German decimal separators (comma instead of period).
        """
        try:
            self.debug_log.append("\n=== VALUE CONVERSION ===")
            self.debug_log.append(f"Converting column '{self.value_col}' to float")
            
            # Check if the column exists
            if self.value_col not in self.df.columns:
                error_msg = f"Column '{self.value_col}' not found in DataFrame. Available columns: {list(self.df.columns)}"
                self.debug_log.append(f"ERROR: {error_msg}")
                raise ValueError(error_msg)
            
            self.debug_log.append(f"Sample values before conversion: {self.df[self.value_col].head(3).tolist()}")
            self.debug_log.append(f"Data type before conversion: {self.df[self.value_col].dtype}")
            
            # Check if column is already numeric
            if pd.api.types.is_numeric_dtype(self.df[self.value_col]):
                self.debug_log.append("Column is already numeric, no conversion needed")
                return
            
            # Convert German decimal numbers
            self.df[self.value_col] = (
                self.df[self.value_col]
                    .astype(str)
                    .str.replace('.', '', regex=False)   # Remove thousand separators
                    .str.replace(',', '.', regex=False)  # Comma → Period
                    .astype(float)
            )
            
            self.debug_log.append(f"Sample values after conversion: {self.df[self.value_col].head(3).tolist()}")
            self.debug_log.append(f"Data type after conversion: {self.df[self.value_col].dtype}")
            self.debug_log.append(f"Value range after conversion: {self.df[self.value_col].min():.4f} to {self.df[self.value_col].max():.4f}")
        
        except Exception as e:
            error_message = f"Conversion failed: {str(e)}"
            self.debug_log.append(f"ERROR: {error_message}")
            raise ValueError(f"Error converting values column '{self.value_col}' to float: {str(e)}")
        
    def _calculate_group_statistics(self, group_name, group_data):
        """Calculate basic statistics for a group and return as log entries."""
        stats = []
        values = group_data[self.value_col].dropna()
        stats.append(f"\n--- Group '{group_name}' Statistics ---")
        stats.append(f"  Count: {len(values)}")
        
        if len(values) > 0:
            stats.append(f"  Min: {values.min():.4f}")
            stats.append(f"  Max: {values.max():.4f}")
            stats.append(f"  Mean: {values.mean():.4f}")
            stats.append(f"  Median: {values.median():.4f}")
            stats.append(f"  Std Dev: {values.std():.4f}")
        else:
            stats.append("  No valid values in this group")
        
        return stats
    
    @staticmethod
    def _grubbs_iterative(vals: np.ndarray,
                           alpha: float = 0.05,
                           two_sided: bool = True):
        """
        Iteratively remove the most extreme point (if G_obs > G_crit),
        repeat until no further outliers are detected.
        Returns list of relative indices into the original vals array.
        """
        vals = np.asarray(vals, dtype=float)
        # track the original positions
        idxs = np.arange(vals.size)
        out_rel_idxs = []
        
        while vals.size >= 3:
            G_obs, G_crit, rel_idx = OutlierDetector._grubbs_statistic(
                vals, alpha=alpha, two_sided=two_sided
            )
            if G_obs <= G_crit:
                break
            # record and remove
            out_rel_idxs.append(int(idxs[rel_idx]))
            mask = np.ones(vals.size, dtype=bool)
            mask[rel_idx] = False
            vals = vals[mask]
            idxs = idxs[mask]
        
        return out_rel_idxs

    @staticmethod
    def _grubbs_statistic(x: np.ndarray,
                          alpha: float = 0.05,
                          two_sided: bool = True):
        """
        Compute Grubbs' G statistic and critical value for array x.
        Returns (G_obs, G_crit, outlier_rel_index).
        """
        n = x.size
        if n < 3:
            raise ValueError("Grubbs' test requires at least 3 values")
        
        mu = x.mean()
        s  = x.std(ddof=1)
        # maximum absolute deviation
        diffs = np.abs(x - mu)
        rel_idx = int(np.argmax(diffs))
        G_obs = diffs[rel_idx] / s
        
        # two‐sided splits alpha into 2 tails
        tail = 2 if two_sided else 1
        # student-t critical value
        t_crit = t.ppf(1 - alpha/(tail * n), df=n-2)
        # critical G
        G_crit = ((n - 1) / np.sqrt(n)
                  * np.sqrt(t_crit**2 / ( (n - 2) + t_crit**2 )))
        
        return G_obs, G_crit, rel_idx

    def run_grubbs(self, alpha: float = 0.05, iterate: bool = True):
        self.debug_log.append("\n=== GRUBBS OUTLIER DETECTION ===")
        self.debug_log.append(f"alpha={alpha}, iterate={iterate}")

        self.df["Grubbs_Outlier"] = False
        self.active_test = "Grubbs"
        total_out = 0

        for gname, gdf in self.df.groupby(self.group_col):
            self.debug_log.append(f"\n--- Group '{gname}' ({len(gdf)}) ---")

            # original indices and clean values
            idxs = gdf.index.values
            vals = gdf[self.value_col].astype(float).values
            valid = ~np.isnan(vals)
            idxs_valid = idxs[valid]
            vals_valid = vals[valid]

            if len(vals_valid) < 3:
                self.debug_log.append("  n < 3 → skipping Grubbs")
                continue

            # choose iterative vs. single removal
            if iterate:
                rel_outs = OutlierDetector._grubbs_iterative(
                    vals_valid, alpha=alpha, two_sided=True
                )
            else:
                G_obs, G_crit, rel_idx = OutlierDetector._grubbs_statistic(
                    vals_valid, alpha=alpha, two_sided=True
                )
                rel_outs = [rel_idx] if G_obs > G_crit else []

            if not rel_outs:
                self.debug_log.append("  No outliers detected")
            else:
                # map back to global indices
                global_outs = idxs_valid[rel_outs].tolist()
                for gi in global_outs:
                    val = self.df.at[gi, self.value_col]
                    self.debug_log.append(f"  Outlier at idx={gi}, value={val:.4f}")
                self.df.loc[global_outs, "Grubbs_Outlier"] = True
                total_out += len(global_outs)
                self.debug_log.append(f"  ==> {len(global_outs)} outliers in group")

        self.debug_log.append("\n=== GRUBBS SUMMARY ===")
        self.debug_log.append(f"Total outliers: {total_out}")

    def run_grubbs_test(self, alpha: float = 0.05, iterate: bool = True):
        """
        Alias for run_grubbs method for compatibility.
        
        Parameters:
        -----------
        alpha : float
            Significance level for the test (default: 0.05)
        iterate : bool
            Whether to iteratively remove outliers (default: True)
        """
        return self.run_grubbs(alpha=alpha, iterate=iterate)

    def run_mod_z_score(self, threshold=3.5, iterate=False):
        """
        Performs Modified Z-Score outlier detection for each group.
        
        Parameters:
        -----------
        threshold : float
            Threshold for modified Z-scores to be considered outliers (default: 3.5)
        iterate : bool
            Whether to iteratively remove outliers and recompute scores
        """
        self.debug_log.append("\n=== MODIFIED Z-SCORE OUTLIER DETECTION EXECUTION ===")
        self.debug_log.append(f"Test parameters: threshold={threshold}, iterate={iterate}")
        self.debug_log.append("Test principle: Modified Z-Score uses median absolute deviation (MAD) for robustness against outliers")
        
        # Create the Modified Z-Score column and mark this test as active
        self.df['ModZ_Outlier'] = False
        self.active_test = 'ModZ'
        
        total_outliers = 0
        
        for group_name, group_df in self.df.groupby(self.group_col):
            self.debug_log.append(f"\n--- Processing group '{group_name}' ---")
            self.debug_log.append(f"Group size: {len(group_df)} observations")
            
            indices = group_df.index.tolist()
            values = group_df[self.value_col].values.copy()
            
            # Filter out NaN values
            valid_mask = ~np.isnan(values)
            valid_values = values[valid_mask]
            valid_indices = [idx for mask, idx in zip(valid_mask, indices) if mask]
            
            if len(valid_values) == 0:
                self.debug_log.append(f"Group '{group_name}' has no valid values (all NaN), skipping")
                continue
                
            outlier_indices = []
            
            # Log basic group statistics
            mean_val = np.mean(valid_values)
            median_val = np.median(valid_values)
            self.debug_log.append(f"Group statistics: min={np.min(valid_values):.3f}, max={np.max(valid_values):.3f}, mean={mean_val:.3f}, median={median_val:.3f}")
            
            def detect_single_round(vals, indices_list):
                if len(vals) <= 1:
                    return [], "No outliers found (too few values for MAD calculation)"
                    
                # Calculate median and MAD
                median = np.median(vals)
                mad = np.median(np.abs(vals - median))
                
                # Avoid division by zero
                if mad == 0:
                    return [], "MAD = 0, cannot compute modified Z-scores"
                    
                # Calculate modified Z-scores
                mod_z_scores = 0.6745 * (vals - median) / mad
                
                # Find outliers
                outlier_mask = np.abs(mod_z_scores) > threshold
                outlier_idx = [idx for mask, idx in zip(outlier_mask, indices_list) if mask]
                
                # Log this round
                self.debug_log.append("  Modified Z-Score Analysis:")
                self.debug_log.append(f"    Median: {median:.4f}")
                self.debug_log.append(f"    MAD: {mad:.4f}")
                self.debug_log.append(f"    Threshold: {threshold} (absolute value)")
                
                if len(outlier_idx) > 0:
                    self.debug_log.append(f"    {len(outlier_idx)} outliers found in this round")
                    return outlier_idx, None
                else:
                    self.debug_log.append("    No outliers found in this round")
                    return [], "No outliers found (all modified Z-scores within threshold)"
            
            # Iterative outlier detection
            vals_copy = valid_values.copy()
            idx_copy = valid_indices.copy() 
            round_count = 0
            
            while True:
                round_count += 1
                self.debug_log.append(f"  Testing round {round_count} with {len(vals_copy)} values")
                
                round_outliers, stop_message = detect_single_round(vals_copy, idx_copy)
                
                if round_outliers:
                    outlier_indices.extend(round_outliers)
                    if iterate:
                        # Remove outliers for next iteration
                        keep_mask = ~np.isin(idx_copy, round_outliers)
                        vals_copy = vals_copy[keep_mask]
                        idx_copy = [idx for keep, idx in zip(keep_mask, idx_copy) if keep]
                    else:
                        self.debug_log.append("  No more iterations requested, terminating test")
                        break
                else:
                    self.debug_log.append(f"  {stop_message if stop_message else 'No more outliers found'}, terminating test")
                    break
                    
                if len(vals_copy) <= 1:
                    self.debug_log.append("  Too few values left for MAD calculation, terminating test")
                    break
            
            # Mark all found indices in main DataFrame
            if outlier_indices:
                self.df.loc[outlier_indices, 'ModZ_Outlier'] = True
                total_outliers += len(outlier_indices)
                self.debug_log.append(f"  RESULT: {len(outlier_indices)} outliers found in group '{group_name}'")
            else:
                self.debug_log.append(f"  RESULT: No outliers found in group '{group_name}'")
        
        # Final summary
        self.debug_log.append("\n=== MODIFIED Z-SCORE TEST SUMMARY ===")
        self.debug_log.append(f"Total outliers detected across all groups: {total_outliers}")
        self.debug_log.append("Test completed successfully")


    def save_results(self, output_path):
        """Write the outlier-detection results as a self-contained HTML report."""
        from export.outlier_html_exporter import export_single
        output_path = os.path.abspath(output_path)
        if output_path.lower().endswith((".xlsx", ".xls")):
            output_path = os.path.splitext(output_path)[0] + ".html"
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        export_single(self, output_path)
        self.debug_log.append(f"Results saved to: {output_path}")
        logger.info(f"Outlier report saved to: {output_path}")
        return output_path

    def get_summary(self):
        """
        Creates a summary of found outliers.
        """
        self.debug_log.append("\n=== CREATING SUMMARY ===")
        
        total_rows = len(self.df)
        grubbs_outliers = self.df['Grubbs_Outlier'].sum() if 'Grubbs_Outlier' in self.df.columns else 0
        modz_outliers = self.df['ModZ_Outlier'].sum() if 'ModZ_Outlier' in self.df.columns else 0
        
        self.debug_log.append(f"Total rows: {total_rows}")
        self.debug_log.append(f"Grubbs outliers: {grubbs_outliers}")
        self.debug_log.append(f"Modified Z-Score outliers: {modz_outliers}")
        
        any_outliers = 0
        
        # Check which columns exist before combining
        if 'Grubbs_Outlier' in self.df.columns and 'ModZ_Outlier' in self.df.columns:
            any_outliers = (self.df['Grubbs_Outlier'] | self.df['ModZ_Outlier']).sum()
        elif 'Grubbs_Outlier' in self.df.columns:
            any_outliers = grubbs_outliers
        elif 'ModZ_Outlier' in self.df.columns:
            any_outliers = modz_outliers
        
        self.debug_log.append(f"Combined outliers: {any_outliers}")
        
        # Outliers per group
        group_summary = {}
        for group_name, group_df in self.df.groupby(self.group_col):
            group_total = len(group_df)
            group_grubbs = group_df['Grubbs_Outlier'].sum() if 'Grubbs_Outlier' in group_df.columns else 0
            group_modz = group_df['ModZ_Outlier'].sum() if 'ModZ_Outlier' in group_df.columns else 0
            
            group_any = 0
            if 'Grubbs_Outlier' in group_df.columns and 'ModZ_Outlier' in group_df.columns:
                group_any = (group_df['Grubbs_Outlier'] | group_df['ModZ_Outlier']).sum()
            elif 'Grubbs_Outlier' in group_df.columns:
                group_any = group_grubbs
            elif 'ModZ_Outlier' in group_df.columns:
                group_any = group_modz
            
            self.debug_log.append(f"Group '{group_name}': total={group_total}, outliers={group_any}")
            
            group_summary[group_name] = {
                'total': group_total,
                'grubbs_outliers': group_grubbs,
                'modz_outliers': group_modz,
                'any_outliers': group_any
            }
        
        summary = {
            'total_rows': total_rows,
            'grubbs_outliers': grubbs_outliers,
            'modz_outliers': modz_outliers,
            'any_outliers': any_outliers,
            'groups': group_summary
        }
        
        self.debug_log.append(f"Summary completed: {summary}")
        return summary

    @staticmethod
    def run_multi_dataset_outlier_detection(df, group_col, dataset_columns, alpha=0.05, 
                                            iterate=False, run_grubbs=True, run_modz=True, 
                                            grubbs_alpha=0.05, modz_threshold=3.5,
                                            output_path="multi_dataset_outlier_results.html"):
        """
        Run outlier detection on multiple datasets (columns) and create a combined HTML report.
        """
        logger.debug(f"DEBUG: Current working directory before export: {os.getcwd()}")
        logger.debug("DEBUG: Starting multi-dataset outlier detection")
        logger.debug(f"DEBUG: Datasets to analyze: {dataset_columns}")
        logger.debug(f"DEBUG: Group column: {group_col}")
        
        all_results = {}
        failed_datasets = {}

        if output_path.lower().endswith((".xlsx", ".xls")):
            output_path = os.path.splitext(output_path)[0] + ".html"

        try:
            # Analyze each dataset
            for i, dataset_col in enumerate(dataset_columns):
                logger.debug(f"DEBUG: Analyzing dataset {i+1}/{len(dataset_columns)}: {dataset_col}")
                
                try:
                    # Create detector for this dataset
                    detector = OutlierDetector(
                        df=df.copy(),
                        group_col=group_col,
                        value_col=dataset_col
                    )
                    
                    # Run requested tests
                    if run_grubbs:
                        logger.debug(f"DEBUG: Running Grubbs test for {dataset_col}")
                        detector.run_grubbs(alpha=grubbs_alpha, iterate=iterate)
                    
                    if run_modz:
                        logger.debug(f"DEBUG: Running Modified Z-Score test for {dataset_col}")
                        detector.run_mod_z_score(threshold=modz_threshold, iterate=iterate)
                    
                    # Store results
                    all_results[dataset_col] = {
                        'detector': detector,
                        'summary': detector.get_summary()
                    }
                    
                    logger.debug(f"DEBUG: Successfully analyzed {dataset_col}")
                    
                except Exception as e:
                    error_msg = f"Error analyzing {dataset_col}: {str(e)}"
                    failed_datasets[dataset_col] = error_msg
                    logger.debug(f"DEBUG: ERROR: {error_msg}")
                        
            # Write the combined HTML report
            from export.outlier_html_exporter import export_multi
            export_multi(all_results, failed_datasets, dataset_columns, output_path)
            logger.debug(f"DEBUG: Combined results saved to: {output_path}")

            # Return summary
            return {
                "type": "multi_dataset_outlier_detection",
                "successful_datasets": list(all_results.keys()),
                "failed_datasets": failed_datasets,
                "total_datasets": len(dataset_columns),
                "output_file": output_path,
                "summary": {
                    "success_count": len(all_results),
                    "failure_count": len(failed_datasets),
                    "success_rate": f"{len(all_results)/len(dataset_columns)*100:.1f}%"
                }
            }
            
        except Exception as e:
            error_msg = f"Critical error in multi-dataset analysis: {str(e)}"
            logger.debug(f"DEBUG: CRITICAL ERROR: {error_msg}")
            raise RuntimeError(error_msg)
