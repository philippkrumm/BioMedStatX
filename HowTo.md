## Statistical Analyzer How-To Guide

Welcome to the Statistical Analyzer application! This guide walks you through everything you need to know to get started with the standalone `.exe` version of the program, from launching the app to exporting your results to Excel.

---

## Table of Contents

1. [Launching the Application](#launching-the-application)
2. [Main Workflow Overview](#main-workflow-overview)
3. [Key Functions & Features](#key-functions--features)
   - Data Import
   - Data Transformation
   - Assumption Checks & Test Selection
   - Statistical Tests
   - Post-Hoc Analyses
   - Decision Tree Visualization
4. [Dialog & Configuration Windows](#dialog--configuration-windows)
5. [Exporting Results to Excel](#exporting-results-to-excel)
6. [Visualization & Plots](#visualization--plots)
7. [Tips & Troubleshooting](#tips--troubleshooting)

---

## Launching the Application

1. Ensure you have the latest `.exe` file for Statistical Analyzer.
2. Double-click the `.exe` to launch the GUI. No Python or PowerShell commands are required.
3. The main window will appear with menus: **File**, **Analysis**, and **Help**.

---

## Main Workflow Overview

1. **Import Data**: Use **File → Import** to load CSV or Excel files.
2. **Select Variables & Groups**:
   - Choose measurement columns and grouping variables via selection dialogs.
3. **Configure Tests**:
   - Pick the type of analysis: two-group comparison, multi-group ANOVA, mixed/repeated-measures, or two-way ANOVA.
4. **Run Analysis**:
   - The program automatically checks normality and variance (Shapiro-Wilk, Levene’s tests).
   - If assumptions fail, you’ll be prompted to apply a transformation (log, Box-Cox) or switch to non-parametric methods.
5. **View & Customize Plots**:
   - After tests, inspect generated plots; customize labels, colors, comparisons, and more.
6. **Export**:
   - Use **File → Export Results** to save statistical summaries, test details, post-hoc tables, and plots into an Excel workbook.
7. **Decision Tree**:
   - Visualize the testing logic path via **Analysis → Show Decision Tree**. A PNG is generated, highlighting which branches were taken.

---

## Key Functions & Features

### Data Import
- **`DataImporter.import_data(path, sheet=None, group_cols=[], value_cols=[])`**
  - Loads CSV/Excel, combines multiple sheets or files, and returns:
    - A dict of sample arrays by group
    - A pandas DataFrame in long format
- **Helper**: `dict_to_long_format(samples: dict, groups: list) → DataFrame`

### Data Transformation
- **`no_transform(data)`**: Leaves data unchanged.
- **`log_transform(data)`**: Applies natural log.
- **`boxcox_transform(data)`**: Finds optimal Box-Cox lambda.

### Assumption Checks & Test Selection
- **Normality**: Shapiro–Wilk test.
- **Homogeneity of variance**: Levene’s test.
- **Decision Logic**:
  - If normality & equal variances pass → parametric tests.
  - If one or both fail → prompt for transformation or switch to non-parametric.

### Statistical Tests
- **Two-Group, Independent**: Student’s t-test, Welch’s t-test, Mann–Whitney U.
- **Multi-Group, Independent**: One-way ANOVA, Welch’s ANOVA, Kruskal–Wallis.
- **Two-Group, Paired**: Paired t-test, Wilcoxon signed-rank.
- **Repeated Measures / Mixed**: Repeated-measures ANOVA, mixed-effects models (via Pingouin if installed).
- **Two-Way ANOVA**: Full factorial analysis.

### Post-Hoc Analyses
- Available methods:
  - Tukey’s HSD
  - Dunn’s test with Bonferroni correction
  - Dunnett’s test (comparison to control)
- Automatically invoked when overall test is significant.
- **Factory**: `PostHocFactory` chooses the appropriate analyzer class.

### Decision Tree Visualization
- **Class**: `DecisionTreeVisualizer`
- Shows the sequence of checks and chosen tests as a flowchart.
- Generates a PNG highlighting the actual path taken.
- Accessible via **Analysis → Show Decision Tree**.

---

## Dialog & Configuration Windows

- **GroupSelectionDialog**: Pick which groups to include.
- **ColumnSelectionDialog**: Select measurement and grouping columns.
- **PairwiseComparisonDialog**: Configure two-group tests.
- **TwoWayAnovaDialog**: Set factors for two-way ANOVA.
- **AdvancedTestDialog**: Drag-and-drop design for mixed/repeated-measures.
- **PlotConfigDialog**: Customize plot appearance and significance annotations.
- **TransformationDialog**: Choose or confirm transformation if assumptions fail.
- **OutlierDetectionDialog**: Identify and remove outliers.

---

## Exporting Results to Excel

- **`ResultsExporter`** compiles:
  - Raw data tables
  - Assumption test results
  - Main test summary with statistics and p-values
  - Post-hoc tables
  - Any applied transformations
  - Embedded plots
- Generates a multi-sheet Excel file (`.xlsx`) using `xlsxwriter`.
- Sheets are named for easy navigation (e.g., `Data`, `Assumptions`, `ANOVA`, `PostHoc`).

---

## Visualization & Plots

- Boxplots, barplots, violin plots, and error bars via Seaborn/Matplotlib.
- Customizable via **PlotConfigDialog**:
  - Axis labels, titles, figure size
  - Colors, hatches
  - Annotation of significance (p-values, asterisks)
  - Reset comparisons or add new ones

---

## Tips & Troubleshooting

- **Missing Values**: The importer automatically drops NaNs.
- **Large Datasets**: Use grouping options to filter out unwanted categories.
- **Transformations**: Log and Box-Cox can improve normality—experiment if prompted.
- **Excel Export**: Ensure the target folder is writable; any existing file with the same name will be overwritten.
- **Error Dialogs**: Detailed messages will help track down issues (e.g., incorrect factor levels).

---