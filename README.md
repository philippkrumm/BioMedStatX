# BioMedStatX

BioMedStatX is a novel tool for the statistical analysis of your experiments.

## **Features**

- **Data Import**  
  Read CSV and Excel files with multiple sheets and columns.

- **Assumption Checks**  
  Automatic Shapiro–Wilk normality tests and Levene’s variance homogeneity tests, with optional log/Box–Cox transformations on failure.

- **Parametric & Non-parametric Tests**  
  Student’s t-tests, Welch’s t-tests, ANOVA (one-way, two-way, repeated measures, mixed), Mann–Whitney U, Kruskal–Wallis, and more.

- **Post-Hoc Analyses**  
  Tukey HSD, Dunnett, Dunn, and dependent-sample analyses via a unified factory interface.

- **Visualization**  
  - Static plots (boxplots, bar graphs, swarm plots) with Seaborn/Matplotlib.  
  - Interactive decision-tree flowchart highlighting the actual analysis path (via NetworkX & Matplotlib).

- **GUI Front-End**  
  PyQt5 dialogs for streamlined data/variable selection, plot configuration, and advanced design setups.

- **Excel Export**  
  Automated creation of detailed result workbooks (via XlsxWriter).

## **Installation**

Simply download the pre-built binary for your operating system:

- **Windows**: Download and run the `.exe` installer.  
- **macOS**: Download the `.app` bundle (packed in a `.zip`).

_All features are identical across both platforms._  
