# BioMedStatX
 
- **App‑Download:** [Releases-Seite](https://github.com/philippkrumm/BioMedStatX/releases/latest) 

## 📖 User Guide
- Read the full tutorial (with embedded screenshots) here: [docs/HowTo.md](docs/HowTo.md)
- Read the advances anovas tutorial here: [ADVANCED_ANOVA_GUIDE.md](ADVANCED_ANOVA_GUIDE.md)


## **Features**
BioMedStatX is an open-source Python application for biomedical and clinical data analysis. It offers a graphical user interface (GUI) for loading experimental data, performing statistical tests, visualizing results, and exporting publication-ready outputs.
Designed for researchers and students in the life sciences, BioMedStatX streamlines statistical workflows without requiring coding skills.

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
**This is a one-folder build. Therefore, the BioMedStatX folder must remain in the same folder as the _internal folder.**

- **Windows**: Download and unpack the .zip file, start the .exe from the unpacked folder. 
- **macOS**: Download the `.app` bundle (packed in a `.zip`).

_All features are identical across both platforms._  

## **Citation & Metadata**

- **Repository**: https://github.com/philippkrumm/BioMedStatX
- **Author**: Philipp Krumm
- **Contact**: pkrumm@ukaachen.de
- **Keywords**: biomedical statistics, statistical GUI, Python data analysis, ANOVA, t-tests, post-hoc, non-parametric tests
- **Citation**: Krumm, Philipp and Böttcher, Nicole and Ottermanns, Richard and Pufe, Thomas and Fragoulis, Athanassios, BioMedStatX – Statistical Workflows for Reliable Biomedical Data Analysis. Available at SSRN: https://ssrn.com/abstract=5407324 or http://dx.doi.org/10.2139/ssrn.5407324





