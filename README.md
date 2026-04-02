# BioMedStatX

A comprehensive, GUI-based tool for statistical analysis of experimental data.  
Users can import Excel or CSV files, define groups, and let BioMedStatX handle the rest - from outlier detection and assumption checks, to guided data transformations, selection of appropriate tests, post-hoc analyses, and generation of fully documented reports.

> **Repository:** [philippkrumm/BioMedStatX](https://github.com/philippkrumm/BioMedStatX)  
> **Releases (Download ready-to-use app):** https://github.com/philippkrumm/BioMedStatX/releases  
> **License:** [MIT License](./LICENSE)

---

## Features

BioMedStatX is designed for experimental and biomedical research workflows:

- **Intuitive GUI**  
  Load data, select groups and variables, and trigger analyses without writing code.

- **Automated statistical pipeline**  
  - Outlier detection  
  - Assumption checks (normality, variance homogeneity, etc.)  
  - Guided data transformations where appropriate  
  - Automatic selection of parametric vs. nonparametric tests for supported designs  
  - Guided post-hoc analyses when needed

- **Rich output**  
  - Publication-ready plots  
  - Detailed Excel report with all intermediate steps, assumptions, and test decisions  
  - Clear documentation of which test was selected and why

- **Excel/CSV support**  
  - Direct import of `.xlsx` and `.csv` files  
  - Provided template: [`StatisticalAnalyzer_Excel_Template.xlsx`](./docs/StatisticalAnalyzer_Excel_Template.xlsx)

- **Correlation & Regression**
  - Pearson / Spearman correlation with 95% confidence intervals (auto-selected via Shapiro-Wilk)
  - Simple and multiple linear regression (OLS) with full residual diagnostics (Ramsey RESET, Breusch-Pagan, Shapiro-Wilk on residuals)
  - Exploratory correlation matrix across all numeric variables with FDR (Benjamini-Hochberg) or Bonferroni correction, pairwise deletion, and optional stratification

- **Subgroup analysis via Filter-Bucket**
  - Drag any categorical column into the Filter bucket to restrict the analysis to a subset of rows (e.g. On-Pump patients only)

- **Transparent methodology**
  - Advanced explanations for ANOVA workflows: see [Advanced ANOVA Guide](./docs/ADVANCED_ANOVA_GUIDE.md)
  - Correlation and regression methodology: see [Correlation & Regression Guide](./docs/CORRELATION_REGRESSION_GUIDE.md)

---

## Installation

BioMedStatX is distributed as a standalone application for end users and as source code for developers.

### Option 1: Recommended for most users - Download from Releases

1. Go to the GitHub Releases page:  
   -> https://github.com/philippkrumm/BioMedStatX/releases
2. Download the latest release (e.g. a `.zip` file containing `BioMedStatX.exe`).
3. Extract the archive to a folder of your choice. **Please note that this is a one-folder packaging and the BioMedStatX.exe stays always together with the _internal file in one folder**
4. Start the application by double-clicking `BioMedStatX.exe`.

That's it - no Python installation or command line usage is required for running the app.

For a step-by-step walkthrough of the GUI (with screenshots), see:  
-> [How to use BioMedStatX (User Guide with screenshots)](./docs/HowTo.md)

### Option 2: For developers and contributors – Run from source

If you want to inspect or modify the source code, or contribute to the project:

1. **Clone the repository**

```bash
git clone https://github.com/philippkrumm/BioMedStatX.git
cd BioMedStatX
```

For information about the helper scripts included in this repository, including [`Start_BioMedStatX_on_Linux.sh`](./Start_BioMedStatX_on_Linux.sh) and [`start.bat`](./start.bat), see: [docs/SCRIPTS.md](./docs/SCRIPTS.md)


---

## Quick Start

BioMedStatX provides a GUI-based workflow for statistical analysis.  
A detailed, step-by-step **User Guide with screenshots and numbered button references** is available here:

-> [How to use BioMedStatX (User Guide with screenshots)](./docs/HowTo.md)

### Basic workflow (short version)

1. **Start BioMedStatX**  
   Launch the main application (e.g., via your Python entry point or executable - see the User Guide for details).

2. **Load your dataset**  
   - Import an Excel or CSV file.  
   - Optionally use the provided template: [`StatisticalAnalyzer_Excel_Template.xlsx`](./docs/StatisticalAnalyzer_Excel_Template.xlsx).

3. **Define groups and variables**  
   - Select the sheet (for Excel files).  
   - Choose grouping variables and measurement columns.  
   - Optionally restrict groups via **Select Groups For Analysis**.
   - Optionally restrict rows via **Filter** bucket.

4. **Configure analysis options (optional)**  
   - Choose plots and statistics to generate.  
   - Adjust settings as needed (see the [User Guide](./docs/HowTo.md) for screenshots).

5. **Run the analysis**  
   - Start the analysis and let BioMedStatX automatically:
     - detect outliers,
     - check assumptions,
     - select the appropriate supported test,
       - run post-hoc tests when needed.
    - You decide when prompted:
       - whether to apply offered transformations,
       - which post-hoc procedure to run when multiple valid options exist.

6. **Inspect the output**  
   - Review plots and statistical results.  
   - Open the generated Excel report for a fully documented analysis pipeline.

For a complete, screenshot-based walkthrough, including which button to click at each step, see the [User Guide](./docs/HowTo.md).

---

## Documentation & Guides

- **User Guide (GUI, step-by-step with screenshots):**
   -> [docs/HowTo.md](./docs/HowTo.md)

   Includes:
   - first-time orientation (what the app decides vs what the user decides),
   - minimum data structure requirements,
   - end-to-end analysis order,
   - mapping-to-test logic and export interpretation.

- **Advanced ANOVA methodology and interpretation:**
   -> [docs/ADVANCED_ANOVA_GUIDE.md](./docs/ADVANCED_ANOVA_GUIDE.md)

- **Correlation & Regression methodology and interpretation:**
   -> [docs/CORRELATION_REGRESSION_GUIDE.md](./docs/CORRELATION_REGRESSION_GUIDE.md)

Additional documentation can be added to the [`docs/`](./docs) folder.

---

## Current Support Notes

- Standard nonparametric workflows are available for common one-factor designs such as Mann-Whitney, Wilcoxon, Kruskal-Wallis, Friedman, and related post-hoc analyses.
- Advanced parametric workflows are available for Two-Way ANOVA, Repeated Measures ANOVA, and Mixed ANOVA.
- Nonparametric fallbacks for advanced ANOVA designs are fully implemented: Friedman test (Repeated Measures fallback), Freedman-Lane permutation test (Two-Way ANOVA fallback), and Brunner-Langer ATS (Mixed ANOVA fallback), each with appropriate post-hoc comparisons.
- ANCOVA (One-Way and Two-Way) is supported when continuous covariates are placed in the Covariates bucket alongside a categorical Factor 1.
- Linear Mixed Models (LMM) and Logistic Regression are available for longitudinal and binary outcome designs via the Auto-pilot.
- Correlation (Pearson/Spearman) and linear regression (OLS) are supported via the Auto-pilot when a continuous variable is assigned to the Factor 1 bucket.
- Exploratory correlation matrices are available via the dedicated button in the Auto-pilot panel.
- Exploratory correlation matrices are available via **Analysis -> Exploratory Correlation Matrix**.
- Subgroup analyses can be performed using the Filter bucket to restrict any analysis to a subset of rows.
- For Windows and macOS end users, the recommended path is to use the packaged application from the GitHub Releases page. The repository launcher scripts are mainly intended for source-based usage.

---

## Repository Structure

A brief overview of the repository layout:

```text
BioMedStatX/
├─ README.md                      # Landing page (this file)
├─ LICENSE                        # MIT License
├─ CONTRIBUTING.md                # Detailed contributing guidelines
├─ CODE_OF_CONDUCT.md             # Contributor Covenant Code of Conduct
├─ Start_BioMedStatX_on_Linux.sh  # Launcher for Linux/macOS source/binary startup
├─ start.bat                      # Launcher for Windows source/binary startup
├─ src/                   # Main application source code
├─ docs/                          # User-facing documentation
│  ├─ HowTo.md                           # Screenshot-based user guide (GUI)
│  ├─ ADVANCED_ANOVA_GUIDE.md            # Advanced ANOVA explanations
│  └─ CORRELATION_REGRESSION_GUIDE.md    # Correlation & regression methodology
└─ .github/
   └─ ISSUE_TEMPLATE/
      ├─ bug_report.yml           # Bug report issue template
      └─ feature_request.yml      # Feature request issue template
```

---

## Contributing & Issue Reporting

BioMedStatX is developed as an open-source academic project.  
We welcome bug reports, feature requests, and code contributions.

- Contribution workflow and coding guidelines: [CONTRIBUTING.md](./CONTRIBUTING.md)  
- Community rules: [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md)  
- License: [LICENSE](./LICENSE)

### Reporting bugs and requesting features

Please use the structured issue templates on GitHub:

-> [Create a new issue](https://github.com/philippkrumm/BioMedStatX/issues/new/choose)

This will guide you through our predefined templates for:

- Bug reports
- Feature requests
- (Any additional templates you may add in the future)

Using the templates helps us reproduce issues more easily and keep the project maintainable.

### Contributing code

If you plan to contribute code:

1. Fork the repository and create a feature branch.
2. Implement your changes and add tests where applicable.
3. Ensure code style and formatting follow the guidelines.
4. Open a Pull Request (PR) with a clear description of your changes.

For full details, including branch naming conventions, commit message guidelines, testing expectations, and how to add new statistical functions, please read:

-> [CONTRIBUTING.md](./CONTRIBUTING.md)

---

## License

BioMedStatX is released under the **MIT License**.

-> See the full license text in [LICENSE](./LICENSE).

---

## Citation

If you use BioMedStatX in a scientific publication, please cite the software repository or the published paper once a formal citation is available. This section will be updated when the final reference and DOI are available.

---

## Paper

A direct publication link will be added here once the corresponding paper is publicly available.

---

## Contact

For questions regarding the software, collaboration requests, or feedback, you can reach the maintainer at:

- **Email:** pkrumm@ukaachen.de  
- **GitHub:** [@philippkrumm](https://github.com/philippkrumm)

Please use [GitHub Issues](https://github.com/philippkrumm/BioMedStatX/issues/new/choose) for bug reports and feature requests so that the discussion remains transparent and searchable.

## ToDos

In this section, we provide some ideas that we think should be implemented, but the maintainers, have not had the time to. If you have the resources to fulfill any of these ToDos, we would love your contribution.

➡️ [Contributing & Issue Reporting](#contributing--issue-reporting)

- Improve support for running from source on Linux, especially for workflows that depend on Excel-specific behavior.
