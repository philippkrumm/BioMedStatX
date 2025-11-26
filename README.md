# BioMedStatX

- **App‑Download:** [Releases-Seite](https://github.com/philippkrumm/BioMedStatX/releases/latest) 

## Table of contents

- [User Guide](#user-guide)
- [Features](#features)
- [Installation](#installation)
- [Open-Source Development Workflow](#open-source-development-workflow)
- [Contributing](#contributing)
- [Reporting Issues](#reporting-issues)
- [Citation & Metadata](#citation--metadata)
- [Licensing](#licensing)
- [Links](#links)

## User Guide
- Read the full tutorial (with embedded screenshots) here: [docs/HowTo.md](docs/HowTo.md)
- Read the advanced ANOVAs tutorial here: [ADVANCED_ANOVA_GUIDE.md](ADVANCED_ANOVA_GUIDE.md)

## Features
BioMedStatX is an open-source Python application for biomedical and clinical data analysis. It offers a graphical user interface (GUI) for loading experimental data and performing statistical tests. Designed for researchers and students in the life sciences, BioMedStatX streamlines statistical workflows without requiring coding skills.

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

## Installation

Simply download the pre-built binary for your operating system:
**This is a one-folder build. Therefore, the BioMedStatX folder must remain in the same folder as the _internal folder.**

- **Windows**: Download and unpack the .zip file, start the .exe from the unpacked folder. 
- **macOS**: Download the `.app` bundle (packed in a `.zip`).

_All features are identical across both platforms._  

## Open-Source Development Workflow

We welcome issues and contributions. To get involved:

- Reporting issues: When reporting problems please use the provided issue templates (Bug report and Feature request) — GitHub will automatically suggest them when creating a new issue. Use the Bug report template for reproducible problems and include environment details and steps to reproduce.
- Fork-and-pull-request workflow: Fork the repository, create a topic branch (see CONTRIBUTING.md for naming conventions), implement your changes, run tests and formatters locally, then open a Pull Request against the main repository. Link to the relevant issue if one exists.
- Contribution guide: See [CONTRIBUTING.md](CONTRIBUTING.md) for full contribution instructions including PR structure, code style, and testing expectations.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute, coding style, testing expectations and branch naming conventions.

## Reporting Issues

To report a bug or request a feature, click "New issue" on GitHub — the platform will prompt you to use one of the provided templates:

- Bug report — include steps to reproduce, expected vs actual behavior, environment (OS, Python version, BioMedStatX version) and optional attachments.
- Feature request — describe the use case, proposed solution and related references.

## Citation & Metadata

- **Repository**: https://github.com/philippkrumm/BioMedStatX
- **Author**: Philipp Krumm
- **Contact**: pkrumm@ukaachen.de
- **Keywords**: biomedical statistics, statistical GUI, Python data analysis, ANOVA, t-tests, post-hoc, non-parametric tests
- **Citation**: Krumm, Philipp and Böttcher, Nicole and Ottermanns, Richard and Pufe, Thomas and Fragoulis, Athanassios, BioMedStatX – Statistical Workflows for Reliable Biomedical Data Analysis...

## Licensing

BioMedStatX is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Links

- Contribution guidelines: [CONTRIBUTING.md](CONTRIBUTING.md)
- Issue templates: use the GitHub "New issue" dialog to access the Bug report and Feature request templates (stored under `.github/ISSUE_TEMPLATE/`).
