# Statistical Analyzer for Windows

This README explains the installation and setup steps for the Statistical Analyzer, a tool for statistical analysis of Excel/CSV data with a graphical user interface.

## Prerequisites

### Python
- Python 3.7 or higher
- Required Python packages:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scipy
  - statsmodels
  - xlsxwriter
  - PyQt5
  - scikit_posthocs
  - pingouin

#### Installation with pip:
```
pip install numpy pandas matplotlib seaborn scipy statsmodels xlsxwriter PyQt5 scikit_posthocs pingouin
```

## Project Files
The following files must be present in the project directory:

- `statistical_analyzer.py` → Main program with UI
- `stats_functions.py` → Statistical functions and helper functions
- `StyleSheet.qss` → CSS style for the user interface
- `Institutslogo.ico` → (optional) Icon for the executable file

## Running the Program

### Development Environment
To run the program directly:
```
python statistical_analyzer.py
```

### Creating an EXE with PyInstaller
1. Install PyInstaller:
   ```
   pip install pyinstaller
   ```

2. The spec file (StatisticalAnalyzer.spec) should already contain all necessary settings:
   - Paths to required files
   - Definition of hidden imports
   - Icon setting

3. Create executable file:
   ```
   pyinstaller StatisticalAnalyzer.spec
   ```

4. After successful creation, the executable file will be in the dist folder.

## Dependent Samples Option
The Statistical Analyzer supports the analysis of dependent samples. When activating the "Show individual connection lines" option, the following visualizations are created:

- A line graph showing the individual connections between measurements of the same subject
- The lines connect measured values that belong to the same subject
- This option is particularly useful for visualizing individual changes across different conditions or time points

## Troubleshooting

### PyInstaller cannot find dependencies:
- Add missing modules to the hiddenimports list in the spec file

### Errors with dependent samples:
- Ensure all groups have the same number of measurements
- The measurements must be in the same order (1st measurement of group A corresponds to 1st measurement of group B)

## Contact
For problems or questions, please contact the developer.