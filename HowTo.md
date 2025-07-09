
# Statistical Analyzer User Guide (HowTo.md)

This guide explains how to use the standalone `.exe` for Statistical Analyzer—from launching the app to importing data, running analyses, and exporting results—without needing any Python commands.

---

## 1. Launching the Application

- Locate the `StatisticalAnalyzer.exe` file.
- Double‑click to open. A Qt GUI window appears with the menus: **File**, **Analysis**, and **Help**.

---

## 2. Importing Data

1. In the **File** menu, choose **Browse** to select an Excel (`.xlsx`/`.xls`) or CSV (`.csv`) file.
2. Upon selection, the file’s sheets (for Excel) populate the **Worksheet** dropdown; CSV skips this step.
3. Click **Load** (or close the dialog) to read data.
---

## 3. Selecting Groups & Measurement Columns

* **GroupSelectionDialog**: Pick which factor/column defines your groups.
* **ColumnSelectionDialog**: Choose one or more numeric columns for analysis. If multiple and **combine\_columns** is enabled, values across columns are merged per group.

---

## 4. Assumption Checking & Transformations

* Before any statistical test, the app runs:

  * **Shapiro–Wilk test** for normality
  * **Levene’s test** for homogeneity of variances
* If either assumption fails, the **TransformationDialog** appears, offering:

  * Log₁₀ transform
  * Box‑Cox transform
  * Arcsine‑sqrt transform

---

## 5. Statistical Tests

* **Two-Group, Independent**: Student’s t‑test, Welch’s t‑test, Mann–Whitney U
* **Two-Group, Paired**: Paired t‑test, Wilcoxon signed‑rank
* **Multi-Group, Independent**: One‑way ANOVA, Welch ANOVA, Kruskal–Wallis
* **Advanced Analyses**: Two‑Way ANOVA, Repeated Measures ANOVA, Mixed ANOVA via **AdvancedTestDialog** 

---

## 6. Post‑Hoc Comparisons

When overall tests are significant, **PostHocFactory** dispatches:

* Tukey’s HSD
* Dunn’s or Bonferroni‑corrected comparisons
* Dunnett’s test (control vs others)

Results appear in separate sheets and on plots.

---

## 7. Decision Tree Visualization

Visualize the statistical decision process via **Analysis → Show Decision Tree**, which calls:

Image is saved as a temporary PNG and highlighted path shows actual branches.

---

## 8. Exporting Results to Excel

After analysis, an excel multi-sheet `.xlsx` is created with:

* **Summary** of tests and p‑values
* **Assumptions** (normality, variance)
* **Main Results** (statistics, effect sizes)
* **Descriptive Statistics** per group
* **Decision Tree**
* **Raw Data**
* **Pairwise Comparisons**
* **Analysis Log** (chronological steps)

Each sheet is clearly named for easy navigation.

---

## 9. Plotting & Customization

Plots are generated with Matplotlib (via Seaborn palettes) and include:

* Bar charts with SD/SEM error bars
* Overlayed individual data points (and connection lines for paired data)
* Violin or boxplots when selected

Use **PlotConfigDialog** to adjust:

* Titles & axis labels
* Figure dimensions
* Colors & hatches per group
* Significance annotations or custom comparisons

Plots can be saved automatically as PDF/PNG alongside Excel.

---

## 10. Outlier Detection (Optional)

Under **Analysis → Detect Outliers**, configure and run:

* Modified Z‑Score Test
* Grubbs’ Test
* Single‑pass or iterative mode

Results export to a specified Excel file via `OutlierDetectionDialog` and `OutlierDetector` in **stats\_functions.py**. 


### Tips & Best Practices

* Ensure your group column has consistent labels.
* Use Box‑Cox or log transforms when skew is severe.
* For paired designs, confirm equal sample sizes per group.
* Consult the **Analysis Log** sheet for troubleshooting and detailed steps.

Happy analyzing!

```
```
