"""Static recipe content for the in-app Help Hub."""

from __future__ import annotations


HELP_RECIPES = [
    {
        "id": "one_way_anova",
        "title": "One-Way ANOVA",
        "summary": "One categorical factor with 3+ independent groups.",
        "keywords": ["anova", "one-way", "group", "factor 1", "long format"],
        "example_tsv": "Group\tValue\nControl\t12.4\nControl\t11.8\nTreatmentA\t15.2\nTreatmentA\t14.9\nTreatmentB\t17.1\nTreatmentB\t16.4",
        "html": """
<h2>Recipe: One-Way ANOVA</h2>
<h3>1. Required data layout (long format) <span class='badge badge-good'>Required</span></h3>
<p>Use one row per observation. Keep the group label in one column and the measurement in one numeric column.</p>
<table>
<tr><th>Group</th><th>Value</th></tr>
<tr><td>Control</td><td>12.4</td></tr>
<tr><td>Control</td><td>11.8</td></tr>
<tr><td>TreatmentA</td><td>15.2</td></tr>
<tr><td>TreatmentA</td><td>14.9</td></tr>
<tr><td>TreatmentB</td><td>17.1</td></tr>
<tr><td>TreatmentB</td><td>16.4</td></tr>
</table>
<h3>2. Common wrong layout (wide format) <span class='badge badge-bad'>Avoid</span></h3>
<table>
<tr><th>Control</th><th>TreatmentA</th><th>TreatmentB</th></tr>
<tr><td>12.4</td><td>15.2</td><td>17.1</td></tr>
<tr><td>11.8</td><td>14.9</td><td>16.4</td></tr>
</table>
<p><b>Why this fails:</b> The app expects one factor column and one value column, not one measurement column per group.</p>
<h3>3. Bucket mapping</h3>
<ul>
<li><b>Dependent Variable:</b> Value</li>
<li><b>Factor 1:</b> Group</li>
<li><b>Factor 2 / Subject ID / Covariates:</b> leave empty</li>
</ul>
<h3>4. Quick validation checklist</h3>
<ul>
<li>Exactly one row per measurement.</li>
<li>Group names repeat across rows.</li>
<li>No pre-aggregated means in the Value column.</li>
</ul>
""",
    },
    {
        "id": "two_way_anova",
        "title": "Two-Way ANOVA",
        "summary": "Two categorical between-subject factors.",
        "keywords": ["anova", "two-way", "factor 2", "interaction", "long format"],
        "example_tsv": "Group\tDose\tValue\nControl\tLow\t10.2\nControl\tHigh\t12.1\nTreatment\tLow\t14.6\nTreatment\tHigh\t18.3\nControl\tLow\t9.8\nTreatment\tHigh\t17.9",
        "html": """
<h2>Recipe: Two-Way ANOVA</h2>
<h3>1. Required data layout (long format) <span class='badge badge-good'>Required</span></h3>
<p>Use one row per subject/observation with both factor columns present in every row.</p>
<table>
<tr><th>Group</th><th>Dose</th><th>Value</th></tr>
<tr><td>Control</td><td>Low</td><td>10.2</td></tr>
<tr><td>Control</td><td>High</td><td>12.1</td></tr>
<tr><td>Treatment</td><td>Low</td><td>14.6</td></tr>
<tr><td>Treatment</td><td>High</td><td>18.3</td></tr>
<tr><td>Control</td><td>Low</td><td>9.8</td></tr>
<tr><td>Treatment</td><td>High</td><td>17.9</td></tr>
</table>
<h3>2. Common wrong layout (wide format) <span class='badge badge-bad'>Avoid</span></h3>
<table>
<tr><th>SubjectID</th><th>Control_Low</th><th>Control_High</th><th>Treatment_Low</th><th>Treatment_High</th></tr>
<tr><td>S01</td><td>10.2</td><td>12.1</td><td>14.6</td><td>18.3</td></tr>
<tr><td>S02</td><td>9.8</td><td>11.7</td><td>14.1</td><td>17.9</td></tr>
</table>
<p><b>Why this fails:</b> Factor levels are encoded in column names. The app cannot map Factor 1 and Factor 2 from header text.</p>
<h3>3. Bucket mapping</h3>
<ul>
<li><b>Dependent Variable:</b> Value</li>
<li><b>Factor 1:</b> Group</li>
<li><b>Factor 2:</b> Dose</li>
<li><b>Subject ID:</b> leave empty for between-subject design</li>
</ul>
<h3>4. Quick validation checklist</h3>
<ul>
<li>Both factors are categorical columns.</li>
<li>Every row has Group + Dose + Value.</li>
<li>No factor combinations are hidden in column headers.</li>
</ul>
""",
    },
    {
        "id": "repeated_measures_anova",
        "title": "Repeated Measures ANOVA",
        "summary": "One within-subject factor measured repeatedly per subject.",
        "keywords": ["repeated", "within", "subject", "timepoint", "long format"],
        "example_tsv": "SubjectID\tTimepoint\tValue\nS01\tBaseline\t5.2\nS01\tWeek4\t6.0\nS01\tWeek8\t6.4\nS02\tBaseline\t4.9\nS02\tWeek4\t5.8\nS02\tWeek8\t6.1",
        "html": """
<h2>Recipe: Repeated Measures ANOVA</h2>
<h3>1. Required data layout (long format) <span class='badge badge-good'>Required</span></h3>
<p>Use one row per measurement, with SubjectID repeated across timepoints.</p>
<table>
<tr><th>SubjectID</th><th>Timepoint</th><th>Value</th></tr>
<tr><td>S01</td><td>Baseline</td><td>5.2</td></tr>
<tr><td>S01</td><td>Week4</td><td>6.0</td></tr>
<tr><td>S01</td><td>Week8</td><td>6.4</td></tr>
<tr><td>S02</td><td>Baseline</td><td>4.9</td></tr>
<tr><td>S02</td><td>Week4</td><td>5.8</td></tr>
<tr><td>S02</td><td>Week8</td><td>6.1</td></tr>
</table>
<h3>2. Common wrong layout (wide format) <span class='badge badge-bad'>Avoid</span></h3>
<table>
<tr><th>SubjectID</th><th>Baseline</th><th>Week4</th><th>Week8</th></tr>
<tr><td>S01</td><td>5.2</td><td>6.0</td><td>6.4</td></tr>
<tr><td>S02</td><td>4.9</td><td>5.8</td><td>6.1</td></tr>
</table>
<p><b>Why this fails:</b> Timepoint must be a categorical column, not multiple value columns.</p>
<h3>3. Bucket mapping</h3>
<ul>
<li><b>Dependent Variable:</b> Value</li>
<li><b>Factor 1:</b> Timepoint</li>
<li><b>Subject ID:</b> SubjectID</li>
<li><b>Factor 2:</b> leave empty</li>
</ul>
<h3>4. Quick validation checklist</h3>
<ul>
<li>Each subject appears in multiple rows.</li>
<li>Timepoint is a single column with repeated labels.</li>
<li>No duplicated SubjectID-Timepoint pairs.</li>
</ul>
""",
    },
    {
        "id": "mixed_anova",
        "title": "Mixed ANOVA",
        "summary": "One between-subject factor plus one within-subject factor.",
        "keywords": ["mixed", "between", "within", "subject id", "long format"],
        "example_tsv": "SubjectID\tGroup\tTimepoint\tValue\nS01\tControl\tBaseline\t8.1\nS01\tControl\tWeek4\t8.7\nS02\tControl\tBaseline\t7.9\nS02\tControl\tWeek4\t8.5\nS11\tTreatment\tBaseline\t8.5\nS11\tTreatment\tWeek4\t10.2",
        "html": """
<h2>Recipe: Mixed ANOVA</h2>
<h3>1. Required data layout (long format) <span class='badge badge-good'>Required</span></h3>
<p>Use one row per measurement. Keep between-factor, within-factor, and SubjectID as separate columns.</p>
<table>
<tr><th>SubjectID</th><th>Group</th><th>Timepoint</th><th>Value</th></tr>
<tr><td>S01</td><td>Control</td><td>Baseline</td><td>8.1</td></tr>
<tr><td>S01</td><td>Control</td><td>Week4</td><td>8.7</td></tr>
<tr><td>S02</td><td>Control</td><td>Baseline</td><td>7.9</td></tr>
<tr><td>S02</td><td>Control</td><td>Week4</td><td>8.5</td></tr>
<tr><td>S11</td><td>Treatment</td><td>Baseline</td><td>8.5</td></tr>
<tr><td>S11</td><td>Treatment</td><td>Week4</td><td>10.2</td></tr>
</table>
<h3>2. Common wrong layout (wide format) <span class='badge badge-bad'>Avoid</span></h3>
<table>
<tr><th>SubjectID</th><th>Group</th><th>Baseline</th><th>Week4</th></tr>
<tr><td>S01</td><td>Control</td><td>8.1</td><td>8.7</td></tr>
<tr><td>S11</td><td>Treatment</td><td>8.5</td><td>10.2</td></tr>
</table>
<p><b>Why this fails:</b> Within-factor levels must be values inside one Timepoint column, not separate columns.</p>
<h3>3. Bucket mapping</h3>
<ul>
<li><b>Dependent Variable:</b> Value</li>
<li><b>Factor 1:</b> Group (between)</li>
<li><b>Factor 2:</b> Timepoint (within)</li>
<li><b>Subject ID:</b> SubjectID</li>
</ul>
<h3>4. Quick validation checklist</h3>
<ul>
<li>SubjectID repeats over within-factor levels.</li>
<li>Group stays constant per subject.</li>
<li>Timepoint is a single categorical column.</li>
</ul>
""",
    },
    {
        "id": "ancova",
        "title": "ANCOVA",
        "summary": "Categorical group factor plus continuous covariate(s).",
        "keywords": ["ancova", "covariate", "slope homogeneity", "long format"],
        "example_tsv": "Treatment\tPost_Weight\tPre_Weight\nControl\t25.4\t24.8\nControl\t24.9\t24.4\nDrug\t27.8\t25.1\nDrug\t27.2\t24.9\nDrug\t28.0\t25.3",
        "html": """
<h2>Recipe: ANCOVA</h2>
<h3>1. Required data layout (subject-level rows) <span class='badge badge-good'>Required</span></h3>
<p>Use one row per subject with one outcome column and one or more covariate columns.</p>
<table>
<tr><th>Treatment</th><th>Post_Weight</th><th>Pre_Weight</th></tr>
<tr><td>Control</td><td>25.4</td><td>24.8</td></tr>
<tr><td>Control</td><td>24.9</td><td>24.4</td></tr>
<tr><td>Drug</td><td>27.8</td><td>25.1</td></tr>
<tr><td>Drug</td><td>27.2</td><td>24.9</td></tr>
<tr><td>Drug</td><td>28.0</td><td>25.3</td></tr>
</table>
<h3>2. Common wrong layout (aggregated summary) <span class='badge badge-bad'>Avoid</span></h3>
<table>
<tr><th>Treatment</th><th>Mean_Post_Weight</th><th>Mean_Pre_Weight</th></tr>
<tr><td>Control</td><td>25.15</td><td>24.60</td></tr>
<tr><td>Drug</td><td>27.67</td><td>25.10</td></tr>
</table>
<p><b>Why this fails:</b> ANCOVA needs subject-level variance. Group means remove the required residual structure.</p>
<h3>3. Bucket mapping</h3>
<ul>
<li><b>Dependent Variable:</b> Post_Weight</li>
<li><b>Factor 1:</b> Treatment</li>
<li><b>Covariates:</b> Pre_Weight</li>
<li><b>Factor 2 / Subject ID:</b> usually empty</li>
</ul>
<h3>4. Quick validation checklist</h3>
<ul>
<li>No pre-aggregated means.</li>
<li>Covariates are numeric columns.</li>
<li>Treatment is categorical.</li>
</ul>
""",
    },
    {
        "id": "correlation",
        "title": "Correlation",
        "summary": "Continuous predictor and continuous outcome without covariates.",
        "keywords": ["correlation", "pearson", "spearman", "long format"],
        "example_tsv": "Biomarker\tOutcome\n3.2\t41.0\n4.1\t46.2\n5.3\t50.8\n6.0\t54.1\n6.8\t56.3",
        "html": """
<h2>Recipe: Correlation</h2>
<h3>1. Required data layout (subject-level rows) <span class='badge badge-good'>Required</span></h3>
<p>Use one row per subject with one predictor and one outcome value.</p>
<table>
<tr><th>Biomarker</th><th>Outcome</th></tr>
<tr><td>3.2</td><td>41.0</td></tr>
<tr><td>4.1</td><td>46.2</td></tr>
<tr><td>5.3</td><td>50.8</td></tr>
<tr><td>6.0</td><td>54.1</td></tr>
<tr><td>6.8</td><td>56.3</td></tr>
</table>
<h3>2. Common wrong layout (binned summary) <span class='badge badge-bad'>Avoid</span></h3>
<table>
<tr><th>Biomarker_Bin</th><th>Mean_Outcome</th></tr>
<tr><td>Low</td><td>42.8</td></tr>
<tr><td>Medium</td><td>50.8</td></tr>
<tr><td>High</td><td>55.2</td></tr>
</table>
<p><b>Why this fails:</b> Correlation requires raw paired observations. Bin means change the correlation structure.</p>
<h3>3. Bucket mapping</h3>
<ul>
<li><b>Dependent Variable:</b> Outcome</li>
<li><b>Factor 1:</b> Biomarker (continuous)</li>
<li><b>Covariates:</b> leave empty</li>
<li><b>Subject ID:</b> leave empty for standard correlation</li>
</ul>
<h3>4. Quick validation checklist</h3>
<ul>
<li>Both columns are numeric.</li>
<li>Each row is one subject or sample.</li>
<li>No grouped means or bins.</li>
</ul>
""",
    },
    {
        "id": "linear_regression",
        "title": "Linear Regression (OLS)",
        "summary": "Continuous predictor with optional additional covariates.",
        "keywords": ["regression", "ols", "covariates", "residual", "long format"],
        "example_tsv": "Pump_Time\tNK_Cells\tAge\tBMI\n72\t180\t61\t27.1\n88\t164\t66\t29.0\n54\t205\t59\t25.8\n95\t152\t70\t30.2\n79\t171\t64\t28.0",
        "html": """
<h2>Recipe: Linear Regression (OLS)</h2>
<h3>1. Required data layout (subject-level rows) <span class='badge badge-good'>Required</span></h3>
<p>Use one row per subject with one outcome and one or more numeric predictors.</p>
<table>
<tr><th>Pump_Time</th><th>NK_Cells</th><th>Age</th><th>BMI</th></tr>
<tr><td>72</td><td>180</td><td>61</td><td>27.1</td></tr>
<tr><td>88</td><td>164</td><td>66</td><td>29.0</td></tr>
<tr><td>54</td><td>205</td><td>59</td><td>25.8</td></tr>
<tr><td>95</td><td>152</td><td>70</td><td>30.2</td></tr>
<tr><td>79</td><td>171</td><td>64</td><td>28.0</td></tr>
</table>
<h3>2. Common wrong layout (pre-computed model output) <span class='badge badge-bad'>Avoid</span></h3>
<table>
<tr><th>Term</th><th>Estimate</th><th>P_Value</th></tr>
<tr><td>Intercept</td><td>250.1</td><td>0.001</td></tr>
<tr><td>Pump_Time</td><td>-0.92</td><td>0.004</td></tr>
</table>
<p><b>Why this fails:</b> The app needs raw rows to fit diagnostics and verify assumptions.</p>
<h3>3. Bucket mapping</h3>
<ul>
<li><b>Dependent Variable:</b> NK_Cells</li>
<li><b>Factor 1:</b> Pump_Time (continuous main predictor)</li>
<li><b>Covariates:</b> Age, BMI (optional)</li>
<li><b>Subject ID:</b> leave empty for standard OLS</li>
</ul>
<h3>4. Quick validation checklist</h3>
<ul>
<li>Outcome is numeric.</li>
<li>Main predictor is numeric.</li>
<li>Covariates are numeric and not duplicated columns.</li>
</ul>
""",
    },
    {
        "id": "logistic_regression",
        "title": "Logistic Regression",
        "summary": "Binary outcome (0/1) with numeric/categorical predictors.",
        "keywords": ["logistic", "binary", "odds ratio", "long format"],
        "example_tsv": "Event\tBiomarker\tAge\n0\t1.8\t59\n1\t2.7\t63\n0\t2.0\t57\n1\t3.1\t68\n1\t2.9\t65",
        "html": """
<h2>Recipe: Logistic Regression</h2>
<h3>1. Required data layout (subject-level rows) <span class='badge badge-good'>Required</span></h3>
<p>Use one row per subject with a binary outcome coded as 0/1 and predictor columns.</p>
<table>
<tr><th>Event</th><th>Biomarker</th><th>Age</th></tr>
<tr><td>0</td><td>1.8</td><td>59</td></tr>
<tr><td>1</td><td>2.7</td><td>63</td></tr>
<tr><td>0</td><td>2.0</td><td>57</td></tr>
<tr><td>1</td><td>3.1</td><td>68</td></tr>
<tr><td>1</td><td>2.9</td><td>65</td></tr>
</table>
<h3>2. Common wrong layout (counts only) <span class='badge badge-bad'>Avoid</span></h3>
<table>
<tr><th>Group</th><th>Events</th><th>NonEvents</th></tr>
<tr><td>Low Biomarker</td><td>4</td><td>18</td></tr>
<tr><td>High Biomarker</td><td>13</td><td>9</td></tr>
</table>
<p><b>Why this fails:</b> Logistic regression in the app expects row-level outcomes, not aggregated counts.</p>
<h3>3. Bucket mapping</h3>
<ul>
<li><b>Dependent Variable:</b> Event (binary 0/1)</li>
<li><b>Factor 1:</b> Biomarker (or main predictor)</li>
<li><b>Covariates:</b> additional predictors such as Age</li>
</ul>
<h3>4. Quick validation checklist</h3>
<ul>
<li>Outcome has exactly two levels (0/1 recommended).</li>
<li>Each row is one observation.</li>
<li>No summary count tables.</li>
</ul>
""",
    },
]
