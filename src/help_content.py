"""Static recipe content for the in-app Help Hub."""

from __future__ import annotations


HELP_RECIPES = [
    {
        "id": "one_way_anova",
        "title": "One-Way ANOVA",
        "summary": "One categorical factor with 3+ independent groups.",
        "keywords": ["anova", "one-way", "group", "factor 1"],
        "example_tsv": "Group\tValue\nControl\t12.4\nControl\t11.8\nTreatmentA\t15.2\nTreatmentA\t14.9\nTreatmentB\t17.1\nTreatmentB\t16.4",
        "html": """
<h2>Recipe: One-Way ANOVA</h2>
<h3>1. Data layout</h3>
<p>You need one categorical group column and one numeric measurement column.</p>
<table border='1' cellspacing='0' cellpadding='4'>
<tr><th>Group</th><th>Value</th></tr>
<tr><td>Control</td><td>12.4</td></tr>
<tr><td>TreatmentA</td><td>15.2</td></tr>
<tr><td>TreatmentB</td><td>17.1</td></tr>
</table>
<h3>2. Bucket mapping</h3>
<ul>
<li><b>Dependent Variable:</b> Value</li>
<li><b>Factor 1:</b> Group</li>
<li><b>Factor 2 / Subject ID / Covariates:</b> leave empty</li>
</ul>
<h3>3. What the app does automatically</h3>
<ul>
<li>Checks normality and variance homogeneity.</li>
<li>Runs One-Way ANOVA if assumptions are met.</li>
<li>Switches to Kruskal-Wallis fallback when assumptions are violated.</li>
</ul>
<h3>4. What the user decides</h3>
<ul>
<li>Accept or skip transformation prompt if shown.</li>
<li>Select post-hoc method when significant effects are found.</li>
</ul>
""",
    },
    {
        "id": "two_way_anova",
        "title": "Two-Way ANOVA",
        "summary": "Two categorical between-subject factors.",
        "keywords": ["anova", "two-way", "factor 2", "interaction"],
        "example_tsv": "Group\tDose\tValue\nControl\tLow\t10.2\nControl\tHigh\t12.1\nTreatment\tLow\t14.6\nTreatment\tHigh\t18.3",
        "html": """
<h2>Recipe: Two-Way ANOVA</h2>
<h3>1. Data layout</h3>
<p>You need two categorical factor columns plus one numeric measurement column.</p>
<table border='1' cellspacing='0' cellpadding='4'>
<tr><th>Group</th><th>Dose</th><th>Value</th></tr>
<tr><td>Control</td><td>Low</td><td>10.2</td></tr>
<tr><td>Treatment</td><td>High</td><td>18.3</td></tr>
</table>
<h3>2. Bucket mapping</h3>
<ul>
<li><b>Dependent Variable:</b> Value</li>
<li><b>Factor 1:</b> Group</li>
<li><b>Factor 2:</b> Dose</li>
<li><b>Subject ID:</b> leave empty for between-subject design</li>
</ul>
<h3>3. What the app does automatically</h3>
<ul>
<li>Runs assumption checks and interaction-capable model path.</li>
<li>Uses parametric two-way model if valid.</li>
<li>Uses nonparametric fallback path when needed.</li>
</ul>
<h3>4. What the user decides</h3>
<ul>
<li>Transformation acceptance if prompted.</li>
<li>Post-hoc strategy if significant main effects/interactions are present.</li>
</ul>
""",
    },
    {
        "id": "repeated_measures_anova",
        "title": "Repeated Measures ANOVA",
        "summary": "One within-subject factor measured repeatedly per subject.",
        "keywords": ["repeated", "within", "subject", "timepoint"],
        "example_tsv": "SubjectID\tTimepoint\tValue\nS01\tBaseline\t5.2\nS01\tWeek4\t6.0\nS02\tBaseline\t4.9\nS02\tWeek4\t5.8",
        "html": """
<h2>Recipe: Repeated Measures ANOVA</h2>
<h3>1. Data layout</h3>
<p>You need repeated rows per subject across levels of one within factor.</p>
<table border='1' cellspacing='0' cellpadding='4'>
<tr><th>SubjectID</th><th>Timepoint</th><th>Value</th></tr>
<tr><td>S01</td><td>Baseline</td><td>5.2</td></tr>
<tr><td>S01</td><td>Week4</td><td>6.0</td></tr>
</table>
<h3>2. Bucket mapping</h3>
<ul>
<li><b>Dependent Variable:</b> Value</li>
<li><b>Factor 1:</b> Timepoint</li>
<li><b>Subject ID:</b> SubjectID</li>
<li><b>Factor 2:</b> leave empty</li>
</ul>
<h3>3. What the app does automatically</h3>
<ul>
<li>Detects repeated structure from Subject ID.</li>
<li>Runs repeated-measures path and sphericity checks where applicable.</li>
<li>Switches to Friedman fallback if assumptions fail.</li>
</ul>
<h3>4. What the user decides</h3>
<ul>
<li>Transformation acceptance when offered.</li>
<li>Post-hoc options for significant results.</li>
</ul>
""",
    },
    {
        "id": "mixed_anova",
        "title": "Mixed ANOVA",
        "summary": "One between-subject factor plus one within-subject factor.",
        "keywords": ["mixed", "between", "within", "subject id"],
        "example_tsv": "SubjectID\tGroup\tTimepoint\tValue\nS01\tControl\tBaseline\t8.1\nS01\tControl\tWeek4\t8.7\nS11\tTreatment\tBaseline\t8.5\nS11\tTreatment\tWeek4\t10.2",
        "html": """
<h2>Recipe: Mixed ANOVA</h2>
<h3>1. Data layout</h3>
<p>You need a subject identifier, one between factor, one within factor, and one numeric outcome.</p>
<table border='1' cellspacing='0' cellpadding='4'>
<tr><th>SubjectID</th><th>Group</th><th>Timepoint</th><th>Value</th></tr>
<tr><td>S01</td><td>Control</td><td>Baseline</td><td>8.1</td></tr>
<tr><td>S11</td><td>Treatment</td><td>Week4</td><td>10.2</td></tr>
</table>
<h3>2. Bucket mapping</h3>
<ul>
<li><b>Dependent Variable:</b> Value</li>
<li><b>Factor 1:</b> Group (between)</li>
<li><b>Factor 2:</b> Timepoint (within)</li>
<li><b>Subject ID:</b> SubjectID</li>
</ul>
<h3>3. What the app does automatically</h3>
<ul>
<li>Recognizes mixed design from Factor 1 + Factor 2 + Subject ID.</li>
<li>Evaluates assumptions and runs mixed ANOVA path.</li>
<li>Uses Brunner-Langer style fallback when assumptions are not met.</li>
</ul>
<h3>4. What the user decides</h3>
<ul>
<li>Transformation acceptance if prompted.</li>
<li>Post-hoc method choice for significant effects.</li>
</ul>
""",
    },
    {
        "id": "ancova",
        "title": "ANCOVA",
        "summary": "Categorical group factor plus continuous covariate(s).",
        "keywords": ["ancova", "covariate", "slope homogeneity"],
        "example_tsv": "Treatment\tPost_Weight\tPre_Weight\nControl\t25.4\t24.8\nControl\t24.9\t24.4\nDrug\t27.8\t25.1\nDrug\t27.2\t24.9",
        "html": """
<h2>Recipe: ANCOVA</h2>
<h3>1. Data layout</h3>
<p>You need one categorical group factor, one numeric outcome, and one or more numeric covariates.</p>
<table border='1' cellspacing='0' cellpadding='4'>
<tr><th>Treatment</th><th>Post_Weight</th><th>Pre_Weight</th></tr>
<tr><td>Control</td><td>25.4</td><td>24.8</td></tr>
<tr><td>Drug</td><td>27.8</td><td>25.1</td></tr>
</table>
<h3>2. Bucket mapping</h3>
<ul>
<li><b>Dependent Variable:</b> Post_Weight</li>
<li><b>Factor 1:</b> Treatment</li>
<li><b>Covariates:</b> Pre_Weight</li>
<li><b>Factor 2 / Subject ID:</b> leave empty unless your design requires them</li>
</ul>
<h3>3. What the app does automatically</h3>
<ul>
<li>Recognizes categorical factor + numeric covariate pattern as ANCOVA path.</li>
<li>Checks assumptions including slope homogeneity diagnostics.</li>
<li>Produces adjusted inference output in export sheets.</li>
</ul>
<h3>4. What the user decides</h3>
<ul>
<li>Transformation acceptance if assumptions are violated.</li>
<li>Post-hoc selection when applicable.</li>
</ul>
""",
    },
    {
        "id": "correlation",
        "title": "Correlation",
        "summary": "Continuous predictor and continuous outcome without covariates.",
        "keywords": ["correlation", "pearson", "spearman"],
        "example_tsv": "Biomarker\tOutcome\n3.2\t41.0\n4.1\t46.2\n5.3\t50.8\n6.0\t54.1",
        "html": """
<h2>Recipe: Correlation</h2>
<h3>1. Data layout</h3>
<p>You need two numeric columns: predictor and outcome.</p>
<table border='1' cellspacing='0' cellpadding='4'>
<tr><th>Biomarker</th><th>Outcome</th></tr>
<tr><td>3.2</td><td>41.0</td></tr>
<tr><td>6.0</td><td>54.1</td></tr>
</table>
<h3>2. Bucket mapping</h3>
<ul>
<li><b>Dependent Variable:</b> Outcome</li>
<li><b>Factor 1:</b> Biomarker (continuous)</li>
<li><b>Covariates:</b> leave empty</li>
<li><b>Subject ID:</b> leave empty for basic correlation</li>
</ul>
<h3>3. What the app does automatically</h3>
<ul>
<li>Chooses Pearson or Spearman based on normality checks.</li>
<li>Computes coefficient, confidence intervals, p-value, and sample size details.</li>
</ul>
<h3>4. What the user decides</h3>
<ul>
<li>Whether to apply a subgroup filter before running.</li>
<li>Whether to use exploratory matrix from the Analysis menu for screening.</li>
</ul>
""",
    },
    {
        "id": "linear_regression",
        "title": "Linear Regression (OLS)",
        "summary": "Continuous predictor with optional additional covariates.",
        "keywords": ["regression", "ols", "covariates", "residual"],
        "example_tsv": "Pump_Time\tNK_Cells\tAge\tBMI\n72\t180\t61\t27.1\n88\t164\t66\t29.0\n54\t205\t59\t25.8\n95\t152\t70\t30.2",
        "html": """
<h2>Recipe: Linear Regression (OLS)</h2>
<h3>1. Data layout</h3>
<p>You need one numeric outcome and one or more numeric predictors.</p>
<table border='1' cellspacing='0' cellpadding='4'>
<tr><th>Pump_Time</th><th>NK_Cells</th><th>Age</th><th>BMI</th></tr>
<tr><td>72</td><td>180</td><td>61</td><td>27.1</td></tr>
<tr><td>95</td><td>152</td><td>70</td><td>30.2</td></tr>
</table>
<h3>2. Bucket mapping</h3>
<ul>
<li><b>Dependent Variable:</b> NK_Cells</li>
<li><b>Factor 1:</b> Pump_Time (continuous main predictor)</li>
<li><b>Covariates:</b> Age, BMI (optional additional predictors)</li>
<li><b>Subject ID:</b> leave empty for standard OLS</li>
</ul>
<h3>3. What the app does automatically</h3>
<ul>
<li>Runs OLS model and reports coefficients and model fit metrics.</li>
<li>Runs residual diagnostics (normality, homoscedasticity, linearity checks).</li>
</ul>
<h3>4. What the user decides</h3>
<ul>
<li>Which covariates to include for adjustment.</li>
<li>Whether to run subgroup-filtered models.</li>
</ul>
""",
    },
    {
        "id": "logistic_regression",
        "title": "Logistic Regression",
        "summary": "Binary outcome (0/1) with numeric/categorical predictors.",
        "keywords": ["logistic", "binary", "odds ratio"],
        "example_tsv": "Event\tBiomarker\tAge\n0\t1.8\t59\n1\t2.7\t63\n0\t2.0\t57\n1\t3.1\t68",
        "html": """
<h2>Recipe: Logistic Regression</h2>
<h3>1. Data layout</h3>
<p>You need a binary outcome column coded as 0/1 and one or more predictors.</p>
<table border='1' cellspacing='0' cellpadding='4'>
<tr><th>Event</th><th>Biomarker</th><th>Age</th></tr>
<tr><td>0</td><td>1.8</td><td>59</td></tr>
<tr><td>1</td><td>3.1</td><td>68</td></tr>
</table>
<h3>2. Bucket mapping</h3>
<ul>
<li><b>Dependent Variable:</b> Event (binary 0/1)</li>
<li><b>Factor 1:</b> Biomarker (or main predictor)</li>
<li><b>Covariates:</b> additional predictors such as Age</li>
</ul>
<h3>3. What the app does automatically</h3>
<ul>
<li>Detects binary outcome and follows logistic model path.</li>
<li>Reports model coefficients and odds-ratio style inference outputs.</li>
</ul>
<h3>4. What the user decides</h3>
<ul>
<li>Predictor/covariate set to include.</li>
<li>Optional subgroup filtering before model fitting.</li>
</ul>
""",
    },
]
