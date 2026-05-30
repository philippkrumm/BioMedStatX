"""Static recipe content for the in-app Help Hub."""

from __future__ import annotations


HELP_RECIPES = [
    {
        "id": "getting_started",
        "title": "▶  Getting Started — Read This First",
        "summary": "What are the buckets? What does long format mean? Start here.",
        "keywords": ["start", "bucket", "factor", "dependent variable", "subject id", "covariate", "format", "long", "wide", "beginner"],
        "html": """
<h2>Getting Started</h2>
<p>BioMedStatX works through <b>six drag-and-drop buckets</b> in the center of the screen. You drag your column names into these buckets to tell the app what role each column plays. The app then selects the right statistical test automatically — you never pick a test manually.</p>

<h3>The six buckets — in plain English</h3>
<table>
<tr><th>Bucket</th><th>What it means</th><th>Real-world example</th></tr>
<tr><td><b>Dependent Variable</b></td><td>The number you measured. The outcome you care about.</td><td>Blood pressure, body weight, cell count, test score</td></tr>
<tr><td><b>Factor 1</b></td><td>The main thing you are comparing or the main predictor.<br>If you have groups → drag the group column here.<br>If you want to see a relationship between two numbers → drag the predictor here.</td><td>Group (Control / Treatment), Genotype (WT / KO), Age, Dosage</td></tr>
<tr><td><b>Factor 2</b></td><td>A second grouping column. Only needed when you have two separate ways of splitting your data at once.</td><td>Sex (Male / Female), Diet (Low fat / High fat), Timepoint (Pre / Post)</td></tr>
<tr><td><b>Subject ID</b></td><td>Who the measurement belongs to. Only needed when the same person or animal appears more than once in your data.</td><td>PatientID (P001, P002 ...), MouseID, Participant_Number</td></tr>
<tr><td><b>Covariates</b></td><td>A background variable you want to mathematically correct for before comparing groups. You do not interpret it directly — you just want to remove its influence from the result.</td><td>Age or Baseline_Blood_Pressure — when you know they differ between your groups and might distort the comparison</td></tr>
<tr><td><b>Filter</b></td><td>Restricts the entire analysis to one subgroup only.</td><td>Analyse only male patients; analyse only older mice</td></tr>
</table>

<h3>The single most important rule: one row = one measurement</h3>
<p>BioMedStatX expects your data in what statisticians call <b>long format</b>: <b>each row is one measurement from one subject</b>. Most people are used to wide format, where each condition has its own column. Wide format does not work here.</p>

<table>
<tr><th colspan="2">✅ Correct — long format</th></tr>
<tr><th>Group</th><th>Blood_Pressure</th></tr>
<tr><td>Control</td><td>120</td></tr>
<tr><td>Control</td><td>118</td></tr>
<tr><td>Treatment_A</td><td>135</td></tr>
<tr><td>Treatment_A</td><td>142</td></tr>
<tr><td>Treatment_B</td><td>145</td></tr>
<tr><td>Treatment_B</td><td>150</td></tr>
</table>

<table>
<tr><th colspan="3">❌ Wrong — wide format (one column per group)</th></tr>
<tr><th>Control</th><th>Treatment_A</th><th>Treatment_B</th></tr>
<tr><td>120</td><td>135</td><td>145</td></tr>
<tr><td>118</td><td>142</td><td>150</td></tr>
<tr><td>122</td><td>138</td><td>148</td></tr>
<tr><td>119</td><td>140</td><td>144</td></tr>
<tr><td>121</td><td>141</td><td>147</td></tr>
</table>

<p><b>Why?</b> In long format, there is a "Group" column that the app can map to Factor 1. In wide format, the group names are hidden inside the column headers — the app cannot extract them from there.</p>

<h3>How the app decides which test to run</h3>
<p>You never pick a statistical test. The app decides based on what you drag where:</p>
<ul>
<li>Factor 1 = group labels (Control / Treatment / ...) → <b>t-Test or ANOVA</b></li>
<li>Factor 1 = numbers (age, dosage, ...) → <b>Correlation or Regression</b></li>
<li>Factor 1 + Factor 2 both filled → <b>Two-Way ANOVA or Mixed ANOVA</b></li>
<li>Subject ID filled → <b>paired or repeated-measures design</b></li>
<li>Covariates filled → <b>ANCOVA or Multiple Regression</b></li>
<li>Outcome has exactly two values (0/1 or Yes/No) → <b>Logistic Regression</b></li>
</ul>
<p>The grey status line below the buckets always shows which test would run right now, before you click Start.</p>
""",
    },
    {
        "id": "one_way_anova",
        "title": "Comparing groups (t-Test / One-Way ANOVA)",
        "summary": "Are 2 or more separate, independent groups different from each other?",
        "keywords": ["anova", "one-way", "t-test", "group", "factor 1", "independent", "between", "compare"],
        "html": """
<h2>Comparing independent groups — t-Test or One-Way ANOVA</h2>

<h3>When do you use this?</h3>
<p>You have <b>two or more groups</b>. Each person or animal is in <b>exactly one group</b>. You want to know: <b>are the measured values different between groups?</b></p>
<p>Examples: "Do Control, Treatment A, and Treatment B mice have different body weights?" &nbsp;·&nbsp; "Is there a blood pressure difference between three genotypes?" &nbsp;·&nbsp; "Do patients from two hospitals differ in recovery time?"</p>
<p>The app uses a t-Test automatically when there are exactly 2 groups, and ANOVA when there are 3 or more. You do not choose — it happens automatically.</p>

<h3>What your data must look like</h3>
<p>One row per measurement. Two columns minimum: one with the group label, one with the measured value.</p>
<table>
<tr><th>Group</th><th>Blood_Pressure</th></tr>
<tr><td>Control</td><td>120</td></tr>
<tr><td>Control</td><td>118</td></tr>
<tr><td>Control</td><td>122</td></tr>
<tr><td>Treatment_A</td><td>135</td></tr>
<tr><td>Treatment_A</td><td>142</td></tr>
<tr><td>Treatment_A</td><td>138</td></tr>
<tr><td>Treatment_B</td><td>145</td></tr>
<tr><td>Treatment_B</td><td>150</td></tr>
<tr><td>Treatment_B</td><td>148</td></tr>
</table>

<h3>Common mistake — groups as column names</h3>
<table>
<tr><th>Control</th><th>Treatment_A</th><th>Treatment_B</th></tr>
<tr><td>120</td><td>135</td><td>145</td></tr>
<tr><td>118</td><td>142</td><td>150</td></tr>
<tr><td>122</td><td>138</td><td>148</td></tr>
<tr><td>119</td><td>140</td><td>144</td></tr>
<tr><td>121</td><td>141</td><td>147</td></tr>
</table>
<p><b>Why this fails:</b> "Control", "Treatment_A", and "Treatment_B" are the group names — they should appear as values inside a single column, not as column headers.</p>

<h3>What to drag where</h3>
<table>
<tr><th>Bucket</th><th>What to drag here</th><th>In this example</th></tr>
<tr><td><b>Dependent Variable</b></td><td>Your measurement column — the numbers you measured</td><td>Blood_Pressure</td></tr>
<tr><td><b>Factor 1</b></td><td>Your group label column — the column that says which group each row belongs to</td><td>Group</td></tr>
<tr><td>Factor 2</td><td>Leave empty</td><td>—</td></tr>
<tr><td>Subject ID</td><td>Leave empty — nobody is measured twice in this design</td><td>—</td></tr>
<tr><td>Covariates</td><td>Leave empty</td><td>—</td></tr>
</table>

<h3>Before you click Start — checklist</h3>
<ul>
<li>One row per measurement — not one row per subject with multiple columns.</li>
<li>Group names are spelled identically across rows. "Control" and "control" are treated as different groups.</li>
<li>The measurement column contains only numbers — no units, no text mixed in.</li>
<li>No subject appears more than once. If they do → use Repeated Measures ANOVA instead.</li>
</ul>
""",
    },
    {
        "id": "two_way_anova",
        "title": "Two independent grouping factors (Two-Way ANOVA)",
        "summary": "Two separate ways of grouping — e.g. Treatment AND Sex.",
        "keywords": ["anova", "two-way", "factor 2", "interaction", "between", "crossed", "two factors"],
        "html": """
<h2>Two-Way ANOVA — two independent grouping factors</h2>

<h3>When do you use this?</h3>
<p>You have <b>two separate ways of grouping your subjects</b> — for example Treatment (Control / Drug) and Sex (Male / Female) — and every combination of the two has been measured. You want to know: does each factor have an effect, and do they interact (i.e. does the treatment work differently in males vs. females)?</p>
<p>Examples: "Does the treatment effect depend on sex?" &nbsp;·&nbsp; "Does diet interact with exercise level to affect weight loss?"</p>
<p>No subject appears more than once. If the same subjects are measured at multiple time points or conditions, use Mixed ANOVA instead.</p>

<h3>What your data must look like</h3>
<p>One row per subject. <b>Both</b> group columns are present in every row.</p>
<table>
<tr><th>Treatment</th><th>Sex</th><th>Score</th></tr>
<tr><td>Control</td><td>Male</td><td>45</td></tr>
<tr><td>Control</td><td>Male</td><td>48</td></tr>
<tr><td>Control</td><td>Male</td><td>46</td></tr>
<tr><td>Control</td><td>Female</td><td>42</td></tr>
<tr><td>Control</td><td>Female</td><td>44</td></tr>
<tr><td>Control</td><td>Female</td><td>41</td></tr>
<tr><td>Drug</td><td>Male</td><td>65</td></tr>
<tr><td>Drug</td><td>Male</td><td>62</td></tr>
<tr><td>Drug</td><td>Female</td><td>85</td></tr>
<tr><td>Drug</td><td>Female</td><td>88</td></tr>
</table>

<h3>Common mistake — both factors hidden in column names</h3>
<table>
<tr><th>SubjectID</th><th>Control_Male</th><th>Control_Female</th><th>Drug_Male</th><th>Drug_Female</th></tr>
<tr><td>S01</td><td>45</td><td>42</td><td>65</td><td>85</td></tr>
<tr><td>S02</td><td>48</td><td>44</td><td>62</td><td>88</td></tr>
<tr><td>S03</td><td>46</td><td>43</td><td>64</td><td>86</td></tr>
<tr><td>S04</td><td>44</td><td>41</td><td>66</td><td>89</td></tr>
</table>
<p><b>Why this fails:</b> Treatment and Sex are mixed into the column headers. The app cannot separate them into Factor 1 and Factor 2.</p>

<h3>What to drag where</h3>
<table>
<tr><th>Bucket</th><th>What to drag here</th><th>In this example</th></tr>
<tr><td><b>Dependent Variable</b></td><td>Your measurement column</td><td>Score</td></tr>
<tr><td><b>Factor 1</b></td><td>First group label column</td><td>Treatment</td></tr>
<tr><td><b>Factor 2</b></td><td>Second group label column</td><td>Sex</td></tr>
<tr><td>Subject ID</td><td>Leave empty — each subject appears only once</td><td>—</td></tr>
<tr><td>Covariates</td><td>Leave empty</td><td>—</td></tr>
</table>

<h3>Before you click Start — checklist</h3>
<ul>
<li>Both group columns contain text labels, not numbers.</li>
<li>Every row has a value in both group columns.</li>
<li>Each subject appears exactly once in the entire dataset.</li>
<li>No group name is embedded in a column header.</li>
</ul>
""",
    },
    {
        "id": "repeated_measures_anova",
        "title": "Same subjects measured multiple times (Repeated Measures ANOVA)",
        "summary": "The same people or animals measured at several time points — one group only.",
        "keywords": ["repeated", "within", "subject", "timepoint", "longitudinal", "pre post", "same subjects", "one group"],
        "html": """
<h2>Repeated Measures ANOVA — same subjects, multiple measurements</h2>

<h3>When do you use this?</h3>
<p>The <b>same subjects are measured at multiple time points or conditions</b>, and all subjects belong to a <b>single group</b>. You want to know: does the measurement change over time or across conditions?</p>
<p>Examples: "Does blood pressure change from Baseline to Week 4 to Week 8 in our patients?" &nbsp;·&nbsp; "Does heart rate change across three exercise intensities in the same athletes?"</p>
<p><b>Key distinction:</b> all subjects are in one group only. If they are also split into different groups (e.g. Treatment vs. Control), use Mixed ANOVA instead.</p>

<h3>What your data must look like</h3>
<p>One row per measurement. Three columns: subject identifier, the time point or condition label, and the measured value. Each subject appears once per time point.</p>
<table>
<tr><th>SubjectID</th><th>Timepoint</th><th>Blood_Pressure</th></tr>
<tr><td>P001</td><td>Baseline</td><td>140</td></tr>
<tr><td>P001</td><td>Week_4</td><td>130</td></tr>
<tr><td>P001</td><td>Week_8</td><td>125</td></tr>
<tr><td>P002</td><td>Baseline</td><td>145</td></tr>
<tr><td>P002</td><td>Week_4</td><td>135</td></tr>
<tr><td>P002</td><td>Week_8</td><td>132</td></tr>
<tr><td>P003</td><td>Baseline</td><td>142</td></tr>
<tr><td>P003</td><td>Week_4</td><td>133</td></tr>
<tr><td>P003</td><td>Week_8</td><td>128</td></tr>
</table>
<p>Notice: P001 appears three times — once per time point. That is correct and expected.</p>

<h3>Common mistake — one column per time point</h3>
<table>
<tr><th>SubjectID</th><th>Baseline</th><th>Week_4</th><th>Week_8</th></tr>
<tr><td>P001</td><td>140</td><td>130</td><td>125</td></tr>
<tr><td>P002</td><td>145</td><td>135</td><td>132</td></tr>
<tr><td>P003</td><td>142</td><td>133</td><td>128</td></tr>
<tr><td>P004</td><td>138</td><td>128</td><td>122</td></tr>
<tr><td>P005</td><td>144</td><td>134</td><td>130</td></tr>
</table>
<p><b>Why this fails:</b> The time points must be values inside a "Timepoint" column. When they are column headers, the app cannot map Timepoint to Factor 1.</p>

<h3>What to drag where</h3>
<table>
<tr><th>Bucket</th><th>What to drag here</th><th>In this example</th></tr>
<tr><td><b>Dependent Variable</b></td><td>Your measurement column</td><td>Blood_Pressure</td></tr>
<tr><td><b>Factor 1</b></td><td>The time point or condition column — the thing that changes across rows for the same subject</td><td>Timepoint</td></tr>
<tr><td><b>Subject ID</b></td><td>The column that says who each measurement belongs to</td><td>SubjectID</td></tr>
<tr><td>Factor 2</td><td>Leave empty — subjects are all in one group here</td><td>—</td></tr>
<tr><td>Covariates</td><td>Leave empty</td><td>—</td></tr>
</table>

<h3>Before you click Start — checklist</h3>
<ul>
<li>Each subject appears exactly once per time point. No duplicate SubjectID + Timepoint combinations.</li>
<li>Ideally, all subjects have measurements at all time points. If &gt;5% of subjects have missing data, the app automatically switches to a robust Linear Mixed Model (LMM) instead of excluding subjects (listwise deletion).</li>
<li>The Timepoint column contains text labels — "Baseline", "Week_4" — not numbers like 0, 4, 8.</li>
<li>Subject ID values repeat across rows — this is correct and expected.</li>
</ul>
""",
    },
    {
        "id": "mixed_anova",
        "title": "Different groups, each measured multiple times (Mixed ANOVA)",
        "summary": "Treatment vs. Control AND multiple time points — the same subjects within each group.",
        "keywords": ["mixed", "between", "within", "subject id", "longitudinal", "group", "timepoint", "repeated", "groups over time"],
        "html": """
<h2>Mixed ANOVA — groups × repeated measurements</h2>

<h3>When do you use this?</h3>
<p>You have <b>two or more groups</b>, and the <b>same subjects within each group are measured multiple times</b>. You want to know: does the measurement change over time, does it differ between groups, and — most interestingly — does the change over time look different across groups?</p>
<p>Examples: "Do Treatment and Control patients show different recovery trajectories from Baseline to Week 8?" &nbsp;·&nbsp; "Do WT and KO mice respond differently across three dose levels?"</p>

<h3>What your data must look like</h3>
<p>One row per measurement. Four columns: subject identifier, the independent group, the time point or condition, and the measurement value.</p>
<table>
<tr><th>SubjectID</th><th>Group</th><th>Timepoint</th><th>Blood_Pressure</th></tr>
<tr><td>P001</td><td>Control</td><td>Baseline</td><td>140</td></tr>
<tr><td>P001</td><td>Control</td><td>Week_8</td><td>138</td></tr>
<tr><td>P002</td><td>Control</td><td>Baseline</td><td>145</td></tr>
<tr><td>P002</td><td>Control</td><td>Week_8</td><td>142</td></tr>
<tr><td>P010</td><td>Treatment</td><td>Baseline</td><td>142</td></tr>
<tr><td>P010</td><td>Treatment</td><td>Week_8</td><td>120</td></tr>
<tr><td>P011</td><td>Treatment</td><td>Baseline</td><td>144</td></tr>
<tr><td>P011</td><td>Treatment</td><td>Week_8</td><td>122</td></tr>
</table>
<p>P001 appears twice (Baseline + Week_8). Their Group (Control) stays the same across both rows — that is correct.</p>

<h3>Common mistake — time points as separate columns</h3>
<table>
<tr><th>SubjectID</th><th>Group</th><th>Baseline</th><th>Week_8</th></tr>
<tr><td>P001</td><td>Control</td><td>140</td><td>138</td></tr>
<tr><td>P002</td><td>Control</td><td>145</td><td>142</td></tr>
<tr><td>P010</td><td>Treatment</td><td>142</td><td>120</td></tr>
<tr><td>P011</td><td>Treatment</td><td>144</td><td>122</td></tr>
<tr><td>P012</td><td>Treatment</td><td>146</td><td>125</td></tr>
</table>
<p><b>Why this fails:</b> Timepoint must be a single column with the labels as values — not separate columns for each time point.</p>

<h3>What to drag where</h3>
<table>
<tr><th>Bucket</th><th>What to drag here</th><th>In this example</th></tr>
<tr><td><b>Dependent Variable</b></td><td>Your measurement column</td><td>Blood_Pressure</td></tr>
<tr><td><b>Factor 1</b></td><td>The <i>repeated</i> factor — the thing that changes across rows for the same subject (time point, condition)</td><td>Timepoint</td></tr>
<tr><td><b>Factor 2</b></td><td>The <i>independent</i> group — subjects are fully in one group only</td><td>Group</td></tr>
<tr><td><b>Subject ID</b></td><td>The column that identifies each subject</td><td>SubjectID</td></tr>
<tr><td>Covariates</td><td>Leave empty</td><td>—</td></tr>
</table>
<p><b>Important:</b> Factor 1 = the repeated/within factor (time points). Factor 2 = the between-subjects group (Treatment / Control). Getting them swapped reverses the labels in the output.</p>

<h3>Before you click Start — checklist</h3>
<ul>
<li>Each subject appears exactly once per time point.</li>
<li>A subject's Group value is identical across all their rows.</li>
<li>Ideally, all subjects have measurements at all time points. If &gt;5% of subjects have missing data, the app automatically switches to a robust Linear Mixed Model (LMM).</li>
<li>Subject ID, Factor 1 (within), and Factor 2 (between) are all filled — this is what distinguishes Mixed ANOVA from Two-Way ANOVA.</li>
</ul>
""",
    },
    {
        "id": "ancova",
        "title": "Comparing groups while correcting for a background variable (ANCOVA)",
        "summary": "Like ANOVA, but you control for an additional numeric variable that might distort results.",
        "keywords": ["ancova", "covariate", "baseline", "adjust", "correct", "confound", "control for"],
        "html": """
<h2>ANCOVA — correcting for a background variable before comparing groups</h2>

<h3>When do you use this?</h3>
<p>You want to compare groups, but you suspect that another variable — one you did not control — differs between your groups and may distort the comparison. ANCOVA adjusts the group averages mathematically to account for the background variable, allowing for a fairer comparison.</p>
<p><b>Important:</b> The covariate must not be affected by the treatment itself (e.g. do not use weight after treatment as a covariate). Otherwise, you risk mathematically removing the actual treatment effect (overadjustment bias).</p>
<p>Example: You compare test scores between Treatment and Control groups. But Control patients happened to be older than Treatment patients. A simple ANOVA would partly reflect that pre-existing age difference, not just the treatment effect. Putting Age into Covariates corrects for it.</p>
<p>The correcting variable goes into the <b>Covariates</b> bucket. It must be a number (not a group label).</p>

<h3>What your data must look like</h3>
<p>One row per subject. One measurement column (outcome), one group column, and one or more numeric covariate columns.</p>
<table>
<tr><th>Group</th><th>Score_post</th><th>Score_baseline</th><th>Age</th></tr>
<tr><td>Control</td><td>45</td><td>42</td><td>62</td></tr>
<tr><td>Control</td><td>48</td><td>40</td><td>71</td></tr>
<tr><td>Control</td><td>46</td><td>41</td><td>65</td></tr>
<tr><td>Treatment</td><td>65</td><td>45</td><td>65</td></tr>
<tr><td>Treatment</td><td>70</td><td>48</td><td>58</td></tr>
<tr><td>Treatment</td><td>68</td><td>46</td><td>61</td></tr>
</table>

<h3>Common mistake — pre-computed group means</h3>
<table>
<tr><th>Group</th><th>Mean_Score_post</th><th>Mean_Age</th></tr>
<tr><td>Control</td><td>46.3</td><td>66.0</td></tr>
<tr><td>Treatment</td><td>67.6</td><td>61.3</td></tr>
</table>
<p><b>Why this fails:</b> ANCOVA needs the raw individual values to estimate how much the covariate influences the outcome. Group means destroy that information.</p>

<h3>What to drag where</h3>
<table>
<tr><th>Bucket</th><th>What to drag here</th><th>In this example</th></tr>
<tr><td><b>Dependent Variable</b></td><td>Your outcome measurement</td><td>Score_post</td></tr>
<tr><td><b>Factor 1</b></td><td>Your group label column</td><td>Group</td></tr>
<tr><td><b>Covariates</b></td><td>The numeric variable(s) to correct for</td><td>Score_baseline, Age</td></tr>
<tr><td>Factor 2</td><td>Leave empty (unless you also have a second group factor → Two-Way ANCOVA)</td><td>—</td></tr>
<tr><td>Subject ID</td><td>Leave empty — each subject appears once</td><td>—</td></tr>
</table>

<h3>Before you click Start — checklist</h3>
<ul>
<li>One row per subject — not one row per time point.</li>
<li>Covariates are number columns, not group label columns.</li>
<li>Factor 1 is a group label column, not a number.</li>
<li>Raw individual values — not pre-aggregated group means.</li>
</ul>
""",
    },
    {
        "id": "correlation",
        "title": "Do two measurements go up and down together? (Correlation)",
        "summary": "Measure the relationship between two continuous variables. No groups.",
        "keywords": ["correlation", "pearson", "spearman", "relationship", "continuous", "scatter", "association"],
        "html": """
<h2>Correlation — do two measurements move together?</h2>

<h3>When do you use this?</h3>
<p>You have <b>two number columns</b> — one measurement per subject in each — and you want to know: <b>when one goes up, does the other tend to go up (or down) too?</b> There are no groups. Both columns are numbers.</p>
<p>Examples: "Is dosage related to blood pressure after treatment?" &nbsp;·&nbsp; "Does protein expression correlate with tumour size?" &nbsp;·&nbsp; "Is age associated with recovery speed?"</p>
<p>Correlation measures the strength and direction of a relationship. It does not give you a prediction formula and does not prove causation. For a specific slope or for controlling additional variables, use Regression instead.</p>

<h3>What your data must look like</h3>
<p>One row per subject. Two numeric columns — one for each variable.</p>
<table>
<tr><th>Dosage_mg</th><th>Blood_Pressure</th></tr>
<tr><td>10</td><td>140</td></tr>
<tr><td>20</td><td>135</td></tr>
<tr><td>30</td><td>125</td></tr>
<tr><td>40</td><td>115</td></tr>
<tr><td>50</td><td>110</td></tr>
<tr><td>60</td><td>102</td></tr>
<tr><td>70</td><td>98</td></tr>
</table>

<h3>Common mistake — averaged groups instead of individual values</h3>
<table>
<tr><th>Dosage_category</th><th>Mean_Blood_Pressure</th></tr>
<tr><td>Low</td><td>138</td></tr>
<tr><td>Medium</td><td>125</td></tr>
<tr><td>High</td><td>112</td></tr>
</table>
<p><b>Why this fails:</b> Correlation needs one paired value per subject. Averaging into groups changes the apparent strength of the relationship and loses statistical power.</p>

<h3>What to drag where</h3>
<table>
<tr><th>Bucket</th><th>What to drag here</th><th>In this example</th></tr>
<tr><td><b>Dependent Variable</b></td><td>The outcome — the measurement you want to explain</td><td>Blood_Pressure</td></tr>
<tr><td><b>Factor 1</b></td><td>The predictor — the other numeric variable</td><td>Dosage_mg</td></tr>
<tr><td>Covariates</td><td>Leave empty — adding anything here switches to Regression</td><td>—</td></tr>
<tr><td>Subject ID</td><td>Leave empty — adding this switches to a Mixed Model</td><td>—</td></tr>
<tr><td>Factor 2, Filter</td><td>Not needed for basic correlation</td><td>—</td></tr>
</table>
<p>The app picks Pearson or Spearman automatically based on whether the data is normally distributed. You do not need to choose.</p>

<h3>Before you click Start — checklist</h3>
<ul>
<li>Both columns are numbers — no group labels, no text.</li>
<li>One row per subject — no repeated measurements.</li>
<li>Raw individual values — no pre-computed averages or bins.</li>
<li>Covariates and Subject ID are both empty.</li>
</ul>
""",
    },
    {
        "id": "linear_regression",
        "title": "Predicting a measurement from other measurements (Linear Regression)",
        "summary": "How much does the outcome change per unit of the predictor? Add control variables.",
        "keywords": ["regression", "ols", "predict", "covariates", "beta", "coefficient", "slope", "linear"],
        "html": """
<h2>Linear Regression — predicting one measurement from others</h2>

<h3>When do you use this?</h3>
<p>You want to know: <b>how much does the outcome change for each unit increase in the predictor — and by exactly how much?</b> Unlike correlation, regression gives you a specific number (the slope). You can also include additional variables to control for — which correlation cannot do.</p>
<p>Examples: "For every additional mg of dosage, how much does blood pressure drop?" &nbsp;·&nbsp; "Predict test score from study hours, controlling for age and baseline score."</p>

<h3>Two ways to trigger regression</h3>
<ul>
<li><b>Simple regression</b> (one predictor, no additional controls): Drag only Factor 1. Then tick the checkbox <i>"Analyse as Linear Regression"</i> that appears below the buckets. Without that tick, the app runs Correlation instead.</li>
<li><b>Multiple regression</b> (predictor + control variables): Drag Factor 1 and one or more Covariates. The Regression mode activates automatically when Covariates is populated.</li>
</ul>

<h3>Variable Transformations</h3>
<p>When regression mode is active, you will see dropdowns to transform your X or Y variables (e.g. into log10 or square root). Use this if your data is highly skewed or heteroscedastic. <b>Important:</b> Transforming changes how you interpret the result! If you log-transform Y, the effect is multiplicative rather than additive. If you log-transform <b>both X and Y (Log-Log)</b>, β represents an <i>elasticity</i> (a 1% increase in X yields a β% change in Y).</p>

<h3>What your data must look like</h3>
<p>One row per subject. One numeric outcome column, one or more numeric predictor columns.</p>
<table>
<tr><th>Dosage_mg</th><th>Blood_Pressure</th><th>Age</th><th>BP_baseline</th></tr>
<tr><td>10</td><td>140</td><td>62</td><td>145</td></tr>
<tr><td>20</td><td>135</td><td>71</td><td>142</td></tr>
<tr><td>30</td><td>125</td><td>59</td><td>130</td></tr>
<tr><td>40</td><td>115</td><td>70</td><td>125</td></tr>
<tr><td>50</td><td>110</td><td>65</td><td>120</td></tr>
<tr><td>60</td><td>102</td><td>68</td><td>112</td></tr>
<tr><td>70</td><td>98</td><td>63</td><td>108</td></tr>
</table>

<h3>What to drag where</h3>
<table>
<tr><th>Bucket</th><th>What to drag here</th><th>In this example</th></tr>
<tr><td><b>Dependent Variable</b></td><td>The outcome you want to predict</td><td>Blood_Pressure</td></tr>
<tr><td><b>Factor 1</b></td><td>The main numeric predictor</td><td>Dosage_mg</td></tr>
<tr><td><b>Covariates</b></td><td>Additional numeric variables to control for</td><td>Age, BP_baseline</td></tr>
<tr><td>Factor 2</td><td>Leave empty</td><td>—</td></tr>
<tr><td>Subject ID</td><td>Leave empty — each subject appears once</td><td>—</td></tr>
</table>

<h3>Reading the result</h3>
<p>The main output is the <b>coefficient (β)</b> for each predictor. If β = −2.5 for Dosage_mg, it means: for every additional mg of dosage, blood pressure decreases by 2.5 points on average — holding age and baseline BP constant.</p>

<h3>Before you click Start — checklist</h3>
<ul>
<li>All predictor and covariate columns are numbers.</li>
<li>The outcome column is a number.</li>
<li>One row per subject.</li>
<li>No column appears in both Factor 1 and Covariates simultaneously.</li>
</ul>
""",
    },
    {
        "id": "logistic_regression",
        "title": "Predicting a yes/no outcome (Logistic Regression)",
        "summary": "Your outcome is binary — 0/1, yes/no, event/no-event.",
        "keywords": ["logistic", "binary", "0/1", "yes no", "event", "odds ratio", "predict", "complication", "survival"],
        "html": """
<h2>Logistic Regression — predicting a yes/no outcome</h2>

<h3>When do you use this?</h3>
<p>Your outcome can only be <b>one of two values</b>: yes/no, 0/1, survived/died, complication/no-complication. You want to know which measurements predict that outcome.</p>
<p>Examples: "Which pre-operative measurements predict post-operative complications (yes/no)?" &nbsp;·&nbsp; "Does dosage predict whether a patient recovers completely (0/1)?"</p>
<p>The app detects this automatically: if your Dependent Variable column has exactly two distinct values, logistic regression runs without any manual selection.</p>

<h3>What your data must look like</h3>
<p>One row per subject. The outcome column must have exactly two values — 0 and 1 is the clearest format. Text labels (e.g. "Yes" / "No") also work, as long as there are only two.</p>
<table>
<tr><th>Side_Effect</th><th>Dosage_mg</th><th>Age</th><th>BP_baseline</th></tr>
<tr><td>0</td><td>10</td><td>62</td><td>145</td></tr>
<tr><td>1</td><td>40</td><td>71</td><td>120</td></tr>
<tr><td>0</td><td>20</td><td>59</td><td>135</td></tr>
<tr><td>1</td><td>50</td><td>68</td><td>115</td></tr>
<tr><td>0</td><td>30</td><td>63</td><td>125</td></tr>
<tr><td>1</td><td>60</td><td>72</td><td>110</td></tr>
<tr><td>1</td><td>70</td><td>65</td><td>108</td></tr>
</table>

<h3>Common mistake — outcome has more than two values</h3>
<table>
<tr><th>Severity</th><th>Dosage_mg</th></tr>
<tr><td>None</td><td>10</td></tr>
<tr><td>Mild</td><td>30</td></tr>
<tr><td>Severe</td><td>50</td></tr>
</table>
<p><b>Why this fails:</b> Three outcome levels (None / Mild / Severe) cannot be handled by standard logistic regression. Consider collapsing to two levels first — e.g. None vs. Any side effect.</p>

<h3>What to drag where</h3>
<table>
<tr><th>Bucket</th><th>What to drag here</th><th>In this example</th></tr>
<tr><td><b>Dependent Variable</b></td><td>The binary outcome column (0/1 or two-level text)</td><td>Side_Effect</td></tr>
<tr><td><b>Factor 1</b></td><td>The main numeric predictor</td><td>Dosage_mg</td></tr>
<tr><td><b>Covariates</b></td><td>Additional predictors to include in the model</td><td>Age, BP_baseline</td></tr>
<tr><td>Factor 2</td><td>Leave empty</td><td>—</td></tr>
<tr><td>Subject ID</td><td>Leave empty — each subject appears once</td><td>—</td></tr>
</table>

<h3>Reading the result</h3>
<p>The main output is the <b>Odds Ratio (OR)</b> per predictor. OR = 2.5 means: for every one-unit increase in that predictor, the odds of the event are 2.5 times higher. OR &lt; 1 means the predictor reduces the odds. OR = 1 means no effect.</p>
<p>The AUC (area under the ROC curve) tells you how well the model discriminates between the two outcomes: 0.5 = no better than chance; 0.70–0.80 = acceptable; above 0.80 = good.</p>

<h3>Before you click Start — checklist</h3>
<ul>
<li>Dependent Variable has exactly two distinct values.</li>
<li>All predictors and covariates are numbers.</li>
<li>One row per subject.</li>
<li>Rule of thumb: at least 10 events (rows where outcome = 1) per predictor included in the model.</li>
</ul>
""",
    },
]
