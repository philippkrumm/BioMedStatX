"""Static recipe content for the in-app Help Hub."""

from __future__ import annotations

CATEGORY_ORDER = [
    "Start here",
    "Choosing a test",
    "Concepts",
    "Workflow & Output",
]

HELP_RECIPES = [
    {
        "id": "getting_started",
        "category": "Start here",
        "title": "Getting started: read this first",
        "summary": "What are the buckets? What does long format mean? Start here.",
        "keywords": ["start", "bucket", "factor", "dependent variable", "subject id", "covariate", "format", "long", "wide", "beginner"],
        "html": """
<h2>Getting started: read this first</h2>
<p>BioMedStatX works through <b>six drag-and-drop buckets</b> in the center of the screen. You drag your column names into these buckets to tell the app what role each column plays. The app then selects the right statistical test automatically. You never pick a test manually.</p>

<h3>The six buckets, in plain English</h3>
<table>
<tr><th>Bucket</th><th>What it means</th><th>Real-world example</th></tr>
<tr><td><b>Dependent Variable</b></td><td>The number you measured. The outcome you care about.</td><td>Blood pressure, body weight, cell count, test score</td></tr>
<tr><td><b>Factor 1</b></td><td>The main thing you are comparing or the main predictor.<br>If you have groups, drag the group column here.<br>If you want to see a relationship between two numbers, drag the predictor here.</td><td>Group (Control / Treatment), Genotype (WT / KO), Age, Dosage</td></tr>
<tr><td><b>Factor 2</b></td><td>A second grouping column. Only needed when you have two separate ways of splitting your data at once.</td><td>Sex (Male / Female), Diet (Low fat / High fat), Timepoint (Pre / Post)</td></tr>
<tr><td><b>Subject ID</b></td><td>Who the measurement belongs to. Only needed when the same person or animal appears more than once in your data.</td><td>PatientID (P001, P002 ...), MouseID, Participant_Number</td></tr>
<tr><td><b>Covariates</b></td><td>A background variable you want to mathematically correct for before comparing groups. You do not interpret it directly. You just want to remove its influence from the result.</td><td>Age or Baseline_Blood_Pressure, when you know they differ between your groups and might distort the comparison</td></tr>
<tr><td><b>Filter</b></td><td>Restricts the entire analysis to one subgroup only.</td><td>Analyse only male patients; analyse only older mice</td></tr>
</table>

<h3>The single most important rule: one row = one measurement</h3>
<p>BioMedStatX expects your data in what statisticians call <b>long format</b>: <b>each row is one measurement from one subject</b>. Most people are used to wide format, where each condition has its own column. For group comparisons, wide format does not work: put your group labels in a column, not in the headers.</p>
<p>There is one exception. If your file looks like a paired or repeated-measures layout, one subject ID column plus a few numeric condition columns, and no group column, the app melts it to long format for you automatically and tells you it did so. For everything else you arrange the data in long format yourself.</p>

<table>
<tr><th colspan="2">Correct: long format</th></tr>
<tr><th>Group</th><th>Blood_Pressure</th></tr>
<tr><td>Control</td><td>120</td></tr>
<tr><td>Control</td><td>118</td></tr>
<tr><td>Treatment_A</td><td>135</td></tr>
<tr><td>Treatment_A</td><td>142</td></tr>
<tr><td>Treatment_B</td><td>145</td></tr>
<tr><td>Treatment_B</td><td>150</td></tr>
</table>

<table>
<tr><th colspan="3">Wrong: wide format (one column per group)</th></tr>
<tr><th>Control</th><th>Treatment_A</th><th>Treatment_B</th></tr>
<tr><td>120</td><td>135</td><td>145</td></tr>
<tr><td>118</td><td>142</td><td>150</td></tr>
<tr><td>122</td><td>138</td><td>148</td></tr>
<tr><td>119</td><td>140</td><td>144</td></tr>
<tr><td>121</td><td>141</td><td>147</td></tr>
</table>

<p><b>Why?</b> In long format, there is a "Group" column that the app can map to Factor 1. In wide format, the group names are hidden inside the column headers, so the app cannot extract them from there.</p>

<h3>How the app decides which test to run</h3>
<p>You never pick a statistical test. The app decides based on what you drag where:</p>
<ul>
<li>Factor 1 = group labels (Control / Treatment / ...) gives a <b>t-Test or ANOVA</b></li>
<li>Factor 1 = numbers (age, dosage, ...) gives a <b>Correlation or Regression</b></li>
<li>Factor 1 and Factor 2 both filled gives a <b>Two-Way ANOVA or Mixed ANOVA</b></li>
<li>Subject ID filled gives a <b>paired or repeated-measures design</b></li>
<li>Covariates filled gives an <b>ANCOVA or Multiple Regression</b></li>
<li>Outcome has exactly two values (0/1 or Yes/No) gives <b>Logistic Regression</b></li>
</ul>
<p>The status line below the buckets checks your mapping each time you change it. When it says the mapping looks valid, the Start button turns on. The full decision path, including the exact test that ran, appears in the results after you click Start.</p>

<h3>Want a file to start from?</h3>
<p>Open the <b>Help</b> menu and choose <b>Save Example Template...</b>. That writes a small Excel file already laid out in long format, so you can see the shape the app expects and replace the numbers with your own.</p>
""",
    },
    {
        "id": "one_way_anova",
        "category": "Choosing a test",
        "title": "Comparing groups (t-Test / One-Way ANOVA)",
        "summary": "Are 2 or more separate, independent groups different from each other?",
        "keywords": ["anova", "one-way", "t-test", "group", "factor 1", "independent", "between", "compare"],
        "html": """
<h2>Comparing independent groups: t-Test or One-Way ANOVA</h2>

<h3>When do you use this?</h3>
<p>You have <b>two or more groups</b>. Each person or animal is in <b>exactly one group</b>. You want to know: <b>are the measured values different between groups?</b></p>
<p>Examples: "Do Control, Treatment A, and Treatment B mice have different body weights?" "Is there a blood pressure difference between three genotypes?" "Do patients from two hospitals differ in recovery time?"</p>
<p>The app uses a t-Test automatically when there are exactly 2 groups, and ANOVA when there are 3 or more. You do not choose. It happens automatically.</p>

<h3>What the app checks and runs</h3>
<p>The app checks whether your data is normally distributed and whether the groups have similar spread. If the data looks normal, it defaults to Welch's version of the t-Test or ANOVA, since that stays valid whether or not the groups have equal variance. If the data is not normal, the app first tries a transformation, then falls back to a rank-based test.</p>
<p>When the result is significant with three or more groups, the app automatically compares each pair of groups and adjusts the p-values for the number of comparisons.</p>

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

<h3>Common mistake: groups as column names</h3>
<table>
<tr><th>Control</th><th>Treatment_A</th><th>Treatment_B</th></tr>
<tr><td>120</td><td>135</td><td>145</td></tr>
<tr><td>118</td><td>142</td><td>150</td></tr>
<tr><td>122</td><td>138</td><td>148</td></tr>
<tr><td>119</td><td>140</td><td>144</td></tr>
<tr><td>121</td><td>141</td><td>147</td></tr>
</table>
<p><b>Why this fails:</b> "Control", "Treatment_A", and "Treatment_B" are the group names. They should appear as values inside a single column, not as column headers.</p>

<h3>What to drag where</h3>
<table>
<tr><th>Bucket</th><th>What to drag here</th><th>In this example</th></tr>
<tr><td><b>Dependent Variable</b></td><td>Your measurement column, the numbers you measured</td><td>Blood_Pressure</td></tr>
<tr><td><b>Factor 1</b></td><td>Your group label column, the column that says which group each row belongs to</td><td>Group</td></tr>
<tr><td>Factor 2</td><td>Leave empty</td><td>-</td></tr>
<tr><td>Subject ID</td><td>Leave empty. Nobody is measured twice in this design</td><td>-</td></tr>
<tr><td>Covariates</td><td>Leave empty</td><td>-</td></tr>
</table>

<h3>Before you click Start: checklist</h3>
<ul>
<li>One row per measurement, not one row per subject with multiple columns.</li>
<li>Group names are spelled identically across rows. "Control" and "control" are treated as different groups.</li>
<li>The measurement column contains only numbers, with no units and no text mixed in.</li>
<li>Each group has at least three measurements. The app blocks the test for any group with fewer.</li>
<li>No subject appears more than once. If a subject is measured twice, use Repeated Measures ANOVA instead.</li>
</ul>
""",
    },
    {
        "id": "two_way_anova",
        "category": "Choosing a test",
        "title": "Two independent grouping factors (Two-Way ANOVA)",
        "summary": "Two separate ways of grouping, e.g. Treatment AND Sex.",
        "keywords": ["anova", "two-way", "factor 2", "interaction", "between", "crossed", "two factors"],
        "html": """
<h2>Two-Way ANOVA: two independent grouping factors</h2>

<h3>When do you use this?</h3>
<p>You have <b>two separate ways of grouping your subjects</b>, for example Treatment (Control / Drug) and Sex (Male / Female), and every combination of the two has been measured. You want to know: does each factor have an effect, and do they interact (i.e. does the treatment work differently in males vs. females)?</p>
<p>Examples: "Does the treatment effect depend on sex?" "Does diet interact with exercise level to affect weight loss?"</p>
<p>No subject appears more than once. If the same subjects are measured at multiple time points or conditions, use Mixed ANOVA instead.</p>

<h3>What the app checks and runs</h3>
<p>The app checks whether your data is normally distributed and whether the combinations of the two factors have similar spread. If the data looks normal, it runs a two-way ANOVA and reports a main effect for each factor plus the interaction between them. If the data is not normal, the app first tries a transformation, then falls back to a permutation-based test. When a result is significant, it compares the relevant groups and adjusts the p-values for the number of comparisons.</p>

<h3>What your data must look like</h3>
<p>One row per measurement. <b>Both</b> group columns are present in every row, alongside the measured value.</p>
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

<h3>Common mistake: both factors hidden in column names</h3>
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
<tr><td>Subject ID</td><td>Leave empty. No subject is measured more than once</td><td>-</td></tr>
<tr><td>Covariates</td><td>Leave empty</td><td>-</td></tr>
</table>

<h3>Before you click Start: checklist</h3>
<ul>
<li>Both group columns hold category labels, one per row. Numbers used as group codes are fine as long as they mark a handful of distinct groups and not a continuous measurement.</li>
<li>Every row has a value in both group columns.</li>
<li>No subject is measured more than once. If a subject is measured under several conditions, use Mixed ANOVA instead.</li>
<li>Each combination of the two factors has at least a few measurements.</li>
<li>No group name is embedded in a column header.</li>
</ul>
""",
    },
    {
        "id": "repeated_measures_anova",
        "category": "Choosing a test",
        "title": "Same subjects measured multiple times (Repeated Measures ANOVA)",
        "summary": "The same people or animals measured at several time points, one group only.",
        "keywords": ["repeated", "within", "subject", "timepoint", "longitudinal", "pre post", "same subjects", "one group"],
        "html": """
<h2>Repeated Measures ANOVA: same subjects, multiple measurements</h2>

<h3>When do you use this?</h3>
<p>The <b>same subjects are measured at multiple time points or conditions</b>, and all subjects belong to a <b>single group</b>. You want to know: does the measurement change over time or across conditions?</p>
<p>Examples: "Does blood pressure change from Baseline to Week 4 to Week 8 in our patients?" "Does heart rate change across three exercise intensities in the same athletes?"</p>
<p><b>Key distinction:</b> all subjects are in one group only. If they are also split into different groups (e.g. Treatment vs. Control), use Mixed ANOVA instead.</p>

<h3>What the app checks and runs</h3>
<p>The app checks whether your data is normally distributed and whether the differences between time points are stable enough to compare them on equal footing. If the data looks normal, it runs a repeated measures ANOVA. If that equal-footing assumption cannot be confirmed, the app applies a conservative correction rather than assuming it holds. If the data is not normal, the app falls back to a rank-based test for repeated measures. When the overall result is significant, it compares the time points and adjusts the p-values for the number of comparisons.</p>

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
<p>Notice: P001 appears three times, once per time point. That is correct and expected.</p>

<h3>One column per time point also works</h3>
<table>
<tr><th>SubjectID</th><th>Baseline</th><th>Week_4</th><th>Week_8</th></tr>
<tr><td>P001</td><td>140</td><td>130</td><td>125</td></tr>
<tr><td>P002</td><td>145</td><td>135</td><td>132</td></tr>
<tr><td>P003</td><td>142</td><td>133</td><td>128</td></tr>
<tr><td>P004</td><td>138</td><td>128</td><td>122</td></tr>
<tr><td>P005</td><td>144</td><td>134</td><td>130</td></tr>
</table>
<p>If you load this wide layout, with one subject identifier and one numeric column per time point, the app recognizes it and reshapes it into the long format above for you. The time points become the values of a new condition column. Either layout is fine to load.</p>

<h3>What to drag where</h3>
<p>If the app already reshaped a wide file, the reshaped columns are what you map here.</p>
<table>
<tr><th>Bucket</th><th>What to drag here</th><th>In this example</th></tr>
<tr><td><b>Dependent Variable</b></td><td>Your measurement column</td><td>Blood_Pressure</td></tr>
<tr><td><b>Factor 1</b></td><td>The time point or condition column, the thing that changes across rows for the same subject</td><td>Timepoint</td></tr>
<tr><td><b>Subject ID</b></td><td>The column that says who each measurement belongs to</td><td>SubjectID</td></tr>
<tr><td>Factor 2</td><td>Leave empty. Subjects are all in one group here</td><td>-</td></tr>
<tr><td>Covariates</td><td>Leave empty</td><td>-</td></tr>
</table>

<h3>Before you click Start: checklist</h3>
<ul>
<li>Each subject appears exactly once per time point. No duplicate SubjectID + Timepoint combinations.</li>
<li>Ideally, all subjects have measurements at all time points. If some measurements are missing, the app switches to a mixed model that uses every subject instead of dropping anyone with a gap.</li>
<li>The Timepoint column can hold text labels such as "Baseline" and "Week_4", or numbers such as 0, 4, and 8. Both work.</li>
<li>Subject ID values repeat across rows. This is correct and expected.</li>
</ul>
""",
    },
    {
        "id": "mixed_anova",
        "category": "Choosing a test",
        "title": "Different groups, each measured multiple times (Mixed ANOVA)",
        "summary": "Treatment vs. Control AND multiple time points, the same subjects within each group.",
        "keywords": ["mixed", "between", "within", "subject id", "longitudinal", "group", "timepoint", "repeated", "groups over time"],
        "html": """
<h2>Mixed ANOVA: groups and repeated measurements</h2>

<h3>When do you use this?</h3>
<p>You have <b>two or more groups</b>, and the <b>same subjects within each group are measured multiple times</b>. You want to know three things: does the measurement change over time, does it differ between groups, and does the change over time look different from one group to another? That last question, the interaction, is what this test is built around and what it reports as the main result.</p>
<p>Examples: "Do Treatment and Control patients show different recovery trajectories from Baseline to Week 8?" "Do WT and KO mice respond differently across three dose levels?"</p>

<h3>What the app checks and runs</h3>
<p>The app checks whether your data is normally distributed, whether the groups have similar spread, and whether the differences between time points are stable enough to compare on equal footing. If everything looks fine, it runs a mixed ANOVA that separates the group effect, the time effect, and the interaction between them. If that equal-footing assumption for the time factor cannot be confirmed, the app applies a conservative correction rather than assuming it holds. If the data is not normal, it falls back to a rank-based test for this design. When the interaction is significant, the app compares the group-and-time combinations and adjusts the p-values for the number of comparisons.</p>

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
<p>P001 appears twice (Baseline + Week_8). Their Group (Control) stays the same across both rows, which is correct.</p>

<h3>Common mistake: time points as separate columns</h3>
<table>
<tr><th>SubjectID</th><th>Group</th><th>Baseline</th><th>Week_8</th></tr>
<tr><td>P001</td><td>Control</td><td>140</td><td>138</td></tr>
<tr><td>P002</td><td>Control</td><td>145</td><td>142</td></tr>
<tr><td>P010</td><td>Treatment</td><td>142</td><td>120</td></tr>
<tr><td>P011</td><td>Treatment</td><td>144</td><td>122</td></tr>
<tr><td>P012</td><td>Treatment</td><td>146</td><td>125</td></tr>
</table>
<p><b>Why this fails:</b> the app only reshapes a wide file when there is no group column, so the automatic reshape does not apply here. It keeps the table as is, and there is no single Timepoint column to map to Factor 1. Put the time points in one column with the labels as values, and keep the group in its own column, as in the long layout above.</p>

<h3>What to drag where</h3>
<table>
<tr><th>Bucket</th><th>What to drag here</th><th>In this example</th></tr>
<tr><td><b>Dependent Variable</b></td><td>Your measurement column</td><td>Blood_Pressure</td></tr>
<tr><td><b>Factor 1</b></td><td>The <i>repeated</i> factor, the thing that changes across rows for the same subject (time point, condition)</td><td>Timepoint</td></tr>
<tr><td><b>Factor 2</b></td><td>The <i>independent</i> group. Subjects are fully in one group only</td><td>Group</td></tr>
<tr><td><b>Subject ID</b></td><td>The column that identifies each subject</td><td>SubjectID</td></tr>
<tr><td>Covariates</td><td>Leave empty</td><td>-</td></tr>
</table>
<p><b>Important:</b> Factor 1 = the repeated/within factor (time points). Factor 2 = the between-subjects group (Treatment / Control). Getting them swapped reverses the labels in the output.</p>

<h3>Before you click Start: checklist</h3>
<ul>
<li>Each subject appears exactly once per time point. No duplicate SubjectID + Timepoint combinations within a group.</li>
<li>A subject's Group value is identical across all their rows. The app uses this to decide which factor is the group and which is the repeated one, so a subject that appears under more than one group breaks the design.</li>
<li>Ideally, all subjects have measurements at all time points. If any subject is missing even one time point, the app switches to a mixed model that uses every subject instead of dropping anyone with a gap.</li>
<li>Subject ID, Factor 1 (within), and Factor 2 (between) are all filled. This is what distinguishes Mixed ANOVA from Two-Way ANOVA.</li>
</ul>
""",
    },
    {
        "id": "ancova",
        "category": "Choosing a test",
        "title": "Comparing groups while correcting for a background variable (ANCOVA)",
        "summary": "Like ANOVA, but you control for an additional numeric variable that might distort results.",
        "keywords": ["ancova", "covariate", "baseline", "adjust", "correct", "confound", "control for"],
        "html": """
<h2>ANCOVA: correcting for a background variable before comparing groups</h2>

<h3>When do you use this?</h3>
<p>You want to compare groups, but you suspect that another variable, one you did not control, differs between your groups and may distort the comparison. ANCOVA adjusts the group averages mathematically to account for the background variable, allowing for a fairer comparison.</p>
<p><b>Important:</b> The covariate must not be affected by the treatment itself (e.g. do not use weight after treatment as a covariate). Otherwise, you risk mathematically removing the actual treatment effect (overadjustment bias).</p>
<p>Example: You compare test scores between Treatment and Control groups. But Control patients happened to be older than Treatment patients. A simple ANOVA would partly reflect that pre-existing age difference, not just the treatment effect. Putting Age into Covariates corrects for it.</p>
<p>The correcting variable goes into the <b>Covariates</b> bucket. It must be a number (not a group label). You can add more than one.</p>

<h3>What the app checks and runs</h3>
<p>The app fits the ANCOVA model and reports each group's average after adjusting for the covariate, next to the raw unadjusted average. It also checks whether the covariate affects every group the same way, which is the assumption ANCOVA relies on. If that assumption fails, it runs a follow-up analysis that shows over which range of the covariate the groups actually differ. When the group effect comes out significant, the app compares the adjusted group averages against each other and adjusts the p-values for the number of comparisons.</p>

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

<h3>Common mistake: pre-computed group means</h3>
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
<tr><td>Factor 2</td><td>Leave empty (unless you also have a second group factor, which gives a Two-Way ANCOVA)</td><td>-</td></tr>
<tr><td>Subject ID</td><td>Leave empty. Each subject appears once</td><td>-</td></tr>
</table>

<h3>Before you click Start: checklist</h3>
<ul>
<li>One row per subject, not one row per time point.</li>
<li>Covariates are number columns, not group label columns.</li>
<li>Factor 1 is a group label column, not a number.</li>
<li>Raw individual values, not pre-aggregated group means.</li>
</ul>
""",
    },
    {
        "id": "correlation",
        "category": "Choosing a test",
        "title": "Do two measurements go up and down together? (Correlation)",
        "summary": "Measure the relationship between two continuous variables. No groups.",
        "keywords": ["correlation", "pearson", "spearman", "relationship", "continuous", "scatter", "association"],
        "html": """
<h2>Correlation: do two measurements move together?</h2>

<h3>When do you use this?</h3>
<p>You have <b>two number columns</b>, one measurement per subject in each, and you want to know: <b>when one goes up, does the other tend to go up (or down) too?</b> There are no groups. Both columns are numbers.</p>
<p>Examples: "Is dosage related to blood pressure after treatment?" "Does protein expression correlate with tumour size?" "Is age associated with recovery speed?"</p>
<p>Correlation measures the strength and direction of a relationship. It does not give you a prediction formula and does not prove causation. For a specific slope or for controlling additional variables, use Regression instead.</p>

<h3>What your data must look like</h3>
<p>One row per subject. Two numeric columns, one for each variable.</p>
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

<h3>Common mistake: averaged groups instead of individual values</h3>
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
<tr><td><b>Dependent Variable</b></td><td>The outcome, the measurement you want to explain</td><td>Blood_Pressure</td></tr>
<tr><td><b>Factor 1</b></td><td>The predictor, the other numeric variable</td><td>Dosage_mg</td></tr>
<tr><td>Covariates</td><td>Leave empty. Adding anything here switches to Regression</td><td>-</td></tr>
<tr><td>Subject ID</td><td>Leave empty. Adding this switches to a Mixed Model</td><td>-</td></tr>
<tr><td>Factor 2, Filter</td><td>Not needed for basic correlation</td><td>-</td></tr>
</table>
<p>The app picks Pearson or Spearman automatically based on whether the data is normally distributed. You do not need to choose.</p>

<h3>Before you click Start: checklist</h3>
<ul>
<li>Both columns are numbers, with no group labels and no text.</li>
<li>One row per subject, with no repeated measurements.</li>
<li>Raw individual values, with no pre-computed averages or bins.</li>
<li>Covariates and Subject ID are both empty.</li>
</ul>
""",
    },
    {
        "id": "linear_regression",
        "category": "Choosing a test",
        "title": "Predicting a measurement from other measurements (Linear Regression)",
        "summary": "How much does the outcome change per unit of the predictor? Add control variables.",
        "keywords": ["regression", "ols", "predict", "covariates", "beta", "coefficient", "slope", "linear"],
        "html": """
<h2>Linear Regression: predicting one measurement from others</h2>

<h3>When do you use this?</h3>
<p>You want to know: <b>how much does the outcome change for each unit increase in the predictor, and by exactly how much?</b> Unlike correlation, regression gives you a specific number (the slope). You can also include additional variables to control for, which correlation cannot do.</p>
<p>Examples: "For every additional mg of dosage, how much does blood pressure drop?" "Predict test score from study hours, controlling for age and baseline score."</p>

<h3>Two ways to trigger regression</h3>
<ul>
<li><b>Simple regression</b> (one predictor, no additional controls): Drag only Factor 1. Then tick the checkbox <i>"Als Lineare Regression analysieren (Y = a + bX)"</i> that appears below the buckets. Without that tick, the app runs Correlation instead.</li>
<li><b>Multiple regression</b> (predictor + control variables): Drag Factor 1 and one or more Covariates. The Regression mode activates automatically when Covariates is populated.</li>
</ul>

<h3>Variable Transformations</h3>
<p>When regression mode is active, you will see dropdowns to transform your X or Y variables. The options are log10, log10(x+1), square root, and Box-Cox. Use one if your data is highly skewed or the spread of the outcome grows with the predictor. <b>Important:</b> Transforming changes how you interpret the result. If you log-transform Y, the effect is multiplicative rather than additive. If you log-transform <b>both X and Y (Log-Log)</b>, β represents an <i>elasticity</i> (a 1% increase in X yields a β% change in Y).</p>

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
<tr><td>Factor 2</td><td>Leave empty</td><td>-</td></tr>
<tr><td>Subject ID</td><td>Leave empty. Each subject appears once</td><td>-</td></tr>
</table>

<h3>Reading the result</h3>
<p>The main output is the <b>coefficient (β)</b> for the primary predictor, with its p-value. If β = -2.5 for Dosage_mg, it means: for every additional mg of dosage, blood pressure decreases by 2.5 points on average, holding age and baseline BP constant.</p>
<p>The report also shows <b>R²</b>: the share of variation in the outcome the model accounts for. A significant β with a low R² means the predictor matters but leaves most of the outcome unexplained.</p>

<h3>Before you click Start: checklist</h3>
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
        "category": "Choosing a test",
        "title": "Predicting a yes/no outcome (Logistic Regression)",
        "summary": "Your outcome is binary: 0/1, yes/no, event/no-event.",
        "keywords": ["logistic", "binary", "0/1", "yes no", "event", "odds ratio", "predict", "complication", "survival"],
        "html": """
<h2>Logistic Regression: predicting a yes/no outcome</h2>

<h3>When do you use this?</h3>
<p>Your outcome can only be <b>one of two values</b>: yes/no, 0/1, survived/died, complication/no-complication. You want to know which measurements predict that outcome.</p>
<p>Examples: "Which pre-operative measurements predict post-operative complications (yes/no)?" "Does treatment group predict whether a patient recovers completely (0/1)?"</p>
<p>The app detects this automatically: when your Dependent Variable column has exactly two distinct values, logistic regression runs without any manual selection.</p>

<h3>What your data must look like</h3>
<p>One row per subject. The outcome column must have exactly two values. Using 0 and 1 is the clearest format, but two text labels (e.g. "Yes" and "No") work too. The app picks one value as the reference and models the odds of the other.</p>
<table>
<tr><th>Side_Effect</th><th>Treatment</th><th>Age</th><th>BP_baseline</th></tr>
<tr><td>0</td><td>Placebo</td><td>62</td><td>145</td></tr>
<tr><td>1</td><td>Drug</td><td>71</td><td>120</td></tr>
<tr><td>0</td><td>Placebo</td><td>59</td><td>135</td></tr>
<tr><td>1</td><td>Drug</td><td>68</td><td>115</td></tr>
<tr><td>0</td><td>Placebo</td><td>63</td><td>125</td></tr>
<tr><td>1</td><td>Drug</td><td>72</td><td>110</td></tr>
<tr><td>1</td><td>Drug</td><td>65</td><td>108</td></tr>
</table>

<h3>Common mistake: outcome has more than two values</h3>
<table>
<tr><th>Severity</th><th>Dosage_mg</th></tr>
<tr><td>None</td><td>10</td></tr>
<tr><td>Mild</td><td>30</td></tr>
<tr><td>Severe</td><td>50</td></tr>
</table>
<p><b>Why this fails:</b> Three outcome levels (None / Mild / Severe) cannot be handled by standard logistic regression. Collapse them to two levels first, for example None vs. any side effect.</p>

<h3>What to drag where</h3>
<p>Factor 1 and Covariates are handled differently. A column in <b>Factor 1</b> is treated as a category: each of its values is compared against a reference value. A column in <b>Covariates</b> is treated as a number and enters the model on a per-unit scale. Put a grouping predictor (treatment, sex) in Factor 1, and put a measurement you want a per-unit odds ratio for (age, dose, blood pressure) in Covariates.</p>
<table>
<tr><th>Bucket</th><th>What to drag here</th><th>In this example</th></tr>
<tr><td><b>Dependent Variable</b></td><td>The binary outcome column (0/1 or two-level text)</td><td>Side_Effect</td></tr>
<tr><td><b>Factor 1</b></td><td>A grouping predictor, compared level by level</td><td>Treatment</td></tr>
<tr><td><b>Covariates</b></td><td>Numeric predictors, each on a per-unit scale</td><td>Age, BP_baseline</td></tr>
<tr><td>Factor 2</td><td>Leave empty</td><td>-</td></tr>
<tr><td>Subject ID</td><td>Leave empty. Each subject appears once</td><td>-</td></tr>
</table>

<h3>Reading the result</h3>
<p>The main output is the <b>Odds Ratio (OR)</b> for each predictor. For a Factor 1 group, the OR compares that group against the reference group: OR = 2.5 means the event is 2.5 times as likely in that group. For a numeric covariate, the OR is per one-unit increase: OR = 1.05 for Age means the odds rise 5% for each additional year. OR &lt; 1 means lower odds; OR = 1 means no effect. The report lists every OR with its 95% confidence interval.</p>
<p>The AUC (area under the ROC curve) measures how well the model separates the two outcomes. The report shows the ROC curve and reads the AUC as: below 0.6 poor, 0.6 to 0.7 acceptable, 0.7 to 0.8 good, 0.8 to 0.9 excellent, 0.9 and above outstanding. An AUC of 0.5 is no better than chance.</p>

<h3>Before you click Start: checklist</h3>
<ul>
<li>Dependent Variable has exactly two distinct values.</li>
<li>Every covariate is numeric; a Factor 1 predictor can be text or numeric.</li>
<li>One row per subject.</li>
<li>Rule of thumb: at least 10 events (rows where the outcome occurred) per predictor in the model.</li>
</ul>
""",
    },
    {
        "id": "dependent_samples",
        "category": "Concepts",
        "title": "Dependent (paired) samples",
        "summary": "When measurements are paired or repeated on the same subjects, and which tests apply.",
        "keywords": ["paired", "dependent", "repeated", "wilcoxon", "friedman", "matched"],
        "html": (
            "<h2>Dependent (paired) samples</h2>"
            "<h3>When are samples dependent?</h3>"
            "<p>Dependent samples arise when:</p>"
            "<ul>"
            "<li>Measurements are taken on the <b>same subject</b> at different time points</li>"
            "<li>Measurements are naturally paired (e.g. left and right eye)</li>"
            "<li>Experiments are conducted with repeated measurements</li>"
            "</ul>"
            "<h3>Data structure for dependent tests</h3>"
            "<p>For dependent tests, each group must:</p>"
            "<ul>"
            "<li>Contain the <b>same number</b> of measurements</li>"
            "<li>Have measurements in <b>matching order</b></li>"
            "</ul>"
            "<p>Example: Measurement 1 in group A and measurement 1 in group B must be from the same subject</p>"
            "<h3>Available tests</h3>"
            "<ul>"
            "<li><b>Two groups:</b> Paired t-test or Wilcoxon signed-rank test</li>"
            "<li><b>More than two groups:</b> Repeated Measures ANOVA or Friedman test</li>"
            "</ul>"
        ),
    },
    {
        "id": "graph_visualization",
        "category": "Workflow & Output",
        "title": "Graph visualization",
        "summary": "How to configure and export plots from an analysis result.",
        "keywords": ["plot", "graph", "chart", "figure", "visualization", "export", "bar", "box"],
        "html": """
            <h2>Graph visualization</h2>
            <ul>
                <li><b>Plot types:</b> Bar, box, violin, and strip plots are generated from your data. Each type visualizes group distributions differently:
                    <ul>
                        <li><b>Bar:</b> Shows group means with error bars.</li>
                        <li><b>Box:</b> Displays medians, quartiles, and outliers.</li>
                        <li><b>Violin:</b> Combines boxplot with a kernel density estimate.</li>
                        <li><b>Strip:</b> Shows all individual data points as dots.</li>
                    </ul>
                </li>
                <li><b>Switching plot types:</b> Use the plot configuration or appearance dialog to select your preferred plot type.</li>
                <li><b>Appearance adjustments:</b>
                    <ul>
                        <li>Change <b>colors</b> and <b>hatches</b> for each group.</li>
                        <li>Choose <b>error bar type</b>: Standard deviation (SD) or standard error (SEM).</li>
                        <li>Set <b>error bar style</b>: With caps or line only.</li>
                        <li>Customize <b>fonts</b>, <b>axes</b>, and <b>grid lines</b> for clarity.</li>
                    </ul>
                </li>
                <li><b>Overlay features:</b>
                    <ul>
                        <li>Show <b>individual data points</b> on box, violin, or strip plots.</li>
                        <li>Add <b>statistical annotations</b>: Letters (grouping) or bars (significance lines) to highlight significant differences.</li>
                    </ul>
                </li>
            </ul>
        """,
    },
    {
        "id": "statistical_tests_html",
        "category": "Workflow & Output",
        "title": "Statistical tests and HTML report",
        "summary": "How tests are chosen and what the exported HTML report contains.",
        "keywords": ["report", "html", "export", "results", "tests", "output"],
        "html": """
            <h2>Statistical tests and HTML report</h2>
            <ul>
                <li><b>How does the program select the test?</b>
                    <ul>
                        <li>The program automatically detects the appropriate test based on group count and data structure.</li>
                        <li><b>Two independent groups:</b>
                            <ul>
                                <li><b>t-Test</b> (parametric): Used when data is normally distributed and variances are comparable.</li>
                                <li><b>Mann-Whitney-U Test</b> (non-parametric): Used when assumptions for t-test are not met.</li>
                            </ul>
                        </li>
                        <li><b>Two dependent groups (e.g. paired measurements):</b>
                            <ul>
                                <li><b>Paired t-Test</b> (parametric): For normally distributed differences.</li>
                                <li><b>Wilcoxon signed-rank test</b> (non-parametric): For non-normally distributed differences.</li>
                            </ul>
                        </li>
                        <li><b>More than two independent groups:</b>
                            <ul>
                                <li><b>One-Way ANOVA</b> (parametric): For normally distributed data with equal variances.</li>
                                <li><b>Kruskal-Wallis Test</b> (non-parametric): When ANOVA assumptions are violated.</li>
                            </ul>
                        </li>
                        <li>The decision is based on normality tests (Shapiro-Wilk) and variance homogeneity (Levene test). When assumptions are violated, a non-parametric test is automatically selected.</li>
                        <li>Post-hoc tests (e.g. pairwise comparisons) are automatically added when significant differences are found.</li>
                        <li><i>Note: for detailed data templates, including long-format examples, see the related recipes in this hub.</i></li>
                    </ul>
                </li>
                <li><b>Interpreting results:</b>
                    <ul>
                        <li><b>p-values</b> indicate the probability that observed differences are due to chance.</li>
                        <li><b>Significance indicators</b> (letters or bars) show which groups differ significantly.</li>
                        <li>Key statistics (means, standard deviations, test statistics) are clearly displayed.</li>
                    </ul>
                </li>
                <li><b>HTML report export:</b>
                    <ul>
                        <li>Results are written to a self-contained HTML report covering each analysis.</li>
                        <li>Sections reflect the test or plot type (e.g. "ANOVA Results", "Pairwise Comparisons").</li>
                        <li>Each section shows group names, means, test statistics, p-values, and significance markers.</li>
                        <li>Open the report in any web browser to review, print, or share results.</li>
                    </ul>
                </li>
            </ul>
            <p style='color:gray; font-size:90%'>Note: see the related recipes in this hub for detailed long-format templates covering both advanced and basic models.</p>
        """,
    },
]
