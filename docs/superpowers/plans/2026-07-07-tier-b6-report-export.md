# Tier B6: Report/Export Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close a reproducible HTML/script-injection path in the exported report (RE1), add the
shared escaping helper that fix needs so it isn't inlined three times (RE5), and apply the same
escaping consistently at the other unescaped Plotly label sites (RE2).

**Architecture:** RE5 first (the shared `_esc()` helper other tasks depend on), then RE1 (the
one reproducible injection, highest severity), then RE2 (same pattern, lower-confidence
exploitability, applied for consistency). All three land in `src/export/`, which is otherwise
untouched since round 1 of the audit (confirmed via `git log --oneline b16cf24..HEAD --
src/export/` returning zero commits as of the round-2 audit).

**Tech Stack:** Python, Jinja2 (via the existing `HTMLExporter`), pytest, statsmodels (to
reproduce the injection against a real fitted model, matching how round 1's audit originally
demonstrated it).

---

### Task 1: RE5 — add a shared HTML-escaping helper to `_FormattingMixin`

**Files:**
- Modify: `src/export/report_formatting.py` (imports + `_FormattingMixin` class)
- Test: `tests/test_formatting_mixin_esc.py`

`_FormattingMixin` has no HTML-escaping helper, unlike `outlier_html_exporter.py`'s existing,
correct `_esc()` (`html.escape(str(value))`). RE1's fix (Task 2) needs a shared helper so the
same escaping logic isn't inlined 3 separate times.

- [ ] **Step 1: Write the failing test**

```python
"""_FormattingMixin has no shared HTML-escaping helper, unlike outlier_html_exporter.py's
existing _esc(). RE1's fix needs one reusable helper instead of inlining html.escape() three
times.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from export.report_formatting import _FormattingMixin


def test_esc_escapes_html_special_characters():
    assert _FormattingMixin._esc("<script>alert(1)</script>") == "&lt;script&gt;alert(1)&lt;/script&gt;"


def test_esc_handles_non_string_input():
    assert _FormattingMixin._esc(42) == "42"
    assert _FormattingMixin._esc(None) == "None"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/test_formatting_mixin_esc.py -v`
Expected: FAIL with `AttributeError: type object '_FormattingMixin' has no attribute '_esc'`.

- [ ] **Step 3: Add the `import html` and the `_esc` staticmethod**

Read `report_formatting.py`'s current imports fresh (`sed -n '1,35p' src/export/report_formatting.py`),
then change:

```python
import math
import random
from pathlib import Path
from typing import Any
```

to:

```python
import html
import math
import random
from pathlib import Path
from typing import Any
```

Then add this staticmethod to `_FormattingMixin`, immediately after the class docstring
(`"""Stateless formatting / numeric helpers mixed into ``HTMLExporter``."""`) and before the
existing `_normalize_for_json` method:

```python
    @staticmethod
    def _esc(value: Any) -> str:
        """HTML-escape a value for safe interpolation into a raw f-string HTML
        block (i.e. anywhere NOT already covered by Jinja's autoescape=True -
        see report_association.py's chart-table builders, which render via
        `{{ chart.html | safe }}` and therefore bypass autoescaping)."""
        return html.escape(str(value))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/test_formatting_mixin_esc.py -v`
Expected: PASS (both cases).

- [ ] **Step 5: Commit**

```bash
git add tests/test_formatting_mixin_esc.py src/export/report_formatting.py
git commit -m "feat(export): add shared HTML-escaping helper to _FormattingMixin"
```

---

### Task 2: RE1 — escape the `parameter` cell in the 3 coefficient-table builders

**Files:**
- Modify: `src/export/report_association.py:34,87,135`
- Test: `tests/test_html_injection_coefficient_tables.py`

`_build_or_table_html`, `_build_beta_coefficient_table_html`, and
`_build_linear_regression_coefficient_table_html` all interpolate `row.get('parameter', '')`
into a raw HTML f-string with no escaping. These render via `{{ chart.html | safe }}`
(`src/templates/report_single.html.j2:32`, `report_multi.html.j2:32`), which explicitly bypasses
Jinja's `autoescape=True`. `parameter` comes from a fitted model's `.params.index` — for a
categorical predictor, patsy encodes the raw category value the user typed into a group/category
column directly into the parameter name (`C(group)[T.<value>]`), so a group literally named
`Zebra<script>alert(1)</script>` reaches the exported `.html` file as a live `<script>` tag.

- [ ] **Step 1: Write the failing test — reproduce the injection against a real statsmodels fit**

```python
"""RE1: report_association.py's 3 coefficient-table builders interpolate a fitted model's raw
parameter name into an HTML f-string with no escaping. Reproduces this end-to-end against a
real statsmodels logit fit with a malicious category value, matching how round 1's audit
originally demonstrated the injection - not a hand-crafted dict, a real patsy-encoded parameter
name.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import pytest
import statsmodels.formula.api as smf

from export.report_association import _AssociationMixin

_MALICIOUS_GROUP = "Zebra<script>alert(1)</script>"


def _fit_logit_with_malicious_group_name():
    df = pd.DataFrame({
        "y": [0, 0, 0, 1, 1, 1, 0, 1],
        "group": (
            ["ctrl"] * 4
            + [_MALICIOUS_GROUP] * 4
        ),
    })
    return smf.logit("y ~ C(group)", data=df).fit(disp=0)


def test_patsy_really_does_leak_the_raw_group_name_into_the_parameter():
    model = _fit_logit_with_malicious_group_name()
    param_names = list(model.params.index)
    assert any(_MALICIOUS_GROUP in name for name in param_names), (
        f"sanity check failed - patsy's parameter names don't contain the raw group name: "
        f"{param_names}"
    )


def test_or_table_html_escapes_the_malicious_parameter_name():
    model = _fit_logit_with_malicious_group_name()
    or_table = [
        {
            "parameter": str(name),
            "odds_ratio": 1.5,
            "ci_lower": 1.0,
            "ci_upper": 2.0,
            "z_value": 1.0,
            "p_value": 0.1,
        }
        for name in model.params.index
        if name != "Intercept"
    ]

    bundle = _AssociationMixin._build_or_table_html({"odds_ratios": or_table})

    assert "<script>alert(1)</script>" not in bundle["html"], (
        "raw <script> tag from the group name survived unescaped into the exported HTML"
    )
    assert "&lt;script&gt;" in bundle["html"], "the escaped form must still be present"


def test_beta_coefficient_table_html_escapes_the_malicious_parameter_name():
    coef_table = [
        {
            "parameter": _MALICIOUS_GROUP,
            "coefficient": 0.5,
            "std_err": 0.1,
            "z_value": 5.0,
            "p_value": 0.001,
            "ci_lower": 0.3,
            "ci_upper": 0.7,
        }
    ]
    bundle = _AssociationMixin._build_beta_coefficient_table_html({"coefficients": coef_table})
    assert "<script>alert(1)</script>" not in bundle["html"]


def test_linear_regression_coefficient_table_html_escapes_the_malicious_parameter_name():
    coef_table = [
        {
            "parameter": _MALICIOUS_GROUP,
            "coefficient": 0.5,
            "std_err": 0.1,
            "t_value": 5.0,
            "p_value": 0.001,
            "ci_lower": 0.3,
            "ci_upper": 0.7,
        }
    ]
    bundle = _AssociationMixin._build_linear_regression_coefficient_table_html({"coefficient_table": coef_table})
    assert "<script>alert(1)</script>" not in bundle["html"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/test_html_injection_coefficient_tables.py -v`
Expected: `test_patsy_really_does_leak_the_raw_group_name_into_the_parameter` PASSES (it's the
sanity check, not testing the fix). The 3 `test_*_escapes_the_malicious_parameter_name` tests
FAIL — `"<script>alert(1)</script>"` is present verbatim in `bundle["html"]`.

- [ ] **Step 3: Fix all 3 builders**

Read `report_association.py` fresh (`grep -n "row.get('parameter', '')" src/export/report_association.py`)
to confirm the current 3 line numbers, then change each of the 3 occurrences of:

```python
                f"<td>{row.get('parameter', '')}</td>"
```

to:

```python
                f"<td>{_FormattingMixin._esc(row.get('parameter', ''))}</td>"
```

(`_FormattingMixin` is already imported at the top of this file — `from export.report_formatting
import _FormattingMixin` — no new import needed. All 3 sites are inside
`_build_or_table_html`, `_build_beta_coefficient_table_html`, and
`_build_linear_regression_coefficient_table_html` respectively; make the same one-line change in
each.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/test_html_injection_coefficient_tables.py -v`
Expected: PASS (all 4 cases).

- [ ] **Step 5: Commit**

```bash
git add tests/test_html_injection_coefficient_tables.py src/export/report_association.py
git commit -m "fix(export): escape the parameter cell in coefficient-table HTML builders"
```

---

### Task 3: RE2 — apply the same escaping at the remaining unescaped Plotly label sites

**Files:**
- Modify: `src/export/report_charts.py:288-289,299,535-536,546,740-741`,
  `src/export/report_summaries.py:816`
- Test: `tests/test_plotly_label_escaping.py`

`report_charts.py:740-741` already escapes a group name via inline `html.escape(str(group_name))`
before using it as a Plotly trace `name=`. Five other sites take factor-level strings straight
from data dict keys with no escaping: `report_charts.py:288-289` (interaction-plot
`hovertext`), `:299` (`name=`), `:535-536` (mixed-profile `hovertext`), `:546` (`name=`), and
`report_summaries.py:816` (`name=str(group_name)`).

- [ ] **Step 1: Write the failing test**

```python
"""RE2: report_charts.py's own group-comparison chart already escapes group_name via
html.escape() before using it as a Plotly name= - 5 other sites in report_charts.py and
report_summaries.py take factor-level strings straight from data without escaping. Applies the
shared _FormattingMixin._esc() helper (added in RE5) at all 5 sites for consistency.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from export.report_formatting import _FormattingMixin

_MALICIOUS = "<script>alert(1)</script>"


def test_esc_helper_is_what_report_charts_and_summaries_should_use():
    # This pins down the exact helper the fix must route through - the real
    # regression coverage is the full-suite run in Step 5, since these 5
    # sites are deep inside large Plotly-figure-building functions that would
    # need a disproportionately large harness to unit-test each in isolation.
    assert _FormattingMixin._esc(_MALICIOUS) == "&lt;script&gt;alert(1)&lt;/script&gt;"
```

**Note:** unlike RE1 (where the vulnerable code is 3 short, easily-isolated `@staticmethod`
functions), RE2's 5 sites are embedded inside large Plotly-figure-construction functions
(`_build_interaction_plot_chart`-style functions spanning hundreds of lines with many
DataFrame/dict preconditions). Building a full reproduction harness for each is disproportionate
to a MEDIUM-severity, lower-confidence-exploitability consistency fix (Plotly.js typically
renders `name=`/`hovertext` as browser-escaped SVG `<text>`, not `innerHTML` — see RE2's finding
text in `docs/superpowers/audit-notes/release-2.0-audit/06-report-export-layer.md`). Verify via
the grep-based check in Step 4 and the full-suite run in Step 5 instead.

- [ ] **Step 2: Run the test to verify it passes already (sanity check, not RED)**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/test_plotly_label_escaping.py -v`
Expected: PASS already (this pins down the helper from Task 1, not the fix itself).

- [ ] **Step 3: Apply `_FormattingMixin._esc()` at all 5 unescaped sites, plus the already-escaped one for consistency**

Read each site fresh before editing (line numbers below are from planning; confirm via
`grep -n 'factor_line}={line_level}\|factor_between}={b_level}\|html.escape' src/export/report_charts.py`
and `grep -n 'name=str(group_name)' src/export/report_summaries.py`).

In `report_charts.py`, change (interaction-plot hovertext, ~line 288):
```python
                        hover_texts.append(
                            f"{factor_x}={x_val}, {factor_line}={line_level}<br>"
                            f"Mean: {cell['mean']:.3f} ± {cell['se']:.3f} SE<br>n={cell['n']}"
                        )
```
to:
```python
                        hover_texts.append(
                            f"{_FormattingMixin._esc(factor_x)}={_FormattingMixin._esc(x_val)}, "
                            f"{_FormattingMixin._esc(factor_line)}={_FormattingMixin._esc(line_level)}<br>"
                            f"Mean: {cell['mean']:.3f} ± {cell['se']:.3f} SE<br>n={cell['n']}"
                        )
```

and (~line 299):
```python
                    name=f"{factor_line}={line_level}",
```
to:
```python
                    name=f"{_FormattingMixin._esc(factor_line)}={_FormattingMixin._esc(line_level)}",
```

and (mixed-profile hovertext, ~line 535):
```python
                        hover_texts.append(
                            f"{factor_between}={b_level}, {factor_within}={w_level}<br>"
                            f"Mean: {cell['mean']:.3f} ± {cell['se']:.3f} SE<br>n={cell['n']}"
                        )
```
to:
```python
                        hover_texts.append(
                            f"{_FormattingMixin._esc(factor_between)}={_FormattingMixin._esc(b_level)}, "
                            f"{_FormattingMixin._esc(factor_within)}={_FormattingMixin._esc(w_level)}<br>"
                            f"Mean: {cell['mean']:.3f} ± {cell['se']:.3f} SE<br>n={cell['n']}"
                        )
```

and (~line 546):
```python
                    name=f"{factor_between}={b_level}",
```
to:
```python
                    name=f"{_FormattingMixin._esc(factor_between)}={_FormattingMixin._esc(b_level)}",
```

and, for consistency (route the already-correct site through the shared helper instead of an
inline `import html`), change (~lines 740-741):
```python
                import html
                escaped_group_name = html.escape(str(group_name))
```
to:
```python
                escaped_group_name = _FormattingMixin._esc(group_name)
```

In `report_summaries.py`, change (~line 816):
```python
                        name=str(group_name),
```
to:
```python
                        name=_FormattingMixin._esc(group_name),
```

- [ ] **Step 4: Confirm every raw f-string `name=`/`hovertext=` site in these 2 files is now escaped**

Run:
```bash
cd /Users/philippkrumm/Documents/BioMedStatX
git grep -n 'name=f"{factor\|hover_texts.append(\s*$\|name=str(group_name)' src/export/report_charts.py src/export/report_summaries.py
```
Expected: no remaining unescaped occurrence of a factor/group name interpolated directly (every
`name=f"..."` and `hover_texts.append(f"...")` site should now read through
`_FormattingMixin._esc(...)`).

- [ ] **Step 5: Run the full test suite**

Run: `cd /Users/philippkrumm/Documents/BioMedStatX && python3 -m pytest tests/ -q --tb=no`
Expected: same pass count as before this task plus the new test (the 1 pre-existing unrelated
`test_convergence.py::test_convergence_keys` failure is expected). Pay particular attention to
any existing chart-building tests
(`grep -rln "_build_interaction_plot_chart\|_build_.*_chart\|_build_distribution_dashboard_chart" tests/`)
— run those explicitly with `-v` since this task touches shared chart-building code paths.

- [ ] **Step 6: Commit**

```bash
git add tests/test_plotly_label_escaping.py src/export/report_charts.py src/export/report_summaries.py
git commit -m "fix(export): apply consistent HTML escaping to remaining Plotly label sites"
```

---

## Self-review notes

- **Spec coverage:** RE5 (Task 1), RE1 (Task 2), RE2 (Task 3) — all 3 findings assigned to this
  package are covered, in the dependency order RE5 → RE1 → RE2 the master summary's own
  remediation order specifies.
- **RE1's test reproduces the injection against a REAL statsmodels fit**, not a hand-crafted
  dict — `test_patsy_really_does_leak_the_raw_group_name_into_the_parameter` proves the
  vulnerability's actual mechanism (patsy's `C(col)[T.<value>]` encoding) before the escaping
  test asserts the fix closes it, matching how round 1's audit originally demonstrated this
  end-to-end rather than assuming the mechanism.
- **RE2's scope-limitation is stated explicitly, not silently under-tested** — Task 3 documents
  why its 5 sites get a lighter test (grep-based verification + full-suite run) than RE1's
  isolated-function sites, rather than presenting a thin test as if it were equivalent coverage.
